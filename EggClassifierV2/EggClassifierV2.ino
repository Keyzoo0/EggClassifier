/*
 * EggClassifier V2 — Firmware Terpadu (RTOS + N16R8 Maximized)
 * Board  : Freenove ESP32-S3-WROOM CAM (FNK0085, N16R8) + microSD 4GB
 *
 * Library (Tools → Manage Libraries):
 *   1. TensorFlowLite_ESP32 (by tanakamasayuki)
 *   (AXP313A tidak diperlukan — kamera Freenove selalu berdaya)
 *
 * Board Settings (Tools):
 *   Board            : ESP32S3 Dev Module
 *   Flash Size       : 16MB (128Mb)   ← varian N16R8 (esptool detect 16MB)
 *   Partition Scheme : Default 16MB with spiffs (6.25MB APP/3.43MB SPIFFS)
 *                      ← dual-OTA: update firmware via web (/ota), tanpa USB
 *   PSRAM            : OPI PSRAM
 *   CPU Frequency    : 240MHz
 *   USB CDC On Boot  : Enabled
 *
 * ── TATA KELOLA MEMORI ─────────────────────────────────────────
 *   Flash app (2×6.25MB OTA) : firmware — update via web /ota
 *   LittleFS (3.43MB)        : SEMUA yang dibutuhkan alat agar berfungsi
 *                              mandiri tanpa SD: web UI + model (PRIMER).
 *                              Update per-file via /fs/upload (tanpa plugin).
 *   SD card (4GB, FAT32)     : data besar/tumbuh: /dataset/, log prediksi
 *                              /predict_log.csv, backup model (auto-restore
 *                              ke LittleFS saat boot bila LittleFS kosong).
 *   NVS                      : pengaturan kamera (Preferences "camcfg")
 *   PSRAM (8MB OPI)          : frame buffer kamera 2×VGA, rgbArena 921KB,
 *                              buffer model ~315KB
 *   SRAM internal (512KB)    : tensor arena bila muat (akses jauh lebih
 *                              cepat dari PSRAM → inferensi lebih singkat),
 *                              fallback otomatis ke PSRAM
 *
 * Arsitektur dual-core:
 *   Core 0 (prio 5) — inferenceTask: TFLite Invoke
 *   Core 1 (prio 1) — loop(): WebServer.handleClient + LED + SD I/O
 *
 * Endpoints:
 *   GET  /              → Web UI (LittleFS; halaman pemulihan bila kosong)
 *   GET  /capture       → JPEG frame live (preview & download dataset)
 *   GET  /predict       → inference → JSON hasil (+ catat ke log SD)
 *   GET  /model_info    → JSON: {loaded, size_kb, arena_kb, arena_loc, model_loc, err}
 *   POST /upload_model  → .tflite → LittleFS (+ backup SD) → restart
 *   GET  /sd/info       → JSON: {mounted, good, bad, used_mb, total_mb}
 *   POST /sd/capture    → ?label=good|bad → simpan JPEG ke SD
 *   GET  /sd/list       → JSON daftar file dataset di SD
 *   GET  /sd/file       → ?name=good_0001.jpg → stream JPEG dari SD
 *   POST /sd/delete     → ?name=good_0001.jpg → hapus file
 *   POST /sd/relabel    → ?name=good_0001.jpg → rename ke label lawan
 *   POST /sd/upload     → ?label=good|bad + multipart JPEG dari PC
 *   GET  /camera/get    → JSON semua parameter sensor OV2640
 *   POST /camera/set    → ?var=brightness&val=1 → set + simpan ke NVS
 *   POST /camera/reset  → hapus setting custom → restart (default pabrik)
 *   GET  /sys/info      → JSON statistik memori/penyimpanan/firmware
 *   POST /ota           → update firmware via web (butuh partisi dual-OTA)
 *   POST /fs/upload     → upload file web UI ke LittleFS (ganti plugin)
 *   GET  /log           → CSV log prediksi dari SD (?tail=1 → 16KB terakhir)
 *   POST /log/clear     → hapus log prediksi
 */

#include <WiFi.h>
#include <WebServer.h>
#include <ESPmDNS.h>
#include <LittleFS.h>
#include <SD_MMC.h>
#include <Preferences.h>
#include <Update.h>
#include <esp_ota_ops.h>
#include <esp_heap_caps.h>
#include <time.h>
#include <new>
#include "esp_camera.h"
#include "img_converters.h"
#include "TensorFlowLite_ESP32.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/semphr.h>

// ── Credentials ───────────────────────────────────────────────
#define WIFI_SSID  "ROSI1"
#define WIFI_PASS  "20517420"

#define FW_VERSION "2.2.0"

// ── Pin Camera — Freenove FNK0085 (profil ESP32S3_EYE) ───────
#define CAM_PWDN   -1
#define CAM_RESET  -1
#define CAM_XCLK   15
#define CAM_SIOD    4
#define CAM_SIOC    5
#define CAM_D7     16   // Y9
#define CAM_D6     17   // Y8
#define CAM_D5     18   // Y7
#define CAM_D4     12   // Y6
#define CAM_D3     10   // Y5
#define CAM_D2      8   // Y4
#define CAM_D1      9   // Y3
#define CAM_D0     11   // Y2
#define CAM_VSYNC   6
#define CAM_HREF    7
#define CAM_PCLK   13

// ── Pin Board ─────────────────────────────────────────────────
#define RGB_LED_PIN 48  // WS2812 onboard: biru kedip=WiFi connecting, hijau=OK
#define BOOT_BTN     0

// ── Pin SD card (SDMMC 1-bit, fixed di PCB Freenove) ─────────
#define SD_PIN_CMD  38
#define SD_PIN_CLK  39
#define SD_PIN_D0   40
#define SD_DATASET_DIR "/dataset"

// ── TFLite ────────────────────────────────────────────────────
#define ARENA_SIZE (200 * 1024)   // 200 KB dari PSRAM

// ── Kamera & Inferensi ────────────────────────────────────────
// Camera selalu VGA; tidak pernah ganti resolusi saat inferensi.
// Frame VGA di-scale nearest-neighbor → 96×96 di PSRAM buffer.
#define RGB_MAX_W  640
#define RGB_MAX_H  480
#define CROP_SZ     96

// ── Global ────────────────────────────────────────────────────
WebServer server(80);

// SD card state
bool     sdMounted = false;
uint32_t sdCntGood = 0;   // jumlah file AKTUAL per label (bukan max index)
uint32_t sdCntBad  = 0;

// Bitmap index terpakai per label — file baru mengisi celah index
// terendah (bekas hapus dipakai ulang), bukan lanjut dari max.
#define SD_MAX_IDX 9999
static uint8_t sdGoodMap[(SD_MAX_IDX / 8) + 1];
static uint8_t sdBadMap [(SD_MAX_IDX / 8) + 1];

static inline void sdIdxMark(bool good, uint32_t n, bool used) {
  if (n < 1 || n > SD_MAX_IDX) return;
  uint8_t* m = good ? sdGoodMap : sdBadMap;
  if (used) m[n >> 3] |=  (1 << (n & 7));
  else      m[n >> 3] &= ~(1 << (n & 7));
}
static inline bool sdIdxUsed(bool good, uint32_t n) {
  const uint8_t* m = good ? sdGoodMap : sdBadMap;
  return m[n >> 3] & (1 << (n & 7));
}
static uint32_t sdNextIdx(bool good) {   // index kosong terendah, 0 = penuh
  for (uint32_t n = 1; n <= SD_MAX_IDX; n++)
    if (!sdIdxUsed(good, n)) return n;
  return 0;
}
// Parse "good_0012.jpg" → label + index
static bool sdParseName(const char* base, bool* isGood, uint32_t* idx) {
  unsigned long n = 0;
  if (sscanf(base, "good_%lu", &n) == 1) { *isGood = true;  *idx = n; return true; }
  if (sscanf(base, "bad_%lu",  &n) == 1) { *isGood = false; *idx = n; return true; }
  return false;
}

// TFLite
static uint8_t*                           tensorArena   = nullptr;
static bool                               arenaInternal = false;  // arena di SRAM?
static const char*                        modelLoc      = "none"; // littlefs|sd
static tflite::MicroMutableOpResolver<12> resolver;

// Error reporter yang menyimpan pesan terakhir — alasan gagal load
// model (mis. op tidak terdaftar) bisa dilihat di web via /model_info
class BufErrorReporter : public tflite::ErrorReporter {
 public:
  char last[96] = "";
  int Report(const char* format, va_list args) override {
    vsnprintf(last, sizeof(last), format, args);
    Serial.printf("[TF] %s\n", last);
    return 0;
  }
};
static BufErrorReporter tfliteErrReporter;
static tflite::MicroInterpreter*          interpreter = nullptr;
static TfLiteTensor*                      inTensor    = nullptr;
static TfLiteTensor*                      outTensor   = nullptr;
bool   modelLoaded = false;
size_t modelSizeKB = 0;

// Slot interpreter (placement-new) — interpreter bisa dibangun ulang saat
// arena dipindah dari PSRAM ke SRAM internal tanpa alokasi dinamis baru
alignas(16) static uint8_t interpSlot[sizeof(tflite::MicroInterpreter)];

// PSRAM decode buffer — pre-alokasi sekali, tidak malloc/free tiap inferensi
static uint8_t* rgbArena = nullptr;

// Dual-core sync: inferensi di Core 0, HTTP di Core 1
volatile bool classifying = false;

struct InferResult {
  float    score;
  bool     good;
  uint32_t time_ms;
  bool     ok;
  char     err[32];
};
static SemaphoreHandle_t inferTrigSem    = NULL;
static SemaphoreHandle_t inferDoneSem    = NULL;
static TaskHandle_t      inferTaskHandle = NULL;
static InferResult       inferResult;

File uploadFile;

// ─────────────────────────────────────────────────────────────
// LED status — WS2812 onboard GPIO48 (non-blocking)
//   Biru kedip = WiFi connecting/reconnect, hijau redup = WiFi OK
//   Hijau/merah terang sesaat = hasil klasifikasi
// ─────────────────────────────────────────────────────────────
static volatile uint32_t ledHoldUntil = 0;

void ledShow(uint8_t r, uint8_t g, uint8_t b, uint32_t holdMs) {
  ledHoldUntil = millis() + holdMs;
  neopixelWrite(RGB_LED_PIN, r, g, b);
}

void updateLED() {
  static uint32_t lastBlink = 0;
  static bool     blinkOn   = false;
  if (millis() < ledHoldUntil) return;   // sedang menampilkan hasil

  if (WiFi.status() == WL_CONNECTED) {
    neopixelWrite(RGB_LED_PIN, 0, 12, 0);          // hijau redup
  } else {
    if (millis() - lastBlink >= 250) {
      lastBlink = millis();
      blinkOn   = !blinkOn;
      neopixelWrite(RGB_LED_PIN, 0, 0, blinkOn ? 32 : 0);  // biru kedip
    }
  }
}

// ─────────────────────────────────────────────────────────────
// Camera — VGA double-buffer di PSRAM, tidak pernah ganti resolusi
// ─────────────────────────────────────────────────────────────
bool initCamera() {
  camera_config_t cfg = {};
  cfg.ledc_channel = LEDC_CHANNEL_0;
  cfg.ledc_timer   = LEDC_TIMER_0;
  cfg.pin_d0 = CAM_D0; cfg.pin_d1 = CAM_D1; cfg.pin_d2 = CAM_D2;
  cfg.pin_d3 = CAM_D3; cfg.pin_d4 = CAM_D4; cfg.pin_d5 = CAM_D5;
  cfg.pin_d6 = CAM_D6; cfg.pin_d7 = CAM_D7;
  cfg.pin_xclk     = CAM_XCLK;
  cfg.pin_pclk     = CAM_PCLK;
  cfg.pin_vsync    = CAM_VSYNC;
  cfg.pin_href     = CAM_HREF;
  cfg.pin_sccb_sda = CAM_SIOD;
  cfg.pin_sccb_scl = CAM_SIOC;
  cfg.pin_pwdn     = CAM_PWDN;
  cfg.pin_reset    = CAM_RESET;
  cfg.xclk_freq_hz = 20000000;
  cfg.pixel_format = PIXFORMAT_JPEG;
  cfg.grab_mode    = CAMERA_GRAB_LATEST;  // selalu ambil frame terbaru
  cfg.frame_size   = FRAMESIZE_VGA;       // 640×480, tidak pernah berubah
  cfg.jpeg_quality = 6;
  cfg.fb_count     = 2;                   // double-buffer di PSRAM
  cfg.fb_location  = CAMERA_FB_IN_PSRAM;

  if (esp_camera_init(&cfg) != ESP_OK) {
    Serial.println("[ERROR] Camera init gagal");
    return false;
  }
  sensor_t* s = esp_camera_sensor_get();
  if (s) {
    s->set_brightness(s, 1);
    s->set_contrast(s, 1);
    s->set_awb_gain(s, 1);
  }
  Serial.println("[OK] Camera: VGA 640×480 JPEG double-buffer PSRAM");
  return true;
}

// ─────────────────────────────────────────────────────────────
// Camera settings — set parameter sensor + persist di NVS
// Setting tersimpan permanen agar exposure/WB konsisten antara
// sesi pengambilan dataset dan prediksi.
// ─────────────────────────────────────────────────────────────
Preferences camPrefs;

static const char* const CAM_VARS[] = {
  "quality", "brightness", "contrast", "saturation", "special_effect",
  "wb_mode", "awb", "awb_gain", "aec", "aec2", "ae_level", "aec_value",
  "agc", "agc_gain", "gainceiling", "bpc", "wpc", "raw_gma", "lenc",
  "hmirror", "vflip", "dcw", "colorbar",
};
static const size_t CAM_VARS_N = sizeof(CAM_VARS) / sizeof(CAM_VARS[0]);

bool applyCameraSetting(sensor_t* s, const String& var, int val) {
  if (var == "quality")        return s->set_quality(s, val)        == 0;
  if (var == "brightness")     return s->set_brightness(s, val)     == 0;
  if (var == "contrast")       return s->set_contrast(s, val)       == 0;
  if (var == "saturation")     return s->set_saturation(s, val)     == 0;
  if (var == "special_effect") return s->set_special_effect(s, val) == 0;
  if (var == "wb_mode")        return s->set_wb_mode(s, val)        == 0;
  if (var == "awb")            return s->set_whitebal(s, val)       == 0;
  if (var == "awb_gain")       return s->set_awb_gain(s, val)       == 0;
  if (var == "aec")            return s->set_exposure_ctrl(s, val)  == 0;
  if (var == "aec2")           return s->set_aec2(s, val)           == 0;
  if (var == "ae_level")       return s->set_ae_level(s, val)       == 0;
  if (var == "aec_value")      return s->set_aec_value(s, val)      == 0;
  if (var == "agc")            return s->set_gain_ctrl(s, val)      == 0;
  if (var == "agc_gain")       return s->set_agc_gain(s, val)       == 0;
  if (var == "gainceiling")    return s->set_gainceiling(s, (gainceiling_t)val) == 0;
  if (var == "bpc")            return s->set_bpc(s, val)            == 0;
  if (var == "wpc")            return s->set_wpc(s, val)            == 0;
  if (var == "raw_gma")        return s->set_raw_gma(s, val)        == 0;
  if (var == "lenc")           return s->set_lenc(s, val)           == 0;
  if (var == "hmirror")        return s->set_hmirror(s, val)        == 0;
  if (var == "vflip")          return s->set_vflip(s, val)          == 0;
  if (var == "dcw")            return s->set_dcw(s, val)            == 0;
  if (var == "colorbar")       return s->set_colorbar(s, val)       == 0;
  return false;
}

// Terapkan setting tersimpan di NVS — dipanggil sekali setelah initCamera
void loadCameraSettings() {
  sensor_t* s = esp_camera_sensor_get();
  if (!s) return;
  camPrefs.begin("camcfg", true);
  int applied = 0;
  for (size_t i = 0; i < CAM_VARS_N; i++) {
    if (camPrefs.isKey(CAM_VARS[i])) {
      if (applyCameraSetting(s, CAM_VARS[i], camPrefs.getInt(CAM_VARS[i]))) applied++;
    }
  }
  camPrefs.end();
  if (applied) Serial.printf("[OK] Camera: %d setting custom dari NVS\n", applied);
}

// ─────────────────────────────────────────────────────────────
// HTTP: baca semua parameter kamera (dari register sensor)
// ─────────────────────────────────────────────────────────────
void handleCamGet() {
  sensor_t* s = esp_camera_sensor_get();
  if (!s) { server.send(503, "application/json", "{\"error\":\"no_camera\"}"); return; }

  char resp[640];
  snprintf(resp, sizeof(resp),
    "{\"quality\":%d,\"brightness\":%d,\"contrast\":%d,\"saturation\":%d,"
    "\"special_effect\":%d,\"wb_mode\":%d,\"awb\":%d,\"awb_gain\":%d,"
    "\"aec\":%d,\"aec2\":%d,\"ae_level\":%d,\"aec_value\":%d,"
    "\"agc\":%d,\"agc_gain\":%d,\"gainceiling\":%d,"
    "\"bpc\":%d,\"wpc\":%d,\"raw_gma\":%d,\"lenc\":%d,"
    "\"hmirror\":%d,\"vflip\":%d,\"dcw\":%d,\"colorbar\":%d}",
    s->status.quality, s->status.brightness, s->status.contrast,
    s->status.saturation, s->status.special_effect, s->status.wb_mode,
    s->status.awb, s->status.awb_gain, s->status.aec, s->status.aec2,
    s->status.ae_level, s->status.aec_value, s->status.agc,
    s->status.agc_gain, s->status.gainceiling, s->status.bpc,
    s->status.wpc, s->status.raw_gma, s->status.lenc,
    s->status.hmirror, s->status.vflip, s->status.dcw, s->status.colorbar);
  server.send(200, "application/json", resp);
}

// ─────────────────────────────────────────────────────────────
// HTTP: set satu parameter (/camera/set?var=brightness&val=1)
// ─────────────────────────────────────────────────────────────
void handleCamSet() {
  sensor_t* s = esp_camera_sensor_get();
  if (!s) { server.send(503, "application/json", "{\"error\":\"no_camera\"}"); return; }

  String var = server.arg("var");
  if (!server.hasArg("var") || !server.hasArg("val")) {
    server.send(400, "application/json", "{\"error\":\"missing_args\"}"); return;
  }
  int val = server.arg("val").toInt();

  if (!applyCameraSetting(s, var, val)) {
    server.send(400, "application/json", "{\"error\":\"bad_var_or_val\"}"); return;
  }

  // Persist ke NVS — kecuali colorbar (pola tes, tidak perlu permanen)
  if (var != "colorbar") {
    camPrefs.begin("camcfg", false);
    camPrefs.putInt(var.c_str(), val);
    camPrefs.end();
  }

  char resp[80];
  snprintf(resp, sizeof(resp), "{\"ok\":true,\"var\":\"%s\",\"val\":%d}",
           var.c_str(), val);
  server.send(200, "application/json", resp);
}

// ─────────────────────────────────────────────────────────────
// HTTP: reset semua setting kamera → default pabrik (restart)
// ─────────────────────────────────────────────────────────────
void handleCamReset() {
  camPrefs.begin("camcfg", false);
  camPrefs.clear();
  camPrefs.end();
  server.send(200, "application/json", "{\"ok\":true,\"restarting\":true}");
  vTaskDelay(pdMS_TO_TICKS(500));
  Serial.println("[INFO] Camera setting direset, restart...");
  ESP.restart();
}

// ─────────────────────────────────────────────────────────────
// Model storage — LittleFS PRIMER (alat berfungsi tanpa SD),
// SD card = backup otomatis. Saat boot kedua salinan disinkron:
//   LittleFS kosong + SD ada  → pulihkan SD → LittleFS
//   keduanya ada, ukuran beda → LittleFS menang → salin ke SD
// Jadi model selamat dari upload ulang LittleFS (plugin lama)
// MAUPUN dari SD card dicabut/diganti.
// ─────────────────────────────────────────────────────────────
#define MODEL_PATH       "/egg_model.tflite"
#define PREDICT_LOG_PATH "/predict_log.csv"

static bool copyFile(fs::FS& src, fs::FS& dst, const char* path) {
  File in = src.open(path, "r");
  if (!in) return false;
  File out = dst.open(path, "w");
  if (!out) { in.close(); return false; }
  static uint8_t buf[4096];
  bool ok = true;
  while (in.available()) {
    size_t n = in.read(buf, sizeof(buf));
    if (n == 0 || out.write(buf, n) != n) { ok = false; break; }
  }
  in.close();
  out.close();
  if (!ok) dst.remove(path);   // jangan tinggalkan salinan korup
  return ok;
}

static size_t fileSize(fs::FS& fs, const char* path) {
  File f = fs.open(path, "r");
  if (!f) return 0;
  size_t s = f.size();
  f.close();
  return s;
}

void syncModelStorage() {
  bool inLfs = LittleFS.exists(MODEL_PATH);
  bool inSd  = sdMounted && SD_MMC.exists(MODEL_PATH);

  if (!inLfs && inSd) {
    Serial.println(copyFile(SD_MMC, LittleFS, MODEL_PATH)
      ? "[TF] Model dipulihkan dari backup SD → LittleFS"
      : "[WARN] Gagal memulihkan model dari SD");
  } else if (inLfs && sdMounted &&
             fileSize(LittleFS, MODEL_PATH) != fileSize(SD_MMC, MODEL_PATH)) {
    Serial.println(copyFile(LittleFS, SD_MMC, MODEL_PATH)
      ? "[TF] Backup model → SD card"
      : "[WARN] Gagal backup model ke SD");
  }
}

// ─────────────────────────────────────────────────────────────
// TFLite: load model + alokasi arena
// Pass 1: arena di PSRAM (pasti muat). Pass 2: bila SRAM internal
// cukup, interpreter dibangun ulang dengan arena pas-ukuran di SRAM
// — akses SRAM jauh lebih cepat dari PSRAM (bus OPI) sehingga
// inferensi lebih singkat. Sisakan ≥120 KB SRAM untuk WiFi + HTTP.
// ─────────────────────────────────────────────────────────────
static bool buildInterpreter(const tflite::Model* model, uint8_t* arena, size_t arenaSz) {
  if (interpreter) { interpreter->~MicroInterpreter(); interpreter = nullptr; }
  memset(arena, 0, arenaSz);
  interpreter = new (interpSlot)
    tflite::MicroInterpreter(model, resolver, arena, arenaSz, &tfliteErrReporter);
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    interpreter->~MicroInterpreter();
    interpreter = nullptr;
    return false;
  }
  inTensor  = interpreter->input(0);
  outTensor = interpreter->output(0);
  return true;
}

bool initTFLite() {
  File f;
  if (LittleFS.exists(MODEL_PATH)) {
    f = LittleFS.open(MODEL_PATH, "r");
    modelLoc = "littlefs";
    Serial.println("[TF] Model dari LittleFS");
  } else if (sdMounted && SD_MMC.exists(MODEL_PATH)) {
    f = SD_MMC.open(MODEL_PATH, "r");
    modelLoc = "sd";
    Serial.println("[TF] Model dari SD card");
  } else {
    Serial.println("[TF] Tidak ada model (LittleFS/SD)");
    return false;
  }
  size_t sz = f.size();
  modelSizeKB = sz / 1024;

  uint8_t* buf = (uint8_t*) ps_malloc(sz);
  if (!buf) { f.close(); Serial.println("[TF] ps_malloc model gagal"); return false; }
  f.read(buf, sz);
  f.close();

  const tflite::Model* model = tflite::GetModel(buf);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    snprintf(tfliteErrReporter.last, sizeof(tfliteErrReporter.last),
             "schema v%d != v%d", (int)model->version(), TFLITE_SCHEMA_VERSION);
    Serial.printf("[TF] %s\n", tfliteErrReporter.last);
    free(buf); return false;
  }

  // Pass 1 — arena di PSRAM
  if (!tensorArena) {
    tensorArena = (uint8_t*) ps_malloc(ARENA_SIZE);
    if (!tensorArena) {
      snprintf(tfliteErrReporter.last, sizeof(tfliteErrReporter.last),
               "ps_malloc arena gagal");
      free(buf); Serial.println("[TF] ps_malloc arena gagal"); return false;
    }
  }
  if (!buildInterpreter(model, tensorArena, ARENA_SIZE)) {
    // Detail (mis. op tidak terdaftar) sudah tertangkap di errReporter.last
    Serial.println("[TF] AllocateTensors gagal — cek op resolver / ARENA_SIZE");
    return false;
  }

  // Pass 2 — pindahkan arena ke SRAM internal bila muat
  size_t need = interpreter->arena_used_bytes() + 16 * 1024;
  if (heap_caps_get_free_size(MALLOC_CAP_INTERNAL) >= need + 120 * 1024) {
    uint8_t* sram = (uint8_t*) heap_caps_malloc(need, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (sram && buildInterpreter(model, sram, need)) {
      free(tensorArena);            // arena PSRAM tak terpakai lagi
      tensorArena   = sram;
      arenaInternal = true;
    } else {
      if (sram) free(sram);
      if (!buildInterpreter(model, tensorArena, ARENA_SIZE)) return false;
    }
  }

  tfliteErrReporter.last[0] = '\0';   // sukses — bersihkan error lama
  modelLoaded = true;

  Serial.printf("[OK] TFLite: %d KB (%s) | arena %d KB di %s | input [%d,%d,%d,%d]\n",
    (int)modelSizeKB, modelLoc,
    (int)(interpreter->arena_used_bytes() / 1024),
    arenaInternal ? "SRAM internal" : "PSRAM",
    inTensor->dims->data[0], inTensor->dims->data[1],
    inTensor->dims->data[2], inTensor->dims->data[3]);
  return true;
}

// ─────────────────────────────────────────────────────────────
// SD card — mount SDMMC 1-bit + scan counter dataset
// ─────────────────────────────────────────────────────────────
bool initSD() {
  SD_MMC.setPins(SD_PIN_CLK, SD_PIN_CMD, SD_PIN_D0);
  if (!SD_MMC.begin("/sdcard", true)) {   // true = mode 1-bit
    Serial.println("[WARN] SD card tidak terdeteksi — dataset mode download");
    return false;
  }
  if (!SD_MMC.exists(SD_DATASET_DIR)) SD_MMC.mkdir(SD_DATASET_DIR);

  // Scan dataset: hitung file aktual per label + tandai index terpakai
  File dir = SD_MMC.open(SD_DATASET_DIR);
  File f;
  while ((f = dir.openNextFile())) {
    const char* base = strrchr(f.name(), '/');
    base = base ? base + 1 : f.name();
    bool g; uint32_t n;
    if (!f.isDirectory() && sdParseName(base, &g, &n)) {
      sdIdxMark(g, n, true);
      if (g) sdCntGood++; else sdCntBad++;
    }
    f.close();
  }
  dir.close();

  Serial.printf("[OK] SD: %llu MB (good=%lu, bad=%lu)\n",
    SD_MMC.totalBytes() / (1024ULL * 1024ULL),
    (unsigned long)sdCntGood, (unsigned long)sdCntBad);
  return true;
}

// Tolak nama file dengan path traversal
static bool sdNameValid(const String& name) {
  return name.length() > 0 && name.indexOf('/') < 0 && name.indexOf("..") < 0;
}

// ─────────────────────────────────────────────────────────────
// HTTP: SD info — dipolling web UI untuk counter dataset
// ─────────────────────────────────────────────────────────────
void handleSDInfo() {
  char resp[160];
  snprintf(resp, sizeof(resp),
    "{\"mounted\":%s,\"good\":%lu,\"bad\":%lu,\"used_mb\":%lu,\"total_mb\":%lu}",
    sdMounted ? "true" : "false",
    (unsigned long)sdCntGood, (unsigned long)sdCntBad,
    sdMounted ? (unsigned long)(SD_MMC.usedBytes()  / (1024UL * 1024UL)) : 0,
    sdMounted ? (unsigned long)(SD_MMC.totalBytes() / (1024UL * 1024UL)) : 0);
  server.send(200, "application/json", resp);
}

// ─────────────────────────────────────────────────────────────
// HTTP: capture → simpan langsung ke SD (/sd/capture?label=good)
// ─────────────────────────────────────────────────────────────
void handleSDCapture() {
  if (!sdMounted)  { server.send(503, "application/json", "{\"error\":\"no_sd\"}"); return; }
  if (classifying) { server.send(503, "application/json", "{\"error\":\"busy\"}");  return; }

  String label = server.arg("label");
  if (label != "good" && label != "bad") {
    server.send(400, "application/json", "{\"error\":\"bad_label\"}"); return;
  }

  const bool isGood = (label == "good");
  uint32_t n = sdNextIdx(isGood);   // isi celah terendah (bekas hapus)
  if (n == 0) { server.send(507, "application/json", "{\"error\":\"dataset_full\"}"); return; }

  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) { server.send(503, "application/json", "{\"error\":\"camera_failed\"}"); return; }

  char path[48];
  snprintf(path, sizeof(path), SD_DATASET_DIR "/%s_%04lu.jpg",
           label.c_str(), (unsigned long)n);

  File f = SD_MMC.open(path, FILE_WRITE);
  if (!f) {
    esp_camera_fb_return(fb);
    server.send(500, "application/json", "{\"error\":\"sd_write_failed\"}");
    return;
  }
  size_t written = f.write(fb->buf, fb->len);
  f.close();
  esp_camera_fb_return(fb);

  if (written == 0) {
    SD_MMC.remove(path);
    server.send(500, "application/json", "{\"error\":\"sd_write_failed\"}");
    return;
  }

  sdIdxMark(isGood, n, true);
  if (isGood) sdCntGood++; else sdCntBad++;

  char resp[160];
  snprintf(resp, sizeof(resp),
    "{\"ok\":true,\"file\":\"%s_%04lu.jpg\",\"good\":%lu,\"bad\":%lu,\"size_kb\":%lu}",
    label.c_str(), (unsigned long)n,
    (unsigned long)sdCntGood, (unsigned long)sdCntBad,
    (unsigned long)(written / 1024));
  server.send(200, "application/json", resp);
}

// ─────────────────────────────────────────────────────────────
// HTTP: daftar file dataset di SD (JSON, dikirim chunked)
// ─────────────────────────────────────────────────────────────
void handleSDList() {
  if (!sdMounted) { server.send(503, "application/json", "{\"error\":\"no_sd\"}"); return; }

  server.setContentLength(CONTENT_LENGTH_UNKNOWN);
  server.send(200, "application/json", "");
  server.sendContent("[");

  File dir = SD_MMC.open(SD_DATASET_DIR);
  File f;
  bool first = true;
  char item[96];
  while ((f = dir.openNextFile())) {
    if (!f.isDirectory()) {
      const char* base = strrchr(f.name(), '/');
      base = base ? base + 1 : f.name();
      snprintf(item, sizeof(item), "%s{\"n\":\"%s\",\"s\":%lu}",
               first ? "" : ",", base, (unsigned long)f.size());
      server.sendContent(item);
      first = false;
    }
    f.close();
  }
  dir.close();
  server.sendContent("]");
  server.sendContent("");   // akhiri chunked response
}

// ─────────────────────────────────────────────────────────────
// HTTP: stream satu file dataset dari SD (/sd/file?name=...)
// ─────────────────────────────────────────────────────────────
void handleSDFile() {
  if (!sdMounted) { server.send(503, "text/plain", "no sd"); return; }
  String name = server.arg("name");
  if (!sdNameValid(name)) { server.send(400, "text/plain", "bad name"); return; }

  String path = String(SD_DATASET_DIR) + "/" + name;
  if (!SD_MMC.exists(path)) { server.send(404, "text/plain", "not found"); return; }

  File f = SD_MMC.open(path, FILE_READ);
  server.streamFile(f, "image/jpeg");
  f.close();
}

// ─────────────────────────────────────────────────────────────
// HTTP: hapus satu file dataset (/sd/delete?name=...)
// ─────────────────────────────────────────────────────────────
void handleSDDelete() {
  if (!sdMounted) { server.send(503, "application/json", "{\"error\":\"no_sd\"}"); return; }
  String name = server.arg("name");
  if (!sdNameValid(name)) { server.send(400, "application/json", "{\"error\":\"bad_name\"}"); return; }

  String path = String(SD_DATASET_DIR) + "/" + name;
  if (!SD_MMC.exists(path)) { server.send(404, "application/json", "{\"error\":\"not_found\"}"); return; }

  bool ok = SD_MMC.remove(path);
  if (ok) {
    // Bebaskan index agar bisa dipakai capture berikutnya + koreksi jumlah
    bool g; uint32_t idx;
    if (sdParseName(name.c_str(), &g, &idx)) {
      sdIdxMark(g, idx, false);
      if (g) { if (sdCntGood) sdCntGood--; }
      else   { if (sdCntBad)  sdCntBad--;  }
    }
  }
  server.send(ok ? 200 : 500, "application/json",
              ok ? "{\"ok\":true}" : "{\"error\":\"delete_failed\"}");
}

// ─────────────────────────────────────────────────────────────
// HTTP: pindah label (/sd/relabel?name=good_0001.jpg)
// Rename ke index berikutnya pada label lawan: good ↔ bad
// ─────────────────────────────────────────────────────────────
void handleSDRelabel() {
  if (!sdMounted) { server.send(503, "application/json", "{\"error\":\"no_sd\"}"); return; }
  String name = server.arg("name");
  if (!sdNameValid(name)) { server.send(400, "application/json", "{\"error\":\"bad_name\"}"); return; }

  bool fromGood; uint32_t oldIdx;
  if (!sdParseName(name.c_str(), &fromGood, &oldIdx)) {
    server.send(400, "application/json", "{\"error\":\"bad_label\"}"); return;
  }

  String from = String(SD_DATASET_DIR) + "/" + name;
  if (!SD_MMC.exists(from)) { server.send(404, "application/json", "{\"error\":\"not_found\"}"); return; }

  uint32_t n = sdNextIdx(!fromGood);   // celah terendah di label tujuan
  if (n == 0) { server.send(507, "application/json", "{\"error\":\"dataset_full\"}"); return; }
  char newName[32];
  snprintf(newName, sizeof(newName), "%s_%04lu.jpg",
           fromGood ? "bad" : "good", (unsigned long)n);

  if (!SD_MMC.rename(from, String(SD_DATASET_DIR) + "/" + newName)) {
    server.send(500, "application/json", "{\"error\":\"rename_failed\"}"); return;
  }
  sdIdxMark(fromGood,  oldIdx, false);
  sdIdxMark(!fromGood, n,      true);
  if (fromGood) { if (sdCntGood) sdCntGood--; sdCntBad++; }
  else          { if (sdCntBad)  sdCntBad--;  sdCntGood++; }

  char resp[128];
  snprintf(resp, sizeof(resp),
    "{\"ok\":true,\"file\":\"%s\",\"good\":%lu,\"bad\":%lu}",
    newName, (unsigned long)sdCntGood, (unsigned long)sdCntBad);
  server.send(200, "application/json", resp);
}

// ─────────────────────────────────────────────────────────────
// HTTP: upload JPEG dari PC ke dataset (/sd/upload?label=good)
// Nama file ditentukan alat (index berikutnya) agar tidak bentrok
// ─────────────────────────────────────────────────────────────
static File     sdUpFile;
static String   sdUpName;
static uint32_t sdUpIdx   = 0;
static bool     sdUpIsGood = false;
static bool     sdUpError  = false;

void handleSDUploadChunk() {
  HTTPUpload& up = server.upload();
  if (up.status == UPLOAD_FILE_START) {
    sdUpError = false;
    sdUpName  = "";
    String label = server.arg("label");
    if (!sdMounted || (label != "good" && label != "bad")) { sdUpError = true; return; }

    sdUpIsGood = (label == "good");
    sdUpIdx    = sdNextIdx(sdUpIsGood);   // isi celah terendah
    if (sdUpIdx == 0) { sdUpError = true; return; }
    char path[48];
    snprintf(path, sizeof(path), SD_DATASET_DIR "/%s_%04lu.jpg",
             label.c_str(), (unsigned long)sdUpIdx);
    sdUpFile = SD_MMC.open(path, FILE_WRITE);
    if (!sdUpFile) { sdUpError = true; return; }
    sdUpName = String(label) + "_";
    char num[12]; snprintf(num, sizeof(num), "%04lu.jpg", (unsigned long)sdUpIdx);
    sdUpName += num;
  } else if (up.status == UPLOAD_FILE_WRITE) {
    if (sdUpFile && !sdUpError) sdUpFile.write(up.buf, up.currentSize);
  } else if (up.status == UPLOAD_FILE_END) {
    if (sdUpFile) sdUpFile.close();
    if (!sdUpError && up.totalSize > 0) {
      sdIdxMark(sdUpIsGood, sdUpIdx, true);
      if (sdUpIsGood) sdCntGood++; else sdCntBad++;
    }
  } else if (up.status == UPLOAD_FILE_ABORTED) {
    if (sdUpFile) sdUpFile.close();
    sdUpError = true;
  }
}

void handleSDUploadDone() {
  if (sdUpError || sdUpName.length() == 0) {
    server.send(500, "application/json", "{\"error\":\"upload_failed\"}");
    return;
  }
  char resp[128];
  snprintf(resp, sizeof(resp),
    "{\"ok\":true,\"file\":\"%s\",\"good\":%lu,\"bad\":%lu}",
    sdUpName.c_str(), (unsigned long)sdCntGood, (unsigned long)sdCntBad);
  server.send(200, "application/json", resp);
}

// ─────────────────────────────────────────────────────────────
// HTTP: file dari LittleFS
// Bila index.html belum ada (mis. flash pertama setelah ganti
// partition scheme), sajikan halaman pemulihan tertanam di firmware:
// upload web UI ke LittleFS / OTA langsung dari browser, tanpa plugin.
// ─────────────────────────────────────────────────────────────
static const char RECOVERY_HTML[] PROGMEM = R"rawliteral(
<!DOCTYPE html><html lang="id"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Egg Classifier — Pemulihan</title>
<style>body{font-family:sans-serif;background:#0f172a;color:#e2e8f0;max-width:480px;margin:40px auto;padding:0 16px}
.card{background:#1e293b;border-radius:12px;padding:20px;margin-bottom:16px}
h1{font-size:20px}h2{font-size:15px;margin:0 0 8px}p{font-size:13px;color:#94a3b8}
code{color:#a5b4fc}input,button{width:100%;margin-top:8px;padding:10px;border-radius:8px;border:0;font-size:14px;box-sizing:border-box}
button{background:#6366f1;color:#fff;font-weight:600;cursor:pointer}
.st{font-size:13px;margin-top:8px;color:#a5b4fc}</style></head><body>
<h1>🥚 Egg Classifier — Mode Pemulihan</h1>
<p>File web UI belum ada di LittleFS. Upload <b>index.html, app.js, style.css</b>
dari folder <code>EggClassifierV2/data/</code>, halaman akan dimuat ulang otomatis.</p>
<div class="card"><h2>Upload Web UI</h2>
<input type="file" id="f" multiple accept=".html,.js,.css">
<button onclick="up()">Upload ke LittleFS</button><div class="st" id="st"></div></div>
<div class="card"><h2>Update Firmware (OTA)</h2>
<input type="file" id="fw" accept=".bin"><button onclick="ota()">Flash Firmware</button>
<div class="st" id="st2"></div></div>
<script>
async function up(){const fs=document.getElementById('f').files,s=document.getElementById('st');
if(!fs.length){s.textContent='Pilih file dulu';return}
for(let i=0;i<fs.length;i++){s.textContent='Upload '+fs[i].name+'…';
const fd=new FormData();fd.append('file',fs[i]);
const r=await fetch('/fs/upload',{method:'POST',body:fd});
if(!r.ok){s.textContent='Gagal: '+fs[i].name;return}}
s.textContent='Selesai! Memuat ulang…';setTimeout(()=>location.reload(),800)}
async function ota(){const f=document.getElementById('fw').files[0],s=document.getElementById('st2');
if(!f){s.textContent='Pilih file .bin dulu';return}
s.textContent='Flashing… jangan matikan alat';
const fd=new FormData();fd.append('firmware',f);
const r=await fetch('/ota',{method:'POST',body:fd});const j=await r.json().catch(()=>({}));
s.textContent=j.ok?'Sukses! Alat restart…':'Gagal: '+(j.error||r.status)}
</script></body></html>
)rawliteral";

void serveFile(const char* path, const char* mime) {
  if (!LittleFS.exists(path)) { server.send(404, "text/plain", "Not found"); return; }
  File f = LittleFS.open(path, "r");
  server.streamFile(f, mime);
  f.close();
}
void handleRoot() {
  if (!LittleFS.exists("/index.html")) {
    server.send_P(200, "text/html", RECOVERY_HTML);
    return;
  }
  serveFile("/index.html", "text/html");
}
void handleAppJS() { serveFile("/app.js",     "application/javascript"); }
void handleCSS()   { serveFile("/style.css",  "text/css"); }

// ─────────────────────────────────────────────────────────────
// HTTP: upload file web UI → LittleFS (/fs/upload)
// Pengganti plugin "LittleFS Data Upload": per-file via web, partisi
// tidak ditimpa total → model & file lain aman.
// ─────────────────────────────────────────────────────────────
static File   fsUpFile;
static String fsUpName;
static bool   fsUpError = false;

void handleFSUploadChunk() {
  HTTPUpload& up = server.upload();
  if (up.status == UPLOAD_FILE_START) {
    fsUpError = false;
    fsUpName  = up.filename;
    int slash = fsUpName.lastIndexOf('/');
    if (slash >= 0) fsUpName = fsUpName.substring(slash + 1);
    if (fsUpName.length() == 0 || fsUpName.indexOf("..") >= 0) { fsUpError = true; return; }
    fsUpFile = LittleFS.open("/" + fsUpName, "w");
    if (!fsUpFile) fsUpError = true;
  } else if (up.status == UPLOAD_FILE_WRITE) {
    if (fsUpFile && !fsUpError) fsUpFile.write(up.buf, up.currentSize);
  } else if (up.status == UPLOAD_FILE_END) {
    if (fsUpFile) fsUpFile.close();
    if (!fsUpError)
      Serial.printf("[FS] %s → LittleFS (%u bytes)\n",
                    fsUpName.c_str(), (unsigned)up.totalSize);
  } else if (up.status == UPLOAD_FILE_ABORTED) {
    if (fsUpFile) fsUpFile.close();
    fsUpError = true;
  }
}

void handleFSUploadDone() {
  if (fsUpError || fsUpName.length() == 0) {
    server.send(500, "application/json", "{\"ok\":false,\"error\":\"upload_failed\"}");
    return;
  }
  char resp[96];
  snprintf(resp, sizeof(resp), "{\"ok\":true,\"file\":\"%s\"}", fsUpName.c_str());
  server.send(200, "application/json", resp);
}

// ─────────────────────────────────────────────────────────────
// HTTP: OTA firmware via web (/ota) — manfaat partisi dual-OTA 16MB.
// Arduino IDE: Sketch → Export Compiled Binary → upload .ino.bin.
// Gagal otomatis bila partition scheme tanpa slot OTA.
// ─────────────────────────────────────────────────────────────
static bool otaOk = false;

void handleOTAChunk() {
  HTTPUpload& up = server.upload();
  if (up.status == UPLOAD_FILE_START) {
    otaOk = Update.begin(UPDATE_SIZE_UNKNOWN);
    Serial.printf("[OTA] Mulai %s → %s\n", up.filename.c_str(),
                  otaOk ? "ok" : Update.errorString());
  } else if (up.status == UPLOAD_FILE_WRITE) {
    if (otaOk && Update.write(up.buf, up.currentSize) != up.currentSize) {
      otaOk = false;
      Serial.printf("[OTA] Write gagal: %s\n", Update.errorString());
    }
  } else if (up.status == UPLOAD_FILE_END) {
    if (otaOk) otaOk = Update.end(true);
    Serial.printf("[OTA] Selesai: %s (%u bytes)\n",
                  otaOk ? "sukses" : Update.errorString(), (unsigned)up.totalSize);
  } else if (up.status == UPLOAD_FILE_ABORTED) {
    Update.abort();
    otaOk = false;
  }
}

void handleOTADone() {
  if (!otaOk) {
    char resp[160];
    snprintf(resp, sizeof(resp), "{\"ok\":false,\"error\":\"%s\"}", Update.errorString());
    server.send(500, "application/json", resp);
    return;
  }
  server.send(200, "application/json", "{\"ok\":true,\"restarting\":true}");
  vTaskDelay(pdMS_TO_TICKS(800));
  Serial.println("[OTA] Restart ke firmware baru...");
  ESP.restart();
}

// ─────────────────────────────────────────────────────────────
// HTTP: statistik sistem (/sys/info) — untuk tab Sistem di web UI
// ─────────────────────────────────────────────────────────────
void handleSysInfo() {
  const esp_partition_t* run = esp_ota_get_running_partition();
  bool otaReady = esp_ota_get_next_update_partition(NULL) != NULL;

  char resp[512];
  snprintf(resp, sizeof(resp),
    "{\"fw\":\"" FW_VERSION "\",\"built\":\"" __DATE__ " " __TIME__ "\","
    "\"part\":\"%s\",\"ota\":%s,"
    "\"heap_kb\":%u,\"heap_min_kb\":%u,\"heap_total_kb\":%u,"
    "\"psram_kb\":%u,\"psram_free_kb\":%u,"
    "\"lfs_kb\":%u,\"lfs_used_kb\":%u,"
    "\"sd_mb\":%lu,\"sd_used_mb\":%lu,"
    "\"model_loc\":\"%s\",\"arena_loc\":\"%s\","
    "\"rssi\":%d,\"uptime_s\":%lu}",
    run ? run->label : "?", otaReady ? "true" : "false",
    (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024),
    (unsigned)(heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL) / 1024),
    (unsigned)(heap_caps_get_total_size(MALLOC_CAP_INTERNAL) / 1024),
    (unsigned)(ESP.getPsramSize() / 1024),
    (unsigned)(ESP.getFreePsram() / 1024),
    (unsigned)(LittleFS.totalBytes() / 1024),
    (unsigned)(LittleFS.usedBytes() / 1024),
    sdMounted ? (unsigned long)(SD_MMC.totalBytes() / (1024ULL * 1024ULL)) : 0,
    sdMounted ? (unsigned long)(SD_MMC.usedBytes()  / (1024ULL * 1024ULL)) : 0,
    modelLoc, arenaInternal ? "sram" : "psram",
    (int)WiFi.RSSI(), (unsigned long)(millis() / 1000));
  server.send(200, "application/json", resp);
}

// ─────────────────────────────────────────────────────────────
// Log prediksi → CSV di SD card (bukti pengujian untuk laporan TA)
// ─────────────────────────────────────────────────────────────
static void fmtNow(char* out, size_t n) {
  time_t now = time(nullptr);
  if (now > 1700000000) {           // NTP sudah sinkron
    struct tm t;
    localtime_r(&now, &t);
    strftime(out, n, "%Y-%m-%d %H:%M:%S", &t);
  } else {
    snprintf(out, n, "boot+%lus", (unsigned long)(millis() / 1000));
  }
}

static void logPrediction(bool good, float score, uint32_t timeMs) {
  if (!sdMounted) return;
  bool fresh = !SD_MMC.exists(PREDICT_LOG_PATH);
  File f = SD_MMC.open(PREDICT_LOG_PATH, FILE_APPEND);
  if (!f) return;
  if (fresh) f.println("waktu,label,skor,waktu_ms");
  char ts[32];
  fmtNow(ts, sizeof(ts));
  f.printf("%s,%s,%.4f,%lu\n", ts, good ? "BAGUS" : "CACAT",
           score, (unsigned long)timeMs);
  f.close();
}

void handleLog() {
  if (!sdMounted || !SD_MMC.exists(PREDICT_LOG_PATH)) {
    server.send(404, "text/plain", "belum ada log");
    return;
  }
  File f = SD_MMC.open(PREDICT_LOG_PATH, FILE_READ);
  size_t start = 0;
  if (server.hasArg("tail") && f.size() > 16384) start = f.size() - 16384;
  f.seek(start);
  server.setContentLength(f.size() - start);
  server.send(200, "text/csv", "");
  uint8_t buf[1024];
  size_t n;
  while ((n = f.read(buf, sizeof(buf))) > 0) server.client().write(buf, n);
  f.close();
}

void handleLogClear() {
  if (sdMounted) SD_MMC.remove(PREDICT_LOG_PATH);
  server.send(200, "application/json", "{\"ok\":true}");
}

// ─────────────────────────────────────────────────────────────
// HTTP: capture JPEG live — dipakai preview & download dataset
// ─────────────────────────────────────────────────────────────
void handleCapture() {
  if (classifying) { server.send(503, "text/plain", "busy"); return; }
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) { server.send(503, "text/plain", "camera failed"); return; }
  server.sendHeader("Cache-Control", "no-cache, no-store");
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.setContentLength(fb->len);
  server.send(200, "image/jpeg", "");
  server.client().write(fb->buf, fb->len);
  esp_camera_fb_return(fb);
}

// ─────────────────────────────────────────────────────────────
// HTTP: info model
// ─────────────────────────────────────────────────────────────
void handleModelInfo() {
  // Pesan error TFLite bisa mengandung kutip ganda — ganti agar JSON valid
  char err[96];
  strlcpy(err, tfliteErrReporter.last, sizeof(err));
  for (char* p = err; *p; p++) if (*p == '"') *p = '\'';

  char resp[320];
  snprintf(resp, sizeof(resp),
    "{\"loaded\":%s,\"size_kb\":%d,\"arena_kb\":%d,"
    "\"arena_loc\":\"%s\",\"model_loc\":\"%s\",\"err\":\"%s\"}",
    modelLoaded ? "true" : "false",
    (int)modelSizeKB,
    modelLoaded ? (int)(interpreter->arena_used_bytes() / 1024) : 0,
    arenaInternal ? "sram" : "psram", modelLoc, err);
  server.send(200, "application/json", resp);
}

// ─────────────────────────────────────────────────────────────
// Inference worker — dipanggil di Core 0 oleh inferenceTask
//
// Optimasi vs versi sebelumnya:
//   - Tidak ada ganti resolusi kamera → hemat ~160ms
//   - Tidak ada warmup frame → hemat 2×40ms
//   - rgbArena pre-alokasi di PSRAM → zero malloc/free
//   - Scale nearest-neighbor seluruh frame → 96×96
// ─────────────────────────────────────────────────────────────
static void _runInference() {
  InferResult r = {};

  // Ambil frame VGA terbaru (kamera selalu stabil, tidak perlu warmup)
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    snprintf(r.err, sizeof(r.err), "camera_failed");
    inferResult = r; return;
  }
  if ((uint32_t)fb->width * fb->height * 3 > (uint32_t)RGB_MAX_W * RGB_MAX_H * 3) {
    esp_camera_fb_return(fb);
    snprintf(r.err, sizeof(r.err), "frame_too_large");
    inferResult = r; return;
  }

  // Decode JPEG → RGB888 ke pre-alokasi PSRAM buffer
  const bool conv = fmt2rgb888(fb->buf, fb->len, PIXFORMAT_JPEG, rgbArena);
  const int  fbW  = (int)fb->width;
  const int  fbH  = (int)fb->height;
  esp_camera_fb_return(fb);

  if (!conv) {
    snprintf(r.err, sizeof(r.err), "decode_failed");
    inferResult = r; return;
  }

  // Scale nearest-neighbor: fbW×fbH → CROP_SZ×CROP_SZ + kuantisasi INT8
  // q = round(pixel/255 / inScale) + inZP
  const float   inScale = inTensor->params.scale;
  const int     inZP    = inTensor->params.zero_point;
  int8_t* const dst     = inTensor->data.int8;

  for (int dy = 0; dy < CROP_SZ; dy++) {
    const int sy = dy * fbH / CROP_SZ;
    for (int dx = 0; dx < CROP_SZ; dx++) {
      const int sx = dx * fbW / CROP_SZ;
      const int si = (sy * fbW + sx) * 3;
      const int di = (dy * CROP_SZ + dx) * 3;
      for (int c = 0; c < 3; c++) {
        const int q = (int)(rgbArena[si + c] / (inScale * 255.0f) + 0.5f) + inZP;
        dst[di + c] = (int8_t)(q < -128 ? -128 : q > 127 ? 127 : q);
      }
    }
  }

  // TFLite Invoke di Core 0
  const uint32_t t0 = millis();
  if (interpreter->Invoke() != kTfLiteOk) {
    snprintf(r.err, sizeof(r.err), "invoke_failed");
    inferResult = r; return;
  }
  r.time_ms = millis() - t0;

  r.score = (outTensor->data.int8[0] - outTensor->params.zero_point)
            * outTensor->params.scale;
  r.good  = (r.score >= 0.5f);
  r.ok    = true;

  Serial.printf("[Core0] %s  score=%.3f  %lu ms\n",
    r.good ? "BAGUS" : "TIDAK BAGUS", r.score, (unsigned long)r.time_ms);

  // LED hasil: hijau terang = BAGUS, merah = TIDAK BAGUS (1.5 detik)
  if (r.good) ledShow(0, 64, 0, 1500);
  else        ledShow(64, 0, 0, 1500);

  inferResult = r;
}

// ─────────────────────────────────────────────────────────────
// Core 0 — inference task: tunggu trigger, jalankan, kirim done
// ─────────────────────────────────────────────────────────────
void inferenceTask(void* pv) {
  (void)pv;
  while (true) {
    xSemaphoreTake(inferTrigSem, portMAX_DELAY);
    _runInference();
    xSemaphoreGive(inferDoneSem);
  }
}

// ─────────────────────────────────────────────────────────────
// HTTP: jalankan inferensi (Core 1 trigger Core 0, tunggu hasil)
// ─────────────────────────────────────────────────────────────
void handlePredict() {
  if (!modelLoaded) { server.send(503, "application/json", "{\"error\":\"no_model\"}"); return; }
  if (classifying)  { server.send(503, "application/json", "{\"error\":\"busy\"}"); return; }

  classifying = true;
  xSemaphoreGive(inferTrigSem);  // bangunkan Core 0

  // Timeout 5 detik (tidak ada warmup/switch resolusi)
  if (xSemaphoreTake(inferDoneSem, pdMS_TO_TICKS(5000)) != pdTRUE) {
    classifying = false;
    server.send(503, "application/json", "{\"error\":\"timeout\"}");
    return;
  }
  classifying = false;

  if (!inferResult.ok) {
    char resp[80];
    snprintf(resp, sizeof(resp), "{\"error\":\"%s\"}", inferResult.err);
    server.send(503, "application/json", resp);
    return;
  }

  char resp[160];
  snprintf(resp, sizeof(resp),
    "{\"label\":\"%s\",\"score\":%.4f,\"time_ms\":%lu,\"good\":%s}",
    inferResult.good ? "BAGUS" : "TIDAK BAGUS",
    inferResult.score, (unsigned long)inferResult.time_ms,
    inferResult.good ? "true" : "false");
  server.send(200, "application/json", resp);

  // Catat ke log SD setelah response terkirim (tidak menambah latensi)
  logPrediction(inferResult.good, inferResult.score, inferResult.time_ms);
}

// ─────────────────────────────────────────────────────────────
// HTTP: upload model .tflite → LittleFS (primer) + backup ke SD
// → restart otomatis. Bila LittleFS gagal (penuh), fallback ke SD.
// ─────────────────────────────────────────────────────────────
void handleUploadDone() {
  if (uploadFile) uploadFile.close();

  bool inLfs = LittleFS.exists(MODEL_PATH);
  bool inSd  = sdMounted && SD_MMC.exists(MODEL_PATH);
  if (!inLfs && !inSd) {
    server.send(500, "application/json", "{\"ok\":false,\"error\":\"save_failed\"}");
    return;
  }

  // Backup otomatis ke SD — model selamat walau LittleFS ditimpa plugin
  if (inLfs && sdMounted) copyFile(LittleFS, SD_MMC, MODEL_PATH);

  size_t sz = inLfs ? fileSize(LittleFS, MODEL_PATH) : fileSize(SD_MMC, MODEL_PATH);

  char resp[96];
  snprintf(resp, sizeof(resp),
    "{\"ok\":true,\"size_kb\":%d,\"restarting\":true}", (int)(sz / 1024));
  server.send(200, "application/json", resp);

  vTaskDelay(pdMS_TO_TICKS(800));  // beri waktu TCP kirim response
  Serial.println("[INFO] Model diupload, restart...");
  ESP.restart();
}

void handleUploadChunk() {
  HTTPUpload& up = server.upload();
  if (up.status == UPLOAD_FILE_START) {
    uploadFile = LittleFS.open(MODEL_PATH, "w");
    if (!uploadFile && sdMounted)            // LittleFS penuh/error → SD
      uploadFile = SD_MMC.open(MODEL_PATH, "w");
    Serial.printf("[UPLOAD] Start: %s → %s\n", up.filename.c_str(),
                  uploadFile ? "LittleFS" : "GAGAL");
  } else if (up.status == UPLOAD_FILE_WRITE) {
    if (uploadFile) uploadFile.write(up.buf, up.currentSize);
  } else if (up.status == UPLOAD_FILE_END) {
    if (uploadFile) uploadFile.close();
    Serial.printf("[UPLOAD] Selesai: %d bytes\n", (int)up.totalSize);
  }
}

// ─────────────────────────────────────────────────────────────
// Setup
// ─────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  vTaskDelay(pdMS_TO_TICKS(1000));
  Serial.println("\n╔══════════════════════════════════════╗");
  Serial.println("║  EggClassifier V2 — Freenove S3 CAM  ║");
  Serial.println("║  RTOS + PSRAM + SD Optimized         ║");
  Serial.println("╚══════════════════════════════════════╝");
  Serial.printf("[INFO] PSRAM: %lu KB | Heap: %lu KB\n",
    (unsigned long)(ESP.getPsramSize() / 1024),
    (unsigned long)(ESP.getFreeHeap() / 1024));

  if (ESP.getPsramSize() == 0) {
    Serial.println("[ERROR] PSRAM tidak aktif! Tools → PSRAM → OPI PSRAM");
    return;
  }

  pinMode(BOOT_BTN, INPUT_PULLUP);
  neopixelWrite(RGB_LED_PIN, 0, 0, 0);

  // 1. LittleFS
  if (!LittleFS.begin(true)) { Serial.println("[ERROR] LittleFS gagal"); return; }
  Serial.println("[OK] LittleFS");

  // 2. SD card (opsional — tanpa SD, dataset jatuh ke mode download)
  sdMounted = initSD();

  // 2b. Sinkron model LittleFS ↔ SD (pulihkan/backup otomatis)
  syncModelStorage();

  // 3. Kamera — VGA double-buffer, tidak pernah berganti resolusi
  //    (Freenove: kamera selalu berdaya, tanpa AXP313A)
  if (!initCamera()) return;
  loadCameraSettings();   // terapkan setting custom tersimpan di NVS

  // 4. Pre-alokasi PSRAM decode buffer (921 KB) — zero malloc/free saat inferensi
  rgbArena = (uint8_t*) ps_malloc((size_t)RGB_MAX_W * RGB_MAX_H * 3);
  if (!rgbArena) { Serial.println("[ERROR] ps_malloc rgbArena gagal"); return; }
  Serial.printf("[OK] rgbArena: %d KB di PSRAM\n",
    (int)((size_t)RGB_MAX_W * RGB_MAX_H * 3 / 1024));

  // 5. TFLite ops (registrasi satu kali sebelum load model)
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddAveragePool2D();
  resolver.AddMean();      // GlobalAveragePooling2D → MEAN di TF ≥2.x
  resolver.AddPad();       // ZeroPadding2D pada blok stride-2 MobileNetV1
  resolver.AddReshape();
  resolver.AddFullyConnected();
  resolver.AddLogistic();
  resolver.AddQuantize();
  resolver.AddDequantize();
  initTFLite();

  // 6. Dual-core: semaphore + inference task di Core 0 (prio 5)
  inferTrigSem = xSemaphoreCreateBinary();
  inferDoneSem = xSemaphoreCreateBinary();
  xTaskCreatePinnedToCore(
    inferenceTask, "Infer", 16384, NULL, 5, &inferTaskHandle, 0
  );
  Serial.println("[OK] Inference task → Core 0 (prio 5)");

  // 7. WiFi — event-driven reconnect (tidak perlu polling di loop)
  WiFi.onEvent([](WiFiEvent_t event) {
    if (event == ARDUINO_EVENT_WIFI_STA_DISCONNECTED) {
      WiFi.reconnect();
    }
  });
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.printf("[..] Connecting to '%s'", WIFI_SSID);
  for (int t = 0; t < 60 && WiFi.status() != WL_CONNECTED; t++) {
    vTaskDelay(pdMS_TO_TICKS(250));
    updateLED();
    if (t % 2 == 0) Serial.print(".");
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\n[OK] WiFi: http://%s\n", WiFi.localIP().toString().c_str());
    neopixelWrite(RGB_LED_PIN, 0, 12, 0);
  } else {
    Serial.println("\n[WARN] WiFi gagal — akan retry otomatis via event");
  }

  // 7b. NTP (WIB, UTC+7) — timestamp asli untuk log prediksi di SD
  configTime(7 * 3600, 0, "pool.ntp.org", "time.google.com");

  // 8. mDNS
  if (MDNS.begin("telur")) Serial.println("[OK] mDNS: http://telur.local");

  // 9. Routes HTTP
  server.on("/",             HTTP_GET,  handleRoot);
  server.on("/index.html",   HTTP_GET,  handleRoot);
  server.on("/app.js",       HTTP_GET,  handleAppJS);
  server.on("/style.css",    HTTP_GET,  handleCSS);
  server.on("/capture",      HTTP_GET,  handleCapture);
  server.on("/predict",      HTTP_GET,  handlePredict);
  server.on("/model_info",   HTTP_GET,  handleModelInfo);
  server.on("/upload_model", HTTP_POST, handleUploadDone, handleUploadChunk);
  server.on("/sd/info",      HTTP_GET,  handleSDInfo);
  server.on("/sd/capture",   HTTP_POST, handleSDCapture);
  server.on("/sd/list",      HTTP_GET,  handleSDList);
  server.on("/sd/file",      HTTP_GET,  handleSDFile);
  server.on("/sd/delete",    HTTP_POST, handleSDDelete);
  server.on("/sd/relabel",   HTTP_POST, handleSDRelabel);
  server.on("/sd/upload",    HTTP_POST, handleSDUploadDone, handleSDUploadChunk);
  server.on("/camera/get",   HTTP_GET,  handleCamGet);
  server.on("/camera/set",   HTTP_POST, handleCamSet);
  server.on("/camera/reset", HTTP_POST, handleCamReset);
  server.on("/sys/info",     HTTP_GET,  handleSysInfo);
  server.on("/ota",          HTTP_POST, handleOTADone, handleOTAChunk);
  server.on("/fs/upload",    HTTP_POST, handleFSUploadDone, handleFSUploadChunk);
  server.on("/log",          HTTP_GET,  handleLog);
  server.on("/log/clear",    HTTP_POST, handleLogClear);
  server.begin();

  Serial.println("[OK] HTTP server");
  const esp_partition_t* runPart = esp_ota_get_running_partition();
  Serial.printf("[INFO] Partisi app: %s | OTA via web: %s\n\n",
    runPart ? runPart->label : "?",
    esp_ota_get_next_update_partition(NULL)
      ? "siap" : "TIDAK ADA — pakai partition scheme Default 16MB");
  Serial.println("╔══════════════════════════════════════╗");
  Serial.printf( "║  http://telur.local                  ║\n");
  Serial.printf( "║  http://%-30s║\n",
    (WiFi.localIP().toString() + "/").c_str());
  Serial.println("╠══════════════════════════════════════╣");
  Serial.printf( "║  Model  : %-28s║\n",
    modelLoaded ? "Loaded ✓" : "Belum ada — upload via web");
  Serial.printf( "║  PSRAM  : %-28s║\n",
    (String(ESP.getPsramSize() / 1024) + " KB total").c_str());
  Serial.printf( "║  Dataset: %-28s║\n",
    sdMounted ? "SD card (/dataset)" : "Download mode (SD tidak ada)");
  Serial.println("╚══════════════════════════════════════╝\n");
}

// ─────────────────────────────────────────────────────────────
// Loop — Core 1: WebServer + LED. WiFi reconnect via event handler.
// ─────────────────────────────────────────────────────────────
void loop() {
  server.handleClient();
  updateLED();
}
