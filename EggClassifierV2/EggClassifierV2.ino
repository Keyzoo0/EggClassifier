/*
 * EggClassifier V2 — Firmware Terpadu (RTOS + PSRAM Optimized)
 * Board  : Freenove ESP32-S3-WROOM CAM (FNK0085, N16R8) + microSD 4GB
 *
 * Library (Tools → Manage Libraries):
 *   1. TensorFlowLite_ESP32 (by tanakamasayuki)
 *   (AXP313A tidak diperlukan — kamera Freenove selalu berdaya)
 *
 * Board Settings (Tools):
 *   Board            : ESP32S3 Dev Module
 *   Flash Size       : 16MB (128Mb)   ← board ini varian N16R8 (esptool detect 16MB)
 *   Partition Scheme : Huge APP (3MB No OTA/1MB SPIFFS)
 *   PSRAM            : OPI PSRAM
 *   CPU Frequency    : 240MHz
 *   USB CDC On Boot  : Enabled
 *
 * SD card: microSD 4GB, format FAT32 (allocation unit 16K),
 *   SDMMC 1-bit — CMD=38, CLK=39, D0=40. Dataset di /dataset/.
 *
 * Arsitektur dual-core:
 *   Core 0 (prio 5) — inferenceTask: TFLite Invoke
 *   Core 1 (prio 1) — loop(): WebServer.handleClient + LED + SD I/O
 *
 * PRIORITAS #1 — web bisa diakses client (REACHABILITY, bukan kecepatan):
 *   - WiFi power-save DEFAULT (modem sleep ON) → arus lebih rendah; mematikannya
 *     bikin brownout pada catu USB pas-pasan → paket IP hilang (device connect
 *     tapi tak terjangkau). Gejala utama: associate OK tapi ping/TCP 100% loss.
 *     → Akar paling sering = DAYA. Pakai catu 5V kuat + kabel bagus.
 *   - loop() tidak busy-spin (delay 2ms) agar task WiFi/lwIP dapat CPU
 *   - Reconnect pakai auto bawaan stack, mDNS diumumkan ulang tiap dapat IP
 *   - WebServer sinkron 1 koneksi → web UI (app.js) mengantre request
 *   - Model dimuat sebelum kamera agar buffer model dapat PSRAM tak terfragmentasi
 *
 * PSRAM layout (~1.4 MB dari 8 MB tersedia):
 *   rgbArena    640×480×3 = 921 KB  — decode JPEG→RGB888 (pre-alokasi)
 *   tensorArena 200 KB              — TFLite working area
 *   modelBuf    ~315 KB             — flat buffer model .tflite
 *
 * Endpoints:
 *   GET  /              → Web UI (LittleFS)
 *   GET  /capture       → JPEG frame live (preview & download dataset)
 *   GET  /predict       → inference → JSON hasil
 *   GET  /model_info    → JSON: {loaded, size_kb, arena_kb, err}
 *   POST /upload_model  → .tflite → SD card (fallback LittleFS) → restart
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
 *   GET  /flash/get     → JSON status flash: {on, bri}
 *   POST /flash/set     → ?on=1&bri=200 → flash kamera (8 LED RGB GPIO47)
 */

#include <WiFi.h>
#include <WebServer.h>
#include <ESPmDNS.h>
#include <LittleFS.h>
#include <SD_MMC.h>
#include <Preferences.h>
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
#include <Adafruit_NeoPixel.h>   // flash kamera: 8 LED RGB di GPIO47

// ── Credentials ───────────────────────────────────────────────
#define WIFI_SSID  "ROSI1"
#define WIFI_PASS  "20517420"

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

// ── Flash kamera — strip 8 LED WS2812 RGB di GPIO47 ──────────
#define FLASH_PIN   47
#define FLASH_NUM    8

// ── Pin SD card (SDMMC 1-bit, fixed di PCB Freenove) ─────────
#define SD_PIN_CMD  38
#define SD_PIN_CLK  39
#define SD_PIN_D0   40
#define SD_DATASET_DIR "/dataset"

// ── TFLite ────────────────────────────────────────────────────
#define ARENA_SIZE (200 * 1024)   // 200 KB dari PSRAM

// ── Kamera & Inferensi ────────────────────────────────────────
// Camera QVGA (320×240) — model hanya 96×96, jadi VGA = pemborosan murni:
// frame JPEG VGA ~35KB memblokir server sinkron lama saat dikirim, dan butuh
// rgbArena 900KB. QVGA: frame ~10KB (preview gesit, server tak lama terblokir),
// rgbArena 230KB, decode lebih cepat. Tetap di-scale nearest-neighbor → 96×96.
#define RGB_MAX_W  320
#define RGB_MAX_H  240
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
static uint8_t*                           tensorArena = nullptr;
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
// Flash kamera — strip 8 LED WS2812 RGB di GPIO47 (penerangan)
//   On = semua LED putih pada tingkat kecerahan flashBri (0–255).
//   Kecerahan tinggi = arus besar (8×~60mA bisa ~480mA penuh) →
//   bisa memicu brownout; default sedang. Kecerahan disimpan di NVS.
// ─────────────────────────────────────────────────────────────
Adafruit_NeoPixel flashStrip(FLASH_NUM, FLASH_PIN, NEO_GRB + NEO_KHZ800);
bool    flashOn  = false;
uint8_t flashBri = 96;   // 0–255

void applyFlash() {
  flashStrip.setBrightness(flashBri);
  uint32_t c = flashOn ? flashStrip.Color(255, 255, 255) : 0;
  for (int i = 0; i < FLASH_NUM; i++) flashStrip.setPixelColor(i, c);
  flashStrip.show();
}

// Isi LED satu per satu dengan satu warna (efek wipe)
void flashColorWipe(uint32_t color, int wait) {
  for (int i = 0; i < FLASH_NUM; i++) {
    flashStrip.setPixelColor(i, color);
    flashStrip.show();
    delay(wait);
  }
}

// Animasi awal setup: colorWipe merah → hijau → biru, lalu mati
void flashBootIntro() {
  flashStrip.setBrightness(50);   // 50 cukup terang & tidak menyilaukan
  flashColorWipe(flashStrip.Color(255, 0, 0), 50);   // merah
  flashColorWipe(flashStrip.Color(0, 255, 0), 50);   // hijau
  flashColorWipe(flashStrip.Color(0, 0, 255), 50);   // biru
  flashStrip.clear();
  flashStrip.show();
}

// Tanda setup selesai: kedip putih 2× sebelum masuk loop
void flashBootReady() {
  flashStrip.setBrightness(60);
  for (int k = 0; k < 2; k++) {
    for (int i = 0; i < FLASH_NUM; i++)
      flashStrip.setPixelColor(i, flashStrip.Color(255, 255, 255));
    flashStrip.show();
    delay(150);
    flashStrip.clear();
    flashStrip.show();
    delay(150);
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
  cfg.frame_size   = FRAMESIZE_QVGA;      // 320×240 — cukup utk model 96×96
  cfg.jpeg_quality = 12;  // 0=terbaik; q12 di QVGA → frame ~8-10KB (preview gesit)
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
  Serial.println("[OK] Camera: QVGA 320×240 JPEG double-buffer PSRAM");
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
// HTTP: flash kamera — status & set on/off + kecerahan
//   GET  /flash/get               → {on, bri}
//   POST /flash/set?on=1&bri=200  → set (kedua arg opsional)
// Kecerahan disimpan permanen (NVS), status on/off TIDAK (default off
// saat boot agar tidak langsung menarik arus besar).
// ─────────────────────────────────────────────────────────────
void handleFlashGet() {
  char resp[48];
  snprintf(resp, sizeof(resp), "{\"on\":%s,\"bri\":%d}",
           flashOn ? "true" : "false", flashBri);
  server.send(200, "application/json", resp);
}

void handleFlashSet() {
  if (server.hasArg("on"))  flashOn = (server.arg("on").toInt() != 0);
  if (server.hasArg("bri")) {
    int b = server.arg("bri").toInt();
    flashBri = b < 0 ? 0 : (b > 255 ? 255 : b);
    camPrefs.begin("camcfg", false);
    camPrefs.putUChar("flbri", flashBri);   // kecerahan persisten
    camPrefs.end();
  }
  applyFlash();
  handleFlashGet();
}

// ─────────────────────────────────────────────────────────────
// TFLite: load model → PSRAM
// Model disimpan di SD card (tahan terhadap upload ulang LittleFS
// yang menimpa seluruh partisi); LittleFS hanya fallback tanpa SD.
// ─────────────────────────────────────────────────────────────
#define MODEL_PATH "/egg_model.tflite"
#define MIN_MODEL_SIZE 1024   // file < 1KB pasti bukan .tflite valid (mis. 0 byte
                              //   sisa deploy gagal) — jangan dicoba dimuat
static fs::FS& modelFS() { return sdMounted ? (fs::FS&)SD_MMC : (fs::FS&)LittleFS; }

// Buka model dari satu FS hanya bila ukurannya wajar (bukan kosong/rusak)
static File openValidModel(fs::FS& fs, const char* label) {
  File f;
  if (!fs.exists(MODEL_PATH)) return f;
  f = fs.open(MODEL_PATH, "r");
  if (f && f.size() >= MIN_MODEL_SIZE) {
    Serial.printf("[TF] Model dari %s (%u KB)\n", label, (unsigned)(f.size() / 1024));
    return f;
  }
  Serial.printf("[TF] Model di %s kosong/rusak (%u B) — dilewati\n",
                label, (unsigned)(f ? f.size() : 0));
  if (f) f.close();
  return File();
}

bool initTFLite() {
  // SD dulu (sumber utama), jatuh ke LittleFS bila SD kosong/rusak/tak ada.
  // File 0-byte di SD tidak lagi menutupi model valid di LittleFS.
  File f;
  if (sdMounted) f = openValidModel(SD_MMC, "SD card");
  if (!f)        f = openValidModel(LittleFS, "LittleFS");
  if (!f) {
    Serial.println("[TF] Tidak ada model valid (SD/LittleFS) — pasang via tab Training");
    snprintf(tfliteErrReporter.last, sizeof(tfliteErrReporter.last),
             "model belum dipasang / file kosong");
    return false;
  }
  size_t sz = f.size();
  modelSizeKB = sz / 1024;

  uint8_t* buf = (uint8_t*) ps_malloc(sz);
  if (!buf) {
    f.close();
    // Ungkap penyebab: kalau 'minta' wajar (~315KB) tapi 'alloc-maks' kecil →
    // PSRAM terfragmentasi (load model setelah kamera). Kalau 'minta' raksasa →
    // file model di SD rusak/ukuran salah.
    snprintf(tfliteErrReporter.last, sizeof(tfliteErrReporter.last),
             "ps_malloc %uKB gagal (PSRAM bebas %uKB, blok-maks %uKB)",
             (unsigned)(sz / 1024), (unsigned)(ESP.getFreePsram() / 1024),
             (unsigned)(ESP.getMaxAllocPsram() / 1024));
    Serial.printf("[TF] %s\n", tfliteErrReporter.last);
    return false;
  }
  f.read(buf, sz);
  f.close();

  const tflite::Model* model = tflite::GetModel(buf);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    snprintf(tfliteErrReporter.last, sizeof(tfliteErrReporter.last),
             "schema v%d != v%d", (int)model->version(), TFLITE_SCHEMA_VERSION);
    Serial.printf("[TF] %s\n", tfliteErrReporter.last);
    free(buf); return false;
  }

  if (!tensorArena) {
    tensorArena = (uint8_t*) ps_malloc(ARENA_SIZE);
    if (!tensorArena) {
      snprintf(tfliteErrReporter.last, sizeof(tfliteErrReporter.last),
               "ps_malloc arena gagal");
      free(buf); Serial.println("[TF] ps_malloc arena gagal"); return false;
    }
  }
  memset(tensorArena, 0, ARENA_SIZE);

  static tflite::MicroInterpreter interp(
    model, resolver, tensorArena, ARENA_SIZE, &tfliteErrReporter
  );
  interpreter = &interp;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    // Detail (mis. op tidak terdaftar) sudah tertangkap di errReporter.last
    Serial.println("[TF] AllocateTensors gagal — cek op resolver / ARENA_SIZE");
    return false;
  }
  tfliteErrReporter.last[0] = '\0';   // sukses — bersihkan error lama

  inTensor  = interpreter->input(0);
  outTensor = interpreter->output(0);
  modelLoaded = true;

  Serial.printf("[OK] TFLite: %d KB | arena: %d KB | input [%d,%d,%d,%d]\n",
    (int)modelSizeKB, (int)(interpreter->arena_used_bytes() / 1024),
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
// HTTP: file statik dari LittleFS (web UI di-host di alat, tanpa CDN)
// Satu handler generik (dipasang sebagai onNotFound) menyajikan semua
// aset — index.html, app.js, style.css, tw.css, chart.min.js — jadi
// menambah file UI baru tidak perlu ubah firmware.
// ─────────────────────────────────────────────────────────────
static const char* mimeFor(const String& p) {
  if (p.endsWith(".html")) return "text/html";
  if (p.endsWith(".js"))   return "application/javascript";
  if (p.endsWith(".css"))  return "text/css";
  if (p.endsWith(".json")) return "application/json";
  if (p.endsWith(".svg"))  return "image/svg+xml";
  if (p.endsWith(".png"))  return "image/png";
  if (p.endsWith(".jpg") || p.endsWith(".jpeg")) return "image/jpeg";
  if (p.endsWith(".ico"))  return "image/x-icon";
  return "application/octet-stream";
}

void handleStatic() {
  String path = server.uri();
  if (path == "/") path = "/index.html";
  if (path.indexOf("..") >= 0) { server.send(400, "text/plain", "bad path"); return; }
  if (!LittleFS.exists(path)) { server.send(404, "text/plain", "Not found"); return; }

  // chart.min.js = vendor besar & stabil → boleh di-cache lama browser
  // (load berikutnya instan). File UI lain TIDAK di-cache agar perubahan
  // langsung terlihat setelah upload ulang LittleFS.
  if (path == "/chart.min.js") server.sendHeader("Cache-Control", "max-age=604800, immutable");
  else                         server.sendHeader("Cache-Control", "no-cache");

  File f = LittleFS.open(path, "r");
  server.streamFile(f, mimeFor(path));
  f.close();
}

// ─────────────────────────────────────────────────────────────
// HTTP: capture JPEG live — dipakai preview & download dataset
// ─────────────────────────────────────────────────────────────
// Timing capture terakhir (untuk /sys) — ungkap di mana server terblokir:
//   grab = lama esp_camera_fb_get (Core1), send = lama kirim frame ke TCP
volatile uint32_t capGrabMs = 0, capSendMs = 0, capLen = 0;

void handleCapture() {
  if (classifying) { server.send(503, "text/plain", "busy"); return; }
  uint32_t t0 = millis();
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) { server.send(503, "text/plain", "camera failed"); return; }
  uint32_t t1 = millis();
  server.sendHeader("Cache-Control", "no-cache, no-store");
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.setContentLength(fb->len);
  server.send(200, "image/jpeg", "");
  server.client().write(fb->buf, fb->len);
  capGrabMs = t1 - t0;
  capSendMs = millis() - t1;
  capLen    = fb->len;
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

  char resp[256];
  snprintf(resp, sizeof(resp),
    "{\"loaded\":%s,\"size_kb\":%d,\"arena_kb\":%d,\"err\":\"%s\"}",
    modelLoaded ? "true" : "false",
    (int)modelSizeKB,
    modelLoaded ? (int)(interpreter->arena_used_bytes() / 1024) : 0,
    err);
  server.send(200, "application/json", resp);
}

// ─────────────────────────────────────────────────────────────
// HTTP: diagnostik sistem — RAM, PSRAM, RSSI, uptime, timing capture
// Untuk debug performa: lihat heap menyusut/leak, RSSI lemah, atau capture
// (grab/send) yang lama memblokir server.
// ─────────────────────────────────────────────────────────────
void handleSys() {
  char resp[320];
  snprintf(resp, sizeof(resp),
    "{\"heap_free\":%u,\"heap_min\":%u,\"heap_maxblk\":%u,"
    "\"psram_free\":%u,\"psram_maxblk\":%u,"
    "\"rssi\":%d,\"uptime_s\":%lu,"
    "\"cap_grab_ms\":%u,\"cap_send_ms\":%u,\"cap_len\":%u}",
    (unsigned)ESP.getFreeHeap(), (unsigned)ESP.getMinFreeHeap(),
    (unsigned)ESP.getMaxAllocHeap(),
    (unsigned)ESP.getFreePsram(), (unsigned)ESP.getMaxAllocPsram(),
    (int)WiFi.RSSI(), (unsigned long)(millis() / 1000),
    (unsigned)capGrabMs, (unsigned)capSendMs, (unsigned)capLen);
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
}

// ─────────────────────────────────────────────────────────────
// HTTP: upload model .tflite → LittleFS → restart otomatis
// ─────────────────────────────────────────────────────────────
void handleUploadDone() {
  if (uploadFile) uploadFile.close();

  if (!modelFS().exists(MODEL_PATH)) {
    server.send(500, "application/json", "{\"ok\":false,\"error\":\"save_failed\"}");
    return;
  }
  File f  = modelFS().open(MODEL_PATH, "r");
  size_t sz = f.size(); f.close();

  // Tolak file kosong/terlalu kecil (mis. unduhan model gagal) — JANGAN restart
  // dengan model rusak; hapus agar tidak menutupi model lama yang valid.
  if (sz < MIN_MODEL_SIZE) {
    modelFS().remove(MODEL_PATH);
    Serial.printf("[UPLOAD] Ditolak: model hanya %u B (rusak)\n", (unsigned)sz);
    server.send(400, "application/json", "{\"ok\":false,\"error\":\"model_too_small\"}");
    return;
  }

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
    Serial.printf("[UPLOAD] Start: %s → %s\n", up.filename.c_str(),
                  sdMounted ? "SD card" : "LittleFS");
    uploadFile = modelFS().open(MODEL_PATH, "w");
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

  // Flash kamera: init strip + animasi awal (colorWipe merah→hijau→biru)
  flashStrip.begin();
  flashBootIntro();
  camPrefs.begin("camcfg", true);
  if (camPrefs.isKey("flbri")) flashBri = camPrefs.getUChar("flbri", flashBri);
  camPrefs.end();
  flashOn = false;   // strip dikendalikan web; mati dulu (kedip putih di akhir setup)

  // 1. LittleFS
  if (!LittleFS.begin(true)) { Serial.println("[ERROR] LittleFS gagal"); return; }
  Serial.println("[OK] LittleFS");

  // 2. SD card (opsional — tanpa SD, dataset jatuh ke mode download)
  sdMounted = initSD();

  // 3. TFLite — muat model DULU (sebelum kamera) agar buffer model + arena
  //    dapat blok PSRAM yang masih segar. Bila dialokasikan setelah kamera +
  //    rgbArena, PSRAM bisa terfragmentasi → ps_malloc model gagal walau total
  //    bebas masih banyak.
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

  // 4. Kamera — VGA double-buffer, tidak pernah berganti resolusi
  //    (Freenove: kamera selalu berdaya, tanpa AXP313A)
  if (!initCamera()) return;
  loadCameraSettings();   // terapkan setting custom tersimpan di NVS

  // 5. Pre-alokasi PSRAM decode buffer (921 KB) — zero malloc/free saat inferensi
  rgbArena = (uint8_t*) ps_malloc((size_t)RGB_MAX_W * RGB_MAX_H * 3);
  if (!rgbArena) { Serial.println("[ERROR] ps_malloc rgbArena gagal"); return; }
  Serial.printf("[OK] rgbArena: %d KB di PSRAM\n",
    (int)((size_t)RGB_MAX_W * RGB_MAX_H * 3 / 1024));

  // 6. Dual-core: semaphore + inference task di Core 0 (prio 5)
  inferTrigSem = xSemaphoreCreateBinary();
  inferDoneSem = xSemaphoreCreateBinary();
  xTaskCreatePinnedToCore(
    inferenceTask, "Infer", 16384, NULL, 5, &inferTaskHandle, 0
  );
  Serial.println("[OK] Inference task → Core 0 (prio 5)");

  // 7. WiFi — kombinasi yang menang: modem sleep OFF (RTT cepat, tanpa tunggu
  //    DTIM ~100ms tiap round-trip → throughput naik) DIGABUNG TX power rendah
  //    (lonjakan arus TX kecil → tidak brownout). Brownout dulu terjadi karena
  //    sleep-off DENGAN TX power PENUH; di sini TX sudah 13dBm. setSleep(false)
  //    dipasang di GOT_IP (setelah terhubung, bukan saat asosiasi).
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.onEvent([](WiFiEvent_t event) {
    if (event == ARDUINO_EVENT_WIFI_STA_GOT_IP) {
      WiFi.setSleep(false);   // radio nyala terus → tak ada jeda DTIM per paket
      Serial.printf("[WiFi] IP: %s\n", WiFi.localIP().toString().c_str());
      // IP DHCP bisa berubah antar boot — umumkan ulang telur.local
      MDNS.end();
      if (MDNS.begin("telur")) Serial.println("[WiFi] mDNS: http://telur.local");
    }
  });
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  // Turunkan TX power dari ~19.5dBm → 13dBm: lonjakan arus saat transmit jauh
  // berkurang (sumber utama brownout di board kamera). Aman karena AP dekat/kuat.
  // Kalau sinyal jadi lemah/putus, naikkan ke WIFI_POWER_15dBm atau _17dBm.
  WiFi.setTxPower(WIFI_POWER_13dBm);
  Serial.printf("[..] Connecting to '%s'", WIFI_SSID);
  for (int t = 0; t < 80 && WiFi.status() != WL_CONNECTED; t++) {
    vTaskDelay(pdMS_TO_TICKS(250));
    updateLED();
    if (t % 2 == 0) Serial.print(".");
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\n[OK] WiFi: http://%s\n", WiFi.localIP().toString().c_str());
    neopixelWrite(RGB_LED_PIN, 0, 12, 0);
  } else {
    Serial.println("\n[WARN] WiFi gagal connect — stack auto-reconnect jalan terus");
  }

  // Catatan: dicoba arahkan alokasi WebServer ke PSRAM (heap_caps_malloc_extmem_
  // enable) — terukur TIDAK ada bedanya (heap internal ~193KB tetap, loop 333Hz),
  // karena alokasi WebServer cuma beberapa KB sesaat. PSRAM lebih lambat & bisa
  // perlambat stack jaringan, jadi alokasi dibiarkan di RAM internal yang cepat.

  // 8. Routes HTTP — aset statik (/, *.html, *.js, *.css) lewat onNotFound
  server.on("/capture",      HTTP_GET,  handleCapture);
  server.on("/predict",      HTTP_GET,  handlePredict);
  server.on("/model_info",   HTTP_GET,  handleModelInfo);
  server.on("/sys",          HTTP_GET,  handleSys);
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
  server.on("/flash/get",    HTTP_GET,  handleFlashGet);
  server.on("/flash/set",    HTTP_POST, handleFlashSet);
  server.onNotFound(handleStatic);   // aset web UI dari LittleFS
  server.begin();

  Serial.println("[OK] HTTP server\n");
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

  // Setup selesai → kedip putih 2× sebagai tanda siap, lalu kembalikan
  // strip ke kendali web (mati, kecerahan dari NVS), baru masuk loop().
  flashBootReady();
  applyFlash();
}

// ─────────────────────────────────────────────────────────────
// Loop — Core 1: WebServer + LED. WiFi (re)connect ditangani stack.
// ─────────────────────────────────────────────────────────────
void loop() {
  server.handleClient();
  updateLED();

  // Hitung loop-rate (kalau drop drastis = ada yang memblokir/macet)
  static uint32_t loopCnt = 0, lastHb = 0;
  loopCnt++;
  if (millis() - lastHb >= 3000) {
    uint32_t hz = loopCnt * 1000 / (millis() - lastHb);
    lastHb = millis();
    loopCnt = 0;
    // Heartbeat debug: heap (leak?), RSSI (RF lemah?), status WiFi, loop-rate.
    // Tetap tercetak walau WiFi tak terjangkau dari LAN.
    Serial.printf("[HB] up=%lus loop=%luHz heap=%uKB min=%uKB psram=%uKB "
                  "wifi=%d RSSI=%ddBm IP=%s\n",
      (unsigned long)(millis() / 1000), (unsigned long)hz,
      (unsigned)(ESP.getFreeHeap() / 1024), (unsigned)(ESP.getMinFreeHeap() / 1024),
      (unsigned)(ESP.getFreePsram() / 1024),
      (int)WiFi.status(), (int)WiFi.RSSI(),
      WiFi.localIP().toString().c_str());
  }

  delay(2);   // jangan busy-spin Core 1 — beri napas ke task sistem (WiFi/lwIP,
              // IDLE) & turunkan konsumsi daya/panas; HTTP tetap responsif
}
