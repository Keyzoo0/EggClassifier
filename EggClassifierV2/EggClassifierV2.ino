/*
 * EggClassifier V2 — Firmware Terpadu (RTOS + PSRAM Optimized)
 * Board  : DFRobot FireBeetle 2 ESP32-S3 N16R8 v1.0
 *
 * Library (Tools → Manage Libraries):
 *   1. DFRobot_AXP313A
 *   2. TensorFlowLite_ESP32 (by tanakamasayuki)
 *
 * Board Settings (Tools):
 *   Board            : ESP32S3 Dev Module
 *   Flash Size       : 16MB (128Mb)
 *   Partition Scheme : Huge APP (3MB No OTA/1MB SPIFFS)  ← WAJIB
 *   PSRAM            : OPI PSRAM
 *   CPU Frequency    : 240MHz
 *   USB CDC On Boot  : Enabled
 *
 * Arsitektur dual-core:
 *   Core 0 (prio 5) — inferenceTask: TFLite Invoke
 *   Core 1 (prio 1) — loop(): WebServer.handleClient + LED
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
 *   GET  /model_info    → JSON: {loaded, size_kb, arena_kb}
 *   POST /upload_model  → .tflite → LittleFS → restart
 */

#include <Wire.h>
#include <WiFi.h>
#include <WebServer.h>
#include <ESPmDNS.h>
#include <LittleFS.h>
#include "DFRobot_AXP313A.h"
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

// ── Pin Camera ────────────────────────────────────────────────
#define CAM_PWDN   -1
#define CAM_RESET  -1
#define CAM_XCLK   45
#define CAM_SIOD    1
#define CAM_SIOC    2
#define CAM_D7     48
#define CAM_D6     46
#define CAM_D5      8
#define CAM_D4      7
#define CAM_D3      4
#define CAM_D2     41
#define CAM_D1     40
#define CAM_D0     39
#define CAM_VSYNC   6
#define CAM_HREF   42
#define CAM_PCLK    5

// ── Pin Board ─────────────────────────────────────────────────
#define LED_PIN   21   // solid=WiFi OK, blink=reconnecting
#define BOOT_BTN   0

// ── TFLite ────────────────────────────────────────────────────
#define ARENA_SIZE (200 * 1024)   // 200 KB dari PSRAM

// ── Kamera & Inferensi ────────────────────────────────────────
// Camera selalu VGA; tidak pernah ganti resolusi saat inferensi.
// Frame VGA di-scale nearest-neighbor → 96×96 di PSRAM buffer.
#define RGB_MAX_W  640
#define RGB_MAX_H  480
#define CROP_SZ     96

// ── Global ────────────────────────────────────────────────────
DFRobot_AXP313A axp;
WebServer        server(80);

// TFLite
static uint8_t*                           tensorArena = nullptr;
static tflite::MicroMutableOpResolver<10> resolver;
static tflite::MicroErrorReporter        tfliteErrReporter;
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
// LED WiFi indicator (non-blocking)
// ─────────────────────────────────────────────────────────────
void updateLED() {
  static uint32_t lastBlink = 0;
  if (WiFi.status() == WL_CONNECTED) {
    digitalWrite(LED_PIN, HIGH);
  } else {
    if (millis() - lastBlink >= 250) {
      lastBlink = millis();
      digitalWrite(LED_PIN, !digitalRead(LED_PIN));
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
// TFLite: load model dari LittleFS → PSRAM
// ─────────────────────────────────────────────────────────────
bool initTFLite() {
  if (!LittleFS.exists("/egg_model.tflite")) {
    Serial.println("[TF] Tidak ada model di LittleFS");
    return false;
  }

  File f  = LittleFS.open("/egg_model.tflite", "r");
  size_t sz = f.size();
  modelSizeKB = sz / 1024;

  uint8_t* buf = (uint8_t*) ps_malloc(sz);
  if (!buf) { f.close(); Serial.println("[TF] ps_malloc model gagal"); return false; }
  f.read(buf, sz);
  f.close();

  const tflite::Model* model = tflite::GetModel(buf);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("[TF] Schema version mismatch");
    free(buf); return false;
  }

  if (!tensorArena) {
    tensorArena = (uint8_t*) ps_malloc(ARENA_SIZE);
    if (!tensorArena) { free(buf); Serial.println("[TF] ps_malloc arena gagal"); return false; }
  }
  memset(tensorArena, 0, ARENA_SIZE);

  static tflite::MicroInterpreter interp(
    model, resolver, tensorArena, ARENA_SIZE, &tfliteErrReporter
  );
  interpreter = &interp;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("[TF] AllocateTensors gagal — coba perbesar ARENA_SIZE");
    return false;
  }

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
// HTTP: file dari LittleFS
// ─────────────────────────────────────────────────────────────
void serveFile(const char* path, const char* mime) {
  if (!LittleFS.exists(path)) { server.send(404, "text/plain", "Not found"); return; }
  File f = LittleFS.open(path, "r");
  server.streamFile(f, mime);
  f.close();
}
void handleRoot()  { serveFile("/index.html", "text/html"); }
void handleAppJS() { serveFile("/app.js",     "application/javascript"); }
void handleCSS()   { serveFile("/style.css",  "text/css"); }

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
  char resp[128];
  snprintf(resp, sizeof(resp),
    "{\"loaded\":%s,\"size_kb\":%d,\"arena_kb\":%d}",
    modelLoaded ? "true" : "false",
    (int)modelSizeKB,
    modelLoaded ? (int)(interpreter->arena_used_bytes() / 1024) : 0);
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

  if (!LittleFS.exists("/egg_model.tflite")) {
    server.send(500, "application/json", "{\"ok\":false,\"error\":\"save_failed\"}");
    return;
  }
  File f  = LittleFS.open("/egg_model.tflite", "r");
  size_t sz = f.size(); f.close();

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
    Serial.printf("[UPLOAD] Start: %s\n", up.filename.c_str());
    uploadFile = LittleFS.open("/egg_model.tflite", "w");
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
  Serial.println("║    EggClassifier V2 — FireBeetle 2   ║");
  Serial.println("║    RTOS + PSRAM Optimized            ║");
  Serial.println("╚══════════════════════════════════════╝");
  Serial.printf("[INFO] PSRAM: %lu KB | Heap: %lu KB\n",
    (unsigned long)(ESP.getPsramSize() / 1024),
    (unsigned long)(ESP.getFreeHeap() / 1024));

  if (ESP.getPsramSize() == 0) {
    Serial.println("[ERROR] PSRAM tidak aktif! Tools → PSRAM → OPI PSRAM");
    return;
  }

  pinMode(LED_PIN, OUTPUT);
  pinMode(BOOT_BTN, INPUT_PULLUP);
  digitalWrite(LED_PIN, LOW);

  // 1. LittleFS
  if (!LittleFS.begin(true)) { Serial.println("[ERROR] LittleFS gagal"); return; }
  Serial.println("[OK] LittleFS");

  // 2. AXP313A (power kamera)
  Wire.begin(CAM_SIOD, CAM_SIOC);
  if (axp.begin() != 0) { Serial.println("[ERROR] AXP313A tidak ditemukan"); return; }
  axp.enableCameraPower(axp.eOV2640);
  vTaskDelay(pdMS_TO_TICKS(100));
  Serial.println("[OK] AXP313A");

  // 3. Kamera — VGA double-buffer, tidak pernah berganti resolusi
  if (!initCamera()) return;

  // 4. Pre-alokasi PSRAM decode buffer (921 KB) — zero malloc/free saat inferensi
  rgbArena = (uint8_t*) ps_malloc((size_t)RGB_MAX_W * RGB_MAX_H * 3);
  if (!rgbArena) { Serial.println("[ERROR] ps_malloc rgbArena gagal"); return; }
  Serial.printf("[OK] rgbArena: %d KB di PSRAM\n",
    (int)((size_t)RGB_MAX_W * RGB_MAX_H * 3 / 1024));

  // 5. TFLite ops (registrasi satu kali sebelum load model)
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddAveragePool2D();
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
    digitalWrite(LED_PIN, HIGH);
  } else {
    Serial.println("\n[WARN] WiFi gagal — akan retry otomatis via event");
  }

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
  Serial.printf( "║  Dataset: %-28s║\n", "Download mode (via browser)");
  Serial.println("╚══════════════════════════════════════╝\n");
}

// ─────────────────────────────────────────────────────────────
// Loop — Core 1: WebServer + LED. WiFi reconnect via event handler.
// ─────────────────────────────────────────────────────────────
void loop() {
  server.handleClient();
  updateLED();
}
