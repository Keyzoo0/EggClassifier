/*
 * EggClassifier V2 — Firmware Terpadu
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
 * SD Card Wiring:
 *   VCC → 3.3V  |  GND → GND
 *   SCK  → GPIO 17  (label board: SCK)
 *   MOSI → GPIO 15  (label board: MO)
 *   MISO → GPIO 16  (label board: MI)
 *   CS   → GPIO 10  (label board: D10, kanan bawah)
 *
 * Endpoints:
 *   GET  /              → Web UI
 *   GET  /capture       → JPEG frame live
 *   GET  /sd_stats      → JSON: {good, bad, free_mb, total_mb}
 *   GET  /save_image    → query: ?label=good|bad → simpan ke SD, return JSON
 *   GET  /predict       → inference → JSON hasil
 *   GET  /model_info    → JSON: {loaded, size_kb}
 *   POST /upload_model  → terima .tflite → simpan LittleFS → restart
 */

#include <Wire.h>
#include <WiFi.h>
#include <WebServer.h>
#include <ESPmDNS.h>
#include <LittleFS.h>
#include <SD.h>
#include <SPI.h>
#include "DFRobot_AXP313A.h"
#include "esp_camera.h"
#include "img_converters.h"
#include "TensorFlowLite_ESP32.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ── Credentials ───────────────────────────────────────────────
#define WIFI_SSID  "Warmindo Samndut Suhat. 4G"
#define WIFI_PASS  "Tanyakasir"

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

// ── Pin SD Card (SPI2/HSPI) ───────────────────────────────────
#define SD_CS    10
#define SD_SCK   17
#define SD_MOSI  15
#define SD_MISO  16

// ── Pin Board ─────────────────────────────────────────────────
#define LED_PIN   21   // Onboard LED: solid=WiFi OK, blink=reconnecting
#define BOOT_BTN   0

// ── TFLite ────────────────────────────────────────────────────
#define ARENA_SIZE (200 * 1024)  // 200 KB dari PSRAM

// ── Inferensi: QQVGA center crop ke 96×96 ─────────────────────
#define INFER_W  160
#define INFER_H  120
#define CROP_SZ   96
#define CROP_X   ((INFER_W - CROP_SZ) / 2)   // 32
#define CROP_Y   ((INFER_H - CROP_SZ) / 2)   // 12

// ── Global ────────────────────────────────────────────────────
DFRobot_AXP313A axp;
WebServer        server(80);
SPIClass         spiSD(HSPI);

// TFLite
static uint8_t                          tensorArena[ARENA_SIZE] __attribute__((aligned(16)));
static tflite::MicroMutableOpResolver<10> resolver;
static tflite::MicroInterpreter*          interpreter = nullptr;
static TfLiteTensor*                      inTensor    = nullptr;
static TfLiteTensor*                      outTensor   = nullptr;
bool   modelLoaded   = false;
size_t modelSizeKB   = 0;

// SD
bool sdReady      = false;
int  sdGoodCount  = 0;
int  sdBadCount   = 0;

// Flags
bool     classifying  = false;
bool     pendingRestart = false;
uint32_t restartAt    = 0;

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
// WiFi auto-reconnect
// ─────────────────────────────────────────────────────────────
void checkWiFi() {
  static uint32_t lastCheck = 0;
  if (millis() - lastCheck < 5000) return;
  lastCheck = millis();
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WIFI] Putus, mencoba reconnect...");
    WiFi.reconnect();
  }
}

// ─────────────────────────────────────────────────────────────
// Camera
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
  cfg.grab_mode    = CAMERA_GRAB_LATEST;

  bool hasPSRAM = (ESP.getPsramSize() > 0);
  cfg.frame_size   = hasPSRAM ? FRAMESIZE_VGA  : FRAMESIZE_QVGA;
  cfg.jpeg_quality = hasPSRAM ? 6              : 8;
  cfg.fb_count     = hasPSRAM ? 2              : 1;
  cfg.fb_location  = hasPSRAM ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;

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
  Serial.printf("[OK] Camera: %s JPEG\n", hasPSRAM ? "VGA 640x480" : "QVGA 320x240");
  return true;
}

// ─────────────────────────────────────────────────────────────
// SD Card
// ─────────────────────────────────────────────────────────────
int countFilesInDir(const char* path) {
  File dir = SD.open(path);
  if (!dir || !dir.isDirectory()) return 0;
  int n = 0;
  File f = dir.openNextFile();
  while (f) { if (!f.isDirectory()) n++; f = dir.openNextFile(); }
  return n;
}

bool initSD() {
  spiSD.begin(SD_SCK, SD_MISO, SD_MOSI, SD_CS);
  if (!SD.begin(SD_CS, spiSD)) {
    Serial.println("[WARN] SD card tidak ditemukan");
    return false;
  }
  SD.mkdir("/dataset");
  SD.mkdir("/dataset/good");
  SD.mkdir("/dataset/bad");
  sdGoodCount = countFilesInDir("/dataset/good");
  sdBadCount  = countFilesInDir("/dataset/bad");
  Serial.printf("[OK] SD card: %d good, %d bad | %.0f MB bebas\n",
    sdGoodCount, sdBadCount,
    (float)(SD.totalBytes() - SD.usedBytes()) / 1024 / 1024);
  return true;
}

// ─────────────────────────────────────────────────────────────
// TFLite: load dari LittleFS → PSRAM
// ─────────────────────────────────────────────────────────────
bool initTFLite() {
  if (!LittleFS.exists("/egg_model.tflite")) {
    Serial.println("[TF] Tidak ada model di LittleFS");
    return false;
  }

  File f = LittleFS.open("/egg_model.tflite", "r");
  size_t sz = f.size();
  modelSizeKB = sz / 1024;

  uint8_t* buf = (uint8_t*) ps_malloc(sz);
  if (!buf) { f.close(); Serial.println("[TF] ps_malloc gagal"); return false; }

  f.read(buf, sz);
  f.close();

  const tflite::Model* model = tflite::GetModel(buf);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("[TF] Schema version mismatch");
    free(buf); return false;
  }

  static tflite::MicroInterpreter interp(model, resolver, tensorArena, ARENA_SIZE);
  interpreter = &interp;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("[TF] AllocateTensors gagal — coba perbesar ARENA_SIZE");
    return false;
  }

  inTensor  = interpreter->input(0);
  outTensor = interpreter->output(0);
  modelLoaded = true;

  Serial.printf("[OK] TFLite: %d KB | arena: %d KB | input [%d,%d,%d,%d]\n",
    (int)modelSizeKB,
    (int)(interpreter->arena_used_bytes() / 1024),
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

// ─────────────────────────────────────────────────────────────
// HTTP: capture JPEG live
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
// HTTP: simpan foto ke SD card
// GET /save_image?label=good|bad
// ─────────────────────────────────────────────────────────────
void handleSaveImage() {
  if (!sdReady) {
    server.send(503, "application/json", "{\"error\":\"sd_not_ready\"}");
    return;
  }
  String label = server.arg("label");
  if (label != "good" && label != "bad") {
    server.send(400, "application/json", "{\"error\":\"label harus good atau bad\"}");
    return;
  }

  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) { server.send(503, "application/json", "{\"error\":\"camera_failed\"}"); return; }

  int* cnt = (label == "good") ? &sdGoodCount : &sdBadCount;
  (*cnt)++;

  char path[64];
  snprintf(path, sizeof(path), "/dataset/%s/%s_%04d.jpg",
           label.c_str(), label.c_str(), *cnt);

  File f = SD.open(path, FILE_WRITE);
  bool ok = false;
  if (f) { f.write(fb->buf, fb->len); f.close(); ok = true; }
  esp_camera_fb_return(fb);

  if (ok) {
    char resp[128];
    snprintf(resp, sizeof(resp),
      "{\"ok\":true,\"filename\":\"%s_%04d.jpg\",\"count\":%d}",
      label.c_str(), *cnt, *cnt);
    server.send(200, "application/json", resp);
  } else {
    (*cnt)--;
    server.send(500, "application/json", "{\"error\":\"sd_write_failed\"}");
  }
}

// ─────────────────────────────────────────────────────────────
// HTTP: statistik SD card
// ─────────────────────────────────────────────────────────────
void handleSDStats() {
  char resp[192];
  if (sdReady) {
    snprintf(resp, sizeof(resp),
      "{\"good\":%d,\"bad\":%d,\"free_mb\":%llu,\"total_mb\":%llu,\"ready\":true}",
      sdGoodCount, sdBadCount,
      (unsigned long long)((SD.totalBytes() - SD.usedBytes()) / 1024 / 1024),
      (unsigned long long)(SD.totalBytes() / 1024 / 1024));
  } else {
    snprintf(resp, sizeof(resp),
      "{\"good\":0,\"bad\":0,\"free_mb\":0,\"total_mb\":0,\"ready\":false}");
  }
  server.send(200, "application/json", resp);
}

// ─────────────────────────────────────────────────────────────
// HTTP: info model
// ─────────────────────────────────────────────────────────────
void handleModelInfo() {
  char resp[96];
  snprintf(resp, sizeof(resp),
    "{\"loaded\":%s,\"size_kb\":%d}",
    modelLoaded ? "true" : "false", (int)modelSizeKB);
  server.send(200, "application/json", resp);
}

// ─────────────────────────────────────────────────────────────
// HTTP: inferensi
// ─────────────────────────────────────────────────────────────
void handlePredict() {
  if (!modelLoaded) {
    server.send(503, "application/json", "{\"error\":\"no_model\"}");
    return;
  }
  classifying = true;

  // Switch ke QQVGA untuk hemat memori saat decode
  sensor_t* s = esp_camera_sensor_get();
  s->set_framesize(s, FRAMESIZE_QQVGA);
  delay(80);

  // Buang 2 frame awal agar AWB stabil
  for (int i = 0; i < 2; i++) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (fb) esp_camera_fb_return(fb);
    delay(40);
  }

  camera_fb_t* fb = esp_camera_fb_get();
  s->set_framesize(s, FRAMESIZE_VGA);  // kembalikan resolusi ASAP

  if (!fb) {
    classifying = false;
    server.send(503, "application/json", "{\"error\":\"camera_failed\"}");
    return;
  }

  // Decode JPEG → RGB888 di PSRAM
  uint8_t* rgb = (uint8_t*) ps_malloc(INFER_W * INFER_H * 3);
  if (!rgb) {
    esp_camera_fb_return(fb);
    classifying = false;
    server.send(503, "application/json", "{\"error\":\"psram_full\"}");
    return;
  }

  bool conv = fmt2rgb888(fb->buf, fb->len, PIXFORMAT_JPEG, rgb);
  esp_camera_fb_return(fb);

  if (!conv) {
    free(rgb);
    classifying = false;
    server.send(503, "application/json", "{\"error\":\"decode_failed\"}");
    return;
  }

  // Preprocess: center crop 96×96 → INT8 input tensor
  float  inScale = inTensor->params.scale;
  int    inZP    = inTensor->params.zero_point;
  int8_t* dst    = inTensor->data.int8;

  for (int y = 0; y < CROP_SZ; y++) {
    for (int x = 0; x < CROP_SZ; x++) {
      int src = ((CROP_Y + y) * INFER_W + (CROP_X + x)) * 3;
      int out = (y * CROP_SZ + x) * 3;
      for (int c = 0; c < 3; c++) {
        float f = rgb[src + c] / 255.0f;
        dst[out + c] = (int8_t) constrain((int)roundf(f / inScale) + inZP, -128, 127);
      }
    }
  }
  free(rgb);

  // Inferensi
  uint32_t t0 = millis();
  TfLiteStatus st = interpreter->Invoke();
  uint32_t dt = millis() - t0;
  classifying = false;

  if (st != kTfLiteOk) {
    server.send(503, "application/json", "{\"error\":\"invoke_failed\"}");
    return;
  }

  float score = (outTensor->data.int8[0] - outTensor->params.zero_point)
                * outTensor->params.scale;
  bool isGood = (score >= 0.5f);

  char resp[160];
  snprintf(resp, sizeof(resp),
    "{\"label\":\"%s\",\"score\":%.4f,\"time_ms\":%lu,\"good\":%s}",
    isGood ? "BAGUS" : "TIDAK BAGUS", score, (unsigned long)dt,
    isGood ? "true" : "false");
  server.send(200, "application/json", resp);

  Serial.printf("[PREDICT] %s  score=%.3f  %lu ms\n",
    isGood ? "BAGUS" : "TIDAK BAGUS", score, (unsigned long)dt);
}

// ─────────────────────────────────────────────────────────────
// HTTP: upload model .tflite → LittleFS → restart
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

  // Flush TCP buffer lalu restart
  server.client().flush();
  delay(500);
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
  delay(1000);
  Serial.println("\n╔══════════════════════════════════════╗");
  Serial.println("║    EggClassifier V2 — FireBeetle 2   ║");
  Serial.println("╚══════════════════════════════════════╝");
  Serial.printf("[INFO] PSRAM: %lu KB | Heap: %lu KB\n",
    (unsigned long)(ESP.getPsramSize() / 1024),
    (unsigned long)(ESP.getFreeHeap() / 1024));

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
  delay(100);
  Serial.println("[OK] AXP313A");

  // 3. Kamera
  if (!initCamera()) return;

  // 4. SD Card
  sdReady = initSD();

  // 5. TFLite (ops registrasi satu kali)
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddAveragePool2D();
  resolver.AddReshape();
  resolver.AddFullyConnected();
  resolver.AddLogistic();
  resolver.AddQuantize();
  resolver.AddDequantize();
  initTFLite();

  // 6. WiFi
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.printf("[..] Connecting to '%s'", WIFI_SSID);
  for (int t = 0; t < 30 && WiFi.status() != WL_CONNECTED; t++) {
    delay(500); Serial.print(".");
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\n[OK] WiFi: http://%s\n", WiFi.localIP().toString().c_str());
    digitalWrite(LED_PIN, HIGH);
  } else {
    Serial.println("\n[WARN] WiFi gagal — akan retry di loop");
  }

  // 7. mDNS
  if (MDNS.begin("telur")) Serial.println("[OK] mDNS: http://telur.local");

  // 8. Routes HTTP
  server.on("/",             HTTP_GET,  handleRoot);
  server.on("/index.html",   HTTP_GET,  handleRoot);
  server.on("/app.js",       HTTP_GET,  handleAppJS);
  server.on("/capture",      HTTP_GET,  handleCapture);
  server.on("/save_image",   HTTP_GET,  handleSaveImage);
  server.on("/sd_stats",     HTTP_GET,  handleSDStats);
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
  Serial.printf( "║  Model : %-29s║\n", modelLoaded ? "Loaded ✓" : "Belum ada — upload via web");
  Serial.printf( "║  SD    : %-29s║\n", sdReady ? "Ready ✓" : "Tidak terhubung ✗");
  Serial.println("╚══════════════════════════════════════╝\n");
}

// ─────────────────────────────────────────────────────────────
// Loop
// ─────────────────────────────────────────────────────────────
void loop() {
  server.handleClient();
  updateLED();
  checkWiFi();
}
