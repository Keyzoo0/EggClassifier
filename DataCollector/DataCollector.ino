/*
 * Phase 2 — Data Collection Server
 * Board  : DFRobot FireBeetle 2 ESP32-S3 N16R8 (v1.0)
 *
 * Akses web:
 *   http://telur.local       ← mDNS (Windows butuh Bonjour/iTunes)
 *   http://<IP>              ← langsung via IP (lebih andal)
 *
 * Endpoint:
 *   GET /          → halaman dashboard (dari LittleFS)
 *   GET /app.js    → JavaScript (dari LittleFS)
 *   GET /capture   → satu frame JPEG live
 *
 * Library yang harus terinstall:
 *   - DFRobot_AXP313A
 *
 * Upload filesystem (WAJIB sebelum pertama kali pakai):
 *   1. Install plugin: https://github.com/earlephilhower/arduino-littlefs-upload
 *      (Arduino IDE 2.x: letakkan .vsix di folder plugins, restart IDE)
 *   2. Tools → ESP32 LittleFS Data Upload
 *   3. Baru Upload sketch seperti biasa
 *
 * Arduino IDE Board Settings:
 *   Board            : ESP32S3 Dev Module
 *   Flash Size       : 16MB (128Mb)
 *   Partition Scheme : Default 4MB with spiffs
 *   PSRAM            : OPI PSRAM   ← ubah dari Disabled ke OPI PSRAM!
 *   CPU Frequency    : 240MHz
 *   USB CDC On Boot  : Enabled
 *   Upload Speed     : 921600
 *
 * Dengan OPI PSRAM aktif:
 *   - Resolusi dataset naik ke VGA 640x480 (kualitas lebih baik untuk training)
 *   - Frame buffer di PSRAM, DRAM bebas untuk WiFi + WebServer
 *   - Double buffering untuk preview lebih smooth
 */

#include <Wire.h>
#include <WiFi.h>
#include <WebServer.h>
#include <ESPmDNS.h>
#include <LittleFS.h>
#include "DFRobot_AXP313A.h"
#include "esp_camera.h"

// === GANTI SESUAI JARINGAN ANDA ===
#define WIFI_SSID "sahrul"
#define WIFI_PASS "12345678"
// ==================================

// Pin mapping FireBeetle 2 ESP32-S3 N16R8
#define CAM_PIN_PWDN    -1
#define CAM_PIN_RESET   -1
#define CAM_PIN_XCLK    45
#define CAM_PIN_SIOD     1
#define CAM_PIN_SIOC     2
#define CAM_PIN_D7      48
#define CAM_PIN_D6      46
#define CAM_PIN_D5       8
#define CAM_PIN_D4       7
#define CAM_PIN_D3       4
#define CAM_PIN_D2      41
#define CAM_PIN_D1      40
#define CAM_PIN_D0      39
#define CAM_PIN_VSYNC    6
#define CAM_PIN_HREF    42
#define CAM_PIN_PCLK     5

DFRobot_AXP313A axp;
WebServer server(80);

// ── Filesystem helpers ────────────────────────────────────────
String getContentType(const String& path) {
  if (path.endsWith(".html")) return "text/html";
  if (path.endsWith(".js"))   return "application/javascript";
  if (path.endsWith(".css"))  return "text/css";
  if (path.endsWith(".ico"))  return "image/x-icon";
  return "text/plain";
}

void serveFile(const String& path) {
  if (!LittleFS.exists(path)) {
    server.send(404, "text/plain", "File tidak ditemukan: " + path);
    return;
  }
  File f = LittleFS.open(path, "r");
  server.streamFile(f, getContentType(path));
  f.close();
}

// ── HTTP Handlers ─────────────────────────────────────────────
void handleRoot()    { serveFile("/index.html"); }
void handleAppJS()   { serveFile("/app.js"); }

void handleCapture() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    server.send(503, "text/plain", "Camera capture gagal");
    return;
  }
  server.sendHeader("Cache-Control", "no-cache, no-store, must-revalidate");
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.setContentLength(fb->len);
  server.send(200, "image/jpeg", "");
  server.client().write(fb->buf, fb->len);
  esp_camera_fb_return(fb);
}

// ── Camera Init ───────────────────────────────────────────────
bool initCamera() {
  bool hasPSRAM = (ESP.getPsramSize() > 0);

  camera_config_t config;
  config.ledc_channel  = LEDC_CHANNEL_0;
  config.ledc_timer    = LEDC_TIMER_0;
  config.pin_d0        = CAM_PIN_D0;
  config.pin_d1        = CAM_PIN_D1;
  config.pin_d2        = CAM_PIN_D2;
  config.pin_d3        = CAM_PIN_D3;
  config.pin_d4        = CAM_PIN_D4;
  config.pin_d5        = CAM_PIN_D5;
  config.pin_d6        = CAM_PIN_D6;
  config.pin_d7        = CAM_PIN_D7;
  config.pin_xclk      = CAM_PIN_XCLK;
  config.pin_pclk      = CAM_PIN_PCLK;
  config.pin_vsync     = CAM_PIN_VSYNC;
  config.pin_href      = CAM_PIN_HREF;
  config.pin_sccb_sda  = CAM_PIN_SIOD;
  config.pin_sccb_scl  = CAM_PIN_SIOC;
  config.pin_pwdn      = CAM_PIN_PWDN;
  config.pin_reset     = CAM_PIN_RESET;
  config.xclk_freq_hz  = 20000000;
  config.pixel_format  = PIXFORMAT_JPEG;
  config.grab_mode     = CAMERA_GRAB_LATEST;

  if (hasPSRAM) {
    // PSRAM tersedia: resolusi tinggi + double buffer
    config.frame_size   = FRAMESIZE_VGA;        // 640x480
    config.jpeg_quality = 6;                    // kualitas tinggi (0-63, makin kecil makin bagus)
    config.fb_count     = 2;                    // double buffering → preview lebih smooth
    config.fb_location  = CAMERA_FB_IN_PSRAM;
    Serial.printf("[OK] Camera mode: VGA 640x480, quality=6, 2 buffer (PSRAM)\n");
  } else {
    // Fallback tanpa PSRAM
    config.frame_size   = FRAMESIZE_QVGA;       // 320x240
    config.jpeg_quality = 8;
    config.fb_count     = 1;
    config.fb_location  = CAMERA_FB_IN_DRAM;
    Serial.printf("[OK] Camera mode: QVGA 320x240, quality=8, 1 buffer (DRAM)\n");
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[ERROR] Camera init gagal: 0x%x\n", err);
    return false;
  }
  return true;
}

// ── Setup ─────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n=== FireBeetle 2 ESP32-S3 — Phase 2 Data Collection ===");
  Serial.printf("[INFO] Free DRAM : %lu KB\n", (unsigned long)(ESP.getFreeHeap() / 1024));
  Serial.printf("[INFO] PSRAM     : %lu KB%s\n",
                (unsigned long)(ESP.getPsramSize() / 1024),
                ESP.getPsramSize() > 0 ? " ✓" : " (tidak aktif — set Tools→PSRAM→OPI PSRAM)");

  // 1. LittleFS
  if (!LittleFS.begin(true)) {
    Serial.println("[ERROR] LittleFS gagal mount. Jalankan 'ESP32 LittleFS Data Upload' dulu.");
    return;
  }
  Serial.println("[OK] LittleFS mounted");

  // 2. AXP313A — wajib untuk board v1.0
  Wire.begin(CAM_PIN_SIOD, CAM_PIN_SIOC);
  if (axp.begin() != 0) {
    Serial.println("[ERROR] AXP313A tidak terdeteksi!");
    return;
  }
  axp.enableCameraPower(axp.eOV2640);
  delay(100);
  Serial.println("[OK] AXP313A: camera power on");

  // 3. Camera
  if (!initCamera()) {
    Serial.println("[ERROR] Camera init gagal");
    return;
  }
  delay(300);
  Serial.println("[OK] Camera ready");

  // 4. WiFi
  Serial.printf("[..] Connecting to '%s'", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  int t = 0;
  while (WiFi.status() != WL_CONNECTED && t++ < 30) {
    delay(500);
    Serial.print(".");
  }
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("\n[ERROR] WiFi gagal. Cek SSID/Password.");
    return;
  }
  Serial.println("\n[OK] WiFi terhubung");

  // 5. mDNS
  if (MDNS.begin("telur")) {
    Serial.println("[OK] mDNS: http://telur.local");
  } else {
    Serial.println("[WARN] mDNS gagal start");
  }

  // 6. HTTP Server
  server.on("/",          handleRoot);
  server.on("/index.html",handleRoot);
  server.on("/app.js",    handleAppJS);
  server.on("/capture",   handleCapture);
  server.begin();

  Serial.println("[OK] HTTP server running\n");
  Serial.println("╔══════════════════════════════════════╗");
  Serial.printf( "║  http://telur.local                  ║\n");
  Serial.printf( "║  http://%-30s║\n", (WiFi.localIP().toString() + "/").c_str());
  Serial.println("╚══════════════════════════════════════╝");
}

// ── Loop ──────────────────────────────────────────────────────
void loop() {  
  server.handleClient();
}
