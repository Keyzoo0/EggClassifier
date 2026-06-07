/*
 * Phase 2 — Data Collection Server
 * Board  : DFRobot FireBeetle 2 ESP32-S3 N16R8 (v1.0)
 *
 * Endpoint HTTP:
 *   GET /         → halaman preview live (auto-refresh di browser)
 *   GET /capture  → satu frame JPEG (dipakai Python script)
 *
 * Library yang harus terinstall:
 *   - DFRobot_AXP313A
 *
 * Arduino IDE Board Settings: sama seperti Phase 1
 *   Board: ESP32S3 Dev Module | Flash: 16MB | PSRAM: Disabled
 *   USB CDC On Boot: Enabled | CPU: 240MHz
 */

#include <Wire.h>
#include <WiFi.h>
#include <WebServer.h>
#include "DFRobot_AXP313A.h"
#include "esp_camera.h"

// === GANTI SESUAI JARINGAN ANDA ===
#define WIFI_SSID "YOUR_SSID"
#define WIFI_PASS "YOUR_PASSWORD"
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

// Halaman preview live — gambar di-refresh tiap 800ms via JavaScript
const char PAGE_HTML[] = R"(<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Egg Classifier - Preview</title>
  <style>
    body { background:#1a1a1a; color:#eee; font-family:sans-serif;
           text-align:center; padding:30px; margin:0; }
    h2   { color:#4CAF50; margin-bottom:6px; }
    img  { max-width:640px; width:95%; border:3px solid #444;
           border-radius:6px; margin:16px 0; }
    p    { color:#888; font-size:14px; }
    span { color:#4CAF50; font-weight:bold; }
  </style>
</head>
<body>
  <h2>Egg Classifier — Live Preview</h2>
  <p>Gunakan Python script untuk capture dataset</p>
  <img id="cam" src="/capture" alt="camera feed">
  <p>Frame: <span id="cnt">0</span> | Status: <span id="status">OK</span></p>
  <script>
    var cnt = 0;
    function refresh() {
      var img = new Image();
      img.onload = function() {
        document.getElementById('cam').src = this.src;
        document.getElementById('cnt').textContent = ++cnt;
        document.getElementById('status').textContent = 'OK';
      };
      img.onerror = function() {
        document.getElementById('status').textContent = 'ERROR';
      };
      img.src = '/capture?' + Date.now();
    }
    setInterval(refresh, 800);
  </script>
</body>
</html>)";

void handleRoot() {
  server.send(200, "text/html", PAGE_HTML);
}

void handleCapture() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    server.send(503, "text/plain", "Capture failed");
    return;
  }

  server.sendHeader("Cache-Control", "no-cache, no-store, must-revalidate");
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.setContentLength(fb->len);
  server.send(200, "image/jpeg", "");
  server.client().write(fb->buf, fb->len);

  esp_camera_fb_return(fb);
}

bool initCamera() {
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
  config.frame_size    = FRAMESIZE_QVGA;  // 320x240 — sama dgn resolusi inferensi
  config.jpeg_quality  = 8;               // kualitas lebih tinggi untuk dataset
  config.fb_count      = 1;
  config.fb_location   = CAMERA_FB_IN_DRAM;
  config.grab_mode     = CAMERA_GRAB_LATEST;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[ERROR] Camera init gagal: 0x%x\n", err);
    return false;
  }
  return true;
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n=== FireBeetle 2 ESP32-S3 — Phase 2 Data Collection ===");

  // AXP313A — wajib untuk board v1.0
  Wire.begin(CAM_PIN_SIOD, CAM_PIN_SIOC);
  if (axp.begin() != 0) {
    Serial.println("[ERROR] AXP313A tidak terdeteksi!");
    return;
  }
  axp.enableCameraPower(axp.eOV2640);
  delay(100);
  Serial.println("[OK] AXP313A: camera power on");

  // Camera
  if (!initCamera()) {
    Serial.println("[ERROR] Camera init gagal");
    return;
  }
  delay(300);
  Serial.println("[OK] Camera ready");

  // WiFi
  Serial.printf("[..] Connecting to '%s'", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  int timeout = 0;
  while (WiFi.status() != WL_CONNECTED && timeout < 30) {
    delay(500);
    Serial.print(".");
    timeout++;
  }

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("\n[ERROR] WiFi gagal terhubung. Cek SSID/Password.");
    return;
  }

  Serial.println("\n[OK] WiFi terhubung");
  Serial.printf("\n╔══════════════════════════════════╗\n");
  Serial.printf("║  Preview : http://%-15s║\n", (WiFi.localIP().toString() + "/").c_str());
  Serial.printf("║  Capture : http://%-15s║\n", (WiFi.localIP().toString() + "/capture").c_str());
  Serial.printf("╚══════════════════════════════════╝\n\n");
  Serial.printf("Jalankan di PC:\n");
  Serial.printf("  python training/collect_data.py --ip %s\n\n", WiFi.localIP().toString().c_str());

  // HTTP server
  server.on("/", handleRoot);
  server.on("/capture", handleCapture);
  server.begin();
  Serial.println("[OK] HTTP server running. Menunggu koneksi...");
}

void loop() {
  server.handleClient();
}
