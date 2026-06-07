/*
 * Phase 1 — Camera Test
 * Board  : DFRobot FireBeetle 2 ESP32-S3 N16R8
 * Tujuan : Verifikasi kamera OV2640 berfungsi sebelum lanjut ke training
 *
 * Arduino IDE Board Settings:
 *   Board           : ESP32S3 Dev Module
 *   Flash Size      : 16MB (128Mb)
 *   Partition Scheme: Default 4MB with spiffs (cukup untuk Phase 1)
 *   PSRAM           : Disabled
 *   CPU Frequency   : 240MHz
 *   USB CDC On Boot : Enabled  ← WAJIB agar Serial muncul di monitor
 *   Upload Speed    : 921600
 */

#include "esp_camera.h"

// --- Pin Mapping FireBeetle 2 ESP32-S3 N16R8 ---
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
  config.frame_size    = FRAMESIZE_QVGA;  // 320x240
  config.jpeg_quality  = 12;
  config.fb_count      = 1;
  config.fb_location   = CAMERA_FB_IN_DRAM; // PSRAM dinonaktifkan
  config.grab_mode     = CAMERA_GRAB_LATEST;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[ERROR] Camera init gagal: 0x%x\n", err);
    Serial.println("  Kemungkinan penyebab:");
    Serial.println("  - Kabel FPC kamera belum terpasang");
    Serial.println("  - USB CDC On Boot belum di-Enable di board settings");
    return false;
  }
  return true;
}

void setup() {
  Serial.begin(115200);
  delay(2000); // beri waktu Serial monitor terhubung

  Serial.println("\n=== FireBeetle 2 ESP32-S3 N16R8 — Phase 1 Camera Test ===");
  Serial.printf("Free heap : %lu bytes\n", (unsigned long)ESP.getFreeHeap());
  Serial.printf("PSRAM size: %lu bytes\n", (unsigned long)ESP.getPsramSize());

  Serial.println("\nInisialisasi kamera...");

  if (!initCamera()) {
    Serial.println("\n[GAGAL] Tidak bisa lanjut. Periksa koneksi dan board settings.");
    return;
  }

  delay(300); // tunggu AWB OV2640 stabil

  Serial.println("[OK] Kamera siap!\n");
  Serial.println("Mulai capture 5 frame test...");
  Serial.println("-------------------------------------------");

  int berhasil = 0;
  for (int i = 1; i <= 5; i++) {
    unsigned long t0 = micros();
    camera_fb_t* fb = esp_camera_fb_get();
    unsigned long durasi = micros() - t0;

    if (!fb) {
      Serial.printf("Frame %d : GAGAL\n", i);
      continue;
    }

    Serial.printf("Frame %d : %dx%d | %5u bytes | %5.1f ms\n",
                  i, fb->width, fb->height, fb->len, durasi / 1000.0f);

    esp_camera_fb_return(fb);
    berhasil++;
    delay(300);
  }

  Serial.println("-------------------------------------------");
  Serial.printf("Hasil    : %d/5 frame berhasil\n\n", berhasil);

  if (berhasil == 5) {
    Serial.println("=== PHASE 1 PASSED: Kamera berfungsi normal ===");
    Serial.println("Lanjut ke Phase 2: pengumpulan dataset.");
  } else {
    Serial.println("=== PHASE 1 FAILED: Periksa koneksi kamera ===");
  }
}

void loop() {
  // Phase 1 hanya test sekali di setup()
  delay(10000);
}
