#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_chip_info.h"
#include "esp_camera.h"

static const char *TAG = "CAM_TEST";

// --- Pin mapping FireBeetle 2 ESP32-S3 N16R8 ---
#define CAM_PIN_PWDN    -1
#define CAM_PIN_RESET   -1
#define CAM_PIN_XCLK    45
#define CAM_PIN_SIOD     1   // SDA
#define CAM_PIN_SIOC     2   // SCL
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

static void print_memory_info(void)
{
    esp_chip_info_t chip;
    esp_chip_info(&chip);

    ESP_LOGI(TAG, "---------- Memory Info ----------");
    ESP_LOGI(TAG, "Chip        : ESP32-S3, %d core(s)", chip.cores);
    ESP_LOGI(TAG, "PSRAM total : %lu KB",
             (unsigned long)(heap_caps_get_total_size(MALLOC_CAP_SPIRAM) / 1024));
    ESP_LOGI(TAG, "PSRAM free  : %lu KB",
             (unsigned long)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));
    ESP_LOGI(TAG, "SRAM free   : %lu KB",
             (unsigned long)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024));
    ESP_LOGI(TAG, "---------------------------------");

    if (heap_caps_get_total_size(MALLOC_CAP_SPIRAM) == 0) {
        ESP_LOGE(TAG, "PSRAM tidak terdeteksi!");
        ESP_LOGE(TAG, "Periksa sdkconfig.defaults: CONFIG_SPIRAM=y dan CONFIG_SPIRAM_MODE_OCT=y");
    }
}

static esp_err_t camera_init_test(void)
{
    camera_config_t config = {
        .pin_pwdn     = CAM_PIN_PWDN,
        .pin_reset    = CAM_PIN_RESET,
        .pin_xclk     = CAM_PIN_XCLK,
        .pin_sccb_sda = CAM_PIN_SIOD,
        .pin_sccb_scl = CAM_PIN_SIOC,
        .pin_d7       = CAM_PIN_D7,
        .pin_d6       = CAM_PIN_D6,
        .pin_d5       = CAM_PIN_D5,
        .pin_d4       = CAM_PIN_D4,
        .pin_d3       = CAM_PIN_D3,
        .pin_d2       = CAM_PIN_D2,
        .pin_d1       = CAM_PIN_D1,
        .pin_d0       = CAM_PIN_D0,
        .pin_vsync    = CAM_PIN_VSYNC,
        .pin_href     = CAM_PIN_HREF,
        .pin_pclk     = CAM_PIN_PCLK,

        .xclk_freq_hz  = 20000000,
        .ledc_timer    = LEDC_TIMER_0,
        .ledc_channel  = LEDC_CHANNEL_0,

        .pixel_format  = PIXFORMAT_JPEG,
        .frame_size    = FRAMESIZE_QVGA,   // 320x240
        .jpeg_quality  = 12,
        .fb_count      = 1,
        // PSRAM tidak tersedia di board ini -> frame buffer di DRAM internal
        .fb_location   = CAMERA_FB_IN_DRAM,
        .grab_mode     = CAMERA_GRAB_LATEST,
    };

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_camera_init gagal: 0x%x", err);
        ESP_LOGE(TAG, "Kemungkinan penyebab:");
        ESP_LOGE(TAG, "  - Kabel FPC kamera belum terpasang");
        ESP_LOGE(TAG, "  - Pin mapping salah");
        ESP_LOGE(TAG, "  - PSRAM belum aktif (frame buffer dialokasi ke PSRAM)");
    }
    return err;
}

static void capture_test_frames(void)
{
    // Tunggu AWB OV2640 stabil (known quirk)
    vTaskDelay(pdMS_TO_TICKS(300));

    ESP_LOGI(TAG, "Mulai capture 5 frame...");

    int success = 0;
    for (int i = 0; i < 5; i++) {
        int64_t t0 = esp_timer_get_time();
        camera_fb_t *fb = esp_camera_fb_get();
        int64_t t1 = esp_timer_get_time();

        if (!fb) {
            ESP_LOGE(TAG, "Frame %d: GAGAL", i + 1);
            continue;
        }

        ESP_LOGI(TAG, "Frame %d: %dx%d, %zu bytes, %.1f ms",
                 i + 1,
                 fb->width, fb->height,
                 fb->len,
                 (float)(t1 - t0) / 1000.0f);

        esp_camera_fb_return(fb);
        success++;
        vTaskDelay(pdMS_TO_TICKS(200));
    }

    ESP_LOGI(TAG, "Hasil: %d/5 frame berhasil", success);

    if (success == 5) {
        ESP_LOGI(TAG, "=== PHASE 1 PASSED: Kamera berfungsi normal ===");
    } else {
        ESP_LOGE(TAG, "=== PHASE 1 FAILED: Ada masalah pada kamera ===");
    }
}

void app_main(void)
{
    // Beri waktu USB-CDC re-enumerate agar log awal tidak hilang di monitor
    vTaskDelay(pdMS_TO_TICKS(2000));

    ESP_LOGI(TAG, "=== FireBeetle 2 ESP32-S3 N16R8 — Camera Test ===");

    print_memory_info();

    ESP_LOGI(TAG, "Inisialisasi kamera...");
    esp_err_t cam_ok = camera_init_test();
    if (cam_ok == ESP_OK) {
        ESP_LOGI(TAG, "Kamera berhasil diinisialisasi");
        capture_test_frames();
    }

    // Loop status agar serial monitor bisa connect kapan saja
    while (1) {
        ESP_LOGI(TAG, "[heartbeat] PSRAM free=%lu KB, SRAM free=%lu KB, camera=%s",
                 (unsigned long)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024),
                 (unsigned long)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024),
                 cam_ok == ESP_OK ? "OK" : "FAIL");
        vTaskDelay(pdMS_TO_TICKS(3000));
    }
}
