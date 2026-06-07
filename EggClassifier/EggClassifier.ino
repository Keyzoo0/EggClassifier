/*
 * Phase 5 — Egg Classification Inference
 * Board  : DFRobot FireBeetle 2 ESP32-S3 N16R8 v1.0
 * Model  : MobileNetV1 α=0.25, 96×96, INT8
 *
 * Library yang harus diinstall (Tools → Manage Libraries):
 *   1. DFRobot_AXP313A
 *   2. Cari "TensorFlow" → install "TensorFlowLite_ESP32" by tanakamasayuki
 *
 * Arduino IDE Board Settings:
 *   Board            : ESP32S3 Dev Module
 *   Flash Size       : 16MB (128Mb)
 *   Partition Scheme : Huge APP (3MB No OTA/1MB SPIFFS)  ← wajib, model 315KB
 *   PSRAM            : OPI PSRAM
 *   CPU Frequency    : 240MHz
 *   USB CDC On Boot  : Enabled
 *
 * Cara pakai:
 *   1. Upload sketch (egg_model.h sudah ada di folder ini)
 *   2. Buka Serial Monitor 115200
 *   3. Tekan tombol BOOT atau kirim 'c' untuk klasifikasi
 */

#include <Wire.h>
#include "DFRobot_AXP313A.h"
#include "esp_camera.h"
#include "TensorFlowLite_ESP32.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "egg_model.h"

// ── Pin mapping FireBeetle 2 ESP32-S3 N16R8 ──────────────────
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

#define BOOT_BTN        0    // Tombol BOOT ESP32
#define BLINK_LED      21    // LED onboard (opsional)

// ── TFLite ────────────────────────────────────────────────────
// Model 315 KB → arena 200 KB di PSRAM untuk activations
#define ARENA_SIZE (200 * 1024)

static uint8_t*                  tensor_arena     = nullptr;
static tflite::MicroInterpreter* interpreter      = nullptr;
static TfLiteTensor*             input_tensor     = nullptr;
static TfLiteTensor*             output_tensor    = nullptr;

// ── Image params ──────────────────────────────────────────────
// Kamera: QQVGA 160×120 RGB565
// Model input: 96×96 → center crop dari 160×120
#define SRC_W   160
#define SRC_H   120
#define CROP_W   96
#define CROP_H   96
#define CROP_X  ((SRC_W - CROP_W) / 2)   // 32
#define CROP_Y  ((SRC_H - CROP_H) / 2)   // 12

DFRobot_AXP313A axp;

// ── Forward declarations ──────────────────────────────────────
bool initCamera();
bool initTFLite();
void runClassify();

// ─────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n╔═══════════════════════════════════════╗");
  Serial.println("║  Egg Classifier — Phase 5 Inference   ║");
  Serial.println("╚═══════════════════════════════════════╝");
  Serial.printf("[INFO] PSRAM     : %lu KB%s\n",
    (unsigned long)(ESP.getPsramSize() / 1024),
    ESP.getPsramSize() > 0 ? " ✓" : " ✗ (wajib aktif!)");
  Serial.printf("[INFO] Free heap : %lu KB\n",
    (unsigned long)(ESP.getFreeHeap() / 1024));

  pinMode(BOOT_BTN, INPUT_PULLUP);
  pinMode(BLINK_LED, OUTPUT);

  // 1. AXP313A
  Wire.begin(CAM_PIN_SIOD, CAM_PIN_SIOC);
  if (axp.begin() != 0) {
    Serial.println("[ERROR] AXP313A tidak terdeteksi!");
    return;
  }
  axp.enableCameraPower(axp.eOV2640);
  delay(100);
  Serial.println("[OK] AXP313A: camera power on");

  // 2. Camera
  if (!initCamera()) return;

  // 3. TFLite
  if (!initTFLite()) return;

  Serial.println("\n[READY] Tekan tombol BOOT atau kirim 'c' untuk klasifikasi.");
  Serial.println("─────────────────────────────────────────\n");
}

void loop() {
  // Trigger via tombol BOOT (active low)
  if (digitalRead(BOOT_BTN) == LOW) {
    delay(50);                         // debounce
    if (digitalRead(BOOT_BTN) == LOW) {
      runClassify();
      while (digitalRead(BOOT_BTN) == LOW) delay(10);  // tunggu lepas
    }
  }

  // Trigger via Serial ('c')
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 'c' || c == 'C') runClassify();
  }

  delay(50);
}

// ─────────────────────────────────────────────────────────────
bool initCamera() {
  camera_config_t cfg;
  cfg.ledc_channel  = LEDC_CHANNEL_0;
  cfg.ledc_timer    = LEDC_TIMER_0;
  cfg.pin_d0        = CAM_PIN_D0;
  cfg.pin_d1        = CAM_PIN_D1;
  cfg.pin_d2        = CAM_PIN_D2;
  cfg.pin_d3        = CAM_PIN_D3;
  cfg.pin_d4        = CAM_PIN_D4;
  cfg.pin_d5        = CAM_PIN_D5;
  cfg.pin_d6        = CAM_PIN_D6;
  cfg.pin_d7        = CAM_PIN_D7;
  cfg.pin_xclk      = CAM_PIN_XCLK;
  cfg.pin_pclk      = CAM_PIN_PCLK;
  cfg.pin_vsync     = CAM_PIN_VSYNC;
  cfg.pin_href      = CAM_PIN_HREF;
  cfg.pin_sccb_sda  = CAM_PIN_SIOD;
  cfg.pin_sccb_scl  = CAM_PIN_SIOC;
  cfg.pin_pwdn      = CAM_PIN_PWDN;
  cfg.pin_reset     = CAM_PIN_RESET;
  cfg.xclk_freq_hz  = 20000000;
  cfg.pixel_format  = PIXFORMAT_RGB565;  // langsung raw, tidak perlu decode JPEG
  cfg.frame_size    = FRAMESIZE_QQVGA;   // 160×120 — paling dekat dengan 96×96
  cfg.jpeg_quality  = 12;
  cfg.fb_count      = 1;
  cfg.grab_mode     = CAMERA_GRAB_LATEST;
  cfg.fb_location   = (ESP.getPsramSize() > 0) ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;

  esp_err_t err = esp_camera_init(&cfg);
  if (err != ESP_OK) {
    Serial.printf("[ERROR] Camera init gagal: 0x%x\n", err);
    return false;
  }

  // Turunkan sensor gain/expo untuk kondisi dalam ruangan
  sensor_t* s = esp_camera_sensor_get();
  if (s) {
    s->set_brightness(s, 1);
    s->set_contrast(s, 1);
    s->set_saturation(s, 0);
    s->set_whitebal(s, 1);
    s->set_awb_gain(s, 1);
    s->set_aec2(s, 0);
    s->set_ae_level(s, 0);
  }

  Serial.printf("[OK] Camera: QQVGA %d×%d RGB565\n", SRC_W, SRC_H);
  return true;
}

// ─────────────────────────────────────────────────────────────
bool initTFLite() {
  // Arena wajib di PSRAM karena 200 KB terlalu besar untuk SRAM
  if (ESP.getPsramSize() == 0) {
    Serial.println("[ERROR] PSRAM tidak aktif. Set Tools→PSRAM→OPI PSRAM.");
    return false;
  }
  tensor_arena = (uint8_t*) ps_malloc(ARENA_SIZE);
  if (!tensor_arena) {
    Serial.println("[ERROR] ps_malloc arena gagal.");
    return false;
  }
  memset(tensor_arena, 0, ARENA_SIZE);

  const tflite::Model* model = tflite::GetModel(g_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("[ERROR] TFLite schema mismatch: model=%d, lib=%d\n",
                  model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }

  // Ops yang digunakan MobileNetV1 + head binary classification
  static tflite::MicroMutableOpResolver<10> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddAveragePool2D();
  resolver.AddReshape();
  resolver.AddFullyConnected();
  resolver.AddLogistic();       // sigmoid output
  resolver.AddQuantize();
  resolver.AddDequantize();

  static tflite::MicroInterpreter static_interp(
    model, resolver, tensor_arena, ARENA_SIZE
  );
  interpreter = &static_interp;

  TfLiteStatus status = interpreter->AllocateTensors();
  if (status != kTfLiteOk) {
    Serial.println("[ERROR] AllocateTensors gagal. Coba perbesar ARENA_SIZE.");
    return false;
  }

  input_tensor  = interpreter->input(0);
  output_tensor = interpreter->output(0);

  Serial.println("[OK] TFLite model loaded");
  Serial.printf("     Input  : [%d,%d,%d,%d] type=%d\n",
    input_tensor->dims->data[0], input_tensor->dims->data[1],
    input_tensor->dims->data[2], input_tensor->dims->data[3],
    input_tensor->type);
  Serial.printf("     Output : [%d] type=%d\n",
    output_tensor->dims->data[1], output_tensor->type);
  Serial.printf("     Arena  : %lu KB dipakai dari %d KB\n",
    (unsigned long)(interpreter->arena_used_bytes() / 1024),
    ARENA_SIZE / 1024);

  return true;
}

// ─────────────────────────────────────────────────────────────
void runClassify() {
  Serial.println("─────────────────────────────────────────");
  Serial.println("[CAM] Capturing...");

  // Beri waktu kamera auto-expose (terutama setelah lama diam)
  // Buang 2 frame pertama agar AWB stabil
  for (int i = 0; i < 2; i++) {
    camera_fb_t* dummy = esp_camera_fb_get();
    if (dummy) esp_camera_fb_return(dummy);
    delay(30);
  }

  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("[ERROR] Camera capture gagal.");
    return;
  }

  if (fb->width != SRC_W || fb->height != SRC_H) {
    Serial.printf("[ERROR] Frame size tidak sesuai: %dx%d (expected %dx%d)\n",
                  fb->width, fb->height, SRC_W, SRC_H);
    esp_camera_fb_return(fb);
    return;
  }

  // ── Preprocess: RGB565 160×120 → INT8 96×96 ──────────────
  // Input quantization params dari model
  const float in_scale = input_tensor->params.scale;
  const int   in_zp    = input_tensor->params.zero_point;

  uint16_t* src = (uint16_t*) fb->buf;
  int8_t*   dst = input_tensor->data.int8;

  for (int y = 0; y < CROP_H; y++) {
    for (int x = 0; x < CROP_W; x++) {
      uint16_t px = src[(CROP_Y + y) * SRC_W + (CROP_X + x)];

      // RGB565 → RGB888
      uint8_t r = ((px >> 11) & 0x1F) * 255 / 31;
      uint8_t g = ((px >>  5) & 0x3F) * 255 / 63;
      uint8_t b = ((px >>  0) & 0x1F) * 255 / 31;

      int base = (y * CROP_W + x) * 3;
      // Float [0,1] → INT8 via model's quantization params
      dst[base + 0] = (int8_t) constrain((int)roundf(r / 255.0f / in_scale) + in_zp, -128, 127);
      dst[base + 1] = (int8_t) constrain((int)roundf(g / 255.0f / in_scale) + in_zp, -128, 127);
      dst[base + 2] = (int8_t) constrain((int)roundf(b / 255.0f / in_scale) + in_zp, -128, 127);
    }
  }
  esp_camera_fb_return(fb);

  // ── Inference ─────────────────────────────────────────────
  unsigned long t0 = millis();
  TfLiteStatus status = interpreter->Invoke();
  unsigned long dt    = millis() - t0;

  if (status != kTfLiteOk) {
    Serial.println("[ERROR] Inference gagal.");
    return;
  }

  // ── Baca output ───────────────────────────────────────────
  // Output INT8 sigmoid → float [0,1]
  // 0 = BAD (TIDAK BAGUS), 1 = GOOD (BAGUS)
  const float out_scale = output_tensor->params.scale;
  const int   out_zp    = output_tensor->params.zero_point;
  float score = (output_tensor->data.int8[0] - out_zp) * out_scale;

  const bool  is_good   = (score >= 0.5f);
  const char* label     = is_good ? "BAGUS" : "TIDAK BAGUS";
  const char* confidence =
    (score > 0.85f || score < 0.15f) ? "TINGGI" :
    (score > 0.70f || score < 0.30f) ? "SEDANG" : "RENDAH";

  // ── Output ────────────────────────────────────────────────
  Serial.println();
  Serial.println("┌─────────────────────────────────┐");
  Serial.printf( "│  Hasil    : %-20s │\n", label);
  Serial.printf( "│  Score    : %.3f                 │\n", score);
  Serial.printf( "│  Keyakinan: %-8s               │\n", confidence);
  Serial.printf( "│  Waktu    : %lu ms                │\n", dt);
  Serial.println("└─────────────────────────────────┘");

  // LED blink (opsional): 1x = BAGUS, 3x = TIDAK BAGUS
  int blinks = is_good ? 1 : 3;
  for (int i = 0; i < blinks; i++) {
    digitalWrite(BLINK_LED, HIGH);
    delay(200);
    digitalWrite(BLINK_LED, LOW);
    delay(200);
  }

  Serial.println("\n[READY] Tekan BOOT atau kirim 'c' untuk klasifikasi berikutnya.");
}
