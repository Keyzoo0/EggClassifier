# Deep Research: Klasifikasi Telur pada ESP32-S3-CAM N16R8
## On-Device Binary Classification (Bagus / Tidak Bagus)

---

## 1. Profil Hardware — Apa yang Kamu Punya

| Spesifikasi | Detail |
|---|---|
| **MCU** | ESP32-S3 dual-core Xtensa LX7 @ 240 MHz |
| **Internal SRAM** | 512 KB (usable ~390–420 KB setelah RTOS overhead) |
| **PSRAM** | 8 MB Octal SPI (OPI), DDR mode — bandwidth ~40 MB/s |
| **Flash** | 16 MB (N16) — cukup untuk model + firmware + OTA partition |
| **Vector Instructions** | PIE (Processor Instruction Extension) — SIMD int8/int16 |
| **Camera** | OV2640 (2MP, max 1600×1200), output JPEG/YUV422/RGB565 |
| **PSRAM Mode** | Octal DDR — **jauh lebih cepat** dari Quad SPI biasa |

**Keunggulan utama N16R8 vs N8R2:**
- 8 MB PSRAM → tensor arena besar bisa dialokasi tanpa masalah
- 16 MB Flash → model sampai ~4 MB bisa masuk, plus OTA dual-partition
- Octal PSRAM → ~2–3× bandwidth vs Quad, kritis untuk akses tensor besar

---

## 2. Arsitektur Sistem yang Direkomendasikan

```
┌──────────────────────────────────────────────────────────┐
│                    ESP32-S3-CAM N16R8                     │
│                                                          │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐   │
│  │ OV2640   │───▶│ Preprocessing│───▶│  TFLite Micro │   │
│  │ Camera   │    │ (resize,     │    │  Interpreter  │   │
│  │ JPEG/RGB │    │  normalize,  │    │  + ESP-NN     │   │
│  │          │    │  int8 quant) │    │  Optimized    │   │
│  └──────────┘    └──────────────┘    └───────┬───────┘   │
│                                              │           │
│                                     ┌────────▼────────┐  │
│                                     │   Output:       │  │
│                                     │   [0] = Bagus   │  │
│                                     │   [1] = Tidak   │  │
│                                     │        Bagus    │  │
│                                     └────────┬────────┘  │
│                                              │           │
│                              ┌───────────────▼────────┐  │
│                              │ Aksi: LED / Buzzer /   │  │
│                              │ WebSocket / MQTT / LCD │  │
│                              └────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

---

## 3. Strategi Model — Pilihan & Trade-Off

### 3.1 Pilihan Arsitektur CNN

Untuk binary classification (bagus/tidak bagus) pada MCU, ada 3 opsi realistis:

#### Opsi A: MobileNetV1 96×96 α=0.25 (REKOMENDASI UTAMA)

| Metrik | Nilai |
|---|---|
| Model size (int8) | ~100–130 KB |
| RAM (tensor arena) | ~100–150 KB |
| Inference time (ESP-NN, SRAM) | ~50–80 ms |
| Inference time (PSRAM arena) | ~150–300 ms |
| Akurasi tipikal (binary, dataset kecil) | 92–97% |

**Kenapa ini best choice:**
- Depthwise separable convolutions = dioptimasi penuh oleh ESP-NN assembly
- Cukup kecil untuk tensor arena di **internal SRAM** → kecepatan maksimal
- α=0.25 artinya channel count dikecilkan 4×, sangat ringan tapi masih powerful untuk binary task

#### Opsi B: MobileNetV2 96×96 α=0.35

| Metrik | Nilai |
|---|---|
| Model size (int8) | ~200–320 KB |
| RAM (tensor arena) | ~200–375 KB |
| Inference time (ESP-NN, SRAM) | ~100–200 ms |
| Akurasi tipikal | 94–98% |

**Trade-off:** Inverted residuals lebih akurat tapi butuh arena lebih besar. Masih bisa masuk SRAM kalau α kecil. Kalau model > 300 KB arena, terpaksa ke PSRAM dan inference melambat.

#### Opsi C: Custom 4-Layer CNN (Paling Ringan)

| Metrik | Nilai |
|---|---|
| Model size (int8) | ~20–50 KB |
| RAM (tensor arena) | ~30–80 KB |
| Inference time | ~20–40 ms |
| Akurasi tipikal | 85–93% |

**Trade-off:** Tercepat tapi akurasi bisa lebih rendah, terutama kalau variasi telur banyak. Cocok kalau mau >10 FPS real-time.

### 3.2 Rekomendasi Final

**Pakai MobileNetV1 96×96 α=0.25 dengan int8 full quantization.**

Alasan:
1. Tensor arena ~100–150 KB → muat di internal SRAM 512 KB
2. ESP-NN assembly kernels memberikan speedup ~40× dibanding ANSI C (Person Detection: 2300ms → 54ms)
3. Transfer learning dari ImageNet → fine-tune hanya top layers dengan dataset telur kamu
4. Binary task (2 kelas) = network head sangat kecil, overhead minimal

---

## 4. Pipeline Training — Step by Step

### 4.1 Pengumpulan Dataset

**Target minimum: 200–400 foto per kelas (total 400–800 foto)**

Tips pengambilan foto telur:
- Gunakan background **putih atau hitam solid** — konsisten
- Pencahayaan: diffused LED strip, hindari shadow keras
- Jarak kamera: tetap (10–15 cm), pakai jig/holder
- Ambil dari ESP32-S3-CAM langsung kalau bisa → menghindari domain gap antara training data vs inference data
- Resolusi capture: 320×240 JPEG sudah cukup, nanti di-resize ke 96×96

**Struktur folder:**
```
datasets/
├── good/          # Telur bagus (utuh, bersih, bentuk normal)
│   ├── good_001.jpg
│   ├── good_002.jpg
│   └── ...
└── bad/           # Telur tidak bagus (retak, kotor, bentuk abnormal)
    ├── bad_001.jpg
    ├── bad_002.jpg
    └── ...
```

### 4.2 Augmentasi Data (Kritis untuk Dataset Kecil!)

```python
import tensorflow as tf

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.3),       # ±30°
    tf.keras.layers.RandomBrightness(0.2),      # variasi cahaya
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomZoom((-0.1, 0.1)),    # slight zoom
])
```

Augmentasi ini penting karena:
- Telur bisa di-orient mana saja di conveyor
- Pencahayaan di lapangan beda dengan saat training
- Dataset kecil (ratusan) butuh variasi buatan supaya generalize

### 4.3 Training Script (Transfer Learning)

```python
import tensorflow as tf
import numpy as np

# === Config ===
IMG_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 30
ALPHA = 0.25

# === Load Dataset ===
train_ds = tf.keras.utils.image_dataset_from_directory(
    "datasets/",
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="binary"   # 0=good, 1=bad
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "datasets/",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

# === Augmentation ===
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomBrightness(0.15),
    tf.keras.layers.RandomContrast(0.15),
])

# === Model: MobileNetV1 α=0.25, Transfer Learning ===
base_model = tf.keras.applications.MobileNet(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    alpha=ALPHA,
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False   # Freeze base layers

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    data_augmentation,
    tf.keras.layers.Rescaling(1./127.5, offset=-1),  # [-1, 1] range
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation="sigmoid")    # Binary output
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# === Phase 1: Train head only ===
model.fit(train_ds, validation_data=val_ds, epochs=15)

# === Phase 2: Fine-tune top layers ===
base_model.trainable = True
# Freeze semua kecuali 20 layer terakhir
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # LR rendah!
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
model.fit(train_ds, validation_data=val_ds, epochs=15)

model.save("egg_classifier_mobilenet")
```

### 4.4 Konversi ke TFLite INT8 (Full Integer Quantization)

```python
import tensorflow as tf
import numpy as np

# Representative dataset untuk kalibrasi quantization
def representative_dataset():
    for images, _ in train_ds.take(50):  # 50 batch
        for img in images:
            yield [tf.cast(tf.expand_dims(img, 0), tf.float32)]

# === Convert ===
converter = tf.lite.TFLiteConverter.from_saved_model("egg_classifier_mobilenet")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8    # Input int8
converter.inference_output_type = tf.int8   # Output int8

tflite_model = converter.convert()

# Simpan
with open("egg_classifier_int8.tflite", "wb") as f:
    f.write(tflite_model)

print(f"Model size: {len(tflite_model) / 1024:.1f} KB")
# Ekspektasi: ~100-130 KB untuk MobileNetV1 α=0.25
```

**Kenapa HARUS int8 full quantization:**
- ESP-NN assembly kernels HANYA bekerja optimal dengan int8
- Tanpa int8 → fallback ke ANSI C → inference 40× lebih lambat
- Person Detection benchmark: **54 ms (int8+ESP-NN)** vs **2300 ms (tanpa)**

### 4.5 Konversi ke C Header

```bash
xxd -i egg_classifier_int8.tflite > egg_model.h
```

Atau pakai Python:
```python
with open("egg_classifier_int8.tflite", "rb") as f:
    data = f.read()

with open("egg_model.h", "w") as f:
    f.write(f"const unsigned char egg_model[] = {{\n")
    for i, byte in enumerate(data):
        f.write(f"0x{byte:02x}, ")
        if (i + 1) % 12 == 0:
            f.write("\n")
    f.write(f"\n}};\n")
    f.write(f"const unsigned int egg_model_len = {len(data)};\n")
```

---

## 5. Optimasi Memory — Kunci Performa di ESP32-S3

### 5.1 Memory Map Strategy

```
┌─────────────────────────────────────────────────┐
│              FLASH (16 MB)                       │
│  ┌──────────────────────────────────────────┐   │
│  │ Firmware partition    (~2 MB)             │   │
│  │ Model weights (XIP mapped, const array)  │   │
│  │ OTA partition         (~2 MB)             │   │
│  │ NVS + SPIFFS          (~1 MB)             │   │
│  └──────────────────────────────────────────┘   │
├─────────────────────────────────────────────────┤
│          INTERNAL SRAM (512 KB)                  │
│  ┌──────────────────────────────────────────┐   │
│  │ RTOS stacks + heap    (~100 KB)          │   │
│  │ ★ Tensor Arena        (~150 KB) ← TARUH │   │
│  │   DI SINI UNTUK KECEPATAN MAKS          │   │
│  │ Camera DMA buffer     (~50 KB)           │   │
│  │ WiFi/BT stack         (~60 KB)           │   │
│  │ Sisa free heap        (~80-100 KB)       │   │
│  └──────────────────────────────────────────┘   │
├─────────────────────────────────────────────────┤
│          PSRAM (8 MB Octal)                      │
│  ┌──────────────────────────────────────────┐   │
│  │ Camera frame buffer JPEG (~40-100 KB)    │   │
│  │ RGB565 decode buffer   (~150 KB utk      │   │
│  │                         320×240)         │   │
│  │ Preprocessing buffer   (~27 KB utk       │   │
│  │                         96×96×3)         │   │
│  │ Web server assets      (optional)        │   │
│  │ Sisa: ~7.5 MB free                       │   │
│  └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

### 5.2 Aturan Emas Alokasi Memory

1. **Model weights → Flash (XIP):** Deklarasi sebagai `const` array → otomatis di-map dari flash, TIDAK disalin ke RAM
2. **Tensor arena → Internal SRAM:** Ini yang bikin beda 5-10× kecepatan inference
3. **Camera buffers → PSRAM:** Frame buffer besar, tapi tidak perlu kecepatan SRAM
4. **Preprocessing buffer → PSRAM:** Temporary, akses sequential

```c
// Alokasi tensor arena di INTERNAL SRAM (KRITIS!)
static uint8_t tensor_arena[150 * 1024]
    __attribute__((aligned(16)));

// Camera frame buffer ke PSRAM
uint8_t* camera_fb = (uint8_t*)heap_caps_malloc(
    320 * 240 * 2,  // RGB565
    MALLOC_CAP_SPIRAM
);

// Preprocessing buffer ke PSRAM
int8_t* input_buffer = (int8_t*)heap_caps_malloc(
    96 * 96 * 3,
    MALLOC_CAP_SPIRAM
);
```

### 5.3 sdkconfig Konfigurasi Kritis

```ini
# === PSRAM ===
CONFIG_SPIRAM=y
CONFIG_SPIRAM_MODE_OCT=y          # Octal mode untuk N16R8!
CONFIG_SPIRAM_SPEED_80M=y         # 80 MHz clock
CONFIG_SPIRAM_USE_MALLOC=y        # heap_caps bisa alokasi ke PSRAM

# === Flash ===
CONFIG_ESPTOOLPY_FLASHSIZE_16MB=y
CONFIG_ESPTOOLPY_FLASHMODE_QIO=y
CONFIG_ESPTOOLPY_FLASHFREQ_80M=y

# === CPU ===
CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_240=y

# === Camera ===
CONFIG_CAMERA_TASK_STACK_SIZE=4096

# === Memory ===
CONFIG_ESP32S3_DATA_CACHE_64KB=y       # Max data cache
CONFIG_ESP32S3_DATA_CACHE_LINE_64B=y   # 64-byte cache lines
CONFIG_ESP32S3_INSTRUCTION_CACHE_32KB=y
```

---

## 6. Firmware ESP-IDF — Kode Lengkap

### 6.1 Struktur Project

```
egg-classifier/
├── CMakeLists.txt
├── sdkconfig.defaults
├── main/
│   ├── CMakeLists.txt
│   ├── main.c
│   ├── camera_init.h
│   ├── camera_init.c
│   ├── egg_classifier.h
│   ├── egg_classifier.cc     # C++ untuk TFLite API
│   └── egg_model.h           # Model binary (xxd output)
├── components/
│   └── esp-tflite-micro/     # dari ESP Component Registry
└── partitions.csv
```

### 6.2 CMakeLists.txt (main)

```cmake
idf_component_register(
    SRCS "main.c" "camera_init.c" "egg_classifier.cc"
    INCLUDE_DIRS "."
    REQUIRES esp_timer esp_camera
)

# ESP-NN optimizations (KRITIS!)
target_compile_options(${COMPONENT_LIB} PRIVATE -DESP_NN)
```

### 6.3 Camera Init (camera_init.c)

```c
#include "esp_camera.h"
#include "camera_init.h"

// Pin mapping untuk AI-Thinker ESP32-S3-CAM
// SESUAIKAN dengan board kamu!
#define CAM_PIN_PWDN    -1
#define CAM_PIN_RESET   -1
#define CAM_PIN_XCLK    15
#define CAM_PIN_SIOD    4
#define CAM_PIN_SIOC    5
#define CAM_PIN_D7      16
#define CAM_PIN_D6      17
#define CAM_PIN_D5      18
#define CAM_PIN_D4      12
#define CAM_PIN_D3      10
#define CAM_PIN_D2      8
#define CAM_PIN_D1      9
#define CAM_PIN_D0      11
#define CAM_PIN_VSYNC   6
#define CAM_PIN_HREF    7
#define CAM_PIN_PCLK    13

esp_err_t camera_init(void)
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

        .xclk_freq_hz = 20000000,
        .ledc_timer   = LEDC_TIMER_0,
        .ledc_channel = LEDC_CHANNEL_0,

        // RGB565 untuk langsung proses tanpa decode JPEG
        .pixel_format = PIXFORMAT_RGB565,
        .frame_size   = FRAMESIZE_QVGA,  // 320×240
        .fb_count     = 1,               // 1 buffer cukup
        .fb_location  = CAMERA_FB_IN_PSRAM,
        .grab_mode    = CAMERA_GRAB_LATEST,
    };

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        printf("Camera init failed: 0x%x\n", err);
    }

    // Tunggu AWB stabilize (known issue OV2640)
    vTaskDelay(pdMS_TO_TICKS(200));

    return err;
}
```

### 6.4 Egg Classifier (egg_classifier.cc)

```cpp
#include "egg_classifier.h"
#include "egg_model.h"  // const unsigned char egg_model[]

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Tensor arena di INTERNAL SRAM — jangan pindah ke PSRAM!
static uint8_t tensor_arena[150 * 1024] __attribute__((aligned(16)));

static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input_tensor = nullptr;
static TfLiteTensor* output_tensor = nullptr;

// Hanya register operator yang dipakai MobileNetV1
static tflite::MicroMutableOpResolver<10> resolver;

bool egg_classifier_init(void)
{
    const tflite::Model* model = tflite::GetModel(egg_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Model schema mismatch!\n");
        return false;
    }

    // Register HANYA operator yang diperlukan (hemat memory)
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddReshape();
    resolver.AddSoftmax();
    resolver.AddFullyConnected();
    resolver.AddAveragePool2D();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddPad();
    resolver.AddMean();

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, sizeof(tensor_arena));
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        printf("AllocateTensors failed!\n");
        printf("Arena used: %zu bytes\n",
               interpreter->arena_used_bytes());
        return false;
    }

    input_tensor  = interpreter->input(0);
    output_tensor = interpreter->output(0);

    printf("Model loaded. Arena used: %zu / %zu bytes\n",
           interpreter->arena_used_bytes(), sizeof(tensor_arena));
    printf("Input: %dx%dx%d (type=%d, scale=%.6f, zp=%d)\n",
           input_tensor->dims->data[1],
           input_tensor->dims->data[2],
           input_tensor->dims->data[3],
           input_tensor->type,
           input_tensor->params.scale,
           input_tensor->params.zero_point);

    return true;
}

// Preprocessing: RGB565 (320×240) → int8 (96×96×3)
void preprocess_image(const uint8_t* rgb565_data,
                      int src_w, int src_h,
                      int8_t* dst, int dst_size)
{
    const int DST_W = 96;
    const int DST_H = 96;

    float scale = input_tensor->params.scale;
    int   zp    = input_tensor->params.zero_point;

    for (int dy = 0; dy < DST_H; dy++) {
        int sy = dy * src_h / DST_H;
        for (int dx = 0; dx < DST_W; dx++) {
            int sx = dx * src_w / DST_W;
            int src_idx = (sy * src_w + sx) * 2;  // RGB565 = 2 bytes

            uint16_t pixel = (rgb565_data[src_idx + 1] << 8)
                           |  rgb565_data[src_idx];

            // Extract RGB dari RGB565
            uint8_t r = ((pixel >> 11) & 0x1F) << 3;  // 5 bit → 8 bit
            uint8_t g = ((pixel >> 5)  & 0x3F) << 2;  // 6 bit → 8 bit
            uint8_t b = ((pixel)       & 0x1F) << 3;  // 5 bit → 8 bit

            int dst_idx = (dy * DST_W + dx) * 3;

            // Normalize ke [-1, 1] lalu quantize ke int8
            // Formula: int8_val = (float_val / scale) + zero_point
            dst[dst_idx + 0] = (int8_t)((r / 127.5f - 1.0f) / scale + zp);
            dst[dst_idx + 1] = (int8_t)((g / 127.5f - 1.0f) / scale + zp);
            dst[dst_idx + 2] = (int8_t)((b / 127.5f - 1.0f) / scale + zp);
        }
    }
}

// Return: confidence 0.0–1.0 bahwa telur BAGUS
float egg_classify(const uint8_t* rgb565_frame, int w, int h)
{
    // Copy preprocessed data ke input tensor
    preprocess_image(rgb565_frame, w, h,
                     input_tensor->data.int8,
                     input_tensor->bytes);

    // INFERENCE — ini yang di-accelerate ESP-NN
    uint32_t t0 = esp_timer_get_time();
    TfLiteStatus status = interpreter->Invoke();
    uint32_t t1 = esp_timer_get_time();

    printf("Inference time: %lu us\n", (unsigned long)(t1 - t0));

    if (status != kTfLiteOk) {
        printf("Invoke failed!\n");
        return -1.0f;
    }

    // Output: int8, de-quantize ke float
    int8_t raw_output = output_tensor->data.int8[0];
    float confidence = (raw_output - output_tensor->params.zero_point)
                     * output_tensor->params.scale;

    // Sigmoid output: >0.5 = bad, <0.5 = good
    // Return confidence bahwa telur BAGUS
    return 1.0f - confidence;  // Invert jika label 1 = bad
}
```

### 6.5 Main Loop (main.c)

```c
#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "esp_camera.h"
#include "camera_init.h"
#include "egg_classifier.h"

#define GOOD_THRESHOLD 0.70f   // >70% confidence = bagus
#define LED_PIN        2       // LED indicator

void app_main(void)
{
    printf("=== Egg Classifier ESP32-S3 ===\n");

    // Init PSRAM info
    printf("PSRAM size: %d bytes\n",
           heap_caps_get_total_size(MALLOC_CAP_SPIRAM));
    printf("Free SRAM:  %d bytes\n",
           heap_caps_get_free_size(MALLOC_CAP_INTERNAL));

    // Init camera
    ESP_ERROR_CHECK(camera_init());

    // Init classifier
    if (!egg_classifier_init()) {
        printf("FATAL: Classifier init failed!\n");
        return;
    }

    // GPIO untuk LED indicator
    gpio_set_direction(LED_PIN, GPIO_MODE_OUTPUT);

    printf("System ready. Starting classification loop...\n\n");

    while (1) {
        // Capture frame
        camera_fb_t* fb = esp_camera_fb_get();
        if (!fb) {
            printf("Camera capture failed\n");
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }

        // Classify
        float good_confidence = egg_classify(fb->buf, fb->width, fb->height);

        // Decision
        bool is_good = (good_confidence >= GOOD_THRESHOLD);

        printf("[%s] Confidence: %.1f%%\n",
               is_good ? "BAGUS" : "TIDAK BAGUS",
               good_confidence * 100.0f);

        // LED: ON = bagus, OFF = tidak bagus
        gpio_set_level(LED_PIN, is_good ? 1 : 0);

        // Return frame buffer
        esp_camera_fb_return(fb);

        // Delay antar klasifikasi (sesuaikan kebutuhan)
        vTaskDelay(pdMS_TO_TICKS(500));
    }
}
```

---

## 7. Benchmark Performa yang Diharapkan

### 7.1 Dengan ESP-NN Enabled (Data dari Espressif Resmi)

| Skenario | Inference Time | FPS Efektif |
|---|---|---|
| Person Detection 96×96 int8, **SRAM arena** | **54 ms** | ~18 FPS |
| Person Detection 96×96 int8, PSRAM arena | ~200–400 ms | ~3–5 FPS |
| Tanpa ESP-NN (ANSI C fallback) | **2300 ms** | <1 FPS |

### 7.2 Proyeksi untuk Egg Classifier

MobileNetV1 α=0.25 lebih ringan dari Person Detection model, jadi:

| Konfigurasi | Estimasi |
|---|---|
| Arena di SRAM + ESP-NN | **40–70 ms** (~15–25 FPS) |
| Arena di PSRAM + ESP-NN | ~150–250 ms (~4–6 FPS) |
| Total pipeline (capture + preprocess + infer) | ~100–200 ms |

**Untuk sorting telur di conveyor, 4–5 FPS sudah lebih dari cukup.**

---

## 8. Alternatif Toolchain: Edge Impulse (Lebih Mudah)

Kalau tidak mau repot setup training pipeline manual, Edge Impulse menyediakan end-to-end workflow:

### Langkah-langkah:

1. **Buat project** di studio.edgeimpulse.com → pilih "Images" project type
2. **Upload dataset** JPG ke Data Acquisition → label "good" dan "bad"
3. **Create Impulse:**
   - Image data → 96×96
   - Processing block: Image (RGB atau Grayscale)
   - Learning block: Transfer Learning (MobileNetV1 96×96 0.25)
4. **Train** — Edge Impulse otomatis quantize ke int8
5. **Set target device:** ESP-EYE (profil serupa ESP32-S3)
6. **Deploy** → Download sebagai Arduino Library atau ESP-IDF C++ SDK
7. **Integrate** ke firmware dengan contoh kode yang disediakan

### Keuntungan Edge Impulse:
- Visual monitoring RAM/latency budget di dashboard
- Auto-quantization + profiling
- Export langsung ke format yang compatible dengan esp-tflite-micro
- Bisa test model di browser sebelum deploy

### Kekurangan:
- Free tier terbatas (developer account cukup untuk project ini)
- Kurang fleksibel untuk custom preprocessing
- Model architecture pilihan terbatas

---

## 9. Tips Praktis & Gotchas

### 9.1 Pencahayaan (Paling Sering Bikin Gagal!)

Klasifikasi telur sangat sensitif terhadap pencahayaan. Solusi:
- Buat **enclosure tertutup** dengan LED ring diffused
- Pakai background putih matte → refleksi merata
- Hindari ambient light masuk → bikin hasil inkonsisten
- **Training dan inference harus di kondisi cahaya yang sama**

### 9.2 OV2640 Quirks

- AWB (Auto White Balance) butuh ~150–200 ms setelah init untuk stabil
- Jangan switch format (JPEG↔RGB) di runtime → sering hang
- Pilih RGB565 langsung kalau mau proses ML → skip JPEG decode overhead
- Kalau pakai JPEG → decode dulu, tapi ini butuh CPU time ekstra

### 9.3 PSRAM Pitfall

- Jangan alokasi tensor arena ke PSRAM kecuali model terlalu besar
- PSRAM octal N16R8 sudah ~2–3× lebih cepat dari Quad, tapi masih ~5× lebih lambat dari SRAM
- Kalau perlu PSRAM arena, set cache lines ke 64 bytes: `CONFIG_ESP32S3_DATA_CACHE_LINE_64B=y`

### 9.4 MicroMutableOpResolver vs AllOpsResolver

- **Jangan pakai AllOpsResolver** → register semua 100+ operator → buang ~20–30 KB RAM
- Pakai `MicroMutableOpResolver<N>` → register hanya operator yang dipakai model
- Cara tahu operator apa saja: buka model di [Netron](https://netron.app), lihat tiap layer

### 9.5 Debug Memory

```c
// Cek sebelum dan sesudah init model
printf("Free internal: %d\n",
       heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
printf("Free PSRAM:    %d\n",
       heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
printf("Arena used:    %zu\n",
       interpreter->arena_used_bytes());
```

---

## 10. Opsi Grayscale — Lebih Hemat Lagi

Kalau variasi warna telur tidak penting (misalnya semua telur coklat, defect hanya crack/kotor), bisa pakai grayscale:

| Aspek | RGB | Grayscale |
|---|---|---|
| Input size 96×96 | 27,648 bytes | 9,216 bytes |
| Model size (int8) | ~100-130 KB | ~60-80 KB |
| Arena | ~150 KB | ~80-100 KB |
| Inference | ~50-70 ms | ~25-40 ms |
| Akurasi | Lebih tinggi | Sedikit lebih rendah |

Untuk egg damage detection (crack, dirt), grayscale sering sudah cukup. Warna shell jarang jadi faktor pembeda "bagus vs tidak bagus."

---

## 11. Ringkasan Rekomendasi

| Keputusan | Rekomendasi | Alasan |
|---|---|---|
| **Model** | MobileNetV1 96×96 α=0.25 int8 | Optimal size/speed/accuracy |
| **Input** | 96×96 RGB (atau grayscale) | Cukup detail untuk egg defect |
| **Quantization** | Full int8 (input & output) | ESP-NN acceleration |
| **Tensor arena** | Internal SRAM (~150 KB) | 5-10× lebih cepat dari PSRAM |
| **Camera format** | RGB565 langsung | Skip JPEG decode |
| **Frame buffer** | PSRAM | Besar, tidak perlu cepat |
| **Training** | Transfer learning + augmentasi | Dataset kecil (ratusan) cukup |
| **Framework** | ESP-IDF + esp-tflite-micro | ESP-NN auto-enabled |
| **Toolchain alt** | Edge Impulse | Kalau mau cepat & visual |

**Target performa realistis: inference ~50-80 ms, total pipeline ~150 ms, akurasi >95% untuk binary classification.**

---

## Referensi & Resource

- [esp-tflite-micro (Espressif)](https://github.com/espressif/esp-tflite-micro) — component resmi
- [ESP-NN optimized kernels](https://github.com/espressif/esp-nn) — assembly SIMD
- [Edge Impulse Transfer Learning Docs](https://docs.edgeimpulse.com/docs/edge-impulse-studio/learning-blocks/transfer-learning-images)
- [Mendeley Egg Dataset (Good/Bad)](https://data.mendeley.com/datasets/mdty358x8m/1) — 1000 foto + augmentasi
- [ESP-IDF PSRAM Config Guide](https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/api-guides/flash_psram_config.html)
- [Netron Model Viewer](https://netron.app) — inspeksi operator model
