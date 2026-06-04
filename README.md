# Klasifikasi Telur — ESP32-S3-CAM N16R8

Sistem klasifikasi telur **on-device** (Bagus / Tidak Bagus) menggunakan kamera OV2640 dan inferensi TFLite Micro langsung di ESP32-S3 tanpa koneksi cloud.

---

## Hardware

| Komponen | Detail |
|---|---|
| Board | DFRobot FireBeetle 2 ESP32-S3 N16R8 |
| MCU | ESP32-S3 dual-core Xtensa LX7 @ 240 MHz |
| Flash | 16 MB |
| PSRAM | 8 MB Octal (OPI) |
| Kamera | OV2640 2MP (onboard) |

---

## Arsitektur Sistem

```
OV2640 Camera (RGB565 320×240)
        ↓
  Preprocessing (resize → 96×96, normalize → int8)
        ↓
  TFLite Micro Interpreter + ESP-NN
  Model: MobileNetV1 96×96 α=0.25 INT8
        ↓
  Output: BAGUS / TIDAK BAGUS
        ↓
  LED / Serial / MQTT
```

**Target performa:** inferensi ~50–80 ms, akurasi >95%

---

## Struktur Direktori

```
klasifikasiTelur_ESP32_S3_CAM/
├── README.md
├── dokumentasi_proyek.md       # Riset mendalam arsitektur & pipeline
├── ESP32-S3-EggClassifier/     # Firmware PlatformIO (ESP-IDF)
│   ├── platformio.ini          # Konfigurasi board & build
│   ├── sdkconfig.defaults      # PSRAM Octal, Flash 16MB, CPU 240MHz
│   ├── src/
│   │   ├── main.c              # Aplikasi utama
│   │   ├── CMakeLists.txt
│   │   └── idf_component.yml   # Dependency: esp32-camera
│   └── dependencies.lock       # Lock file komponen
└── training/                   # (akan dibuat) Script Python training
    ├── train.py
    ├── convert_to_tflite.py
    └── datasets/               # Tidak di-track git
```

---

## Cara Build & Flash

### Prasyarat

- [VS Code](https://code.visualstudio.com/) + extension [PlatformIO](https://platformio.org/install/ide?install=vscode)
- Python 3.10+ (untuk training)

### Build dan Upload

```bash
cd ESP32-S3-EggClassifier

# Build
~/.platformio/penv/bin/platformio run

# Upload ke board
~/.platformio/penv/bin/platformio run -t upload

# Buka serial monitor
~/.platformio/penv/bin/platformio device monitor
```

Atau gunakan tombol di status bar VS Code: **Build (✓)** → **Upload (→)** → **Monitor (🔌)**

---

## Status Fase Pengembangan

| Fase | Deskripsi | Status |
|---|---|---|
| **Phase 1** | Verifikasi hardware — camera test | 🔄 In Progress |
| **Phase 2** | Data collection firmware (capture → WiFi → PC) | ⏳ Pending |
| **Phase 3** | Pengumpulan dataset (300–500 foto/kelas) | ⏳ Pending |
| **Phase 4** | Training model di PC (Python + TFLite int8) | ⏳ Pending |
| **Phase 5** | Inference firmware lengkap | ⏳ Pending |

---

## Output Serial Monitor (Phase 1)

```
[CAM_TEST] === FireBeetle 2 ESP32-S3 N16R8 — Camera Test ===
[CAM_TEST] Chip        : ESP32-S3, 2 core(s)
[CAM_TEST] PSRAM total : 8192 KB
[CAM_TEST] PSRAM free  : 8100 KB
[CAM_TEST] SRAM free   : 350 KB
[CAM_TEST] Kamera berhasil diinisialisasi
[CAM_TEST] Frame 1: 320x240, 8456 bytes, 45.2 ms
[CAM_TEST] Frame 2: 320x240, 8312 bytes, 44.8 ms
[CAM_TEST] Frame 3: 320x240, 8391 bytes, 45.1 ms
[CAM_TEST] Frame 4: 320x240, 8274 bytes, 44.9 ms
[CAM_TEST] Frame 5: 320x240, 8318 bytes, 45.0 ms
[CAM_TEST] Hasil: 5/5 frame berhasil
[CAM_TEST] === PHASE 1 PASSED: Kamera berfungsi normal ===
```

---

## Pin Mapping Kamera (FireBeetle 2 ESP32-S3)

| Signal | GPIO |
|---|---|
| XCLK | 45 |
| SDA (SCCB) | 1 |
| SCL (SCCB) | 2 |
| D0–D7 | 39, 40, 41, 4, 7, 8, 46, 48 |
| VSYNC | 6 |
| HREF | 42 |
| PCLK | 5 |

---

## Referensi

- [esp-tflite-micro — Espressif](https://github.com/espressif/esp-tflite-micro)
- [esp32-camera — Espressif](https://github.com/espressif/esp32-camera)
- [ESP-NN optimized kernels](https://github.com/espressif/esp-nn)
- [DFRobot FireBeetle 2 Wiki](https://wiki.dfrobot.com/SKU_DFR0975_FireBeetle_2_Board_ESP32_S3)
- [Dokumentasi Riset Lengkap](./dokumentasi_proyek.md)
