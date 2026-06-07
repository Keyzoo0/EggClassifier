# RANCANG BANGUN ALAT KLASIFIKASI TELUR BERDASARKAN TEKSTUR CANGKANG MENGGUNAKAN ESP-32 S3 CAM BERBASIS IOT

---

<div align="center">

## LAPORAN TUGAS AKHIR

Laporan ini diajukan untuk memenuhi salah satu syarat Menyelesaikan  
Pendidikan Program Diploma III Pada Jurusan  
Teknik Komputer

**Oleh :**

**Muhammad Reka Alviandi**  
**062330701499**

**POLITEKNIK NEGERI SRIWIJAYA**  
**PALEMBANG**  
**2026**

</div>

---

## Deskripsi Proyek

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

### Dimensi Board

![Dimensi Board](docs/dimesion.jpg)

### Skematik Rangkaian

![Skematik Rangkaian](docs/schematic.jpg)

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
├── docs/
│   ├── dimesion.jpg            # Dimensi board FireBeetle 2
│   └── schematic.jpg           # Skematik rangkaian
├── CameraTest/                 # Arduino sketch — Phase 1
│   └── CameraTest.ino          # Uji kamera: capture 5 frame, cek memori
└── training/                   # (akan dibuat) Script Python training
    ├── train.py
    ├── convert_to_tflite.py
    └── datasets/               # Tidak di-track git
```

---

## Cara Build & Flash (Arduino IDE)

### Prasyarat

1. [Arduino IDE 2.x](https://www.arduino.cc/en/software)
2. Tambahkan board ESP32 — buka **File → Preferences**, tambahkan URL berikut ke *Additional boards manager URLs*:
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
3. Buka **Tools → Board → Boards Manager**, cari `esp32` by Espressif, install.

### Konfigurasi Board di Arduino IDE

Buka sketch `CameraTest/CameraTest.ino`, lalu set di menu **Tools**:

| Pengaturan | Nilai |
|---|---|
| Board | `ESP32S3 Dev Module` |
| Flash Size | `16MB (128Mb)` |
| Partition Scheme | `Default 4MB with spiffs` |
| PSRAM | `Disabled` |
| CPU Frequency | `240MHz` |
| **USB CDC On Boot** | **`Enabled`** ← wajib untuk Serial |
| Upload Speed | `921600` |
| Port | `/dev/ttyACM0` (Linux) atau `COMx` (Windows) |

### Upload

1. Sambungkan board via USB-C
2. Pilih port yang benar di **Tools → Port**
3. Klik **Upload (→)**
4. Buka **Serial Monitor** (baud: 115200)

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
