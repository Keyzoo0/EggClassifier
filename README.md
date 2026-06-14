# Rancang Bangun Alat Klasifikasi Telur Berdasarkan Tekstur Cangkang Menggunakan ESP-32 S3 CAM Berbasis IoT

<div align="center">

**Tugas Akhir — Politeknik Negeri Sriwijaya**

Muhammad Reka Alviandi · NIM 062330701499

Jurusan Teknik Komputer · 2026

</div>

---

## Deskripsi

Sistem klasifikasi telur **on-device** yang mendeteksi kualitas cangkang telur (BAGUS / TIDAK BAGUS) secara real-time menggunakan kamera OV2640 dan model TFLite Micro yang berjalan langsung di ESP32-S3 — tanpa koneksi cloud. Pengguna mengakses antarmuka web melalui WiFi untuk melihat live preview kamera, mengumpulkan dataset, dan menjalankan inferensi.

---

## Hardware

| Komponen | Spesifikasi |
|---|---|
| Board | Freenove ESP32-S3-WROOM CAM (FNK0085, varian N16R8) |
| MCU | ESP32-S3 Dual-Core Xtensa LX7 @ 240 MHz |
| Flash | 16 MB (terdeteksi esptool) |
| PSRAM | 8 MB Octal (OPI) |
| Kamera | OV2640 2MP FOV 66.5° (onboard, selalu berdaya — tanpa AXP) |
| Penyimpanan | microSD 4 GB, FAT32 (allocation unit 16K), SDMMC 1-bit |
| LED status | WS2812 RGB onboard (GPIO 48) |
| Flash kamera | Strip **8 LED WS2812 RGB** di GPIO 47 (penerangan, kontrol via web) |
| Daya | 5V / maks 2A (gunakan adaptor/port USB yang kuat) |

> **Catatan jaringan:** ESP32-CAM sensitif terhadap kualitas jaringan. Hindari
> WiFi repeater/penguat yang menambah latency & membuang paket unicast — gunakan
> koneksi langsung ke router atau **hotspot HP** untuk web yang lancar.

<div align="center">

![Skematik Rangkaian](docs/schematic.jpg)

*Skematik Rangkaian*

</div>

---

## Arsitektur Sistem

```
OV2640 (JPEG VGA 640×480, kualitas 40)
        │  live stream / dataset
        ├─────────────────────────→  Web Browser (WiFi)
        │  frame VGA                        ↕ HTTP API (non-blocking)
        ↓                          ESPAsyncWebServer (task AsyncTCP)
  JPEG decode → RGB888             serve aset/SD chunked, paralel
  (rgbArena 900KB PSRAM)                   │
        ↓                                  │ /predict (trigger semaphore)
  Scale nearest-neighbor               inferTrigSem
  VGA → 96×96                             ↓
        ↓                     Inference Task (Core 0 prio 5)
  Kuantisasi → INT8 tensor          MicroInterpreter
        ↓                           MobileNetV1 α=0.25
  TFLite Invoke                          │
        ↓                          inferDoneSem
  Output: BAGUS / TIDAK BAGUS            │
        ↓                                ↓
   LED + flash strip         JSON response → Browser
```

**Arsitektur server & RTOS:**
- **ESPAsyncWebServer** (non-blocking): melayani banyak koneksi paralel, file besar
  di-stream tanpa memblokir — `loop()` tidak lagi memanggil `handleClient()`.
  (Versi sinkron lama memblokir per-request → file besar timeout di link lambat.)
- Inference di **task Core 0 (prio 5)**; handler `/predict` async memicu + menunggu
  semaphore. `rgbArena` 900KB pre-alokasi PSRAM (zero malloc/free saat inferensi).
- Kamera dimuat **sebelum** kamera-init agar buffer model dapat blok PSRAM tak
  terfragmentasi; model di SD card (fallback LittleFS), tahan upload ulang LittleFS.
- `/capture` async: frame disalin ke buffer (`shared_ptr`, bebas otomatis saat
  selesai/disconnect — anti-leak) karena response mereferensi buffer, tak menyalin.

**Spesifikasi model:** MobileNetV1 α=0.25, input 96×96 RGB INT8, ~315 KB

---

## Preprocessing & Hasil Training

<div align="center">

![Preprocessing Dataset](docs/prepocessing%20dataset.png)

*Pipeline preprocessing dataset: capture → resize 96×96 → augmentasi*

</div>

<div align="center">

![Hasil Training](docs/trainResult.png)

*Hasil training: Accuracy & Loss curve (Fase 1 head + Fase 2 fine-tune)*

</div>

---

## Struktur Direktori

```
klasifikasiTelur_ESP32_S3_CAM/
├── README.md
├── .github/workflows/
│   └── train.yml                      # Workflow training otomatis (Actions)
├── dataset/                           # Dataset sinkron dari SD (via web UI)
├── model/                             # Hasil training CI: egg_model.tflite + info
├── docs/
│   ├── schematic.jpg                  # Skematik rangkaian
│   ├── prepocessing dataset.png       # Pipeline preprocessing
│   └── trainResult.png                # Kurva training
├── DataCollector/
│   └── DataCollector.ino              # Pengumpul dataset awal (legacy)
├── training/
│   ├── KlasifikasiTelur_Training.ipynb  # Google Colab notebook (alternatif)
│   ├── train.py                         # Script training (lokal & CI)
│   ├── requirements.txt
│   └── dataset/                         # Foto telur lokal (tidak di-track)
└── EggClassifierV2/                   # ← Firmware utama
    ├── EggClassifierV2.ino
    ├── web-build/                     # Build Tailwind purged (di luar data/)
    └── data/                          # LittleFS — web interface (di-host di alat, TANPA CDN)
        ├── index.html
        ├── app.js
        ├── style.css
        ├── tw.css                     # Tailwind purged ~8KB (ganti Play CDN 400KB+JIT)
        ├── chart.min.js               # Chart.js lokal — lazy-load saat tab Prediksi
        └── secrets.js                 # ⚠️ GITIGNORED — PAT GitHub (window.GH_TOKEN)
```

> **Web tampil cepat tanpa internet:** semua aset (CSS, JS, font) di-host di
> LittleFS, bukan dari CDN. Tailwind dipakai sebagai build purged (`tw.css`),
> bukan Play CDN yang berat. Regenerasi `tw.css` bila menambah class Tailwind baru:
> jalankan `EggClassifierV2/web-build/build.sh` (scan index.html + app.js → `data/tw.css`).
> Chart.js di-lazy-load hanya saat tab Prediksi.
>
> **`secrets.js` (opsional):** berisi `window.GH_TOKEN = "github_pat_..."` agar user
> tak perlu mengetik token di tab Training. File ini **di-gitignore** — token tak
> pernah ter-push ke repo publik (token di repo publik dicabut otomatis GitHub).
> Tanpa file ini, web fallback meminta token via prompt sekali.

---

## Setup & Cara Pakai (EggClassifierV2)

### 1. Install Library (Arduino IDE → Tools → Manage Libraries)

| Library | Author |
|---|---|
| `TensorFlowLite_ESP32` | tanakamasayuki |
| `ESPAsyncWebServer` | ESP32Async (fork ≥3.6.0) |
| `AsyncTCP` | ESP32Async (≥3.3.2) |
| `Adafruit NeoPixel` | Adafruit (flash 8 LED GPIO47) |

> Board Freenove tidak memakai AXP313A — kamera selalu berdaya, library DFRobot tidak diperlukan.

### 2. Konfigurasi Board (Tools)

| Pengaturan | Nilai |
|---|---|
| Board | `ESP32S3 Dev Module` |
| Flash Size | `16MB (128Mb)` |
| **Partition Scheme** | **`Huge APP (3MB No OTA/1MB SPIFFS)`** ← wajib |
| PSRAM | `OPI PSRAM` |
| CPU Frequency | `240MHz` |
| USB CDC On Boot | `Disabled` (Serial lewat chip UART/CH343) |

> **USB CDC On Boot:** set **Disabled** bila memakai port USB-UART (CH343, muncul
> sebagai `/dev/ttyACM0`/`COMx`) — serial lebih andal. Set **Enabled** hanya bila
> Serial lewat USB-OTG native. Pilih sesuai kabel yang Anda colok.

### 3. Set WiFi Credentials

Edit di `EggClassifierV2.ino` (disarankan **hotspot HP** untuk koneksi stabil):
```cpp
#define WIFI_SSID  "nama_wifi_kamu"
#define WIFI_PASS  "password_wifi"
```

### 4. Upload Web Interface ke LittleFS

1. Install plugin **arduino-littlefs-upload** (earlephilhower) untuk Arduino IDE 2.x:
   - Download `.vsix` dari [earlephilhower/arduino-littlefs-upload](https://github.com/earlephilhower/arduino-littlefs-upload)
   - Salin ke `~/.arduinoIDE/plugins/`
2. Buka sketch `EggClassifierV2/EggClassifierV2.ino`
3. **`Ctrl+Shift+P` → "Upload LittleFS to ..."** (tutup Serial Monitor dulu)
4. Tunggu sampai "Completed upload"

> Plugin mengunggah seluruh isi folder `data/` (termasuk `tw.css`, `chart.min.js`,
> dan `secrets.js` bila ada). **Perubahan pada file web (mis. tombol UI) baru
> terlihat setelah LittleFS di-upload ulang** + hard-refresh browser (`Ctrl+Shift+R`).

### 5. Upload Firmware

Klik **Upload (→)**, tunggu selesai. Serial Monitor 115200 (port UART). Perubahan
firmware (default kamera, server, animasi flash) butuh flash firmware — bukan LittleFS.

### 6. Akses Web Interface

Setelah board terhubung WiFi, buka browser:
```
http://telur.local       ← via mDNS (Windows perlu Bonjour)
http://<IP_ADDRESS>      ← IP tampil di Serial Monitor
```

### 7. Pasang Model

Model dipasang dari tab **🚀 Training** (satu pintu):
jalankan pipeline penuh, atau klik **"Pasang Model Ini ke Alat"** pada kartu
Model Terakhir di GitHub. Board restart otomatis dan model tersimpan di **SD card**
(fallback LittleFS bila SD tidak terpasang) — tahan terhadap upload ulang LittleFS
yang menimpa seluruh partisi web.

> Untuk model dari luar pipeline (mis. hasil Colab), gunakan endpoint langsung:
> `curl -F "model=@egg_model.tflite" http://telur.local/upload_model`

---

## Web Interface

### Tab Dataset — Kumpulkan Data (`1`)

| Aksi | Shortcut |
|---|---|
| Capture foto BAGUS | Klik tombol hijau atau tekan `G` |
| Capture foto CACAT | Klik tombol merah atau tekan `B` |

Dengan SD card terpasang, foto tersimpan langsung di alat (`/dataset/good_0001.jpg` dst.)
dan counter dihitung dari isi SD — tidak hilang saat ganti browser.
Tanpa SD card, otomatis jatuh ke mode lama: foto diunduh ke PC.

### Tab Kelola — CRUD Dataset di SD (`2`)

Galeri seluruh foto dataset di SD card dengan operasi lengkap:

| Operasi | Cara |
|---|---|
| **Create** | Upload JPEG dari PC (multi-file) sebagai BAGUS / CACAT |
| **Read** | Galeri thumbnail (filter Semua/Bagus/Cacat, lazy-load 24/batch), ketuk untuk ukuran penuh |
| **Update** | 🔁 pindah label good ↔ bad (file di-rename di SD) |
| **Delete** | 🗑️ hapus foto (dengan konfirmasi) |

Saat tab ini terbuka, **preview kamera dimatikan** (polling berhenti + area preview
disembunyikan) agar bandwidth ESP32 fokus memuat foto dari SD.
Saat sinkron training berikutnya, perubahan ikut diterapkan ke repo GitHub —
foto yang dihapus di SD juga dihapus dari repo (SD = sumber kebenaran).

### Tab Training — Pipeline Otomatis (`3`)

Preview kamera dimatikan di tab ini; sebagai gantinya tampil ringkasan dataset
(jumlah foto BAGUS / CACAT + pemakaian SD).

Satu tombol **"Sinkron + Training + Pasang Model"** menjalankan seluruh pipeline:

1. **Sinkron** — strategi *wipe + reupload*: **hapus semua** dataset di repo, lalu
   **upload ulang semua** dari SD (anti-conflict — tak ada update in-place yang butuh
   sha cocok). SD = sumber kebenaran.
2. **Training** — workflow GitHub Actions (`train.yml`) melatih MobileNetV1 α=0.25 INT8
3. **Pasang** — model hasil training diunduh dan dipasang otomatis ke ESP32 (ke SD card)

Repo target (`Keyzoo0/EggClassifier@main`) sudah tertanam. Token GitHub diambil dari
`secrets.js` (`window.GH_TOKEN`) bila ada → **tidak perlu input token**. Tanpa file itu,
web meminta token sekali via prompt (tersimpan di `localStorage` browser, tidak pernah
menyentuh ESP32).

### Tab Prediksi — Klasifikasi Real-time (`4`)

| Aksi | Shortcut |
|---|---|
| Jalankan klasifikasi | Klik tombol biru atau tekan `C` |

Output: label (BAGUS / TIDAK BAGUS), skor sigmoid, tingkat keyakinan, waktu inferensi.

### Tab Kamera — Pengaturan Sensor (`5`)

Kontrol penuh semua parameter OV2640: brightness, contrast, saturation, efek khusus,
kualitas JPEG, white balance (auto/manual + mode), exposure (AEC auto/manual, AE level),
gain (AGC, gain ceiling), koreksi pixel/gamma/lensa, mirror & flip, dan color bar.
Plus **kontrol Flash Kamera** (on/off + slider kecerahan).

Semua perubahan langsung terlihat di preview dan **tersimpan permanen di NVS** —
tidak hilang saat alat dimatikan. **"Reset Default"** mengembalikan ke setting default
firmware (bukan pabrik OV2640): brightness/contrast/saturation 0, AWB & AEC & AGC auto,
AE level −2, gain ceiling 2×, quality 40, koreksi pixel/gamma/lensa on.

> 💡 **Tips akurasi:** atur pencahayaan tetap, matikan Auto Exposure dan Auto White
> Balance, lalu gunakan setting yang sama saat mengumpulkan dataset dan prediksi.
> Sensor yang konsisten = model yang akurat.

### Flash Kamera — Penerangan 8 LED RGB (GPIO 47)

Strip **8 LED WS2812 RGB** sebagai lampu penerangan. Tombol **⚡ Flash** muncul di
area kamera (tab Dataset/Prediksi/Kamera) dan di tab Kamera; slider **kecerahan**
(0–255) di tab Kamera. Default **OFF saat boot** (tidak langsung menarik arus),
kecerahan tersimpan di NVS.

> ⚠️ 8 LED putih penuh menarik ~480 mA — kecerahan tinggi bisa memicu brownout.
> Default kecerahan 96; naikkan secukupnya.

---

## API Endpoints

Server **ESPAsyncWebServer** (non-blocking). Aset web (`/`, `*.css`, `*.js`) disajikan
via `serveStatic` dari LittleFS; `chart.min.js` di-cache lama, file lain `no-cache`.

| Method | Endpoint | Deskripsi |
|---|---|---|
| GET | `/` | Web interface utama (LittleFS) |
| GET | `/capture` | JPEG frame live (async chunked, untuk preview & dataset) |
| GET | `/predict` | Jalankan inferensi → JSON `{label, score, time_ms, good}` |
| GET | `/model_info` | Status model `{loaded, size_kb, arena_kb, err}` |
| GET | `/sys` | Diagnostik `{heap_free, heap_min, psram_free, rssi, uptime_s, cap_*}` |
| POST | `/upload_model` | Upload `.tflite` → SD (fallback LittleFS), tolak <1KB → restart |
| GET | `/sd/info` | Status SD `{mounted, good, bad, used_mb, total_mb}` |
| POST | `/sd/capture?label=good\|bad` | Capture → simpan JPEG ke SD `/dataset/` |
| GET | `/sd/list` | Daftar file dataset di SD (JSON) |
| GET | `/sd/file?name=...` | Kirim satu foto dataset dari SD (async) |
| POST | `/sd/delete?name=...` | Hapus satu foto dataset |
| POST | `/sd/relabel?name=...` | Pindah label good ↔ bad (rename di SD) |
| POST | `/sd/upload?label=good\|bad` | Upload JPEG dari PC ke dataset SD |
| GET | `/camera/get` | JSON semua parameter sensor OV2640 |
| POST | `/camera/set?var=...&val=...` | Set parameter kamera → tersimpan di NVS |
| POST | `/camera/reset` | Reset setting kamera → default firmware (restart) |
| GET | `/flash/get` | Status flash `{on, bri}` |
| POST | `/flash/set?on=1&bri=200` | Set flash on/off + kecerahan (8 LED GPIO47) |

---

## Training

### Cara 1 — Otomatis via Web UI (GitHub Actions) ← direkomendasikan

Repo target (`Keyzoo0/EggClassifier` @ `main`) sudah tertanam di web UI — tidak ada
form pengaturan. Butuh **fine-grained PAT** (GitHub → Settings → Developer settings →
Fine-grained tokens → akses repo ini saja, permission **Contents: Read & Write** +
**Actions: Read & Write**). Dua cara menyediakan token:

- **Disarankan** — taruh di `EggClassifierV2/data/secrets.js` (gitignored):
  ```js
  window.GH_TOKEN = "github_pat_...";
  ```
  upload LittleFS → tab Training langsung jalan tanpa input token.
- **Tanpa secrets.js** — web meminta token sekali via prompt, tersimpan di `localStorage`.

> ⚠️ Token **JANGAN** ditaruh di `app.js`/`.ino` atau file lain yang ter-commit —
> token di repo publik **dicabut otomatis** GitHub secret-scanning + tercatat permanen
> di history. `secrets.js` di-gitignore khusus agar token tak pernah ter-push.

Selanjutnya cukup satu klik **"Sinkron + Training + Pasang Model"**.
Training berjalan di runner GitHub (gratis untuk repo publik, ±20–40 menit CPU),
hasilnya di-commit sebagai `model/egg_model.tflite` + `model/model_info.json`,
lalu otomatis dipasang ke alat. Progres bisa dipantau di tab Actions GitHub.

### Cara 2 — Manual via Google Colab

Buka notebook [`training/KlasifikasiTelur_Training.ipynb`](training/KlasifikasiTelur_Training.ipynb) di Google Colab,
lalu pasang model hasilnya via `curl -F "model=@egg_model.tflite" http://telur.local/upload_model`.

### Pipeline Training (kedua cara sama)

1. **Dataset** — format nama file: `good_0001.jpg`, `bad_0001.jpg`, dst.
2. **Augmentasi** — flip, brightness, contrast, saturation, hue, crop-resize
3. **Fase 1** — Train head only (base MobileNetV1 frozen)
4. **Fase 2** — Fine-tune 20 layer terakhir (lr=1e-5)
5. **Konversi INT8** — full integer quantization, representative dataset
6. **Output** — `egg_model.tflite` (CI: + `model_info.json` berisi akurasi & metadata)

### Hasil Training (Dataset 25 foto)

| Metrik | Fase 1 | Fase 2 |
|---|---|---|
| Best Val Accuracy | 100% | 100% |
| Early Stopping | Epoch 23 | Epoch 16 |

> ⚠️ Val accuracy 100% pada dataset 25 foto tidak menjamin performa di dunia nyata. **Target minimal 100 foto per kelas** untuk akurasi yang representatif.

---

## LED Indikator

**LED status onboard (WS2812, GPIO 48):**

| Kondisi | LED |
|---|---|
| Booting / Connecting WiFi | Biru kedip (250ms) |
| WiFi terhubung | Hijau redup solid |
| WiFi putus / reconnecting | Biru kedip |
| Hasil klasifikasi BAGUS | Hijau terang 1,5 detik |
| Hasil klasifikasi TIDAK BAGUS | Merah terang 1,5 detik |

**Strip flash (8 LED WS2812, GPIO 47)** — indikator ikut status WiFi (task `wifiFlashTask`):

| Kondisi | Animasi |
|---|---|
| Connecting / WiFi putus | colorWipe merah → hijau → biru berulang |
| Baru tersambung | Kedip putih 2× → strip diserahkan ke kendali web (flash) |

---

## Pin Mapping — Freenove FNK0085

**Kamera (profil `CAMERA_MODEL_ESP32S3_EYE`):**

| Signal | GPIO |
|---|---|
| XCLK | 15 |
| SDA (SCCB) | 4 |
| SCL (SCCB) | 5 |
| D0–D7 (Y2–Y9) | 11, 9, 8, 10, 12, 18, 17, 16 |
| VSYNC | 6 |
| HREF | 7 |
| PCLK | 13 |

**SD card (SDMMC 1-bit) & LED:**

| Signal | GPIO |
|---|---|
| SD CMD | 38 |
| SD CLK | 39 |
| SD D0 | 40 |
| WS2812 LED status (onboard) | 48 |
| WS2812 strip flash (8 LED) | 47 |

> Tidak ada konflik pin kamera ↔ SD card di board ini — keduanya aktif bersamaan.

## Status Pengembangan

| Fase | Deskripsi | Status |
|---|---|---|
| Phase 1 | Verifikasi hardware — camera test | ✅ DONE |
| Phase 2 | DataCollector — web UI + LittleFS | ✅ DONE |
| Phase 3 | Pengumpulan dataset | ⚠️ 25 foto (target 100+/kelas) |
| Phase 4 | Training MobileNetV1 α=0.25 INT8 | ✅ DONE |
| Phase 5 | EggClassifierV2 — firmware + web + dual-core | ✅ DONE |
| Phase 6 | Optimasi RTOS + PSRAM — zero-alloc inference | ✅ DONE |
| Phase 7 | Migrasi Freenove FNK0085 + SD card + pipeline training GitHub Actions | ✅ DONE |
| Phase 8 | Server async (ESPAsyncWebServer) + flash kamera + token secrets.js + default kamera | ✅ DONE |

---

## Referensi

- [Freenove ESP32-S3-WROOM Board (FNK0085)](https://github.com/Freenove/Freenove_ESP32_S3_WROOM_Board)
- [Dokumentasi FNK0085](https://docs.freenove.com/projects/fnk0085/en/latest/)
- [esp32-camera — Espressif](https://github.com/espressif/esp32-camera)
- [ESPAsyncWebServer — ESP32Async](https://github.com/ESP32Async/ESPAsyncWebServer)
- [arduino-littlefs-upload — earlephilhower](https://github.com/earlephilhower/arduino-littlefs-upload)
- [TensorFlowLite_ESP32 — tanakamasayuki](https://github.com/tanakamasayuki/Arduino_TensorFlowLite_ESP32)
- [MobileNets — Howard et al.](https://arxiv.org/abs/1704.04861)
