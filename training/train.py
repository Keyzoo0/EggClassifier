#!/usr/bin/env python3
"""
Phase 3 — Training MobileNetV1 α=0.25 untuk Klasifikasi Telur
Dataset : flat folder, nama file = label_XXXX.jpg (good_/bad_)
Output  : models/egg_classifier_int8.tflite + models/egg_model.h

Cara pakai lokal (perilaku default tidak berubah):
  cd training/
  source venv/bin/activate
  python train.py

Cara pakai CI (GitHub Actions, dari root repo):
  python training/train.py --dataset dataset --out training/models \
    --repeat 8 --epochs1 30 --epochs2 20 --batch 16 --export-dir model
"""

import os, sys, shutil, struct, pathlib, argparse, json
from datetime import datetime, timezone
import numpy as np

# ── Cek dependensi ────────────────────────────────────────────
try:
    import tensorflow as tf
    from PIL import Image
except ImportError as e:
    print(f"[ERROR] Dependensi kurang: {e}")
    print("  Jalankan: pip install tensorflow pillow numpy")
    sys.exit(1)

print(f"TensorFlow {tf.__version__} | Python {sys.version.split()[0]}\n")

# ── Konfigurasi (default = perilaku lama untuk pemakaian lokal) ──
ap = argparse.ArgumentParser(description="Training klasifikasi telur")
ap.add_argument("--dataset",    default="dataset", help="folder dataset flat (good_*.jpg / bad_*.jpg)")
ap.add_argument("--out",        default="models",  help="folder output model")
ap.add_argument("--batch",      type=int, default=4)
ap.add_argument("--epochs1",    type=int, default=60, help="fase 1: train head")
ap.add_argument("--epochs2",    type=int, default=40, help="fase 2: fine-tune")
ap.add_argument("--repeat",     type=int, default=20, help="augmentasi: repeat dataset per epoch")
ap.add_argument("--export-dir", default=None,
                help="jika diisi: salin egg_model.tflite + model_info.json ke folder ini (untuk CI)")
args = ap.parse_args()

IMG_SIZE   = 96             # Input model ESP32: 96×96
BATCH_SIZE = args.batch
EPOCHS_1   = args.epochs1   # Fase 1: hanya head
EPOCHS_2   = args.epochs2   # Fase 2: fine-tune top layers
ALPHA      = 0.25           # MobileNetV1 width multiplier
DATASET_DIR = pathlib.Path(args.dataset)
MODEL_DIR   = pathlib.Path(args.out)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Baca dataset dari flat folder ─────────────────────────
print("=" * 55)
print("1. MEMBACA DATASET")
print("=" * 55)

all_files, all_labels = [], []
for f in sorted(DATASET_DIR.iterdir()):
    if not f.suffix.lower() in ('.jpg', '.jpeg', '.png'):
        continue
    name = f.name.lower()
    if name.startswith('good'):
        label = 1
    elif name.startswith('bad'):
        label = 0
    else:
        print(f"  [SKIP] File tidak dikenal: {f.name}")
        continue
    all_files.append(str(f))
    all_labels.append(label)

n_good = sum(all_labels)
n_bad  = len(all_labels) - n_good
print(f"  GOOD : {n_good} gambar")
print(f"  BAD  : {n_bad} gambar")
print(f"  Total: {len(all_files)} gambar\n")

if len(all_files) < 4:
    print("[ERROR] Dataset terlalu sedikit (minimal 4 gambar).")
    sys.exit(1)

# ── 2. Split train/val (80/20, stratified manual) ─────────────
print("=" * 55)
print("2. SPLIT TRAIN / VALIDASI")
print("=" * 55)

rng = np.random.default_rng(42)
idx_good = [i for i, l in enumerate(all_labels) if l == 1]
idx_bad  = [i for i, l in enumerate(all_labels) if l == 0]

def split_indices(idx, val_ratio=0.2):
    idx = rng.permutation(idx).tolist()
    n_val = max(1, round(len(idx) * val_ratio))
    return idx[n_val:], idx[:n_val]   # train, val

tr_good, va_good = split_indices(idx_good)
tr_bad,  va_bad  = split_indices(idx_bad)

train_idx = tr_good + tr_bad
val_idx   = va_good + va_bad
rng.shuffle(train_idx)

train_files  = [all_files[i]  for i in train_idx]
train_labels = [all_labels[i] for i in train_idx]
val_files    = [all_files[i]  for i in val_idx]
val_labels   = [all_labels[i] for i in val_idx]

print(f"  Train : {len(train_files)} gambar  (GOOD={sum(train_labels)}, BAD={len(train_labels)-sum(train_labels)})")
print(f"  Val   : {len(val_files)} gambar  (GOOD={sum(val_labels)}, BAD={len(val_labels)-sum(val_labels)})\n")

# ── 3. Pipeline dataset ───────────────────────────────────────
print("=" * 55)
print("3. MEMBUAT PIPELINE AUGMENTASI")
print("=" * 55)

def load_image(path, label):
    raw   = tf.io.read_file(path)
    img   = tf.image.decode_jpeg(raw, channels=3)
    img   = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img   = tf.cast(img, tf.float32) / 255.0
    return img, tf.cast(label, tf.float32)

def augment(img, label):
    # Flip
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    # Warna
    img = tf.image.random_brightness(img, max_delta=0.4)
    img = tf.image.random_contrast(img, lower=0.6, upper=1.4)
    img = tf.image.random_saturation(img, lower=0.6, upper=1.4)
    img = tf.image.random_hue(img, max_delta=0.1)
    # Rotasi + zoom via crop-resize
    img = tf.image.random_crop(tf.pad(img, [[8,8],[8,8],[0,0]]), [IMG_SIZE, IMG_SIZE, 3])
    # Clamp
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label

train_files_t  = tf.constant(train_files)
train_labels_t = tf.constant(train_labels, dtype=tf.int32)
val_files_t    = tf.constant(val_files)
val_labels_t   = tf.constant(val_labels,   dtype=tf.int32)

AUTOTUNE = tf.data.AUTOTUNE

# Repeat dataset agar augmentasi menghasilkan lebih banyak sampel
REPEAT = args.repeat   # Tiap epoch = N× data asli

train_ds = (
    tf.data.Dataset.from_tensor_slices((train_files_t, train_labels_t))
    .map(load_image, num_parallel_calls=AUTOTUNE)
    .cache()
    .repeat(REPEAT)
    .map(augment, num_parallel_calls=AUTOTUNE)
    .shuffle(512)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((val_files_t, val_labels_t))
    .map(load_image, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

steps_per_epoch = (len(train_files) * REPEAT) // BATCH_SIZE
print(f"  Augmentasi repeat × {REPEAT}  →  ~{len(train_files)*REPEAT} sampel/epoch")
print(f"  Steps per epoch : {steps_per_epoch}\n")

# ── 4. Bangun model ───────────────────────────────────────────
print("=" * 55)
print("4. MEMBANGUN MODEL MobileNetV1 α=0.25")
print("=" * 55)

base = tf.keras.applications.MobileNet(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    alpha=ALPHA,
    include_top=False,
    weights='imagenet',
)
base.trainable = False   # Fase 1: freeze semua

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs, outputs)
model.summary(line_length=60)
print()

# ── 5. Fase 1: Train head ─────────────────────────────────────
print("=" * 55)
print(f"5. FASE 1 — Train head ({EPOCHS_1} epoch, base frozen)")
print("=" * 55)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy'],
)

cb_best1 = tf.keras.callbacks.ModelCheckpoint(
    str(MODEL_DIR / "best_phase1.h5"),
    monitor='val_accuracy', save_best_only=True, verbose=0
)
cb_early1 = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', patience=20, restore_best_weights=True, verbose=1
)
cb_lr1 = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1
)

hist1 = model.fit(
    train_ds,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS_1,
    validation_data=val_ds,
    callbacks=[cb_best1, cb_early1, cb_lr1],
    verbose=1,
)

best_val1 = max(hist1.history.get('val_accuracy', [0]))
print(f"\n  Best val accuracy fase 1: {best_val1*100:.1f}%\n")

# ── 6. Fase 2: Fine-tune top layers ──────────────────────────
print("=" * 55)
print(f"6. FASE 2 — Fine-tune top 20 layers ({EPOCHS_2} epoch)")
print("=" * 55)

base.trainable = True
# Bekukan semua kecuali 20 layer terakhir
for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy'],
)

cb_best2 = tf.keras.callbacks.ModelCheckpoint(
    str(MODEL_DIR / "best_phase2.h5"),
    monitor='val_accuracy', save_best_only=True, verbose=0
)
cb_early2 = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1
)

hist2 = model.fit(
    train_ds,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS_2,
    validation_data=val_ds,
    callbacks=[cb_best2, cb_early2],
    verbose=1,
)

best_val2 = max(hist2.history.get('val_accuracy', [0]))
print(f"\n  Best val accuracy fase 2: {best_val2*100:.1f}%")
final_val  = max(best_val1, best_val2)
print(f"  Final best val accuracy : {final_val*100:.1f}%\n")

# ── 7. Konversi ke TFLite INT8 ────────────────────────────────
print("=" * 55)
print("7. KONVERSI TFLite FLOAT32 + INT8")
print("=" * 55)

# Simpan model Keras dulu
keras_path = str(MODEL_DIR / "egg_classifier.h5")
model.save(keras_path)
print(f"  Keras model tersimpan: {keras_path}")

# 7a. Float32 TFLite
converter_f = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_f    = converter_f.convert()
path_f32    = MODEL_DIR / "egg_classifier_float32.tflite"
path_f32.write_bytes(tflite_f)
print(f"  Float32 TFLite: {path_f32}  ({len(tflite_f)/1024:.1f} KB)")

# 7b. INT8 quantization — butuh representative dataset
def representative_dataset():
    for path in all_files:
        img = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32) / 255.0
        yield [tf.expand_dims(img, 0)]

converter_q = tf.lite.TFLiteConverter.from_keras_model(model)
converter_q.optimizations             = [tf.lite.Optimize.DEFAULT]
converter_q.representative_dataset    = representative_dataset
converter_q.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_q.inference_input_type      = tf.int8
converter_q.inference_output_type     = tf.int8

tflite_q = converter_q.convert()
path_int8 = MODEL_DIR / "egg_classifier_int8.tflite"
path_int8.write_bytes(tflite_q)
print(f"  INT8 TFLite:    {path_int8}  ({len(tflite_q)/1024:.1f} KB)\n")

# ── 8. Generate C header ──────────────────────────────────────
print("=" * 55)
print("8. GENERATE C HEADER  egg_model.h")
print("=" * 55)

model_data = path_int8.read_bytes()
hex_lines  = []
for i in range(0, len(model_data), 12):
    chunk = model_data[i:i+12]
    hex_lines.append("  " + ", ".join(f"0x{b:02x}" for b in chunk))

header = f"""// Auto-generated by train.py — JANGAN EDIT MANUAL
// Model : MobileNetV1 α={ALPHA}, 96×96 INT8
// Val accuracy: {final_val*100:.1f}%
// Size  : {len(model_data)} bytes ({len(model_data)/1024:.1f} KB)

#pragma once
#include <stdint.h>

const unsigned int g_model_len = {len(model_data)};

alignas(8) const uint8_t g_model_data[] = {{
{chr(10).join(hex_lines)}
}};
"""

header_path = MODEL_DIR / "egg_model.h"
header_path.write_text(header)
print(f"  Header: {header_path}")
print(f"  Array : g_model_data[{len(model_data)}]  ({len(model_data)/1024:.1f} KB)\n")

# ── 9. Evaluasi akhir ─────────────────────────────────────────
print("=" * 55)
print("9. EVALUASI AKHIR")
print("=" * 55)

loss, acc = model.evaluate(val_ds, verbose=0)
print(f"  Val Loss     : {loss:.4f}")
print(f"  Val Accuracy : {acc*100:.1f}%")

# ── 10. Export untuk pipeline CI (--export-dir) ───────────────
if args.export_dir:
    export_dir = pathlib.Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(path_int8, export_dir / "egg_model.tflite")
    info = {
        "val_accuracy": round(float(final_val), 4),
        "size_kb":      round(len(tflite_q) / 1024, 1),
        "n_good":       n_good,
        "n_bad":        n_bad,
        "alpha":        ALPHA,
        "img_size":     IMG_SIZE,
        "trained_at":   datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
    (export_dir / "model_info.json").write_text(json.dumps(info, indent=2))
    print(f"\n  [CI] Export: {export_dir}/egg_model.tflite + model_info.json")

print("\n" + "=" * 55)
print("SELESAI!")
print("=" * 55)
print(f"  Model INT8  : {path_int8}")
print(f"  C Header    : {header_path}")
print()
print("  Salin egg_model.h ke folder sketch Arduino,")
print("  lalu lanjut ke Phase 5 — Inference Firmware.")
if final_val < 0.85:
    print()
    print("  [PERINGATAN] Akurasi < 85% — dataset terlalu sedikit.")
    print("  Tambah lebih banyak foto (target 100+ per kelas)")
    print("  lalu jalankan train.py lagi.")
print("=" * 55)
