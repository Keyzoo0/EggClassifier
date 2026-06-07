#!/usr/bin/env python3
"""
Phase 2 — Data Collection Script

Cara penggunaan:
  1. Upload DataCollector.ino ke ESP32, lihat IP di Serial Monitor
  2. Buka browser ke http://<IP> untuk preview live kamera
  3. Jalankan script ini:
       python collect_data.py --ip 192.168.x.x
  4. Atur telur di depan kamera, lalu tekan:
       g + Enter  = simpan sebagai GOOD (telur bagus)
       b + Enter  = simpan sebagai BAD  (telur cacat/retak/kotor)
       q + Enter  = keluar
"""

import argparse
import os
import sys
import time

try:
    import requests
except ImportError:
    print("[ERROR] Library 'requests' belum terinstall.")
    print("  Jalankan: pip install requests")
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(description="Egg Classifier — Data Collection")
    p.add_argument("--ip",     required=True, help="IP address ESP32 (contoh: 192.168.1.10)")
    p.add_argument("--output", default="datasets", help="Folder output (default: datasets/)")
    p.add_argument("--port",   default=80, type=int, help="Port HTTP (default: 80)")
    return p.parse_args()


def capture(base_url):
    try:
        r = requests.get(f"{base_url}/capture", timeout=5)
        if r.status_code == 200 and r.headers.get("Content-Type", "").startswith("image"):
            return r.content
        print(f"  [WARN] Response status: {r.status_code}")
    except requests.exceptions.ConnectionError:
        print("  [ERROR] Tidak bisa konek ke ESP32. Pastikan WiFi sama dan IP benar.")
    except requests.exceptions.Timeout:
        print("  [ERROR] Timeout. ESP32 tidak merespons.")
    return None


def count_files(folder):
    if not os.path.exists(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.lower().endswith('.jpg')])


def main():
    args = parse_args()

    base_url  = f"http://{args.ip}:{args.port}"
    good_dir  = os.path.join(args.output, "good")
    bad_dir   = os.path.join(args.output, "bad")

    os.makedirs(good_dir, exist_ok=True)
    os.makedirs(bad_dir,  exist_ok=True)

    good_count = count_files(good_dir)
    bad_count  = count_files(bad_dir)

    print(f"\n{'='*48}")
    print(f"  Egg Classifier — Phase 2 Data Collection")
    print(f"{'='*48}")
    print(f"  ESP32 IP     : {args.ip}")
    print(f"  Preview      : {base_url}  (buka di browser)")
    print(f"  Output       : {os.path.abspath(args.output)}/")
    print(f"  Dataset awal : {good_count} GOOD | {bad_count} BAD")
    print(f"{'='*48}")
    print(f"  g + Enter = capture GOOD (telur bagus)")
    print(f"  b + Enter = capture BAD  (telur cacat)")
    print(f"  q + Enter = keluar")
    print(f"{'='*48}\n")

    # Cek koneksi
    print("Mengecek koneksi ke ESP32...", end=" ")
    img = capture(base_url)
    if img is None:
        print("GAGAL")
        print(f"Pastikan ESP32 sudah running dan terhubung ke WiFi yang sama.")
        sys.exit(1)
    print(f"OK ({len(img)} bytes)\n")

    # Loop utama
    while True:
        status = f"[GOOD: {good_count:3d} | BAD: {bad_count:3d}]"
        try:
            cmd = input(f"{status} Perintah (g/b/q): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nDihentikan.")
            break

        if cmd == 'q':
            break

        if cmd not in ('g', 'b'):
            print("  Perintah tidak dikenal. Gunakan g, b, atau q.")
            continue

        print("  Capturing...", end=" ", flush=True)
        t0  = time.time()
        img = capture(base_url)
        elapsed = (time.time() - t0) * 1000

        if img is None:
            continue

        if cmd == 'g':
            good_count += 1
            path = os.path.join(good_dir, f"good_{good_count:04d}.jpg")
            label = "GOOD"
        else:
            bad_count += 1
            path = os.path.join(bad_dir, f"bad_{bad_count:04d}.jpg")
            label = "BAD "

        with open(path, 'wb') as f:
            f.write(img)

        print(f"[{label}] {os.path.basename(path)}  ({len(img):,} bytes, {elapsed:.0f}ms)")

    print(f"\n{'='*48}")
    print(f"  Selesai!")
    print(f"  Total GOOD : {good_count} foto  →  {good_dir}/")
    print(f"  Total BAD  : {bad_count} foto  →  {bad_dir}/")
    target = 300
    if good_count < target or bad_count < target:
        print(f"\n  [INFO] Target minimum: {target} foto/kelas")
        print(f"         Masih perlu: GOOD +{max(0, target-good_count)} | BAD +{max(0, target-bad_count)}")
    else:
        print(f"\n  [OK] Dataset sudah mencukupi untuk training!")
    print(f"{'='*48}\n")


if __name__ == "__main__":
    main()
