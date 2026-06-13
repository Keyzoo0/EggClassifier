#!/usr/bin/env bash
# Regenerasi data/tw.css setelah menambah/ubah class Tailwind di UI.
# Jalankan dari folder ini: ./build.sh
set -e
cd "$(dirname "$0")"
npx -y tailwindcss@3.4.17 -c tailwind.config.js -i input.css -o ../data/tw.css --minify
echo "OK → ../data/tw.css ($(wc -c < ../data/tw.css) bytes)"
