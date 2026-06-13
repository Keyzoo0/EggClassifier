// Build Tailwind purged untuk web UI ESP32 (di-host di LittleFS, bukan CDN).
// Scan class yang dipakai di index.html + app.js → hanya itu yang di-output.
module.exports = {
  content: ['../data/index.html', '../data/app.js'],
  corePlugins: { preflight: true },
  theme: { extend: {} },
};
