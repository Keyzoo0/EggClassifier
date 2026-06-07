// ─────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────
const TARGET       = 300;
const PREVIEW_MS   = 800;
let counts         = { good: 0, bad: 0 };
let activeTab      = 'dataset';
let isCapturing    = false;
let isClassifying  = false;
let frameCount     = 0;
let previewOk      = false;
let predictHistory = [];

// ─────────────────────────────────────────────────────────────
// Tab
// ─────────────────────────────────────────────────────────────
function switchTab(tab) {
  activeTab = tab;
  ['dataset', 'predict'].forEach(t => {
    document.getElementById('tab-' + t).classList.toggle('hidden', t !== tab);
    const btn = document.getElementById('btn-' + t);
    btn.className = btn.className.replace(/tab-(active|inactive)/g, '');
    btn.classList.add(t === tab ? 'tab-active' : 'tab-inactive');
  });
}

// ─────────────────────────────────────────────────────────────
// Live Preview
// ─────────────────────────────────────────────────────────────
function refreshPreview() {
  if (isClassifying) return;
  const tmp = new Image();
  tmp.onload = () => {
    const el = document.getElementById('preview');
    el.src = tmp.src;
    document.getElementById('frame-count').textContent = ++frameCount;
    if (!previewOk) {
      previewOk = true;
      document.getElementById('overlay-loading').style.display = 'none';
      setWifiStatus(true);
    }
  };
  tmp.onerror = () => { if (previewOk) setWifiStatus(false); };
  tmp.src = '/capture?t=' + Date.now();
}

function previewError() {
  setWifiStatus(false);
}

function setWifiStatus(ok) {
  document.getElementById('wifi-dot').className =
    'w-2 h-2 rounded-full ' + (ok ? 'bg-green-400' : 'bg-red-400 animate-pulse');
  document.getElementById('wifi-text').textContent = ok ? 'Terhubung' : 'Tidak terhubung';
}

setInterval(refreshPreview, PREVIEW_MS);

// ─────────────────────────────────────────────────────────────
// Dataset: capture → simpan ke SD (server-side)
// ─────────────────────────────────────────────────────────────
async function captureImage(label) {
  if (isCapturing) return;
  isCapturing = true;

  const btn = document.getElementById(label + '-btn');
  btn.disabled = true;
  btn.classList.add(label === 'good' ? 'flash-good' : 'flash-bad');
  setTimeout(() => btn.classList.remove('flash-good', 'flash-bad'), 400);

  try {
    const res  = await fetch('/save_image?label=' + label + '&t=' + Date.now());
    const data = await res.json();

    if (!res.ok || data.error) {
      const msg = data.error === 'sd_not_ready'
        ? 'SD card tidak terhubung!'
        : 'Gagal simpan: ' + (data.error || res.status);
      showToast(msg, 'error');
      return;
    }

    counts[label] = data.count;
    updateCounters();
    addThumb(label, data.count);
    showToast(
      (label === 'good' ? '✅ BAGUS' : '❌ CACAT') + ' #' + data.count + ' → SD card',
      label
    );
  } catch (e) {
    showToast('Error: ' + e.message, 'error');
  } finally {
    btn.disabled = false;
    isCapturing  = false;
  }
}

function updateCounters() {
  ['good', 'bad'].forEach(l => {
    document.getElementById(l + '-count').textContent = counts[l];
    const pct = Math.min(100, (counts[l] / TARGET) * 100).toFixed(1);
    document.getElementById(l + '-bar').style.width = pct + '%';
  });

  const remaining = Math.max(0, TARGET - counts.good) + Math.max(0, TARGET - counts.bad);
  const info = document.getElementById('progress-info');
  if (counts.good >= TARGET && counts.bad >= TARGET) {
    info.innerHTML = '<span class="text-green-400 font-semibold">✓ Dataset lengkap! Siap training.</span>';
  } else {
    info.textContent = 'Perlu ' + remaining + ' foto lagi untuk mencapai target 300/kelas';
  }
}

// ─────────────────────────────────────────────────────────────
// SD Stats (polling 10 detik)
// ─────────────────────────────────────────────────────────────
async function loadSDStats() {
  try {
    const res  = await fetch('/sd_stats');
    const data = await res.json();

    document.getElementById('sd-warning').classList.toggle('hidden', data.ready);

    if (data.ready) {
      counts.good = data.good;
      counts.bad  = data.bad;
      updateCounters();
      document.getElementById('sd-badge').textContent =
        data.free_mb + ' MB bebas';
    }
  } catch (_) {}
}

loadSDStats();
setInterval(loadSDStats, 10000);

// ─────────────────────────────────────────────────────────────
// Thumbnails
// ─────────────────────────────────────────────────────────────
function addThumb(label, n) {
  const grid  = document.getElementById('thumb-grid');
  const empty = document.getElementById('thumb-empty');
  empty.classList.add('hidden');

  const tmp = new Image();
  tmp.onload = () => {
    const wrap = document.createElement('div');
    wrap.className = 'relative rounded-lg overflow-hidden bg-slate-900 aspect-square';
    wrap.innerHTML =
      `<img src="${tmp.src}" class="w-full h-full object-cover">` +
      `<span class="absolute bottom-0 inset-x-0 text-center text-xs font-bold py-0.5 opacity-90
                    ${label === 'good' ? 'bg-green-700' : 'bg-red-700'} text-white">#${n}</span>`;
    grid.insertBefore(wrap, grid.firstChild);
    while (grid.children.length > 6) grid.removeChild(grid.lastChild);
  };
  tmp.src = '/capture?t=' + Date.now();
}

// ─────────────────────────────────────────────────────────────
// Predict: klasifikasi real-time
// ─────────────────────────────────────────────────────────────
async function runPredict() {
  if (isClassifying) return;
  isClassifying = true;

  const btn = document.getElementById('predict-btn');
  btn.disabled = true;
  btn.textContent = '⏳ Menganalisis...';

  try {
    const res  = await fetch('/predict?t=' + Date.now());
    const data = await res.json();

    if (data.error) {
      if (data.error === 'no_model') {
        showToast('Upload model .tflite dulu!', 'error');
        document.getElementById('no-model-warning').classList.remove('hidden');
      } else {
        showToast('Error: ' + data.error, 'error');
      }
      return;
    }

    showResult(data);
    addHistory(data);

  } catch (e) {
    showToast('Gagal: ' + e.message, 'error');
  } finally {
    isClassifying = false;
    btn.disabled  = false;
    btn.innerHTML = '🔍 Klasifikasi  <span class="text-sm font-normal opacity-70">(C)</span>';
  }
}

function showResult(data) {
  const card  = document.getElementById('result-card');
  const label = document.getElementById('result-label');
  const score = document.getElementById('result-score');
  const ring  = document.getElementById('result-score-ring');
  const conf  = document.getElementById('result-confidence');
  const time  = document.getElementById('result-time');

  card.classList.remove('hidden');

  label.textContent = data.label;
  label.className   = 'text-4xl font-black ' +
    (data.good ? 'text-green-400' : 'text-red-400');

  const pct = (data.score * 100).toFixed(1);
  score.textContent = pct + '%';
  ring.className    = ring.className.replace(/border-\S+/g, '') +
    ' border-' + (data.good ? 'green-500' : 'red-500');

  const confLevel =
    data.score > 0.85 || data.score < 0.15 ? 'TINGGI' :
    data.score > 0.70 || data.score < 0.30 ? 'SEDANG' : 'RENDAH';

  conf.textContent = confLevel;
  conf.className   = 'font-semibold ' + (
    confLevel === 'TINGGI' ? 'text-green-400' :
    confLevel === 'SEDANG' ? 'text-yellow-400' : 'text-red-400');

  time.textContent = data.time_ms + ' ms';

  showToast(
    (data.good ? '✅ ' : '❌ ') + data.label + ' (' + pct + '%)',
    data.good ? 'good' : 'bad'
  );
}

function addHistory(data) {
  predictHistory.unshift(data);
  if (predictHistory.length > 6) predictHistory.pop();

  const box = document.getElementById('history-box');
  const list = document.getElementById('history-list');
  box.classList.remove('hidden');
  list.innerHTML = '';

  predictHistory.forEach(d => {
    const el = document.createElement('div');
    el.className = 'flex justify-between items-center px-3 py-1.5 rounded-lg bg-slate-700/40 text-sm';
    el.innerHTML =
      `<span class="font-semibold ${d.good ? 'text-green-400' : 'text-red-400'}">${d.label}</span>` +
      `<span class="text-slate-400">${(d.score*100).toFixed(1)}% · ${d.time_ms} ms</span>`;
    list.appendChild(el);
  });
}

// ─────────────────────────────────────────────────────────────
// Model: info + upload
// ─────────────────────────────────────────────────────────────
async function loadModelInfo() {
  try {
    const res  = await fetch('/model_info');
    const data = await res.json();
    const el   = document.getElementById('model-status');
    const warn = document.getElementById('no-model-warning');

    if (data.loaded) {
      el.textContent = 'egg_model.tflite (' + data.size_kb + ' KB) ✓';
      el.className   = 'text-xs text-green-400';
      warn.classList.add('hidden');
    } else {
      el.textContent = 'Tidak ada model';
      el.className   = 'text-xs text-red-400';
      warn.classList.remove('hidden');
    }
  } catch (_) {}
}

function onFileSelected(input) {
  const file = input.files[0];
  document.getElementById('file-label').textContent =
    file ? file.name + ' (' + (file.size / 1024).toFixed(0) + ' KB)' : 'Pilih file .tflite ...';
  document.getElementById('upload-btn').disabled = !file;
}

async function uploadModel() {
  const fileInput = document.getElementById('model-file');
  const file = fileInput.files[0];
  if (!file) { showToast('Pilih file .tflite dulu', 'error'); return; }

  const btn      = document.getElementById('upload-btn');
  const progress = document.getElementById('upload-progress');
  const fill     = document.getElementById('upload-fill');
  const status   = document.getElementById('upload-status');

  btn.disabled = true;
  progress.classList.remove('hidden');
  fill.style.width = '0%';
  status.textContent = 'Mengunggah...';

  try {
    const fd  = new FormData();
    fd.append('model', file);

    const result = await new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.upload.onprogress = e => {
        if (e.lengthComputable) {
          const pct = Math.round(e.loaded / e.total * 100);
          fill.style.width = pct + '%';
          status.textContent = 'Mengunggah... ' + pct + '%';
        }
      };
      xhr.onload  = () => {
        try { resolve(JSON.parse(xhr.responseText)); }
        catch { resolve({ ok: false, error: 'parse_error' }); }
      };
      xhr.onerror = () => reject(new Error('Network error'));
      xhr.open('POST', '/upload_model');
      xhr.send(fd);
    });

    if (result.ok) {
      fill.style.width = '100%';
      status.textContent = '✅ Tersimpan (' + result.size_kb + ' KB) — Board restart...';
      showToast('Model diupload! Menunggu restart...', 'good');

      // Poll sampai board kembali online
      await waitForRestart();
      status.textContent = '✅ Model aktif!';
      showToast('✅ Model baru aktif!', 'good');
      loadModelInfo();
      previewOk = false;
      document.getElementById('overlay-loading').style.display = '';
    } else {
      status.textContent = 'Gagal: ' + (result.error || 'unknown');
      showToast('Upload gagal', 'error');
      btn.disabled = false;
    }

  } catch (e) {
    status.textContent = 'Error: ' + e.message;
    showToast('Error: ' + e.message, 'error');
    btn.disabled = false;
  }
}

async function waitForRestart(maxWait = 15000) {
  const t0 = Date.now();
  await new Promise(r => setTimeout(r, 2500));  // beri waktu restart
  while (Date.now() - t0 < maxWait) {
    try {
      await fetch('/capture?t=' + Date.now());
      return;  // board online lagi
    } catch (_) {
      await new Promise(r => setTimeout(r, 800));
    }
  }
}

loadModelInfo();

// ─────────────────────────────────────────────────────────────
// Toast
// ─────────────────────────────────────────────────────────────
function showToast(msg, type) {
  const colors = { good:'bg-green-700', bad:'bg-red-700', error:'bg-amber-700' };
  const el = document.createElement('div');
  el.className = `${colors[type] || 'bg-slate-700'} text-white text-sm px-4 py-2
                  rounded-lg shadow-lg pointer-events-auto transition-all duration-300
                  translate-y-2 opacity-0`;
  el.textContent = msg;
  document.getElementById('toast-container').appendChild(el);
  requestAnimationFrame(() => el.classList.remove('translate-y-2', 'opacity-0'));
  setTimeout(() => {
    el.classList.add('opacity-0', 'translate-y-2');
    setTimeout(() => el.remove(), 300);
  }, 2800);
}

// ─────────────────────────────────────────────────────────────
// Keyboard shortcuts
// ─────────────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.repeat) return;
  if (e.key === 'g' || e.key === 'G') captureImage('good');
  if (e.key === 'b' || e.key === 'B') captureImage('bad');
  if (e.key === 'c' || e.key === 'C') runPredict();
  if (e.key === '1') switchTab('dataset');
  if (e.key === '2') switchTab('predict');
});
