// ─── State ────────────────────────────────────────────────────
const TARGET      = 300;
const PREVIEW_MS  = 850;
let counts        = { good: 0, bad: 0 };
let activeTab     = 'dataset';
let isCapturing   = false;
let isClassifying = false;
let frameCount    = 0;
let previewOk     = false;
let history       = [];
let scoreChart    = null;

// ─── Score Chart (Chart.js donut) ────────────────────────────
function initChart() {
  const ctx = document.getElementById('score-chart').getContext('2d');
  scoreChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      datasets: [{
        data: [0, 100],
        backgroundColor: ['#6366f1', 'rgba(255,255,255,0.05)'],
        borderWidth: 0,
        borderRadius: 3,
      }]
    },
    options: {
      cutout: '76%',
      animation: { duration: 700, easing: 'easeOutQuart' },
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
    }
  });
}

function updateChart(score, isGood) {
  if (!scoreChart) return;
  const pct = Math.round(score * 100);
  const filled = isGood ? pct : (100 - pct);
  const color  = isGood ? '#4ade80' : '#f87171';
  scoreChart.data.datasets[0].data = [filled, 100 - filled];
  scoreChart.data.datasets[0].backgroundColor = [color, 'rgba(255,255,255,0.05)'];
  scoreChart.update();
}

// ─── Tab switching ────────────────────────────────────────────
function switchTab(tab) {
  activeTab = tab;
  ['dataset', 'predict'].forEach(t => {
    document.getElementById('tab-' + t).classList.toggle('hidden', t !== tab);
    document.getElementById('btn-' + t).classList.toggle('active', t === tab);
  });
  if (tab === 'predict' && !scoreChart) initChart();
}

// ─── Live Preview ─────────────────────────────────────────────
function refreshPreview() {
  if (isClassifying) return;
  const tmp = new Image();
  tmp.onload = () => {
    document.getElementById('preview').src = tmp.src;
    document.getElementById('frame-count').textContent = ++frameCount;
    if (!previewOk) {
      previewOk = true;
      document.getElementById('overlay-loading').style.display = 'none';
      setWifi(true);
    }
  };
  tmp.onerror = () => { if (previewOk) setWifi(false); };
  tmp.src = '/capture?t=' + Date.now();
}

function previewError() { setWifi(false); }

function setWifi(ok) {
  const dot  = document.getElementById('wifi-dot');
  const text = document.getElementById('wifi-text');
  if (ok) {
    dot.style.background = '#4ade80';
    dot.classList.remove('dot-pulse');
    text.textContent = 'Terhubung';
    text.style.color = '#86efac';
  } else {
    dot.style.background = '#f87171';
    dot.classList.add('dot-pulse');
    text.textContent = 'Tidak terhubung';
    text.style.color = '#fca5a5';
  }
}

setInterval(refreshPreview, PREVIEW_MS);

// ─── Dataset: capture & save ──────────────────────────────────
async function captureImage(label) {
  if (isCapturing) return;
  isCapturing = true;

  const btn = document.getElementById(label + '-btn');
  btn.disabled = true;

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

  const rem  = Math.max(0, TARGET - counts.good) + Math.max(0, TARGET - counts.bad);
  const info = document.getElementById('progress-info');
  if (counts.good >= TARGET && counts.bad >= TARGET) {
    info.innerHTML =
      '<span style="color:var(--success);font-weight:700">✓ Dataset lengkap! Siap training.</span>';
  } else {
    info.textContent = 'Perlu ' + rem + ' foto lagi untuk target 300/kelas';
  }
}

// ─── SD Stats ─────────────────────────────────────────────────
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
        data.free_mb + ' MB bebas · total ' + data.total_mb + ' MB';
    }
  } catch (_) {}
}

loadSDStats();
setInterval(loadSDStats, 10000);

// ─── Thumbnails ───────────────────────────────────────────────
function addThumb(label, n) {
  const grid  = document.getElementById('thumb-grid');
  const empty = document.getElementById('thumb-empty');
  empty.classList.add('hidden');

  const img = new Image();
  img.onload = () => {
    const wrap = document.createElement('div');
    wrap.className = 'thumb';
    const imgEl = document.createElement('img');
    imgEl.src = img.src;
    imgEl.alt = label;
    const lbl = document.createElement('div');
    lbl.className = 'thumb-label ' + (label === 'good' ? 'thumb-label-good' : 'thumb-label-bad');
    lbl.textContent = '#' + n;
    wrap.appendChild(imgEl);
    wrap.appendChild(lbl);
    grid.insertBefore(wrap, grid.firstChild);
    while (grid.children.length > 6) grid.removeChild(grid.lastChild);
  };
  img.src = '/capture?t=' + Date.now();
}

// ─── Predict ──────────────────────────────────────────────────
async function runPredict() {
  if (isClassifying) return;
  if (!scoreChart) initChart();
  isClassifying = true;

  const btn = document.getElementById('predict-btn');
  btn.disabled = true;
  btn.textContent = '⏳  Menganalisis...';
  btn.classList.add('btn-classify-busy');

  document.getElementById('busy-badge').classList.remove('hidden');

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
    btn.innerHTML =
      '🔍 &nbsp;Klasifikasi' +
      '<span style="font-size:14px;font-weight:400;opacity:0.55;margin-left:4px">(C)</span>';
    btn.classList.remove('btn-classify-busy');
    document.getElementById('busy-badge').classList.add('hidden');
  }
}

function showResult(data) {
  const card = document.getElementById('result-card');
  card.classList.remove('hidden');
  card.classList.remove('result-reveal');
  void card.offsetWidth;  // reflow to retrigger animation
  card.classList.add('result-reveal');

  // Donut chart
  updateChart(data.score, data.good);

  // Pulse ring on chart wrap
  const wrap = document.getElementById('chart-wrap');
  wrap.classList.remove('ring-good', 'ring-bad');
  void wrap.offsetWidth;
  wrap.classList.add(data.good ? 'ring-good' : 'ring-bad');

  // Score text
  const pct = (data.score * 100).toFixed(1);
  document.getElementById('result-score').textContent = pct + '%';
  document.getElementById('result-score').style.color =
    data.good ? 'var(--success)' : 'var(--danger)';

  // Label
  const label = document.getElementById('result-label');
  label.textContent = data.label;
  label.style.color = data.good ? 'var(--success)' : 'var(--danger)';

  // Confidence
  const confLevel =
    data.score > 0.85 || data.score < 0.15 ? 'TINGGI' :
    data.score > 0.70 || data.score < 0.30 ? 'SEDANG' : 'RENDAH';
  const confEl = document.getElementById('result-confidence');
  confEl.textContent = confLevel;
  confEl.style.color =
    confLevel === 'TINGGI' ? 'var(--success)' :
    confLevel === 'SEDANG' ? 'var(--warn)'    : 'var(--danger)';

  // Time
  document.getElementById('result-time').textContent = data.time_ms + ' ms';

  showToast(
    (data.good ? '✅ ' : '❌ ') + data.label + ' — ' + pct + '%',
    data.good ? 'good' : 'bad'
  );
}

function addHistory(data) {
  history.unshift(data);
  if (history.length > 8) history.pop();

  const box  = document.getElementById('history-box');
  const list = document.getElementById('history-list');
  box.classList.remove('hidden');
  list.innerHTML = '';

  history.forEach((d, i) => {
    const pct  = (d.score * 100).toFixed(1);
    const isGd = d.good;
    const el   = document.createElement('div');
    el.className = 'history-item';
    el.innerHTML =
      `<div class="flex items-center gap-2">` +
        `<span class="hist-badge" style="` +
          `background:${isGd ? 'rgba(74,222,128,0.12)' : 'rgba(248,113,113,0.12)'};` +
          `color:${isGd ? '#86efac' : '#fca5a5'};` +
          `border:1px solid ${isGd ? 'rgba(74,222,128,0.25)' : 'rgba(248,113,113,0.25)'}">` +
          `${isGd ? 'BAGUS' : 'CACAT'}` +
        `</span>` +
        `<span style="font-size:12px;color:var(--text-2)">#${history.length - i}</span>` +
      `</div>` +
      `<div class="text-right" style="font-size:12px">` +
        `<span class="font-bold font-mono" style="color:${isGd ? 'var(--success)' : 'var(--danger)'}">${pct}%</span>` +
        `<span style="color:var(--text-3);margin-left:8px">${d.time_ms} ms</span>` +
      `</div>`;
    list.appendChild(el);
  });
}

// ─── Model: info + upload ─────────────────────────────────────
async function loadModelInfo() {
  try {
    const res  = await fetch('/model_info');
    const data = await res.json();
    const el   = document.getElementById('model-status');
    const warn = document.getElementById('no-model-warning');

    if (data.loaded) {
      el.textContent = '✓ ' + data.size_kb + ' KB';
      el.className   = 'badge-loaded';
      warn.classList.add('hidden');
    } else {
      el.textContent = 'Tidak ada model';
      el.className   = 'badge-missing';
      warn.classList.remove('hidden');
    }
  } catch (_) {}
}

function onFileSelected(input) {
  const file = input.files[0];
  const lbl  = document.getElementById('file-label');
  const btn  = document.getElementById('upload-btn');
  if (file) {
    lbl.textContent = file.name + ' — ' + (file.size / 1024).toFixed(0) + ' KB';
    lbl.style.color = 'var(--text-1)';
  } else {
    lbl.textContent = 'Pilih atau drop file .tflite';
    lbl.style.color = 'var(--text-2)';
  }
  btn.disabled = !file;
}

async function uploadModel() {
  const file = document.getElementById('model-file').files[0];
  if (!file) { showToast('Pilih file .tflite dulu', 'error'); return; }

  const btn      = document.getElementById('upload-btn');
  const progress = document.getElementById('upload-progress');
  const fill     = document.getElementById('upload-fill');
  const status   = document.getElementById('upload-status');

  btn.disabled = true;
  progress.classList.remove('hidden');
  fill.style.width   = '0%';
  status.textContent = 'Mengunggah...';

  try {
    const fd = new FormData();
    fd.append('model', file);

    const result = await new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.upload.onprogress = e => {
        if (e.lengthComputable) {
          const pct = Math.round(e.loaded / e.total * 100);
          fill.style.width   = pct + '%';
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
      fill.style.width   = '100%';
      status.textContent = '✅ Tersimpan (' + result.size_kb + ' KB) — Menunggu restart...';
      showToast('Model diupload! Menunggu restart...', 'good');

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
  await new Promise(r => setTimeout(r, 2500));
  while (Date.now() - t0 < maxWait) {
    try { await fetch('/capture?t=' + Date.now()); return; }
    catch (_) { await new Promise(r => setTimeout(r, 800)); }
  }
}

loadModelInfo();

// ─── Drag-and-drop on dropzone ───────────────────────────────
const dz = document.getElementById('dropzone-area');
if (dz) {
  dz.addEventListener('dragover', e => {
    e.preventDefault();
    dz.classList.add('drag-over');
  });
  dz.addEventListener('dragleave', () => dz.classList.remove('drag-over'));
  dz.addEventListener('drop', e => {
    e.preventDefault();
    dz.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.name.endsWith('.tflite')) {
      const input = document.getElementById('model-file');
      const dt = new DataTransfer();
      dt.items.add(file);
      input.files = dt.files;
      onFileSelected(input);
    } else {
      showToast('File harus berekstensi .tflite', 'error');
    }
  });
}

// ─── Toast ────────────────────────────────────────────────────
function showToast(msg, type) {
  const container = document.getElementById('toast-container');
  const el = document.createElement('div');
  el.className = 'toast toast-' + (type === 'good' ? 'good' : type === 'bad' ? 'bad' : 'error');
  el.textContent = msg;
  container.appendChild(el);

  setTimeout(() => {
    el.classList.add('toast-exit');
    setTimeout(() => el.remove(), 220);
  }, 2800);
}

// ─── Keyboard shortcuts ───────────────────────────────────────
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.repeat) return;
  const k = e.key;
  if (k === 'g' || k === 'G') captureImage('good');
  if (k === 'b' || k === 'B') captureImage('bad');
  if (k === 'c' || k === 'C') { switchTab('predict'); runPredict(); }
  if (k === '1') switchTab('dataset');
  if (k === '2') switchTab('predict');
});
