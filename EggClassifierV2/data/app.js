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

// Counter download — persisten di localStorage agar tidak reset saat refresh
const dlCounts = {
  good: parseInt(localStorage.getItem('dl_good') || '0'),
  bad:  parseInt(localStorage.getItem('dl_bad')  || '0'),
};

// Init counters dari localStorage
counts.good = dlCounts.good;
counts.bad  = dlCounts.bad;

const sleep = ms => new Promise(r => setTimeout(r, ms));

// ─── Antrian request ke alat ─────────────────────────────────
// WebServer di firmware sinkron: hanya melayani SATU koneksi pada satu
// waktu — socket lain ditahan sampai 5 detik. Browser senang membuka
// socket paralel (preview + tombol + galeri sekaligus), dan itulah yang
// membuat web terasa macet. Solusi: SEMUA request ke alat lewat antrian
// ini; maksimal satu berjalan, sisanya menunggu giliran di browser.
// (Request ke api.github.com TIDAK lewat sini — beda server.)
let espChain = Promise.resolve();

function espQueue(job) {
  const p = espChain.then(job, job);
  espChain = p.then(() => {}, () => {});
  return p;
}

const ESP_TIMEOUT_MS = 15000;   // satu request macet tak boleh membekukan UI

function espFetch(url, opts, precheck) {
  return espQueue(async () => {
    if (precheck && !precheck()) throw new Error('dibatalkan');
    // Timeout: bila alat tak merespons (mis. sedang restart), batalkan agar
    // antrian lanjut ke request berikutnya, bukan menggantung selamanya
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), ESP_TIMEOUT_MS);
    try {
      const res = await fetch(url, { ...(opts || {}), signal: ctrl.signal });
      // Habiskan body DI DALAM antrian — request berikutnya baru jalan
      // setelah transfer ini benar-benar selesai
      const buf = await res.arrayBuffer();
      return new Response(buf, { status: res.status, headers: res.headers });
    } finally {
      clearTimeout(timer);
    }
  });
}

// Muat gambar dari alat lewat antrian yang sama. img.src biasa membuat
// browser membuka socket sendiri di luar antrian — di sini gambar
// diambil sebagai blob lalu dipasang sebagai object URL.
async function espImage(imgEl, url, precheck) {
  const res = await espFetch(url, undefined, precheck);
  if (!res.ok) throw new Error('HTTP ' + res.status);
  const blob = await res.blob();
  const old  = imgEl.dataset.blobUrl || '';
  const u    = URL.createObjectURL(blob);
  imgEl.src  = u;
  imgEl.dataset.blobUrl = u;
  if (old) URL.revokeObjectURL(old);
}

function revokeBlobImgs(container) {
  container.querySelectorAll('img').forEach(i => {
    if (i.dataset.blobUrl) URL.revokeObjectURL(i.dataset.blobUrl);
  });
}

// ─── Score Chart (Chart.js donut) ────────────────────────────
// Chart.js (~205 KB) di-host di alat tapi di-lazy-load: hanya diunduh saat
// tab Prediksi pertama kali dibuka, jadi tidak membebani load awal halaman.
let chartJsPromise = null;
function ensureChartJs() {
  if (window.Chart) return Promise.resolve();
  if (!chartJsPromise) {
    chartJsPromise = new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = '/chart.min.js';
      s.onload = resolve;
      s.onerror = () => { chartJsPromise = null; reject(new Error('chart.min.js gagal dimuat')); };
      document.head.appendChild(s);
    });
  }
  return chartJsPromise;
}

async function initChart() {
  if (scoreChart) return;
  await ensureChartJs();
  if (scoreChart) return;   // bisa keburu dibuat saat menunggu skrip
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
  ['dataset', 'manage', 'training', 'predict', 'camera'].forEach(t => {
    document.getElementById('tab-' + t).classList.toggle('hidden', t !== tab);
    document.getElementById('btn-' + t).classList.toggle('active', t === tab);
  });
  // Kolom kamera disembunyikan di tab Kelola & Training (frame tak ditampilkan,
  // preview juga berhenti) → konten pakai lebar penuh, bandwidth fokus ke galeri.
  const camOff = (tab === 'manage' || tab === 'training');
  document.getElementById('workspace').classList.toggle('no-cam', camOff);
  if (tab === 'predict' && !scoreChart) initChart();
  if (tab === 'camera' && !camLoaded) loadCamSettings();
  if (tab === 'manage') loadDatasetManager();
  if (tab === 'training') loadSDInfo();   // refresh angka dataset
}

// ─── Live Preview — self-paced ────────────────────────────────
// Bukan setInterval: frame berikutnya baru diminta setelah frame
// sebelumnya benar-benar selesai, jadi koneksi tidak pernah menumpuk
// saat link lambat (frame rate menyesuaikan sendiri).
// Kamera menyala di tab Dataset/Prediksi/Kamera, DIMATIKAN di Kelola & Training
// (tab itu butuh bandwidth untuk galeri/sync, frame kamera cuma membebani).
// Juga jeda saat klasifikasi (kamera dipakai inferensi) atau tab disembunyikan.
async function previewLoop() {
  const img = document.getElementById('preview');
  while (true) {
    const camOff = isClassifying || activeTab === 'manage' || activeTab === 'training';
    if (document.hidden || camOff) { await sleep(350); continue; }

    const t0 = Date.now();
    try {
      await espImage(img, '/capture?t=' + Date.now());
      document.getElementById('frame-count').textContent = ++frameCount;
      if (!previewOk) {
        previewOk = true;
        document.getElementById('overlay-loading').style.display = 'none';
      }
      setWifi(true);
    } catch (_) {
      if (previewOk) setWifi(false);
    }
    await sleep(Math.max(PREVIEW_MS - (Date.now() - t0), 200));
  }
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

previewLoop();

// ─── Dataset: capture → SD card (fallback: download ke PC) ───
let sdMounted = false;

async function captureImage(label) {
  if (isCapturing) return;
  isCapturing = true;
  const btn = document.getElementById(label + '-btn');
  btn.disabled = true;

  try {
    if (sdMounted) {
      // Mode SD: foto disimpan langsung di alat
      const res  = await espFetch('/sd/capture?label=' + label, { method: 'POST' });
      const data = await res.json();
      if (data.error) throw new Error(data.error);

      counts.good = data.good;
      counts.bad  = data.bad;
      updateCounters();
      addThumb(label, label === 'good' ? data.good : data.bad,
               '/sd/file?name=' + data.file + '&t=' + Date.now());
      showToast(
        (label === 'good' ? '✅ BAGUS' : '❌ CACAT') + ' → SD: ' + data.file +
        ' (' + data.size_kb + ' KB)',
        label
      );
    } else {
      // Mode download: perilaku lama (tanpa SD card)
      const res = await espFetch('/capture?t=' + Date.now());
      if (!res.ok) throw new Error('Camera gagal (' + res.status + ')');
      const blob = await res.blob();

      dlCounts[label]++;
      localStorage.setItem('dl_' + label, dlCounts[label]);

      const n     = dlCounts[label];
      const fname = label + '_' + String(n).padStart(4, '0') + '.jpg';

      const url = URL.createObjectURL(blob);
      const a   = document.createElement('a');
      a.href     = url;
      a.download = fname;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => URL.revokeObjectURL(url), 1500);

      counts[label] = n;
      updateCounters();
      addThumb(label, n);
      showToast(
        (label === 'good' ? '✅ BAGUS' : '❌ CACAT') + ' #' + n + ' → ' + fname,
        label
      );
    }
  } catch (e) {
    showToast('Error: ' + e.message, 'error');
  } finally {
    btn.disabled = false;
    isCapturing  = false;
  }
}

// ─── SD info: counter dataset + mode penyimpanan ──────────────
async function loadSDInfo() {
  const badge = document.getElementById('sd-badge');
  const title = document.getElementById('storage-mode-title');
  const desc  = document.getElementById('storage-mode-desc');
  try {
    const res  = await espFetch('/sd/info');
    const data = await res.json();
    sdMounted  = !!data.mounted;

    if (sdMounted) {
      counts.good = data.good;
      counts.bad  = data.bad;
      updateCounters();
      badge.textContent = '💾 SD ' + data.used_mb + '/' + data.total_mb + ' MB';
      title.textContent = '💾 Mode SD Card Aktif';
      desc.textContent  =
        'Foto tersimpan di SD card alat (/dataset). Buka tab Training untuk ' +
        'sinkron ke GitHub dan training otomatis.';
    } else {
      badge.textContent = '📥 Download';
      title.textContent = '📥 Mode Download Aktif';
      desc.textContent  =
        'SD card tidak terdeteksi — foto diunduh ke PC. Pasang microSD ' +
        '(FAT32) lalu restart alat untuk menyimpan dataset di alat.';
    }

    // Angka dataset di tab Training (pengganti preview kamera)
    document.getElementById('train-good').textContent = counts.good;
    document.getElementById('train-bad').textContent  = counts.bad;
    document.getElementById('train-total').textContent =
      (counts.good + counts.bad) + ' foto total' +
      (sdMounted ? ' · SD ' + data.used_mb + '/' + data.total_mb + ' MB' : '');
  } catch (_) {}
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

// ─── Thumbnails ───────────────────────────────────────────────
function addThumb(label, n, src) {
  const grid  = document.getElementById('thumb-grid');
  const empty = document.getElementById('thumb-empty');
  empty.classList.add('hidden');

  const wrap  = document.createElement('div');
  wrap.className = 'thumb';
  const imgEl = document.createElement('img');
  imgEl.alt = label;
  const lbl = document.createElement('div');
  lbl.className = 'thumb-label ' + (label === 'good' ? 'thumb-label-good' : 'thumb-label-bad');
  lbl.textContent = '#' + n;
  wrap.appendChild(imgEl);
  wrap.appendChild(lbl);

  espImage(imgEl, src || ('/capture?t=' + Date.now())).then(() => {
    grid.insertBefore(wrap, grid.firstChild);
    while (grid.children.length > 6) {
      revokeBlobImgs(grid.lastChild);
      grid.removeChild(grid.lastChild);
    }
  }).catch(() => {});
}

// ─── Predict ──────────────────────────────────────────────────
async function runPredict() {
  if (isClassifying) return;
  await initChart();   // pastikan Chart.js termuat sebelum updateChart
  isClassifying = true;

  const btn = document.getElementById('predict-btn');
  btn.disabled = true;
  btn.textContent = '⏳  Menganalisis...';
  btn.classList.add('btn-classify-busy');

  document.getElementById('busy-badge').classList.remove('hidden');

  try {
    const res  = await espFetch('/predict?t=' + Date.now());
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
  void card.offsetWidth;
  card.classList.add('result-reveal');

  updateChart(data.score, data.good);

  const wrap = document.getElementById('chart-wrap');
  wrap.classList.remove('ring-good', 'ring-bad');
  void wrap.offsetWidth;
  wrap.classList.add(data.good ? 'ring-good' : 'ring-bad');

  const pct = (data.score * 100).toFixed(1);
  document.getElementById('result-score').textContent = pct + '%';
  document.getElementById('result-score').style.color =
    data.good ? 'var(--success)' : 'var(--danger)';

  const label = document.getElementById('result-label');
  label.textContent = data.label;
  label.style.color = data.good ? 'var(--success)' : 'var(--danger)';

  const confLevel =
    data.score > 0.85 || data.score < 0.15 ? 'TINGGI' :
    data.score > 0.70 || data.score < 0.30 ? 'SEDANG' : 'RENDAH';
  const confEl = document.getElementById('result-confidence');
  confEl.textContent = confLevel;
  confEl.style.color =
    confLevel === 'TINGGI' ? 'var(--success)' :
    confLevel === 'SEDANG' ? 'var(--warn)'    : 'var(--danger)';

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
    const res  = await espFetch('/model_info');
    const data = await res.json();
    const el   = document.getElementById('model-status');
    const warn = document.getElementById('no-model-warning');

    const errEl = document.getElementById('model-err');
    if (data.loaded) {
      el.textContent = '✓ ' + data.size_kb + ' KB';
      el.className   = 'badge-loaded';
      warn.classList.add('hidden');
    } else {
      el.textContent = data.err ? 'Model gagal dimuat' : 'Tidak ada model';
      el.className   = 'badge-missing';
      warn.classList.remove('hidden');
      // Tampilkan alasan dari firmware (mis. op tidak terdaftar / arena kurang)
      errEl.classList.toggle('hidden', !data.err);
      if (data.err) errEl.textContent = '⛔ ' + data.err;
    }
  } catch (_) {}
}

// Upload model manual dihapus dari UI — pemasangan model satu pintu
// lewat tab Training (pipeline / tombol "Pasang Model Ini ke Alat").
// Endpoint POST /upload_model tetap ada dan dipakai installModelToDevice().

async function waitForRestart(maxWait = 15000) {
  const t0 = Date.now();
  await new Promise(r => setTimeout(r, 2500));
  while (Date.now() - t0 < maxWait) {
    try { await espFetch('/capture?t=' + Date.now()); return; }
    catch (_) { await new Promise(r => setTimeout(r, 800)); }
  }
}

// ═══════════════════════════════════════════════════════════
//  KELOLA DATASET — galeri CRUD foto di SD card
//  Preview kamera dimatikan selama tab ini terbuka.
// ═══════════════════════════════════════════════════════════
const DS_BATCH = 24;          // foto per "muat lebih banyak"
let dsFiles  = [];            // [{n: nama, s: size}]
let dsFilter = 'all';
let dsShown  = 0;
let dsGen    = 0;             // naik tiap render ulang — pembatalan antrian
                              // gambar tile yang sudah tidak relevan

async function loadDatasetManager() {
  const empty = document.getElementById('ds-empty');
  try {
    const res  = await espFetch('/sd/list');
    const data = await res.json();
    if (data.error) {
      dsFiles = [];
      empty.textContent = data.error === 'no_sd'
        ? 'SD card tidak terpasang.' : 'Error: ' + data.error;
      document.getElementById('ds-grid').innerHTML = '';
      document.getElementById('ds-more').classList.add('hidden');
      return;
    }
    dsFiles = data
      .filter(f => /\.(jpg|jpeg)$/i.test(f.n))
      .sort((a, b) => b.n.localeCompare(a.n));   // terbaru dulu
    dsRender(true);
  } catch (e) {
    empty.textContent = 'Gagal memuat: ' + e.message;
  }
}

function dsSetFilter(f) {
  dsFilter = f;
  ['all', 'good', 'bad'].forEach(x =>
    document.getElementById('ds-f-' + x).classList.toggle('active', x === f));
  dsRender(true);
}

function dsFiltered() {
  if (dsFilter === 'all') return dsFiles;
  return dsFiles.filter(f => f.n.startsWith(dsFilter + '_'));
}

function dsRender(reset) {
  const grid  = document.getElementById('ds-grid');
  const empty = document.getElementById('ds-empty');
  const more  = document.getElementById('ds-more');
  const list  = dsFiltered();

  if (reset) {
    dsGen++;                  // batalkan muatan gambar tile yang masih antre
    revokeBlobImgs(grid);
    grid.innerHTML = '';
    dsShown = 0;
  }

  const slice = list.slice(dsShown, dsShown + DS_BATCH);
  slice.forEach(f => grid.appendChild(dsTile(f)));
  dsShown += slice.length;

  const nGood = dsFiles.filter(f => f.n.startsWith('good_')).length;
  const nBad  = dsFiles.filter(f => f.n.startsWith('bad_')).length;
  document.getElementById('ds-count').textContent =
    nGood + ' bagus · ' + nBad + ' cacat';

  empty.style.display = list.length ? 'none' : '';
  if (!list.length) empty.textContent = 'Tidak ada foto.';
  more.classList.toggle('hidden', dsShown >= list.length);
  if (dsShown < list.length)
    more.textContent = 'Muat lebih banyak ▾ (' + (list.length - dsShown) + ' lagi)';
}

function dsRenderMore() { dsRender(false); }

function dsTile(f) {
  const good = f.n.startsWith('good_');
  const el = document.createElement('div');
  el.className = 'ds-tile';
  el.id = 'ds-' + f.n;

  const img = document.createElement('img');
  img.alt = f.n;
  const fileUrl = '/sd/file?name=' + encodeURIComponent(f.n);
  // Muat lewat antrian (berurutan, tidak membanjiri alat); batal otomatis
  // bila galeri sudah dirender ulang sebelum gilirannya tiba
  const gen = dsGen;
  espImage(img, fileUrl, () => gen === dsGen && img.isConnected).catch(() => {});
  img.onclick = () => window.open(fileUrl, '_blank');
  el.appendChild(img);

  const badge = document.createElement('div');
  badge.className = 'thumb-label ' + (good ? 'thumb-label-good' : 'thumb-label-bad');
  badge.textContent = f.n.replace('.jpg', '');
  el.appendChild(badge);

  const acts = document.createElement('div');
  acts.className = 'ds-actions';

  const btnRe = document.createElement('button');
  btnRe.className = 'ds-act';
  btnRe.title = 'Pindah label';
  btnRe.textContent = '🔁';
  btnRe.onclick = e => { e.stopPropagation(); dsRelabel(f.n); };
  acts.appendChild(btnRe);

  const btnDel = document.createElement('button');
  btnDel.className = 'ds-act ds-act-danger';
  btnDel.title = 'Hapus';
  btnDel.textContent = '🗑️';
  btnDel.onclick = e => { e.stopPropagation(); dsDelete(f.n); };
  acts.appendChild(btnDel);

  el.appendChild(acts);
  return el;
}

async function dsDelete(name) {
  if (!confirm('Hapus ' + name + ' dari SD card?')) return;
  try {
    const res  = await espFetch('/sd/delete?name=' + encodeURIComponent(name), { method: 'POST' });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    dsFiles = dsFiles.filter(f => f.n !== name);
    const tile = document.getElementById('ds-' + name);
    if (tile) { revokeBlobImgs(tile); tile.remove(); }
    dsRender(false);
    loadSDInfo();
    showToast('🗑️ ' + name + ' dihapus', 'good');
  } catch (e) {
    showToast('Hapus gagal: ' + e.message, 'error');
  }
}

async function dsRelabel(name) {
  try {
    const res  = await espFetch('/sd/relabel?name=' + encodeURIComponent(name), { method: 'POST' });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    const f = dsFiles.find(x => x.n === name);
    if (f) f.n = data.file;
    dsRender(true);
    loadSDInfo();
    showToast('🔁 ' + name + ' → ' + data.file, 'good');
  } catch (e) {
    showToast('Pindah label gagal: ' + e.message, 'error');
  }
}

// Upload dari PC: pilih label dulu, lalu file picker (boleh multi-file)
let dsUploadLabel = 'good';

function dsPickUpload(label) {
  dsUploadLabel = label;
  document.getElementById('ds-upload-input').click();
}

async function dsUploadFiles(input) {
  const files = [...input.files];
  input.value = '';
  if (!files.length) return;

  const status = document.getElementById('ds-upload-status');
  status.classList.remove('hidden');
  let ok = 0;

  for (let i = 0; i < files.length; i++) {
    status.textContent = 'Upload ' + (i + 1) + '/' + files.length + ': ' + files[i].name;
    try {
      const fd = new FormData();
      fd.append('photo', files[i]);
      const res  = await espFetch('/sd/upload?label=' + dsUploadLabel, { method: 'POST', body: fd });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      dsFiles.unshift({ n: data.file, s: files[i].size });
      ok++;
    } catch (e) {
      showToast(files[i].name + ' gagal: ' + e.message, 'error');
    }
  }

  status.textContent = '✅ ' + ok + '/' + files.length + ' foto terupload sebagai ' +
    (dsUploadLabel === 'good' ? 'BAGUS' : 'CACAT');
  dsRender(true);
  loadSDInfo();
}

// ═══════════════════════════════════════════════════════════
//  TRAINING PIPELINE — GitHub Actions
//  Browser = jembatan: SD card → GitHub repo → Actions → model
//  Token PAT hanya di localStorage browser, tidak ke ESP32.
// ═══════════════════════════════════════════════════════════
const GH_API   = 'https://api.github.com';
const WORKFLOW = 'train.yml';

// Repo target — terisi tetap, tidak perlu form pengaturan
const ghCfg = { owner: 'Keyzoo0', repo: 'EggClassifier', branch: 'main' };

// Token TIDAK ditanam di kode: token yang ter-push ke repo otomatis
// dicabut GitHub (secret scanning). Diminta sekali via popup → localStorage.
let ghToken = localStorage.getItem('gh_token') || '';

function ensureToken(forceAsk) {
  if (ghToken && !forceAsk) return true;
  const t = prompt(
    'Tempel GitHub Personal Access Token (cukup sekali, tersimpan di browser).\n\n' +
    'Cara buat: GitHub → Settings → Developer settings → Fine-grained tokens\n' +
    '• Repo     : ' + ghCfg.owner + '/' + ghCfg.repo + '\n' +
    '• Permission: Contents (Read & Write) + Actions (Read & Write)',
    ghToken || ''
  );
  if (t && t.trim()) {
    ghToken = t.trim();
    localStorage.setItem('gh_token', ghToken);
    showToast('🔑 Token tersimpan di browser ini', 'good');
  }
  return !!ghToken;
}

function changeToken() { ensureToken(true); }
function ghReady()     { return ensureToken(false); }

async function gh(path, opts = {}) {
  const res = await fetch(GH_API + '/repos/' + ghCfg.owner + '/' + ghCfg.repo + path, {
    ...opts,
    headers: {
      'Authorization': 'Bearer ' + ghToken,
      'Accept':        'application/vnd.github+json',
      ...(opts.headers || {}),
    },
  });
  if (res.status === 401) {
    ghToken = '';
    localStorage.removeItem('gh_token');
    throw new Error('Token GitHub salah/kedaluwarsa — klik tombol lagi untuk isi ulang');
  }
  return res;
}

function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload  = () => resolve(fr.result.split(',')[1]);
    fr.onerror = reject;
    fr.readAsDataURL(blob);
  });
}

// ─── Step indicator helpers ──────────────────────────────────
function setStep(id, state, info) {
  const row = document.getElementById(id);
  row.classList.remove('step-active', 'step-done', 'step-fail');
  if (state === 'active') row.classList.add('step-active');
  if (state === 'done')   row.classList.add('step-done');
  if (state === 'fail')   row.classList.add('step-fail');
  const st = row.querySelector('.step-state');
  st.textContent = info || '—';
  st.style.color =
    state === 'done' ? 'var(--success)' :
    state === 'fail' ? 'var(--danger)'  :
    state === 'active' ? 'var(--primary)' : 'var(--text-3)';
}

function trainStatus(msg, pct) {
  document.getElementById('train-progress').classList.remove('hidden');
  document.getElementById('train-status').textContent = msg;
  if (pct !== undefined) document.getElementById('train-fill').style.width = pct + '%';
}

// ─── Step 1: sinkron dataset SD → repo GitHub ────────────────
// SD card = sumber kebenaran: foto baru di-upload, foto yang sudah
// dihapus/dipindah label di SD ikut dihapus dari repo.
async function syncDataset() {
  // Isi repo saat ini (nama → sha+size; sha perlu untuk update/delete)
  const repoFiles = new Map();
  const lsRes = await gh('/contents/dataset?ref=' + ghCfg.branch);
  if (lsRes.ok) (await lsRes.json()).forEach(f =>
    repoFiles.set(f.name, { sha: f.sha, size: f.size }));
  else if (lsRes.status !== 404) throw new Error('Gagal baca repo (' + lsRes.status + ')');

  const sdRes  = await espFetch('/sd/list');
  const sdList = await sdRes.json();
  if (sdList.error) throw new Error('SD: ' + sdList.error);
  const sdNames = new Set(sdList.map(f => f.n));

  // Upload jika: nama belum ada di repo, ATAU nama sama tapi ukuran beda
  // (index bekas hapus dipakai ulang oleh foto baru → isi file berubah)
  const toUpload = sdList.filter(f => {
    const r = repoFiles.get(f.n);
    return !r || r.size !== f.s;
  });
  const toDelete = [...repoFiles.keys()]
    .filter(n => /\.(jpg|jpeg)$/i.test(n) && !sdNames.has(n));
  const total = toUpload.length + toDelete.length;
  let done = 0;

  for (const f of toUpload) {
    const img = await espFetch('/sd/file?name=' + encodeURIComponent(f.n));
    if (!img.ok) throw new Error('Gagal baca ' + f.n + ' dari SD');
    const b64 = await blobToBase64(await img.blob());

    const body = {
      message: 'dataset: tambah/update ' + f.n,
      content: b64,
      branch:  ghCfg.branch,
    };
    const prev = repoFiles.get(f.n);
    if (prev) body.sha = prev.sha;   // update file lama (nama dipakai ulang)

    const put = await gh('/contents/dataset/' + encodeURIComponent(f.n), {
      method: 'PUT',
      body: JSON.stringify(body),
    });
    if (!put.ok) throw new Error('Upload ' + f.n + ' gagal (' + put.status + ')');

    done++;
    setStep('step-sync', 'active', done + '/' + total);
    trainStatus('Upload ' + f.n + ' (' + done + '/' + total + ')',
                Math.round(done / total * 100));
  }

  for (const n of toDelete) {
    const del = await gh('/contents/dataset/' + encodeURIComponent(n), {
      method: 'DELETE',
      body: JSON.stringify({
        message: 'dataset: hapus ' + n + ' (tidak ada lagi di SD)',
        sha:     repoFiles.get(n),
        branch:  ghCfg.branch,
      }),
    });
    if (!del.ok) throw new Error('Hapus ' + n + ' di repo gagal (' + del.status + ')');

    done++;
    setStep('step-sync', 'active', done + '/' + total);
    trainStatus('Hapus ' + n + ' dari repo (' + done + '/' + total + ')',
                Math.round(done / total * 100));
  }

  return { uploaded: toUpload.length, deleted: toDelete.length };
}

// ─── Step 2: trigger workflow + tunggu selesai ───────────────
async function dispatchTraining() {
  const res = await gh('/actions/workflows/' + WORKFLOW + '/dispatches', {
    method: 'POST',
    body: JSON.stringify({ ref: ghCfg.branch }),
  });
  if (res.status !== 204) {
    if (res.status === 404)
      throw new Error('Workflow train.yml tidak ditemukan — sudah di-push ke GitHub?');
    throw new Error('Trigger training gagal (' + res.status + ')');
  }
}

async function waitForTraining(dispatchedAt) {
  const t0      = Date.now();
  const maxMs   = 90 * 60 * 1000;   // batas 90 menit
  const fmtMin  = ms => Math.floor(ms / 60000) + 'm ' +
                        Math.floor((ms % 60000) / 1000) + 's';

  while (Date.now() - t0 < maxMs) {
    await new Promise(r => setTimeout(r, 10000));
    const elapsed = fmtMin(Date.now() - t0);

    let run = null;
    try {
      const res = await gh('/actions/workflows/' + WORKFLOW +
                           '/runs?per_page=1&branch=' + ghCfg.branch);
      if (res.ok) {
        const runs = (await res.json()).workflow_runs || [];
        // Hanya run yang dibuat setelah dispatch kita
        if (runs.length && new Date(runs[0].created_at).getTime() >= dispatchedAt - 90000)
          run = runs[0];
      }
    } catch (_) { /* network blip — coba lagi */ }

    if (!run) {
      setStep('step-train', 'active', 'antri… ' + elapsed);
      trainStatus('Menunggu runner GitHub Actions… (' + elapsed + ')');
      continue;
    }
    if (run.status === 'completed') {
      if (run.conclusion === 'success') return run;
      throw new Error('Training gagal (' + run.conclusion + ') — cek log di GitHub Actions');
    }
    setStep('step-train', 'active', run.status + ' ' + elapsed);
    trainStatus('Training berjalan di GitHub… (' + elapsed + ') — boleh ditinggal, ' +
                'jangan tutup tab ini', undefined);
  }
  throw new Error('Timeout 90 menit — cek manual di GitHub Actions');
}

// ─── Step 3: ambil model dari repo → pasang ke alat ──────────
async function fetchRemoteModel() {
  const res = await gh('/contents/model/egg_model.tflite?ref=' + ghCfg.branch);
  if (res.status === 404) throw new Error('model/egg_model.tflite belum ada di repo');
  if (!res.ok) throw new Error('Gagal ambil model (' + res.status + ')');
  const data = await res.json();
  const bin  = atob(data.content.replace(/\s/g, ''));
  const buf  = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return new Blob([buf], { type: 'application/octet-stream' });
}

async function installModelToDevice(blob, onProgress) {
  const fd = new FormData();
  fd.append('model', blob, 'egg_model.tflite');

  // XHR (demi progress-bar upload) — tetap lewat antrian alat
  const result = await espQueue(() => new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.upload.onprogress = e => {
      if (e.lengthComputable && onProgress)
        onProgress(Math.round(e.loaded / e.total * 100));
    };
    xhr.onload  = () => {
      try { resolve(JSON.parse(xhr.responseText)); }
      catch { resolve({ ok: false, error: 'parse_error' }); }
    };
    xhr.onerror = () => reject(new Error('Network error ke ESP32'));
    xhr.open('POST', '/upload_model');
    xhr.send(fd);
  }));

  if (!result.ok) throw new Error('Upload ke alat gagal: ' + (result.error || '?'));
  await waitForRestart();
  loadModelInfo();
}

// ─── Info model hasil training terakhir di GitHub ────────────
async function loadRemoteModelInfo() {
  if (!ghReady()) { showToast('Token GitHub diperlukan', 'error'); return; }
  const el = document.getElementById('remote-model-info');
  el.textContent = 'Memeriksa…';
  try {
    const res = await gh('/contents/model/model_info.json?ref=' + ghCfg.branch);
    if (res.status === 404) {
      el.textContent = 'Belum ada model hasil training di repo.';
      return;
    }
    const info = JSON.parse(atob((await res.json()).content.replace(/\s/g, '')));
    el.innerHTML =
      'Val accuracy: <b style="color:var(--success)">' +
      (info.val_accuracy * 100).toFixed(1) + '%</b> · ' +
      info.size_kb + ' KB · dataset ' + info.n_good + ' bagus / ' + info.n_bad +
      ' cacat<br><span style="color:var(--text-3)">Dilatih: ' + info.trained_at + '</span>';
    document.getElementById('deploy-btn').classList.remove('hidden');
  } catch (e) {
    el.textContent = 'Error: ' + e.message;
  }
}

async function deployRemoteModel() {
  const btn = document.getElementById('deploy-btn');
  btn.disabled = true;
  try {
    showToast('Mengunduh model dari GitHub…', 'good');
    const blob = await fetchRemoteModel();
    await installModelToDevice(blob, p => trainStatus('Upload ke alat… ' + p + '%', p));
    trainStatus('✅ Model baru aktif!', 100);
    showToast('✅ Model baru aktif!', 'good');
  } catch (e) {
    showToast('Gagal: ' + e.message, 'error');
  } finally {
    btn.disabled = false;
  }
}

// ─── Pipeline penuh: sync → train → deploy ───────────────────
let pipelineRunning = false;

async function runFullPipeline() {
  if (pipelineRunning) return;
  if (!ghReady()) {
    showToast('Token GitHub diperlukan untuk training', 'error');
    return;
  }
  pipelineRunning = true;
  const btn = document.getElementById('train-btn');
  btn.disabled = true;
  document.getElementById('train-steps').classList.remove('hidden');
  setStep('step-sync',   'active', '');
  setStep('step-train',  null, '');
  setStep('step-deploy', null, '');

  try {
    // 1 — Sinkron dataset
    if (sdMounted) {
      trainStatus('Membandingkan SD card dengan repo…', 0);
      const r = await syncDataset();
      setStep('step-sync', 'done',
        '+' + r.uploaded + (r.deleted ? ' / −' + r.deleted : '') + ' foto');
    } else {
      setStep('step-sync', 'done', 'dilewati (tanpa SD)');
    }

    // 2 — Training di GitHub Actions
    setStep('step-train', 'active', 'memicu…');
    trainStatus('Memicu workflow training…', 0);
    const dispatchedAt = Date.now();
    await dispatchTraining();
    const run = await waitForTraining(dispatchedAt);
    setStep('step-train', 'done', 'sukses');

    // 3 — Pasang model ke alat
    setStep('step-deploy', 'active', 'unduh…');
    trainStatus('Mengunduh model hasil training…');
    const blob = await fetchRemoteModel();
    await installModelToDevice(blob, p => {
      setStep('step-deploy', 'active', p + '%');
      trainStatus('Upload model ke alat… ' + p + '%', p);
    });
    setStep('step-deploy', 'done', 'aktif');
    trainStatus('✅ Selesai — model baru sudah aktif di alat!', 100);
    showToast('✅ Pipeline selesai! Model baru aktif.', 'good');
    loadRemoteModelInfo();

  } catch (e) {
    // Tandai step yang sedang aktif sebagai gagal
    ['step-sync', 'step-train', 'step-deploy'].forEach(id => {
      if (document.getElementById(id).classList.contains('step-active'))
        setStep(id, 'fail', 'gagal');
    });
    trainStatus('❌ ' + e.message);
    showToast('❌ ' + e.message, 'error');
  } finally {
    pipelineRunning = false;
    btn.disabled = false;
  }
}

// ═══════════════════════════════════════════════════════════
//  PENGATURAN KAMERA — semua parameter sensor OV2640
//  Kontrol digenerate dari skema; nilai disimpan firmware di NVS.
// ═══════════════════════════════════════════════════════════
let camLoaded = false;

const CAM_SCHEMA = [
  { group: '🖼️ Gambar', items: [
    { id: 'brightness', label: 'Brightness',  type: 'range', min: -2, max: 2 },
    { id: 'contrast',   label: 'Contrast',    type: 'range', min: -2, max: 2 },
    { id: 'saturation', label: 'Saturation',  type: 'range', min: -2, max: 2 },
    { id: 'special_effect', label: 'Efek Khusus', type: 'select',
      options: ['Normal', 'Negative', 'Grayscale', 'Red Tint', 'Green Tint', 'Blue Tint', 'Sepia'] },
    { id: 'quality', label: 'Kualitas JPEG', type: 'range', min: 4, max: 40,
      hint: 'kecil = lebih bagus & file besar' },
  ]},
  { group: '⚪ White Balance', items: [
    { id: 'awb',      label: 'Auto White Balance', type: 'switch' },
    { id: 'awb_gain', label: 'AWB Gain',           type: 'switch' },
    { id: 'wb_mode',  label: 'Mode WB', type: 'select',
      options: ['Auto', 'Sunny', 'Cloudy', 'Office', 'Home'] },
  ]},
  { group: '☀️ Exposure', items: [
    { id: 'aec',       label: 'Auto Exposure (AEC)', type: 'switch' },
    { id: 'aec2',      label: 'AEC DSP (night mode)', type: 'switch' },
    { id: 'ae_level',  label: 'AE Level',  type: 'range', min: -2, max: 2,
      hint: 'kompensasi exposure saat AEC aktif' },
    { id: 'aec_value', label: 'Exposure Manual', type: 'range', min: 0, max: 1200,
      hint: 'aktif saat AEC dimatikan' },
  ]},
  { group: '📈 Gain', items: [
    { id: 'agc',         label: 'Auto Gain (AGC)', type: 'switch' },
    { id: 'gainceiling', label: 'Gain Ceiling', type: 'select',
      options: ['2x', '4x', '8x', '16x', '32x', '64x', '128x'],
      hint: 'batas atas gain saat AGC aktif' },
    { id: 'agc_gain',    label: 'Gain Manual', type: 'range', min: 0, max: 30,
      hint: 'aktif saat AGC dimatikan' },
  ]},
  { group: '🔧 Koreksi & Orientasi', items: [
    { id: 'bpc',      label: 'Black Pixel Correction', type: 'switch' },
    { id: 'wpc',      label: 'White Pixel Correction', type: 'switch' },
    { id: 'raw_gma',  label: 'Gamma (RAW GMA)',        type: 'switch' },
    { id: 'lenc',     label: 'Koreksi Lensa (LENC)',   type: 'switch' },
    { id: 'dcw',      label: 'Downsize (DCW)',         type: 'switch' },
    { id: 'hmirror',  label: 'Mirror Horizontal',      type: 'switch' },
    { id: 'vflip',    label: 'Flip Vertikal',          type: 'switch' },
    { id: 'colorbar', label: 'Color Bar (pola tes)',   type: 'switch',
      hint: 'hanya untuk uji — tidak disimpan permanen' },
  ]},
];

// Aturan enable/disable antar kontrol (meniru perilaku CameraWebServer)
function updateCamDeps() {
  const v = id => {
    const el = document.getElementById('cam-' + id);
    return el ? (el.type === 'checkbox' ? (el.checked ? 1 : 0) : parseInt(el.value)) : 0;
  };
  const dis = (id, d) => {
    const el  = document.getElementById('cam-' + id);
    if (!el) return;
    el.disabled = d;
    el.closest('.setting-row').style.opacity = d ? 0.4 : 1;
  };
  dis('ae_level',    v('aec') === 0);
  dis('aec_value',   v('aec') === 1);
  dis('gainceiling', v('agc') === 0);
  dis('agc_gain',    v('agc') === 1);
  dis('wb_mode',     v('awb_gain') === 0);
}

async function camSet(id, val) {
  try {
    const res  = await espFetch('/camera/set?var=' + id + '&val=' + val, { method: 'POST' });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    updateCamDeps();
  } catch (e) {
    showToast('Set ' + id + ' gagal: ' + e.message, 'error');
  }
}

function renderCamSettings(values) {
  const box = document.getElementById('cam-settings');
  box.innerHTML = '';

  CAM_SCHEMA.forEach(grp => {
    const card = document.createElement('div');
    card.className = 'card p-4 space-y-2';
    card.innerHTML = '<h3 class="text-sm font-semibold mb-1" style="color:var(--text-1)">' +
                     grp.group + '</h3>';

    grp.items.forEach(it => {
      const val = values[it.id];
      const row = document.createElement('div');
      row.className = 'setting-row';

      const lbl = document.createElement('div');
      lbl.className = 'setting-label';
      lbl.innerHTML = it.label +
        (it.hint ? '<span class="setting-hint">' + it.hint + '</span>' : '');
      row.appendChild(lbl);

      if (it.type === 'range') {
        const wrap = document.createElement('div');
        wrap.className = 'setting-ctl flex items-center gap-2';
        const out = document.createElement('span');
        out.className = 'range-value';
        out.textContent = val;
        const inp = document.createElement('input');
        inp.type = 'range';
        inp.id   = 'cam-' + it.id;
        inp.min  = it.min; inp.max = it.max; inp.value = val;
        inp.className = 'range';
        inp.oninput  = () => { out.textContent = inp.value; };
        inp.onchange = () => camSet(it.id, inp.value);
        wrap.appendChild(inp);
        wrap.appendChild(out);
        row.appendChild(wrap);

      } else if (it.type === 'select') {
        const sel = document.createElement('select');
        sel.id = 'cam-' + it.id;
        sel.className = 'input setting-ctl';
        it.options.forEach((o, i) => {
          const op = document.createElement('option');
          op.value = i; op.textContent = o;
          if (i === val) op.selected = true;
          sel.appendChild(op);
        });
        sel.onchange = () => camSet(it.id, sel.value);
        row.appendChild(sel);

      } else { // switch
        const sw = document.createElement('label');
        sw.className = 'switch';
        const inp = document.createElement('input');
        inp.type    = 'checkbox';
        inp.id      = 'cam-' + it.id;
        inp.checked = val === 1;
        inp.onchange = () => camSet(it.id, inp.checked ? 1 : 0);
        const track = document.createElement('span');
        track.className = 'switch-track';
        sw.appendChild(inp);
        sw.appendChild(track);
        row.appendChild(sw);
      }

      card.appendChild(row);
    });
    box.appendChild(card);
  });

  updateCamDeps();
}

async function loadCamSettings() {
  const box = document.getElementById('cam-settings');
  try {
    const res  = await espFetch('/camera/get');
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    renderCamSettings(data);
    camLoaded = true;
  } catch (e) {
    box.innerHTML = '<p class="text-xs text-center py-6" style="color:var(--danger)">' +
                    'Gagal memuat: ' + e.message + '</p>';
  }
}

async function resetCamSettings() {
  if (!confirm('Reset semua pengaturan kamera ke default pabrik?\nAlat akan restart.')) return;
  try {
    await espFetch('/camera/reset', { method: 'POST' });
    showToast('Reset! Menunggu alat restart…', 'good');
    camLoaded = false;
    await waitForRestart();
    loadCamSettings();
    showToast('✅ Kamera kembali ke default', 'good');
  } catch (e) {
    showToast('Error: ' + e.message, 'error');
  }
}

// ─── Sidebar collapse ─────────────────────────────────────────
function toggleSidebar() {
  const shell = document.getElementById('app-shell');
  const collapsed = shell.classList.toggle('sb-collapsed');
  localStorage.setItem('sb_collapsed', collapsed ? '1' : '0');
  document.getElementById('sb-toggle-ico').textContent = collapsed ? '⏵' : '⏴';
}
if (localStorage.getItem('sb_collapsed') === '1') {
  document.getElementById('app-shell').classList.add('sb-collapsed');
  document.getElementById('sb-toggle-ico').textContent = '⏵';
}

// ─── Flash kamera (8 NeoPixel GPIO47) ─────────────────────────
// Tombol on/off ada di area kamera (tab Dataset/Prediksi/Kamera) dan di
// tab Kamera; slider kecerahan hanya di tab Kamera. Semua sinkron.
let flashState = { on: false, bri: 96 };

function renderFlash() {
  const b1 = document.getElementById('flash-btn');
  if (b1) {
    b1.textContent = '⚡ Flash ' + (flashState.on ? 'ON' : 'OFF');
    b1.classList.toggle('flash-on', flashState.on);
  }
  const b2 = document.getElementById('flash-btn2');
  if (b2) {
    b2.textContent = flashState.on ? 'ON' : 'OFF';
    b2.classList.toggle('flash-on', flashState.on);
  }
  const sl = document.getElementById('flash-bri');
  if (sl) sl.value = flashState.bri;
  const sv = document.getElementById('flash-bri-val');
  if (sv) sv.textContent = flashState.bri;
}

async function loadFlash() {
  try {
    const res = await espFetch('/flash/get');
    const d = await res.json();
    flashState.on = !!d.on; flashState.bri = d.bri;
    renderFlash();
  } catch (_) {}
}

async function flashApply() {
  try {
    const res = await espFetch(
      '/flash/set?on=' + (flashState.on ? 1 : 0) + '&bri=' + flashState.bri,
      { method: 'POST' });
    const d = await res.json();
    flashState.on = !!d.on; flashState.bri = d.bri;
    renderFlash();
  } catch (e) {
    showToast('Flash gagal: ' + e.message, 'error');
  }
}

function toggleFlash() {
  flashState.on = !flashState.on;
  renderFlash();        // umpan balik instan
  flashApply();
}

function flashBriPreview(v) {          // saat geser slider (belum lepas)
  flashState.bri = parseInt(v) || 0;
  const sv = document.getElementById('flash-bri-val');
  if (sv) sv.textContent = flashState.bri;
}

function setFlashBri(v) {               // saat slider dilepas → kirim ke alat
  flashState.bri = parseInt(v) || 0;
  flashApply();
}

// ─── Init ─────────────────────────────────────────────────────
loadModelInfo();
updateCounters();
loadSDInfo();
loadFlash();
setInterval(() => { if (!document.hidden) loadSDInfo(); }, 30000);

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
  if (k === '2') switchTab('manage');
  if (k === '3') switchTab('training');
  if (k === '4') switchTab('predict');
  if (k === '5') switchTab('camera');
});
