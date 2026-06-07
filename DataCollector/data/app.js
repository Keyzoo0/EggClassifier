const TARGET = 300;
const REFRESH_MS = 800;

let counts = JSON.parse(localStorage.getItem('eggCounts') || '{"good":0,"bad":0}');
let isCapturing = false;
let frameCount = 0;
let previewOk = false;

// ── UI Update ────────────────────────────────────────────────
function updateUI() {
  document.getElementById('good-count').textContent = counts.good;
  document.getElementById('bad-count').textContent  = counts.bad;

  const gPct = Math.min(100, (counts.good / TARGET) * 100);
  const bPct = Math.min(100, (counts.bad  / TARGET) * 100);

  document.getElementById('good-bar').style.width = gPct + '%';
  document.getElementById('bad-bar').style.width  = bPct + '%';

  const remaining = Math.max(0, TARGET - counts.good) + Math.max(0, TARGET - counts.bad);
  const info = document.getElementById('target-info');
  if (counts.good >= TARGET && counts.bad >= TARGET) {
    info.innerHTML = '<span class="text-green-400 font-semibold">✓ Dataset lengkap!</span>';
  } else {
    info.textContent = `Perlu ${remaining} foto lagi untuk mencapai target`;
  }
}

// ── Live Preview ─────────────────────────────────────────────
function refreshPreview() {
  const tmp = new Image();
  tmp.onload = () => {
    document.getElementById('preview').src = tmp.src;
    document.getElementById('frame-count').textContent = ++frameCount;

    if (!previewOk) {
      previewOk = true;
      document.getElementById('overlay-loading').style.display = 'none';
      document.getElementById('preview').style.opacity = '1';
      setStatus(true);
    }
  };
  tmp.onerror = () => {
    setStatus(false);
  };
  tmp.src = '/capture?t=' + Date.now();
}

function setStatus(ok) {
  const dot  = document.getElementById('status-dot');
  const text = document.getElementById('status-text');
  dot.className  = 'w-2 h-2 rounded-full ' + (ok ? 'bg-green-400' : 'bg-red-400');
  text.textContent = ok ? 'Terhubung' : 'Tidak terhubung';
}

setInterval(refreshPreview, REFRESH_MS);

// ── Capture & Save ───────────────────────────────────────────
async function captureImage(label) {
  if (isCapturing) return;
  isCapturing = true;

  const btn = document.getElementById(label + '-btn');
  btn.disabled = true;
  btn.style.opacity = '0.6';

  try {
    const res = await fetch('/capture?t=' + Date.now());
    if (!res.ok) throw new Error('HTTP ' + res.status);

    const blob = await res.blob();
    if (!blob.type.startsWith('image')) throw new Error('Bukan gambar');

    counts[label]++;
    localStorage.setItem('eggCounts', JSON.stringify(counts));

    // Trigger browser download
    const filename = `${label}_${String(counts[label]).padStart(4, '0')}.jpg`;
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 1000);

    // Visual flash
    btn.classList.add(label === 'good' ? 'flash-good' : 'flash-bad');
    setTimeout(() => btn.classList.remove('flash-good', 'flash-bad'), 400);

    updateUI();
    addRecentThumb(blob, label, counts[label]);
    showToast(
      (label === 'good' ? '✅ GOOD' : '❌ BAD') + ` #${counts[label]} tersimpan`,
      label
    );

  } catch (e) {
    showToast('Gagal capture: ' + e.message, 'error');
    setStatus(false);
  } finally {
    btn.disabled = false;
    btn.style.opacity = '';
    isCapturing = false;
  }
}

// ── Recent Thumbnails ────────────────────────────────────────
function addRecentThumb(blob, label, n) {
  const grid  = document.getElementById('recent-grid');
  const empty = document.getElementById('recent-empty');
  empty.style.display = 'none';

  const url = URL.createObjectURL(blob);
  const wrap = document.createElement('div');
  wrap.className = 'relative rounded-lg overflow-hidden aspect-square bg-slate-900';
  wrap.innerHTML = `
    <img src="${url}" class="w-full h-full object-cover">
    <span class="absolute bottom-0 inset-x-0 text-center text-xs font-bold py-0.5
                 ${label === 'good' ? 'bg-green-600' : 'bg-red-600'} text-white opacity-90">
      #${n}
    </span>`;
  grid.insertBefore(wrap, grid.firstChild);

  // Max 8 thumbnails
  while (grid.children.length > 8) grid.removeChild(grid.lastChild);
}

// ── Toast ────────────────────────────────────────────────────
function showToast(msg, type) {
  const color = {
    good:  'bg-green-600',
    bad:   'bg-red-600',
    error: 'bg-amber-600',
  }[type] || 'bg-slate-600';

  const el = document.createElement('div');
  el.className = `${color} text-white text-sm px-4 py-2 rounded-lg shadow-lg
                  transform transition-all duration-300 translate-y-2 opacity-0`;
  el.textContent = msg;
  document.getElementById('toast-container').appendChild(el);

  requestAnimationFrame(() => {
    el.classList.remove('translate-y-2', 'opacity-0');
  });
  setTimeout(() => {
    el.classList.add('opacity-0', 'translate-y-2');
    setTimeout(() => el.remove(), 300);
  }, 2500);
}

// ── Keyboard Shortcuts ───────────────────────────────────────
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (e.repeat) return;
  if (e.key === 'g' || e.key === 'G') captureImage('good');
  if (e.key === 'b' || e.key === 'B') captureImage('bad');
});

// ── Reset ────────────────────────────────────────────────────
function resetCounts() {
  if (!confirm('Reset counter? (File yang sudah didownload tidak dihapus)')) return;
  counts = { good: 0, bad: 0 };
  localStorage.setItem('eggCounts', JSON.stringify(counts));
  document.getElementById('recent-grid').innerHTML = '';
  document.getElementById('recent-empty').style.display = '';
  updateUI();
  showToast('Counter direset', 'error');
}

// ── Init ─────────────────────────────────────────────────────
updateUI();
