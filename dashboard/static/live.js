/* Live analysis of the hero <video>: browser camera (getUserMedia) or real recorded footage (/videos/<file>)
   -> JPEG -> POST /api/analyze -> canvas overlay. Same loop for both sources. */
(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const vf = $('vframe'), cam = $('cam'), ov = $('ov'), ctx = ov.getContext('2d');
  const store = { get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }, set(k, v) { try { localStorage.setItem(k, v); } catch (e) {} } };
  const rid = () => (crypto.getRandomValues ? Array.from(crypto.getRandomValues(new Uint8Array(8)), b => b.toString(16).padStart(2, '0')).join('') : String(Math.random()).slice(2));
  const esc = t => String(t == null ? '' : t).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const CAM_SESSION = rid();
  let SESSION = CAM_SESSION, SOURCE = null;         // SOURCE = 'video:<file>' in recorded-footage mode
  let mode = null;                                   // null | 'cam' | 'file'
  let clips = null, clipIdx = -1, loopClip = store.get('loopClip') === '1';
  const MAXW = 640, QUAL = 0.7, DEPTH = 2;

  let stream = null, running = false, abort = null;
  let gen = 0, lastTick = 0, inflight = 0, encoding = false, seq = 0, applied = 0, sentW = 0, sentH = 0;
  let mirror = store.get('mirror') !== '0';
  let status = null, retryAt = 0, notReady = false;
  const done = [];            // timestamps of completed responses
  let rtt = null, srvMs = null, banner = null, bannerAt = 0, respW = 0, respH = 0, lastResp = 0;
  let items = [], nextKey = 1;
  const off = document.createElement('canvas'), octx = off.getContext('2d');

  /* ---------- UI helpers ---------- */
  function showErr(msg) { $('camErrTxt').textContent = msg; $('camErr').hidden = false; }
  $('camErrX').onclick = () => { $('camErr').hidden = true; };
  function setUi(on) {
    const isCam = on && mode === 'cam', isFile = on && mode === 'file';
    vf.classList.toggle('cam', on); document.body.classList.toggle('camlive', on);
    $('camTxt').innerHTML = isCam ? 'Stop camera' : isFile ? 'Stop video' : 'Go live &mdash; use camera';
    $('camSel').hidden = !isCam || $('camSel').options.length < 2;
    $('btnMirror').hidden = !isCam;
    ['btnPlay', 'btnNext', 'btnLoop'].forEach(id => { $(id).hidden = !isFile; });
    $('btnFiles').innerHTML = isFile ? 'Clips <kbd>V</kbd>' : 'Real footage <kbd>V</kbd>';
    $('hudLive').hidden = !on; $('hudLb').hidden = !on; $('hudRes').hidden = !on; $('hudAttr').hidden = !isFile;
    $('btnCam').setAttribute('aria-pressed', on);
    applyMirror();
  }
  function applyMirror() { cam.classList.toggle('mirror', mirror && mode === 'cam'); $('btnMirror').setAttribute('aria-pressed', mirror); }
  $('btnMirror').onclick = () => { mirror = !mirror; store.set('mirror', mirror ? '1' : '0'); applyMirror(); };
  applyMirror();
  $('video').addEventListener('error', () => vf.classList.add('off'));

  function describe(err) {
    const n = err && err.name;
    if (!window.isSecureContext) return 'The camera needs a secure context. Open this page via http://localhost:' + (location.port || '80') + ' or over https (not a LAN IP over http).';
    if (n === 'NotAllowedError' || n === 'SecurityError') return 'Camera permission was denied. Allow camera access for this site in the browser address bar, then try again.';
    if (n === 'NotFoundError' || n === 'OverconstrainedError') return 'No camera was found (or the selected one is unavailable). Connect a camera and try again.';
    if (n === 'NotReadableError' || n === 'AbortError') return 'The camera is busy or cannot be read. Close other apps that use it (Zoom, Teams, other tabs) and try again.';
    return 'Could not start the camera: ' + ((err && (err.message || err.name)) || 'unknown error');
  }

  /* ---------- start / stop ---------- */
  async function populate(curId) {
    try {
      const devs = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
      const sel = $('camSel'); sel.innerHTML = '';
      devs.forEach((d, i) => { const o = document.createElement('option'); o.value = d.deviceId; o.textContent = d.label || 'Camera ' + (i + 1); if (d.deviceId === curId) o.selected = true; sel.appendChild(o); });
      sel.hidden = !vf.classList.contains('cam') || devs.length < 2;
    } catch (e) {}
  }
  async function open(deviceId) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) throw Object.assign(new Error('no mediaDevices'), { name: 'Insecure' });
    const video = { width: { ideal: 1280 }, height: { ideal: 720 } };
    if (deviceId) video.deviceId = { exact: deviceId }; else video.facingMode = 'user';
    return navigator.mediaDevices.getUserMedia({ video, audio: false });
  }
  function halt() {
    running = false; if (abort) abort.abort(); abort = null;
    if (stream) stream.getTracks().forEach(t => t.stop()); stream = null;
    cam.srcObject = null;
    if (cam.getAttribute('src')) { cam.pause(); cam.removeAttribute('src'); cam.load(); }
    mode = null;
  }
  function begin() {            // shared: reset loop state and start capture + render
    if (window.setMjpeg) window.setMjpeg(false);
    items = []; done.length = 0; rtt = srvMs = banner = null; inflight = 0; encoding = false; seq = applied = 0; notReady = false; lastResp = 0; sentW = sentH = 0;
    gen++; abort = new AbortController(); running = true;
    loadStatus(); layout(); render(); schedule();
  }
  async function start(deviceId) {
    $('camErr').hidden = true;
    if (running) halt();
    let s;
    try { s = await open(deviceId); }
    catch (e) {
      if (deviceId && e.name === 'OverconstrainedError') { try { s = await open(null); } catch (e2) { showErr(describe(e2)); return; } }
      else { showErr(window.isSecureContext && navigator.mediaDevices ? describe(e) : describe({ name: 'Insecure' })); return; }
    }
    mode = 'cam'; SESSION = CAM_SESSION; SOURCE = null; closeMenu();
    stream = s; cam.loop = false; cam.srcObject = s;
    try { await cam.play(); } catch (e) {}
    const track = s.getVideoTracks()[0], cur = track && track.getSettings().deviceId;
    if (cur) store.set('camDev', cur);
    track.addEventListener('ended', () => { if (stream === s) { stop(); showErr('The camera was disconnected.'); } });
    setUi(true); await populate(cur);
    begin();
  }
  function stop() {
    halt(); setUi(false); $('banner').classList.remove('on'); ctx.clearRect(0, 0, ov.width, ov.height); items = [];
    if (window.setMjpeg) window.setMjpeg(true);
  }
  function toggle() { if (running) stop(); else start(store.get('camDev')); }
  $('btnCam').onclick = toggle; $('ctaLive').onclick = () => start(store.get('camDev'));

  /* ---------- recorded footage ---------- */
  async function loadClips() {
    try { const r = await fetch('/api/videos', { cache: 'no-store' }); clips = r.ok ? await r.json() : []; } catch (e) { clips = []; }
    if (!Array.isArray(clips)) clips = [];
    return clips;
  }
  const ttl = c => String(c.title || c.file || '').replace(/\.(webm|ogv|ogg|mp4|mov)$/i, '');
  const dur = d => d ? Math.floor(d / 60) + ':' + String(Math.round(d % 60)).padStart(2, '0') : '';
  function renderMenu() {
    const box = $('clipList');
    if (!clips.length) { box.innerHTML = '<div class="none">No recorded clips available yet (data/videos/manifest.json is missing or empty).</div>'; return; }
    box.innerHTML = clips.map((c, i) => {
      const hs = c.helmets == null ? '' : String(c.helmets), cls = /^(no\b|none|false|0$)/i.test(hs) ? 'nh' : /^(all|yes|true)\b/i.test(hs) ? 'h' : 'hm';
      const h = hs ? '<span class="' + cls + '">helmets: ' + esc(hs) + '</span>' : '';
      return `<button type="button" class="clip${i === clipIdx && mode === 'file' ? ' cur' : ''}" data-i="${i}"><b>${esc(ttl(c))}</b><span class="d">${esc(dur(c.duration_s))}</span>` +
        `<span class="m">${[esc(c.provider), esc(c.license), c.scene ? esc(c.scene) : '', h].filter(Boolean).join(' &middot; ')}</span></button>`;
    }).join('');
    box.querySelectorAll('.clip').forEach(b => { b.onclick = () => playClip(+b.dataset.i); });
  }
  function closeMenu() { $('clipMenu').hidden = true; vf.classList.remove('menuopen'); $('btnFiles').setAttribute('aria-expanded', 'false'); }
  async function openMenu() {
    $('clipMenu').hidden = false; vf.classList.add('menuopen'); $('btnFiles').setAttribute('aria-expanded', 'true');
    $('clipList').innerHTML = '<div class="none">Loading clips…</div>';
    await loadClips(); renderMenu();
    const f = $('clipList').querySelector('.clip.cur') || $('clipList').querySelector('.clip'); if (f) f.focus();
  }
  function toggleMenu() { if ($('clipMenu').hidden) openMenu(); else closeMenu(); }
  function playClip(i) {
    if (!clips || !clips.length) return;
    clipIdx = ((i % clips.length) + clips.length) % clips.length;
    const c = clips[clipIdx];
    $('camErr').hidden = true; closeMenu();
    if (running) halt();
    mode = 'file'; SESSION = rid(); SOURCE = 'video:' + c.file;      // fresh session per clip: trackers never mix
    cam.srcObject = null; cam.loop = loopClip; cam.muted = true;
    cam.src = '/videos/' + encodeURIComponent(c.file).replace(/%2F/g, '/');
    cam.play().catch(() => {});
    setUi(true); renderAttr(c); $('btnPlay').textContent = 'Pause';
    begin();
  }
  async function cycleClip() {
    if (!clips) await loadClips();
    if (!clips.length) { openMenu(); return; }
    playClip(mode === 'file' ? clipIdx + 1 : Math.max(0, clipIdx));
  }
  function renderAttr(c) {
    const link = c.source_page_url && /^https?:\/\//i.test(c.source_page_url) ? ` &middot; <a href="${esc(c.source_page_url)}" target="_blank" rel="noopener noreferrer">source &#8599;</a>` : '';
    $('hudAttr').innerHTML = [c.provider && 'via ' + esc(c.provider), c.license && esc(c.license)].filter(Boolean).join(' &middot; ') + link;
  }
  cam.addEventListener('ended', () => { if (mode === 'file' && !cam.loop) playClip(clipIdx + 1); });
  cam.addEventListener('error', () => { if (mode === 'file') showErr('Could not play this clip (' + ((((clips || [])[clipIdx]) || {}).file || '?') + '). Try another one.'); });
  cam.addEventListener('play', () => { $('btnPlay').textContent = 'Pause'; });
  cam.addEventListener('pause', () => { $('btnPlay').textContent = 'Play'; });
  $('btnFiles').onclick = toggleMenu; $('ctaFiles').onclick = openMenu;
  $('btnPlay').onclick = () => { if (cam.paused) cam.play().catch(() => {}); else cam.pause(); };
  $('btnNext').onclick = () => playClip(clipIdx + 1);
  $('btnLoop').setAttribute('aria-pressed', loopClip);
  $('btnLoop').onclick = () => { loopClip = !loopClip; cam.loop = loopClip; store.set('loopClip', loopClip ? '1' : '0'); $('btnLoop').setAttribute('aria-pressed', loopClip); };
  document.addEventListener('click', e => { if (!$('clipMenu').hidden && !$('clipMenu').contains(e.target) && !$('btnFiles').contains(e.target) && !$('ctaFiles').contains(e.target)) closeMenu(); });
  $('camSel').onchange = e => { store.set('camDev', e.target.value); start(e.target.value); };
  document.addEventListener('keydown', e => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (/^(INPUT|TEXTAREA|SELECT)$/.test((e.target || {}).tagName)) return;
    const k = e.key.toLowerCase();
    if (k === 'c') { if (mode === 'cam') stop(); else start(store.get('camDev')); }
    else if (k === 'v') cycleClip();
    else if (k === ' ' && mode === 'file') { e.preventDefault(); $('btnPlay').click(); }
    else if (k === 'escape' && !$('clipMenu').hidden) closeMenu();
  });

  async function loadStatus() {
    try {
      const r = await fetch('/api/live/status', { cache: 'no-store' });
      status = r.ok ? await r.json() : null;
    } catch (e) { status = null; }
    if (status && status.ready === false) { notReady = true; showErr('The live detector is not ready yet' + (status.error ? ' (' + status.error + ')' : '') + '. Retrying automatically.'); }
    else if (!status) { notReady = false; }
  }

  /* ---------- capture loop ---------- */
  function schedule() {
    if (!running) return;
    if (cam.requestVideoFrameCallback) cam.requestVideoFrameCallback(tick); else setTimeout(() => tick(), 33);
  }
  setInterval(() => { if (running && performance.now() - lastTick > 1000) tick(true); }, 1000);   // watchdog: rVFC can stall after a source switch
  function tick(kick) {
    if (!running) return;
    lastTick = performance.now();
    if (kick !== true) schedule();                       // watchdog kicks capture once without forking the rVFC chain
    if (cam.readyState < 2 || !cam.videoWidth) return;
    if (inflight + (encoding ? 1 : 0) >= DEPTH || encoding) return;   // drop frame instead of queueing
    if (performance.now() < retryAt) return;
    const vw = cam.videoWidth, vh = cam.videoHeight, sc = Math.min(1, MAXW / vw);
    const w = Math.round(vw * sc), h = Math.round(vh * sc);
    if (off.width !== w || off.height !== h) { off.width = w; off.height = h; }
    octx.drawImage(cam, 0, 0, w, h);
    encoding = true; const my = ++seq, g = gen;
    off.toBlob(blob => {
      encoding = false;
      if (!blob || !running || g !== gen) return;          // frame from a previous clip/camera: never mix sessions
      send(blob, my, w, h);
    }, 'image/jpeg', QUAL);
  }
  async function send(blob, my, w, h) {
    inflight++; const t0 = performance.now(), ac = abort, sess = SESSION;
    const hdrs = { 'Content-Type': 'image/jpeg', 'X-Session': SESSION }; if (SOURCE) hdrs['X-Source'] = SOURCE;
    try {
      const r = await fetch('/api/analyze', { method: 'POST', body: blob, headers: hdrs, signal: ac.signal, cache: 'no-store' });
      if (r.status === 503) { retryAt = performance.now() + 2000; notReady = true; showErr('The live detector is not ready (503). The server is still loading the model; retrying automatically.'); return; }
      if (!r.ok) { retryAt = performance.now() + 1500; showErr('Analyze request failed: HTTP ' + r.status); return; }
      const j = await r.json();
      if (!running || !j || j.ok === false) { if (j && j.ok === false) { retryAt = performance.now() + 1500; showErr('Server error: ' + (j.error || 'unknown')); } return; }
      if (notReady || !$('camErr').hidden) { notReady = false; $('camErr').hidden = true; if (!status) loadStatus(); }
      const now = performance.now();
      if (my < applied || sess !== SESSION) return;           // out-of-order / previous clip
      applied = my; sentW = w; sentH = h;
      rtt = now - t0; srvMs = typeof j.ms === 'number' ? j.ms : null; lastResp = now;
      done.push(now);
      ingest(j, w, h, now);
    } catch (e) {
      if (e.name === 'AbortError') return;
      retryAt = performance.now() + 1500; showErr('Cannot reach /api/analyze: ' + e.message);
    } finally { inflight--; }
  }

  /* ---------- smoothing ---------- */
  function iou(a, b) {
    const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]), x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
    const i = Math.max(0, x2 - x1) * Math.max(0, y2 - y1), u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i;
    return u > 0 ? i / u : 0;
  }
  function ingest(j, w, h, now) {
    respW = j.w || w; respH = j.h || h;
    const fresh = [];
    (j.people || []).forEach(p => fresh.push({ kind: 'p', id: p.id == null ? null : String(p.id), n: 'person', b: p.b, viol: !!p.violation, labels: p.labels || [] }));
    const hasPeople = fresh.length > 0;
    (j.dets || []).forEach(d => { if (d.n === 'person' && hasPeople) return; fresh.push({ kind: 'd', n: d.n, s: d.s, b: d.b }); });
    const used = new Set();
    fresh.forEach(f => {
      let best = null, bs = 0;
      items.forEach(o => {
        if (used.has(o) || o.kind !== f.kind || o.n !== f.n) return;
        let sc = 0;
        if (f.kind === 'p' && f.id != null && o.id === f.id) sc = 2; else sc = iou(o.t, f.b);
        if (sc > bs && sc > (f.kind === 'p' ? 0.15 : 0.2)) { bs = sc; best = o; }
      });
      if (best) { used.add(best); Object.assign(best, { t: f.b.slice(), viol: f.viol, labels: f.labels, s: f.s, id: f.id, seen: now }); }
      else { const o = Object.assign({}, f, { t: f.b.slice(), b: f.b.slice(), seen: now, born: now, key: nextKey++ }); items.push(o); used.add(o); }
    });
    banner = j.banner || null; if (banner) bannerAt = now;
  }

  /* ---------- geometry ---------- */
  let R = { x: 0, y: 0, w: 0, h: 0, dpr: 1 };
  function layout() {
    const fw = vf.clientWidth, fh = vf.clientHeight, vw = cam.videoWidth || 16, vh = cam.videoHeight || 9;
    const s = Math.min(fw / vw, fh / vh), w = vw * s, h = vh * s, dpr = Math.min(window.devicePixelRatio || 1, 2);
    R = { x: (fw - w) / 2, y: (fh - h) / 2, w, h, dpr };
    ov.style.left = R.x + 'px'; ov.style.top = R.y + 'px'; ov.style.width = w + 'px'; ov.style.height = h + 'px';
    const pw = Math.round(w * dpr), ph = Math.round(h * dpr);
    if (ov.width !== pw || ov.height !== ph) { ov.width = pw; ov.height = ph; }
  }
  if (window.ResizeObserver) new ResizeObserver(layout).observe(vf);
  window.addEventListener('resize', layout);
  cam.addEventListener('loadedmetadata', layout); cam.addEventListener('resize', layout);

  /* ---------- drawing ---------- */
  const FONT = '600 12px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace';
  function chip(c, text, x, y, fill, fg, size) {
    c.font = '700 ' + size + 'px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace';
    const tw = c.measureText(text).width, ph = size * 0.55, hh = size + ph, ww = tw + ph * 1.6;
    x = Math.max(0, Math.min(x, ov.width / R.dpr - ww)); y = Math.max(hh, y);
    c.fillStyle = fill; c.beginPath(); c.roundRect ? c.roundRect(x, y - hh, ww, hh, 4) : c.rect(x, y - hh, ww, hh); c.fill();
    c.fillStyle = fg; c.textBaseline = 'middle'; c.fillText(text, x + ph * 0.8, y - hh / 2 + 1);
  }
  function brackets(c, x, y, w, h, len, lw, color) {
    c.strokeStyle = color; c.lineWidth = lw; c.lineCap = 'round'; c.lineJoin = 'round';
    len = Math.min(len, w / 2, h / 2); c.beginPath();
    c.moveTo(x, y + len); c.lineTo(x, y); c.lineTo(x + len, y);
    c.moveTo(x + w - len, y); c.lineTo(x + w, y); c.lineTo(x + w, y + len);
    c.moveTo(x + w, y + h - len); c.lineTo(x + w, y + h); c.lineTo(x + w - len, y + h);
    c.moveTo(x + len, y + h); c.lineTo(x, y + h); c.lineTo(x, y + h - len); c.stroke();
  }
  function px(o) {         // source-pixel box -> overlay css px (mirror aware)
    const sw = respW || sentW || 1, sh = respH || sentH || 1, kx = R.w / sw, ky = R.h / sh;
    let x1 = o.b[0] * kx, x2 = o.b[2] * kx; const y1 = o.b[1] * ky, y2 = o.b[3] * ky;
    if (mirror && mode === 'cam') { const a = R.w - x2, b = R.w - x1; x1 = a; x2 = b; }
    return [x1, y1, x2 - x1, y2 - y1];
  }
  const pretty = s => String(s).replace(/_/g, ' ').toUpperCase();
  const isBad = n => /^no[_ ]|^none$|^without/i.test(n);
  let last = performance.now(), lastHud = '';
  function render() {
    if (!running) return;
    requestAnimationFrame(render);
    const now = performance.now(), dt = Math.min(0.1, (now - last) / 1000); last = now;
    const k = 1 - Math.exp(-dt * 20);
    const frozen = mode === 'file' && cam.paused;        // paused clip: keep last result on screen
    if (frozen) { items.forEach(o => { o.seen = now; }); if (banner) bannerAt = now; }
    // hold each box for ~2.5 response intervals (min 450 ms, max 1.5 s) so slow links don't flicker
    const gap = done.length > 1 ? (done[done.length - 1] - done[0]) / (done.length - 1) : 0;
    const hold = Math.max(450, Math.min(1500, gap * 2.5));
    items = items.filter(o => now - o.seen < hold);
    const c = ctx; c.setTransform(R.dpr, 0, 0, R.dpr, 0, 0); c.clearRect(0, 0, R.w, R.h);
    const pulse = 0.5 + 0.5 * Math.sin(now / 150);
    // draw PPE items first, people on top
    items.sort((a, b) => (a.kind === 'p') - (b.kind === 'p'));
    for (const o of items) {
      for (let i = 0; i < 4; i++) o.b[i] += (o.t[i] - o.b[i]) * k;
      const age = now - o.seen, fade = Math.min(1, (hold - age) / 250, (now - o.born) / 120 + 0.3);
      c.globalAlpha = Math.max(0, Math.min(1, fade));
      const [x, y, w, h] = px(o);
      if (o.kind === 'p') {
        const bad = o.viol, col = bad ? '#ff4d5a' : '#34e08a';
        c.save();
        if (bad) { c.shadowColor = 'rgba(255,60,75,' + (0.45 + 0.55 * pulse) + ')'; c.shadowBlur = 12 + 22 * pulse; }
        else { c.shadowColor = 'rgba(52,224,138,.5)'; c.shadowBlur = 8; }
        c.fillStyle = bad ? 'rgba(255,77,90,' + (0.06 + 0.07 * pulse) + ')' : 'rgba(52,224,138,.04)'; c.fillRect(x, y, w, h);
        brackets(c, x, y, w, h, Math.max(14, Math.min(w, h) * 0.18), bad ? 3.5 : 3, col);
        c.restore();
        const lab = bad ? (o.labels.length ? o.labels.map(pretty).join(' + ') : 'VIOLATION') : 'PPE OK';
        chip(c, (o.id != null ? '#' + o.id + ' ' : '') + lab, x, y - 3, col, bad ? '#fff' : '#04140b', 14);
      } else {
        const bad = isBad(o.n), col = bad ? '#ff6a75' : (/helmet|hat/.test(o.n) ? '#34e08a' : '#2fd7ff');
        c.lineWidth = 1.5; c.strokeStyle = col; c.strokeRect(x, y, w, h);
        c.fillStyle = bad ? 'rgba(255,77,90,.08)' : 'rgba(47,215,255,.05)'; c.fillRect(x, y, w, h);
        brackets(c, x, y, w, h, Math.min(10, w / 3, h / 3), 2.5, col);
        chip(c, pretty(o.n) + (o.s != null ? ' ' + Math.round(o.s * 100) : ''), x, y, col, '#041014', 11);
      }
    }
    c.globalAlpha = 1;
    // banner (held briefly to avoid flicker)
    const on = banner && now - bannerAt < 600;
    const bn = $('banner'); if (on) bn.textContent = banner; bn.classList.toggle('on', !!on);
    // HUD
    while (done.length && now - done[0] > 2000) done.shift();
    const fps = done.length > 1 ? (done.length - 1) / ((done[done.length - 1] - done[0]) / 1000) : null;
    const stale = lastResp && now - lastResp > 2500;
    const clip = mode === 'file' ? clips[clipIdx] || {} : null;
    const head = clip ? '<span class="chip rec" title="' + esc(ttl(clip)) + '"><span class="live demo"><i></i>RECORDED VIDEO &middot; ' + esc(ttl(clip)) + '</span></span>'
      : '<span class="chip live"><span class="live on"><i></i>LIVE CAMERA</span></span>';
    const html = head + (frozen ? '<span class="chip paused">PAUSED</span>' : '') +
      '<span class="chip">' + (fps != null && !stale && !frozen ? fps.toFixed(1) + ' FPS' : '-- FPS') + '</span>' +
      '<span class="chip">' + (rtt != null ? 'rtt ' + Math.round(rtt) + ' ms' : 'rtt --') + (srvMs != null ? ' &middot; server ' + Math.round(srvMs) + ' ms' : '') + '</span>';
    if (html !== lastHud) { $('hudLive').innerHTML = html; lastHud = html; }
    const meta = status ? [status.backend, status.model, status.remote ? 'remote' : (status.remote === false ? 'local' : null)].filter(Boolean).join('  ·  ') : (notReady ? 'detector not ready' : 'backend unknown');
    const res = cam.videoWidth + '×' + cam.videoHeight + ' → ' + (sentW ? sentW + '×' + sentH : '--');
    if ($('hudLb').textContent !== meta) $('hudLb').textContent = meta;
    if ($('hudRes').textContent !== res) $('hudRes').textContent = res;
  }
})();
