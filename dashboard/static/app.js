(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const reduce = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
  const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const store = { get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }, set(k, v) { try { localStorage.setItem(k, v); } catch (e) {} } };

  /* ---------- count-up ---------- */
  const shown = new WeakMap();
  function countTo(el, to, fmt) {
    fmt = fmt || (v => Math.round(v).toString());
    const from = shown.has(el) ? shown.get(el) : 0;
    if (el._raf) cancelAnimationFrame(el._raf);
    if (reduce || from === to) { shown.set(el, to); el.textContent = fmt(to); return; }
    const t0 = performance.now(), dur = 900;
    const step = now => {
      const p = Math.min(1, (now - t0) / dur), e = p === 1 ? 1 : 1 - Math.pow(2, -10 * p), v = from + (to - from) * e;
      shown.set(el, v); el.textContent = fmt(v); if (p < 1) el._raf = requestAnimationFrame(step);
    };
    el._raf = requestAnimationFrame(step);
  }

  /* ---------- sound ---------- */
  let actx = null, sound = false;
  function beep() {
    if (!sound || !actx) return;
    [0, .16, .32].forEach(d => {
      const o = actx.createOscillator(), g = actx.createGain(); o.type = 'sine'; o.frequency.value = 880;
      g.gain.setValueAtTime(0, actx.currentTime + d); g.gain.linearRampToValueAtTime(.16, actx.currentTime + d + .01); g.gain.exponentialRampToValueAtTime(.001, actx.currentTime + d + .13);
      o.connect(g); g.connect(actx.destination); o.start(actx.currentTime + d); o.stop(actx.currentTime + d + .14);
    });
  }
  $('btnSound').onclick = () => {
    sound = !sound;
    if (sound) { actx = actx || new (window.AudioContext || window.webkitAudioContext)(); actx.resume(); beep(); }
    $('btnSound').setAttribute('aria-pressed', sound); $('soundTxt').textContent = sound ? 'Sound on' : 'Sound off';
  };
  function flash() { const f = $('flash'); f.classList.remove('go'); void f.offsetWidth; f.classList.add('go'); }

  /* ---------- video HUD ---------- */
  const vid = $('video');
  let mjpeg = true;
  function startVideo() { if (mjpeg) vid.src = '/video?t=' + Date.now(); }
  vid.onerror = () => setTimeout(startVideo, 2500);
  window.setMjpeg = on => { mjpeg = on; if (on) startVideo(); else { vid.removeAttribute('src'); } };
  startVideo();
  const base = p => String(p == null ? '' : p).split(/[\\/]/).pop();
  let lastVideo = {};
  function video(v) {
    if (!v) return; lastVideo = v;
    $('vframe').classList.toggle('off', !v.live);
    const lb = $('liveBadge'), isCam = !!v.source && v.source.startsWith('camera');
    lb.className = 'live ' + (v.live ? (isCam ? 'on' : 'demo') : '');
    $('liveTxt').textContent = !v.live ? 'OFFLINE' : isCam ? 'LIVE CAMERA' : 'RECORDED VIDEO' + (v.source ? ': ' + base(v.source) : '');
    $('hFps').textContent = v.live ? v.fps.toFixed(1) + ' FPS' : '-- FPS';
    $('hFps').parentElement.style.display = v.live ? '' : 'none';
    const m = [v.backend, v.model && base(v.model), v.infer_fps != null ? 'infer ' + Number(v.infer_fps).toFixed(1) + ' fps' : null].filter(Boolean);
    $('hMeta').textContent = v.live ? m.join('  ·  ') : '';
    $('hRes').textContent = v.live && v.width ? v.width + '×' + v.height : '';
    $('pSrc').textContent = v.source ? (isCam ? 'live camera' : 'recorded video: ' + base(v.source)) : '—';
    $('pModel').textContent = [v.model && base(v.model), v.backend].filter(Boolean).join(' on ') || '—';
    $('pFps').textContent = v.live ? 'video ' + v.fps.toFixed(1) + ' fps measured' + (v.infer_fps != null ? ', inference ' + Number(v.infer_fps).toFixed(1) + ' fps' : '') : 'no frames arriving';
  }

  /* ---------- KPIs ---------- */
  let lastStats = null;
  function stats(s) {
    lastStats = s;
    countTo($('kPers'), s.unique_persons || 0); countTo($('kViol'), s.unique_violators || 0); countTo($('kInc'), s.incidents || 0);
    $('kViol').classList.toggle('hot', s.unique_violators > 0); $('kInc').classList.toggle('hot', s.incidents > 0);
    const c = s.compliance, rv = $('ringV'), kc = $('kComp');
    if (c == null) { kc.textContent = '—'; shown.delete(kc); rv.setAttribute('stroke-dasharray', '0 100'); }
    else {
      countTo(kc, c, v => (c % 1 ? v.toFixed(1) : Math.round(v)) + '%');
      rv.setAttribute('stroke-dasharray', c + ' 100'); rv.setAttribute('class', 'rv' + (c >= 90 ? '' : c >= 70 ? ' warn' : ' bad'));
    }
    if (s.timeline) spark(s.timeline);
    video(s.video); status(s.status || {});
    $('drRaw').textContent = JSON.stringify({ messages: s.messages, label_counts: s.label_counts, status: s.status, video: s.video }, null, 2);
  }
  function spark(a) {
    const W = 300, H = 80, n = a.length, mx = Math.max(1, ...a), bw = W / n, y = v => H - 4 - v / mx * (H - 14);
    let s = '<defs><linearGradient id="tg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ff4d5a" stop-opacity=".55"/><stop offset="1" stop-color="#ff4d5a" stop-opacity="0"/></linearGradient></defs>';
    for (let i = 1; i < 4; i++) s += `<line x1="0" x2="${W}" y1="${H * i / 4}" y2="${H * i / 4}" stroke="rgba(255,255,255,.05)" vector-effect="non-scaling-stroke"/>`;
    const pts = a.map((v, i) => [i * bw + bw / 2, y(v)]);
    s += `<polygon points="0,${H} ${pts.map(p => p.join(',')).join(' ')} ${W},${H}" fill="url(#tg)"/>`;
    s += `<polyline points="${pts.map(p => p.join(',')).join(' ')}" fill="none" stroke="#ff4d5a" stroke-width="1.6" stroke-linejoin="round" vector-effect="non-scaling-stroke"/>`;
    const li = a.length - 1; s += `<line x1="${pts[li][0]}" x2="${pts[li][0]}" y1="0" y2="${H}" stroke="rgba(255,176,0,.5)" stroke-dasharray="2 3" vector-effect="non-scaling-stroke"/>`;
    $('tlSvg').innerHTML = s; $('tlPeak').textContent = 'peak ' + (a.some(x => x) ? Math.max(...a) : 0) + '/min';
  }

  /* ---------- connection ---------- */
  let sseOk = false, srcStatus = {};
  function status(st) { srcStatus = st; conn(); }
  function conn() {
    const e = Object.entries(srcStatus), ok = sseOk && (!e.length || e.every(([k, v]) => /^(connected|tailing)/.test(v)));
    $('conn').className = 'pill ' + (!sseOk ? '' : ok ? 'ok' : 'warn');
    $('ctext').textContent = !sseOk ? 'Reconnecting…' : e.length ? (ok ? 'Live · ' + e.map(([k]) => k).join(', ') : 'Source problem') : 'Live · no event source';
    $('conn').title = e.map(([k, v]) => k + ': ' + v).join(' | ');
  }

  /* ---------- incident feed ---------- */
  let cardCount = 0, lastInc = [];
  function when(c) { return new Date(c.received * 1000).toLocaleTimeString([], { hour12: false }); }
  function addCard(c, isNew) {
    const empty = $('empty'); if (empty) empty.remove();
    const b = document.createElement('button'); b.type = 'button'; b.className = 'card' + (isNew ? ' new' : '');
    const conf = c.conf != null && !isNaN(c.conf) ? Math.round(c.conf * 100) + '%' : 'n/a';
    b.innerHTML = `<div class="thumb">${c.evidence ? `<img alt="Evidence for track ${esc(c.id)}" loading="lazy" src="${esc(c.evidence)}">` : 'no img'}</div>
      <div class="cinfo"><h4>${esc(String(c.label).replace(/_/g, ' '))}</h4>
      <div class="cmeta"><span>track <b>${esc(c.id == null ? 'n/a' : c.id)}</b></span><span>conf <b>${conf}</b></span><span class="mono">${when(c)}</span></div></div><span class="cn mono">#${c.n}</span>`;
    const im = b.querySelector('img'); if (im) im.onerror = () => { im.parentElement.textContent = 'no img'; };
    b.onclick = () => openLb(c);
    const box = $('cards'); box.prepend(b);
    while (box.children.length > 60) box.lastChild.remove();
    cardCount = box.querySelectorAll('.card').length; $('feedCount').textContent = cardCount;
  }
  function clearFeed() {
    $('cards').innerHTML = '<div class="empty" id="empty"><b>All clear</b><span>No violations detected.</span></div>'; $('feedCount').textContent = 0;
  }

  /* ---------- lightbox ---------- */
  let lbFrom = null;
  function openLb(c) {
    lbFrom = document.activeElement; const box = $('lbImg'), wrap = box.parentElement;
    wrap.classList.remove('none'); box.onerror = () => wrap.classList.add('none');
    if (c.evidence) box.src = c.evidence; else wrap.classList.add('none');
    $('lbTitle').textContent = String(c.label).replace(/_/g, ' ');
    const rows = [['track', c.id], ['confidence', c.conf != null ? (c.conf * 100).toFixed(1) + '%' : 'n/a'], ['received', new Date(c.received * 1000).toLocaleString()], ['frame', c.frame], ['source', c.source], ['incident', '#' + c.n]];
    $('lbDl').innerHTML = rows.map(r => `<dt>${r[0]}</dt><dd>${esc(r[1] == null ? 'n/a' : r[1])}</dd>`).join('');
    $('lbRaw').textContent = JSON.stringify(c, null, 2);
    $('lb').hidden = false; $('lbClose').focus();
  }
  function closeLb() { $('lb').hidden = true; if (lbFrom && lbFrom.focus) lbFrom.focus(); }
  $('lbClose').onclick = closeLb; $('lb').onclick = e => { if (e.target === $('lb')) closeLb(); };

  /* ---------- drawer / modes / reset ---------- */
  function drawer(open) {
    const d = $('drawer'); d.classList.toggle('open', open); d.setAttribute('aria-hidden', !open); $('btnHow').setAttribute('aria-expanded', open);
    if (open) $('drClose').focus();
  }
  $('btnHow').onclick = () => drawer(!$('drawer').classList.contains('open')); $('drClose').onclick = () => { drawer(false); $('btnHow').focus(); };
  function pres(on) { document.body.classList.toggle('pres', on); $('btnPres').setAttribute('aria-pressed', on); store.set('pres', on ? '1' : '0'); setTimeout(() => window.dispatchEvent(new Event('resize')), 50); }
  $('btnPres').onclick = () => pres(!document.body.classList.contains('pres'));
  function fs() { if (document.fullscreenElement) document.exitFullscreen(); else if (document.documentElement.requestFullscreen) document.documentElement.requestFullscreen().catch(() => {}); }
  $('btnFs').onclick = fs;
  let rstT = null;
  $('btnReset').onclick = () => {
    const b = $('btnReset');
    if (!rstT) { b.textContent = 'Confirm reset?'; rstT = setTimeout(() => { b.textContent = 'Reset'; rstT = null; }, 3000); return; }
    clearTimeout(rstT); rstT = null; b.textContent = 'Reset'; fetch('/api/reset', { method: 'POST' }).catch(() => {});
  };
  document.addEventListener('keydown', e => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key.toLowerCase();
    if (k === 'escape') { if (!$('lb').hidden) closeLb(); else drawer(false); }
    else if (k === 'p') $('btnPres').click(); else if (k === 'f') fs();
  });
  if (store.get('pres') === '1') pres(true);

  /* magnetic hover */
  if (!reduce) document.querySelectorAll('.btn').forEach(b => {
    b.addEventListener('pointermove', e => { const r = b.getBoundingClientRect(); b.style.transform = `translate(${(e.clientX - r.left - r.width / 2) * .15}px,${(e.clientY - r.top - r.height / 2) * .25}px)`; });
    b.addEventListener('pointerleave', () => { b.style.transform = ''; });
  });

  /* ---------- twin ---------- */
  const tmode = window.Twin ? Twin.init($('twinCanvas'), $('twinWrap'), n => { $('twinCount').textContent = n + ' tracked'; $('twinEmpty').classList.toggle('hide', n > 0); }) : 'none';
  $('twinTag').textContent = tmode === 'gl' ? 'WebGL · three.js' : '2D canvas fallback';

  /* ---------- data ---------- */
  function clearAll() { clearFeed(); if (window.Twin) Twin.clear(); }
  fetch('/api/state').then(r => r.json()).then(j => {
    stats(j.stats); if (!$('cards').querySelector('.card')) j.incidents.slice().reverse().forEach(c => addCard(c, false));
  }).catch(() => {});
  function connect() {
    const es = new EventSource('/stream');
    es.onopen = () => { sseOk = true; conn(); };
    es.onerror = () => { sseOk = false; conn(); if (es.readyState === 2) setTimeout(connect, 2000); };
    es.onmessage = ev => {
      sseOk = true; let m; try { m = JSON.parse(ev.data); } catch (e) { return; }
      switch (m.type) {
        case 'stats': stats(m.data); break;
        case 'track': if (window.Twin) Twin.update(m.data); break;
        case 'incident': addCard(m.data, true); flash(); beep(); break;
        case 'reset': clearAll(); stats(m.data); break;
      }
    };
  }
  connect();
  setInterval(() => { if (lastStats && !sseOk) return; fetch('/api/state').then(r => r.json()).then(j => stats(j.stats)).catch(() => {}); }, 10000);
})();
