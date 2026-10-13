/* Digital twin: three.js scene driven only by SSE 'track' events (bbox + frame size).
   Schematic projection: bbox bottom-centre -> floor X/Z, bbox height -> person scale. */
(function () {
  'use strict';
  const FADE_AFTER = 1.5, FADE_LEN = .5;
  const SITE_W = 14, SITE_D = 9;
  const reduce = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
  const tracks = new Map();
  let canvas, wrap, mode = 'none', R = {}, last = performance.now(), nudge = 0, focus = null, onCount = () => {};
  let gl2d = null, running = true;

  function project(t) {
    const W = (t.size && t.size[0]) || 1920, H = (t.size && t.size[1]) || 1080, b = t.bbox;
    const cx = (b[0] + b[2]) / 2 / W, by = Math.min(1, Math.max(0, b[3] / H)), bh = Math.max(0, (b[3] - b[1]) / H);
    return { x: (cx - .5) * SITE_W, z: (by - .5) * SITE_D * 1.15 - .3, s: Math.min(1.35, Math.max(.62, .55 + bh * 1.5)) };
  }

  function glowTex(THREE, rgb) {
    const c = document.createElement('canvas'); c.width = c.height = 128; const g = c.getContext('2d');
    const gr = g.createRadialGradient(64, 64, 0, 64, 64, 64);
    gr.addColorStop(0, `rgba(${rgb},.95)`); gr.addColorStop(.35, `rgba(${rgb},.28)`); gr.addColorStop(1, `rgba(${rgb},0)`);
    g.fillStyle = gr; g.fillRect(0, 0, 128, 128); return new THREE.CanvasTexture(c);
  }
  function labelTex(THREE, text) {
    const c = document.createElement('canvas'); c.width = 256; c.height = 80; const g = c.getContext('2d');
    g.font = '600 36px ui-monospace,Menlo,monospace'; g.textAlign = 'center'; g.textBaseline = 'middle';
    const w = g.measureText(text).width + 40; g.fillStyle = 'rgba(8,9,11,.72)';
    g.beginPath(); g.roundRect ? g.roundRect(128 - w / 2, 12, w, 56, 28) : g.rect(128 - w / 2, 12, w, 56); g.fill();
    g.strokeStyle = 'rgba(255,255,255,.25)'; g.lineWidth = 2; g.stroke();
    g.fillStyle = '#fff'; g.fillText(text, 128, 41); const t = new THREE.CanvasTexture(c); t.minFilter = THREE.LinearFilter; return t;
  }

  function initGL() {
    const THREE = window.THREE;
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'high-performance' });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    const scene = new THREE.Scene(); scene.fog = new THREE.FogExp2(0x07080a, .045);
    const cam = new THREE.PerspectiveCamera(42, 1, .1, 80);
    scene.add(new THREE.HemisphereLight(0xbcc8ff, 0x1a1408, .9));
    const key = new THREE.DirectionalLight(0xfff0d0, 1.1); key.position.set(4, 9, 6); scene.add(key);
    const grid = new THREE.GridHelper(SITE_W + 6, 28, 0x3a3f4a, 0x1c2027); scene.add(grid);
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(SITE_W + 6, SITE_W + 6), new THREE.MeshBasicMaterial({ color: 0x0b0d11, transparent: true, opacity: .75 }));
    floor.rotation.x = -Math.PI / 2; floor.position.y = -.01; scene.add(floor);
    // site boundary
    const hw = SITE_W / 2, hd = SITE_D / 2 * 1.15;
    const bp = [[-hw, -hd], [hw, -hd], [hw, hd], [-hw, hd], [-hw, -hd]].map(p => new THREE.Vector3(p[0], .01, p[1] - .3));
    scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(bp), new THREE.LineBasicMaterial({ color: 0xffb000, transparent: true, opacity: .55 })));
    // camera marker + view wedge (near edge = bottom of frame)
    const camMark = new THREE.Group(); camMark.position.set(0, .25, hd + 1.1);
    camMark.add(new THREE.Mesh(new THREE.BoxGeometry(.5, .3, .35), new THREE.MeshStandardMaterial({ color: 0x2a2f38, flatShading: true })));
    const lens = new THREE.Mesh(new THREE.CylinderGeometry(.09, .09, .2, 12), new THREE.MeshBasicMaterial({ color: 0xffb000 })); lens.rotation.x = Math.PI / 2; lens.position.z = -.25; camMark.add(lens);
    scene.add(camMark);
    const wedge = [[0, .02, hd + .9], [-hw, .02, -hd - .3], [hw, .02, -hd - .3]].map(p => new THREE.Vector3(...p));
    const wg = new THREE.BufferGeometry().setFromPoints([wedge[0], wedge[1], wedge[0], wedge[2]]);
    scene.add(new THREE.LineSegments(wg, new THREE.LineBasicMaterial({ color: 0xffb000, transparent: true, opacity: .13 })));
    // dust
    const N = 260, pos = new Float32Array(N * 3), vel = new Float32Array(N);
    for (let i = 0; i < N; i++) { pos[i * 3] = (Math.random() - .5) * 18; pos[i * 3 + 1] = Math.random() * 6; pos[i * 3 + 2] = (Math.random() - .5) * 14; vel[i] = .05 + Math.random() * .12; }
    const dg = new THREE.BufferGeometry(); dg.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const dust = new THREE.Points(dg, new THREE.PointsMaterial({ size: .06, map: glowTex(THREE, '255,214,140'), transparent: true, opacity: .55, depthWrite: false, blending: THREE.AdditiveBlending }));
    scene.add(dust);
    // shared geometry/textures
    const G = {
      body: new THREE.CylinderGeometry(.2, .26, 1.0, 7), head: new THREE.IcosahedronGeometry(.17, 0),
      hat: new THREE.SphereGeometry(.215, 8, 5, 0, Math.PI * 2, 0, Math.PI / 2), brim: new THREE.CylinderGeometry(.27, .27, .03, 10),
      ring: new THREE.RingGeometry(.55, .68, 48), beam: new THREE.CylinderGeometry(.05, .05, 4.5, 8, 1, true), legs: new THREE.CylinderGeometry(.16, .18, .55, 6)
    };
    const T = { green: glowTex(THREE, '52,224,138'), red: glowTex(THREE, '255,77,90') };
    R = { THREE, renderer, scene, cam, dust, dustPos: pos, dustVel: vel, G, T, camMark };
    resize();
  }

  function makeWorker(id) {
    const { THREE, G, T, scene } = R;
    const skin = new THREE.MeshStandardMaterial({ color: 0xd9b48f, flatShading: true, transparent: true });
    const vest = new THREE.MeshStandardMaterial({ color: 0x2c3340, flatShading: true, transparent: true, emissive: 0x0a1a12 });
    const pants = new THREE.MeshStandardMaterial({ color: 0x1b1f27, flatShading: true, transparent: true });
    const hatM = new THREE.MeshStandardMaterial({ color: 0xffb000, flatShading: true, transparent: true, emissive: 0x442b00 });
    const g = new THREE.Group(), inner = new THREE.Group(); g.add(inner);
    const legs = new THREE.Mesh(G.legs, pants); legs.position.y = .275;
    const body = new THREE.Mesh(G.body, vest); body.position.y = 1.05;
    const head = new THREE.Mesh(G.head, skin); head.position.y = 1.78;
    const hat = new THREE.Group(); const dome = new THREE.Mesh(G.hat, hatM); const brim = new THREE.Mesh(G.brim, hatM); brim.position.y = .005; hat.add(dome, brim); hat.position.y = 1.82;
    inner.add(legs, body, head, hat);
    const ringM = new THREE.MeshBasicMaterial({ color: 0xff4d5a, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false, blending: THREE.AdditiveBlending });
    const ring = new THREE.Mesh(G.ring, ringM); ring.rotation.x = -Math.PI / 2; ring.position.y = .03; g.add(ring);
    const beamM = new THREE.MeshBasicMaterial({ color: 0xff4d5a, transparent: true, opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide });
    const beam = new THREE.Mesh(G.beam, beamM); beam.position.y = 2.25; g.add(beam);
    const glowM = new THREE.SpriteMaterial({ map: T.green, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, opacity: .55 });
    const glow = new THREE.Sprite(glowM); glow.scale.set(1.8, 1.8, 1); glow.position.y = .06; g.add(glow);
    const lblM = new THREE.SpriteMaterial({ map: labelTex(THREE, '#' + id), transparent: true, depthWrite: false, depthTest: false });
    const lbl = new THREE.Sprite(lblM); lbl.scale.set(1.25, .39, 1); g.add(lbl);
    scene.add(g);
    return { g, inner, hat, hatM, skin, vest, pants, ring, ringM, beamM, glowM, glow, lbl, lblM, parts: [skin, vest, pants, hatM] };
  }

  function disposeWorker(w) { R.scene.remove(w.g); w.lblM.map.dispose(); [w.skin, w.vest, w.pants, w.hatM, w.ringM, w.beamM, w.glowM, w.lblM].forEach(m => m.dispose()); }

  function update(list) {
    const now = performance.now() / 1000;
    for (const t of list || []) {
      if (!t.bbox || t.id == null) continue;
      if (t.person === false && !t.violation) continue;       // e.g. hardhat box: not a person
      const key = String(t.id), p = project(t);
      let tr = tracks.get(key);
      if (!tr) {
        tr = { id: key, x: p.x, z: p.z, s: p.s, tx: p.x, tz: p.z, ts: p.s, born: now, viol: false, w: null, vt: 0, alpha: 0 };
        tracks.set(key, tr);
      }
      tr.tx = p.x; tr.tz = p.z; tr.ts = p.s; tr.seen = now;
      const v = !!t.violation;
      if (v && !tr.viol) { nudge = 1; focus = tr; tr.vt = 0; }
      tr.viol = v;
    }
    if (mode === '2d') draw2d();
    onCount(tracks.size);
  }

  function frame(tm) {
    requestAnimationFrame(frame);
    if (!running || document.hidden) { last = tm; return; }
    const dt = Math.min(.05, (tm - last) / 1000); last = tm; const now = tm / 1000;
    if (mode === '2d') { draw2d(); return; }
    if (mode !== 'gl') return;
    const { THREE, scene, cam, renderer } = R;
    const k = 1 - Math.exp(-dt * 9);
    for (const [id, tr] of tracks) {
      const age = now - tr.seen;
      if (age > FADE_AFTER + FADE_LEN) { if (tr.w) disposeWorker(tr.w); tracks.delete(id); if (focus === tr) focus = null; onCount(tracks.size); continue; }
      if (!tr.w) tr.w = makeWorker(tr.id);
      const w = tr.w;
      tr.x += (tr.tx - tr.x) * k; tr.z += (tr.tz - tr.z) * k; tr.s += (tr.ts - tr.s) * k;
      const fade = age > FADE_AFTER ? 1 - (age - FADE_AFTER) / FADE_LEN : 1;
      tr.alpha += (fade - tr.alpha) * Math.min(1, dt * 8);
      const a = Math.max(0, tr.alpha);
      w.g.position.set(tr.x, 0, tr.z); w.inner.scale.setScalar(tr.s);
      w.inner.position.y = reduce ? 0 : Math.sin(now * 1.6 + tr.born) * .012;
      w.parts.forEach(m => m.opacity = a); w.lblM.opacity = a;
      w.lbl.position.y = 2.35 * tr.s + .25;
      const v = tr.viol; tr.vt += dt;
      w.hat.visible = !v;
      w.skin.emissive.setHex(v ? 0x661018 : 0x000000);
      w.vest.emissive.setHex(v ? 0x4a0a10 : 0x0a1a12);
      const pulse = reduce ? .6 : .5 + .5 * Math.sin(tr.vt * 6);
      w.ringM.opacity = v ? a * (.5 + .5 * pulse) : 0; const rs = v ? 1 + (reduce ? .3 : (tr.vt * .9 % 1.4)) : 1; w.ring.scale.set(rs * tr.s, rs * tr.s, 1);
      w.beamM.opacity = v ? a * (.16 + .14 * pulse) : 0;
      w.glowM.map = v ? R.T.red : R.T.green; w.glowM.opacity = a * (v ? .85 : .4);
    }
    // camera: slow orbit + violation nudge
    nudge = Math.max(0, nudge - dt * .8);
    const ang = reduce ? 0 : Math.sin(now * .09) * .5, rad = 12.8;
    const fx = focus && focus.w ? focus.x * .25 : 0, fz = focus && focus.w ? focus.z * .15 : 0;
    const zoom = 1 - nudge * .12;
    cam.position.set(Math.sin(ang) * rad * zoom + fx * nudge, 6.6 * zoom, Math.cos(ang) * rad * zoom);
    cam.lookAt(fx * nudge, .6, fz * nudge + .2);
    if (!reduce) {
      const p = R.dustPos; for (let i = 0; i < p.length / 3; i++) { p[i * 3 + 1] += R.dustVel[i] * dt; p[i * 3] += Math.sin(now * .3 + i) * dt * .05; if (p[i * 3 + 1] > 6) p[i * 3 + 1] = 0; }
      R.dust.geometry.attributes.position.needsUpdate = true;
    }
    renderer.render(scene, cam);
  }

  function resize() {
    if (!wrap) return; const w = wrap.clientWidth, h = wrap.clientHeight; if (!w || !h) return;
    if (mode === 'gl') { R.renderer.setSize(w, h, false); R.cam.aspect = w / h; R.cam.updateProjectionMatrix(); }
    else if (mode === '2d') { const d = Math.min(window.devicePixelRatio || 1, 2); canvas.width = w * d; canvas.height = h * d; draw2d(); }
  }

  /* 2D fallback: top-down site map on canvas */
  function draw2d() {
    const c = gl2d || (gl2d = canvas.getContext('2d')); const W = canvas.width, H = canvas.height; if (!W) return;
    c.clearRect(0, 0, W, H); const d = W / (wrap.clientWidth || W); c.save(); c.scale(d, d);
    const w = W / d, h = H / d, now = performance.now() / 1000;
    c.strokeStyle = 'rgba(255,255,255,.07)'; c.lineWidth = 1;
    for (let x = 0; x < w; x += 32) { c.beginPath(); c.moveTo(x, 0); c.lineTo(x, h); c.stroke(); }
    for (let y = 0; y < h; y += 32) { c.beginPath(); c.moveTo(0, y); c.lineTo(w, y); c.stroke(); }
    for (const [id, tr] of [...tracks]) {
      const age = now - (tr.seen || now); if (age > FADE_AFTER + FADE_LEN) { tracks.delete(id); continue; }
      const a = age > FADE_AFTER ? 1 - (age - FADE_AFTER) / FADE_LEN : 1;
      tr.x += (tr.tx - tr.x) * .2; tr.z += (tr.tz - tr.z) * .2;
      const px = w / 2 + tr.x / SITE_W * w * .9, py = h / 2 + tr.z / (SITE_D * 1.2) * h * .85, col = tr.viol ? '255,77,90' : '52,224,138';
      c.globalAlpha = a; const gr = c.createRadialGradient(px, py, 0, px, py, 30); gr.addColorStop(0, `rgba(${col},.5)`); gr.addColorStop(1, `rgba(${col},0)`);
      c.fillStyle = gr; c.fillRect(px - 30, py - 30, 60, 60); c.fillStyle = `rgb(${col})`; c.beginPath(); c.arc(px, py, 6, 0, 7); c.fill();
      c.fillStyle = '#fff'; c.font = '11px ui-monospace,monospace'; c.fillText('#' + tr.id, px + 10, py + 4);
    }
    c.restore();
  }

  window.Twin = {
    init(cv, wr, cb) {
      canvas = cv; wrap = wr; onCount = cb || onCount;
      let ok = false;
      if (window.THREE) { try { initGL(); mode = 'gl'; ok = true; } catch (e) { console.warn('WebGL unavailable, using 2D fallback', e); } }
      if (!ok) { mode = '2d'; }
      new ResizeObserver(resize).observe(wrap); resize();
      requestAnimationFrame(frame);
      return mode;
    },
    update, clear() { for (const t of tracks.values()) if (t.w) disposeWorker(t.w); tracks.clear(); onCount(0); },
    get mode() { return mode; }, get size() { return tracks.size; }
  };
})();
