// Dronomucha 3D: the search of the real-scale simulation, drawn as a game draws its world.
// Load the scene (python -m wingloop.dronomucha.sym3d.eksport), build the valley in the
// chosen style, and play any platform's flight -- or, recording, render the frame asked for.
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import * as S from './swiat.js';
import * as D from './drony.js';
import { STYLE } from './style.js';
import { KLIPY } from './klipy.js';

const params = new URLSearchParams(location.search);
const BEZ = new Set((params.get('bez') || '').split(',')); // stages turned off, for measuring
const NAGRANIE = params.has('nagranie');
const KLIP = params.get('klip') ? KLIPY[params.get('klip')] : null;
const HASH = location.hash.replace('#', '');
const STYL_NAZWA = (KLIP && KLIP.styl) || params.get('styl') || (HASH in STYLE ? HASH : 'realizm');
const STYL = STYLE[STYL_NAZWA];
const W = Number(params.get('w') || 0), H = Number(params.get('h') || 0);
document.documentElement.dataset.styl = STYL_NAZWA;
if (NAGRANIE) document.body.classList.add('nagrywanie');

const FAZY = ['PRZESZUKIWANIE', 'POTWIERDZANIE', 'POWRÓT NA PAS', 'POWRÓT DO BAZY'];
const NAZWY = { samolot: 'Samolot piankowy', quad: 'Quadrocopter', vtol: 'Tiltrotor' };

// ---------------------------------------------------------------- the flight data
function b64f(s) { const bin = atob(s); const u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return new Float32Array(u.buffer); }

class Lot {
  constructor(d) {
    this.d = d; this.nazwa = d.nazwa; this.k = d.kolumny.length; this.n = d.n;
    this.w = b64f(d.wiersze);
    this.zd = d.zdarzenia.slice().sort((a, b) => a.t - b.t);
    this.koniec = this.w[(this.n - 1) * this.k];
  }
  col(i, c) { return this.w[i * this.k + c]; }
  idx(t) {
    let lo = 0, hi = this.n - 1;
    if (t <= this.col(0, 0)) return [0, 0];
    if (t >= this.col(hi, 0)) return [hi, 0];
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (this.col(m, 0) <= t) lo = m; else hi = m; }
    const f = (t - this.col(lo, 0)) / Math.max(1e-6, this.col(lo + 1, 0) - this.col(lo, 0));
    return [lo, f];
  }
  // Where it is at t: x, y, z, kurs, v, zawis, krazy, faza, bateria, agl.
  w_chwili(t) {
    const [i, f] = this.idx(t), j = Math.min(i + 1, this.n - 1);
    const L = (c) => this.col(i, c) * (1 - f) + this.col(j, c) * f;
    let k0 = this.col(i, 4), k1 = this.col(j, 4);
    let dk = ((k1 - k0 + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
    return { t, x: L(1), y: L(2), z: L(3), kurs: k0 + dk * f, v: L(5), zawis: this.col(f < 0.5 ? i : j, 6) > 0.5, krazy: this.col(i, 7), faza: this.col(i, 8) | 0, bateria: L(9), agl: L(10) };
  }
  // The heading averaged over a few seconds: what a chasing camera follows.
  kursSredni(t, okno = 2.5) {
    let sx = 0, sy = 0;
    for (let k = 0; k <= 10; k++) { const s = this.w_chwili(t - okno + okno * k / 10); sx += Math.cos(s.kurs) * (1 + k * 0.2); sy += Math.sin(s.kurs) * (1 + k * 0.2); }
    return Math.atan2(sy, sx);
  }
  kandydaci(t) {
    const m = new Map();
    for (const e of this.zd) { if (e.t > t) break; m.set(e.id, { ...e, od: e.t }); }
    return [...m.values()];
  }
}

// ---------------------------------------------------------------- the renderer and scene
const canvas = document.getElementById('scena');
// No logarithmic depth: writing the depth from the fragment shader turns off the early depth
// test, and a software renderer then shades every hidden fragment of the valley too.
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, preserveDrawingBuffer: true, powerPreference: 'high-performance' });
const width = W || window.innerWidth, height = H || window.innerHeight;
renderer.setPixelRatio(W ? 1 : Math.min(window.devicePixelRatio, 1.5));
renderer.setSize(width, height, false);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = STYL.ekspozycja;
renderer.shadowMap.enabled = !new URLSearchParams(location.search).get('bez')?.includes('cienie');
renderer.shadowMap.type = THREE.PCFShadowMap;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(48, width / height, 0.25, 40000);

async function wczytaj() {
  const url = params.get('dane') || 'scena.json';
  const r = await fetch(url);
  return r.json();
}

const dane = await wczytaj();
const tr = new S.Teren(dane.teren);
const tex = S.przygotujTekstury(tr);
const loty = Object.fromEntries(Object.entries(dane.loty).map(([k, v]) => [k, new Lot(v)]));
let lotNazwa = (KLIP && KLIP.lot) || params.get('lot') || 'samolot';

// Light: the sun (or the moon), the sky's light, and the sky itself.
const sunDir = S.kierunekSlonca(STYL.slonce.az, STYL.slonce.el);
const sun = new THREE.DirectionalLight(STYL.slonce.kolor, STYL.slonce.moc);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.6;
const sc = sun.shadow.camera; sc.left = -220; sc.right = 220; sc.top = 220; sc.bottom = -220; sc.near = 10; sc.far = 4000;
scene.add(sun, sun.target);
scene.add(new THREE.HemisphereLight(STYL.polkula.niebo, STYL.polkula.ziemia, STYL.polkula.moc));
scene.fog = new THREE.FogExp2(STYL.mgla.kolor, STYL.mgla.gestosc);

const pmrem = new THREE.PMREMGenerator(renderer);
let env;
if (STYL.niebo) {
  const sky = S.niebo(STYL);
  sky.material.uniforms.sunPosition.value.copy(sunDir.clone().multiplyScalar(1000));
  scene.add(sky);
  const skyScene = new THREE.Scene(); const sky2 = S.niebo(STYL);
  sky2.material.uniforms.sunPosition.value.copy(sunDir.clone().multiplyScalar(1000)); skyScene.add(sky2);
  env = pmrem.fromScene(skyScene, 0.02).texture;
} else {
  scene.background = new THREE.Color(STYL.tlo);
  const e = new THREE.Scene(); e.background = new THREE.Color(0x1a2440);
  env = pmrem.fromScene(e).texture;
  scene.add(S.gwiazdy());
  const moon = new THREE.Mesh(new THREE.SphereGeometry(900, 24, 16), new THREE.MeshBasicMaterial({ color: 0xeef2ff, fog: false }));
  moon.position.copy(sunDir.clone().multiplyScalar(60000)); scene.add(moon);
}
scene.environment = env;
if (STYL.chmury.ile) scene.add(S.chmury(STYL));

// The ground, the mountains round it, what grows and lies on it, the tarn, the hut.
const gruntMat = S.materialGruntu(tr, tex, STYL.paleta, { flat: STYL.plaski });
const grunt = new THREE.Mesh(S.siatkaGruntu(tr), gruntMat);
grunt.receiveShadow = true; grunt.castShadow = true;
scene.add(grunt);
const gory = new THREE.Mesh(S.siatkaGor(tr), S.materialGruntu(tr, tex, STYL.paleta, { daleko: true, flat: STYL.plaski }));
gory.receiveShadow = true;
if (!BEZ.has('gory')) scene.add(gory);
const kosodrzewina = S.roslinnosc(tr, STYL.roslinnosc, !!STYL.plaski);
if (!BEZ.has('roslinnosc')) scene.add(kosodrzewina);
if (!STYL.plaski) scene.add(S.kamienie(tr, STYL.kamienie));
const woda = S.staw(tr, env, STYL.woda);
scene.add(woda);
const schronisko = S.baza(tr, !!STYL.noc);
scene.add(schronisko);

// ---------------------------------------------------------------- the thermal world
// Every surface, as warm as it is: drawn as a temperature into a float picture.
const TERMO_VS = /* glsl */`
#include <common>
#include <logdepthbuf_pars_vertex>
varying vec3 vWPos;
void main() {
  vec4 p = vec4(position, 1.0);
  #ifdef USE_INSTANCING
    p = instanceMatrix * p;
  #endif
  vec4 wp = modelMatrix * p;
  vWPos = wp.xyz;
  gl_Position = projectionMatrix * viewMatrix * wp;
  #include <logdepthbuf_vertex>
}`;
const termoStala = (t) => new THREE.ShaderMaterial({
  uniforms: { uT: { value: t } }, vertexShader: TERMO_VS,
  fragmentShader: `#include <common>\n#include <logdepthbuf_pars_fragment>\nuniform float uT; void main(){ gl_FragColor = vec4(uT, 0.0, 0.0, 1.0);\n#include <logdepthbuf_fragment>\n}`,
});
const termoGrunt = (dodaj) => new THREE.ShaderMaterial({
  uniforms: { uTemp: { value: tex.temp }, uSzum: { value: tex.szum }, uMin: { value: tr.tmin }, uMax: { value: tr.tmax }, uPolowa: { value: tr.polowa }, uKrok: { value: tr.krok }, uN: { value: tr.n }, uDodaj: { value: dodaj }, uNoc: { value: STYL.noc ? 1 : 0 } },
  vertexShader: TERMO_VS,
  fragmentShader: /* glsl */`
#include <common>
#include <logdepthbuf_pars_fragment>
uniform sampler2D uTemp, uSzum; uniform float uMin, uMax, uPolowa, uKrok, uN, uDodaj, uNoc; varying vec3 vWPos;
void main() {
  vec2 xy = vec2(vWPos.x, -vWPos.z);
  vec2 uv = ((xy + uPolowa) / uKrok + 0.5) / uN;
  float t = mix(uMin, uMax, texture2D(uTemp, clamp(uv, 0.0, 1.0)).r);
  t += 0.6 * (texture2D(uSzum, xy / 6.0).r - 0.5) + 0.3 * (texture2D(uSzum, xy / 1.3).a - 0.5);
  if (uNoc > 0.5) t = mix(t, -9.0, 0.55);
  gl_FragColor = vec4(t + uDodaj, 0.0, 0.0, 1.0);
  #include <logdepthbuf_fragment>
}`,
});
{
  const g = new THREE.Mesh(grunt.geometry, termoGrunt(0)); g.layers.set(S.TERMO); scene.add(g);
  const k = new THREE.InstancedMesh(kosodrzewina.geometry, termoGrunt(0.25), kosodrzewina.count);
  for (let i = 0; i < kosodrzewina.count; i++) { const m = new THREE.Matrix4(); kosodrzewina.getMatrixAt(i, m); k.setMatrixAt(i, m); }
  k.layers.set(S.TERMO); scene.add(k);
  const water = new THREE.Mesh(woda.geometry, termoStala(1.5)); water.position.copy(woda.position); water.layers.set(S.TERMO); scene.add(water);
  const skyT = new THREE.Mesh(new THREE.SphereGeometry(30000, 16, 12), termoStala(-30)); skyT.material.side = THREE.BackSide; skyT.layers.set(S.TERMO); scene.add(skyT);
}
const { grp: rzeczy, twins } = S.obiekty(tr, dane.obiekty, termoStala);
scene.add(rzeczy, twins);

const TW = 256, TH = 192;
const termoRT = new THREE.WebGLRenderTarget(TW * 2, TH * 2, { type: THREE.FloatType, format: THREE.RGBAFormat, depthBuffer: true });
const vfov = 2 * Math.atan(Math.tan(THREE.MathUtils.degToRad(dane.kamera.fov / 2)) * TH / TW);
const termoCam = new THREE.PerspectiveCamera(THREE.MathUtils.radToDeg(vfov), TW / TH, 0.5, 40000);
termoCam.layers.set(S.TERMO);
const termoBuf = new Float32Array(TW * 2 * TH * 2 * 4);
const termoImg = new Float32Array(TW * TH);
const IRONBOW = [[0, 0, 0], [40, 0, 90], [120, 10, 140], [200, 40, 90], [240, 110, 20], [250, 200, 40], [255, 255, 220]];
let zakres = null;

// ---------------------------------------------------------------- the drones
const modele = {};
for (const k of Object.keys(loty)) { modele[k] = D.zbuduj(k); modele[k].traverse((o) => { o.castShadow = true; }); scene.add(modele[k]); }
let reflektor = null;
if (STYL.hud.reflektor) {
  reflektor = new THREE.SpotLight(0xfff2d8, 3200, 140, 0.26, 0.6, 1.6);
  reflektor.castShadow = false;
  scene.add(reflektor, reflektor.target);
}
// Beams over the candidates, as a game marks where to go.
const beams = new Map();
function wiazka(kolor) {
  const g = new THREE.CylinderGeometry(0.3, 0.3, 90, 16, 1, true); g.translate(0, 45, 0);
  const m = new THREE.ShaderMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
    uniforms: { uC: { value: new THREE.Color(kolor) }, uA: { value: 0.0 } },
    vertexShader: `varying float vY; void main(){ vY = position.y / 90.0; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
    fragmentShader: `uniform vec3 uC; uniform float uA; varying float vY; void main(){ gl_FragColor = vec4(uC * (1.0 - vY) * 1.6, (1.0 - vY) * (1.0 - vY) * uA); }` });
  return new THREE.Mesh(g, m);
}

// The camera's reach: rays to the picture's corners, its outline on the slope, the laser.
const linieMat = new LineMaterial({ color: STYL.hud.kolorLinii, linewidth: 1.4, transparent: true, opacity: 0.55, dashed: false });
const obrysMat = new LineMaterial({ color: STYL.hud.kolorLinii, linewidth: 2.4, transparent: true, opacity: 0.85 });
const laserMat = new LineMaterial({ color: 0xff3b30, linewidth: 2.6, transparent: true, opacity: 0.95 });
for (const m of [linieMat, obrysMat, laserMat]) m.resolution.set(width, height);
const linie = new LineSegments2(new LineSegmentsGeometry(), linieMat);
const obrys = new LineSegments2(new LineSegmentsGeometry(), obrysMat);
const laser = new LineSegments2(new LineSegmentsGeometry(), laserMat);
const kropka = new THREE.Mesh(new THREE.SphereGeometry(0.5, 12, 8), new THREE.MeshBasicMaterial({ color: 0xff3b30, fog: false }));
[linie, obrys, laser].forEach((l) => { l.frustumCulled = false; scene.add(l); });
scene.add(kropka);
for (const o of [linie, obrys, laser, kropka]) o.visible = !!STYL.hud.linie;

// ---------------------------------------------------------------- post-processing
const rt = new THREE.WebGLRenderTarget(width, height, { type: THREE.HalfFloatType, samples: 2 });
const composer = new EffectComposer(renderer, rt);
composer.setPixelRatio(renderer.getPixelRatio());
composer.setSize(width, height);
composer.addPass(new RenderPass(scene, camera));
if (STYL.bloom.moc > 0 && !BEZ.has('bloom')) composer.addPass(new UnrealBloomPass(new THREE.Vector2(width, height), STYL.bloom.moc, STYL.bloom.promien, STYL.bloom.prog));
composer.addPass(new OutputPass());
const grade = new ShaderPass({
  uniforms: { tDiffuse: { value: null }, uKontrast: { value: STYL.grade.kontrast }, uNas: { value: STYL.grade.nasycenie }, uTint: { value: new THREE.Vector3(...STYL.grade.tint) }, uLift: { value: new THREE.Vector3(...STYL.grade.lift) }, uWin: { value: STYL.grade.winieta }, uZiarno: { value: STYL.grade.ziarno }, uAb: { value: STYL.grade.aberracja }, uCzas: { value: 0 }, uRes: { value: new THREE.Vector2(width, height) } },
  vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
  fragmentShader: /* glsl */`
uniform sampler2D tDiffuse; uniform float uKontrast, uNas, uWin, uZiarno, uAb, uCzas; uniform vec3 uTint, uLift; uniform vec2 uRes; varying vec2 vUv;
void main() {
  vec2 d = (vUv - 0.5) * uAb * length(vUv - 0.5) * 4.0;
  vec3 c = vec3(texture2D(tDiffuse, vUv + d).r, texture2D(tDiffuse, vUv).g, texture2D(tDiffuse, vUv - d).b);
  c *= uTint; c += uLift * (1.0 - c);
  float l = dot(c, vec3(0.2126, 0.7152, 0.0722)); c = mix(vec3(l), c, uNas);
  c = (c - 0.5) * uKontrast + 0.5;
  float v = smoothstep(0.95, 0.3, length((vUv - 0.5) * vec2(uRes.x / uRes.y, 1.0)) * 0.9);
  c *= mix(1.0, v, uWin);
  float n = fract(sin(dot(vUv * uRes + uCzas * 61.7, vec2(12.9898, 78.233))) * 43758.5453);
  c += (n - 0.5) * uZiarno;
  gl_FragColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}`,
});
if (!BEZ.has('grade')) composer.addPass(grade);

// ---------------------------------------------------------------- the head-up display
const $ = (id) => document.getElementById(id);
for (const [id, on] of [['termo', STYL.hud.termo], ['mapa', STYL.hud.mapa]]) $(id).style.display = on ? '' : 'none';
$('h-kam').textContent = dane.kamera.nazwa.replace('Sipeed ', '');
const mapaCtx = $('h-mapa').getContext('2d');
const mapaTlo = (() => {
  // The map's ground: the cover's colours, shaded by the slope, north up.
  const c = document.createElement('canvas'), N = 480; c.width = c.height = N;
  const x = c.getContext('2d'), img = x.createImageData(N, N), R = 560;
  const kol = [[236, 240, 246], [112, 108, 102], [150, 145, 136], [42, 70, 38], [150, 140, 82], [30, 70, 96], [150, 116, 80]];
  for (let py = 0; py < N; py++) for (let px = 0; px < N; px++) {
    const wx = -R + 2 * R * px / N, wy = R - 2 * R * py / N;
    const c0 = Math.max(0, tr.cover(wx, wy)), h = tr.h(wx, wy);
    const dx = tr.h(wx + 4, wy) - tr.h(wx - 4, wy), dy = tr.h(wx, wy + 4) - tr.h(wx, wy - 4);
    const shade = Math.max(0.35, Math.min(1.25, 0.85 + (-dx * 0.7 + dy * 0.7) / 16));
    const k = (py * N + px) * 4, base = kol[c0] || kol[4];
    const tint = 0.85 + 0.15 * ((h - tr.zmin) / (tr.zmax - tr.zmin));
    img.data[k] = base[0] * shade * tint; img.data[k + 1] = base[1] * shade * tint; img.data[k + 2] = base[2] * shade * tint; img.data[k + 3] = 255;
  }
  x.putImageData(img, 0, 0);
  return c;
})();
function mapaXY(x, y) { const R = 560, N = 480; return [(x + R) / (2 * R) * N, (R - y) / (2 * R) * N]; }

function rysujMape(lot, s, kandydaci, rogi) {
  const x = mapaCtx, acc = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#ffc23d';
  x.drawImage(mapaTlo, 0, 0);
  x.strokeStyle = 'rgba(80,200,255,0.9)'; x.lineWidth = 2;
  const [a0, b0] = mapaXY(-tr.pole, tr.pole), [a1, b1] = mapaXY(tr.pole, -tr.pole);
  x.strokeRect(a0, b0, a1 - a0, b1 - b0);
  x.strokeStyle = 'rgba(255,255,255,0.28)'; x.lineWidth = 1;
  for (const lane of lot.d.pasy) { x.beginPath(); lane.forEach(([px, py], i) => { const [u, v] = mapaXY(px, py); i ? x.lineTo(u, v) : x.moveTo(u, v); }); x.stroke(); }
  x.strokeStyle = acc; x.lineWidth = 2.5; x.beginPath();
  for (let tt = 0, first = true; tt <= s.t; tt += 2) { const p = lot.w_chwili(tt); const [u, v] = mapaXY(p.x, p.y); first ? x.moveTo(u, v) : x.lineTo(u, v); first = false; }
  x.stroke();
  if (rogi) { x.fillStyle = 'rgba(255,220,90,0.25)'; x.beginPath(); rogi.forEach((r, i) => { const [u, v] = mapaXY(r[0], r[1]); i ? x.lineTo(u, v) : x.moveTo(u, v); }); x.closePath(); x.fill(); }
  for (const c of kandydaci) {
    const [u, v] = mapaXY(c.x, c.y);
    x.fillStyle = c.stan === 'potwierdzony' ? '#4dffa0' : c.stan === 'odrzucony' ? 'rgba(200,200,200,0.6)' : c.stan === 'do_sprawdzenia' ? '#7ec8ff' : acc;
    x.beginPath(); x.arc(u, v, 6, 0, 6.3); x.fill(); x.strokeStyle = '#000'; x.lineWidth = 1; x.stroke();
  }
  const [u, v] = mapaXY(s.x, s.y);
  x.save(); x.translate(u, v); x.rotate(-s.kurs); x.fillStyle = '#fff'; x.strokeStyle = '#000'; x.lineWidth = 2;
  x.beginPath(); x.moveTo(14, 0); x.lineTo(-9, 8); x.lineTo(-4, 0); x.lineTo(-9, -8); x.closePath(); x.stroke(); x.fill(); x.restore();
  const [bx, by] = mapaXY(...tr.baza); x.fillStyle = '#ffc23d'; x.font = '700 20px Rajdhani'; x.fillText('H', bx - 6, by + 7);
}

const kompasCtx = $('h-kompas').getContext('2d');
function rysujKompas(kursDeg, znaczniki) {
  const x = kompasCtx, Wc = 1120, Hc = 92, hud = getComputedStyle(document.documentElement).getPropertyValue('--hud').trim() || '#fff';
  x.clearRect(0, 0, Wc, Hc);
  const pxPerDeg = 6.2, names = { 0: 'N', 45: 'NE', 90: 'E', 135: 'SE', 180: 'S', 225: 'SW', 270: 'W', 315: 'NW' };
  x.fillStyle = hud; x.strokeStyle = hud; x.textAlign = 'center';
  for (let d = Math.floor(kursDeg - 95); d <= kursDeg + 95; d++) {
    if (d % 5) continue;
    const dd = ((d % 360) + 360) % 360, u = Wc / 2 + (d - kursDeg) * pxPerDeg;
    const big = dd % 15 === 0;
    x.globalAlpha = 1 - Math.min(1, Math.abs(d - kursDeg) / 95) * 0.8;
    x.lineWidth = big ? 3 : 2; x.beginPath(); x.moveTo(u, Hc - 6); x.lineTo(u, Hc - (big ? 30 : 18)); x.stroke();
    if (names[dd] !== undefined) { x.font = '700 30px Rajdhani'; x.fillText(names[dd], u, 34); }
    else if (dd % 30 === 0) { x.font = '600 22px Rajdhani'; x.fillText(String(dd), u, 32); }
  }
  x.globalAlpha = 1;
  for (const z of znaczniki) {
    let off = ((z.deg - kursDeg + 540) % 360) - 180; if (Math.abs(off) > 90) continue;
    const u = Wc / 2 + off * pxPerDeg; x.fillStyle = z.kolor; x.beginPath(); x.moveTo(u, Hc - 34); x.lineTo(u + 9, Hc - 46); x.lineTo(u - 9, Hc - 46); x.closePath(); x.fill();
  }
  x.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#ffc23d';
  x.beginPath(); x.moveTo(Wc / 2, Hc - 2); x.lineTo(Wc / 2 + 10, Hc - 16); x.lineTo(Wc / 2 - 10, Hc - 16); x.closePath(); x.fill();
}

const termoCtx = $('h-termo').getContext('2d'), termoImgData = termoCtx.createImageData(TW, TH);
const nakCtx = $('h-termo-nak').getContext('2d');
function ironbow(v) {
  const x = Math.min(1, Math.max(0, v)) ** 0.85 * (IRONBOW.length - 1), i = Math.min(IRONBOW.length - 2, Math.floor(x)), f = x - i;
  return [0, 1, 2].map((c) => IRONBOW[i][c] * (1 - f) + IRONBOW[i + 1][c] * f);
}
let szumTermo = S.los(77);
function gauss() { let u = 0, v = 0; while (u === 0) u = szumTermo(); v = szumTermo(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(6.2832 * v); }

function obrazTermo(s, kandydaci, ocenaFn) {
  // The thermal camera, as the drone's: straight down, heading up in the picture.
  termoCam.position.copy(S.w2t(s.x, s.y, s.z - 0.1));
  termoCam.up.set(Math.cos(s.kurs), 0, -Math.sin(s.kurs));
  termoCam.lookAt(S.w2t(s.x, s.y, s.z - 100));
  termoCam.updateMatrixWorld();
  const tm = renderer.toneMapping; renderer.toneMapping = THREE.NoToneMapping;
  renderer.setRenderTarget(termoRT); renderer.setClearColor(0x000000, 1); renderer.clear(); renderer.render(scene, termoCam);
  renderer.readRenderTargetPixels(termoRT, 0, 0, TW * 2, TH * 2, termoBuf);
  renderer.setRenderTarget(null); renderer.toneMapping = tm;
  // Each pixel the mean of its patch (the sensor mixes what it sees), and the noise.
  const vals = [];
  for (let y = 0; y < TH; y++) for (let x = 0; x < TW; x++) {
    let s4 = 0;
    for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) s4 += termoBuf[(((TH - 1 - y) * 2 + (1 - dy)) * TW * 2 + x * 2 + dx) * 4];
    const v = s4 / 4 + 0.05 * gauss(); termoImg[y * TW + x] = v; if ((x + y) % 7 === 0) vals.push(v);
  }
  vals.sort((a, b) => a - b);
  const lo = vals[Math.floor(vals.length * 0.01)], hi = Math.max(vals[Math.floor(vals.length * 0.997)], lo + 4);
  zakres = zakres ? [zakres[0] + 0.25 * (lo - zakres[0]), zakres[1] + 0.25 * (hi - zakres[1])] : [lo, hi];
  const [a, b] = zakres;
  for (let k = 0; k < TW * TH; k++) { const c = ironbow((termoImg[k] - a) / (b - a)); termoImgData.data[4 * k] = c[0]; termoImgData.data[4 * k + 1] = c[1]; termoImgData.data[4 * k + 2] = c[2]; termoImgData.data[4 * k + 3] = 255; }
  termoCtx.putImageData(termoImgData, 0, 0);
  $('h-zakres').textContent = `${a >= 0 ? '+' : '−'}${Math.abs(a).toFixed(0)}…${b >= 0 ? '+' : '−'}${Math.abs(b).toFixed(0)} °C`;
  const mid = termoImg[(TH >> 1) * TW + (TW >> 1)];
  $('h-temp').textContent = `${mid >= 0 ? '+' : '−'}${Math.abs(mid).toFixed(1)}°`;
  // The reticle, and a box round each candidate in the picture.
  const X = nakCtx, Wn = 768, Hn = 576, acc = '#7dffb0';
  X.clearRect(0, 0, Wn, Hn); X.strokeStyle = acc; X.lineWidth = 3;
  [[-46, 0, -14, 0], [14, 0, 46, 0], [0, -46, 0, -14], [0, 14, 0, 46]].forEach(([a1, b1, a2, b2]) => { X.beginPath(); X.moveTo(Wn / 2 + a1, Hn / 2 + b1); X.lineTo(Wn / 2 + a2, Hn / 2 + b2); X.stroke(); });
  X.font = '700 26px Rajdhani';
  for (const c of kandydaci) {
    const p = S.w2t(c.x, c.y, c.z).project(termoCam);
    if (Math.abs(p.x) > 1 || Math.abs(p.y) > 1 || p.z > 1) continue;
    const u = (p.x * 0.5 + 0.5) * Wn, v = (-p.y * 0.5 + 0.5) * Hn, hs = 30;
    const col = c.stan === 'potwierdzony' ? '#4dffa0' : c.stan === 'odrzucony' ? '#cccccc' : '#ffd23d';
    X.strokeStyle = col; X.fillStyle = col; X.lineWidth = 3; X.strokeRect(u - hs, v - hs, 2 * hs, 2 * hs);
    X.fillText(c.stan === 'potwierdzony' ? 'CZŁOWIEK' : c.stan === 'odrzucony' ? 'NIE' : 'CZŁOWIEK?', u + hs + 8, v - hs + 20);
  }
}

const znacznikiEl = $('znaczniki');
function rysujZnaczniki(s, kandydaci) {
  let html = '';
  if (STYL.hud.znaczniki) for (const c of kandydaci) {
    const p = S.w2t(c.x, c.y, c.z + 4).project(camera);
    if (p.z > 1 || Math.abs(p.x) > 1.05 || Math.abs(p.y) > 1.05) continue;
    const u = (p.x * 0.5 + 0.5) * 100, v = (-p.y * 0.5 + 0.5) * 100;
    const d = Math.hypot(c.x - s.x, c.y - s.y, c.z - s.z);
    const opis = { czeka: 'KANDYDAT', potwierdzony: 'CZŁOWIEK', odrzucony: 'ODRZUCONY', do_sprawdzenia: 'DO SPRAWDZENIA' }[c.stan];
    html += `<div class="znacznik ${c.stan}" style="left:${u}%;top:${v}%"><div class="romb"></div><div class="opis">${opis}</div><div class="odl">${d.toFixed(0)} m</div></div>`;
  }
  znacznikiEl.innerHTML = html;
}

function rysujToasty(lot, t) {
  const box = $('toasty'); let html = '';
  const recent = lot.zd.filter((e) => e.t <= t && t - e.t < 9).slice(-3).reverse();
  for (const e of recent) {
    const age = t - e.t, op = Math.min(1, age * 3) * Math.min(1, (9 - age) / 1.5);
    const [title, sub, cls] = {
      czeka: ['KANDYDAT WYKRYTY', `ocena mózgu muchy ${e.ocena >= 0 ? '+' : '−'}${Math.abs(e.ocena).toFixed(3)} · leci obejrzeć z bliska`, ''],
      potwierdzony: ['POTWIERDZONO: CZŁOWIEK', `zgłoszenie do ekipy · ${e.x.toFixed(0)}, ${e.y.toFixed(0)} m`, 'ok'],
      odrzucony: ['ODRZUCONY Z BLISKA', 'to nie człowiek', ''],
      do_sprawdzenia: ['DO SPRAWDZENIA', 'nie udało się obejrzeć: do operatora', ''],
    }[e.stan];
    html += `<div class="toast ${cls}" style="opacity:${op};transform:translateX(${(1 - Math.min(1, age * 4)) * 20}px)">${title}<small>${sub}</small></div>`;
  }
  box.innerHTML = html;
}

function hudTekst(lot, s, kandydaci) {
  $('h-platforma').textContent = NAZWY[lot.nazwa];
  let tryb = FAZY[s.faza];
  if (lot.nazwa === 'vtol') tryb += s.zawis ? ' · ZAWIS' : ' · PRZELOT';
  if (s.krazy) tryb += ' · KRĄŻY';
  $('h-tryb').textContent = tryb;
  const mm = Math.floor(s.t / 60), ss = Math.floor(s.t % 60);
  $('h-czas').textContent = `${String(mm).padStart(2, '0')}:${String(ss).padStart(2, '0')}`;
  $('h-zgl').textContent = kandydaci.filter((c) => c.stan === 'potwierdzony').length;
  $('h-kand').textContent = kandydaci.length;
  $('h-bat').style.width = `${Math.max(0, Math.min(1, s.bateria)) * 100}%`;
  $('h-agl').textContent = s.agl.toFixed(0);
  $('h-v').textContent = s.v.toFixed(1);
  $('h-amsl').textContent = s.z.toFixed(0);
}

// ---------------------------------------------------------------- one frame
const tmpV = new THREE.Vector3();
function stanDrona(lot, t) {
  const s = lot.w_chwili(t), a = lot.w_chwili(t - 0.5), b = lot.w_chwili(t + 0.5);
  let dk = ((b.kurs - a.kurs + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
  const omega = dk / 1.0, vz = (b.z - a.z) / 1.0;
  let przechyl = 0, pochyl = 0;
  if (lot.nazwa === 'quad' || (lot.nazwa === 'vtol' && s.zawis)) { pochyl = -Math.min(0.32, s.v / 8 * 0.28); przechyl = -Math.max(-0.25, Math.min(0.25, omega * 0.25)); }
  else { przechyl = -Math.max(-0.65, Math.min(0.65, Math.atan(s.v * omega / 9.81))); pochyl = Math.atan2(vz, Math.max(s.v, 1)) * 0.8; }
  return { ...s, pos: S.w2t(s.x, s.y, s.z), przechyl, pochyl };
}

function zasieg(s) {
  // The rays to the picture's corners and the outline on the ground, and the laser.
  const tw = Math.tan(THREE.MathUtils.degToRad(dane.kamera.fov / 2)), th = tw * TH / TW;
  const f = [Math.cos(s.kurs), Math.sin(s.kurs)], l = [-Math.sin(s.kurs), Math.cos(s.kurs)];
  const dir = (a, b) => { const d = [a * l[0] + b * f[0], a * l[1] + b * f[1], -1]; const n = Math.hypot(...d); return d.map((v) => v / n); };
  const o = [s.x, s.y, s.z - 0.12];
  const edge = [[tw, th], [-tw, th], [-tw, -th], [tw, -th]];
  const pts = [], corners = [];
  for (let e = 0; e < 4; e++) {
    const [a0, b0] = edge[e], [a1, b1] = edge[(e + 1) % 4];
    for (let k = 0; k < 14; k++) { const q = k / 14; const h = tr.traf(o, dir(a0 + (a1 - a0) * q, b0 + (b1 - b0) * q)); if (h) pts.push(h); if (k === 0) corners.push(h); }
  }
  const seg = [], segO = [];
  const lens = S.w2t(...o);
  for (const c of corners) if (c) { const p = S.w2t(c[0], c[1], c[2] + 0.3); seg.push(lens.x, lens.y, lens.z, p.x, p.y, p.z); }
  for (let k = 0; k < pts.length; k++) { const a = pts[k], b = pts[(k + 1) % pts.length]; const pa = S.w2t(a[0], a[1], a[2] + 0.6), pb = S.w2t(b[0], b[1], b[2] + 0.6); segO.push(pa.x, pa.y, pa.z, pb.x, pb.y, pb.z); }
  if (seg.length) linie.geometry.setPositions(seg);
  if (segO.length) obrys.geometry.setPositions(segO);
  const hit = tr.traf(o, [0, 0, -1]);
  if (hit) { const p = S.w2t(hit[0], hit[1], hit[2] + 0.05); laser.geometry.setPositions([lens.x, lens.y, lens.z, p.x, p.y, p.z]); kropka.position.copy(p); }
  return corners.filter(Boolean);
}

function kamera(klip, s, lot, tc) {
  const k = klip ? klip.kamera : { typ: trybKamery };
  const head = lot.kursSredni(s.t);
  const fwd = new THREE.Vector3(Math.cos(head), 0, -Math.sin(head));
  if (k.typ === 'poscig') {
    const dist = k.dist ?? 8, up = k.wys ?? 1.8, side = k.bok ?? 1.2;
    const right = new THREE.Vector3(Math.sin(head), 0, Math.cos(head));
    camera.position.copy(s.pos).addScaledVector(fwd, -dist).addScaledVector(right, side).add(new THREE.Vector3(0, up, 0));
    const ground = tr.h(camera.position.x, -camera.position.z);
    camera.position.y = Math.max(camera.position.y, ground + 1.5);
    camera.lookAt(s.pos.clone().addScaledVector(fwd, k.przod ?? 28).add(new THREE.Vector3(0, k.celWys ?? -5, 0)));
  } else if (k.typ === 'orbita') {
    const c = k.srodek ? S.w2t(k.srodek[0], k.srodek[1], tr.h(k.srodek[0], k.srodek[1]) + (k.srodek[2] || 0)) : s.pos;
    const ang = (k.kat0 || 0) + (k.obrot || 0.15) * tc;
    const o = k.wokolDrona ? s.pos : c; // round the drone, or round a place
    camera.position.set(o.x + Math.cos(ang) * k.r, o.y + k.wys, o.z + Math.sin(ang) * k.r);
    camera.lookAt(k.patrz !== undefined ? s.pos.clone().lerp(c, k.patrz) : k.patrzNaDrona ? s.pos.clone().lerp(c, 0.5) : c);
  } else if (k.typ === 'kran') {
    const nad = ([x, y, dz]) => S.w2t(x, y, tr.h(x, y) + dz);
    const a = nad(k.od), b = nad(k.do), q = tc / (klip.dlugosc || 5);
    camera.position.copy(a).lerp(b, q * q * (3 - 2 * q));
    camera.lookAt(k.cel ? nad(k.cel) : s.pos);
  } else if (k.typ === 'swobodna') {
    controls.target.lerp(s.pos, 0.08); controls.update();
  }
  if (k.fov) { camera.fov = k.fov; camera.updateProjectionMatrix(); }
}

const czasy = {};
let klatkaNr = 0;
const sync = () => { const gl = renderer.getContext(); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(4)); };
const mierz = (k, f) => { const a = performance.now(); const r = f(); sync(); czasy[k] = (czasy[k] || 0) + performance.now() - a; return r; };
function klatka(tSim, tKlip) {
  const lot = loty[lotNazwa];
  for (const [k, m] of Object.entries(modele)) m.visible = KLIP && KLIP.wszystkie ? true : k === lotNazwa;
  for (const [k, l] of Object.entries(loty)) if (modele[k].visible) { D.ustaw(modele[k], stanDrona(l, tSim)); modele[k].scale.setScalar(STYL.skalaDronow || 1); }
  const s = stanDrona(lot, tSim);
  const kand = lot.kandydaci(tSim);
  kamera(KLIP, s, lot, tKlip);
  // The shadows follow what is looked at.
  const focus = s.pos.clone();
  sun.position.copy(focus).addScaledVector(sunDir, 1500); sun.target.position.copy(focus); sun.target.updateMatrixWorld();
  if (reflektor) { reflektor.position.copy(s.pos).add(new THREE.Vector3(0, -0.3, 0)); reflektor.target.position.copy(S.w2t(s.x + Math.cos(s.kurs) * 6, s.y + Math.sin(s.kurs) * 6, tr.h(s.x, s.y))); reflektor.target.updateMatrixWorld(); }
  // The tarn's ripples, the hut's beacon.
  woda.userData.normalna.offset.set(tKlip * 0.004, tKlip * 0.0025);
  schronisko.traverse((o) => { if (o.userData.migacz) o.material.emissiveIntensity = (tKlip % 1.5) < 0.2 ? 14 : 1; });
  // Beams over candidates.
  const seenIds = new Set();
  for (const c of kand) {
    if (c.stan === 'odrzucony') continue;
    seenIds.add(c.id);
    if (!beams.has(c.id)) { const b = wiazka(0xffffff); b.position.copy(S.w2t(c.x, c.y, tr.h(c.x, c.y))); scene.add(b); beams.set(c.id, b); }
    const b = beams.get(c.id);
    const far = Math.min(1, Math.max(0, (camera.position.distanceTo(b.position) - 50) / 100)); // for the distance
    b.material.uniforms.uC.value.set(c.stan === 'potwierdzony' ? 0x4dffa0 : c.stan === 'do_sprawdzenia' ? 0x7ec8ff : 0xffc23d);
    b.material.uniforms.uA.value = (STYL.hud.znaczniki ? 0.28 : 0) * far * Math.min(1, (tSim - c.od) / 2 + 0.2);
  }
  for (const [id, b] of beams) b.visible = seenIds.has(id);
  const rogi = STYL.hud.linie || STYL.hud.mapa ? mierz('zasieg', () => zasieg(s)) : null;
  // Live, the thermal picture is read back every third frame: it waits on the graphics card.
  if (STYL.hud.termo && !BEZ.has('termo') && (NAGRANIE || (klatkaNr++ % 3 === 0))) mierz('termo', () => obrazTermo(s, kand));
  grade.uniforms.uCzas.value = tKlip;
  mierz('render', () => composer.render());
  hudTekst(lot, s, kand);
  if (STYL.hud.mapa) mierz('mapa', () => rysujMape(lot, s, kand, rogi));
  const kursDeg = ((90 - THREE.MathUtils.radToDeg(s.kurs)) % 360 + 360) % 360;
  rysujKompas(kursDeg, kand.filter((c) => c.stan !== 'odrzucony').map((c) => ({ deg: ((90 - THREE.MathUtils.radToDeg(Math.atan2(c.y - s.y, c.x - s.x))) % 360 + 360) % 360, kolor: c.stan === 'potwierdzony' ? '#4dffa0' : '#ffc23d' })));
  rysujZnaczniki(s, kand);
  rysujToasty(lot, tSim);
  if (KLIP && KLIP.tytul) {
    const a = Math.min(1, tKlip / 0.6) * Math.min(1, ((KLIP.dlugosc || 5) - tKlip) / 0.6);
    $('tytul').innerHTML = `<div class="duzy" style="opacity:${a}">${KLIP.tytul}</div><div class="maly" style="opacity:${a}">${KLIP.podtytul || ''}</div>`;
  }
}

// ---------------------------------------------------------------- playing, or recording
let trybKamery = 'poscig', tSim = Number(params.get('t') || 118), tempo = 3, gra = true;
const controls = new OrbitControls(camera, canvas); controls.enabled = false; controls.enableDamping = true;
$('ladowanie').remove();
window.SIM = {
  klip: KLIP,
  czasy,
  // Render the clip's frame at tKlip seconds; resolves once drawn.
  async klatka(tKlip) {
    const t = KLIP.t0 + tKlip * (KLIP.tempo || 1);
    klatka(t, tKlip);
    // Wait until the frame is really drawn: a pixel read back makes the GL finish.
    const gl = renderer.getContext(); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(4));
    await new Promise((r) => requestAnimationFrame(() => r()));
    return true;
  },
};

if (!NAGRANIE) {
  const lotSel = $('s-lot'), stylSel = $('s-styl'), suw = $('s-czas');
  lotSel.value = lotNazwa; stylSel.value = STYL_NAZWA;
  lotSel.onchange = () => { lotNazwa = lotSel.value; tSim = 30; };
  // A style builds its own world: the page is drawn again, the style kept in its #hash.
  stylSel.onchange = () => { location.hash = stylSel.value; location.reload(); };
  window.addEventListener('hashchange', () => { if (location.hash.replace('#', '') in STYLE) location.reload(); });
  const kamery = ['poscig', 'swobodna'];
  $('s-kam').onclick = () => { trybKamery = kamery[(kamery.indexOf(trybKamery) + 1) % kamery.length]; controls.enabled = trybKamery === 'swobodna'; $('s-kam').textContent = trybKamery === 'poscig' ? 'Pościg' : 'Swobodna'; };
  $('s-play').onclick = () => { gra = !gra; $('s-play').textContent = gra ? '❚❚' : '▶'; $('s-play').classList.toggle('on', gra); };
  const tempa = [1, 3, 10, 30];
  $('s-tempo').textContent = '×' + tempo;
  $('s-tempo').onclick = () => { tempo = tempa[(tempa.indexOf(tempo) + 1) % tempa.length]; $('s-tempo').textContent = '×' + tempo; };
  suw.oninput = () => { tSim = loty[lotNazwa].koniec * suw.value / 1000; };
  window.addEventListener('resize', () => {
    const w = window.innerWidth, h = window.innerHeight;
    renderer.setSize(w, h, false); composer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix();
    for (const m of [linieMat, obrysMat, laserMat]) m.resolution.set(w, h);
  });
  let last = performance.now();
  const petla = (now) => {
    const dt = Math.min(0.1, (now - last) / 1000); last = now;
    if (gra) tSim = (tSim + dt * tempo) % loty[lotNazwa].koniec;
    suw.value = String(Math.round(1000 * tSim / loty[lotNazwa].koniec));
    klatka(tSim, now / 1000);
    requestAnimationFrame(petla);
  };
  requestAnimationFrame(petla);
}
