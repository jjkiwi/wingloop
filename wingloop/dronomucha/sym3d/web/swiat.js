// The world: the valley's ground and what grows on it, the mountains round it, the tarn,
// the sky, the people, animals and boulders -- and their twins for the thermal camera.
// Coordinates: the search's x east, y north, z up become three's X = x, Y = z, Z = -y.
import * as THREE from 'three';
import { Sky } from 'three/addons/objects/Sky.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

export const SNIEG = 0, SKALA = 1, PIARG = 2, KOSO = 3, TRAWA = 4, WODA = 5, SZLAK = 6;
export const TERMO = 1; // the thermal camera's layer

export function w2t(x, y, z) { return new THREE.Vector3(x, z, -y); }

function b64(s, Type) {
  const bin = atob(s);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Type(bytes.buffer);
}

// A seeded random, so every film of the same style is the same film.
export function los(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------- the ground's data
export class Teren {
  constructor(d) {
    this.n = d.n; this.krok = d.krok; this.polowa = d.polowa; this.pole = d.pole;
    this.z = b64(d.wysokosci, Float32Array);
    this.pokrycie = b64(d.pokrycie, Uint8Array);
    this.temp = b64(d.temperatura, Float32Array);
    this.baza = d.baza; this.staw = d.staw; this.szlak = d.szlak;
    let lo = Infinity, hi = -Infinity, tlo = Infinity, thi = -Infinity;
    for (let i = 0; i < this.z.length; i++) {
      lo = Math.min(lo, this.z[i]); hi = Math.max(hi, this.z[i]);
      tlo = Math.min(tlo, this.temp[i]); thi = Math.max(thi, this.temp[i]);
    }
    this.zmin = lo; this.zmax = hi; this.tmin = tlo; this.tmax = thi;
    // The tarn's water: the middle of its cells' heights.
    const w = [];
    for (let i = 0; i < this.pokrycie.length; i++) if (this.pokrycie[i] === WODA) w.push(this.z[i]);
    w.sort((a, b) => a - b);
    this.lustro = w.length ? w[Math.floor(w.length * 0.5)] + 0.15 : lo;
  }
  // Height at (x, y), bilinear on the grid -- the same surface the mesh is.
  h(x, y) {
    const n = this.n;
    let j = (x + this.polowa) / this.krok, i = (y + this.polowa) / this.krok;
    j = Math.min(Math.max(j, 0), n - 1.001); i = Math.min(Math.max(i, 0), n - 1.001);
    const j0 = Math.floor(j), i0 = Math.floor(i), fx = j - j0, fy = i - i0;
    const a = this.z[i0 * n + j0], b = this.z[i0 * n + j0 + 1];
    const c = this.z[(i0 + 1) * n + j0], d = this.z[(i0 + 1) * n + j0 + 1];
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
  }
  cover(x, y) {
    const n = this.n;
    const j = Math.round((x + this.polowa) / this.krok), i = Math.round((y + this.polowa) / this.krok);
    if (i < 0 || j < 0 || i >= n || j >= n) return -1;
    return this.pokrycie[i * n + j];
  }
  // Where a ray from o (world x, y, z) along d first meets the ground; null past `far`.
  traf(o, d, far = 3000) {
    let t = 0, step = 2.0, prev = o[2] - this.h(o[0], o[1]);
    for (let k = 0; k < 4000 && t < far; k++) {
      const tn = t + step;
      const x = o[0] + d[0] * tn, y = o[1] + d[1] * tn, z = o[2] + d[2] * tn;
      const above = z - this.h(x, y);
      if (above <= 0) {
        let a = t, b = tn;
        for (let s = 0; s < 18; s++) {
          const m = 0.5 * (a + b);
          const ab = o[2] + d[2] * m - this.h(o[0] + d[0] * m, o[1] + d[1] * m);
          if (ab > 0) a = m; else b = m;
        }
        return [o[0] + d[0] * b, o[1] + d[1] * b, o[2] + d[2] * b, b];
      }
      prev = above; t = tn; step = Math.min(20, Math.max(1.0, above * 0.5));
    }
    return null;
  }
}

// ---------------------------------------------------------------- textures made here
// Tileable value noise, four kinds in four channels, for the ground's detail.
export function teksturaSzumu(size = 256, seed = 7) {
  const rnd = los(seed);
  const lattice = (f) => { const a = new Float32Array(f * f); for (let i = 0; i < a.length; i++) a[i] = rnd(); return a; };
  const sample = (L, f, u, v) => {
    const x = u * f, y = v * f, x0 = Math.floor(x), y0 = Math.floor(y);
    const fx = x - x0, fy = y - y0, sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy);
    const g = (i, j) => L[((j % f + f) % f) * f + ((i % f + f) % f)];
    return (g(x0, y0) * (1 - sx) + g(x0 + 1, y0) * sx) * (1 - sy) + (g(x0, y0 + 1) * (1 - sx) + g(x0 + 1, y0 + 1) * sx) * sy;
  };
  const fbm = (freqs) => { const Ls = freqs.map(lattice); return (u, v) => { let s = 0, w = 0, a = 1; for (let k = 0; k < freqs.length; k++) { s += a * sample(Ls[k], freqs[k], u, v); w += a; a *= 0.55; } return s / w; }; };
  const A = fbm([4, 8, 16, 32, 64]), B = fbm([8, 16, 32, 64]), C = fbm([16, 32, 64, 128]), D = fbm([32, 64, 128]);
  const data = new Uint8Array(size * size * 4);
  for (let j = 0; j < size; j++) for (let i = 0; i < size; i++) {
    const u = i / size, v = j / size, k = (j * size + i) * 4;
    data[k] = 255 * A(u, v);
    data[k + 1] = 255 * Math.min(1, Math.abs(B(u, v) - 0.5) * 2.4);
    data[k + 2] = 255 * C(u, v);
    data[k + 3] = 255 * D(u, v);
  }
  const t = new THREE.DataTexture(data, size, size, THREE.RGBAFormat);
  t.wrapS = t.wrapT = THREE.RepeatWrapping; t.magFilter = THREE.LinearFilter; t.minFilter = THREE.LinearMipmapLinearFilter;
  t.generateMipmaps = true; t.anisotropy = 8; t.needsUpdate = true;
  return t;
}

// The cover as soft weights: snow, rock, scree, pine | meadow, water, trail.
function teksturyPokrycia(tr) {
  const n = tr.n, A = new Uint8Array(n * n * 4), B = new Uint8Array(n * n * 4);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const acc = new Float32Array(8); let w = 0;
    for (let di = -1; di <= 1; di++) for (let dj = -1; dj <= 1; dj++) {
      const ii = Math.min(n - 1, Math.max(0, i + di)), jj = Math.min(n - 1, Math.max(0, j + dj));
      const c = tr.pokrycie[ii * n + jj], ww = di === 0 && dj === 0 ? 2 : 1;
      acc[c] += ww; w += ww;
    }
    const k = (i * n + j) * 4;
    A[k] = 255 * acc[0] / w; A[k + 1] = 255 * acc[1] / w; A[k + 2] = 255 * acc[2] / w; A[k + 3] = 255 * acc[3] / w;
    B[k] = 255 * acc[4] / w; B[k + 1] = 255 * acc[5] / w; B[k + 2] = 255 * acc[6] / w; B[k + 3] = 0;
  }
  const mk = (d) => { const t = new THREE.DataTexture(d, n, n, THREE.RGBAFormat); t.magFilter = t.minFilter = THREE.LinearFilter; t.needsUpdate = true; return t; };
  return [mk(A), mk(B)];
}

function teksturaTemperatury(tr) {
  const n = tr.n, d = new Uint8Array(n * n * 4), span = Math.max(tr.tmax - tr.tmin, 1e-3);
  for (let k = 0; k < n * n; k++) { const v = 255 * (tr.temp[k] - tr.tmin) / span; d[4 * k] = v; d[4 * k + 3] = 255; }
  const t = new THREE.DataTexture(d, n, n, THREE.RGBAFormat); t.magFilter = t.minFilter = THREE.LinearFilter; t.needsUpdate = true;
  return t;
}

// ---------------------------------------------------------------- the ground's look
const GLSL_POKRYCIE = /* glsl */`
uniform sampler2D uSplatA, uSplatB, uSzum;
uniform float uPolowa, uKrok, uN, uDaleko, uDetal, uPaleta, uFlat;
uniform vec3 cSnieg, cSkala, cPiarg, cKoso, cTrawa, cWoda, cSzlak, cSkala2, cTrawa2;
varying vec3 vWPos; varying vec3 vWNor;
vec4 szum(vec2 p) { return texture2D(uSzum, p); }
// The ground's colour and roughness at a point, and how bumpy it is there.
vec3 gruntKolor(out float rough, out float bump) {
  vec2 xy = vec2(vWPos.x, -vWPos.z);
  vec2 uv = ((xy + uPolowa) / uKrok + 0.5) / uN;
  vec4 a = texture2D(uSplatA, uv), b = texture2D(uSplatB, uv);
  float slope = 1.0 - clamp(vWNor.y, 0.0, 1.0);
  float h = vWPos.y;
  vec4 n1 = szum(xy / 180.0), n2 = szum(xy / 31.0), n3 = szum(xy / 5.3), n4 = szum(xy / 1.1);
  // Outside the valley's grid: the mountains round it, covered by height and slope.
  float inside = step(abs(xy.x), uPolowa - 2.0) * step(abs(xy.y), uPolowa - 2.0);
  if (uDaleko > 0.5 || inside < 0.5) {
    float snow = smoothstep(2050.0, 2120.0, h + 120.0 * (n1.r - 0.5)) * (1.0 - smoothstep(0.55, 0.75, slope));
    float rock = smoothstep(0.42, 0.62, slope + 0.2 * (n2.r - 0.5)) + smoothstep(2150.0, 2350.0, h) * 0.6;
    float pine = (1.0 - smoothstep(1750.0, 1850.0, h + 60.0 * (n1.g - 0.5))) * (1.0 - smoothstep(0.35, 0.5, slope));
    a = vec4(snow, clamp(rock, 0.0, 1.0) * (1.0 - snow), 0.25 * (1.0 - snow), pine * (1.0 - snow));
    b = vec4(max(0.0, 1.0 - a.x - a.y - a.w) * 0.8, 0.0, 0.0, 0.0);
  }
  float wsum = a.x + a.y + a.z + a.w + b.x + b.y + b.z + 1e-3;
  // Each cover's own texture: snow wind-blown and bluish in its hollows, rock in strata,
  // scree speckled, dwarf pine in dark clumps, autumn meadow tufted, a worn trail.
  float strata = 0.5 + 0.5 * sin(h * 0.9 + n2.g * 6.0 + n1.r * 4.0);
  vec3 snieg = cSnieg * (0.9 + 0.1 * n3.b) - vec3(0.06, 0.03, 0.0) * (1.0 - n2.r);
  vec3 skala = mix(cSkala, cSkala2, strata * 0.6 + n2.b * 0.4) * (0.75 + 0.45 * n3.r) * (0.8 + 0.25 * n4.a);
  vec3 piarg = cPiarg * (0.65 + 0.55 * n4.r) * (0.85 + 0.2 * n3.g);
  vec3 koso = cKoso * (0.55 + 0.7 * smoothstep(0.3, 0.75, n3.b)) * (0.8 + 0.3 * n2.a);
  vec3 trawa = mix(cTrawa, cTrawa2, smoothstep(0.35, 0.7, n2.r)) * (0.75 + 0.35 * n4.b) * (0.9 + 0.15 * n3.a);
  vec3 woda = cWoda;
  vec3 szlak = cSzlak * (0.8 + 0.3 * n4.g);
  // Steep ground shows its rock whatever covers it.
  float bare = smoothstep(0.62, 0.85, slope) * (1.0 - uFlat);
  vec3 col = (snieg * a.x + skala * a.y + piarg * a.z + koso * a.w + trawa * b.x + woda * b.y + szlak * b.z) / wsum;
  col = mix(col, skala, bare * 0.7);
  col = mix(col, col * (0.85 + 0.3 * n1.a), uDetal);
  if (uFlat > 0.5) col = (cSnieg * a.x + cSkala * a.y + cPiarg * a.z + cKoso * a.w + cTrawa * b.x + cWoda * b.y + cSzlak * b.z) / wsum;
  rough = (0.55 * a.x + 0.85 * a.y + 0.95 * a.z + 0.9 * a.w + 0.95 * b.x + 0.1 * b.y + 0.9 * b.z) / wsum;
  bump = uDetal * ((n4.r * 0.5 + n3.r * 0.5) * (0.25 * a.x + 1.2 * a.y + 1.6 * a.z + 1.4 * a.w + 0.8 * b.x + 0.6 * b.z) / wsum);
  return col;
}
`;

export function materialGruntu(tr, tex, paleta, opcje = {}) {
  const m = new THREE.MeshStandardMaterial({ roughness: 0.9, metalness: 0.0, flatShading: !!opcje.flat });
  const u = {
    uSplatA: { value: tex.splat[0] }, uSplatB: { value: tex.splat[1] }, uSzum: { value: tex.szum },
    uPolowa: { value: tr.polowa }, uKrok: { value: tr.krok }, uN: { value: tr.n },
    uDaleko: { value: opcje.daleko ? 1 : 0 }, uDetal: { value: opcje.flat ? 0 : 1 }, uPaleta: { value: 0 }, uFlat: { value: opcje.flat ? 1 : 0 },
  };
  for (const [k, v] of Object.entries(paleta)) u[k] = { value: new THREE.Color(v) };
  m.userData.uniforms = u;
  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, u);
    sh.vertexShader = 'varying vec3 vWPos; varying vec3 vWNor;\n' + sh.vertexShader.replace('#include <worldpos_vertex>',
      `#include <worldpos_vertex>
       vWPos = (modelMatrix * vec4(transformed, 1.0)).xyz;
       vWNor = normalize(mat3(modelMatrix) * objectNormal);`);
    sh.fragmentShader = GLSL_POKRYCIE + sh.fragmentShader
      .replace('#include <map_fragment>', `float gRough, gBump; vec3 gCol = gruntKolor(gRough, gBump); diffuseColor.rgb = gCol;`)
      .replace('#include <roughnessmap_fragment>', `float roughnessFactor = gRough;`)
      .replace('#include <normal_fragment_maps>', `#include <normal_fragment_maps>
        {
          vec3 sx = dFdx(-vViewPosition), sy = dFdy(-vViewPosition);
          vec3 r1 = cross(sy, normal), r2 = cross(normal, sx);
          float det = dot(sx, r1);
          float hb = gBump * 0.35;
          vec3 grad = sign(det) * (dFdx(hb) * r1 + dFdy(hb) * r2);
          normal = normalize(abs(det) * normal - grad);
        }`);
  };
  m.customProgramCacheKey = () => 'grunt' + (opcje.flat ? 'f' : '') + (opcje.daleko ? 'd' : '');
  return m;
}

// The valley's mesh, every grid point a vertex; and the mountains round it.
export function siatkaGruntu(tr, co = 1) {
  const n = Math.floor((tr.n - 1) / co) + 1;
  const pos = new Float32Array(n * n * 3);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const ii = i * co, jj = j * co, k = (i * n + j) * 3;
    pos[k] = -tr.polowa + jj * tr.krok; pos[k + 1] = tr.z[ii * tr.n + jj]; pos[k + 2] = -(-tr.polowa + ii * tr.krok);
  }
  const idx = new Uint32Array((n - 1) * (n - 1) * 6);
  let p = 0;
  for (let i = 0; i < n - 1; i++) for (let j = 0; j < n - 1; j++) {
    const a = i * n + j, b = a + 1, c = a + n, d = c + 1;
    idx[p++] = a; idx[p++] = b; idx[p++] = c; idx[p++] = b; idx[p++] = d; idx[p++] = c;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  g.computeVertexNormals();
  return g;
}

export function siatkaGor(tr, seed = 3) {
  // 16 km square; the valley's edge carried out and raised into ridges and peaks.
  const size = 16000, n = 128, rnd = los(seed);
  const peaks = [];
  for (let k = 0; k < 40; k++) {
    const ang = rnd() * Math.PI * 2, r = 1300 + rnd() * 5500;
    peaks.push([Math.cos(ang) * r, Math.sin(ang) * r, 250 + rnd() * 650, 500 + rnd() * 1200]);
  }
  const ridge = (x, y) => {
    let s = 0;
    for (const [px, py, hh, w] of peaks) { const d = Math.hypot(x - px, y - py) / w; s = Math.max(s, hh * Math.exp(-d * d * 1.6) * (0.8 + 0.2 * Math.sin(x * 0.004 + y * 0.003))); }
    return s + 90 * Math.sin(x * 0.0021 + 1.3) * Math.sin(y * 0.0017 + 0.4);
  };
  const pos = new Float32Array(n * n * 3);
  const P = tr.polowa - tr.krok;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const x = -size / 2 + size * j / (n - 1), y = -size / 2 + size * i / (n - 1);
    const cx = Math.max(-P, Math.min(P, x)), cy = Math.max(-P, Math.min(P, y));
    const edge = tr.h(cx, cy), out = Math.hypot(x - cx, y - cy);
    const f = Math.min(1, out / 1800);
    let h = edge + (ridge(x, y) + 260 * f) * f - (1 - Math.min(1, out / 30)) * 3;
    if (Math.abs(x) < tr.polowa - 30 && Math.abs(y) < tr.polowa - 30) h = tr.zmin - 60; // under the valley
    const k = (i * n + j) * 3;
    pos[k] = x; pos[k + 1] = h; pos[k + 2] = -y;
  }
  const idx = [];
  for (let i = 0; i < n - 1; i++) for (let j = 0; j < n - 1; j++) {
    const a = i * n + j, b = a + 1, c = a + n, d = c + 1;
    idx.push(a, b, c, b, d, c);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setIndex(idx);
  g.computeVertexNormals();
  return g;
}

// ---------------------------------------------------------------- what grows and lies there
function kepaKosodrzewiny(stylizowana) {
  if (stylizowana) {
    const c = new THREE.ConeGeometry(1.1, 2.6, 6, 1); c.translate(0, 1.3, 0);
    return c;
  }
  const parts = [];
  const r = los(11);
  // Three low blobs: some 60 triangles a clump -- there are thousands of clumps.
  for (let k = 0; k < 3; k++) {
    const s = new THREE.IcosahedronGeometry(0.8 + r() * 0.5, 0);
    s.scale(1.0, 0.55 + r() * 0.2, 1.0);
    s.translate((r() - 0.5) * 1.6, 0.35 + r() * 0.3, (r() - 0.5) * 1.6);
    parts.push(s);
  }
  const g = mergeGeometries(parts.map((p) => (p.index ? p.toNonIndexed() : p)));
  // A little noise on the surface: needles, not a ball.
  const p = g.attributes.position;
  for (let i = 0; i < p.count; i++) { const f = 0.88 + 0.24 * r(); p.setXYZ(i, p.getX(i) * f, p.getY(i) * (0.9 + 0.2 * r()), p.getZ(i) * f); }
  g.computeVertexNormals();
  return g;
}

export function roslinnosc(tr, kolor, stylizowana = false, zasieg = 700) {
  const r = los(5);
  const geo = kepaKosodrzewiny(stylizowana);
  const mat = new THREE.MeshStandardMaterial({ color: kolor, roughness: 0.95, flatShading: stylizowana });
  const pts = [];
  for (let i = 0; i < tr.n; i++) for (let j = 0; j < tr.n; j++) {
    if (tr.pokrycie[i * tr.n + j] !== KOSO) continue;
    const x = -tr.polowa + j * tr.krok, y = -tr.polowa + i * tr.krok;
    if (Math.abs(x) > zasieg || Math.abs(y) > zasieg) continue;
    if (r() > (stylizowana ? 0.16 : 0.34)) continue;
    pts.push([x + (r() - 0.5) * tr.krok, y + (r() - 0.5) * tr.krok]);
  }
  const mesh = new THREE.InstancedMesh(geo, mat, pts.length);
  const m = new THREE.Matrix4(), q = new THREE.Quaternion(), s = new THREE.Vector3(), c = new THREE.Color(kolor);
  pts.forEach(([x, y], k) => {
    const sc = stylizowana ? 0.8 + r() * 0.9 : 0.9 + r() * 1.3;
    q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), r() * 6.28);
    s.set(sc * (0.9 + r() * 0.4), sc * (stylizowana ? 1.0 + r() * 0.8 : 0.7 + r() * 0.5), sc);
    m.compose(w2t(x, y, tr.h(x, y) - 0.15), q, s);
    mesh.setMatrixAt(k, m);
    const shade = 0.75 + r() * 0.45;
    mesh.setColorAt(k, c.clone().multiplyScalar(shade));
  });
  mesh.castShadow = true; mesh.receiveShadow = true;
  mesh.userData.punkty = pts;
  return mesh;
}

export function kamienie(tr, kolor, zasieg = 650) {
  const r = los(9);
  const geo = new THREE.DodecahedronGeometry(1, 0);
  const mat = new THREE.MeshStandardMaterial({ color: kolor, roughness: 0.92, flatShading: true });
  const pts = [];
  for (let i = 0; i < tr.n; i++) for (let j = 0; j < tr.n; j++) {
    const c = tr.pokrycie[i * tr.n + j];
    if (c !== PIARG && c !== SKALA) continue;
    const x = -tr.polowa + j * tr.krok, y = -tr.polowa + i * tr.krok;
    if (Math.abs(x) > zasieg || Math.abs(y) > zasieg) continue;
    if (r() > (c === PIARG ? 0.22 : 0.04)) continue;
    pts.push([x + (r() - 0.5) * tr.krok, y + (r() - 0.5) * tr.krok]);
  }
  const mesh = new THREE.InstancedMesh(geo, mat, pts.length);
  const m = new THREE.Matrix4(), q = new THREE.Quaternion(), s = new THREE.Vector3(), cc = new THREE.Color(kolor);
  pts.forEach(([x, y], k) => {
    const sc = 0.25 + Math.pow(r(), 2.2) * 1.4;
    q.setFromEuler(new THREE.Euler(r() * 3, r() * 3, r() * 3));
    s.set(sc * (0.8 + r() * 0.5), sc * (0.5 + r() * 0.4), sc * (0.8 + r() * 0.5));
    m.compose(w2t(x, y, tr.h(x, y) + sc * 0.2), q, s);
    mesh.setMatrixAt(k, m);
    mesh.setColorAt(k, cc.clone().multiplyScalar(0.7 + r() * 0.5));
  });
  mesh.castShadow = true; mesh.receiveShadow = true;
  return mesh;
}

// The people, the chamois, the foxes, the boulders: their parts, as the thermal camera
// sees them too (each mesh has a twin on the thermal layer, as warm as the part).
export function obiekty(tr, lista, termoMat) {
  const grp = new THREE.Group(), twins = new THREE.Group();
  for (const o of lista) {
    const g = new THREE.Group(), gt = new THREE.Group();
    const ground = tr.h(o.x, o.y);
    g.position.copy(w2t(o.x, o.y, ground)); g.rotation.y = o.yaw;
    gt.position.copy(g.position); gt.rotation.y = o.yaw;
    for (const c of o.czesci) {
      let geo;
      if (c.typ === 'kapsula') {
        const a = new THREE.Vector3(c.a[0], c.a[2], -c.a[1]), b = new THREE.Vector3(c.b[0], c.b[2], -c.b[1]);
        const len = a.distanceTo(b);
        geo = new THREE.CapsuleGeometry(c.r, Math.max(len, 1e-3), 4, 10);
        const mid = a.clone().add(b).multiplyScalar(0.5);
        const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), b.clone().sub(a).normalize());
        geo.applyQuaternion(q); geo.translate(mid.x, mid.y, mid.z);
      } else {
        geo = new THREE.SphereGeometry(1, 14, 10);
        geo.scale(c.r[0], c.r[2], c.r[1]);
        geo.translate(c.a[0], c.a[2], -c.a[1]);
      }
      const rodzaj = o.rodzaj.startsWith('glaz');
      const mat = new THREE.MeshStandardMaterial({ color: new THREE.Color(...c.kolor), roughness: rodzaj ? 0.9 : 0.75, flatShading: rodzaj });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.castShadow = true; mesh.receiveShadow = true;
      if (c.snieg) { mesh.position.y -= 0.12; }
      g.add(mesh);
      if (!c.snieg) {
        const tw = new THREE.Mesh(geo, termoMat(c.temp));
        tw.layers.set(TERMO);
        gt.add(tw);
      }
    }
    g.userData = { ...o, ground };
    grp.add(g); twins.add(gt);
  }
  return { grp, twins };
}

// ---------------------------------------------------------------- the tarn
export function staw(tr, env, kolor) {
  const [sx, sy, sr] = tr.staw;
  const geo = new THREE.CircleGeometry(sr + 70, 96);
  geo.rotateX(-Math.PI / 2);
  // Ripples: a normal map made here, slowly moved.
  const size = 256, d = new Uint8Array(size * size * 4), rr = los(4);
  const waves = Array.from({ length: 24 }, () => [rr() * 6.28, 2 + Math.floor(rr() * 10), 2 + Math.floor(rr() * 10), 0.3 + rr()]);
  for (let j = 0; j < size; j++) for (let i = 0; i < size; i++) {
    let dx = 0, dy = 0;
    for (const [ph, kx, ky, a] of waves) { const arg = 6.2832 * (kx * i + ky * j) / size + ph; dx += a * kx * Math.cos(arg); dy += a * ky * Math.cos(arg); }
    const k = (j * size + i) * 4, s = 0.012;
    const nx = -dx * s, ny = -dy * s, nz = 1, l = Math.hypot(nx, ny, nz);
    d[k] = 127.5 * (nx / l + 1); d[k + 1] = 127.5 * (ny / l + 1); d[k + 2] = 127.5 * (nz / l + 1); d[k + 3] = 255;
  }
  const nt = new THREE.DataTexture(d, size, size, THREE.RGBAFormat);
  nt.wrapS = nt.wrapT = THREE.RepeatWrapping; nt.repeat.set(8, 8); nt.generateMipmaps = true; nt.minFilter = THREE.LinearMipmapLinearFilter; nt.needsUpdate = true;
  const mat = new THREE.MeshPhysicalMaterial({ color: kolor, roughness: 0.06, metalness: 0.0, normalMap: nt, normalScale: new THREE.Vector2(0.35, 0.35), envMap: env, envMapIntensity: 1.2, clearcoat: 1.0, clearcoatRoughness: 0.05, transparent: true, opacity: 0.94 });
  const m = new THREE.Mesh(geo, mat);
  m.position.copy(w2t(sx, sy, tr.lustro));
  m.receiveShadow = true;
  m.userData.normalna = nt;
  return m;
}

// ---------------------------------------------------------------- the hut, the base
export function baza(tr, noc) {
  const g = new THREE.Group();
  const [bx, by] = tr.baza;
  const z = tr.h(bx + 18, by);
  const wood = new THREE.MeshStandardMaterial({ color: 0x6b4a2f, roughness: 0.85 });
  const stone = new THREE.MeshStandardMaterial({ color: 0x8a8580, roughness: 0.95 });
  const roof = new THREE.MeshStandardMaterial({ color: 0x3a3f46, roughness: 0.7, metalness: 0.2 });
  const hut = new THREE.Mesh(new THREE.BoxGeometry(14, 4, 9), wood); hut.position.set(0, 3.2, 0);
  const plinth = new THREE.Mesh(new THREE.BoxGeometry(14.6, 1.4, 9.6), stone); plinth.position.set(0, 0.7, 0);
  const rg = new THREE.CylinderGeometry(0.01, 6.6, 4.2, 4, 1); rg.rotateY(Math.PI / 4); rg.scale(1.55, 1, 1.05);
  const r = new THREE.Mesh(rg, roof); r.position.set(0, 7.3, 0);
  const win = new THREE.MeshStandardMaterial({ color: 0x222222, emissive: noc ? 0xffb35c : 0x000000, emissiveIntensity: noc ? 6 : 0 });
  for (let k = -2; k <= 2; k++) { const w = new THREE.Mesh(new THREE.BoxGeometry(1.1, 1.2, 0.1), win); w.position.set(k * 2.6, 3.4, 4.56); g.add(w); }
  const pad = new THREE.Mesh(new THREE.CylinderGeometry(4, 4, 0.2, 40), new THREE.MeshStandardMaterial({ color: 0x2b2f33, roughness: 0.6 }));
  pad.position.set(-16, tr.h(bx - 16 + 18, by) - z + 0.1, 3);
  const h = new THREE.Mesh(new THREE.RingGeometry(2.6, 3.1, 40), new THREE.MeshStandardMaterial({ color: 0xffc23d, emissive: noc ? 0xffc23d : 0, emissiveIntensity: noc ? 2 : 0 }));
  h.rotation.x = -Math.PI / 2; h.position.set(-16, pad.position.y + 0.12, 3);
  const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.12, 12, 8), stone); mast.position.set(8, 6, -6);
  const beacon = new THREE.Mesh(new THREE.SphereGeometry(0.25, 10, 8), new THREE.MeshStandardMaterial({ color: 0xff2a2a, emissive: 0xff2a2a, emissiveIntensity: 8 }));
  beacon.position.set(8, 12.2, -6); beacon.userData.migacz = true;
  [hut, plinth, r, pad, h, mast, beacon].forEach((m) => { m.castShadow = true; m.receiveShadow = true; g.add(m); });
  g.position.copy(w2t(bx + 18, by, z));
  return g;
}

// ---------------------------------------------------------------- sky and clouds
export function niebo(styl) {
  const sky = new Sky();
  sky.scale.setScalar(450000);
  const u = sky.material.uniforms;
  u.turbidity.value = styl.niebo.turbidity; u.rayleigh.value = styl.niebo.rayleigh;
  u.mieCoefficient.value = styl.niebo.mie; u.mieDirectionalG.value = styl.niebo.mieG;
  return sky;
}

export function kierunekSlonca(az, el) {
  const a = THREE.MathUtils.degToRad(az), e = THREE.MathUtils.degToRad(el);
  // az from north, clockwise; north is -Z, east +X
  return new THREE.Vector3(Math.sin(a) * Math.cos(e), Math.sin(e), -Math.cos(a) * Math.cos(e));
}

function teksturaChmury(seed) {
  const c = document.createElement('canvas'); c.width = c.height = 256;
  const x = c.getContext('2d'), r = los(seed);
  for (let k = 0; k < 28; k++) {
    const cx = 128 + (r() - 0.5) * 150, cy = 140 + (r() - 0.5) * 60, rad = 30 + r() * 60;
    const g = x.createRadialGradient(cx, cy, 0, cx, cy, rad);
    g.addColorStop(0, 'rgba(255,255,255,0.55)'); g.addColorStop(1, 'rgba(255,255,255,0)');
    x.fillStyle = g; x.beginPath(); x.arc(cx, cy, rad, 0, 6.3); x.fill();
  }
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}

export function chmury(styl) {
  const g = new THREE.Group(), r = los(21);
  for (let k = 0; k < styl.chmury.ile; k++) {
    const mat = new THREE.SpriteMaterial({ map: teksturaChmury(k + 1), color: styl.chmury.kolor, transparent: true, opacity: styl.chmury.krycie * (0.6 + 0.4 * r()), depthWrite: false, fog: false });
    const s = new THREE.Sprite(mat);
    const ang = r() * Math.PI * 2, dist = 2500 + r() * 6000;
    s.position.set(Math.cos(ang) * dist, 2700 + r() * 900, Math.sin(ang) * dist);
    const sc = 1600 + r() * 2600; s.scale.set(sc, sc * 0.45, 1);
    g.add(s);
  }
  return g;
}

export function gwiazdy() {
  const r = los(33), n = 2500, p = new Float32Array(n * 3), c = new Float32Array(n * 3);
  for (let k = 0; k < n; k++) {
    const u = r() * 2 - 1, th = r() * 6.283, el = Math.asin(Math.abs(u) * 0.95 + 0.05);
    const R = 90000;
    p[3 * k] = R * Math.cos(el) * Math.cos(th); p[3 * k + 1] = R * Math.sin(el); p[3 * k + 2] = R * Math.cos(el) * Math.sin(th);
    const b = 0.4 + 0.6 * Math.pow(r(), 3); c[3 * k] = b; c[3 * k + 1] = b; c[3 * k + 2] = b * (0.9 + 0.2 * r());
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(p, 3)); g.setAttribute('color', new THREE.BufferAttribute(c, 3));
  return new THREE.Points(g, new THREE.PointsMaterial({ size: 1.6, sizeAttenuation: false, vertexColors: true, fog: false, transparent: true, opacity: 0.9, depthWrite: false }));
}

export function przygotujTekstury(tr) {
  return { splat: teksturyPokrycia(tr), szum: teksturaSzumu(), temp: teksturaTemperatury(tr) };
}
