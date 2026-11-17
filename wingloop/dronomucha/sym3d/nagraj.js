// Record a clip of the 3-D simulator, frame by frame, to an mp4:
//   node nagraj.js KLIP scena.json out.mp4 [fps] [width] [height]
// The page and everything it loads are served from this folder and from local copies of
// three.js and the fonts (SYM3D_DEPS: a folder with node_modules/three, @fontsource/*).
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const [klip, dane, out, fpsArg, wArg, hArg] = process.argv.slice(2);
const fps = Number(fpsArg || 30), W = Number(wArg || 1280), H = Number(hArg || 720);
const WEB = path.join(__dirname, 'web');
const DEPS = process.env.SYM3D_DEPS || path.join(__dirname, 'node_modules', '..');
const FFMPEG = process.env.FFMPEG || 'ffmpeg';
const TYPES = { '.js': 'text/javascript', '.html': 'text/html', '.json': 'application/json', '.css': 'text/css', '.woff2': 'font/woff2', '.woff': 'font/woff' };
const type = (p) => TYPES[path.extname(p)] || 'application/octet-stream';

function fontCss() {
  const css = [];
  for (const [fam, ws] of [['rajdhani', [500, 600, 700]], ['inter', [400, 600]]]) {
    for (const w of ws) {
      const f = path.join(DEPS, 'node_modules', '@fontsource', fam, `${w}.css`);
      css.push(fs.readFileSync(f, 'utf8').replace(/url\(\.\/files\//g, `url(https://fonts.local/${fam}/files/`));
    }
  }
  return css.join('\n');
}

(async () => {
  const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--disable-gpu-vsync', '--disable-gpu-compositing'] });
  const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });
  page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') console.error('[strona]', m.text()); });
  page.on('pageerror', (e) => console.error('[błąd strony]', e.message));
  await page.route('**/*', async (route) => {
    const u = new URL(route.request().url());
    let file = null, body = null;
    if (u.host === 'sym3d.local') file = u.pathname === '/scena.json' ? dane : path.join(WEB, u.pathname);
    else if (u.host === 'cdn.jsdelivr.net' && u.pathname.startsWith('/npm/three@0.160.0/')) file = path.join(DEPS, 'node_modules', 'three', u.pathname.slice('/npm/three@0.160.0/'.length));
    else if (u.host === 'fonts.googleapis.com') body = fontCss();
    else if (u.host === 'fonts.local') { const [, fam, , name] = u.pathname.split('/'); file = path.join(DEPS, 'node_modules', '@fontsource', fam, 'files', name); }
    if (body !== null) return route.fulfill({ status: 200, contentType: 'text/css', body });
    if (file && fs.existsSync(file)) return route.fulfill({ status: 200, contentType: type(file), body: fs.readFileSync(file) });
    return route.fulfill({ status: 404, body: '' });
  });
  const t0 = Date.now();
  await page.goto(`http://sym3d.local/index.html?nagranie=1&klip=${klip}&w=${W}&h=${H}`);
  await page.waitForFunction(() => window.SIM, null, { timeout: 600000 });
  await page.evaluate(() => document.fonts.ready);
  const dl = await page.evaluate(() => window.SIM.klip.dlugosc || 5);
  const n = Math.round(dl * fps);
  // Part of a long clip, as frames to a folder (KLATKI), several recorders side by side:
  // ZAKRES="from:to". A few frames before it first, unrecorded, settle the thermal gain.
  if (process.env.ZAKRES) {
    const [k0, k1] = process.env.ZAKRES.split(':').map(Number), dir = process.env.KLATKI;
    fs.mkdirSync(dir, { recursive: true });
    for (let k = Math.max(0, k0 - 8); k < k0; k++) await page.evaluate((t) => window.SIM.klatka(t), k / fps);
    const tStart = Date.now();
    for (let k = k0; k < Math.min(k1, n); k++) {
      const name = path.join(dir, `${String(k).padStart(5, '0')}.png`);
      if (fs.existsSync(name)) continue;
      await page.evaluate((t) => window.SIM.klatka(t), k / fps);
      fs.writeFileSync(name + '.tmp', await page.screenshot({ type: 'png' }));
      fs.renameSync(name + '.tmp', name);
      if ((k - k0) % 25 === 0) console.log(`  ${k}/${n}  ${((Date.now() - tStart) / 1000 / (k - k0 + 1)).toFixed(1)} s/klatkę`);
    }
    await browser.close();
    console.log(`-> ${dir} ${k0}:${k1}`);
    return;
  }
  console.log(`klip ${klip}: ${n} klatek, scena gotowa po ${((Date.now() - t0) / 1000).toFixed(0)} s`);
  const ff = spawn(FFMPEG, ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(fps), '-c:v', 'png', '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '17', '-preset', 'slow', '-movflags', '+faststart', out]);
  ff.stderr.on('data', (d) => process.stderr.write(d));
  const tStart = Date.now();
  for (let k = 0; k < n; k++) {
    const ta = Date.now();
    await page.evaluate((t) => window.SIM.klatka(t), k / fps);
    const tb = Date.now();
    const png = await page.screenshot({ type: 'png' });
    if (k < 3) console.log(`  klatka ${k}: strona ${tb - ta} ms, zrzut ${Date.now() - tb} ms`);
    if (!ff.stdin.write(png)) await new Promise((r) => ff.stdin.once('drain', r));
    if (k === 0 || k === Math.floor(n / 2) || k === n - 1) fs.writeFileSync(out.replace(/\.mp4$/, `_${k}.png`), png);
    if (k % 15 === 0) console.log(`  ${k}/${n}  ${((Date.now() - tStart) / 1000 / (k + 1)).toFixed(1)} s/klatkę`, JSON.stringify(await page.evaluate(() => Object.fromEntries(Object.entries(window.SIM.czasy).map(([a, b]) => [a, Math.round(b)])))));
  }
  ff.stdin.end();
  await new Promise((r) => ff.on('close', r));
  await browser.close();
  console.log(`-> ${out} (${((Date.now() - t0) / 1000).toFixed(0)} s)`);
})().catch((e) => { console.error(e); process.exit(1); });
