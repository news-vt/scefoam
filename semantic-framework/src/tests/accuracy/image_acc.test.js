/* eslint-env jest */
/* eslint-disable no-console */
const fs            = require('fs');
const path          = require('path');
const os            = require('os');
const { spawnSync } = require('child_process');
const SemanticFramework = require('../../semantic_framework').default;
const { setGlobalDispatcher, Agent } = require('undici');

const rootDir      = path.resolve(__dirname, '..');
const dataDir      = path.join(rootDir, 'test_data', 'images');
const resultDir    = path.join(rootDir, 'result');
const publicDir    = path.join(rootDir, 'test_public');
const tmpDir       = fs.mkdtempSync(path.join(os.tmpdir(), 'img_lpips_'));
const kbPath       = path.join(publicDir, 'kb_img_acc.json');
const csvPath      = path.join(resultDir, 'image_acc.csv');
const reconOutDir  = path.join(resultDir, 'acc_reconstruct');

jest.setTimeout(60_000);

// Make Undici use short-lived connections to avoid "other side closed" 
let agent;
beforeAll(() => {
  agent = new Agent({ keepAliveTimeout: 1, keepAliveMaxTimeout: 1, connections: 1 });
  setGlobalDispatcher(agent);
});
afterAll(async () => { try { await agent.close(); } catch {} });

// FS prep 
if (!fs.readdirSync(dataDir).some(f => /\.jpe?g$/i.test(f))) {
  throw new Error(`No JPEGs in ${dataDir}`);
}
[ publicDir, resultDir, reconOutDir ].forEach(d => fs.mkdirSync(d, { recursive: true }));
fs.writeFileSync(kbPath, '{}');
fs.readdirSync(dataDir).forEach(f =>
  fs.copyFileSync(path.join(dataDir, f), path.join(tmpDir, f))
);

// Helpers 
function lpipsScore(refPath, reconPath) {
  const out = spawnSync('python', [ path.join(__dirname, 'lpips_cli.py'), refPath, reconPath ], {
    encoding: 'utf8'
  });
  if (out.status !== 0) throw new Error(out.stderr || 'lpips_cli failed');

  const lines = out.stdout.trim().split(/\r?\n/).filter(l => l);
  const val   = parseFloat(lines.pop());
  return Number.isFinite(val) ? val : 1.0;
}

const sleep = ms => new Promise(r => setTimeout(r, ms));
const isTransient = (e) =>
  /other side closed|UND_ERR_SOCKET|ECONNRESET|EPIPE/i.test(String(e && e.message || e));

async function withRetryAsync(fn, { tries = 3, backoffMs = 150 } = {}) {
  let last;
  for (let i = 0; i < tries; i++) {
    try { return await fn(); }
    catch (e) {
      if (!isTransient(e) || i === tries - 1) throw e;
      last = e; await sleep(backoffMs * (i + 1));
    }
  }
  throw last;
}
function withRetrySync(fn, { tries = 3 } = {}) {
  let last;
  for (let i = 0; i < tries; i++) {
    try { return fn(); }
    catch (e) {
      if (!isTransient(e) || i === tries - 1) throw e;
      // tiny backoff without timers
      Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 50 * (i + 1));
      last = e;
    }
  }
  throw last;
}

// Tests 
let sf;
beforeAll(() => {
  sf = new SemanticFramework({ kbPath, imgDir: tmpDir });
});

const rows = [];
for (const file of fs.readdirSync(dataDir).filter(f => /\.jpe?g$/i.test(f))) {
  test(`LPIPS for ${file}`, async () => {
    const ext       = path.extname(file);
    const base      = path.basename(file, ext);
    const reconFile = `${base}_recon${ext}`;

    const refPath   = path.join(tmpDir, file);
    const vec       = withRetrySync(() => sf.encode_vec(`<img:${file}>`));
    const buf       = await withRetryAsync(() => sf.decode_vec(vec));
    const reconPath = path.join(tmpDir, reconFile);

    fs.writeFileSync(reconPath, buf);
    fs.copyFileSync(reconPath, path.join(reconOutDir, reconFile)); // save for review

    const score = lpipsScore(refPath, reconPath);
    rows.push({ file, lpips: score });
  });
}

afterAll(() => {
  if (!rows.length) return;
  const header = 'fileName,accuracy\n';
  const body   = rows.map(r => `${r.file},${r.lpips.toFixed(4)}`).join('\n');
  fs.writeFileSync(csvPath, header + body, 'utf8');

  const avg = rows.reduce((sum, r) => sum + r.lpips, 0) / rows.length;
  console.log(`\n→ image_acc.csv written – avg LPIPS = ${avg.toFixed(4)}\n`);
});
