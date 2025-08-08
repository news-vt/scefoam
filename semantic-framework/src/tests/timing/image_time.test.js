/* eslint-disable no-console */
const fs   = require('fs');
const path = require('path');
const SemanticFramework = require('../../semantic_framework').default;
const { setGlobalDispatcher, Agent } = require('undici');

// Project paths
const rootDir   = path.resolve(__dirname, '..');     // adjust if your tree differs
const dataDir   = path.join(rootDir, 'test_data', 'images');
const resultDir = path.join(__dirname, '..', 'result');
const publicDir = path.join(__dirname, '..', 'test_public');
const tmpDir    = path.join(process.cwd(), '__tmp');
const kbFile    = path.join(publicDir, 'kb_images_test.json');
const csvPath   = path.join(resultDir, 'image_timing.csv');

// Give a bit more room for slower CI machines
jest.setTimeout(60_000);

// Use a fresh TCP connection per request (avoid stale keep-alive sockets)
// NOTE: keepAliveTimeout must be > 0, so we use 1ms to effectively not keep connections alive.
let agent;
beforeAll(() => {
  agent = new Agent({ keepAliveTimeout: 1, keepAliveMaxTimeout: 1, connections: 1 });
  setGlobalDispatcher(agent);
});

afterAll(async () => {
  try { await agent.close(); } catch {}
});

// Prepare folders & KB
beforeAll(() => {
  [publicDir, resultDir, tmpDir].forEach(d => fs.mkdirSync(d, { recursive: true }));
  if (!fs.existsSync(kbFile)) fs.writeFileSync(kbFile, '{}');
});

let sf;
beforeAll(() => { sf = new SemanticFramework({ kbPath: kbFile, imgDir: tmpDir }); });

// Collect fixtures
const jpgFiles = fs.readdirSync(dataDir).filter(f => f.endsWith('.jpg'));
if (jpgFiles.length === 0) throw new Error(`No .jpg fixtures in ${dataDir}`);

// Copy them to tmp so <img:...> resolves relative to imgDir
beforeAll(() => {
  jpgFiles.forEach(f =>
    fs.copyFileSync(path.join(dataDir, f), path.join(tmpDir, f)));
});

// ---- transient-error helpers ------------------------------------------------
const ns2ms = ns => Number(ns) / 1e6;
const encTimes = [];
const decTimes = [];

const sleep = ms => new Promise(r => setTimeout(r, ms));
const isTransient = (e) =>
  /other side closed|UND_ERR_SOCKET|ECONNRESET|EPIPE/i.test(String(e && e.message || e));

async function withRetryAsync(fn, { tries = 3, backoffMs = 150 } = {}) {
  let last;
  for (let i = 0; i < tries; i++) {
    try { return await fn(); } catch (e) {
      if (!isTransient(e) || i === tries - 1) throw e;
      last = e; await sleep(backoffMs * (i + 1));
    }
  }
  throw last;
}

function withRetrySync(fn, { tries = 3 } = {}) {
  let last;
  for (let i = 0; i < tries; i++) {
    try { return fn(); } catch (e) {
      if (!isTransient(e) || i === tries - 1) throw e;
      // tiny backoff without timers
      Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 50 * (i + 1));
      last = e;
    }
  }
  throw last;
}

// ---- tests ------------------------------------------------------------------
jpgFiles.forEach(file => {
  const token = `<img:${file}>`;

  test(`encode ${file}`, () => {
    const t0 = process.hrtime.bigint();
    const v  = withRetrySync(() => sf.encode_vec(token));
    const t1 = process.hrtime.bigint();
    encTimes.push(ns2ms(t1 - t0));
    expect(Array.isArray(v)).toBe(true);
  });

  test(`decode ${file}`, async () => {
    const v  = withRetrySync(() => sf.encode_vec(token));
    const t0 = process.hrtime.bigint();
    await withRetryAsync(() => sf.decode_vec(v));
    const t1 = process.hrtime.bigint();
    decTimes.push(ns2ms(t1 - t0));
  });
});

// ---- results ----------------------------------------------------------------
afterAll(() => {
  if (!encTimes.length) return;

  const avg = arr => arr.reduce((s, v) => s + v, 0) / arr.length;
  const csv = `metric,avgMilliseconds\n` +
              `embedImage,${avg(encTimes).toFixed(2)}\n` +
              `decodeImage,${avg(decTimes).toFixed(2)}\n`;

  fs.writeFileSync(csvPath, csv, 'utf8');
  console.log(`\n→ Wrote CSV to ${csvPath}\n`);
});
