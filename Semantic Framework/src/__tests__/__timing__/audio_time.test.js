// src/__tests__/__timing__/audio_time.test.js
/* eslint-disable no-console */
const fs   = require('fs');
const path = require('path');
const SemanticFramework = require('../../SemanticFramework').default;

/* ─────────── canonical folders ─────────── */
const rootDir   = path.resolve(__dirname, '..');      // project root
const dataDir   = path.join(rootDir, '__test_data__', 'audio');
const resultDir = path.join(rootDir, '__result__');
const publicDir = path.join(rootDir, '__test_public__');
const tmpDir    = path.join(process.cwd(), '__tmp');
const kbFile    = path.join(publicDir, 'kb_audio_test.json');
const csvPath   = path.join(resultDir, 'audio_timing.csv');

/* ─────────── env prep ─────────── */
beforeAll(() => {
  [publicDir, resultDir, tmpDir].forEach(d => fs.mkdirSync(d, { recursive: true }));
  if (!fs.existsSync(kbFile)) fs.writeFileSync(kbFile, '{}');
});

/* ─────────── SemanticFramework ─── */
let sf;
beforeAll(() => { sf = new SemanticFramework({ kbPath: kbFile }); });

/* ─────────── fixtures ─────────── */
const mp3Files = fs.readdirSync(dataDir).filter(f => f.endsWith('.mp3'));
if (mp3Files.length === 0) throw new Error(`No .mp3 fixtures in ${dataDir}`);

beforeAll(() => {
  mp3Files.forEach(f =>
    fs.copyFileSync(path.join(dataDir, f), path.join(tmpDir, f)));
});

/* ─────────── timers ─────────── */
const ns2ms   = ns => Number(ns) / 1e6;
const encMs   = [];
const decMs   = [];

/* ─────────── timed tests ─────────── */
mp3Files.forEach(file => {
  const token = `<audio:${file}>`;

  test(`encode ${file}`, () => {
    const t0 = process.hrtime.bigint();
    const v  = sf.encode_vec(token);
    const t1 = process.hrtime.bigint();
    encMs.push(ns2ms(t1 - t0));
    expect(Array.isArray(v)).toBe(true);
  });

  test(`decode ${file}`, async () => {
    const v  = sf.encode_vec(token);
    const t0 = process.hrtime.bigint();
    await sf.decode_vec(v);
    const t1 = process.hrtime.bigint();
    decMs.push(ns2ms(t1 - t0));
  });
});

/* ─────────── CSV export ─────────── */
afterAll(() => {
  if (!encMs.length) return;

  const avg = arr => arr.reduce((s, v) => s + v, 0) / arr.length;
  const csv = `metric,avgMilliseconds\n` +
              `embedAudio,${avg(encMs).toFixed(2)}\n` +
              `decodeAudio,${avg(decMs).toFixed(2)}\n`;

  fs.writeFileSync(csvPath, csv, 'utf8');
  console.log(`\n→ Wrote CSV to ${csvPath}\n`);
});
