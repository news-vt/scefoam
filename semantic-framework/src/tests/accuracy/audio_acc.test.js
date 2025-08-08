/* eslint-env jest */
/* eslint-disable no-console */
const fs            = require('fs');
const path          = require('path');
const os            = require('os');
const crypto        = require('crypto');
const { spawnSync } = require('child_process');
const SemanticFramework = require('../../semantic_framework').default;
const { setGlobalDispatcher, Agent } = require('undici');

jest.setTimeout(60_000);

const rootDir      = path.resolve(__dirname, '..');
const dataDir      = path.join(rootDir, 'test_data', 'audio');
const resultDir    = path.join(rootDir, 'result');
const publicDir    = path.join(rootDir, 'test_public');
const tmpDir       = fs.mkdtempSync(path.join(os.tmpdir(), 'aud_xcorr_'));
const kbPath       = path.join(publicDir, 'kb_audio_acc.json');
const csvPath      = path.join(resultDir, 'audio_acc.csv');
const reconOutDir  = path.join(resultDir, 'acc_reconstruct');

// Undici agent: 1 ms keep-alive = effectively disabled 
let agent;
beforeAll(() => {
  agent = new Agent({ keepAliveTimeout: 1, keepAliveMaxTimeout: 1, connections: 1 });
  setGlobalDispatcher(agent);
});
afterAll(async () => { try { await agent.close(); } catch {} });

// prep dirs and copy fixtures
[ publicDir, resultDir, reconOutDir ].forEach(d => fs.mkdirSync(d, { recursive: true }));
fs.writeFileSync(kbPath, '{}');
fs.readdirSync(dataDir).forEach(f =>
  fs.copyFileSync(path.join(dataDir, f), path.join(tmpDir, f))
);

// helper: convert anything → .wav via ffmpeg
function toWav(absFile) {
  if (absFile.toLowerCase().endsWith('.wav')) return absFile;
  const wavFile = absFile.replace(/\.\w+$/, '.wav');
  const { status } = spawnSync('ffmpeg', [
    '-nostdin','-loglevel','error','-y',
    '-i', absFile, wavFile
  ]);
  if (status !== 0) throw new Error('ffmpeg conversion failed');
  return wavFile;
}

// helper: x-corr score via python script
function xcorrScore(refWav, degWav) {
  const out = spawnSync(
    'python', [ path.join(__dirname, 'xcorr_cli.py'), refWav, degWav ],
    { encoding: 'utf8' }
  );
  if (out.status !== 0) throw new Error(out.stderr || 'xcorr_cli failed');
  const val = parseFloat(out.stdout.trim());
  return Number.isFinite(val) ? val : 0.0;
}

// transient-socket retry helpers
const sleep = ms => new Promise(r => setTimeout(r, ms));
const isTransient = e =>
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
      Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 50 * (i + 1));
      last = e;
    }
  }
  throw last;
}

const sf = new SemanticFramework({ kbPath, imgDir: tmpDir });

const rows = [];

for (const file of fs.readdirSync(dataDir).filter(f => /\.(wav|mp3|flac)$/i.test(f))) {
  test(`XCorr for ${file}`, async () => {
    const origTmp = path.join(tmpDir, file);
    const refWav  = toWav(origTmp);

    /* reconstruct via framework */
    const vec       = withRetrySync(() => sf.encode_vec(`<audio:${path.basename(refWav)}>`));
    const buf       = await withRetryAsync(() => sf.decode_vec(vec));
    const reconFile = path.basename(refWav, '.wav') + '_recon.wav';
    const reconWav  = path.join(tmpDir, reconFile);

    fs.writeFileSync(reconWav, buf);
    fs.copyFileSync(reconWav, path.join(reconOutDir, reconFile));

    const score = xcorrScore(refWav, reconWav);
    rows.push({ file, accuracy: score });
  });
}

afterAll(() => {
  if (!rows.length) return;
  const header = 'fileName,accuracy\n';
  const body   = rows.map(r => `${r.file},${r.accuracy.toFixed(4)}`).join('\n');
  fs.writeFileSync(csvPath, header + body, 'utf8');

  const avg = rows.reduce((sum, r) => sum + r.accuracy, 0) / rows.length;
  console.log(`\n→ audio_acc.csv written – avg XCorr = ${avg.toFixed(4)}\n`);
});
