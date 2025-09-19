/* eslint-env jest */
/* eslint-disable no-console */

// image_vid_pred.test.js
// Integration test: learn on ISS_NASA.mp4 frames, then restore
// ISS_NASA_packetloss.mp4 by replacing corrupted frames (from
// ISS_NASA_packetloss.csv) with predicted context frames generated
// from t-1 using the SemanticFramework (SCE-FOAM).
//
// Expected inputs (under the tests root):
//   src/tests/test_data/video_feed_predict/ISS_NASA.mp4
//   src/tests/test_data/video_feed_predict/ISS_NASA_packetloss.mp4
//   src/tests/test_data/video_feed_predict/ISS_NASA_packetloss.csv
//
// Output:
//   src/tests/result/ISS_NASA_restored.mp4
//
// Requirements:
// - ffmpeg and ffprobe must be on PATH
// - image_translator.py server is spawned by this test (port 8082)
// - semantic_framework.js + image_controller.js already part of repo

const fs = require('fs');
const path = require('path');
const { spawn, spawnSync } = require('child_process');
const SemanticFramework = require('../../semantic_framework').default;

jest.setTimeout(1_800_000); // up to 30 minutes for full videos

// ---- Tunables ----
const SAMPLE_EVERY = 1;          // use every Nth frame
const MAX_FRAMES   = Infinity;   // set to limit processing for speed
const DOWNSTREAM_TIMEOUT_MS = 45_000;  // per predict() call

// ---- Small helpers ----
function ensure(d) { fs.mkdirSync(d, { recursive: true }); }
const sleep = ms => new Promise(r => setTimeout(r, ms));

function which(cmds) {
  for (const c of Array.isArray(cmds) ? cmds : [cmds]) {
    const r = spawnSync(process.platform === 'win32' ? 'where' : 'which', [c], { encoding: 'utf8' });
    if (r.status === 0 && r.stdout.trim()) return r.stdout.trim().split(/\r?\n/)[0];
  }
  return null;
}

// Walk up from __dirname until we find a folder literally named "tests"
function findTestsRoot(startDir) {
  let dir = path.resolve(startDir);
  for (let i = 0; i < 8; i++) {
    if (path.basename(dir) === 'tests') return dir;
    const parent = path.dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  // Fallback: assume parent of current dir (works for src/tests/<any_subdir>)
  return path.resolve(startDir, '..');
}

const testsRoot = findTestsRoot(__dirname);

// Candidates to locate your data folder from various plausible locations
const dataCandidates = [];
(function buildDataCandidates() {
  // Try under the discovered tests root
  dataCandidates.push(path.join(testsRoot, 'test_data', 'video_feed_predict'));

  // Try walking further up a few levels (just in case)
  let dir = testsRoot;
  for (let i = 0; i < 4; i++) {
    dataCandidates.push(path.join(dir, 'test_data', 'video_feed_predict'));
    dir = path.resolve(dir, '..');
  }

  // Also consider the current test folder and its parents
  dir = __dirname;
  for (let i = 0; i < 4; i++) {
    dataCandidates.push(path.join(dir, 'test_data', 'video_feed_predict'));
    dir = path.resolve(dir, '..');
  }
})();

function findDataDir() {
  for (const d of dataCandidates) {
    const a = path.join(d, 'ISS_NASA.mp4');
    const b = path.join(d, 'ISS_NASA_packetloss.mp4');
    const c = path.join(d, 'ISS_NASA_packetloss.csv');
    if (fs.existsSync(a) && fs.existsSync(b) && fs.existsSync(c)) return d;
  }
  return null;
}

const dataDir = findDataDir();
if (!dataDir) {
  console.error('[image_vid_pred.test] Looked in:');
  for (const d of dataCandidates) console.error('  -', d);
  throw new Error('Could not find ISS_NASA.* files. Please place them under src/tests/test_data/video_feed_predict/');
}

const resultDir  = path.join(testsRoot, 'result');
const publicDir  = path.join(testsRoot, 'test_public');
const tmpDir     = path.join(testsRoot, '__tmp_vid'); // keep temp under tests root
const framesDir  = path.join(tmpDir, 'frames_clean');
const glitchDir  = path.join(tmpDir, 'frames_glitch');
const restoreDir = path.join(tmpDir, 'frames_restore');

const cleanVid   = path.join(dataDir, 'ISS_NASA.mp4');
const glitchVid  = path.join(dataDir, 'ISS_NASA_packetloss.mp4');
const lossCsv    = path.join(dataDir, 'ISS_NASA_packetloss.csv');
const restoredMp4= path.join(resultDir, 'ISS_NASA_restored.mp4');

// ---- ffmpeg helpers ----
function run(cmd, args, opts = {}) {
  const r = spawnSync(cmd, args, { encoding: 'utf8', ...opts });
  if (r.status !== 0) {
    const err = (r.stderr || r.stdout || 'unknown error').toString();
    throw new Error(`[${cmd} ${args.join(' ')}] failed: ${err}`);
  }
  return r.stdout;
}

function ffprobeFPS(fp) {
  const out = run('ffprobe',[
    '-v','error','-select_streams','v:0','-show_entries','stream=r_frame_rate',
    '-of','default=noprint_wrappers=1:nokey=1', fp
  ]);
  const rate = (out.trim().split(/\r?\n/)[0] || '').trim();
  const m = /^(\d+)\s*\/\s*(\d+)$/.exec(rate);
  if (m) return Number(m[1]) / Number(m[2] || 1);
  const n = Number(rate);
  return isFinite(n) && n > 0 ? n : 30;
}

function ffmpegExtractFrames(mp4, outDir, pattern = 'frame_%06d.jpg') {
  ensure(outDir);
  run('ffmpeg',[ '-y', '-i', mp4, path.join(outDir, pattern) ]);
}

function ffmpegEncodeFromFrames(inDir, fps, outMp4, globPattern = 'frame_%06d.jpg') {
  ensure(path.dirname(outMp4));
  run('ffmpeg', [
    '-y', '-framerate', String(fps),
    '-i', path.join(inDir, globPattern),
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', outMp4
  ]);
}

// Minimal CSV reader: we only need frame_index (first column)
function loadCorruptIndices(csvText) {
  const lines = csvText.trim().split(/\r?\n/).filter(Boolean);
  const start = lines[0].toLowerCase().startsWith('frame_index') ? 1 : 0;
  const set = new Set();
  for (let i = start; i < lines.length; i++) {
    const first = lines[i].split(',', 1)[0];
    const idx = Number(first);
    if (Number.isInteger(idx)) set.add(idx);
  }
  return set;
}

// ---- Robust prediction helper (retries on “connection lost”) ----
async function predictJPEGWithRetry(framework, seedTok, {
  wantJPEG = true,
  timeout  = DOWNSTREAM_TIMEOUT_MS,
  tries    = 4,
  backoffMs= 250
} = {}) {
  let lastErr;
  for (let attempt = 1; attempt <= tries; attempt++) {
    try {
      const pred = await framework.predict(seedTok, 'image', { wantJPEG, timeout });

      // Prefer JPEG direct
      if (pred && pred.jpeg && Buffer.isBuffer(pred.jpeg) && pred.jpeg.length > 0) {
        return { jpeg: pred.jpeg, from: 'jpeg' };
      }

      // Fallback: decode latent if present
      if (pred && pred.latent) {
        const jpg = await framework.imgCodec.decode(pred.latent);
        if (jpg && Buffer.isBuffer(jpg) && jpg.length > 0) {
          return { jpeg: jpg, from: 'latent+decode' };
        }
      }

      throw new Error('empty prediction');
    } catch (e) {
      lastErr = e;
      const msg = (e && e.message) ? e.message : String(e);
      const transient = /other side closed|ECONNRESET|socket hang up|ClientOSError|ReadTimeout/i.test(msg);
      if (!transient || attempt === tries) break;

      // Warm-up: re-open a fresh HTTP session by encoding the same context
      try { framework.encode_vec(seedTok); } catch {}

      // Backoff before retrying same context
      await sleep(backoffMs * attempt);
    }
  }
  throw lastErr;
}

// ---- Test scaffolding ----
const KB_teacher = path.join(publicDir, 'kb_teacher_vid.json');
const KB_apprent = path.join(publicDir, 'kb_apprent_vid.json');

let serverProc = null;
let teacher, apprentice;
let fps = 30;

beforeAll(async () => {
  [resultDir, publicDir, tmpDir, framesDir, glitchDir, restoreDir].forEach(ensure);

  // Check deps
  if (!which(['ffmpeg']))  throw new Error('ffmpeg not found in PATH');
  if (!which(['ffprobe'])) throw new Error('ffprobe not found in PATH');

  // Extract frames for both clean and glitched videos
  fps = ffprobeFPS(cleanVid);
  ffmpegExtractFrames(cleanVid,  framesDir);
  ffmpegExtractFrames(glitchVid, glitchDir);

  // Start the image codec server
  const py = which(['python3','python']) || 'python3';
  const pyScript = path.resolve(__dirname, '../../image_translator.py'); // src/image_translator.py
  serverProc = spawn(py, [pyScript, '--host','127.0.0.1','--port','8082'], { stdio: 'ignore' });

  // Initialize frameworks and KBs
  const blankKB = j => fs.writeFileSync(j, JSON.stringify({ map:{}, receivedData:[], hexHistory:[] }));
  [KB_teacher, KB_apprent].forEach(blankKB);
  teacher    = new SemanticFramework({ kbPath: KB_teacher, imgDir: framesDir });
  apprentice = new SemanticFramework({ kbPath: KB_apprent, imgDir: framesDir });

  // Wait for codec server to be responsive by attempting a tiny encode
  const firstFrame = fs.readdirSync(framesDir).find(f => /\.jpe?g$/i.test(f));
  if (!firstFrame) throw new Error('No frames extracted from clean video');

  let ok = false;
  for (let tries = 0; tries < 40; tries++) {
    try {
      const v = teacher.encode_vec(`<img:${firstFrame}>`);
      if (Array.isArray(v) && v.length > 10) { ok = true; break; }
    } catch { /* not ready yet */ }
    await sleep(500);
  }
  if (!ok) throw new Error('Image codec server did not become ready');
});

afterAll(() => {
  try { serverProc && serverProc.kill(); } catch {}
});

test('video restoration via context prediction on corrupted frames', async () => {
  // 1) Prepare list of clean frames we’ll use for teaching + as source for t-1
  const frameFiles = fs.readdirSync(framesDir)
    .filter(f => /\.jpe?g$/i.test(f))
    .sort()
    .filter((_, idx) => idx % SAMPLE_EVERY === 0);

  const totalFrames = Math.min(frameFiles.length, MAX_FRAMES);
  if (totalFrames < 2) throw new Error('Need at least 2 frames from the clean video');

  // 2) Train/learn: decode sequence through apprentice (teacher->apprentice)
  for (let i = 0; i < totalFrames; i++) {
    const tok = `<img:${frameFiles[i]}>`;
    await apprentice.receive( teacher.send([tok]) );
  }

  // 3) Corrupted frame indices from CSV
  const lossCSV = fs.readFileSync(lossCsv, 'utf8');
  const corruptIdx = loadCorruptIndices(lossCSV);

  // 4) Walk frames. If index i is corrupted, predict from clean(i-1);
  //    otherwise, keep the glitched frame as-is.
  const glitchFrames = fs.readdirSync(glitchDir).filter(f => /\.jpe?g$/i.test(f)).sort();
  if (!glitchFrames.length) throw new Error('No frames extracted from glitched video');

  for (let i = 0; i < totalFrames; i++) {
    const outPath = path.join(restoreDir, frameFiles[i]);

    // Not corrupted or first frame — keep the glitched frame
    if (!corruptIdx.has(i) || i === 0) {
      fs.copyFileSync(path.join(glitchDir, glitchFrames[i] || glitchFrames[glitchFrames.length - 1]), outPath);
      continue;
    }

    // Predict context using t-1 (clean), with retry on connection loss
    const prevTok = `<img:${frameFiles[i-1]}>`;
    let wrote = false;
    try {
      const { jpeg, from } = await predictJPEGWithRetry(
        apprentice,
        prevTok,
        { wantJPEG: true, timeout: DOWNSTREAM_TIMEOUT_MS, tries: 4, backoffMs: 250 }
      );
      fs.writeFileSync(outPath, jpeg);
      wrote = true;
      // optional: console.log(`predicted frame ${i} via ${from}`);
    } catch (e) {
      console.warn(`predict failed at frame ${i}: ${e.message || e}`);
    }

    // Fallback: if predict failed, copy glitched frame
    if (!wrote) {
      fs.copyFileSync(path.join(glitchDir, glitchFrames[i] || glitchFrames[glitchFrames.length - 1]), outPath);
    }
  }

  // 5) Encode restored frames back into mp4
  ensure(path.dirname(restoredMp4));
  ffmpegEncodeFromFrames(restoreDir, fps, restoredMp4);
  console.log(`\n→ Restored video written to ${restoredMp4}\n`);
});
