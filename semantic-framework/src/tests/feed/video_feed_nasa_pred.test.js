/* eslint-env jest */
/* eslint-disable no-console */

// video_feed_nasa_pred.test.js
//
// Trains on sequential frames from ISS_NASA.mp4 (clean) in frames_controll/,
// then restores ISS_NASA_packetloss.mp4 by replacing frames listed in
// ISS_NASA_packetloss.csv. For any contiguous run [..., n-2, n-1] of loss,
// we take the NEXT known-good frame n (from frames_pktloss/) as context,
// predict n-1, save it, then chain backward to predict n-2, etc.
// If a run reaches the end (no next good), we forward-fill from the previous good.
//
// Inputs (under tests root):
//   src/tests/test_data/video_feed_predict/ISS_NASA.mp4
//   src/tests/test_data/video_feed_predict/ISS_NASA_packetloss.mp4
//   src/tests/test_data/video_feed_predict/ISS_NASA_packetloss.csv
//
// Folders created under tests root:
//   __tmp_vid/frames_controll     (training frames from clean video)
//   __tmp_vid/frames_pktloss      (frames from packet-loss video)
//   __tmp_vid/frames_restored     (final reconstructed frame sequence)
//
// Output MP4:
//   src/tests/result/ISS_NASA_restored.mp4
//
// Requires: ffmpeg, ffprobe in PATH; image_translator.py present.

const fs = require('fs');
const path = require('path');
const { spawn, spawnSync } = require('child_process');
const SemanticFramework = require('../../semantic_framework').default;

jest.setTimeout(1_800_000); // up to 30 minutes

// ---------- Tunables ----------
const TRAIN_CYCLES = 3;        // how many passes over the clean frames
const SAMPLE_EVERY = 1;        // use every Nth frame from clean for train
const MAX_FRAMES   = Infinity; // cap if needed (e.g. 2400 ~ 80s @ 30fps)
const PREDICT_TIMEOUT_MS = 45_000;

// ---------- Path helpers ----------
function ensure(d) { fs.mkdirSync(d, { recursive: true }); }

function which(cmds) {
  for (const c of (Array.isArray(cmds) ? cmds : [cmds])) {
    const r = spawnSync(process.platform === 'win32' ? 'where' : 'which', [c], { encoding: 'utf8' });
    if (r.status === 0 && r.stdout.trim()) return r.stdout.trim().split(/\r?\n/)[0];
  }
  return null;
}

// Find the tests root (folder literally named "tests")
function findTestsRoot(startDir) {
  let dir = path.resolve(startDir);
  for (let i = 0; i < 8; i++) {
    if (path.basename(dir) === 'tests') return dir;
    const parent = path.dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  return path.resolve(startDir, '..');
}

const testsRoot = findTestsRoot(__dirname);

// Candidate data paths
const dataCandidates = [
  path.join(testsRoot, 'test_data', 'video_feed_predict'),
  path.join(__dirname, '..', 'test_data', 'video_feed_predict'),
  path.join(__dirname, 'test_data', 'video_feed_predict'),
];

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
  console.error('[video_feed_nasa_pred.test] Searched:');
  for (const d of dataCandidates) console.error('  -', d);
  throw new Error('Could not find ISS_NASA.* files. Place them under src/tests/test_data/video_feed_predict/');
}

const resultDir   = path.join(testsRoot, 'result');
const publicDir   = path.join(testsRoot, 'test_public');
const tmpRoot     = path.join(testsRoot, '__tmp_vid');

const framesControll = path.join(tmpRoot, 'frames_controll'); // train on clean
const framesPktloss  = path.join(tmpRoot, 'frames_pktloss');  // extracted packet-loss frames (mixed clean/corrupt)
const framesRestored = path.join(tmpRoot, 'frames_restored'); // final sequence (predicted + copied)

const cleanVid   = path.join(dataDir, 'ISS_NASA.mp4');
const glitchVid  = path.join(dataDir, 'ISS_NASA_packetloss.mp4');
const lossCsv    = path.join(dataDir, 'ISS_NASA_packetloss.csv');
const restoredMp4= path.join(resultDir, 'ISS_NASA_restored.mp4');

// ---------- ffmpeg helpers ----------
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

// ---------- CSV ----------
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

// Build contiguous runs from a Set of indices
function contiguousRuns(corruptSet, lastIdxExclusive) {
  const runs = [];
  let i = 0;
  while (i < lastIdxExclusive) {
    if (!corruptSet.has(i)) { i++; continue; }
    const start = i;
    while (i < lastIdxExclusive && corruptSet.has(i)) i++;
    const end = i - 1; // inclusive
    runs.push([start, end]);
  }
  return runs;
}

// ---------- Test scaffolding ----------
const KB_teacher   = path.join(publicDir, 'kb_teacher_vid.json');
const KB_apprent   = path.join(publicDir, 'kb_apprent_vid.json');
const KB_restorer  = path.join(publicDir, 'kb_restorer_vid.json');

let serverProc = null;
let teacher, apprentice, restorer;
let fps = 30;

beforeAll(async () => {
  [resultDir, publicDir, tmpRoot, framesControll, framesPktloss, framesRestored].forEach(ensure);

  // deps
  if (!which(['ffmpeg']))  throw new Error('ffmpeg not found in PATH');
  if (!which(['ffprobe'])) throw new Error('ffprobe not found in PATH');

  // Extract frames
  fps = ffprobeFPS(cleanVid);
  ffmpegExtractFrames(cleanVid,  framesControll); // training frames (clean)
  ffmpegExtractFrames(glitchVid, framesPktloss);  // packet-loss timeline

  // Start the image codec server (Python)
  const py = which(['python3','python']) || 'python3';
  const pyScript = path.resolve(__dirname, '../../image_translator.py'); // relative to src/tests/prediction/
  serverProc = spawn(py, [pyScript, '--host','127.0.0.1','--port','8082'], { stdio: 'ignore' });

  // Init frameworks (+ clean KBs)
  const blankKB = j => fs.writeFileSync(j, JSON.stringify({ map:{}, receivedData:[], hexHistory:[] }));
  [KB_teacher, KB_apprent, KB_restorer].forEach(blankKB);

  // Train clients use frames_controll (clean)
  teacher   = new SemanticFramework({ kbPath: KB_teacher,  imgDir: framesControll });
  apprentice= new SemanticFramework({ kbPath: KB_apprent, imgDir: framesControll });

  // Restorer uses frames_restored as its imgDir so we can seed with a good frame,
  // then reference predicted frames on the next step (chained predictions).
  restorer  = new SemanticFramework({ kbPath: KB_restorer, imgDir: framesRestored });

  // Wait for codec server to be responsive (probe one encode from frames_controll)
  const firstClean = fs.readdirSync(framesControll).find(f => /\.jpe?g$/i.test(f));
  if (!firstClean) throw new Error('No frames extracted from clean video');
  let ok = false;
  for (let tries = 0; tries < 40; tries++) {
    try {
      const v = teacher.encode_vec(`<img:${firstClean}>`);
      if (Array.isArray(v) && v.length > 10) { ok = true; break; }
    } catch { /* not ready */ }
    await new Promise(r => setTimeout(r, 500));
  }
  if (!ok) throw new Error('Image codec server did not become ready');
});

afterAll(() => {
  try { serverProc && serverProc.kill(); } catch {}
});

// ---------- Main test ----------
test('restore packet-loss video with chained context prediction', async () => {
  // Frame lists; assume both extractions produce same count and naming (frame_XXXXXX.jpg)
  const cleanFrames  = fs.readdirSync(framesControll).filter(f => /\.jpe?g$/i.test(f)).sort();
  const pktFrames    = fs.readdirSync(framesPktloss).filter(f => /\.jpe?g$/i.test(f)).sort();

  const total = Math.min(cleanFrames.length, pktFrames.length, MAX_FRAMES);
  if (total < 2) throw new Error('Need at least 2 frames after extraction');

  // TRAIN: multiple passes over clean frames (frames_controll)
  const TOK = cleanFrames.filter((_, i) => i % SAMPLE_EVERY === 0).map(f => `<img:${f}>`);
  for (let c = 1; c <= TRAIN_CYCLES; c++) {
    for (const tok of TOK) {
      await apprentice.receive( teacher.send([tok]) );
    }
  }

  // Parse CSV to get corrupt indices (0-based frame_index)
  const lossCSV = fs.readFileSync(lossCsv, 'utf8');
  const corruptIdx = loadCorruptIndices(lossCSV);

  // 1) Initialize frames_restored by copying ALL known (packet-loss timeline) frames.
  //    We'll overwrite the corrupted ones as we restore them.
  for (let i = 0; i < total; i++) {
    const src = path.join(framesPktloss, pktFrames[i]);
    const dst = path.join(framesRestored, pktFrames[i]); // same naming
    ensure(path.dirname(dst));
    fs.copyFileSync(src, dst);
  }

  // 2) Build contiguous corrupted runs on [0..total)
  const runs = contiguousRuns(corruptIdx, total);

    async function predictJPEGWithRetry(framework, seedTok, {
    wantJPEG = true,
    timeout  = 45000,
    tries    = 4,
    backoffMs= 250
  } = {}) {
    let lastErr;
    for (let attempt = 1; attempt <= tries; attempt++) {
      try {
        // Ask the server to predict a frame from the seed token
        const pred = await framework.predict(seedTok, 'image', { wantJPEG, timeout });

        // Prefer JPEG direct
        if (pred && pred.jpeg && Buffer.isBuffer(pred.jpeg) && pred.jpeg.length > 0) {
          return { jpeg: pred.jpeg, from: 'jpeg' };
        }

        // Fallback: decode latent (if provided)
        if (pred && pred.latent) {
          const jpg = await framework.imgCodec.decode(pred.latent);
          if (jpg && Buffer.isBuffer(jpg) && jpg.length > 0) {
            return { jpeg: jpg, from: 'latent+decode' };
          }
        }

        throw new Error('empty prediction');
      } catch (e) {
        lastErr = e;
        const msg = (e && e.message) || String(e);
        // Common transient errors from uvicorn/requests libs:
        const transient =
          /other side closed|ECONNRESET|socket hang up|ClientOSError|ReadTimeout/i.test(msg);
        if (!transient || attempt === tries) break;

        // Warm-up: do a quick encode on the same seed to reopen a fresh HTTP session
        try { framework.encode_vec(seedTok); } catch {}

        // Backoff before retry
        await sleep(backoffMs * attempt);
      }
    }
    throw lastErr;
  }

  // 3) For each run [a..b], prefer back-fill from next good frame n=b+1 (if exists).
  for (const [a, b] of runs) {
    const nextGood = (() => {
      const n = b + 1;
      return (n < total && !corruptIdx.has(n)) ? n : null;
    })();

    const prevGood = (() => {
      const p = a - 1;
      return (p >= 0 && !corruptIdx.has(p)) ? p : null;
    })();

    if (nextGood !== null) {
      // Seed with next good frame (from frames_pktloss which is already copied to frames_restored)
      let seedName = pktFrames[nextGood];
      let seedTok  = `<img:${seedName}>`; // restorer.imgDir is frames_restored, which already has seedName
      // Chain backwards: n-1, n-2, ..., a
      for (let t = nextGood - 1; t >= a; t--) {
        const outName = pktFrames[t];
        const outPath = path.join(framesRestored, outName);
        try {
            const { jpeg, from } = await predictJPEGWithRetry(
              restorer,
              seedTok,
              { wantJPEG: true, timeout: PREDICT_TIMEOUT_MS, tries: 4, backoffMs: 250 }
            );
            fs.writeFileSync(outPath, jpeg);
            wrote = true;
            // optional: progress log
            // console.log(`predicted frame ${i} via ${from}`);
          } catch (e) {
            console.warn(`predict failed at frame ${i}: ${e.message || e}`);
          }
        // Next step uses the just-produced frame as the new seed
        seedName = outName;
        seedTok  = `<img:${seedName}>`;
      }
    } else if (prevGood !== null) {
      // No next good (run at end) → forward fill from previous good frame
      let seedName = pktFrames[prevGood];
      let seedTok  = `<img:${seedName}>`;
      for (let t = prevGood + 1; t <= b; t++) {
        const outName = pktFrames[t];
        const outPath = path.join(framesRestored, outName);
        try {
          const pred = await restorer.predict(seedTok, 'image', { wantJPEG: true, timeout: PREDICT_TIMEOUT_MS });
          if (pred && pred.jpeg && Buffer.isBuffer(pred.jpeg) && pred.jpeg.length > 0) {
            fs.writeFileSync(outPath, pred.jpeg);
          } else {
            console.warn(`predict returned no jpeg at t=${t}; kept pktloss frame`);
          }
        } catch (e) {
          console.warn(`predict failed at t=${t}: ${e.message || e}; kept pktloss frame`);
        }
        seedName = outName;
        seedTok  = `<img:${seedName}>`;
      }
    } else {
      // Entire clip corrupted? We have no anchor; nothing to do beyond initial copy
      console.warn(`run [${a},${b}] has no prev or next good frame; left pktloss frames as-is`);
    }
  }

  // 4) Encode frames_restored back to MP4
  ensure(path.dirname(restoredMp4));
  ffmpegEncodeFromFrames(framesRestored, fps, restoredMp4);
  console.log(`\n→ Restored video written to ${restoredMp4}\n`);

  expect(fs.existsSync(restoredMp4)).toBe(true);
});
