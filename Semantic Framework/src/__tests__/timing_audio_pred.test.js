/* eslint-env jest */
/**
 * Bidirectional audio-pair fine-tuning test
 *
 * odd  i ⇒ teacher sends  [audio1 , audio2]  (twice)    → predict from audio1
 * even i ⇒ teacher sends  [audio2 , audio1]  (twice)    → predict from audio2
 *
 * Saving every predicted WAV lets you listen for convergence.
 */

const fs = require('fs');
const path = require('path');
const SemanticFramework = require('../SemanticFramework').default;

/* ─── config ─── */
jest.setTimeout(900_000);
const ITERATIONS = 20;  // change as you like, but 5 is a good start

/* ─── paths ─── */
const testRoot = __dirname;
const dataDir = path.join(testRoot, '__test_data');
const publicDir = path.join(testRoot, '__test_public');
const tmpDir = path.join(process.cwd(), '__tmp_pred');

const files = { audio1: 'test_audio_1.mp3', audio2: 'test_audio_2.mp3' };
const teacherKB = path.join(publicDir, 'kb_teacher_audio.json');
const apprenticeKB = path.join(publicDir, 'kb_apprentice_audio.json');
const test_sfKB = path.join(publicDir, 'kb_test_sf_audio.json');

/* ─── helpers ─── */
const ensureDir = d => fs.mkdirSync(d, { recursive: true });
const blankKB = fp => fs.writeFileSync(
  fp,
  JSON.stringify(
    { map: {}, receivedData: [], hexHistory: [], predictedText: '', modelReady: false },
    null, 2
  )
);
const save = (name, payload) => {
  fs.writeFileSync(
    path.join(publicDir, name),
    Buffer.isBuffer(payload) ? payload : JSON.stringify(payload, null, 2)
  );
};

/* ─── frameworks ─── */
let teacher, apprentice, test_sf;

beforeAll(() => {
  ensureDir(publicDir); ensureDir(tmpDir);
  blankKB(teacherKB); blankKB(apprenticeKB); blankKB(test_sfKB);

  /* copy MP3 fixtures into tmp */
  Object.values(files).forEach(fn =>
    fs.copyFileSync(path.join(dataDir, fn), path.join(tmpDir, fn))
  );

  teacher   = new SemanticFramework({ kbPath: teacherKB,   imgDir: tmpDir });
  apprentice= new SemanticFramework({ kbPath: apprenticeKB, imgDir: tmpDir });
  test_sf   = new SemanticFramework({ kbPath: test_sfKB,    imgDir: tmpDir });
});

/* ─── sanity-check: codec round-trip of the two sources ─── */
test('embed & reconstruct originals once', async () => {
  for (const src of [files.audio1, files.audio2]) {
    const tok   = `<audio:${src}>`;

    // ① embed (vectorize) the local MP3 → latent
    console.time(`embed-${src}`);
    const latent = test_sf.vectorize(tok);          // sync
    console.timeEnd(`embed-${src}`);

    // ② decode latent back to WAV
    console.time(`reconstruct-${src}`);
    const wavBuf = await test_sf.decodeAsync(latent);   // async
    console.timeEnd(`reconstruct-${src}`);

    // ③ save for manual listening
    const wavName = src.replace(/\.mp3$/, '.wav');
    save(`reconstructed_${wavName}`, wavBuf);
  }
});

test(`cycle audio1→audio2 then audio2→audio1 for ${ITERATIONS} iterations`, async () => {
  // helper to decode raw latent vectors into WAV
  async function decodeLatent(latent) {
    // if the model returned just [v0, v1, …] (no header), prepend a dummy header
    const vec = latent.length >= 3
      ? latent
      : [24_000, 8, 1, ...latent];
    return apprentice.audioCodec.decode(vec);
  }

  const tok1 = `<audio:${files.audio1}>`;
  const tok2 = `<audio:${files.audio2}>`;

  for (let i = 1; i <= ITERATIONS; ++i) {
    // ── 1) Teach & predict audio1 → audio2 ──────────────────────
    await apprentice.receive( teacher.send([tok1]) );
    await apprentice.receive( teacher.send([tok2]) );

    console.time(`predict audio1→audio2 iter ${i}`);
    const latent12 = await apprentice.predict(tok1, 'audio');
    console.timeEnd(`predict audio1→audio2 iter ${i}`);

    const wav12 = await decodeLatent(latent12);
    save(`predicted_audio_iter${i}_audio1_to_audio2.wav`, wav12);

    // ── 2) Teach & predict audio2 → audio1 ──────────────────────
    await apprentice.receive( teacher.send([tok2]) );
    await apprentice.receive( teacher.send([tok1]) );

    console.time(`predict audio2→audio1 iter ${i}`);
    const latent21 = await apprentice.predict(tok2, 'audio');
    console.timeEnd(`predict audio2→audio1 iter ${i}`);

    const wav21 = await decodeLatent(latent21);
    save(`predicted_audio_iter${i}_audio2_to_audio1.wav`, wav21);
  }

  console.log(`📂  Audio outputs in: ${publicDir}`);
});
