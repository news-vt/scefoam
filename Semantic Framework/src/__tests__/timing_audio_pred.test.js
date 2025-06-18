/* eslint-env jest */
/**
 * Bidirectional audio-pair fine-tuning test
 *
 * odd  i ⇒ teacher sends  [audio1 , audio2]  (twice)    → predict from audio1
 * even i ⇒ teacher sends  [audio2 , audio1]  (twice)    → predict from audio2
 *
 * Saving every predicted WAV lets you listen for convergence.
 */

const fs   = require('fs');
const path = require('path');
const SemanticFramework = require('../SemanticFramework').default;

/* ─── config ─── */
jest.setTimeout(900_000);
const ITERATIONS = 1;

/* ─── paths ─── */
const testRoot  = __dirname;
const dataDir   = path.join(testRoot, '__test_data');
const publicDir = path.join(testRoot, '__test_public');
const tmpDir    = path.join(process.cwd(), '__tmp_pred');

const files = { audio1: 'test_audio_1.mp3', audio2: 'test_audio_2.mp3' };
const teacherKB    = path.join(publicDir, 'kb_teacher_audio.json');
const apprenticeKB = path.join(publicDir, 'kb_apprentice_audio.json');

/* ─── helpers ─── */
const ensureDir = d => fs.mkdirSync(d, { recursive: true });
const blankKB = fp => fs.writeFileSync(
  fp,
  JSON.stringify(
    { map:{}, receivedData:[], hexHistory:[], predictedText:'', modelReady:false },
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
let teacher, apprentice;

beforeAll(() => {
  ensureDir(publicDir); ensureDir(tmpDir);
  blankKB(teacherKB);   blankKB(apprenticeKB);

  /* copy MP3 fixtures into tmp */
  Object.values(files).forEach(fn =>
    fs.copyFileSync(path.join(dataDir, fn), path.join(tmpDir, fn))
  );

  teacher    = new SemanticFramework({ kbPath: teacherKB,    imgDir: tmpDir });
  apprentice = new SemanticFramework({ kbPath: apprenticeKB, imgDir: tmpDir });
});

/* ─── test ─── */
test(`cycle audio1→audio2 and audio2→audio1 for ${ITERATIONS} iterations`, async () => {
for (let i = 1; i <= ITERATIONS; ++i) {
  const forward = i % 2 === 1;              // odd = 1→2   even = 2→1
  const srcTok  = forward ? `<audio:${files.audio1}>`
                          : `<audio:${files.audio2}>`;
  const tgtTok  = forward ? `<audio:${files.audio2}>`
                          : `<audio:${files.audio1}>`;

  /* ① teacher sends ONLY the source clip twice */
  await apprentice.receive( teacher.send([srcTok]) );   // first pass
  await apprentice.receive( teacher.send([srcTok]) );   // second pass → Δ ready

  /* ② predict continuation of srcTok */
  const latent = await apprentice.predict(srcTok, 'audio');
  const wav    = await apprentice.audioCodec.decode(
                   latent.length >= 3 ? latent : [24_000,8,1,...latent]);

  save(`predicted_audio_iter${i}_${forward?'cont1':'cont2'}.wav`, wav);

  /* ③ now send the target clip once to keep Δ fresh for next round */
  await apprentice.receive( teacher.send([tgtTok]) );
}


  console.log(`📂  Audio outputs in: ${publicDir}`);
});
