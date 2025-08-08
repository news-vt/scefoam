// src/tests/prediction/videofeed_pred.test.js
/* eslint-env jest */
/* eslint-disable no-console */
const fs   = require('fs');
const path = require('path');
const SemanticFramework = require('../../semantic_framework').default;

jest.setTimeout(900_000);          // plenty of time for 10× encode / predict
const CYCLES        = 20;          // teach loop
const PRED_SRC_IDX  = 0;           // frame‑4 (0‑based)
const PRED_TGT_IDX  = 1;           // we expect frame‑5

const rootDir   = path.resolve(__dirname, '..');                     // src/tests
const dataRoot  = path.join(rootDir, 'test_data', 'video_feed','park_feed');
const resultDir = path.join(rootDir, 'result');
const publicDir = path.join(rootDir, 'test_public');
const tmpDir    = path.join(process.cwd(), '__tmp');                 // codec scratch

let frames = fs.readdirSync(dataRoot)
               .filter(f => /\.(png|jpe?g)$/i.test(f))
               .sort((a, b) => {
                 const n = s => Number((s.match(/(\d+)/) || [])[1] || 0);
                 return n(a) - n(b);
               });

if (frames.length < PRED_TGT_IDX + 1)
  throw new Error(`Need ≥${PRED_TGT_IDX + 1} frames in ${dataRoot}`);

const ensure  = d => fs.mkdirSync(d, { recursive: true });
const blankKB = f => fs.writeFileSync(f, JSON.stringify({
  map:{}, receivedData:[], hexHistory:[]
}, null, 2));

const teacherKB    = path.join(publicDir, 'kb_teacher_vf.json');
const apprenticeKB = path.join(publicDir, 'kb_apprentice_vf.json');
const probeKB      = path.join(publicDir, 'kb_probe_vf.json');

let teacher, apprentice, probe;
beforeAll(() => {
  [publicDir, resultDir, tmpDir].forEach(ensure);
  [teacherKB, apprenticeKB, probeKB].forEach(blankKB);

  frames.forEach(fn =>
    fs.copyFileSync(path.join(dataRoot, fn), path.join(tmpDir, fn))
  );

  teacher    = new SemanticFramework({ kbPath: teacherKB,    imgDir: tmpDir });
  apprentice = new SemanticFramework({ kbPath: apprenticeKB, imgDir: tmpDir });
  probe      = new SemanticFramework({ kbPath: probeKB,      imgDir: tmpDir });
});

const TOK = frames.map(f => `<img:${f}>`);

test(`teach ${frames.length} frames × ${CYCLES} cycles`, async () => {
  for (let c = 1; c <= CYCLES; c++) {
    for (const tok of TOK) {
      // “send then receive” for each frame, per user instructions
      await apprentice.receive( teacher.send([tok]) );
    }
  }
});

afterAll(async () => {
  const srcTok = TOK[PRED_SRC_IDX];   // frame 4 token
  const tgtTok = TOK[PRED_TGT_IDX];   // frame 5 token (for sanity / naming)

  // Ask the apprentice to predict.  ImageCodec can hand us a JPEG directly
  // when we pass {wantJPEG:true} …
  const pred = await apprentice.predict(srcTok, 'image', { wantJPEG: true });

  // If server included a JPEG buffer we can use it; otherwise decode locally
  const jpgBuf = pred.jpeg
    ? pred.jpeg
    : await teacher.imgCodec.decode(pred.latent);  // decode_vec → ImageCodec.decode

  const outName = `pred_${frames[PRED_TGT_IDX].replace(/\.(png|jpe?g)$/i,'')}.jpg`;
  const outPath = path.join(resultDir, outName);
  fs.writeFileSync(outPath, jpgBuf);

  expect(fs.existsSync(outPath)).toBe(true);
  console.log(`\n→ Predicted frame written to ${outPath}\n`);
});
