// src/__tests__/audio_pred.test.js
/* eslint-env jest */
/* eslint-disable no-console */
const fs   = require('fs');
const path = require('path');
const SemanticFramework = require('../../SemanticFramework').default;

/* ───────── config ───────── */
jest.setTimeout(900_000);
const ITERATIONS = 20;

/* ───────── folders ───────── */
const rootDir   = path.resolve(__dirname, '..');
const dataDir   = path.join(rootDir, '__test_data__', 'audio');
const resultDir = path.join(rootDir, '__result__');
const publicDir = path.join(rootDir, '__test_public__');
const tmpDir    = path.join(process.cwd(), '__tmp_pred');
const csvPath   = path.join(resultDir, 'audio_prediction.csv');

/* ───────── discover “*_1.mp3” & “*_2.mp3” ───────── */
const mp3s  = fs.readdirSync(dataDir).filter(f => f.endsWith('.mp3'));
const file1 = mp3s.find(f => /_1\.mp3$/i.test(f)) || mp3s[0];
const file2 = mp3s.find(f => /_2\.mp3$/i.test(f)) || mp3s.find(f => f !== file1);
if (!file1 || !file2) throw new Error('Need two .mp3 fixtures ending _1/_2');

const files = { a1: file1, a2: file2 };

/* ───────── helpers ───────── */
const ensure = d => fs.mkdirSync(d, { recursive: true });
const blank  = fp => fs.writeFileSync(fp, JSON.stringify({
  map:{}, receivedData:[], hexHistory:[], predictedText:'', modelReady:false
}, null,2));

const cosine = (u, v) => {
  let dot=0, nu=0, nv=0;
  for (let i=0;i<u.length;i++){ dot+=u[i]*v[i]; nu+=u[i]*u[i]; nv+=v[i]*v[i]; }
  return dot/ (Math.sqrt(nu)*Math.sqrt(nv));
};
const ms = (startNs, endNs) => Number(endNs - startNs) / 1e6;

/* ───────── KB files & frameworks ───────── */
const teacherKB    = path.join(publicDir, 'kb_teacher_audio.json');
const apprenticeKB = path.join(publicDir, 'kb_apprentice_audio.json');
const probeKB      = path.join(publicDir, 'kb_probe_audio.json');

let teacher, apprentice, probe;

beforeAll(() => {
  [publicDir, resultDir, tmpDir].forEach(ensure);
  [teacherKB, apprenticeKB, probeKB].forEach(blank);

  Object.values(files).forEach(fn =>
    fs.copyFileSync(path.join(dataDir, fn), path.join(tmpDir, fn)));

  teacher    = new SemanticFramework({ kbPath: teacherKB,    imgDir: tmpDir });
  apprentice = new SemanticFramework({ kbPath: apprenticeKB, imgDir: tmpDir });
  probe      = new SemanticFramework({ kbPath: probeKB,      imgDir: tmpDir });
});

/* ───────── metric rows ───────── */
const rows = [];

/* ───────── tokens ───────── */
const tok1 = `<audio:${files.a1}>`;
const tok2 = `<audio:${files.a2}>`;

/* ───────── prediction loop ───────── */
test(`bidirectional prediction for ${ITERATIONS} iterations`, async () => {
  for (let i = 1; i <= ITERATIONS; i++) {

    /* teach & predict 1 → 2 */
    await apprentice.receive( teacher.send([tok1]) );
    await apprentice.receive( teacher.send([tok2]) );
    let t0 = process.hrtime.bigint();
    const pred12 = await apprentice.predict(tok1, 'audio');
    let t1 = process.hrtime.bigint();
    const tMs12 = ms(t0, t1);
    const sim12 = cosine(pred12, probe.encode_vec(tok2));

    /* teach & predict 2 → 1 */
    await apprentice.receive( teacher.send([tok2]) );
    await apprentice.receive( teacher.send([tok1]) );
    t0 = process.hrtime.bigint();
    const pred21 = await apprentice.predict(tok2, 'audio');
    t1 = process.hrtime.bigint();
    const tMs21 = ms(t0, t1);
    const sim21 = cosine(pred21, probe.encode_vec(tok1));

    /* save per-iteration averages */
    rows.push({
      iter: i,
      avgPredictMs: ((tMs12 + tMs21) / 2).toFixed(2),
      cosineSim:   ((sim12 + sim21) / 2).toFixed(4)
    });
  }
});

/* ───────── CSV export ───────── */
afterAll(() => {
  if (!rows.length) return;

  const header = 'iteration,avgPredictMs,cosineSim\n';
  const body   = rows.map(r => `${r.iter},${r.avgPredictMs},${r.cosineSim}`).join('\n');

  fs.writeFileSync(csvPath, header + body, 'utf8');
  console.log(`\n→ Prediction metrics written to ${csvPath}\n`);
});
