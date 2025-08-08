// src/tests/prediction/image_pred.test.js
/* eslint-env jest */
/* eslint-disable no-console */
const fs   = require('fs');
const path = require('path');
const SemanticFramework = require('../../semantic_framework').default;

jest.setTimeout(900_000);
const ITERATIONS = 20;

const rootDir   = path.resolve(__dirname, '..');     // src/tests
const dataRoot  = path.join(rootDir, 'test_data');
const imgDir    = fs.existsSync(path.join(dataRoot, 'images'))
                    ? path.join(dataRoot, 'images')
                    : dataRoot;

const resultDir = path.join(rootDir, 'result');
const publicDir = path.join(rootDir, 'test_public');
const tmpDir    = path.join(process.cwd(), '__tmp');
const csvPath   = path.join(resultDir, 'image_prediction.csv');

let jpgs = fs.readdirSync(imgDir).filter(f => /\.jpe?g$/i.test(f));
if (jpgs.length < 2) throw new Error(`Need ≥2 .jpg files in ${imgDir}`);

const img1 = jpgs.find(f => /_1\.jpe?g$/i.test(f)) || jpgs[0];
jpgs = jpgs.filter(f => f !== img1);
const img2 = jpgs.find(f => /_2\.jpe?g$/i.test(f)) || jpgs[0];

const files = { img1, img2 };

const ensure  = d => fs.mkdirSync(d, { recursive: true });
const blankKB = f => fs.writeFileSync(f, JSON.stringify({
  map:{}, receivedData:[], hexHistory:[], predictedText:'', modelReady:false
}, null,2));

const elapsedMs = (startNs, endNs) => Number(endNs - startNs) / 1e6;

const cosine = (a, b) => {
  let dot=0, na=0, nb=0;
  for (let i=0;i<a.length;i++){
    dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i];
  }
  return dot / (Math.sqrt(na)*Math.sqrt(nb));
};

const teacherKB    = path.join(publicDir, 'kb_teacher_img.json');
const apprenticeKB = path.join(publicDir, 'kb_apprentice_img.json');
const probeKB      = path.join(publicDir, 'kb_probe_img.json');

let teacher, apprentice, probe;
beforeAll(() => {
  [publicDir, resultDir, tmpDir].forEach(ensure);
  [teacherKB, apprenticeKB, probeKB].forEach(blankKB);

  Object.values(files).forEach(fn =>
    fs.copyFileSync(path.join(imgDir, fn), path.join(tmpDir, fn))
  );

  teacher    = new SemanticFramework({ kbPath: teacherKB,    imgDir: tmpDir });
  apprentice = new SemanticFramework({ kbPath: apprenticeKB, imgDir: tmpDir });
  probe      = new SemanticFramework({ kbPath: probeKB,      imgDir: tmpDir });
});

const timesMs = [];
const sims    = [];

const tok1 = `<img:${files.img1}>`;
const tok2 = `<img:${files.img2}>`;

const getVec = out => Array.isArray(out) ? out : out.latent;

test(`bidirectional prediction for ${ITERATIONS} iterations`, async () => {
  for (let i = 1; i <= ITERATIONS; i++) {

    /* teach & predict img1 → img2 */
    await apprentice.receive( teacher.send([tok1]) );
    await apprentice.receive( teacher.send([tok2]) );

    let t0 = process.hrtime.bigint();
    const pred12 = await apprentice.predict(tok1, 'image');
    let t1 = process.hrtime.bigint();
    timesMs.push(elapsedMs(t0, t1));

    sims.push(cosine(getVec(pred12), probe.encode_vec(tok2)));
  }
});

afterAll(() => {
  if (!timesMs.length) return;

  const avg = arr => arr.reduce((s,v)=>s+v,0)/arr.length;
  const csv =
    `metric,value\n` +
    `avgPredictMs,${avg(timesMs).toFixed(2)}\n` +
    `avgCosineSim,${avg(sims).toFixed(4)}\n`;

  fs.writeFileSync(csvPath, csv, 'utf8');
  console.log(`\n→ Prediction metrics written to ${csvPath}\n`);
});
