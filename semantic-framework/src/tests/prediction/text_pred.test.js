// src/tests/prediction/text_pred.test.js
/* eslint-env jest */
/* eslint-disable no-console */
const fs   = require('fs');
const path = require('path');
const SemanticFramework = require('../../semantic_framework').default;

jest.setTimeout(6_000_000);
const ITERATIONS = 5;

const rootDir   = path.resolve(__dirname, '..');          // src/tests
const dataRoot  = path.join(rootDir, 'test_data');
const txtDir    = fs.existsSync(path.join(dataRoot, 'text'))
                    ? path.join(dataRoot, 'text')
                    : dataRoot;

const resultDir = path.join(rootDir, 'result');
const publicDir = path.join(rootDir, 'test_public');
const tmpDir    = path.join(process.cwd(), '__tmp');
const csvPath   = path.join(resultDir, 'text_prediction.csv');

let txts = fs.readdirSync(txtDir).filter(f => f.endsWith('.txt'));
if (txts.length < 2) throw new Error(`Need ≥2 .txt files in ${txtDir}`);

const text1 = txts.find(f => /_1\.txt$/i.test(f)) || txts[0];
txts = txts.filter(f => f !== text1);
const text2 = txts.find(f => /_2\.txt$/i.test(f)) || txts[0];

const files = { text1, text2 };

const ensure = d => fs.mkdirSync(d, { recursive: true });
const blankKB = fp => fs.writeFileSync(fp, JSON.stringify({
  map:{}, receivedData:[], hexHistory:[], predictedText:'', modelReady:false
}, null, 2));

const save = (name, payload) =>
  fs.writeFileSync(path.join(publicDir, name), String(payload));

// Levenshtein similarity 0-1
function levSim(a, b) {
  const m = a.length, n = b.length;
  const dp = Array.from({ length: m + 1 }, () => new Array(n + 1));
  for (let i=0;i<=m;i++) dp[i][0] = i;
  for (let j=0;j<=n;j++) dp[0][j] = j;
  for (let i=1;i<=m;i++){
    for (let j=1;j<=n;j++){
      dp[i][j] = Math.min(
        dp[i-1][j] + 1,
        dp[i][j-1] + 1,
        dp[i-1][j-1] + (a[i-1] === b[j-1] ? 0 : 1)
      );
    }
  }
  const lev = dp[m][n];
  return 1 - lev / Math.max(m, n);
}

// timing util
const elapsedMs = (t0, t1) => Number(t1 - t0) / 1e6;

const teacherKB    = path.join(publicDir, 'kb_teacher_text.json');
const apprenticeKB = path.join(publicDir, 'kb_apprentice_text.json');
const probeKB      = path.join(publicDir, 'kb_probe_text.json');

let teacher, apprentice, probe;

beforeAll(() => {
  [publicDir, resultDir, tmpDir].forEach(ensure);
  [teacherKB, apprenticeKB, probeKB].forEach(blankKB);

  /* copy fixtures into tmp so <text:…> resolves */
  Object.values(files).forEach(fn =>
    fs.copyFileSync(path.join(txtDir, fn), path.join(tmpDir, fn))
  );

  teacher    = new SemanticFramework({ kbPath: teacherKB,    imgDir: tmpDir });
  apprentice = new SemanticFramework({ kbPath: apprenticeKB, imgDir: tmpDir });
  probe      = new SemanticFramework({ kbPath: probeKB,      imgDir: tmpDir });
});

const timesMs = [];
const sims    = [];

const tok1 = `<text:${files.text1}>`;
const tok2 = `<text:${files.text2}>`;

test(`bidirectional prediction for ${ITERATIONS} iterations`, async () => {
  for (let i=1; i<=ITERATIONS; i++) {

    // teach & predict text1 → text2 
    await apprentice.receive( teacher.send([tok1]) );
    await apprentice.receive( teacher.send([tok2]) );

    let t0 = process.hrtime.bigint();
    const { text: pred12 } = await apprentice.predict(tok1, 'text', { wantText:true });
    let t1 = process.hrtime.bigint();
    timesMs.push(elapsedMs(t0, t1));

    sims.push(levSim(pred12, fs.readFileSync(path.join(tmpDir, files.text2), 'utf8')));

    // teach & predict text2 → text1 
    await apprentice.receive( teacher.send([tok2]) );
    await apprentice.receive( teacher.send([tok1]) );

    t0 = process.hrtime.bigint();
    const { text: pred21 } = await apprentice.predict(tok2, 'text', { wantText:true });
    t1 = process.hrtime.bigint();
    timesMs.push(elapsedMs(t0, t1));

    sims.push(levSim(pred21, fs.readFileSync(path.join(tmpDir, files.text1), 'utf8')));

    save(`pred_iter${i}_${i%2?'1to2':'2to1'}.txt`, i%2 ? pred12 : pred21);
  }
});

afterAll(() => {
  if (!timesMs.length) return;

  const avg = arr => arr.reduce((s,v)=>s+v,0)/arr.length;
  const csv =
    `metric,value\n` +
    `avgPredictMs,${avg(timesMs).toFixed(2)}\n` +
    `avgLevSim,${avg(sims).toFixed(4)}\n`;

  fs.writeFileSync(csvPath, csv, 'utf8');
  console.log(`\n→ Prediction metrics written to ${csvPath}\n`);
});
