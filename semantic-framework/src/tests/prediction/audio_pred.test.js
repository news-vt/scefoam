/* eslint-env jest */
/* eslint-disable no-console */
const fs            = require('fs');
const path          = require('path');
const SemanticFramework = require('../../semantic_framework').default;
const { setGlobalDispatcher, Agent } = require('undici');

jest.setTimeout(900_000);                    // 15 min max for 20 iterations
const ITERATIONS = 20;

//  paths
const rootDir   = path.resolve(__dirname, '..');
const dataDir   = path.join(rootDir, 'test_data', 'audio');
const resultDir = path.join(rootDir, 'result');
const publicDir = path.join(rootDir, 'test_public');
const tmpDir    = path.join(process.cwd(), '__tmp');
const csvPath   = path.join(resultDir, 'audio_prediction.csv');

// Agent: short-lived connections (fixes “other side closed”)
let agent;
beforeAll(() => {
  agent = new Agent({ keepAliveTimeout: 1, keepAliveMaxTimeout: 1, connections: 1 });
  setGlobalDispatcher(agent);
});
afterAll(async () => { try { await agent.close(); } catch {} });

//  fixtures & dirs 
const mp3s  = fs.readdirSync(dataDir).filter(f => f.endsWith('.mp3'));
const file1 = mp3s.find(f => /_1\.mp3$/i.test(f)) || mp3s[0];
const file2 = mp3s.find(f => /_2\.mp3$/i.test(f)) || mp3s.find(f => f !== file1);
if (!file1 || !file2) throw new Error('Need two .mp3 fixtures ending _1/_2');

const files = { a1: file1, a2: file2 };

const ensure = d => fs.mkdirSync(d, { recursive: true });
[publicDir, resultDir, tmpDir].forEach(ensure);

// blank KBs for three roles
const teacherKB    = path.join(publicDir, 'kb_teacher_audio.json');
const apprenticeKB = path.join(publicDir, 'kb_apprentice_audio.json');
const probeKB      = path.join(publicDir, 'kb_probe_audio.json');
[teacherKB, apprenticeKB, probeKB].forEach(fp => fs.writeFileSync(fp, '{}'));

// copy audio into tmpDir
Object.values(files).forEach(fn =>
  fs.copyFileSync(path.join(dataDir, fn), path.join(tmpDir, fn)));

//  helpers
const cosine = (u, v) => {
  let dot = 0, nu = 0, nv = 0;
  for (let i = 0; i < u.length; i++) { dot += u[i]*v[i]; nu += u[i]*u[i]; nv += v[i]*v[i]; }
  return dot / (Math.sqrt(nu) * Math.sqrt(nv));
};
const ms = (ns0, ns1) => Number(ns1 - ns0) / 1e6;

const sleep = ms_ => new Promise(r => setTimeout(r, ms_));
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

// initialise frameworks 
let teacher, apprentice, probe;
beforeAll(() => {
  teacher    = new SemanticFramework({ kbPath: teacherKB,    imgDir: tmpDir });
  apprentice = new SemanticFramework({ kbPath: apprenticeKB, imgDir: tmpDir });
  probe      = new SemanticFramework({ kbPath: probeKB,      imgDir: tmpDir });
});

// test body
const rows = [];
const tok1 = `<audio:${files.a1}>`;
const tok2 = `<audio:${files.a2}>`;

test(`bidirectional prediction for ${ITERATIONS} iterations`, async () => {
  for (let i = 1; i <= ITERATIONS; i++) {
    // teach & predict 1 → 2
    await apprentice.receive( teacher.send([tok1]) );
    await apprentice.receive( teacher.send([tok2]) );
    let t0 = process.hrtime.bigint();
    const pred12 = await withRetryAsync(() => apprentice.predict(tok1, 'audio'));
    let t1 = process.hrtime.bigint();
    const tMs12 = ms(t0, t1);
    const sim12 = cosine(pred12, withRetrySync(() => probe.encode_vec(tok2)));

    // teach & predict 2 → 1
    await apprentice.receive( teacher.send([tok2]) );
    await apprentice.receive( teacher.send([tok1]) );
    t0 = process.hrtime.bigint();
    const pred21 = await withRetryAsync(() => apprentice.predict(tok2, 'audio'));
    t1 = process.hrtime.bigint();
    const tMs21 = ms(t0, t1);
    const sim21 = cosine(pred21, withRetrySync(() => probe.encode_vec(tok1)));

    rows.push({
      iter: i,
      avgPredictMs: ((tMs12 + tMs21) / 2).toFixed(2),
      cosineSim:    ((sim12 + sim21) / 2).toFixed(4)
    });
  }
});

// write CSV 
afterAll(() => {
  if (!rows.length) return;

  const header = 'iteration,avgPredictMs,cosineSim\n';
  const body   = rows.map(r => `${r.iter},${r.avgPredictMs},${r.cosineSim}`).join('\n');
  fs.writeFileSync(csvPath, header + body, 'utf8');

  console.log(`\n→ Prediction metrics written to ${csvPath}\n`);
});
