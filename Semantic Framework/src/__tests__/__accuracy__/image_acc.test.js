/* eslint-env jest */
/* eslint-disable no-console */
const fs   = require('fs');
const path = require('path');
const os   = require('os');
const crypto = require('crypto');
const SemanticFramework = require('../../SemanticFramework').default;

/* ─── folders ─── */
const rootDir   = path.resolve(__dirname, '..');
const dataDir   = path.join(rootDir, '__test_data__', 'images');
const resultDir = path.join(rootDir, '__result__');
const publicDir = path.join(rootDir, '__test_public__');
const tmpDir    = fs.mkdtempSync(path.join(os.tmpdir(), 'img_acc_'));
const kbFile    = path.join(publicDir, 'kb_img_acc.json');
const csvPath   = path.join(resultDir, 'image_acc.csv');

/* ─── sanity ─── */
if (!fs.readdirSync(dataDir).some(f => /\.jpe?g$/i.test(f))) {
  throw new Error(`No .jpg files in ${dataDir}`);
}

/* ─── prep ─── */
beforeAll(() => {
  [publicDir, resultDir].forEach(d => fs.mkdirSync(d, { recursive: true }));
  fs.writeFileSync(kbFile, '{}');
  /* copy originals so SF can find them */
  fs.readdirSync(dataDir).forEach(f =>
    fs.copyFileSync(path.join(dataDir, f), path.join(tmpDir, f)));
});

/* ─── helpers ─── */
const cosine = (A,B) => {
  let dot=0, na=0, nb=0;
  for (let i=0;i<A.length;i++){
    dot += A[i]*B[i]; na += A[i]*A[i]; nb += B[i]*B[i];
  }
  return dot / (Math.sqrt(na)*Math.sqrt(nb));
};

/* ─── SemanticFramework ─── */
let sf;
beforeAll(() => { sf = new SemanticFramework({ kbPath: kbFile, imgDir: tmpDir }); });

/* ─── measure every JPEG ─── */
const rows = [];
for (const file of fs.readdirSync(dataDir).filter(f => /\.jpe?g$/i.test(f))) {
  test(`reconstruct ${file}`, async () => {
    const tok         = `<img:${file}>`;
    const origLat     = sf.encode_vec(tok);

    /* decode → Buffer(JPEG) */
    const reconBuf    = await sf.decode_vec(origLat);
    const reconFile   = crypto.randomBytes(6).toString('hex') + '.jpg';
    fs.writeFileSync(path.join(tmpDir, reconFile), reconBuf);

    const reconLat    = sf.encode_vec(`<img:${reconFile}>`);
    const acc         = cosine(origLat, reconLat);

    rows.push({ file, accuracy: acc });
  });
}

/* ─── CSV + average ─── */
afterAll(() => {
  if (!rows.length) return;
  const header = 'fileName,accuracy\n';
  const body   = rows.map(r => `${r.file},${r.accuracy.toFixed(4)}`).join('\n');
  fs.writeFileSync(csvPath, header + body, 'utf8');

  const avgAcc = rows.reduce((s,r)=>s+r.accuracy,0) / rows.length;
  console.log(`\n→ image_acc.csv written – avgAcc=${avgAcc.toFixed(4)}\n`);
});
