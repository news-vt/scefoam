/* eslint-env jest */
/* eslint-disable no-console */
const fs   = require('fs');
const path = require('path');
const os   = require('os');
const SemanticFramework = require('../../SemanticFramework').default;

/* ── give every test up to 2 minutes ─────────────────────────────────── */
jest.setTimeout(120_000);

/* ── folders ─────────────────────────────────────────────────────────── */
const rootDir   = path.resolve(__dirname, '..');
const dataDir   = path.join(rootDir, '__test_data__', 'text');
const resultDir = path.join(rootDir, '__result__');
const publicDir = path.join(rootDir, '__test_public__');
const tmpDir    = fs.mkdtempSync(path.join(os.tmpdir(), 'txt_cos_'));
const kbPath    = path.join(publicDir, 'kb_text_cos.json');
const csvPath   = path.join(resultDir, 'text_acc.csv');
const reconDir  = path.join(resultDir, 'recon');

[resultDir, reconDir, publicDir].forEach(d => fs.mkdirSync(d, { recursive: true }));
fs.writeFileSync(kbPath, '{}');

/* copy fixtures so that <text:FILE> works inside SemanticFramework */
fs.readdirSync(dataDir).forEach(f =>
  fs.copyFileSync(path.join(dataDir, f), path.join(tmpDir, f))
);

/* ── helper: cosine between two Float32Array buffers ─────────────────── */
function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; ++i) {
    dot += a[i] * b[i];
    na  += a[i] ** 2;
    nb  += b[i] ** 2;
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

/* ── Semantic Framework instance ─────────────────────────────────────── */
const sf = new SemanticFramework({ kbPath, imgDir: tmpDir });

/* ── run the similarity loop ─────────────────────────────────────────── */
const rows = [];

for (const file of fs.readdirSync(dataDir).filter(f => f.endsWith('.txt'))) {
  test(
    `semantic-embedding similarity for ${file}`,
    async () => {
      /* read reference text ------------------------------------------- */
      const refStr = fs.readFileSync(path.join(dataDir, file), 'utf8').trim();

      /* encode reference directly (bypasses <text:FILE>) -------------- */
      const refVec = sf.encode_vec(refStr);              // Float32Array

      /* encode → latent → decode -------------------------------------- */
      const latent = sf.encode_vec(`<text:${file}>`);
      const buf    = await sf.decode_vec(latent);
      const hypStr = buf.toString('utf8').trim();        // reconstruction

      /* encode reconstruction ----------------------------------------- */
      const hypVec = sf.encode_vec(hypStr);

      /* cosine similarity --------------------------------------------- */
      const sim = cosine(refVec, hypVec);
      const pct = (sim * 100).toFixed(2);

      fs.writeFileSync(
        path.join(reconDir, file.replace('.txt', '_recon.txt')),
        hypStr,
        'utf8'
      );

      console.log(`\n[${file}] cosine = ${pct}%`);
      console.log('  ↳ first 240 chars of HYP:\n', hypStr.slice(0, 240), '…\n');

      rows.push({ file, cosine: sim });
    },
    120_000
  );
}

/* ── CSV + average summary ───────────────────────────────────────────── */
afterAll(() => {
  if (!rows.length) return;
  const header = 'fileName,accuracy\n';
  const body   = rows.map(r => `${r.file},${r.cosine.toFixed(4)}`).join('\n');
  fs.writeFileSync(csvPath, header + body, 'utf8');

  const avg = rows.reduce((s, r) => s + r.cosine, 0) / rows.length;
  console.log(`\n→ text_cosine.csv written - avg cosine = ${(avg * 100).toFixed(2)}%`);
  console.log(`→ full reconstructions saved to: ${reconDir}\n`);
});
