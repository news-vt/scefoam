/* eslint-env jest */
/* eslint-disable no-console */
const fs         = require('fs');
const path       = require('path');
const os         = require('os');
const crypto     = require('crypto');
const SemanticFramework = require('../../SemanticFramework').default;

/* ── give every test up to 2 minutes ──────────────────────────────────── */
jest.setTimeout(120_000);

/* ── folders ─────────────────────────────────────────────────────────── */
const rootDir   = path.resolve(__dirname, '..');
const dataDir   = path.join(rootDir, '__test_data__', 'text');
const resultDir = path.join(rootDir, '__result__');
const publicDir = path.join(rootDir, '__test_public__');
const tmpDir    = fs.mkdtempSync(path.join(os.tmpdir(), 'txt_acc_'));
const kbPath    = path.join(publicDir, 'kb_text_acc.json');
const csvPath   = path.join(resultDir, 'text_acc.csv');

/* ── bootstrap ───────────────────────────────────────────────────────── */
[publicDir, resultDir].forEach(d => fs.mkdirSync(d, { recursive: true }));
fs.writeFileSync(kbPath, '{}');

/* copy fixtures to tmp so <text:FILE> resolves */
fs.readdirSync(dataDir).forEach(f =>
  fs.copyFileSync(path.join(dataDir, f), path.join(tmpDir, f))
);

/* ── cosine helper ───────────────────────────────────────────────────── */
const cosine = (A, B) => {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < A.length; i++) {
    dot += A[i] * B[i];  na += A[i] * A[i];  nb += B[i] * B[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
};

/* ── Semantic Framework instance ─────────────────────────────────────── */
const sf = new SemanticFramework({ kbPath, imgDir: tmpDir });

/* ── run the accuracy loop ───────────────────────────────────────────── */
const rows = [];

for (const file of fs.readdirSync(dataDir).filter(f => f.endsWith('.txt'))) {
  test(
    `reconstruct ${file}`,
    async () => {
      /* original latent ------------------------------------------------- */
      const tok      = `<text:${file}>`;
      const origLat  = sf.encode_vec(tok);

      /* decode latent → Buffer(utf-8) ----------------------------------- */
      const buf      = await sf.decode_vec(origLat);
      const reconStr = buf.toString('utf8').trim();

      /* re-encode ------------------------------------------------------- */
      const reconLat = sf.encode_vec(reconStr);

      const acc      = cosine(origLat, reconLat);
      rows.push({ file, accuracy: acc });
    },
    120_000  // per-test timeout
  );
}

/* ── CSV + average summary ───────────────────────────────────────────── */
afterAll(() => {
  if (!rows.length) return;
  const header = 'fileName,accuracy\n';
  const body   = rows.map(r => `${r.file},${r.accuracy.toFixed(4)}`).join('\n');
  fs.writeFileSync(csvPath, header + body, 'utf8');

  const avg = rows.reduce((s, r) => s + r.accuracy, 0) / rows.length;
  console.log(`\n→ text_acc.csv written – avgAcc=${avg.toFixed(4)}\n`);
});
