/* eslint-env jest */
/* eslint-disable no-console */
const fs            = require('fs');
const path          = require('path');
const os            = require('os');
const { spawnSync } = require('child_process');
const SemanticFramework = require('../../semantic_framework').default;

jest.setTimeout(120_000);

const rootDir      = path.resolve(__dirname, '..');
const dataDir      = path.join(rootDir, 'test_data', 'text');
const resultDir    = path.join(rootDir, 'result');
const publicDir    = path.join(rootDir, 'test_public');
const tmpDir       = fs.mkdtempSync(path.join(os.tmpdir(), 'txt_bert_'));
const kbPath       = path.join(publicDir, 'kb_text_acc.json');
const csvPath      = path.join(resultDir, 'text_acc.csv');
const reconDir     = path.join(resultDir, 'recon');
const reconOutDir  = path.join(resultDir, 'acc_reconstruct');

[ resultDir, reconDir, publicDir, reconOutDir ].forEach(d => fs.mkdirSync(d, { recursive: true }));
fs.writeFileSync(kbPath, '{}');

fs.readdirSync(dataDir).forEach(f =>
  fs.copyFileSync(path.join(dataDir, f), path.join(tmpDir, f))
);

function bertscore(refPath, hypPath) {
  const out = spawnSync(
    'python',
    [ path.join(__dirname, 'bertscore_cli.py'), refPath, hypPath ],
    { encoding: 'utf8' }
  );
  if (out.status !== 0) throw new Error(out.stderr);
  return parseFloat(out.stdout.trim());
}

const sf   = new SemanticFramework({ kbPath, imgDir: tmpDir });
const rows = [];

for (const file of fs.readdirSync(dataDir).filter(f => f.endsWith('.txt'))) {
  test(`BERTScore F1 for ${file}`, async () => {
    const refPath = path.join(tmpDir, file);

    // reconstruct text
    const latent   = sf.encode_vec(`<text:${file}>`);
    const buf      = await sf.decode_vec(latent);
    const hypStr   = buf.toString('utf8').trim();

    // name with _reconstruction suffix
    const ext      = path.extname(file);               // ".txt"
    const base     = path.basename(file, ext);         // e.g. "words_100"
    const reconFile= `${base}_recon${ext}`;   // "words_100_reconstruction.txt"

    // save for any downstream tooling
    const hypPath  = path.join(reconDir, reconFile);
    fs.writeFileSync(hypPath, hypStr, 'utf8');

    // also dump into acc_reconstruct for manual review
    fs.copyFileSync(hypPath, path.join(reconOutDir, reconFile));

    // score!
    const score = bertscore(refPath, hypPath);
    rows.push({ file, accuracy: score });
  });
}

afterAll(() => {
  if (!rows.length) return;
  const header = 'fileName,accuracy\n';
  const body   = rows.map(r => `${r.file},${r.accuracy.toFixed(4)}`).join('\n');
  fs.writeFileSync(csvPath, header + body, 'utf8');

  const avg = rows.reduce((sum, r) => sum + r.accuracy, 0) / rows.length;
  console.log(`\n→ text_acc.csv written – avg BERTScore F1 = ${avg.toFixed(4)}\n`);
});
