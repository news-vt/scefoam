// src/__metrics__/text_size.test.js
/* eslint-disable no-console */
const fs   = require('fs');
const path = require('path');
const SemanticFramework = require('../../SemanticFramework').default;

/* ─────────────── paths ─────────────── */
const rootDir  = path.resolve(__dirname, '..');
const dataDir   = path.join(rootDir, '__test_data__', 'text');
const tmpDir    = path.join(process.cwd(), '__tmp');     
const resultDir = path.join(rootDir, '__result__');
const csvPath   = path.join(resultDir, 'text_sizes.csv');
const publicDir = path.join(rootDir, '__test_public__');
const kbFile    = path.join(publicDir, 'kb_text_test.json');

/* ─── ensure fixtures exist ─── */
if (!fs.existsSync(dataDir) || !fs.readdirSync(dataDir).some(f => f.endsWith('.txt'))) {
  throw new Error(`No .txt fixtures in ${dataDir}`);
}

/* ─── prep folders & copy files ─── */
beforeAll(() => {
  [tmpDir, publicDir, resultDir].forEach(d => fs.mkdirSync(d, { recursive: true }));
  fs.readdirSync(dataDir)
    .filter(f => f.endsWith('.txt'))
    .forEach(f => fs.copyFileSync(path.join(dataDir, f), path.join(tmpDir, f)));

  if (!fs.existsSync(kbFile)) fs.writeFileSync(kbFile, '{}');
});

/* ─── framework ─── */
let sf;
beforeAll(() => { sf = new SemanticFramework({ kbPath: kbFile }); });

/* ─── measure each file ─── */
const rows = [];
for (const fileName of fs.readdirSync(dataDir).filter(f => f.endsWith('.txt'))) {
  test(`measure ${fileName}`, () => {
    const txtBytes = fs.statSync(path.join(dataDir, fileName)).size;

    const vec = sf.encode_vec(`<text:${fileName}>`);
    expect(Array.isArray(vec)).toBe(true);

    const latentBytes = Buffer.from(Float32Array.from(vec).buffer).length;
    rows.push({ fileName, txtBytes, latentBytes });

  });
}

/* ─── write CSV ─── */
afterAll(() => {
  if (!rows.length) return;

  const header = 'fileName,txtBytes,latentBytes\n';
  const body   = rows.map(({fileName,txtBytes,latentBytes}) =>
                  `${fileName},${txtBytes},${latentBytes}`).join('\n');

  fs.writeFileSync(csvPath, header + body, 'utf8');
  console.log(`\n→ Wrote CSV to ${csvPath}\n`);
});
