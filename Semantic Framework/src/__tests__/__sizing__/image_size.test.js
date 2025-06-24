// src/__metrics__/image_size.test.js
/* eslint-disable no-console */
const fs   = require('fs');
const path = require('path');
const SemanticFramework = require('../../SemanticFramework').default;

/* ─────────────────────────── directories ────────────────────────── */
const rootDir = path.resolve(__dirname, '..');
const resultDir = path.join(rootDir, '__result__');
const publicDir = path.join(rootDir, '__test_public__');
const dataDir   = path.join(rootDir, '__test_data__', 'images');
const tmpDir    = path.join(process.cwd(), '__tmp');
const kbFile  = path.join(publicDir, 'kb_images_test.json');
const csvPath = path.join(resultDir, 'image_sizes.csv');


/* ────────────────────────── fixture sanity check ────────────────── */
if (!fs.existsSync(dataDir) || fs.readdirSync(dataDir).length === 0) {
  throw new Error(`No image fixtures found in ${dataDir}`);
}

/* ───────── copy fixtures to __tmp so SF can load them ───────────── */
beforeAll(() => {
  [tmpDir, publicDir, resultDir].forEach(d => fs.mkdirSync(d, { recursive: true }));
  for (const f of fs.readdirSync(dataDir)) {
    fs.copyFileSync(path.join(dataDir, f), path.join(tmpDir, f));
  }
  if (!fs.existsSync(kbFile)) fs.writeFileSync(kbFile, '{}');
});

/* ─────────────────────── SemanticFramework init ─────────────────── */
let sf;
beforeAll(() => { sf = new SemanticFramework({ kbPath: kbFile }); });

/* ─────────────────────────────── test logic ─────────────────────── */
const resolutions = ['4k', '1440p', '1080p', '720p'];
const exts        = ['png', 'jpg', 'webp', 'jxl', 'avif']; // now supported columns
const rows        = [];

resolutions.forEach(res => {
  test(`measure ${res}`, () => {
    const row = { resolution: res };

    exts.forEach(ext => {
      const matcher  = new RegExp(`_${res}\\.${ext}$`, 'i');
      const files    = fs.readdirSync(dataDir).filter(f => matcher.test(f));
      expect(files.length).toBeGreaterThan(0);

      const file     = files[0];
      const size     = fs.statSync(path.join(dataDir, file)).size;
      row[`${ext}Bytes`] = size;

      if (ext === 'jpg') {
        const vec        = sf.encode_vec(`<img:${file}>`);
        expect(Array.isArray(vec)).toBe(true);
        row.latentBytes  = Buffer.byteLength(JSON.stringify(vec), 'utf8');
      }
    });

    rows.push(row);
  });
});

/* ──────────────────────────── CSV export ────────────────────────── */
afterAll(() => {
  if (rows.length === 0) {
    console.warn('\n⚠️  No measurements recorded – CSV not written.\n');
    return;
  }

  /* 1️⃣  keep every key except latentBytes in original order */
  const firstRow = rows[0];
  const colNames = Object.keys(firstRow).filter(k => k !== 'latentBytes');

  /* 2️⃣  then force latentBytes to the end */
  colNames.push('latentBytes');

  const header = colNames.join(',') + '\n';

  const body = rows
    .map(r => colNames.map(c => r[c] ?? '').join(','))
    .join('\n');

  fs.writeFileSync(csvPath, header + body, 'utf8');
  console.log(`\n→ Wrote CSV to ${csvPath}\n`);
});

