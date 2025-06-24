// src/__metrics__/audio_size.test.js
/* eslint-disable no-console */
const fs   = require('fs');
const path = require('path');
const SemanticFramework = require('../../SemanticFramework').default;

/* ─────────── directories ─────────── */
const rootDir = path.resolve(__dirname, '..');
const dataDir   = path.join(rootDir, '__test_data__', 'audio');
const resultDir = path.join(rootDir, '__result__');
const publicDir = path.join(rootDir, '__test_public__');
const tmpDir    = path.join(process.cwd(), '__tmp');
const kbFile  = path.join(publicDir, 'kb_audio_test.json');
const csvPath = path.join(resultDir, 'audio_sizes.csv');


/* ─────────── setup folders & KB ───── */
beforeAll(() => {
  [publicDir, resultDir].forEach(d => fs.mkdirSync(d, { recursive: true }));
  if (!fs.existsSync(kbFile)) fs.writeFileSync(kbFile, '{}');
});

/* ─────────── SemanticFramework ────── */
let sf;
beforeAll(() => { sf = new SemanticFramework({ kbPath: kbFile }); });

/* formats we track  */
const exts = ['wav', 'mp3', 'ogg', 'opus', 'flac'];   // ⇦ new
const bases = ['test_audio_1'];       // add more if you have them
const rows = [];

/* ─────────── test loop ────────────── */
bases.forEach(base => {
  exts.forEach(ext => {
    const file = `${base}.${ext}`;
    const fp   = path.join(dataDir, file);

    test(`measure ${file}`, () => {
      const size = fs.statSync(fp).size;
      let latent = '';

      if (ext === 'mp3') {                     // framework encodes MP3
        const vec = sf.encode_vec(`<audio:${file}>`);
        expect(Array.isArray(vec)).toBe(true);
        latent = Buffer.byteLength(JSON.stringify(vec), 'utf8');
        expect(latent).toBeLessThan(size);
      } else {                                // other formats just size-check
        expect(size).toBeGreaterThan(0);
      }

      rows.push({ base, ext, size, latent });
    });
  });
});

/* ─────────── CSV export ───────────── */
afterAll(() => {
  if (rows.length === 0) return;

  // pivot: one row per base
  const grouped = {};
  rows.forEach(({ base, ext, size, latent }) => {
    grouped[base] ??= { base };
    grouped[base][`${ext}Bytes`] = size;
    if (latent) grouped[base].latentBytes = latent;
  });

  const colOrder = ['wavBytes','mp3Bytes','oggBytes','opusBytes','flacBytes','latentBytes'];
  const header = ['baseName', ...colOrder].join(',') + '\n';

  const body = Object.values(grouped)
    .map(r => ['base', ...colOrder].map(c => r[c] ?? '').join(','))
    .join('\n');

  fs.writeFileSync(csvPath, header + body, 'utf8');
  console.log(`\n→ Wrote CSV to ${csvPath}\n`);
});
