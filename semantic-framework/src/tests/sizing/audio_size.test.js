// src/tests/audio_size.test.js
/* eslint-disable no-console */
const fs   = require('fs');
const path = require('path');
const { request } = require('undici');

const rootDir   = path.resolve(__dirname, '..');
const dataDir   = path.join(rootDir, 'test_data', 'audio');
const resultDir = path.join(rootDir, 'result');
const publicDir = path.join(rootDir, 'test_public');
const csvPath   = path.join(resultDir, 'audio_sizes.csv');

const AUDIO_BASE = process.env.AUDIO_BASE || 'http://127.0.0.1:8081';

beforeAll(() => {
  [publicDir, resultDir].forEach(d => fs.mkdirSync(d, { recursive: true }));
});

async function httpEncodeBytesAudio(absPath) {
  const buf = fs.readFileSync(absPath);
  const { statusCode, body } = await request(`${AUDIO_BASE}/encode`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/octet-stream' },
    body: buf,
  });
  expect(statusCode).toBe(200);
  const chunks = [];
  for await (const c of body) chunks.push(c);
  const raw = Buffer.concat(chunks); // exact bytes received
  JSON.parse(raw.toString('utf8'));
  return raw.length;
}

const exts  = ['wav', 'mp3', 'ogg', 'opus', 'flac'];
const bases = ['test_audio_1', 'test_audio_2', 'test_audio_3'];
const rows  = [];

bases.forEach(base => {
  exts.forEach(ext => {
    const file = `${base}.${ext}`;
    const fp   = path.join(dataDir, file);

    test(`measure ${file}`, async () => {
      const size = fs.statSync(fp).size;

      let latent = '';
      if (ext === 'wav') {
        latent = await httpEncodeBytesAudio(fp);
        expect(latent).toBeGreaterThan(0);
        expect(latent).toBeLessThan(size);
      } else {
        expect(size).toBeGreaterThan(0);
      }

      rows.push({ baseName: base, [ext]: size, ...(latent !== '' ? { latent } : {}) });
    }, 30000);
  });
});

afterAll(() => {
  if (!rows.length) return;

  const grouped = {};
  rows.forEach(r => {
    grouped[r.baseName] ??= { baseName: r.baseName };
    Object.entries(r).forEach(([k, v]) => {
      if (k !== 'baseName') grouped[r.baseName][k] = v;
    });
  });

  const colOrder = ['wav','mp3','ogg','opus','flac','latent'];
  const header   = ['baseName', ...colOrder].join(',') + '\n';

  const body = Object.values(grouped)
    .map(r => ['baseName', ...colOrder].map(c => r[c] ?? '').join(','))
    .join('\n');

  fs.writeFileSync(csvPath, header + body, 'utf8');
  console.log(`\n→ Wrote CSV to ${csvPath}\n`);
});
