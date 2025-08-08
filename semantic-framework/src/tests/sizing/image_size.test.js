// src/metrics/image_size.test.js
/* eslint-disable no-console */
const fs   = require('fs');
const path = require('path');
const { request } = require('undici');

const rootDir   = path.resolve(__dirname, '..');
const resultDir = path.join(rootDir, 'result');
const publicDir = path.join(rootDir, 'test_public');
const dataDir   = path.join(rootDir, 'test_data', 'images');
const csvPath   = path.join(resultDir, 'image_sizes.csv');

if (!fs.existsSync(dataDir) || fs.readdirSync(dataDir).length === 0) {
  throw new Error(`No image fixtures found in ${dataDir}`);
}

const IMAGE_BASE = process.env.IMAGE_BASE || 'http://127.0.0.1:8082';
const DS         = parseInt(process.env.IMG_DS || '64', 10);

beforeAll(() => {
  [publicDir, resultDir].forEach(d => fs.mkdirSync(d, { recursive: true }));
});

async function httpEncodeBytesImage(absPath) {
  const buf = fs.readFileSync(absPath);
  const { statusCode, body } = await request(`${IMAGE_BASE}/encode?ds=${DS}`, {
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

const resolutions = ['4k', '1440p', '1080p', '720p'];
const exts        = ['png', 'jpg', 'webp', 'jxl', 'avif'];
const rows        = [];

resolutions.forEach(res => {
  test(`measure ${res}`, async () => {
    const row = { resolution: res };

    let latentMeasured = false;
    for (const ext of exts) {
      const matcher = new RegExp(`_${res}\\.${ext}$`, 'i');
      const files   = fs.readdirSync(dataDir).filter(f => matcher.test(f));
      expect(files.length).toBeGreaterThan(0);

      const file = path.join(dataDir, files[0]);
      row[ext] = fs.statSync(file).size; // remove "Bytes" from key

      if (!latentMeasured && ext === 'jpg') {
        row.latent = await httpEncodeBytesImage(file); // remove "Bytes"
        latentMeasured = true;
      }
    }

    rows.push(row);
  }, 30000);
});

afterAll(() => {
  if (!rows.length) {
    console.warn('\n⚠️  No measurements recorded – CSV not written.\n');
    return;
  }

  const firstRow = rows[0];
  const colNames = Object.keys(firstRow).filter(k => k !== 'latent');
  colNames.push('latent');

  const header = colNames.join(',') + '\n';
  const body   = rows.map(r => colNames.map(c => r[c] ?? '').join(',')).join('\n');

  fs.writeFileSync(csvPath, header + body, 'utf8');
  console.log(`\n→ Wrote CSV to ${csvPath}\n`);
});
