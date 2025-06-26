/* eslint-env jest */
/* eslint-disable no-console */
const fs         = require('fs');
const path       = require('path');
const os         = require('os');
const crypto     = require('crypto');
const { spawnSync } = require('child_process');
const SemanticFramework = require('../../SemanticFramework').default;

/* ─── 20-second timeout per test ─── */
jest.setTimeout(20_000);

/* ─── folders ─── */
const rootDir   = path.resolve(__dirname, '..');
const dataDir   = path.join(rootDir, '__test_data__', 'audio');
const resultDir = path.join(rootDir, '__result__');
const publicDir = path.join(rootDir, '__test_public__');
const tmpDir    = fs.mkdtempSync(path.join(os.tmpdir(), 'aud_acc_'));
const kbPath    = path.join(publicDir, 'kb_audio_acc.json');
const csvPath   = path.join(resultDir, 'audio_acc.csv');

/* ─── bootstrap ─── */
[publicDir, resultDir].forEach(d => fs.mkdirSync(d, { recursive: true }));
fs.writeFileSync(kbPath, '{}');

/* copy fixtures to tmp so SF can resolve tokens */
fs.readdirSync(dataDir).forEach(f =>
  fs.copyFileSync(path.join(dataDir, f), path.join(tmpDir, f))
);

/* ─── cosine helper ─── */
const cosine = (a, b) => {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
};

/* ─── ensure WAV for every fixture ─── */
function toWav(absFile) {
  if (absFile.toLowerCase().endsWith('.wav')) return absFile;
  const wavFile = absFile.replace(/\.\w+$/, '') + '.wav';
  const { status } = spawnSync(
    'ffmpeg',
    ['-nostdin', '-loglevel', 'error', '-y', '-i', absFile, wavFile]
  );
  if (status !== 0) throw new Error('ffmpeg conversion failed');
  return wavFile;
}

/* ─── run the accuracy loop ─── */
const sf = new SemanticFramework({ kbPath, imgDir: tmpDir });

const rows = [];

for (const file of fs.readdirSync(dataDir).filter(f => /\.(mp3|wav)$/i.test(f))) {
  test(`reconstruct ${file}`, async () => {
    /* token must point at the file inside tmpDir */
    const absInTmp = path.join(tmpDir, file);
    const wavPath  = toWav(absInTmp);

    const tok      = `<audio:${path.basename(wavPath)}>`;
    const origLat  = sf.encode_vec(tok);

    /* decode latent → WAV buffer */
    const wavBuf   = await sf.decode_vec(origLat);
    const reconFile = crypto.randomBytes(6).toString('hex') + '.wav';
    fs.writeFileSync(path.join(tmpDir, reconFile), wavBuf);

    const reconLat = sf.encode_vec(`<audio:${reconFile}>`);
    const acc      = cosine(origLat, reconLat);

    rows.push({ file, accuracy: acc });
  });
}

/* ─── CSV + average ─── */
afterAll(() => {
  if (!rows.length) return;
  const header = 'fileName,accuracy\n';
  const body   = rows.map(r => `${r.file},${r.accuracy.toFixed(4)}`).join('\n');
  fs.writeFileSync(csvPath, header + body, 'utf8');

  const avg = rows.reduce((s, r) => s + r.accuracy, 0) / rows.length;
  console.log(`\n→ audio_acc.csv written – avgAcc=${avg.toFixed(4)}\n`);
});
