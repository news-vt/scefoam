/* eslint-env jest */

const fs   = require('fs');
const path = require('path');

const SemanticFramework    = require('../SemanticFramework').default;
const PredictionController = require('../prediction_controller');

/* plenty of time for I/O + model warm-up */
jest.setTimeout(600_000);

/* paths */
const publicDir = path.join(__dirname, '__test_public');
const dataDir   = path.join(__dirname, '__test_data');
const tmpDir    = path.join(process.cwd(), '__tmp_pred');

const files = {
  text : 'test_text_1.txt',
  audio: 'test_audio_1.mp3',
  img  : 'test_image_1.jpg'
};

const teacherKB    = path.join(publicDir, 'kb_teacher_pred.json');
const apprenticeKB = path.join(publicDir, 'kb_apprentice_pred.json');

/* globals */
let teacher, apprentice;

/* ------------------------------------------------------------------ */
beforeAll(() => {
  for (const d of [publicDir, tmpDir]) fs.mkdirSync(d, { recursive: true });

  const blank = JSON.stringify(
    { map:{}, receivedData:[], hexHistory:[], predictedText:'', modelReady:false },
    null, 2
  );
  fs.writeFileSync(teacherKB,    blank);
  fs.writeFileSync(apprenticeKB, blank);

  for (const fn of Object.values(files)) {
    fs.copyFileSync(path.join(dataDir, fn), path.join(tmpDir, fn));
  }

  teacher = new SemanticFramework({ kbPath: teacherKB, imgDir: tmpDir });

  apprentice = new SemanticFramework({
    kbPath : apprenticeKB,
    imgDir : tmpDir,
    model  : new PredictionController({ kbPath: apprenticeKB })
  });
});

afterAll(() => {
  apprentice.model.close();
});

/* helpers ----------------------------------------------------------- */
function saveRecon(kind, srcName, decoded) {
  let buf, out;
  if (kind === 'text') {
    buf = Buffer.from(String(decoded), 'utf8');
    out = srcName;
  } else if (kind === 'audio') {
    buf = decoded;                              // WAV
    out = srcName.replace(/\.mp3$/i, '.wav');
  } else {                                      // image
    buf = decoded;                              // JPEG
    out = srcName;
  }
  fs.writeFileSync(path.join(publicDir, `reconstructed_${out}`), buf);
}

/* helper: try to decode; if it fails, save the latent JSON instead ---- */
async function savePrediction(kind, latent) {
  let okBuf     = null;                 // Buffer or string when decode works
  let outName   = '';

  try {
    if (kind === 'text') {
      outName = 'predicted_text.txt';

      const dec = (await apprentice.receive([['p', latent, 'text']]))[0];
      if (dec) okBuf = Buffer.from(String(dec), 'utf8');

    } else if (kind === 'image') {      // <- tag must be "image"
      outName = 'predicted_image.jpg';

      // prepend [W,H] so VAE knows the size
      const ready = [512, 512, ...latent];
      const dec   = (await apprentice.receive([['p', ready, 'image']]))[0];
      if (dec) okBuf = dec;             // Buffer

    } else if (kind === 'audio') {
      outName = 'predicted_audio.wav';

      const sr = 24_000, nq = 8;
      const codesInt = latent.map(v => Math.max(0, Math.min(1023, v | 0)));
      const frames   = Math.floor(codesInt.length / nq);
      if (frames > 0) {
        const ready = [sr, nq, frames, ...codesInt.slice(0, nq * frames)];
        const dec   = (await apprentice.receive([['p', ready, 'audio']]))[0];
        if (dec) okBuf = dec;           // Buffer
      }
    }
  } catch { /* decoder failed — fall back below */ }

  /* Decoder failed or returned nothing — persist latent for inspection */
  if (!okBuf) {
    outName = `predicted_${kind}_latent.json`;
    okBuf   = Buffer.from(JSON.stringify(latent, null, 2), 'utf8');
  }

  fs.writeFileSync(path.join(publicDir, outName), okBuf);
}

/* ------------------------------------------------------------------ */
test('teacher → apprentice; decode predictions when possible', async () => {
  const tokens = [
    `<text:${files.text}>`,
    `<audio:${files.audio}>`,
    `<img:${files.img}>`
  ];
  const kinds = ['text', 'audio', 'image'];   // <- use "image"

  /* original round-trip ------------------------------------------------ */
  const outs1 = await apprentice.receive(teacher.send(tokens));
  outs1.forEach((dec, i) =>
    saveRecon(kinds[i], files[kinds[i] === 'image' ? 'img' : kinds[i]], dec)
  );

  /* predictions ------------------------------------------------------- */
  for (let i = 0; i < tokens.length; ++i) {
    const predLatent = await apprentice.predict(tokens[i], kinds[i]);
    await savePrediction(kinds[i], predLatent);
  }

  /* basic sanity: hex-only still decodes ------------------------------ */
  const outs2 = await apprentice.receive(teacher.send(tokens));
  expect(outs2).toHaveLength(tokens.length);
});
