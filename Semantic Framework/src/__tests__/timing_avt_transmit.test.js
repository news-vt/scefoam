/* eslint-env jest */

const fs            = require('fs');
const path          = require('path');
const SemanticFramework = require('../SemanticFramework').default;

/* ─── plenty of time for audio decode / I-O ──────────────────────────── */
jest.setTimeout(600_000);

/* ─── fixtures & scratch paths ───────────────────────────────────────── */
const publicDir = path.join(__dirname, '__test_public');
const dataDir   = path.join(__dirname, '__test_data');
const tmpDir    = path.join(process.cwd(), '__tmp');

const files = {
  text : 'test_text_1.txt',
  audio: 'test_audio_1.mp3',
  img  : 'test_image_1.jpg'
};

const teacherKB    = path.join(publicDir, 'kb_teacher_avt.json');
const apprenticeKB = path.join(publicDir, 'kb_apprentice_avt.json');

/* ─── global setup ───────────────────────────────────────────────────── */
let teacher, apprentice;
beforeAll(() => {
  for (const d of [publicDir, tmpDir]) fs.mkdirSync(d, { recursive: true });

  const blankKB = JSON.stringify(
    { map:{}, receivedData:[], hexHistory:[], predictedText:'', modelReady:false },
    null, 2
  );
  fs.writeFileSync(teacherKB,    blankKB);
  fs.writeFileSync(apprenticeKB, blankKB);

  /* copy fixtures to tmp so the originals stay pristine */
  for (const fn of Object.values(files)) {
    fs.copyFileSync(path.join(dataDir, fn), path.join(tmpDir, fn));
  }

  teacher = new SemanticFramework({
    kbPath : teacherKB,
    imgDir : tmpDir
  });

  apprentice = new SemanticFramework({
    kbPath : apprenticeKB,
    imgDir : tmpDir
  });
});

/* ─── helper: persist reconstructions so you can eyeball them later ──── */
function saveRecon(kind, srcName, decoded) {
  let buf, outName;

  if (kind === 'text') {
    const txt = Array.isArray(decoded) ? decoded.join(' ') : String(decoded);
    buf      = Buffer.from(txt, 'utf8');
    outName  = srcName;
  } else if (kind === 'audio') {
    buf      = decoded;                             // WAV from AudioCodec
    outName  = srcName.replace(/\.mp3$/i, '.wav');
  } else {                                          // image
    buf      = decoded;                             // JPEG
    outName  = srcName;
  }

  const outPath = path.join(publicDir, `reconstructed_${outName}`);
  fs.writeFileSync(outPath, buf);
  expect(fs.existsSync(outPath)).toBe(true);
}

/* ─── the actual round-trip test ─────────────────────────────────────── */
test('teacher → apprentice, twice (first with latents, then hex-only)', async () => {
  const tokens = [
    `<text:${files.text}>`,
    `<audio:${files.audio}>`,
    `<img:${files.img}>`
  ];

  /* ── 1st transmission: teacher sends [hex, latent] ─────────────────── */
  console.time('send-1');
  const payload1 = teacher.send(tokens);            // [[hex, latent, mod], …]
  console.timeEnd('send-1');

  expect(payload1.length).toBe(tokens.length);
  /* sanity: each triple should be at least length-2 */
  payload1.forEach(item => expect(item.length).toBeGreaterThan(1));

  console.time('receive-1');
  const outs1 = await apprentice.receive(payload1); // [decoded,*3]
  console.timeEnd('receive-1');
  expect(outs1.length).toBe(tokens.length);

  /* persist the reconstructions */
  outs1.forEach((dec, idx) => {
    const kind = ['text','audio','img'][idx];
    saveRecon(kind, files[kind], dec);
  });

  /* ── 2nd transmission: teacher now believes apprentice has the latents */
  console.time('send-2');
  const payload2 = teacher.send(tokens);            // [[hex] …]  (hex-only)
  console.timeEnd('send-2');

  /* We expect the payload to be shorter (no latent blobs) */
  expect(payload2.length).toBe(tokens.length);
  payload2.forEach(item => expect(item.length).toBe(1)); // hex-only

  /* size check: serialized payload2 should be much smaller */
  const bytes1 = Buffer.byteLength(JSON.stringify(payload1), 'utf8');
  const bytes2 = Buffer.byteLength(JSON.stringify(payload2), 'utf8');
  console.log(`payload sizes: first=${bytes1} B, second=${bytes2} B`);
  expect(bytes2).toBeLessThan(bytes1 * 0.2);        // heuristic gate

  /* Apprentice receives hex-only references and fetches latents from its KB */
  console.time('receive-2');
  const outs2 = await apprentice.receive(payload2); // [decoded,*3]  (instant)
  console.timeEnd('receive-2');
  expect(outs2.length).toBe(tokens.length);

  // /* outputs from 1st and 2nd pass must match byte-for-byte */
  // outs1.forEach((buf1, idx) => {
  //   const buf2 = outs2[idx];
  //   if (Buffer.isBuffer(buf1) && Buffer.isBuffer(buf2)) {
  //     expect(buf2.equals(buf1)).toBe(true);
  //   } else {
  //     expect(buf2).toEqual(buf1);
  //   }
  // });

  // /* ── KB sanity on both sides ───────────────────────────────────────── */
  // tokens.forEach(tok => {
  //   const vec   = teacher.vectorize(tok);
  //   const hex   = teacher.findClosestHexByVector(vec, 0.9999);
  //   const hexAp = apprentice.findClosestHexByVector(vec, 0.9999);
  //   expect(hex).toBe(hexAp);
  // });
});
