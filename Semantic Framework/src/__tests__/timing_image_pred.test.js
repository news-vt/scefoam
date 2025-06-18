/* eslint-env jest */
/**
 * Bidirectional image-pair fine-tuning test
 *
 * For ITERATIONS loops we alternate:
 *   odd  i ⇒ teacher sends [img1,img2], predict from img1
 *   even i ⇒ teacher sends [img2,img1], predict from img2
 *
 * Predicted JPEGs are written to
 *   __tests__/__test_public/predicted_image_iter<i>_<dir>.jpg
 */

const fs   = require('fs');
const path = require('path');
const SemanticFramework = require('../SemanticFramework').default;

/* ─── config ─── */
jest.setTimeout(900_000);
const ITERATIONS = 2;                 // change as you like

/* ─── paths ─── */
const testRoot  = __dirname;
const dataDir   = path.join(testRoot, '__test_data');
const publicDir = path.join(testRoot, '__test_public');
const tmpDir    = path.join(process.cwd(), '__tmp_pred');

const files = { img1: 'test_image_1.jpg', img2: 'test_image_2.jpg' };
const teacherKB    = path.join(publicDir, 'kb_teacher_img.json');
const apprenticeKB = path.join(publicDir, 'kb_apprentice_img.json');

/* ─── helpers ─── */
const ensureDir = d => fs.mkdirSync(d, { recursive: true });
const blankKB = fp => fs.writeFileSync(
  fp,
  JSON.stringify(
    { map:{}, receivedData:[], hexHistory:[], predictedText:'', modelReady:false },
    null, 2
  )
);
const save = (name, payload) => {
  const out = path.join(publicDir, name);
  fs.writeFileSync(
    out,
    Buffer.isBuffer(payload) ? payload :
    typeof payload === 'string' ? payload :
    JSON.stringify(payload, null, 2)
  );
};

/* ─── global frameworks ─── */
let teacher, apprentice;

/* ------------------------------------------------------------------ */
beforeAll(() => {
  ensureDir(publicDir); ensureDir(tmpDir);
  blankKB(teacherKB);    blankKB(apprenticeKB);

  /* copy fixture JPEGs into tmp */
  Object.values(files).forEach(fn =>
    fs.copyFileSync(path.join(dataDir, fn), path.join(tmpDir, fn))
  );

  teacher    = new SemanticFramework({ kbPath: teacherKB,    imgDir: tmpDir });
  apprentice = new SemanticFramework({ kbPath: apprenticeKB, imgDir: tmpDir });
});

/* ------------------------------------------------------------------ */
test(`cycle img1→img2 and img2→img1 for ${ITERATIONS} iterations`, async () => {

  for (let i = 1; i <= ITERATIONS; ++i) {
    const forward = i % 2 === 1;                       // odd = 1→2, even = 2→1
    const dirTag  = forward ? '1to2' : '2to1';
    const imgTokens = forward
      ? [`<img:${files.img1}>`, `<img:${files.img2}>`]
      : [`<img:${files.img2}>`, `<img:${files.img1}>`];

    /* ① teacher sends */
    const packets = teacher.send(imgTokens);
    expect(packets.length).toBe(2);

    /* ② apprentice receives (trains) */
    const decs = await apprentice.receive(packets);
    expect(decs.length).toBe(2);

    /* save reconstructions first time we see this direction */
    if (i === 1) {
      decs.forEach((buf, idx) =>
        save(`reconstructed_image_${idx + 1}_${dirTag}.jpg`, buf)
      );
    }

    /* small pause for any async work (affects transformer version) */
    await new Promise(r => setTimeout(r, 300));

    /* ③ predict next latent from first token */
    const pred = await apprentice.predict(imgTokens[0], 'image', { wantJPEG:true });

    /* ④ persist artefacts */
    save(`predicted_image_iter${i}_${dirTag}.jpg`, pred.jpeg);
    // save(`predicted_image_latent_iter${i}_${dirTag}.json`, pred.latent);

    /* quick sanity */
    expect(Array.isArray(pred.latent)).toBe(true);
    expect(Buffer.isBuffer(pred.jpeg)).toBe(true);
  }

  console.log(`📂  Outputs written to: ${publicDir}`);
});
