/* eslint-env jest */
/**
 * Bidirectional text-pair fine-tuning test
 *
 * odd  i ⇒ teacher sends [text1 , text2]  → predict from text1
 * even i ⇒ teacher sends [text2 , text1]  → predict from text2
 *
 * Predicted sentences are saved as .txt files so you can read how the
 * Fast-Transformer evolves over time.
 */

const fs   = require('fs');
const path = require('path');
const SemanticFramework = require('../SemanticFramework').default;

/* ─── config ─── */
jest.setTimeout(600_000);
const ITERATIONS = 2;

/* ─── paths ─── */
const testRoot  = __dirname;
const dataDir   = path.join(testRoot, '__test_data');
const publicDir = path.join(testRoot, '__test_public');
const tmpDir    = path.join(process.cwd(), '__tmp_pred');

const files = { text1: 'test_text_1.txt', text2: 'test_text_2.txt' };
const teacherKB    = path.join(publicDir, 'kb_teacher_text.json');
const apprenticeKB = path.join(publicDir, 'kb_apprentice_text.json');

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
  fs.writeFileSync(out, Buffer.isBuffer(payload) ? payload : String(payload));
};

/* ─── frameworks ─── */
let teacher, apprentice;

beforeAll(() => {
  ensureDir(publicDir); ensureDir(tmpDir);
  blankKB(teacherKB);   blankKB(apprenticeKB);

  /* copy fixture TXT files into tmp */
  Object.values(files).forEach(fn =>
    fs.copyFileSync(path.join(dataDir, fn), path.join(tmpDir, fn))
  );

  teacher    = new SemanticFramework({ kbPath: teacherKB,    imgDir: tmpDir });
  apprentice = new SemanticFramework({ kbPath: apprenticeKB, imgDir: tmpDir });
});

/* ─── test ─── */
test(`cycle text1→text2 and text2→text1 for ${ITERATIONS} iterations`, async () => {

  for (let i = 1; i <= ITERATIONS; ++i) {
    const forward = i % 2 === 1;                     // odd = 1→2
    const dirTag  = forward ? '1to2' : '2to1';
    const tokens  = forward
      ? [`<text:${files.text1}>`, `<text:${files.text2}>`]
      : [`<text:${files.text2}>`, `<text:${files.text1}>`];

    /* ① teacher sends both sentences */
    const pkts = teacher.send(tokens);
    expect(pkts.length).toBe(2);

    /* ② apprentice receives & decodes (trains async) */
    const outs = await apprentice.receive(pkts);
    expect(outs.length).toBe(2);

    if (i === 1) {
      outs.forEach((buf, idx) =>
        save(`reconstructed_text_${idx + 1}.txt`, buf.toString('utf8'))
      );
    }

    /* wait a bit so the 5-epoch Fast-Transformer fine-tune finishes */
    await new Promise(r => setTimeout(r, 400));

    /* ③ predict next sentence from first token */
    const { latent, text } = await apprentice.predict(
      tokens[0],
      'text',
      { wantText: true }
    );

    /* ④ persist artefacts */
    save(`predicted_text_iter${i}_${dirTag}.txt`, text);
    // save latent if you want: save(`predicted_text_latent_iter${i}_${dirTag}.json`, JSON.stringify(latent));

    /* sanity */
    expect(Array.isArray(latent)).toBe(true);
    expect(typeof text).toBe('string');
    expect(text.length).toBeGreaterThan(0);
  }

  console.log(`📂  Text outputs saved to: ${publicDir}`);
});
