// timing_avt.test.js
const fs   = require('fs');
const path = require('path');
const SemanticFramework = require('../SemanticFramework').default;

describe('SemanticFramework Multimodal (Audio • Vision • Text)', () => {

  /* ───────────────────────────── paths & fixtures ───────────────────────────── */
  const publicDir = path.join(__dirname, '__test_public');
  const dataDir   = path.join(__dirname, '__test_data');
  const tmpDir    = path.join(process.cwd(), '__tmp');
  const kbFile    = path.join(publicDir, 'knowledge_base_avt_test.json');

  /* sample inputs (one per modality is enough) */
  const files = {
    text : 'test_text_1.txt',
    audio: 'test_audio_1.mp3',
    img  : 'test_image_1.jpg',
  };

  /* stash temp copies so the tests never touch the originals in git */
  beforeAll(() => {
    for (const d of [publicDir, tmpDir]) fs.mkdirSync(d, { recursive: true });
    if (!fs.existsSync(kbFile)) {
      fs.writeFileSync(kbFile, JSON.stringify(
        { map: {}, receivedData: [], hexHistory: [], predictedText: '', modelReady: false },
        null, 2
      ));
    }
    /* copy fixtures */
    for (const [type, fn] of Object.entries(files)) {
      fs.copyFileSync(
        path.join(dataDir, fn),
        path.join(tmpDir, fn)
      );
    }
  });

  /* single SF instance for all three modalities */
  let sf;
  beforeAll(() => {
    sf = new SemanticFramework({ kbPath: kbFile });
  });

  /* ───────────────────────────────── tests ─────────────────────────────────── */

  test('embed each modality', () => {
    for (const [kind, fn] of Object.entries(files)) {
      const tag = kind === 'img' ? 'img' : kind;        // audio|text stay the same
      console.time(`embed-${kind}`);
      const vec = sf.vectorize(`<${tag}:${fn}>`);
      console.timeEnd(`embed-${kind}`);
      expect(Array.isArray(vec)).toBe(true);
      // cache in KB for later retrieval test
      sf._register([vec]);
    }
  });

  test('reconstruct each modality & save', async () => {
    for (const [kind, fn] of Object.entries(files)) {
      const tag = kind === 'img' ? 'img' : kind;
      const vec = sf.vectorize(`<${tag}:${fn}>`);

      console.time(`recon-${kind}`);
      const buf = await sf.decodeAsync(vec);
      console.timeEnd(`recon-${kind}`);

      /* pick sensible output extension per type */
      const outName = (() => {
        if (kind === 'audio') return fn.replace(/\.mp3$/i, '.wav');
        return fn;                                       // text & img keep same
      })();
      const outPath = path.join(publicDir, `reconstructed_${outName}`);
      fs.writeFileSync(outPath, buf);
      expect(fs.existsSync(outPath)).toBe(true);
    }
  }, 300_000);       // generous timeout (5 min) for slow CI boxes

  test('KB lookup across modalities', () => {
    /* take text vector, ask KB for closest, expect itself back quickly */
    const textVec = sf.vectorize(`<text:${files.text}>`);
    console.time('sift-kb-multimodal');
    const nearest = sf.findClosestHexByVector(textVec);
    console.timeEnd('sift-kb-multimodal');
    expect(nearest).toBeDefined();
  });

});
