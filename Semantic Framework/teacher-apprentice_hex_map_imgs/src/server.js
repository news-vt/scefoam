// server.js

/* eslint-disable no-console */
// ──────────────────────────────────────────────────────────────
// Express backend – latent‐only image transport, with auto‐created KB file
// ──────────────────────────────────────────────────────────────
import express           from 'express';
import cors              from 'cors';
import fs                from 'fs';
import path              from 'path';
import { fileURLToPath } from 'url';
import { spawnSync }     from 'child_process';
const PY = process.env.PYTHON || 'python';

import { SemanticFramework } from './SemanticFramework.js';
import { TransformerModel }  from './model.js';
import ImageCodec            from './image_embedding.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

/* ───────── CLI port ───────── */
const idx  = process.argv.indexOf('--port');
const PORT = idx !== -1 ? Number(process.argv[idx + 1]) : 3000;

/* ───────── folders ───────── */
const publicDir = path.join(__dirname, '..', 'public');
const imgDir    = path.join(publicDir, 'images');
const tmpDir    = path.join(process.cwd(), '__tmp'); // for temporary storage

for (const d of [publicDir, imgDir, tmpDir]) {
  fs.mkdirSync(d, { recursive: true });
}

/* ───────── KB file path ───────── */
const kbFile = path.join(publicDir, `knowledge_base_${PORT}.json`);

/* ───────── Auto‐create KB file if missing ───────── */
if (!fs.existsSync(kbFile)) {
  const initialWrapper = {
    map: {},
    receivedData: [],
    hexHistory: [],
    predictedText: '',
    modelReady: false
  };
  fs.writeFileSync(kbFile, JSON.stringify(initialWrapper, null, 2));
  console.log(`✨ Created new knowledge base at ${kbFile}`);
}

/* ───────── INSTANTIATE EMBEDDING & MODEL ───────── */
const imageEmbedding = new ImageCodec(/* any config if needed */);
const model = new TransformerModel({ kbPath: kbFile, port: PORT });

/* ───────── CREATE SEMANTIC FRAMEWORK ───────── */
const sf = new SemanticFramework({
  kbPath:    kbFile,
  embedding: imageEmbedding,
  model:     model,
  imgDir:    imgDir
});

/* Helper to persist if disk‐backed */
const persist = () => {
  if (!sf.kbPath) {
    sf.kbPath = kbFile;
  }
  fs.writeFileSync(kbFile, JSON.stringify(sf.exportKB(), null, 2));
};

/* ───────── Express ───────── */
const app = express();
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.static(publicDir));

/* ───────── routes ───────── */
app.get('/ping', (_q, r) => r.json({ ok: true, port: PORT }));
app.get('/knowledge_base', (_q, r) => r.sendFile(kbFile));

/* SEND (teacher) */
app.post('/send', (req, res) => {
  try {
    const { imageData, text } = req.body;
    if (!imageData && !text) {
      return res.status(400).json({ error: 'No payload provided.' });
    }

    let payload;
    if (imageData) {
      // 1) Strip off “data:…;base64,” if present, then decode:
      const b64 = imageData.includes(',') ? imageData.split(',').pop() : imageData;
      const jpegBuffer = Buffer.from(b64, 'base64');

      // 2) Spawn Python in “encode-bytes” mode, piping raw JPEG → stdin:
      const script = path.join(__dirname, 'image_codec_diffusers.py');
      const proc = spawnSync(
        PY,
        [script, 'encode-bytes'],
        {
          input:    jpegBuffer,
          encoding: 'utf8',
          maxBuffer: 100_000_000
        }
      );

      if (proc.status !== 0) {
        console.error('Python encode‐bytes stderr:', proc.stderr);
        return res.status(500).json({ error: `Encoder failed: ${proc.stderr}` });
      }

      // 3) Parse the JSON‐encoded latent array (floats):
      let latentArray;
      try {
        latentArray = JSON.parse(proc.stdout);
      } catch (e) {
        console.error('Failed to parse JSON from encoder stdout:', e, proc.stdout);
        return res.status(500).json({ error: 'Invalid JSON from encoder.' });
      }

      // 4) Pass the raw latent array into sf.send(...) AS A SINGLE ELEMENT:
      payload = sf.send([latentArray]);

    } else {
      // If you also want to support text:
      const vec = sf.vectorize(text);
      payload = sf.send([vec]);
    }

    persist();
    return res.json({ ok: true, payload });

  } catch (e) {
    console.error('/send error:', e);
    return res.status(500).json({ error: e.message });
  }
});

/* RECEIVE (apprentice) */
app.post('/receive', (req, res) => {
  try {
    const { payload } = req.body;
    if (!Array.isArray(payload)) {
      return res.status(400).json({ error: 'Bad payload: expected an array' });
    }
    sf.receive(payload)
      .then(() => {
        persist();
        res.json({ ok: true });
      })
      .catch(e => {
        console.error('/receive error:', e);
        res.status(500).json({ error: e.message });
      });
  } catch (e) {
    console.error('/receive error:', e);
    res.status(500).json({ error: e.message });
  }
});

/* PREDICT */
app.post('/predict', (_q, res) => {
  try {
    const lastHex = sf.hexHistory.at(-1);
    if (!lastHex) return res.status(400).json({ error: 'History empty.' });

    const nextHex = sf.predict(lastHex);
    sf.hexHistory.push(nextHex);
    persist();
    res.json({ nextHex, nextText: sf.predictedText });
  } catch (e) {
    console.error('/predict', e);
    res.status(500).json({ error: e.message });
  }
});

/* CLEAR */
app.post('/clear', (_q, res) => {
  sf.clear();
  persist();
  res.json({ ok: true });
});

/* SPA fallback */
app.get('*', (_req, res) => {
  res.sendFile(path.join(publicDir, 'index.html'));
});

app.listen(PORT, () => {
  console.log(`Dashboard → http://localhost:${PORT}`);
});
