/* eslint-disable no-console */
// ────────────────────────────────────────────────────────────────────────────────
// File: src/server.js
// Express backend for the teacher‑apprentice demo – *single* canonical version
// that guarantees:
//   • outgoing messages are split into WORDS → hex‑payload
//   • only that hex‑payload is ever forwarded to peers
//   • /receive gets exactly that payload, so hexHistory stays clean
//   • KB wrapper is always persisted after every mutating call
// ────────────────────────────────────────────────────────────────────────────────
import express from 'express';
import cors    from 'cors';
import bodyParser from 'body-parser';
import fs   from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

import { SemanticFramework } from './SemanticFramework.js';
import GloveEmbedding        from './embedding.js';
import { TransformerModel }  from './model.js';           // ← already exists


/* -------------------------------------------------------------------------- */
/*  Resolve paths & CLI port                                                  */
/* -------------------------------------------------------------------------- */
const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

const idx  = process.argv.indexOf('--port');
const PORT = idx !== -1 ? Number(process.argv[idx + 1]) : 3000;

/* -------------------------------------------------------------------------- */
/*  Per‑instance public / KB files                                            */
/* -------------------------------------------------------------------------- */
const publicDir = path.join(__dirname, '..', 'public');
fs.mkdirSync(publicDir, { recursive: true });

const kbFile = path.join(publicDir, `knowledge_base_${PORT}.json`);
if (!fs.existsSync(kbFile)) {
  fs.writeFileSync(kbFile, JSON.stringify({
    map          : {},
    rawData      : [],
    receivedData : [],
    hexHistory   : [],
    predictedText: ''
  }, null, 2));
}

/* ---------- build the model FIRST so it already knows the port ----------- */
const model = new TransformerModel({
  kbPath : kbFile,     // knowledge_base_<PORT>.json
  port   : PORT        // explicit is safest
});

/* -------------------------------------------------------------------------- */
/*  Instantiate the semantic framework                                        */
/* -------------------------------------------------------------------------- */
const sf = new SemanticFramework({
  kbPath   : kbFile,
  embedding: new GloveEmbedding(path.join(__dirname, 'glove.6B.100d.txt')),
  model    : model
});
const persist = () => fs.writeFileSync(kbFile, JSON.stringify(sf.exportKB(), null, 2));

/* -------------------------------------------------------------------------- */
/*  Express                                                                   */
/* -------------------------------------------------------------------------- */
const app = express();
app.use(cors());
app.use(bodyParser.json({ limit: '15mb' }));
app.use(express.static(publicDir));

/* ───────────────────────────── HELPERS ─────────────────────────────────── */
const splitWords = txt => txt.trim().split(/\s+/).filter(Boolean);

/* ───────────────────────────── ROUTES ──────────────────────────────────── */
app.get('/ping', (_q, r) => r.json({ ok: true, port: PORT }));
app.get('/knowledge_base', (_q, r) => r.sendFile(kbFile));

/* ----------------------------- SEND (teacher) ---------------------------- */
app.post('/send', (req, res) => {
  try {
    const { data, imageData } = req.body;
    if (!data && !imageData) {
      return res.status(400).json({ error: 'No payload provided.' });
    }

    let payload;
    if (imageData) {
      /* handle base‑64 images as a single special token                         */
      const imgDir = path.join(publicDir, 'images');
      fs.mkdirSync(imgDir, { recursive: true });
      const fname = `img_${Date.now()}.png`;
      const base64 = imageData.split(',').pop();
      fs.writeFileSync(path.join(imgDir, fname), Buffer.from(base64, 'base64'));
      payload = sf.send([`<img:${fname}>`]);
    } else {
      /* normal text → split into *words*                                       */
      const words = splitWords(data);
      payload = sf.send(words);               // [hex | [hex,vec] …]
    }

    persist();                                // make sure KB is on disk
    res.json({ ok: true, payload });          // *forward exactly this* to peers
  } catch (e) {
    console.error('/send', e);
    res.status(500).json({ error: e.message });
  }
});

/* --------------------------- RECEIVE (apprentice) ------------------------ */
app.post('/receive', (req, res) => {
  try {
    const { payload } = req.body;
    if (!Array.isArray(payload) || payload.length === 0) {
      return res.status(400).json({ error: 'Bad payload.' });
    }

    sf.receive(payload);                      // store + (re)train
    persist();
    res.json({ ok: true });
  } catch (e) {
    console.error('/receive', e);
    res.status(500).json({ error: e.message });
  }
});

/* ------------------------------ PREDICT ---------------------------------- */
app.post('/predict', async (_q, res) => {
  try {
    const lastHex = sf.hexHistory.at(-1);
    if (!lastHex) return res.status(400).json({ error: 'History empty.' });

    const nextHex = await sf.predict(lastHex);
    sf.hexHistory.push(nextHex);
    persist();
    res.json({ nextHex, nextText: sf.predictedText });
  } catch (e) {
    console.error('/predict', e);
    res.status(500).json({ error: e.message });
  }
});

/* ------------------------------- CLEAR ----------------------------------- */
app.post('/clear', (_q, res) => {
  sf.clear();
  persist();
  res.json({ ok: true });
});

/* SPA fallback */
app.get('*', (_q, r) => r.sendFile(path.join(publicDir, 'index.html')));

app.listen(PORT, () => console.log(`Dashboard → http://localhost:${PORT}`));
