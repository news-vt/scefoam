/* eslint-disable no-console */
// ──────────────────────────────────────────────────────────────────────────────
// File: src/SemanticFramework.js
// Uses *individual* word embeddings (no message-level mean pooling)
// ──────────────────────────────────────────────────────────────────────────────
import fs from 'fs';
import GloveEmbedding from './embedding.js';
import TransformerModel from './model.js';

/* cheap hex key generator – no counter file */
function nextHex(keys) {
  let max = -1;
  for (const k of keys) max = Math.max(max, parseInt(k.slice(1), 16));
  return `#${(max + 1).toString(16).padStart(8, '0')}`;
}

export class SemanticFramework {
  constructor({
    kbPath = 'knowledge_base.json',
    embedding = new GloveEmbedding(),
    model = new TransformerModel(),
  } = {}) {
    this.embedding = embedding;
    this.model = model;
    this.kbPath = kbPath;

    /* ── load wrapper ─────────────────────────────────────────── */
    let wrapper = {};
    if (fs.existsSync(kbPath)) {
      try { wrapper = JSON.parse(fs.readFileSync(kbPath, 'utf8')); }
      catch { /* corrupted → start fresh */ }
    }
    const mapObj = wrapper.map ?? wrapper;           // back-compat
    this.kb = new Map(Object.entries(mapObj));
    this.rawData = wrapper.rawData ?? [];
    this.receivedData = wrapper.receivedData ?? [];
    this.hexHistory = wrapper.hexHistory ?? [];
    this.hexHistory = this.hexHistory.filter(
      h => typeof h === 'string' && h.startsWith('#')
    );

    this.predictedText = wrapper.predictedText ?? '';
    this.modelReady = this.hexHistory.length >= 2;

    /* let the model know where the KB lives */
    this.model.kbPath = kbPath;
  }

  /* ───────── helpers ───────── */
  _persist() {
    fs.writeFileSync(
      this.kbPath,
      JSON.stringify({
        map: Object.fromEntries(this.kb),
        rawData: this.rawData,
        receivedData: this.receivedData,
        hexHistory: this.hexHistory,
        predictedText: this.predictedText
      }, null, 2)
    );
  }
  vectorize(tok) { return this.embedding.embed(tok); }
  devectorize(vec) { return this.embedding.devectorize(vec); }

  /* ------------------------------------------------------------------ */
  /*  private helper – register vectors and give back their hex keys    */
  /* ------------------------------------------------------------------ */
  _register(vectors) {
    const newKeys = new Set();
    const hexes = vectors.map(vec => {
      const sig = vec.join(',');
      const hit = [...this.kb.entries()]
        .find(([, v]) => Array.isArray(v) && v.join(',') === sig);
      let hex = hit ? hit[0] : undefined;

      if (!hex) {
        hex = nextHex(this.kb.keys());
        this.kb.set(hex, vec);
        newKeys.add(hex);
      }
      return hex;
    });
    return { hexes, newKeys };
  }

  /* ----------------------------- SEND (teacher) ---------------------------- */
  send(words) {
    const vecs = words.map(w => this.vectorize(w));
    const { hexes, newKeys } = this._register(vecs);

    const stamp = new Date().toISOString().slice(0, 16).replace('T', ' ');
    this.rawData.push(`[${stamp}] ${words.join(' ')}`);

    this.hexHistory.push(...hexes);
    this._persist();
    if (this.hexHistory.length >= 2) { this.model.fit(); this.modelReady = true; }

    return hexes.map((h, i) => newKeys.has(h) ? [h, vecs[i]] : h);
  }

  /* --------------------------- RECEIVE (apprentice) ------------------------ */
  receive(payload) {
    /* 0️⃣  store every (hex→vector) pair we see ---------------------------- */
    for (const item of payload) {
      if (Array.isArray(item)) {
        const [hex, vec] = item;
        this.kb.set(hex, vec);
      }
    }

    /* 1️⃣  translate hexes → words (ignore fails that start with '#') ------ */
    const hexes = payload.map(p => Array.isArray(p) ? p[0] : p);
    const words = hexes
      .map(h => this.devectorize(this.kb.get(h)))
      .filter(w => typeof w === 'string' && !w.startsWith('#'));

    /* 2️⃣  single, pretty log line ---------------------------------------- */
    const stamp = new Date().toISOString().slice(0, 16).replace('T', ' ');
    if (words.length)                    // only log when at least one real word
      this.receivedData.push(`[${stamp}] ${words.join(' ')}`);

    /* 3️⃣  update model timeline & (re)train ------------------------------- */
    this.hexHistory.push(...hexes);
    this._persist();
    if (this.hexHistory.length >= 2) { this.model.fit(); this.modelReady = true; }
  }


  /* ------------------------------ PREDICT ---------------------------------- */
  predict(lastHex) {
    /* 0️⃣  look up the embedding for the last token -------------------- */
    const prevVec = this.kb.get(lastHex);
    if (!Array.isArray(prevVec))
      throw new Error(`Unknown key ${lastHex}`);

    /* 1️⃣  transformer forecasts the *next* embedding ------------------ */
    const rawNextVec = this.model.predict(prevVec);

    /* 2️⃣  nearest real word in *embedding* space ---------------------- */
    const word = this.devectorize(rawNextVec);   // ← GloVe NN search
    const canonicalVec = this.vectorize(word);           // exact 100-d vector

    /* 3️⃣  does that vector already live in the KB? -------------------- */
    const sig = canonicalVec.join(',');
    let hex = [...this.kb.entries()]
      .find(([, v]) => Array.isArray(v) && v.join(',') === sig)
      ?.[0];                                  // 1st element = key

    /* 4️⃣  if not: mint a fresh key & insert --------------------------- */
    if (!hex) {
      hex = nextHex(this.kb.keys());
      this.kb.set(hex, canonicalVec);
    }

    /* 5️⃣  UX niceties -------------------------------------------------- */
    this.predictedText = word;    // show the actual word in the dashboard
    this._persist();              // commit to disk
    return hex;                   // hand back to /predict
  }

  /* --------------------------- housekeeping ------------------------------- */
  clear() {
    this.kb.clear();
    this.rawData = this.receivedData = this.hexHistory = [];
    this.predictedText = ''; this.modelReady = false;
    this._persist();
  }

  exportKB() {
    return {
      map: Object.fromEntries(this.kb),
      rawData: this.rawData,
      receivedData: this.receivedData,
      hexHistory: this.hexHistory,
      predictedText: this.predictedText,
      modelReady: this.modelReady
    };
  }
}               //  ← end of class  ✅

export default SemanticFramework;
