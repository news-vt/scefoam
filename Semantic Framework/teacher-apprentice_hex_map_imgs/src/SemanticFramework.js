// SemanticFramework.js

import fs from 'fs';
import path from 'path';
import ImageCodec from './image_embedding.js';
import TransformerModel from './model.js';

/* simple hex key generator */
function nextHex(keys) {
  let max = -1;
  for (const k of keys) {
    max = Math.max(max, parseInt(k.slice(1), 16));
  }
  return `#${(max + 1).toString(16).padStart(8, '0')}`;
}

function cosineSimilarity(A, B) {
  if (!Array.isArray(A) || !Array.isArray(B) || A.length !== B.length) {
    throw new Error("cosineSimilarity: vectors must be equal‐length arrays.");
  }

  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < A.length; i++) {
    const a = A[i], b = B[i];
    if (typeof a !== "number" || typeof b !== "number") {
      throw new Error("cosineSimilarity: array elements must be numbers.");
    }
    dot += a * b;
    normA += a * a;
    normB += b * b;
  }

  if (normA === 0 || normB === 0) {
    return 0;
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

export class SemanticFramework {
  /**
   * @param {Object} options
   * @param {Map<string, number[]>} [options.kb]
   *   – If you pass a preloaded Map, we'll bypass file loading entirely.
   * @param {string} [options.kbPath]
   *   – Path to JSON KB. Ignored if `kb` is provided.
   * @param {ImageCodec} options.embedding
   *   – Any embedding instance (must have .embed() and .devectorize()).
   * @param {TransformerModel} options.model
   *   – Any model instance (must have .predict(), .fit(), etc.).
   * @param {string} [options.imgDir='public/images']
   *   – Where to write decoded PNGs (for <img:…> tokens).
   */
  constructor({
    kb = null,
    kbPath = 'knowledge_base.json',
    embedding,
    model,
    imgDir = 'public/images'
  } = {}) {
    if (!embedding) {
      throw new Error('SemanticFramework: you must provide an embedding instance.');
    }
    if (!model) {
      throw new Error('SemanticFramework: you must provide a model instance.');
    }

    this.embedding = embedding;
    this.imgCodec = new ImageCodec();
    this.model = model;
    this.imgDir = imgDir;

    // ── If caller gave us an in‐memory Map for KB, use it; else load from disk:
    if (kb instanceof Map) {
      this.kb = kb;
      this.receivedData = [];
      this.hexHistory = [];
      this.predictedText = '';
      this.modelReady = false;
      this.kbPath = null; // signal that we’re in “memory‐only” mode
    } else {
      this.kbPath = kbPath;
      let wrapper = {};
      if (fs.existsSync(kbPath)) {
        try {
          wrapper = JSON.parse(fs.readFileSync(kbPath, 'utf8'));
        } catch {
          // corrupted ⇒ start fresh
        }
      }
      const mapObj = wrapper.map ?? wrapper;
      this.kb = new Map(Object.entries(mapObj));
      this.receivedData = wrapper.receivedData ?? [];
      this.hexHistory = (wrapper.hexHistory ?? []).filter(h => typeof h === 'string' && h.startsWith('#'));
      this.predictedText = wrapper.predictedText ?? '';
      this.modelReady = this.hexHistory.length >= 2;
    }

    // If we do have a disk‐backed KB, let the model know where to persist:
    if (this.kbPath) {
      this.model.kbPath = this.kbPath;
    }
  }

  /* ───────── embedding helpers ───────── */
  vectorize(tok) {
    // If tok is already a numeric array (raw latent), return it directly:
    if (Array.isArray(tok) && tok.every(x => typeof x === 'number')) {
      return tok;
    }

    // If tok is of the form "<img:filename>" decode that image file into a latent vector:
    if (typeof tok === 'string' && tok.startsWith('<img:')) {
      const fname = tok.slice(5, -1); // e.g. "frame_12345.jpg"
      const absPath = path.join(process.cwd(), '__tmp', fname);
      return this.imgCodec.embed(absPath);
    }

    // Otherwise, treat tok as a text token and use the injected embedding:
    return this.embedding.embed(tok);
  }

  devectorize(vec) {
    return this.embedding.devectorize(vec);
  }

  findClosestHexByVector(vec, threshold = 0.999) {
    if (this.kb.size === 0) return null;
    let bestHex = null;
    let bestSim = -1;

    for (const [hexKey, existingVec] of this.kb.entries()) {
      if (!Array.isArray(existingVec) || existingVec.length !== vec.length) {
        continue;
      }
      const sim = cosineSimilarity(existingVec, vec);
      if (sim > bestSim) {
        bestSim = sim;
        bestHex = hexKey;
      }
      if (bestSim >= threshold) {
        return bestHex;
      }
    }

    return bestSim >= threshold ? bestHex : null;
  }

  /* ───────── internal KB register ───────── */
  _register(vectors) {
    const newKeys = new Set();
    const hexes = vectors.map((vec) => {
      const sig = vec.join(',');
      const hit = [...this.kb.entries()].find(
        ([, v]) => Array.isArray(v) && v.join(',') === sig
      );
      let hex = hit ? hit[0] : null;
      if (!hex) {
        hex = nextHex(this.kb.keys());
        this.kb.set(hex, vec);
        newKeys.add(hex);
      }
      return hex;
    });
    return { hexes, newKeys };
  }

  /* ───────── SEND (teacher) ───────── */
  send(tokens) {
    // 1) Vectorize every token → latent vector
    const vecs = tokens.map(t => this.vectorize(t));

    // 2a) Prepare an array to hold final hex for each token:
    const hexes = new Array(vecs.length);
    // 2b) Keep track of which indices/vecs must be registered from scratch
    const toRegister = [];
    const newlyAssigned = {};

    // 2c) For each vec: look for a cosine‐close match in this.kb
    const SIM_THRESHOLD = 0.90;
    for (let i = 0; i < vecs.length; i++) {
      const v = vecs[i];
      const existingHex = this.findClosestHexByVector(v, SIM_THRESHOLD);
      if (existingHex) {
        hexes[i] = existingHex;
      } else {
        toRegister.push({ index: i, vec: v });
      }
    }

    // 3) Register only the truly new vectors via _register()
    if (toRegister.length > 0) {
      const newVectors = toRegister.map(x => x.vec);
      const { hexes: regHexes, newKeys } = this._register(newVectors);

      for (let k = 0; k < toRegister.length; k++) {
        const idx = toRegister[k].index;
        const assignedHex = regHexes[k];
        hexes[idx] = assignedHex;
        newlyAssigned[assignedHex] = true;
      }
    }

    // 5) Update hexHistory and persist
    this.hexHistory.push(...hexes);
    this._persist();

    // 6) Build the payload: newly assigned → [hex, vec], others → hex
    return hexes.map((h, i) => {
      if (newlyAssigned[h]) {
        return [h, vecs[i]];
      } else {
        return h;
      }
    });
  }

  /* ───────── RECEIVE (apprentice) ───────── */
  async receive(payload) {
    const stamp = new Date().toISOString().slice(0, 16).replace('T', ' ');
    const decodeTasks = [];

    for (const item of payload) {
      if (
        Array.isArray(item) &&
        item.length === 2 &&
        typeof item[0] === 'string' &&
        Array.isArray(item[1])
      ) {
        const [hex, vec] = item;

        // ➊ register new latent if not already in KB
        if (!this.kb.has(hex)) {
          this.kb.set(hex, vec);
        }

        // ➋ decode → PNG if file doesn’t exist
        const keyNoHash = hex.slice(1);
        const outPath = path.join(this.imgDir, `${keyNoHash}.png`);
        if (!fs.existsSync(outPath)) {
          const task = this.decodeAsync(vec)
            .then(pngBuf => ({ outPath, pngBuf }))
            .catch(err => {
              console.error(`Error decoding latent for ${hex}:`, err);
              return null;
            });
          decodeTasks.push(task);
        }

        // ➌ log receipt
        this.receivedData.push(`[${stamp}] <img:${keyNoHash}>`);
      } else if (typeof item === 'string') {
        const hex = item;
        const vec = this.kb.get(hex);
        if (!Array.isArray(vec)) {
          console.warn(`DEBUG: no latent in KB for ${hex}, skipping decode.`);
          continue;
        }
        const keyNoHash = hex.slice(1);
        const outPath = path.join(this.imgDir, `${keyNoHash}.png`);
        if (!fs.existsSync(outPath)) {
          const task = this.decodeAsync(vec)
            .then(pngBuf => ({ outPath, pngBuf }))
            .catch(err => {
              console.error(`Error decoding cached latent for ${hex}:`, err);
              return null;
            });
          decodeTasks.push(task);
        }
        this.receivedData.push(`[${stamp}] <img:${keyNoHash}>`);
      }
      // ignore any other payload items
    }

    // ➃ write all decoded PNGs
    const results = await Promise.all(decodeTasks);
    for (const result of results) {
      if (result && result.pngBuf) {
        try {
          await fs.promises.writeFile(result.outPath, result.pngBuf);
        } catch (ioErr) {
          console.error(`Failed to write ${result.outPath}:`, ioErr);
        }
      }
    }

    // ➄ update hexHistory & retrain if needed
    const justHexes = payload.map(p => (Array.isArray(p) ? p[0] : p));
    this.hexHistory.push(...justHexes);
    this._persist();

    if (this.hexHistory.length >= 2) {
      this.model.fit();
      this.modelReady = true;
    }
  }

  /* ───────── helper: decode a latent array asynchronously ───────── */
  decodeAsync(vec) {
    return new Promise((resolve, reject) => {
      try {
        const pngBuf = this.imgCodec.decode(vec);
        resolve(pngBuf);
      } catch (e) {
        reject(e);
      }
    });
  }

  /* ───────── PREDICT ───────── */
  predict(lastHex) {
    const prevVec = this.kb.get(lastHex);
    if (!Array.isArray(prevVec)) {
      throw new Error(`Unknown key ${lastHex}`);
    }

    const rawNext = this.model.predict(prevVec);
    const word = this.devectorize(rawNext);
    const canon = this.vectorize(word);

    let hex = [...this.kb.entries()]
      .find(([, v]) => Array.isArray(v) && v.join(',') === canon.join(','))?.[0];

    if (!hex) {
      hex = nextHex(this.kb.keys());
      this.kb.set(hex, canon);
    }
    this.predictedText = word;
    this._persist();
    return hex;
  }

  /* ───────── housekeeping ───────── */
  clear() {
    this.kb.clear();
    this.receivedData = [];
    this.hexHistory = [];
    this.predictedText = '';
    this.modelReady = false;
    this._persist();
  }

  _persist() {
    if (!this.kbPath) return;
    fs.writeFileSync(this.kbPath, JSON.stringify(this.exportKB(), null, 2));
  }

  exportKB() {
    return {
      map: Object.fromEntries(this.kb),
      receivedData: this.receivedData,
      hexHistory: this.hexHistory,
      predictedText: this.predictedText,
      modelReady: this.modelReady
    };
  }
}

export default SemanticFramework;
