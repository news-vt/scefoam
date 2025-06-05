/* eslint-disable no-console */
// ──────────────────────────────────────────────────────────────
// Image‐aware SemanticFramework – latent‐only transport
// ──────────────────────────────────────────────────────────────
import fs from 'fs';
import path from 'path';
import GloveEmbedding from './embedding.js';
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
    // If one vector is zero‐length, treat similarity as 0.
    return 0;
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}


export class SemanticFramework {
  /**
   * @param {Object}   options
   * @param {string}   options.kbPath       – path to knowledge‐base JSON
   * @param {GloveEmbedding} options.embedding
   * @param {TransformerModel} options.model
   * @param {string}   options.imgDir       – **NEW**: folder where decoded PNGs go
   */
  constructor({
    kbPath = 'knowledge_base.json',
    embedding = new GloveEmbedding(),
    model = new TransformerModel(),
    imgDir = 'public/images'    // ← default in case you forget to pass it
  } = {}) {
    this.embedding = embedding;
    this.imgCodec = new ImageCodec();
    this.model = model;
    this.kbPath = kbPath;
    this.imgDir = imgDir;     // <— store imgDir so receive() can use it

    /* ── load persisted KB wrapper ───────────────────────────── */
    let wrapper = {};
    if (fs.existsSync(kbPath)) {
      try { wrapper = JSON.parse(fs.readFileSync(kbPath, 'utf8')); }
      catch { /* corrupted → start fresh */ }
    }
    const mapObj = wrapper.map ?? wrapper;

    this.kb = new Map(Object.entries(mapObj));
    this.rawData = wrapper.rawData ?? [];
    this.receivedData = wrapper.receivedData ?? [];
    this.hexHistory = wrapper.hexHistory ?? [];
    this.hexHistory = this.hexHistory.filter(
      (h) => typeof h === 'string' && h.startsWith('#')
    );
    this.predictedText = wrapper.predictedText ?? '';
    this.modelReady = this.hexHistory.length >= 2;

    this.model.kbPath = kbPath;
  }

  /* ───────── embedding helpers ───────── */
  vectorize(tok) {
    // If tok is already a numeric array (raw latent), return it directly:
    if (Array.isArray(tok) && tok.every(x => typeof x === 'number')) {
      return tok;
    }

    if (typeof tok === 'string' && tok.startsWith('<img:')) {
      const fname = tok.slice(5, -1);                     // e.g. "frame_12345.jpg"
      const abs = path.join(process.cwd(), '__tmp', fname);
      return this.imgCodec.embed(abs);                    // returns latent array
    }

    // Otherwise tok is assumed to be a text token:
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
        continue; // skip if dimensionality mismatch (shouldn't happen if all vectors share dims)
      }
      const sim = cosineSimilarity(existingVec, vec);
      if (sim > bestSim) {
        bestSim = sim;
        bestHex = hexKey;
      }
      if (bestSim >= threshold) {
        // Once we hit the threshold exactly, we can stop early:
        return bestHex;
      }
    }

    // After scanning, only return if bestSim ≥ threshold:
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
    const toRegister = [];        // array of { index: i, vec: vecs[i] }
    const newlyAssigned = {};     // map hex → true for those we’ll newly register

    // 2c) For each vec: look for a cosine‐close match in this.kb
    const SIM_THRESHOLD = 0.90; // tweak if you want “almost bit‐for‐bit” or “nearer”
    for (let i = 0; i < vecs.length; i++) {
      const v = vecs[i];
      // a) search by vector, not by hex
      const existingHex = this.findClosestHexByVector(v, SIM_THRESHOLD);
      if (existingHex) {
        // “Close enough” latent found → reuse that hex
        hexes[i] = existingHex;
      } else {
        // No match ⇒ mark this index to register after
        toRegister.push({ index: i, vec: v });
      }
    }

    // 3) Register only the truly new vectors via _register()
    if (toRegister.length > 0) {
      const newVectors = toRegister.map(x => x.vec);
      // _register(newVectors) will assign new hexes only for these
      const { hexes: regHexes, newKeys } = this._register(newVectors);

      // Plug those newly assigned hexes back into the right slot in hexes[]
      for (let k = 0; k < toRegister.length; k++) {
        const idx = toRegister[k].index;
        const assignedHex = regHexes[k];
        hexes[idx] = assignedHex;
        // mark that we created this one now:
        newlyAssigned[assignedHex] = true;
      }
    }

    // 4) Log a timestamp+rawData entry exactly as before
    const stamp = new Date().toISOString().slice(0, 16).replace('T', ' ');
    this.rawData.push(`[${stamp}] ${tokens.join(' ')}`);

    // 5) Update hexHistory with every hex (old or new) and persist
    this.hexHistory.push(...hexes);
    this._persist();

    // 6) Build the payload: only those in newlyAssigned get [hex, vec], others remain hex‐only
    return hexes.map((h, i) => {
      if (newlyAssigned[h]) {
        // newly created registration ⇒ send hex+vector so receiver can store/decode
        return [h, vecs[i]];
      } else {
        // already existed in KB ⇒ send only the hex string
        return h;
      }
    });
  }
  /* ───────── RECEIVE (apprentice) ───────── */
  async receive(payload) {
    const stamp = new Date().toISOString().slice(0, 16).replace('T', ' ');
    const decodeTasks = []; // array of Promises that each write one PNG file

    for (const item of payload) {
      // — If item is [hex, vec], maybe‐store & decode
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

        // ➋ decode → PNG, but only if file doesn’t exist yet
        const keyNoHash = hex.slice(1);
        const outPath = path.join(this.imgDir, `${keyNoHash}.png`);

        if (!fs.existsSync(outPath)) {
          // schedule an async decode of `vec` → Buffer
          const task = this.decodeAsync(vec)
            .then(pngBuf => ({ outPath, pngBuf, hex: keyNoHash }))
            .catch(err => {
              console.error(`Error decoding latent for ${hex}:`, err);
              return null;
            });
          decodeTasks.push(task);
        }

        // ➌ log receipt
        this.receivedData.push(`[${stamp}] <img:${keyNoHash}>`);
      }

      // — Else if item is just a bare "hex"
      else if (typeof item === 'string') {
        const hex = item;
        const vec = this.kb.get(hex);

        if (!Array.isArray(vec)) {
          console.warn(`DEBUG: no latent in KB for ${hex}, skipping decode.`);
          continue;
        }
        const keyNoHash = hex.slice(1);
        const outPath = path.join(this.imgDir, `${keyNoHash}.png`);

        if (!fs.existsSync(outPath)) {
          // schedule an async decode
          const task = this.decodeAsync(vec)
            .then(pngBuf => ({ outPath, pngBuf, hex: keyNoHash }))
            .catch(err => {
              console.error(`Error decoding cached latent for ${hex}:`, err);
              return null;
            });
          decodeTasks.push(task);
        }

        this.receivedData.push(`[${stamp}] <img:${keyNoHash}>`);
      }
      // otherwise, ignore any other payload items
    }

    // ➃ await *all* decodes in parallel, then write each Buffer
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
    if (!Array.isArray(prevVec)) throw new Error(`Unknown key ${lastHex}`);

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
    this.rawData = [];
    this.receivedData = [];
    this.hexHistory = [];
    this.predictedText = '';
    this.modelReady = false;
    this._persist();
  }

  _persist() {
    fs.writeFileSync(this.kbPath, JSON.stringify(this.exportKB(), null, 2));
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
}

export default SemanticFramework;
