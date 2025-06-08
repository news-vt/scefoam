const fs = require('fs');
const path = require('path');
const ImageCodec = require('./image_embedding');
const TransformerModel = require('./model');

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
    throw new Error("cosineSimilarity: vectors must be equal-length arrays.");
  }
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < A.length; i++) {
    const a = A[i], b = B[i];
    if (typeof a !== 'number' || typeof b !== 'number') {
      throw new Error('cosineSimilarity: array elements must be numbers.');
    }
    dot += a * b;
    normA += a * a;
    normB += b * b;
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

class SemanticFramework {
  constructor({ kb = null, kbPath = 'knowledge_base.json', embedding, model } = {}) {
    if (!embedding) throw new Error('SemanticFramework: embedding instance required.');
    if (!model) throw new Error('SemanticFramework: model instance required.');

    this.embedding = embedding;
    this.imgCodec = new ImageCodec();
    this.model = model;

    if (kb instanceof Map) {
      this.kb = kb;
      this.receivedData = [];
      this.hexHistory = [];
      this.predictedText = '';
      this.modelReady = false;
      this.kbPath = null;
    } else {
      this.kbPath = kbPath;
      let wrapper = {};
      if (fs.existsSync(kbPath)) {
        try {
          wrapper = JSON.parse(fs.readFileSync(kbPath, 'utf8'));
        } catch {}
      }
      const mapObj = wrapper.map ?? wrapper;
      this.kb = new Map(Object.entries(mapObj));
      this.receivedData = wrapper.receivedData ?? [];
      this.hexHistory = (wrapper.hexHistory ?? []).filter(h => typeof h === 'string' && h.startsWith('#'));
      this.predictedText = wrapper.predictedText ?? '';
      this.modelReady = this.hexHistory.length >= 2;
    }

    if (this.kbPath) this.model.kbPath = this.kbPath;
  }

  vectorize(tok) {
    if (Array.isArray(tok) && tok.every(x => typeof x === 'number')) return tok;
    if (typeof tok === 'string' && tok.startsWith('<img:')) {
      const fname = tok.slice(5, -1);
      const absPath = path.join(process.cwd(), '__tmp', fname);
      return this.imgCodec.embed(absPath);
    }
    return this.embedding.embed(tok);
  }

  devectorize(vec) {
    return this.embedding.devectorize(vec);
  }

  findClosestHexByVector(vec, threshold = 0.999) {
    if (this.kb.size === 0) return null;
    let bestHex = null, bestSim = -1;
    for (const [hexKey, existingVec] of this.kb.entries()) {
      if (!Array.isArray(existingVec) || existingVec.length !== vec.length) continue;
      const sim = cosineSimilarity(existingVec, vec);
      if (sim > bestSim) {
        bestSim = sim;
        bestHex = hexKey;
      }
      if (bestSim >= threshold) return bestHex;
    }
    return bestSim >= threshold ? bestHex : null;
  }

  _register(vectors) {
    const newKeys = new Set();
    const hexes = vectors.map(vec => {
      const sig = vec.join(',');
      const hit = [...this.kb.entries()].find(([, v]) => Array.isArray(v) && v.join(',') === sig);
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

  send(tokens) {
    const vecs = tokens.map(t => this.vectorize(t));
    const hexes = new Array(vecs.length);
    const toRegister = [];
    const newly = {};
    const SIM = 0.9;

    for (let i = 0; i < vecs.length; i++) {
      const found = this.findClosestHexByVector(vecs[i], SIM);
      if (found) hexes[i] = found;
      else toRegister.push({ index: i, vec: vecs[i] });
    }
    if (toRegister.length) {
      const { hexes: reg, newKeys } = this._register(toRegister.map(x => x.vec));
      toRegister.forEach((t, i) => { hexes[t.index] = reg[i]; newly[reg[i]] = true; });
    }

    this.hexHistory.push(...hexes);
    this._persist();
    return hexes.map((h, i) => newly[h] ? [h, vecs[i]] : h);
  }

  async receive(payload) {
    const stamp = new Date().toISOString().slice(0, 16).replace('T', ' ');

    const images = payload.map(item => {
      let vec;
      let hexKey;
      if (Array.isArray(item) && item.length === 2) {
        [hexKey, vec] = item;
        if (!this.kb.has(hexKey)) this.kb.set(hexKey, vec);
      } else if (typeof item === 'string') {
        hexKey = item;
        vec = this.kb.get(hexKey);
      } else {
        return null;
      }

      try {
        return this.imgCodec.decode(vec);
      } catch (e) {
        console.error(`Error decoding latent for ${hexKey}:`, e);
        return null;
      }
    }).filter(buf => buf);

    const justHexes = payload.map(p => Array.isArray(p) ? p[0] : p);
    this.receivedData.push(...justHexes.map(h => `[${stamp}] ${h}`));
    this.hexHistory.push(...justHexes);
    this._persist();

    if (this.hexHistory.length >= 2) {
      this.model.fit();
      this.modelReady = true;
    }

    return images;
  }

  decodeAsync(vec) {
    return Promise.resolve(this.imgCodec.decode(vec));
  }

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

module.exports = SemanticFramework;