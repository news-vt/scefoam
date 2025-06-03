/* eslint-disable no-console */
// ────────────────────────────────────────────────────────────────────────────────
// File: src/embedding.js
// GloVe 100‑d embedding backend with simple average pooling & nearest‑neighbor decode.
// ────────────────────────────────────────────────────────────────────────────────
import fs from 'fs';
import path from 'path';

export class GloveEmbedding {
  constructor(glovePath = path.join(process.cwd(), 'glove.6B.100d.txt')) {
    this.dim = 100;
    this.vocab = new Map();

    for (const line of fs.readFileSync(glovePath, 'utf-8').split(/\r?\n/)) {
      const parts = line.trim().split(' ');
      if (parts.length === this.dim + 1) {
        this.vocab.set(parts[0], parts.slice(1).map(Number));
      }
    }
    if (!this.vocab.size) {
      throw new Error(`Failed to load GloVe vectors from ${glovePath}`);
    }
    this._matrix = Array.from(this.vocab.values());
    this._words = Array.from(this.vocab.keys());
  }

  /** Embed arbitrary string → 100‑d vector (Array<number>) */
  embed(text) {
    const tokens = text.toLowerCase().split(/\W+/).filter(Boolean);
    if (!tokens.length) return Array(this.dim).fill(0);
    const sum = Array(this.dim).fill(0);
    let count = 0;
    for (const tok of tokens) {
      const vec = this.vocab.get(tok);
      if (vec) {
        vec.forEach((v, i) => { sum[i] += v; });
        count += 1;
      }
    }
    if (count === 0) return Array(this.dim).fill(0);
    return sum.map(v => v / count);
  }

  /** Devectorize – naive nearest neighbor word */
  devectorize(vec) {
    let bestIdx = -1;
    let bestDist = Infinity;
    for (let i = 0; i < this._matrix.length; i++) {
      const wv = this._matrix[i];
      let dist = 0;
      for (let j = 0; j < this.dim; j++) {
        const diff = wv[j] - vec[j];
        dist += diff * diff;
      }
      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = i;
      }
    }
    return this._words[bestIdx] || '';
  }
}

export default GloveEmbedding;
