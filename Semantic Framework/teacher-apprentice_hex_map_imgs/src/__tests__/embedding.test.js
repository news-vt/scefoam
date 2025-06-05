/* eslint-env jest */
/* -------------------------------------------------------------------------- */
/*  File: tests/embedding.test.js                                             */
/*  Unit tests for GloveEmbedding using the real GloVe 6B‑100d file.          */
/*  The test expects ../glove.6B.100d.txt to be present (one level up from    */
/*  the tests/ directory).                                                    */
/*  Run with:  npx jest tests/embedding.test.js                               */
/* -------------------------------------------------------------------------- */
import fs   from 'fs';
import path from 'path';
import { GloveEmbedding } from '../embedding.js';

// Path to the full 100‑dimensional GloVe file shipped separately
const GLOVE_PATH = path.join(__dirname, '..', 'glove.6B.100d.txt');

if (!fs.existsSync(GLOVE_PATH)) {
  // Skip entire suite if the real file isn't in repo
  console.warn(`\n[embedding.test] ${GLOVE_PATH} not found → tests skipped`);
  test.skip('GloVe file missing', () => {});
} else {
  // Load once – reading 400k rows can take a couple seconds
  const embedding = new GloveEmbedding(GLOVE_PATH, 100);

  describe('GloveEmbedding with real 100‑d vectors', () => {
    test('embed("hello") returns 100‑length non‑zero vector', () => {
      const vec = embedding.embed('hello');
      expect(vec.length).toBe(100);
      expect(vec.some((v) => v !== 0)).toBe(true);
    });

    test('embed of unknown token returns all zeros', () => {
      const vec = embedding.embed('zzxtqvv');
      expect(vec.every((v) => v === 0)).toBe(true);
    });

    test('embedding of two words equals average of individual embeddings', () => {
      const vHello = embedding.embed('hello');
      const vWorld = embedding.embed('world');
      const vBoth  = embedding.embed('hello world');
      const avg    = vHello.map((v, i) => (v + vWorld[i]) / 2);
      const maxErr = Math.max(...avg.map((v, i) => Math.abs(v - vBoth[i])));
      expect(maxErr).toBeLessThan(1e-6);
    });

    test('devectorize reproduces word when vector exists', () => {
      const vWorld = embedding.embed('world');
      const word   = embedding.devectorize(vWorld);
      expect(word).toBe('world');
    });

    test('devectorize of zero vector returns some nearest token', () => {
      const word = embedding.devectorize(Array(100).fill(0));
      expect(typeof word).toBe('string');
      expect(word.length).toBeGreaterThan(0);
    });
  });
}
