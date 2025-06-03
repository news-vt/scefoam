/* eslint-env jest */
/* -------------------------------------------------------------------------- */
/*  File: tests/demo.test.js                                                   */
/*  Integration tests for SemanticFramework – KB growth, key reuse & lookup    */
/*  Run with:  npx jest tests/demo.test.js                                      */
/* -------------------------------------------------------------------------- */
import fs from 'fs';
import path from 'path';
import { SemanticFramework } from '../SemanticFramework.js';

/* ---------------- Dummy embedding backend (deterministic) ------------------------- */
class DummyEmbedding {
  /* maps string → fixed‑length numeric vector (char codes + padding) */
  embed(raw) {
    const MAX = 8;
    const vec = new Array(MAX).fill(0);
    [...String(raw).slice(0, MAX)].forEach((c, i) => {
      vec[i] = c.charCodeAt(0);
    });
    return vec;
  }
  /* back to string (trim zero padding) */
  devectorize(vec) {
    return vec
      .filter((v) => v !== 0)
      .map((v) => String.fromCharCode(v))
      .join('');
  }
}

/* ---------------- Dummy model backend (echo) ------------------------------ */
class EchoModel {
  async predict(vec) {
    return vec; // simply echoes the same vector
  }
  async fit() {
    /* no‑op */
  }
  async save() {
    /* no‑op */
  }
}

/* ---------------- Temp KB setup ------------------------------------------- */
const TMP_DIR = path.join(process.cwd(), '.tmp_tests');
const KB_PATH = path.join(TMP_DIR, 'kb_demo.json');

beforeAll(() => {
  if (!fs.existsSync(TMP_DIR)) fs.mkdirSync(TMP_DIR);
});

afterAll(() => {
  fs.rmSync(TMP_DIR, { recursive: true, force: true });
});

/* ---------------- Test suite ---------------------------------------------- */
describe('SemanticFramework – KB growth, key reuse & lookup', () => {
  const sf = new SemanticFramework({
    kbPath: KB_PATH,
    embedding: new DummyEmbedding(),
    model: new EchoModel(),
  });

  const sent1 = 'Hello World, This is my name Alex';
  const sent2 = 'Hello Earth, That is my alias John';
  let keys1 = [];

  test('first sentence populates KB, verifies embeddings, and returns new keys', () => {
    const words1 = sent1.split(' ');
    const sizeBefore = sf.kb.size;
    const payload1   = sf.send(words1);
    const sizeAfter  = sf.kb.size;

    console.log('\nKB size before:', sizeBefore, 'after:', sizeAfter);
    expect(sizeAfter).toBeGreaterThan(sizeBefore);

    // Extract hex keys and verify each key maps to correct embedding
    keys1 = payload1.map((item, idx) => {
      if (Array.isArray(item)) {
        const [hex, vec] = item;
        const storedVec = sf.kb.get(hex);
        expect(storedVec).toEqual(vec);
        const dev = sf.embedding.devectorize(storedVec);
        expect(dev).toBe(words1[idx]);
        return hex;
      } else {
        // existing key (unlikely on first run)
        const storedVec = sf.kb.get(item);
        const dev = sf.embedding.devectorize(storedVec);
        expect(dev).toBe(words1[idx]);
        return item;
      }
    });

    console.log('New keys for first sentence:', keys1);
    // All returned symbols should be unique
    expect(new Set(keys1).size).toBe(keys1.length);
  });

  test('second sentence reuses keys, adds new ones, and verifies embeddings', () => {
    const words2 = sent2.split(' ');
    const sizeBefore = sf.kb.size;
    const payload2   = sf.send(words2);
    const sizeAfter  = sf.kb.size;

    console.log('KB size before:', sizeBefore, 'after:', sizeAfter);
    expect(sizeAfter).toBeGreaterThanOrEqual(sizeBefore);

    const keys2 = payload2.map((item, idx) => {
      if (Array.isArray(item)) {
        const [hex, vec] = item;
        const storedVec = sf.kb.get(hex);
        expect(storedVec).toEqual(vec);
        const dev = sf.embedding.devectorize(storedVec);
        expect(dev).toBe(words2[idx]);
        return hex;
      } else {
        // lookup existing vector and verify
        const storedVec = sf.kb.get(item);
        const dev = sf.embedding.devectorize(storedVec);
        expect(dev).toBe(words2[idx]);
        return item;
      }
    });

    const overlap = keys1.filter((k) => keys2.includes(k));
    expect(overlap.length).toBeGreaterThan(0); // at least 'Hello' reused
    console.log('Overlapping keys:', overlap);
  });
});