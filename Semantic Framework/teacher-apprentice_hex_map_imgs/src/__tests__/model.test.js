/* eslint-env jest */
/* -------------------------------------------------------------------------- */
/*  File: tests/model.test.js                                                 */
/*  Unit tests for the updated TransformerModel (src/model.js)               */
/*  These tests stub the _invoke() method instead of spawning dummy Python   */
/*  back‑ends, so we test JS logic in isolation.                             */
/*  Run with:  npx jest tests/model.test.js                                  */
/* -------------------------------------------------------------------------- */
import fs   from 'fs';
import path from 'path';
import { TransformerModel } from '../model.js';

/* ---------------- Temp KB ------------------------------------------------- */
const TMP_DIR = path.join(process.cwd(), '.tmp_model_tests');
const KB_PATH = path.join(TMP_DIR, 'kb.json');

beforeAll(() => {
  if (!fs.existsSync(TMP_DIR)) fs.mkdirSync(TMP_DIR);
  fs.writeFileSync(KB_PATH, JSON.stringify({ centroidHistory: [] }), 'utf8');
});

afterAll(() => {
  fs.rmSync(TMP_DIR, { recursive: true, force: true });
});

/* ---------------- Test suite --------------------------------------------- */
describe('TransformerModel wrapper (JS‑only)', () => {
  afterEach(() => {
    // Restore any spies to avoid cross‑test pollution
    jest.restoreAllMocks();
  });

  test('fit() calls _invoke("train") without throwing', () => {
    // Arrange: stub _invoke → pretend python returns {status:"trained"}
    const spy = jest
      .spyOn(TransformerModel.prototype, '_invoke')
      .mockImplementation((cmd) => {
        expect(cmd).toBe('train');
        return { status: 'trained' };
      });

    const mdl = new TransformerModel({ kbPath: KB_PATH });

    // Act & Assert: should not throw, spy is hit once
    expect(() => mdl.fit()).not.toThrow();
    expect(spy).toHaveBeenCalledTimes(1);
  });

  test('predict() returns hex when _invoke returns {hex}', () => {
    jest
      .spyOn(TransformerModel.prototype, '_invoke')
      .mockImplementation((cmd) => {
        expect(cmd).toBe('predict');
        return { hex: '#0000000a' };
      });

    const mdl = new TransformerModel({ kbPath: KB_PATH });
    const hex = mdl.predict();
    expect(hex).toBe('#0000000a');
  });

  test('predict() returns first hex from {hexes:[...] }', () => {
    jest
      .spyOn(TransformerModel.prototype, '_invoke')
      .mockReturnValue({ hexes: ['#0000000b', '#0000000c'] });

    const mdl = new TransformerModel({ kbPath: KB_PATH });
    expect(mdl.predict()).toBe('#0000000b');
  });

  test('predict() converts centroid id to hex for legacy payload', () => {
    jest
      .spyOn(TransformerModel.prototype, '_invoke')
      .mockReturnValue({ centroids: { '10': [1, 2, 3] } }); // id "10" → 0xa

    const mdl = new TransformerModel({ kbPath: KB_PATH });
    expect(mdl.predict()).toBe('#0000000a');
  });

  test('predict() throws when backend omits hex information', () => {
    jest
      .spyOn(TransformerModel.prototype, '_invoke')
      .mockReturnValue({ foo: 'bar' });

    const mdl = new TransformerModel({ kbPath: KB_PATH });
    expect(() => mdl.predict()).toThrow('hex prediction');
  });
});