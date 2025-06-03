/* eslint-disable no-console */
/* ──────────────────────────────────────────────────────────────
   File: src/model.js
   JS wrapper around src/transformer.py
   – automatically extracts <PORT> from the KB filename
   – therefore you can keep using  new TransformerModel()
   – each running instance still gets its own .pt checkpoint
   ------------------------------------------------------------ */
import { spawnSync } from 'child_process';
import path          from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

/* helper:  "…_4500.json"  →  "4500"  */
function portFromKB(kbPath) {
  const m = kbPath.match(/_(\d+)\.json$/);
  return m ? m[1] : '';      // empty string for the teacher instance
}

class TransformerModel {
  /**
   * @param {object} opts
   * @param {string} [opts.transformerPath]  defaults to ./transformer.py
   * @param {string} [opts.kbPath]           knowledge_base_<PORT>.json
   * @param {string|number} [opts.port]      optional; if omitted it is
   *                                         read from kbPath via portFromKB()
   */
  constructor({
    transformerPath = path.join(__dirname, 'transformer.py'),
    kbPath          = 'knowledge_base.json',
    port            = null
  } = {}) {
    this.kbPath          = kbPath;
    this.port            = String(port ?? portFromKB(kbPath));   // auto-infer
    this.transformerPath = transformerPath;
  }

  /* -------- call transformer.py -------------------------------------- */
  _invoke(cmd) {
    const env = { ...process.env, PORT_ARG: this.port };
    const { status, stdout, stderr } = spawnSync(
      'python', [this.transformerPath, cmd, this.kbPath],
      { encoding: 'utf-8', env }
    );
    if (status !== 0) throw new Error(stderr || `transformer.py exited with ${status}`);
    return JSON.parse(stdout.trim() || '{}');
  }

  /* -------- public API ----------------------------------------------- */
  fit() {
    try { this._invoke('train'); }
    catch (e) { console.error('[transformer] train:', e.message); }
  }

  /** @returns {number[]}  – next 100-d embedding vector */
  predict() {
    const res = this._invoke('predict');
    if (res.error) throw new Error(res.error);
    if (Array.isArray(res.vector)) return res.vector;
    throw new Error('transformer.py did not return a "vector" field');
  }

  /* checkpoints are written by transformer.py itself */
  save() {}
}

export default TransformerModel;
export { TransformerModel };
