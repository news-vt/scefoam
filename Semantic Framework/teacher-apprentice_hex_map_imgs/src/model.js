// src/model.js

const { spawnSync } = require('child_process');
const path = require('path');

const PY = process.env.PYTHON || 'python';

/**
 * Extracts port number from a filename like "..._<PORT>.json"
 */
function portFromKB(kbPath) {
  const m = kbPath.match(/_(\d+)\.json$/);
  return m ? m[1] : '';
}

class TransformerModel {
  /**
   * @param {object} opts
   * @param {string} [opts.transformerPath]  // defaults to ./transformer.py
   * @param {string} [opts.kbPath]           // e.g. knowledge_base_<PORT>.json
   * @param {string|number} [opts.port]      // if omitted, inferred from kbPath
   */
  constructor({
    transformerPath = path.join(__dirname, 'transformer.py'),
    kbPath = 'knowledge_base.json',
    port = null
  } = {}) {
    this.kbPath = kbPath;
    this.port = String(port ?? portFromKB(kbPath));
    this.transformerPath = transformerPath;
  }

  _invoke(cmd) {
    const env = { ...process.env, PORT_ARG: this.port };
    const { status, stdout, stderr } = spawnSync(
      PY,
      [this.transformerPath, cmd, this.kbPath],
      { encoding: 'utf8', env }
    );
    if (status !== 0) {
      throw new Error(stderr || `transformer.py exited with ${status}`);
    }
    return JSON.parse(stdout.trim() || '{}');
  }

  fit() {
    try {
      this._invoke('train');
    } catch (e) {
      console.error('[transformer] train:', e.message);
    }
  }

  /** @returns {number[]} – next 100-dim embedding vector */
  predict() {
    const res = this._invoke('predict');
    if (res.error) throw new Error(res.error);
    if (Array.isArray(res.vector)) return res.vector;
    throw new Error('transformer.py did not return a "vector" field');
  }

  save() {
    // no-op: Python script handles checkpoints
  }
}

module.exports = TransformerModel;
