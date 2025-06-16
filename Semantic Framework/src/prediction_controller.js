// prediction_controller.js – spawns & talks to prediction_server.py on :8084
// -------------------------------------------------------------------------
const path        = require('path');
const { spawn }   = require('child_process');
const { request } = require('undici');

const PORT = 8084;
const HOST = `http://127.0.0.1:${PORT}`;

/* ── helper: flatten & modality-aware trim --------------------------- */
function prepareLatent(raw, modality) {
  // deep-flatten & cast typed arrays ⇒ plain JS list
  const flat = (function recur(x) {
    if (ArrayBuffer.isView(x)) return Array.from(x);
    if (Array.isArray(x))      return x.flatMap(recur);
    return [x];
  })(raw);

  if (modality === 'audio') {
    // keep EnCodec header [sr,nq,T] but cap to maxFrames (1024)
    if (flat.length < 3) return flat;            // malformed, pass through
    const [sr, nq, T] = flat;
    const codes       = flat.slice(3);
    const maxFrames   = 1024;                    // ≈ 6 s at 24 kHz
    const keepFrames  = Math.min(T, maxFrames);
    const keepCodes   = codes.slice(0, nq * keepFrames);
    return [sr, nq, keepFrames, ...keepCodes];
  }

  /* text / image caps */
  const cap = modality === 'text' ? 1024 : 16_384;     // image = 4×64×64
  return flat.length > cap ? flat.slice(0, cap) : flat;
}

/* ── controller class ----------------------------------------------- */
class PredictionController {
  constructor({ kbPath, latentsPath, pythonBin = 'python3' } = {}) {
    if (!kbPath) throw new Error('kbPath is required');
    this.base   = HOST;
    this._ready = this._spawnAndInit({ kbPath, latentsPath, pythonBin });
  }

  async predict(latent, modality) {
    await this._ready;
    const vec = prepareLatent(latent, modality);

    const { statusCode, body } = await request(
      `${this.base}/predict`,
      {
        method : 'POST',
        body   : Buffer.from(JSON.stringify({ modality, latent: vec })),
        headers: { 'Content-Type': 'application/json' }
      }
    );
    if (statusCode !== 200) throw new Error('predict failed');
    return JSON.parse(await body.text());
  }

  close() { if (this._proc && !this._proc.killed) this._proc.kill(); }

  /* ---------- internals ---------- */
  async _spawnAndInit({ kbPath, latentsPath, pythonBin }) {
    const script = path.join(__dirname, 'prediction_server.py');
    this._proc   = spawn(pythonBin, [script, '--port', String(PORT)],
                         { stdio: ['ignore', 'inherit', 'inherit'] });

    const deadline = Date.now() + 10_000;
    while (true) {
      try {
        const { statusCode } = await request(`${HOST}/health`, { method: 'GET' });
        if (statusCode === 200) break;
      } catch {}
      if (Date.now() > deadline) throw new Error('prediction server init timeout');
      await new Promise(r => setTimeout(r, 200));
    }

    await request(`${HOST}/load`, {
      method : 'POST',
      body   : Buffer.from(JSON.stringify({
        kb  : kbPath,
        lat : latentsPath ?? path.join(path.dirname(kbPath), 'latents.bin')
      })),
      headers: { 'Content-Type': 'application/json' }
    });
  }
}

module.exports = PredictionController;
