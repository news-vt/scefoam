/* SemanticFramework.js – SIMD search + append-only store + multi-modal support
   Updated: integrates ImageCodec.predict() for next-step image inference    */

import fs from 'fs';
import path from 'path';
import sdot from '@stdlib/blas-base-sdot';
import ImageCodec from './image_controller.js';
import AudioCodec from './audio_controller.js';
import TextCodec from './text_controller.js';

/* ─── knobs ─── */
// const LATENT_FILE   = 'latents.bin';
const PERSIST_EVERY = 0;

/* ─── helpers ─── */
const cosSim = (A, B) => sdot(A.length, A, 1, B, 1);
const nextHex = keys => {
  let m = -1;
  for (const k of keys) m = Math.max(m, parseInt(k.slice(1), 16));
  return `#${(m + 1).toString(16).padStart(8, '0')}`;
};
const toUnitF32 = v => {
  const f32 = Float32Array.from(v);
  let n2 = 0;
  for (let i = 0; i < f32.length; i++) n2 += f32[i] * f32[i];
  if (n2) {
    const s = 1 / Math.sqrt(n2);
    for (let i = 0; i < f32.length; i++) f32[i] *= s;
  }
  return f32;
};
const appendRaw = (fd, raw) => {
  const buf = Buffer.from(new Float32Array(raw).buffer);
  const offset = fs.fstatSync(fd).size;
  fs.writeSync(fd, buf, 0, buf.length, offset);
  return { offset, length: raw.length };
};
const readRaw = (fd, { offset, length }) => {
  const buf = Buffer.allocUnsafe(length * 4);
  fs.readSync(fd, buf, 0, buf.length, offset);
  return Array.from(new Float32Array(buf.buffer));
};
const isImageToken = tok => typeof tok === 'string' && tok.startsWith('<img:');
const isAudioToken = tok => typeof tok === 'string' && tok.startsWith('<audio:');
const isTextToken = tok => typeof tok === 'string' && tok.startsWith('<text:');

/* ─── class ─── */
export default class SemanticFramework {
  constructor({
    kb = null,
    kbPath = 'knowledge_base.json',
    model = null,
    imgDir = path.join(process.cwd(), '__tmp')
  } = {}) {
    fs.mkdirSync(imgDir, { recursive: true });

    /* per-modality codecs (each now handles its own training + prediction) */
    this.imgCodec = new ImageCodec();
    this.audioCodec = new AudioCodec();
    this.textCodec = new TextCodec();

    this.model = model;          // optional external language / meta model
    this.imgDir = imgDir;

    this.kb = new Map();    // hex -> {offset,length,…}
    this.unitMap = new Map();    // hex -> normalized vector
    this.kbPath = kbPath;

    /* ── derive latent-blob filename from kbPath ─────────────────────── */
    const kbDir = path.dirname(kbPath);
    const kbBase = path.basename(kbPath, path.extname(kbPath));
    const latentFile = `${kbBase}_latents.bin`;

    fs.mkdirSync(kbDir, { recursive: true });
    this.latentFD = fs.openSync(path.join(kbDir, latentFile), 'a+');

    if (fs.existsSync(kbPath)) {
      const j = JSON.parse(fs.readFileSync(kbPath, 'utf8'));
      for (const [h, idx] of Object.entries(j.map || {})) {
        this.kb.set(h, idx);
        this.unitMap.set(h, toUnitF32(readRaw(this.latentFD, idx)));
      }
      this.hexHistory = j.hexHistory ?? [];
      this.receivedData = j.receivedData ?? [];
      this.predictedText = j.predictedText ?? '';
      this.modelReady = j.modelReady ?? false;
    } else {
      this.hexHistory = [];
      this.receivedData = [];
      this.predictedText = '';
      this.modelReady = false;
    }

    /* Tell any external model where persistence files live */
    if (model && typeof model === 'object') {
      model.kbPath = this.kbPath;
      model.latentsPath = path.join(kbDir, latentFile);
    }

    this._ops = 0;
    this.tokenCache = new Map();
  }

  async predict(src, modality = "text", opts = {}) {
    /* —— image ───────────────────────────────────────── */
    if (modality === "image") {
      const latent = Array.isArray(src) ? src : this.vectorize(src);
      return this.imgCodec.predict(latent, !!opts.wantJPEG);
    }

    /* —— audio ───────────────────────────────────────── */
    if (modality === "audio") {
      const latent = Array.isArray(src) ? src : this.vectorize(src);
      return this.audioCodec.predict(latent, !!opts.wantWav);
    }

    /* —— text ────────────────────────────────────────── */
    if (modality === "text") {
      const latent = Array.isArray(src) ? src : this.vectorize(src);
      return this.textCodec.predict(latent, !!opts.wantText);
    }

    /* —— fallback to external model (other modalities) ─ */
    if (!this.model || typeof this.model.predict !== "function")
      throw new Error("model lacks predict()");
    const latent = Array.isArray(src) ? src : this.vectorize(src);
    return this.model.predict(latent, modality, opts);
  }


  /* helper: accept a token or latent and always return a flat latent array */
  _latentFor(item) {
    const raw = (typeof item === 'string' ||
      (Array.isArray(item) && typeof item[0] !== 'number'))
      ? this.vectorize(item)
      : item;

    /* deep-flatten and cast typed arrays to plain lists */
    const flat = (function f(v) {
      if (ArrayBuffer.isView(v)) return Array.from(v);
      if (Array.isArray(v)) return v.flatMap(f);
      return [v];
    })(raw);

    return flat.length >= 512
      ? flat.slice(0, 512)
      : flat.concat(Array(512 - flat.length).fill(0));
  }

  /* ------ vector helpers ------ */
  vectorize(tok) {
    /* latent already? */
    if (Array.isArray(tok)) return tok;

    /* ---- image file ------------------------------------------------------- */
    if (isImageToken(tok)) {
      const fp = path.join(this.imgDir, tok.slice(5, -1));
      return this.imgCodec.embed(fp);
    }

    /* ---- audio file ------------------------------------------------------- */
    if (isAudioToken(tok)) {
      const fp = path.join(this.imgDir, tok.slice(7, -1));
      return this.audioCodec.embed(fp);
    }

    /* ---- text file -------------------------------------------------------- */
    if (isTextToken(tok)) {
      const fp = path.join(this.imgDir, tok.slice(6, -1));
      const text = fs.readFileSync(fp, 'utf8');
      return this.textCodec.embed(text);
    }

    /* ---- plain text token ------------------------------------------------- */
    return this.textCodec.embed(tok);
  }

  /* ------ fast cosine search ------ */
  _closest(unit, thr) {
    let best = -1, bHex = null;
    for (const [h, u] of this.unitMap) {
      if (u.length !== unit.length) continue;
      const s = cosSim(u, unit);
      if (s > best) { best = s; bHex = h; }
      if (best >= thr) break;
    }
    return best >= thr ? bHex : null;
  }
  findClosestHexByVector(raw, thr = 0.999) {
    return this._closest(toUnitF32(raw), thr);
  }

  /* … unchanged CRUD / send / receive / decode helpers follow … */
  /* (everything below this comment is identical to your previous
     SemanticFramework.js except for whitespace tweaks) */

  /** Store a latent in the KB and return the record object */
  _add(hex, raw, mod = 'unknown') {
    const isMatrix = Array.isArray(raw[0]);
    const flat = isMatrix ? raw.flat() : raw;
    const rec = appendRaw(this.latentFD, flat);

    const record = { ...rec, mod };
    if (isMatrix) { record.rows = raw.length; record.cols = raw[0].length; }

    this.kb.set(hex, record);
    this.unitMap.set(hex, toUnitF32(flat));

    if (typeof this.model?.refresh === 'function') {
      this.model.refresh().catch(() => { });
    }
    return record;
  }

  _register(raws) {
    return raws.map(r => {
      const h = nextHex(this.kb.keys()); this._add(h, r); return h;
    });
  }

  /* ───────── Teacher side ───────── */
  send(tokens, thr = 0.999) {
    const payload = [];

    for (const tok of tokens) {
      if (this.tokenCache.has(tok)) {
        payload.push([this.tokenCache.get(tok)]);
        continue;
      }

      const raw = this.vectorize(tok);
      const unit = toUnitF32(raw);
      const mod = isImageToken(tok) ? 'image'
        : isAudioToken(tok) ? 'audio' : 'text';

      let hex = this._closest(unit, thr);
      let pkt;

      if (hex) { pkt = [hex]; }
      else {
        hex = nextHex(this.kb.keys());
        this._add(hex, raw, mod);
        pkt = [hex, raw, mod];
      }

      this.tokenCache.set(tok, hex);
      this.hexHistory.push(hex);
      payload.push(pkt);
    }
    this._persist(++this._ops);
    return payload;
  }

  /* ---------------- Apprentice ---------------- */
  async receive(packet) {
    const stamp = new Date().toISOString().slice(0, 16).replace('T', ' ');
    const outs = [];

    for (const item of packet) {
      if (Array.isArray(item) && item.length === 3) {
        const [hex, raw, mod] = item;
        this._add(hex, raw, mod);
        try { outs.push(await this._decodeByMod(raw, mod)); } catch { }
        this.receivedData.push(`[${stamp}] ${hex} (${mod})`);
        continue;
      }

      if (Array.isArray(item) && item.length === 1) {
        const hex = item[0];
        const rec = this.kb.get(hex);
        if (!rec) continue;

        const flat = readRaw(this.latentFD, rec);
        let latent = flat;
        if (rec.mod === 'text' && rec.rows && rec.cols) {
          const { rows, cols } = rec;
          latent = Array.from({ length: rows },
            (_, r) => flat.slice(r * cols, (r + 1) * cols));
        }
        try { outs.push(await this._decodeByMod(latent, rec.mod)); } catch { }
        this.receivedData.push(`[${stamp}] ${hex} (${rec.mod})`);
      }
    }
    this.hexHistory.push(...packet.map(p => Array.isArray(p) ? p[0] : null));
    this._persist(++this._ops);
    return outs;
  }

  /* helper */
  _decodeByMod(raw, mod) {
    if (mod === 'image') return this.imgCodec.decode(raw);
    if (mod === 'audio') return this.audioCodec.decode(raw);
    return this.textCodec.decode(raw);
  }

  /* ---------------- persistence ---------------- */
  _persist(n) {
    if (!this.kbPath) return;
    if (PERSIST_EVERY && n % PERSIST_EVERY) return;
    const j = {
      map: Object.fromEntries(this.kb),
      hexHistory: this.hexHistory,
      receivedData: this.receivedData,
      predictedText: this.predictedText,
      modelReady: this.modelReady
    };
    fs.writeFileSync(this.kbPath, JSON.stringify(j, null, 2));
  }

  /* misc */
  async decodeAsync(item) {
    if (!Array.isArray(item)) throw new Error('decodeAsync(): invalid item');

    if (item.length === 3 && typeof item[2] === 'string') {
      const [, raw, mod] = item;
      if (mod === 'image') return this.imgCodec.decode(raw);
      if (mod === 'audio') return this.audioCodec.decode(raw);
      if (mod === 'text') {
        const arr = await this.textCodec.decode(raw);
        return Buffer.from(arr.join(' '), 'utf8');
      }
      throw new Error(`decodeAsync(): unknown modality ${mod}`);
    }

    const raw = item;
    if (raw.length === this.imgCodec.latentLength + 2)
      return this.imgCodec.decode(raw);

    if (Array.isArray(raw[0])) {
      const arr = await this.textCodec.decode(raw);
      return Buffer.from(arr.join(' '), 'utf8');
    }

    try {
      const any = await this.textCodec.decode(raw);
      const arr = Array.isArray(any) ? any : [any];
      return Buffer.from(arr.join(' '), 'utf8');
    } catch {
      return this.audioCodec.decode(raw);
    }
  }

  clear() {
    this.kb.clear();
    this.unitMap.clear();
    this.hexHistory = [];
    this.receivedData = [];
    fs.truncateSync(this.latentFD, 0);
    this._persist(++this._ops);
  }
}
