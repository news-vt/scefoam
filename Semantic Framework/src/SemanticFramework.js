/* SemanticFramework.js – SIMD search + append-only store + multi-modal support */

import fs from 'fs';
import path from 'path';
import sdot from '@stdlib/blas-base-sdot';
import ImageCodec from './image_controller.js';
import AudioCodec from './audio_controller.js';
import TextCodec from './text_controller.js';
import TransformerModel from './model.js';

/* ─── knobs ─── */
const LATENT_FILE = 'latents.bin';
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
    model,
    imgDir = path.join(process.cwd(), '__tmp')
  } = {}) {
    if (!model) throw new Error('model instance required');

    fs.mkdirSync(imgDir, { recursive: true });

    this.imgCodec = new ImageCodec();
    this.audioCodec = new AudioCodec();
    this.textCodec = new TextCodec();
    this.model = model;
    this.imgDir = imgDir;

    this.kb = new Map();      // hex -> {offset,length}
    this.unitMap = new Map();      // hex -> normalized vector
    this.kbPath = kbPath;

    const dir = path.dirname(kbPath);
    fs.mkdirSync(dir, { recursive: true });
    this.latentFD = fs.openSync(path.join(dir, LATENT_FILE), 'a+');

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

    if (this.kbPath) this.model.kbPath = this.kbPath;
    this._ops = 0;

    this.tokenCache = new Map();
  }

  /* ------ vector helpers ------ */
  vectorize(tok) {
    // If we've already got a latent vector, just return it
    if (Array.isArray(tok)) return tok;

    // ---- image file ---------------------------------------------------------
    if (isImageToken(tok)) {
      const fp = path.join(this.imgDir, tok.slice(5, -1));
      return this.imgCodec.embed(fp);     // raw latent array
    }

    // ---- audio file ---------------------------------------------------------
    if (isAudioToken(tok)) {
      const fp = path.join(this.imgDir, tok.slice(7, -1));
      return this.audioCodec.embed(fp);
    }

    // ---- text file ----------------------------------------------------------
    if (isTextToken(tok)) {
      const fp = path.join(this.imgDir, tok.slice(6, -1));
      const text = fs.readFileSync(fp, "utf8");

      // NEW: naïve sentence split → [ "s1.", "s2.", … ]
      const sents = text
        .split(/(?<=[.!?])\s+/)   // keep terminators, break on whitespace
        .map(s => s.trim())
        .filter(Boolean);

      // return this.textCodec.embedMany(sents);   // (N × 1024) matrix
      return this.textCodec.embed(text);   // (N × 1024) matrix

    }

    // ---- plain text token ---------------------------------------------------
    return this.textCodec.embed(tok);  // treat tok itself as a sentence
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

  /* ------ storage helpers ------ */
  // _add(hex, raw) {
  //   const flat = Array.isArray(raw[0]) ? raw.flat() : raw;
  //   this.kb.set(hex, appendRaw(this.latentFD, flat));
  //   this.unitMap.set(hex, toUnitF32(flat));
  // }

  /** Store a latent in the KB and return the record object */
  _add(hex, raw, mod = 'unknown') {
    const isMatrix = Array.isArray(raw[0]);          // text = matrix
    const flat = isMatrix ? raw.flat() : raw;
    const rec = appendRaw(this.latentFD, flat); // { offset, length }

    /* enrich with modality (+ shape for text) */
    const record = { ...rec, mod };      // ← fixed spread-operator typo
    if (isMatrix) {
      record.rows = raw.length;
      record.cols = raw[0].length;       // ← NEW: remember true width
    }

    this.kb.set(hex, record);
    this.unitMap.set(hex, toUnitF32(flat));
    return record;
  }


  _register(raws) {
    return raws.map(r => {
      const h = nextHex(this.kb.keys());
      this._add(h, r);
      return h;
    });
  }

  /* ───────── Teacher side ───────── */
  send(tokens, thr = 0.999) {
    const payload = [];

    for (const tok of tokens) {
      /* fast path: exact token already sent once → hex-only */
      if (this.tokenCache.has(tok)) {
        payload.push([this.tokenCache.get(tok)]);
        continue;
      }

      /* embed & (maybe) reuse existing vector */
      const raw = this.vectorize(tok);
      const unit = toUnitF32(raw);

      /* detect modality once */
      const mod = isImageToken(tok)
        ? 'image'
        : isAudioToken(tok)
          ? 'audio'
          : 'text';

      /* try vector match in KB */
      let hex = this._closest(unit, thr);
      let pkt;

      if (hex) {                               // latent already known
        pkt = [hex];                           // → slim packet
      } else {                                 // brand-new latent
        hex = nextHex(this.kb.keys());
        this._add(hex, raw, mod);              // store with modality
        pkt = [hex, raw, mod];                 // → full triplet
      }

      /* remember for next time */
      this.tokenCache.set(tok, hex);
      this.hexHistory.push(hex);
      payload.push(pkt);
    }

    this._persist(++this._ops);
    return payload;                            // [[hex] | [hex,latent,mod], …]
  }


  /* ---------------- Apprentice ---------------- */
  async receive(packet) {
    const stamp = new Date().toISOString().slice(0, 16).replace('T', ' ');
    const outs = [];

    for (const item of packet) {
      /* ---------- triplet [hex, latent, mod] ---------- */
      if (Array.isArray(item) && item.length === 3) {
        const [hex, raw, mod] = item;
        this._add(hex, raw, mod);                 // keep modality
        try { outs.push(await this._decodeByMod(raw, mod)); }
        catch { }
        this.receivedData.push(`[${stamp}] ${hex} (${mod})`);
        continue;
      }

      /* ---------- hex-only [hex] ---------- */
      if (Array.isArray(item) && item.length === 1) {
        const hex = item[0];
        const rec = this.kb.get(hex);            // {offset,len,mod,rows?}
        if (!rec) continue;

        const flat = readRaw(this.latentFD, rec);

        /* restore shape for text: rows × 1024 */
        let latent = flat;
        if (rec.mod === 'text' && rec.rows && rec.cols) {
          const { rows, cols } = rec;
          latent = Array.from({ length: rows },
            (_, r) => flat.slice(r * cols, (r + 1) * cols));
        }
        try { outs.push(await this._decodeByMod(latent, rec.mod)); }
        catch { }
        this.receivedData.push(`[${stamp}] ${hex} (${rec.mod})`);
      }
    }

    this.hexHistory.push(...packet.map(p => Array.isArray(p) ? p[0] : null));
    this._persist(++this._ops);
    return outs;
  }

  /* helper inside the class */
  _decodeByMod(raw, mod) {
    if (mod === 'image') return this.imgCodec.decode(raw);
    if (mod === 'audio') return this.audioCodec.decode(raw);
    /* default or 'text' */
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
  /* misc */
  async decodeAsync(item) {
    if (!Array.isArray(item))
      throw new Error("decodeAsync(): invalid item");

    // ── Triplet form ─────────────────────────────────────────────────────────
    if (item.length === 3 && typeof item[2] === "string") {
      const [, raw, mod] = item;

      if (mod === "image") return this.imgCodec.decode(raw);
      if (mod === "audio") return this.audioCodec.decode(raw);

      if (mod === "text") {
        const outArr = await this.textCodec.decode(raw);   // ["s1", "s2", …]
        const joined = outArr.join(" ");
        return Buffer.from(joined, "utf8");
      }
      throw new Error(`decodeAsync(): unknown modality ${mod}`);
    }

    // ── Raw-only  (figure out which modality) ───────────────────────────────
    const raw = item;

    // image latent has fixed length
    if (raw.length === this.imgCodec.latentLength + 2)
      return this.imgCodec.decode(raw);

    // if first element is an array → it's a matrix from text
    if (Array.isArray(raw[0])) {
      const outArr = await this.textCodec.decode(raw); // ["…"]
      return Buffer.from(outArr.join(" "), "utf8");
    }

    // otherwise try single-sentence embedding → text, and fall back to audio
    try {
      /* TextCodec may return *array* (["s1", …]) *or* plain string.   */
      const any = await this.textCodec.decode(raw);
      const arr = Array.isArray(any) ? any : [any];      // ← normalize
      return Buffer.from(arr.join(" "), "utf8");
    } catch {
      /* only if text-decode truly fails, fall back to audio */
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
