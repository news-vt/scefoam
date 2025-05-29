// File: src/main.js
import fs               from 'fs';
import path             from 'path';
import { execFileSync } from 'child_process';

/* ───────────────── helpers ─────────────────────────────────────── */
function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];                 // a·b
    na  += a[i] * a[i];                 // |a|²
    nb  += b[i] * b[i];                 // |b|²
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
}

/* ───────────────── SemanticFramework ───────────────────────────── */
export class SemanticFramework {
  constructor(
    kbPath      = 'knowledge_base.json',
    glovePath   = path.join(process.cwd(), 'glove.6B.100d.txt'),
    cluster     = 'xmeans_cluster.py',
    transformer = 'transformer.py'
  ) {
    /* paths & port-scoping ---------------------------------------- */
    this.kbPath        = kbPath;
    this.clusterScript = cluster;
    this.transformerPy = transformer;

    const m     = kbPath.match(/_(\d+)\.json$/);   // “…_4500.json”
    this.port   = m ? m[1] : '';                  // "" → teacher
    this.modelPath = this.port
      ? `transformer_model_${this.port}.pt`
      : `transformer_model.pt`;

    /* static parameters ------------------------------------------- */
    this.dimension = 100;                         // GloVe dim

    /* in-memory state --------------------------------------------- */
    this.glove           = new Map();             // word → vec
    this.rawData         = [];
    this.embeddings      = [];
    this.centroids       = {};
    this.centroidLabels  = {};
    this.receivedData    = [];
    this.centroidHistory = [];
    this.futureCentroids = {};
    this.modelReady      = fs.existsSync(this.modelPath);

    /* load GloVe --------------------------------------------------- */
    for (const ln of fs.readFileSync(glovePath, 'utf-8').split(/\r?\n/)) {
      const p = ln.split(' ');
      if (p.length === 101) this.glove.set(p[0], p.slice(1).map(Number));
    }

    this._loadKB();          // hydrate from disk
  }

  /* ── persistence helpers ──────────────────────────────────────── */
  _loadKB() {
    if (!fs.existsSync(this.kbPath)) return;
    try {
      const j = JSON.parse(fs.readFileSync(this.kbPath, 'utf-8'));
      Object.assign(this, {
        rawData        : j.rawData        ?? [],
        embeddings     : j.embeddings     ?? [],
        centroids      : j.centroids      ?? {},
        centroidLabels : j.centroidLabels ?? {},
        receivedData   : j.receivedData   ?? [],
        centroidHistory: j.centroidHistory?? [],
        futureCentroids: j.futureCentroids?? {}
      });
    } catch {/* corrupt JSON → start fresh */}
  }

  _saveKB() {
    fs.writeFileSync(
      this.kbPath,
      JSON.stringify({
        rawData        : this.rawData,
        embeddings     : this.embeddings,
        centroids      : this.centroids,
        centroidLabels : this.centroidLabels,
        receivedData   : this.receivedData,
        centroidHistory: this.centroidHistory,
        futureCentroids: this.futureCentroids
      }, null, 2),
      'utf-8'
    );
  }

  /* ── SEND (tokenise → embed → cluster) ────────────────────────── */
  send(text) {
    /* 1. tokenise */
    const toks = text.toLowerCase()
                     .replace(/[^a-z0-9_]+/g, ' ')
                     .split(/\s+/).filter(Boolean);

    /* 2. add new words + embeddings */
    for (const t of toks)
      if (!this.rawData.includes(t)) {
        this.rawData.push(t);
        this.embeddings.push(this.glove.get(t) ?? Array(this.dimension).fill(0));
      }
    this._saveKB();

    if (!this.embeddings.length)
      return { centroids: this.centroids, hitIds: [], hitLabels: [] };

    /* 3. recluster whole KB  */
    const rawOut = execFileSync(
      'python',
      [this.clusterScript, this.kbPath],
      { encoding: 'utf-8', maxBuffer: 32e6 }
    ).trim();
    const { centroids: centers, error } = JSON.parse(rawOut);
    if (error) throw new Error(error);

    this.centroids = {};
    centers.forEach((v, i) => { this.centroids[i + 1] = v; });

    /* 4. label each centroid by nearest token */
    this.centroidLabels = {};
    for (const [cid, cvec] of Object.entries(this.centroids)) {
      let best = '', bestSim = -Infinity;
      for (let i = 0; i < this.rawData.length; i++) {
        const sim = cosine(cvec, this.embeddings[i]);
        if (sim > bestSim) { bestSim = sim; best = this.rawData[i]; }
      }
      this.centroidLabels[cid] = best || `C${cid}`;
    }

    /* 5. determine which centroids this message hit */
    const hit = new Set();
    for (const t of toks) {
      const idx = this.rawData.indexOf(t); if (idx === -1) continue;
      const emb = this.embeddings[idx];

      let bestId = null, bestSim = -Infinity;
      for (const [cid, cvec] of Object.entries(this.centroids)) {
        const sim = cosine(cvec, emb);
        if (sim > bestSim) { bestSim = sim; bestId = cid; }
      }
      if (bestId) hit.add(bestId);
    }
    const hitIds    = [...hit];
    const hitLabels = hitIds.map(id => `C${id}(${this.centroidLabels[id]})`);

    /* 6. snapshot & (re)train transformer when ≥2 snapshots */
    this.centroidHistory.push(JSON.parse(JSON.stringify(this.centroids)));
    if (this.centroidHistory.length >= 2) {
      try {
        execFileSync(
          'python',
          [this.transformerPy, 'train', this.kbPath],
          {
            stdio   : 'ignore',
            maxBuffer: 32e6,
            env     : { ...process.env, PORT_ARG: this.port }
          }
        );
        this.modelReady = true;
      } catch (e) {
        console.warn('[SF] transformer training failed:', e.message);
        this.modelReady = false;
      }
    }

    this._saveKB();
    return { centroids: this.centroids, hitIds, hitLabels };
  }

  /* ── RECEIVE: just append readable log line ───────────────────── */
  receive(line) {
    const ts = new Date().toISOString().replace('T', ' ').slice(0, 19);
    this.receivedData.push(`[${ts}] ${line}`);
    this._saveKB();
  }

  /* ── ONE-STEP PREDICTION ──────────────────────────────────────── */
  statistical_ref() {
    if (!this.modelReady)
      throw new Error('Transformer not trained yet.');

    let raw;
    try {
      raw = execFileSync(
        'python',
        [this.transformerPy, 'predict', this.kbPath],
        {
          encoding : 'utf-8',
          maxBuffer: 32e6,
          env      : { ...process.env, PORT_ARG: this.port }
        }
      ).trim();
    } catch (e) {
      const msg = (e.stdout || e.stderr || '').trim() || e.message;
      throw new Error(`Predict failed: ${msg}`);
    }

    let res;
    try { res = JSON.parse(raw); }
    catch { throw new Error(`Predict returned non-JSON: ${raw || '<empty>'}`); }

    if (res.error) throw new Error(res.error);

    /* keep only IDs we recognise */
    const next = {};
    for (const [cid, vec] of Object.entries(res.centroids || {}))
      if (this.centroidLabels[cid]) next[cid] = vec;

    this.futureCentroids = next;
    this._saveKB();
    return next;
  }

  /* getters */
  getCentroids       () { return this.centroids;        }
  getRawData         () { return this.rawData;          }
  getLabels          () { return this.centroidLabels;   }
  getReceivedData    () { return this.receivedData;     }
  getFutureCentroids () { return this.futureCentroids;  }
  getModelReady      () { return this.modelReady;       }
}
