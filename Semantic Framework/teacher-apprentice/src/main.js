import fs from 'fs';
import path from 'path';
import { execFileSync } from 'child_process';

/* cosine-similarity helper */
function cosine(a,b){
  let d=0,na=0,nb=0;
  for(let i=0;i<a.length;i++){
    d+=a[i]*b[i]; na+=a[i]*a[i]; nb+=b[i]*b[i];
  }
  return d/(Math.sqrt(na)*Math.sqrt(nb)+1e-8);
}

/* ================================================================ */
export class SemanticFramework{
  constructor(
    kbPath='knowledge_base.json',
    glovePath=path.join(process.cwd(),'glove.6B.100d.txt'),
    cluster='xmeans_cluster.py',
    transformer='transformer.py'
  ){
    this.kbPath=kbPath;
    this.clusterScript=cluster;
    this.transformerScript=transformer;
    this.dimension=100;

    /* in-memory state */
    this.glove=new Map();
    this.rawData=[]; this.embeddings=[];
    this.centroids={}; this.centroidLabels={};
    this.receivedData=[];
    this.centroidHistory=[];
    this.futureCentroids={};
    this.modelReady=fs.existsSync('transformer_model.pt');

    /* GloVe load */
    for(const line of fs.readFileSync(glovePath,'utf-8').split(/\r?\n/)){
      const p=line.split(' ');
      if(p.length===101) this.glove.set(p[0],p.slice(1).map(Number));
    }
    this._loadKB();
  }

  /* persistence helpers */
  _loadKB(){
    if(!fs.existsSync(this.kbPath)) return;
    try{
      const j=JSON.parse(fs.readFileSync(this.kbPath,'utf-8'));
      Object.assign(this,{
        rawData:j.rawData||[], embeddings:j.embeddings||[],
        centroids:j.centroids||{}, centroidLabels:j.centroidLabels||{},
        receivedData:j.receivedData||[],
        centroidHistory:j.centroidHistory||[],
        futureCentroids:j.futureCentroids||{}
      });
    }catch{}
  }
  _saveKB(){
    fs.writeFileSync(this.kbPath,JSON.stringify({
      rawData:this.rawData, embeddings:this.embeddings,
      centroids:this.centroids, centroidLabels:this.centroidLabels,
      receivedData:this.receivedData,
      centroidHistory:this.centroidHistory,
      futureCentroids:this.futureCentroids
    },null,2),'utf-8');
  }

/* ── SEND (embed → cluster → label → hit-set) ─────────────────────── */
send (text) {
  /* -------------------------------------------------- 1. tokenise --- */
  const toks = text
    .toLowerCase()
    .replace(/[^a-z0-9_]+/g, ' ')
    .split(/\s+/)
    .filter(Boolean);

  /* -------------------------------------------------- 2. extend KB --- */
  for (const t of toks) {
    if (!this.rawData.includes(t)) {
      this.rawData.push(t);
      this.embeddings.push(
        this.glove.get(t) ?? Array(this.dimension).fill(0)
      );
    }
  }
  this._saveKB();
  if (!this.embeddings.length) {              // nothing to cluster yet
    return { centroids: this.centroids, hitIds: [], hitLabels: [] };
  }

  /* ------------------------------------------- 3. re-cluster KB --- */
  const out = execFileSync(
    'python',
    [this.clusterScript, this.kbPath],
    { encoding: 'utf-8', maxBuffer: 32 * 1024 * 1024 }
  ).trim();

  const r = JSON.parse(out);
  if (r.error) throw new Error(r.error);

  /* ------------------------------------------------ 4. rebuild maps --- */
  this.centroids = {};
  r.centroids.forEach((vec, i) => { this.centroids[i + 1] = vec; });

  this.centroidLabels = {};
  for (const [cid, cvec] of Object.entries(this.centroids)) {
    let best = '', bestSim = -Infinity;
    for (let i = 0; i < this.rawData.length; i++) {
      const sim = cosine(cvec, this.embeddings[i]);
      if (sim > bestSim) { bestSim = sim; best = this.rawData[i]; }
    }
    this.centroidLabels[cid] = best || `C${cid}`;
  }

  /* ----------------------------- 5. map current message → centroids --- */
  const hit = new Set();                      // Set of centroid-IDs hit
  for (const t of toks) {
    const idx = this.rawData.indexOf(t);
    if (idx === -1) continue;                 // safety
    const emb = this.embeddings[idx];

    let bestId = null, bestSim = -Infinity;
    for (const [cid, cvec] of Object.entries(this.centroids)) {
      const sim = cosine(cvec, emb);
      if (sim > bestSim) { bestSim = sim; bestId = cid; }
    }
    if (bestId) hit.add(bestId);
  }

  const hitIds    = [...hit];                         // e.g. [ '1', '5' ]
  const hitLabels = hitIds.map(
    id => `C${id}(${this.centroidLabels[id]})`
  );                                                  // e.g. [ 'C1(Nature)' … ]

  /* ------------------------------------------------ 6. transformer --- */
  this.centroidHistory.push(JSON.parse(JSON.stringify(this.centroids)));
  if (this.centroidHistory.length >= 2) {
    try {
      execFileSync(
        'python',
        [this.transformerScript, 'train', this.kbPath],
        { stdio: 'ignore', maxBuffer: 32 * 1024 * 1024 }
      );
      this.modelReady = true;
    } catch (e) {
      console.warn('[SF] transformer training failed:', e.message);
      this.modelReady = false;
    }
  }

  /* ------------------------------------------------ 7. persist & out --- */
  this._saveKB();
  return {
    centroids : this.centroids,   // full table for the local dashboard
    hitIds,                       // raw centroid IDs             (optional)
    hitLabels                      // de-duplicated pretty names   ← **used**
  };
}


  /* ── RECEIVE (teacher → apprentice) ─────────────── */
  receive(payload){
    const now=new Date();
    const stamp=`[${
      now.toLocaleDateString('en-US',{month:'2-digit',day:'2-digit',year:'numeric'})} ${
      now.toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit',hour12:false})}]`;
    this.receivedData.push(`${stamp} ${payload}`);
    this._saveKB();
  }

  /* ── PREDICT ───────────────────────────────────── */
  statistical_ref(){
    if(!this.modelReady) throw new Error('Transformer not trained yet.');
    const out=execFileSync('python',
      [this.transformerScript,'predict',this.kbPath],
      {encoding:'utf-8',maxBuffer:32*1024*1024}).trim();
    const r=JSON.parse(out); if(r.error) throw new Error(r.error);

    /* keep only IDs we already recognise */
    const recognised={};
    for(const [id,vec] of Object.entries(r.centroids))
      if(this.centroidLabels[id]) recognised[id]=vec;

    this.futureCentroids=recognised;
    this._saveKB();
    return this.futureCentroids;
  }

  /* getters */
  getCentroids()       {return this.centroids;}
  getRawData()         {return this.rawData;}
  getLabels()          {return this.centroidLabels;}
  getReceivedData()    {return this.receivedData;}
  getFutureCentroids() {return this.futureCentroids;}
  getModelReady()      {return this.modelReady;}
}
