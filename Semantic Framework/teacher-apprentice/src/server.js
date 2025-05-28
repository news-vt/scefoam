import express from 'express';
import cors from 'cors';
import bodyParser from 'body-parser';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { execSync } from 'child_process';
import yargs from 'yargs/yargs';
import { hideBin } from 'yargs/helpers';
import { SemanticFramework } from './main.js';

/* ▸ resolve paths & port */
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PORT = yargs(hideBin(process.argv))
  .option('port', {
    alias: 'p', type: 'number',
    default: process.env.PORT ? Number(process.env.PORT) : 3000
  })
  .parse().port;

/* ▸ per-instance files */
const publicDir = path.join(__dirname, '..', 'public');
const kbFile = path.join(publicDir, `knowledge_base_${PORT}.json`);
const tsneFile = path.join(publicDir, `tsne_data_${PORT}.json`);

const clusterScript = path.join(__dirname, 'xmeans_cluster.py');
const transformerScript = path.join(__dirname, 'transformer.py');
const tsneScript = path.join(__dirname, '..', 'compute_tsne.py');

fs.mkdirSync(publicDir, { recursive: true });
if (!fs.existsSync(kbFile))
  fs.writeFileSync(kbFile, JSON.stringify({
    rawData: [], embeddings: [], centroids: {}, centroidLabels: {},
    receivedData: [], centroidHistory: [], futureCentroids: {}
  }, null, 2));
if (!fs.existsSync(tsneFile))
  fs.writeFileSync(tsneFile, JSON.stringify({
    tokens: [], coords: [], centroid_ids: [],
    centroid_labels: [], centroid_coords: []
  }, null, 2));

/* ▸ Express & framework */
const app = express();
app.use(cors());
app.use(bodyParser.json());
app.use(express.static(publicDir));

const sf = new SemanticFramework(
  kbFile,
  path.join(__dirname, 'glove.6B.100d.txt'),
  clusterScript,
  transformerScript
);

/* ▸ t-SNE recompute */
function recomputeTSNE() {
  const tsnePublic = path.join(path.dirname(tsneScript), 'public');
  fs.mkdirSync(tsnePublic, { recursive: true });
  fs.copyFileSync(kbFile, path.join(tsnePublic, 'knowledge_base.json'));
  try {
    execSync(`python "${tsneScript}"`, { stdio: 'ignore' });
    const generic = path.join(tsnePublic, 'tsne_data.json');
    if (fs.existsSync(generic)) fs.copyFileSync(generic, tsneFile);
  } catch { }
}

/* ▸ API */
app.get('/ping', (_q, r) => r.json({ ok: true, port: PORT }));

app.get('/model_status', (_q, r) => r.json({
  ready: sf.getModelReady(),
  snapshots: sf.centroidHistory.length
}));

/* ■ SEND – returns hitIds */
app.post('/send', (req, res) => {
  const { data } = req.body;
  if (!data) return res.status(400).json({ error: 'Missing "data".' });
  try {
    // const { centroids, hitIds } = sf.send(data);
    const {centroids,hitLabels}=sf.send(data);
    recomputeTSNE();
    sf._saveKB();               
    // res.json({ centroids, hitIds });
    res.json({centroids,hitLabels});
  } catch (e) {
    console.error('/send', e);
    res.status(500).json({ error: e.message });
  }
});

/* ■ RECEIVE – accepts {centroids:[ "C1(Nature)", … ]} OR {data:"…"} */
app.post('/receive', (req, res) => {
  const { data, centroids } = req.body;

  /* basic validation */
  const hasText  = typeof data === 'string' && data.trim().length;
  const hasCenters = Array.isArray(centroids) && centroids.length;

  if (!hasText && !hasCenters) {
    return res.status(400).json({ error: 'Missing payload.' });
  }

  try {
    if (hasCenters) {
      /* already of form "C3(Computer)" etc. — log as-is */
      sf.receive(centroids.join(', '));
    } else {
      sf.receive(data.trim());
    }
    recomputeTSNE();
    res.json({ ok: true });
  } catch (e) {
    console.error('/receive', e);
    res.status(500).json({ error: e.message });
  }
});


/* ■ PREDICT */
app.post('/predict', (_q, res) => {
  try { res.json({ centroids: sf.statistical_ref() }); }
  catch (e) {
    console.error('/predict', e);
    res.status(500).json({ error: e.message });
  }
});

/* ■ CLEAR */
app.post('/clear', (_q, res) => {
  fs.writeFileSync(kbFile, JSON.stringify({
    rawData: [], embeddings: [], centroids: {}, centroidLabels: {},
    receivedData: [], centroidHistory: [], futureCentroids: {}
  }, null, 2));
  fs.writeFileSync(tsneFile, JSON.stringify({
    tokens: [], coords: [], centroid_ids: [],
    centroid_labels: [], centroid_coords: []
  }, null, 2));
  Object.assign(sf, {
    rawData: [], embeddings: [], centroids: {}, centroidLabels: {},
    receivedData: [], centroidHistory: [], futureCentroids: {}
  });
  res.json({ ok: true });
});

/* SPA fallback */
app.get('*', (_q, res) => res.sendFile(path.join(publicDir, 'index.html')));

app.listen(PORT, () => console.log(`Dashboard → http://localhost:${PORT}`));
