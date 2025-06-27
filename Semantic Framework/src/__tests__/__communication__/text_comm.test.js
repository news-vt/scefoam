/* eslint-env jest */
/* eslint-disable no-console */
const fs   = require('fs');
const path = require('path');
const os   = require('os');
const SemanticFramework = require('../../SemanticFramework').default;

jest.setTimeout(120_000);

/* ── directories ───────────────────────────────────────────────────── */
const rootDir   = path.resolve(__dirname, '..');
const dataDir   = path.join(rootDir, '__test_data__', 'text');
const resultDir = path.join(rootDir, '__result__');
const publicDir = path.join(rootDir, '__test_public__');
const tmpDir    = fs.mkdtempSync(path.join(os.tmpdir(), 'txt_comm_'));
const kbPath    = path.join(publicDir, 'kb_text_comm.json');

[resultDir, publicDir].forEach(d => fs.mkdirSync(d, { recursive: true }));
fs.writeFileSync(kbPath, '{}');

/* ── guard: skip whole suite if no fixtures ────────────────────────── */
if (!fs.existsSync(dataDir)) {
  console.warn(`⚠️  ${dataDir} missing – skipping text_comm tests`);
  describe.skip('text_comm fixtures missing', () => {
    test('skipped', () => {});
  });
} else {
  /* copy fixtures so that <text:FILE> resolves */
  fs.readdirSync(dataDir).forEach(f =>
    fs.copyFileSync(path.join(dataDir, f), path.join(tmpDir, f))
  );

  const sf = new SemanticFramework({ kbPath, imgDir: tmpDir });

  const commRows   = [];   // packet-size log
  const lookupRows = [];   // KB-latency log
  const reprRows   = [];   // KPI rows (all three variants)
  let   pktId      = 0;

  /* helper returns # of source symbols (fallback = word count) */
  const symbolCount = token =>
    sf.symbol_count ? sf.symbol_count(token)
                    : fs.readFileSync(path.join(dataDir, token.slice(6,-1)), 'utf8')
                         .trim().split(/\s+/).length;

  for (const file of fs.readdirSync(dataDir).filter(f => f.endsWith('.txt'))) {
    test(`send ${file}`, async () => {
      const token   = `<text:${file}>`;
      const payload = sf.send([token]);

      /* 1 ▶ packet sizes (proposed) ---------------------------------- */
      payload.forEach(pkt =>
        commRows.push({ packet: ++pktId,
                        bytes: Buffer.byteLength(JSON.stringify(pkt)) })
      );

      /* 2 ▶ representation vector, src size, complexity -------------- */
      const vec         = sf.encode_vec(token);
      const reprBits    = vec.length * 32;
      const srcBytes    = fs.statSync(path.join(dataDir, file)).size;
      const contentNats = srcBytes * 8 * Math.LN2;
      const packetsProp = payload.length;

      /* 3 ▶ push three KPI rows -------------------------------------- */
      const rows = [
        { variant: 'classical',
          reprBits: srcBytes * 8,
          packets : 1 },
        { variant: 'semanticAll',
          reprBits,
          packets : symbolCount(token) },
        { variant: 'proposed',
          reprBits,
          packets : packetsProp },
      ];
      rows.forEach(r =>
        reprRows.push({
          file,
          variant:     r.variant,
          contentNats,
          reprBits:    r.reprBits,
          packets:     r.packets,
          impact:      r.packets / r.reprBits
        })
      );

      /* 4 ▶ KB-lookup time (optional) -------------------------------- */
      const t0 = Date.now();
      await sf.receive(payload);
      lookupRows.push({ file, lookupMs: Date.now() - t0 });
    });
  }

  /* ── write CSVs after suite ──────────────────────────────────────── */
  afterAll(() => {
    fs.writeFileSync(
      path.join(resultDir, 'text_comm.csv'),
      'packet,bytes\n' +
        commRows.map(r => `${r.packet},${r.bytes}`).join('\n') + '\n'
    );
    fs.writeFileSync(
      path.join(resultDir, 'text_knowledgebase.csv'),
      'fileName,lookupMs\n' +
        lookupRows.map(r => `${r.file},${r.lookupMs}`).join('\n') + '\n'
    );
    fs.writeFileSync(
      path.join(resultDir, 'text_semantic_kpis.csv'),
      'fileName,variant,contentNats,reprBits,packets,impact\n' +
        reprRows.map(r =>
          [r.file,
           r.variant,
           r.contentNats.toExponential(3),
           r.reprBits,
           r.packets,
           r.impact.toExponential(3)].join(',')
        ).join('\n') + '\n'
    );
    console.log('\n→ wrote text_* CSVs');
  });
}
