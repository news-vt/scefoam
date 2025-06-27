/* eslint-env jest */
/* eslint-disable no-console */
const fs   = require('fs');
const path = require('path');
const os   = require('os');
const SemanticFramework = require('../../SemanticFramework').default;

jest.setTimeout(120_000);

/* paths & dirs ------------------------------------------------------- */
const rootDir   = path.resolve(__dirname, '..');
const dataDir   = path.join(rootDir, '__test_data__', 'audio');
const resultDir = path.join(rootDir, '__result__');
const publicDir = path.join(rootDir, '__test_public__');
const tmpDir    = fs.mkdtempSync(path.join(os.tmpdir(), 'aud_comm_'));
const kbPath    = path.join(publicDir, 'kb_audio_comm.json');

[resultDir, publicDir].forEach(d => fs.mkdirSync(d, { recursive: true }));
fs.writeFileSync(kbPath, '{}');

if (!fs.existsSync(dataDir)) {
  console.warn(`⚠️  ${dataDir} missing – skipping audio_comm tests`);
  describe.skip('audio_comm fixtures missing', () => {
    test('skipped', () => {});
  });
} else {
  fs.readdirSync(dataDir).forEach(f =>
    fs.copyFileSync(path.join(dataDir, f), path.join(tmpDir, f))
  );

  const sf = new SemanticFramework({ kbPath, imgDir: tmpDir });

  const commRows = [];
  const lookupRows = [];
  const reprRows = [];
  let   pktId = 0;

  const symbolCount = token =>
    sf.symbol_count ? sf.symbol_count(token) : 1000;  // fallback

  for (const file of fs.readdirSync(dataDir).filter(f => f.match(/\.(wav|flac)$/i))) {
    test(`send ${file}`, async () => {
      const token   = `<audio:${file}>`;
      const payload = sf.send([token]);

      payload.forEach(pkt =>
        commRows.push({ packet: ++pktId,
                        bytes: Buffer.byteLength(JSON.stringify(pkt)) })
      );

      const vec         = sf.encode_vec(token);
      const reprBits    = vec.length * 32;
      const srcBytes    = fs.statSync(path.join(dataDir, file)).size;
      const contentNats = srcBytes * 8 * Math.LN2;

      [
        { variant:'classical',   reprBits:srcBytes*8, packets:1 },
        { variant:'semanticAll', reprBits, packets:symbolCount(token) },
        { variant:'proposed',    reprBits, packets:payload.length },
      ].forEach(r =>
        reprRows.push({
          file, variant:r.variant,
          contentNats,
          reprBits : r.reprBits,
          packets  : r.packets,
          impact   : r.packets / r.reprBits
        })
      );

      const t0 = Date.now();
      await sf.receive(payload);
      lookupRows.push({ file, lookupMs: Date.now() - t0 });
    });
  }

  afterAll(() => {
    fs.writeFileSync(
      path.join(resultDir, 'audio_comm.csv'),
      'packet,bytes\n' + commRows.map(r => `${r.packet},${r.bytes}`).join('\n') + '\n'
    );
    fs.writeFileSync(
      path.join(resultDir, 'audio_knowledgebase.csv'),
      'fileName,lookupMs\n' + lookupRows.map(r => `${r.file},${r.lookupMs}`).join('\n') + '\n'
    );
    fs.writeFileSync(
      path.join(resultDir, 'audio_semantic_kpis.csv'),
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
    console.log('\n→ wrote audio_* CSVs');
  });
}
