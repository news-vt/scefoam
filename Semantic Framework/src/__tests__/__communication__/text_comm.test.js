const fs     = require("fs");
const path   = require("path");
const os     = require("os");
const SemanticFramework = require("../../SemanticFramework").default;

jest.setTimeout(180_000);

const rootDir   = path.resolve(__dirname, ".."); 
const dataDir   = path.join(rootDir, "__test_data__", "text");
const resultDir = path.join(rootDir, "__result__");
const publicDir = path.join(rootDir, "__test_public__");
const tmpSend   = fs.mkdtempSync(path.join(os.tmpdir(), "txt_sem_tx_"));
const tmpRecv   = fs.mkdtempSync(path.join(os.tmpdir(), "txt_sem_rx_"));

[resultDir, publicDir].forEach(d => fs.mkdirSync(d, { recursive: true }));

for (const f of fs.readdirSync(dataDir).filter(f => f.endsWith(".txt"))) {
  fs.copyFileSync(path.join(dataDir, f), path.join(tmpSend, f));
}

const senderKB = path.join(publicDir, "kb_txt_sender.json");
const recvKB   = path.join(publicDir, "kb_txt_receiver.json");
fs.writeFileSync(senderKB, "{}");
fs.writeFileSync(recvKB, "{}");

const sender = new SemanticFramework({ kbPath: senderKB, imgDir: tmpSend });
const recv   = new SemanticFramework({ kbPath: recvKB });

const commRows   = [];             // { packet, bytes }
const lookupRows = [];             // { file, ms  }

const hrMs = (t0, t1) => Number(t1 - t0) / 1e6;

let pktId = 0;

for (const file of fs.readdirSync(dataDir).filter(f => f.endsWith(".txt"))) {
  test(`send ${file}`, async () => {
    const tok     = `<text:${file}>`;
    const payload = sender.send([tok]);

    for (const pkt of payload) {
      const bytes = Buffer.byteLength(JSON.stringify(pkt));
      commRows.push({ packet: ++pktId, bytes });
    }
    await recv.receive(payload);     
  });
}

for (const file of fs.readdirSync(dataDir).filter(f => f.endsWith(".txt"))) {
  test(`lookup ${file}`, () => {
    const vec = sender.encode_vec(`<text:${file}>`);
    const t0  = process.hrtime.bigint();
    recv.findClosestHexByVector(vec);   
    const t1  = process.hrtime.bigint();
    lookupRows.push({ file, ms: hrMs(t0, t1).toFixed(2) });
  });
}

afterAll(() => {
  const commCsv = "packet,bytes\n" +
    commRows.map(r => `${r.packet},${r.bytes}`).join("\n") + "\n";
  fs.writeFileSync(path.join(resultDir, "text_comm.csv"), commCsv, "utf8");

  const kbCsv = "fileName,lookupMs\n" +
    lookupRows.map(r => `${r.file},${r.ms}`).join("\n") + "\n";
  fs.writeFileSync(
    path.join(resultDir, "text_knowledgebase.csv"),
    kbCsv,
    "utf8"
  );

  console.log(
    `\n→ Wrote ${commRows.length} packet sizes to text_comm.csv ` +
    `and ${lookupRows.length} KB timings to text_knowledgebase.csv\n`
  );
});
