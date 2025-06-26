const fs         = require("fs");
const path       = require("path");
const os         = require("os");
const SemanticFramework = require("../../SemanticFramework").default;

jest.setTimeout(180_000);

const rootDir   = path.resolve(__dirname, "..");     
const dataDir   = path.join(rootDir, "__test_data__", "images");
const resultDir = path.join(rootDir, "__result__");
const publicDir = path.join(rootDir, "__test_public__");
const tmpSend   = fs.mkdtempSync(path.join(os.tmpdir(), "img_sem_tx_"));
const tmpRecv   = fs.mkdtempSync(path.join(os.tmpdir(), "img_sem_rx_"));

[resultDir, publicDir].forEach(d => fs.mkdirSync(d, { recursive: true }));

for (const f of fs.readdirSync(dataDir).filter(f => /\.jpe?g$/i.test(f))) {
  fs.copyFileSync(path.join(dataDir, f), path.join(tmpSend, f));
}

const senderKB  = path.join(publicDir, "kb_img_sender.json");
const recvKB    = path.join(publicDir, "kb_img_receiver.json");
fs.writeFileSync(senderKB, "{}");
fs.writeFileSync(recvKB, "{}");

const sender = new SemanticFramework({ kbPath: senderKB, imgDir: tmpSend });
const recv   = new SemanticFramework({ kbPath: recvKB });

const commRows   = [];      // { packet, bytes }
const lookupRows = [];      // { file, ms }

const ms = (start, end) => Number(end - start) / 1e6;

let packetCounter = 0;

for (const file of fs.readdirSync(dataDir).filter(f => /\.jpe?g$/i.test(f))) {
  test(`send ${file}`, async () => {
    const tok = `<img:${file}>`;

    const payload = sender.send([tok]);

    for (const pkt of payload) {
      const bytes = Buffer.byteLength(JSON.stringify(pkt));
      commRows.push({ packet: ++packetCounter, bytes });
    }

    await recv.receive(payload);
  });
}

for (const file of fs.readdirSync(dataDir).filter(f => /\.jpe?g$/i.test(f))) {
  test(`lookup ${file}`, () => {
    const tok   = `<img:${file}>`;
    const vec   = sender.encode_vec(tok);          
    const t0    = process.hrtime.bigint();
    recv.findClosestHexByVector(vec);             
    const t1    = process.hrtime.bigint();
    lookupRows.push({ file, ms: ms(t0, t1).toFixed(2) });
  });
}

afterAll(() => {
  const commCsv =
    "packet,bytes\n" +
    commRows.map(r => `${r.packet},${r.bytes}`).join("\n") + "\n";
  fs.writeFileSync(path.join(resultDir, "image_comm.csv"), commCsv, "utf8");

  const kbCsv =
    "fileName,lookupMs\n" +
    lookupRows.map(r => `${r.file},${r.ms}`).join("\n") + "\n";
  fs.writeFileSync(path.join(resultDir, "image_knowledgebase.csv"), kbCsv, "utf8");

  console.log(
    `\n→ Wrote ${commRows.length} packet sizes to image_comm.csv ` +
    `and ${lookupRows.length} KB timings to image_knowledgebase.csv\n`
  );
});
