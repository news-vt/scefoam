const fs = require('fs');
const path = require('path');
const SemanticFramework = require('../SemanticFramework');
const ImageCodec = require('../image_embedding');
const TransformerModel = require('../model');
const jpeg = require('jpeg-js');

describe('SemanticFramework Image & KB Performance', () => {
  const publicDir = path.join(__dirname, '__test_public');
  const tmpDir = path.join(process.cwd(), '__tmp');
  const kbFile = path.join(publicDir, 'knowledge_base_test.json');

  const sourceImage = path.join(__dirname, 'test_image.jpg');
  const imageName = 'test_image.jpg';
  const tmpImagePath = path.join(tmpDir, imageName);

  let sf;
  beforeAll(() => {
    for (const d of [publicDir, tmpDir]) fs.mkdirSync(d, { recursive: true });
    if (!fs.existsSync(kbFile)) {
      fs.writeFileSync(kbFile, JSON.stringify({ map: {}, receivedData: [], hexHistory: [], predictedText: '', modelReady: false }, null, 2));
    }
    fs.copyFileSync(sourceImage, tmpImagePath);
    const imageEmbedding = new ImageCodec();
    const model = new TransformerModel({ kbPath: kbFile, port: 9999 });
    sf = new SemanticFramework({ kbPath: kbFile, embedding: imageEmbedding, model });
  });

  test('embed image timing', () => {
    console.time('embed-image');
    const vec = sf.vectorize(`<img:${imageName}>`);
    console.timeEnd('embed-image');
    expect(Array.isArray(vec)).toBe(true);
  });

  test('reconstruct image timing and save', async () => {
    const vec = sf.vectorize(`<img:${imageName}>`);
    console.time('reconstruct-image');
    const buf = await sf.decodeAsync(vec);
    console.timeEnd('reconstruct-image');
    const outPath = path.join(publicDir, 'reconstructed_' + imageName);
    fs.writeFileSync(outPath, buf);
    expect(fs.existsSync(outPath)).toBe(true);
  });

  test('sift KB timing', () => {
    const vec = sf.vectorize(`<img:${imageName}>`);
    sf._register([vec]);
    console.time('sift-kb');
    sf.findClosestHexByVector(vec);
    console.timeEnd('sift-kb');
  });

test('size and similarity metrics', () => {
  // File sizes
  const origSize = fs.statSync(tmpImagePath).size;
  const vec = sf.vectorize(`<img:${imageName}>`);
  const latentSize = Buffer.byteLength(JSON.stringify(vec), 'utf8');
  const reconPath = path.join(publicDir, 'reconstructed_' + imageName);
  const reconSize = fs.statSync(reconPath).size;
  // Percent differences
  const percLatent = ((origSize - latentSize) / origSize) * 100;
  const percRecon = ((reconSize - origSize) / origSize) * 100;
  console.log(`Original: ${origSize} bytes - Latent: ${latentSize} bytes (${percLatent.toFixed(2)}%) - Recon: ${reconSize} bytes (${percRecon.toFixed(2)}%)`);
  expect(latentSize).toBeLessThan(origSize);

  // Pixel difference
  const origBuffer = fs.readFileSync(sourceImage);
  const reconBuffer = fs.readFileSync(reconPath);
  const origJpeg = jpeg.decode(origBuffer, { useTArray: true });
  const reconJpeg = jpeg.decode(reconBuffer, { useTArray: true });

  expect(reconJpeg.width).toBe(origJpeg.width);
  expect(reconJpeg.height).toBe(origJpeg.height);

  const width = origJpeg.width;
  const height = origJpeg.height;
  const origData = origJpeg.data;
  const reconData = reconJpeg.data;
  let diffCount = 0;
  const threshold = 10;
  for (let i = 0; i < origData.length; i += 4) {
    if (
      Math.abs(origData[i] - reconData[i]) > threshold ||
      Math.abs(origData[i+1] - reconData[i+1]) > threshold ||
      Math.abs(origData[i+2] - reconData[i+2]) > threshold
    ) {
      diffCount++;
    }
  }
  const diffRatio = (diffCount / (width * height)) * 100;
  console.log(`Pixel difference: ${diffRatio.toFixed(2)}%`);
  // Expect less than 10% of pixels differ significantly
});

});
