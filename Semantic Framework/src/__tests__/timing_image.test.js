// timing.test.js
const fs = require('fs');
const path = require('path');
const SemanticFramework = require('../SemanticFramework').default;

describe('SemanticFramework Image & KB Performance', () => {

    const publicDir = path.join(__dirname, '__test_public');
    const dataDir          = path.join(__dirname, '__test_data');
    
    const tmpDir = path.join(process.cwd(), '__tmp');
    const kbFile = path.join(publicDir, 'knowledge_base_images_test.json');

    const sourceImage1 = path.join(dataDir, 'test_image_1.jpg');
    const sourceImage2 = path.join(dataDir, 'test_image_2.jpg');
    const imageName1 = 'test_image_1.jpg';
    const imageName2 = 'test_image_2.jpg';
    const tmpImagePath1 = path.join(tmpDir, imageName1);
    const tmpImagePath2 = path.join(tmpDir, imageName2);

    let sf;
    beforeAll(() => {
        for (const d of [publicDir, tmpDir]) fs.mkdirSync(d, { recursive: true });
        if (!fs.existsSync(kbFile)) {
            fs.writeFileSync(kbFile, JSON.stringify({ map: {}, receivedData: [], hexHistory: [], predictedText: '', modelReady: false }, null, 2));
        }
        fs.copyFileSync(sourceImage1, tmpImagePath1);
        fs.copyFileSync(sourceImage2, tmpImagePath2);
        sf = new SemanticFramework({ kbPath: kbFile });
    });

    test('embed image timing (1)', () => {
        console.time('embed-image');
        const vec = sf.vectorize(`<img:${imageName1}>`);
        console.timeEnd('embed-image');
        expect(Array.isArray(vec)).toBe(true);
    });

    test('embed image timing (2)', () => {
        console.time('embed-image');
        const vec = sf.vectorize(`<img:${imageName2}>`);
        console.timeEnd('embed-image');
        expect(Array.isArray(vec)).toBe(true);
    });

    test('reconstruct image timing and save (1)', async () => {
        const vec = sf.vectorize(`<img:${imageName1}>`);
        console.time('reconstruct-image');
        const buf = await sf.decodeAsync(vec);
        console.timeEnd('reconstruct-image');
        const outPath = path.join(publicDir, 'reconstructed_' + imageName1);
        fs.writeFileSync(outPath, buf);
        expect(fs.existsSync(outPath)).toBe(true);
    });

    test('reconstruct image timing and save (2)', async () => {
        const vec = sf.vectorize(`<img:${imageName2}>`);
        console.time('reconstruct-image');
        const buf = await sf.decodeAsync(vec);
        console.timeEnd('reconstruct-image');
        const outPath = path.join(publicDir, 'reconstructed_' + imageName2);
        fs.writeFileSync(outPath, buf);
        expect(fs.existsSync(outPath)).toBe(true);
    });

    test('sift KB timing (1)', () => {
        const vec = sf.vectorize(`<img:${imageName1}>`);
        sf._register([vec]);
        console.time('sift-kb');
        sf.findClosestHexByVector(vec);
        console.timeEnd('sift-kb');
    });

    test('sift KB timing (2)', () => {
        const vec = sf.vectorize(`<img:${imageName2}>`);
        sf._register([vec]);
        console.time('sift-kb');
        sf.findClosestHexByVector(vec);
        console.timeEnd('sift-kb');
    });

    // test('size and similarity metrics', () => {
    //     // File sizes
    //     const origSize = fs.statSync(tmpImagePath).size;
    //     const vec = sf.vectorize(`<img:${imageName}>`);
    //     const latentSize = Buffer.byteLength(JSON.stringify(vec), 'utf8');
    //     const reconPath = path.join(publicDir, 'reconstructed_' + imageName);
    //     const reconSize = fs.statSync(reconPath).size;
    //     // Percent differences
    //     const percLatent = ((origSize - latentSize) / origSize) * 100;
    //     const percRecon = ((reconSize - origSize) / origSize) * 100;
    //     console.log(`Original: ${origSize} bytes - Latent: ${latentSize} bytes (${percLatent.toFixed(2)}%) - Recon: ${reconSize} bytes (${percRecon.toFixed(2)}%)`);
    //     expect(latentSize).toBeLessThan(origSize);

    //     // Pixel difference
    //     const origBuffer = fs.readFileSync(sourceImage);
    //     const reconBuffer = fs.readFileSync(reconPath);
    //     const origJpeg = jpeg.decode(origBuffer, { useTArray: true });
    //     const reconJpeg = jpeg.decode(reconBuffer, { useTArray: true });

    //     expect(reconJpeg.width).toBe(origJpeg.width);
    //     expect(reconJpeg.height).toBe(origJpeg.height);

    //     const width = origJpeg.width;
    //     const height = origJpeg.height;
    //     const origData = origJpeg.data;
    //     const reconData = reconJpeg.data;
    //     let diffCount = 0;
    //     const threshold = 10;
    //     for (let i = 0; i < origData.length; i += 4) {
    //         if (
    //             Math.abs(origData[i] - reconData[i]) > threshold ||
    //             Math.abs(origData[i + 1] - reconData[i + 1]) > threshold ||
    //             Math.abs(origData[i + 2] - reconData[i + 2]) > threshold
    //         ) {
    //             diffCount++;
    //         }
    //     }
    //     const diffRatio = (diffCount / (width * height)) * 100;
    //     console.log(`Pixel difference: ${diffRatio.toFixed(2)}%`);
    //     // Expect less than 10% of pixels differ significantly
    // });

});
