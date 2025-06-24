// timing_audio.test.js
const fs = require('fs');
const path = require('path');
const SemanticFramework = require('../../SemanticFramework').default;

describe('SemanticFramework Audio & KB Performance', () => {

    const testRoot = path.resolve(__dirname, '..');
    const publicDir = path.join(testRoot, '__test_public__');
    const dataDir = path.join(testRoot, '__test_data__');
    const tmpDir = path.join(process.cwd(), '__tmp');
    const kbFile = path.join(publicDir, 'knowledge_base_audio_test.json');

    const sourceAudio1 = path.join(dataDir, 'test_audio_1.mp3');
    const sourceAudio2 = path.join(dataDir, 'test_audio_2.mp3');
    const audioName1 = 'test_audio_1.mp3';
    const audioName2 = 'test_audio_2.mp3';
    const tmpAudioPath1 = path.join(tmpDir, audioName1);
    const tmpAudioPath2 = path.join(tmpDir, audioName2);

    let sf;
    beforeAll(() => {
        // ensure directories
        for (const d of [publicDir, tmpDir]) {
            fs.mkdirSync(d, { recursive: true });
        }
        // init KB
        if (!fs.existsSync(kbFile)) {
            fs.writeFileSync(
                kbFile,
                JSON.stringify({
                    map: {}, receivedData: [], hexHistory: [],
                    predictedText: '', modelReady: false
                }, null, 2)
            );
        }
        // copy test inputs
        fs.copyFileSync(sourceAudio1, tmpAudioPath1);
        fs.copyFileSync(sourceAudio2, tmpAudioPath2);

        sf = new SemanticFramework({
            kbPath: kbFile,
        });
    });

    test('embed audio timing (1)', () => {
        console.time('embed-audio');
        const vec = sf.encode_vec(`<audio:${audioName1}>`);
        console.timeEnd('embed-audio');
        expect(Array.isArray(vec)).toBe(true);
    });

    test('embed audio timing (2)', () => {
        console.time('embed-audio');
        const vec = sf.encode_vec(`<audio:${audioName2}>`);
        console.timeEnd('embed-audio');
        expect(Array.isArray(vec)).toBe(true);
    });

    test('reconstruct audio timing and save (1)', async () => {
        const vec = sf.encode_vec(`<audio:${audioName1}>`);
        console.time('reconstruct-audio');
        const buf = await sf.decode_vec(vec);
        console.timeEnd('reconstruct-audio');
        const outName = audioName1.replace(/\.mp3$/, '.wav');
        const outPath = path.join(publicDir, 'reconstructed_' + outName);
        fs.writeFileSync(outPath, buf);
        expect(fs.existsSync(outPath)).toBe(true);
    });

    test('reconstruct audio timing and save (2)', async () => {
        const vec = sf.encode_vec(`<audio:${audioName2}>`);
        console.time('reconstruct-audio');
        const buf = await sf.decode_vec(vec);
        console.timeEnd('reconstruct-audio');
        const outName = audioName2.replace(/\.mp3$/, '.wav');
        const outPath = path.join(publicDir, 'reconstructed_' + outName);
        fs.writeFileSync(outPath, buf);
        expect(fs.existsSync(outPath)).toBe(true);
    });

    test('sift KB timing (1)', () => {
        const vec = sf.encode_vec(`<audio:${audioName1}>`);
        sf._register([vec]);
        console.time('sift-kb');
        sf.findClosestHexByVector(vec);
        console.timeEnd('sift-kb');
    });

    test('sift KB timing (2)', () => {
        const vec = sf.encode_vec(`<audio:${audioName2}>`);
        sf._register([vec]);
        console.time('sift-kb');
        sf.findClosestHexByVector(vec);
        console.timeEnd('sift-kb');
    });

    test('size and similarity metrics', () => {
        // Original MP3 on disk
        const origSize = fs.statSync(tmpAudioPath1).size;

        // Latent JSON size
        const vec = sf.encode_vec(`<audio:${audioName1}>`);
        const latentSize = Buffer.byteLength(JSON.stringify(vec), 'utf8');

        // Reconstructed WAV size (replace .mp3 → .wav)
        const outName = audioName1.replace(/\.mp3$/, '.wav');
        const reconPath = path.join(publicDir, 'reconstructed_' + outName);
        const reconSize = fs.statSync(reconPath).size;

        // Percent differences
        const percLatent = ((origSize - latentSize) / origSize) * 100;
        const percRecon = ((reconSize - origSize) / origSize) * 100;

        console.log(
            `Original: ${origSize} B — Latent: ${latentSize} B (${percLatent.toFixed(2)}%)` +
            ` — Recon: ${reconSize} B (${percRecon.toFixed(2)}%)`
        );

        // latent should always be smaller than the original
        expect(latentSize).toBeLessThan(origSize);
    });

});
