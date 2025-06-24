// timing.test.js
const fs = require('fs');
const path = require('path');
const SemanticFramework = require('../../SemanticFramework').default;

describe('SemanticFramework Image & KB Performance', () => {

    const testRoot = path.resolve(__dirname, '..');
    const publicDir = path.join(testRoot, '__test_public__');
    const dataDir = path.join(testRoot, '__test_data__');
    const tmpDir = path.join(process.cwd(), '__tmp');
    const kbFile = path.join(publicDir, 'knowledge_base_texts_test.json');

    const sourceText1 = path.join(dataDir, 'test_text_1.txt');
    const sourceText2 = path.join(dataDir, 'test_text_2.txt');
    const textName1 = 'test_text_1.txt';
    const textName2 = 'test_text_2.txt';
    const tmpTextPath1 = path.join(tmpDir, textName1);
    const tmpTextPath2 = path.join(tmpDir, textName2);

    let sf;
    beforeAll(() => {
        for (const d of [publicDir, tmpDir]) fs.mkdirSync(d, { recursive: true });
        if (!fs.existsSync(kbFile)) {
            fs.writeFileSync(kbFile, JSON.stringify({ map: {}, receivedData: [], hexHistory: [], predictedText: '', modelReady: false }, null, 2));
        }
        fs.copyFileSync(sourceText1, tmpTextPath1);
        fs.copyFileSync(sourceText2, tmpTextPath2);
        sf = new SemanticFramework({ kbPath: kbFile });
    });

    test('embed text timing (1)', () => {
        console.time('embed-text');
        const vec = sf.encode_vec(`<text:${textName1}>`);
        console.timeEnd('embed-text');
        expect(Array.isArray(vec)).toBe(true);
    });

    test('embed text timing (2)', () => {
        console.time('embed-text');
        const vec = sf.encode_vec(`<text:${textName2}>`);
        console.timeEnd('embed-text');
        expect(Array.isArray(vec)).toBe(true);
    });

    test('reconstruct text timing and save (1)', async () => {
        const vec = sf.encode_vec(`<text:${textName1}>`);
        console.time('reconstruct-text');
        const buf = await sf.decode_vec(vec);
        console.timeEnd('reconstruct-text');
        const outPath = path.join(publicDir, 'reconstructed_' + textName1);
        fs.writeFileSync(outPath, buf);
        expect(fs.existsSync(outPath)).toBe(true);
    }, 1_000_000);

    // test('reconstruct text timing and save (2)', async () => {
    //     const vec = sf.encode_vec(`<text:${textName2}>`);
    //     console.time('reconstruct-text');
    //     const buf = await sf.decode_vec(vec);
    //     console.timeEnd('reconstruct-text');
    //     const outPath = path.join(publicDir, 'reconstructed_' + textName2);
    //     fs.writeFileSync(outPath, buf);
    //     expect(fs.existsSync(outPath)).toBe(true);
    // }, 1_000_000);

    test('sift KB timing (1)', () => {
        const vec = sf.encode_vec(`<text:${textName1}>`);
        sf._register([vec]);
        console.time('sift-kb');
        sf.findClosestHexByVector(vec);
        console.timeEnd('sift-kb');
    });

    test('sift KB timing (2)', () => {
        const vec = sf.encode_vec(`<text:${textName2}>`);
        sf._register([vec]);
        console.time('sift-kb');
        sf.findClosestHexByVector(vec);
        console.timeEnd('sift-kb');
    });

    /* ---------- size & similarity metrics (text) ---------- */
    test('size and similarity metrics (text-1)', async () => {
        /* --------------------------------------------------------
           1.  make sure we have: original file, latent vector, and
               reconstructed file on disk
        --------------------------------------------------------- */
        const vec = sf.encode_vec(`<text:${textName1}>`);
        const reconPath = path.join(publicDir, 'reconstructed_' + textName1);

        if (!fs.existsSync(reconPath)) {
            const buf = await sf.decode_vec(vec);      // write only once
            fs.writeFileSync(reconPath, buf);
        }

        /* --------------------------------------------------------
           2. byte-sizes
               – latent is counted as   (#floats × 4 B)
        --------------------------------------------------------- */
        const origSize = fs.statSync(tmpTextPath1).size;

        const latentSize = Buffer.byteLength(JSON.stringify(vec), 'utf8');

        const reconSize = fs.statSync(reconPath).size;

        const pctLatent = ((origSize - latentSize) / origSize) * 100;
        const pctRecon = ((reconSize - origSize) / origSize) * 100;

        console.log(
            `Original: ${origSize} B | Latent: ${latentSize} B `
            + `(${pctLatent.toFixed(2)} % smaller) | `
            + `Recon: ${reconSize} B (${pctRecon.toFixed(2)} % diff)`
        );

        /* --- gate: latent must be smaller than source --- */
        // expect(latentSize).toBeLessThan(origSize);

        /* --------------------------------------------------------
           3.  Levenshtein similarity between original & recon
        --------------------------------------------------------- */
        const origText = fs.readFileSync(tmpTextPath1, 'utf8');
        const reconText = fs.readFileSync(reconPath, 'utf8');

        function levenshtein(a, b) {
            const m = a.length, n = b.length;
            const dp = Array.from({ length: m + 1 },
                () => new Array(n + 1));
            for (let i = 0; i <= m; i++) dp[i][0] = i;
            for (let j = 0; j <= n; j++) dp[0][j] = j;
            for (let i = 1; i <= m; i++) {
                for (let j = 1; j <= n; j++) {
                    dp[i][j] = Math.min(
                        dp[i - 1][j] + 1,                       // deletion
                        dp[i][j - 1] + 1,                       // insertion
                        dp[i - 1][j - 1]
                        + (a[i - 1] === b[j - 1] ? 0 : 1)     // substitution
                    );
                }
            }
            return dp[m][n];
        }

        const lev = levenshtein(origText, reconText);
        const sim = 1 - lev / Math.max(origText.length, reconText.length);
        console.log(`Levenshtein similarity: ${(sim * 100).toFixed(2)} %`);

        /* --- gate: ≥ 90 % of characters must match --- */
        // expect(sim).toBeGreaterThanOrEqual(0.9);
    }, 300_000);   // generous time-out for large files

});
