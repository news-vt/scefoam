// src/image_embedding.js

import { spawnSync } from 'child_process';
import path         from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);
const PY         = process.env.PYTHON || 'python';

/**
 * ImageCodec
 *
 * • embed(absPath: string)       → returns a latent‐vector (array of floats)
 * • decode(item: [hex:string, latentArray:number[]] | latentArray:number[])  
 *                               → returns Buffer(PNG)
 */
class ImageCodec {
  constructor() {
    // nothing to initialize
  }

  embed(absPath) {
    // Call Python to encode <absPath> → latent array
    const script = path.join(__dirname, 'image_codec_diffusers.py');
    const args   = ['encode', absPath];
    const { status, stdout, stderr } = spawnSync(
      PY,
      [script, ...args],
      { encoding: 'utf8' }
    );

    if (status !== 0) {
      throw new Error(stderr.trim());
    }
    try {
      return JSON.parse(stdout);
    } catch (e) {
      throw new Error('Could not parse JSON from embed(): ' + e.message);
    }
  }

  decode(item) {
    // item must be either:
    //   [hex:string, latentArray:number[]]   (→ strip off the hex)
    //   OR
    //   latentArray:number[]
    if (item == null) {
      throw new Error("ImageCodec.decode(): received null/undefined");
    }

    let vec;
    // Case A: [hexString, latentArray]
    if (
      Array.isArray(item) &&
      item.length === 2 &&
      typeof item[0] === 'string' &&
      Array.isArray(item[1])
    ) {
      vec = item[1];
    }
    // Case B: [float, float, …]
    else if (Array.isArray(item) && item.every(x => typeof x === 'number')) {
      vec = item;
    } else {
      throw new Error(
        "ImageCodec.decode(): invalid argument, must be [hex, latentArray] or latentArray"
      );
    }

    const script    = path.join(__dirname, 'image_codec_diffusers.py');
    const jsonVector = JSON.stringify(vec);

    // Pipe JSON on stdin rather than as a giant command‐line argument.
    const { status, stdout, stderr } = spawnSync(
      PY,
      [script, 'decode'],
      {
        input: jsonVector,
        encoding: 'binary',
        maxBuffer: 50_000_000
      }
    );

    if (status !== 0) {
      // If Python prints something to stderr, we throw that error now:
      throw new Error(Buffer.from(stderr, 'binary').toString());
    }

    if (!stdout || stdout.length === 0) {
      throw new Error("ImageCodec.decode(): Python returned no PNG bytes");
    }

    // stdout is a 'binary'‐encoded string of PNG bytes
    return Buffer.from(stdout, 'binary');
  }
}

export default ImageCodec;
