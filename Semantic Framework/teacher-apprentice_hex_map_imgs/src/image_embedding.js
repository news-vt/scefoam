// image_embedding.js
const { spawnSync } = require('child_process');
const path           = require('path');

class ImageCodec {
  constructor({ downsample = 64 } = {}) {
    this.downsample   = downsample;
    this.latentLength = 4 * downsample * downsample;
  }

  embed(absPath) {
    const script = path.join(__dirname, 'image_codec_diffusers.py');
    const args   = ['encode-file', absPath, String(this.downsample)];
    const { status, stdout, stderr } = spawnSync(
      process.env.PYTHON || 'python',
      [script, ...args],
      { encoding: 'utf8' }
    );
    if (status !== 0) throw new Error(stderr.trim());
    return JSON.parse(stdout);
  }

  decode(item) {
    if (!item) throw new Error('decode(): null/undefined');
    const fullLen = this.latentLength + 2;
    let vecArray;
    if (Array.isArray(item) && (item.length === fullLen || item.length === this.latentLength)) {
      vecArray = item.length === fullLen ? item : [0,0,...item];
    } else {
      throw new Error(`decode(): invalid latent length (${item.length})`);
    }
    const script = path.join(__dirname, 'image_codec_diffusers.py');
    const { status, stdout, stderr } = spawnSync(
      process.env.PYTHON || 'python',
      [script, 'decode'],
      { input: JSON.stringify(vecArray), encoding: 'binary', maxBuffer: 50_000_000 }
    );
    if (status !== 0) throw new Error(Buffer.from(stderr, 'binary').toString());
    return Buffer.from(stdout, 'binary');
  }
}

module.exports = ImageCodec;