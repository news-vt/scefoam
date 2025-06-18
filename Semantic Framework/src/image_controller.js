// image_controller.js – client for the enhanced image codec server
// Supports: embed, decode, predict
import { spawnSync } from "child_process";
import path from "path";
import fs from "fs/promises";
import { request, FormData } from "undici";

class ImageCodec {
  /**
   * @param {object} opts
   * @param {number} opts.downsample  Latent H=W size (default 64 → 16 384 dims)
   * @param {string} opts.baseURL     Base URL of the image codec server
   * @param {number} opts.timeout     ms before HTTP abort
   */
  constructor({ downsample = 64, baseURL = "http://127.0.0.1:8082",
                timeout = 30000 } = {}) {
    this.downsample   = downsample;
    this.latentLength = 4 * downsample * downsample;
    this.base         = baseURL.replace(/\/$/, "");
    this.timeout      = timeout;
  }

  /*──────────────── synchronous embed (curl) ─────────────*/
  embed(absPath) {
    const res = spawnSync(
      "curl",
      [
        "-fsS",
        "-X", "POST",
        "-H", "Content-Type: application/octet-stream",
        "--data-binary", `@${absPath}`,
        `${this.base}/encode?ds=${this.downsample}`,
      ],
      { encoding: "utf8", maxBuffer: 50_000_000 }
    );
    if (res.status !== 0) {
      throw new Error(res.stderr.trim() || "curl failed");
    }
    return JSON.parse(res.stdout.trim());
  }

  /*──────────────── async embed (raw bytes) ───────────────*/
  async embedBytes(buf) {
    const { statusCode, body } = await request(
      `${this.base}/encode?ds=${this.downsample}`,
      { method: "POST",
        body: buf,
        headers: { "Content-Type": "application/octet-stream" },
        timeout: this.timeout
      }
    );
    if (statusCode !== 200) throw new Error(`encode failed (${statusCode})`);
    return JSON.parse(await body.text());
  }

  /*──────────────── async embedFile (multipart) ───────────*/
  async embedFile(absPath) {
    const buf = await fs.readFile(absPath);
    const form = new FormData();
    form.append("file", new Blob([buf]), path.basename(absPath));
    const { statusCode, body } = await request(
      `${this.base}/encode?ds=${this.downsample}`,
      { method: "POST", body: form, headers: form.headers, timeout: this.timeout }
    );
    if (statusCode !== 200) throw new Error(`encode failed (${statusCode})`);
    return JSON.parse(await body.text());
  }

  /*──────────────── async decode latent → JPEG buf ───────*/
  async decode(item) {
    const full = this.latentLength + 2;           // w,h + flat latent
    const vec =
      Array.isArray(item) && item.length === full     ? item :
      Array.isArray(item) && item.length === full - 2 ? [0, 0, ...item] :
      (() => { throw new Error("decode(): invalid latent length"); })();

    const { statusCode, body } = await request(
      `${this.base}/decode`,
      { method: "POST",
        body: Buffer.from(JSON.stringify(vec)),
        headers: { "Content-Type": "application/json" },
        timeout: this.timeout
      }
    );
    if (statusCode !== 200) throw new Error(`decode failed (${statusCode})`);
    const chunks = [];
    for await (const c of body) chunks.push(c);
    return Buffer.concat(chunks);
  }

  /*──────────────── async predict latent(t+1) ─────────────*/
  /**
   * @param {Array<number>} latent  The current latent vector (t)
   * @param {boolean} [wantJPEG]    If true, also return decoded JPEG
   * @returns {Promise<Array<number>|{latent:Array, jpeg:Buffer}>}
   */
  async predict(latent, wantJPEG = false) {
    const payload = { vec: latent };
    const url = `${this.base}/predict${wantJPEG ? "?jpeg=true" : ""}`;
    const { statusCode, body } = await request(
      url,
      { method: "POST",
        body: JSON.stringify(payload),          // ▼ keep the {"vec": …} wrapper
        headers: { "Content-Type": "application/json" },
        timeout: this.timeout
      }
    );
    const txt = await body.text();
    if (statusCode !== 200) throw new Error(`predict failed (${statusCode}): ${txt}`);
    const parsed = JSON.parse(txt);
    if (wantJPEG && parsed.jpeg) {
      // convert hex back to Buffer
      const jpgBuf = Buffer.from(parsed.jpeg.split(",")[1], "hex");
      return { latent: parsed.latent, jpeg: jpgBuf };
    }
    return parsed;               // predicted latent list
  }
}

export default ImageCodec;
