// image_controller.js — pure ESM client for the VAE‑codec server
import { spawnSync } from "child_process";
import path from "path";
import fs from "fs/promises";
import { request, FormData } from "undici";

class ImageCodec {
  constructor({ downsample = 64, baseURL = "http://127.0.0.1:8082" } = {}) {
    this.downsample = downsample;
    this.latentLength = 4 * downsample * downsample;
    this.base = baseURL.replace(/\/$/, "");
  }

  /** Legacy blocking path (curl) for test timing */
  embed(absPath) {
    const res = spawnSync(
      "curl", [
        "-sS", "-X", "POST",
        "-H", "Content-Type: application/octet-stream",
        "--data-binary", `@${absPath}`,
        `${this.base}/encode?ds=${this.downsample}`,
      ],
      { encoding: "utf8", maxBuffer: 50_000_000 },
    );
    if (res.status !== 0) throw new Error(res.stderr.trim() || "curl failed");
    return JSON.parse(res.stdout.trim());
  }

  /** Async encode: send *raw bytes* (octet‑stream). */
  async embedBytes(buf) {
    const { statusCode, body } = await request(
      `${this.base}/encode?ds=${this.downsample}`,
      {
        method: "POST",
        body: buf,
        headers: { "Content-Type": "application/octet-stream" },
      },
    );
    if (statusCode !== 200) throw new Error(`encode failed (${statusCode})`);
    return JSON.parse(await body.text());
  }

  /** Optional async helper that uploads a local file via multipart */
  async embedFile(absPath) {
    const buf = await fs.readFile(absPath);
    const form = new FormData();
    // undici requires a Blob/File for multipart; Buffer ⇒ Blob
    form.append("file", new Blob([buf]), path.basename(absPath));

    const { statusCode, body } = await request(
      `${this.base}/encode?ds=${this.downsample}`,
      { method: "POST", body: form, headers: form.headers },
    );
    if (statusCode !== 200) throw new Error(`encode failed (${statusCode})`);
    return JSON.parse(await body.text());
  }

  /** Async decode latent → JPEG Buffer */
  async decode(item) {
    const full = this.latentLength + 2;
    const vec =
      Array.isArray(item) && item.length === full     ? item :
      Array.isArray(item) && item.length === full - 2 ? [0, 0, ...item] :
      (() => { throw new Error("decode(): invalid latent length"); })();

    const { statusCode, body } = await request(
      `${this.base}/decode`,
      {
        method: "POST",
        body: Buffer.from(JSON.stringify(vec)),
        headers: { "Content-Type": "application/json" },
      },
    );
    if (statusCode !== 200) throw new Error(`decode failed (${statusCode})`);
    const chunks = [];
    for await (const c of body) chunks.push(c);
    return Buffer.concat(chunks);
  }
}

export default ImageCodec;