// audio_controller.js – client for EnCodec Δ-forecaster (encode, decode, predict)
import fs from "fs/promises";
import path from "path";
import { request, FormData } from "undici";
import { spawnSync } from "child_process";

export default class AudioCodec {
  constructor({ baseURL = "http://127.0.0.1:8081", timeout = 30000 } = {}) {
    this.base = baseURL.replace(/\/$/, "");
    this.timeout = timeout;
  }

  /* ---------- helpers ---------- */
  async _json(res) {
    const txt = await res.body.text();
    if (res.statusCode !== 200) throw new Error(`HTTP ${res.statusCode}: ${txt}`);
    return JSON.parse(txt);
  }

  /* ---------- encode ---------- */
  async embedFile(absPath) {
    const buf = await fs.readFile(absPath);
    const form = new FormData();
    form.append("file", buf, { filename: path.basename(absPath) });
    const res = await request(`${this.base}/encode`, { method: "POST", body: form, headers: form.headers, timeout: this.timeout });
    return this._json(res);
  }

  embed(absPath) {                      // blocking curl variant
    const { status, stdout, stderr } = spawnSync("curl", ["-fsS", "-X", "POST", "-H", "Content-Type: application/octet-stream", "--data-binary", `@${absPath}`, `${this.base}/encode`], { encoding: "utf8", maxBuffer: 200e6 });
    if (status !== 0) throw new Error(stderr.trim() || "curl failed");
    return JSON.parse(stdout.trim());
  }

  /* ---------- decode ---------- */
  async decode(vec) {
    const res = await request(`${this.base}/decode`, { method: "POST", body: JSON.stringify(vec), headers: { "Content-Type": "application/json" }, timeout: this.timeout });
    if (res.statusCode !== 200) throw new Error(`decode failed (${res.statusCode})`);
    const chunks = []; for await (const c of res.body) chunks.push(c);
    return Buffer.concat(chunks);
  }

  /* ---------- predict ---------- */
  async predict(latent, wantWav = false) {
    const url = `${this.base}/predict${wantWav ? "?wav=true" : ""}`;
    const res = await request(url, { method: "POST", body: JSON.stringify({ vec: latent }), headers: { "Content-Type": "application/json" }, timeout: this.timeout });
    const txt = await res.body.text();
    if (res.statusCode !== 200) throw new Error(`predict failed (${res.statusCode}): ${txt}`);
    const parsed = JSON.parse(txt);
    if (wantWav && parsed.wav) {
      const wavBuf = Buffer.from(parsed.wav.split(",")[1], "hex");
      return { latent: parsed.latent, wav: wavBuf };
    }
    return parsed;                     // latent only
  }
}
