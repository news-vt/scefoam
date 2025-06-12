// text_controller.js — pure ESM client for the SONAR text-codec server
import { spawnSync } from "child_process";
import { request } from "undici";

/**
 * FastAPI endpoints:
 *   POST /encode   { "texts": ["..."] }          → [[1024-floats], ...]
 *   POST /decode   { "embeddings": [[1024]...] } → ["sentence", ...]
 */
export default class TextCodec {
  constructor({ baseURL = "http://127.0.0.1:8080" } = {}) {
    this.base = baseURL.replace(/\/$/, "");
  }

  /*──────────────────────── helper ────────────────────────*/
  _curlJson(payload, endpoint) {
    const res = spawnSync(
      "curl",
      [
        "-sS", "-X", "POST",
        "-H", "Content-Type: application/json",
        "-d", JSON.stringify(payload),
        `${this.base}/${endpoint}`
      ],
      { encoding: "utf8", maxBuffer: 50_000_000 }
    );
    if (res.status !== 0) {
      throw new Error(res.stderr.trim() || res.stdout.trim() || "curl failed");
    }
    return JSON.parse(res.stdout);
  }

  /*──────────────────────── synchronous embed (blocking) ──*/
  embed(sentence) {
    const out = this._curlJson({ texts: [String(sentence)] }, "encode");
    return out[0];                    // [768]
  }

  /*──────────────────────── synchronous embedMany ─────────*/
  embedMany(arr) {
    return this._curlJson({ texts: arr }, "encode"); // [[768]×N]
  }

  /*──────────────────────── asynchronous embed ────────────*/
  async embedAsync(sentence) {
    const payload = { texts: [String(sentence)] };
    const { statusCode, body } = await request(`${this.base}/encode`, {
      method: "POST",
      body: JSON.stringify(payload),
      headers: { "Content-Type": "application/json" }
    });
    const txt = await body.text();
    if (statusCode !== 200) {
      throw new Error(`encode failed (${statusCode}): ${txt}`);
    }
    return JSON.parse(txt)[0];
  }

  /*──────────────────────── asynchronous decode ───────────*/
  async decode(emb) {
    const matrix = Array.isArray(emb[0]) ? emb : [emb];
    const { statusCode, body } = await request(`${this.base}/decode`, {
      method: "POST",
      body: JSON.stringify({ embeddings: matrix }),
      headers: { "Content-Type": "application/json" }
    });
    const txt = await body.text();
    if (statusCode !== 200) throw new Error(`decode failed (${statusCode}): ${txt}`);
    const sentences = JSON.parse(txt);          // ["..."] × N
    return Array.isArray(emb[0]) ? sentences : sentences[0];
  }
}
