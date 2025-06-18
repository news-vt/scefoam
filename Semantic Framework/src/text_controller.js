// text_controller.js — pure-ESM client for the SONAR text-codec server
import { spawnSync } from "child_process";
import { request } from "undici";

/**
 * FastAPI endpoints:
 *   POST /encode   {texts:[...]}          → [[1024-int8], ...]
 *   POST /decode   {embeddings:[[1024]…]} → ["sentence", ...]
 *   POST /predict  {vec:[1024]}           → [1024]  OR {latent:[1024],text:"..."}
 */
export default class TextCodec {
  constructor({ baseURL = "http://127.0.0.1:8080" } = {}) {
    this.base = baseURL.replace(/\/$/, "");
  }

  /*──────────────── helper (sync via curl) ───────────────*/
  _curlJson(payload, ep) {
    const url = `${this.base}/${ep}`;
    const res = spawnSync("curl", ["-s", "-XPOST", "-H", "Content-Type: application/json", "-d", JSON.stringify(payload), url]);
    if (res.status !== 0) throw new Error(`curl failed: ${res.stderr}`);
    return JSON.parse(res.stdout);
  }

  /*──────────────── synchronous embed ────────────────────*/
  embed(sentence)         { return this._curlJson({texts:[String(sentence)]},"encode")[0]; }
  embedMany(arr)          { return this._curlJson({texts:arr}, "encode"); }

  /*──────────────── asynchronous embed ───────────────────*/
  async embedAsync(sentence) {
    const { statusCode, body } = await request(`${this.base}/encode`, {
      method: "POST",
      body: JSON.stringify({ texts: [String(sentence)] }),
      headers: { "Content-Type": "application/json" }
    });
    const txt = await body.text();
    if (statusCode !== 200) throw new Error(`encode failed (${statusCode}): ${txt}`);
    return JSON.parse(txt)[0];
  }

  /*──────────────── asynchronous decode ─────────────────*/
  async decode(emb) {
    const matrix = Array.isArray(emb[0]) ? emb : [emb];
    const { statusCode, body } = await request(`${this.base}/decode`, {
      method: "POST",
      body: JSON.stringify({ embeddings: matrix }),
      headers: { "Content-Type": "application/json" }
    });
    const txt = await body.text();
    if (statusCode !== 200) throw new Error(`decode failed (${statusCode}): ${txt}`);
    const sentences = JSON.parse(txt);
    return Array.isArray(emb[0]) ? sentences : sentences[0];
  }

  /*──────────────── asynchronous predict ────────────────*/
  /**
   * Forecast next latent.  If wantText=true the server also decodes it.
   * @param {number[]} latent
   * @param {boolean}  [wantText=false]
   * @returns {Promise<Array|{latent:Array,text:string}>}
   */
  async predict(latent, wantText = false) {
    const url = `${this.base}/predict${wantText ? "?text=true" : ""}`;
    const { statusCode, body } = await request(url, {
      method: "POST",
      body: JSON.stringify({ vec: latent }),
      headers: { "Content-Type": "application/json" }
    });
    const txt = await body.text();
    if (statusCode !== 200) throw new Error(`predict failed (${statusCode}): ${txt}`);
    return JSON.parse(txt);
  }
}
