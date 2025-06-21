// Imports
import { request } from "undici";
import { spawnSync } from "child_process";

/**
 * AudioCodec endpoints:
 *   POST /embed   (binary)               → [24000,n_q,T_LEN,…codes] (length=3 + n_q*T_LEN ints)
 *   POST /decode   {vec:[…]}              → audio/wav binary
 *   POST /predict  {vec:[…]}              → [24000,n_q,T_LEN,…] OR {latent:[…],wav:"..."}
 */
export default class AudioCodec {
  constructor({ baseURL = "http://127.0.0.1:8081", timeout = 30000 } = {}) {
    this.base = baseURL.replace(/\/$/, "");
    this.timeout = timeout;
  }

  embed(absPath) {                    
    const { status, stdout, stderr } = spawnSync("curl", ["-fsS", "-X", "POST", "-H", "Content-Type: application/octet-stream", "--data-binary", `@${absPath}`, `${this.base}/encode`], { encoding: "utf8", maxBuffer: 200e6 });
    if (status !== 0) throw new Error(stderr.trim() || "curl failed");
    return JSON.parse(stdout.trim());
  }

  async decode(vec) {
    const res = await request(`${this.base}/decode`, { method: "POST", body: JSON.stringify(vec), headers: { "Content-Type": "application/json" }, timeout: this.timeout });
    if (res.statusCode !== 200) throw new Error(`decode failed (${res.statusCode})`);
    const chunks = []; for await (const c of res.body) chunks.push(c);
    return Buffer.concat(chunks);
  }

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
    return parsed; 
  }
}
