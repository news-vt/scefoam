// Imports
import { spawnSync } from "child_process";
import { request } from "undici";

/**
 * ImageCodec endpoints:
 *   POST /embed   (binary)               → [w,h,…] (length=2 + 4*ds² = 16 386 floats)
 *   POST /decode   {vec:[16 386]}         → image/jpeg binary
 *   POST /predict  {vec:[16 386]}         → [16 386] OR {latent:[16 386],jpeg:"..."}
 */
class ImageCodec {
  constructor({ downsample = 64, baseURL = "http://127.0.0.1:8082",
                timeout = 30000 } = {}) {
    this.downsample   = downsample;
    this.latentLength = 4 * downsample * downsample;
    this.base         = baseURL.replace(/\/$/, "");
    this.timeout      = timeout;
  }

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

  async decode(item) {
    const full = this.latentLength + 2; 
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

  async predict(latent, wantJPEG = false) {
    const payload = { vec: latent };
    const url = `${this.base}/predict${wantJPEG ? "?jpeg=true" : ""}`;
    const { statusCode, body } = await request(
      url,
      { method: "POST",
        body: JSON.stringify(payload), 
        headers: { "Content-Type": "application/json" },
        timeout: this.timeout
      }
    );
    const txt = await body.text();
    if (statusCode !== 200) throw new Error(`predict failed (${statusCode}): ${txt}`);
    const parsed = JSON.parse(txt);
    if (wantJPEG && parsed.jpeg) {
      const jpgBuf = Buffer.from(parsed.jpeg.split(",")[1], "hex");
      return { latent: parsed.latent, jpeg: jpgBuf };
    }
    return parsed;  
  }
}

export default ImageCodec;
