// audio_controller.js — pure ESM client for the flat-array audio VAE server
import fs from "fs/promises";
import path from "path";
import { request, FormData } from "undici";
import { Blob } from "buffer";
import { spawnSync } from "child_process";

export default class AudioCodec {
  /**
   * @param {object} opts
   * @param {string} opts.baseURL  Base URL of the audio codec server
   * @param {number} opts.timeout  Milliseconds before HTTP requests abort
   */
  constructor({ baseURL = "http://127.0.0.1:8081", timeout = 30000 } = {}) {
    this.base    = baseURL.replace(/\/$/, "");
    this.timeout = timeout;
  }

  /** Internal helper: parse JSON or throw with full text on error */
  async _parseJsonOrThrow(res) {
    const text = await res.body.text();
    if (res.statusCode !== 200) {
      throw new Error(`HTTP ${res.statusCode}: ${text}`);
    }
    try {
      return JSON.parse(text);
    } catch (err) {
      throw new Error(`Invalid JSON response: ${text}`);
    }
  }

  /** Legacy blocking path (curl) for test timing */
  embed(absPath) {
    const res = spawnSync(
      "curl",
      [
        "-fsS", // fail on HTTP ≥400, stay silent on stdout
        "-X", "POST",
        "-H", "Content-Type: application/octet-stream",
        "--data-binary", `@${absPath}`,
        `${this.base}/encode`,
      ],
      { encoding: "utf8", maxBuffer: 200_000_000 }
    );
    if (res.status !== 0) {
      const msg = (res.stderr || res.stdout || "curl failed").trim();
      throw new Error(`curl error: ${msg}`);
    }
    try {
      return JSON.parse(res.stdout.trim());
    } catch {
      throw new Error(`Invalid JSON response: ${res.stdout.trim()}`);
    }
  }

  /** Async encode: send raw bytes (octet‐stream). */
  async embedBytes(buf) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);
    let res;
    try {
      res = await request(
        `${this.base}/encode`,
        {
          method: "POST",
          body: buf,
          headers: { "Content-Type": "application/octet-stream" },
          signal: controller.signal
        }
      );
    } catch (err) {
      throw new Error(`Request failed: ${err.message}`);
    } finally {
      clearTimeout(timeoutId);
    }
    return this._parseJsonOrThrow(res);
  }

  /** Async helper: upload a local file via multipart/form-data. */
  async embedFile(absPath) {
    const buf = await fs.readFile(absPath);
    const form = new FormData();
    // Undici accepts Buffer + filename directly
    form.append("file", buf, { filename: path.basename(absPath) });

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);
    let res;
    try {
      res = await request(
        `${this.base}/encode`,
        {
          method: "POST",
          body: form,
          headers: form.headers,
          signal: controller.signal
        }
      );
    } catch (err) {
      throw new Error(`Request failed: ${err.message}`);
    } finally {
      clearTimeout(timeoutId);
    }
    return this._parseJsonOrThrow(res);
  }

  /** Async decode flat-array → WAV Buffer */
  async decode(vec) {
    if (!Array.isArray(vec) || vec.length < 3) {
      throw new Error("decode(): invalid payload");
    }
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);
    let res;
    try {
      res = await request(
        `${this.base}/decode`,
        {
          method: "POST",
          body: Buffer.from(JSON.stringify(vec)),
          headers: { "Content-Type": "application/json" },
          signal: controller.signal
        }
      );
    } catch (err) {
      throw new Error(`Request failed: ${err.message}`);
    } finally {
      clearTimeout(timeoutId);
    }
    if (res.statusCode !== 200) {
      const text = await res.body.text();
      throw new Error(`decode failed (${res.statusCode}): ${text}`);
    }
    const chunks = [];
    for await (const chunk of res.body) {
      chunks.push(chunk);
    }
    return Buffer.concat(chunks);
  }
}
