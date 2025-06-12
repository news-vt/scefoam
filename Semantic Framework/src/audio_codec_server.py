#!/usr/bin/env python3
"""
audio_codec_server.py  ▸ FastAPI server for AudioCraft EnCodec (flat-array API)
-------------------------------------------------------------------------------
With improved multipart/raw dispatch + exception logging.
"""
import argparse, logging, warnings, os, traceback
from io import BytesIO
import tempfile
from fastapi import Body

import torch
import torchaudio
import soundfile as sf

from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, Response
import uvicorn

from audiocraft.models.encodec import CompressionModel

# ─── Setup ───────────────────────────────────────────────────────────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
CODEC_NAME = "facebook/encodec_24khz"
TARGET_SR  = 24000

print(f"Loading EnCodec ({CODEC_NAME}) to {DEVICE} …", flush=True)
codec = CompressionModel.get_pretrained(CODEC_NAME, device=DEVICE).eval()
print(f"Audio Codec Server Ready on {DEVICE} - Access API via: http://<host>:8081", flush=True)


# ─── encode/decode ──────────────────────────────────────────────────────────
def _encode(audio_bytes: bytes):
    with tempfile.NamedTemporaryFile(suffix=".mp3") as tf:
        tf.write(audio_bytes); tf.flush()
        waveform, sr = torchaudio.load(tf.name, normalize=True)  # [C,T], float32   
    # ── MIX TO MONO ───────────────────────────────────────────────────────────
    # Model expects 1 channel. If stereo (>1), average channels:
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # now [1, T]
    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(sr, TARGET_SR).to(DEVICE)
        waveform = resampler(waveform.to(DEVICE)).cpu()
        sr = TARGET_SR
    x = waveform.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        codes, scale = codec.encode(x)
    _, n_q, T = codes.shape
    flat = codes.cpu().numpy().reshape(-1).tolist()
    sflat = (scale.cpu().numpy().reshape(-1).tolist()
             if scale is not None else [])
    return [sr, n_q, T] + flat + sflat

def _decode(vec):
    sr, n_q, T = map(int, vec[:3])
    flat = vec[3:]
    codes_flat, scale_flat = flat[:n_q*T], flat[n_q*T:]
    codes = torch.tensor(codes_flat, dtype=torch.int64, device=DEVICE)\
                .reshape(1, n_q, T)
    scale = (torch.tensor(scale_flat, dtype=torch.float32, device=DEVICE)
             .reshape(1, n_q, T)) if len(scale_flat)==n_q*T else None
    with torch.no_grad():
        audio = codec.decode(codes, scale)
    wav = audio.squeeze(0).cpu().numpy().T
    buf = BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    return buf.getvalue()

# ─── FastAPI ────────────────────────────────────────────────────────────────
app = FastAPI(title="AudioCraft EnCodec (flat-array)", docs_url=None, redoc_url=None)

@app.post("/encode")
async def encode_endpoint(
    request: Request,
    file: UploadFile = File(None)
):
    try:
        # choose multipart file or raw body
        if file is not None:
            payload = await file.read()
        else:
            payload = await request.body()
        if not payload:
            raise HTTPException(400, "No audio data provided")

        vec = _encode(payload)
        return JSONResponse(vec)
    except HTTPException:
        raise
    except Exception as e:
        # log full traceback for debugging
        traceback.print_exc()
        raise HTTPException(500, f"Encode error: {e}")

@app.post("/decode")
async def decode_endpoint(vec: list = Body(...)):
    try:
        wav = _decode(vec)
        return Response(content=wav, media_type="audio/wav")
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Decode error: {e}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host",    default="0.0.0.0")
    p.add_argument("--port",    type=int, default=8081)
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                workers=args.workers,
                log_level="warning")
