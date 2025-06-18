#!/usr/bin/env python3
"""
audio_codec_server.py – EnCodec + shape-aware Δ-forecaster
• Keeps a separate exponential-moving Δ for each header shape.
• /predict uses the Δ that matches the incoming latent’s header.
"""

import argparse, logging, warnings, os, math, asyncio, tempfile, traceback
from io import BytesIO

import torch, torchaudio, soundfile as sf
from fastapi import FastAPI, Request, UploadFile, HTTPException, Body, File
from fastapi.responses import JSONResponse, Response
import uvicorn
from audiocraft.models.encodec import CompressionModel

# ─── Silence noisy logs ─────────────────────────────────────────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

# ─── codec ──────────────────────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
CODEC_NAME = "facebook/encodec_24khz"
TARGET_SR  = 24_000
print(f"Loading EnCodec ({CODEC_NAME}) on {DEVICE} …", flush=True)
codec = CompressionModel.get_pretrained(CODEC_NAME, device=DEVICE).eval()

# ─── encode / decode helpers ────────────────────────────────────────────
def _encode(audio_bytes: bytes):
    with tempfile.NamedTemporaryFile(suffix=".mp3") as tf:
        tf.write(audio_bytes); tf.flush()
        wav, sr = torchaudio.load(tf.name, normalize=True)  # [C,T]
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)                    # mono
    if sr != TARGET_SR:
        wav = torchaudio.transforms.Resample(sr, TARGET_SR)(wav)
        sr = TARGET_SR
    with torch.no_grad():
        codes, scale = codec.encode(wav.unsqueeze(0).to(DEVICE))  # (1,n_q,T)
    _, n_q, T = codes.shape
    flat   = codes.cpu().view(-1).tolist()
    sflat  = scale.cpu().view(-1).tolist() if scale is not None else []
    return [sr, n_q, T] + flat + sflat

def _decode(vec):
    sr, n_q, T = map(int, vec[:3])
    body       = vec[3:]
    codes_flat, scale_flat = body[:n_q*T], body[n_q*T:]
    codes  = torch.tensor(codes_flat, dtype=torch.int64, device=DEVICE)\
             .view(1, n_q, T)
    scale  = (torch.tensor(scale_flat, dtype=torch.float32, device=DEVICE)
              .view(1, n_q, T)) if scale_flat else None
    with torch.no_grad():
        audio = codec.decode(codes, scale)                 # (1,1,T')
    buf = BytesIO()
    sf.write(buf, audio.squeeze(0).cpu().numpy().T, sr, format='WAV')
    return buf.getvalue()

# ─── shape-aware Δ-forecaster ───────────────────────────────────────────
HISTORY = []                       # [(header tuple, tensor(body))]
DELTAS  = {}                       # header tuple -> EMA Δ tensor
ALPHA   = 0.5                      # smoothing factor

def _split(vec):
    header, body = tuple(map(int, vec[:3])), vec[3:]
    return header, torch.tensor(body, dtype=torch.float32, device=DEVICE)

def _update_delta():
    if len(HISTORY) < 2:
        return
    (h2, b2), (h1, b1) = HISTORY[-1], HISTORY[-2]
    if h2 != h1 or b2.numel() != b1.numel():
        # new shape → start fresh EMA for this header
        return
    delta = b2 - b1
    if h2 in DELTAS:
        DELTAS[h2] = ALPHA * delta + (1 - ALPHA) * DELTAS[h2]
    else:
        DELTAS[h2] = delta

def _predict(vec, need_wav=False):
    header, body = _split(vec)
    if header not in DELTAS:
        raise HTTPException(400, "Forecaster lacks Δ for this shape "
                                  "(need ≥2 decode calls with same header)")
    next_body = (body + DELTAS[header]).round().clamp(0)   # keep ints ≥0
    next_vec  = list(header) + next_body.cpu().tolist()
    if need_wav:
        return next_vec, _decode(next_vec)
    return next_vec, None

# ─── FastAPI wiring ─────────────────────────────────────────────────────
app = FastAPI(title="EnCodec Δ forecaster (shape-aware)", docs_url=None, redoc_url=None)

@app.post("/encode")
async def encode_ep(request: Request, file: UploadFile = File(None)):
    data = await file.read() if file else await request.body()
    if not data:
        raise HTTPException(400, "No audio supplied")
    try:
        return JSONResponse(_encode(data))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Encode error: {e}")

@app.post("/decode")
async def decode_ep(vec: list = Body(...)):
    try:
        header, body = _split(vec)
        HISTORY.append((header, body))
        _update_delta()
        wav = _decode(vec)
        return Response(content=wav, media_type="audio/wav")
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Decode error: {e}")

@app.post("/predict")
async def predict_ep(vec: list = Body(..., embed=True), wav: bool = False):
    try:
        next_vec, wav_buf = _predict(vec, wav)
        if wav:
            return JSONResponse({"latent": next_vec,
                                 "wav": "data:audio/wav;base64," +
                                        BytesIO(wav_buf).getvalue().hex()})
        return JSONResponse(next_vec)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Predict error: {e}")

# ─── CLI ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8081)
    args = p.parse_args()
    print(f"Audio Codec Ready on {DEVICE} – API: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
