#!/usr/bin/env python3
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XFORMERS_MORE_DETAILS"] = "0"

import builtins, sys
_orig_print = builtins.print
def _safe_print(*args, **kwargs):
    try:
        _orig_print(*args, **kwargs)
    except BrokenPipeError:
        pass
builtins.print = _safe_print

# Imports
import argparse
import asyncio
import tempfile
import math
import random
from io import BytesIO
from typing import List
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
from fastapi import FastAPI, Request, UploadFile, File, Body, HTTPException
from fastapi.responses import JSONResponse, Response
import uvicorn
from audiocraft.models.encodec import CompressionModel

from contextlib import nullcontext
DTYPE   = torch.bfloat16 if torch.cuda.is_available() else torch.float32
if torch.cuda.is_available():
    from torch.amp import autocast          # future-proof import
    autocast = autocast(device_type="cuda", dtype=DTYPE)
else:
    autocast = nullcontext()

# ─── constants & device ─────────────────────────────────────────────────
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SR    = 24_000
CODEC_ID     = "facebook/encodec_24khz"
SEC_LIMIT    = 30       # prediction length in seconds
EMB          = 48
HEADS        = 6
LAYERS       = 4
LR           = 2e-4
EPOCHS       = 50
MAX_TOKENS   = 2048     # max total tokens per window = n_q * T_window
HISTORY_K    = 6   
acc_threshold = 99
torch.set_float32_matmul_precision("medium")

# ─── load EnCodec & compute frames-per-SEC_LIMIT ────────────────────────
codec = CompressionModel.get_pretrained(CODEC_ID, device=DEVICE).eval()
_dummy = torch.zeros(1, 1, TARGET_SR * SEC_LIMIT, device=DEVICE)
with torch.no_grad():
    codes_dummy, _ = codec.encode(_dummy)   # [1, n_q, T_LEN]
_, N_Q, T_LEN = codes_dummy.shape
CODEBOOK_SIZE = codec.cardinality

# ─── Seq2Seq latent→latent predictor ────────────────────────────────────
class Latent2Latent(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed       = nn.Embedding(CODEBOOK_SIZE, EMB)
        self.transformer = nn.Transformer(
            d_model=EMB, nhead=HEADS,
            num_encoder_layers=LAYERS, num_decoder_layers=LAYERS,
            batch_first=True
        )
        self.head        = nn.Linear(EMB, CODEBOOK_SIZE)

    def forward(self, src: torch.LongTensor, tgt: torch.LongTensor) -> torch.Tensor:
        B, n_q, T = src.shape
        seq_len    = n_q * T

        x_src = self.embed(src).view(B, seq_len, EMB)
        x_tgt = self.embed(tgt).view(B, seq_len, EMB)

        # sinusoidal positional encoding
        device = x_src.device
        position = torch.arange(seq_len, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, EMB, 2, device=device) * (-math.log(10000.0) / EMB)
        )
        pe = torch.zeros(seq_len, EMB, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, seq_len, EMB]

        x_src = x_src + pe
        x_tgt = x_tgt + pe

        out = self.transformer(x_src, x_tgt)  # [B, seq_len, EMB]
        logits = self.head(out)               # [B, seq_len, CODEBOOK_SIZE]
        return logits.view(B, n_q, T, CODEBOOK_SIZE)

# ─── instantiate model, optimizer, loss, lock & history ────────────────
MODEL        = Latent2Latent().to(DEVICE)
OPT          = torch.optim.AdamW(MODEL.parameters(), lr=LR)
LOSS         = nn.CrossEntropyLoss()
LOCK         = asyncio.Lock()
HIST: List[torch.LongTensor] = []

# ─── EnCodec I/O helpers ────────────────────────────────────────────────
def _wav_to_latent(wav_raw_bytes: bytes) -> List[float]:
    with tempfile.NamedTemporaryFile(suffix=".bin") as tf:
        tf.write(wav_raw_bytes); tf.flush()
        wav, sr = torchaudio.load(tf.name, normalize=True)
    if wav.size(0) > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.transforms.Resample(sr, TARGET_SR)(wav)
    with torch.no_grad(), autocast:
        codes, _ = codec.encode(wav.unsqueeze(0).to(DEVICE))
    _, n_q, T = codes.shape
    return [TARGET_SR, n_q, T] + codes.cpu().view(-1).tolist()

def _latent_to_wav(vec: list) -> torch.Tensor:
    sr, n_q, T = map(int, vec[:3])
    body = vec[3:]
    codes = torch.tensor(body, dtype=torch.int64, device=DEVICE).view(1, n_q, T)
    with torch.no_grad(), autocast:
        wav = codec.decode(codes).squeeze(0)
    return wav.float()

# ─── synchronous, random-window training ───────────────────────────────
async def _maybe_train():
    if len(HIST) < 2:
        return

    # sample two consecutive latents
    max_pairs = len(HIST) - 1
    K = min(max_pairs, HISTORY_K)
    idx = (len(HIST) - 1 - K) + random.randint(0, K - 1) if K > 1 else 0
    c1, c2 = HIST[idx], HIST[idx + 1]

    # 1) crop to equal T
    if c1.shape != c2.shape:
        T1, T2 = c1.shape[2], c2.shape[2]
        m = min(T1, T2)
        c1, c2 = c1[:, :, :m], c2[:, :, :m]

    # 2) clamp & random-window crop
    _, n_q, Tcur = c1.shape
    if n_q * Tcur > MAX_TOKENS:
        T_new = MAX_TOKENS // n_q
        start = random.randint(0, Tcur - T_new) if Tcur > T_new else 0
        c1 = c1[:, :, start:start + T_new]
        c2 = c2[:, :, start:start + T_new]

    c1, c2 = c1.contiguous(), c2.contiguous()

    async with LOCK:
        def _train_loop():
            MODEL.train()
            for epoch in range(EPOCHS):
                OPT.zero_grad()
                logits = MODEL(c1, c2)                # [1, n_q, T, V]
                B, n_q, T, V = logits.shape
                flat_logits = logits.view(-1, V)
                flat_target = c2.reshape(-1)

                loss = LOSS(flat_logits, flat_target)
                loss.backward()
                OPT.step()

                # compute and print accuracy
                with torch.no_grad():
                    preds = flat_logits.argmax(-1)
                    acc = (preds == flat_target).float().mean().item() * 100
                if acc>acc_threshold:
                    break
            # print(f"[Train-loop] epoch - loss={loss:.4f} – acc={acc:.1f}%", flush=True)
            MODEL.eval()

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _train_loop)

# ─── FastAPI app & endpoints ───────────────────────────────────────────
app = FastAPI(title="EnCodec Latent2Latent Seq2Seq", docs_url=None, redoc_url=None)

@app.post("/encode")
async def encode_ep(req: Request, file: UploadFile = File(None)):
    wav_raw_bytes = await (file.read() if file else req.body())
    if not wav_raw_bytes:
        raise HTTPException(400, "No audio supplied")
    return JSONResponse(_wav_to_latent(wav_raw_bytes))

@app.post("/decode")
async def decode_ep(vec: list = Body(...)):
    sr, n_q, T = map(int, vec[:3])
    codes = torch.tensor(vec[3:], dtype=torch.int64, device=DEVICE).view(1, n_q, T)
    HIST.append(codes)
    await _maybe_train()
    wav = _latent_to_wav(vec)
    buf = BytesIO()
    sf.write(buf, wav.cpu().numpy().reshape(-1,1),
             TARGET_SR, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return Response(content=buf.read(), media_type="audio/wav")

@app.post("/predict")
async def predict_ep(vec: list = Body(..., embed=True), wav: bool = False):
    # 1) parse input latent
    sr, n_q, T_in = map(int, vec[:3])
    flat = vec[3:]
    codes_in = torch.tensor(flat, dtype=torch.int64, device=DEVICE).view(1, n_q, T_in)

    # 2) build full-length latent of exactly T_LEN frames
    if T_in >= T_LEN:
        codes_full = codes_in[:, :, :T_LEN]
    else:
        pad_len = T_LEN - T_in
        pad = torch.zeros((1, n_q, pad_len), dtype=torch.int64, device=DEVICE)
        codes_full = torch.cat([codes_in, pad], dim=2)

    # 3) sliding-window inference
    window = MAX_TOKENS // n_q
    preds: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, T_LEN, window):
            end = min(start + window, T_LEN)
            chunk = codes_full[:, :, start:end]
            # pad last chunk if shorter than window
            if chunk.shape[2] < window:
                pad2 = torch.zeros((1, n_q, window - chunk.shape[2]),
                                   dtype=torch.int64, device=DEVICE)
                chunk_in = torch.cat([chunk, pad2], dim=2)
                pred_chunk = MODEL(chunk_in, chunk_in).argmax(-1)[:, :, :chunk.shape[2]]
            else:
                pred_chunk = MODEL(chunk, chunk).argmax(-1)
            preds.append(pred_chunk.cpu())

    full_pred = torch.cat(preds, dim=2)  # [1, n_q, T_LEN]
    flat_pred = full_pred.view(-1).tolist()
    out_vec = [TARGET_SR, n_q, T_LEN] + flat_pred

    if wav:
        wav_out = codec.decode(full_pred.to(DEVICE)).squeeze(0).float()
        buf = BytesIO()
        sf.write(buf, wav_out.cpu().numpy().reshape(-1,1),
                 TARGET_SR, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return JSONResponse({
            "latent": out_vec,
            "wav":   "data:audio/wav;base64," + buf.read().hex()
        })

    return JSONResponse(out_vec)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8081)
    args = p.parse_args()
    print(f"Audio Codec Ready on {DEVICE} – API: http://{args.host}:{args.port}", file=sys.stderr, flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
