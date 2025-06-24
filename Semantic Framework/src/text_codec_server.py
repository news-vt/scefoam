#!/usr/bin/env python3

import argparse, traceback, sys, asyncio
from typing import List
from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import torch.optim as optim
import uvicorn

from sonar.inference_pipelines.text import (
    TextToEmbeddingModelPipeline,
    EmbeddingToTextModelPipeline,
)

# ── Configuration ────────────────────────────────────────────────────────────
ENCODER_NAME = "text_sonar_basic_encoder"
DECODER_NAME = "text_sonar_basic_decoder"
LANG         = "eng_Latn"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ── Initialize SONAR pipelines & infer embedding dim ─────────────────────────
print(f"Loading SONAR text encoder (‘{ENCODER_NAME}’) on {DEVICE}…", file=sys.stderr)
t2vec = TextToEmbeddingModelPipeline(
    encoder=ENCODER_NAME,
    tokenizer=ENCODER_NAME,
    device=torch.device("cuda"),
    dtype=torch.float16,
)

print(f"Loading SONAR text decoder (‘{DECODER_NAME}’) on {DEVICE}…", file=sys.stderr)
v2t = EmbeddingToTextModelPipeline(
    decoder=DECODER_NAME,
    tokenizer=ENCODER_NAME,
    device=torch.device("cuda"),
    dtype=torch.float16,
)

# dummy call to get D
_dummy = t2vec.predict([""], source_lang=LANG)
if isinstance(_dummy, torch.Tensor):
    EMB_SIZE = _dummy.size(1)
else:
    EMB_SIZE = _dummy.shape[1]
print(f"Detected embedding size: {EMB_SIZE}", file=sys.stderr)

# ──────────────────────────────────────────────────
#  FastFormer building block
# ──────────────────────────────────────────────────
class GlobalAdditiveAttn(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d = d_model; self.h = n_heads; head_dim = d_model//n_heads
        self.q = nn.Linear(d_model,d_model)
        self.k = nn.Linear(d_model,d_model)
        self.v = nn.Linear(d_model,d_model)
        self.fc = nn.Linear(d_model,d_model)
        self.scale = head_dim ** -0.5
    def forward(self,x):                # x:(B,N,d)
        B,N,_=x.shape
        q=self.q(x).view(B,N,self.h,-1)           # (B,N,h,dh)
        k=self.k(x).view(B,N,self.h,-1)
        v=self.v(x).view(B,N,self.h,-1)
        # global query / key
        q_g = (q.softmax(dim=1)*q).sum(dim=1,keepdim=True)   # (B,1,h,dh)
        k_g = (k.softmax(dim=1)*k).sum(dim=1,keepdim=True)
        # element-wise modulation
        y = v * (q_g * self.scale) + v * (k_g * self.scale)
        y = y.view(B,N,self.d)
        return self.fc(y)

# ── Latent‐Forecasting Model Setup ────────────────────────────────────────────
HISTORY: List[torch.Tensor]        = []
MODEL:       nn.Module | None      = None
OPT:         optim.Optimizer | None= None
LOSS = nn.MSELoss()
LR, EPOCHS = 3e-4, 20
TRAIN_LOCK = asyncio.Lock()

class LatentFastFormer1D(nn.Module):
    """Fast-Former over a *sequence* of EMB_SIZE scalar tokens."""
    def __init__(self, d_model: int = 32, n_heads: int = 4):
        super().__init__()
        self.inp   = nn.Linear(1, d_model)
        self.pos   = nn.Parameter(torch.randn(1, EMB_SIZE, d_model))
        self.attn  = GlobalAdditiveAttn(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.mlp   = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.out   = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, EMB_SIZE)
        h = x.view(x.size(0), EMB_SIZE, 1)
        h = self.inp(h) + self.pos
        h = self.norm1(self.attn(h))
        h = self.norm2(h + self.mlp(h))
        return self.out(h).view(x.size(0), EMB_SIZE)

async def _maybe_train():
    """Background training on HISTORY → builds/updates MODEL."""
    global MODEL, OPT
    if len(HISTORY) < 2:
        return
    # prepare pairs (H[t] -> H[t+1])
    xs = torch.stack(HISTORY[:-1])  # (N-1, EMB_SIZE)
    ys = torch.stack(HISTORY[1:])
    if MODEL is None:
        MODEL = LatentFastFormer1D().to(DEVICE)
        OPT   = optim.AdamW(MODEL.parameters(), lr=LR)
    async with TRAIN_LOCK:
        MODEL.train()
        for _ in range(EPOCHS):
            OPT.zero_grad()
            LOSS(MODEL(xs.to(DEVICE)), ys.to(DEVICE)).backward()
            OPT.step()
        MODEL.eval()

# ── FastAPI app ─────────────────────────────────────────────────────────────
app = FastAPI(title="SONAR Text Codec (with latent forecasting)", docs_url=None, redoc_url=None)

@app.post("/encode")
async def encode(texts: List[str] = Body(..., embed=True)):
    try:
        emb = t2vec.predict(texts, source_lang=LANG)
        if isinstance(emb, torch.Tensor):
            out = emb.cpu().tolist()
        else:
            import numpy as np
            out = emb.astype(np.float32).tolist()
        return JSONResponse(out)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Encode error: {e}")

@app.post("/decode")
async def decode(embeddings: List[List[float]] = Body(..., embed=True)):
    try:
        tensor = torch.tensor(embeddings, dtype=torch.float32, device=DEVICE)
        if tensor.ndim != 2 or tensor.size(1) != EMB_SIZE:
            raise HTTPException(400, f"Each embedding must be length {EMB_SIZE}")
        texts = v2t.predict(tensor, target_lang=LANG, max_seq_len=512)

        # ➡️ record for forecasting
        for vec in tensor:
            HISTORY.append(vec.detach().cpu())
        # ➡️ train in background
        asyncio.create_task(_maybe_train())

        return JSONResponse(texts)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Decode error: {e}")

@app.post("/predict")
async def predict(vec: List[float] = Body(..., embed=True), text: bool = False):
    """
    Forecast the *next* embedding given `vec`. 
    If ?text=true, also decode that forecast to text.
    """
    try:
        tensor = torch.tensor([vec], dtype=torch.float32, device=DEVICE)  # (1, EMB_SIZE)
        if tensor.size(1) != EMB_SIZE:
            raise HTTPException(400, f"Embedding must be length {EMB_SIZE}")

        # ensure at least one train pass
        if (MODEL is None) and len(HISTORY) >= 2:
            import asyncio as _asyncio
            _asyncio.run(_maybe_train())

        if MODEL is None:
            raise HTTPException(400, "Need ≥ 2 decode calls first to build prediction model")

        with torch.no_grad():
            nxt = MODEL(tensor).squeeze(0).cpu()

        latent = nxt.tolist()
        if text:
            txt = v2t.predict(nxt.unsqueeze(0), target_lang=LANG, max_seq_len=512)[0]
            return JSONResponse({"latent": latent, "text": txt})
        else:
            return JSONResponse(latent)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Predict error: {e}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()
    print(f"SONAR Text Codec Ready on {DEVICE} – http://{args.host}:{args.port}",
          file=sys.stderr, flush=True)
    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level="warning")
