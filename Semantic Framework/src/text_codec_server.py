#!/usr/bin/env python3

# Imports
import argparse, traceback, asyncio
from typing import List
import sys
import torch
import torch.nn as nn
from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput
import uvicorn
from image_codec_server import GlobalAdditiveAttn 

# Constants
MODEL_PATH = "facebook/bart-large"
EMB_SIZE   = 1024
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

class T5Autoencoder:
    def __init__(self, path: str = "facebook/bart-large", device="cpu"):
        self.device = device

        # tokenizer stays the same
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, model_max_length=512
        )

        # ⚠️ Use the Seq2SeqLM class, not the CausalLM one
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path
        ).to(self.device)
        self.model.eval()

        # ── NEW ── set these two so generate() can reference them
        self.d_model = self.model.config.d_model                  # 512 for t5-small
        self.decoder_start_token_id = self.model.config.decoder_start_token_id

    # ────────────────────────────────────────────────────────────────────
    #  Codec API
    # ────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def embed(self, text: str) -> torch.FloatTensor:
        """
        Return a 512-dim float32 latent for a whole document via mean-pool.
        (Same signature as before.)
        """
        # tokenize + encode
        inp = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        # get encoder hidden states: (1, seq_len, d_model)
        # enc = self.model.get_encoder()(**inp).last_hidden_state
        enc = self.model.model.encoder(**inp).last_hidden_state 
        # mean-pool over tokens → (1, d_model) → (d_model,)
        return enc.mean(dim=1).squeeze(0)
    
    @torch.no_grad()
    def generate(self,
                 latent_f32: torch.Tensor,
                 max_len: int = 512,
                 temp: float = 1.0
                ) -> str:
        """
        Reconstruct text from a 512-dim latent (float32).
        (Same signature as before.)
        """
        # expand latent to (1,1,d_model)
        expanded = latent_f32.view(1, 1, self.d_model)
        enc_out = BaseModelOutput(last_hidden_state=expanded)

        # start from the decoder_start token
        decoder_input_ids = torch.tensor(
            [[self.decoder_start_token_id]],
            device=self.device
        )

        out_ids = self.model.generate(
            encoder_outputs=enc_out,
            decoder_input_ids=decoder_input_ids,
            max_length=max_len,
            temperature=temp,
            top_p=0.9,
            do_sample=True,
            use_cache=True,
        )
        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True)

def quantize_to_int8(lat_f32: torch.Tensor) -> List[int]:
    return (lat_f32.clamp(-1, 1) * 127).round().to(torch.int8).tolist()

def dequantize_from_int8(lat_int8: List[int]) -> torch.FloatTensor:
    return (torch.tensor(lat_int8, dtype=torch.int8, device=DEVICE) / 127.0)

AE = T5Autoencoder(MODEL_PATH, DEVICE)

HISTORY : list[torch.Tensor] = []
MODEL   : nn.Module | None = None
OPT     : torch.optim.Optimizer | None = None
LOSS = nn.MSELoss()
LR, EPOCHS = 3e-4, 20
TRAIN_LOCK = asyncio.Lock()

class LatentFastFormer1D(nn.Module):
    """Fast-Former over a *sequence* of 1024 scalar tokens."""
    def __init__(self, d_model: int = 32, n_heads: int = 4):
        super().__init__()
        self.inp  = nn.Linear(1, d_model)
        self.pos  = nn.Parameter(torch.randn(1, EMB_SIZE, d_model))
        self.attn = GlobalAdditiveAttn(d_model, n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.mlp  = nn.Sequential(
            nn.Linear(d_model, 4*d_model), nn.GELU(),
            nn.Linear(4*d_model, d_model))
        self.out  = nn.Linear(d_model, 1)

    def forward(self, x):                 # x:(B,1024)
        h = x.view(x.size(0), EMB_SIZE, 1)
        h = self.inp(h) + self.pos
        h = self.norm(self.attn(h))
        h = self.norm(h + self.mlp(h))
        return self.out(h).view(x.size(0), EMB_SIZE)

async def _maybe_train():
    """
    Background task that (re)trains the Fast-Former whenever we have
    at least one (x → y) pair in HISTORY.
    """
    global MODEL, OPT
    if len(HISTORY) < 2:            # need just one successive pair
        return

    xs, ys = zip(*[(HISTORY[i], HISTORY[i + 1])
                   for i in range(len(HISTORY) - 1)])
    x = torch.stack(xs)
    y = torch.stack(ys)

    if MODEL is None:
        MODEL = LatentFastFormer1D().to(DEVICE)
        OPT   = torch.optim.AdamW(MODEL.parameters(), lr=LR)

    async with TRAIN_LOCK:
        MODEL.train()
        for _ in range(EPOCHS):
            OPT.zero_grad()
            LOSS(MODEL(x), y).backward()
            OPT.step()
        MODEL.eval()

def _forecast(vec_int8: list[int], want_text: bool = False):
    """
    Predict the next latent (and optionally decode it), ensuring that
    the model is trained at least once before use.
    """
    # If a scheduled background train hasn’t finished yet, do a
    # quick synchronous pass so the first /predict never fails.
    if (MODEL is None) and len(HISTORY) >= 2:
        import asyncio
        asyncio.run(_maybe_train())

    if MODEL is None:
        raise HTTPException(400, "Need ≥ 2 decode calls first")

    x = dequantize_from_int8(vec_int8).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        nxt_f32 = MODEL(x).squeeze(0)

    nxt_int8 = quantize_to_int8(nxt_f32)
    txt = AE.generate(nxt_f32) if want_text else None
    return nxt_int8, txt

app = FastAPI(title="Contra-BT5-small codec (int8 latent)",
              docs_url=None, redoc_url=None)

@app.post("/encode")
async def encode(texts: List[str] = Body(..., embed=True)):
    try:
        out = [quantize_to_int8(AE.embed(t).cpu()) for t in texts]
        return JSONResponse(out)
    except Exception as e:
        traceback.print_exc(); raise HTTPException(500, f"Encode error: {e}")

@app.post("/decode")
async def decode(embeddings: List[List[int]] = Body(..., embed=True)):
    try:
        outs = []
        for vec in embeddings:
            if len(vec) != EMB_SIZE:
                raise HTTPException(400, f"Each embedding must be {EMB_SIZE} numbers")
            lat_f32 = dequantize_from_int8(vec)
            HISTORY.append(lat_f32)              
            outs.append(AE.generate(lat_f32))
        asyncio.create_task(_maybe_train())     
        return JSONResponse(outs)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc(); raise HTTPException(500, f"Decode error: {e}")

@app.post("/predict")
async def predict_ep(vec: List[int] = Body(..., embed=True), text: bool = False):
    try:
        nxt, sent = _forecast(vec, text)
        return (JSONResponse({"latent": nxt, "text": sent})
                if text else JSONResponse(nxt))
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc(); raise HTTPException(500, f"Predict error: {e}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    print(f"Text Codec Ready on {DEVICE} – API: http://{args.host}:{args.port}", file=sys.stderr, flush=True)
    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level="warning")
