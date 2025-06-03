#!/usr/bin/env python3
# File: src/transformer.py
"""
Sequence-to-sequence forecaster used by the apprentice.
Trains on consecutive 100-d word-vectors stored in knowledge_base_<PORT>.json
and returns the next-step vector as JSON:
    { "vector": [ … 100 floats … ] }
"""

from __future__ import annotations
import os, sys, json, math, warnings
from pathlib import Path
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", category=UserWarning)     # silence Torch

# one checkpoint per running instance (distinguished by PORT)
PORT        = os.getenv("PORT_ARG", "")                     # "" for teacher
MODEL_PATH  = Path(f"transformer_model_{PORT}.pt")

# ───────────────────────── encoder-decoder model ──────────────────────────
class Seq2SeqForecast(nn.Module):
    def __init__(self, feat: int, layers: int = 2,
                 heads: int = 4, ff: int = 256):
        super().__init__()
        enc = nn.TransformerEncoderLayer(feat, heads, ff)
        dec = nn.TransformerDecoderLayer(feat, heads, ff)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.decoder = nn.TransformerDecoder(dec, layers)
        self.in_proj  = nn.Linear(feat, feat)
        self.out_proj = nn.Linear(feat, feat)

        # fixed sinusoidal positional encoding (length 500 is plenty)
        pe = torch.zeros(500, feat)
        pos = torch.arange(0, 500, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, feat, 2) * (-math.log(1e4) / feat))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pos_emb", pe)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        S, T = src.size(0), tgt.size(0)
        src = self.in_proj(src) + self.pos_emb[:S].unsqueeze(1)
        tgt = self.in_proj(tgt) + self.pos_emb[:T].unsqueeze(1)
        memory = self.encoder(src)
        return self.out_proj(self.decoder(tgt, memory))

# ─────────────────────────── KB helpers ────────────────────────────
def load_kb(kb_path: str) -> dict:
    """Return wrapper with keys: map, hexHistory."""
    obj = json.loads(Path(kb_path).read_text(encoding="utf-8"))
    if "map" not in obj:                    # extremely old format
        obj = {"map": obj, "hexHistory": []}
    obj.setdefault("hexHistory", [])
    return obj

def seq_from_history(kb: dict) -> torch.Tensor:
    """
    Return the (T,F) tensor built from hexHistory, silently skipping
    entries that are not valid keys in kb["map"].
    """
    vecs = [kb["map"][h] for h in kb["hexHistory"] if h in kb["map"]]
    return torch.tensor(vecs, dtype=torch.float32)


# ─────────────────────────── I/O helpers ───────────────────────────
def save_ckpt(model: nn.Module, feat: int) -> None:
    torch.save({"feature_size": feat, "state": model.state_dict()}, MODEL_PATH)

def load_ckpt(feat: int) -> Seq2SeqForecast:
    ck = torch.load(MODEL_PATH, map_location="cpu")
    if ck.get("feature_size") != feat:
        raise ValueError("feature-size mismatch")
    net = Seq2SeqForecast(feat)
    net.load_state_dict(ck["state"])
    return net

# ───────────────────────── train & predict ─────────────────────────
def train(kb_path: str) -> None:
    try:
        _train_internal(kb_path)
        print(json.dumps({"status": "trained"}))
    except Exception as e:
        print(json.dumps({"error": str(e)}))

def _train_internal(kb_path: str) -> Seq2SeqForecast:
    kb   = load_kb(kb_path)
    seq  = seq_from_history(kb)                 # (T,F)
    if seq.size(0) < 2:
        raise RuntimeError("Need ≥2 timesteps to train")

    src    = seq.unsqueeze(1)                   # (T,1,F)
    tgt_in = seq[:-1].unsqueeze(1)              # (T-1,1,F)
    tgt_out= seq[1:].unsqueeze(1)               # (T-1,1,F)

    F     = seq.size(1)
    model = Seq2SeqForecast(F)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss  = nn.MSELoss()

    model.train()
    for _ in range(200):
        opt.zero_grad()
        loss(model(src, tgt_in), tgt_out).backward()
        opt.step()

    save_ckpt(model, F)
    return model

def predict(kb_path: str) -> None:
    kb   = load_kb(kb_path)
    seq  = seq_from_history(kb)
    if seq.size(0) < 2:
        print(json.dumps({"error": "Model not trained yet"}))
        return

    F     = seq.size(1)
    src   = seq.unsqueeze(1)
    tgt   = seq[-1:].unsqueeze(1)               # seed decoder with last step

    try:
        model = load_ckpt(F)
    except Exception:                           # no ckpt yet → train now
        model = _train_internal(kb_path)

    model.eval()
    with torch.no_grad():
        nxt = model(src, tgt)[0, 0].tolist()    # (F,)

    print(json.dumps({"vector": nxt}))

# ───────────────────────── command-line glue ───────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(json.dumps({"error": "Usage: transformer.py [train|predict] <kb>"}))
        sys.exit(1)

    # propagate port number so JS can set it when spawning
    os.environ["PORT_ARG"] = Path(sys.argv[2]).stem.rsplit('_', 1)[-1]

    {"train": train, "predict": predict}[sys.argv[1]](sys.argv[2])
