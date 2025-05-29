#!/usr/bin/env python3
# File: src/transformer.py   ← full replacement

from __future__ import annotations
import os
import sys
import json
import math
import warnings
from pathlib import Path

import torch
import torch.nn as nn

warnings.filterwarnings("ignore", category=UserWarning)  # hush torch notes

# every apprentice instance passes its own port via ENV → model file is unique
PORT       = os.getenv("PORT_ARG", "")                       # "" for teacher
MODEL_PATH = Path(f"transformer_model_{PORT}.pt")            # per-port file


# ─────────────────────── simple encoder–decoder ──────────────────────
class Seq2SeqForecast(nn.Module):
    def __init__(self, feat: int, layers: int = 2, heads: int = 4, ff: int = 256):
        super().__init__()
        # encoder
        enc_layer = nn.TransformerEncoderLayer(d_model=feat, nhead=heads,
                                               dim_feedforward=ff)
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        # decoder
        dec_layer = nn.TransformerDecoderLayer(d_model=feat, nhead=heads,
                                               dim_feedforward=ff)
        self.decoder = nn.TransformerDecoder(dec_layer, layers)
        # input/output projections
        self.input_proj  = nn.Linear(feat, feat)
        self.output_proj = nn.Linear(feat, feat)

        # positional embeddings (optional but usually helpful)
        pe = torch.zeros(500, feat)
        pos = torch.arange(0, 500, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, feat, 2) * (-math.log(10000.0) / feat))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pos_emb", pe)

    def forward(self, src, tgt):
        # src: (S, B, F), tgt: (T, B, F)
        S, B, F = src.shape
        T = tgt.shape[0]
        src = self.input_proj(src) + self.pos_emb[:S].unsqueeze(1)
        tgt = self.input_proj(tgt) + self.pos_emb[:T].unsqueeze(1)
        memory = self.encoder(src)
        out    = self.decoder(tgt, memory)
        return self.output_proj(out)


# ──────────────────────── helpers ───────────────────────────────────
def load_kb(p: str) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _flatten_snapshots(hist: list[dict]) -> tuple[torch.Tensor, list[str]]:
    """
    Convert a list of centroid‐snapshots [ {cid:vec}, … ]
    into a tensor of shape (T, F) and ordered list of IDs.
    """
    # collect all IDs in history, sorted numerically
    ids = sorted({cid for snap in hist for cid in snap}, key=lambda k: int(k))
    # infer vector dim
    any_vec = next(v for snap in hist for v in snap.values())
    dim     = len(any_vec)
    zeros   = [0.0] * dim

    rows = []
    for snap in hist:
        row = []
        for cid in ids:
            row.extend(snap.get(cid, zeros))
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32), ids


def save_ckpt(net: nn.Module, feat: int) -> None:
    torch.save({"feature_size": feat, "state": net.state_dict()}, MODEL_PATH)


def load_ckpt(feat: int) -> Seq2SeqForecast:
    chk = torch.load(MODEL_PATH, map_location="cpu")
    if chk.get("feature_size") != feat:
        raise ValueError("feature-size mismatch")
    model = Seq2SeqForecast(feat)
    model.load_state_dict(chk["state"])
    return model


# ───────────────────────── training core ────────────────────────────
def _train_internal(kb_path: str) -> Seq2SeqForecast:
    hist = load_kb(kb_path).get("centroidHistory", [])
    if len(hist) < 2:
        raise RuntimeError("Not enough history to train")

    seq, ids = _flatten_snapshots(hist)      # (T, F)
    S, F     = seq.shape
    # teacher‐forced input: feed all steps except the last as tgt_in,
    # and predict steps 1..T as tgt_out
    src    = seq.unsqueeze(1)                # (T,1,F)
    tgt_in = seq[:-1].unsqueeze(1)           # (T-1,1,F)
    tgt_out= seq[1:].unsqueeze(1)            # (T-1,1,F)

    model  = Seq2SeqForecast(F)
    opt    = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn= nn.MSELoss()

    model.train()
    for _ in range(200):                     # small data → more epochs
        opt.zero_grad()
        pred = model(src, tgt_in)            # (T-1,1,F)
        loss_fn(pred, tgt_out).backward()
        opt.step()

    save_ckpt(model, F)
    return model


# ───────────────────── public CLI commands ─────────────────────────
def train(kb: str) -> None:
    try:
        _train_internal(kb)
        print(json.dumps({"status": "trained"}))
    except Exception as e:
        print(json.dumps({"error": str(e)}))


def predict(kb: str) -> None:
    kbdata = load_kb(kb)
    hist   = kbdata.get("centroidHistory", [])
    if len(hist) < 2:
        print(json.dumps({"error": "Model not trained yet"}))
        return

    seq, ids = _flatten_snapshots(hist)     # (T, F)
    F         = seq.size(1)

    # prepare src tensor (full history)
    src = seq.unsqueeze(1)                  # (T,1,F)
    # decoder start: use the last vector as first input
    tgt = seq[-1].view(1,1,F)               # (1,1,F)

    # load or train
    try:
        model = load_ckpt(F)
    except Exception:
        model = _train_internal(kb)

    model.eval()
    with torch.no_grad():
        out = model(src, tgt)                # (1,1,F)

    nxt = out[0,0].tolist()                 # flat length‐F
    step = F // len(ids)
    fut  = {cid: nxt[i*step:(i+1)*step] for i,cid in enumerate(ids)}
    print(json.dumps({"centroids": fut}))


# ─────────────────────────── CLI wiring ────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(json.dumps({"error": "Usage: transformer.py [train|predict] <kb>"}))
        sys.exit(1)

    # expose port before anything else reads it
    os.environ["PORT_ARG"] = Path(sys.argv[2]).stem.rsplit('_',1)[-1]
    {"train": train, "predict": predict}[sys.argv[1]](sys.argv[2])
