#!/usr/bin/env python3
# File: transformer.py  (drop-in replacement)

from __future__ import annotations
import sys, json, math, warnings, torch
import torch.nn as nn
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)          # hush torch notes
MODEL_PATH = Path("transformer_model.pt")

# ───────────────── positional encoding ─────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):                           # (L,B,D)
        return x + self.pe[: x.size(0)]

# ───────────────── sequence forecaster ─────────────────────────────
class TransformerForecast(nn.Module):
    def __init__(self, feat: int, layers: int = 2, heads: int = 4, ff: int = 256):
        super().__init__()
        self.in_proj  = nn.Linear(feat, feat)
        enc_layer     = nn.TransformerEncoderLayer(d_model=feat, nhead=heads,
                                                   dim_feedforward=ff)
        self.encoder  = nn.TransformerEncoder(enc_layer, layers)
        self.out_proj = nn.Linear(feat, feat)

    def forward(self, src):                         # (L,B,D)
        return self.out_proj(self.encoder(self.in_proj(src)))

# ───────────────── helpers ─────────────────────────────────────────
def load_kb(p: str) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))

def flatten(hist: list[dict]) -> tuple[torch.Tensor, list[str]]:
    """history[list[{cid:vec}]] → (T,F) tensor  &  ordered centroid-id list"""
    ids  = sorted(hist[0], key=lambda k: int(k))
    rows = [[v for cid in ids for v in snap[cid]] for snap in hist]
    return torch.tensor(rows, dtype=torch.float32), ids

def save_ckpt(net: nn.Module, feat: int) -> None:
    torch.save({"feature_size": feat, "state": net.state_dict()}, MODEL_PATH)

def load_ckpt(feat: int) -> TransformerForecast:
    chk = torch.load(MODEL_PATH, map_location="cpu")
    if chk.get("feature_size") != feat:
        raise ValueError("feature-size mismatch")
    net = TransformerForecast(feat); net.load_state_dict(chk["state"]); return net

# ───────────────── training core ───────────────────────────────────
def _train_internal(kb_path: str) -> TransformerForecast:
    hist = load_kb(kb_path).get("centroidHistory", [])
    if len(hist) < 2:
        raise RuntimeError("Not enough history to train")

    seq, _      = flatten(hist)                # (T,F)
    src, tgt    = seq[:-1].unsqueeze(1), seq[1:].unsqueeze(1)
    feat        = seq.size(1)

    net         = TransformerForecast(feat)
    opt         = torch.optim.AdamW(net.parameters(), lr=1e-3)
    loss_fn     = nn.MSELoss()

    net.train()
    for _ in range(50):
        opt.zero_grad(set_to_none=True)
        loss_fn(net(src), tgt).backward()
        opt.step()

    save_ckpt(net, feat)
    return net

# ───────────────── public CLI commands ─────────────────────────────
def train(kb: str) -> None:
    try:
        _train_internal(kb)
        print(json.dumps({"status": "trained"}))
    except Exception as e:
        print(json.dumps({"error": str(e)}))

def predict(kb: str) -> None:
    hist = load_kb(kb).get("centroidHistory", [])
    if not hist:
        print(json.dumps({"error": "No history to predict"})); return

    seq, ids = flatten(hist); feat = seq.size(1)

    try:
        net = load_ckpt(feat)
    except Exception:                            # no ckpt or wrong size → train
        try:
            net = _train_internal(kb)
        except Exception as e:
            print(json.dumps({"error": f"Training failed: {e}"})); return

    net.eval()
    with torch.no_grad():
        out = net(seq.unsqueeze(1))              # (T,1,F)

    nxt  = out[-1, 0].tolist()
    step = feat // len(ids)
    fut  = {cid: nxt[i*step:(i+1)*step] for i, cid in enumerate(ids)}
    print(json.dumps({"centroids": fut}))

# ───────────────── CLI entrypoint ──────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(json.dumps({"error": "Usage: transformer.py [train|predict] <kb>"}))
        sys.exit(1)

    {"train": train, "predict": predict}.get(
        sys.argv[1],
        lambda _: print(json.dumps({"error": f"Unknown cmd {sys.argv[1]}"}))
    )(sys.argv[2])
