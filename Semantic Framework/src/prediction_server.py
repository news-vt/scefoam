#!/usr/bin/env python3
# prediction_server.py – Continual-Transformer micro-service
import argparse, warnings, uvicorn, torch, torch.nn as nn
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
import continual

warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="CT-Predict", docs_url=None, redoc_url=None)

# modality-specific bottleneck widths
PROJ = {"text": 256, "image": 1024, "audio": 512}

# swap (B,D,T) ↔ (B,T,D)
class SwapLast(nn.Module):
    def forward(self, x): return x.transpose(-1, -2)

def make_ct(d_lat: int, proj: int, depth: int = 2, seq_len: int = 128):
    nhead = next((h for h in (8, 4, 2) if proj % h == 0), 1)
    def fac(_):
        return continual.RetroactiveTransformerEncoderLayer(
            d_model=proj, nhead=nhead,
            dim_feedforward=4*proj, sequence_len=seq_len,
        )
    ct = continual.TransformerEncoder(fac, num_layers=depth)
    return nn.Sequential(
        SwapLast(), nn.Linear(d_lat, proj, bias=False),
        SwapLast(), ct,
        SwapLast(), nn.Linear(proj, d_lat, bias=False),
        SwapLast()
    ).to(DEVICE)

MODELS, LAST = {}, {}
print("[CT-Predict] ready – models build lazily", flush=True)

@app.get("/health")
async def health(): return JSONResponse({"ok": True})

@app.post("/load")
async def load(cfg: dict = Body(...)): return JSONResponse({"ok": True})

@app.post("/predict")
async def predict(req: dict = Body(...)):
    mod  = req["modality"]                 # "text" | "audio" | "image"
    vec  = torch.tensor(req["latent"], dtype=torch.float32, device=DEVICE)
    dlat = vec.numel()
    key  = (mod, dlat)

    if key not in MODELS:
        print(f"[CT] building {mod} model (latent={dlat}, proj={PROJ[mod]})",
              flush=True)
        MODELS[key] = make_ct(dlat, PROJ[mod])
        LAST[key]   = None

    x = vec.view(1, dlat, 1)               # (B,D,T=1)
    with torch.no_grad():
        y   = MODELS[key](x)               # (B,D,T)
        out = y.squeeze(0).squeeze(-1)     # (D,)
    LAST[key] = x.detach()
    return JSONResponse(out.cpu().tolist())

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--port", type=int, default=8084)
    port = p.parse_args().port
    print(f"[CT-Predict] running on http://0.0.0.0:{port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
