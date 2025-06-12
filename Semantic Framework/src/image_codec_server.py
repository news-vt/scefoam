#!/usr/bin/env python3
"""
image_codec.py  ▸  FastAPI server edition
--------------------------------------------------
Runs a lightweight HTTP server that keeps the Stable-Diffusion VAE resident
between requests.  Start it once:

    $ python image_codec.py --host 0.0.0.0 --port 8080

…and point `image_controller.js` at http://<host>:8080 .
"""
import argparse
import logging
import warnings
import os
import math
from io import BytesIO

import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, Body, Request, HTTPException
from fastapi.responses import JSONResponse, Response
import uvicorn

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL

# ──────────────────────────────────────────────────────────────────────────────
#  One-time model load
# ──────────────────────────────────────────────────────────────────────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")      # silence TF deps
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LATENT_CH = 4  # (B,4,H,W)

print("Loading SD-VAE to", DEVICE, "…", flush=True)
VAE = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(DEVICE).eval()
print("Loaded VAE", flush=True)

# ──────────────────────────────────────────────────────────────────────────────
#  TorchScript wrappers
# ──────────────────────────────────────────────────────────────────────────────
class EncodeWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return the latent mean
        return self.vae.encode(x).latent_dist.mean

class DecodeWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # return the decoded image tensor
        return self.vae.decode(z).sample

# trace both paths once at startup
print("Tracing VAE for TorchScript…", flush=True)
_example = torch.randn(1, 3, 512, 512, device=DEVICE)
encode_wrap = EncodeWrapper(VAE).to(DEVICE)
scripted_encode = torch.jit.trace(encode_wrap, _example, strict=False)

_dummy_lat = scripted_encode(_example)
decode_wrap = DecodeWrapper(VAE).to(DEVICE)
scripted_decode = torch.jit.trace(decode_wrap, _dummy_lat, strict=False)
print("Image Codec Ready! - Access API via: http://<host>:8082", flush=True)

# ──────────────────────────────────────────────────────────────────────────────
#  Helper functions (encode / decode)
# ──────────────────────────────────────────────────────────────────────────────
def _preprocess(img: Image.Image) -> torch.Tensor:
    img = img.resize((512, 512), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

def _encode(img_bytes: bytes, ds: int):
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    w, h = img.size
    with torch.no_grad():
        # use the scripted encoder
        lat64 = scripted_encode(_preprocess(img))        # (1,4,64,64)
        lat_ds = F.interpolate(lat64, size=(ds, ds),
                               mode="bilinear", align_corners=False)
    flat = lat_ds.cpu().numpy().reshape(-1).tolist()
    return [w, h] + flat

def _decode(vec):
    if len(vec) > 2 and isinstance(vec[0], (int, float)):
        w, h, *latent = vec
    else:
        w = h = None
        latent = vec
    n = len(latent)
    ds = int(math.sqrt(n / LATENT_CH))
    lat = torch.as_tensor(latent, dtype=torch.float32,
                          device=DEVICE).reshape(1, LATENT_CH, ds, ds)
    with torch.no_grad():
        lat64 = F.interpolate(lat, size=(64, 64),
                              mode="bilinear", align_corners=False)
        img_t = scripted_decode(lat64)                  # (1,3,512,512)
    arr = ((img_t.cpu().numpy()[0].transpose(1, 2, 0) + 1) *
           127.5).clip(0, 255).astype("uint8")
    img = Image.fromarray(arr)
    if w and h:
        img = img.resize((w, h), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()

# ──────────────────────────────────────────────────────────────────────────────
#  FastAPI wiring
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="SD VAE Codec", docs_url=None, redoc_url=None)

@app.post("/encode")
async def encode_endpoint(request: Request, ds: int = 64):
    """Accept either raw bytes or multipart upload named 'file'."""
    ctype = request.headers.get("content-type", "")
    if ctype.startswith("multipart/form-data"):
        form = await request.form()
        up = form.get("file")
        if up is None:
            raise HTTPException(400, "No file field in form")
        img_bytes = await up.read()
    else:
        img_bytes = await request.body()
    vec = _encode(img_bytes, ds)
    return JSONResponse(vec)

@app.post("/decode")
async def decode_endpoint(vec: list = Body(...)):
    img_bytes = _decode(vec)
    return Response(content=img_bytes, media_type="image/jpeg")

# ──────────────────────────────────────────────────────────────────────────────
#  Entry-point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",   default="0.0.0.0")
    parser.add_argument("--port",   type=int, default=8082)
    parser.add_argument("--workers",type=int, default=1)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
