#!/usr/bin/env python3
"""
Diffusers‐based Image Codec using Stable Diffusion’s VAE (AutoencoderKL),
modified to produce a smaller latent by downsampling the 64×64 latents
to 32×32 before sending.

CLI:
  python image_codec_diffusers.py encode-file <frame.jpg>
      → read JPEG from disk, encode → JSON list of floats (4×32×32 = 4096 floats)

  python image_codec_diffusers.py encode-bytes
      → read raw JPEG bytes from stdin, encode → JSON list of floats

  python image_codec_diffusers.py decode
      → read latent JSON from stdin (length = 4096), upsample → 64×64,
        decode → PNG (512×512) to stdout
"""

import sys, json, os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # silence TF warnings

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from io import BytesIO
from PIL import Image

from diffusers import AutoencoderKL

# ─────────────────────────── constants ───────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VAE_ID = "stabilityai/sd-vae-ft-ema"

# ─────────────────────── load the AutoencoderKL ───────────────────────
def load_vae():
    global VAE
    try:
        VAE = AutoencoderKL.from_pretrained(VAE_ID).to(DEVICE)
        VAE.eval()
    except Exception as e:
        sys.stderr.write(f"Error loading VAE '{VAE_ID}': {e}\n")
        sys.exit(1)

load_vae()


# ──────────────────── helper: preprocess a PIL image ────────────────────
def preprocess_pil_image(img: Image.Image) -> torch.FloatTensor:
    """
    Given a PIL Image (RGB), resize to 512×512, scale to [-1,1],
    and return torch.FloatTensor of shape (1,3,512,512).
    """
    img_resized = img.resize((512, 512), resample=Image.LANCZOS)
    arr = np.array(img_resized, dtype=np.float32) / 127.5 - 1.0  # [-1,1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,512,512)
    return tensor.to(DEVICE)


# ─────────────────── encode from a file path ───────────────────
def encode_from_file(path: Path) -> list[float]:
    img = Image.open(path).convert("RGB")
    x   = preprocess_pil_image(img)
    with torch.no_grad():
        lat_dist = VAE.encode(x).latent_dist
        lat64   = lat_dist.mean   # shape [1,4,64,64]
    # Downsample to 32×32:
    lat32 = F.interpolate(lat64, size=(32, 32), mode='bilinear', align_corners=False)  # [1,4,32,32]
    arr   = lat32.cpu().numpy().astype("float32")
    return arr.reshape(-1).tolist()  # length = 4 * 32 * 32 = 4096


# ─────────────────── encode from raw bytes ───────────────────
def encode_from_bytes(raw_bytes: bytes) -> list[float]:
    img = Image.open(BytesIO(raw_bytes)).convert("RGB")
    x   = preprocess_pil_image(img)
    with torch.no_grad():
        lat_dist = VAE.encode(x).latent_dist
        lat64   = lat_dist.mean  # [1,4,64,64]
    # Downsample to 32×32:
    lat32 = F.interpolate(lat64, size=(32, 32), mode='bilinear', align_corners=False)  # [1,4,32,32]
    arr = lat32.cpu().numpy().astype("float32")
    return arr.reshape(-1).tolist()  # 4096 floats


# ──────────────────── decode a latent vector ────────────────────
def decode(vec: list[float]) -> Image.Image:
    arr    = np.asarray(vec, dtype=np.float32)
    length = arr.size
    # Expect length = 4*32*32 = 4096
    if length != 4 * 32 * 32:
        raise ValueError(f"Expected latent length 4096, got {length}")
    # Reshape to [1,4,32,32]
    lat32 = arr.reshape(1, 4, 32, 32).astype("float32")
    lat32 = torch.from_numpy(lat32).to(DEVICE)

    # Upsample back to [1,4,64,64]
    lat64 = F.interpolate(lat32, size=(64, 64), mode='bilinear', align_corners=False)

    with torch.no_grad():
        decoded = VAE.decode(lat64).sample  # [1,3,512,512], range [-1,1]
    decoded = decoded.cpu().numpy()[0]      # shape [3,512,512]
    decoded = ((decoded + 1.0) * 127.5).clip(0, 255).astype("uint8")
    img_arr = decoded.transpose(1, 2, 0)    # [512,512,3]
    return Image.fromarray(img_arr, mode="RGB")


# ───────────────────────── CLI entrypoint ─────────────────────────
if __name__ == "__main__":

    if len(sys.argv) < 2:
        sys.exit("usage:\n  encode-file <frame.jpg>\n  encode-bytes\n  decode")

    mode = sys.argv[1]

    if mode == "encode-file":
        if len(sys.argv) != 3:
            sys.exit("usage: image_codec_diffusers.py encode-file <frame.jpg>")
        flat = encode_from_file(Path(sys.argv[2]))
        print(json.dumps(flat))

    elif mode == "encode-bytes":
        # — read raw JPEG bytes from stdin —
        raw = sys.stdin.buffer.read()
        if not raw:
            sys.exit("No bytes on stdin for encode-bytes")
        flat = encode_from_bytes(raw)
        print(json.dumps(flat))

    elif mode == "decode":
        # — read latent JSON array from stdin, decode → PNG to stdout —
        raw = sys.stdin.read()
        try:
            vec = json.loads(raw)
        except Exception as e:
            sys.exit(f"Failed to parse JSON from stdin: {e}")
        img = decode(vec)
        buf = BytesIO()
        img.save(buf, format="PNG")
        sys.stdout.buffer.write(buf.getvalue())

    else:
        sys.exit(f"Unknown mode '{mode}'; use encode-file, encode-bytes, or decode.")
