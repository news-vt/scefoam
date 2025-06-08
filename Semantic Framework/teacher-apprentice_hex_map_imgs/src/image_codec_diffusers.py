#!/usr/bin/env python3
import sys, json, os, math, logging, time, warnings

# suppress all logs so stdout is only image data
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore", message=".*flash attention.*")

import numpy as np
from pathlib import Path
from io import BytesIO
from PIL import Image

# redirect stderr for debug logs
err_log = open("perf_log.txt", "a")
sys.stderr = err_log

# Try ONNX Runtime first
USE_ONNX = False
try:
    import onnxruntime as ort  # type: ignore
    USE_ONNX = True

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    enc_sess = ort.InferenceSession("vae_encoder.onnx", sess_options=sess_opts, providers=providers)
    dec_sess = ort.InferenceSession("vae_decoder.onnx", sess_options=sess_opts, providers=providers)

    print("ONNX providers:", enc_sess.get_providers(), file=sys.stderr)
except ImportError:
    pass

# PyTorch fallback
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL

device = "cuda" if torch.cuda.is_available() else "cpu"
if not USE_ONNX:
    VAE = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    VAE.eval()

LATENT_CH = 4

# Utility to log timings into perf_log.txt
def log_time(ms: float):
    with open("perf_log.txt", "a") as f:
        f.write(f"{ms:.1f}\n")

# Preprocess image
def preprocess(img: Image.Image):
    img = img.resize((512,512), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
    if USE_ONNX:
        return arr.transpose(2,0,1)[None, ...]
    t = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)
    return t.to(device)

# Shared encode logic with full-timing
def _encode(orig: Image.Image, ds: int):
    t_start = time.perf_counter()

    if USE_ONNX:
        inp = preprocess(orig)
        lat_out = enc_sess.run(None, {"pixel_values": inp})[0]
        if ds != 64:
            import cv2
            m = lat_out[0].transpose(1,2,0)
            m = cv2.resize(m, (ds,ds), interpolation=cv2.INTER_LINEAR)
            lat_out = m.transpose(2,0,1)[None, ...]

        result = lat_out.reshape(-1).tolist()
    else:
        t = preprocess(orig)
        with torch.no_grad():
            lat64 = VAE.encode(t).latent_dist.mean
        lat_ds = F.interpolate(lat64, size=(ds,ds), mode='bilinear', align_corners=False)
        result = lat_ds.cpu().numpy().reshape(-1).tolist()

    delta = (time.perf_counter() - t_start) * 1000
    print(f"Total encode time: {delta:.1f} ms", file=sys.stderr)
    return result

def encode_from_file(path: Path, ds: int):
    return _encode(Image.open(path).convert("RGB"), ds)

def encode_from_bytes(raw: bytes, ds: int):
    return _encode(Image.open(BytesIO(raw)).convert("RGB"), ds)

# Decode API with full-timing
def decode_to_image(latent_list, w=None, h=None):
    t_start = time.perf_counter()

    n = len(latent_list)
    ds = int(math.sqrt(n / LATENT_CH))
    arr = np.array(latent_list, dtype=np.float32)

    if USE_ONNX:
        lat = arr.reshape(1, LATENT_CH, ds, ds)
        out = dec_sess.run(None, {"latent": lat})[0]
        img = Image.fromarray(((out[0].transpose(1,2,0)+1)*127.5).clip(0,255).astype('uint8'))
    else:
        lat_t = torch.from_numpy(arr.reshape(1, LATENT_CH, ds, ds)).to(device)
        lat64 = F.interpolate(lat_t, size=(64,64), mode='bilinear', align_corners=False)
        with torch.no_grad():
            out = VAE.decode(lat64).sample
        img = Image.fromarray(((out.cpu().numpy()[0].transpose(1,2,0)+1)*127.5).clip(0,255).astype('uint8'))

    if w and h:
        img = img.resize((w,h), Image.LANCZOS)

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    image_bytes = buf.getvalue()

    delta = (time.perf_counter() - t_start) * 1000
    print(f"Total decode time: {delta:.1f} ms", file=sys.stderr)

    return image_bytes

# CLI
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: encode-file <jpg> [ds] | encode-bytes [ds] | decode")
    mode = sys.argv[1]
    if mode == "encode-file":
        path, ds = Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 32
        w, h = Image.open(path).size
        flat = encode_from_file(path, ds)
        sys.stdout.write(json.dumps([w, h] + flat))
        sys.stdout.flush()
    elif mode == "encode-bytes":
        ds = int(sys.argv[2]) if len(sys.argv) > 2 else 32
        raw = sys.stdin.buffer.read()
        w, h = Image.open(BytesIO(raw)).size
        flat = encode_from_bytes(raw, ds)
        sys.stdout.write(json.dumps([w, h] + flat))
        sys.stdout.flush()
    elif mode == "decode":
        arr = json.loads(sys.stdin.read())
        if isinstance(arr, list) and len(arr) > 2 and isinstance(arr[0], (int, float)):
            w, h = int(arr[0]), int(arr[1])
            latent = arr[2:]
        else:
            w = h = None
            latent = arr
        sys.stdout.buffer.write(decode_to_image(latent, w, h))
    else:
        sys.exit("Unknown mode; use encode-file, encode-bytes, or decode.")
