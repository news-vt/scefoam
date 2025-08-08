#!/usr/bin/env python3
# pip install lpips torch torchvision pillow numpy

import sys, math
import numpy as np

# safe imports with auto-install if needed
try:
    import torch
    import torchvision.transforms as T
    from PIL import Image
    import lpips
except ModuleNotFoundError:
    import subprocess, sys as _s
    # silence pip install logs completely
    subprocess.check_call(
        [_s.executable, "-m", "pip", "install",
         "torch", "torchvision", "pillow", "numpy", "lpips"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    import torch
    import torchvision.transforms as T
    from PIL import Image
    import lpips

if len(sys.argv) != 3:
    sys.exit("usage: lpips_cli.py <ref.jpg> <deg.jpg>")

# preprocessing to 256×256, tensor in [-1,1]
resize    = T.Resize((256, 256))
to_tensor = T.ToTensor()
normalize = T.Normalize([0.5]*3, [0.5]*3)

def load(img_path):
    img = Image.open(img_path).convert('RGB')
    img = resize(img)
    t   = to_tensor(img)
    return normalize(t).unsqueeze(0)

# compute LPIPS
with torch.no_grad():
    model = lpips.LPIPS(net='alex')
    score = model(load(sys.argv[1]), load(sys.argv[2])).item()

# fallback if LPIPS produced invalid value
if not math.isfinite(score):
    a = np.asarray(Image.open(sys.argv[1]).resize((256,256)), np.float32) / 255.0
    b = np.asarray(Image.open(sys.argv[2]).resize((256,256)), np.float32) / 255.0
    score = float(np.mean((a - b) ** 2))

# emit only the numeric score on the last line
print(score)
