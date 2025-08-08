#!/usr/bin/env python3
# pip install numpy soundfile scipy librosa

import sys
import numpy as np
import soundfile as sf
from scipy.signal import correlate

# lazy‑install librosa if needed
try:
    import librosa
except ImportError:
    import subprocess, sys as _s
    subprocess.check_call(
        [_s.executable, "-m", "pip", "install", "librosa"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    import librosa

if len(sys.argv) != 3:
    sys.exit("usage: xcorr_cli.py <ref.wav> <deg.wav>")

def read_mono(path):
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32), sr

# load both files
ref, sr_ref = read_mono(sys.argv[1])
deg, sr_deg = read_mono(sys.argv[2])

# resample deg → sr_ref if needed
if sr_ref != sr_deg:
    deg = librosa.resample(deg, orig_sr=sr_deg, target_sr=sr_ref)
    sr_deg = sr_ref

# pad to equal length
L = max(len(ref), len(deg))
ref = np.pad(ref, (0, L - len(ref)), 'constant')
deg = np.pad(deg, (0, L - len(deg)), 'constant')

# zero‑mean normalize
ref -= ref.mean()
deg -= deg.mean()

# full normalized cross‑correlation
corr = correlate(ref, deg, mode='full')
denom = np.linalg.norm(ref) * np.linalg.norm(deg)
score = float(np.max(np.abs(corr)) / denom) if denom > 0 else 0.0

# emit score in [0,1]
print(score)
