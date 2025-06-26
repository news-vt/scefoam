# Semantic Comm Framework

# How to use
on windows:
wsl python3 image_codec_server.py
wsl python3 audio_codec_server.py
wsl python3 text_codec_server.py

wsl npm test src/__tests__/timing_image.test.js
wsl npm test src/__tests__/timing_audio.test.js
wsl npm test src/__tests__/timing_text.test.js

### Python Virtual Env. commands
- Downloading uv: `pip install uv`
- Creates (or updates) python uv virtual env: `uv sync`
- Open up a Vir. Env. uv shell: `uv shell`
- Downloading a library: `uv pip install PackageName`
- Run a server for jupyter notebook: `uv run jupyter notebook`
- python3 -m uv sync
source .venv/bin/activate

wsl python3 -m uv run python src/text_codec_server.py
wsl python3 src/image_codec_server.py
wsl python3 src/audio_codec_server.py

python3 -m uv pip install --extras cpu -e .

wsl python3 -m uv run python3 src/text_codec_server.py

wsl npm test -- --runInBand



Create a new resolv.conf pointing at Google’s (or Cloudflare’s) DNS
wsl sudo bash -c 'cat > /etc/resolv.conf <<EOF
nameserver 8.8.8.8
nameserver 1.1.1.1
EOF'


sudo rm /etc/resolv.conf


wsl python3 -m uv run python3 src/text_codec_server.py

# Set up environment and install CPU dependencies
uv sync --extra cpu --extra eval --extra data

# For GPU support (example: Torch 2.5.1 + CUDA 12.1)
uv pip install torch==2.5.1 --extra-index-url https://download.pytorch.org/whl/cu121 --upgrade
uv pip install fairseq2==v0.3.0rc1 --pre --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/rc/pt2.5.1/cu121 --upgrade











# 0) fresh virtual-env + bootstrap tooling
python3 -m venv .venv
source .venv/bin/activate                     # ← every shell session!
python   -m ensurepip --upgrade               # pip gets installed
python   -m pip install --upgrade pip uv      # uv lives in the venv

# 1) install the project with its *default* (CPU) extras
uv sync                                        # reads pyproject.toml

# 2) switch the DL stack to CUDA *11.8*  (choose cu124 if that’s your wheelhouse)
# 2-a) replace torch + torchaudio with the CUDA build
uv pip install --upgrade --force-reinstall \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  torch==2.5.1+cu118 torchaudio==2.5.1+cu118

# 2-b) remove the CPU fairseq2n that came from uv sync …
uv pip uninstall fairseq2n                 # uv asks for confirmation, -y = yes

# 2-c) …and install the matching CUDA build
uv pip install --pre \
  --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/rc/pt2.5.1/cu118 \
  fairseq2n==0.3.0rc1                         # note the **+cu118** tag inside the wheel

# 3) sanity-check the result
python - <<'PY'
import torch, import importlib.metadata as im
print("torch     :", torch.__version__, torch.version.cuda)
print("fairseq2n :", im.version("fairseq2n"))
print("CUDA OK?  :", torch.cuda.is_available())
PY
# should print something like:
# torch     : 2.5.1+cu118 11.8
# fairseq2n : 0.3.0rc1+cu118
# CUDA OK?  : True

# 4) run the server
python src/text_codec_server.py               # now starts on “cuda”
