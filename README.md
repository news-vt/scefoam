# SCE-FOAM: Semantic Context-aware Framework for Adaptive Multimodal Reasoning
## Media

[![SCE-FOAM Demo](https://img.youtube.com/vi/Mh4FryIMvKk/0.jpg)](https://youtu.be/Mh4FryIMvKk?si=DL-SmaBZGGXF-cgG)



## Paper
**Title:** Artificial Intelligence (AI)-based Semantic Communications with Multimodal Data: Framework and Implementation  
**Author:** Jean-Luc Aristide DeRieux  
**Publisher:** Virginia Tech  
**Date:** September 22, 2025  
**URL:** [Access Paper](https://vtechworks.lib.vt.edu/items/eec7b677-ccbf-4579-b23e-7618b78bd724)
**Abstract:** Semantic communication (SC) has emerged as an effective paradigm for reducing the bandwidth needs of wireless services by exploiting the so-called "semantics" or meaning behind the data. To date, existing works in this area either focus on multimodal approaches only and omit context-aware recovery or embed it in cross-modal settings, such as audio-to-video, rather than providing a unified, modality-agnostic method. These works also impose substantial architecture redesigns for additional modalities support and are not easily extensible. In contrast to prior work, in this thesis, a novel semantic framework called the semantic context-aware framework for adaptive multimodal reasoning (SCE-FOAM) is proposed. SCE-FOAM is a multimodal semantic framework that enables compact transmission, efficient reconstruction, and contextually-aware predictions using a unique microservice-based architecture. This unique design simultaneously offers an extensible and modular platform for incorporating new modalities and enabling scalable deployment strategies. Experimental results show that SCE-FOAM can achieve data reductions up to 50% for text, 94.56% for audio, and 98.70% for images, respectively. Lastly, the proposed contextual‑prediction model achieves an average accuracy of 90% across all modalities. In addition, a heuristic-based extension of the deferred acceptance (DA) matching algorithm is proposed. The extension enables network node matches that incorporate exploration, coverage, and diversification heuristics. In summary, this thesis presents a unified, extensible, context-aware, multimodal SC framework and a heuristic extension to the DA matching algorithm.

## Quickstart

```bash
# Set up environment
python3 -m venv .venv
source .venv/bin/activate
pip install uv
uv sync

# Run translators
wsl python3 src/text_translator.py
wsl python3 src/image_translator.py
wsl python3 src/audio_translator.py
```

---

## How to Use

### Running Translators (Windows via WSL)
```bash
wsl python3 src/image_translator.py
wsl python3 src/audio_translator.py
wsl python3 src/text_translator.py
```

### Running Tests
```bash
wsl npm test src/tests/timing_image.test.js
wsl npm test src/tests/timing_audio.test.js
wsl npm test src/tests/timing_text.test.js
wsl npm test -- --runInBand
```

---

## Python Virtual Environment (using `uv`)

### Setup
```bash
# Install uv
pip install uv

# Create or update a virtual environment
uv sync

# Activate uv shell
uv shell
```

### Common Commands
```bash
# Install a package
uv pip install PackageName

# Run Jupyter Notebook server
uv run jupyter notebook

# Install CPU dependencies
uv sync --extra cpu --extra eval --extra data

# Install project with extras
python3 -m uv pip install --extras cpu -e .
```

### Running Translators from Virtual Env
```bash
source .venv/bin/activate
python src/text_translator.py

# Alternative
wsl python3 -m uv run python src/text_translator.py
```

---

## Networking (WSL DNS Setup)

To fix DNS resolution issues inside WSL:

```bash
wsl sudo bash -c 'cat > /etc/resolv.conf <<EOF
nameserver 8.8.8.8
nameserver 1.1.1.1
EOF'

sudo rm /etc/resolv.conf
```

---

## GPU Support

### Install Torch + Fairseq2 (CUDA 12.1 example)
```bash
uv pip install torch==2.5.1   --extra-index-url https://download.pytorch.org/whl/cu121 --upgrade

uv pip install fairseq2==v0.3.0rc1 --pre   --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/rc/pt2.5.1/cu121 --upgrade
```

---

## Full Setup Workflow

### 0) Fresh Virtual Env + Bootstrap
```bash
python3 -m venv .venv
source .venv/bin/activate             # Run in every new shell session
python -m ensurepip --upgrade
python -m pip install --upgrade pip uv
```

### 1) Install Project (CPU defaults)
```bash
uv sync
```

### 2) Switch to CUDA Build (example: CUDA 11.8)
```bash
# Replace torch + torchaudio with CUDA build
uv pip install --upgrade --force-reinstall   --extra-index-url https://download.pytorch.org/whl/cu118   torch==2.5.1+cu118 torchaudio==2.5.1+cu118

# Remove CPU fairseq2n
uv pip uninstall fairseq2n

# Install CUDA fairseq2n build
uv pip install --pre   --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/rc/pt2.5.1/cu118   fairseq2n==0.3.0rc1
```

### 3) Verify Installation
```bash
python - <<'PY'
import torch, import importlib.metadata as im
print("torch     :", torch.__version__, torch.version.cuda)
print("fairseq2n :", im.version("fairseq2n"))
print("CUDA OK?  :", torch.cuda.is_available())
PY
```

Expected output:
```
torch     : 2.5.1+cu118 11.8
fairseq2n : 0.3.0rc1+cu118
CUDA OK?  : True
```

### 4) Run Server
```bash
python src/text_translator.py
```
