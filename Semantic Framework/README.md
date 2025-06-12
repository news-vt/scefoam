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







Create a new resolv.conf pointing at Google’s (or Cloudflare’s) DNS
wsl sudo bash -c 'cat > /etc/resolv.conf <<EOF
nameserver 8.8.8.8
nameserver 1.1.1.1
EOF'


sudo rm /etc/resolv.conf
