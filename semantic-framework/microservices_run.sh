#!/usr/bin/env bash
# run_servers.sh — start text (in .venv), audio & image (system python3)
# Prints only the PIDs + final status; all other output goes to logs.

set -euo pipefail

# --clear to wipe logs first
CLEAR_LOGS=false
if [[ "${1-}" == "--clear" ]]; then
  CLEAR_LOGS=true
  shift
fi

# repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# logs
$CLEAR_LOGS && rm -rf logs
mkdir -p logs

# clean shutdown
trap 'kill $(jobs -p) 2>/dev/null || true; exit 0' INT TERM

# ensure venv exists for text
if [[ ! -x ".venv/bin/python" ]]; then
  echo "Error: .venv not found. Create it with: python3 -m venv .venv" >&2
  exit 1
fi

# launch text (venv)
(
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.venv/bin/activate"
  exec "$SCRIPT_DIR/.venv/bin/python" -u "$SCRIPT_DIR/src/text_translator.py"
) >> logs/text_translator.log 2>&1 &
pid_text=$!

# launch audio (system python3)
 /usr/bin/python3 -u "$SCRIPT_DIR/src/audio_translator.py" \
   >> logs/audio_translator.log 2>&1 &
pid_audio=$!

# launch image (system python3)
 /usr/bin/python3 -u "$SCRIPT_DIR/src/image_translator.py" \
   >> logs/image_translator.log 2>&1 &
pid_image=$!

# the only stdout lines:
echo "[PID] text server = $pid_text"
echo "[PID] audio server = $pid_audio"
echo "[PID] image server = $pid_image"
echo "All servers launched. Waiting… (Ctrl+C to stop)"

wait
