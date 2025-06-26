#!/usr/bin/env bash
# run_servers.sh — start text, audio, and image codec servers in parallel,
# with per-server logs and debug headers.
#
# Usage:
#   ./run_servers.sh          # appends to existing logs
#   ./run_servers.sh --clear  # clears logs/ before launching

set -euo pipefail

# parse --clear flag
CLEAR_LOGS=false
if [[ "${1-}" == "--clear" ]]; then
  CLEAR_LOGS=true
  shift
fi

# 1) Ensure we’re in the repo root (where this script lives)
cd "$(dirname "$0")"

# 2) Prepare logs directory
if $CLEAR_LOGS; then
  rm -rf logs
fi
mkdir -p logs

# 3) Clean shutdown on Ctrl+C
trap '
  echo "[STOP] $(date): Killing all servers…"
  kill $(jobs -p) 2>/dev/null
  exit 0
' INT TERM

# 4) Server list (now including text)
declare -A SERVERS=(
  # [text]=src/text_codec_server.py    # run with uv run
  [audio]=src/audio_codec_server.py  # plain python
  [image]=src/image_codec_server.py  # plain python
)

# 5) Launch each one
for name in "${!SERVERS[@]}"; do
  script="${SERVERS[$name]}"
  logfile="logs/${name}_codec_server.log"

  # Write debug header
  {
    echo "======================================"
    echo "[START] $(date): Launching ${name} codec server"
    if [[ "$name" == "text" ]]; then
      echo "Command: python3 -m uv run python3 ${script}"
    else
      echo "Command: python3 -u ${script}"
    fi
    echo "Log file: ${logfile}"
    echo "--------------------------------------"
  } >> "$logfile"

  # Start in background
  if [[ "$name" == "text" ]]; then
    python3 -m uv run python3 "$script" >> "$logfile" 2>&1 &
  else
    python3 -u "$script" >> "$logfile" 2>&1 &
  fi

  pid=$!
  echo "[PID] ${name} server = $pid" >> "$logfile"
  echo "→ Launched ${name} (PID $pid) — logging to ${logfile}"
done

# 6) Now wait for all of them
echo "All servers launched. Waiting… (Ctrl+C to stop)"
wait
