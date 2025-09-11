#!/usr/bin/env bash
# microservices_run.sh
set -euo pipefail

CLEAR_LOGS=false
TIMEOUT=60      # seconds; use --timeout 0 to wait indefinitely
SHOW_TAIL=true  # show last 50 log lines on timeout

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clear) CLEAR_LOGS=true; shift ;;
    --timeout) TIMEOUT="${2:-60}"; shift 2 ;;
    --no-tail) SHOW_TAIL=false; shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

$CLEAR_LOGS && rm -rf logs
mkdir -p logs

cleanup(){ kill $(jobs -p) 2>/dev/null || true; }
trap cleanup INT TERM

now_ms(){
  if date +%s%3N >/dev/null 2>&1; then date +%s%3N
  else perl -MTime::HiRes=time -e 'printf("%.0f\n",time()*1000)'
  fi
}
fmt_ms(){
  local ms="${1:-0}"          # fix: safe default under set -u
  local s=$(( ms / 1000 ))
  local msr=$(( ms % 1000 ))
  local h=$(( s / 3600 ))
  local m=$(( (s % 3600) / 60 ))
  local sec=$(( s % 60 ))
  printf "%02d:%02d:%02d.%03d" "$h" "$m" "$sec" "$msr"
}

wait_ready() {
  # wait_ready <name> <logfile> <regex> <elapsed_varname>
  local name="$1" log="$2" pattern="$3" outvar="$4"
  : > "$log"                               # ensure file exists
  local start="$(now_ms)"

  local ok=1
  if (( TIMEOUT > 0 )); then
    # -q: quiet (don’t echo the matched line)
    if timeout "${TIMEOUT}s" bash -c "stdbuf -oL -eL tail -n0 -F \"$log\" | grep -E -m1 -q -- \"$pattern\""; then
      ok=0
    fi
  else
    if bash -c "stdbuf -oL -eL tail -n0 -F \"$log\" | grep -E -m1 -q -- \"$pattern\""; then
      ok=0
    fi
  fi

  local elapsed=$(( $(now_ms) - start ))
  if (( ok == 0 )); then
    printf "[READY]  %-6s in %s\n" "$name" "$(fmt_ms "$elapsed")"
  else
    printf "[TIMEOUT] %-6s after %s (pattern not seen)\n" "$name" "$(fmt_ms "$elapsed")"
    if $SHOW_TAIL; then
      echo "---- last 50 lines of $log ----"; tail -n 50 "$log" || true; echo "--------------------------------"
    fi
  fi
  printf -v "$outvar" '%s' "$elapsed"
  return $ok
}

# Preconditions
if [[ ! -x ".venv/bin/python" ]]; then
  echo "Error: .venv not found. Create it with: python3 -m venv .venv" >&2
  exit 1
fi

# Launch
(
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.venv/bin/activate"
  exec "$SCRIPT_DIR/.venv/bin/python" -u "$SCRIPT_DIR/src/text_translator.py"
) >> logs/text_translator.log 2>&1 & pid_text=$!

/usr/bin/python3 -u "$SCRIPT_DIR/src/audio_translator.py" \
  >> logs/audio_translator.log 2>&1 & pid_audio=$!

/usr/bin/python3 -u "$SCRIPT_DIR/src/image_translator.py" \
  >> logs/image_translator.log 2>&1 & pid_image=$!

echo "[PID]   text  = $pid_text"
echo "[PID]   audio = $pid_audio"
echo "[PID]   image = $pid_image"
echo "All servers launched. Waiting for readiness… (Ctrl+C to stop)"

# Regex (tolerates ASCII/Unicode dashes)
dash_class='[-–—]'
pat_text='Text[[:space:]]+Codec[[:space:]]+Ready([[:space:]]+on[[:space:]]+\w+)?[[:space:]]+'"$dash_class"'[[:space:]]+API:'
pat_audio='Audio[[:space:]]+Codec[[:space:]]+Ready([[:space:]]+on[[:space:]]+\w+)?[[:space:]]+'"$dash_class"'[[:space:]]+API:'
pat_image='Image[[:space:]]+Codec[[:space:]]+Ready([[:space:]]+on[[:space:]]+\w+)?[[:space:]]+'"$dash_class"'[[:space:]]+API:'

# Wait in parallel
elapsed_text=0 elapsed_audio=0 elapsed_image=0
wait_ready "audio" "logs/audio_translator.log" "$pat_audio" elapsed_audio & w_audio=$!
wait_ready "image" "logs/image_translator.log" "$pat_image" elapsed_image & w_image=$!
wait_ready "text"  "logs/text_translator.log"  "$pat_text"  elapsed_text  & w_text=$!

wait "$w_audio" || true
wait "$w_image" || true
wait "$w_text"  || true

# Summary (overall = max of elapsed)
overall=$elapsed_text
(( elapsed_audio > overall )) && overall=$elapsed_audio
(( elapsed_image > overall )) && overall=$elapsed_image

echo "----------------------------------------"
echo "Startup timing summary:"
printf "  audio: %s\n" "$(fmt_ms "$elapsed_audio")"
printf "  image: %s\n" "$(fmt_ms "$elapsed_image")"
printf "  text : %s\n" "$(fmt_ms "$elapsed_text")"
echo "----------------------------------------"
printf "Overall ready (slowest): %s\n" "$(fmt_ms "$overall")"
echo "Servers are running. (Ctrl+C to stop)"
wait
