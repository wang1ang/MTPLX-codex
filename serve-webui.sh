#!/usr/bin/env bash
# Start MTPLX serve (a local MTPLX model) + Open WebUI connected to it.
# Usage: ./serve-webui.sh [model_path]   no arg -> pick from local models.
# Stop:  ./serve-webui.sh stop
set -euo pipefail

MTPLX_DIR="$HOME/models/MTPLX"
VENV="$MTPLX_DIR/.venv-mtplx"
# Locate the open-webui binary (it lives in a conda env named open-webui, which
# the default shell PATH usually can't find). Priority: (1) current PATH ->
# (2) bin of each conda env -> (3) hard-coded fallback path.
# To override: export OPEN_WEBUI=/path/to/open-webui before running (:= skips probing).
_find_open_webui() {
  command -v open-webui 2>/dev/null && return 0
  local base hit
  base="$(conda info --base 2>/dev/null || echo /opt/homebrew/Caskroom/miniconda/base)"
  for hit in "$base"/envs/*/bin/open-webui; do
    [ -x "$hit" ] && { echo "$hit"; return 0; }
  done
  local fallback="/opt/homebrew/Caskroom/miniconda/base/envs/open-webui/bin/open-webui"
  [ -x "$fallback" ] && { echo "$fallback"; return 0; }
  return 1
}
OPEN_WEBUI="${OPEN_WEBUI:-$(_find_open_webui || true)}"
MODELS_DIR="$HOME/.mtplx/models"   # downloaded MTPLX models live here (one subdir per model)

SERVE_PORT=8000
WEBUI_PORT=3000

# --- stop ---
if [ "${1:-}" = "stop" ]; then
  pkill -f "mtplx serve" 2>/dev/null || true
  pkill -f "open-webui serve" 2>/dev/null || true
  echo "Stopped MTPLX serve + Open WebUI"
  exit 0
fi

# --- pick model ---
# If a path is passed explicitly, use it; otherwise scan MODELS_DIR subdirs:
# one -> use it directly, many -> prompt to choose.
if [ -n "${1:-}" ]; then
  MODEL="$1"
else
  models=()
  while IFS= read -r d; do
    models+=("$d")
  done < <(find -L "$MODELS_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort)

  if [ ${#models[@]} -eq 0 ]; then
    echo "Error: no model directories found under $MODELS_DIR" >&2
    exit 1
  elif [ ${#models[@]} -eq 1 ]; then
    MODEL="${models[0]}"
    echo "Using: $(basename "$MODEL")"
  else
    echo "Available models:"
    for i in "${!models[@]}"; do
      printf "  [%d] %s\n" "$((i+1))" "$(basename "${models[$i]}")"
    done
    echo ""
    read -rp "Pick a model [1-${#models[@]}]: " choice
    if ! [[ "$choice" =~ ^[0-9]+$ ]] || (( choice < 1 || choice > ${#models[@]} )); then
      echo "Invalid choice" >&2
      exit 1
    fi
    MODEL="${models[$((choice-1))]}"
  fi
fi

# --- 1. MTPLX serve (background) ---
source "$VENV/bin/activate"
# Models like Qwythos ship their own <think> chat template and are reasoning
# models: they MUST use the model's own template (tokenizer) instead of the
# default local_qwen36, with the qwen3 reasoning parser enabled. Otherwise the
# model emits its step-by-step reasoning as the answer and never stops, and with
# a small max_tokens it can also trip a session-cache deadlock.
# To use a non-reasoning model: export CHAT_TEMPLATE_PROFILE=local_qwen36 REASONING_PARSER=none
CHAT_TEMPLATE_PROFILE="${CHAT_TEMPLATE_PROFILE:-tokenizer}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
echo "Starting MTPLX serve  ->  http://127.0.0.1:$SERVE_PORT  (model: $(basename "$MODEL"))"
echo "  chat-template-profile=$CHAT_TEMPLATE_PROFILE  reasoning-parser=$REASONING_PARSER"
mtplx serve --model "$MODEL" --host 127.0.0.1 --port "$SERVE_PORT" \
  --chat-template-profile "$CHAT_TEMPLATE_PROFILE" \
  --reasoning-parser "$REASONING_PARSER" &

# wait for /v1/models to be ready
echo "Waiting for serve to be ready..."
for i in $(seq 1 120); do
  if curl -s -o /dev/null "http://127.0.0.1:$SERVE_PORT/v1/models" 2>/dev/null; then
    echo "serve is ready."
    break
  fi
  sleep 2
done

# --- 2. Open WebUI (background), connected to MTPLX serve; logs to a file, not the terminal ---
# If open-webui isn't found, skip the web UI (API still works); don't fail the script.
WEBUI_LOG="$MTPLX_DIR/open-webui.log"
if [ -x "$OPEN_WEBUI" ]; then
  echo "Starting Open WebUI  ->  http://127.0.0.1:$WEBUI_PORT  (open-webui: $OPEN_WEBUI, log: $WEBUI_LOG)"
  # Disable WebUI's automatic background tasks (title/tags/follow-up generation):
  # after each turn they silently fire extra requests, and a reasoning model
  # (e.g. Qwythos) spends hundreds-to-thousands of tokens thinking on each one,
  # clogging serve's serial queue -- the symptom is "the second question just
  # spins forever". Disabling them makes multi-turn smooth.
  # ENABLE_PERSISTENT_CONFIG=False makes the env vars below take effect every
  # run; otherwise WebUI uses stale values from its DB (env is persisted on
  # first launch) and the disable-task settings have no effect.
  OPENAI_API_BASE_URL="http://127.0.0.1:$SERVE_PORT/v1" \
  OPENAI_API_KEY="dummy" \
  WEBUI_AUTH=False \
  ENABLE_PERSISTENT_CONFIG=False \
  ENABLE_TITLE_GENERATION=False \
  ENABLE_TAGS_GENERATION=False \
  ENABLE_FOLLOW_UP_GENERATION=False \
  ENABLE_AUTOCOMPLETE_GENERATION=False \
    "$OPEN_WEBUI" serve --host 127.0.0.1 --port "$WEBUI_PORT" >"$WEBUI_LOG" 2>&1 &
else
  echo "WARNING: open-webui not found, skipping the web UI (API only)."
  echo "         To set it manually: export OPEN_WEBUI=/path/to/open-webui and re-run."
fi

echo ""
[ -x "$OPEN_WEBUI" ] && echo "  Web chat : http://127.0.0.1:$WEBUI_PORT"
echo "  API      : http://127.0.0.1:$SERVE_PORT/v1"
echo "  Stop     : $0 stop"
echo ""
if [ -x "$OPEN_WEBUI" ]; then
  echo "NOTE: reasoning models (e.g. Qwythos) think before answering, spending"
  echo "      hundreds of tokens. If Open WebUI's default max_tokens is too small,"
  echo "      the thinking gets truncated (symptom: \"stuck / reasoning but no answer\")."
  echo "      Raise max_tokens in the WebUI to >=2048:"
  echo "        top-right Settings -> General/Advanced Params -> Max Tokens (num_predict) -> 2048+"
  echo "      (this value lives in the WebUI's own settings; the script can't bake it in.)"
  echo ""
fi
echo "---- MTPLX serve output below ----"
wait
