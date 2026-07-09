#!/bin/bash
# Interactive tune: list local MTPLX models, pick one, run `mtplx tune` on it.
# No arguments — just run ./tune.sh and choose from the menu.
set -euo pipefail

MTPLX_REPO="$(cd "$(dirname "$0")" && pwd)"   # repo root (script lives here)
ROOTS=(
  "$HOME/Documents/MTPLX/models"
  "$HOME/.mtplx/models"
)

# --- collect unique models (a dir with mtplx_runtime.json), dedup by realpath ---
declare -a PATHS=()
declare -a SEEN=()
for root in "${ROOTS[@]}"; do
  [ -d "$root" ] || continue
  for d in "$root"/*/; do
    [ -f "$d/mtplx_runtime.json" ] || continue
    real="$(cd "$d" && pwd -P)"
    dup=0
    for s in "${SEEN[@]:-}"; do [ "$s" = "$real" ] && dup=1 && break; done
    [ "$dup" = 1 ] && continue
    SEEN+=("$real")
    PATHS+=("$real")
  done
done

if [ "${#PATHS[@]}" -eq 0 ]; then
  echo "No MTPLX models found under: ${ROOTS[*]}"
  exit 1
fi

# --- print menu with a bit of metadata ---
echo "Local MTPLX models:"
echo
i=1
for p in "${PATHS[@]}"; do
  name="$(basename "$p")"
  info="$(python3 - "$p" <<'PY'
import json, sys, os
p = sys.argv[1]
try:
    rt = json.load(open(os.path.join(p, "mtplx_runtime.json")))
except Exception:
    rt = {}
arch = rt.get("arch_id", "?")
depth = rt.get("mtp_depth_max", "?")
se = rt.get("speed_evidence", {}) or {}
verdict = se.get("verdict", "?")
print(f"arch={arch}  saved_depth_max={depth}  verdict={verdict}")
PY
)"
  printf "  [%d] %s\n      %s\n      %s\n\n" "$i" "$name" "$p" "$info"
  i=$((i+1))
done

# --- prompt for selection ---
printf "Pick a model [1-%d]: " "${#PATHS[@]}"
read -r choice
if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt "${#PATHS[@]}" ]; then
  echo "Invalid choice."; exit 1
fi
MODEL="${PATHS[$((choice-1))]}"
echo
echo "Selected: $MODEL"

# --- tune options ---
printf "Re-tune from scratch (ignore saved result)? [Y/n]: "
read -r retune_ans
RETUNE=""
case "${retune_ans:-Y}" in [Nn]*) RETUNE="" ;; *) RETUNE="--retune" ;; esac

printf "Save the winning depth to the model? [Y/n]: "
read -r save_ans
NOSAVE=""
case "${save_ans:-Y}" in [Nn]*) NOSAVE="--no-save" ;; *) NOSAVE="" ;; esac

echo
echo "Running: mtplx tune --model <selected> $RETUNE $NOSAVE --verbose"
echo "(this loads the model and benchmarks each depth; fans will ramp)"
echo
cd "$MTPLX_REPO"
exec python3 -m mtplx.cli tune --model "$MODEL" $RETUNE $NOSAVE --verbose
