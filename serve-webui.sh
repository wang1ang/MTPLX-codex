#!/usr/bin/env bash
# 起 MTPLX serve(本地 27B SABER 模型)+ Open WebUI 连它。
# 用法: ./serve-webui.sh [model_path]   不传则用下面的默认 Ornstein snapshot。
# 停止: ./serve-webui.sh stop
set -euo pipefail

MTPLX_DIR="$HOME/models/MTPLX"
VENV="$MTPLX_DIR/.venv-mtplx"
# 自动定位 open-webui 二进制(它装在名为 open-webui 的 conda 环境里,默认 shell 的
# PATH 通常找不到)。优先级:① 当前 PATH → ② conda 各 env 的 bin → ③ 写死兜底路径。
# 想手动指定:运行前 export OPEN_WEBUI=/path/to/open-webui 即可(此处 := 会跳过探测)。
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
MODELS_DIR="$HOME/.mtplx/models"   # 下载的 MTPLX 模型都放这里(每个模型一个子目录)

SERVE_PORT=8000
WEBUI_PORT=3000

# 旧默认: Ornstein 27B SABER 6bit(实测 ~24.3 tok/s)
#   $HOME/.cache/huggingface/hub/models--samuelfaj--Ornstein3.6-27B-MTP-NSC-ACE-SABER-6bit-MTPLX-Optimized-Speed/snapshots/faa56f1ba7c8e94f5cbe60250d130f8a4b54c262

# --- stop ---
if [ "${1:-}" = "stop" ]; then
  pkill -f "mtplx serve" 2>/dev/null || true
  pkill -f "open-webui serve" 2>/dev/null || true
  echo "已停止 MTPLX serve + Open WebUI"
  exit 0
fi

# --- 选模型 ---
# 显式传了路径就直接用;否则扫描 MODELS_DIR 下的子目录:单个直接用,多个让你选。
if [ -n "${1:-}" ]; then
  MODEL="$1"
else
  models=()
  while IFS= read -r d; do
    models+=("$d")
  done < <(find "$MODELS_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort)

  if [ ${#models[@]} -eq 0 ]; then
    echo "Error: $MODELS_DIR 下没找到任何模型目录" >&2
    exit 1
  elif [ ${#models[@]} -eq 1 ]; then
    MODEL="${models[0]}"
    echo "使用: $(basename "$MODEL")"
  else
    echo "可用模型:"
    for i in "${!models[@]}"; do
      printf "  [%d] %s\n" "$((i+1))" "$(basename "${models[$i]}")"
    done
    echo ""
    read -rp "选择模型 [1-${#models[@]}]: " choice
    if ! [[ "$choice" =~ ^[0-9]+$ ]] || (( choice < 1 || choice > ${#models[@]} )); then
      echo "无效选择" >&2
      exit 1
    fi
    MODEL="${models[$((choice-1))]}"
  fi
fi

# --- 1. MTPLX serve (后台) ---
source "$VENV/bin/activate"
echo "启动 MTPLX serve  →  http://127.0.0.1:$SERVE_PORT  (model: $(basename "$MODEL"))"
mtplx serve --model "$MODEL" --host 127.0.0.1 --port "$SERVE_PORT" &

# 等 /v1/models 就绪
echo "等待 serve 就绪…"
for i in $(seq 1 120); do
  if curl -s -o /dev/null "http://127.0.0.1:$SERVE_PORT/v1/models" 2>/dev/null; then
    echo "serve 就绪。"
    break
  fi
  sleep 2
done

# --- 2. Open WebUI (后台), 连 MTPLX serve;输出写到日志文件,不占终端 ---
# 没找到 open-webui 就跳过,只起 serve(API 仍可用),不让脚本因此挂掉。
WEBUI_LOG="$MTPLX_DIR/open-webui.log"
if [ -x "$OPEN_WEBUI" ]; then
  echo "启动 Open WebUI  →  http://127.0.0.1:$WEBUI_PORT  (open-webui: $OPEN_WEBUI, 日志: $WEBUI_LOG)"
  OPENAI_API_BASE_URL="http://127.0.0.1:$SERVE_PORT/v1" \
  OPENAI_API_KEY="dummy" \
  WEBUI_AUTH=False \
    "$OPEN_WEBUI" serve --host 127.0.0.1 --port "$WEBUI_PORT" >"$WEBUI_LOG" 2>&1 &
else
  echo "⚠️  没找到 open-webui,跳过网页界面(只起 API)。"
  echo "    手动指定: export OPEN_WEBUI=/path/to/open-webui 再重跑。"
fi

echo ""
[ -x "$OPEN_WEBUI" ] && echo "  网页聊天 : http://127.0.0.1:$WEBUI_PORT"
echo "  API      : http://127.0.0.1:$SERVE_PORT/v1"
echo "  停止     : $0 stop"
echo ""
echo "—— 以下为 MTPLX serve 输出 ——"
wait
