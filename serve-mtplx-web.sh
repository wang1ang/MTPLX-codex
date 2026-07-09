#!/usr/bin/env bash
# 起 MTPLX 自带的浏览器聊天界面(mtplx start web)——零第三方依赖,不需要 Open WebUI。
# 对比 serve-webui.sh:那个是 mtplx serve(纯 API)+ 第三方 Open WebUI;本脚本一条
# `mtplx start web` 就同时拉起 API server 和 MTPLX 自带的 web chat。
#
# 用法:
#   ./serve-mtplx-web.sh                  扫描 ~/.mtplx/models 让你选模型,然后起 web
#   ./serve-mtplx-web.sh <model_path>     显式指定模型路径(跳过菜单)
#   ./serve-mtplx-web.sh stop             停掉跑在 SERVE_PORT 上的 MTPLX
set -euo pipefail

MTPLX_DIR="$HOME/models/MTPLX"
VENV="$MTPLX_DIR/.venv-mtplx"
MODELS_DIR="$HOME/.mtplx/models"   # 下载的 MTPLX 模型都放这里(每个模型一个子目录)

SERVE_PORT=8000   # 与 llama / serve-webui.sh 统一到 8000(三套服务互斥跑)

source "$VENV/bin/activate"

# --- stop ---:用 mtplx 自带的 stop(按端口找进程,优雅 SIGTERM→SIGKILL)
if [ "${1:-}" = "stop" ]; then
  mtplx stop --port "$SERVE_PORT" 2>/dev/null || pkill -f "mtplx start" 2>/dev/null || true
  echo "已停止跑在 :$SERVE_PORT 上的 MTPLX"
  exit 0
fi

# --- 选模型 ---:与 serve-webui.sh 同样的逻辑。
# 显式传了路径就直接用;否则扫描 MODELS_DIR:单个直接用,多个让你选。
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

# --- 起 MTPLX 自带 web chat ---
# web      : 走浏览器聊天界面(start 的默认 surface,显式写出来更清楚)
# --model  : 用上面选好的本地路径,不走 onboarding 的模型问询
# --yes    : 用默认值,跳过交互式 onboarding(否则首跑会卡在向导)
# --port   : 统一到 8000
echo "启动 MTPLX web  →  http://127.0.0.1:$SERVE_PORT  (model: $(basename "$MODEL"))"
echo "  停止: $0 stop"
echo ""
mtplx start web --model "$MODEL" --yes --port "$SERVE_PORT"
