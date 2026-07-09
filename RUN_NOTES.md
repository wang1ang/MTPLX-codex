# MTPLX 运行命令

## 安装

```bash
cd ~/models/MTPLX
uv venv .venv-mtplx --python 3.12
source .venv-mtplx/bin/activate
uv pip install -e .
```

## tune（用本地 snapshot 路径，避免重下模型）

```bash
mtplx tune --retune --model \
  /Users/yang.wang/.cache/huggingface/hub/models--samuelfaj--Ornstein3.6-27B-MTP-NSC-ACE-SABER-6bit-MTPLX-Optimized-Speed/snapshots/faa56f1ba7c8e94f5cbe60250d130f8a4b54c262
```

## benchmark

```bash
# 默认跑法：default suite + Sustained profile
mtplx bench run \
  --model ~/.mtplx/models/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed \
  --output outputs/bench_qwen36_27b_$(date +%Y%m%d_%H%M%S).json
```

- `--model` 指向本地权重目录（注意是 `~/.mtplx/models/...`，不是空的 HF 缓存目录）。
- `--output` 把汇总 JSON 写到文件；不加也会打印到终端，产物仍落在 `outputs/cli/bench/<run_id>/`。
- 加 `--dry-run` 只打印将用的配置（profile / seed / 环境变量），不真正跑。

其他跑法：

```bash
mtplx bench run --model <路径> --profile performance-cold      # 冷启动峰值速度
mtplx bench run --model <路径> --profile performance-cold --max # Burst 模式(max-fan lane, 最高 8K 上下文)
mtplx bench prefill-ladder --model <路径> --contexts 512,2k,32k # 长上下文 prefill 曲线
mtplx bench run --model <路径> --suite long_code               # 贴近编码 agent 的长任务
```

### 实测基线 (2026-06-22, default suite)

| 模型 / 模式 | tok/s | first64 | last64 | 质量 |
|---|---|---|---|---|
| Qwen3.6-27B — Sustained | 36.73 | 33.81 | 41.16 | ✅ |
| Qwen3.6-27B — Burst (`performance-cold --max`) | 27.71 | 38.52 | 28.64 | ❌ 质量未过 |
| Ornstein3.6-27B SABER 6bit — Sustained | 24.29 | 23.78 | 24.70 | ✅ |

- Qwen3.6 Sustained 比 Ornstein 快约 1.5x，是当前 `serve-webui.sh` 默认。
- Burst 首段瞬时最快(38.5)，但持续吞吐更低、且本次 code 任务质量 gate 未过(括号未闭合 / 语法非法)——速度数不作有效对比，需要时重跑确认。

## Youssofal 模型变体选型 (来自 HF 模型卡)

### 27B — Speed vs Quality

| 维度 | Optimized-Speed (本地在用) | Optimized-Quality |
|---|---|---|
| 主体量化 | flat 4-bit affine | 8-bit MLX affine (Flat8, group64) |
| draft / sidecar | 3-bit draft head + INT4 MTP sidecar | INT8 proposal sidecar + BF16 辅助张量 |
| 体积 / 内存 | 16.4 GB | 27.6 GiB (标 30GB) |
| depth | 3, profile sustained | 3 |
| 峰值 tok/s (卡) | ~63 (M5 Max, 192 prompt) | decode ~33.6 |
| AR baseline | 60.1 | — |
| 接受率 D1/D2/D3 | — | 95.6% / 85.3% / 74.1% |

- Speed: 4-bit，省内存(16GB 够)、快约 1.9x，日常编码默认选它。
- Quality: 8-bit，质量更高，但 ~2x 内存 + ~半速；需 32GB+ 余量。

### 35B-A3B — Speed vs Balance

| 维度 | Optimized-Speed | Optimized-Balance |
|---|---|---|
| 主体量化 | 4-bit affine (group64) | 6-bit affine (group64) |
| MTP 专家权重 | INT4 (group32) | INT4 (group64) |
| 体积 | 21 GB | 29.6 GB |
| 默认 depth | D1 | D2 |
| 默认档 tok/s (卡) | 138.39 (接受 0.886) | 126.43 (接受 0.813/0.505) |
| AR baseline | 94.46 | 86.30 |

- Speed: 4-bit、21GB、最快(138)。
- Balance: 6-bit、29.6GB、质量更稳、略慢(126)；需内存放得下 29.6GB。

> 各卡 benchmark 非同机同条件，绝对值仅供参考，看相对趋势。

## 其他

```bash
mtplx inspect <本地路径>
mtplx serve --port 8000
mtplx chat
```

找其他模型的本地路径：`ls -d ~/.cache/huggingface/hub/models--<org>--<repo>/snapshots/*/`
