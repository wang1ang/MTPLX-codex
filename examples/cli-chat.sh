#!/usr/bin/env bash
set -euo pipefail

mtplx chat --model "${MTPLX_MODEL:-mtplx/Qwen3.6-27B-MTPLX-GDN8-Speed4-CyanKiwiMTP}"
