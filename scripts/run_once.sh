#!/usr/bin/env bash
# 一键运行：安装（首次）→ 执行每日流水线 → 启动 Streamlit
set -euo pipefail
cd "$(dirname "$0")/.."

pip install -r requirements.txt
pip install -e .

python -m oilcast.pipeline.main "$@"
echo "报告已生成于 reports/latest/，启动交互页面…"
streamlit run src/oilcast/app.py
