#!/usr/bin/env bash
# =====================================================================
# OilCast 本地一键启动（macOS / Linux）
#   用法：
#     bash scripts/run_local.sh                      # 直连
#     bash scripts/run_local.sh http://127.0.0.1:7890  # 首次就指定代理
#   也可事先： export OILCAST_PROXY=socks5://127.0.0.1:1080
#
#   流程：创建 .venv 虚拟环境 → 安装依赖 → 连通性自检 → 采集真实数据
#         并训练/生成报告 → 启动本地网页控制台（http://localhost:8501）。
#   采集时若某海外信源网络不可达，会暂停并倒计时 60 秒，等待你切换代理后
#   回车重试；不操作则倒计时结束自动跳过、继续后续步骤（缺失数据如实标记）。
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"

if [ "${1:-}" != "" ]; then export OILCAST_PROXY="$1"; fi

if [ ! -d .venv ]; then
  echo ">> 创建虚拟环境 .venv"
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo ">> 安装/更新依赖"
python -m pip install --upgrade pip -q
pip install -r requirements.txt -q
pip install -e . -q

echo ">> 代理连通性自检（不通过可稍后在网页侧栏或按运行提示更换）"
python -m oilcast.pipeline.main --proxy-test || \
  echo "   ! 当前通道访问探测地址失败；若海外信源采不到，请准备好代理。"

echo ">> 采集真实数据、训练模型并生成报告（首次约 8~15 分钟）"
python -m oilcast.pipeline.main || \
  echo "   ! 部分数据本次采集失败（已如实标记），仍启动控制台供查看。"

echo ">> 启动本地控制台，浏览器打开 http://localhost:8501 （Ctrl+C 退出）"
exec streamlit run src/oilcast/app.py
