"""Streamlit Cloud 入口：把 src/ 加入导入路径后运行 oilcast 应用。

Streamlit Cloud 主文件路径请填仓库根目录的 app.py（本文件）。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

# oilcast.app 的页面代码在导入时执行（Streamlit 脚本式模块）
import oilcast.app  # noqa: E402,F401
