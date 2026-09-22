@echo off
rem =====================================================================
rem OilCast 本地一键启动（Windows / cmd 或 PowerShell 均可双击/调用）
rem   用法：
rem     scripts\run_local.bat                       （直连）
rem     scripts\run_local.bat http://127.0.0.1:7890 （首次指定代理）
rem   流程：创建 .venv -> 安装依赖 -> 连通性自检 -> 采集/训练/生成报告
rem         -> 启动本地控制台 http://localhost:8501
rem   采集中海外信源不可达时，会暂停并倒计时 60 秒等待更换代理后回车重试；
rem   不操作则倒计时结束自动跳过、继续后续步骤。
rem =====================================================================
chcp 65001 >nul
cd /d "%~dp0\.."

if not "%~1"=="" set OILCAST_PROXY=%~1

where python >nul 2>nul
if errorlevel 1 (
  echo [错误] 未找到 python，请先安装 Python 3.10+ 并勾选 "Add Python to PATH"。
  pause
  exit /b 1
)

if not exist .venv (
  echo ^>^> 创建虚拟环境 .venv
  python -m venv .venv
)
call .venv\Scripts\activate.bat

echo ^>^> 安装/更新依赖
python -m pip install --upgrade pip -q
pip install -r requirements.txt -q
pip install -e . -q

echo ^>^> 代理连通性自检
python -m oilcast.pipeline.main --proxy-test
if errorlevel 1 echo    ! 当前通道访问探测地址失败；若海外信源采不到，请准备好代理。

echo ^>^> 采集真实数据、训练模型并生成报告（首次约 8~15 分钟）
python -m oilcast.pipeline.main
if errorlevel 1 echo    ! 部分数据本次采集失败（已如实标记），仍启动控制台供查看。

echo ^>^> 启动本地控制台，浏览器打开 http://localhost:8501 （Ctrl+C 退出）
streamlit run src\oilcast\app.py
pause
