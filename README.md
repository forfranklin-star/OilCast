# OilCast — Streamlit Community Cloud 部署版

多因素油价智能分析与预测系统（WTI / 布伦特 / 上海原油 INE / 美燃油 / 伦敦柴油）。
本包用于部署到 **Streamlit Community Cloud**；本地 Windows 一键版请用另一个包
（`OilCast-Local.zip`，双击 `scripts/run_local.bat`）。

## 一、部署到 Streamlit Cloud（只用网页操作）

1. **推到 GitHub**：在 GitHub 网页新建公开仓库（如 `OilCast`），进入仓库
   `Add file → Upload files`，把本包解压后的**全部内容**（含 `src/`、`app.py`、
   `requirements.txt`、`.github/` 等）拖入并提交，保持目录结构。
2. **创建 Streamlit 应用**：打开 https://share.streamlit.io 登录 →
   `Create app` → 选择刚才的仓库与分支 → **Main file path 填 `app.py`**
   （仓库根目录的入口）→ `Deploy!`。
3. 等待依赖安装完成，左侧栏即可看到与本地一致的界面（历史存档、价格标的切换、
   模型备份与恢复等）。

## 二、数据每日自动更新（持久化闭环）

Streamlit Cloud 容器自身的文件系统**不持久**（重启会还原），因此权威数据由
**GitHub Actions** 维护：

- 仓库 `Actions` 中 `daily-oil-report` 工作流每日 **UTC 01:00 = 北京 09:00**
  自动采集真实数据、训练/预测、生成报告，并把数据、模型工件、历史报告
  **commit 回仓库**，同时把最新静态网页发布到 **GitHub Pages**。
- 仓库更新后 Streamlit 会自动重部署；也可在应用右上菜单 `☰ → Reboot`
  立即读到最新数据。
- 首次使用需在仓库 `Settings → Actions → General` 确认 Actions 已启用；
  fork 的仓库需在 `Actions` 页手动启用工作流。

### 可选：EIA 数据密钥
仓库 `Settings → Secrets and variables → Actions → New repository secret`，
名称填 `EIA_API_KEY`，值填你的 EIA key（用于 EIA 现货/期货数据）。

## 三、重要说明（避免误解）

- Streamlit 页面上的"立即重新生成报告"在云端可触发一次真实采集，但**容器重启后
  不保留**该次产生的数据；跨天的权威数据与报告以 Actions 提交的内容和 Pages
  为准。需要完全持久、可离线，请使用本地版 `OilCast-Local.zip`。
- 只用真实、可追溯、带观测日期的数据；任何源不可达/过期/样本不足，对应模块
  标记为"不可用"，绝不合成补齐。
- 系统定位为**概率区间 / 校准 / 情景 / 风险与事件监控的决策支持**，不是日频
  自动交易信号发生器。
