# OilCast · 多因素油价智能分析与预测系统

每天 **北京时间 09:00（UTC 01:00）** 自动完成「真实数据采集 → 质量门 → 特征工程 →
权重学习/模型训练 → 短中长期预测 → 交互式报告发布」，覆盖 **WTI、布伦特原油、上海原油（INE SC，人民币计价并按真实汇率换算美元）、美燃油（NYMEX 超低硫柴油 HO）与伦敦柴油（ICE Gasoil）** 五个价格标的，
输出带概率区间、事件量化影响、**分品种因素敏感度**、跨品种升贴水与文字解读的分析报告：

- **静态 HTML 报告**（图表引擎本地内联共享、不依赖外网 CDN，自动发布到 GitHub Pages，即"公开网页"）；
- **Streamlit 交互页面**（可部署 Streamlit Community Cloud，支持历史回溯）；
- **SQLite + CSV 双存储**（原始数据、数据谱系、特征、预测、权重全部可审计回溯）。

---

## 0. 最高原则：只用真实、可追溯、带观测日期的数据

> **模型只建立在真实观测之上；任何缺失、过期或无法验证的数据都保持缺失，
> 绝不用插值、外推或合成值"补齐"成假数据。**

落地机制：

1. **只认真实源、无演示/合成入口**：系统只抓取真实源；每个字段经**质量门**判定
   `ok / stale（过期）/ insufficient（样本不足）/ unavailable（不可达）`，状态写入报告
   顶部「数据谱系表」与 SQLite `data_lineage` 表（来源、URL、抓取时刻、首末观测日、
   样本数、滞后工作日）。产品代码不含任何合成数据兜底（合成生成器仅作为 pytest 离线夹具
   存在于 `tests/`，不进入发布包，也无法从 CLI/页面/Actions 触发）。
2. **缺口不连线、不填零**：历史价格缺口在图上断开（`connectgaps=False`）；缺失因素不填 0
   冒充"中性"，而是从权重归一化中剔除并标注；样本不足的模型直接输出
   `unavailable + 原因`，不硬算。
3. **月频 vintage**：CPI/非农等月频指标在两次发布之间沿用「最近一次真实发布值」，
   底层逐点保留其原始发布日期（vintage），报告显示该值的观测日而非当天。
4. **当前值标注观测日**：每个现价都写明"截至 YYYY-MM-DD 真实观测"，不把滞后发布说成当日价。
5. **数据新鲜度守门（防止"旧数据冒充今日新报告"）**：每期对比"本期最新交易日 vs 上一期
   训练截止 vs 报告日"，分三档并在报告顶部/Streamlit 显式标注——`fresh`（已取到最近交易日、
   训练截止向前推进，正常不打扰）；`unchanged`（红色：本期没拿到任何新交易日，训练截止未推进、
   模型没学到新行情，数值会与上期雷同，明确告知这不是最新判断并提示重跑/查谱系）；`lagging`
   （橙色：最新收盘落后报告日 ≥2 个工作日，提示主数据源未取到最新）。这样"运行了但数据没更新"
   不会再被误当成一次有效更新；主源 CNBC 的 HTTP 重试也提高到 2 次以对抗偶发抖动。

---

## 1. 系统架构

```
                          ┌──────────────────────────────────────────┐
                          │        data_sources 数据层（strict）       │
 CNBC近月期货 ─► 四价格标的│ @CL.1/@LCO.1/@HO.1/@GAS.1（新鲜度主源） │
 FRED ─────────► 宏观/汇率 │ DGS10/DGS2/DTWEXBGS/CPIAUCSL/PAYEMS/…   │
 GPRD ─────────► 地缘风险  │ Caldara-Iacoviello 日频指数              │
 GoogleNews RSS ► 事件/机构│ 不可达即留空，绝不伪造                    │
 yfinance/EIA ──► 备份/可选│ PoliteSession：UA、延时、重试、守 robots │
                          └───────────────┬──────────────────────────┘
                                          ▼
        quality 质量门 + lineage 谱系（状态/来源/观测日期/样本数）
                                          ▼
   storage（SQLite: 原始/谱系/事件/观点/预测/权重/报告 + CSV 快照）
                                          ▼
   features：九大类因素 → "利多为正"、无未来泄漏、缺失保持 NaN 不填零
             ▲ 交易时段 as-of 对齐：各市场真实收盘时刻不同（上海 INE 约 UTC07、
             WTI/Brent 等约 UTC21），跨品种/宏观特征按目标品种收盘时刻 backward
             as-of，只取已落定行情——上海 t 日只可见 t-1 海外收盘，杜绝跨时区
             隐性未来函数与升贴水错位（engineering.align_exogenous_to_target）
                                          ▼
   models ┬─ weights     RF + LASSO 融合（缺失因素剔除）→ 先验收缩 → 跨日 EMA
          ├─ short_term  Direct 多步梯度提升（10 交易日）+ 样本外残差区间 + ARIMA 基准
          ├─ mid_term    动态内生变量 VAR（66 交易日）+ 残差块 bootstrap
          ├─ long_term   高/中/低情景 + 2000 条蒙特卡洛路径（252 交易日）
          └─ evaluation  滚动原点回测（MAE/RMSE/方向命中率 vs 随机游走）
           （任一模型真实样本不足 → 显式 unavailable，不输出假预测）
                                          ▼
   reporting ┬─ static_html  自包含 HTML（谱系表/缺口断开/不可用明示）→ Pages
             ├─ narratives   每个数字可追溯的中文解读；不可用即说明原因
             └─ app.py       Streamlit 交互报告（历史存档/手动重跑/模型备份恢复）
                                          ▲
              GitHub Actions cron 01:00 UTC（北京09:00）每日触发并 commit 回仓库
```

## 2. 目录结构

```
oilcast/
├── config/config.yaml            # 全部可调参数：数据原则/质量门/窗口/先验/情景
├── src/oilcast/
│   ├── config.py / utils.py
│   ├── data_sources/
│   │   ├── calendar.py          # 交易日历：真实交易日为轴，非交易日不占位/不算缺口
│   │   ├── cnbc_client.py       # CNBC 近月连续期货（交易日当天即更新，价格新鲜度主源）
│   │   ├── fred_client.py       # FRED 公开 CSV 客户端（现货/宏观备源）
│   │   ├── prices.py            # 四价格标的多源链 + 新鲜度优先选源
│   │   ├── macro.py             # 真实宏观 + 月频 vintage 对齐
│   │   ├── quality.py           # 数据谱系 FieldLineage + 质量门
│   │   ├── events.py / institutional.py
│   │   └── collector.py         # 真实数据采集总调度（固定严格真实，无合成兜底）
│   ├── storage/database.py      # SQLite（含 data_lineage 谱系表）+ CSV 快照
│   ├── features/engineering.py  # 缺失不填零、因素可用性判定
│   ├── models/                  # short/mid/long/weights/evaluation/review/registry
│   ├── reporting/               # 静态 HTML、文字解读、Streamlit app
│   └── pipeline/main.py
├── models/artifacts/            # 跨期模型工件 joblib + manifest 版本链（可导入导出）
├── .github/workflows/daily_report.yml
├── scripts/（backfill.py、run_once.sh）
├── tests/（test_smoke.py；synth_fixture.py 为仅测试用的离线数据夹具，不进入发布包）
├── data/、reports/、requirements.txt、pyproject.toml、README.md
```

## 3. 本地快速开始

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.10+
pip install -r requirements.txt && pip install -e .

# 1) 生成一期报告（只用真实数据，缺失/过期会在报告中明示）
python -m oilcast.pipeline.main
#   --require-prices    WTI/Brent 真实价格均不可用时以退出码 2 失败（供 CI 告警）
#   --as-of 2026-09-02  指定基准日期
#   --list-models / --export-models bundle.zip / --import-models bundle.zip（见第 5 节）

# 2) 交互式报告（侧边栏可一键导出/导入学习快照）
streamlit run src/oilcast/app.py
```

产物：`reports/latest/index.html`（浏览器直接打开）、`latest.json`、
`reports/archive/YYYY-MM-DD.{json,html}`、`data/oilcast.db`。
一键脚本：`bash scripts/run_once.sh`。

## 4. 真实数据源与多源优先级链（failover）

每个字段都在 `config.yaml → data_sources.source_chains / news_rss_feeds` 中配置一张
**有序源列表**，运行时逐个尝试：宏观序列取第一个足量源；**价格序列采用『新鲜度优先』——全部尝试后选末次观测日期最新的源**（官方现货发布滞后时由交易日当天更新的近月期货自动顶上，不同口径只整条采用单一源、绝不拼接）；每次尝试
（源名、成功/失败、观测条数、耗时、失败原因）都写入数据谱系的"数据源优先级尝试链"列
与数据库 `data_lineage.tried_sources`，全程可审计。源之间**绝不混合拼接**；所有源都
失败才标记 `unavailable`，绝不造数。

### 4.1 价格 / 宏观源链（world=全球稳定可达；us=海外机房可达）
| 字段 | 优先级链（从左到右依次尝试） | Key |
|---|---|---|
| WTI | **CNBC `@CL.1` 近月连续期货(world,交易日当天即更新)** → FRED `DCOILWTICO` 现货(world,官方滞后1~3日) → EIA(需key) → Yahoo → yfinance；**新鲜度优先**自动选末次观测最新者 | 否 |
| Brent | **CNBC `@LCO.1` 近月连续期货(world)** → FRED `DCOILBRENTEU` 现货(world) → EIA(需key) → Yahoo `BZ=F` → yfinance；同样新鲜度优先 | 否 |
| 美燃油 NYMEX ULSD | **CNBC `@HO.1` 近月连续期货(world，美元/加仑)** → Yahoo `HO=F`(us) → yfinance；替代无法核验的国内0#柴油 | 否 |
| 伦敦柴油 ICE Gasoil | **CNBC `@GAS.1` 近月连续期货(world，美元/吨)**；海外可完整取数 | 否 |
| 上海原油 INE SC | **新浪国内期货 `SC0` 主力连续日K(world，人民币/桶，2018-03 上市至今)** → 东方财富 `113.sc0`(us 备源)；以真实 USDCNY 换算美元后再与 Brent 算升贴水 | 否 |
| 美债 10Y/2Y | FRED `DGS10/DGS2`(world) → **美国财政部官方收益率曲线 CSV**(world) | 否 |
| 美元指数 | FRED `DTWEXBGS` 广义美元(world) → Yahoo `DX-Y.NYB`(us) | 否 |
| 人民币/日元汇率 | **ECB Frankfurter 参考汇率（`api.frankfurter.dev`，免费无 key、海外稳定，工作日日频、回溯到 2018）** → FRED `DEXCHUS`(USDCNY)/`DEXJPUS`(USDJPY) → Yahoo `CNY=X`/`JPY=X`；前者用于上海原油美元换算，后者作为日元汇率敏感度因素。报告"同一真实时刻"面板显式标注所用汇率值与末次观测日，可在谱系表 `usd_cny` 行核验 | 否 |
| CPI/非农/联邦基金/需求 | FRED `CPIAUCSL`/`PAYEMS`/`FEDFUNDS`/`INDPRO`（月频，逐点带发布日期 vintage） | 否 |
| 地缘风险 GPRD | GPRD 当前 CSV → GPRD `data_gpr_export.xls` 备份地址；均不可达时用**多源真实事件计数代理**（明确标注口径） | 否 |

### 4.2 事件 / 机构观点 RSS 源链（按序累积、去重，直到拿满或源用尽）
| 顺序 | 源 | 类型 | 可达性 | 说明 |
|---|---|---|---|---|
| 1 | **OilPrice** `oilprice.com/rss/main` | 能源垂直 feed | world | 油气专业媒体，相关度最高，无需检索词 |
| 2 | Google News RSS | 关键词检索 | us | 海外机房可达 |
| 3 | Bing News RSS | 关键词检索 | us | 海外机房可达 |
| 4 | WSJ Markets RSS | 综合财经 feed | world | 兜底 |
| 5 | MarketWatch RSS | 综合财经 feed | world | 兜底 |

标题经主题归类 + 多空词典打分，输出"若该事件单独主导的单日价格影响（美元/桶）"。
源可达但当天确无某主题事件记为真实 0；**某主题在整个窗口一次都未出现则该因素保持缺失**
（不把"没监测到"伪装成"恒为 0 可建模"）。

### 4.3 成品油标的：以美燃油/伦敦柴油替代国内柴油
- 国内 0# 柴油（人民币/吨）没有可稳定核验、合规免费的**海外**公开序列，按数据原则取消该标的，
  不以国际油价固定比率推算冒充。
- 改以两个可完整取数、口径清晰的**成品油近月期货**替代，并各自标注报价单位、不与原油混用：
  - **美燃油**：NYMEX 超低硫柴油/取暖油 ULSD Heating Oil，CNBC `@HO.1`，单位 **美元/加仑**；
  - **伦敦柴油**：ICE Gasoil，CNBC `@GAS.1`，单位 **美元/吨**。
- 二者与 WTI/Brent 走同一套多源优先级链、交易日轴、短/中/长期模型与持久化热启动。
- CNBC 对 403/429 限频做了递增退避重试；每日一次的低频任务不会触发，若仍被限流则该标的
  诚实降级为 unavailable 并在谱系写明尝试链。

### 4.4 上海原油 INE SC、汇率换算与跨品种升贴水
- **上海原油**以人民币/桶报价（新浪 `SC0` 主力连续，东财备源），与美元报价品种**不直接做水平价差**；
  先用当日真实 **USDCNY（首选 ECB Frankfurter 每日参考汇率，FRED/Yahoo 依次兜底；缺失按自然日前向桥接 ≤7 天并带观测日，超期留缺）** 把 SC 换算成"美元/桶"，
  再与 Brent 计算对数价差 `spread_brent` 及其 5 日变化，刻画**上海原油换算美元后相对布伦特的升水/贴水波动**。换算方向为 log(元/桶 ÷ 元/美元)=log(美元/桶)，报告同时刻面板标注所用汇率值与观测日以便复核。
- 对每个品种统一构造 8 个跨品种特征：对 Brent、WTI 两个固定基准各 4 个——5/20 日相对强弱
  `rs_*_5d/20d`（无量纲，任何品种可用）、美元口径对数价差 `spread_*` 与 5 日变化（仅同为
  美元/桶口径的 WTI/Brent/上海原油(换算后) 有值；成品油单位不同，水平价差列保持缺失、绝不混用口径）。
- **分品种因素敏感度**：每个可用品种都用**自身真实样本独立学一套因素权重**并分别落库
  （`factor_weights` 主键为 报告日+品种+因素），报告以敏感度热力图横向对比——例如上海原油
  对全球需求前景、美债收益率与人民币汇率更敏感，而 WTI/Brent 对美联储预期更敏感；权重不做跨品种平均。
- **日元汇率（USDJPY）**作为第 10 个因素 `fx_jpy` 入模（z 分数 + 5 日变化），各品种对其敏感度分别学习。
- 迪拜/阿曼原油（DME）经核查**无合规免费的海外日度公开源**（官网/主流行情站均为付费或不可达），
  按"无信源不造数"原则**不纳入**，不以内插或固定价差冒充。

### 4.5 盘中分时与"同一真实时刻"对齐（解决夜盘同步）
- **问题**：日 K 的日期标签不等于同一时刻。上海 INE 除日盘(北京 09:00–11:30/13:30–15:00)外还有
  **夜盘(21:00–次日 02:30)**，该窗口与 WTI/Brent 欧美电子盘真正重叠、同一时刻都在成交；只用日 K
  收盘对齐会把"与海外同步的夜盘"和"次日日盘"捆在一根里，升贴水/传导口径粗糙（实证：上海标签日 D
  日收益与海外 **D-1** 日收益相关性最高，WTI 0.53、Brent 0.55，同日仅 0.24/0.29）。
- **分时采集**（`data_sources/intraday_client.py`，落库 `raw_intraday(symbol,ts,...)`，每日增量 upsert）：
  上海 SC 用新浪 **15 分钟 K**（时间戳即北京时），WTI/Brent/美燃油/伦敦柴油用 CNBC `1H`（UTC→北京时）。
  免费源分时历史有限（新浪约 5 个月、CNBC 约 3.3 个月），**拿不到更早的绝不回填造数**，靠每日积累变长。
  - **为何不用新浪 60 分 K**：新浪 60 分 K 会把夜盘最后一根（02:30 收盘）错标成当日 09:30（周五夜盘
    物理在周六凌晨 02:30 收盘，被错标成周六 09:30，而上期所周六不开盘），曾导致真实夜盘收盘价被截断
    逻辑误删、日频停在日盘 15:00；15 分 K 时间戳正确（02:30 整点）。另用 `ine_sc_session_mask` 上期所
    合法交易时段白名单（日盘 09:00–11:30/13:30–15:00、夜盘 21:00–次日 02:30，并按星期排除周日晚/周六
    白天等不开盘时段）剔除行情商错标的幽灵 K（如周一 00:00、周六 09:30），单测锁定。
- **统计主节点＝上海夜盘 02:30 收盘**：上海一个连续交易时段的真正收盘是**夜盘 02:30**（周五夜盘物理在
  周六凌晨、归属周五交易日）。`features/intraday_align.py` 在 **02:30（主）** 与 **15:00（日盘对照）** 两个
  锚点，用 `merge_asof(direction=backward, tolerance=60min)` 取各品种当时**已成交**的最近一根 K（北京 02:30
  ＝美东 14:30／伦敦 19:30，欧美电子盘仍在交易）：只取过去/当时、不取未来，锚点前 60 分钟无成交则缺失、
  不拿数小时前陈旧价硬桥（单测锁定"不把白天价带到夜里"）。
- **双口径融合、标注不混用**：上海原油在分时覆盖日，**自身收盘价、跨市场 Brent/WTI、美元换算与对布升贴水
  统一用 02:30 同时刻价**（原始日 K＝日盘 15:00 收盘仍落库保留、不改写）；更早日期维持 4.4/交易时段 as-of
  日频口径。海外四品种同处美/欧时区、日 K 同日已对齐，不做分时替换。
- **无泄漏 A/B 裁决**：在分时完全覆盖窗口滚动样本外回测，同时刻口径校准后 MAE 不劣于日频方予保留，变差即
  回退（`build_features(intraday_session=...)` 可一键关闭）。报告设"交易时段·同一真实时刻"面板：02:30
  同时刻升贴水时序图 + 02:30（主）/15:00（对照）双锚点截面 + 分时根数/来源，全程可审计。
- **夜盘交易日归属（`apply_session_rollover`）**：以"日盘日期"锚定一个交易日——北京时凌晨 00:00–07:59
  的 K 线是前一自然日晚开盘的夜盘（上海夜盘到次日 02:30、欧美电子盘到北京时凌晨），日期回退一天，使
  "周五 15:00 日盘 + 周六凌晨 02:30 夜盘收盘"同归周五，面板不再冒出周六空行；归属后仍晚于 `--as-of`
  的越界前瞻点（接口偶发返回的下一交易时段脏点）一律剔除，单测锁定。

### 4.6 爬虫礼仪与扩展
统一 `PoliteSession`：可识别的研究 UA（规避被部分官网丢弃的 "bot" 字样，可按源覆盖浏览器
头）、请求间隔、有限重试、分级超时预算、**同主机连续失败熔断**避免不可达源拖垮流水线。
新增数据源前先检查 `/robots.txt` 与服务条款；新增一个源只需在 config 链表里加一项并在
对应 client 实现抓取，优先级、留痕、质量门、熔断自动生效。

## 5. 模型方法论

- **时间轴以真实交易日为准（非交易时间不计算、不算缺口）**：历史时间轴直接取真实
  观测出现过的交易日（`data_sources/calendar.py`），不生成"周一到周五"硬网格去
  reindex——硬网格会把交易所假日排成交易日、凭空制造 NaN 假缺口。未来外推沿真实
  交易日推进，**周六周日绝不计入步长**，固定假日由"历史同一月-日从未交易"识别跳过，
  离线不可预知的临时休市等真实数据到来后自然校正。**走势图 X 轴采用交易日类别轴
  （plotly category，按交易日出现顺序等距排列）而非自然日历 date 轴**——非交易日根本不占
  横坐标，因此周末/节假日不会在图上留白形成断点，曲线在交易日之间严格连续；周末/假日
  运行报告时，滞后判定以最近工作日为基准。

- **因素体系（10 类）**：供给/OPEC+、地缘风险、美元、美债10Y、CPI 意外、非农、
  美联储政策预期、**日元汇率(USDJPY)**、需求前景、机构观点；统一"数值越大越利多油价"。
  另有跨品种相对强弱/升贴水、技术面等**内部特征组**（下划线前缀，进模型但不参与因素权重归一化）。
- **权重学习（逐品种各学一套）**：RandomForest 重要性 ×0.6 + |LASSO 系数| ×0.4 聚合到因素，
  **只在真实数据可用的因素间归一化**；向人工先验收缩、与该品种上一轮权重 EMA 平滑。
  **每个价格标的用自身真实样本独立学习，体现不同油种对局域冲突、美联储政策、CPI、非农、
  日元汇率等变量的不同敏感度**（如上海原油更受亚太需求/人民币汇率驱动，WTI 更受美国货币
  政策驱动），分别落库、报告以热力图横向对比，不做跨品种平均。真实样本不足时退回先验并标注。
- **地缘时变因子（刻画"边际升级"，非常年常数）**：
  - `china_supply_risk_5d`：从真实新闻标题正则识别霍尔木兹海峡/伊朗/油轮/封锁/扣押/断供等
    （中英文关键词），按事件真实强度做 5 日滚动聚合——专门捕捉**对华供油受扰的边际升级**
    （中国是伊朗及中东含硫原油主要买家，上海 INE SC 可交割中东含硫油，故对此最敏感）；
    源不可达整列缺失、可达但无命中记真实 0，归入"地缘风险"因素组随模型学习权重。
  - **重大事件 regime 分级（`data/reference/major_events.yaml` 的 `phases`）**：一场长期冲突
    内部允许"升级/缓和阶段"取不同强度（如 2026 美以伊战事长期对峙基准 0.75、开战封锁首月与
    9 月油轮战升级阶段 1.0），避免长期事件被压成无方差常数、树模型无法利用；每条阶段都要求
    公开可核验日期与来源。
  - **长期平均权重 ≠ 当前事件期敏感度**：因素热力图是近 500 交易日全窗平均，会把当前冲突期的
    高敏感度稀释；报告另用真实价格独立统计"重大事件期条件敏感度"（事件期/非事件期波动比、对
    滞后地缘溢价的传导 β 及其抬升 `beta_lift`、事件期累计涨跌），并配对比图——例如上海 SC 长期
    平均地缘权重不高，但当前霍尔木兹/伊朗窗口 β 抬升与累计涨幅居三大原油之首，两口径并列、不互相冒充。
- **短期（10 交易日）**：Direct multi-step 梯度提升（每步长一个模型，避免误差累积），
  区间来自 rolling-origin 样本外残差分位数，ARIMA 自动降阶作基准；点预测向随机游走收缩。
- **中期（66 交易日）**：油价 + 真实覆盖充分的宏观变量动态组建 VAR，AIC 选阶并校验
  特征根在单位圆外（稳定性），残差块 bootstrap 500 条路径叠加情景漂移。
- **长期（252 交易日）**：三情景 softmax 概率（缺失因素不投票、权重重新归一化），
  2000 条几何布朗路径；真实机构目标价中位数作外部锚并列展示。
- **每日重训 + 模型持久化热启动（持续学习、重启不从零）**：默认抓取并**用满近 10 年
  （`history_days=3650`，CNBC 日K实测可回溯 25 年，需要更长直接调大）真实历史训练**——
  短期 direct 梯度提升模型不再写死截断（旧版只取最近 540 行、数据量偏少，现已默认
  `max_train_rows=0` = 用满全部采集样本，约 2500+ 交易日）；`rolling_window=180` 仅作为
  样本外残差锚点、`resid_train_window=750` 控制残差回看长度与计算量。每期用截至当日最新
  真实数据训练；短期模型、样本外残差、ARIMA 参数等
  **工件以 joblib 落盘到 `models/artifacts/`，`manifest.json` 记录版本链（parent_version）、
  训练区间、样本行数、特征列与学习方式**。下一期（或进程/容器重启后）自动加载上期模型：
  特征 schema 一致时在旧模型上 `warm_start` **增量加树**、残差分位/波动与上期 EMA 融合，
  累计迭代到上限（`max_cum_iter`）才冷重建；仅当特征 schema 改变或无历史工件时才冷启动。
  - **固定特征 schema（不因数据源抖动从零）**：事件类特征在"本期无事件/新闻源一时不可达"
    时会整列缺失，或在事件稀疏时仅有零星几个取值；若动态删列会使特征数逐期变化、热启动
    频繁失效。模型因此采用固定列集合，对**有效样本不足的零信息列（整列全 NaN，或非空个数
    少于 `MIN_FEATURE_NONNULL=20` 的稀疏列）做常数 0 占位**——无方差、不会被树选作分裂，
    故不造假也不影响预测；仍有足量取值的列，其零星缺失交模型原生处理。这同时根除了新版
    scikit-learn(≥1.6)/numpy2 对全 NaN 或稀疏列分箱报
    `window shape cannot be larger than input array shape` 的问题（已在 sklearn 1.4 与
    1.9、numpy 1.26 与 2.5 双环境实测），预测时严格按训练时判定的零信息列一致占位。
  训练区间、有效行数、重训时刻、`vN 热启动自 vN-1`、本期被零占位的列都写入报告"模型学习"
  面板；权重向新数据学习并与上一期 EMA 平滑，历史权重在 `factor_weights` 表逐期可追溯。
- **完整学习快照的导入 / 导出（删库、换机、数据丢失后仍不从零）**：导出的 zip 不只是模型，
  而是**全部可学习状态**——① 各标的短期模型工件 + 版本链 manifest；② SQLite 中的因素权重
  历史 `factor_weights`（权重 EMA 续学）、历史预测 `forecasts`（到期复测/复盘）、报告日期
  索引 `reports`（定位"上一期"）。
  ```bash
  python -m oilcast.pipeline.main --list-models              # 查看已持久化模型版本
  python -m oilcast.pipeline.main --export-models bundle.zip # 导出完整学习快照
  python -m oilcast.pipeline.main --import-models bundle.zip # 整体恢复（模型+权重+历史）
  # 等价子模块：python -m oilcast.models.registry list|export|import
  ```
  **图形界面入口（无需命令行）**：`streamlit run src/oilcast/app.py` 后，在左侧
  「模型备份与恢复」面板：点 **① 生成学习快照 → ② 下载 bundle.zip 到本地** 即完成本地备份；
  在另一套环境（如云端 Streamlit）同一面板 **上传 bundle.zip → 确认导入并恢复** 即可。
  **本地 ↔ 云端迁移**：在已有学习积累的一侧导出 zip，到另一侧导入，之后生成报告即在原模型
  版本与权重上热启动续学（云端 Actions 与本地共用同一快照格式，可双向搬运）。
  **灾难恢复**：即使 `models/` 与 `data/oilcast.db` 同时丢失，导入快照后再跑一期，模型会在
  原版本上热启动（parent_version 连续）、权重 EMA 也能找到上一期续算，不会退回冷启动/先验。
  GitHub Actions 每日把 `models/` 与 `data/`（含上述三表）随 reports 一起 commit 回仓库，
  次日 CI 检出即自动续学；`--import-models/--export-models` 不依赖 plotly，最小环境也能恢复。
- **滚动样本外回测（复测①）**：最近若干滚动原点样本外检验 MAE/RMSE/方向命中率，
  并与随机游走基准对比、明确标注是否跑赢；价格缺口处不计误差。
- **方向双判据门控 + 经济价值考核（胜率≠盈利能力）**：方向立场为看涨/看跌/中性三分类，
  由两条**独立、严格无泄漏**的通道判定，任一成立才明确表态、否则诚实中性：
  ① *ML 概率通道*——交叉拟合 isotonic 校准后，高置信表态**胜率**显著高于 50%（二项检验）；
  ② *时序动量通道*——近 1 月动量(`mom_21`)强度越过其扩展中位数时顺势持有，按**经济价值**
  判据（平均收益>0、盈亏比≥1.15、t 检验 p≤0.10）。趋势跟随常"胜率不高但盈亏比>1、期望为
  正"，单看胜率会误杀，故回测/复测同时报告表态子集的**平均收益、盈亏比、近似年化夏普**与
  "无条件持有"对照。`edge_basis` 标注该周期 edge 来源（胜率/趋势期望）。九轮无泄漏诊断的
  硬结论：仅靠价格技术面+宏观+地缘，WTI/布伦特日度方向胜率围绕 50%（弱有效市场），系统不
  以放松门控或标签泄漏制造虚高命中；要系统性提高方向 edge 需引入价格之外的新信息（期货期限
  结构/基差、EIA 周度库存、CFTC 持仓，已在数据源规划中，采集到真实历史后由同一门控自动判定）。
- **历史预测复盘（复测②，learning loop 闭环）**：每期预测落 `forecasts` 表；之后每日
  运行时，把目标日已到期的历史预测与**现已实现的真实收盘价**逐条对照，统计平均误差、
  方向命中率、95% 区间覆盖率，并在报告列出最近的"预测 vs 实际"明细表；目标日尚无真实
  价格的预测不参与，绝不用填充值充当实际。复测误差是下一期重训的客观反馈。
  短期预测把未来 10 个交易日的**逐日路径全部落库**（而非只存两周终点），
  因此发布后次日起第 1 步预测即陆续到期、每天都有新复测样本，不必等满两周。
- **入模特征稳健化（跨版本兼容、不删列）**：短期模型训练前不剔除特征列，而是把【有效样本
  不足】的列（整列全 NaN，或非空少于 20 的稀疏事件列）常数 0 占位，保持固定 schema；这同时
  规避新版 scikit-learn(≥1.6)/numpy2 对全 NaN、稀疏列分箱报 `window shape cannot be larger
  than input array shape`（旧实现只挡全 NaN，事件稀疏时仍会复现，现已覆盖稀疏情形），并在
  sklearn 1.4 与 1.9 / numpy 1.26 与 2.5 双环境各跑通全部单测。
- **健壮性降级**：慢源（如 GPRD）单独限时，不拖累 FRED 主宏观；VAR 在内生变量不足、
  任一预测模型在建模边界异常时，该标的/周期优雅降级为 unavailable 并写明原因，
  绝不因局部失败中断整份报告，更不以估算值顶替。

## 6. 自动化部署（GitHub Actions + Pages / Streamlit Cloud）

1. GitHub 新建 **public** 仓库并推送；
2. **Settings → Pages → Build and deployment → Source 选 "GitHub Actions"**；
3. （可选）Settings → Secrets and variables → Actions 添加 `EIA_API_KEY`；
4. 每天 **UTC 01:00（北京 09:00）** 自动运行，结果 commit 回仓库并把整个 `reports/`
   部署为公开网页：站点根路径自动跳转到 `latest/index.html`，历史报告在 `archive/`，
   图表引擎 `assets/plotly.min.js` 随仓库发布，**不引用任何外网 CDN**；
5. Streamlit Cloud 连同一仓库、入口填 `src/oilcast/app.py`，每日 push 后自动刷新。

**确保定时任务真的会自动运行（GitHub Actions 机制，务必逐项确认）：**
- 定时表达式是 **UTC**：`on.schedule: cron: "0 1 * * *"` = 北京 09:00；GitHub 高峰期可能
  延迟数分钟到十几分钟，属正常现象（不是故障）。
- `schedule` **只在默认分支（通常 main）生效**；workflow 文件必须已合并进默认分支。
- **fork 或新建仓库后**第一次要到仓库 **Actions 标签页**点 "I understand my workflows,
  enable them" 启用；可先在 Actions 页选 `daily-oil-report` → **Run workflow** 手动跑一次验证。
- 工作流已声明 `permissions: contents: write`，用内置 `GITHUB_TOKEN` 自动 commit/push，
  无需额外配置 PAT；每日 commit 也让仓库保持活跃，避免 GitHub 对连续 60 天无活动仓库自动
  暂停定时任务。每次运行还会校验 `reports/latest/index.html` 是否真的生成，缺失即任务失败、
  可在 Actions 页看到红色叉号并接邮件提醒。

> 测试/容器隔离：设置环境变量 `OILCAST_HOME=/path` 可把数据库与报告重定向到该目录
> （pytest 已用临时目录自动隔离，测试夹具数据绝不会写入真实 `data/`）。

## 7. 常用配置（config/config.yaml）

| 参数 | 含义 | 默认 |
|---|---|---|
| `data_sources.mode` | 固定 strict（只认真实数据，无演示模式） | strict |
| `quality_gate.price_daily` | 价格最少观测/最大滞后工作日 | 120 / 7 |
| `quality_gate.macro_monthly.max_stale_days` | 月频发布滞后容忍 | 60 |
| `model.history_days` | 抓取历史长度（自然日，CNBC可回溯25年） | 3650（约10年） |
| `model.max_train_rows` | 短期训练最大行数，0=用满全部历史 | 0 |
| `model.rolling_window` / `resid_train_window` / `min_train_obs` | 残差锚点窗口 / 残差回看长度 / 建模最少真实样本 | 180 / 750 / 250 |
| `model.short/mid/long_horizon_td` | 三周期步长（交易日） | 10 / 66 / 252 |
| `model.weight_ema_alpha` / `prior_shrinkage` | 跨日平滑 / 先验收缩 | 0.6 / 0.7 |
| `model.persist_models` / `warm_start` | 工件落盘 / 旧模型增量热启动 | true / true |
| `model.max_cum_iter` / `residual_ema_alpha` | 单模型累计迭代上限 / 残差跨期融合 | 900 / 0.7 |

## 8. 测试

```bash
pytest -q   # 离线主链路（tests/synth_fixture 测试夹具）+ 数据原则（质量门/vintage/缺失不填零/样本不足拒绝训练）
```

## 9. 局限与免责

- 免费公开源可能限流或改版：对应字段表现为 `unavailable/stale` 并在报告明示，
  而不是用假数据保持"页面完整"；配置 `--require-prices` 可让 CI 在价格全失时显式失败告警。
- 国内0#柴油因无海外可核验真实源已取消，改以美燃油/伦敦柴油期货替代；迪拜/阿曼 DME 原油因无
  合规免费海外日度源同样不纳入；上海原油 INE SC 取新浪/东财主力连续，按真实 USDCNY 换算美元后
  才与 Brent 比较升贴水。任何源失败仍诚实标 unavailable，绝不内插造数。
- 规则词典情感打分为可解释基线，可替换为 FinBERT（保持输出表结构即可）。
- 预测区间反映历史波动与模型不确定性，**不构成投资建议**；第三方数据版权归原方所有。
