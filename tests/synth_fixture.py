"""【仅测试使用】结构自洽的合成时间序列夹具。

产品代码不含任何演示/合成兜底（严格只用真实可追溯数据）；本模块只服务 pytest，
让端到端测试在离线、不联网的情况下也有结构合理的数据可跑，绝不进入发布包。
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd

from oilcast.config import get_config
from oilcast.utils import get_logger, seeded_rng

LOG = get_logger(__name__)

GEO_THEMES = [
    ("红海航运袭击加剧", "geopolitical_risk", 1.0),
    ("中东局势升级，产油设施遇袭", "geopolitical_risk", 1.2),
    ("俄罗斯原油出口遭新制裁", "supply_disruption", 0.9),
    ("OPEC+宣布延长自愿减产", "supply_disruption", 0.8),
    ("产油国政局动荡", "geopolitical_risk", 0.7),
    ("委内瑞拉/伊朗供应受限", "supply_disruption", 0.6),
]
MACRO_THEMES = [
    ("美国CPI超预期，降息预期降温", "cpi_surprise", -0.8),
    ("美国非农就业强于预期", "jobs_surprise", -0.5),
    ("美国非农走弱，衰退担忧升温", "jobs_surprise", 0.6),
    ("EIA原油库存意外下降", "demand_outlook", 0.5),
    ("IEA下调全球需求增速", "demand_outlook", -0.7),
    ("美联储释放降息信号", "fed_policy_expectation", 0.7),
]


def _ar1(n: int, phi: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros(n)
    eps = rng.normal(0, sigma, n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return x


def build_synthetic_bundle(as_of: datetime, history_days: int) -> Dict[str, pd.DataFrame]:
    """生成 prices / macro / events / views 四张表。"""
    cfg = get_config()
    rng = seeded_rng()
    dates = pd.bdate_range(end=pd.Timestamp(as_of).normalize(), periods=int(history_days * 5 / 7))
    n = len(dates)

    # ---------- 潜变量 ----------
    demand = _ar1(n, 0.95, 0.004, rng)                 # 需求景气（零附近均值回归）
    supply_cut = _ar1(n, 0.90, 0.004, rng)             # 供给收缩压力
    # GPR 地缘风险指数：直接在真实量纲（中枢 100）上做 AR(1) + 跳跃 + 指数衰减
    gpr = np.zeros(n)
    gpr[0] = 100.0
    g_noise = rng.normal(0, 1.2, n)
    for t in range(1, n):
        gpr[t] = 100 + 0.90 * (gpr[t - 1] - 100) + g_noise[t]
    jump_idx = rng.choice(n, size=max(6, n // 90), replace=False)
    for j in jump_idx:
        size = rng.uniform(15, 45)                     # 单次地缘冲击抬升 15~45 点
        decay = np.exp(-np.arange(n - j) / 8)
        gpr[j:] += size * decay
    gpr = np.maximum(60, gpr)

    # 美元指数：随机游走 + 与需求负相关（日波动受控，避免单一因素方差碾压）
    dxy_ret = -0.15 * demand + rng.normal(0, 0.0022, n)
    dxy = 103 * np.exp(np.cumsum(dxy_ret))
    # 美债10Y：与 DXY 同向、慢变，水平 3.5~5.0
    us10y = 4.2 + 1.5 * (dxy - dxy.mean()) / dxy.std() + _ar1(n, 0.99, 0.015, rng)

    # ---------- WTI：带均值回归(OU)的多因子对数价格过程 ----------
    # 中枢 78 美元；均值回归防止长样本随机游走发散到 0 或天价，
    # 各因子日波动量级经过平衡，避免单一因素在重要性学习中"机械碾压"。
    # GPR 日跳升 30 点 ≈ 利多油价 3%（系数 0.001），符合地缘风险溢价经验量级
    gpr_shock = np.diff(gpr, prepend=gpr[0]) * 0.001
    supply_shock = np.diff(supply_cut, prepend=supply_cut[0]) * 4
    factor_drift = (0.5 * demand + supply_shock + gpr_shock
                    - 3.0 * dxy_ret + rng.normal(0, 0.010, n))
    p_bar, kappa = np.log(78.0), 0.025     # OU 半衰期约 28 个交易日，拉住长期偏离
    log_p = np.zeros(n)
    log_p[0] = np.log(75.0)
    for t in range(1, n):
        log_p[t] = log_p[t - 1] + factor_drift[t] - kappa * (log_p[t - 1] - p_bar)
    wti = np.exp(log_p)
    # Brent 价差（时变 2.5~5 美元）
    spread = 3.4 + _ar1(n, 0.97, 0.08, rng)
    brent = wti + np.clip(spread, 2.0, 5.5)

    # --- 成品油（仅 demo 合成；strict 模式一律来自 CNBC 真实近月期货，不做推算）---
    # 美燃油 NYMEX ULSD：美元/加仑 ≈ 原油美元/桶 ÷42 × 裂解系数
    crack_ho = 2.05 + _ar1(n, 0.98, 0.01, rng)
    heating_oil = (wti / 42.0 * crack_ho) * (1 + rng.normal(0, 0.004, n))
    # 伦敦柴油 ICE Gasoil：美元/吨 ≈ 原油美元/桶 × 吨桶比7.45 × 裂解系数
    crack_go = 1.95 + _ar1(n, 0.98, 0.01, rng)
    gasoil = (brent * 7.45 * crack_go) * (1 + rng.normal(0, 0.004, n))
    # 上海原油 INE SC（人民币/桶）：≈ 美元油价 × USDCNY 汇率(带缓慢漂移)，叠加自身噪声
    usdcny = 7.1 + np.cumsum(rng.normal(0, 0.0006, n))
    shanghai_crude = brent * usdcny * (1 + rng.normal(0, 0.005, n))

    prices = pd.DataFrame(
        {"wti": wti, "brent": brent,
         "shanghai_crude": shanghai_crude,
         "heating_oil": heating_oil, "gasoil": gasoil},
        index=dates,
    )

    # ---------- 宏观序列 ----------
    # CPI 同比：慢变，受油价成本推动（滞后 20 日）
    oil_cost_push = pd.Series(wti).pct_change(60).fillna(0).to_numpy()
    cpi_yoy = 2.8 + 1.2 * oil_cost_push + _ar1(n, 0.995, 0.02, rng)
    cpi_yoy = np.clip(cpi_yoy, 1.0, 6.0)
    # 非农意外：月度（每月第一个工作日"公布"），其余日前向填充
    nfp = np.full(n, np.nan)
    month_first = pd.Series(dates).groupby([dates.year, dates.month]).head(1).index
    nfp[month_first] = rng.normal(0, 60, len(month_first))      # 单位：千人
    nonfarm_surprise = pd.Series(nfp, index=dates).ffill().bfill()
    # 降息/加息预期指数：CPI 越高越偏鹰（-1 强加息 … +1 强降息），惯性平滑避免日跳变
    fed_target = -0.8 * (cpi_yoy - cpi_yoy.mean()) / cpi_yoy.std()
    fed_noise = rng.normal(0, 0.01, n)
    fed_expectation = np.zeros(n)
    for t in range(1, n):
        fed_expectation[t] = 0.92 * fed_expectation[t - 1] + 0.08 * fed_target[t] + fed_noise[t]
    fed_expectation = np.clip(fed_expectation, -1, 1)
    demand_proxy = demand * 3   # 放大到可读量纲
    # 汇率夹具：USDCNY 与上面生成 SC 用的同一序列；USDJPY 慢变随机游走（140~170）
    usdjpy = 150.0 + np.cumsum(rng.normal(0, 0.05, n))

    macro = pd.DataFrame(
        {
            "dxy": dxy,
            "us10y": us10y,
            "cpi_yoy": cpi_yoy,
            "nonfarm_surprise": nonfarm_surprise.to_numpy(),
            "fed_expectation": fed_expectation,
            "demand_proxy": demand_proxy,
            "gpr_index": gpr,
            "usdcny": usdcny,
            "usdjpy": usdjpy,
        },
        index=dates,
    )

    # ---------- 事件表（最近 60 天，供报告展示） ----------
    daily_ret = np.diff(log_p, prepend=log_p[0])
    events = _synthesize_events(dates, gpr, daily_ret, wti, rng)
    # ---------- 机构观点（最近 90 天） ----------
    views = _synthesize_views(dates, wti, rng)

    LOG.info("已生成合成兜底数据：%d 个交易日，%d 条事件", n, len(events))
    return {"prices": prices, "macro": macro, "events": events, "views": views}


def _synthesize_events(dates, gpr, daily_ret, prices_wti, rng) -> pd.DataFrame:
    rows = []
    gpr_diff = np.abs(np.diff(gpr, prepend=gpr[0]))
    recent = np.where((np.arange(len(dates)) >= len(dates) - 75) & (gpr_diff > 8))[0]
    for i in recent:
        title, theme, sign = GEO_THEMES[rng.integers(0, len(GEO_THEMES))]
        intensity = float(np.clip(gpr_diff[i] / 45, 0.1, 1.0))
        # 经验弹性：用当日对数收益 × 当时价位估算美元/桶影响
        impact = float(np.clip(daily_ret[i] * prices_wti[i] * 2.2, -6, 6)) if sign > 0 else 0.0
        rows.append(dict(date=dates[i], title=title, source="synthetic-news",
                         theme=theme, sentiment="negative" if "袭击" in title or "制裁" in title else "neutral",
                         intensity=round(intensity, 3), est_price_impact=round(impact, 2)))
    # 宏观事件固定落在每月初
    macro_pub = pd.Series(dates).groupby([dates.year, dates.month]).head(1).index
    for i in macro_pub[-6:]:
        title, theme, sign = MACRO_THEMES[rng.integers(0, len(MACRO_THEMES))]
        rows.append(dict(date=dates[i], title=title, source="synthetic-calendar",
                         theme=theme, sentiment="neutral",
                         intensity=round(float(rng.uniform(0.2, 0.6)), 3),
                         est_price_impact=round(sign * float(rng.uniform(0.4, 1.8)), 2)))
    ev = pd.DataFrame(rows).sort_values("date", ascending=False).reset_index(drop=True)
    return ev


def _synthesize_views(dates, wti, rng) -> pd.DataFrame:
    banks = [
        ("高盛", 0.06), ("摩根大通", 0.03), ("瑞银", -0.02),
        ("IEA", 0.0), ("EIA", -0.01), ("OPEC", 0.05),
    ]
    rows = []
    pub_days = np.linspace(len(dates) - 90, len(dates) - 1, 12, dtype=int)
    for k, i in enumerate(pub_days):
        name, bias = banks[k % len(banks)]
        cur = wti[i]
        target = float(cur * (1 + bias + rng.uniform(-0.03, 0.05)))
        stance = "看涨" if target > cur * 1.01 else ("看跌" if target < cur * 0.99 else "中性")
        rows.append(dict(date=dates[i], institution=name,
                         target_wti=round(target, 1), stance=stance,
                         note=f"{name}调整WTI目标价至{target:.1f}美元/桶（合成）"))
    return pd.DataFrame(rows).sort_values("date", ascending=False).reset_index(drop=True)


def make_test_bundle(as_of, history_days=1095):
    """组装一个仅用于测试的 DataBundle（字段标注 SYNTHETIC-TEST，严禁当真实数据）。"""
    from oilcast.data_sources.base import DataBundle
    from oilcast.data_sources.collector import DISPLAY
    from oilcast.data_sources.quality import FieldLineage
    synth = build_synthetic_bundle(as_of, int(history_days))
    lineage = {f: FieldLineage(f, d, source_name="SYNTHETIC-TEST（测试夹具，非真实观测）",
                               url="", frequency="test", status="ok",
                               note="仅用于单元测试").to_dict() for f, d in DISPLAY.items()}
    return DataBundle(synth["prices"], synth["macro"], synth["events"], synth["views"],
                      lineage=lineage, mode="strict")
