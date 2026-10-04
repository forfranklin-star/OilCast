"""主要用油地区长假 / 出行旺季日历因子（真实日期、可追溯、可延伸）。

长假是【需求侧】的可预期事件窗口，方向因品种而异，故本模块只负责把"是否处于某长假窗口"
如实展开为 0/1 状态，**不预设涨跌**，由各品种模型自行学习其敏感度：

- 中国春节（农历，公历浮动）：节前备货/返乡（出行、汽油/航煤脉冲），节中工业与物流停工
  （柴油/工业用油骤降）——拆 pre / now 两个窗口，让模型分别学；
- 暑假 / 驾驶旺季（7–8 月）：中美欧学生假期叠加北美驾驶季，出行、汽油、航煤需求高峰；
- 中国国庆黄金周（10/1–10/7）：出行脉冲、工业停工；
- 年末圣诞新年（12/24–12/31）：出行与假期。

口径与 features.engineering 的历史 regime 一致：按自然日期区间生成，再映射到交易日 index
（非交易日天然不占位）；窗口只依赖公开日历事实，缺失不造假。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..utils import safe_log

# 农历正月初一对应的公历日期（公开农历可核对）。2016–2025 为确定值；2026 年起为农历推算，
# 可在官方放假安排发布后校正（仅影响未来窗口，不改变已实现的历史特征）。
CN_SPRING_FESTIVAL: Dict[int, str] = {
    2016: "2016-02-08",
    2017: "2017-01-28",
    2018: "2018-02-16",
    2019: "2019-02-05",
    2020: "2020-01-25",
    2021: "2021-02-12",
    2022: "2022-02-01",
    2023: "2023-01-22",
    2024: "2024-02-10",
    2025: "2025-01-29",
    2026: "2026-02-17",
    2027: "2027-02-06",
    2028: "2028-01-26",
    2029: "2029-02-13",
    2030: "2030-02-03",
}

# 输出列名（features.engineering 把它们归入 demand_outlook 因素组）
HOLIDAY_COLUMNS: List[str] = [
    "holiday_cn_spring_pre",   # 春节前一周（备货/返乡）
    "holiday_cn_spring_now",   # 除夕至正月初七（停工/返程）
    "holiday_summer_peak",     # 暑假 / 驾驶旺季 7/1–8/31
    "holiday_cn_national",     # 国庆黄金周 10/1–10/7
    "holiday_yearend",         # 圣诞新年 12/24–12/31
]


def _in_window(index: pd.DatetimeIndex,
               ranges: List[Tuple[str, str]]) -> pd.Series:
    """交易日 index 中，落在任一 [start,end] 自然日区间内取 1，否则 0。"""
    s = pd.Series(0.0, index=index)
    d = pd.DatetimeIndex(index).normalize()
    mask = np.zeros(len(d), dtype=bool)
    for a, b in ranges:
        mask |= (d >= pd.Timestamp(a)) & (d <= pd.Timestamp(b))
    s.loc[mask] = 1.0
    return s


def build_holiday_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """对交易日 index 展开主要用油地区长假窗口状态（0/1）。"""
    index = pd.DatetimeIndex(index)
    years = sorted(set(index.year))

    pre_ranges: List[Tuple[str, str]] = []
    now_ranges: List[Tuple[str, str]] = []
    for d in CN_SPRING_FESTIVAL.values():
        d0 = pd.Timestamp(d)
        pre_ranges.append((d0 - pd.Timedelta(days=7),
                           d0 - pd.Timedelta(days=2)))
        now_ranges.append((d0 - pd.Timedelta(days=1),
                           d0 + pd.Timedelta(days=7)))

    out: Dict[str, pd.Series] = {
        "holiday_cn_spring_pre": _in_window(index, pre_ranges),
        "holiday_cn_spring_now": _in_window(index, now_ranges),
        "holiday_summer_peak": _in_window(
            index, [(f"{y}-07-01", f"{y}-08-31") for y in years]),
        "holiday_cn_national": _in_window(
            index, [(f"{y}-10-01", f"{y}-10-07") for y in years]),
        "holiday_yearend": _in_window(
            index, [(f"{y}-12-24", f"{y}-12-31") for y in years]),
    }
    return pd.DataFrame(out, index=index)[HOLIDAY_COLUMNS]


def holiday_window_impact(prices: pd.DataFrame, instruments: List[str],
                          fwd_h: int = 5) -> List[Dict[str, object]]:
    """各长假窗口的【历史描述性】条件收益（真实、可追溯、带样本数）。

    对每个品种用其自身交易日展开假日窗口，统计"进入窗口的交易日持有 fwd_h 日"的实际对数
    收益均值与上涨概率。仅描述历史发生了什么，**不作为方向预测信号**（样本有限、且整体
    季节性检验不显著）；用于让长假因子在报告中可见、可核对。非交易日天然不参与。
    """
    rows: List[Dict[str, object]] = []
    for t in instruments:
        if t not in prices.columns:
            continue
        s = pd.to_numeric(prices[t], errors="coerce").dropna()
        if s.empty:
            continue
        idx = pd.DatetimeIndex(s.index)
        hdf = build_holiday_features(idx)
        fwd = safe_log(s).shift(-fwd_h) - safe_log(s)
        for c in HOLIDAY_COLUMNS:
            m = (hdf[c] == 1)
            n = int(m.sum())
            r = fwd[m].dropna()
            if not n:
                continue
            rows.append({
                "holiday": c, "instrument": t, "n": n,
                "mean_ret_pct": round(float(r.mean()) * 100, 2) if len(r) else None,
                "up_prob": round(float((r > 0).mean()), 2) if len(r) else None,
            })
    return rows
