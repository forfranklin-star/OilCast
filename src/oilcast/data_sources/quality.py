"""数据谱系（lineage）与质量门（quality gate）。

铁律：模型只消费通过质量门的真实观测。每个字段都必须能回答四个问题：
    1) 来源是谁（source_name / url，可回溯核验）；
    2) 值是哪一天观测/发布的（last_observed，区别于抓取时刻）；
    3) 什么时候抓的（retrieved_at）；
    4) 是否新鲜、样本是否足够（status）。

状态：
    ok            通过质量门，可进入模型
    stale         最后观测过旧（超过容忍窗口）——展示但禁止用于训练/预测
    insufficient  真实样本数不足
    unavailable   源不可达 / 不存在可核验观测 —— 保持缺失，严禁补齐
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from .calendar import normalize_as_of_to_trading

from ..config import get_config
from ..utils import now_beijing


@dataclass
class FieldLineage:
    field: str
    display: str
    source_name: str = ""
    url: str = ""
    frequency: str = ""              # daily_business / monthly / event
    retrieved_at: str = ""
    n_obs: int = 0
    first_observed: Optional[str] = None
    last_observed: Optional[str] = None
    stale: Optional[int] = None      # 滞后：日频=工作日数，月频=自然日数
    status: str = "unavailable"      # ok / stale / insufficient / unavailable
    note: str = ""
    tried_sources: str = ""          # 多源优先级链完整尝试记录（含失败原因），可审计

    def to_dict(self) -> dict:
        return asdict(self)


def business_days_between(a: pd.Timestamp, b: pd.Timestamp) -> int:
    return int(len(pd.bdate_range(a, b)) - 1)


def assess_daily(field: str, display: str, series: Optional[pd.Series], as_of: pd.Timestamp,
                 source_name: str, url: str, gate: dict,
                 retrieved_at: Optional[str] = None, note: str = "",
                 tried_sources: str = "") -> FieldLineage:
    """评估日频（工作日）字段。series 只允许包含真实观测，不得预先插值。"""
    lin = FieldLineage(field=field, display=display, source_name=source_name, url=url,
                       frequency="daily_business",
                       retrieved_at=retrieved_at or now_beijing().strftime("%Y-%m-%d %H:%M:%S"),
                       note=note, tried_sources=tried_sources)
    if series is None:
        lin.status = "unavailable"
        lin.note = (lin.note + "；源未返回数据").strip("；")
        return lin
    s = series.dropna()
    lin.n_obs = int(len(s))
    if len(s) == 0:
        lin.status = "unavailable"
        return lin
    lin.first_observed = s.index.min().strftime("%Y-%m-%d")
    lin.last_observed = s.index.max().strftime("%Y-%m-%d")
    # 周末/非交易日运行时以最近工作日为基准，休市时间不计为滞后
    _ref = normalize_as_of_to_trading(as_of)
    lin.stale = business_days_between(s.index.max(), _ref)

    if lin.n_obs < int(gate["min_obs"]):
        lin.status = "insufficient"
        lin.note += f"；真实观测仅{lin.n_obs}条，少于{gate['min_obs']}条门槛"
    elif lin.stale > int(gate["max_stale_bdays"]):
        lin.status = "stale"
        lin.note += f"；最后观测滞后{lin.stale}个工作日，超过{gate['max_stale_bdays']}天"
    else:
        lin.status = "ok"
    return lin


def assess_monthly(field: str, display: str, series: Optional[pd.Series], as_of: pd.Timestamp,
                   source_name: str, url: str, gate: dict,
                   retrieved_at: Optional[str] = None, tried_sources: str = "") -> FieldLineage:
    """评估月频字段（CPI/非农/联邦基金利率）。发布滞后按自然日计。"""
    lin = FieldLineage(field=field, display=display, source_name=source_name, url=url,
                       frequency="monthly",
                       retrieved_at=retrieved_at or now_beijing().strftime("%Y-%m-%d %H:%M:%S"),
                       tried_sources=tried_sources)
    if series is None:
        lin.status = "unavailable"
        return lin
    s = series.dropna()
    lin.n_obs = int(len(s))
    if len(s) == 0:
        lin.status = "unavailable"
        return lin
    lin.first_observed = s.index.min().strftime("%Y-%m-%d")
    lin.last_observed = s.index.max().strftime("%Y-%m-%d")
    lin.stale = int((pd.Timestamp(as_of).normalize() - s.index.max()).days)
    lin.status = "ok" if lin.stale <= int(gate["max_stale_days"]) else "stale"
    return lin


def align_monthly_with_vintage(monthly: pd.Series, bday_index: pd.DatetimeIndex,
                               ffill_limit: int = 25) -> tuple[pd.Series, pd.Series]:
    """把月频真实发布值对齐到工作日索引。

    这不是"造数"：两次发布之间沿用的是【最近一次真实发布值】，同时返回
    vintage 序列，逐点记录该值的实际发布日期，报告必须展示该观测日期。
    """
    aligned = monthly.reindex(bday_index, method="ffill", limit=ffill_limit)
    # 每个对齐值对应的原始观测（发布）日期
    vint = pd.Series(monthly.index[monthly.index.searchsorted(aligned.index, side="right") - 1],
                     index=aligned.index)
    valid = aligned.notna()
    vint = vint.where(valid)
    vint = vint.dt.strftime("%Y-%m-%d") if hasattr(vint.dt, "strftime") else vint
    return aligned, vint


def limited_ffill_daily(series: pd.Series, limit: int) -> pd.Series:
    """日频字段只允许填补节假日级短缺口（默认≤3 工作日），长缺口保持 NaN（图上断开）。"""
    return series.reindex(sorted(series.index.unique())).ffill(limit=limit)


def lineage_table(lineages: dict) -> pd.DataFrame:
    rows = [L.to_dict() for L in lineages.values()]
    return pd.DataFrame(rows)
