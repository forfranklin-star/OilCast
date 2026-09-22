"""交易日历（trading-day calendar）。

最高原则：**时间轴以真实交易日为准，非交易时间（周末、交易所假日）不占位、
不计入步长，更不算"数据缺口"。** 只有"本来应当交易、却缺真实观测"的交易日
才是缺口。

- 历史时间轴：直接取真实价格/宏观中出现过的日期（observed trading days），
  不预生成"周一到周五"的硬网格去 reindex——硬网格会把公共假日也排成交易日，
  凭空造出 NaN 假缺口。
- 未来时间轴：沿真实交易日向外推，周六周日绝不计入；公共假日借助"历史上同一
  月-日从未交易"识别并跳过。离线无法预知的临时休市，等真实数据到来后自然校正，
  绝不预填。
"""
from __future__ import annotations

from typing import Iterable, Optional, Set

import pandas as pd


def _to_index(obj) -> pd.DatetimeIndex:
    if obj is None:
        return pd.DatetimeIndex([])
    idx = pd.DatetimeIndex(pd.to_datetime(obj.index)).normalize()
    if isinstance(obj, pd.DataFrame):
        obj = obj.dropna(how="all")
        idx = pd.DatetimeIndex(pd.to_datetime(obj.index)).normalize()
    elif isinstance(obj, pd.Series):
        idx = pd.DatetimeIndex(pd.to_datetime(obj.dropna().index)).normalize()
    return idx


def observed_trading_index(*objs) -> pd.DatetimeIndex:
    """合并一个或多个真实序列/表的观测日，得到历史真实交易日轴（升序、去重）。

    非交易日从不出现在结果里，因此后续 reindex 不会在周末/假日制造 NaN。
    """
    days: Set[pd.Timestamp] = set()
    for obj in objs:
        for d in _to_index(obj):
            days.add(pd.Timestamp(d))
    return pd.DatetimeIndex(sorted(days))


def trade_weekdays(hist_index: Optional[pd.DatetimeIndex]) -> Set[int]:
    """历史上发生过交易的星期几（0=周一）。原油近月期货为周一到周五。"""
    if hist_index is None or len(hist_index) == 0:
        return {0, 1, 2, 3, 4}
    idx = pd.DatetimeIndex(hist_index)
    cnt = pd.Series(1, index=idx).groupby(idx.weekday).size()
    weekdays = {int(w) for w in cnt[cnt >= 1].index}
    return weekdays or {0, 1, 2, 3, 4}


def closed_holiday_mmdd(hist_index: Optional[pd.DatetimeIndex],
                        min_span_days: int = 300) -> Set[str]:
    """识别交易所假日：历史跨度足够长时，属于常规交易星期几、但同一 mm-dd
    从未出现过交易，判定为该市场假日（如元旦、独立日、感恩节）。

    历史跨度不足 min_span_days 时不做假日推断（避免样本太短误杀正常交易日）。
    返回 {"01-01", "12-25", ...}。
    """
    if hist_index is None or len(hist_index) == 0:
        return set()
    idx = pd.DatetimeIndex(sorted(pd.DatetimeIndex(hist_index).normalize().unique()))
    if (idx.max() - idx.min()).days < min_span_days:
        return set()
    twd = trade_weekdays(idx)
    observed = set(idx.strftime("%m-%d"))
    closed: Set[str] = set()
    for d in pd.bdate_range(idx.min(), idx.max()):
        if int(d.weekday()) in twd and d.strftime("%m-%d") not in observed:
            closed.add(d.strftime("%m-%d"))
    return closed


def next_trading_days(last_date, periods: int,
                      hist_index: Optional[pd.DatetimeIndex] = None,
                      closed_mmdd: Optional[Set[str]] = None) -> pd.DatetimeIndex:
    """从 last_date 的次日起，生成 periods 个【未来交易日】。

    - 周六、周日绝不计入（非交易时间）；
    - closed_mmdd 中的已知假日跳过（可由 closed_holiday_mmdd 预算）；
    - 离线不可预知的临时休市无法提前排除，待真实数据到来后校正，不预填价格。
    """
    last = pd.Timestamp(last_date).normalize()
    twd = trade_weekdays(hist_index)
    closed = closed_mmdd if closed_mmdd is not None else closed_holiday_mmdd(hist_index)
    out = []
    d = last + pd.Timedelta(days=1)
    guard = 0
    max_guard = periods * 4 + 40        # 周末+假日最多让日历多扫若干天，设保护性上限
    while len(out) < periods and guard < max_guard:
        guard += 1
        if int(d.weekday()) in twd and d.strftime("%m-%d") not in closed:
            out.append(d)
        d += pd.Timedelta(days=1)
    return pd.DatetimeIndex(out)


def normalize_as_of_to_trading(as_of) -> pd.Timestamp:
    """把基准时间归一到不晚于它的最近工作日：周末运行报告时，滞后判定以周五为
    基准，避免把"周末本来就不开市"误判成数据滞后。"""
    d = pd.Timestamp(as_of).normalize()
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    return d
