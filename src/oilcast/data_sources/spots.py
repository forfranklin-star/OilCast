"""现货价格聚合：WTI / 布伦特（日度，FRED/EIA）+ 迪拜（日度，仅本地 OilPriceAPI）。

主价格序列（wti/brent）口径是近月期货；本模块【额外】采集现货，用于现货走势与
"期货-现货价差"，不与主序列混合。
- 本地版：OilPriceAPI 提供 WTI/Brent/Dubai（含迪拜），FRED 补齐 WTI/Brent；
- Streamlit 版：不含 oilprice_client，只用 FRED DCOILWTICO/DCOILBRENTEU（无需 key）。
任何源不可达/缺日一律保留 NaN，绝不造数。
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional, Tuple

import pandas as pd

from ..config import get_config
from ..utils import get_logger
from .fred_client import fetch_fred

LOG = get_logger(__name__)

SPOT_COLUMNS = ["wti_spot", "brent_spot", "dubai_spot"]


def fetch_spots(as_of: datetime, lookback_days: Optional[int] = None
                ) -> Tuple[pd.DataFrame, dict]:
    """返回 (现货表[wti_spot,brent_spot,dubai_spot], 来源元信息)；缺失保留 NaN。"""
    cfg = get_config()
    lb = int(lookback_days or cfg["data_sources"]["spot_lookback_days"])
    start, end = pd.Timestamp(as_of) - pd.Timedelta(days=lb), pd.Timestamp(as_of)
    meta = {}

    # 1) OilPriceAPI（仅本地版含该模块/key；唯一提供日度迪拜）
    op = None
    try:
        from .oilprice_client import fetch_oilprice_spots
        op = fetch_oilprice_spots(start, end)
        if op is not None and len(op):
            if op.index.tz is not None:        # 统一为 naive UTC，与 FRED/交易日轴一致
                op.index = op.index.tz_convert("UTC").tz_localize(None)
            meta["oilpriceapi"] = {"n": int(len(op)),
                                   "last": op.index.max().strftime("%Y-%m-%d")}
    except Exception as exc:  # noqa: BLE001 - Streamlit 无此模块/异常时降级 FRED
        LOG.info("OilPriceAPI 不可用，现货改用 FRED：%s", exc)

    # 2) FRED 日度现货（WTI/Brent；Streamlit 主源，本地用于补齐）
    fred = pd.DataFrame()
    try:
        w = fetch_fred("DCOILWTICO", start, end)
        b = fetch_fred("DCOILBRENTEU", start, end)
        fred = pd.DataFrame({"wti_spot": w, "brent_spot": b})
        if len(fred):
            meta["fred"] = {
                "n": int(fred.notna().any(axis=1).sum()),
                "last": fred.dropna(how="all").index.max().strftime("%Y-%m-%d")}
    except Exception as exc:  # noqa: BLE001
        LOG.warning("FRED 现货失败：%s", exc)

    # 合并：OilPrice 优先（含迪拜）；其缺失点用 FRED 补（combine_first 不覆盖已有值）
    spots = pd.DataFrame()
    if op is not None and len(op):
        spots = op.copy()
    if len(fred):
        spots = fred.copy() if not len(spots) else \
            spots.combine_first(fred[["wti_spot", "brent_spot"]])

    for c in SPOT_COLUMNS:                      # 缺列补 NA（Streamlit 无迪拜）
        if c not in spots.columns:
            spots[c] = pd.NA
    spots = spots[SPOT_COLUMNS].sort_index()
    spots.index = spots.index.normalize()
    spots.index.name = "date"
    LOG.info("现货聚合：%d 个交易日；迪拜有效点 %d",
             len(spots), int(spots["dubai_spot"].notna().sum()))
    return spots, meta


# 期货主列 -> (对应现货列, 价差列)；迪拜无期货、不在此映射
_BASIS_MAP = {"wti": ("wti_spot", "wti_basis"),
              "brent": ("brent_spot", "brent_basis")}


def attach_basis(spots: pd.DataFrame, futures: pd.DataFrame) -> pd.DataFrame:
    """按【同一交易日】对齐近月期货与现货，basis=期货-现货。

    只在该日两者都有真实值时计算（dropna），任一缺失则该日 basis 留空、绝不造数；
    非交易日不占位。期货与现货均为 naive 交易日索引。
    """
    out = spots.copy()
    for bc in ("wti_basis", "brent_basis"):
        out[bc] = pd.NA
    for f, (sc, bc) in _BASIS_MAP.items():
        if f in futures.columns and sc in out.columns:
            j = pd.DataFrame({"fut": futures[f], "spot": out[sc]}).dropna()
            if len(j):
                out.loc[j.index, bc] = (j["fut"] - j["spot"]).round(3)
    return out
