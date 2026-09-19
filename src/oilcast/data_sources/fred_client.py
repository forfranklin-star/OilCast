"""FRED（圣路易斯联储）公开 CSV 客户端：无需 API key。

每个序列都可在 https://fred.stlouisfed.org/series/<ID> 核验口径、观测日期与
原始发布机构（DCOILWTICO/DCOILBRENTEU 的源头是 EIA）。
"""
from __future__ import annotations

import io
from datetime import datetime
from typing import Optional

import pandas as pd

from ..config import get_config
from ..utils import PoliteSession, get_logger

LOG = get_logger(__name__)


def fred_url(series_id: str) -> str:
    return f"https://fred.stlouisfed.org/series/{series_id}"


def fetch_fred(series_id: str, start: datetime, end: datetime,
               sess: Optional[PoliteSession] = None) -> Optional[pd.Series]:
    """抓取单个 FRED 序列；缺失值（"."）保留为 NaN，绝不填充。返回 None 表示不可达。"""
    cfg = get_config()
    base = str(cfg["data_sources"]["fred_base_csv"])
    own = sess is None
    sess = sess or PoliteSession()
    resp = sess.get(base, params={"id": series_id,
                                  "cosd": pd.Timestamp(start).strftime("%Y-%m-%d"),
                                  "coed": pd.Timestamp(end).strftime("%Y-%m-%d")})
    if own:
        pass
    if resp is None:
        return None
    try:
        df = pd.read_csv(io.StringIO(resp.text), index_col=0, parse_dates=True)
        s = df.iloc[:, 0].replace(".", pd.NA).astype("float64").dropna()
        s.name = series_id
        s.index = s.index.normalize()
        return s.sort_index()
    except Exception as exc:
        LOG.warning("FRED %s 解析失败：%s", series_id, exc)
        return None
