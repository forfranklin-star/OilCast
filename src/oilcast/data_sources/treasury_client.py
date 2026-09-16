"""美国财政部官方国债收益率曲线 CSV 客户端（无 key，全球可达）。
是 FRED DGS10/DGS2 的【官方源头备份】：FRED 利率序列本就转自财政部。
URL 按年份返回该年每个交易日的全部期限（"2 Yr"/"10 Yr" 等）。
不可达返回 None，绝不造数。
"""
from __future__ import annotations
import io
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
from ..config import get_config
from ..utils import PoliteSession, get_logger
from .sources import BROWSER_HEADERS

LOG = get_logger(__name__)


def fetch_treasury_curve(tenor_col: str, start: datetime, end: datetime,
                         sess: Optional[PoliteSession] = None) -> Optional[pd.Series]:
    """拉取指定期限（如 '10 Yr' / '2 Yr'）在 [start,end] 的日频收益率。"""
    sess = sess or PoliteSession(extra_headers=BROWSER_HEADERS)
    tmpl = str(get_config()["data_sources"]["treasury_yield_csv"])
    years = list(range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1))
    frames = []
    for yr in years:
        resp = sess.get(tmpl.format(year=yr))
        if resp is None:
            continue
        try:
            df = pd.read_csv(io.StringIO(resp.text))
            if "Date" not in df.columns or tenor_col not in df.columns:
                continue
            frames.append(df[["Date", tenor_col]])
        except Exception as exc:
            LOG.warning("Treasury %s 年解析失败：%s", yr, exc)
    if not frames:
        return None
    raw = pd.concat(frames, ignore_index=True)
    raw["Date"] = pd.to_datetime(raw["Date"], format="%m/%d/%Y", errors="coerce")
    s = pd.to_numeric(raw[tenor_col], errors="coerce")
    out = pd.Series(s.values, index=raw["Date"]).dropna().sort_index()
    out = out[(out.index >= pd.Timestamp(start)) & (out.index <= pd.Timestamp(end))]
    out.name = tenor_col
    return out if len(out) > 0 else None
