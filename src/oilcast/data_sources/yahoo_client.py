"""Yahoo Finance 原生 chart API 客户端（不依赖 yfinance 第三方库）。
海外网络（含 GitHub Actions 美国机房）通常可直接访问，无需 key。
作为 FRED 主源失效时的价格/指数备份源；不可达返回 None，绝不造数。
"""
from __future__ import annotations
import time
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import quote
import pandas as pd
from ..config import get_config
from ..utils import PoliteSession, get_logger
from .sources import BROWSER_HEADERS

LOG = get_logger(__name__)


def fetch_yahoo_chart(symbol: str, start: datetime, end: datetime,
                      sess: Optional[PoliteSession] = None) -> Optional[pd.Series]:
    """取日线收盘价序列。期货 CL=F/BZ=F/HO=F、指数 DX-Y.NYB/^TNX 均适用。"""
    sess = sess or PoliteSession(extra_headers=BROWSER_HEADERS)
    p1 = int(time.mktime(pd.Timestamp(start).timetuple()))
    p2 = int(time.mktime((pd.Timestamp(end) + timedelta(days=1)).timetuple()))
    tmpl = str(get_config()["data_sources"]["yahoo_chart"])
    url = tmpl.format(symbol=quote(symbol))
    resp = sess.get(url, params={"period1": p1, "period2": p2, "interval": "1d",
                                 "events": "history"})
    if resp is None:
        return None
    try:
        j = resp.json()
        result = j["chart"]["result"][0]
        ts = result["timestamp"]
        close = result["indicators"]["quote"][0]["close"]
        s = pd.Series(close, index=pd.to_datetime(ts, unit="s").normalize())
        s = s.dropna().sort_index()
        s.name = symbol
        return s if len(s) > 0 else None
    except Exception as exc:
        LOG.warning("Yahoo chart %s 解析失败：%s", symbol, exc)
        return None
