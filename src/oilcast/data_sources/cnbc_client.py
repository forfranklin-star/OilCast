"""CNBC 行情历史 K 线客户端（world 可达、无需 API Key、交易所近月连续期货）。

为什么需要它：FRED 的 DCOILWTICO/DCOILBRENTEU 是 EIA【现货】，官方发布通常滞后
1~3 个工作日；而原油定价与本系统预测的对象对【近月连续期货】更敏感，且期货在
交易日当天即有收盘。CNBC 的 ts-api 直接给出交易所日 K（路径里的 adjusted 表示
连续合约已做换月处理），正常情况下末次观测就是最近一个交易日，比 FRED 现货更新鲜。

口径诚实：本源返回的是【近月连续期货收盘价（美元/桶，后复权连续）】，与 FRED/EIA
现货存在基差，源链最终只整条采用单一源、绝不把期货与现货拼成一条混合序列。
不可达 / 解析失败一律返回 None，交由源链尝试下一个源，绝不造数。
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional, Tuple
import pandas as pd
from ..utils import PoliteSession, get_logger
from .sources import BROWSER_HEADERS
LOG = get_logger(__name__)
# CNBC 标的符号：WTI 近月 @CL.1（NYMEX）、布伦特近月 @LCO.1（ICE）
CNBC_BARS_URL = ("https://ts-api.cnbc.com/harmony/app/bars/{symbol}/1D/"
                 "{start}000000/{end}235959/adjusted/EST5EDT.json")
def fetch_cnbc_bars(symbol: str, start: datetime, end: datetime,
                    sess: Optional[PoliteSession] = None
                    ) -> Optional[Tuple[pd.Series, dict]]:
    """取日收盘序列。返回 (Series(按日归一化、升序、去空), meta)；失败返回 None。"""
    sess = sess or PoliteSession(extra_headers=BROWSER_HEADERS)
    url = CNBC_BARS_URL.format(symbol=symbol,
                               start=pd.Timestamp(start).strftime("%Y%m%d"),
                               end=pd.Timestamp(end).strftime("%Y%m%d"))
    resp = sess.get(url)
    if resp is None:
        return None
    try:
        j = resp.json()
        bars = j["barData"]["priceBars"]
        if not bars:
            return None
        idx = pd.to_datetime([b["tradeTimeinMills"] for b in bars], unit="ms").normalize()
        close = pd.to_numeric(pd.Series([b["close"] for b in bars]), errors="coerce")
        s = pd.Series(close.to_numpy(), index=idx).dropna().sort_index()
        s = s[~s.index.duplicated(keep="last")]
        s.name = symbol
        if s.empty:
            return None
        meta = {"last_observed": s.index.max().strftime("%Y-%m-%d"),
                "caliber": "CNBC近月连续期货(交易所收盘,换月调整)",
                "url": url}
        return s, meta
    except Exception as exc:
        LOG.warning("CNBC bars %s 解析失败：%s", symbol, exc)
        return None
