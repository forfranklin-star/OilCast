"""中国期货市场真实日 K 客户端（上海原油 INE SC 主力连续等）。

为什么需要它：上海国际能源交易中心（INE）的原油期货以人民币计价、反映亚太/中东
原油到岸供需，是与 WTI/Brent 相互独立的价格信号，且其可交割品以迪拜/阿曼等中东
中质原油为锚。海外可访问的免费日 K 源：
  * 主源 新浪财经 InnerFuturesNewService.getDailyKLine（本项目运行环境实测可达，
    返回该品种上市以来全部日 K，本地按 start/end 过滤，主力连续 symbol=SC0）；
  * 备源 东方财富 push2his 日 K（secid=113.sc0，113 为上期所/能源中心板块）。
口径诚实：返回的是【主力连续合约日 K 收盘价（人民币/桶，即日盘 15:00 收盘）】，与美元
计价的 WTI/Brent 不直接做水平价差，只在特征层做无量纲的相对强弱比较；任何源不可达/解析
失败返回 None，交由源链尝试下一个源，绝不造数。
注意：新浪日 K 某交易日标签对应【日盘 15:00】收盘（周五晚~周六凌晨 02:30 的夜盘按交易所
归属下一交易日、不并入当日日 K）。系统统计主节点为上海【夜盘 02:30 收盘】，该值由
intraday_client 的 15 分 K 提供、在 pipeline 中覆盖日频最新值（原始日 K 仍落库保留），
因此报告最新上海价是夜盘 02:30 收盘而非此处日盘 15:00 收盘，两种口径分别标注、不混用。
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional, Tuple
import pandas as pd
from ..utils import PoliteSession, get_logger
from .sources import BROWSER_HEADERS

LOG = get_logger(__name__)

SINA_KLINE_URL = ("https://stock2.finance.sina.com.cn/futures/api/json.php/"
                  "InnerFuturesNewService.getDailyKLine?symbol={symbol}")
# 113 = 上海期货交易所/上海国际能源交易中心板块；fields2: f51日期 f52开 f53收
EM_KLINE_URL = ("https://push2his.eastmoney.com/api/qt/stock/kline/get?"
                "secid={secid}&fields1=f1,f2,f3&fields2=f51,f52,f53,f54,f55,f56,f57"
                "&klt=101&fqt=0&beg={beg}&end={end}&lmt=100000")


def _finalize(idx, close, symbol, caliber, url) -> Optional[Tuple[pd.Series, dict]]:
    vals = pd.to_numeric(pd.Series(list(close)), errors="coerce").to_numpy()
    s = pd.Series(vals,
                  index=pd.DatetimeIndex(pd.to_datetime(idx)).normalize())
    s = s.dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s.name = symbol
    if s.empty:
        return None
    return s, {"last_observed": s.index.max().strftime("%Y-%m-%d"),
               "caliber": caliber, "url": url}


def fetch_sina_inner(symbol: str, start: datetime, end: datetime,
                     sess: Optional[PoliteSession] = None
                     ) -> Optional[Tuple[pd.Series, dict]]:
    """新浪国内期货主力连续日 K（人民币计价）。"""
    sess = sess or PoliteSession(extra_headers=BROWSER_HEADERS)
    url = SINA_KLINE_URL.format(symbol=symbol)
    resp = sess.get(url)
    if resp is None:
        return None
    try:
        arr = resp.json()
        if not arr:
            return None
        df = pd.DataFrame(arr)
        caliber = "新浪国内期货主力连续(人民币计价,日收盘)"
        out = _finalize(df["d"], df["c"], symbol, caliber, url)
        if out is None:
            return None
        s = out[0]
        s = s.loc[(s.index >= pd.Timestamp(start).normalize()) &
                  (s.index <= pd.Timestamp(end).normalize() + pd.Timedelta(days=1))]
        return (s, out[1]) if not s.empty else None
    except Exception as exc:
        LOG.warning("新浪国内期货 %s 解析失败：%s", symbol, exc)
        return None


def fetch_eastmoney_inner(secid: str, start: datetime, end: datetime,
                          sess: Optional[PoliteSession] = None
                          ) -> Optional[Tuple[pd.Series, dict]]:
    """东方财富国内期货日 K（备源，secid 形如 113.sc0，人民币计价）。"""
    sess = sess or PoliteSession(extra_headers={**BROWSER_HEADERS,
                                                "Referer": "https://quote.eastmoney.com/"})
    url = EM_KLINE_URL.format(
        secid=secid,
        beg=pd.Timestamp(start).strftime("%Y%m%d"),
        end=pd.Timestamp(end).strftime("%Y%m%d"))
    resp = sess.get(url)
    if resp is None:
        return None
    try:
        data = resp.json().get("data")
        if not data or not data.get("klines"):
            return None
        rows = [k.split(",") for k in data["klines"]]
        dates = [r[0] for r in rows]
        closes = [r[2] for r in rows]   # f53=收盘
        caliber = "东方财富国内期货主力连续(人民币计价,日收盘)"
        return _finalize(dates, closes, secid, caliber, url)
    except Exception as exc:
        LOG.warning("东财国内期货 %s 解析失败：%s", secid, exc)
        return None
