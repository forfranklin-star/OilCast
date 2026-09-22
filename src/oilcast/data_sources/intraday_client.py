"""盘中（分时）K 线采集：用于跨市场"同一真实时刻"对齐。

背景：日 K 的日期标签不等于同一时刻。上海 INE 原油除日盘(09:00-11:30/13:30-15:00)外还有
夜盘(北京 21:00-次日 02:30)，该窗口与 WTI/Brent 欧美电子盘真正重叠、同一时刻都有成交价。
仅用日 K 收盘做对齐，会把"与海外同步的夜盘"和"次日日盘"捆在一根里，升贴水/传导口径粗糙。
本模块采集各品种【带北京时间戳的分时 K】，供特征层在重叠窗口取严格同时刻价格。

数据源（均海外可达、无需 key、已实测）：
  * 上海原油：新浪 InnerFuturesNewService.getFewMinLine(symbol=SC0, type=15)，15 分钟，
    时间戳本身即北京时间，服务端保留最近约 5 个月。【为何不用 60 分 K】新浪 60 分 K 会把
    夜盘最后一根（02:30 收盘）错标成当日 09:30（周六凌晨收盘被错标成周六 09:30，而上期所
    周六根本不开盘），曾导致真实夜盘收盘价被截断逻辑误删；15 分 K 时间戳正确（02:30 整点）。
    另用上期所 SC 合法交易时段白名单 _ine_sc_session_mask 剔除行情商错标的幽灵 K（如周一
    00:00、周六 09:30 等非交易时刻）。
  * WTI/Brent/美燃油/伦敦柴油：CNBC ts-api 周期 1H（大写），UTC 毫秒戳，转北京时间，
    服务端保留最近约 3.3 个月、近 24 小时连续电子盘。

统计节点：上海原油一个连续交易时段的真正收盘是【夜盘 02:30】（周五夜盘物理发生在周六凌晨
02:30，经 apply_session_rollover 回退归属到周五交易日），特征层以 02:30 为统计主节点，在该
真实时刻对齐 WTI/Brent（北京 02:30 = 美东 14:30 / 伦敦 19:30，欧美电子盘仍在交易）。

诚实口径与积累机制：免费源分时历史有限（拿不到更早的，绝不伪造回填）；本系统每日定时增量
拉取并 upsert 入库(raw_intraday)，同时刻面板自部署之日起随时间持续变长，配合模型热启动，
越用样本越多。任一源失败只跳过该品种、返回已取到的真实数据，绝不造数。
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from ..utils import PoliteSession, get_logger
from .sources import BROWSER_HEADERS

LOG = get_logger(__name__)

# 用 15 分 K：60 分 K 会把夜盘 02:30 收盘那根错标成 09:30（时间戳 bug），15 分 K 正确。
SINA_INTRADAY_URL = ("https://stock2.finance.sina.com.cn/futures/api/json.php/"
                     "InnerFuturesNewService.getFewMinLine?symbol={symbol}&type=15")
CNBC_1H_URL = ("https://ts-api.cnbc.com/harmony/app/bars/{symbol}/1H/"
               "{start}000000/{end}235959/adjusted/EST5EDT.json")

# 分时品种 -> CNBC 代码（上海原油走新浪，不在此表）
CNBC_1H_SYMBOLS: Dict[str, str] = {
    "wti": "@CL.1", "brent": "@LCO.1",
    "heating_oil": "@HO.1", "gasoil": "@GAS.1",
}
SINA_INTRADAY = {"shanghai_crude": "SC0"}

_COLS = ["symbol", "ts", "open", "high", "low", "close", "volume", "source"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=_COLS)


def ine_sc_session_mask(ts: pd.Series) -> pd.Series:
    """上期所原油 SC 合法交易时段白名单（时间戳为北京时间、K 线结束时刻）。

    日盘 09:00-11:30 / 13:30-15:00（周一~周五）；夜盘 21:00-23:59（周一~周五晚）与
    00:00-02:30（周二~周六凌晨，承接前一晚夜盘）。用于剔除行情商错标的幽灵 K：
    例如把周六凌晨 02:30 收盘错标成周六 09:30、或凭空出现周一 00:00（周日晚无夜盘）。
    weekday(): 周一=0 … 周日=6。
    """
    ts = pd.to_datetime(ts)
    wd = ts.dt.weekday
    hm = ts.dt.hour * 60 + ts.dt.minute
    day = (wd <= 4) & (hm.between(9 * 60, 11 * 60 + 30)
                       | hm.between(13 * 60 + 30, 15 * 60))
    evening = (wd <= 4) & hm.between(21 * 60, 24 * 60 - 1)      # 周一~周五晚
    dawn = wd.between(1, 5) & hm.between(0, 2 * 60 + 30)        # 周二~周六凌晨
    return day | evening | dawn


def fetch_sina_intraday(symbol: str = "shanghai_crude", code: str = "SC0",
                        sess: Optional[PoliteSession] = None) -> pd.DataFrame:
    """新浪国内期货 15 分钟 K（时间戳即北京时间），并按上期所交易时段过滤幽灵 K。"""
    sess = sess or PoliteSession(extra_headers=BROWSER_HEADERS)
    url = SINA_INTRADAY_URL.format(symbol=code)
    resp = sess.get(url)
    if resp is None:
        return _empty()
    try:
        arr = resp.json()
        if not arr:
            return _empty()
        df = pd.DataFrame(arr)
        out = pd.DataFrame({
            "symbol": symbol,
            "ts": pd.to_datetime(df["d"]),                 # 已是北京时
            "open": pd.to_numeric(df["o"], errors="coerce"),
            "high": pd.to_numeric(df["h"], errors="coerce"),
            "low": pd.to_numeric(df["l"], errors="coerce"),
            "close": pd.to_numeric(df["c"], errors="coerce"),
            "volume": pd.to_numeric(df.get("v"), errors="coerce"),
            "source": "新浪:INE SC0 15min",
        })
        out = out.dropna(subset=["ts", "close"]).sort_values("ts")
        before = len(out)
        out = out[ine_sc_session_mask(out["ts"])].reset_index(drop=True)
        dropped = before - len(out)
        if dropped:
            LOG.info("新浪 SC 分时按合法交易时段剔除 %d 根错标/幽灵 K", dropped)
        return out
    except Exception as exc:
        LOG.warning("新浪分时 %s 解析失败：%s", code, exc)
        return _empty()


# 向后兼容旧名
fetch_sina_60m = fetch_sina_intraday


def fetch_cnbc_1h(symbol: str, code: str,
                  start: datetime, end: datetime,
                  sess: Optional[PoliteSession] = None) -> pd.DataFrame:
    """CNBC 近月连续期货 1 小时 K，UTC 毫秒戳转北京时间（naive，Asia/Shanghai）。"""
    sess = sess or PoliteSession(extra_headers=BROWSER_HEADERS)
    url = CNBC_1H_URL.format(symbol=code,
                             start=pd.Timestamp(start).strftime("%Y%m%d"),
                             end=pd.Timestamp(end).strftime("%Y%m%d"))
    resp = sess.get(url)
    if resp is None:
        return _empty()
    try:
        bars = resp.json()["barData"]["priceBars"]
        if not bars:
            return _empty()
        ts = pd.to_datetime([b["tradeTimeinMills"] for b in bars], unit="ms", utc=True) \
            .tz_convert("Asia/Shanghai").tz_localize(None)
        out = pd.DataFrame({
            "symbol": symbol,
            "ts": ts,
            "open": pd.to_numeric([b.get("open") for b in bars], errors="coerce"),
            "high": pd.to_numeric([b.get("high") for b in bars], errors="coerce"),
            "low": pd.to_numeric([b.get("low") for b in bars], errors="coerce"),
            "close": pd.to_numeric([b.get("close") for b in bars], errors="coerce"),
            "volume": pd.to_numeric([b.get("volume") for b in bars], errors="coerce"),
            "source": f"CNBC:{code} 1H",
        })
        return out.dropna(subset=["ts", "close"]).sort_values("ts")
    except Exception as exc:
        LOG.warning("CNBC分时 %s 解析失败：%s", code, exc)
        return _empty()


def apply_session_rollover(df: pd.DataFrame, as_of) -> pd.DataFrame:
    """交易日归属 + as_of 截断（纯函数，便于测试）。

    本系统以"日盘日期"锚定一个交易日：北京时凌晨 00:00~07:59 的 K 线是前一自然日晚
    开盘的夜盘（上海夜盘到次日 02:30、欧美电子盘到北京时凌晨），把其日期回退一天，使
    "周五 15:00 日盘 + 周六凌晨夜盘"归入同一交易日；归属后仍晚于 as_of 的越界前瞻点剔除。
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts"])
    overnight = out["ts"].dt.hour < 8
    out.loc[overnight, "ts"] = out.loc[overnight, "ts"] - pd.Timedelta(days=1)
    as_of = pd.Timestamp(as_of).normalize()
    out = out[out["ts"].dt.normalize() <= as_of]
    return out


def collect_intraday(as_of: Optional[datetime] = None,
                     lookback_days: int = 200) -> pd.DataFrame:
    """采集全部分时品种，返回统一长表；start 给得足够早以拿满服务端保留窗口。

    lookback_days 仅决定请求起点（免费源会自行截断到其保留窗口），不代表能取到这么久；
    更长的历史依赖每日增量入库积累。"""
    as_of = pd.Timestamp(as_of or pd.Timestamp.now())
    start = as_of - pd.Timedelta(days=lookback_days)
    sess = PoliteSession(extra_headers=BROWSER_HEADERS)
    frames: List[pd.DataFrame] = []
    for sym, code in SINA_INTRADAY.items():
        frames.append(fetch_sina_intraday(sym, code, sess))
    for sym, code in CNBC_1H_SYMBOLS.items():
        frames.append(fetch_cnbc_1h(sym, code, start, as_of, sess))
    out = pd.concat([f for f in frames if f is not None and not f.empty],
                    ignore_index=True) if frames else _empty()
    if out.empty:
        return _empty()
    # 凌晨夜盘归属前一交易日、按 as_of 截断越界前瞻点（详见 apply_session_rollover）
    out = apply_session_rollover(out, as_of)
    out = out[_COLS].drop_duplicates(["symbol", "ts"], keep="last").sort_values(["symbol", "ts"])
    LOG.info("分时采集完成：%d 根，覆盖 %s",
             len(out),
             ", ".join(f"{s}:{n}" for s, n in out.groupby('symbol').size().items()))
    return out
