"""上海 INE 原油：持仓量主力连续（同合约换月调整）日 K。

为什么需要它
------------
新浪 ``SC0`` 是"主力连续"行情，它把不同到期月份的合约在换月日**直接拼接**：
换月前一日还是旧主力、当日切成新主力，两根 K 线之间的价差（月差）被错误地计成
当日涨跌。临近交割的近月合约还常出现逼仓/流动性塌陷（例如 2026-09 近月 SC2610
被异常拉高），SC0 把这段异动也带进连续序列，制造单日 ±7% 的**虚假跳空**，污染
日收益、5/20/60 日变化率与模型训练。

修法（可复现、不造数）
----------------------
1. 枚举月份合约 ``SC + YYMM``（INE 原油 2018-03 上市，至未来约 12 个月），逐合约
   拉新浪日 K（收盘 ``c`` 与持仓量 ``p``），按合约缓存到本地 CSV：首次运行回填
   全部历史合约，之后每日只增量刷新最近若干个合约（旧合约已摘牌、数据不再变化）。
2. 每个交易日选**持仓量最大**的合约为当日主力（交易所通用的持仓量主力规则，
   通常比行情商 SC0 的切换更早，天然避开近月交割前的逼仓段）。
3. 换月日用**新合约自身**的跨日收益（新合约在换月前后都有成交，可取其前一日收盘），
   并做比例复权（ratio adjustment）得到连续价 ``adj_close``：最新日复权因子为 1，
   向过去在每个换月边界乘以 ``新合约前一日收盘 / 旧合约前一日收盘``。这样
   ``adj_close`` 的日收益恒等于"同一合约"的真实日收益，同时价格水平锚定最新主力，
   原始各合约真实收盘价与持仓量也保留在缓存中，全程可审计。

任何合约拉取失败只缺该合约、不造数；若可用合约过少、无法重建连续序列，返回
``None``，由调用方回退到原始 ``SC0``（口径如实标注为"未换月调整"）。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..config import get_config
from ..utils import PoliteSession, get_logger
from .cn_futures_client import SINA_KLINE_URL
from .sources import BROWSER_HEADERS

LOG = get_logger(__name__)

# INE 原油首个合约月份（2018-03 上市）
FIRST_YM = (2018, 3)
# 向未来枚举的合约月数（覆盖当前主力之后的次主力/远月）
FUTURE_MONTHS = 12
# 每日运行时强制重新拉取的"最近合约"数量（旧合约摘牌后数据不变，读缓存即可）
REFRESH_RECENT = 10

_COLS = ["date", "open", "high", "low", "close", "volume", "oi"]


def _cache_dir() -> Path:
    sqlite_path = Path(get_config()["storage"]["sqlite_path"])
    d = sqlite_path.parent / "cache" / "sc_contracts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def contract_symbols(as_of: Optional[pd.Timestamp] = None) -> List[str]:
    """枚举从上市到 as_of 未来 FUTURE_MONTHS 的全部月份合约代码（SC + YYMM）。"""
    as_of = pd.Timestamp(as_of or pd.Timestamp.now())
    end = (as_of.year + (as_of.month + FUTURE_MONTHS) // 12,
           (as_of.month + FUTURE_MONTHS) % 12)
    # 上面的月取模在余 0 时需修正为 12
    if end[1] == 0:
        end = (end[0] - 1, 12)
    syms: List[str] = []
    y, m = FIRST_YM
    while (y, m) <= end:
        syms.append(f"SC{y % 100:02d}{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return syms


def fetch_contract_daily(code: str, sess: Optional[PoliteSession] = None
                         ) -> Optional[pd.DataFrame]:
    """拉取单个月份合约的全部日 K，返回标准列 DataFrame；失败返回 None。"""
    sess = sess or PoliteSession(extra_headers=BROWSER_HEADERS)
    resp = sess.get(SINA_KLINE_URL.format(symbol=code))
    if resp is None:
        return None
    try:
        arr = resp.json()
        if not arr:
            return None
        df = pd.DataFrame(arr)
        out = pd.DataFrame({
            "date": pd.to_datetime(df["d"]),
            "open": pd.to_numeric(df.get("o"), errors="coerce"),
            "high": pd.to_numeric(df.get("h"), errors="coerce"),
            "low": pd.to_numeric(df.get("l"), errors="coerce"),
            "close": pd.to_numeric(df.get("c"), errors="coerce"),
            "volume": pd.to_numeric(df.get("v"), errors="coerce"),
            # 新浪国内期货日 K 的持仓量字段为 p（open interest）
            "oi": pd.to_numeric(df.get("p"), errors="coerce"),
        })
        out = out.dropna(subset=["date", "close"]).sort_values("date")
        out = out.drop_duplicates("date", keep="last").set_index("date")
        return out if not out.empty else None
    except Exception as exc:  # 解析失败按不可达处理，绝不造数
        LOG.warning("上海合约 %s 日 K 解析失败：%s", code, exc)
        return None


def _cache_path(code: str) -> Path:
    return _cache_dir() / f"{code}.csv"


def load_contract_panel(as_of: Optional[pd.Timestamp] = None,
                        force_refresh: bool = False
                        ) -> Dict[str, pd.DataFrame]:
    """加载（必要时联网回填/增量）全部月份合约日 K，返回 {合约: DataFrame}。

    旧合约读本地缓存（摘牌后数据不变）；最近 REFRESH_RECENT 个合约每次重新拉取，
    以纳入当前主力与次主力的最新成交/持仓。force_refresh=True 时忽略缓存全部重拉。
    """
    as_of = pd.Timestamp(as_of or pd.Timestamp.now())
    symbols = contract_symbols(as_of)
    recent = set(symbols[-REFRESH_RECENT:])
    sess = PoliteSession(extra_headers=BROWSER_HEADERS)
    panel: Dict[str, pd.DataFrame] = {}
    n_fetched = 0
    for code in symbols:
        path = _cache_path(code)
        df: Optional[pd.DataFrame] = None
        if path.exists() and not force_refresh and code not in recent:
            try:
                cached = pd.read_csv(path, parse_dates=["date"]).set_index("date")
                if not cached.empty:
                    df = cached
            except Exception:
                df = None
        if df is None:
            df = fetch_contract_daily(code, sess)
            if df is not None:
                try:
                    df.to_csv(path)
                    n_fetched += 1
                except Exception as exc:
                    LOG.warning("合约 %s 缓存写入失败：%s", code, exc)
        if df is not None:
            panel[code] = df
    LOG.info("上海主力连续：%d 个月份合约（本次联网刷新 %d 个，缓存目录 %s）",
             len(panel), n_fetched, _cache_dir())
    return panel


def build_dominant_continuous(panel: Dict[str, pd.DataFrame]
                              ) -> Optional[pd.DataFrame]:
    """按每日持仓量最大选主力，同合约收益 + 比例复权，构造连续序列。

    返回列：``main``（当日主力合约）、``raw_close``（该主力真实收盘）、
    ``adj_factor``（比例复权因子，最新日=1）、``adj_close``（复权连续收盘）、
    ``adj_ret``（同合约日收益）。数据不足返回 None。
    """
    frames = []
    for code, df in panel.items():
        d = df[["close", "oi"]].copy()
        d.columns = ["close", "oi"]
        d["code"] = code
        d = d.reset_index()
        d = d.rename(columns={d.columns[0]: "date"})   # 兼容无名索引的面板
        frames.append(d)
    if not frames:
        return None
    long = pd.concat(frames, ignore_index=True)
    # 持仓量缺失（极个别早期日）用 0 处理，使其不被选为主力
    long["oi"] = long["oi"].fillna(0.0)
    # 每日持仓量最大的合约即主力；持仓并列时取收盘较新/代码较大者（确定性 tie-break）
    long = long.sort_values(["date", "oi", "code"], ascending=[True, False, True])
    dom = long.groupby("date", as_index=False).first().set_index("date").sort_index()
    dom = dom.rename(columns={"code": "main", "close": "raw_close"})
    if len(dom) < 30:
        return None

    # 各合约按日的收盘宽表，用于换月日取"新合约前一日收盘"
    close_wide = long.pivot_table(index="date", columns="code", values="close",
                                  aggfunc="last").sort_index()

    # 同合约日收益：换月日用新合约自身的前一日收盘
    same_ret = []
    dates = list(dom.index)
    for i, d in enumerate(dates):
        if i == 0:
            same_ret.append(float("nan"))
            continue
        prev = dates[i - 1]
        cur_main, prev_main = dom.at[d, "main"], dom.at[prev, "main"]
        c_t = float(dom.at[d, "raw_close"])
        if cur_main == prev_main:
            base = float(dom.at[prev, "raw_close"])
        else:
            base = float(close_wide.at[prev, cur_main]) if cur_main in close_wide.columns \
                else float("nan")
        same_ret.append(c_t / base - 1.0 if base and pd.notna(base) and base != 0
                        else float("nan"))
    dom["same_ret"] = same_ret

    # 比例复权因子：最新日=1，逆序在每个换月边界乘以 新合约前收/旧合约前收
    factor = pd.Series(1.0, index=dom.index)
    roll_dates = []
    for i in range(len(dates) - 1, 0, -1):
        d, prev = dates[i], dates[i - 1]
        f_next = factor.at[d]
        if dom.at[d, "main"] != dom.at[prev, "main"]:
            new_code, old_code = dom.at[d, "main"], dom.at[prev, "main"]
            c_new_prev = float(close_wide.at[prev, new_code]) \
                if new_code in close_wide.columns else float("nan")
            c_old_prev = float(dom.at[prev, "raw_close"])
            if pd.notna(c_new_prev) and c_old_prev:
                f_next = f_next * (c_new_prev / c_old_prev)
                roll_dates.append(d)
        factor.at[prev] = f_next
    dom["adj_factor"] = factor
    dom["adj_close"] = (dom["raw_close"] * dom["adj_factor"]).astype(float)
    # 复权连续价的日收益（数值上应等于同合约收益，留它做交叉校验）
    dom["adj_ret"] = dom["adj_close"].pct_change(fill_method=None)
    dom.attrs["n_rolls"] = len(roll_dates)
    dom.attrs["roll_dates"] = [pd.Timestamp(x).strftime("%Y-%m-%d") for x in sorted(roll_dates)]
    return dom


def fetch_sc_dominant_continuous(start: datetime, end: datetime,
                                 force_refresh: bool = False
                                 ) -> Optional[Tuple[pd.Series, dict]]:
    """对外主入口：返回 (复权连续收盘 series, meta)；无法重建返回 None（回退 SC0）。"""
    try:
        panel = load_contract_panel(pd.Timestamp(end), force_refresh=force_refresh)
        dom = build_dominant_continuous(panel)
        if dom is None:
            return None
        s = dom["adj_close"].dropna().sort_index()
        s = s.loc[(s.index >= pd.Timestamp(start).normalize()) &
                  (s.index <= pd.Timestamp(end).normalize() + pd.Timedelta(days=1))]
        if len(s) < 30:
            return None
        s.name = "shanghai_crude"
        meta = {
            "last_observed": s.index.max().strftime("%Y-%m-%d"),
            "caliber": ("新浪INE持仓量主力连续(同合约换月比例复权,人民币计价,日收盘;"
                        "统计主节点02:30由分时覆盖)"),
            "url": "https://finance.sina.com.cn/futures/quotes/SC0.shtml",
            "n_contracts": int(len(panel)),
            "n_rolls": int(dom.attrs.get("n_rolls", 0)),
        }
        LOG.info("上海换月调整连续：%d 个交易日、%d 次换月，末日 %s",
                 len(s), meta["n_rolls"], meta["last_observed"])
        return s, meta
    except Exception as exc:
        LOG.warning("上海持仓量主力连续重建失败，将回退原始SC0：%s", exc)
        return None


def sc_adjustment_factor_by_date(as_of: Optional[pd.Timestamp] = None
                                 ) -> Optional[pd.Series]:
    """返回每个交易日的比例复权因子（index=日, 值=adj_factor），供分时 SC0 连续
    在换月日做同一套比例复权，消除盘中序列的换月跳空。数据不足返回 None。"""
    try:
        panel = load_contract_panel(pd.Timestamp(as_of or pd.Timestamp.now()))
        dom = build_dominant_continuous(panel)
        if dom is None:
            return None
        return dom["adj_factor"].copy()
    except Exception as exc:
        LOG.warning("上海复权因子计算失败：%s", exc)
        return None
