"""国际原油【真实】价格采集 —— 多数据源优先级链（failover）。

每个标的按 config.data_sources.source_chains 中的顺序依次尝试：
    FRED（EIA 现货转发，无 key）→ EIA v2（可选 key）→ Yahoo 原生 API → yfinance，
第一个返回足量真实观测的源被采用，尝试全过程写入谱系；源之间绝不混合拼接。
任何源都不可达 -> 该列保持缺失，由质量门判定 unavailable，严禁估算补齐。
"""
from __future__ import annotations
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
from ..config import get_config, _project_root
from ..utils import get_logger, run_with_timeout
from .cnbc_client import fetch_cnbc_bars
from .cn_futures_client import fetch_sina_inner, fetch_eastmoney_inner
from .fred_client import fetch_fred, fred_url
from .sc_contracts import fetch_sc_dominant_continuous
from .sources import run_chain
from .treasury_client import fetch_treasury_curve  # noqa: F401  (供宏观复用)
from .yahoo_client import fetch_yahoo_chart

LOG = get_logger(__name__)


# ----------------------------------------------------------- yfinance 备份
import threading
import time


class _YfThrottle:
    """两次【真实】yfinance 网络请求之间的最小间隔，串行节流、避免高频触发 429。"""
    def __init__(self, gap: float = 2.0):
        self.gap, self._last, self._lock = gap, 0.0, threading.Lock()

    def wait(self) -> None:
        with self._lock:
            d = time.monotonic() - self._last
            if d < self.gap:
                time.sleep(self.gap - d)
            self._last = time.monotonic()


_yf_throttle = _YfThrottle(gap=2.0)


def _yf_cache_path(ticker: str):
    safe = "".join(ch if ch.isalnum() else "_" for ch in ticker)
    d = _project_root() / "data" / "cache" / "yf"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{safe}.csv"


def _read_yf_cache(ticker: str) -> Optional[pd.Series]:
    """读 yfinance 已下载序列的本地真实副本（可追溯，非合成）。"""
    p = _yf_cache_path(ticker)
    if not p.exists():
        return None
    try:
        s = pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0]
        s.index = pd.to_datetime(s.index).normalize()
        return pd.to_numeric(s, errors="coerce").dropna().sort_index()
    except Exception:
        return None


def _write_yf_cache(ticker: str, s: pd.Series) -> None:
    try:
        pd.DataFrame({ticker: s}).to_csv(_yf_cache_path(ticker))
    except Exception as exc:
        LOG.warning("yfinance 缓存写入失败 %s：%s", ticker, exc)


def _download_one(ticker: str, start: datetime, end: datetime) -> Optional[pd.Series]:
    # Windows 中文安装路径下，先把 CA 束指到纯 ASCII 路径，规避 libcurl curl(77)。
    try:
        from ..certfix import ensure_ca_bundle
        ensure_ca_bundle()
    except Exception:
        pass
    import requests
    import yfinance as yf
    # 显式传标准 requests.Session：yfinance 默认的 curl_cffi(libcurl) 在含中文路径下
    # 加载 certifi 失败(curl 77)；requests 用 Python open() 读 CA，不受路径字符集影响。
    sess = requests.Session()
    _yf_throttle.wait()   # 串行节流：与上一次真实请求至少间隔 2s
    df = yf.download(ticker, start=pd.Timestamp(start).strftime("%Y-%m-%d"),
                     end=(pd.Timestamp(end) + timedelta(days=1)).strftime("%Y-%m-%d"),
                     progress=False, auto_adjust=True, threads=False, session=sess,
                     timeout=20)
    if df is None or df.empty:
        return None
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.squeeze().dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    return close.rename(ticker)


def fetch_yf_series(ticker: str, start: datetime, end: datetime) -> Optional[pd.Series]:
    """带磁盘增量缓存的 yfinance 拉取：
    - 请求区间已被缓存完整覆盖 -> 直接返回，【不发请求】；
    - 否则只增量拉取缓存末尾之后的新数据（必要时向前补早段），合并写回缓存。
    显著减少重复请求与数据量，配合节流降低 429 概率。缓存为真实下载副本，可追溯。"""
    try:
        import yfinance  # noqa: F401
    except ImportError:
        return None
    s0 = pd.Timestamp(start).normalize()
    e0 = pd.Timestamp(end).normalize()
    cached = _read_yf_cache(ticker)

    def _cut(s: pd.Series) -> Optional[pd.Series]:
        s = s[(s.index >= s0) & (s.index <= e0)]
        return s.rename(ticker) if len(s) else None

    # 完整命中：零请求
    if cached is not None and cached.index.min() <= s0 and cached.index.max() >= e0:
        return _cut(cached)

    pieces: List[pd.Series] = []
    # 向后增量（最常见：只拉最近未缓存的几天）
    back_start = s0 if cached is None else (cached.index.max() + timedelta(days=1))
    if cached is None or back_start <= e0:
        try:
            new = run_with_timeout(_download_one, (ticker, back_start, e0),
                                   timeout_sec=25)
        except Exception as exc:
            LOG.warning("yfinance %s 增量失败：%s", ticker, exc)
            new = None
        if new is not None and len(new):
            pieces.append(new)
    # 向前补早段（罕见：缓存起点晚于请求起点）
    if cached is not None and cached.index.min() > s0:
        try:
            pre = run_with_timeout(_download_one, (ticker, s0,
                           cached.index.min() - timedelta(days=1)), timeout_sec=25)
        except Exception as exc:
            LOG.warning("yfinance %s 前置补取失败：%s", ticker, exc)
            pre = None
        if pre is not None and len(pre):
            pieces.append(pre)
    if cached is not None:
        pieces.append(cached)
    if not pieces:
        return None
    full = pd.concat(pieces).groupby(level=0).last().sort_index()
    _write_yf_cache(ticker, full)
    return _cut(full)


def _load_eia_key() -> Optional[str]:
    """EIA key 读取顺序：环境变量 → 本地 key 文件（data/secrets/，不入库、不打包）。

    API key 是敏感凭证，绝不硬编码进公开代码/仓库；本机放在 data/secrets/ 下，
    云端由 GitHub Actions 从 Secret 注入环境变量。
    """
    env = str(get_config()["data_sources"]["eia_api_key_env"])
    key = os.getenv(env)
    if key and key.strip():
        return key.strip()
    root = _project_root()
    for p in (root / "data" / "secrets" / "eia_api_key.txt",
              root / "data" / "secrets" / "eia_key.txt",
              root / "data" / "eia_api_key.txt"):
        try:
            if p.exists():
                txt = p.read_text(encoding="utf-8-sig").strip()
                if txt:
                    return txt
        except Exception:
            continue
    return None


def fetch_eia_series(series_id: str, start: datetime, end: datetime) -> Optional[pd.Series]:
    """EIA v2 API（可选；key 来自环境变量 EIA_API_KEY 或本地 data/secrets 文件）。"""
    key = _load_eia_key()
    if not key:
        LOG.info("EIA 源跳过：未检测到 key（设环境变量 %s，或新建 "
                 "%s/data/secrets/eia_api_key.txt 并写入 key）",
                 get_config()["data_sources"]["eia_api_key_env"], _project_root())
        return None
    import requests
    url = (f"https://api.eia.gov/v2/seriesid/{series_id}?api_key={key}"
           f"&start={pd.Timestamp(start):%Y-%m-%d}&end={pd.Timestamp(end):%Y-%m-%d}")
    try:
        j = requests.get(url, timeout=12).json()["response"]["data"]
        return pd.Series({pd.Timestamp(r["period"]): float(r["value"]) for r in j}).sort_index()
    except Exception as exc:
        LOG.warning("EIA %s 失败：%s", series_id, exc)
        return None


def _provider_url(kind: str, ref: str) -> str:
    if kind == "fred":
        return fred_url(ref)
    if kind in ("yahoo", "yfinance"):
        return f"https://finance.yahoo.com/quote/{ref}"
    if kind == "eia":
        return f"https://www.eia.gov/opendata/browser/{ref.split('.')[0].lower()}"
    return ""


def _with_meta(s, caliber: str):
    """把单一 series 包成 (series, meta)，携带口径与末次观测，供新鲜度选源。"""
    if s is None or len(s) == 0:
        return None
    ss = s.dropna()
    return (s, {"caliber": caliber,
                "last_observed": ss.index.max().strftime("%Y-%m-%d")})


def _cnbc_tuple(got, caliber: str):
    if got is None:
        return None
    s, meta = got
    if caliber:
        meta["caliber"] = caliber
    return s, meta


def _build_providers(chain: List[dict], start, end) -> List[Tuple[str, callable]]:
    """把 config 中的有序源描述翻译成 run_chain 需要的 (name, fn) 列表。"""
    providers = []
    for item in chain:
        kind, ref, name = item["kind"], str(item["ref"]), item["name"]
        caliber = item.get("caliber", "")
        if kind == "cnbc":   # CNBC 已自带 (series, meta)
            providers.append((name, lambda r=ref, c=caliber:
                              _cnbc_tuple(fetch_cnbc_bars(r, start, end), c)))
        elif kind == "fred":
            providers.append((name, lambda r=ref, c=caliber: _with_meta(fetch_fred(r, start, end), c)))
        elif kind == "eia":
            # 检测到 key 时，标签去掉"(需EIA_API_KEY)"，避免已配置的用户误以为系统在索要 key
            display = (name.replace("(需EIA_API_KEY)", "").strip()
                       if _load_eia_key() else name)
            providers.append((display, lambda r=ref, c=caliber:
                              _with_meta(fetch_eia_series(r, start, end), c)))
        elif kind == "yahoo":
            providers.append((name, lambda r=ref, c=caliber: _with_meta(fetch_yahoo_chart(r, start, end), c)))
        elif kind == "yfinance":
            if bool(get_config()["data_sources"].get("yfinance_backup", True)):
                providers.append((name, lambda r=ref, c=caliber: _with_meta(fetch_yf_series(r, start, end), c)))
        elif kind == "sina_inner":   # 客户端已返回 (series, meta)
            providers.append((name, lambda r=ref: fetch_sina_inner(r, start, end)))
        elif kind == "eastmoney_inner":
            providers.append((name, lambda r=ref: fetch_eastmoney_inner(r, start, end)))
    return providers


def fetch_prices(as_of: datetime, history_days: int
                 ) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """返回 (真实价格表, 每字段来源元信息)。缺口保留 NaN，不做填充。
    价格标的：WTI、布伦特原油 + 美燃油(NYMEX ULSD 取暖油)、伦敦柴油(ICE Gasoil)。
    国内0#柴油因无可核验海外真实源已取消，改以两个可完整取数的成品油期货替代。"""
    cfg = get_config()
    start, end = pd.Timestamp(as_of) - timedelta(days=history_days), pd.Timestamp(as_of)
    # 不预生成"周一~周五"硬网格：那会把交易所假日排成交易日、凭空造 NaN 假缺口。
    # 价格序列只保留源返回的真实交易日，最终统一对齐到 observed 交易日轴。
    chains = dict(cfg["data_sources"]["source_chains"])
    # 品种顺序固定（原油在前、成品油在后）；实际取哪些取决于 config 里配置了源链的品种
    price_order = ["wti", "brent", "shanghai_crude",
                   "heating_oil", "gasoil"]
    price_fields = [f for f in price_order if f in chains]
    meta: Dict[str, dict] = {}
    out = {}
    for field in price_fields:
        # 上海原油：优先使用"持仓量最大主力 + 同合约换月比例复权"的连续日 K，消除 SC0 主力
        # 连续在换月日拼接不同月份合约造成的虚假跳空（含近月交割前逼仓异动）；原始各合约
        # 真实收盘/持仓缓存于 data/cache/sc_contracts 可审计。重建失败才回退到源链原始 SC0，
        # 回退时口径如实标注为"未换月调整"，绝不造数。
        if field == "shanghai_crude":
            sc = fetch_sc_dominant_continuous(start, end)
            if sc is not None:
                s_sc, sc_meta = sc
                s_sc = s_sc.sort_index().loc[
                    lambda x: (pd.to_datetime(x.index) >= start) &
                              (pd.to_datetime(x.index) <= end + timedelta(days=1))]
                if len(s_sc) >= 20:
                    out[field] = s_sc
                    meta[field] = {
                        "source_name": "新浪INE持仓量主力连续",
                        "url": sc_meta.get("url", ""),
                        "frequency": "daily_business",
                        "attempts": [],
                        "caliber": sc_meta.get("caliber", ""),
                        "note": (f"口径：{sc_meta.get('caliber','')}；{sc_meta.get('n_contracts',0)}"
                                 f"个月份合约、{sc_meta.get('n_rolls',0)}次换月，按每日持仓量最大"
                                 f"定主力、同合约收益比例复权；统计主节点02:30由分时覆盖"),
                    }
                    continue
        result = run_chain(_build_providers(chains.get(field, []), start, end),
                           field_name=field, min_obs=20, chain_timeout_sec=120,
                           select="freshest")
        if result.ok:
            out[field] = result.payload.sort_index().loc[
                lambda x: (pd.to_datetime(x.index) >= start) &
                          (pd.to_datetime(x.index) <= end + pd.Timedelta(days=1))]
            kind = next((i["kind"] for i in chains[field] if i["name"] == result.used), "")
            ref = next((str(i["ref"]) for i in chains[field] if i["name"] == result.used), "")
            caliber = result.meta.get("caliber", "")
            meta[field] = {
                "source_name": result.used,
                "url": _provider_url(kind, ref),
                "frequency": "daily_business",
                "attempts": result.attempts,
                "caliber": caliber,
                "note": f"口径：{caliber}；新鲜度优先选源链：{result.trail_text()}",
            }
        else:
            out[field] = pd.Series(dtype=float)   # 无真实交易日，保持缺失，绝不造数
            meta[field] = {"source_name": "UNAVAILABLE", "url": "",
                           "frequency": "daily_business",
                           "attempts": result.attempts,
                           "note": f"全部价格源均失败：{result.trail_text()}"}
    return pd.DataFrame(out)[price_fields], meta
