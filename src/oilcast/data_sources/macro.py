"""宏观与金融市场【真实】数据，多数据源优先级链，全部可核验、带观测日期。
日频（工作日）：10Y/2Y 国债（FRED→美国财政部）、广义美元（FRED→Yahoo）、
    GPRD 地缘风险（双地址依次尝试，均失败时由 collector 注入真实事件计数代理）。
月频（发布日对齐到工作日，携带 vintage 原始发布日期）：
    CPIAUCSL→CPI同比；PAYEMS→非农新增及意外z；FEDFUNDS；INDPRO→工业产出同比。
铁律：源不可达则该列保持 NaN 并由质量门标记，绝不合成。
"""
from __future__ import annotations
import io
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from ..config import get_config
from ..utils import PoliteSession, get_logger, run_with_timeout
from .fred_client import fetch_fred, fred_url
from .ecb_fx_client import fetch_ecb_fx
from .quality import align_monthly_with_vintage, limited_ffill_daily
from .sources import run_chain
from .treasury_client import fetch_treasury_curve
from .yahoo_client import fetch_yahoo_chart

LOG = get_logger(__name__)


def fetch_gpr(start: pd.Timestamp, end: pd.Timestamp,
              sess: PoliteSession) -> Optional[pd.Series]:
    """Caldara-Iacoviello 日频地缘风险指数；按配置多地址依次尝试，全失败返回 None。"""
    urls = list(get_config()["data_sources"].get("gprd_urls", []))
    for url in urls:
        resp = sess.get(url)
        if resp is None:
            continue
        try:
            if url.lower().endswith(".csv"):
                df = pd.read_csv(io.StringIO(resp.text))
            else:  # .xls 备份地址
                df = pd.read_excel(io.BytesIO(resp.content))
            dcol = next(c for c in df.columns if c.lower() in ("date", "day", "month"))
            vcol = next(c for c in df.columns if "gprd" in c.lower() or
                        (c.lower() == "gpr"))
            s = pd.Series(df[vcol].values, index=pd.to_datetime(df[dcol]),
                          name="gpr_index").sort_index()
            s = s.loc[(s.index >= start) & (s.index <= end)]
            if len(s) > 30:
                LOG.info("GPRD 命中地址 %s（%d 条）", url, len(s))
                return s
        except Exception as exc:
            LOG.warning("GPRD 地址 %s 解析失败：%s", url, exc)
    return None


def _daily_chain(chain_key: str, start, end, sess) -> "object":
    """按 source_chains[chain_key] 依次尝试（FRED→财政部/Yahoo），返回 ChainResult。"""
    chains = dict(get_config()["data_sources"]["source_chains"])

    def make(item):
        kind, ref, name = item["kind"], str(item["ref"]), item["name"]
        if kind == "fred":
            return (name, lambda: _t(fetch_fred(ref, start, end, sess)))
        if kind == "ecb":
            return (name, lambda: _t(fetch_ecb_fx(ref, start, end, sess)))
        if kind == "treasury":
            return (name, lambda: _t(fetch_treasury_curve(ref, start, end, sess)))
        if kind == "yahoo":
            return (name, lambda: _t(fetch_yahoo_chart(ref, start, end, sess)))
        return None

    providers = [p for it in chains.get(chain_key, []) if (p := make(it)) is not None]
    return run_chain(providers, field_name=chain_key, min_obs=20, chain_timeout_sec=80)


def _t(s):
    return (s, {}) if s is not None and len(s) > 0 else None


def fetch_macro(as_of: datetime, history_days: int
                ) -> Tuple[pd.DataFrame, Dict[str, dict], Dict[str, pd.Series]]:
    """返回 (日频宏观表, 字段来源元信息, vintage 月频发布日期)。"""
    cfg = get_config()
    start, end = pd.Timestamp(as_of) - timedelta(days=history_days), pd.Timestamp(as_of)
    bdays = pd.bdate_range(start, end)
    smap = dict(cfg["data_sources"]["fred_series_map"])
    ffill_lim = int(dict(cfg["quality_gate"]["macro_daily"])["ffill_limit_bdays"])
    sess = PoliteSession()
    meta: Dict[str, dict] = {}
    vintage: Dict[str, pd.Series] = {}
    frame = pd.DataFrame(index=bdays, dtype=float)

    def put_daily_chain(meta_key, chain_key, col_name):
        """日频字段：走多源优先级链。meta_key=谱系因素键, chain_key=源链键。"""
        result = _daily_chain(chain_key, start, end, sess)
        if result.ok:
            frame[col_name] = limited_ffill_daily(result.payload, ffill_lim).reindex(bdays)
            meta[meta_key] = {"source_name": result.used, "url": "",
                              "frequency": "daily_business", "attempts": result.attempts,
                              "note": f"源优先级链：{result.trail_text()}"}
        else:
            frame[col_name] = np.nan
            meta[meta_key] = {"source_name": "UNAVAILABLE", "url": "",
                              "frequency": "daily_business", "attempts": result.attempts,
                              "note": f"全部源失败：{result.trail_text()}"}

    def put_monthly(field, sid, col_name, transform):
        s = fetch_fred(sid, start - timedelta(days=400), end, sess) if sid else None
        if s is None:
            frame[col_name] = np.nan
            meta[field] = {"source_name": "UNAVAILABLE", "url": "", "frequency": "monthly"}
            return
        s = transform(s).dropna()
        aligned, vint = align_monthly_with_vintage(s, bdays)
        frame[col_name] = aligned
        vintage[col_name] = vint
        meta[field] = {"source_name": f"FRED:{sid}（月频，值为最近发布值并附发布日期）",
                       "url": fred_url(sid), "frequency": "monthly"}

    # ---- 日频（多源链）----
    put_daily_chain("us_treasury_10y", "us10y", "us10y")
    put_daily_chain("us2y", "us2y", "us2y")   # 辅助：供政策预期推导
    put_daily_chain("usd_index", "dxy", "dxy")
    # 汇率：USDCNY 用于把上海原油换算为美元/桶；USDJPY 作为分品种敏感度外生变量
    put_daily_chain("usd_cny", "usdcny", "usdcny")
    put_daily_chain("usd_jpy", "usdjpy", "usdjpy")
    # ---- 月频（FRED 官方序列，发布滞后可核验）----
    put_monthly("cpi_surprise", smap.get("cpi_level"), "cpi_yoy",
                lambda s: s.pct_change(12) * 100)

    def _nfp(s):   # 非农新增相对其 12 个月均值的标准化意外（真实统计代理）
        chg = s.diff()
        return (chg - chg.rolling(12, min_periods=3).mean()) / \
               chg.rolling(12, min_periods=3).std() * 60

    put_monthly("jobs_surprise", smap.get("payems"), "nonfarm_surprise", _nfp)
    put_monthly("fedfunds", smap.get("fedfunds"), "fedfunds", lambda s: s)
    put_monthly("demand_outlook", smap.get("demand_proxy"), "demand_proxy",
                lambda s: s.pct_change(12) * 100)
    # ---- 降息/加息预期：由真实短端利率走势推导（2Y 优先，FEDFUNDS 备选）----
    if frame["us2y"].notna().sum() > 60:
        frame["fed_expectation"] = -((frame["us2y"].diff(20)) /
                                     frame["us2y"].rolling(60, min_periods=20).std()
                                     ).clip(-2, 2) / 2
        meta["fed_policy_expectation"] = {"source_name": f"由 {meta['us2y']['source_name']} 短端利率推导",
                                          "url": meta["us2y"].get("url", ""),
                                          "frequency": "daily_business"}
    elif frame["fedfunds"].notna().sum() > 60:
        frame["fed_expectation"] = -((frame["fedfunds"].diff(20)) /
                                     frame["fedfunds"].rolling(60, min_periods=20).std()
                                     ).clip(-2, 2) / 2
        meta["fed_policy_expectation"] = {"source_name": "由 FRED:FEDFUNDS 推导",
                                          "url": fred_url(smap.get("fedfunds")),
                                          "frequency": "monthly"}
    else:
        frame["fed_expectation"] = np.nan
        meta["fed_policy_expectation"] = {"source_name": "UNAVAILABLE", "url": "",
                                          "frequency": "na"}
    # ---- GPR 地缘风险（多地址；单独限时，慢源不得拖垮 FRED 主宏观；
    #      均不可达/超时先置 NaN，collector 注入真实事件代理，绝不合成）----
    gpr = run_with_timeout(fetch_gpr, (start, end, sess), timeout_sec=45)
    if gpr is not None and len(gpr) > 30:
        frame["gpr_index"] = limited_ffill_daily(gpr, ffill_lim).reindex(bdays)
        meta["geopolitical_risk"] = {"source_name": "GPRD(Caldara&Iacoviello)",
                                     "url": str(cfg["data_sources"]["gprd_urls"][0]),
                                     "frequency": "daily_business"}
    else:
        frame["gpr_index"] = np.nan
        meta["geopolitical_risk"] = {"source_name": "UNAVAILABLE", "url": "",
                                     "frequency": "daily_business",
                                     "note": "GPRD 各地址均不可达"}
    LOG.info("真实宏观表非空观测：%s",
             {c: int(frame[c].notna().sum()) for c in frame.columns})
    return frame, meta, vintage
