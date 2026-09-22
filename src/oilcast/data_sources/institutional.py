"""机构观点提取：高盛/摩根大通/瑞银/IEA/EIA/OPEC 等的目标价与方向。
复用多 RSS 源链（OilPrice / Google / Bing / WSJ / MarketWatch），用正则从标题
抽取"raises/lowers ... $NN ... brent/wti forecast"类结构化信息。
官网月报（IEA OMR / EIA STEO / OPEC MOMR）可在此扩展。抓取不到返回 None。
"""
from __future__ import annotations
import re
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from urllib.parse import quote_plus
import pandas as pd
from ..config import get_config
from ..utils import PoliteSession, get_logger
from .events import _entry_date
from .sources import BROWSER_HEADERS

LOG = get_logger(__name__)

INSTITUTIONS = {
    "goldman": "高盛", "goldman sachs": "高盛",
    "jpmorgan": "摩根大通", "jp morgan": "摩根大通",
    "morgan stanley": "摩根士丹利", "ubs": "瑞银", "citi": "花旗",
    "barclays": "巴克莱", "iea": "IEA", "eia": "EIA", "opec": "OPEC",
}
RAISE_WORDS = re.compile(r"(raises?|hikes?|boosts?|upgrades?|上调)", re.I)
LOWER_WORDS = re.compile(r"(lowers?|cuts? .{0,20}forecast|downgrades?|slashes?|下调)", re.I)
PRICE = re.compile(r"\$?\s?(\d{2,3}(?:\.\d)?)\s?(?:dollars|usd)?", re.I)
QUERIES = ["Goldman Sachs oil price forecast", "JPMorgan Brent forecast",
           "UBS oil price target", "IEA oil demand outlook", "OPEC oil demand outlook"]


def parse_view(title: str, source: str, date) -> Optional[dict]:
    lower = title.lower()
    inst_cn = next((cn for en, cn in INSTITUTIONS.items() if en in lower), None)
    if inst_cn is None:
        return None
    is_raise = bool(RAISE_WORDS.search(title))
    is_lower = bool(LOWER_WORDS.search(title))
    m = PRICE.search(title)
    target = float(m.group(1)) if m else None
    stance = "看涨" if is_raise and not is_lower else ("看跌" if is_lower and not is_raise else "中性")
    if target is None and stance == "中性":
        return None
    return {"date": date, "institution": inst_cn, "target_wti": target,
            "stance": stance, "note": title, "source": source}


def _search_urls(feed: dict, since: str) -> List[str]:
    kind = feed["kind"]
    if kind == "rss_feed":
        return [feed["url"]]
    out = []
    for q in QUERIES:
        qq = f"{q} after:{since}"
        if kind == "google":
            base = get_config()["data_sources"]["google_news_rss"]
            out.append(f"{base}?q={quote_plus(qq)}&hl=en-US&gl=US&ceid=US:en")
        elif kind == "bing":
            base = get_config()["data_sources"]["bing_news_rss"]
            out.append(f"{base}?q={quote_plus(qq)}&format=RSS&setmkt=en-US&setlang=en-US")
    return out


def fetch_institutional_views(as_of: datetime, lookback_days: int = 90
                              ) -> Tuple[Optional[pd.DataFrame], List[dict]]:
    try:
        import feedparser
    except ImportError:
        return None, [{"source": "all", "ok": False, "reason": "feedparser未安装"}]
    sess = PoliteSession(extra_headers=BROWSER_HEADERS)
    since = (pd.Timestamp(as_of) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    rows, seen, attempts = [], set(), []
    for feed in get_config()["data_sources"]["news_rss_feeds"]:
        name, kept = feed["name"], 0
        for url in _search_urls(feed, since):
            resp = sess.get(url)
            if resp is None:
                continue
            for ent in feedparser.parse(resp.content).entries:
                title = ent.get("title", "")
                if title in seen:
                    continue
                parsed = parse_view(title, ent.get("source", {}).get("title", name),
                                    _entry_date(ent, as_of))
                if parsed is None:
                    continue
                seen.add(title)
                rows.append(parsed)
                kept += 1
        attempts.append({"source": name, "ok": kept > 0, "n_kept": kept,
                         "reason": "" if kept else "无机构观点条目"})
        if len(rows) >= 20:
            break
    if not rows:
        return None, attempts
    df = pd.DataFrame(rows).drop_duplicates(subset=["note"])
    df = df.sort_values("date", ascending=False).head(20).reset_index(drop=True)
    LOG.info("多源真实机构观点提取：%d 条", len(df))
    return df, attempts
