"""地缘政治与宏观突发事件挖掘 —— 多 RSS 源优先级链。

源（config.data_sources.news_rss_feeds，按序尝试、累积去重）：
  1) OilPrice 主 feed：能源垂直专业媒体，相关度最高，全球稳定可达；
  2) Google News / Bing News：按关键词检索（海外网络可达）；
  3) WSJ Markets / MarketWatch：综合财经，兜底补充。
处理：标题关键词 → 主题归类；多空词典 → 情感与强度；按主题经验弹性估算
"若该事件单独主导，当日约影响多少美元/桶"。这是可解释的规则基线，后续可
替换为 FinBERT（保持输出表结构即可）。任何源都不可达才返回空，绝不伪造事件。
"""
from __future__ import annotations
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus
import pandas as pd
from ..config import get_config
from ..utils import PoliteSession, get_logger
from .sources import BROWSER_HEADERS

LOG = get_logger(__name__)

# 主题关键词（小写匹配）；顺序即优先级
THEME_KEYWORDS: Dict[str, List[str]] = {
    "supply_disruption": ["opec", "production cut", "output cut", "supply disruption",
                          "sanction", "embargo", "halt export", "voluntary cut", "supply cut"],
    "geopolitical_risk": ["red sea", "houthi", "attack", "strike", "war", "military",
                          "iran", "israel", "gaza", "ukraine", "russia", "tanker",
                          "political unrest", "coup", "conflict", "escalat", "hormuz"],
    "demand_outlook": ["demand", "inventory", "stockpile", "iea", "slowdown",
                       "refinery", "crude stock", "trade"],
    "cpi_surprise": ["cpi", "inflation", "consumer price"],
    "jobs_surprise": ["nonfarm", "payroll", "jobs report", "unemployment"],
    "fed_policy_expectation": ["fed", "rate cut", "rate hike", "fomc", "powell",
                               "interest rate", "central bank"],
    "institutional_view": ["goldman", "jpmorgan", "morgan stanley", "ubs", "citi",
                           "forecast", "price target", "raises estimate", "cuts estimate"],
}
# 能源相关性：垂直主 feed 用它筛掉与油气无关的条目
ENERGY_HINT = ("oil", "crude", "opec", "brent", "wti", "petrol", "gasoline", "diesel",
                "heating oil", "ulsd", "gasoil", "distillate",
               "refinery", "iran", "russia", "saudi", "energy", "barrel", "tanker",
               "natural gas", "shale", "hormuz", "iea", "fuel")
# 利多 / 利空词强度（对油价方向）
BULLISH = {"surge": 1.0, "soar": 1.0, "rally": 0.8, "jump": 0.7, "spike": 0.9,
           "disruption": 1.0, "attack": 0.9, "sanction": 0.9, "halt": 0.7,
           "cut supply": 1.0, "production cut": 1.0, "shortage": 0.9,
           "escalat": 0.8, "raise": 0.5, "upgrade": 0.6, "rate cut": 0.8,
           "tighter": 0.6, "drop in export": 0.8, "craters": 0.8}
BEARISH = {"plunge": 1.0, "slump": 0.9, "tumble": 0.9, "fall": 0.5, "drop": 0.4,
           "surplus": 0.8, "recession": 0.9, "weak demand": 1.0, "rate hike": 0.8,
           "downgrade": 0.7, "lower forecast": 0.9, "output raise": 0.9,
           "pump more": 0.8, "ease supply": 0.6, "risk-off": 0.5}
THEME_ELASTICITY = {
    "supply_disruption": 3.5, "geopolitical_risk": 3.0, "demand_outlook": 1.8,
    "cpi_surprise": 1.4, "jobs_surprise": 1.2, "fed_policy_expectation": 1.6,
    "institutional_view": 2.0,
}


def classify_theme(title_lower: str) -> str:
    for theme, kws in THEME_KEYWORDS.items():
        if any(kw in title_lower for kw in kws):
            return theme
    return "demand_outlook"


def score_title(title_lower: str) -> Tuple[float, float]:
    bull = sum(w for kw, w in BULLISH.items() if kw in title_lower)
    bear = sum(w for kw, w in BEARISH.items() if kw in title_lower)
    net = bull - bear
    intensity = min(1.0, abs(net) / 2.5)
    return net, intensity


def _entry_date(ent, as_of) -> pd.Timestamp:
    pp = ent.get("published_parsed")
    return pd.Timestamp(datetime(*pp[:6])) if pp else pd.Timestamp(as_of)


def _parse_entries(entries, source_name: str, as_of, energy_only: bool) -> List[dict]:
    """把 feedparser 条目统一打分成事件行。"""
    rows = []
    for ent in entries:
        title = ent.get("title", "").strip()
        if not title:
            continue
        lower = title.lower()
        if energy_only and not any(h in lower for h in ENERGY_HINT):
            continue
        theme = classify_theme(lower)
        net, intensity = score_title(lower)
        if intensity < 0.05:        # 过滤无明显多空信息的标题
            continue
        impact = (1 if net >= 0 else -1) * intensity * THEME_ELASTICITY.get(theme, 1.5)
        rows.append({
            "date": _entry_date(ent, as_of),
            "title": title,
            "source": ent.get("source", {}).get("title", source_name) or source_name,
            "feed": source_name,
            "theme": theme,
            "sentiment": "positive" if net > 0 else ("negative" if net < 0 else "neutral"),
            "intensity": round(intensity, 3),
            "est_price_impact": round(impact, 2),
            "url": ent.get("link", ""),
        })
    return rows


def _feed_urls(feed: dict, keywords: List[str], since: str) -> List[str]:
    """根据源类型生成要请求的 URL 列表。"""
    kind = feed["kind"]
    if kind == "rss_feed":
        return [feed["url"]]
    urls = []
    for kw in keywords:
        q = f"{kw} after:{since}"
        if kind == "google":
            base = get_config()["data_sources"]["google_news_rss"]
            urls.append(f"{base}?q={quote_plus(q)}&hl=en-US&gl=US&ceid=US:en")
        elif kind == "bing":
            base = get_config()["data_sources"]["bing_news_rss"]
            urls.append(f"{base}?q={quote_plus(kw)}&format=RSS&setmkt=en-US&setlang=en-US")
    return urls


def fetch_events(as_of: datetime, lookback_days: int = 30
                 ) -> Tuple[Optional[pd.DataFrame], List[dict]]:
    """按优先级依次尝试多个 RSS 源，累积去重；返回 (事件表, 每源尝试记录)。"""
    try:
        import feedparser
    except ImportError:
        LOG.warning("未安装 feedparser，跳过事件抓取")
        return None, [{"source": "all", "ok": False, "n": 0, "reason": "feedparser未安装"}]
    cfg = get_config()
    sess = PoliteSession(extra_headers=BROWSER_HEADERS)
    keywords = list(cfg["data_sources"]["news_keywords"])
    since = (pd.Timestamp(as_of) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    cutoff = pd.Timestamp(as_of) - timedelta(days=lookback_days)
    target_n = 40
    rows, seen, attempts = [], set(), []
    for feed in cfg["data_sources"]["news_rss_feeds"]:
        name, energy_focus = feed["name"], bool(feed.get("energy_focus"))
        before = len(rows)
        feed_ok, feed_raw = False, 0
        for url in _feed_urls(feed, keywords, since):
            resp = sess.get(url)
            if resp is None:
                continue
            parsed = feedparser.parse(resp.content)
            feed_raw += len(parsed.entries)
            new = _parse_entries(parsed.entries, name, as_of, energy_only=energy_focus)
            for r in new:
                if r["title"] in seen or r["date"] < cutoff:
                    continue
                seen.add(r["title"])
                rows.append(r)
            if new:
                feed_ok = True
        attempts.append({"source": name, "ok": feed_ok,
                         "n_raw": feed_raw, "n_kept": len(rows) - before,
                         "reason": "" if feed_ok else ("无有效条目/不可达")})
        LOG.info("事件源 %s：原始%d条，累计保留%d条", name, feed_raw, len(rows))
        if len(rows) >= target_n:
            break
    if not rows:
        return None, attempts
    ev = pd.DataFrame(rows).sort_values("date", ascending=False).head(target_n).reset_index(drop=True)
    LOG.info("多源真实新闻事件挖掘完成：%d 条（来自 %s）",
             len(ev), ",".join(sorted(ev["feed"].unique())))
    return ev, attempts


def geopolitical_proxy(events: Optional[pd.DataFrame], index: pd.DatetimeIndex) -> pd.Series:
    """GPRD 不可达时的【真实事件计数代理】：用真实地缘/供给主题事件的日度
    强度计数做滚动 z-score。口径不同于 GPRD，谱系中会明确标注为"事件衍生代理"。"""
    import numpy as np
    daily = pd.Series(0.0, index=index)
    if events is None or len(events) == 0:
        return pd.Series(np.nan, index=index)
    geo = events[events["theme"].isin(["geopolitical_risk", "supply_disruption"])]
    for _, r in geo.iterrows():
        d = pd.Timestamp(r["date"]).normalize()
        if d in daily.index:
            daily.loc[d] += float(r.get("intensity", 0.5))
    roll = daily.rolling(60, min_periods=20)
    z = (daily - roll.mean()) / roll.std()
    return (z * 25 + 100).clip(0, 300)   # 缩放到与 GPRD 相近的量纲
