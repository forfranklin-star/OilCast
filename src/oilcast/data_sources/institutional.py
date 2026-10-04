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
    "bofa": "美银", "bank of america": "美银",
    "barclays": "巴克莱", "iea": "IEA", "eia": "EIA", "opec": "OPEC",
}
# 方向词必须与"价格预测/目标价"邻近，避免把产量动作误判：
#  "raises/hikes ... price target/forecast" → 看涨；"OPEC hikes output/production"（增产）不匹配
RAISE_WORDS = re.compile(
    r"\b(?:raise|hike|boost|upgrade|lift)[a-z]*\b[^.;]{0,22}?"
    r"\b(?:price target|forecast|target|estimate|pt)\b|上调", re.I)
#  "cuts/lowers ... price target/forecast" → 看跌；"OPEC cuts production/supply"（减产利多）不匹配
LOWER_WORDS = re.compile(
    r"\b(?:lower|cut|slash|downgrade)[a-z]*\b[^.;]{0,22}?"
    r"\b(?:price target|forecast|target|estimate|pt)\b|下调", re.I)
# 价格【运动方向 + 目标价位】直接表述（必须绑定价位，避免把产量动作误判）：
#  "Brent could surge past $150" / "oil may fall below $70"。运动动词后接(可选)方向
#  介词 + 价位数字；break 的方向由 above/below 决定，"break to" 同时匹配两边→冲突中性。
UP_MOVE = re.compile(
    r"\b(?:surge|soar|spike|rally|jump|climb|rise|break)[a-z]*\s+"
    r"(?:past|above|through|over|to|toward)?\s*\$?\s?\d{2,3}\b", re.I)
DOWN_MOVE = re.compile(
    r"\b(?:plunge|tumble|slump|sink|drop|fall|decline|slide|break)[a-z]*\s+"
    r"(?:below|under|beneath|to|toward)?\s*\$?\s?\d{2,3}\b", re.I)
# 标题【必须】命中原油/油市实体，否则即便含机构名也属个股评级、一律不采
OIL_WORDS = re.compile(
    r"\b(oil|crude|brent|wti|opec\+?|petroleum|gasoil|diesel|heating oil)\b|油价|原油|石油", re.I)
# 个股语境词
STOCK_WORDS = re.compile(r"\b(stock|stocks|shares|equity|ticker|ipo)\b", re.I)
# 油价目标价必须与品种/桶【明确绑定】：
#  A) 品种词(brent/wti/oil/crude) 后 28 字符内出现数字（且该片段含 barrel/bbl 或品种为 brent/wti）
OIL_TARGET_FIRST = re.compile(
    r"\b(brent|wti|oil|crude)\b[^.;]{0,28}?\$?\s?(\d{2,3}(?:\.\d)?)\s?"
    r"(?:per barrel|/bbl|a barrel|/barrel)?", re.I)
#  B) 数字后紧跟 per barrel / /bbl / a barrel
OIL_TARGET_BBL = re.compile(
    r"\$?\s?(\d{2,3}(?:\.\d)?)\s?(?:per barrel|/bbl|a barrel|/barrel)", re.I)
# 区间口语 "low $100s / high $70s" 不是点目标价，不抽取
RANGE_VAGUE = re.compile(r"(?:low|mid|high)\s*\$\d{2,3}s\b", re.I)
QUERIES = ["Goldman Sachs oil price forecast", "JPMorgan Brent forecast",
           "UBS oil price target", "IEA oil demand outlook", "OPEC oil demand outlook"]


def _extract_target(seg: str):
    """从一个子句抽取与油/桶明确绑定的点目标价；区间口语/无绑定返回 None。"""
    if RANGE_VAGUE.search(seg):
        return None
    m = OIL_TARGET_FIRST.search(seg)
    if m:
        s = m.group(0)
        if re.search(r"barrel|/bbl", s, re.I) or re.search(r"\b(brent|wti)\b", s, re.I):
            return float(m.group(2))
    mb = OIL_TARGET_BBL.search(seg)
    return float(mb.group(1)) if mb else None


def parse_view(title: str, source: str, date) -> Optional[dict]:
    if not OIL_WORDS.search(title):
        return None  # 个股 PT（Darling/Alaska Air/Meesho/IOVA…）即使含机构名也排除
    # 按分号/竖线拆子句：目标价所在子句才代表"该机构的该观点"，机构/品种/方向均取该
    # 子句，避免"BofA 警告…；Goldman 另称…"被后半句机构带偏。无目标价则用整标题。
    parts = [p.strip() for p in re.split(r"\s*[;|]\s*", title) if p.strip()]
    tseg = next((p for p in parts if _extract_target(p) is not None), title)
    tlow, low = tseg.lower(), title.lower()
    inst_cn = next((cn for en, cn in INSTITUTIONS.items() if en in tlow), None) \
        or next((cn for en, cn in INSTITUTIONS.items() if en in low), None)
    if inst_cn is None:
        return None
    # 品种归属：目标价子句优先，回退整标题
    instrument = None
    for scope in (tlow, low):
        if re.search(r"\bbrent\b", scope):
            instrument = "brent"; break
        if re.search(r"\bwti\b", scope):
            instrument = "wti"; break
    target = _extract_target(tseg)
    # 无明确油价目标价、且标题是个股语境 → 丢弃
    if target is None and STOCK_WORDS.search(title):
        return None
    # 方向：目标价子句内，"上调目标价/价格上行运动"→看涨；"下调/下行运动"→看跌；
    # 同时出现（冲突，如 break to）或都无 → 中性。
    is_raise = bool(RAISE_WORDS.search(tseg)) or bool(UP_MOVE.search(tseg))
    is_lower = bool(LOWER_WORDS.search(tseg)) or bool(DOWN_MOVE.search(tseg))
    stance = "看涨" if is_raise and not is_lower else (
        "看跌" if is_lower and not is_raise else "中性")
    # target 不明确绑定则为 None，绝不拿标题里无关数字填充
    return {"date": date, "institution": inst_cn, "target_wti": target,
            "instrument": instrument, "stance": stance, "note": title, "source": source}


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
