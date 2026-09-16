"""特征工程。

设计原则：
1. **无未来泄漏**：第 t 行特征只使用 ≤t 的信息；标签为 t+1..t+h 累计收益；
2. **方向统一**：所有因素特征处理成"数值越大 → 越利多油价"，
   模型系数/重要性聚合后符号可直接解释；
3. **缺失不造假**：源不可达导致的缺失保留 NaN，绝不 fillna(0) 伪装成"中性"；
   训练时只使用真实观测齐全的样本，整列缺失的因素在权重层显式剔除；
4. **两层结构**：细粒度特征进模型，按 FACTOR_GROUPS 聚合为九大因素权重。
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# 细特征 -> 因素（技术面/跨品种归以下划线开头的内部组，不参与因素权重展示）
FACTOR_GROUPS: Dict[str, list] = {
    "supply_disruption": ["ev_supply_5d"],
    "geopolitical_risk": ["gpr_level_z", "gpr_chg_5d", "ev_geo_5d", "geo_premium_chg5",
                          "china_supply_risk_5d"],
    "usd_index": ["dxy_chg_5d_dir"],
    "us_treasury_10y": ["us10y_chg_5d_dir"],
    "cpi_surprise": ["cpi_surprise_dir", "ev_cpi_5d"],
    "jobs_surprise": ["nfp_surprise_dir", "ev_jobs_5d"],
    "fed_policy_expectation": ["fed_exp_level", "fed_exp_chg", "ev_fed_5d"],
    "fx_jpy": ["usdjpy_z", "usdjpy_chg_5d"],
    "demand_outlook": ["demand_chg_5d", "ev_demand_5d"],
    "institutional_view": ["inst_bias", "ev_view_5d"],
    "_technical_": [
        "ret_1d", "ret_5d", "ma_gap", "vol_20",
        "major_event_regime",   # 可审计重大事件状态（新冠/俄乌/红海/2026美以伊）
    ],
}
# 可统一到"美元/桶"口径的原油品种：WTI/Brent 本就是美元/桶；上海原油为人民币/桶，
# 需除以真实 USDCNY 汇率换算成美元/桶后，才能与布伦特计算升贴水（水平价差）。
# 成品油（美元/加仑、美元/吨）单位不同，只做无量纲相对强弱、不做水平价差。
USD_BARREL_EQUIV = {"wti", "brent", "shanghai_crude"}
# 跨品种对照基准（顺序固定 → 每个品种模型的跨品种特征列集合固定，保证热启动 schema 稳定）
PEER_BENCHMARKS = ("brent", "wti")


def cross_market_columns() -> list:
    """固定的跨品种特征列名（与数据是否取到无关，缺数据时该列保持 NaN 占位）。"""
    cols = []
    for peer in PEER_BENCHMARKS:
        cols += [f"rs_{peer}_5d", f"rs_{peer}_20d",
                 f"spread_{peer}", f"spread_{peer}_chg5"]
    return cols


# 把跨品种 8 列登记到内部组：进模型，但不参与因素权重展示
FACTOR_GROUPS["_crossmarket_"] = cross_market_columns()


# ----------------------------------------------------------------------------
# 交易时段对齐（解决各油种/市场收盘时刻不一致导致的隐性未来函数）
#
# 日 K 的"日期标签"并不等于"同一时刻"：上海 INE SC 日盘 15:00（北京）收盘，即 UTC 07:00；
# WTI/Brent/美燃油/伦敦柴油及美债、美元指数等美国市场按美东交易日，要到 UTC 21:00 前后才
# 收盘（≈北京次日凌晨）。若按日期标签"同日"对齐，那么在为上海原油构造 t 日特征时，会用到
# 它收盘后十几个小时才落定的海外行情——既是跨市场价差/升贴水错位，也是隐性未来函数，会同时
# 污染波动传导、因素敏感度与回测。这里把每根日 K 还原到其【真实收盘 UTC 时刻】，对目标品种
# 每个交易日的收盘时刻做 backward as-of：只取"在该时点已经收盘落定"的最近一根外生 K 线。
# 方向由收盘先后自动决定：上海(07) 看海外只能取到前一交易日；海外(21) 看上海同日即可（上海
# 已领先收盘）；海外之间同时刻、仍为同日。仅用固定日历规则，不使用任何未来价格数值，无泄漏。
# ----------------------------------------------------------------------------
# 各品种日 K 标签日的真实收盘时刻（UTC 小时；海外统一取偏晚的 21:00，最防未来函数，
# 且海外品种之间偏移相同、彼此对齐关系不变；不随夏令时切换，日频"差一天"结论保持稳健）
PRICE_CLOSE_UTC_HOUR: Dict[str, float] = {
    "shanghai_crude": 7.0,     # 北京 15:00 日盘收盘 = UTC 07:00（夜盘计入下一标签日）
    "wti": 21.0, "brent": 21.0, "heating_oil": 21.0, "gasoil": 21.0,
}
# 宏观/金融序列标签日的"落定 UTC 小时"：人民币定盘在北京白天(07)；GPR 学术指数发布时点
# 不定取正午(12)；其余为美国市场/数据（DXY、美债、CPI、非农、联邦基金、需求代理、USDJPY）
MACRO_CLOSE_UTC_HOUR: Dict[str, float] = {
    "usdcny": 7.0, "gpr_index": 12.0,
    "dxy": 21.0, "us10y": 21.0, "us2y": 21.0, "cpi_yoy": 21.0,
    "nonfarm_surprise": 21.0, "fedfunds": 21.0, "fed_expectation": 21.0,
    "demand_proxy": 21.0, "usdjpy": 21.0,
}
_MACRO_DEFAULT_HOUR = 21.0      # 未显式列出的宏观列默认按美国市场收盘
_PRICE_DEFAULT_HOUR = 21.0
_MAX_STALE_DAYS = 7             # as-of 桥接上限：超过 7 个自然日仍无更近收盘则判缺失


def _asof_series(s: pd.Series, target_idx: pd.DatetimeIndex,
                 src_hour: float, tgt_hour: float,
                 max_stale_days: int = _MAX_STALE_DAYS) -> pd.Series:
    """把源序列（日期标签索引）按其真实收盘时刻 backward as-of 到目标交易日收盘时刻。

    即对目标每个交易日，只保留在该目标收盘时刻【已经落定】的最近源值；源休市时自然回取
    前一根（受控于 max_stale_days，不用陈旧值硬桥）。"""
    ss = pd.Series(s).dropna()
    if ss.empty:
        return pd.Series(np.nan, index=target_idx, dtype=float)
    t_close = pd.DatetimeIndex(pd.to_datetime(target_idx)).normalize() \
        + pd.Timedelta(hours=tgt_hour)
    s_close = pd.DatetimeIndex(pd.to_datetime(ss.index)).normalize() \
        + pd.Timedelta(hours=src_hour)
    order = np.argsort(s_close.values)
    s_close_v, s_val = s_close.values[order], ss.values[order]
    left = pd.DataFrame({"_t": t_close}).sort_values("_t")
    right = pd.DataFrame({"_v": s_val}, index=pd.DatetimeIndex(s_close_v)).sort_index()
    merged = pd.merge_asof(left, right, left_on="_t", right_index=True, direction="backward")
    out = pd.Series(merged["_v"].values, index=target_idx, dtype=float)
    # 陈旧度：匹配到的源收盘时刻距目标收盘时刻超过上限 → 置缺失（不拿很久以前的值桥接）
    pos = np.searchsorted(s_close_v, t_close.values, side="right") - 1
    age_days = np.full(len(t_close), np.inf)
    for i, mi in enumerate(pos):
        if mi >= 0:
            age_days[i] = (t_close[i] - pd.Timestamp(s_close_v[mi])).total_seconds() / 86400.0
    out[age_days > max_stale_days] = np.nan
    return out


def align_exogenous_to_target(prices: pd.DataFrame, macro: pd.DataFrame,
                              target: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (对齐后价格面板, 对齐后宏观面板)：目标品种自身价格保持原值，其余品种与全部宏观
    序列按真实收盘 UTC 时刻 as-of 到目标品种的每个交易日（只含目标收盘时已知的信息）。"""
    idx = pd.DatetimeIndex(prices.index)
    tgt_h = PRICE_CLOSE_UTC_HOUR.get(target, _PRICE_DEFAULT_HOUR)
    ap = prices.copy()
    for c in ap.columns:
        if c == target:
            continue                      # 目标自身：自身收盘对自身收盘，无错配，保持原样
        ap[c] = _asof_series(prices[c], idx,
                             PRICE_CLOSE_UTC_HOUR.get(c, _PRICE_DEFAULT_HOUR), tgt_h).values
    am = pd.DataFrame(index=idx)
    for c in macro.columns:
        am[c] = _asof_series(macro[c], idx,
                             MACRO_CLOSE_UTC_HOUR.get(c, _MACRO_DEFAULT_HOUR), tgt_h).values
    return ap, am


def _bridge_7_calendar_days(s: pd.Series, idx) -> pd.Series:
    """按【自然日】向前桥接 ≤7 天的最近已知真实值（用于汇率等极稳、发布偶有滞后的序列）；
    超过 7 天仍无新值则留 NaN，不拿陈旧值硬凑、不造数。"""
    ss = pd.Series(s).dropna()
    ss.index = pd.to_datetime(ss.index).normalize()
    ss = ss[~ss.index.duplicated(keep="last")].sort_index()
    if ss.empty:
        return pd.Series(np.nan, index=idx, dtype=float)
    daily = pd.date_range(ss.index.min(), pd.DatetimeIndex(idx).max(), freq="D")
    return ss.reindex(daily).ffill(limit=7).reindex(pd.DatetimeIndex(idx))


def _usd_barrel_log(prices: pd.DataFrame, idx, usdcny) -> dict:
    """各原油品种的美元/桶对数价。上海原油用真实 USDCNY 换算；汇率缺失则该品种留缺。"""
    out = {}
    for c in ("wti", "brent"):
        if c in prices:
            out[c] = np.log(prices[c].reindex(idx).ffill(limit=3))
    if "shanghai_crude" in prices and usdcny is not None:
        fx = _bridge_7_calendar_days(usdcny, idx)   # FRED 定盘偶滞后，7 自然日内桥接
        cny = prices["shanghai_crude"].reindex(idx).ffill(limit=3)
        out["shanghai_crude"] = np.log(cny / fx)   # 元/桶 ÷ (元/美元) = 美元/桶
    return out


def cross_market_features(prices: pd.DataFrame, target: str,
                          usdcny: Optional[pd.Series] = None) -> pd.DataFrame:
    """目标品种相对基准（尤其布伦特）的相对强弱与美元口径升贴水。

    * rs_{peer}_k ：目标与基准近 k 日本币【对数收益之差】，无量纲，刻画近期相对强弱；
    * spread_{peer} / _chg5 ：把两者统一到美元/桶口径后的对数价差水平及 5 日变化。
      WTI/Brent 直接相减；上海原油先按真实 USDCNY 换算成美元/桶再减布伦特，
      即"上海原油换算美元后对布伦特的升水/贴水"波动（用户明确要求纳入学习）；
      成品油单位不同、或汇率缺失时，水平价差整列保持 NaN（绝不混用口径）。
    """
    idx = prices.index
    out = pd.DataFrame(index=idx)
    log_t_local = np.log(prices[target].ffill(limit=3)) if target in prices else None
    usd_log = _usd_barrel_log(prices, idx, usdcny)
    for peer in PEER_BENCHMARKS:
        c5, c20 = f"rs_{peer}_5d", f"rs_{peer}_20d"
        cs, cs5 = f"spread_{peer}", f"spread_{peer}_chg5"
        if log_t_local is None or peer not in prices:
            out[c5] = out[c20] = out[cs] = out[cs5] = np.nan
            continue
        log_p_local = np.log(prices[peer].reindex(idx).ffill(limit=3))
        out[c5] = log_t_local.diff(5) - log_p_local.diff(5)
        out[c20] = log_t_local.diff(20) - log_p_local.diff(20)
        # 美元口径升贴水（上海原油在此完成汇率换算后比较）
        if target in USD_BARREL_EQUIV and peer in USD_BARREL_EQUIV \
                and target in usd_log and peer in usd_log:
            spread = usd_log[target] - usd_log[peer]
            out[cs] = spread
            out[cs5] = spread.diff(5)
        else:
            out[cs] = np.nan
            out[cs5] = np.nan
    return out
THEME_TO_COL = {
    "supply_disruption": "ev_supply_5d",
    "geopolitical_risk": "ev_geo_5d",
    "demand_outlook": "ev_demand_5d",
    "cpi_surprise": "ev_cpi_5d",
    "jobs_surprise": "ev_jobs_5d",
    "fed_policy_expectation": "ev_fed_5d",
    "institutional_view": "ev_view_5d",
}
# 因素整列非空率低于该阈值 → 判定该因素数据缺失，不参与权重归一化
MIN_FACTOR_COVERAGE = 0.20


def _zscore(s: pd.Series, win: int = 120) -> pd.Series:
    mu = s.rolling(win, min_periods=20).mean()
    sd = s.rolling(win, min_periods=20).std().replace(0, np.nan)
    return (s - mu) / sd


@lru_cache(maxsize=1)
def _load_major_events():
    """加载可审计的重大事件年表（公开事实日期+来源）。文件缺失返回 None（不造数）。"""
    import yaml
    for base in (Path(__file__).resolve().parents[3], Path.cwd()):
        f = base / "data" / "reference" / "major_events.yaml"
        if f.exists():
            try:
                doc = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
                return doc.get("events", []) or []
            except Exception:
                return []
    return None


def major_event_regime(index: pd.DatetimeIndex) -> pd.Series:
    """重大事件 regime 哑变量：处于某重大事件（疫情/战争等）影响窗口内取其强度（默认1），
    否则 0；无年表文件时整列 NaN（显式缺失，绝不凭空生成）。

    事件起止为公开、可核验的事实日期（见 yaml 内 source），第 t 行只反映"截至 t 事件是否
    正在发生"，无未来泄漏；滚动重训使该因素权重随局势动态变化。
    """
    events = _load_major_events()
    if not events:
        return pd.Series(np.nan, index=index, name="major_event_regime")
    regime = pd.Series(0.0, index=index)
    for e in events:
        try:
            st = pd.Timestamp(e["start"])
            en = pd.Timestamp(e["end"]) if e.get("end") else index.max()  # end 空=持续至今
            intensity = float(e.get("intensity", 1.0))
        except Exception:
            continue
        mask = (index >= st) & (index <= en)
        regime.loc[mask] = np.maximum(regime.loc[mask], intensity)
        # 事件内部的升级/缓和阶段（phases）：用子区间强度在 base 之上取 max，
        # 使一场长期冲突内部仍有"升级台阶"（如开战封锁、油轮战升级），避免长期事件
        # 被压成无方差常数而无法被模型利用。每个 phase 同样要求公开可核验日期与来源。
        for ph in (e.get("phases") or []):
            try:
                pst = pd.Timestamp(ph["start"])
                pen = pd.Timestamp(ph["end"]) if ph.get("end") else index.max()
                pval = float(ph.get("intensity", intensity))
            except Exception:
                continue
            pmask = (index >= pst) & (index <= pen) & mask
            regime.loc[pmask] = np.maximum(regime.loc[pmask], pval)
    return regime


def conditional_geo_sensitivity(prices: pd.DataFrame, targets: list,
                                recent_window: int = 500) -> Dict[str, dict]:
    """事件期【条件敏感度】归因——回答"当前地缘冲突中各品种到底有多敏感"。

    因素权重（模型特征重要性）是近 500 交易日的全窗平均，会把当前事件期的高敏感度
    稀释；本函数只用真实价格做可审计统计，独立于树模型：
      * vol_ratio ：重大事件期年化波动 / 非事件期年化波动（>1 表示冲突中波动放大）；
      * geo_beta ：日收益对"布伦特-WTI 价差日变化"（地缘供给溢价的市场定价）的 OLS 斜率，
                   分别给事件期 / 非事件期，二者之差即冲突带来的敏感度抬升；
      * event_cumret_pct：事件期累计涨跌幅；in_event_now：观测日是否仍处事件期。
    样本不足（<20）的统计项返回 None，绝不造数。
    """
    prices = prices.sort_index()
    idx = prices.index
    regime = major_event_regime(idx)
    out: Dict[str, dict] = {}
    if "brent" not in prices or "wti" not in prices:
        return {t: {"available": False, "reason": "缺 brent/wti，无法构造地缘溢价"} for t in targets}
    # 用【滞后一期的 5 日地缘溢价变化】做横轴：既测"地缘冲击→次日收益传导"，又避免
    # 当日 bw≈r_brent-r_wti 与 Brent/WTI 自身收益的机械恒等。
    bw = (np.log(prices["brent"].ffill(limit=3)) -
          np.log(prices["wti"].ffill(limit=3))).diff(5).shift(1)
    win_idx = idx[-recent_window:]
    ev_mask = (regime.reindex(win_idx).fillna(0.0) > 0)

    def _beta(y: pd.Series, x: pd.Series):
        m = y.notna() & x.notna()
        if m.sum() < 20 or x[m].std(ddof=0) < 1e-12:
            return None, int(m.sum())
        xv, yv = x[m], y[m]
        return float(np.cov(yv, xv, ddof=0)[0, 1] / np.var(xv)), int(m.sum())

    for t in targets:
        if t not in prices:
            out[t] = {"available": False, "reason": "缺价格"}
            continue
        r = np.log(prices[t].ffill(limit=3)).diff().reindex(win_idx)
        x = bw.reindex(win_idx)
        b_ev, n_ev = _beta(r[ev_mask], x[ev_mask])
        b_no, n_no = _beta(r[~ev_mask], x[~ev_mask])
        v_ev = float(r[ev_mask].std() * np.sqrt(252)) if ev_mask.sum() >= 20 and r[ev_mask].notna().sum() >= 20 else None
        v_no = float(r[~ev_mask].std() * np.sqrt(252)) if (~ev_mask).sum() >= 20 and r[~ev_mask].notna().sum() >= 20 else None
        r_ev = r[ev_mask].dropna()
        cum = float(np.expm1(r_ev.sum()) * 100) if len(r_ev) >= 20 else None
        out[t] = {
            "available": True,
            "vol_event": round(v_ev, 3) if v_ev is not None else None,
            "vol_normal": round(v_no, 3) if v_no is not None else None,
            "vol_ratio": round(v_ev / v_no, 2) if (v_ev and v_no) else None,
            "geo_beta_event": round(b_ev, 3) if b_ev is not None else None,
            "geo_beta_normal": round(b_no, 3) if b_no is not None else None,
            "beta_lift": round(b_ev - b_no, 3) if (b_ev is not None and b_no is not None) else None,
            "event_cumret_pct": round(cum, 1) if cum is not None else None,
            "n_event": n_ev, "n_normal": n_no,
            "in_event_now": bool(ev_mask.iloc[-1]) if len(ev_mask) else False,
            "event_share_pct": round(float(ev_mask.mean()) * 100, 1),
        }
    return out


def aggregate_events(events: pd.DataFrame, index: pd.DatetimeIndex,
                     source_available: bool = True) -> pd.DataFrame:
    """事件按主题聚合到日频，再做 5 日滚动强度和。

    事件源不可达时返回全 NaN（"没抓到"不等于"当天零事件"）；
    源可达但当天确无事件，才是真实的 0。
    """
    cols = list(THEME_TO_COL.values())
    if not source_available:
        return pd.DataFrame(np.nan, index=index, columns=cols)
    out = pd.DataFrame(np.nan, index=index, columns=cols)
    if events is None or events.empty:
        return out
    ev = events.copy()
    ev["date"] = pd.to_datetime(ev["date"]).dt.normalize()
    daily = ev.groupby(["date", "theme"])["intensity"].sum().unstack(fill_value=0)
    daily = daily.reindex(index).fillna(0.0)
    for theme, col in THEME_TO_COL.items():
        if theme in daily.columns:
            # 该主题至少真实出现过：无事件日记真实 0，滚动 5 日强度；
            # 从未出现的主题保持 NaN（没有该类信号 ≠ 该因素恒为 0 可建模）
            out[col] = daily[theme].rolling(5, min_periods=1).sum()
    return out


# 对华中东供油链路（霍尔木兹海峡/伊朗）受冲击的标题关键词。上海 INE SC 可交割标的为
# 中东含硫原油、中国是伊朗等产油国主要买家，"油轮被炸/霍尔木兹对峙/封锁"这类事件对上海
# 原油的传导强于一般地缘新闻，故单列一个【时变】因子刻画边际升级（区别于常年=1 的大事件
# regime）。以真实标题文本为准、不依赖主题分类是否打对。
CHINA_SUPPLY_PATTERN = re.compile(
    r"hormuz|strait|iran|iranian|irgc|revolutionary guard|persian gulf|"
    r"tanker|seiz|blockade|embargo|"
    r"霍尔木兹|海峡|伊朗|油轮|油船|运油|扣押|封锁|禁运|断供|波斯湾|袭船",
    re.IGNORECASE)


def china_supply_risk(events: pd.DataFrame, index: pd.DatetimeIndex,
                      source_available: bool = True, window: int = 5) -> pd.Series:
    """从真实新闻标题识别霍尔木兹/伊朗对华供油链路冲击，按事件强度聚合成日频、再做
    window 日滚动和，得到时变风险强度。

    - 事件源不可达 → 全 NaN（"没抓到"不等于"当天零冲击"）；
    - 源可达但无标题命中 → 真实的 0；命中则按该事件真实 intensity 求和（不人为放大）。
    权重仍交由各品种模型自行学习，不预设上海一定更高（但给了模型捕捉该特异性的通道）。"""
    name = f"china_supply_risk_{window}d"
    idx = pd.DatetimeIndex(index)
    if not source_available:
        return pd.Series(np.nan, index=idx, name=name)
    if events is None or events.empty or "title" not in events.columns:
        return pd.Series(0.0, index=idx, name=name)
    e = events.copy()
    e["date"] = pd.to_datetime(e["date"]).dt.normalize()
    hit = e["title"].fillna("").str.contains(CHINA_SUPPLY_PATTERN, regex=True)
    eh = e.loc[hit]
    if eh.empty:
        return pd.Series(0.0, index=idx, name=name)
    w = pd.to_numeric(eh["intensity"] if "intensity" in eh.columns else 1.0,
                      errors="coerce").fillna(0.3)
    by_day = pd.Series(np.asarray(w, dtype=float), index=eh["date"].values).groupby(level=0).sum()
    daily = by_day.reindex(idx.normalize().unique()).fillna(0.0).sort_index()
    roll = daily.rolling(window, min_periods=1).sum()
    roll.index = pd.DatetimeIndex(roll.index)
    return roll.reindex(idx.normalize()).rename(name)


def institutional_bias(views: pd.DataFrame, index: pd.DatetimeIndex,
                       source_available: bool = True) -> pd.Series:
    """机构情绪：净看涨比例的 30 日滚动均值（-1~1）。源不可达返回全 NaN。"""
    if not source_available or views is None or views.empty:
        return pd.Series(np.nan if not source_available else 0.0, index=index, name="inst_bias")
    v = views.copy()
    v["date"] = pd.to_datetime(v["date"]).dt.normalize()
    v["score"] = v["stance"].map({"看涨": 1.0, "看跌": -1.0, "中性": 0.0}).fillna(0)
    # 同一交易日可能有多条机构观点：先按日聚合为当日净看涨均值，保证索引唯一，
    # 否则 rolling 后再 reindex 会抛 duplicate labels（CI 全新采集时复现过）。
    daily = v.set_index("date")["score"].groupby(level=0).mean().sort_index()
    idx_u = pd.DatetimeIndex(index)
    roll = daily.rolling("30D").mean().reindex(idx_u.unique().sort_values(),
                                               method="ffill", limit=22)
    return roll.clip(-1, 1).reindex(idx_u).rename("inst_bias")


def build_features(prices: pd.DataFrame, macro: pd.DataFrame,
                   events: pd.DataFrame, views: pd.DataFrame,
                   target: str = "wti",
                   events_available: bool = True,
                   views_available: bool = True,
                   use_session_align: bool = True,
                   intraday_session: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """返回对齐后的特征矩阵；缺失保持 NaN，不做零值填充。

    所有外生于目标品种的序列（其他油种、宏观/汇率）先按各市场真实收盘 UTC 时刻 as-of 到
    目标品种的交易日，确保目标在 t 日收盘时只看得到已经落定的外生行情（消除跨时区错配与
    隐性未来函数，例如上海原油不会用到同日北京时间次日凌晨才收盘的 WTI/Brent）；目标自身
    技术面仍按其自身连续交易日计算。"""
    # 健壮性：多源合并/重复落库可能让价格或宏观出现重复交易日索引，而下游大量
    # reindex/rolling 要求索引唯一，否则抛 "cannot reindex on an axis with duplicate labels"。
    # 入口统一按交易日去重（重复日保留最后一条）并排序，保证后续全部对齐安全。
    def _dedup_trading_index(df):
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return df
        d = df.copy()
        d.index = pd.to_datetime(d.index)
        d = d[~d.index.duplicated(keep="last")].sort_index()
        return d
    prices = _dedup_trading_index(prices)
    macro = _dedup_trading_index(macro)
    # 取对数前统一清洗：价格与必须为正的汇率(usdjpy)列，把非有限/0/负值脏数据置为 NaN，
    # 交给下游 ffill/缺失处理，杜绝 np.log 对 0/负值报 "invalid value encountered in log"
    # 并产生 -inf 污染特征（不删除交易日、不补值，仅把非法价格点判为缺失，符合不造假约束）。
    prices = prices.apply(
        lambda s: pd.to_numeric(s, errors="coerce").where(
            lambda x: np.isfinite(x.astype(float)) & (x > 0)))
    if "usdjpy" in macro.columns:
        _jpy = pd.to_numeric(macro["usdjpy"], errors="coerce")
        macro = macro.copy()
        macro["usdjpy"] = _jpy.where(np.isfinite(_jpy.astype(float)) & (_jpy > 0))
    # 交易时段对齐：prices/macro 在本函数作用域内替换为"目标视角下时点无泄漏"的面板
    if use_session_align:
        prices, macro = align_exogenous_to_target(prices, macro, target)
    # 同一真实时刻融合（仅上海原油）：统计主节点为上海【夜盘 02:30 收盘】（上海连续交易的真正
    # 收盘点；周五夜盘物理发生在周六凌晨，已在采集层归属到周五交易日）。在有分时的窗口：
    # ①上海自身收盘价用 02:30 价（而非日盘 15:00 日 K）；②海外 peer 用北京 02:30（=美东
    # 14:30/伦敦 19:30）电子盘同刻价替换日 K as-of 值，使跨市场相对强弱/美元升贴水/地缘溢价
    # 都基于同一真实时刻；更早无分时的日期维持日频 as-of。全部 backward 用已成交价、无未来。
    intraday_used = 0
    if intraday_session is not None and target == "shanghai_crude" and not intraday_session.empty:
        idp = intraday_session.copy()
        idp.index = pd.to_datetime(idp.index).normalize()
        idp = idp.reindex(prices.index)
        if target in idp.columns:                       # ① 上海自身：02:30 夜盘收盘覆盖日盘收盘
            m0 = idp[target].notna()
            prices.loc[m0, target] = idp.loc[m0, target]
            intraday_used = int(m0.sum())
        for peer in ("brent", "wti"):                   # ② 海外 peer：同一 02:30 真实时刻价
            if peer in prices.columns and peer in idp.columns:
                mask = idp[peer].notna()
                prices.loc[mask, peer] = idp.loc[mask, peer]
    idx = prices.index
    feats = pd.DataFrame(index=idx)
    feats.attrs["intraday_synchronized_days"] = intraday_used

    # ---------- 目标价格技术面（仅允许节假日级 3 工作日短填充）----------
    log_t = np.log(prices[target].ffill(limit=3))
    daily_ret = log_t.diff(1)
    feats["ret_1d"] = daily_ret
    feats["ret_5d"] = log_t.diff(5)
    feats["ma_gap"] = log_t - log_t.rolling(20, min_periods=5).mean()
    feats["vol_20"] = daily_ret.rolling(20, min_periods=5).std()
    # 多周期时序动量与中期趋势/波动状态：均为 t 时点可得（无未来信息）。日度方向近随机，
    # 但严格无泄漏回测显示近 1 月(mom_21)趋势在月/季尺度存在"胜率不高、盈亏比>1"的正期望，
    # 由模型层独立的"趋势期望"门控通道评估后才启用；mom_63/中期均线用于确认趋势、过滤假突破。
    feats["mom_21"] = log_t.diff(21)
    feats["mom_63"] = log_t.diff(63)
    feats["ma_gap_120"] = log_t - log_t.rolling(120, min_periods=30).mean()
    feats["vol_60"] = daily_ret.rolling(60, min_periods=10).std()

    # 重大事件 regime（新冠疫情、俄乌战争、红海危机、2026 美以伊战事等）：可审计的离散
    # 状态变量，窗口内取事件强度、否则 0，年表见 data/reference/major_events.yaml（公开事实
    # 日期+来源，缺失不造假）。滚动重训使其权重随局势动态调整。注：严格无泄漏回测表明，堆砌
    # 更多连续技术指标对 10 日方向无增量，故只保留这一列可解释的事件状态锚，点预测幅度再由
    # 模型层的样本外 β 校准统一控制。
    feats["major_event_regime"] = major_event_regime(idx)

    # 宏观列在采集层已完成节假日短填充与月频 vintage 对齐，这里不再无限 ffill
    m = macro.reindex(idx)

    feats["dxy_chg_5d_dir"] = -m["dxy"].pct_change(5, fill_method=None) * 100 if "dxy" in m else np.nan
    feats["us10y_chg_5d_dir"] = -m["us10y"].diff(5) if "us10y" in m else np.nan
    feats["cpi_surprise_dir"] = -_zscore(m["cpi_yoy"]) if "cpi_yoy" in m else np.nan
    feats["nfp_surprise_dir"] = _zscore(m["nonfarm_surprise"]) if "nonfarm_surprise" in m else np.nan
    # 非农超强劲→紧缩利空，方向取负
    feats["nfp_surprise_dir"] = -feats["nfp_surprise_dir"]
    if "fed_expectation" in m:
        feats["fed_exp_level"] = m["fed_expectation"]
        feats["fed_exp_chg"] = m["fed_expectation"].diff(5)
    else:
        feats["fed_exp_level"] = np.nan
        feats["fed_exp_chg"] = np.nan
    feats["demand_chg_5d"] = m["demand_proxy"].diff(5) if "demand_proxy" in m else np.nan
    # 日元汇率（USDJPY，日元/美元）：作为外生宏观变量，方向不预设、交由各品种模型
    # 自行学习其敏感度差异；水平用对数 z、动量用 5 日变化率
    if "usdjpy" in m:
        feats["usdjpy_z"] = _zscore(np.log(m["usdjpy"]))
        feats["usdjpy_chg_5d"] = m["usdjpy"].pct_change(5, fill_method=None) * 100
    else:
        feats["usdjpy_z"] = np.nan
        feats["usdjpy_chg_5d"] = np.nan

    # GPR：真实指数优先；不可达时用【真实事件】构造代理；两者皆无则保持缺失
    gpr = m["gpr_index"] if "gpr_index" in m else None
    if gpr is None or gpr.notna().sum() < 30:
        ev_proxy = aggregate_events(events, idx, source_available=events_available).sum(axis=1)
        if events_available:
            gpr = 100 + ev_proxy * 20
        else:
            gpr = pd.Series(np.nan, index=idx)
    # 陈旧检测：GPR 为月频且可能停更（如 2026 年未更新）。若最近 60 个交易日内近乎常数，
    # 说明它对当前局势已无信息量，滚动 zscore 会退化成极端常数尾巴，反而误导模型——
    # 此时把 GPR 两列判为缺失（不拿过期常数冒充当前地缘信号），地缘改由日频市场代理承担。
    def _stale_mask(s: pd.Series, win: int = 60) -> pd.Series:
        recent_nunique = s.rolling(win, min_periods=10).apply(
            lambda x: pd.Series(x).nunique(), raw=False)
        return (recent_nunique < 3)
    gpr_z = _zscore(gpr)
    gpr_5 = gpr.diff(5) / 50
    stale = _stale_mask(gpr)
    feats["gpr_level_z"] = gpr_z.mask(stale)
    feats["gpr_chg_5d"] = gpr_5.mask(stale)

    # 地缘供给溢价（日频、市场真实交易、海外可达，不依赖易被限流的文本源）：
    # 布伦特-WTI 对数价差及其 5 日变化。中东/霍尔木兹断供冲击海运中质油（Brent 锚），
    # 而 WTI 锚定美国本土页岩、自给度高，故冲突升级时 Brent-WTI 价差走阔。
    # 上海 INE SC 可交割标的为中东含硫油、中国是伊朗/中东油主要买家，对此价差最敏感。
    if "brent" in prices and "wti" in prices:
        lb = np.log(prices["brent"].reindex(idx).ffill(limit=3))
        lw = np.log(prices["wti"].reindex(idx).ffill(limit=3))
        bw = (lb - lw)
        feats["geo_premium_chg5"] = bw.diff(5)
    else:
        feats["geo_premium_chg5"] = pd.Series(np.nan, index=idx)

    ev_agg = aggregate_events(events, idx, source_available=events_available)
    for col in ev_agg.columns:
        feats[col] = ev_agg[col]
    # 霍尔木兹/伊朗对华供油链路时变风险（真实标题驱动；各品种自学其权重）
    feats["china_supply_risk_5d"] = china_supply_risk(
        events, idx, source_available=events_available, window=5)
    feats["inst_bias"] = institutional_bias(views, idx, source_available=views_available)
    if "ev_view_5d" not in feats:
        feats["ev_view_5d"] = ev_agg.get("ev_view_5d",
                                         pd.Series(np.nan if not views_available else 0.0, index=idx))

    # 跨品种相对强弱 / 美元口径升贴水（对布伦特、WTI），固定 8 列，缺数据留 NaN
    usdcny = m["usdcny"] if "usdcny" in m else None
    cross = cross_market_features(prices, target, usdcny=usdcny)
    for c in cross.columns:
        feats[c] = cross[c]

    # 只做 3 工作日短填充（衔接节假日），长缺口保留 NaN，绝不填 0 冒充中性
    feats = feats.replace([np.inf, -np.inf], np.nan).ffill(limit=3)
    return feats


def factor_availability(features: pd.DataFrame) -> Dict[str, bool]:
    """逐因素判断真实数据覆盖率是否达标（整列缺失的因素不参与建模/权重）。"""
    avail = {}
    n = len(features)
    for factor, cols in FACTOR_GROUPS.items():
        if factor.startswith("_"):   # 技术面、跨品种等内部组不参与九大因素权重
            continue
        present = [c for c in cols if c in features.columns]
        if not present:
            avail[factor] = False
            continue
        cov = features[present].notna().any(axis=1).mean() if n else 0.0
        avail[factor] = bool(cov >= MIN_FACTOR_COVERAGE)
    return avail


TECHNICAL_COLS = ["ret_1d", "ret_5d", "ma_gap", "vol_20", "mom_21", "mom_63",
                  "ma_gap_120", "vol_60", "major_event_regime"]


def make_supervised(features: pd.DataFrame, target_price: pd.Series,
                    horizon: int) -> Tuple[pd.DataFrame, pd.Series]:
    """构造 (X_t, y_t)：用 t 日特征预测未来 horizon 日累计对数收益。

    样本有效性只要求【标签】与【技术面特征】真实非空；外生因素列允许 NaN——
    HistGradientBoosting 原生处理缺失，这样某因素暂时缺数据不会浪费其余真实样本，
    也不会靠填零制造样本。
    """
    fwd_ret = np.log(target_price).shift(-horizon) - np.log(target_price)
    y = fwd_ret.loc[features.index]
    tech = [c for c in TECHNICAL_COLS if c in features.columns]
    valid = (y.replace([np.inf, -np.inf], np.nan).notna()
             & features[tech].notna().all(axis=1))
    return features.loc[valid], y.loc[valid]


def all_feature_columns() -> list:
    cols = []
    for v in FACTOR_GROUPS.values():
        cols.extend(v)
    return cols
