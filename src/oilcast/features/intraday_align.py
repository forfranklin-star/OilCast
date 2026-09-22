"""盘中分时 → 跨市场"同一真实时刻"对齐面板。

与 engineering.align_exogenous_to_target（日 K 收盘 as-of）的区别：
日 K 只有一根、标签是交易日，无法体现上海原油【夜盘(北京 21:00-次日 02:30)与欧美电子盘
重叠】这一事实；本模块直接用带北京时间戳的分时 K，在固定锚点时刻用 merge_asof 取各品种
当时已经成交的最近一根价格，得到严格同时刻横截面——不依赖开盘/收盘标签、不做"整体滞后一天"
的机械假设。

两个锚点（均按上海 INE 交易日 D 归属，normalize 后日期即 D）：
  * day_1500 ：北京 15:00，上海日盘收盘时刻，取此刻海外电子盘最新价 → 与上海收盘价严格同时刻，
               用于上海原油模型的跨市场外生价、以及"上海换算美元后对布伦特同时刻升贴水"；
  * night_0230：北京 02:30，上海夜盘收盘时刻，欧美盘仍在交易，刻画夜盘内外盘同步/传导。

无泄漏：锚点只 backward 取该时刻【之前/当时】已成交的 K 线，容差 60 分钟（1H K 线），
锚点前 60 分钟无成交（该品种休市）则置 NaN，绝不向前取未来、也不拿数小时前陈旧价硬桥。
免费分时源历史有限，面板只在有分时的窗口有值，更早自然为 NaN（绝不回填造数）。
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

ANCHORS = {
    "day_1500": (15, 0),       # 上海日盘收盘同时刻
    "night_0230": (2, 30),     # 上海夜盘收盘同时刻
}
_TOL = pd.Timedelta(minutes=60)   # 1H K 线容差：只接受锚点前 60 分钟内的成交


def intraday_close_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """分时长表(symbol,ts,close,...) → 宽表(index=北京时 ts, columns=symbol 的 close)。"""
    if long_df is None or long_df.empty:
        return pd.DataFrame()
    d = long_df[["symbol", "ts", "close"]].copy()
    d["ts"] = pd.to_datetime(d["ts"])
    wide = d.pivot_table(index="ts", columns="symbol", values="close", aggfunc="last")
    return wide.sort_index()


def _anchor_one_col(s: pd.Series, grid: pd.DatetimeIndex, tol: pd.Timedelta) -> pd.Series:
    """把一根分时序列 backward as-of 到锚点 grid（真实时间容差 tol），返回以锚点当日为索引。"""
    s = s.dropna().sort_index()
    if s.empty:
        return pd.Series(np.nan, index=grid.normalize(), dtype=float)
    left = pd.DataFrame({"_t": grid}).sort_values("_t")
    right = pd.DataFrame({"_v": s.to_numpy()}, index=pd.DatetimeIndex(s.index)).sort_index()
    m = pd.merge_asof(left, right, left_on="_t", right_index=True,
                      direction="backward", tolerance=tol)
    out = pd.Series(m["_v"].to_numpy(), index=pd.DatetimeIndex(m["_t"]).normalize(), dtype=float)
    out = out[~out.index.duplicated(keep="last")]
    return out


def anchor_cross_section(wide: pd.DataFrame, hour: int, minute: int,
                         tol: pd.Timedelta = _TOL) -> pd.DataFrame:
    """对每个自然日的 (hour:minute) 锚点，取各品种同时刻（backward 容差内）最近成交价。"""
    if wide.empty:
        return pd.DataFrame()
    days = pd.DatetimeIndex(wide.index.normalize().unique()).sort_values()
    grid = pd.DatetimeIndex([d + pd.Timedelta(hours=hour, minutes=minute) for d in days])
    cols = {c: _anchor_one_col(wide[c], grid, tol) for c in wide.columns}
    panel = pd.DataFrame(cols)
    panel.index.name = "date"
    return panel


def synchronized_panels(long_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """返回 {锚点名: 同时刻宽表(index=上海交易日, columns=品种)}。"""
    wide = intraday_close_wide(long_df)
    out = {}
    for name, (h, mi) in ANCHORS.items():
        out[name] = anchor_cross_section(wide, h, mi)
    return out


def roll_adjust_panel_sc(panel: pd.DataFrame, sc_factor: Optional[pd.Series]) -> pd.DataFrame:
    """对同时刻面板中的上海原油（未复权 SC0 连续）按日 K 主力复权因子做换月校正。

    复权因子按交易日对齐（最新日=1），消除盘中序列在换月日的虚假跳空，使盘中上海与日 K
    换月调整连续同一口径；面板缺失或无因子时原样返回，绝不造数。
    """
    if panel is None or panel.empty or sc_factor is None or "shanghai_crude" not in panel:
        return panel
    p = panel.copy()
    p.index = pd.to_datetime(p.index).normalize()
    f = pd.Series(sc_factor).copy()
    f.index = pd.to_datetime(f.index).normalize()
    ff = f.reindex(p.index)
    ok = ff.notna() & p["shanghai_crude"].notna()
    if ok.any():
        p.loc[ok, "shanghai_crude"] = p.loc[ok, "shanghai_crude"] * ff.loc[ok]
    return p


def apply_anchor_panel_to_prices(prices: pd.DataFrame, panel: pd.DataFrame,
                                 sc_factor: Optional[pd.Series] = None) -> pd.DataFrame:
    """把同一真实时刻的锚点面板（如北京 02:30 夜盘收盘）覆盖到日频价格表。

    关键：面板含【全部品种】，覆盖后五品种在分时覆盖窗口内统一到同一统计时刻（此前只更新
    上海、外盘仍用各自美盘日收盘，导致变化率不可比）。若上海面板仍是未复权 SC0，可传入日 K
    持仓量主力重建的比例复权因子 ``sc_factor``（按交易日）在覆盖前校正换月跳空；调用方若已
    用 :func:`roll_adjust_panel_sc` 复权过面板则传 None。仅覆盖有真实成交的格子，缺失保持 NaN。
    返回更新后的副本，不改写传入表；原始日 K 已在数据库另行落库保留。
    """
    out = prices.copy()
    if panel is None or panel.empty:
        return out
    p = roll_adjust_panel_sc(panel, sc_factor)
    p.index = pd.to_datetime(p.index).normalize()
    p = p.reindex(out.index)
    for c in [c for c in out.columns if c in p.columns]:
        m = p[c].notna()
        if m.any():
            out.loc[m, c] = p.loc[m, c]
    return out


def synchronized_day_panel(long_df: pd.DataFrame) -> pd.DataFrame:
    """日盘收盘(15:00)同时刻面板——上海原油跨市场对齐主口径。"""
    return synchronized_panels(long_df).get("day_1500", pd.DataFrame())


def sc_brent_premium_usd(day_panel: pd.DataFrame, usdcny: Optional[pd.Series]) -> pd.DataFrame:
    """日盘收盘同时刻的上海原油(换算美元/桶) 对 布伦特升贴水（美元/桶）及二者美元对数价。

    需要同时刻的上海(人民币/桶)、布伦特(美元/桶)与当日已知 USDCNY；任一缺失则该行留 NaN。
    仅做无量纲/同单位比较，不与成品油混用。返回列 sc_usd, brent_usd, premium_usd,
    sc_brent_logspread（与日频特征同名口径，便于融合）。

    冷启动健壮性：无论面板是否为空、是否缺上海/布伦特列、汇率是否就绪，都【始终返回这 4 列】，
    缺数据的列为全 NaN（绝不造数）。否则下游 dropna(subset=["premium_usd"])/loc 取列会 KeyError
    （CI 首次运行分时历史几乎为空时必现）。"""
    out_cols = ["brent_usd", "sc_usd", "premium_usd", "sc_brent_logspread"]
    idx = getattr(day_panel, "index", None)
    out = pd.DataFrame(index=idx)
    for c in out_cols:
        out[c] = np.nan
    if day_panel is None or day_panel.empty:
        return out
    if "shanghai_crude" not in day_panel or "brent" not in day_panel:
        # 布伦特列若存在仍如实落 brent_usd，上海/升贴水留 NaN
        if "brent" in day_panel:
            out["brent_usd"] = day_panel["brent"]
        return out
    fx = None
    if usdcny is not None and pd.Series(usdcny).notna().any():
        fx = pd.Series(usdcny).copy()
        fx.index = pd.to_datetime(fx.index).normalize()
        fx = fx[~fx.index.duplicated(keep="last")].sort_index().dropna()
        # FRED 在岸定盘(DEXCHUS)常滞后数日：按【自然日】向前桥接 ≤7 天（汇率短期极稳，
        # 对美元/桶升贴水影响可忽略），超过 7 天仍无新定则留 NaN（不拿陈旧值硬凑、不造数）。
        daily = pd.date_range(fx.index.min(), day_panel.index.max(), freq="D")
        fx = fx.reindex(daily).ffill(limit=7).reindex(day_panel.index)
    sc_cny = day_panel["shanghai_crude"]
    out["brent_usd"] = day_panel["brent"]
    if fx is not None and fx.notna().any():
        out["sc_usd"] = sc_cny / fx
        out["premium_usd"] = out["sc_usd"] - out["brent_usd"]
        out["sc_brent_logspread"] = np.log(out["sc_usd"]) - np.log(out["brent_usd"])
    return out
