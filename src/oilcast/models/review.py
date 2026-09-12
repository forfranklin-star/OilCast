"""预测复盘（learning loop 的"复测"环节）。

每日运行时，把【过去做出、且目标日已经有真实价格】的历史预测从 forecasts 表取出，
与现已实现的真实价格逐条对比，量化：
  - 价格误差 |预测-实际|/实际；
  - 方向命中（预测涨跌与实际涨跌是否一致）；
  - 95% 区间覆盖（真实价是否落在预测的 [q05,q95] 内）。
只使用真实价格，目标日尚无真实观测的预测不参与评估，绝不用任何填充值充当实际。
结果用于向用户证明"模型在被持续复测"，并作为自适应迭代是否有效的客观依据。
"""
from __future__ import annotations
import sqlite3
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from ..config import get_config
from ..utils import get_logger
LOG = get_logger(__name__)
def _nearest_real(price: pd.Series, day: pd.Timestamp) -> Optional[float]:
    """取 day 当日真实价；当日非交易日/缺失则向前找最近的真实观测（不向后偷看未来）。"""
    if price is None or price.empty:
        return None
    s = price.dropna()
    s = s[s.index <= pd.Timestamp(day)]
    return float(s.iloc[-1]) if not s.empty else None
def review_predictions(prices: Dict[str, pd.Series], as_of: pd.Timestamp,
                       max_rows: int = 12) -> dict:
    """读取历史落库预测并与已实现真实价格对比。返回汇总指标 + 最近明细。"""
    cfg = get_config()
    conn = sqlite3.connect(cfg["storage"]["sqlite_path"])
    try:
        df = pd.read_sql_query("SELECT * FROM forecasts ORDER BY report_date, target_date", conn)
    finally:
        conn.close()
    if df.empty:
        return {"available": False, "reason": "尚无历史落库预测，运行次日起开始累积复盘样本"}
    rows: List[dict] = []
    for _, r in df.iterrows():
        inst = r["instrument"]
        if inst not in prices:
            continue
        target = pd.Timestamp(r["target_date"])
        # 只复测目标日已经到来、且已有真实价格的预测
        last_real = prices[inst].dropna().index.max()
        if pd.isna(target) or target > last_real:
            continue
        p0 = _nearest_real(prices[inst], pd.Timestamp(r["report_date"]))
        actual = _nearest_real(prices[inst], target)
        mean = r["mean"]
        if p0 is None or actual is None or mean is None or pd.isna(mean) or p0 <= 0:
            continue
        pred_ret = mean / p0 - 1.0
        actual_ret = actual / p0 - 1.0
        q05, q95 = r.get("q05"), r.get("q95")
        covered = None
        if q05 is not None and q95 is not None and not pd.isna(q05) and not pd.isna(q95):
            covered = bool(q05 <= actual <= q95)
        rows.append({
            "report_date": r["report_date"], "target_date": target.strftime("%Y-%m-%d"),
            "horizon": r["horizon"], "instrument": inst,
            "base": round(p0, 2), "pred": round(float(mean), 2), "actual": round(actual, 2),
            "pred_ret_pct": round(pred_ret * 100, 2), "actual_ret_pct": round(actual_ret * 100, 2),
            "abs_err_pct": round(abs(float(mean) - actual) / actual * 100, 2),
            "dir_hit": bool(np.sign(pred_ret) == np.sign(actual_ret)) if actual_ret != 0 else None,
            "covered": covered,
        })
    if not rows:
        return {"available": False, "reason": "历史预测的目标日尚无已实现真实价格，暂无可复盘样本"}
    d = pd.DataFrame(rows)
    by_horizon: Dict[str, dict] = {}
    for hz, g in d.groupby("horizon"):
        cov = g["covered"].dropna().astype(float)
        dh = g["dir_hit"].dropna().astype(float)   # astype(float)：避免 numpy.bool_ 与 None 混合时 object 列 mean 算错
        by_horizon[hz] = {
            "n": int(len(g)),
            "mae_pct": round(float(g["abs_err_pct"].mean()), 2),
            "dir_acc": round(float(dh.mean()) * 100, 1) if len(dh) else None,
            "coverage95": round(float(cov.mean()) * 100, 1) if len(cov) else None,
        }
    overall_dir = d["dir_hit"].dropna().astype(float)
    overall_cov = d["covered"].dropna().astype(float)
    summary = {
        "n": int(len(d)),
        "mae_pct": round(float(d["abs_err_pct"].mean()), 2),
        "dir_acc": round(float(overall_dir.mean()) * 100, 1) if len(overall_dir) else None,
        "coverage95": round(float(overall_cov.mean()) * 100, 1) if len(overall_cov) else None,
        "by_horizon": by_horizon,
    }
    detail = d.sort_values(["report_date", "target_date"], ascending=False).head(max_rows)
    detail = detail.sort_values("report_date")
    return {"available": True, "summary": summary, "detail": detail.to_dict("records")}
