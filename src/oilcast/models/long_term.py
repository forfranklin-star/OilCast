"""长期（12 个月）预测：情景分析 + 蒙特卡洛。

1. 三情景（高/中/低油价）的年化漂移与波动来自 config；
2. 情景先验概率用**当前多因素最新读数**做 softmax 调整：
   地缘/供给压力大、美元走弱、降息预期升温 → 高油价情景概率上升；
3. 按后验情景概率混合抽样几何布朗运动路径，得到长期分布、
   关键节点（1/3/6/12 月）预测与涨跌概率；
4. 机构目标价中位数作为外部锚，与模型分布并列展示（不强行混合）。
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..config import get_config
from ..utils import seeded_rng


def scenario_probabilities(latest_features: pd.Series,
                           temperature: float = 1.4,
                           available: Optional[set] = None) -> Dict[str, float]:
    """情景后验概率。available 指定真实可用的细特征；缺失因素不投票，
    其权重在可用特征之间重新归一化（不把缺失当作中性 0 悄悄稀释信号）。"""
    cfg = get_config()
    sc = cfg["scenarios"]
    # (细特征, 原始权重, 缩放)
    terms = [("gpr_level_z", 0.35, 1.0), ("ev_supply_5d", 0.30, 1.0),
             ("dxy_chg_5d_dir", 0.20, 0.5), ("fed_exp_level", 0.20, 1.0),
             ("demand_chg_5d", 0.25, 1.0), ("inst_bias", 0.20, 1.0)]
    weighted = 0.0
    wsum = 0.0
    for col, w, scale in terms:
        val = latest_features.get(col, np.nan)
        if available is not None and col not in available:
            continue
        if val is None or pd.isna(val):
            continue
        weighted += w * float(val) * scale
        wsum += w
    score = float(np.clip(weighted / wsum if wsum else 0.0, -2, 2))
    raw = {
        "bull": float(sc["bull"]["prob_prior"]) * np.exp(temperature * score / 2),
        "base": float(sc["base"]["prob_prior"]),
        "bear": float(sc["bear"]["prob_prior"]) * np.exp(-temperature * score / 2),
    }
    z = sum(raw.values())
    out = {k: round(v / z, 3) for k, v in raw.items()}
    # 消除四舍五入残差，保证概率严格求和为 1（补到概率最大的情景）
    resid = round(1 - sum(out.values()), 3)
    if abs(resid) >= 1e-9:
        k_max = max(out, key=out.get)
        out[k_max] = round(out[k_max] + resid, 3)
    return out


class LongTermMonteCarlo:
    def fit(self, last_price: float):
        self.last_price = float(last_price)
        return self

    def predict(self, future_dates: pd.DatetimeIndex,
                scenario_probs: Dict[str, float],
                n_sims: Optional[int] = None,
                institution_anchor: Optional[float] = None) -> dict:
        cfg = get_config()
        n_sims = n_sims or int(cfg["model"]["n_monte_carlo"])
        steps = len(future_dates)
        rng = seeded_rng()
        sc = cfg["scenarios"]
        names = list(scenario_probs)
        probs = np.array([scenario_probs[n] for n in names], dtype=float)
        probs = probs / probs.sum()

        mu_d = np.array([np.log1p(float(sc[n]["drift_annual"])) / 252 for n in names])
        sig_d = np.array([float(sc[n]["vol_annual"]) / np.sqrt(252) for n in names])
        pick = rng.choice(len(names), size=n_sims, p=probs)
        mu = mu_d[pick][:, None]
        sig = sig_d[pick][:, None]

        shocks = rng.normal(0, 1, (n_sims, steps))
        log_paths = np.cumsum((mu - 0.5 * sig ** 2) + sig * shocks, axis=1)
        paths = self.last_price * np.exp(log_paths)

        q = np.quantile(paths, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
        mean_path = paths.mean(axis=0)
        path = pd.DataFrame({
            "mean": mean_path, "q05": q[0], "q25": q[1], "q50": q[2],
            "q75": q[3], "q95": q[4]}, index=future_dates)

        end = paths[:, -1]
        prob_up = float((end > self.last_price).mean())
        # 关键节点
        checkpoints = {}
        for label, off in (("1个月", 21), ("3个月", 63), ("6个月", 126), ("12个月", steps)):
            j = min(off, steps) - 1
            checkpoints[label] = {
                "mean": round(float(paths[:, j].mean()), 2),
                "q05": round(float(np.quantile(paths[:, j], 0.05)), 2),
                "q95": round(float(np.quantile(paths[:, j], 0.95)), 2),
            }
        endpoint = {
            "target_date": pd.Timestamp(future_dates[-1]).strftime("%Y-%m-%d"),
            "mean": round(float(mean_path[-1]), 2),
            "q05": round(float(q[0][-1]), 2), "q25": round(float(q[1][-1]), 2),
            "q75": round(float(q[3][-1]), 2), "q95": round(float(q[4][-1]), 2),
            "pct_mean": round((float(mean_path[-1]) / self.last_price - 1) * 100, 2),
            "prob_up": round(prob_up, 3), "prob_down": round(1 - prob_up, 3),
            "checkpoints": checkpoints,
            "scenario_probs": scenario_probs,
            "institution_anchor": institution_anchor,
        }
        return {"path": path, "endpoint": endpoint}
