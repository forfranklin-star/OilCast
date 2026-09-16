"""中期（三个月 / 66 个交易日）预测：向量自回归 VAR。

内生变量（全部平稳化，且只纳入通过质量门、真实覆盖充分的字段）：
    dl        目标油价日对数收益（必选）
    dxy_chg   美元指数日变化 / u10y_chg / gpr_chg / fed_chg（按可用性动态纳入）
VAR 刻画油价与宏观变量的领先/滞后联动；预测区间用保留同期相关性的残差块
bootstrap 递归模拟，再按情景概率叠加年化漂移，得到分布与涨跌概率。

数据原则：不做任何跨缺口填充，dropna 后只在真实观测齐全的行上估计；
真实样本不足时抛 InsufficientData，由编排层标记 unavailable。
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from ..config import get_config
from ..utils import get_logger, seeded_rng
from .errors import InsufficientData

LOG = get_logger(__name__)

# 候选宏观内生列：(输出列名, macro 中的真实列名, 构造方式)
CANDIDATES = [
    ("dxy_chg", "dxy", "pct"),
    ("u10y_chg", "us10y", "diff"),
    ("gpr_chg", "gpr_index", "diff50"),
    ("fed_chg", "fed_expectation", "diff"),
]


class MidTermVAR:
    def fit(self, prices: pd.DataFrame, macro: pd.DataFrame, target: str = "wti",
            max_lags: int = 4, coverage: float = 0.5):
        from statsmodels.tsa.api import VAR
        df = pd.DataFrame(index=prices.index)
        df["dl"] = np.log(prices[target].ffill(limit=3)).diff()
        m = macro.reindex(prices.index)
        self.used_cols = ["dl"]
        for out_col, src_col, kind in CANDIDATES:
            if src_col not in m.columns or m[src_col].notna().mean() < coverage:
                continue
            if kind == "pct":
                df[out_col] = m[src_col].pct_change(fill_method=None)
            elif kind == "diff50":
                df[out_col] = m[src_col].diff() / 50
            else:
                df[out_col] = m[src_col].diff()
            self.used_cols.append(out_col)

        df = df.dropna()      # 只用所有纳入变量同时真实存在的行，不填充
        if df.shape[1] < 2:
            raise InsufficientData(
                f"VAR 除油价自身外无其他真实覆盖充分的宏观内生变量（纳入列 {self.used_cols}），拒绝训练")
        min_rows = int(get_config()["model"].get("min_train_obs", 250))
        if len(df) < min_rows:
            raise InsufficientData(
                f"VAR 真实对齐样本仅 {len(df)} 行（纳入列 {self.used_cols}），"
                f"少于 {min_rows} 行门槛，拒绝训练")
        lo, hi = df.quantile(0.005), df.quantile(0.995)
        self.data = df.clip(lower=lo, upper=hi, axis=1)
        self.last_price = float(prices[target].dropna().iloc[-1])

        try:
            sel = VAR(self.data).select_order(maxlags=max_lags)
            cand = int(np.clip(sel.aic, 1, max_lags))
        except Exception:
            cand = 2
        lags, chosen = cand, None
        while lags >= 1:
            fitted = VAR(self.data).fit(lags)
            if fitted.is_stable(verbose=False):
                chosen = fitted
                break
            lags -= 1
        self.model = chosen or VAR(self.data).fit(1)
        self.lags = max(lags, 1)
        self.resid = self.model.resid
        LOG.info("VAR(%s) lags=%d 样本=%d 稳定=%s 内生列=%s", target, self.lags,
                 len(self.data), self.model.is_stable(verbose=False), self.used_cols)
        return self

    def predict(self, future_dates: pd.DatetimeIndex,
                scenario_probs: Dict[str, float], scenarios_cfg,
                n_sims: int = 500) -> dict:
        steps = len(future_dates)
        rng = seeded_rng()
        k = len(self.used_cols)
        last_y = self.data.values[-self.lags:]
        coefs = self.model.coefs
        intercept = getattr(self.model, "intercept", np.zeros(k))
        resid = self.resid.values

        daily_drift = {name: np.log1p(float(s["drift_annual"])) / 252
                       for name, s in scenarios_cfg.items()}
        names = list(scenario_probs)
        probs = np.array([scenario_probs[n] for n in names], dtype=float)
        probs = probs / probs.sum()
        expected_drift = float((probs * np.array([daily_drift[n] for n in names])).sum())
        draws = rng.choice(len(names), size=n_sims, p=probs)

        history = last_y.copy()
        mean_path = np.zeros(steps)
        for t in range(steps):
            yhat = intercept.copy()
            for L in range(self.lags):
                yhat += coefs[L] @ history[-(L + 1)]
            yhat[0] += expected_drift
            history = np.vstack([history, yhat])
            mean_path[t] = yhat[0]
        mean_cum = np.cumsum(mean_path)

        sim_paths = np.zeros((n_sims, steps))
        for s in range(n_sims):
            hist = last_y.copy()
            drift = daily_drift[names[draws[s]]]
            cum = np.zeros(steps)
            for t in range(steps):
                yhat = intercept.copy()
                for L in range(self.lags):
                    yhat += coefs[L] @ hist[-(L + 1)]
                yhat = yhat + resid[rng.integers(0, len(resid))]
                yhat[0] += drift
                hist = np.vstack([hist, yhat])
                cum[t] = np.clip(cum[t - 1] + yhat[0] if t else yhat[0],
                                 np.log(0.3), np.log(3.5))
            sim_paths[s] = cum
        q = np.quantile(sim_paths, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
        path = pd.DataFrame({
            "mean": self.last_price * np.exp(mean_cum),
            "q05": self.last_price * np.exp(q[0]), "q25": self.last_price * np.exp(q[1]),
            "q50": self.last_price * np.exp(q[2]), "q75": self.last_price * np.exp(q[3]),
            "q95": self.last_price * np.exp(q[4]),
        }, index=future_dates)
        prob_up = float((sim_paths[:, -1] > 0).mean())
        endpoint = {
            "target_date": pd.Timestamp(future_dates[-1]).strftime("%Y-%m-%d"),
            "mean": round(float(path["mean"].iloc[-1]), 2),
            "q05": round(float(path["q05"].iloc[-1]), 2),
            "q25": round(float(path["q25"].iloc[-1]), 2),
            "q75": round(float(path["q75"].iloc[-1]), 2),
            "q95": round(float(path["q95"].iloc[-1]), 2),
            "pct_mean": round((float(path["mean"].iloc[-1]) / self.last_price - 1) * 100, 2),
            "prob_up": round(prob_up, 3), "prob_down": round(1 - prob_up, 3),
        }
        return {"path": path, "endpoint": endpoint}
