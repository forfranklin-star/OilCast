"""因素权重学习。

三路融合，兼顾数据驱动与可解释稳定性：
  1. RandomForest 特征重要性（非线性、捕捉交互）；
  2. LASSO 标准化系数绝对值（线性、稀疏，抑制噪声特征）；
  3. config.prior_weights 人工先验，做贝叶斯式收缩；
再与上一轮权重做指数平滑（EMA），实现"每天追加真实数据后渐进调整"。

数据原则：整列缺失的因素（available=False）不参与任何归一化，权重记为 NaN，
绝不用 0 冒充"该因素无影响"；先验也只在可用因素之间重新归一化。
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

from ..config import get_config
from ..features.engineering import FACTOR_GROUPS
from ..utils import get_logger

LOG = get_logger(__name__)
MIN_TRAIN_ROWS = 120          # 少于该真实样本量不训练数据驱动权重，退回先验


def _feature_importance(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """RF 重要性 与 LASSO |系数| 等权平均，返回归一化细特征重要性。

    整列缺失的因素已在调用前按可用性剔除；剩余零星 NaN（节假日级缺口）仅在
    模型拟合时用【训练集自身中位数】填补——这是标准统计插补，不引入任何外部
    或合成数据，也不影响报告层对缺失的如实呈现。
    """
    from sklearn.impute import SimpleImputer
    X = X.tail(1500)
    y = y.loc[X.index]
    # 丢弃整列缺失（双保险），其余用训练集中位数插补
    X = X.loc[:, X.notna().any(axis=0)]
    Xs = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X))
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                               max_depth=6, n_jobs=-1, random_state=42)
    rf.fit(Xs, y)
    imp_rf = pd.Series(rf.feature_importances_, index=X.columns)
    lasso = LassoCV(cv=5, random_state=42, max_iter=5000)
    lasso.fit(Xs, y)
    imp_lasso = pd.Series(np.abs(lasso.coef_), index=X.columns)
    if imp_lasso.sum() < 1e-8:
        imp_lasso = pd.Series(1 / X.shape[1], index=X.columns)
    return 0.6 * imp_rf / imp_rf.sum() + 0.4 * imp_lasso / imp_lasso.sum()


def aggregate_to_factors(feature_imp: pd.Series, available: Dict[str, bool]) -> pd.Series:
    """细特征重要性聚合到因素；只统计可用因素并在其中归一化。"""
    scores = {}
    for factor, cols in FACTOR_GROUPS.items():
        if factor.startswith("_") or not available.get(factor, False):
            continue
        scores[factor] = float(sum(feature_imp.get(c, 0.0) for c in cols))
    s = pd.Series(scores, dtype=float)
    if len(s) == 0:
        return s
    if s.sum() < 1e-12:
        s[:] = 1 / len(s)
    return s / s.sum()


def learn_factor_weights(X: pd.DataFrame, y: pd.Series,
                         available: Optional[Dict[str, bool]] = None,
                         prev_weights: Optional[pd.Series] = None) -> pd.DataFrame:
    """主入口：返回 DataFrame[factor, weight, model_importance, prior, available]。

    weight 仅对 available=True 的因素计算且和为 1；不可用因素 weight=NaN。
    """
    cfg = get_config()
    all_factors = [f for f in cfg["prior_weights"].keys()]
    if available is None:
        available = {f: True for f in all_factors}
    usable = [f for f in all_factors if available.get(f, False)]

    # 先验只在可用因素间归一化
    priors_all = pd.Series(dict(cfg["prior_weights"]), dtype=float).reindex(all_factors)
    priors = priors_all[usable]
    priors = priors / priors.sum() if priors.sum() > 0 else pd.Series(1 / len(usable), index=usable)

    model_used = len(X) >= MIN_TRAIN_ROWS and len(usable) >= 2
    if model_used:
        # 剔除不可用因素对应的细特征列（缺失因素不参与重要性学习）
        drop_cols = []
        for f in all_factors:
            if not available.get(f, False):
                drop_cols += [c for c in FACTOR_GROUPS.get(f, []) if c in X.columns]
        feature_imp = _feature_importance(X.drop(columns=drop_cols, errors="ignore"), y)
        model_w = aggregate_to_factors(feature_imp, available).reindex(usable).fillna(0)
        model_w = model_w / model_w.sum() if model_w.sum() > 0 else priors
    else:
        LOG.warning("真实训练样本 %d 行或可用因素不足，权重退回人工先验", len(X))
        model_w = priors.copy()

    shrink = float(cfg["model"]["prior_shrinkage"])
    blended = shrink * model_w + (1 - shrink) * priors
    blended = blended / blended.sum()

    if prev_weights is not None:
        alpha = float(cfg["model"]["weight_ema_alpha"])
        prev = prev_weights.reindex(usable).dropna()
        if len(prev) == len(usable) and prev.sum() > 0:
            blended = alpha * prev / prev.sum() + (1 - alpha) * blended
            blended = blended / blended.sum()

    rows = []
    for f in all_factors:
        rows.append({
            "factor": f,
            "weight": round(float(blended[f]), 4) if f in blended.index else np.nan,
            "model_importance": round(float(model_w[f]), 4) if f in model_w.index else np.nan,
            "prior": round(float(priors_all[f]), 4) if pd.notna(priors_all.get(f)) else np.nan,
            "available": bool(available.get(f, False)),
        })
    out = pd.DataFrame(rows).sort_values("weight", ascending=False, na_position="last") \
                            .reset_index(drop=True)
    # 不可用因素（无真实信源）内部用 NaN 排序到末尾，但输出为 None → JSON null，
    # 前端显示"不可用/—"，绝不能把裸 NaN 写进报告显示成 "NaN%"
    for c in ("weight", "model_importance", "prior"):
        out[c] = out[c].astype(object).where(out[c].notna(), None)
    top = out.dropna(subset=["weight"])
    if len(top):
        LOG.info("权重更新（模型%s），主导因素：%s",
                 "已训练" if model_used else "未训练/先验", top.iloc[0]["factor"])
    return out
