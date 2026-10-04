"""稳健概率校准：带先验收缩、近端/regime 自适应的方向概率估计量。

为什么需要它（替代裸 IsotonicRegression）：
严格无泄漏回测发现，方向分类器在**极端/稀疏概率区过度自信**——例如当前点分类器给出
raw≈0.97，但样本外历史上 raw≥0.92 的原点实际胜率仅约 0.43、近端窗口甚至 0 个样本。
裸 isotonic 在这种区域有三个问题：① 可依据的样本极少、估计高方差；② 用全部历史（含
多年前已失效 regime）；③ out_of_bounds="clip" 会把超出范围的 raw 直接贴到边界极端值。
结果是在不该自信时输出 0.98。

本估计量从数学性质上纠正（**不是外部改数字**）：
- **近端/regime 自适应**：每个样本按其距当前的交易日龄做指数衰减权重（半衰期 halflife），
  久远 regime 的证据权重自然变小；
- **经验贝叶斯收缩**：在待校准 raw 的局部窗内计算加权有效样本量 n_eff = W²/Σw²，再把
  单调 base（加权 isotonic）与无信息先验 0.5 按 n_eff 与先验强度 prior_n 加权：
      p = (n_eff·g(r) + prior_n·0.5) / (n_eff + prior_n)
  于是稀疏区（n_eff→0）估计量**自然回到 0.5**；证据充分（n_eff 大）才采信经验胜率。

部署（对未来点，用全部已实现 OOS 样本拟合，无泄漏）与 edge 评估（对历史点，仅用更早
样本拟合）使用**完全相同的估计量与权重口径**，保证概率与方向门控同源、结论一致。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression


class ShrinkageCalibrator:
    """加权 isotonic base + 有效样本量向 0.5 的经验贝叶斯收缩。

    rp/oos raw prob, ad/actual dir(0/1), ages/各样本距当前的交易日龄（None 表示不衰减）。
    """

    def __init__(self, rp: np.ndarray, ad: np.ndarray,
                 ages: Optional[np.ndarray] = None,
                 halflife: float = 120.0, prior_n: float = 12.0,
                 band: float = 0.13) -> None:
        rp = np.asarray(rp, dtype=float)
        ad = np.asarray(ad, dtype=float)
        m = np.isfinite(rp) & np.isfinite(ad)
        rp, ad = rp[m], ad[m]
        if ages is not None:
            ages = np.asarray(ages, dtype=float)[m]
            w = np.exp(-np.log(2.0) * np.maximum(ages, 0.0) / float(halflife))
        else:
            w = np.ones(len(rp), dtype=float)
        self.rp, self.ad, self.w = rp, ad, w
        self.prior_n = float(prior_n)
        self.band = float(band)
        self._iso: Optional[IsotonicRegression] = None
        if len(np.unique(rp)) >= 2 and len(np.unique(ad)) == 2:
            try:
                self._iso = IsotonicRegression(
                    out_of_bounds="clip", y_min=0.02, y_max=0.98).fit(
                    rp, ad, sample_weight=w)
            except Exception:
                self._iso = None

    def _base(self, r: float) -> float:
        """单调 base g(r)：加权 isotonic；信息不足时退局部加权均值，再退 0.5。"""
        if self._iso is not None:
            return float(np.clip(self._iso.predict([r])[0], 0.02, 0.98))
        m = np.abs(self.rp - r) <= self.band
        if self.w[m].sum() > 1e-9:
            return float(np.clip(np.average(self.ad[m], weights=self.w[m]),
                                 0.02, 0.98))
        return 0.5

    def predict(self, r) -> float:
        """校准单个 raw 概率；稀疏区由收缩自然回到 0.5。局部有效样本量用平滑 tri-cubic
        核加权（而非硬窗 0/1），避免窗边界样本进出造成的抖动。"""
        r = float(r)
        d = np.abs(self.rp - r) / self.band
        kern = np.where(d < 1.0, (1.0 - d ** 3) ** 3, 0.0)
        w = self.w * kern
        W = float(w.sum())
        if W <= 1e-9:
            return 0.5
        n_eff = (W * W) / float((w ** 2).sum())
        g = self._base(r)
        p = (n_eff * g + self.prior_n * 0.5) / (n_eff + self.prior_n)
        return float(np.clip(p, 0.02, 0.98))

    def predict_many(self, rs) -> np.ndarray:
        return np.array([self.predict(r) for r in rs], dtype=float)
