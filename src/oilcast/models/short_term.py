"""短期（两周 / 10 个交易日）预测。

主模型：Direct multi-step 梯度提升回归 —— 对每个预测步长 h 单独训练一个模型，
目标是未来 h 日累计对数收益（外生变量即特征工程的多因素矩阵）。
区间：滚动原点（rolling-origin）样本外残差的经验分位数，避免用 in-sample
残差造成的过度乐观。
基准：ARIMA(2,0,2) 拟合日对数收益，作为无外生变量的时间序列对照。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor

from ..config import get_config
from ..utils import get_logger
from .errors import InsufficientData

LOG = get_logger(__name__)
RESIDUAL_HORIZONS = (2, 5, 10)
# 仅对【整列全 NaN（非空个数为 0）】的特征列做常数 0 占位。
# 设计权衡（经真实 walk-forward 对照验证）：HistGradientBoosting 原生把 NaN 当作独立
# 缺失类别、走默认分裂方向，这在"事件/GPR 等列只有零星真实值"时反而是最优处理；若把
# 非空 1~19 的稀疏列也统一填成常数 0，会把"缺失"误标成数值 0 参与分裂，实测会让同一
# 时点的方向预测由正确翻为错误、显著拉低方向命中率。因此阈值取 1：有任意真实值就保留
# NaN 交模型原生处理，只有整列全空（新版 sklearn>=1.6/numpy2 对全 NaN 列分箱会抛
# "window shape cannot be larger than input array shape"）才占位保 schema、防崩溃。
MIN_FEATURE_NONNULL = 1


def neutralize_low_info(X: pd.DataFrame, min_non_null: int = MIN_FEATURE_NONNULL):
    """把【整列全 NaN】（非空个数 < min_non_null，默认即 0 个非空）的特征列填为常数 0，
    返回 (填充后的副本, 被占位列名)。

    同时实现三件事：① 规避新版 sklearn/numpy2 对【全 NaN 列】分箱崩溃；② 保持特征
    schema 跨期恒定（不删列），热启动/滚动重训不因某期源不可达而改变列集合；③ 被占位
    全空列无方差、树不会据其分裂，不携带信息、不构成假数据。仍有哪怕零星真实值的稀疏
    列保留其 NaN，交给 HistGradientBoosting 原生缺失处理（walk-forward 证明这样更准）。"""
    low = [c for c in X.columns if int(X[c].notna().sum()) < min_non_null]
    if not low:
        return X, []
    X = X.copy()
    X[low] = 0.0
    return X, low


# 向后兼容别名
neutralize_all_nan = neutralize_low_info


@dataclass
class PathResult:
    path: pd.DataFrame           # date, mean,q05,q25,q50,q75,q95
    endpoint: dict               # 期末均值/区间/涨跌概率
    benchmark_endpoint: Optional[dict] = None


class ShortTermForecaster:
    def __init__(self, horizon: int = 10, window: int = 180,
                 compute_residuals: bool = True,
                 calib_origins: Optional[int] = None,
                 calib_iter: Optional[int] = None,
                 run_arima: bool = True) -> None:
        self.horizon = horizon
        self.window = window
        self.compute_residuals = compute_residuals
        self.run_arima = run_arima   # 回测时可关闭耗时的 ARIMA 基准
        # 内部样本外校准规模的可选覆盖（None=读 config）；回测/测试可用小值提速
        self.calib_origins_override = calib_origins
        self.calib_iter_override = calib_iter
        self.models: Dict[int, HistGradientBoostingRegressor] = {}
        self.resid_quantiles: Dict[int, np.ndarray] = {}
        self.resid_std: Dict[int, float] = {}
        self.calib_beta: Dict[int, float] = {}   # 各步长样本外校准系数（无信号→0退守RW）
        # 跨期持续学习状态
        self.feature_cols = None
        self.warm_info = None          # 本期热启动所依据的上期信息（None=冷启动）
        self.cum_iters: Dict[int, int] = {}
        self.per_step_iter = 250       # 每个 direct 模型单轮新增迭代数
        self.arima_cumret = None

    # ------------------------------------------------------------- fit
    def fit(self, X: pd.DataFrame, price: pd.Series,
            warm: Optional["ShortTermForecaster"] = None) -> "ShortTermForecaster":
        mcfg = get_config()["model"]
        # 训练样本组织（严格无泄漏 walk-forward 回测结论）：
        #  - direct 预测模型只用最近 train_window 个交易日的【滚动窗】整体重拟合。金融变量
        #    关系非平稳，旧 regime 会稀释当前结构；这是"每天用最近约两年真实数据重估模型"
        #    的滚动在线学习，并非从零——原始数据库持续累积、残差/β 跨期 EMA 继承、工件可
        #    导入导出。注意：日频 10 日方向信噪比很低，严格无泄漏回测下任何模型方向命中都
        #    在 50% 附近，点预测幅度因此再交由样本外 β 校准统一收缩（无信号退守随机游走）。
        #  - max_train_rows 仅作为额外硬上限（默认 0=不再截断）。
        train_window = int(mcfg.get("train_window", 500) or 0)
        max_rows = int(mcfg.get("max_train_rows", 0) or 0)
        if max_rows > 0:
            X = X.tail(max_rows)
        # 价格只允许节假日级 3 工作日短填充，长缺口保持缺失（dropna 时自然剔除）
        price = price.loc[X.index].ffill(limit=3)
        # —— 固定特征 schema + 全空列剔除（不填 0）——
        # self.feature_cols 始终登记全部列（跨期稳定、供因素权重展示与热启动判定）；
        # 但实际入模只用【至少有一个真实值】的 active 列：整列全 NaN 的列直接剔除，既
        # 规避新版 sklearn(>=1.6)/numpy2 对全 NaN 列分箱抛 window shape 错，又避免把缺失
        # 填成常数 0 干扰梯度提升内部分箱（walk-forward 实测填 0 会让方向预测由对转错）。
        # 有零星真实值的稀疏列予以保留，其 NaN 交 HGB 原生缺失处理。
        self.feature_cols = list(X.columns)
        active_cols = [c for c in self.feature_cols if int(X[c].notna().sum()) >= 1]
        self.zero_info_cols = [c for c in self.feature_cols if c not in active_cols]
        self.active_cols = active_cols
        X = X[active_cols]
        if not active_cols:
            raise InsufficientData("无任何含真实值的特征列，拒绝训练")
        # 仅当上期模型同口径(horizon)、特征 schema 完全一致时才继承误差结构，否则冷启动
        usable_warm = None
        if warm is not None and getattr(warm, "horizon", None) == self.horizon and \
                list(getattr(warm, "feature_cols", []) or []) == self.feature_cols:
            usable_warm = warm
        self.warm_info = {
            "warm_started": usable_warm is not None,
            "zero_info_cols": self.zero_info_cols,
            "parent_train_end": getattr(usable_warm, "train_meta", {}).get("train_end")
            if usable_warm is not None else None,
            "parent_fitted_at": getattr(usable_warm, "train_meta", {}).get("fitted_at")
            if usable_warm is not None else None,
        }
        log_p = np.log(price)
        min_rows = int(get_config()["model"].get("min_train_obs", 250))
        valid_mask = log_p.notna()        # 价格真实有效即可训练；特征缺失由模型原生处理
        n_valid = int(valid_mask.sum())
        if n_valid < min_rows:
            raise InsufficientData(
                f"短期模型真实有效样本仅 {n_valid} 行，少于 {min_rows} 行门槛，拒绝训练")
        self.ret_std = float(log_p.diff().std())
        # direct 模型的滚动训练窗（0=用全部可用历史）；残差结构仍在完整历史上估计。
        # 关键：窗口必须按"标签可得的有效 (X,y) 配对"来取——即先在全序列上算未来 h 日
        # 标签、剔除末端 h 个无标签行，再取最近 win 行。绝不能先 tail(win) 再在窗内
        # shift(-h)，那样会把窗口末端 h 个【最新交易日】当无标签丢弃，使模型对最近 h 日
        # 失明、在拐点处方向系统性判反（实测同一时点预测符号因此翻转）。
        self.train_window = win = train_window if train_window > 0 else len(X)
        self.point_shrink = float(mcfg.get("point_shrink", 0.9))
        valid_idx = X.index[valid_mask]
        from ..utils import now_beijing
        self.train_meta = {
            "n_rows": int(len(X)), "n_valid": n_valid,
            "window": int(self.window),
            "train_window": int(win),
            "train_start": valid_idx.min().strftime("%Y-%m-%d"),
            "train_end": valid_idx.max().strftime("%Y-%m-%d"),
            "direct_mode": "rolling_refit",
            "point_shrink": self.point_shrink,
            "n_steps": int(self.horizon),
            "fitted_at": now_beijing().strftime("%Y-%m-%d %H:%M:%S"),
            "warm_started": self.warm_info["warm_started"],
            "parent_train_end": self.warm_info["parent_train_end"],
            "feature_cols": list(self.feature_cols),
            "zero_info_cols": list(getattr(self, "zero_info_cols", [])),
        }
        self._train_direct(X, log_p)
        # 训练后回填 direct 实际拟合区间（以最长步长 horizon 的配对区间为准）
        span = getattr(self, "_direct_span", None)
        if span is not None:
            self.train_meta["direct_window_start"], self.train_meta["direct_window_end"] = span
        if self.compute_residuals:
            self._rolling_residuals(X, log_p)
            if self.run_arima:
                self._arima_benchmark(log_p)
        # 残差分位/波动/基准/校准β与上期做指数平滑：跨期累积、越估越稳（不从零）
        if usable_warm is not None and self.compute_residuals:
            a = float(mcfg.get("residual_ema_alpha", 0.7))
            for h in self.resid_quantiles:
                if h in usable_warm.resid_quantiles:
                    prev_q = np.asarray(usable_warm.resid_quantiles[h], dtype=float)
                    cur_q = np.asarray(self.resid_quantiles[h], dtype=float)
                    # 上期值非有限（旧版本/NaN 修复前工件）时不混合，保留本期，避免 NaN 传染
                    if np.isfinite(prev_q).all() and np.isfinite(cur_q).all():
                        self.resid_quantiles[h] = a * cur_q + (1 - a) * prev_q
                    prev_s = float(usable_warm.resid_std.get(h, np.nan))
                    if np.isfinite(prev_s) and np.isfinite(self.resid_std[h]):
                        self.resid_std[h] = float(a * self.resid_std[h] + (1 - a) * prev_s)
            for h in list(self.calib_beta.keys()):
                if h in getattr(usable_warm, "calib_beta", {}):
                    prev_b = float(usable_warm.calib_beta[h])
                    if np.isfinite(prev_b) and np.isfinite(self.calib_beta[h]):
                        self.calib_beta[h] = float(
                            a * self.calib_beta[h] + (1 - a) * prev_b)
                    elif not np.isfinite(self.calib_beta[h]):
                        self.calib_beta[h] = 0.0
            if getattr(usable_warm, "arima_cumret", None) is not None and                     getattr(self, "arima_cumret", None) is not None:
                self.arima_cumret = float(
                    a * self.arima_cumret + (1 - a) * float(usable_warm.arima_cumret))
        # 最终清洗：任何非有限 β 一律归零（退守随机游走），杜绝 NaN 进入点预测/区间
        for h in list(self.calib_beta.keys()):
            if not np.isfinite(self.calib_beta[h]):
                self.calib_beta[h] = 0.0
        self.train_meta["cum_iters"] = dict(self.cum_iters)
        if self.calib_beta:
            self.train_meta["calib_beta"] = {int(k): round(float(v), 3)
                                            for k, v in self.calib_beta.items()}
            self.train_meta["calib_beta_h"] = round(float(self.calib_beta.get(self.horizon, 0)), 3)
        return self

    def _fallback_quantiles(self, h: int) -> np.ndarray:
        """未计算样本外残差时（如回测模式只取点预测后又调 predict）的正态兜底。"""
        sd = self.ret_std * np.sqrt(h)
        return np.array([-1.645 * sd, -0.674 * sd, 0.674 * sd, 1.645 * sd])

    def _train_direct(self, X: pd.DataFrame, log_p: pd.Series) -> None:
        """对每个步长 h，在【标签可得的最近滚动窗】内训练一个 direct 模型。

        顺序：先在全序列上构造未来 h 日累计对数收益 y=logP[t+h]-logP[t] 并剔除末端
        h 个无标签行，再取最近 train_window 个有效配对。这样训练样本一直覆盖到 t-h，
        完整保留最新 h 个交易日的信息（它们的标签恰为最近 h 日的真实涨跌）。
        """
        iters = int(get_config()["model"].get("direct_max_iter", 300))
        for h in range(1, self.horizon + 1):
            y = (log_p.shift(-h) - log_p)
            valid = y.notna()
            yv, Xv = y[valid], X.loc[y[valid].index]
            if self.train_window > 0:
                yv, Xv = yv.tail(self.train_window), Xv.tail(self.train_window)
            model = HistGradientBoostingRegressor(
                max_depth=4, max_iter=iters, learning_rate=0.05,
                min_samples_leaf=15, random_state=42)
            model.fit(Xv, yv)
            model._oilcast_cum_iter = iters
            self.models[h] = model
            self.cum_iters[h] = iters
            if h == self.horizon:
                self._direct_span = (Xv.index.min().strftime("%Y-%m-%d"),
                                     Xv.index.max().strftime("%Y-%m-%d"))

    def _rolling_residuals(self, X: pd.DataFrame, log_p: pd.Series) -> None:
        """滚动原点样本外检验：origin 每隔若干交易日取一个，回看训练窗拟合，在 origin
        上做严格样本外预测。一次循环同时产出两样东西：
          ① 残差经验分位 → 预测区间宽度（不用 in-sample 残差，避免过度乐观）；
          ② (pred, actual) 序列 → 点预测的在线校准系数 β（James-Stein/岭收缩）。

        β 的意义：严格无泄漏回测表明，日频 10 日方向信噪比极低，模型原始输出在没有样本
        外预测力的时段会因"过度自信"而稳定跑输随机游走。用样本外 pred/actual 估最优线性
        收缩 β=Cov(pred,actual)/Var(pred) 并截断到 [0,1]：无预测力时 β→0、点预测自动退守
        随机游走（机制上保证不跑输基准），有真实趋势信号时 β>0、按可信比例保留方向与幅度。
        """
        n = len(X)
        hs = tuple(h for h in RESIDUAL_HORIZONS if h <= self.horizon) \
            or tuple(range(1, self.horizon + 1))
        max_h = max(hs)
        mcfg = get_config()["model"]
        n_cal = int(self.calib_origins_override
                    if self.calib_origins_override is not None
                    else mcfg.get("calib_origins", 48))     # 内部原点数
        gap_c = int(mcfg.get("calib_gap", 5))
        rw = int(mcfg.get("resid_train_window", 500) or 0)
        origins = list(range(self.window, n - max_h, gap_c))[-n_cal:]
        iters = int(self.calib_iter_override
                    if self.calib_iter_override is not None
                    else mcfg.get("calib_iter", 150))
        resid: Dict[int, list] = {h: [] for h in hs}
        pa: Dict[int, list] = {h: [] for h in hs}      # (pred, actual, origin时点20日波动)
        for t in origins:
            lo = max(0, t - rw) if rw > 0 else 0
            tr_X, tr_log = X.iloc[lo:t], log_p.iloc[lo:t]
            # origin 时点的近期波动状态（只用 ≤t 信息，无泄漏）：用于把 β 按波动 regime 分层
            vol_t = float(tr_log.diff().iloc[-20:].std())
            # 子窗内整列全空的列剔除（不填 0），预测行按同一组列对齐
            keep = [c for c in tr_X.columns if int(tr_X[c].notna().sum()) >= 1]
            tr_X = tr_X[keep]
            for h in hs:
                y = (tr_log.shift(-h) - tr_log).dropna()
                m = HistGradientBoostingRegressor(
                    max_depth=4, max_iter=iters, learning_rate=0.05,
                    min_samples_leaf=15, random_state=42)
                m.fit(tr_X.loc[y.index], y)
                pred = float(m.predict(X[keep].iloc[[t]])[0])
                actual = float(log_p.iloc[t + h] - log_p.iloc[t])
                # 长假日（如国内春节超过 3 个工作日短填充上限）会让个别 actual 为 NaN，
                # 必须丢弃，否则污染 β 协方差与残差分位（上海原油曾因此整列预测 NaN）
                if not (np.isfinite(pred) and np.isfinite(actual) and np.isfinite(vol_t)):
                    continue
                pa[h].append((pred, actual, vol_t))
        # 样本外校准 β：带样本量收缩的单变量回归斜率，截断 [0,1]，不允许反向
        def beta_of(triples) -> float:
            if len(triples) < 8:
                return float(mcfg.get("point_shrink", 0.9))
            p = np.array([q[0] for q in triples]); a = np.array([q[1] for q in triples])
            vp = float(np.var(p))
            if vp < 1e-12 or not np.isfinite(vp):
                return 0.0
            cov = float(np.cov(p, a)[0, 1])
            if not np.isfinite(cov):
                return 0.0
            shrink_n = len(triples) / (len(triples) + float(mcfg.get("calib_prior_n", 20)))
            b = shrink_n * cov / vp
            if not np.isfinite(b):
                return 0.0
            return float(np.clip(b, 0.0, 1.0))
        # 当前（预测时点）波动状态：只用截至最新的真实数据
        cur_vol = float(log_p.iloc[-20:].diff().std())
        beta_anchor, beta_regime_note = {}, {}
        for h in hs:
            triples = pa[h]
            b_all = beta_of(triples)
            # regime 分层：按各 origin 波动中位数分高/低波两组分别估 β。事件/高波动期趋势
            # 延续性通常更强、原始预测更可信，应给更高 β；低波震荡期 β 更低、更贴近随机游走。
            # 任一组样本不足 8 则回退全局 β，绝不硬分。当前处于哪档就用哪档，全程无泄漏。
            hi = [q for q in triples if q[2] >= np.median([q[2] for q in triples])] if triples else []
            lo_ = [q for q in triples if q[2] < np.median([q[2] for q in triples])] if triples else []
            b_hi, b_lo = beta_of(hi), beta_of(lo_)
            if len(hi) < 8: b_hi = b_all
            if len(lo_) < 8: b_lo = b_all
            # regime 分层 β 默认关闭：2026-09 三品种无泄漏 A/B（各24原点）显示分层后 MAE
            # 均轻微劣于单一全局 β（校准原点本就不多、再二分导致每组样本不足、噪声更大）。
            # 保留开关，待校准样本显著增多后可再验证；不得在未被无泄漏回测证实改善前开启。
            use_regime = bool(mcfg.get("use_regime_beta", False))
            if not use_regime:
                pick = b_all            # 关闭 regime 分层时退回单一全局 β（用于 A/B 与回退）
            elif np.isfinite(cur_vol) and triples and cur_vol >= np.median([q[2] for q in triples]):
                pick = b_hi
            else:
                pick = b_lo
            beta_anchor[h] = float(pick)
            beta_regime_note[h] = {"all": round(b_all, 3), "high_vol": round(b_hi, 3),
                                   "low_vol": round(b_lo, 3), "regime": "high" if pick == b_hi else "low"}
        self.train_meta["beta_regime"] = {int(h): beta_regime_note[h] for h in hs}
        # 残差必须相对【β 校准后的预测】计算（actual - β·pred），与最终点预测同基准，
        # 否则 β 很小时区间中心会偏离点预测、出现 q05>mean 的错位
        for h in hs:
            bh = beta_anchor[h]
            resid[h] = [a - bh * p for p, a, _ in pa[h]]
        # 插值补齐全部 h；样本不足以做滚动原点时，用日收益正态分位 ×√h 兜底
        self.ret_std = float(log_p.diff().std())
        z = {"q05": -1.645, "q25": -0.674, "q75": 0.674, "q95": 1.645}
        anchor_h = np.array(hs)
        self.calib_beta = {}
        for h in range(1, self.horizon + 1):
            self.calib_beta[h] = float(np.interp(h, anchor_h,
                                                 [beta_anchor[hh] for hh in hs]))
            if origins:
                qs, stds = [], []
                for q in (0.05, 0.25, 0.75, 0.95):
                    vals = [np.quantile(resid[hh], q) for hh in anchor_h]
                    qs.append(float(np.interp(h, anchor_h, vals)))
                stds = [np.std(resid[hh], ddof=1) for hh in anchor_h]
                self.resid_quantiles[h] = np.array(qs)
                self.resid_std[h] = float(np.interp(h, anchor_h, stds))
            else:
                sd = self.ret_std * np.sqrt(h)
                self.resid_quantiles[h] = np.array([z["q05"] * sd, z["q25"] * sd,
                                                    z["q75"] * sd, z["q95"] * sd])
                self.resid_std[h] = float(sd)

    def _arima_benchmark(self, log_p: pd.Series) -> None:
        """ARIMA 基准：阶数由高到低自动降级，收敛警告视为失败并重试更简模型。"""
        import warnings
        from statsmodels.tsa.arima.model import ARIMA
        rets = log_p.diff().dropna().tail(250)
        for order in ((2, 0, 2), (1, 0, 1), (1, 0, 0)):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ARIMA(rets, order=order).fit()
                if not getattr(model, "mle_retvals", {}).get("converged", True):
                    continue
                fc = model.forecast(steps=self.horizon)
                self.arima_cumret = float(fc.cumsum().iloc[-1])
                return
            except Exception as exc:
                LOG.warning("ARIMA%s 基准拟合失败：%s", order, exc)
        LOG.warning("全部 ARIMA 阶数均未收敛，跳过基准对比")
        self.arima_cumret = None

    # --------------------------------------------------------- predict
    def predict(self, latest_X: pd.DataFrame, last_price: float,
                future_dates: pd.DatetimeIndex) -> PathResult:
        # 训练时整列全空的列未入模（active_cols）；预测按 active 列对齐即可，既不填 0
        # 也不要求全空列存在。完整 schema 仍登记在 feature_cols，供跨期一致性判断。
        active = getattr(self, "active_cols", None) or getattr(self, "feature_cols", None)
        if active is not None:
            missing = [c for c in active if c not in latest_X.columns]
            if missing:
                raise InsufficientData(f"预测输入缺少训练特征列：{missing}")
            latest_X = latest_X[active]
        x_now = latest_X.iloc[[-1]]
        rows = []
        for h, dt in enumerate(future_dates, start=1):
            # 点预测幅度由【样本外校准系数 β】决定（替代旧版拍脑袋的固定 0.7/0.9 收缩）：
            # β 来自训练窗内部严格样本外 pred/actual，无预测力时→0（退守随机游走，机制上
            # 不跑输基准），有真实趋势信号时按可信比例保留方向。区间宽度仍由样本外残差决定。
            beta = float(self.calib_beta.get(
                h, getattr(self, "point_shrink", 0.9))) if hasattr(self, "calib_beta") \
                else float(getattr(self, "point_shrink", 0.9))
            if not np.isfinite(beta):
                beta = 0.0   # 校准系数异常时最保守处理：退守随机游走
            cum = beta * float(self.models[h].predict(x_now)[0])
            qs = self.resid_quantiles.get(h, self._fallback_quantiles(h))
            qs = np.asarray(qs, dtype=float)
            if not np.isfinite(qs).all():      # 残差分位异常时正态兜底，区间永不为 NaN
                qs = self._fallback_quantiles(h)
            raw = cum + qs                      # [q05,q25,q75,q95] 对数空间
            # 单调性保障：小样本残差分位可能抖动，强制 q05≤q25≤点预测≤q75≤q95，区间永不倒挂
            lo = np.sort([min(float(raw[0]), cum), min(float(raw[1]), cum)])
            hi = np.sort([max(float(raw[2]), cum), max(float(raw[3]), cum)])
            q05, q25, q75, q95 = lo[0], lo[1], hi[0], hi[1]
            rows.append({
                "date": dt, "mean": last_price * np.exp(cum),
                "q50": last_price * np.exp(cum),
                "q05": last_price * np.exp(q05), "q25": last_price * np.exp(q25),
                "q75": last_price * np.exp(q75), "q95": last_price * np.exp(q95),
                "_cumret": cum,
            })
        path = pd.DataFrame(rows).set_index("date")

        last = path.iloc[-1]
        std = self.resid_std.get(self.horizon, self.ret_std * np.sqrt(self.horizon))
        if not np.isfinite(std) or std <= 1e-9:
            prob_up = 0.5   # 波动尺度不可用时方向概率取中性，绝不输出 NaN
        else:
            prob_up = float(norm.cdf(last["_cumret"] / std))
        endpoint = {
            "target_date": pd.Timestamp(path.index[-1]).strftime("%Y-%m-%d"),
            "mean": round(float(last["mean"]), 2),
            "q05": round(float(last["q05"]), 2), "q25": round(float(last["q25"]), 2),
            "q75": round(float(last["q75"]), 2), "q95": round(float(last["q95"]), 2),
            "pct_mean": round((float(last["mean"]) / last_price - 1) * 100, 2),
            "prob_up": round(prob_up, 3), "prob_down": round(1 - prob_up, 3),
        }
        bench = None
        if getattr(self, "arima_cumret", None) is not None:
            bench = {"mean": round(last_price * np.exp(self.arima_cumret), 2),
                     "pct_mean": round((np.exp(self.arima_cumret) - 1) * 100, 2)}
        return PathResult(path=path.drop(columns=["_cumret"]), endpoint=endpoint,
                          benchmark_endpoint=bench)
