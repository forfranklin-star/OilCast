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
from scipy.stats import norm, binomtest, ttest_1samp
from sklearn.ensemble import (
    HistGradientBoostingRegressor, HistGradientBoostingClassifier)
from sklearn.isotonic import IsotonicRegression

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


def _hgb_fit_columns(frame: pd.DataFrame) -> list:
    """挑选可安全送入 HistGradientBoosting 分箱的数值列（基于【本次训练切片】）。

    sklearn>=1.6（搭配 numpy2）的分箱器用 sliding_window_view(distinct_values, 2) 取相邻
    均值作为阈值；当某列在该训练切片内"非缺失的不同取值不足 2 个"——整列全空，或非空但
    只有一个唯一值（常数列）——窗口长度 2 大于输入长度，抛
    "window shape cannot be larger than input array shape"（旧版 sklearn 会安全退化为
    无阈值，故本地旧版不复现）。树模型对零方差列本就无法分裂、不含信息，因此逐次拟合前
    剔除这些列；只影响当次拟合，不改动全局 feature_cols/schema，也不丢任何真实信息。"""
    cols = []
    for c in frame.columns:
        s = frame[c]
        if int(s.notna().sum()) >= 1 and int(s.dropna().nunique()) >= 2:
            cols.append(c)
    return cols


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
                 run_arima: bool = False,
                 dir_origins: Optional[int] = None) -> None:
        self.horizon = horizon
        self.window = window
        self.compute_residuals = compute_residuals
        # ARIMA 基准默认关闭：实测在无频率的交易日 index 上长期收敛失败、零贡献却每品种
        # 串行试 3 阶、拖慢每日管线（已有随机游走基准作对照）；确需时显式 run_arima=True。
        self.run_arima = run_arima
        # 内部样本外校准规模的可选覆盖（None=读 config）；回测/测试可用小值提速
        self.calib_origins_override = calib_origins
        self.calib_iter_override = calib_iter
        # 方向门控滚动原点个数的可选覆盖（None=读 config.dir_gate_origins）
        self.dir_origins_override = dir_origins
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
        # —— 方向概率分类层（与幅度回归解耦）——
        # dir_models: 各 anchor 步长的涨跌分类器；dir_calib: 样本外 isotonic 概率校准器；
        # dir_edge: 各步长样本外方向 edge 是否统计成立（命中率/二项p/样本数）。全部随工件
        # joblib 持久化，跨重启保留（持续学习，不从零）。
        self.dir_models: Dict[int, HistGradientBoostingClassifier] = {}
        self.dir_calib: Dict[int, Optional[IsotonicRegression]] = {}
        self.dir_edge: Dict[int, dict] = {}
        # trend_edge: 独立时序动量通道（mom_21）的样本外经济价值证据。与 ML 概率通道并行：
        # 趋势跟随常胜率不高但盈亏比>1、期望为正，单独按经济价值判据门控。随工件持久化。
        self.trend_edge: Dict[int, dict] = {}

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
        # 价格必须为正才能取对数：脏数据里的 0/负值先置为缺失（np.log 对 NaN 不告警、
        # 由后续 valid_mask 自然剔除），避免 RuntimeWarning: invalid value in log 及 -inf 入模。
        log_p = np.log(price.where(price > 0))
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
        # 方向层：长历史样本外概率校准 + edge 显著性门控 + 最终方向分类器（无泄漏）。
        # 门控 edge 是慢变统计（数百非重叠原点、回看约5年），每天全量重算代价大却几乎不变，
        # 故当上期工件在 dir_gate_refresh_days 天内已算过门控时【复用其门控证据】、只重训最终
        # 分类器（秒级），到期/冷启动/特征schema变化才全量重算——日常运行由此大幅提速，且
        # edge 结论仍周期性刷新，原始库与最终分类器每天重训，持续学习不从零。
        if self._reuse_direction_gate(usable_warm):
            self._train_direction(X, log_p)
        else:
            self._direction_walkforward(X, log_p)
            self.train_meta["gate_trained_date"] = self.train_meta["train_end"]
            self.train_meta["gate_reused"] = False
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
        # 方向门控证据（样本外命中率/二项p/是否成立）写入元信息，供页面与审计展示
        if getattr(self, "dir_edge", None):
            self.train_meta["direction_gate"] = {
                int(h): {k: v for k, v in ev.items() if k != "conf_threshold"}
                for h, ev in self.dir_edge.items()}
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
            # 逐 h 按"实际有标签的训练切片"剔除全空/常数列，规避新版 sklearn 分箱崩溃，
            # 并把本次真正入模的列挂到模型上，供 predict 严格按同列对齐（旧工件回退 active）
            fit_cols = _hgb_fit_columns(Xv)
            if not fit_cols:
                raise InsufficientData(
                    f"h={h} 训练窗内无任何含≥2个不同真实值的特征列，拒绝训练")
            model = HistGradientBoostingRegressor(
                max_depth=4, max_iter=iters, learning_rate=0.05,
                min_samples_leaf=15, random_state=42)
            model.fit(Xv[fit_cols], yv)
            model._oilcast_cum_iter = iters
            model._oilcast_fit_cols = list(fit_cols)
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
            for h in hs:
                y = (tr_log.shift(-h) - tr_log).dropna()
                # 按"该 h 实际有标签的拟合切片"逐次剔除全空/常数列（不填 0）：既规避新版
                # sklearn 对常数/全空列分箱抛 window shape，又保证 fit/predict 同列对齐
                fit_frame = tr_X.loc[y.index]
                keep = _hgb_fit_columns(fit_frame)
                if not keep:
                    # 该 origin 此步长窗内无任何含变化的真实特征，跳过，不污染 β/残差
                    continue
                m = HistGradientBoostingRegressor(
                    max_depth=4, max_iter=iters, learning_rate=0.05,
                    min_samples_leaf=15, random_state=42)
                m.fit(fit_frame[keep], y)
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

    def _reuse_direction_gate(self, warm) -> bool:
        """门控证据跨期复用判定。返回 True 表示直接继承 warm 的门控(校准器/胜率edge/趋势edge)，
        调用方只需重训最终方向分类器；False 表示需全量重算门控。

        复用条件（任一不满足即全量重算，保证不漏新出现的 edge）：
          - 不是回测/测试用 dir_origins 显式控制的场景；
          - warm 确有完整门控证据（dir_calib/dir_edge/trend_edge）；
          - warm 门控训练日距本期训练 end 在 dir_gate_refresh_days 个自然日内。
        """
        if self.dir_origins_override is not None or warm is None:
            return False
        for attr in ("dir_calib", "dir_edge", "trend_edge"):
            obj = getattr(warm, attr, None)
            if not obj:
                return False
        if not getattr(warm, "dir_models", None):
            return False
        prev_gate = (getattr(warm, "train_meta", {}) or {}).get("gate_trained_date")
        if not prev_gate:
            return False
        try:
            gap_days = (pd.Timestamp(self.train_meta["train_end"])
                        - pd.Timestamp(prev_gate)).days
        except Exception:
            return False
        refresh = int(get_config()["model"].get("dir_gate_refresh_days", 7))
        if gap_days < 0 or gap_days > refresh:
            return False
        import copy
        self.dir_calib = copy.deepcopy(warm.dir_calib)
        self.dir_edge = copy.deepcopy(warm.dir_edge)
        self.trend_edge = copy.deepcopy(warm.trend_edge)
        self.dir_models = copy.deepcopy(warm.dir_models)
        self.train_meta["gate_trained_date"] = prev_gate
        self.train_meta["gate_reused"] = True
        self.train_meta["gate_age_days"] = int(gap_days)
        return True

    def _direction_walkforward(self, X: pd.DataFrame, log_p: pd.Series) -> None:
        """在较长历史上做严格无泄漏滚动原点，收集方向分类器的样本外涨跌概率，用于
        isotonic 校准与方向 edge 显著性门控，随后训练最终方向分类器。

        方向 edge 是相对稳定的统计性质，不能只靠 β 残差那 30 个近端原点估计（会时灵时
        不灵），故单独用更长的原点序列。每个原点 t 仅用 ≤t 数据训练（训练标签 shift(-h)
        在切片末端自然为 NaN、终点≤t），预测 t→t+h 实际方向，绝无标签越界。"""
        mcfg = get_config()["model"]
        hs = [h for h in RESIDUAL_HORIZONS if h <= self.horizon]
        # dir_origins 显式传 0：跳过门控滚动收集（复用外部已注入的 dir_calib/dir_edge，
        # 供回测逐原点提速），只训练最终方向分类器。
        if self.dir_origins_override == 0:
            self.dir_calib = getattr(self, "dir_calib", None) or {}
            self.dir_edge = getattr(self, "dir_edge", None) or {}
            self.trend_edge = getattr(self, "trend_edge", None) or {}
            self._train_direction(X, log_p)
            return
        # 门控回看长度（交易日）。方向 edge 是稳定统计性质，短窗（如近 300 日）噪声大、
        # 结论会在品种间偶然翻转；用多年历史、且每个步长按 h 非重叠取原点（标签不重叠，
        # 显著性不被高估），结论才与长样本无泄漏回测一致。dir_origins_override>0 时按
        # "回看 origin*h 个交易日"近似，供回测/测试压缩规模。
        win = self.train_window
        # 门控滚动要在数百个非重叠原点上反复拟合，只用于判定 edge 是否存在，用较少迭代即可，
        # 避免每日任务过慢；最终部署用方向分类器在 _train_direction 内用满 dir_clf_max_iter。
        iters = int(mcfg.get("dir_gate_max_iter",
                             min(60, int(mcfg.get("dir_clf_max_iter", 120)))))
        if self.dir_origins_override:
            lookback = self.dir_origins_override * max(hs)
        else:
            lookback = int(mcfg.get("dir_gate_lookback", 1200))
        n = len(X)
        last = n - 1
        dir_pa: Dict[int, list] = {h: [] for h in hs}
        trend_pa: Dict[int, list] = {h: [] for h in hs}
        # 时序动量通道：mom_21 及"截至 t 的扩展中位数"作趋势强度下限（只用 ≤t 数据，无泄漏）
        mom21 = X["mom_21"] if "mom_21" in X.columns else pd.Series(np.nan, index=X.index)
        mom21_cut = mom21.abs().expanding(min_periods=60).median()
        # 每个 h 的门控原点上限：步长始终取 h 的整数倍，抽稀后标签区间仍互不重叠
        # （不破坏二项检验独立性），同时把 h=2 这类过密原点从数百压到上限内，显著降耗时。
        cap = int(mcfg.get("dir_gate_max_origins_per_h", 150))
        for h in hs:
            first = max(win + 2, last - lookback)
            raw_origins = list(range(first, last - h + 1, h))   # 间隔=h，标签非重叠
            if len(raw_origins) > cap:
                k = -(-len(raw_origins) // cap)                 # 向上取整，步长=k*h 仍非重叠
                origins = raw_origins[::k]
            else:
                origins = raw_origins
            for t in origins:
                # 趋势通道：mom_21 强度越过其历史中位数才记录"顺势持有 h"的实际对数收益
                m21 = float(mom21.iloc[t]); cut = mom21_cut.iloc[t]
                fwd_ret = float(log_p.iloc[t + h] - log_p.iloc[t])
                if (np.isfinite(m21) and np.isfinite(cut) and np.isfinite(fwd_ret)
                        and abs(m21) >= cut):
                    trend_pa[h].append((np.sign(m21), fwd_ret))
                lo = max(0, t - win + 1)
                tr_X = X.iloc[lo:t + 1]
                tr_log = log_p.iloc[lo:t + 1]
                yy = tr_log.shift(-h) - tr_log
                valid = yy.notna()
                if int(valid.sum()) < 60:
                    continue
                yd = (yy[valid] > 0).astype(int)
                Xf = tr_X.loc[yd.index]
                if yd.nunique() < 2:
                    continue
                keep = _hgb_fit_columns(Xf)
                if not keep:
                    continue
                try:
                    clf = HistGradientBoostingClassifier(
                        max_depth=4, max_iter=iters, learning_rate=0.05,
                        min_samples_leaf=15, random_state=42)
                    clf.fit(Xf[keep], yd)
                    p_up = float(clf.predict_proba(X[keep].iloc[[t]])[0, 1])
                    actual_dir = int(log_p.iloc[t + h] > log_p.iloc[t])
                    # 同时记录该原点 t→t+h 的实际对数收益，供"经济价值判据"评估期望/盈亏比
                    if np.isfinite(p_up) and np.isfinite(fwd_ret):
                        dir_pa[h].append((p_up, actual_dir, fwd_ret))
                except Exception:
                    continue
        cur_cut = float(mom21_cut.iloc[last]) if np.isfinite(mom21_cut.iloc[last]) else np.nan
        self._fit_direction_gate(hs, dir_pa, mcfg, trend_pa, cur_cut)
        self._train_direction(X, log_p)

    @staticmethod
    def _crossfit_calibrated_prob(rp: np.ndarray, ad: np.ndarray,
                                  n_splits: int = 5) -> np.ndarray:
        """顺序 K 折交叉拟合，返回每个原点的 out-of-fold 校准概率（评估 edge 专用）。"""
        n = len(rp)
        cal = np.empty(n, dtype=float)
        folds = np.array_split(np.arange(n), n_splits)
        for fold in folds:
            mask = np.ones(n, dtype=bool)
            mask[fold] = False
            if len(np.unique(rp[mask])) >= 2 and len(np.unique(ad[mask])) == 2:
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0.02,
                                         y_max=0.98).fit(rp[mask], ad[mask])
                cal[fold] = np.clip(iso.predict(rp[fold]), 0.02, 0.98)
            else:
                cal[fold] = rp[fold]   # 训练折信息不足时回退原始概率，不强行校准
        return cal

    @staticmethod
    def _trend_gate(tarr, strength_cut, min_eng, min_payoff, max_ret_p) -> dict:
        """独立时序动量通道的样本外经济价值检验。tarr 元素=(mom_21 符号, 顺势持有 h 的
        实际对数收益)。胜率可不足 55%，只要平均收益>0、盈亏比达标、t 检验显著即认可。"""
        out = {"has_edge": False, "n": len(tarr), "hit": None, "mean_ret": None,
               "payoff": None, "p": None,
               "strength_cut": (round(float(strength_cut), 5)
                                if np.isfinite(strength_cut) else None)}
        if len(tarr) < min_eng:
            return out
        sgn = np.array([a[0] for a in tarr], dtype=float)
        ret = np.array([a[1] for a in tarr], dtype=float)
        strat = sgn * ret
        hit = float((np.sign(ret) == sgn).mean())
        mean = float(strat.mean())
        wins, losses = strat[strat > 0], strat[strat < 0]
        payoff = (float(wins.mean() / abs(losses.mean()))
                  if len(wins) and len(losses) else np.nan)
        p = float(ttest_1samp(strat, 0.0).pvalue) if len(tarr) >= 3 else 1.0
        ok = bool(mean > 0 and np.isfinite(payoff) and payoff >= min_payoff
                  and p <= max_ret_p)
        out.update(has_edge=ok, hit=round(hit, 3), mean_ret=round(mean * 100, 3),
                   payoff=round(payoff, 3) if np.isfinite(payoff) else None,
                   p=round(p, 3))
        return out

    def _fit_direction_gate(self, hs, dir_pa, mcfg, trend_pa=None,
                            trend_cut: float = np.nan) -> None:
        """方向 edge 由两条【独立通道】检验，任一成立才允许明确表态：

        通道A·ML 概率：样本外 isotonic（交叉拟合）校准后，高置信表态的【胜率】显著>50%。
        通道B·时序动量(mom_21)：不依赖 ML 概率，趋势强度越过历史中位数时顺势持有 h，按
          【经济价值】判据——平均收益>0、盈亏比≥阈值、t 检验显著。趋势跟随常胜率不高
          （可低于 55%）但盈亏比>1、期望为正，单看胜率会误杀，故单列。
        edge_basis 记录成立来源（"胜率"/"趋势期望"）；都不成立则诚实中性。全部数字来自
        严格样本外、非重叠原点，绝不用原点自身校准后再评它。"""
        thr = float(mcfg.get("dir_conf_threshold", 0.62))
        margin = thr - 0.5
        min_n = int(mcfg.get("edge_min_n", 20))
        min_eng = int(mcfg.get("edge_min_engaged", 8))
        min_hit = float(mcfg.get("edge_min_hit", 0.55))
        max_p = float(mcfg.get("edge_max_p", 0.20))
        min_payoff = float(mcfg.get("edge_min_payoff", 1.15))
        max_ret_p = float(mcfg.get("edge_max_ret_p", 0.10))
        trend_pa = trend_pa or {}
        self.dir_calib, self.dir_edge, self.trend_edge = {}, {}, {}
        for h in hs:
            te = self._trend_gate(trend_pa.get(h, []), trend_cut,
                                  min_eng, min_payoff, max_ret_p)
            self.trend_edge[h] = te
            arr = dir_pa.get(h, [])
            base = dict(conf_threshold=thr, n=len(arr))
            trend_fields = {"trend_n": te["n"], "trend_hit": te["hit"],
                            "trend_mean_ret": te["mean_ret"], "trend_payoff": te["payoff"],
                            "trend_p": te["p"]}
            empty = {**base, "has_edge": te["has_edge"],
                     "edge_basis": "趋势期望" if te["has_edge"] else None,
                     "engaged_n": 0, "engaged_hit": None, "engaged_p": None,
                     "all_hit": None, "engaged_mean_ret": None, "engaged_payoff": None,
                     "engaged_ret_p": None, **trend_fields}
            if len(arr) < min_n:
                self.dir_calib[h] = None
                self.dir_edge[h] = {**empty, "reason": "样本外原点不足"}
                continue
            rp = np.array([a[0] for a in arr], dtype=float)
            ad = np.array([a[1] for a in arr], dtype=int)
            fr = np.array([a[2] for a in arr], dtype=float)
            # 部署用校准器：用全部样本外点拟合（预测的是未来新点，无泄漏）
            iso = None
            if len(np.unique(rp)) >= 2 and len(np.unique(ad)) == 2:
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0.02,
                                         y_max=0.98).fit(rp, ad)
            # edge 评估必须用【交叉拟合 out-of-fold】校准概率（理由见方法注释），
            # 否则 isotonic 记忆样本会系统性虚高高置信命中（样本内校准泄漏）。
            cal = self._crossfit_calibrated_prob(rp, ad)
            engaged = np.abs(cal - 0.5) >= margin
            en = int(engaged.sum())
            eh = int(((cal[engaged] >= 0.5) == (ad[engaged] == 1)).sum()) if en else 0
            all_hit = float(((cal >= 0.5) == (ad == 1)).mean())
            hit = eh / en if en else 0.5
            p_val = float(binomtest(eh, en, 0.5).pvalue) if en else 1.0
            # —— 通道A 经济价值：ML 高置信表态子集按方向的对数收益 ——
            strat = np.sign(cal[engaged] - 0.5) * fr[engaged] if en else np.array([])
            wins = strat[strat > 0]; losses = strat[strat < 0]
            mean_ret = float(strat.mean()) if en else 0.0
            payoff = (float(wins.mean() / abs(losses.mean()))
                      if len(wins) and len(losses) else np.nan)
            ret_p = float(ttest_1samp(strat, 0.0).pvalue) if en >= 3 else 1.0
            hit_edge = bool(en >= min_eng and hit >= min_hit and p_val <= max_p)
            econ_edge = bool(en >= min_eng and mean_ret > 0 and np.isfinite(payoff)
                             and payoff >= min_payoff and ret_p <= max_ret_p)
            has_edge = hit_edge or econ_edge or te["has_edge"]
            if hit_edge:
                edge_basis = "胜率"
            elif econ_edge or te["has_edge"]:
                edge_basis = "趋势期望"
            else:
                edge_basis = None
            self.dir_calib[h] = iso
            self.dir_edge[h] = {
                **base, "has_edge": has_edge, "edge_basis": edge_basis,
                "engaged_n": en, "engaged_hit": round(hit, 3),
                "engaged_p": round(p_val, 3), "all_hit": round(all_hit, 3),
                "engaged_mean_ret": round(mean_ret * 100, 3),
                "engaged_payoff": round(float(payoff), 3) if np.isfinite(payoff) else None,
                "engaged_ret_p": round(ret_p, 3), **trend_fields}

    def _train_direction(self, X: pd.DataFrame, log_p: pd.Series) -> None:
        """对每个 anchor 步长用最近滚动窗训练最终涨跌分类器。严格无泄漏：标签
        y=logP[t+h]-logP[t] 由 shift(-h) 构造，末端 h 行标签为 NaN 被剔除，训练样本标签
        终点不超过最新交易日；与对应回归模型使用完全相同的入模列。"""
        mcfg = get_config()["model"]
        iters = int(mcfg.get("dir_clf_max_iter", 200))
        hs = sorted(self.dir_edge.keys()) or [h for h in RESIDUAL_HORIZONS
                                              if h <= self.horizon]
        self.dir_models = {}
        for h in hs:
            yd = (log_p.shift(-h) - log_p > 0).astype(int)
            valid = (log_p.shift(-h) - log_p).notna()
            yv, Xv = yd[valid], X.loc[valid.index[valid]]
            if self.train_window > 0:
                yv, Xv = yv.tail(self.train_window), Xv.tail(self.train_window)
            if yv.nunique() < 2:
                continue
            reg = self.models.get(h)
            keep = getattr(reg, "_oilcast_fit_cols", None) or _hgb_fit_columns(Xv)
            keep = [c for c in keep if c in Xv.columns]
            if not keep:
                continue
            clf = HistGradientBoostingClassifier(
                max_depth=4, max_iter=iters, learning_rate=0.05,
                min_samples_leaf=15, random_state=42)
            clf.fit(Xv[keep], yv)
            clf._oilcast_fit_cols = list(keep)
            self.dir_models[h] = clf

    def direction_at(self, x_row: pd.DataFrame, h: Optional[int] = None):
        """返回某步长（默认 horizon）的 (校准后看涨概率, 方向立场, edge证据)。

        方向立场三分类：仅当该步长样本外方向 edge 统计成立、且校准概率越过置信阈值时才
        明确看涨/看跌，否则一律"中性"。无方向分类器（旧工件/未校准）时安全回退中性。"""
        h = self.horizon if h is None else h
        anchors = sorted(self.dir_models.keys())
        if not anchors:
            return 0.5, "中性", {"has_edge": False, "reason": "无方向分类器"}
        ha = min(anchors, key=lambda k: abs(k - h))
        clf = self.dir_models[ha]
        cols = getattr(clf, "_oilcast_fit_cols", None)
        raw = float(clf.predict_proba(x_row[cols] if cols is not None else x_row)[0, 1])
        iso = self.dir_calib.get(ha)
        if iso is not None:
            p = float(np.clip(iso.predict([raw])[0], 0.02, 0.98))
        else:
            p = float(np.clip(raw, 0.02, 0.98))
        edge = dict(self.dir_edge.get(ha, {})); edge["anchor"] = ha
        thr = float(edge.get("conf_threshold", 0.62))
        has = bool(edge.get("has_edge", False))
        stance = "中性"
        # 通道A：ML 概率（胜率/ML 经济价值）edge 成立且校准概率越过置信带才表态
        if has and p >= thr:
            stance = "看涨"
        elif has and p <= 1 - thr:
            stance = "看跌"
        else:
            # 通道B：独立时序动量趋势补位——ML 概率中性，但 mom_21 趋势的样本外经济价值
            # 成立、且当前趋势强度越过门控期强度下限时，按趋势方向表态（低胜率高盈亏比型）。
            te = self.trend_edge.get(ha, {})
            if te.get("has_edge") and "mom_21" in x_row.columns:
                m_now = float(x_row["mom_21"].iloc[0])
                cut = te.get("strength_cut")
                if np.isfinite(m_now) and cut is not None and abs(m_now) >= cut:
                    stance = "看涨" if m_now > 0 else "看跌"
                    edge = {**edge, "edge_basis": "趋势期望", "trend_hit": te.get("hit"),
                            "trend_mean_ret": te.get("mean_ret"),
                            "trend_payoff": te.get("payoff"), "trend_p": te.get("p")}
        return p, stance, edge

    def _arima_benchmark(self, log_p: pd.Series) -> None:
        """ARIMA 基准：阶数由高到低自动降级，收敛警告视为失败并重试更简模型。"""
        import warnings
        from statsmodels.tsa.arima.model import ARIMA
        # 交易日 DatetimeIndex 无固定频率，statsmodels 会报 "No supported index is
        # available"；重置为整数 RangeIndex 即可（ARIMA 只用序列次序，不依赖日历）。
        rets = log_p.diff().dropna().tail(250).reset_index(drop=True)
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
        x_base = latest_X.iloc[[-1]]
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
            # 与该步长模型实际入模列严格对齐（训练时剔除的窗内常数/全空列这里同样不取）；
            # 旧版本导入的模型没有该标记时回退到 active 列集合
            fit_cols = getattr(self.models[h], "_oilcast_fit_cols", None)
            x_now = x_base[fit_cols] if fit_cols else x_base
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
            mag_prob = 0.5   # 波动尺度不可用时幅度隐含概率取中性，绝不输出 NaN
        else:
            mag_prob = float(norm.cdf(last["_cumret"] / std))
        # 方向以独立的概率分类器（样本外 isotonic 校准 + 显著性门控）为准；幅度隐含概率
        # mag_prob 仅作对照。无分类器（旧工件）时回退到幅度隐含概率、立场中性。
        if getattr(self, "dir_models", None):
            dir_prob, stance, edge = self.direction_at(x_base, self.horizon)
        else:
            dir_prob, stance, edge = mag_prob, "中性", {"has_edge": False,
                                                       "reason": "旧工件无方向分类器"}
        prob_up = dir_prob if getattr(self, "dir_models", None) else mag_prob
        endpoint = {
            "target_date": pd.Timestamp(path.index[-1]).strftime("%Y-%m-%d"),
            "mean": round(float(last["mean"]), 2),
            "q05": round(float(last["q05"]), 2), "q25": round(float(last["q25"]), 2),
            "q75": round(float(last["q75"]), 2), "q95": round(float(last["q95"]), 2),
            "pct_mean": round((float(last["mean"]) / last_price - 1) * 100, 2),
            "prob_up": round(prob_up, 3), "prob_down": round(1 - prob_up, 3),
            "mag_prob_up": round(mag_prob, 3),
            "dir_prob_up": round(dir_prob, 3),
            "dir_stance": stance,
            "dir_has_edge": bool(edge.get("has_edge", False)),
            "dir_edge_basis": edge.get("edge_basis"),
            "dir_edge_hit": edge.get("engaged_hit"),
            "dir_edge_p": edge.get("engaged_p"),
            "dir_edge_n": edge.get("n"),
            "dir_edge_mean_ret": edge.get("engaged_mean_ret"),
            "dir_edge_payoff": edge.get("engaged_payoff"),
        }
        bench = None
        if getattr(self, "arima_cumret", None) is not None:
            bench = {"mean": round(last_price * np.exp(self.arima_cumret), 2),
                     "pct_mean": round((np.exp(self.arima_cumret) - 1) * 100, 2)}
        return PathResult(path=path.drop(columns=["_cumret"]), endpoint=endpoint,
                          benchmark_endpoint=bench)
