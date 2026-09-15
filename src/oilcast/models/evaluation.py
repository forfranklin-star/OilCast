"""滚动原点回测：量化短期模型的样本外表现，并与随机游走基准对比。

方向评估采用【三分类口径】（看涨 / 看跌 / 中性）：
- 方向由独立的概率分类器（样本外 isotonic 校准 + 显著性门控）给出，只有某(品种,周期)
  的样本外方向 edge 统计成立时才明确看涨/看跌，否则判"中性"；
- "中性"是模型在弱有效市场下的诚实选择，**不计为方向错误**。因此评估分别报告：
  明确表态率、表态原点的方向命中率（及相对抛硬币的二项检验 p 值）、中性率、概率
  Brier 分数；另保留底层回归的二分命中与幅度 MAE 作参考。
旧版用 sign(β·pred)==sign(actual) 计方向、β=0 时把中性一律算错，会在震荡市把命中率
系统性压到接近 0，属误导口径，已废弃（字段保留仅为向后兼容，页面不再展示）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .short_term import ShortTermForecaster


def backtest_short(features: pd.DataFrame, price: pd.Series,
                   horizon: int = 10, n_origins: int = 42, gap: int = 3,
                   calib_origins: int = None, calib_iter: int = None,
                   dir_origins: int = None) -> dict:
    """在最近 n_origins 个滚动原点上做严格样本外 horizon 日预测评估。

    每个原点 t：仅用 ≤t 的数据、按生产口径（滚动训练窗）拟合，在 t 预测 t+horizon，
    再与 t+horizon 的真实价格比较；随机游走基准为"价格保持不变"。真实价格缺口处跳过，
    绝不用填充值计算误差。

    方向门控（是否存在统计 edge、概率校准器）在【回测起点之前】一次性无泄漏估计，各
    原点复用，既保证门控不偷看回测期，又避免逐原点重复跑长历史门控导致回测过慢。

    返回幅度误差（MAE/RMSE，对随机游走）、三分类方向指标（表态率/表态命中/二项p/
    中性率/Brier）及是否跑赢随机游走。
    """
    n = len(features)
    if n < 260:
        return {"available": False, "reason": f"样本不足({n}<260)"}
    origins = list(range(n - horizon - n_origins * gap, n - horizon, gap))
    origins = [t for t in origins if t >= 260]
    if not origins:
        return {"available": False, "reason": "可用回测原点不足"}

    # 门控参考模型：只用第一个回测原点【之前】的数据估计方向 edge 与校准器（无泄漏）
    gate_fc = None
    gate_err = None
    try:
        gate_fc = ShortTermForecaster(
            horizon, compute_residuals=False, run_arima=False,
            dir_origins=dir_origins).fit(features.iloc[:origins[0]],
                                         price.iloc[:origins[0]])
    except Exception as exc:  # 门控参考失败不拖垮幅度回测，方向层降级为全中性
        gate_err = repr(exc)

    errs, bench_errs, raw_errs = [], [], []
    pred_ret, actual_ret, raw_ret = [], [], []
    stances, p_ups, a_dirs = [], [], []
    hits, raw_hits, evaluated = 0, 0, 0
    fit_failures = 0
    last_err = None
    for t in origins:
        tr_X, tr_p = features.iloc[:t], price.iloc[:t]
        try:
            # dir_origins=0：逐原点只训练最终方向分类器，门控/校准复用 gate_fc（提速）
            fc = ShortTermForecaster(horizon=horizon, compute_residuals=True,
                                     calib_origins=calib_origins,
                                     calib_iter=calib_iter,
                                     run_arima=False, dir_origins=0).fit(tr_X, tr_p)
            if gate_fc is not None:
                fc.dir_calib = dict(gate_fc.dir_calib)
                fc.dir_edge = dict(gate_fc.dir_edge)
            step_model = fc.models[horizon]
            fit_cols = getattr(step_model, "_oilcast_fit_cols", None)
            if fit_cols is None:
                fit_cols = getattr(fc, "active_cols", None)
            x_now = (tr_X[fit_cols] if fit_cols is not None else tr_X).iloc[[-1]]
            raw = float(step_model.predict(x_now)[0])
            beta = float(fc.calib_beta.get(horizon, getattr(fc, "point_shrink", 0.9)))
            cum_pred = beta * raw
            p_up, stance, _ = fc.direction_at(tr_X.iloc[[-1]], horizon)
        except Exception as exc:  # 单个原点失败不拖垮整体，但计数并保留原因，杜绝静默归零
            fit_failures += 1
            last_err = repr(exc)
            continue
        p_now, p_fut = price.iloc[t], price.iloc[t + horizon]
        if pd.isna(p_now) or pd.isna(p_fut):
            continue   # 真实价格缺口处不计算回测误差，不用填充值
        actual = float(np.log(p_fut) - np.log(p_now))
        errs.append(abs(np.exp(cum_pred) - np.exp(actual)) / np.exp(actual) * 100)
        raw_errs.append(abs(np.exp(raw) - np.exp(actual)) / np.exp(actual) * 100)
        bench_errs.append(abs(1 - np.exp(actual)) * 100)   # random walk: 预测=当前价
        pred_ret.append(cum_pred)
        actual_ret.append(actual)
        raw_ret.append(raw)
        if np.sign(cum_pred) == np.sign(actual) and actual != 0:
            hits += 1
        if np.sign(raw) == np.sign(actual) and actual != 0:
            raw_hits += 1
        stances.append(stance)
        p_ups.append(float(p_up))
        a_dirs.append(int(actual > 0))
        evaluated += 1
    if evaluated < 10:
        diag = f"；{fit_failures} 个原点拟合/预测失败" if fit_failures else ""
        if fit_failures and last_err:
            diag += f"，末例错误：{last_err[:160]}"
        return {"available": False,
                "reason": f"有效回测原点不足({evaluated}<10){diag}"}
    mae = float(np.mean(errs))
    bench_mae = float(np.mean(bench_errs))
    raw_mae = float(np.mean(raw_errs))
    if len(pred_ret) >= 3 and np.std(pred_ret) > 0 and np.std(actual_ret) > 0:
        ic = float(np.corrcoef(pred_ret, actual_ret)[0, 1])
    else:
        ic = 0.0
    # 事后中性带（仅用于把原点按【幅度】分为出击/中性，作参考，不参与拟合）
    pr = np.asarray(pred_ret); ar = np.asarray(actual_ret); rr = np.asarray(raw_ret)
    band = float(np.median(np.abs(ar))) * 0.5 if len(ar) else 0.0
    engaged_amp = np.abs(pr) >= band
    amp_eng_n = int(engaged_amp.sum())
    amp_eng_hits = int(((np.sign(pr[engaged_amp]) == np.sign(ar[engaged_amp]))
                        & (ar[engaged_amp] != 0)).sum())

    # —— 三分类方向口径（核心）——
    st = np.asarray(stances)
    pu = np.asarray(p_ups)
    ad = np.asarray(a_dirs)
    engaged = st != "中性"
    eng_n = int(engaged.sum())
    bull_correct = ((st == "看涨") & (ad == 1))
    bear_correct = ((st == "看跌") & (ad == 0))
    eng_correct = int((bull_correct | bear_correct)[engaged].sum()) if eng_n else 0
    eng_hit = eng_correct / eng_n if eng_n else None
    eng_p = float(binomtest(eng_correct, eng_n, 0.5).pvalue) if eng_n else None
    brier = float(np.mean((pu - ad) ** 2))
    clf_all_hit = float(np.mean(((pu >= 0.5) == (ad == 1))))
    # 门控证据（回测起点前估计，无泄漏）
    gate_edge = {}
    if gate_fc is not None and getattr(gate_fc, "dir_edge", None):
        anchors = sorted(gate_fc.dir_edge.keys())
        ha = min(anchors, key=lambda k: abs(k - horizon))
        gate_edge = gate_fc.dir_edge.get(ha, {})
    return {
        "available": True,
        # 幅度误差
        "mae_pct": round(mae, 2),
        "rmse_pct": round(float(np.sqrt(np.mean(np.square(errs)))), 2),
        "raw_mae_pct": round(raw_mae, 2),
        "benchmark_mae_pct": round(bench_mae, 2),
        "ic": round(ic, 3),
        "beat_random": bool(mae < bench_mae),
        "mae_improve_pct": round((1 - mae / bench_mae) * 100, 1) if bench_mae > 0 else 0.0,
        # 三分类方向（核心，页面展示这套）
        "stance_engagement_rate": round(eng_n / evaluated, 3),
        "stance_neutral_rate": round(1 - eng_n / evaluated, 3),
        "stance_engaged_accuracy": round(eng_hit, 3) if eng_hit is not None else None,
        "stance_engaged_n": eng_n,
        "stance_engaged_p": round(eng_p, 3) if eng_p is not None else None,
        "direction_brier": round(brier, 3),
        "clf_direction_accuracy": round(clf_all_hit, 3),
        "gate_has_edge": bool(gate_edge.get("has_edge", False)),
        "gate_edge_hit": gate_edge.get("engaged_hit"),
        "gate_edge_p": gate_edge.get("engaged_p"),
        "gate_edge_n": gate_edge.get("n"),
        # 参考/向后兼容字段（页面不再以其为主口径）
        "raw_direction_accuracy": round(raw_hits / evaluated, 3),
        "direction_accuracy": round(hits / evaluated, 3),
        "neutral_band": round(float(band), 4),
        "engagement_rate": round(amp_eng_n / evaluated, 3),
        "engaged_direction_accuracy": round(amp_eng_hits / amp_eng_n, 3) if amp_eng_n else None,
        "engaged_n": amp_eng_n,
        "n_origins": evaluated,
        "horizon_td": horizon,
        "window_start": features.index[origins[0]].strftime("%Y-%m-%d"),
        "window_end": features.index[origins[-1]].strftime("%Y-%m-%d"),
        "gate_error": gate_err,
    }
