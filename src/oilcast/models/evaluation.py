"""滚动原点回测：量化短期模型的样本外表现，并与随机游走基准对比。

旧版只在最近 8 个、间隔 5 天的原点上评估（全部挤在最近 40 个交易日），方向命中率
粒度仅 12.5%，撞上某段高波动行情就会出现"50%、跑输随机游走"的偶然结论。现改为在
最近一段更长的窗口内取多个原点（默认约 42 个、间隔 3 个交易日，覆盖最近约半年），
且每个原点都用与生产完全一致的【滚动训练窗】拟合，指标统计上才可信、可复现。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .short_term import ShortTermForecaster


def backtest_short(features: pd.DataFrame, price: pd.Series,
                   horizon: int = 10, n_origins: int = 42, gap: int = 3,
                   calib_origins: int = None, calib_iter: int = None) -> dict:
    """在最近 n_origins 个滚动原点上做严格样本外 horizon 日预测评估。

    每个原点 t：仅用 ≤t 的数据、按生产口径（滚动训练窗）拟合，在 t 预测 t+horizon，
    再与 t+horizon 的真实价格比较；随机游走基准为"价格保持不变"。真实价格缺口处跳过，
    绝不用填充值计算误差。

    返回 MAE/RMSE（百分比价格误差）、方向命中率、随机游走基准 MAE、收益相关 IC，
    以及是否跑赢随机游走。
    """
    n = len(features)
    if n < 260:
        return {"available": False, "reason": f"样本不足({n}<260)"}
    origins = list(range(n - horizon - n_origins * gap, n - horizon, gap))
    origins = [t for t in origins if t >= 260]
    errs, bench_errs, raw_errs = [], [], []
    pred_ret, actual_ret, raw_ret = [], [], []
    hits, raw_hits, evaluated = 0, 0, 0
    for t in origins:
        tr_X, tr_p = features.iloc[:t], price.iloc[:t]
        try:
            # 与生产同一模型类、同一滚动训练窗，并同样估计样本外校准系数 β；
            # 这样回测评估的是【部署后真实口径】（β 校准后的点预测），而非未校准原始输出。
            fc = ShortTermForecaster(horizon=horizon, compute_residuals=True,
                                     calib_origins=calib_origins,
                                     calib_iter=calib_iter,
                                     run_arima=False).fit(tr_X, tr_p)
            active = getattr(fc, "active_cols", None)
            x_now = (tr_X[active] if active is not None else tr_X).iloc[[-1]]
            raw = float(fc.models[horizon].predict(x_now)[0])
            beta = float(fc.calib_beta.get(horizon, getattr(fc, "point_shrink", 0.9)))
            cum_pred = beta * raw
        except Exception:
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
        evaluated += 1
    if evaluated < 10:
        return {"available": False, "reason": f"有效回测原点不足({evaluated}<10)"}
    mae = float(np.mean(errs))
    bench_mae = float(np.mean(bench_errs))
    raw_mae = float(np.mean(raw_errs))
    if len(pred_ret) >= 3 and np.std(pred_ret) > 0 and np.std(actual_ret) > 0:
        ic = float(np.corrcoef(pred_ret, actual_ret)[0, 1])
    else:
        ic = 0.0
    # 公允的方向口径：β→0 时点预测≈持平（模型主动退守、不表达方向）。若把"中性"也按
    # 二分法计为方向错误，会在震荡市系统性压低命中率。故以实际 h 期波动绝对值中位数的一半
    # 为中性带（仅用于事后把原点分为"出击/中性"，不参与拟合与预测，无泄漏）：
    #   engagement_rate = 模型明确表态(预测幅度≥中性带)的占比；
    #   engaged_direction_accuracy = 表态原点中的方向命中率（这才是可与抛硬币比较的量）。
    pr = np.asarray(pred_ret); ar = np.asarray(actual_ret); rr = np.asarray(raw_ret)
    band = float(np.median(np.abs(ar))) * 0.5 if len(ar) else 0.0
    engaged = np.abs(pr) >= band
    eng_n = int(engaged.sum())
    eng_hits = int(((np.sign(pr[engaged]) == np.sign(ar[engaged])) & (ar[engaged] != 0)).sum())
    raw_engaged = np.abs(rr) >= band
    raw_eng_n = int(raw_engaged.sum())
    raw_eng_hits = int(((np.sign(rr[raw_engaged]) == np.sign(ar[raw_engaged])) & (ar[raw_engaged] != 0)).sum())
    return {
        "available": True,
        "mae_pct": round(mae, 2),
        "rmse_pct": round(float(np.sqrt(np.mean(np.square(errs)))), 2),
        "raw_mae_pct": round(raw_mae, 2),
        "raw_direction_accuracy": round(raw_hits / evaluated, 3),
        "benchmark_mae_pct": round(bench_mae, 2),
        "direction_accuracy": round(hits / evaluated, 3),
        "neutral_band": round(float(band), 4),
        "engagement_rate": round(eng_n / evaluated, 3),
        "engaged_direction_accuracy": round(eng_hits / eng_n, 3) if eng_n else None,
        "engaged_n": eng_n,
        "raw_engagement_rate": round(raw_eng_n / evaluated, 3),
        "raw_engaged_direction_accuracy": round(raw_eng_hits / raw_eng_n, 3) if raw_eng_n else None,
        "ic": round(ic, 3),
        "beat_random": bool(mae < bench_mae),
        "mae_improve_pct": round((1 - mae / bench_mae) * 100, 1) if bench_mae > 0 else 0.0,
        "n_origins": evaluated,
        "horizon_td": horizon,
        "window_start": features.index[origins[0]].strftime("%Y-%m-%d"),
        "window_end": features.index[origins[-1]].strftime("%Y-%m-%d"),
    }
