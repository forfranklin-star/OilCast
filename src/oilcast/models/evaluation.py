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
from scipy.stats import binomtest, ttest_1samp

from ..config import get_config
from .short_term import ShortTermForecaster


def _econ_summary(rs, horizon: int) -> dict:
    """一组【按立场取号后的持有期对数收益】的经济价值统计。

    mean 平均对数收益(%)、payoff 盈亏比(平均盈利/平均亏损)、p 单样本 t 检验(相对0)、
    sharpe 年化近似夏普(按 252/horizon 折算)、cum 累计对数收益(%)。样本不足给 None。"""
    rs = np.asarray(rs, dtype=float)
    n = int(len(rs))
    if n < 3:
        return {"n": n, "mean_pct": None, "payoff": None, "p": None,
                "sharpe": None, "cum_pct": None}
    wins, losses = rs[rs > 0], rs[rs < 0]
    payoff = (float(wins.mean() / abs(losses.mean()))
              if len(wins) and len(losses) else np.nan)
    p = float(ttest_1samp(rs, 0.0).pvalue)
    std = float(rs.std(ddof=1)) if n >= 2 else 0.0
    sharpe = float(rs.mean() / std * np.sqrt(252.0 / horizon)) if std > 0 else 0.0
    return {"n": n, "mean_pct": round(float(rs.mean()) * 100, 3),
            "payoff": round(payoff, 3) if np.isfinite(payoff) else None,
            "p": round(p, 3), "sharpe": round(sharpe, 3),
            "cum_pct": round(float(rs.sum()) * 100, 2)}


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

    _mcfg = get_config()["model"]
    _thr = float(_mcfg.get("dir_conf_threshold", 0.62))
    _recent_span = int(_mcfg.get("edge_recent_span", 250))
    _recent_min = int(_mcfg.get("edge_recent_min_eng", 6))
    _recent_p = float(_mcfg.get("edge_recent_max_p", 0.30))
    _k = max(_recent_min, int(np.ceil(_recent_span / horizon)))

    # 门控参考模型：只用第一个回测原点【之前】的数据估计长历史方向 edge（无泄漏，慢变沿用）
    gate_fc = None
    gate_err = None
    try:
        gate_fc = ShortTermForecaster(
            horizon, compute_residuals=False, run_arima=False,
            dir_origins=dir_origins).fit(features.iloc[:origins[0]],
                                         price.iloc[:origins[0]])
    except Exception as exc:  # 门控参考失败不拖垮幅度回测，方向层降级为全中性
        gate_err = repr(exc)

    # 滚动门控探针：在全样本上跑一次【密集】walk-forward，样本外原点须覆盖整个回测窗口并向
    # 前多铺 recent_span 个交易日，使每个回测原点 t 都能取到"最近约 1 年、且 i+h<=t 已实现"
    # 的非重叠点做近端正向确认（原点太稀疏会把近端窗口稀释成长历史、近期反转被摊薄而漏判）。
    # 每个探针点的分类器只用该点之前数据训练，探针内部无泄漏。
    probe = None
    probe_origins = int(np.ceil((_recent_span + n_origins * gap) / horizon)) + 4
    try:
        probe = ShortTermForecaster(horizon, compute_residuals=False, run_arima=False,
                                    dir_origins=probe_origins).fit(features, price)
    except Exception as exc:  # 探针失败退回固定门控，不拖垮幅度回测
        gate_err = (gate_err or "") + f"; 滚动门控探针失败:{repr(exc)[:120]}"

    errs, bench_errs, raw_errs = [], [], []
    pred_ret, actual_ret, raw_ret = [], [], []
    stances, p_ups, a_dirs = [], [], []
    hits, raw_hits, evaluated = 0, 0, 0
    fit_failures = 0
    n_long_edge, n_recent_break = 0, 0   # 长历史ML edge成立原点 / 其中被近端熔断原点
    last_err = None
    for t in origins:
        # 与生产 predict 口径严格一致：训练用 ≤t（含第 t 行，其 h 日标签为未来、被 shift(-h)
        # 剔除故不参与拟合，无泄漏），喂第 t 行特征，预测 t→t+h；actual 也从 price[t] 起算。
        # 旧实现误用 iloc[:t]（止于 t-1）并喂第 t-1 行，actual 却从 t 起，预测与实际错位 1 个
        # 交易日，在拐点行情会系统性拉低方向命中，已修正。
        tr_X, tr_p = features.iloc[:t + 1], price.iloc[:t + 1]
        try:
            # dir_origins=0：逐原点只训练最终方向分类器，门控/校准复用 gate_fc（提速）
            fc = ShortTermForecaster(horizon=horizon, compute_residuals=True,
                                     calib_origins=calib_origins,
                                     calib_iter=calib_iter,
                                     run_arima=False, dir_origins=0).fit(tr_X, tr_p)
            # 校准与长历史 edge 沿用 gate_fc（回测起点前数据拟合，无泄漏、慢变）。
            # 校准映射必须稳定：若在每个原点用最近一二十个探针点小样本重拟合 isotonic，
            # 反转期它会"在线自适应"把概率拉回、抵消并绕过熔断；生产校准本就是长历史、
            # 周期性（约每周）刷新，故回测统一用起点前的长历史校准，才能如实暴露
            # "按长期口径本应高置信、近期却稳定反向/不赚钱"的失效。
            long_calib = None
            if gate_fc is not None:
                fc.dir_calib = dict(gate_fc.dir_calib)
                fc.dir_edge = dict(gate_fc.dir_edge)
                fc.trend_edge = dict(getattr(gate_fc, "trend_edge", {}))
                long_calib = gate_fc.dir_calib.get(horizon)
            # 滚动近端正向确认：用探针中【截至 t 已实现】(i+h<=t) 的样本外点，取最近约
            # recent_span 个非重叠点，按【长历史校准后概率】判高置信表态，再检验其近端
            # 是否仍赚钱/命中过半；不满足即熔断退回中性。长历史 edge 沿用 gate_fc。
            if probe is not None and gate_fc is not None:
                _raw_pts = getattr(probe, "gate_pa_raw", {}).get(horizon, [])
                pts = [x for x in _raw_pts if x[0] + horizon <= t]
                if len(pts) >= _recent_min:
                    rk = pts[-_k:]
                    rrp = np.array([x[1] for x in rk], dtype=float)
                    rad = np.array([x[2] for x in rk], dtype=int)
                    rfr = np.array([x[3] for x in rk], dtype=float)
                    crp = long_calib.predict(rrp) if long_calib is not None else rrp
                    rsign = np.where(crp >= _thr, 1, np.where(crp <= 1 - _thr, -1, 0))
                    rec = ShortTermForecaster._recent_failure(
                        rsign, rfr, _recent_min, _recent_p, ad=rad)
                    _g = gate_fc.dir_edge.get(horizon, {})
                    ml_long = bool(_g.get("ml_long_edge", False))
                    if ml_long:
                        n_long_edge += 1
                        if not rec["recent_ok"]:
                            n_recent_break += 1
                    edge_t = dict(_g)
                    edge_t.update({"anchor": horizon,
                                   "has_edge": bool(ml_long and rec["recent_ok"]),
                                   "ml_recent_ok": rec["recent_ok"],
                                   "ml_recent_engaged_n": rec["recent_engaged_n"],
                                   "ml_recent_hit": rec["recent_hit"],
                                   "ml_recent_mean_ret": rec["recent_mean_ret"],
                                   "ml_recent_reason": rec["recent_reason"]})
                    fc.dir_edge[horizon] = edge_t
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
    # —— 经济价值口径：胜率≠盈利能力。按表态方向取号持有 horizon，统计期望收益/盈亏比/
    # 近似夏普；中性原点不持仓。另给"无条件买入持有"作对照，证明表态是否真有超额价值。
    dir_sign = np.where(st == "看涨", 1.0, np.where(st == "看跌", -1.0, 0.0))
    econ_engaged = _econ_summary((dir_sign * ar)[engaged], horizon)
    econ_buyhold = _econ_summary(ar, horizon)
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
        "gate_edge_basis": gate_edge.get("edge_basis"),
        "gate_edge_hit": gate_edge.get("engaged_hit"),
        "gate_edge_p": gate_edge.get("engaged_p"),
        "gate_edge_n": gate_edge.get("n"),
        # 经济价值（表态子集按立场持有 horizon；买入持有为对照）
        "stance_engaged_mean_ret_pct": econ_engaged["mean_pct"],
        "stance_engaged_payoff": econ_engaged["payoff"],
        "stance_engaged_ret_p": econ_engaged["p"],
        "stance_engaged_sharpe": econ_engaged["sharpe"],
        "stance_engaged_cum_ret_pct": econ_engaged["cum_pct"],
        "buyhold_mean_ret_pct": econ_buyhold["mean_pct"],
        "buyhold_sharpe": econ_buyhold["sharpe"],
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
        # 近端失效熔断透明化：长历史 ML edge 成立的原点中，有多少因近期按表态稳定
        # 亏钱/显著反向被熔断退回中性（regime 自适应，区别于"长历史本就无 edge"）。
        "long_edge_origins": n_long_edge,
        "recent_break_origins": n_recent_break,
        "gate_error": gate_err,
    }
