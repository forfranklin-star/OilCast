"""多时间尺度（5 / 10 / 21 交易日）概率预测与自我进化闭环。

这是系统的"学习中枢"，对 **WTI / Brent** 两个核心品种实现完整工作环：

    预测  →  回测批判  →  根因反思  →  策略进化

1. 预测：用 horizon=21 的 ShortTermForecaster（direct 多步模型覆盖 1..21），一次
   产出 5 / 10 / 21 三个交易日尺度的点预测、概率与置信区间，落库 forecasts
   （horizon = h5 / h10 / h21）。模型以独立工件标签 multi21 持久化、跨期热启动。
2. 回测批判：把 h5/h10/h21 的历史预测与现已实现的真实价格逐条对比，量化
   误差、明确表态的方向命中、95% 区间覆盖（只用真实价格，目标日无真实观测不评估）。
3. 根因反思：基于复测统计量客观判定失败/成功模式（方向 edge 是否成立、区间尾部
   是否低估），不通过任何外部 if 掰数字。
4. 策略进化：把结论写回机器可读、带版本、可回溯的策略状态库：
     strategy/strategy_state.json   —— 证据权重 / 校准参数 / 假设库（每轮覆盖、版本递增）
     strategy/learning_journal.jsonl —— 每轮一条、append-only 的审计轨迹
     strategy/STRATEGY.md           —— 由状态自动渲染的人读策略文档
   证据不足/失效的信号自动降级，新证据增强才升级；假设库据检验结果更新状态。

数据原则与全系统一致：只用真实、可追溯、带观测日期的数据，缺失不补齐、不合成。
"""
from __future__ import annotations
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

from ..config import _project_root, get_config
from ..data_sources.calendar import next_trading_days
from ..features.engineering import _load_major_events
from ..models.registry import load_artifact, save_artifact
from ..models.review import _nearest_real
from ..models.short_term import MULTI_HORIZONS, ShortTermForecaster
from ..utils import get_logger, now_beijing, safe_log

LOG = get_logger(__name__)
ART_TAG = "multi21"
HS: List[int] = list(MULTI_HORIZONS)          # [5, 10, 21]
H_TAGS = [f"h{h}" for h in HS]
INSTS = ("wti", "brent")
INST_CN = {"wti": "WTI", "brent": "布伦特"}
# 门控/反思阈值（可在 config.model 覆盖；这里给默认，避免魔法数字散落）
MIN_ENGAGE_TEST = 5        # 方向 edge 检验所需最少明确表态样本
SIG_P = 0.10               # 方向命中/失效的显著性水平
COV_MIN = 90.0             # 95% 区间应达到的实际覆盖下限（容差后）
COV_N_MIN = 8              # 覆盖率检验所需最少到期样本
SCORE_PRIOR_N = 12         # 证据分的样本量收缩先验
COV_TARGET = 0.95          # 区间名义目标覆盖
WIDTH_MULT_CAP = (0.7, 1.8)  # 宽度乘子的允许范围（兜底，正常校准不应触及）
WIDTH_PRIOR_N = 10         # 宽度乘子更新的样本量收缩先验
# 区间基础临界值（正态）：95% ±1.96、50% ±0.674；厚尾/σ偏差统一由在线 width_mult 吸收
Z95 = float(norm.ppf(0.975))
Z50 = float(norm.ppf(0.875))     # 双侧 50%：Φ^{-1}(.875)≈0.674
# 区间引擎参数（经真机 300 原点 walk-forward 校准，覆盖 91~94%）
EWMA_LAM = 0.94                  # 条件波动率 EWMA 衰减（RiskMetrics，对冲击快速反应）
JUMP_K = 3                       # 事件爆发窗（交易日）
JQ_LEVEL = 0.85                  # 爆发幅度分位
ZROLL = 250                      # 短尺度标准化分位的滚动窗（长尺度用全部历史）


def _build_interval_engine(price_series):
    """构建区间引擎，处理三层不确定性（均由真实数据估计）：

      1. 条件波动率 σ：日对数收益方差 EWMA，波动 regime 转换即时抬升（非无条件窗 std）；
      2. 厚尾临界：标准化 h 日收益 z_h=r_h/(σ_start√h) 的经验分位（去波动率后近似 iid）；
      3. 事件跳跃保险：年表起始频率 λ 与爆发窗超额幅度 J，未来 h 日按 P=1-e^{-λh} 事前
         加尾部（平静时也保留，覆盖不可预见的突发）。
    返回 (logp, half_width(p,h,mult)->(h95,h50 对数半宽), diag)。
    """
    s = price_series.dropna()
    logp = safe_log(s)
    r1 = logp.diff()
    v = max(float(r1.iloc[:30].var()), 1e-8)
    sig_arr = np.zeros(len(logp))
    r1v = r1.to_numpy()
    for i in range(len(logp)):
        sig_arr[i] = v
        if not np.isnan(r1v[i]):
            v = EWMA_LAM * v + (1 - EWMA_LAM) * r1v[i] ** 2
    sig = pd.Series(np.sqrt(sig_arr), index=logp.index)

    ev = _load_major_events() or []
    starts = sorted({pd.Timestamp(e["start"]) for e in ev})
    if len(starts) >= 2:
        yrs = (max(starts) - min(starts)).days / 365.25
        lam = len(starts) / max(yrs, 1e-6)
    else:
        lam = 0.0
    J = []
    for d in starts:
        i0 = logp.index.searchsorted(d)
        if i0 < 1 or i0 + JUMP_K >= len(logp):
            continue
        J.append(float(logp.iloc[i0 + JUMP_K] - logp.iloc[i0 - 1]))
    Jq = float(np.quantile(np.abs(J), JQ_LEVEL)) if J else 0.0

    # 统一日度标准化收益 z_d=r_d/σ（去波动后近似 iid、样本最多）；h 日宽度 = qz×σ√h，
    # 尺度间严格 ∝√h（单调、不倒挂）。厚尾由 qz>1.96 吸收，突发由跳跃保险吸收。
    z_d = r1 / sig.shift(1)

    def half_width(p: int, h: int, mult: float):
        zw = z_d.iloc[max(0, p - ZROLL):p].dropna()
        if len(zw) < 30:
            zw = z_d.iloc[:p].dropna()
        if len(zw) >= 10:
            qz95, qz50 = float(zw.quantile(0.95)), float(zw.quantile(0.75))
        else:                          # 极端样本不足：退回正态临界
            qz95, qz50 = Z95, Z50
        scur = float(sig.iloc[p])
        P = 1.0 - float(np.exp(-lam * h / 252.0))
        jump = P * Jq
        h95 = qz95 * scur * np.sqrt(h) * mult + jump
        h50 = qz50 * scur * np.sqrt(h) * mult
        return h95, h50, P

    diag = {"lam": round(lam, 2), "Jq": round(Jq, 3), "n_jump": len(J)}
    return logp, half_width, diag


def _coverage_error(k: int, n: int):
    """返回 (target−p_hat, p_hat)：p_hat 为覆盖率的 Jeffreys 后验均值 Beta(k+.5,n-k+.5)。

    相比 (k+2)/(n+4)，Jeffreys 均值不把点估计系统性拉向中间：真实覆盖 95%（k≈.95n）
    时 p_hat≈.94、误差≈0、校准自动停止；覆盖 100% 时 p_hat≈.98、误差为负（收窄）；
    覆盖低时误差为正（加宽）。信号以名义覆盖为中心、双向对称，杜绝单边过冲。
    """
    if n <= 0:
        return 0.0, None
    p_hat = (k + 0.5) / (n + 1.0)
    return COV_TARGET - p_hat, p_hat


# ----------------------------------------------------------- 路径与状态
def strategy_dir() -> Path:
    p = _project_root() / "strategy"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _initial_assumptions() -> List[dict]:
    A: List[dict] = []
    for inst in INSTS:
        c = INST_CN[inst]
        for h in HS:
            A.append({
                "id": f"dir_{inst}_{h}", "kind": "direction",
                "statement": f"{c} {h} 交易日方向存在统计可预测 edge（表态命中显著>50%）",
                "first_tested": None, "last_tested": None,
                "status": "pending", "evidence": None})
        A.append({
            "id": f"cov_{inst}_21", "kind": "coverage",
            "statement": f"{c} 95% 区间在 21 交易日实际覆盖 ≥ 90%",
            "first_tested": None, "last_tested": None,
            "status": "pending", "evidence": None})
    return A


def _initial_state() -> dict:
    ts = now_beijing().strftime("%Y-%m-%d %H:%M:%S")
    return {"version": 0, "created_at": ts, "updated_at": ts,
            "evidence": {}, "calibration": {},
            "assumptions": _initial_assumptions()}


def _load_state() -> dict:
    p = strategy_dir() / "strategy_state.json"
    if p.exists():
        try:
            st = json.loads(p.read_text(encoding="utf-8"))
            st.setdefault("assumptions", _initial_assumptions())
            return st
        except Exception as exc:
            LOG.warning("策略状态读取失败，按初始状态重建：%s", exc)
    return _initial_state()


# ----------------------------------------------------------- 1) 多尺度预测
def _forecast_one(inst: str, feats: dict, prices, calendars, last_real_date,
                  rolling_window, persist_on,
                  dir_origins: Optional[int] = None,
                  calib_origins: Optional[int] = None,
                  width_mults: Optional[dict] = None) -> dict:
    warm_obj, prev_entry = (load_artifact(ART_TAG, inst) if persist_on
                            else (None, None))
    fc = ShortTermForecaster(horizon=21, window=int(rolling_window),
                             compute_residuals=True, dir_origins=dir_origins,
                             calib_origins=calib_origins)
    fc.fit(feats[inst], prices[inst], warm=warm_obj)
    last_dt = last_real_date[inst]
    last_price = float(prices[inst].loc[last_dt])
    future = next_trading_days(last_dt, 21, calendars[inst]["hist"],
                               calendars[inst]["closed"])
    res = fc.predict(feats[inst].loc[[last_dt]], last_price, future)
    if persist_on:
        try:
            save_artifact(ART_TAG, inst, fc,
                          meta=getattr(fc, "train_meta", {}),
                          warm_from=prev_entry if fc.warm_info["warm_started"] else None)
        except Exception as exc:
            LOG.warning("多尺度工件保存失败(%s)，不影响闭环：%s", inst, exc)
    width_mults = width_mults or {}
    # 区间引擎：EWMA 条件波动率 + 标准化厚尾分位 + 事件跳跃保险（只用真实价格）
    elogp, half_width, jdiag = _build_interval_engine(prices[inst])
    asof_pos = int(elogp.index.get_loc(last_dt))
    x_base = feats[inst].loc[[last_dt]]
    cards = {}
    records = []
    path = res.path
    for h in HS:
        row = path.iloc[h - 1]
        p, stance, edge = fc.direction_at(x_base, h)
        tgt = pd.Timestamp(row.name).strftime("%Y-%m-%d")
        mean = float(row["mean"])
        mult = float(width_mults.get((inst, h), 1.0))
        h95, h50, Pj = half_width(asof_pos, h, mult)
        q05, q95 = mean * float(np.exp(-h95)), mean * float(np.exp(h95))
        q25, q75 = mean * float(np.exp(-h50)), mean * float(np.exp(h50))
        card = {
            "h": h, "target_date": tgt, "stance": stance,
            "prob_up": round(p, 3), "prob_down": round(1 - p, 3),
            "mean": round(mean, 2),
            "q05": round(q05, 2), "q25": round(q25, 2),
            "q75": round(q75, 2), "q95": round(q95, 2),
            "pct_mean": round((mean / last_price - 1) * 100, 2),
            "width_mult": round(mult, 3), "hazard": round(Pj, 2),
            "has_edge": bool(edge.get("has_edge", False)),
            "basis": edge.get("edge_basis")}
        cards[h] = card
        records.append({
            "horizon": f"h{h}", "instrument": inst, "target_date": tgt,
            "mean": mean, "q05": q05, "q25": q25,
            "q50": mean, "q75": q75, "q95": q95,
            "prob_up": round(p, 3), "prob_down": round(1 - p, 3),
            "dir_stance": stance})
    return {"fc": fc, "cards": cards, "records": records,
            "observed_date": last_dt.strftime("%Y-%m-%d"),
            "warm_started": bool(fc.warm_info["warm_started"]),
            "interval_diag": jdiag}


# ----------------------------------------------------------- 2) 回测批判
def multi_scale_review(prices) -> Dict[str, dict]:
    """逐 (inst,h) 把到期多尺度预测与真实价格对比，返回复测指标（全部来自真实数据）。"""
    cfg = get_config()
    conn = sqlite3.connect(cfg["storage"]["sqlite_path"])
    try:
        df = pd.read_sql_query("SELECT * FROM forecasts", conn)
    finally:
        conn.close()
    df = df[df["horizon"].isin(H_TAGS) & df["instrument"].isin(INSTS)]
    out: Dict[str, dict] = {}
    for inst in INSTS:
        last_real = prices[inst].dropna().index.max()
        for h in HS:
            g = df[(df["instrument"] == inst) & (df["horizon"] == f"h{h}")]
            hits, covers, errs, n_eng = [], [], [], 0
            n_total = 0
            for _, r in g.iterrows():
                target = pd.Timestamp(r["target_date"])
                if pd.isna(target) or target > last_real:
                    continue
                p0 = _nearest_real(prices[inst], pd.Timestamp(r["report_date"]))
                actual = _nearest_real(prices[inst], target)
                mean = r["mean"]
                if p0 is None or actual is None or pd.isna(mean) or p0 <= 0:
                    continue
                n_total += 1
                actual_ret = actual / p0 - 1.0
                errs.append(abs(float(mean) - actual) / actual * 100)
                stance = r.get("dir_stance")
                if stance in ("看涨", "看跌") and actual_ret != 0:
                    n_eng += 1
                    ok = (actual_ret > 0) if stance == "看涨" else (actual_ret < 0)
                    hits.append(bool(ok))
                q05, q95 = r.get("q05"), r.get("q95")
                if q05 is not None and q95 is not None and not pd.isna(q05) \
                        and not pd.isna(q95):
                    covers.append(bool(q05 <= actual <= q95))
            n_hits = int(sum(hits))
            dir_p = float(binomtest(n_hits, n_eng, 0.5).pvalue) if n_eng else None
            out[f"{inst}_{h}"] = {
                "n_due": int(n_total), "engaged_n": int(n_eng),
                "dir_acc": round(float(np.mean(hits)) * 100, 1) if hits else None,
                "dir_p": round(dir_p, 3) if dir_p is not None else None,
                "coverage95": round(float(np.mean(covers)) * 100, 1)
                if covers else None,
                "cov_n": int(len(covers)),
                "mae_pct": round(float(np.mean(errs)), 2) if errs else None}
    return out


# ----------------------------------------------------------- 3) 根因反思
def diagnose(msr: Dict[str, dict]) -> List[dict]:
    """对每 (inst,h) 的复测统计量做客观判定，产出失败/成功/观察结论。"""
    findings: List[dict] = []

    def add(inst, h, dim, observed, bench, kind, verdict, n):
        findings.append({"inst": inst, "h": h, "dimension": dim,
                         "observed": observed, "benchmark": bench,
                         "kind": kind, "verdict": verdict, "n": n})

    for inst in INSTS:
        cn = INST_CN[inst]
        for h in HS:
            m = msr[f"{inst}_{h}"]
            en, acc, dp = m["engaged_n"], m["dir_acc"], m["dir_p"]
            # —— 方向 edge ——
            if en < MIN_ENGAGE_TEST:
                add(inst, h, "方向", f"表态样本 {en} 个", f"≥{MIN_ENGAGE_TEST}",
                    "watch", f"{cn}{h}日明确表态样本不足，方向 edge 尚未证实", en)
            elif acc is not None and acc < 50 and (dp is None or dp <= SIG_P):
                add(inst, h, "方向", f"表态命中 {acc}%、p={dp}", "50%",
                    "failure", f"{cn}{h}日表态命中不高于随机，方向 edge 失效", en)
            elif acc is not None and acc > 50 and dp is not None and dp <= SIG_P:
                add(inst, h, "方向", f"表态命中 {acc}%、p={dp}", "50%",
                    "success", f"{cn}{h}日方向 edge 获统计支持", en)
            else:
                add(inst, h, "方向", f"表态命中 {acc}%、p={dp}", "50%",
                    "watch", f"{cn}{h}日命中略偏但不显著，继续观察", en)
            # —— 区间覆盖 ——
            cov, cn_n = m["coverage95"], m["cov_n"]
            if cn_n < COV_N_MIN:
                add(inst, h, "区间", f"到期 {cn_n} 个", f"≥{COV_N_MIN}",
                    "watch", f"{cn}{h}日区间覆盖样本不足，暂不判定", cn_n)
            elif cov is not None and cov < COV_MIN:
                add(inst, h, "区间", f"95% 覆盖 {cov}%", f"≥{COV_MIN}%",
                    "failure", f"{cn}{h}日95%区间覆盖不足、尾部低估（需加宽）", cn_n)
            elif cov is not None:
                add(inst, h, "区间", f"95% 覆盖 {cov}%", f"≥{COV_MIN}%",
                    "success", f"{cn}{h}日95%区间覆盖可靠", cn_n)
    return findings


# ----------------------------------------------------------- 证据分（估计量）
def _dir_edge_score(hit: Optional[float], n: int) -> float:
    """方向证据强度：样本量收缩 × 归一化胜率 edge，取值 0..1（n 小/无 edge 自然→0）。"""
    if not n or hit is None:
        return 0.0
    shrink = n / (n + SCORE_PRIOR_N)
    return round(float(shrink * max(0.0, hit / 100.0 - 0.5) * 2.0), 3)


def _trend_score(te: dict) -> float:
    """趋势通道证据分：以盈亏比与平均收益构造（趋势靠盈亏比而非胜率）。"""
    if not te.get("has_edge"):
        return 0.0
    payoff = te.get("payoff") or 0.0
    mean = te.get("mean_ret") or 0.0
    n = te.get("n") or 0
    shrink = n / (n + SCORE_PRIOR_N)
    return round(float(shrink * min(1.0, max(0.0, mean) / 2.0)
                       * min(2.0, max(0.0, payoff)) / 2.0 * 2.0), 3)


# ----------------------------------------------------------- 4) 策略进化
def evolve(msr, findings, fitted: Dict[str, object]) -> dict:
    state = _load_state()
    mcfg = get_config()["model"]
    actions: List[str] = []

    for inst in INSTS:
        fc = fitted.get(inst)
        for h in HS:
            m = msr[f"{inst}_{h}"]
            # —— 证据权重 ——
            if fc is not None:
                de = getattr(fc, "dir_edge", {}).get(h, {})
                te = getattr(fc, "trend_edge", {}).get(h, {})
                ml_hit = de.get("engaged_hit")
                # engaged_hit 在 de 中为比例(0..1)，证据分用百分比口径，统一换算
                hit_pct = round(ml_hit * 100, 1) if ml_hit is not None else None
                ml = {
                    "has_edge": bool(de.get("has_edge", False)),
                    "ml_long_edge": bool(de.get("ml_long_edge", False)),
                    "engaged_n": de.get("engaged_n"),
                    "engaged_hit": hit_pct, "engaged_p": de.get("engaged_p"),
                    "all_hit": de.get("all_hit"),
                    "score": _dir_edge_score(hit_pct, de.get("engaged_n") or 0),
                    "status": ("enabled" if de.get("has_edge") else
                               ("watch" if (de.get("engaged_n") or 0) > 0
                                else "disabled"))}
                tr = {
                    "has_edge": bool(te.get("has_edge", False)),
                    "hit": te.get("hit"), "mean_ret": te.get("mean_ret"),
                    "payoff": te.get("payoff"), "p": te.get("p"),
                    "score": _trend_score(te),
                    "status": "enabled" if te.get("has_edge") else "disabled"}
            else:
                ml = {"has_edge": False, "engaged_n": m["engaged_n"],
                      "engaged_hit": m["dir_acc"], "engaged_p": m["dir_p"],
                      "score": _dir_edge_score(m["dir_acc"], m["engaged_n"]),
                      "status": "disabled"}
                tr = {"has_edge": False, "score": 0.0, "status": "disabled"}
            state.setdefault("evidence", {}).setdefault(inst, {})[str(h)] = \
                {"ml": ml, "trend": tr}
            # —— 校准参数（版本化记录）——
            beta = None
            if fc is not None:
                beta = getattr(fc, "calib_beta", {}).get(h)
            # 区间宽度乘子：用上一轮值做 EMA 起点，按本轮样本外覆盖偏差收缩更新
            old_mult = float(state.get("calibration", {})
                             .get(f"{inst}_{h}", {}).get("width_mult", 1.0))
            cov_n = int(m["cov_n"])
            if cov_n > 0 and m["coverage95"] is not None:
                k = int(round(m["coverage95"] / 100.0 * cov_n))
                err, _ = _coverage_error(k, cov_n)
                shrink = cov_n / (cov_n + WIDTH_PRIOR_N)
                # 指数步长、双向对称：误差正(过窄)加宽、负(过宽)收窄，覆盖≈95 时停止
                new_mult = old_mult * float(np.exp(shrink * err))
                new_mult = min(WIDTH_MULT_CAP[1], max(WIDTH_MULT_CAP[0],
                                                       new_mult))
            else:
                new_mult = old_mult
            state.setdefault("calibration", {})[f"{inst}_{h}"] = {
                "point_beta": round(float(beta), 3) if beta is not None else None,
                "dir_halflife": float(mcfg.get("dir_calib_halflife", 120)),
                "dir_prior_n": float(mcfg.get("dir_calib_prior_n", 12)),
                "dir_band": float(mcfg.get("dir_calib_band", 0.13)),
                "width_mult": round(new_mult, 3),
                "oos_mae_pct": m["mae_pct"],
                "oos_cov95": m["coverage95"]}
            if abs(new_mult - old_mult) >= 0.02:
                actions.append(f"{INST_CN[inst]}{h}日 区间宽度乘子 "
                               f"{round(old_mult,3)} → {round(new_mult,3)}"
                               f"（经验覆盖 {m['coverage95']}%）")
            # 变更动作（与上轮比较）
            prev = state.get("evidence", {}).get(inst, {}).get(str(h))
            tag = f"{INST_CN[inst]}{h}日"
            actions.append(f"{tag} ML 证据分 {ml['score']}（{ml['status']}）、"
                           f"趋势 {tr['score']}（{tr['status']}）")

    # —— 假设库状态更新 ——
    today = now_beijing().strftime("%Y-%m-%d")
    fmap = {(f["inst"], f["h"], f["dimension"]): f for f in findings}
    for a in state["assumptions"]:
        kind = a["kind"]
        if kind == "direction":
            parts = a["id"].split("_")
            inst, h = parts[1], int(parts[2])
            f = fmap.get((inst, h, "方向"))
        else:
            parts = a["id"].split("_")
            inst = parts[1]
            f = fmap.get((inst, 21, "区间"))
        if f is None:
            continue
        new_status = {"success": "supported", "failure": "refuted",
                      "watch": "weakened"}[f["kind"]]
        # 覆盖类"weakened"表示尾部低估、方向类样本不足维持 pending
        if f["kind"] == "watch" and kind == "direction" and f["n"] < MIN_ENGAGE_TEST:
            new_status = "pending"
        if a["status"] != new_status:
            actions.append(f"假设[{a['id']}] {a['status']} → {new_status}")
        a["status"] = new_status
        a["last_tested"] = today
        a["first_tested"] = a["first_tested"] or today
        a["evidence"] = f["observed"]

    state["version"] = int(state.get("version", 0)) + 1
    state["updated_at"] = now_beijing().strftime("%Y-%m-%d %H:%M:%S")
    return {"state": state, "actions": actions}


# ----------------------------------------------------------- STRATEGY.md
def render_strategy_md(state: dict) -> str:
    L = ["# OilCast 预测策略状态（自动生成，勿手改）", "",
         f"- 状态版本：v{state['version']}　更新时间：{state['updated_at']}",
         "- 闭环：预测 → 回测批判 → 根因反思 → 策略进化（每轮自动）", ""]
    L.append("## 一、证据权重（按品种 / 尺度）")
    L.append("")
    L.append("| 品种 | 尺度 | ML方向 状态 | ML证据分 | 趋势 状态 | 趋势证据分 |")
    L.append("|---|---|---|---|---|---|")
    for inst in INSTS:
        for h in HS:
            e = state.get("evidence", {}).get(inst, {}).get(str(h), {})
            ml, tr = e.get("ml", {}), e.get("trend", {})
            L.append(f"| {INST_CN[inst]} | {h}日 | {ml.get('status')} | "
                     f"{ml.get('score')} | {tr.get('status')} | {tr.get('score')} |")
    L.append("")
    L.append("## 二、校准参数（点预测 β / 方向概率收缩）")
    L.append("")
    L.append("| 键 | point_beta | 宽度乘子 | dir_halflife | dir_prior_n | dir_band | OOS MAE% | OOS 95%覆盖 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k, c in state.get("calibration", {}).items():
        L.append(f"| {k} | {c.get('point_beta')} | {c.get('width_mult')} | "
                 f"{c.get('dir_halflife')} | {c.get('dir_prior_n')} | "
                 f"{c.get('dir_band')} | {c.get('oos_mae_pct')} | "
                 f"{c.get('oos_cov95')} |")
    L.append("")
    L.append("## 三、假设库")
    L.append("")
    L.append("| ID | 陈述 | 状态 | 最近检验 | 证据 |")
    L.append("|---|---|---|---|---|")
    sm = {"supported": "支持", "refuted": "证伪", "weakened": "削弱",
          "pending": "待检验"}
    for a in state.get("assumptions", []):
        L.append(f"| {a['id']} | {a['statement']} | {sm.get(a['status'], a['status'])} "
                 f"| {a.get('last_tested')} | {a.get('evidence')} |")
    L.append("")
    L.append("> 证据分 = 样本量收缩 × 归一化 edge；无统计 edge 即 0、方向默认中性。")
    return "\n".join(L)


# ----------------------------------------------------------- 主入口
def run_learning_loop(*, report_date: str, prices, feats: dict, calendars: dict,
                      target_ok: dict, last_real_date: dict,
                      rolling_window: int, persist_on: bool, db,
                      dir_origins: Optional[int] = None,
                      calib_origins: Optional[int] = None) -> dict:
    # 取【上一轮】区间宽度校准乘子（只能用历史值缩放本轮区间，杜绝未来函数）
    prev_state = _load_state()
    width_mults = {}
    for _i in INSTS:
        for _h in HS:
            width_mults[(_i, _h)] = float(
                prev_state.get("calibration", {})
                .get(f"{_i}_{_h}", {}).get("width_mult", 1.0))

    multi: Dict[str, dict] = {}
    records: List[dict] = []
    fitted: Dict[str, object] = {}
    for inst in INSTS:
        if not target_ok.get(inst):
            multi[inst] = {"status": "unavailable", "reason": "真实价格不可用"}
            continue
        try:
            r = _forecast_one(inst, feats, prices, calendars, last_real_date,
                              rolling_window, persist_on, dir_origins=dir_origins,
                              calib_origins=calib_origins,
                              width_mults=width_mults)
            fitted[inst] = r["fc"]
            multi[inst] = {"status": "ok", "cards": r["cards"],
                           "observed_date": r["observed_date"],
                           "warm_started": r["warm_started"],
                           "interval_diag": r["interval_diag"]}
            records.extend(r["records"])
        except Exception as exc:
            LOG.exception("多尺度预测失败(%s)：%s", inst, exc)
            multi[inst] = {"status": "unavailable",
                           "reason": f"{type(exc).__name__}: {exc}"}
    rec_list = records
    if rec_list:
        db.save_multiscale_forecasts(report_date, rec_list)

    # 回测批判（落库后读取，含本轮）
    try:
        msr = multi_scale_review(prices)
    except Exception as exc:
        LOG.warning("多尺度复测失败：%s", exc)
        msr = {f"{i}_{h}": {"n_due": 0, "engaged_n": 0, "dir_acc": None,
                             "dir_p": None, "coverage95": None, "cov_n": 0,
                             "mae_pct": None}
               for i in INSTS for h in HS}

    # 根因反思
    findings = diagnose(msr)

    # 策略进化
    ev = evolve(msr, findings, fitted)
    state, actions = ev["state"], ev["actions"]
    d = strategy_dir()
    (d / "strategy_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    with (d / "learning_journal.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(
            {"report_date": report_date,
             "generated_at": now_beijing().strftime("%Y-%m-%d %H:%M:%S"),
             "state_version": state["version"],
             "review": msr, "findings": findings, "actions": actions},
            ensure_ascii=False, default=str) + "\n")
    (d / "STRATEGY.md").write_text(render_strategy_md(state), encoding="utf-8")

    return {"available": True, "multi": multi, "review": msr,
            "findings": findings, "actions": actions,
            "state_version": state["version"],
            "strategy_dir": str(d),
            "retrained_at": now_beijing().strftime("%Y-%m-%d %H:%M:%S")}
