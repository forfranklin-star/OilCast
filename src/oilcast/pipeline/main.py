"""每日主流程：采集 → 存储 → 特征 → 权重/模型 → 预测 → 回测 → 报告存档。

命令行::
    python -m oilcast.pipeline.main              # 只用真实可追溯数据
    python -m oilcast.pipeline.main --require-prices   # 价格不可用时以非零码退出（CI告警）

任何标的/因素真实数据不足时，对应模块输出 status=unavailable 与原因，
绝不用残缺或合成数据硬算预测。
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..config import get_config, ensure_dirs
from ..data_sources.collector import collect
from ..data_sources.calendar import (next_trading_days, closed_holiday_mmdd)
from ..data_sources.intraday_client import collect_intraday
from ..data_sources.sc_contracts import sc_adjustment_factor_by_date
from ..features.intraday_align import (synchronized_panels, sc_brent_premium_usd,
                                       apply_anchor_panel_to_prices, roll_adjust_panel_sc)
from ..features.engineering import (FACTOR_GROUPS, build_features,
                                    factor_availability, make_supervised,
                                    conditional_geo_sensitivity)
from ..models.errors import InsufficientData
from ..models.evaluation import backtest_short
from ..models.long_term import LongTermMonteCarlo, scenario_probabilities
from ..models.mid_term import MidTermVAR
from ..models.review import review_predictions
from ..models.registry import (save_artifact, load_artifact, export_models,
    import_models, list_models)
from ..models.short_term import ShortTermForecaster
from ..models.weights import learn_factor_weights
from ..reporting import narratives as narr
# 报告渲染依赖 plotly，延迟到真正出 HTML 时再 import，使 --list/--export/--import 模型在最小环境可用
from ..storage.database import OilCastDB, save_csv_snapshot
from ..utils import get_logger, now_beijing

LOG = get_logger(__name__)
# 国内0#柴油无可核验海外真实源，已取消；以美燃油(NYMEX ULSD)、伦敦柴油(ICE Gasoil)替代。
# 新增上海原油 INE SC（人民币/桶；模型内按真实 USDCNY 汇率换算为美元后与布伦特算升贴水）。
TARGETS = {"wti": ("WTI原油", "美元/桶"),
           "brent": ("布伦特原油", "美元/桶"),
           "shanghai_crude": ("上海原油INE SC", "元/桶"),
           "heating_oil": ("美燃油·NYMEX超低硫柴油", "美元/加仑"),
           "gasoil": ("伦敦柴油·ICE Gasoil", "美元/吨")}


# ----------------------------------------------------------- 序列化工具
def _path_records(path: pd.DataFrame) -> list:
    out = path.copy()
    out.insert(0, "date", out.index.strftime("%Y-%m-%d"))
    return out.round(2).where(pd.notnull(out), None).to_dict("records")


def _hist_records(prices: pd.DataFrame, n: int = 180) -> list:
    # 最后防线：丢弃四标的全空的非交易日行（绝不把假日排成 NaN 视觉断点）
    p = prices.tail(n).dropna(how="all").copy()
    p.insert(0, "date", p.index.strftime("%Y-%m-%d"))
    return p.where(pd.notnull(p), None).round(2).to_dict("records")


def _events_records(ev: pd.DataFrame, n: int = 30) -> list:
    if ev is None or ev.empty:
        return []
    e = ev.head(n).copy()
    e["date"] = pd.to_datetime(e["date"]).dt.strftime("%Y-%m-%d")
    return e.to_dict("records")


def _views_records(v: pd.DataFrame, n: int = 12) -> list:
    if v is None or v.empty:
        return []
    v = v.head(n).copy()
    v["date"] = pd.to_datetime(v["date"]).dt.strftime("%Y-%m-%d")
    return v.replace({np.nan: None}).to_dict("records")


def future_index(last_date: pd.Timestamp, periods: int,
                 hist_index=None, closed_mmdd=None) -> pd.DatetimeIndex:
    """未来 periods 个交易日：跳过周末与已知假日，非交易时间不计步长。"""
    return next_trading_days(last_date, periods, hist_index, closed_mmdd)


def _unavailable(reason: str) -> dict:
    return {"status": "unavailable", "reason": reason}


def _available_feature_cols(factor_avail: Dict[str, bool]) -> set:
    cols = set()
    for f, ok in factor_avail.items():
        if ok:
            cols.update(FACTOR_GROUPS.get(f, []))
    return cols


# --------------------------------------------------------------- 主流程
def run(as_of: Optional[datetime] = None, require_prices: bool = False,
        bundle=None) -> dict:
    """生成一期报告。bundle 仅用于单元测试离线注入 DataBundle；
    产品/CLI/Actions 路径恒为 None，即只走真实采集（绝不合成）。"""
    cfg = get_config()
    ensure_dirs()
    real_run = bundle is None     # 产品/CLI 走真实采集；测试注入 bundle 为离线
    as_of = pd.Timestamp(as_of or now_beijing())
    if as_of.tzinfo is not None:
        as_of = as_of.tz_convert("Asia/Shanghai").tz_localize(None)
    report_date = as_of.strftime("%Y-%m-%d")
    LOG.info("===== OilCast 每日报告开始：%s =====", report_date)

    # 1) 数据采集与落库（严格真实，缺失不补齐；测试可注入 bundle 以离线运行）
    if bundle is None:
        bundle = collect(as_of=as_of)
    db = OilCastDB()
    db.save_prices(bundle.prices, bundle.mode, bundle.lineage)
    db.save_macro(bundle.macro, bundle.mode, bundle.lineage)
    db.save_events(bundle.events)
    db.save_views(bundle.views)
    db.save_lineage(report_date, bundle.lineage, bundle.mode)
    save_csv_snapshot(bundle.prices, bundle.macro, report_date)
    prices, macro, events, views = bundle.prices, bundle.macro, bundle.events, bundle.views

    # 重大事件年表"自动扩充"：events 表每日 INSERT OR IGNORE、长期累积且从不删除。读取库内
    # 全部真实 RSS 事件并合并本次抓取，作为机器核验重大事件源传入特征层——yaml 人工年表覆盖
    # 可核验历史，库内累积事件让 yaml 结束日之后新发生的重大冲突自动进入历史 regime。
    detected_events = None
    try:
        hist_events = db.read_events()
        frames = [df for df in (hist_events, events) if df is not None and not df.empty]
        if frames:
            detected_events = pd.concat(frames, ignore_index=True)
            detected_events = (detected_events.drop_duplicates(["date", "title"])
                               .sort_values("date").reset_index(drop=True))
            LOG.info("年表自动扩充：机器核验候选真实事件 %d 条（库内累积+本次）",
                     len(detected_events))
    except Exception as exc:
        LOG.warning("累积事件读取失败，本期历史 regime 仅用人工年表：%s", exc)

    # 1b) 盘中分时采集与"同一真实时刻"面板（长期增量积累；失败不阻断，退化为日频 as-of）
    intraday_day = pd.DataFrame()
    intraday_night = pd.DataFrame()
    try:
        if real_run:
            idf = collect_intraday(as_of)
            n_saved = db.save_intraday(idf)
            LOG.info("分时 K 增量落库 %d 行", n_saved)
        ilong = db.read_intraday()
        if not ilong.empty:
            panels = synchronized_panels(ilong)
            intraday_day = panels.get("day_1500", pd.DataFrame())
            intraday_night = panels.get("night_0230", pd.DataFrame())
            LOG.info("同时刻面板：15:00 %d 日、02:30 %d 日",
                     len(intraday_day), len(intraday_night))
            # 上海分时 SC0 是未复权主力连续，换月日有跳空：用日 K 持仓量主力重建的同一套比例
            # 复权因子，在【面板层】校正 15:00 与 02:30 两个同时刻面板的上海列，使下游价格序列、
            # 跨市场外生特征、美元升贴水三处口径一致（失败则沿用未复权 SC0，不造数）。
            _sc_fac = None
            if "shanghai_crude" in intraday_night.columns or "shanghai_crude" in intraday_day.columns:
                try:
                    _sc_fac = sc_adjustment_factor_by_date(as_of)
                except Exception as exc:
                    LOG.warning("上海分时换月复权因子计算失败，沿用未复权SC0：%s", exc)
            intraday_day = roll_adjust_panel_sc(intraday_day, _sc_fac)
            intraday_night = roll_adjust_panel_sc(intraday_night, _sc_fac)
            # 分时覆盖窗口内，把【五个品种】统一锚到北京 02:30（上海夜盘收盘）这一同一真实
            # 时刻：此时欧美电子盘仍在交易，取其同时刻盘中价（用户已确认按盘中价、不要求开/
            # 收盘价）。此前只更新上海一个品种、外盘仍用各自美盘日收盘，导致五品种并非同一时刻、
            # 变化率不可比（如上海日 K -7% 而外盘同刻仅 -1% 的虚假背离）。原始日 K 已落库保留。
            if not intraday_night.empty:
                _before = prices.copy()
                prices = apply_anchor_panel_to_prices(prices, intraday_night, None)
                _updated = [f"{c}:{int(prices[c].reindex(_before.index).ne(_before[c]).sum())}"
                            for c in prices.columns if c in intraday_night.columns]
                LOG.info("分时覆盖窗口五品种统一按北京02:30同时刻更新（更新交易日数）：%s",
                         "，".join(x for x in _updated if not x.endswith(":0")))
    except Exception as exc:
        LOG.warning("分时同时刻面板构建失败，本期跨市场退化为日频 as-of：%s", exc)

    mcfg = cfg["model"]
    h_short, h_mid, h_long = (int(mcfg["short_horizon_td"]),
                              int(mcfg["mid_horizon_td"]),
                              int(mcfg["long_horizon_td"]))

    # 2) 标的可用性（只认真实质量门 ok）
    target_ok = {t: (bundle.field_status(t) == "ok") for t in TARGETS}
    usable_targets = [t for t, ok in target_ok.items() if ok]
    anchor = "wti" if target_ok["wti"] else ("brent" if target_ok["brent"] else None)
    if anchor is None:
        LOG.error("WTI/Brent 真实价格均不可用，本期不产出任何价格预测")
    last_real_date = {t: (prices[t].dropna().index.max() if prices[t].notna().any() else None)
                      for t in TARGETS}
    # 每标的真实交易日历（用于未来外推时跳过周末/假日，非交易时间不计步长）
    cal = {}
    for _t in TARGETS:
        _hidx = pd.DatetimeIndex(sorted(prices[_t].dropna().index))
        cal[_t] = {"hist": _hidx, "closed": closed_holiday_mmdd(_hidx)}

    # 3) 特征矩阵（事件/观点源不可达时对应因素保持缺失）
    ev_ok = bundle.field_status("events") == "ok"
    vw_ok = bundle.field_status("institutional_view") == "ok"
    feats = {t: build_features(prices, macro, events, views, target=t,
                               events_available=ev_ok, views_available=vw_ok,
                               intraday_session=intraday_night,
                               detected_events=detected_events)
             for t in usable_targets}
    if feats:
        Path(cfg["storage"]["processed_dir"]).mkdir(parents=True, exist_ok=True)
        feats[anchor or usable_targets[0]].to_csv(
            Path(cfg["storage"]["processed_dir"]) / f"features_{report_date}.csv",
            encoding="utf-8-sig")

    # 4) 因素权重——【逐品种分别学习】。不同油种对局域冲突、美联储政策、CPI、非农、
    #    日元汇率等变量的敏感度不同（如上海原油更受中东地缘、亚太需求与人民币汇率
    #    驱动，WTI 更受美国库存/货币政策驱动），因此每个可用品种都用自己的样本学一
    #    套权重并分别落库，报告层再横向对比；主锚(anchor)那套供既有叙事/权重变化面板。
    empty_w = pd.DataFrame(columns=["factor", "weight", "model_importance",
                                    "prior", "available"])
    weights_by_target: Dict[str, pd.DataFrame] = {}
    factor_avail_by: Dict[str, dict] = {}
    prev_by_target: Dict[str, pd.Series] = {}
    hist_dates = [d for d in db.list_report_dates() if d < report_date]
    prev_weight_date = hist_dates[0] if hist_dates else None
    for t in usable_targets:
        fav = factor_availability(feats[t])
        factor_avail_by[t] = fav
        X_w, y_w = make_supervised(feats[t], prices[t], h_short)
        prev_w = db.latest_weights(t, report_date)   # 该品种严格早于本期的上一期权重
        prev_by_target[t] = prev_w
        try:
            w_t = learn_factor_weights(X_w, y_w, available=fav, prev_weights=prev_w)
        except Exception as exc:
            LOG.warning("品种 %s 权重学习失败，降级空表：%s", t, exc)
            w_t = empty_w.copy()
        db.save_weights(report_date, w_t, instrument=t)
        weights_by_target[t] = w_t
        LOG.info("因素权重已分品种学习：%s（%d 真实样本）", t, len(X_w))

    factor_avail = factor_avail_by.get(anchor, {f: False for f in cfg["prior_weights"]}) \
        if anchor else {f: False for f in cfg["prior_weights"]}
    weights = weights_by_target.get(anchor, empty_w.copy())
    prev_weights = prev_by_target.get(anchor)

    avail_cols = _available_feature_cols(factor_avail)

    # 5) 短期预测：逐可用标的直接建模（不做跨标的推算）
    short_out: Dict[str, dict] = {}
    persist_entries: Dict[str, dict] = {}
    # 读写模型工件，实现跨期热启动（只用真实数据训练）
    persist_on = bool(mcfg.get("persist_models", True))
    for t in TARGETS:
        if not target_ok[t]:
            lin = bundle.lineage.get(t, {})
            short_out[t] = _unavailable(
                f"{TARGETS[t][0]}真实数据不可用（{lin.get('status','unavailable')}）："
                f"{lin.get('note') or lin.get('source_name','源不可达')}")
            continue
        try:
            warm_obj, prev_entry = (load_artifact("short", t) if persist_on
                                    else (None, None))
            fc = ShortTermForecaster(horizon=h_short, window=int(mcfg["rolling_window"]),
                                     compute_residuals=True)
            fc.fit(feats[t], prices[t], warm=warm_obj)
            last_dt = last_real_date[t]
            res = fc.predict(feats[t].loc[[last_dt]], float(prices[t].loc[last_dt]),
                             future_index(last_dt, h_short, cal[t]['hist'], cal[t]['closed']))
            entry = {}
            if persist_on:
                try:
                    entry = save_artifact(
                        "short", t, fc, meta=getattr(fc, "train_meta", {}),
                        warm_from=prev_entry if fc.warm_info["warm_started"] else None)
                    persist_entries[t] = entry
                except Exception as exc2:
                    LOG.warning("模型工件保存失败(%s)，不影响报告：%s", t, exc2)
            short_out[t] = {"status": "ok", "endpoint": res.endpoint,
                            "path": _path_records(res.path),
                            "arima_benchmark": res.benchmark_endpoint,
                            "train_meta": getattr(fc, "train_meta", {}),
                            "model_entry": {k: entry.get(k) for k in
                                            ("version", "parent_version", "warm_started",
                                             "parent_train_end", "saved_at")},
                            "observed_date": last_dt.strftime("%Y-%m-%d")}
        except Exception as exc:   # 建模边界异常一律降级为 unavailable，不拖垮整份报告
            LOG.exception("短期模型失败(%s)：%s", t, exc)   # 完整堆栈落日志，便于定位
            short_out[t] = _unavailable(f"{type(exc).__name__}: {exc}")
    LOG.info("短期预测完成：%s", {t: short_out[t]["status"] for t in short_out})

    # 6) 中期 VAR：逐可用标的
    mid_out: Dict[str, dict] = {}
    for t in TARGETS:
        if not target_ok[t]:
            mid_out[t] = short_out[t]
            continue
        try:
            var = MidTermVAR().fit(prices, macro, target=t)
            probs = scenario_probabilities(feats[t].loc[last_real_date[t]],
                                           available=avail_cols)
            r = var.predict(future_index(last_real_date[t], h_mid, cal[t]['hist'], cal[t]['closed']), probs, cfg["scenarios"])
            mid_out[t] = {"status": "ok", "endpoint": r["endpoint"],
                          "path": _path_records(r["path"]), "scenario_probs": probs,
                          "observed_date": last_real_date[t].strftime("%Y-%m-%d")}
        except Exception as exc:   # 建模边界异常一律降级为 unavailable，不拖垮整份报告
            mid_out[t] = _unavailable(str(exc))

    # 7) 长期情景 + 蒙特卡洛：逐可用标的
    long_out: Dict[str, dict] = {}
    anchor_inst = None
    if vw_ok and "target_wti" in views.columns:
        vals = pd.to_numeric(views["target_wti"], errors="coerce").dropna()
        anchor_inst = round(float(vals.median()), 1) if len(vals) else None
    for t in TARGETS:
        if not target_ok[t]:
            long_out[t] = short_out[t]
            continue
        try:
            probs = scenario_probabilities(feats[t].loc[last_real_date[t]],
                                           available=avail_cols)
            lt = LongTermMonteCarlo().fit(float(prices[t].loc[last_real_date[t]]))
            r = lt.predict(future_index(last_real_date[t], h_long, cal[t]['hist'], cal[t]['closed']), probs,
                           institution_anchor=anchor_inst if t == "wti" else None)
            long_out[t] = {"status": "ok", "endpoint": r["endpoint"],
                           "path": _path_records(r["path"]),
                           "observed_date": last_real_date[t].strftime("%Y-%m-%d")}
        except Exception as exc:   # 建模边界异常一律降级为 unavailable，不拖垮整份报告
            long_out[t] = _unavailable(str(exc))
    LOG.info("中/长期预测完成")

    # 8) 回测（仅主锚）。回测是对历史的慢变评估，不必每天重算数十个完整拟合：
    #   ① 逐原点用评估专用的小校准规模（生产模型仍用全量，不影响线上预测质量）；
    #   ② 结果按主锚数据末日缓存 backtest_refresh_days 天，期内直接复用并标注截止日，
    #      到期/数据推进才重算——把每日管线从 10+ 分钟压回时限内，且回测结论仍每周刷新。
    _mcfg = get_config()["model"]
    if anchor:
        def _bt_json_default(o):
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.bool_):
                return bool(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return str(o)

        bt = None
        data_end = last_real_date[anchor]
        cache_path = (Path(cfg["storage"]["sqlite_path"]).parent
                      / f"backtest_cache_{anchor}.json")
        refresh_days = int(_mcfg.get("backtest_refresh_days", 7))
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                age_days = int((data_end - pd.Timestamp(cached["data_end"])).days)
                if 0 <= age_days <= refresh_days:
                    bt = cached["bt"]
                    bt["cached"] = True
                    bt["cached_as_of"] = cached["data_end"]
                    LOG.info("回测结果复用缓存（截止 %s，距今 %s 个自然日）",
                             cached["data_end"], age_days)
            except Exception as exc:
                LOG.warning("回测缓存读取失败，将重算：%s", exc)
                bt = None
        if bt is None:
            n_bo = int(_mcfg.get("backtest_origins", 30))
            bt = backtest_short(
                feats[anchor], prices[anchor], horizon=h_short,
                n_origins=n_bo, gap=int(_mcfg.get("backtest_gap", 6)),
                calib_origins=int(_mcfg.get("backtest_calib_origins", 6)),
                calib_iter=int(_mcfg.get("backtest_calib_iter", 50)),
                dir_origins=n_bo)
            bt["cached"] = False
            # 仅缓存【可用】结果；样本不足等不可用结果绝不缓存，否则会被连续复用 7 天
            if bt.get("available"):
                try:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(
                        json.dumps({"data_end": data_end.strftime("%Y-%m-%d"), "bt": bt},
                                   ensure_ascii=False, default=_bt_json_default),
                        encoding="utf-8")
                except Exception as exc:
                    LOG.warning("回测缓存写入失败，不影响报告：%s", exc)
    else:
        bt = {"status": "unavailable", "reason": "无可用真实价格序列，未执行回测"}
    # 8b) 模型学习/更新证据：训练元信息、本轮 vs 上轮权重变化、历史预测复盘
    train_meta = {t: short_out[t].get("train_meta", {})
                  for t in short_out if short_out[t].get("status") == "ok"}
    weight_delta = []
    try:   # 学习/复盘为增强模块，任何异常降级，绝不拖垮报告主体
        if len(weights) and prev_weights is not None:
            for _, r in weights.iterrows():
                old = prev_weights.get(r["factor"])
                new = r.get("weight")
                cn = narr.FACTOR_CN.get(r["factor"], r["factor"])
                if pd.notna(new) and old is not None and not pd.isna(old):
                    weight_delta.append({"factor": cn,
                                         "prev": round(float(old) * 100, 2),
                                         "now": round(float(new) * 100, 2),
                                         "delta_pp": round((float(new) - float(old)) * 100, 2)})
                elif pd.notna(new):
                    weight_delta.append({"factor": cn, "prev": None,
                                         "now": round(float(new) * 100, 2), "delta_pp": None})
    except Exception as exc:
        LOG.warning("权重变化计算失败，已降级：%s", exc)
        weight_delta = []
    try:
        review = review_predictions({t: prices[t] for t in TARGETS if t in prices},
                                    pd.Timestamp(report_date))
    except Exception as exc:
        LOG.warning("历史预测复盘失败，已降级：%s", exc)
        review = {"available": False, "reason": f"复盘计算异常已跳过：{type(exc).__name__}"}
    n_report_runs = len([d for d in db.list_report_dates() if d < report_date]) + 1  # 含本期
    # 主锚品种的样本外校准系数 β（模型当前"进攻性/置信度"），供回测卡展示
    calib_beta_h = None
    if anchor and train_meta.get(anchor):
        calib_beta_h = train_meta[anchor].get("calib_beta_h")
    model_learning = {
        "retrained_at": now_beijing().strftime("%Y-%m-%d %H:%M:%S"),
        "rolling_window": int(mcfg["rolling_window"]),
        "train_window": int(mcfg.get("train_window", 500) or 0),
        "calib_beta_h": calib_beta_h,
        "train_meta": train_meta,
        "prev_weight_date": prev_weight_date,
        "weight_delta": weight_delta,
        "backtest": bt,
        "review": review,
        "n_runs": n_report_runs,
        "persistence": {
            "enabled": persist_on,
            "models": {t: short_out[t].get("model_entry", {})
                       for t in short_out if short_out[t].get("status") == "ok"},
        },
    }

    # 8c) 数据新鲜度守门：明确区分"已取到最新交易日 / 较上期未推进 / 收盘滞后"，
    # 绝不在没拿到新数据时静默产出一份与昨天雷同、却看似"今日已更新"的预测
    freshness = _assess_freshness(anchor, short_out, last_real_date, report_date)
    LOG.info("数据新鲜度：%s（最新观测 %s，上期 %s，滞后工作日 %s，推进=%s）",
             freshness["level"], freshness["latest"], freshness["prev_latest"],
             freshness["lag_bdays"], freshness["advanced"])

    # 8d) 分品种因素敏感度矩阵：行=因素，列=品种，值=该品种学到的权重(%)，
    #     不可用因素/不可用品种为 None，报告层据此对比"谁对哪个变量更敏感"
    sensitivity: Dict[str, Dict[str, float]] = {}
    for t, wt in weights_by_target.items():
        if wt is None or wt.empty:
            continue
        for _, r in wt.iterrows():
            f = r["factor"]
            wv = r.get("weight")
            sensitivity.setdefault(f, {})[t] = (round(float(wv) * 100, 2)
                                                if pd.notna(wv) else None)
    weights_by_target_rec = {t: wt.to_dict("records")
                             for t, wt in weights_by_target.items()}

    # 事件期【条件敏感度】归因：区别于 500 日平均权重，刻画当前地缘冲突中各品种真实敏感度
    cond_sens = conditional_geo_sensitivity(
        prices, list(TARGETS.keys()),
        recent_window=int(mcfg.get("train_window", mcfg.get("rolling_window", 500))),
        usdcny=macro["usdcny"] if "usdcny" in macro.columns else None)

    # 9) 当前值（携带真实观测日期，缺失为 None）
    current, current_meta = {}, {}
    for t in TARGETS:
        lin = bundle.lineage.get(t, {})
        if target_ok[t] and last_real_date[t] is not None:
            current[t] = round(float(prices[t].loc[last_real_date[t]]), 2)
            current_meta[t] = {"value": current[t],
                               "observed_date": last_real_date[t].strftime("%Y-%m-%d"),
                               "status": "ok", "source": lin.get("source_name", "")}
        else:
            current[t] = None
            current_meta[t] = {"value": None, "observed_date": lin.get("last_observed"),
                               "status": lin.get("status", "unavailable"),
                               "source": lin.get("source_name", ""),
                               "reason": lin.get("note", "无真实可核验数据")}

    # 9b) 交易时段"同一真实时刻"面板（盘中分时；只在有分时的窗口有值，每日积累）
    session_sync = _build_session_sync(intraday_day, intraday_night, macro,
                                       feats.get("shanghai_crude"), db)

    report = {
        "report_date": report_date,
        "generated_at": now_beijing().strftime("%Y-%m-%d %H:%M:%S %z"),
        "mode": bundle.mode,
        "lineage": bundle.lineage,
        "factor_availability": factor_avail,
        "current": current,
        "current_meta": current_meta,
        "units": {t: TARGETS[t][1] for t in TARGETS},
        "names": {t: TARGETS[t][0] for t in TARGETS},
        "history": _hist_records(prices),
        "forecasts": {"short": short_out, "mid": mid_out, "long": long_out},
        "weights": weights.to_dict("records") if len(weights) else [],
        "weights_by_target": weights_by_target_rec,
        "sensitivity": sensitivity,
        "conditional_sensitivity": cond_sens,
        "events": _events_records(events),
        "views": _views_records(views),
        "backtest": bt,
        "model_learning": model_learning,
        "session_sync": session_sync,
        "freshness": freshness,
        "narratives": _build_narratives(bundle, prices, current_meta, short_out,
                                        mid_out, long_out, weights, events, bt,
                                        model_learning, sensitivity, cond_sens),
    }

    # 10) 落库 + JSON/HTML 存档
    _persist_forecasts(db, report_date, report)
    archive = Path(cfg["storage"]["archive_dir"])
    latest = Path(cfg["storage"]["latest_dir"])
    archive.mkdir(parents=True, exist_ok=True)
    latest.mkdir(parents=True, exist_ok=True)
    (archive / f"{report_date}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (latest / "latest.json").write_text(
        json.dumps(report, ensure_ascii=False, default=str), encoding="utf-8")
    from ..reporting.static_html import render_static_html  # 延迟加载（依赖 plotly）
    html = render_static_html(report)
    (archive / f"{report_date}.html").write_text(html, encoding="utf-8")
    (latest / "index.html").write_text(html, encoding="utf-8")
    # Pages 根入口：自动跳到 latest（upload 整个 reports 目录时生效）
    (latest.parent / "index.html").write_text(
        '<!doctype html><meta charset="utf-8">'
        '<meta http-equiv="refresh" content="0; url=latest/index.html">'
        '<a href="latest/index.html">最新报告</a>', encoding="utf-8")
    db.register_report(report_date, str(archive / f"{report_date}.json"),
                       str(archive / f"{report_date}.html"))
    LOG.info("报告已存档：%s", archive / f"{report_date}.html")

    if require_prices and anchor is None:
        raise SystemExit(2)
    return report


def _build_session_sync(day: pd.DataFrame, night: pd.DataFrame, macro: pd.DataFrame,
                        sc_feats, db: OilCastDB) -> dict:
    """组装"交易时段同一真实时刻"面板数据（供报告展示与审计）。

    15:00=上海日盘收盘同时刻海外电子盘价；02:30=上海夜盘收盘同时刻欧美盘价；
    上海人民币价按真实 USDCNY 换算美元后与同时刻布伦特算升贴水。仅在有分时窗口有值。"""
    def _clean(v):
        return None if (v is None or pd.isna(v)) else round(float(v), 2)
    if day is None or day.empty:
        return {"available": False, "reason": "分时数据尚未采集到（免费源历史有限，每日增量积累中）"}
    usdcny = macro["usdcny"] if "usdcny" in macro else None
    fx_rate = fx_date = None
    if usdcny is not None and usdcny.notna().any():
        fx_rate = round(float(usdcny.dropna().iloc[-1]), 4)
        fx_date = usdcny.dropna().index[-1].strftime("%Y-%m-%d")
    # 统计主节点=上海夜盘 02:30 收盘：升贴水/美元换算主口径用 night 面板（02:30 同刻），
    # 仅当夜盘分时缺失时回退到 15:00 日盘面板（不造数）。
    primary = night if (night is not None and not night.dropna(how="all").empty) else day
    prem = sc_brent_premium_usd(primary, usdcny)
    ilong = db.read_intraday()
    counts, sources = {}, {}
    if not ilong.empty:
        counts = {s: int(n) for s, n in ilong.groupby("symbol").size().items()}
        sources = {s: str(g["source"].iloc[0]) for s, g in ilong.groupby("symbol")}
    prem_curve = prem.dropna(subset=["premium_usd"]).tail(130)
    prem_series = [{"date": d.strftime("%Y-%m-%d"),
                    "premium_usd": _clean(r.premium_usd),
                    "sc_usd": _clean(r.sc_usd), "brent_usd": _clean(r.brent_usd)}
                   for d, r in prem_curve.iterrows()]
    def _records(panel, with_fx):
        price_cols = [c for c in ("shanghai_crude", "wti", "brent") if c in panel]
        # 只展示至少有一个真实价格的交易日，剔除分时缺失的空行/未来占位行
        p = panel.dropna(how="all", subset=price_cols).tail(8)
        rows = []
        for d, r in p.iterrows():
            row = {"date": d.strftime("%Y-%m-%d"),
                   "sc_cny": _clean(r.get("shanghai_crude")),
                   "wti": _clean(r.get("wti")), "brent": _clean(r.get("brent"))}
            if with_fx:
                if d in prem.index:
                    row["sc_usd"] = _clean(prem.loc[d, "sc_usd"])
                    row["premium_usd"] = _clean(prem.loc[d, "premium_usd"])
            rows.append(row)
        return rows
    model_days = 0
    if sc_feats is not None:
        model_days = int(getattr(sc_feats, "attrs", {}).get("intraday_synchronized_days", 0) or 0)
    both = pd.concat([day.dropna(how="all"), night.dropna(how="all")])
    return {
        "available": True,
        "coverage_start": both.index.min().strftime("%Y-%m-%d") if len(both) else None,
        "coverage_end": both.index.max().strftime("%Y-%m-%d") if len(both) else None,
        "intraday_counts": counts, "sources": sources,
        "premium_series": prem_series,
        "primary_anchor": "night_0230",
        "anchor_night_records": _records(night, True),
        "anchor_day_records": _records(day, False),
        "model_synchronized_days": model_days,
        "fx_rate": fx_rate, "fx_date": fx_date,
    }


def _is_ok(item: dict) -> bool:
    return isinstance(item, dict) and item.get("status", "ok") == "ok"


def _build_narratives(bundle, prices, current_meta, short_out, mid_out, long_out,
                     weights, events, bt, model_learning=None, sensitivity=None,
                     conditional_sensitivity=None):
    def one_narr(item, cn, unit, horizon):
        if not _is_ok(item):
            return (f"{cn}{horizon}预测暂不提供：{item.get('reason','真实数据不可用')}。"
                    f"按数据原则，不以估算或合成数据替代。")
        return narr.forecast_narrative(item["endpoint"], cn, unit, horizon)

    def trend_narr(t):
        meta = current_meta[t]
        if meta["status"] != "ok":
            return (f"{TARGETS[t][0]}本期无真实可核验价格（状态：{meta['status']}；"
                    f"{meta.get('reason','')}），不做走势判断。")
        return narr.trend_narrative(prices[t].dropna(), TARGETS[t][0], TARGETS[t][1])

    return {
        "trends": {t: trend_narr(t) for t in TARGETS},
        "short": {t: one_narr(short_out[t], TARGETS[t][0], TARGETS[t][1], "两周")
                  for t in TARGETS},
        "mid": {t: one_narr(mid_out[t], TARGETS[t][0], TARGETS[t][1], "三个月")
                for t in TARGETS},
        # 长期情景叙事逐品种生成，交互页切换标的时随之切换
        "long": {t: (narr.scenario_narrative(long_out[t]["endpoint"]) if _is_ok(long_out[t])
                     else f"{TARGETS[t][0]}长期预测暂不提供："
                          f"{long_out[t].get('reason','真实数据不可用')}")
                 for t in TARGETS},
        "weights": narr.weights_narrative(weights) if len(weights) else
        "可用真实因素不足，本期不计算因素权重。",
        "sensitivity": narr.sensitivity_narrative(sensitivity or {},
                                                  {t: TARGETS[t][0] for t in TARGETS}),
        "conditional_sensitivity": narr.conditional_sensitivity_narrative(
            conditional_sensitivity or {}, {t: TARGETS[t][0] for t in TARGETS}),
        "learning": narr.learning_narrative(model_learning or {}),
        "events": narr.events_narrative(events),
        "backtest": narr.backtest_narrative(bt) if isinstance(bt, dict) and
                    bt.get("status") != "unavailable" else
                    "无可用真实价格序列，本期未执行滚动回测。",
        "sources": narr.lineage_narrative(bundle),
    }


def _assess_freshness(anchor, short_out, last_real_date, report_date) -> dict:
    """判定本期价格是否真正更新。

    三个等级：
      fresh     —— 已取到最近应有的交易日收盘（较上期推进，或仅隔 0~1 个工作日，
                   对应"当天早上跑、上一交易日收盘为最新"的正常情形）；
      unchanged —— 最新观测日与上一期训练截止相同：本期没拿到任何新交易日，
                   模型训练截止未推进、没有学到新行情，必须显著告知，避免旧预测冒充新报告；
      lagging   —— 最新观测距报告日已过 ≥2 个工作日仍无更近收盘（期间多为正常交易日），
                   提示主数据源未取到最新数据，需查数据谱系尝试链。
    口径只用工作日计数（周末天然不计），交易所个别假日造成的 1 天偏差归入 fresh。
    """
    per = {t: (d.strftime("%Y-%m-%d") if d is not None else None)
           for t, d in last_real_date.items()}
    cur = last_real_date.get(anchor) if anchor else None
    prev = None
    if anchor and short_out.get(anchor, {}).get("status") == "ok":
        pte = short_out[anchor].get("model_entry", {}).get("parent_train_end")
        if pte:
            prev = pd.Timestamp(pte)
    lag_bdays = None
    if cur is not None:
        lag_bdays = max(len(pd.bdate_range(cur.normalize(),
                                          pd.Timestamp(report_date))) - 1, 0)
    advanced = None
    if prev is not None and cur is not None:
        advanced = bool(cur.normalize() > prev.normalize())
    # 各品种"交易日 T 完整收盘"对应的北京时刻（物理发生在 T+1 凌晨），不能用同一个截断边界：
    #   上海 INE：T 日盘 + 夜盘到 T+1 02:30 即完成 T 日全部交易（周五夜盘物理在周六02:30，归周五）；
    #   欧美 WTI/Brent/成品油：T 日电子盘一直交易到 T+1 约 06:00（取较晚的冬令时，宁严勿松）——
    #   即周六02:30 上海已收、欧美仍处周五交易时段。上海统计节点 02:30 处，欧美只取【同时刻盘中价】
    #   做对齐，绝不能取其尚未产生的当日收盘价，也不能按"到了周六"把上海02:30夜盘当周末截断。
    close_offset = {"shanghai_crude": (2, 30)}
    default_close = (6, 0)

    def _expected_asof(now_ts, h, m):
        """截至 now_ts（北京墙钟），最近一个【已完整收盘】的交易日（已跳过周末）。"""
        aa = now_ts.normalize()
        cut = aa + pd.Timedelta(hours=h, minutes=m)
        cand = aa - pd.Timedelta(days=1) if now_ts >= cut else aa - pd.Timedelta(days=2)
        while cand.weekday() > 4:
            cand -= pd.Timedelta(days=1)
        return cand

    nm = TARGETS[anchor][0] if anchor else "主力品种"
    cur_s = per.get(anchor)
    now = pd.Timestamp(now_beijing())
    if now.tzinfo is not None:
        now = now.tz_localize(None)     # now_beijing 已是北京墙钟时间，去 tz 与 naive 日期比较
    wd_cn = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now.weekday()]
    run_clock = now.strftime("%Y-%m-%d %H:%M") + f"（北京时 {wd_cn}）"
    asof = pd.Timestamp(report_date).normalize()
    # 逐品种：各自的应收盘交易日、落后工作日数、报告日 T 是否已完整收盘
    per_status = {}
    for t in TARGETS:
        hh, mm = close_offset.get(t, default_close)
        exp_t = _expected_asof(now, hh, mm)
        obs_t = last_real_date.get(t)
        g_t = None if obs_t is None else max(
            len(pd.bdate_range(obs_t.normalize(), exp_t)) - 1, 0)
        closed_t = now >= (asof + pd.Timedelta(days=1, hours=hh, minutes=mm))
        per_status[t] = {"obs": per.get(t), "expected": exp_t.strftime("%Y-%m-%d"),
                         "gap": g_t, "closed_for_asof": bool(closed_t)}
    ah, am = close_offset.get(anchor, default_close)
    expected = pd.Timestamp(per_status[anchor]["expected"]) if anchor else _expected_asof(now, *default_close)
    gap = per_status[anchor]["gap"] if anchor else None
    early = not per_status[anchor]["closed_for_asof"] if anchor else False
    # 上海已收、欧美仍盘中（周六02:30~06:00，anchor=上海时的关键情形）
    sh_done = per_status.get("shanghai_crude", {}).get("closed_for_asof")
    overseas_open = [TARGETS[t][0] for t in ("wti", "brent", "heating_oil", "gasoil")
                     if t in per_status and not per_status[t]["closed_for_asof"]]

    def _obs(t):
        return per_status[t]["obs"] or "无"

    per_line = "各品种最新观测日：" + "；".join(
        f"{TARGETS[t][0]} {per_status[t]['obs'] or '无'}" for t in TARGETS if t in per_status)
    backfill = asof < (expected - pd.Timedelta(days=4))   # 显式回溯更早历史重算
    if backfill:
        level, msg = "fresh", (f"历史重算（指定 --as-of {report_date}），数据严格截止该日、不向未来取数。"
                               f"{per_line}。")
    elif cur is None:
        level, msg = "lagging", f"{nm}无任何真实价格，未做预测；请查数据谱系尝试链。{per_line}。"
    elif gap == 0:
        # anchor 已到"截至此刻最近应已完整收盘的交易日"→ 数据是新的（同日/周末重跑 advanced=False 也正常）
        level = "fresh"
        if advanced:
            msg = f"价格已更新至最近交易日 {cur_s}，较上一期 {prev:%Y-%m-%d} 向前推进。{per_line}。"
        elif anchor == "shanghai_crude" and sh_done and overseas_open:
            msg = (f"本报告生成于 {run_clock}：上海 INE {report_date} 日夜盘已于次日 02:30 收盘"
                   f"（最新完整收盘 {cur_s}）；此刻 {'、'.join(overseas_open)} 仍处于 {report_date} "
                   f"交易时段（其日 K 收盘要到次日约 05–06 点）。上海统计节点取 02:30，欧美在该真实时刻"
                   f"只取【同时刻盘中价】对齐、不取其尚未产生的收盘价，故欧美日 K 收盘序列最新到 "
                   f"{_obs('brent')} 属正常，并非时钟错误或采集失败；每日北京 09:00 定时任务时各市场均已收完。"
                   f"{per_line}。")
        elif early:
            msg = (f"本报告生成于 {run_clock}（{nm}收盘前/盘中运行）：{report_date} 的交易尚未走完，"
                   f"当前最新【完整收盘】为 {cur_s} 属正常、并非时钟错误或采集失败；待其收盘后由每日"
                   f"北京 09:00 定时任务重跑即推进到 {report_date}。{per_line}。")
        else:
            msg = f"价格已处于最近交易日 {cur_s}（截至 {run_clock} 无更新的已收盘交易日，周末/同日重跑属正常）。{per_line}。"
    elif gap == 1:
        level = "unchanged"
        msg = (f"最近一个已收盘交易日 {expected:%Y-%m-%d} 的{nm}数据尚未取到，最新仍停在 {cur_s}，"
               f"模型训练截止未推进；常见原因为数据源临时不可达（见数据谱系尝试链），请稍后重跑。{per_line}。")
    else:
        level = "lagging"
        msg = (f"{nm}最新收盘为 {cur_s}，截至 {run_clock} 已落后应有交易日 {gap} 个工作日，"
               f"主数据源可能未取到最新数据（见数据谱系尝试链），以下预测未纳入最近行情、请注意时效。{per_line}。")
    return {"level": level, "anchor": anchor, "per_target": per, "latest": cur_s,
            "expected_latest": expected.strftime("%Y-%m-%d"),
            "per_target_status": per_status,
            "prev_latest": prev.strftime("%Y-%m-%d") if prev is not None else None,
            "lag_bdays": lag_bdays, "gap_bdays": gap, "advanced": advanced,
            "message": msg, "run_clock": run_clock, "early_market_open": bool(early)}


def _persist_forecasts(db: OilCastDB, report_date: str, report: dict) -> None:
    def _rec(horizon, inst, p) -> dict:
        return {"horizon": horizon, "instrument": inst,
                "target_date": p.get("target_date") or p.get("date"), "mean": p["mean"],
                "q05": p.get("q05"), "q25": p.get("q25"), "q50": p.get("q50", p["mean"]),
                "q75": p.get("q75"), "q95": p.get("q95"),
                "prob_up": p.get("prob_up"), "prob_down": p.get("prob_down"),
                "dir_stance": p.get("dir_stance", "中性")}
    records = []
    for horizon, pack in report["forecasts"].items():
        for inst, item in pack.items():
            if item.get("status") == "unavailable":
                continue
            if horizon == "short" and item.get("path"):
                # 短期逐日路径全部落库：次日起 h=1 即到期可复盘，学习反馈不必等两周
                for p in item["path"]:
                    if p.get("mean") is not None:
                        records.append(_rec(horizon, inst, p))
            else:
                records.append(_rec(horizon, inst, item["endpoint"]))
    db.save_forecasts(report_date, records)


def main() -> None:
    parser = argparse.ArgumentParser(description="OilCast 每日报告流水线（严格真实数据）")
    parser.add_argument("--require-prices", action="store_true",
                        help="WTI/Brent 真实价格均不可用时以退出码 2 失败（CI 告警）")
    parser.add_argument("--as-of", type=str, default=None, help="基准日期 YYYY-MM-DD")
    parser.add_argument("--list-models", action="store_true", help="列出已持久化模型版本后退出")
    parser.add_argument("--export-models", type=str, default=None, metavar="ZIP",
                        help="导出完整学习快照(模型工件+版本链+因素权重+历史预测)为 zip 后退出")
    parser.add_argument("--import-models", type=str, default=None, metavar="ZIP",
                        help="从学习快照 zip 恢复模型与权重/历史预测(删库/换机后续学不从零)后退出")
    args = parser.parse_args()
    if args.list_models:
        mm = list_models()
        if not mm:
            print("（当前无已保存模型工件）")
        for k, e in mm.items():
            print(f"{k} v{e['version']}｜训练截止 {e.get('train_end','—')}｜"
                  f"{'热启动' if e.get('warm_started') else '冷启动'}｜{e.get('saved_at','—')}")
        return
    if args.export_models:
        print("已导出：", export_models(args.export_models)); return
    if args.import_models:
        m = import_models(args.import_models)
        print(f"已导入，共 {len(m.get('models', {}))} 个模型；"
              f"恢复学习状态表 {m.get('_restored_learning_tables', {})}"); return
    run(as_of=pd.Timestamp(args.as_of) if args.as_of else None,
        require_prices=args.require_prices)


if __name__ == "__main__":
    main()
