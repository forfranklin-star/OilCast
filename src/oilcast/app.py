"""Streamlit 交互式报告页面。

本地运行::
    streamlit run src/oilcast/app.py

部署到 Streamlit Community Cloud 后，每日 GitHub Actions 更新仓库内
reports/latest/latest.json，页面随之刷新；侧边栏可回溯任意历史报告。
只展示真实可追溯数据，缺失即明示，不做任何补齐。
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1])     # .../src
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # 项目根
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import pandas as pd
import streamlit as st

from oilcast.config import get_config
from oilcast.models.registry import export_models, import_models, list_models
from oilcast.reporting.narratives import FACTOR_CN, STATUS_CN
from oilcast.reporting.static_html import (HORIZON_CN, price_figure,
                                           scenario_figure, weights_figure,
                                           sensitivity_figure,
                                           relative_strength_figure)

st.set_page_config(page_title="多因素油价智能分析与预测系统", layout="wide", page_icon="🛢️")
CFG = get_config()


# --------------------------------------------------------------- 数据加载
@st.cache_data(ttl=300, show_spinner=False)
def list_archive_dates() -> list[str]:
    arch = Path(CFG["storage"]["archive_dir"])
    return sorted([p.stem for p in arch.glob("*.json")], reverse=True) if arch.exists() else []


@st.cache_data(ttl=300, show_spinner=False)
def load_report(date: str | None) -> dict | None:
    if date and date != "latest":
        path = Path(CFG["storage"]["archive_dir"]) / f"{date}.json"
    else:
        path = Path(CFG["storage"]["latest_dir"]) / "latest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def run_pipeline() -> None:
    cmd = [sys.executable, "-m", "oilcast.pipeline.main"]
    with st.spinner("正在采集真实数据、训练模型并生成报告；首次冷启动约 8~15 分钟"
                    "（需采集历史数据、训练并完成回测，取决于网络），之后每日增量运行约 3~6 分钟，"
                    "请勿关闭或刷新页面…"):
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT / "src"),
                              capture_output=True, text=True)
    if proc.returncode not in (0,):
        st.error(f"流水线失败（退出码 {proc.returncode}）：{proc.stderr[-1500:]}")
        st.info("若因运行过久被中断（operation was canceled）：每个品种训练完会立即存盘，"
                "再点一次本按钮即可对已完成品种秒级续跑、只补未完成品种，无需从头开始。")
    else:
        st.success("报告已更新")
        st.cache_data.clear()


def _is_ok(item) -> bool:
    return isinstance(item, dict) and item.get("status", "ok") == "ok" and "endpoint" in item


def _export_bundle_bytes() -> bytes:
    """把完整学习快照导出到临时 zip 并读回字节，供浏览器下载到本地电脑。"""
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tf:
        tmp = tf.name
    try:
        export_models(tmp)
        return Path(tmp).read_bytes()
    finally:
        Path(tmp).unlink(missing_ok=True)


# --------------------------------------------------------------- 侧边栏
with st.sidebar:
    st.header("⚙️ 报告控制台")
    dates = list_archive_dates()
    chosen = st.selectbox("历史报告存档", ["latest"] + dates,
                          format_func=lambda x: "最新报告" if x == "latest" else x)
    target = st.radio("价格标的",
                      ["wti", "brent", "shanghai_crude",
                       "heating_oil", "gasoil"],
                      format_func=lambda x: {"wti": "WTI原油", "brent": "布伦特原油",
                                             "shanghai_crude": "上海原油INE SC",
                                             "heating_oil": "美燃油NYMEX超低硫柴油",
                                             "gasoil": "伦敦柴油ICE Gasoil"}[x])
    st.divider()
    if st.button("🔄 立即重新生成报告", width="stretch"):
        run_pipeline()
        st.rerun()
    st.caption("每日北京时间 09:00 由 GitHub Actions 自动运行")

    # -------- 模型学习状态的导出（下载到本地）/ 导入（从本地上传恢复）--------
    st.divider()
    st.markdown("**模型备份与恢复（持续学习不丢失）**")
    n_models = len(list_models())
    st.caption(f"当前已保存 {n_models} 个模型工件。快照含模型、版本链、因素权重、"
               "历史预测与报告索引，可下载到本地备份，或上传到另一套环境在原进度续学。")
    if st.button("① 生成学习快照", width="stretch"):
        with st.spinner("正在打包…"):
            st.session_state["bundle_bytes"] = _export_bundle_bytes()
    if st.session_state.get("bundle_bytes"):
        st.download_button(
            "② 下载 bundle.zip 到本地", st.session_state["bundle_bytes"],
            file_name="oilcast-learning-bundle.zip",
            mime="application/zip", width="stretch")
    up = st.file_uploader("从本地 bundle.zip 导入恢复", type=["zip"],
                          help="上传此前导出的快照，恢复模型与全部学习状态，之后即在原进度热启动")
    if up is not None and st.button("确认导入并恢复", width="stretch", type="primary"):
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tf:
            tf.write(up.read()); in_zip = tf.name
        try:
            with st.spinner("正在导入…"):
                m = import_models(in_zip)
            st.success(f"已恢复 {len(m.get('models', {}))} 个模型；学习状态表 "
                       f"{m.get('_restored_learning_tables', {})}，下次生成即在原进度热启动。")
            st.cache_data.clear()
        finally:
            Path(in_zip).unlink(missing_ok=True)

report = load_report(chosen)
st.title("🛢️ 多因素油价智能分析与预测系统")
if report is None:
    st.warning("尚未找到报告。请在左侧点击「立即重新生成报告」，"
               "或在终端执行 `python -m oilcast.pipeline.main`。")
    st.stop()

st.markdown(f"**报告日期：{report['report_date']}** ｜ 生成于 {report['generated_at']}")
_fr = report.get("freshness") or {}
if _fr.get("level") == "unchanged":
    st.error(f"本期数据未更新：{_fr.get('message','')}")
elif _fr.get("level") == "lagging":
    st.warning(f"价格数据滞后：{_fr.get('message','')}")
_target_order = ["wti", "brent", "shanghai_crude", "heating_oil", "gasoil"]
_target_order = [t for t in _target_order if t in report["current_meta"]]
for col, t in zip(st.columns(len(_target_order)), _target_order):
    meta = report["current_meta"][t]
    if meta["status"] == "ok":
        col.metric(report["names"][t],
                   f"{meta['value']:.2f} {report['units'][t]}",
                   f"观测 {meta['observed_date']}")
    else:
        col.metric(report["names"][t], "不可用")

with st.expander("📋 数据谱系与质量门（字段状态 / 真实来源 / 末次观测日期 / 样本数）"):
    lin = pd.DataFrame(report.get("lineage", {}).values())
    if not lin.empty:
        lin["状态"] = lin["status"].map(STATUS_CN).fillna(lin["status"])
        if "tried_sources" not in lin.columns:
            lin["tried_sources"] = ""
        st.dataframe(lin[["field", "display", "状态", "source_name",
                          "last_observed", "n_obs", "tried_sources"]].rename(
            columns={"field": "字段", "display": "含义", "source_name": "命中来源",
                     "last_observed": "末次观测", "n_obs": "样本数",
                     "tried_sources": "数据源优先级尝试链"}),
            width="stretch", hide_index=True)

# --------------------------------------------------------------- 预测卡片（跟随所选标的）
_sel_cn = report["names"][target]
_sel_unit = report["units"][target]
st.subheader(f"多周期预测卡片（{_sel_cn}）")
cols = st.columns(3)
for col, hz in zip(cols, ("short", "mid", "long")):
    item = report["forecasts"][hz].get(target)
    with col:
        st.markdown(f"#### {HORIZON_CN[hz]}")
        if not _is_ok(item):
            st.warning(f"暂不预测：{item.get('reason', '真实数据不可用')}")
            continue
        ep = item["endpoint"]
        st.caption(f"截至 {ep['target_date']}（观测日 {item.get('observed_date','—')}）")
        st.metric(f"预测均值（{_sel_unit}）", f"{ep['mean']:.2f}", delta=f"{ep['pct_mean']:+.2f}%")
        stance = ep.get("dir_stance", "中性")
        edge_note = ""
        if ep.get("dir_edge_basis") == "趋势期望":
            edge_note = (f"｜趋势跟随（低胜率高盈亏比）：样本外顺势均收 "
                         f"{ep.get('dir_edge_mean_ret', '—')}%、盈亏比 {ep.get('dir_edge_payoff', '—')}")
        elif ep.get("dir_has_edge") and ep.get("dir_edge_hit") is not None:
            edge_note = f"｜样本外表态命中 {ep['dir_edge_hit']*100:.0f}%"
        elif stance == "中性":
            edge_note = "｜方向未达显著门控，以区间为准"
        st.markdown(f"方向立场：**{stance}**{edge_note}")
        st.markdown(f"95%区间 **{ep['q05']} ~ {ep['q95']}**　"
                    f"看涨 {ep['prob_up']*100:.0f}% / 看跌 {ep['prob_down']*100:.0f}%")
        st.progress(float(ep["prob_up"]), text=f"看涨概率 {ep['prob_up']*100:.0f}%")

# --------------------------------------------------------------- 走势图（跟随所选标的）
name_map = {"wti": ("wti", "WTI原油", "美元/桶"),
            "brent": ("brent", "布伦特原油", "美元/桶"),
            "shanghai_crude": ("shanghai_crude", "上海原油INE SC", "元/桶"),
            "heating_oil": ("heating_oil", "美燃油·NYMEX超低硫柴油", "美元/加仑"),
            "gasoil": ("gasoil", "伦敦柴油·ICE Gasoil", "美元/吨")}
key, cn, unit = name_map[target]
st.subheader(f"价格走势、预测曲线与概率区间（当前：{cn}）")
st.plotly_chart(price_figure(report, key, cn, unit), width="stretch")

tab_w, tab_s = st.tabs(["因素权重解释", "长期情景与机构锚"])
with tab_w:
    fig_w = weights_figure(report, target)
    if fig_w is None:
        st.warning("可用真实因素不足，本期不计算因素权重。")
    else:
        st.plotly_chart(fig_w, width="stretch")
    st.caption("下图为各品种独立学到的权重横向对比；下方文字为主锚品种的权重变化说明。")
    st.info(report["narratives"]["weights"])
    fig_s = sensitivity_figure(report)
    if fig_s is not None:
        st.markdown("**分品种因素敏感度差异（各油种独立学习）**")
        st.plotly_chart(fig_s, width="stretch")
    fig_rs = relative_strength_figure(report)
    if fig_rs is not None:
        st.markdown("**各油种相对布伦特的强弱（归一化净值）**")
        st.plotly_chart(fig_rs, width="stretch")
    if report["narratives"].get("sensitivity"):
        st.info(report["narratives"]["sensitivity"])
with tab_s:
    item = report["forecasts"]["long"].get(target)
    if _is_ok(item):
        cc1, cc2 = st.columns(2)
        with cc1:
            st.plotly_chart(scenario_figure(report, target), width="stretch")
        with cc2:
            ep = item["endpoint"]
            st.markdown(f"**长期关键节点（{_sel_cn}，{_sel_unit}）**")
            if "checkpoints" in ep:
                df_cp = pd.DataFrame(ep["checkpoints"]).T
                df_cp.columns = ["均值", "5%分位", "95%分位"]
                st.dataframe(df_cp, width="stretch")
            if ep.get("institution_anchor"):
                st.metric("机构目标价中位数（真实抽取）", f"{ep['institution_anchor']:.1f} {_sel_unit}")
        st.info(report["narratives"].get("long", {}).get(target)
                or report["narratives"].get("long_wti", ""))
    else:
        st.warning(f"长期预测暂不可用：{item.get('reason','真实数据不可用') if item else '该品种无长期预测'}")

# --------------------------------------------------------------- 事件列表
st.subheader("关键事件与量化影响（真实新闻）")
ev = pd.DataFrame(report["events"])
if not ev.empty:
    filt = st.radio("时间范围", ["一周", "一月"], horizontal=True)
    ev["date"] = pd.to_datetime(ev["date"])
    cutoff = pd.Timestamp(report["report_date"]) - pd.Timedelta(days=7 if filt == "一周" else 30)
    show = ev[ev["date"] >= cutoff].copy()
    # RSS 新闻只到“日”、没有时刻：统一显示为 YYYY-MM-DD，避免渲染出无意义的 00:00:00
    show["日期"] = show["date"].dt.strftime("%Y-%m-%d")
    show["因素主题"] = show["theme"].map(FACTOR_CN).fillna(show["theme"])
    show["强度"] = (show["intensity"] * 100).round(0).astype(int).astype(str) + "%"
    st.dataframe(show[["日期", "title", "source", "因素主题", "强度", "est_price_impact"]].rename(
        columns={"title": "事件", "source": "来源",
                 "est_price_impact": "估算影响(美元/桶)"}),
        width="stretch", hide_index=True)
    for t in report["narratives"]["events"][:5]:
        st.markdown(f"- {t}")
else:
    st.caption("事件源不可达或无有效真实条目，按数据原则不展示任何模拟事件。")

# --------------------------------------------------------------- 机构观点
st.subheader("机构观点与目标价（真实抽取）")
vw = pd.DataFrame(report["views"])
if not vw.empty:
    st.dataframe(vw.rename(columns={"date": "日期", "institution": "机构",
                                    "target_wti": "WTI目标价", "stance": "方向",
                                    "note": "摘要"}), width="stretch", hide_index=True)
else:
    _iv = (report.get("lineage") or {}).get("institutional_view") or {}
    _tried = _iv.get("tried_sources") or _iv.get("note") or ""
    st.caption("本期各机构观点源未抽取到可核验的评级/目标价，按数据原则保持空缺、不以模拟观点补齐。"
               + (f"尝试情况：{_tried}" if _tried else ""))

# ----------------------------------------------- 模型学习 / 回测 / 预测复盘
st.subheader("模型学习、回测与预测复盘（每日用最新真实数据重训并复测）")
ml = report.get("model_learning") or {}
if ml:
    if report["narratives"].get("learning"):
        st.info(report["narratives"]["learning"])
    st.caption(f"本期重训时刻：{ml.get('retrained_at','—')}｜拟合滚动窗：最近 "
               f"{ml.get('train_window','—')} 个交易日｜累计真实样本 "
               f"{(next(iter((ml.get('train_meta') or {}).values()), {}) or {}).get('n_valid','—')} 行｜累计第 {ml.get('n_runs','—')} 期")
    tm = ml.get("train_meta", {})
    pers = ml.get("persistence", {}).get("models", {})
    if tm:
        def _ver(k):
            e = pers.get(k, {})
            if not e:
                return "—"
            if e.get("warm_started"):
                return f"v{e.get('version')} 热启动自v{e.get('parent_version')}"
            return f"v{e.get('version')} 冷启动"
        rows = [{"标的": report["names"].get(k, k), "训练区间": f"{v['train_start']} ~ {v['train_end']}",
                 "有效样本行": v["n_valid"], "逐日模型数": v["n_steps"],
                 "模型版本/学习方式": _ver(k)} for k, v in tm.items()]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption("模型每期落盘并在下期热启动、增量训练。可用左侧「模型备份与恢复」一键"
                   "导出下载到本地、或上传 zip 导入恢复（快照含模型+版本链+因素权重+历史预测+"
                   "报告索引），删库/换机后在原进度续学、不从零；命令行等价操作见 README。")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**滚动样本外回测（短期）**")
        bt = ml.get("backtest", {})
        if bt.get("available"):
            b1, b2, b3, b4, b5 = st.columns(5)
            b1.metric("MAE", f"{bt['mae_pct']}%", f"随机游走 {bt['benchmark_mae_pct']}%")
            b2.metric("收益相关IC", f"{bt.get('ic', 0):.2f}")
            eng = bt.get("stance_engaged_accuracy")
            gate = "门控通过" if bt.get("gate_has_edge") else "门控未过/中性为主"
            b3.metric("表态时方向命中",
                      (f"{eng*100:.0f}%" if eng is not None else "—"),
                      f"p={bt.get('stance_engaged_p', '—')}")
            basis = bt.get("gate_edge_basis")
            b4.metric("明确表态占比", f"{bt.get('stance_engagement_rate', 0)*100:.0f}%",
                      f"中性 {bt.get('stance_neutral_rate', 0)*100:.0f}%｜{gate}")
            b5.metric("概率Brier", f"{bt.get('direction_brier', '—')}")
            # 经济价值行：胜率≠盈利能力，单独考核表态子集的期望收益/盈亏比/夏普
            e1, e2, e3 = st.columns(3)
            e1.metric("表态平均收益",
                      (f"{bt['stance_engaged_mean_ret_pct']:+.2f}%"
                       if bt.get("stance_engaged_mean_ret_pct") is not None else "—"),
                      "按立场持有该周期")
            e2.metric("表态盈亏比", f"{bt.get('stance_engaged_payoff', '—')}",
                      "平均盈利/平均亏损")
            e3.metric("表态近似年化夏普", f"{bt.get('stance_engaged_sharpe', '—')}",
                      f"无条件持有均收 {bt.get('buyhold_mean_ret_pct', '—')}%")
            cache_note = (f"回测为慢变历史评估，本期复用 {bt.get('cached_as_of')} 缓存结果（7 天内到期重算）；"
                          if bt.get("cached") else "")
            st.caption(
                cache_note
                + f"{bt.get('window_start','')}~{bt.get('window_end','')} 共 {bt['n_origins']} 个样本外原点；"
                "方向为看涨/看跌/中性三分类，中性是弱有效市场下诚实选择、不计为错误；"
                + (f"本期方向 edge 来源：{basis}；" if basis else "")
                + "门控按【胜率显著高于50%】或【趋势跟随期望收益为正、盈亏比>1】任一判据放行；"
                + (f"幅度 MAE 较随机游走降低 {bt.get('mae_improve_pct', 0)}%。"
                   if bt["mae_pct"] < bt["benchmark_mae_pct"]
                   else "幅度预测在随机游走基准附近，价值主要体现在区间刻画；日频胜率围绕 50% 属正常，需结合盈亏比判断盈利能力。"))
        else:
            st.caption(bt.get("reason", "真实样本积累中，回测暂不可用"))
    with c2:
        st.markdown("**历史预测 vs 已实现真实价（复测）**")
        rv = ml.get("review", {})
        if rv.get("available"):
            sm = rv["summary"]
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("已到期预测", f"{sm['n']} 条",
                      f"明确表态 {sm.get('engaged_n', 0)}、中性 {sm.get('neutral_n', 0)}")
            r2.metric("平均误差", f"{sm['mae_pct']}%")
            r3.metric("表态方向命中", f"{sm['dir_acc']}%" if sm["dir_acc"] is not None else "—")
            r4.metric("表态平均收益/盈亏比",
                      (f"{sm['engaged_mean_ret_pct']:+.2f}%"
                       if sm.get("engaged_mean_ret_pct") is not None else "—"),
                      f"盈亏比 {sm.get('engaged_payoff', '—')}")
            cap = (f"95%区间覆盖真实价比例：{sm['coverage95']}%"
                   if sm["coverage95"] is not None else "区间覆盖统计积累中")
            cap += "；中性预测不参与方向命中，胜率与盈亏比分开考核"
            st.caption(cap)
        else:
            st.caption(rv.get("reason", "历史预测目标日尚未到期，下一期起复测"))
    wd = pd.DataFrame(ml.get("weight_delta", []))
    if not wd.empty:
        st.markdown(f"**权重自适应更新（对比 {ml.get('prev_weight_date') or '上一期'}）**")
        st.dataframe(wd.rename(columns={"factor": "因素", "prev": "上期权重%",
                                        "now": "本期权重%", "delta_pp": "变化(pp)"}),
                     width="stretch", hide_index=True)
    det = pd.DataFrame((ml.get("review") or {}).get("detail", []))
    if not det.empty:
        st.markdown("**最近预测复盘明细（预测 vs 后续真实收盘）**")
        det = det.copy()
        det["instrument"] = det["instrument"].map(report["names"]).fillna(det["instrument"])
        st.dataframe(det.rename(columns={"report_date": "发布日", "target_date": "目标日",
                                         "horizon": "周期", "instrument": "标的", "base": "发布时价",
                                         "pred": "预测", "actual": "实际", "pred_ret_pct": "预测涨跌%",
                                         "actual_ret_pct": "实际涨跌%", "abs_err_pct": "误差%",
                                         "dir_hit": "方向命中", "covered": "落95%区间"}),
                     width="stretch", hide_index=True)
else:
    st.caption("旧版本报告缺少学习记录，重新运行后生成。")

st.subheader("文字解读")
for t in _target_order:
    cn2 = report["names"][t]
    with st.expander(f"{cn2}：历史与两周/三个月预测解读", expanded=(t == target)):
        st.write("**历史走势**：" + report["narratives"]["trends"][t])
        st.write("**两周预测**：" + report["narratives"]["short"][t])
        st.write("**三个月预测**：" + report["narratives"]["mid"][t])

st.caption(report["narratives"]["sources"] +
           " ｜ 缺失、过期或无法验证的数据一律不补齐；预测不构成投资建议。")
