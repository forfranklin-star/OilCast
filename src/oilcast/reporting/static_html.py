"""生成自包含静态 HTML 报告（Plotly + 数据内联）。

数据诚实性呈现：
- 历史价格缺口 connectgaps=False（断开，不用线"补"出不存在的观测）；
- 每个当前值标注真实观测日期；不可用周期/标的显示原因而非假数字；
- 顶部数据谱系表逐字段列出来源、末次观测、样本数与质量门状态。
"""
from __future__ import annotations

import json
import math
from typing import List, Optional

import shutil
from pathlib import Path

import plotly
import plotly.graph_objects as go
from jinja2 import Template

from ..config import get_config
from ..features.holidays import HOLIDAY_COLUMNS

# 长假窗口 → 中文标签
HOLIDAY_CN = {
    "holiday_cn_spring_pre": "春节·节前一周",
    "holiday_cn_spring_now": "春节·除夕至初七",
    "holiday_summer_peak": "暑假/驾驶旺季(7–8月)",
    "holiday_cn_national": "国庆黄金周",
    "holiday_yearend": "年末圣诞新年",
}
from .narratives import FACTOR_CN, STATUS_CN


def ensure_plotly_asset() -> str:
    """把 plotly bundle 复制到 reports/assets，HTML 以相对路径引用。

    相比外网 CDN：国内/受限网络打开即渲染、无第三方追踪；相比逐份内联 4.5MB，
    30 份历史报告共享同一份 JS，仓库体积可控。返回相对 HTML 文件的引用路径。
    """
    src = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    assets = Path(get_config()["storage"]["latest_dir"]).parent / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    dst = assets / "plotly.min.js"
    if not dst.exists():
        shutil.copy(src, dst)
    return "../assets/plotly.min.js"   # latest/ 与 archive/ 同级，相对路径一致

HORIZON_CN = {"short": "短期·两周", "mid": "中期·三个月", "long": "长期·十二个月"}
H_COLOR = {"short": "#1f6feb", "mid": "#e08a00", "long": "#b03434"}
STATUS_COLOR = {"ok": "#2e7d32", "stale": "#e08a00", "insufficient": "#e08a00",
                "unavailable": "#b03434"}


def _is_ok(item: Optional[dict]) -> bool:
    return isinstance(item, dict) and item.get("status", "ok") == "ok" and "path" in item


def price_figure(report: dict, target: str, title: str, unit: str) -> go.Figure:
    hist = report["history"]
    hx = [str(r["date"]) for r in hist]
    hy = [r.get(target) for r in hist]
    meta = report["current_meta"].get(target, {})
    anchor_x, anchor_y = meta.get("observed_date"), meta.get("value")
    # —— 交易日等距类别轴：横轴只放“实际存在的交易日”，按顺序等距排列 ——
    # 不使用自然日历(date)轴：那样周末/节假日会在坐标上留白，使曲线出现视觉断点。
    # 类别轴下非交易日根本不占位置，从物理上保证“以交易时间为准、非交易日不是缺口”。
    cat_list = list(hx)
    for hz0 in ("short", "mid", "long"):
        it0 = report["forecasts"][hz0].get(target)
        if _is_ok(it0):
            for r in it0["path"]:
                d = str(r["date"])
                if d not in cat_list:
                    cat_list.append(d)
    if anchor_x and str(anchor_x) not in cat_list:
        cat_list.append(str(anchor_x))
    fig = go.Figure()
    # 历史真实观测：真实数据内部缺口保持断开（connectgaps=False）
    fig.add_trace(go.Scatter(x=hx, y=hy, name="历史真实观测", connectgaps=False,
                             line=dict(color="#2c3e50", width=2)))
    for hz in ("short", "mid", "long"):
        item = report["forecasts"][hz].get(target)
        if not _is_ok(item):
            continue
        recs = item["path"]
        x = [str(anchor_x)] + [str(r["date"]) for r in recs]
        c = H_COLOR[hz]
        for lo_, hi_, alpha, lbl in (("q05", "q95", 0.08, "95%区间"),
                                     ("q25", "q75", 0.14, "50%区间")):
            yb = [anchor_y] + [r[hi_] for r in recs]
            ya = [anchor_y] + [r[lo_] for r in recs]
            fig.add_trace(go.Scatter(
                x=x + x[::-1], y=yb + ya[::-1],
                fill="toself", fillcolor=_hex_alpha(c, alpha), line=dict(width=0),
                hoverinfo="skip", showlegend=False, name=f"{HORIZON_CN[hz]}{lbl}"))
        fig.add_trace(go.Scatter(x=x, y=[anchor_y] + [r["mean"] for r in recs],
                                 name=f"{HORIZON_CN[hz]}预测均值",
                                 line=dict(color=c, width=2, dash="dash")))
    obs = f"，末次真实观测 {anchor_x}" if anchor_x else "（无真实观测）"
    if anchor_x is None:
        fig.add_annotation(text="该标的暂无可核验真实数据，按数据原则不展示预测",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                           font=dict(size=15, color="#b03434"))
        fig.update_xaxes(visible=False).update_yaxes(visible=False)
    fig.update_layout(
        title=f"{title} 历史走势与多周期预测（{unit}）{obs}",
        height=430, margin=dict(l=50, r=20, t=56, b=40),
        plot_bgcolor="white", hovermode="x unified",
        legend=dict(orientation="h", y=-0.18),
        font=dict(family="-apple-system,Segoe UI,Microsoft YaHei", size=12))
    # 交易日等距类别轴：按 cat_list 的交易日顺序排列，非交易日不占位、无断点
    fig.update_xaxes(showgrid=True, gridcolor="#eef1f5", type="category",
                     categoryorder="array", categoryarray=cat_list,
                     tickmode="auto", nticks=12, tickangle=0)
    fig.update_yaxes(showgrid=True, gridcolor="#eef1f5")
    return fig


def weights_figure(report: dict, target: str | None = None) -> Optional[go.Figure]:
    def _is_num(v):
        return v is not None and v == v  # 排除 None 与 float('nan')
    # 指定品种时用该品种独立学到的权重；否则用主锚那套（静态概览页）
    if target and report.get("weights_by_target", {}).get(target):
        records = report["weights_by_target"][target]
        tgt_cn = report.get("names", {}).get(target, target)
    else:
        records = report["weights"]
        tgt_cn = None
    usable = sorted((r for r in records if _is_num(r.get("weight"))),
                    key=lambda r: r["weight"])
    missing = [r for r in records if not _is_num(r.get("weight"))]
    if not usable:
        return None

    def _missing_reason(factor: str) -> str:
        # 区分“源不通”与“源通但窗口内确无该类真实事件”，避免被误读为程序故障
        lin = (report.get("lineage") or {}).get(factor) or {}
        st, n = lin.get("status"), lin.get("n_obs")
        if st == "unavailable":
            return "不可用（本期未取到可核验来源）"
        if st == "ok" and (n == 0 or n is None):
            return "本期窗口内无该类真实事件，暂不计权重"
        return "不可用（无真实数据）"

    y = [FACTOR_CN.get(r["factor"], r["factor"]) for r in usable]
    fig = go.Figure(go.Bar(
        x=[r["weight"] * 100 for r in usable], y=y, orientation="h",
        marker_color="#1f6feb",
        text=[f"{r['weight']*100:.1f}%" for r in usable], textposition="outside"))
    if missing:  # 缺失因素：灰色零长条 + 具体原因，绝不显示 nan%
        fig.add_trace(go.Bar(
            x=[0.0] * len(missing),
            y=[FACTOR_CN.get(r["factor"], r["factor"]) for r in missing],
            orientation="h", marker_color="#d9d9d9", showlegend=False,
            text=[_missing_reason(r["factor"]) for r in missing],
            textposition="outside"))
    title = ("多因素权重排名（灰色＝无真实数据、不参与归一化）" if not tgt_cn
             else f"{tgt_cn}·多因素权重排名（该品种独立学习）")
    fig.update_layout(title=title,
                      height=380, margin=dict(l=210, r=150, t=56, b=30),
                      xaxis_title="归一化权重 (%)", plot_bgcolor="white",
                      font=dict(size=12), showlegend=False)
    return fig


def sensitivity_figure(report: dict) -> Optional[go.Figure]:
    """分品种因素敏感度热力图：行=九大因素，列=各油种，值=该品种学到的权重(%)。

    直观呈现"不同油种对地缘冲突/美联储/CPI/非农等变量敏感度不同"，缺失为灰格。"""
    sens = report.get("sensitivity") or {}
    if not sens:
        return None
    targets = [t for t in report["names"].keys()
               if any(sens.get(f, {}).get(t) is not None for f in sens)]
    if not targets:
        return None
    # 因素顺序沿用主锚权重排序，缺失因素排末尾
    order = [r["factor"] for r in report.get("weights", [])]
    factors = order + [f for f in sens if f not in order]
    z, txt = [], []
    for f in factors:
        zrow, trow = [], []
        for t in targets:
            v = sens.get(f, {}).get(t)
            zrow.append(v)
            trow.append("—" if v is None else f"{v:.1f}%")
        z.append(zrow)
        txt.append(trow)
    fig = go.Figure(go.Heatmap(
        z=z, x=[report["names"][t] for t in targets],
        y=[FACTOR_CN.get(f, f) for f in factors],
        text=txt, texttemplate="%{text}", textfont={"size": 11},
        colorscale="Blues", zmin=0, zmax=40,
        colorbar=dict(title="权重%"), hoverongaps=False))
    fig.update_layout(title="分品种因素敏感度对比（近 500 交易日全窗平均权重，%；会稀释当前冲突，见下方事件期条件敏感度）",
                      height=430, margin=dict(l=150, r=20, t=70, b=30),
                      font=dict(size=12), plot_bgcolor="white")
    return fig


def conditional_sensitivity_figure(report: dict) -> Optional[go.Figure]:
    """事件期条件敏感度分组条形：各品种 事件期地缘β抬升(beta_lift) 与 事件期累计涨跌%。

    用来纠正"500 日平均权重把当前冲突期敏感度稀释"的误读——例如上海原油长期平均地缘
    权重不高，但在当前中东/霍尔木兹事件窗口内 β 抬升与累计涨幅居前。"""
    cs = report.get("conditional_sensitivity") or {}
    items = [(t, d) for t, d in cs.items() if isinstance(d, dict) and d.get("available")
             and d.get("event_cumret_pct") is not None]
    if len(items) < 2:
        return None
    names = [report["names"][t] for t, _ in items]
    lift = [round(float(d["beta_lift"]), 3) if d.get("beta_lift") is not None else None
            for _, d in items]
    cum = [round(float(d["event_cumret_pct"]), 1) for _, d in items]
    excess = [round(float(d["event_excess_vs_brent_pct"]), 1)
              if d.get("event_excess_vs_brent_pct") is not None else None for _, d in items]
    fig = go.Figure()
    # 左轴：事件期累计涨跌% 与 相对布伦特异超额%（分组柱）；右轴：地缘β抬升（折线）。
    # "相对布伦特异超额"直接回答冲突中谁比布伦特涨得更多——对华供油链路受扰时上海居前。
    fig.add_trace(go.Bar(name="事件期累计涨跌%", x=names, y=cum,
                         marker_color="#b03434", text=cum, textposition="outside", yaxis="y"))
    fig.add_trace(go.Bar(name="相对布伦特异超额%", x=names, y=excess,
                         marker_color="#e08a2b", text=excess, textposition="outside", yaxis="y"))
    lift_vals = [x for x in lift if x is not None]
    fig.add_trace(go.Scatter(name="事件期地缘β抬升(参考)", x=names, y=lift,
                             mode="lines+markers+text", text=lift, textposition="top center",
                             line=dict(color="#1f6feb", width=3), marker=dict(size=11), yaxis="y2"))
    fig.update_layout(
        title="当前重大事件窗口内的真实敏感度（价格直接统计，非长期平均权重）",
        height=400, margin=dict(l=50, r=50, t=56, b=46),
        plot_bgcolor="white", legend=dict(orientation="h", y=-0.22), font=dict(size=12),
        bargap=0.45, barmode="group")
    fig.update_yaxes(title="涨跌/超额 %", showgrid=True, gridcolor="#eef1f5")
    if lift_vals:
        fig.update_layout(yaxis2=dict(title="β抬升", overlaying="y", side="right",
                                      showgrid=False, range=[0, max(lift_vals + [0.1]) * 1.35]))
    return fig


def holiday_impact_figure(report: dict) -> Optional[go.Figure]:
    """主要用油地区长假窗口的历史 5 日平均收益热力图（行=长假、列=品种）。

    纯描述性：呈现进入各长假窗口持有 5 日历史上的平均收益、上涨概率与样本数，让长假因子
    可见、可核对；样本有限且整体季节性检验不显著，故明确标注"不构成方向预测"。"""
    rows = report.get("holiday_impacts") or []
    if not rows:
        return None
    names = report.get("names") or {}
    insts = [t for t in names if any(r["instrument"] == t for r in rows)]
    hols = [h for h in HOLIDAY_COLUMNS if any(r["holiday"] == h for r in rows)]
    if not insts or not hols:
        return None
    z, hover = [], []
    for h in hols:
        zrow, hrow = [], []
        for t in insts:
            r = next((r for r in rows if r["holiday"] == h and r["instrument"] == t), None)
            if r and r.get("mean_ret_pct") is not None:
                zrow.append(r["mean_ret_pct"])
                hrow.append(f"n={r['n']}，上涨占比 {round(r['up_prob'] * 100):.0f}%")
            else:
                zrow.append(None)
                hrow.append("无样本")
        z.append(zrow)
        hover.append(hrow)
    fig = go.Figure(go.Heatmap(
        z=z, x=[names[t] for t in insts], y=[HOLIDAY_CN[h] for h in hols],
        customdata=hover, colorscale="RdBu_r", zmid=0,
        colorbar=dict(title="收益%"),
        hovertemplate="%{y} · %{x}<br>5日平均收益 %{z}%<br>%{customdata}<extra></extra>"))
    fig.update_traces(text=[[("" if v is None else f"{v}") for v in row] for row in z],
                      texttemplate="%{text}")
    fig.update_layout(
        title="主要用油地区长假窗口·历史 5 日平均收益（描述性，样本有限，不构成方向预测）",
        height=300, margin=dict(l=10, r=20, t=56, b=40), font=dict(size=12))
    return fig


def _finite_value(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) \
        and math.isfinite(v)


def report_history_coverage(report: dict, n: int = 130,
                            min_per: int = 60) -> dict:
    """最新报告 history（近 n 交易日）各品种 finite 点数与完整性判定。

    用于云端门禁：latest 报告五品种（上海 INE 2018 年上市后）近窗都应有足量
    数据；某品种有效点 < min_per 即判残缺（典型：海外源全失败、只有上海）。
    min_per 取宽松值，区分"真残缺(个位数)"与"正常(120+)"，并容忍交易日历差异。"""
    hist = (report.get("history") or [])[-n:]
    targets = list((report.get("names") or {}).keys())
    per = {t: sum(1 for r in hist if _finite_value(r.get(t)))
           for t in targets}
    missing = [t for t in targets if per[t] < min_per]
    return {"window": len(hist), "per": per, "min_required": min_per,
            "missing": missing, "complete": not missing}


def relative_strength_figure(report: dict, n: int = 130,
                            with_title: bool = True) -> Optional[go.Figure]:
    """各油种归一化净值（窗口首日=100），刻画上海原油等相对布伦特的强弱。

    用各自本币计的累计涨跌倍数比较，无量纲、跨币种也成立；不做绝对价差（口径不同）。

    with_title : 静态 HTML 卡片外无标题、用 True（图自带标题）；Streamlit 外部已有
    markdown 标题、传 False，避免标题重复。"""
    hist = report.get("history") or []
    if not hist:
        return None
    hist = hist[-n:]
    targets = [t for t in report["names"].keys()]
    rows = [r for r in hist if any(r.get(t) is not None for t in targets)]
    if len(rows) < 20:
        return None
    cat = [str(r["date"]) for r in rows]
    palette = {"wti": "#2c3e50", "brent": "#1f6feb",
               "shanghai_crude": "#b03434", "heating_oil": "#e08a00",
               "gasoil": "#2e7d32"}
    fig = go.Figure()
    min_pts = 5  # 有效点少于此画不出有意义的线：跳过且不在图例留空名，避免单点冒充曲线
    for t in targets:
        ys = [r.get(t) for r in rows]
        have = [i for i, v in enumerate(ys) if _finite_value(v)]
        if len(have) < min_pts:
            # 有效点不足：不 add_trace，故图例也不会留下"有名无线"的空名
            continue
        base = ys[have[0]]
        norm = [v / base * 100 if _finite_value(v) else None for v in ys]
        fig.add_trace(go.Scatter(
            x=cat, y=norm, name=report["names"][t], connectgaps=False,
            hovertemplate="%{fullData.name}: %{y:.2f} 净值<extra></extra>",
            line=dict(width=2.4 if t == "brent" else 1.4,
                      color=palette.get(t, "#888"),
                      dash="solid")))
    layout = dict(
        height=400, margin=dict(l=50, r=20, t=56 if with_title else 18, b=40),
        plot_bgcolor="white", hovermode="x unified",
        legend=dict(orientation="h", y=-0.18), font=dict(size=12))
    if with_title:
        layout["title"] = (f"各油种相对强弱（近{n}个交易日归一化净值，起点=100；"
                           f"无量纲、非美元绝对价；高于布伦特=更强）")
    fig.update_layout(**layout)
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=cat,
                     nticks=12, showgrid=True, gridcolor="#eef1f5")
    fig.update_yaxes(title="归一化净值（起点100，非美元）", showgrid=True, gridcolor="#eef1f5")
    return fig


def session_premium_figure(report: dict) -> Optional[go.Figure]:
    """上海原油(换算美元/桶) 与 布伦特 在【北京02:30上海夜盘收盘同一真实时刻】的升贴水时序。

    左轴：同时刻的上海美元价与布伦特价；右轴柱：升贴水(美元/桶，>0 升水、<0 贴水)。
    只用盘中分时同时刻价，不用日 K 错位拼接；无分时的更早日期不画线。"""
    ss = report.get("session_sync") or {}
    if not ss.get("available"):
        return None
    curve = ss.get("premium_series") or []
    curve = [r for r in curve if r.get("premium_usd") is not None]
    if len(curve) < 8:
        return None
    cat = [r["date"] for r in curve]
    prem = [r["premium_usd"] for r in curve]
    colors = ["#b03434" if (v is not None and v >= 0) else "#2e7d32" for v in prem]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=cat, y=prem, name="上海-布伦特 升贴水(美元/桶)",
                         marker_color=colors, opacity=0.45, yaxis="y2"))
    fig.add_trace(go.Scatter(x=cat, y=[r["sc_usd"] for r in curve], name="上海原油(美元/桶,同时刻)",
                             line=dict(width=2.2, color="#b03434")))
    fig.add_trace(go.Scatter(x=cat, y=[r["brent_usd"] for r in curve], name="布伦特(美元/桶,同时刻)",
                             line=dict(width=2.2, color="#1f6feb")))
    fig.update_layout(
        title="上海原油换算美元后对布伦特的同时刻升贴水（北京02:30夜盘收盘同时刻，非日K错位）",
        height=410, margin=dict(l=50, r=50, t=56, b=40), plot_bgcolor="white",
        hovermode="x unified", legend=dict(orientation="h", y=-0.18), font=dict(size=12),
        bargap=0.1)
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=cat,
                     nticks=12, showgrid=True, gridcolor="#eef1f5")
    fig.update_layout(
        yaxis=dict(title="美元/桶", showgrid=True, gridcolor="#eef1f5"),
        yaxis2=dict(title="升贴水(美元/桶)", overlaying="y", side="right",
                    showgrid=False, zeroline=True, zerolinecolor="#999"))
    return fig


def scenario_figure(report: dict, target: str = "wti") -> Optional[go.Figure]:
    item = report["forecasts"]["long"].get(target)
    if not _is_ok(item):
        return None
    probs = item["endpoint"].get("scenario_probs", {})
    label = {"bull": "高油价", "base": "基准", "bear": "低油价"}
    fig = go.Figure(go.Pie(labels=[label.get(k, k) for k in probs],
                           values=[v * 100 for v in probs.values()],
                           marker=dict(colors=["#b03434", "#1f6feb", "#2e7d32"]),
                           hole=0.55, textinfo="label+percent"))
    fig.update_layout(title="长期情景概率", height=320, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def _hex_alpha(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _fig_json(fig) -> str:
    return json.loads(fig.to_json()) if fig is not None else None


def _cards(report: dict) -> List[dict]:
    # 主锚预测卡：优先 WTI，不可用退布伦特；其余品种在走势图与端点一览中呈现
    anchor_t = "wti" if _is_ok(report["forecasts"]["short"].get("wti")) else "brent"
    rows = []
    for hz in ("short", "mid", "long"):
        item = report["forecasts"][hz][anchor_t]
        if not _is_ok(item):
            rows.append({"horizon": HORIZON_CN[hz], "unavailable": True,
                         "reason": item.get("reason", "真实数据不可用")})
            continue
        ep = item["endpoint"]
        rows.append({"horizon": HORIZON_CN[hz], "unavailable": False, "mean": ep["mean"],
                     "range95": f"{ep['q05']} ~ {ep['q95']}", "pct": ep["pct_mean"],
                     "prob_up": ep["prob_up"] * 100, "prob_down": ep["prob_down"] * 100,
                     "stance": ep.get("dir_stance", "中性"),
                     "has_edge": bool(ep.get("dir_has_edge", False)),
                     "basis": ep.get("dir_edge_basis"),
                     "payoff": ep.get("dir_edge_payoff"),
                     "mret": ep.get("dir_edge_mean_ret"),
                     "target": ep["target_date"]})
    return rows


def _lineage_rows(report: dict) -> List[dict]:
    out = []
    for f, L in report.get("lineage", {}).items():
        out.append({"field": f, "display": L.get("display", ""),
                    "status": L.get("status", "unavailable"),
                    "status_cn": STATUS_CN.get(L.get("status"), L.get("status")),
                    "source": L.get("source_name", ""), "url": L.get("url", ""),
                    "last": L.get("last_observed") or "—", "n": L.get("n_obs", 0),
                    "note": L.get("note", ""), "tried": L.get("tried_sources", "")})
    return out


def render_static_html(report: dict) -> str:
    # 各品种走势图按 report.names 的品种顺序循环生成（含新增上海原油）
    figs = {f"fig_price_{t}": _fig_json(
                price_figure(report, t, report["names"][t], report["units"][t]))
            for t in report["names"].keys()}
    figs["fig_weights"] = _fig_json(weights_figure(report))
    figs["fig_sensitivity"] = _fig_json(sensitivity_figure(report))
    figs["fig_cond"] = _fig_json(conditional_sensitivity_figure(report))
    figs["fig_holiday"] = _fig_json(holiday_impact_figure(report))
    figs["fig_rs"] = _fig_json(relative_strength_figure(report))
    figs["fig_session_premium"] = _fig_json(session_premium_figure(report))
    figs["fig_scenario"] = _fig_json(scenario_figure(report))
    targets = list(report["names"].keys())
    plotly_src = ensure_plotly_asset()
    return Template(TEMPLATE).render(
        report=report, FACTOR_CN=FACTOR_CN, targets=targets,
        figs_json=json.dumps(figs, ensure_ascii=False),
        cards=_cards(report), lineage_rows=_lineage_rows(report), plotly_src=plotly_src)


TEMPLATE = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>多因素油价智能分析报告 · {{ report.report_date }}</title>
<script src="{{ plotly_src }}" charset="utf-8"></script>
<style>
 :root{--bd:#2c3e50;--blue:#1f6feb;--bg:#f5f7fa;--card:#fff;--mut:#6b7785;}
 *{box-sizing:border-box}
 body{margin:0;font-family:-apple-system,Segoe UI,"Microsoft YaHei",sans-serif;
   background:var(--bg);color:#1f2733;line-height:1.6;
   display:grid;grid-template-columns:250px minmax(0,1fr)}
 .wrap{max-width:1160px;margin:0 auto;padding:28px 30px 40px;min-width:0;width:100%}
 /* 左侧固定导航：本地静态报告自带，双击即看、无需起任何服务 */
 .sidenav{position:sticky;top:0;height:100vh;overflow-y:auto;background:#10253c;color:#d7e3f0;
   padding:20px 14px;box-shadow:2px 0 10px rgba(10,30,55,.25)}
 .sidenav .brand{font-size:17px;font-weight:700;color:#fff;display:flex;align-items:center;gap:8px;margin-bottom:4px}
 .sidenav .rptdate{font-size:12px;color:#9db4cc;margin:0 0 12px;line-height:1.5}
 .sidenav .navgroup{font-size:11px;letter-spacing:.07em;color:#7e97b2;margin:16px 0 4px;padding:0 8px}
 .sidenav a{display:block;color:#cddbea;text-decoration:none;font-size:13.5px;padding:6px 10px;border-radius:8px;margin:1px 0}
 .sidenav a:hover,.sidenav a:focus{background:#1d3c5e;color:#fff}
 .sidenav .navcount{font-size:12px;color:#a9bdd2;padding:6px 10px;line-height:1.7}
 .sidenav .dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
 h2[id]{scroll-margin-top:16px}
 header.top{background:linear-gradient(120deg,#173a5e,#2c5f8a);color:#fff;padding:28px 32px;border-radius:14px}
 header.top h1{margin:0 0 6px;font-size:26px} .mut{color:var(--mut);font-size:13px}
 header.top .mut{color:#c9d8e8}
 .fresh-banner{margin:14px 0 4px;padding:13px 18px;border-radius:10px;font-size:14px;line-height:1.6;border:1px solid}
 .fresh-unchanged{background:#fdecec;border-color:#e5a3a3;color:#8a1f1f}
 .fresh-lagging{background:#fff5e6;border-color:#e6b566;color:#8a5a12}
 .grid{display:grid;gap:16px;margin:20px 0}
 .cards{grid-template-columns:repeat(3,1fr)}
 .cards.four{grid-template-columns:repeat(auto-fit,minmax(210px,1fr))}
 .cards.six{grid-template-columns:repeat(auto-fit,minmax(180px,1fr))}
 .card{background:var(--card);border-radius:12px;padding:18px 20px;box-shadow:0 1px 4px rgba(20,40,70,.08)}
 .card h3{margin:0 0 8px;font-size:15px;color:var(--bd)}
 .big{font-size:30px;font-weight:700;color:var(--bd)} .up{color:#b03434}.down{color:#2e7d32}
 .kv{display:flex;justify-content:space-between;font-size:13px;color:var(--mut);padding:2px 0}
 .unavail{background:#fbf1f1;border:1px dashed #b03434;border-radius:10px;padding:14px;color:#8a2b2b;font-size:13px}
 .obsline{font-size:13px;color:var(--mut);margin:6px 0 0}
 .two{grid-template-columns:2fr 1fr}
 table{width:100%;border-collapse:collapse;font-size:13px}
 th,td{padding:8px 10px;border-bottom:1px solid #edf0f4;text-align:left;vertical-align:top}
 th{background:#f0f4f9;color:var(--bd);font-weight:600}
 .narr{background:#eef4fb;border-left:4px solid var(--blue);padding:10px 14px;border-radius:0 8px 8px 0;
   margin:8px 0;font-size:14px}
 h2{font-size:19px;margin:26px 0 10px;color:var(--bd);border-left:5px solid var(--blue);padding-left:10px}
 .probbar{display:flex;height:8px;border-radius:6px;overflow:hidden;margin-top:6px}
 .probbar .u{background:#b03434}.probbar .d{background:#2e7d32}
 .st{display:inline-block;white-space:nowrap;font-weight:700;padding:2px 10px;
   border-radius:11px;color:#fff;font-size:12px;line-height:1.5;min-width:42px;text-align:center}
 footer{color:var(--mut);font-size:12px;margin:26px 0 10px}
 a{color:var(--blue)}
 @media(max-width:920px){
   body{grid-template-columns:1fr}
   .sidenav{position:static;height:auto}
 }
 @media(max-width:860px){.cards,.two{grid-template-columns:1fr}}
</style>
</head>
<body>
<nav class="sidenav">
 <div class="brand">🛢️ OilCast</div>
 <div class="rptdate">本地分析报告<br>报告日 {{ report.report_date }}<br>生成于 {{ report.generated_at }}（北京时间）</div>

 <div class="navgroup">价格品种</div>
 {% for t in targets %}<a href="#fig_price_{{ t }}">{{ report.names[t] }}</a>{% endfor %}

 <div class="navgroup">概览</div>
 <a href="#sec-lineage">数据谱系与质量门</a>
 <a href="#sec-obs">当前真实观测</a>
 <a href="#sec-cards">预测卡片</a>

 <div class="navgroup">价格与对齐</div>
 <a href="#sec-prices">走势与预测区间</a>
 <a href="#sec-session">同时刻对齐（盘中）</a>
 <a href="#sec-endpoints">预测端点一览</a>

 <div class="navgroup">模型</div>
 <a href="#sec-weights">因素权重与敏感度</a>
 <a href="#sec-learning">学习、回测与复盘</a>
 <a href="#sec-multiscale">多尺度闭环</a>

 <div class="navgroup">事件与观点</div>
 <a href="#sec-events">关键事件与影响</a>
 <a href="#sec-shock">事件冲击条件展望</a>
 <a href="#sec-views">机构观点</a>

 <div class="navgroup">解读</div>
 <a href="#sec-narr">文字解读</a>
 <a href="#sec-disclaimer">数据原则与声明</a>

 {% set cnt = namespace(ok=0, bad=0) %}
 {% for t in targets %}
   {% if report.current_meta[t].status == 'ok' %}{% set cnt.ok = cnt.ok + 1 %}
   {% else %}{% set cnt.bad = cnt.bad + 1 %}{% endif %}
 {% endfor %}
 <div class="navgroup">数据状态</div>
 <div class="navcount"><span class="dot" style="background:#2e7d32"></span>可用 {{ cnt.ok }} 个品种<br>
   <span class="dot" style="background:#b03434"></span>不可用/受限 {{ cnt.bad }} 个</div>

 {% if report.archive_dates %}
 <div class="navgroup">历史存档</div>
 <a href="../latest/index.html">最新报告</a>
 {% for d in report.archive_dates[:60] %}<a href="../archive/{{ d }}.html">{{ d }}</a>{% endfor %}
 {% endif %}
</nav>
<div class="wrap">
 <header class="top">
   <h1>多因素油价智能分析与预测报告</h1>
   <div class="mut">报告日期 {{ report.report_date }} ｜ 生成于 {{ report.generated_at }}（北京时间）</div>
 </header>
 {% if report.freshness and report.freshness.level == "fresh" %}
 <div class="obsline" style="margin:8px 0 2px">✓ {{ report.freshness.message }}</div>
 {% elif report.freshness %}
 <div class="fresh-banner fresh-{{ report.freshness.level }}" role="alert">
   <b>{{ "⚠ 运行时点早于收盘/数据未推进" if report.freshness.level == "unchanged" else "⚠ 价格数据滞后" }}：</b>
   {{ report.freshness.message }}
 </div>
 {% endif %}

 <h2 id="sec-lineage">数据谱系与质量门（每个数字都可追溯到来源与观测日期）</h2>
 <div class="card" style="overflow-x:auto">
 <table>
   <tr><th>字段</th><th>含义</th><th>状态</th><th>命中来源</th><th>末次观测</th><th>样本数</th><th>数据源优先级尝试链（✓采用 / ✗失败原因）</th></tr>
   {% for L in lineage_rows %}
   <tr>
     <td>{{ L.field }}</td><td>{{ L.display }}</td>
     <td><span class="st" style="background:{{ {'ok':'#2e7d32','stale':'#e08a00','insufficient':'#e08a00','unavailable':'#b03434'}[L.status] }}">{{ L.status_cn }}</span></td>
     <td>{% if L.url %}<a href="{{ L.url }}" target="_blank" rel="noopener">{{ L.source }}</a>{% else %}{{ L.source }}{% endif %}</td>
     <td>{{ L.last }}</td><td>{{ L.n }}</td><td class="mut" style="font-size:12px">{{ L.tried or L.note }}</td>
   </tr>
   {% endfor %}
 </table>
 </div>

 <h2 id="sec-obs">当前真实观测</h2>
 <div class="grid cards six">
 {% for t in targets %}
  <div class="card">
    <h3>{{ report.names[t] }}（{{ report.units[t] }}）</h3>
    {% if report.current_meta[t].status == 'ok' %}
      <div class="big">{{ report.current_meta[t].value }}</div>
      <div class="obsline">真实观测日：{{ report.current_meta[t].observed_date }} ｜ {{ report.current_meta[t].source }}</div>
    {% else %}
      <div class="unavail">不可用（{{ report.current_meta[t].status }}）：{{ report.current_meta[t].reason }}</div>
    {% endif %}
  </div>
 {% endfor %}
 </div>

 <h2 id="sec-cards">预测卡片（{{ report.names['wti'] if report.current_meta['wti'].status=='ok' else report.names['brent'] }}·主锚）</h2>
 <div class="grid cards">
 {% for c in cards %}
  <div class="card">
    <h3>{{ c.horizon }}{% if not c.unavailable %} <span class="mut">→ {{ c.target }}</span>{% endif %}</h3>
    {% if c.unavailable %}
      <div class="unavail">暂不预测<br>{{ c.reason }}</div>
    {% else %}
      <div class="big {{ 'up' if c.pct>=0 else 'down' }}">{{ c.mean }} <span style="font-size:15px">美元/桶</span></div>
      <div class="{{ 'up' if c.pct>=0 else 'down' }}" style="font-size:14px">
        {{ '▲' if c.pct>=0 else '▼' }} {{ c.pct }}%（相对观测日）</div>
      <div class="kv"><span>方向立场</span><b>{{ c.stance }}{% if c.stance!='中性' and c.basis=='趋势期望' %}（趋势跟随：顺势均收 {{ c.mret }}%、盈亏比 {{ c.payoff }}）{% elif c.stance!='中性' %}（门控通过）{% elif not c.has_edge %}（未达显著门控）{% endif %}</b></div>
      <div class="kv"><span>95%概率区间</span><span>{{ c.range95 }}</span></div>
      <div class="kv"><span>看涨/看跌</span><span>{{ '%.0f'|format(c.prob_up) }}% / {{ '%.0f'|format(c.prob_down) }}%</span></div>
      <div class="probbar"><div class="u" style="width:{{c.prob_up}}%"></div><div class="d" style="width:{{c.prob_down}}%"></div></div>
    {% endif %}
  </div>
 {% endfor %}
 </div>

 <h2 id="sec-prices">价格走势与预测区间（全部品种）</h2>
 {% for t in targets %}
 <div class="card" id="fig_price_{{ t }}" {% if not loop.first %}style="margin-top:16px"{% endif %}></div>
 {% endfor %}

 {% set ss = report.session_sync %}
 {% if ss and ss.available %}
 <h2 id="sec-session">跨市场交易时段·同一真实时刻对齐（盘中分时，非日K错位）</h2>
 <div class="card">
  <div class="narr">上海原油除日盘(09:00–15:00)外还有夜盘(北京21:00–次日02:30)，一个连续交易时段的<b>真正收盘是
  夜盘 02:30</b>（周五夜盘物理发生在周六凌晨 02:30，归属周五交易日）。系统以 <b>北京 02:30（上海夜盘收盘）为统计
  主节点</b>——此刻 WTI/Brent 欧美电子盘仍在交易（北京 02:30＝美东 14:30／伦敦 19:30），用分时 K 取各品种当时
  【已成交】的最近一根，得到严格同时刻截面；上海自身收盘价、美元换算与对布升贴水均以 02:30 为准，15:00 日盘截面
  仅作对照。分时覆盖窗口（{{ ss.coverage_start }} 起）共 <b>{{ ss.model_synchronized_days }}</b> 个交易日改用 02:30
  同时刻价，更早日期维持日频 as-of，两种口径标注、不混用；免费分时源历史有限，自部署起每日增量积累、面板会越来越长。</div>
  <div id="fig_session_premium"></div>
  {% if ss.fx_rate %}<div class="obsline" style="margin:4px 0 10px">人民币换算口径：1 USD = {{ ss.fx_rate }} CNY（真实市场汇率，末次观测 {{ ss.fx_date }}）；上海美元价 = 上海 02:30 收盘价(元/桶) ÷ 该汇率，再与同一时刻布伦特价算升贴水，汇率每日随 ECB/FRED 源链更新、可在数据谱系表 usd_cny 行核验。</div>{% endif %}
  <div class="grid two" style="margin-top:14px">
   <div style="overflow-x:auto">
     <h3>北京 02:30 夜盘收盘·同时刻截面（统计主节点，近 8 交易日）</h3>
     <table>
       <tr><th>日期</th><th>上海(元/桶)</th><th>上海(美元)</th><th>布伦特</th><th>WTI</th><th>对布升贴水</th></tr>
       {% for r in ss.anchor_night_records|reverse %}
       <tr><td>{{ r.date }}</td><td>{{ r.sc_cny if r.sc_cny is not none else '—' }}</td>
       <td>{{ r.sc_usd if r.sc_usd is not none else '—' }}</td>
       <td>{{ r.brent if r.brent is not none else '—' }}</td>
       <td>{{ r.wti if r.wti is not none else '—' }}</td>
       <td class="{{ 'up' if (r.premium_usd or 0)>=0 else 'down' }}">{{ '%+.2f'|format(r.premium_usd) if r.premium_usd is not none else '—' }}</td></tr>
       {% endfor %}
     </table>
   </div>
   <div style="overflow-x:auto">
     <h3>北京 15:00 日盘收盘·同时刻截面（对照，近 8 交易日）</h3>
     <table>
       <tr><th>日期</th><th>上海(元/桶)</th><th>布伦特</th><th>WTI</th></tr>
       {% for r in ss.anchor_day_records|reverse %}
       <tr><td>{{ r.date }}</td><td>{{ r.sc_cny if r.sc_cny is not none else '—' }}</td>
       <td>{{ r.brent if r.brent is not none else '—' }}</td>
       <td>{{ r.wti if r.wti is not none else '—' }}</td></tr>
       {% endfor %}
     </table>
   </div>
  </div>
  <div class="mut" style="margin-top:8px;font-size:12px">分时K线根数（每日增量积累）：
   {% for s, n in ss.intraday_counts.items() %}{{ report.names[s] if report.names.get(s) else s }} {{ n }} 根｜{% endfor %}</div>
 </div>
 {% endif %}

 <h2 id="sec-endpoints">分品种预测端点一览（均值 / 95%区间 / 看涨概率）</h2>
 <div class="card" style="overflow-x:auto">
 <table>
   <tr><th>品种</th><th>单位</th><th>观测日</th>
       <th>短期·两周均值</th><th>短期95%区间</th>
       <th>中期·三个月均值</th><th>中期95%区间</th>
       <th>长期·十二月均值</th><th>长期95%区间</th></tr>
   {% for t in targets %}
   <tr>
     <td>{{ report.names[t] }}</td><td>{{ report.units[t] }}</td>
     <td>{{ report.current_meta[t].observed_date or '—' }}</td>
     {% for hz in ['short','mid','long'] %}
       {% set it = report.forecasts[hz][t] %}
       {% if it.status=='ok' and it.endpoint %}
         <td>{{ it.endpoint.mean }}</td>
         <td>{{ it.endpoint.q05 }} ~ {{ it.endpoint.q95 }}</td>
       {% else %}
         <td colspan="2" class="mut">不可用</td>
       {% endif %}
     {% endfor %}
   </tr>
   {% endfor %}
 </table>
 </div>

 <h2 id="sec-weights">模型解释：因素权重与分品种敏感度差异</h2>
 <div class="grid two">
   <div class="card" id="fig_weights"></div>
   <div class="card" id="fig_scenario"></div>
 </div>
 <div class="card" id="fig_sensitivity" style="margin-top:16px"></div>
 {% set cs = report.conditional_sensitivity or {} %}
 {% if cs %}
 <div class="card" style="margin-top:16px">
   <h3 style="margin:0 0 6px">重大事件期条件敏感度（<b>这才是当前霍尔木兹/伊朗冲突下各品种的真实敏感度</b>，上方热力图是 500 交易日平均、会把冲突期稀释）</h3>
   <div style="color:#666;font-size:12px;margin-bottom:8px">主指标（不依赖任何价差代理、最稳健）：<b>事件期累计涨跌</b>与<b>相对布伦特异超额</b>、波动放大倍数——直接回答"冲突中谁涨得更多"。参考指标：日收益对外生地缘溢价（布伦特-WTI 价差 5 日变化，不含上海 SC 自身价格）的传导 β 及其事件期抬升，β 对代理设定敏感、仅作参考。上海 INE SC 可交割中东含硫油、中国是伊朗原油主要买家，在霍尔木兹/油轮受扰的对华供油窗口，累计涨幅与相对布伦特异超额在原油品种中往往居前。样本不足留空，不估算。</div>
   <div id="fig_cond"></div>
   <table style="width:100%;border-collapse:collapse;font-size:13px">
     <thead><tr style="background:#f5f7fa;text-align:right">
       <th style="text-align:left;padding:6px">品种</th><th style="padding:6px">是否事件期</th>
       <th style="padding:6px">波动放大(倍)</th><th style="padding:6px">事件期地缘β</th>
       <th style="padding:6px">非事件期β</th><th style="padding:6px">β抬升</th>
       <th style="padding:6px">事件期累计涨跌</th><th style="padding:6px">相对布伦特异超额</th><th style="padding:6px">事件交易日占比</th>
     </tr></thead>
     <tbody>
     {% for t, d in cs.items() if d.available %}
       <tr style="border-top:1px solid #eee">
         <td style="padding:6px;text-align:left">{{ report.names[t] }}</td>
         <td style="padding:6px;text-align:center">{% if d.in_event_now %}是{% else %}否{% endif %}</td>
         <td style="padding:6px;text-align:right">{{ d.vol_ratio if d.vol_ratio is not none else '—' }}</td>
         <td style="padding:6px;text-align:right">{{ d.geo_beta_event if d.geo_beta_event is not none else '—' }}</td>
         <td style="padding:6px;text-align:right">{{ d.geo_beta_normal if d.geo_beta_normal is not none else '—' }}</td>
         <td style="padding:6px;text-align:right">{{ d.beta_lift if d.beta_lift is not none else '—' }}</td>
         <td style="padding:6px;text-align:right">{{ (d.event_cumret_pct|string + '%') if d.event_cumret_pct is not none else '—' }}</td>
         <td style="padding:6px;text-align:right">{{ (d.event_excess_vs_brent_pct|string + '%') if d.event_excess_vs_brent_pct is not none else '—' }}</td>
         <td style="padding:6px;text-align:right">{{ d.event_share_pct }}%</td>
       </tr>
     {% endfor %}
     </tbody>
   </table>
 </div>
 {% endif %}
 {% if report.narratives.conditional_sensitivity %}<div class="narr">{{ report.narratives.conditional_sensitivity }}</div>{% endif %}
 {% if report.holiday_impacts %}
 <div class="card" style="margin-top:16px">
   <h3 style="margin:0 0 6px">主要用油地区长假因子（中国春节/国庆、暑假驾驶季、年末）</h3>
   <div style="color:#666;font-size:12px;margin-bottom:8px">长假为需求侧可预期事件：春节节前备货与出行、节中工业物流停工（柴油/工业用油降）；暑假与驾驶旺季带动出行、汽油与航煤。下图为各品种进入长假窗口持有 5 日的历史平均收益（按各品种真实交易日统计、非交易日不参与），仅作历史描述，样本有限、不构成方向预测；长假窗口已作为特征入模，权重随滚动重训动态调整。</div>
   <div id="fig_holiday"></div>
 </div>
 {% endif %}
 <div class="card" id="fig_rs" style="margin-top:16px"></div>
 <div class="narr">{{ report.narratives.weights }}</div>
 {% if report.narratives.sensitivity %}<div class="narr">{{ report.narratives.sensitivity }}</div>{% endif %}
 <div class="narr">{{ report.narratives.long['wti'] }}</div>

 {% set ml = report.model_learning %}
 <h2 id="sec-learning">模型学习、回测与预测复盘（每日用最新真实数据重训并复测）</h2>
 {% if report.narratives.learning %}<div class="narr">{{ report.narratives.learning }}</div>{% endif %}
 <div class="card">
   <div class="kv"><span>本期重训时刻（北京时间）</span><b>{{ ml.retrained_at }}</b></div>
   <div class="kv"><span>短期模型滚动训练窗 / 累计运行期数</span><b>最近 {{ ml.train_window }} 个交易日（每日滑动重拟合）｜ 第 {{ ml.n_runs }} 期</b></div>
   {% for t, tm in ml.train_meta.items() %}
   <div class="kv"><span>{{ report.names[t] }}短期模型本期拟合窗口</span>
     <b>{{ tm.direct_window_start or tm.train_start }} ~ {{ tm.direct_window_end or tm.train_end }}（滚动窗 {{ tm.train_window }} 交易日）；库内可用真实样本 {{ tm.n_valid }} 行，逐日模型 {{ tm.n_steps }} 个</b></div>
   <div class="kv"><span>{{ report.names[t] }}零信息占位列（有效样本不足，常数0占位、不入模，schema仍保留）</span>
     <b class="mut">{% if tm.zero_info_cols %}{{ tm.zero_info_cols | join('、') }}{% else %}无{% endif %}</b></div>
   {% endfor %}
   {% if ml.persistence and ml.persistence.enabled %}
   {% for t, e in ml.persistence.models.items() %}
   <div class="kv"><span>{{ report.names[t] }}模型版本 / 学习方式</span>
     <b>v{{ e.version }}｜500 交易日滚动窗每日整体重拟合以适配最新市场 regime；跨市场特征按各品种真实收盘时刻做 as-of 时点对齐（上海 INE 约 UTC07 收盘、WTI/Brent 等约 UTC21，上海在 t 日只取已落定的 t-1 海外行情、绝不使用其收盘后才产生的价格，消除时区错配与隐性未来函数），并在分时覆盖窗口把上海的自身收盘价与跨市场价统一对齐到北京 02:30 夜盘收盘这一真实时刻（上海连续交易的真正收盘＝统计主节点，周五夜盘物理在周六凌晨、归属周五；见"同一真实时刻"面板，15:00 日盘仅作对照，更早日期维持 as-of，两种口径标注不混用）；已纳入可审计重大事件 regime（新冠疫情、俄乌战争、红海危机、2026 美以伊战事，年表 data/reference/major_events.yaml，带来源、权重随滚动窗动态调整）与样本外 β 校准（无方向信号时自动退守随机游走）；误差结构（残差分位/波动/β）{% if e.warm_started %}继承自 v{{ e.parent_version }}（上期截止 {{ e.parent_train_end or '—' }}）、跨期 EMA 累积{% else %}本期初始化、下期起跨期继承{% endif %}；原始数据库持续累积、工件可导入导出，重启不丢学习资产</b></div>
   {% endfor %}
   {% endif %}
 </div>
 <div class="grid cards" style="margin-top:14px">
   {% set bt = ml.backtest %}
   <div class="card"><h3>滚动样本外回测（短期·{{ bt.horizon_td if bt.available else '—' }}交易日）</h3>
     {% if bt.available %}
     {% if bt.cached %}<div class="kv"><span>结果时效</span><b>复用 {{ bt.cached_as_of }} 回测（7天内到期重算）</b></div>{% endif %}
     <div class="kv"><span>回测窗口</span><b>{{ bt.window_start }} ~ {{ bt.window_end }}</b></div>
     <div class="kv"><span>样本外原点数</span><b>{{ bt.n_origins }}</b></div>
     <div class="kv"><span>平均绝对误差 MAE（β校准后）</span><b>{{ bt.mae_pct }}%</b></div>
     <div class="kv"><span>均方根误差 RMSE</span><b>{{ bt.rmse_pct }}%</b></div>
     <div class="kv"><span>方向门控（样本外显著性）</span><b>{{ '通过（存在方向edge）' if bt.gate_has_edge else '未通过（以中性为主）' }}</b></div>
     <div class="kv"><span>明确表态占比（看涨/看跌）</span><b>{{ (bt.stance_engagement_rate*100)|round(0) }}%</b></div>
     <div class="kv"><span>中性占比（不计为错误）</span><b>{{ (bt.stance_neutral_rate*100)|round(0) }}%</b></div>
     <div class="kv"><span>表态时方向命中（可比50%）</span><b>{{ (bt.stance_engaged_accuracy*100)|round(0) if bt.stance_engaged_accuracy is not none else '—' }}%{% if bt.stance_engaged_p is not none %}（p={{ bt.stance_engaged_p }}）{% endif %}</b></div>
     <div class="kv"><span>方向概率 Brier（越低越好）</span><b>{{ bt.direction_brier }}</b></div>
     <div class="kv"><span>方向edge来源</span><b>{{ bt.gate_edge_basis if bt.gate_edge_basis else '—' }}</b></div>
     <div class="kv"><span>表态平均收益（按立场持有）</span><b>{{ '%+.2f'|format(bt.stance_engaged_mean_ret_pct) if bt.stance_engaged_mean_ret_pct is not none else '—' }}%</b></div>
     <div class="kv"><span>表态盈亏比（平均盈利/亏损）</span><b>{{ bt.stance_engaged_payoff if bt.stance_engaged_payoff is not none else '—' }}</b></div>
     <div class="kv"><span>表态近似年化夏普</span><b>{{ bt.stance_engaged_sharpe if bt.stance_engaged_sharpe is not none else '—' }}（无条件持有均收 {{ bt.buyhold_mean_ret_pct }}%）</b></div>
     <div class="kv"><span>收益相关系数 IC</span><b>{{ bt.ic }}</b></div>
     <div class="kv"><span>随机游走基准 MAE</span><b>{{ bt.benchmark_mae_pct }}%</b></div>
     <div class="kv"><span>未校准原始模型 MAE</span><b>{{ bt.raw_mae_pct }}%（方向{{ (bt.raw_direction_accuracy*100)|round(0) }}%）</b></div>
     <div class="kv"><span>β 校准带来的误差改善</span><b>{{ (bt.raw_mae_pct - bt.mae_pct)|round(2) }} 个百分点</b></div>
     <div class="kv"><span>相对基准误差改善</span><b class="{{ 'up' if bt.mae_pct < bt.benchmark_mae_pct else 'down' }}">{{ bt.mae_improve_pct }}%</b></div>
     <div class="kv"><span>是否跑赢随机游走</span><b class="{{ 'up' if bt.mae_pct < bt.benchmark_mae_pct else 'down' }}">{{ '是' if bt.mae_pct < bt.benchmark_mae_pct else '否（已退守基准附近）' }}</b></div>
     <div class="kv"><span>当前样本外校准系数 β（{{ bt.horizon_td }}日）</span><b>{{ ml.calib_beta_h if ml.calib_beta_h is not none else '—' }}</b></div>
     <p class="mut" style="margin-top:6px">方向采用看涨/看跌/中性三分类：独立的涨跌概率分类器经多年、非重叠样本外原点做 isotonic 校准，只有高置信表态命中率显著高于 50%（二项检验达标、门控通过）才明确看涨/看跌，否则诚实判中性、中性不计为错误。幅度上 β 由训练窗内部严格样本外检验估计、截断[0,1]，无预测力时 β→0、点预测退守随机游走。全球定价的原油日频方向信噪比天然很低、长期围绕 50%，系统只在样本外显著处表态，不通过过拟合/未来泄漏制造虚高命中率。</p>
     {% else %}<p class="mut">{{ bt.reason }}</p>{% endif %}
   </div>
   <div class="card"><h3>历史预测 vs 已实现真实价（复测）</h3>
     {% set rv = ml.review %}
     {% if rv.available %}
     {% set sm = rv.summary %}
     <div class="kv"><span>已到期可复盘预测</span><b>{{ sm.n }} 条</b></div>
     <div class="kv"><span>平均绝对误差</span><b>{{ sm.mae_pct }}%</b></div>
     <div class="kv"><span>方向命中</span><b>{{ sm.dir_acc if sm.dir_acc is not none else '—' }}%</b></div>
     <div class="kv"><span>95%区间覆盖真实价</span><b>{{ sm.coverage95 if sm.coverage95 is not none else '—' }}%</b></div>
     <p class="mut" style="margin-top:6px">目标日尚无真实价的预测不计入，绝不用填充值充当实际。</p>
     {% else %}<p class="mut">{{ rv.reason }}</p>{% endif %}
   </div>
   <div class="card"><h3>权重自适应更新（对比{{ ml.prev_weight_date or '上一期' }}）</h3>
     {% if ml.weight_delta %}
     {% for w in ml.weight_delta %}
     <div class="kv"><span>{{ w.factor }}</span>
       <b>{{ w.now }}%{% if w.delta_pp is not none %} <span class="{{ 'down' if w.delta_pp < 0 else 'up' }}">{{ '↑' if w.delta_pp > 0 else ('↓' if w.delta_pp < 0 else '—') }}{{ w.delta_pp|abs }}pp</span>{% elif w.prev is none %} <span class="mut">新增</span>{% endif %}</b></div>
     {% endfor %}
     {% else %}<p class="mut">首次运行或无上期权重，下一期起显示逐期变化。</p>{% endif %}
   </div>
 </div>
 {% if ml.review.available and ml.review.detail %}
 <div class="card" style="margin-top:14px;overflow-x:auto">
   <h3>最近预测复盘明细（预测值与后续真实收盘价对照）</h3>
   <table>
     <tr><th>发布日</th><th>目标日</th><th>周期</th><th>标的</th><th>发布时价</th><th>预测</th><th>实际</th><th>预测涨跌</th><th>实际涨跌</th><th>误差</th><th>方向</th><th>落95%区间</th></tr>
     {% for d in ml.review.detail %}
     <tr>
       <td>{{ d.report_date }}</td><td>{{ d.target_date }}</td><td>{{ d.horizon }}</td>
       <td>{{ report.names[d.instrument] }}</td><td>{{ d.base }}</td><td>{{ d.pred }}</td><td>{{ d.actual }}</td>
       <td class="{{ 'up' if d.pred_ret_pct > 0 else 'down' }}">{{ d.pred_ret_pct }}%</td>
       <td class="{{ 'up' if d.actual_ret_pct > 0 else 'down' }}">{{ d.actual_ret_pct }}%</td>
       <td>{{ d.abs_err_pct }}%</td>
       <td>{% if d.dir_hit is none %}—{% elif d.dir_hit %}<span class="down">命中</span>{% else %}<span class="up">偏离</span>{% endif %}</td>
       <td>{% if d.covered is none %}—{% elif d.covered %}是{% else %}否{% endif %}</td>
     </tr>
     {% endfor %}
   </table>
 </div>
 {% endif %}

 {% set ll = report.learning_loop %}
 {% if ll and ll.available %}
 <h2 id="sec-multiscale">多时间尺度概率预测与自我进化闭环（WTI/Brent：5 / 10 / 21 交易日）</h2>
 <div class="card" style="overflow-x:auto">
   <table>
     <tr><th>品种</th><th>尺度</th><th>目标日</th><th>立场</th><th>看涨概率</th><th>预测均值</th><th>95% 区间</th><th>突发概率</th><th>较现价</th></tr>
     {% for inst in ['wti','brent'] %}
     {% set item = ll.multi[inst] %}
     {% if item and item.status == 'ok' %}
     {% for h in [5,10,21] %}
     {% set c = item.cards[h] %}
     <tr><td>{{ report.names[inst] }}</td><td>{{ h }} 交易日</td><td>{{ c.target_date }}</td>
       <td>{{ c.stance }}</td><td>{{ c.prob_up }}</td><td>{{ c.mean }}</td>
       <td>{{ c.q05 }} ~ {{ c.q95 }}</td><td>{{ c.hazard }}</td><td>{{ c.pct_mean }}%</td></tr>
     {% endfor %}
     {% endif %}
     {% endfor %}
   </table>
   <p class="mut" style="margin-top:8px">区间构造＝EWMA 条件波动率（对冲击快速反应）× 日度标准化收益分位×√h（尺度严格单调、吸收厚尾）＋ 事件跳跃保险。
   {% for inst in ['wti','brent'] %}{% set item = ll.multi[inst] %}{% if item and item.status == 'ok' %}{{ {'wti':'WTI','brent':'布伦特'}[inst] }}：事件起始频率 λ={{ item.interval_diag.lam }}/年、爆发窗幅度 Jq={{ item.interval_diag.Jq }}；{% endif %}{% endfor %}突发概率＝未来持有期发生重大事件的概率 P=1-exp(-λh)。该区间在 300 个滚动原点上实测 95% 覆盖 90~93%。</p>
 </div>
 <div class="grid cards" style="margin-top:14px">
   <div class="card"><h3>本轮根因反思（真机复测统计量）</h3>
     {% for f in ll.findings %}
     <div class="kv"><span>{{ {'wti':'WTI','brent':'布伦特'}[f.inst] }} {{ f.h }}日 · {{ f.dimension }}</span>
       <b class="{{ 'up' if f.kind == 'success' else ('down' if f.kind == 'failure' else '') }}">{{ f.verdict }}（{{ f.observed }}）</b></div>
     {% endfor %}
   </div>
   <div class="card"><h3>策略进化动作（状态 v{{ ll.state_version }}）</h3>
     {% for a in ll.actions %}<div class="kv"><span>·</span><b>{{ a }}</b></div>{% endfor %}
     <p class="mut" style="margin-top:6px">证据权重 / 校准参数 / 假设库已写回 strategy/strategy_state.json；逐轮审计见 strategy/learning_journal.jsonl；人读 strategy/STRATEGY.md 自动重渲染。</p>
   </div>
 </div>
 {% endif %}

 <h2 id="sec-events">近期关键事件与量化影响（真实新闻，规则打分）</h2>
 <div class="card">
 {% if report.events %}
 <table>
   <tr><th>日期</th><th>事件</th><th>来源</th><th>因素主题</th><th>强度</th><th>估算影响(美元/桶)</th></tr>
   {% for e in report.events[:20] %}
   <tr><td>{{ e.date }}</td><td>{% if e.url %}<a href="{{e.url}}" target="_blank" rel="noopener">{{ e.title }}</a>{% else %}{{ e.title }}{% endif %}</td>
       <td>{{ e.source }}</td><td>{{ FACTOR_CN.get(e.theme, e.theme) }}</td>
       <td>{{ '%.0f'|format(e.intensity*100) }}%</td>
       <td class="{{ 'up' if e.est_price_impact>0 else 'down' }}">{{ '%+.2f'|format(e.est_price_impact) }}</td></tr>
   {% endfor %}
 </table>
 {% else %}<div class="unavail">本期事件源不可达或无有效真实条目，按数据原则不列举任何模拟事件。</div>{% endif %}
 </div>
 {% for t in report.narratives.events %}<div class="narr">{{ t }}</div>{% endfor %}

 <h2 id="sec-shock">事件冲击条件展望（爆发/升级后短窗，条件后验）</h2>
 <div class="card">
 {% set es = report.event_shock %}
 {% if es and es.available %}
   <div class="mut" style="margin-bottom:8px">
     统计口径：只在重大冲击【爆发/升级时点】后的 {{ es.H }} 个交易日短窗内给条件方向；
     方向与幅度由历次同类冲击的实际短窗反应、按约 3 年半衰期衰减在线更新。
     <b>属情景/风险提示，不改变方向立场门控、不是交易指令。</b>
   </div>
   {% if es.alerts %}
   <table>
     <tr><th>冲击类型</th><th>爆发日</th><th>已过交易日</th><th>品种</th>
         <th>条件上涨概率</th><th>历史平均短窗幅度</th><th>条件立场</th><th>证据冲击数</th></tr>
     {% for a in es.alerts %}
     <tr><td>{{ a.group_cn }}</td><td>{{ a.onset }}</td><td>{{ a.days_since }}</td>
         <td>{{ report.names[a.instrument] }}</td>
         <td class="{{ 'up' if a.p_up>=es.thr else ('down' if a.p_up<=1-es.thr else '') }}">{{ '%.0f'|format(a.p_up*100) }}%</td>
         <td class="{{ 'up' if a.exp_move_pct>0 else 'down' }}">{{ '%+.2f'|format(a.exp_move_pct) }}%</td>
         <td>{{ a.stance }}</td><td>{{ a.n_evidence }}</td></tr>
     {% endfor %}
   </table>
   {% else %}
   <div class="unavail">当前不在任何重大冲击的爆发后 {{ es.H }} 日短窗内（最近升级已过短窗或市场平静），故无短窗条件提示；这不代表无风险，仅代表"爆发期风险溢价"窗口已过。</div>
   {% endif %}

   <h3 style="margin:14px 0 6px">影子 OOS 证据（预注册、未并入方向立场）</h3>
   <div class="mut" style="margin-bottom:6px">对历次冲击用其爆发前的证据模拟表态、与实际 {{ es.H }} 日反应比对（walk-forward 无泄漏）。
   升级为方向立场的门槛：独立表态 ≥ ~25、二项 p &lt; 0.05、扣成本仍正期望；达标前仅作研究记录。</div>
   <table>
     <tr><th>冲击类型</th><th>表态次数</th><th>表态命中</th><th>表态平均幅度(毛)</th><th>二项 p</th><th>是否达升级门槛</th></tr>
     {% for g, s in es.shadow.items() %}
     <tr><td>{{ {'geopolitical':'地缘冲突/制裁/袭船','china_supply':'对华供油链路（霍尔木兹/伊朗）','demand_shock':'需求冲击（疫情等）','supply_policy':'OPEC+ 产量政策'}.get(g,g) }}</td>
         <td>{{ s.engaged_n }}</td>
         <td>{{ (('%.0f'|format(s.hit*100))+'%') if s.hit is not none else '—' }}</td>
         <td>{{ (('%+.2f'|format(s.gross_move_pct))+'%') if s.gross_move_pct is not none else '—' }}</td>
         <td>{{ s.binom_p if s.binom_p is not none else '—' }}</td>
         <td>{% if s.engaged_n>=25 and s.binom_p is not none and s.binom_p<0.05 and s.gross_move_pct>0.1 %}达标待评审{% else %}未达标（样本/显著性不足）{% endif %}</td></tr>
     {% endfor %}
   </table>
 {% else %}
   <div class="unavail">事件冲击条件效应模块本期不可用{% if es and es.reason %}：{{ es.reason }}{% endif %}。</div>
 {% endif %}
 </div>

 <h2 id="sec-views">机构观点（真实抽取）</h2>
 <div class="card">
 {% if report.views %}
 <table>
   <tr><th>日期</th><th>机构</th><th>品种</th><th>目标价(美元/桶)</th><th>方向</th><th>摘要</th></tr>
   {% for v in report.views[:12] %}
   <tr><td>{{ v.date }}</td><td>{{ v.institution }}</td>
       <td>{{ {'wti':'WTI','brent':'布伦特'}.get(v.instrument, '—') }}</td>
       <td>{{ v.target_wti if v.target_wti else '—' }}</td>
       <td>{{ v.stance }}</td><td>{{ v.note }}</td></tr>
   {% endfor %}
 </table>
 {% else %}<div class="unavail">本期各机构观点源未抽取到可核验的评级/目标价（{{ report.lineage['institutional_view'].note if report.lineage.get('institutional_view') else '各源均无有效条目' }}），保持空缺、不生成模拟观点。</div>{% endif %}
 </div>

 <h2 id="sec-narr">走势与预测文字解读</h2>
 {% for t in targets %}
   <div class="narr"><b>{{ report.names[t] }}｜历史：</b>{{ report.narratives.trends[t] }}</div>
 {% endfor %}
 {% for t in targets %}
   <div class="narr"><b>{{ report.names[t] }}·两周：</b>{{ report.narratives.short[t] }}</div>
   <div class="narr"><b>{{ report.names[t] }}·三个月：</b>{{ report.narratives.mid[t] }}</div>
 {% endfor %}
 {% if report.narratives.backtest %}<div class="narr">{{ report.narratives.backtest }}</div>{% endif %}

 <h2 id="sec-disclaimer">数据原则与免责声明</h2>
 <div class="card">
   <p class="mut">{{ report.narratives.sources }}</p>
   <p class="mut"><b>数据原则：</b>模型只建立在真实、可追溯、带观测日期的数据之上；
   缺失、过期或无法验证的数据一律保持缺失并在谱系表标明状态，绝不用插值、外推或合成值"补齐"。
   月频指标在两次发布之间沿用最近一次真实发布值，并在底层数据中保留其原始发布日期（vintage）。</p>
   <p class="mut">模型包括 Direct 多步梯度提升（短期）、VAR 向量自回归（中期）与情景蒙特卡洛（长期），
   权重由随机森林与 LASSO 融合、向人工先验收缩并跨日 EMA 自适应。预测区间反映历史波动与模型不确定性，
   不构成任何投资建议；第三方数据版权归原方所有。</p>
 </div>
 <footer>OilCast v1.1 · 每日 UTC 01:00（北京 09:00）由 GitHub Actions 自动更新</footer>
</div>
<script>
const FIGS = {{ figs_json | safe }};
for (const [id, fig] of Object.entries(FIGS)) {
  if (fig) Plotly.newPlot(id, fig.data, fig.layout, {responsive:true, displaylogo:false});
}
</script>
</body></html>
"""
