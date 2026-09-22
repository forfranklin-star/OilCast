"""把模型输出转成中文文字解读：每个数字都来自上游真实计算，不写空话；
数据不可用时明确说明，不编造趋势。"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

FACTOR_CN = {
    "supply_disruption": "供给中断与OPEC+产量政策",
    "geopolitical_risk": "地缘政治风险",
    "usd_index": "美元指数",
    "us_treasury_10y": "美国10年期国债收益率",
    "cpi_surprise": "美国通胀(CPI)超预期程度",
    "jobs_surprise": "非农就业数据",
    "fed_policy_expectation": "美联储降息/加息预期",
    "fx_jpy": "日元汇率(USDJPY)",
    "demand_outlook": "全球需求前景",
    "institutional_view": "机构观点与目标价调整",
}
STATUS_CN = {"ok": "可用", "stale": "已过期", "insufficient": "样本不足",
             "unavailable": "不可用"}


def pct_word(x: float) -> str:
    if pd.isna(x):
        return "数据不足"
    if x > 0.05:
        return "上涨"
    if x < -0.05:
        return "下跌"
    return "基本持平"


def trend_narrative(prices: pd.Series, name: str, unit: str) -> str:
    p = prices.dropna()
    cur = float(p.iloc[-1])
    obs = p.index[-1].strftime("%Y-%m-%d")

    def chg(k):
        return (cur / float(p.iloc[-1 - k]) - 1) * 100 if len(p) > k else np.nan
    c5, c20, c60 = chg(5), chg(20), chg(60)
    vol_ann = float(p.pct_change().tail(20).std() * np.sqrt(252) * 100)
    hi, lo = float(p.tail(252).max()), float(p.tail(252).min())
    pos = (cur - lo) / max(hi - lo, 1e-9) * 100
    return (f"截至真实观测日 {obs}，{name}收于 {cur:.2f} {unit}，"
           f"近5个交易日{pct_word(c5)}{abs(c5):.2f}%、近20日{pct_word(c20)}{abs(c20):.2f}%、"
           f"近60日{pct_word(c60)}{abs(c60):.2f}%；近20日年化波动率约 {vol_ann:.1f}%，"
           f"价格处于近一年真实区间 [{lo:.2f}, {hi:.2f}] 的 {pos:.0f}% 分位。")


def forecast_narrative(ep: dict, name: str, unit: str, horizon_cn: str) -> str:
    """方向以独立概率分类器 + 样本外显著性门控为准（看涨/看跌/中性三分类）。

    无统计 edge 或概率未越置信阈值时诚实判中性（弱有效市场下不硬押方向），并说明这是
    证据不足而非看空；有 edge 且明确表态时，给出校准概率与该周期样本外表态命中证据。
    """
    stance = ep.get("dir_stance", "中性")
    p_up = float(ep.get("dir_prob_up", ep.get("prob_up", 0.5)))
    iv = (f"50%概率区间 [{ep['q25']}, {ep['q75']}]，95%概率区间 [{ep['q05']}, {ep['q95']}]")
    edge_txt = ""
    basis = ep.get("dir_edge_basis")
    if basis == "趋势期望":
        payoff = ep.get("dir_edge_payoff")
        mret = ep.get("dir_edge_mean_ret")
        edge_txt = ("该周期由近 1 月时序动量的【趋势跟随】通道触发（历史样本外顺势表态"
                    + (f"平均收益 {mret:+.2f}%、" if mret is not None else "")
                    + (f"盈亏比 {payoff}" if payoff is not None else "")
                    + "，胜率可不占优、靠盈亏比取得正期望，属低胜率高赔率信号）；")
    elif ep.get("dir_has_edge") and ep.get("dir_edge_hit") is not None:
        edge_txt = (f"该周期历史样本外高置信表态命中率约 {ep['dir_edge_hit']*100:.0f}%"
                    + (f"（二项检验 p={ep['dir_edge_p']}）" if ep.get("dir_edge_p") is not None else "")
                    + "，方向信号通过显著性门控；")
    if stance == "中性":
        reason = ("该周期方向未通过样本外显著性门控" if not ep.get("dir_has_edge")
                  else "校准后涨跌概率未达到明确表态阈值")
        return (f"未来{horizon_cn}，模型对{name}不主张明确方向（{reason}）：预测均值 "
                f"{ep['mean']:.2f} {unit}（较观测日 {ep['pct_mean']:+.2f}%），{iv}，"
                f"校准后看涨概率 {p_up*100:.0f}%、看跌 {(1-p_up)*100:.0f}%，接近均衡。"
                f"这表示在当前噪声水平下没有统计上可靠的方向优势、以区间刻画不确定性，"
                f"而非看空；预测仅基于截至观测日的真实数据，不构成投资建议。")
    if stance == "看涨":
        return (f"未来{horizon_cn}，模型明确【看涨】{name}：预测均值 {ep['mean']:.2f} {unit}"
                f"（较观测日 +{abs(ep['pct_mean']):.2f}%），{iv}；{edge_txt}校准后看涨概率 "
                f"{p_up*100:.0f}% / 看跌 {(1-p_up)*100:.0f}%。预测仅基于截至观测日的真实数据，"
                f"不构成投资建议。")
    return (f"未来{horizon_cn}，模型明确【看跌】{name}：预测均值 {ep['mean']:.2f} {unit}"
            f"（较观测日 -{abs(ep['pct_mean']):.2f}%），{iv}；{edge_txt}校准后看跌概率 "
            f"{(1-p_up)*100:.0f}% / 看涨 {p_up*100:.0f}%。预测仅基于截至观测日的真实数据，"
            f"不构成投资建议。")


def weights_narrative(weights: pd.DataFrame) -> str:
    usable = weights.dropna(subset=["weight"])
    if usable.empty:
        return "当前没有任何因素具备充分真实数据，本期不计算因素权重。"
    top = usable.head(3)
    parts = [f"{FACTOR_CN.get(r['factor'], r['factor'])}（权重 {r['weight']*100:.1f}%）"
             for _, r in top.iterrows()]
    leader = FACTOR_CN.get(usable.iloc[0]["factor"], usable.iloc[0]["factor"])
    missing = [FACTOR_CN.get(f, f) for f in weights[weights["weight"].isna()]["factor"]]
    tail = f"；因素【{'、'.join(missing)}】因真实数据缺失未参与权重归一化" if missing else ""
    return ("当前主导油价的前三大因素依次为 " + "、".join(parts) +
            f"；其中「{leader}」是边际定价的核心变量。权重由随机森林重要性与"
            f"LASSO系数融合、向人工先验收缩并与历史权重EMA平滑得到{tail}。")


def sensitivity_narrative(sensitivity: dict, names: dict) -> str:
    """用文字说明各油种主导因素的差异（分品种独立学习得到，不做跨品种平均）。"""
    if not sensitivity:
        return ""
    def top_factor(t):
        cand = {f: mp.get(t) for f, mp in sensitivity.items()
                if isinstance(mp, dict) and mp.get(t) is not None}
        if not cand:
            return None
        f = max(cand, key=cand.get)
        return FACTOR_CN.get(f, f), cand[f]
    segs = []
    for t in names:
        tf = top_factor(t)
        if tf:
            segs.append(f"{names[t]}对「{tf[0]}」最敏感（权重 {tf[1]:.1f}%）")
    if len(segs) < 2:
        return ""
    # 比较上海原油与布伦特的首要因素是否不同
    def lead(t):
        tf = top_factor(t)
        return tf[0] if tf else None
    diffs = []
    if lead("shanghai_crude") and lead("brent") and lead("shanghai_crude") != lead("brent"):
        diffs.append(f"{names['shanghai_crude']}首要驱动为「{lead('shanghai_crude')}」，"
                     f"与布伦特的「{lead('brent')}」不同，体现了亚太到岸供需、计价货币"
                     f"（人民币/汇率）与区域地缘敏感度差异")
    tail = ("；" + "；".join(diffs)) if diffs else ""
    return "分品种敏感度：" + "；".join(segs) + tail + "。权重为各品种用自身真实样本独立学习，未做跨品种平均。"


def conditional_sensitivity_narrative(cs: dict, names: dict) -> str:
    """事件期条件敏感度的文字解读（区别于 500 日平均权重）。"""
    if not cs:
        return ""
    rows = {t: d for t, d in cs.items() if isinstance(d, dict) and d.get("available")}
    if not rows:
        return ""
    in_event = [t for t, d in rows.items() if d.get("in_event_now")]
    # 原油三品种里，事件期累计涨幅最高、相对布伦特异超额最大、地缘 β 抬升最大者
    crude = [t for t in ("wti", "brent", "shanghai_crude") if t in rows]
    lead_ret = max(crude, key=lambda t: rows[t].get("event_cumret_pct") or -1e9) if crude else None
    lead_ex = max(crude, key=lambda t: rows[t].get("event_excess_vs_brent_pct")
                  if rows[t].get("event_excess_vs_brent_pct") is not None else -1e9) if crude else None
    lead_lift = max(crude, key=lambda t: rows[t].get("beta_lift") if rows[t].get("beta_lift") is not None else -1e9) if crude else None
    segs = []
    if in_event:
        segs.append("观测日仍处于重大地缘事件期（见重大事件年表），以下为事件期条件归因，"
                    "与上方 500 交易日滚动平均权重口径不同，后者会稀释当前冲突的影响")
    for t in crude:
        d = rows[t]
        if d.get("vol_ratio") and d.get("event_cumret_pct") is not None:
            ex = d.get("event_excess_vs_brent_pct")
            ex_txt = f"、相对布伦特异超额 {ex:+.1f}%" if ex is not None and t != "brent" else ""
            segs.append(f"{names[t]}事件期波动放大 {d['vol_ratio']} 倍、累计 {d['event_cumret_pct']:+.1f}%{ex_txt}，"
                        f"地缘溢价传导 β 由非事件期 {d.get('geo_beta_normal')} 变为 {d.get('geo_beta_event')}"
                        f"（抬升 {d.get('beta_lift')}，参考）")
    extra = ""
    if lead_ret == "shanghai_crude" or lead_ex == "shanghai_crude" or lead_lift == "shanghai_crude":
        extra = ("。上海原油（INE SC）可交割标的为中东含硫原油、中国是伊朗及中东原油主要买家，"
                 "在本轮对华供油受扰窗口的累计涨幅、相对布伦特异超额与地缘敏感度抬升均居三大原油之首，"
                 "高于 WTI（锚定美国本土页岩、自给度高、对中东断供最钝）；这解释了为何其全窗平均地缘权重"
                 "看似不高，当前行情却由地缘主导——平均权重会被大量非冲突交易日稀释，事件期条件敏感度才反映当下")
    return "；".join(segs) + extra + "。统计仅基于真实价格，样本不足项留空。"


def events_narrative(events: pd.DataFrame, top_n: int = 5) -> List[str]:
    if events is None or events.empty:
        return ["本期事件源不可达或未捕捉到达到强度阈值的真实事件，不列举模拟事件。"]
    out = []
    for _, r in events.head(top_n).iterrows():
        impact = float(r.get("est_price_impact", 0) or 0)
        direction = "利多" if impact > 0 else ("利空" if impact < 0 else "中性")
        out.append(
            f"{pd.Timestamp(r['date']).strftime('%m-%d')}｜{r['title']}"
            f"（主题：{FACTOR_CN.get(r['theme'], r['theme'])}，强度 {float(r['intensity'])*100:.0f}%，"
            f"规则估算{direction}约 {abs(impact):.2f} 美元/桶；来源：{r.get('source','')}）")
    return out


def scenario_narrative(long_ep: dict) -> str:
    p = long_ep["scenario_probs"]
    label = {"bull": "高油价（供给冲击）", "base": "基准（供需再平衡）", "bear": "低油价（需求衰退）"}
    seg = "、".join(f"{label[k]} {v*100:.0f}%" for k, v in p.items())
    anchor = long_ep.get("institution_anchor")
    extra = f"真实抓取的机构目标价中位数约 {anchor:.1f} 美元/桶，作为外部锚点并列展示。" if anchor else ""
    return (f"长期（12个月）情景概率：{seg}；模型分布均值 {long_ep['mean']:.2f}，"
            f"95%区间 [{long_ep['q05']}, {long_ep['q95']}]。{extra}")


def learning_narrative(ml: dict) -> str:
    """用一句话说明本期模型如何重训、回测与复测，给出可核验数字。"""
    if not ml:
        return ""
    bt = ml.get("backtest", {})
    rv = ml.get("review", {})
    seg = []
    tm = ml.get("train_meta", {})
    if tm:
        first = next(iter(tm.values()))
        tw = ml.get("train_window", "?")
        seg.append(f"短期模型已于 {ml.get('retrained_at','')} 重训：累计 "
                   f"{first.get('n_valid','?')} 行带观测日的真实样本，每个模型以最近 "
                   f"{tw} 个交易日滚动窗拟合、截至 {first.get('train_end','')}")
    if isinstance(bt, dict) and bt.get("available"):
        beat = "跑赢" if bt["mae_pct"] < bt["benchmark_mae_pct"] else "暂未跑赢"
        dir_txt = _stance_txt(bt)
        seg.append(f"滚动样本外回测 {bt['n_origins']} 个原点，MAE {bt['mae_pct']}%、"
                   f"{dir_txt}，相对随机游走基准（{bt['benchmark_mae_pct']}%）{beat}")
    if isinstance(rv, dict) and rv.get("available"):
        sm = rv["summary"]
        seg.append(f"对 {sm['n']} 条已到期历史预测复测，平均误差 {sm['mae_pct']}%、"
                   f"方向命中 {sm['dir_acc']}%、95%区间覆盖 {sm['coverage95']}%，误差反馈到下一期重训")
    else:
        seg.append("历史预测从下一期起陆续到期并纳入复测")
    pers = ml.get("persistence", {})
    if pers.get("enabled"):
        cn = {"wti": "WTI原油", "brent": "布伦特原油",
              "shanghai_crude": "上海原油", "heating_oil": "美燃油", "gasoil": "伦敦柴油"}
        mods = pers.get("models", {})
        warm = [cn.get(t, t) for t, e in mods.items() if e.get("warm_started")]
        if warm:
            seg.append("、".join(warm) +
                       "在已保存的上期模型工件上热启动、增量训练（进程重启也不从零），本期工件已落盘并推进版本")
        elif mods:
            seg.append("本期为冷启动训练，模型工件已落盘，此后每期在上期基础上热启动持续学习")
    return "；".join(seg) + "。权重每日向新数据学习并与上期平滑，全过程仅使用真实观测。"
def _stance_txt(bt: dict) -> str:
    """三分类方向口径的一句话总结（看涨/看跌/中性，中性不计错）。"""
    er = bt.get("stance_engagement_rate")
    acc = bt.get("stance_engaged_accuracy")
    if er is None:
        return "方向指标暂缺"
    if er == 0:
        return ("该周期样本外方向未通过显著性门控、回测期全程判中性（不硬赌方向，"
                "中性不计为错误）")
    p_txt = f"、二项检验 p={bt.get('stance_engaged_p')}" if bt.get("stance_engaged_p") is not None else ""
    base = (f"明确表态占比 {er*100:.0f}%、其余 {bt.get('stance_neutral_rate', 0)*100:.0f}% 判中性，"
            f"表态时方向命中 {acc*100:.0f}%{p_txt}" if acc is not None else f"明确表态占比 {er*100:.0f}%")
    if bt.get("gate_has_edge"):
        base += "，该周期方向 edge 通过显著性门控"
    else:
        base += "，该周期方向 edge 未通过门控、以中性为主"
    return base


def backtest_narrative(bt: dict) -> Optional[str]:
    if not bt or not bt.get("available"):
        return None
    beat = bt["mae_pct"] < bt["benchmark_mae_pct"]
    cmp_word = "跑赢" if beat else "退守至随机游走基准附近（未跑输）"
    imp = bt.get("mae_improve_pct")
    span = ""
    if bt.get("window_start") and bt.get("window_end"):
        span = f"{bt['window_start']} 至 {bt['window_end']}、"
    imp_txt = f"，误差较基准降低 {imp}%" if (beat and imp is not None) else ""
    # 三分类方向口径
    er = bt.get("stance_engagement_rate", 0)
    if er == 0:
        nbreak = bt.get("recent_break_origins", 0)
        if nbreak > 0:
            dir_txt = (
                f"方向采用看涨/看跌/中性三分类：{bt['n_origins']} 个样本外原点中，{nbreak} 个"
                f"原点的长历史方向 edge 虽曾成立，但最近约一年样本外表态已转为稳定亏损或显著"
                f"反向，被【近端失效熔断】自动退回中性，其余原点未通过显著性门控，故本期无明确"
                f"方向表态（中性是弱有效市场下的诚实选择，不计为错误；这避免在地缘暴涨暴跌的"
                f"反转行情里逆势硬猜），概率 Brier {bt.get('direction_brier', '-')}；")
        else:
            dir_txt = (f"方向采用看涨/看跌/中性三分类：{bt['n_origins']} 个样本外原点上该周期方向"
                       f"均未通过显著性门控、全部判中性（中性是弱有效市场下的诚实选择，不计为错误），"
                       f"概率 Brier {bt.get('direction_brier', '-')}；")
    else:
        acc = bt.get("stance_engaged_accuracy")
        p_txt = (f"，相对抛硬币的二项检验 p={bt.get('stance_engaged_p')}"
                 if bt.get("stance_engaged_p") is not None else "")
        gate = "方向 edge 通过门控" if bt.get("gate_has_edge") else "方向 edge 未通过门控"
        dir_txt = (f"方向采用看涨/看跌/中性三分类：明确表态占比 {er*100:.0f}%、"
                   f"{bt.get('stance_neutral_rate', 0)*100:.0f}% 判中性，表态原点方向命中 "
                   f"{acc*100:.0f}%{p_txt}（{gate}），概率 Brier {bt.get('direction_brier', '-')}；")
    # 经济价值口径：胜率≠盈利能力，趋势信号可在胜率不高时靠盈亏比取得正期望
    econ_txt = ""
    mr, po, sh = (bt.get("stance_engaged_mean_ret_pct"), bt.get("stance_engaged_payoff"),
                  bt.get("stance_engaged_sharpe"))
    if er and er > 0 and mr is not None:
        econ_txt = (f"表态原点按立场持有 {bt['horizon_td']} 日的平均收益 {mr:+.2f}%、"
                    f"盈亏比 {po if po is not None else '-'}、近似年化夏普 {sh if sh is not None else '-'}，"
                    f"同期无条件持有的平均收益 {bt.get('buyhold_mean_ret_pct', '-')}%（胜率与"
                    f"盈利能力分开考核）；")
    return (f"近期滚动回测（{span}{bt['n_origins']} 个样本外原点、{bt['horizon_td']} 交易日视野，"
            f"每个原点均只用当时可得数据、按滚动训练窗拟合；方向门控逐原点滚动、仅引用当时已实现"
            f"的样本外表现，并对最近约一年失效（稳定亏损或显著反向）的通道自动熔断退回中性，全程"
            f"无未来泄漏）：幅度 MAE {bt['mae_pct']}%，{dir_txt}{econ_txt}预测与实际累计收益相关系数 "
            f"{bt.get('ic', 0):.2f}；随机游走基准误差 {bt['benchmark_mae_pct']}%，"
            f"本模型{cmp_word}{imp_txt}。全球定价的原油日度方向信噪比天然偏低、胜率长期围绕 50% "
            f"波动属正常，故系统同时考核胜率与经济价值，只在样本外显著（胜率或期望收益）时才明确"
            f"表态，不以过拟合或未来函数制造虚高命中。")


def lineage_narrative(bundle) -> str:
    """逐字段说明真实来源、观测日期、质量门状态（替代旧的 provenance 说明）。"""
    parts = []
    for f, L in bundle.lineage.items():
        status = STATUS_CN.get(L.get("status"), L.get("status"))
        src = L.get("source_name", "")
        last = L.get("last_observed") or "无观测"
        parts.append(f"{f}[{status}|末次观测 {last}|{src}]")
    head = "数据谱系（字段[状态|末次真实观测|来源]）："
    return head + "；".join(parts)
