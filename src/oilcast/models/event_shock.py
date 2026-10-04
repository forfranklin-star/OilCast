"""事件冲击条件效应：重大冲击"爆发/升级时点"后的短窗方向与幅度后验。

根因（真机数据实测）
--------------------
1. 对"事件持续窗口"整体取方向几乎无 edge：冲击爆发后价格很快 priced-in，长达数月的
   窗口内多在回落（地缘事件整窗上涨比例约 0.35-0.51、均不显著）。
2. 方向 edge 集中在冲击【爆发/升级时点】后的约 1 周（h=5）。严格 walk-forward OOS
   （无泄漏、扣约 10bp 双边成本）：地缘 / 对华供油冲击爆发后 h=5，各品种命中约
   0.67-0.83、净期望 +4%~+9%；但十年独立冲击仅十余次、二项 p 不显著。

定位与纪律
----------
- 本模块**不改动保守的方向立场门控**（样本不足不把方向 edge 当交易信号）。
- 只向报告"事件影响 / 情景"提供条件化、随证据在线更新的冲击提示（决策支持，定位 3）：
  出事时明确量化发声（条件方向概率、预期幅度、证据样本数与局限），平静期不提示。
- 同时以**影子模式**逐冲击记录前瞻 OOS（shadow_evaluation），持续累积；只有当独立
  表态样本 ≥ ~25、二项 p < 0.05、扣成本仍正期望，才讨论是否升级为方向立场。

持续学习 / 不丢资产
-------------------
每轮把已到期冲击的实际 h 日反应并入、旧冲击按 halflife 衰减；状态经 registry 持久化、
热启动并随 bundle 导入导出。只用真实价格，缺失/未到期一律不计入，绝不合成。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

INSTRUMENTS = ["wti", "brent", "shanghai_crude", "heating_oil", "gasoil"]

# 分组：键 → 判定(category, china_supply)
GROUPS: Dict[str, callable] = {
    "geopolitical": lambda cat, cs: cat == "geopolitical",
    "china_supply": lambda cat, cs: cs is True,
    "demand_shock": lambda cat, cs: cat == "demand_shock",
    "supply_policy": lambda cat, cs: cat == "supply_policy",
}
GROUP_CN = {
    "geopolitical": "地缘冲突 / 制裁 / 袭船",
    "china_supply": "对华供油链路（霍尔木兹 / 伊朗）",
    "demand_shock": "需求冲击（疫情等）",
    "supply_policy": "OPEC+ 产量政策",
}


class EventShockModel:
    """逐 (分组 × 品种) 累积已实现冲击、给条件后验与当前冲击提示。"""

    def __init__(self,
                 H: int = 5,
                 prior_n: float = 1.5,
                 halflife: int = 750,
                 thr: float = 0.62,
                 state: Optional[dict] = None) -> None:
        self.H = int(H)
        self.prior_n = float(prior_n)
        self.halflife = int(halflife)
        self.thr = float(thr)
        self.state: Dict[str, Dict[str, list]] = state or {
            g: {t: [] for t in INSTRUMENTS} for g in GROUPS}
        self.updated_at: Optional[str] = None

    # -------------------------------------------------- 冲击时点
    @staticmethod
    def onsets_from_events(events: list) -> List[tuple]:
        """年表事件 → 去重排序的冲击时点 (date, category, china_supply)；含 phases 升级日。"""
        out = set()
        for e in events or []:
            cat, cs = e.get("category"), e.get("china_supply") is True
            if e.get("start"):
                out.add((str(e["start"]), cat, cs))
            for ph in e.get("phases", []) or []:
                if ph.get("start"):
                    out.add((str(ph["start"]), cat, cs))
        return sorted(out, key=lambda x: x[0])

    def _pos_on_or_after(self, tdays: pd.DatetimeIndex, d0: str):
        later = tdays[tdays >= pd.Timestamp(d0)]
        return tdays.get_loc(later[0]) if len(later) else None

    @staticmethod
    def _log_price(prices: pd.DataFrame, t: str) -> pd.Series:
        s = pd.to_numeric(prices[t], errors="coerce")
        return np.log(s.where(s > 0))

    # -------------------------------------------------- 每轮更新（持续学习）
    def update(self, prices: pd.DataFrame, events: list) -> None:
        """计算每个冲击对各品种的 h 日反应；已到期（i0+H ≤ 最新交易日）即并入状态。

        已存在的冲击按 onset 去重、以最新真实反应覆盖；跨重启只增不丢（不删除任何
        已实现冲击）。未到期/价格缺失的不并入。
        """
        tdays = prices.index
        cur_max = len(tdays) - 1
        onsets = self.onsets_from_events(events)
        for g, gpred in GROUPS.items():
            sel = [o for o in onsets if gpred(o[1], o[2])]
            for t in INSTRUMENTS:
                if t not in prices.columns:
                    continue
                lp = self._log_price(prices, t)
                have = {r["onset"]: r for r in self.state[g][t]}
                for (d0, cat, cs) in sel:
                    i0 = self._pos_on_or_after(tdays, d0)
                    if i0 is None or i0 - 1 < 0 or i0 + self.H > cur_max:
                        continue
                    ret = float(lp.iloc[i0 + self.H] - lp.iloc[i0 - 1])
                    if not np.isfinite(ret):
                        continue
                    have[d0] = {"onset": d0, "i0": int(i0),
                                "dir": 1 if ret > 0 else -1,
                                "ret": ret, "realized": True}
                self.state[g][t] = sorted(have.values(), key=lambda r: r["i0"])
        self.updated_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    # -------------------------------------------------- 条件后验
    def _posterior(self, g: str, t: str, before_i0: int):
        """仅用 i0 < before_i0 的已实现冲击，时间衰减 → (P(涨), 衰减加权均值幅度, 样本数)。"""
        a = b = self.prior_n
        num = den = 0.0
        n = 0
        for r in self.state[g][t]:
            if not r.get("realized") or r["i0"] >= before_i0:
                continue
            w = 0.5 ** ((before_i0 - r["i0"]) / self.halflife)
            if r["dir"] > 0:
                a += w
            else:
                b += w
            num += w * r["ret"]; den += w; n += 1
        p_up = a / (a + b)
        mu = (num / den) if den > 0 else 0.0
        return p_up, mu, n

    def _stance(self, p_up: float):
        if p_up >= self.thr:
            return "看涨"
        if p_up <= 1 - self.thr:
            return "看跌"
        return "中性"

    # -------------------------------------------------- 当前冲击提示
    def current_alerts(self, prices: pd.DataFrame, events: list) -> List[dict]:
        """最新交易日处于某冲击爆发后 0..H 日窗口内 → 逐品种条件提示。"""
        tdays = prices.index
        cur = len(tdays) - 1
        alerts: List[dict] = []
        for (d0, cat, cs) in self.onsets_from_events(events):
            i0 = self._pos_on_or_after(tdays, d0)
            if i0 is None:
                continue
            age = cur - i0
            if not (0 <= age <= self.H):
                continue
            for g, gpred in GROUPS.items():
                if not gpred(cat, cs):
                    continue
                for t in INSTRUMENTS:
                    if t not in prices.columns:
                        continue
                    # 条件后验不含当前冲击自身（其反应可能尚未实现）
                    p_up, mu, n_ev = self._posterior(g, t, i0)
                    alerts.append({
                        "group": g, "group_cn": GROUP_CN[g],
                        "onset": d0, "days_since": int(age),
                        "instrument": t,
                        "p_up": round(float(p_up), 3),
                        "exp_move_pct": round(float(mu) * 100, 2),
                        "n_evidence": int(n_ev),
                        "stance": self._stance(p_up),
                    })
        return alerts

    # -------------------------------------------------- 影子 OOS
    def shadow_evaluation(self, prices: pd.DataFrame, events: list) -> dict:
        """对每个已到期冲击，用其爆发之前的证据模拟表态并与实际 h 日反应比对。

        返回逐分组汇总（表态数、命中、表态平均毛/净收益、二项 p）。这是前瞻 OOS 的
        累积证据，用于判断是否达到升级为方向立场的门槛，不参与日常方向表态。
        """
        from scipy.stats import binomtest
        tdays = prices.index
        cur_max = len(tdays) - 1
        summary: Dict[str, dict] = {}
        for g, gpred in GROUPS.items():
            sel = [o for o in self.onsets_from_events(events) if gpred(o[1], o[2])]
            eng = []   # (stance, ret)
            for t in [INSTRUMENTS[0]]:   # 逐品种在 report 层另算；这里以锚品种汇总
                lp = self._log_price(prices, t)
                for (d0, cat, cs) in sel:
                    i0 = self._pos_on_or_after(tdays, d0)
                    if i0 is None or i0 - 1 < 0 or i0 + self.H > cur_max:
                        continue
                    ret = float(lp.iloc[i0 + self.H] - lp.iloc[i0 - 1])
                    if not np.isfinite(ret):
                        continue
                    p_up, _, _ = self._posterior(g, t, i0)
                    stv = 1 if p_up >= self.thr else (-1 if p_up <= 1 - self.thr else 0)
                    if stv != 0:
                        eng.append((stv, ret))
            ne = len(eng)
            if ne:
                hit = float(np.mean([1 if (s > 0) == (r > 0) else 0 for s, r in eng]))
                gross = float(np.mean([s * r for s, r in eng]))
                nup = int(sum([1 for s, r in eng if (s > 0) == (r > 0)]))
                p_b = float(binomtest(nup, ne, 0.5).pvalue)
            else:
                hit = gross = p_b = None
            summary[g] = {"engaged_n": ne, "hit": (round(hit, 3) if hit is not None else None),
                          "gross_move_pct": (round(gross * 100, 2) if gross is not None else None),
                          "binom_p": (round(p_b, 3) if p_b is not None else None)}
        return summary

    # -------------------------------------------------- 序列化
    def to_state(self) -> dict:
        return {"H": self.H, "prior_n": self.prior_n, "halflife": self.halflife,
                "thr": self.thr, "state": self.state, "updated_at": self.updated_at}

    @classmethod
    def from_state(cls, blob: dict) -> "EventShockModel":
        m = cls(H=blob.get("H", 5), prior_n=blob.get("prior_n", 1.5),
                halflife=blob.get("halflife", 750), thr=blob.get("thr", 0.62),
                state=blob.get("state"))
        m.updated_at = blob.get("updated_at")
        return m
