"""多数据源优先级链（failover chain）统一框架。
设计原则（与全局数据原则一致）：
- 每个字段配置一张【有序 provider 列表】，按优先级依次尝试；
- 选源策略 select：
  * "first"    —— 第一个返回足量【真实】观测的源被采用（宏观序列常用）；
  * "freshest" —— 尝试预算内所有源，从成功源里选【末次观测日期最新】者
                  （价格序列常用：现货主源发布滞后时，由更新鲜的期货源顶上）；
- 不同源之间绝不混合拼接，最终只采用单一源的整条序列；
- 每次尝试（源名、是否成功、观测条数、末次观测、耗时、失败原因）都记录在案，
  写入数据谱系，做到"为什么用了这个源、其余源为何没被采用"完全可审计；
- 所有源都失败才返回 unavailable，由上层如实标记，严禁造数补齐。
provider 约定：一个无参可调用（用闭包/functools.partial 绑定参数），
成功返回 (payload, meta)，payload 为 pd.Series / pd.DataFrame；meta 可带
last_observed（末次观测日）、caliber（口径，如"近月连续期货"/"现货"）；
无数据或不可达时返回 None，或直接抛异常（本框架会捕获并记录原因）。
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple
import pandas as pd
from ..utils import get_logger
LOG = get_logger(__name__)
# 部分政府/门户站点会丢弃自报机器人身份或缺少常规浏览器头的请求，
# 对这类源统一使用浏览器兼容头（仍是低速、可识别的善意研究访问）。
BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
@dataclass
class Attempt:
    source: str
    ok: bool = False
    n_obs: int = 0
    elapsed_ms: int = 0
    reason: str = ""
    last_observed: Optional[str] = None
    chosen: bool = False
    def line(self) -> str:
        if self.ok:
            tag = "✓" if not self.chosen else "✓采用"
            last = f"/末次{self.last_observed[5:]}" if self.last_observed else ""
            return f"{self.source}{tag}{self.n_obs}条{last}/{self.elapsed_ms}ms"
        return f"{self.source}✗({self.reason})"
@dataclass
class ChainResult:
    payload: Any = None
    used: Optional[str] = None
    meta: dict = field(default_factory=dict)
    attempts: List[Attempt] = field(default_factory=list)
    @property
    def ok(self) -> bool:
        return self.payload is not None
    def trail_text(self) -> str:
        """人类可读的尝试链，写入谱系 note，例如 A✗(超时) → B✓采用740条/末次09-04/2s。"""
        return " → ".join(a.line() for a in self.attempts)
def _count(payload) -> int:
    if payload is None:
        return 0
    if isinstance(payload, pd.Series):
        return int(payload.dropna().shape[0])
    if isinstance(payload, pd.DataFrame):
        return int(len(payload))
    return 1
def _last_date(payload, meta) -> Optional[pd.Timestamp]:
    """从 meta.last_observed 或 payload 索引推断末次真实观测日。"""
    lo = (meta or {}).get("last_observed")
    if lo is not None:
        try:
            return pd.Timestamp(lo)
        except Exception:
            pass
    if isinstance(payload, pd.Series):
        s = payload.dropna()
        if not s.empty:
            return pd.Timestamp(s.index.max())
    if isinstance(payload, pd.DataFrame):
        idx = payload.index
        if len(idx):
            return pd.Timestamp(idx.max())
    return None
def run_chain(providers: List[Tuple[str, Callable[[], Optional[Tuple[Any, dict]]]]],
              field_name: str = "",
              min_obs: int = 1,
              chain_timeout_sec: float = 90.0,
              select: str = "first") -> ChainResult:
    """按优先级依次尝试 providers。
    providers: [(源显示名, 零参 callable -> (payload, meta) | None), ...]
    min_obs:   payload 至少要有多少真实观测才算成功（防止拿到空壳/残缺源）。
    select:    "first" 首个成功即用；"freshest" 全部尝试后选末次观测最新者。
    返回 ChainResult；单个 provider 异常不影响后续源尝试。
    """
    result = ChainResult()
    chain_start = time.time()
    candidates: List[Tuple[str, Any, dict, Attempt, pd.Timestamp]] = []
    for name, fn in providers:
        if time.time() - chain_start > chain_timeout_sec:
            result.attempts.append(Attempt(name, False, reason="整条链超时，跳过后续源"))
            LOG.warning("[%s] 源链总耗时超 %.0fs，停止尝试", field_name, chain_timeout_sec)
            break
        att = Attempt(source=name)
        t0 = time.time()
        try:
            got = fn()
            payload, meta = got if isinstance(got, tuple) else (got, {})
            meta = meta or {}
            att.elapsed_ms = int((time.time() - t0) * 1000)
            n = _count(payload)
            if payload is None or n < min_obs:
                att.ok, att.reason = False, (f"观测不足({n}<{min_obs})"
                                             if payload is not None else "无返回")
                LOG.info("[%s] 源 %s 不可用：%s", field_name, name, att.reason)
            else:
                att.ok, att.n_obs = True, n
                ld = _last_date(payload, meta)
                att.last_observed = ld.strftime("%Y-%m-%d") if ld is not None else None
                candidates.append((name, payload, meta, att, ld))
                if select == "first":
                    att.chosen = True
                    result.payload, result.used, result.meta = payload, name, meta
                    result.attempts.append(att)
                    LOG.info("[%s] 命中源 %s（%d 条, 末次 %s, %dms）",
                             field_name, name, n, att.last_observed, att.elapsed_ms)
                    break
                LOG.info("[%s] 候选源 %s（%d 条, 末次 %s, %dms），继续比较新鲜度",
                         field_name, name, n, att.last_observed, att.elapsed_ms)
        except Exception as exc:  # 单源失败不得拖垮整条链
            att.elapsed_ms = int((time.time() - t0) * 1000)
            att.reason = f"{type(exc).__name__}: {str(exc)[:80]}"
            LOG.warning("[%s] 源 %s 异常：%s", field_name, name, att.reason)
        result.attempts.append(att)
    if select == "freshest" and candidates:
        # 末次观测最新者胜出；并列时取源链中更靠前（优先级更高）者
        best_i = max(range(len(candidates)),
                     key=lambda i: (candidates[i][4] if candidates[i][4] is not None
                                    else pd.Timestamp.min, -i))
        name, payload, meta, att, ld = candidates[best_i]
        att.chosen = True
        result.payload, result.used, result.meta = payload, name, meta
        others = [c[0] for j, c in enumerate(candidates) if j != best_i]
        LOG.info("[%s] 新鲜度选源：采用 %s（末次 %s），其余候选 %s",
                 field_name, name, att.last_observed, others or "无")
    if not result.ok:
        LOG.warning("[%s] 全部 %d 个源均失败，标记 unavailable：%s",
                    field_name, len(providers), result.trail_text())
    return result
