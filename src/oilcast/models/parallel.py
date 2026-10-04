"""可复现的并行执行工具。

背景
----
joblib 多进程 worker 是独立解释器，主进程里 ``sklearn.set_config(...)`` 的全局配置与
全局随机数状态默认不会传过去。自定义并行任务若依赖它们，结果会随机器、worker 数量或
调度顺序漂移；普通回测大概率看不出影响，但严谨的量化实验必须可复现。

实现要点（为何不用 ``sklearn.utils.parallel.delayed``）
----------------------------------------------------
``sklearn.utils.parallel.delayed`` 必须与 ``sklearn.utils.parallel.Parallel`` 配套，
否则 worker 内会反复抛出 ``UserWarning: ...delayed should be used with ...Parallel``。
为彻底规避这一配套要求（并兼容不同 sklearn 版本），这里：

1. 使用 **joblib 原生的** ``joblib.Parallel`` / ``joblib.delayed``（二者天然配套，
   不会触发 sklearn 的 delayed 配套告警）；
2. 每个任务显式接收主进程的 sklearn 配置快照，并在任务体内再 ``set_config`` 一次，
   手动把配置传播进 worker（等价于 sklearn.Parallel 的自动传播，不依赖其内部实现）；
3. 每个任务的随机种子由 ``base_seed`` 与任务【索引】经 ``SeedSequence.spawn`` 确定性
   派生，与 worker 数量、调度顺序无关；任务体内只能用传入 seed 构造随机源，禁止读取
   全局 numpy RNG。

数据量小、无需并行时请用 ``n_jobs=1``（默认）：单进程天然可复现。
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

import joblib
import numpy as np

#: 串行（默认）：估计器/任务在主进程内执行，配置与随机状态不跨进程、严格可复现。
SERIAL_N_JOBS = 1

#: set_config 可安全接受、且影响数值或执行行为的字段（仅传播这些，避免版本差异）。
_CONFIG_KEYS = (
    "assume_finite", "working_memory", "print_changed_only",
    "pairwise_dist_chunk_size", "enable_cython_pairwise_dist",
)


def config_snapshot() -> dict:
    """主进程当前 sklearn 全局配置的可传播子集。"""
    import sklearn
    cfg = sklearn.get_config()
    return {k: cfg[k] for k in _CONFIG_KEYS if k in cfg}


def derive_seeds(base_seed: int, n_tasks: int) -> List[int]:
    """按任务索引确定性派生 n 个种子（只依赖 base_seed，与进程数/顺序无关）。"""
    children = np.random.SeedSequence(int(base_seed)).spawn(int(n_tasks))
    return [int(child.generate_state(1)[0]) for child in children]


def _run_one(cfg: dict, seed: int, fn: Callable,
             index: int, args: tuple, kwargs: dict) -> Any:
    """worker 内执行：先手动恢复主进程 sklearn 配置，再用传入种子运行任务。"""
    import sklearn
    try:  # 手动传播配置，不依赖 sklearn.Parallel 的自动注入
        sklearn.set_config(**cfg)
    except Exception:
        pass
    return fn(index, seed, *args, **kwargs)


def reproducible_run(fn: Callable, n_tasks: int, base_seed: int = 42,
                     n_jobs: int = SERIAL_N_JOBS,
                     args: Tuple[Any, ...] = (),
                     kwargs: Optional[dict] = None) -> List[Any]:
    """对 ``n_tasks`` 个任务做可复现并行，返回按任务【索引】排列的结果列表。

    参数
    ----
    fn :
        任务函数，签名必须为 ``fn(index: int, seed: int, *args, **kwargs)``；函数体内
        只能用 ``np.random.default_rng(seed)`` 等方式使用传入种子，**不得读取全局 RNG**。
    n_tasks :
        任务数量（索引 0..n-1）。
    base_seed :
        派生种子的根种子；相同 base_seed 与 n_tasks 必得到完全相同的种子序列。
    n_jobs :
        并行度；默认 1（串行、最稳）。确需并行时可设 >1，配置与种子仍保证可复现。
    """
    kwargs = kwargs or {}
    cfg = config_snapshot()
    seeds = derive_seeds(base_seed, n_tasks)
    # joblib 原生 Parallel + delayed 配套；_run_one 内手动 set_config 传播 sklearn 配置
    return joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_run_one)(cfg, seeds[i], fn, i, args, kwargs)
        for i in range(int(n_tasks)))
