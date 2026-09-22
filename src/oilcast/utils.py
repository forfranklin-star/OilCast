"""通用工具：日志、带退避与熔断的 HTTP 会话、交易日工具、确定性随机。"""
from __future__ import annotations

import logging
import random
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

from . import net
from .config import get_config


# ---------------------------------------------------------------- logging
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                              datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ------------------------------------------------------------- randomness
def seeded_rng():
    import numpy as np
    seed = int(get_config()["system"]["random_seed"])
    return np.random.default_rng(seed)


# ----------------------------------------------------------- http session
class PoliteSession:
    """带请求头、重试、固定延时、同主机熔断的 requests 会话（爬虫礼仪）。
    - 统一 User-Agent；请求间 sleep 降低站点压力；5xx/网络异常有限退避重试；
    - 熔断：同一 host 连续连接失败达阈值后，本进程一段时间内对该 host 直接快速
      失败，避免不可达源的多个查询逐个超时拖垮流水线（缺失由上层如实标记）。
    """
    _circuit: dict = {}
    _fail_count: dict = {}
    CIRCUIT_FAIL_THRESHOLD = 2
    CIRCUIT_COOLDOWN_SEC = 600

    def __init__(self, extra_headers: Optional[dict] = None) -> None:
        cfg = get_config()["system"]["request"]
        self.timeout = int(cfg["timeout_sec"])
        self.retry = int(cfg["retry"])
        self.sleep_sec = float(cfg["sleep_between_requests_sec"])
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": str(cfg["user_agent"]),
            "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8",
        })
        if extra_headers:                       # 个别源需要浏览器兼容头时按源覆盖
            self.session.headers.update(extra_headers)
        net.apply_to_session(self.session)      # 应用当前代理（直连时显式关闭环境代理）

    def get(self, url: str, params: Optional[dict] = None,
            encoding: Optional[str] = None) -> Optional[requests.Response]:
        from urllib.parse import urlparse
        host = urlparse(url).netloc
        if time.time() < PoliteSession._circuit.get(host, 0):
            get_logger(__name__).info("主机 %s 熔断中，快速跳过", host)
            return None
        for attempt in range(self.retry + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if encoding:
                    resp.encoding = encoding
                resp.raise_for_status()
                PoliteSession._fail_count.pop(host, None)
                time.sleep(self.sleep_sec)
                return resp
            except (requests.ConnectionError, requests.Timeout) as exc:
                fails = PoliteSession._fail_count.get(host, 0) + 1
                PoliteSession._fail_count[host] = fails
                if fails >= self.CIRCUIT_FAIL_THRESHOLD:
                    # 网络层连续不可达（疑似代理失效/被墙）：交互终端暂停倒计时，等用户换
                    # 代理后清熔断、换新通道重试一次；非交互（CI/网页）或超时无响应则照常熔断。
                    handled = False
                    try:
                        handled = net.maybe_pause_for_proxy(host, exc)
                    except Exception:  # noqa: BLE001 - 交互异常不得拖垮采集
                        handled = False
                    if handled:
                        get_logger(__name__).info("用户已更换/处理网络，用新通道重试 %s", host)
                        net.apply_to_session(self.session)
                        net.reset_circuit_for(host)
                        fails = 0
                        attempt = max(0, attempt - 1)   # 不消耗本次重试额度
                        continue
                    PoliteSession._circuit[host] = time.time() + self.CIRCUIT_COOLDOWN_SEC
                    get_logger(__name__).warning(
                        "主机 %s 连续失败 %d 次，熔断 %ds：%s", host, fails,
                        self.CIRCUIT_COOLDOWN_SEC, exc)
                    return None
                if attempt == self.retry:
                    get_logger(__name__).warning("GET 失败 %s: %s", url, exc)
                    return None
                time.sleep(self.sleep_sec * (2 ** attempt) + random.uniform(0, 0.5))
            except requests.RequestException as exc:
                # 403/429 多为临时限频：用更长退避（6s、18s…）再试，避免短间隔连发被拒
                code = getattr(getattr(exc, "response", None), "status_code", None)
                if attempt == self.retry:
                    get_logger(__name__).warning("GET 失败 %s: %s", url, exc)
                    return None
                if code in (403, 429):
                    time.sleep(6 * (attempt + 1) + random.uniform(0, 1.0))
                else:
                    time.sleep(self.sleep_sec * (2 ** attempt) + random.uniform(0, 0.5))
        return None


# ------------------------------------------------------------ timeout guard
def run_with_timeout(func, args=(), kwargs=None, timeout_sec: float = 25):
    """对第三方库调用加总耗时保护，超时返回 None（上层据此标记缺失，不造假）。"""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
    ex = ThreadPoolExecutor(max_workers=1)
    future = ex.submit(func, *args, **(kwargs or {}))
    try:
        return future.result(timeout=timeout_sec)
    except FutureTimeout:
        get_logger(__name__).warning("调用 %s 超过 %.0fs，已跳过",
                                     getattr(func, "__name__", func), timeout_sec)
        return None
    finally:
        ex.shutdown(wait=False)


# ------------------------------------------------------------- date utils
def now_beijing() -> datetime:
    return datetime.now(timezone(timedelta(hours=8)))


def business_day_index(end: datetime, n_periods: int):
    import pandas as pd
    return pd.bdate_range(end=end.normalize(), periods=n_periods)


def trading_offsets(n_days: int) -> int:
    return max(1, int(round(n_days * 5 / 7)))
