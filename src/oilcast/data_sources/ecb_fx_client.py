"""欧洲央行（ECB）参考汇率客户端，经由 Frankfurter 公共镜像，无需 API key。

用途：为上海原油"人民币/桶 → 美元/桶"换算、以及日元外生变量提供**每日更新、带观测
日期、海外稳定可达**的市场汇率。FRED 的 DEXCHUS/DEXJPUS 为在岸定盘、更新常滞后数日
（实测可能停在多日前），故把 ECB 参考汇率作为源链第一优先、FRED 作为兜底。

口径：base=USD，symbols=CNY/JPY，返回"1 美元兑多少人民币/日元"，与 FRED DEXCHUS 同口径；
ECB 每个欧洲工作日约 16:00 CET 发布一次当日参考汇率，周末/节假日无报价（保持缺失，不造数）。
每个数值都可在 https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/
核验；镜像文档 https://frankfurter.dev。
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from ..utils import PoliteSession, get_logger

LOG = get_logger(__name__)
BASE_URL = "https://api.frankfurter.dev/v1"


def fetch_ecb_fx(currency: str, start: datetime, end: datetime,
                 sess: Optional[PoliteSession] = None) -> Optional[pd.Series]:
    """抓取 USD→指定货币（CNY/JPY）的 ECB 每日参考汇率区间序列。

    返回按日期升序、index 为归一化日期的 Series；不可达或无有效数据返回 None（交由源链兜底）。
    """
    currency = currency.upper()
    own = sess is None
    sess = sess or PoliteSession()
    s0, s1 = pd.Timestamp(start).strftime("%Y-%m-%d"), pd.Timestamp(end).strftime("%Y-%m-%d")
    url = f"{BASE_URL}/{s0}..{s1}"
    resp = sess.get(url, params={"base": "USD", "symbols": currency})
    if own:
        pass
    if resp is None:
        return None
    try:
        payload = resp.json()
        rates = payload.get("rates") or {}
        if not rates:
            return None
        rows = [(pd.Timestamp(d), float(v[currency]))
                for d, v in rates.items() if currency in v]
        if not rows:
            return None
        s = pd.Series({d: v for d, v in rows}).sort_index()
        s.index = s.index.normalize()
        s.name = f"ECB:USD{currency}"
        return s
    except Exception as exc:  # 任何解析异常都视为该源失败、交给下一源，绝不抛出拖垮流水线
        LOG.warning("ECB/Frankfurter USD%s 解析失败：%s", currency, exc)
        return None
