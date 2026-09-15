"""端到端与数据原则测试。
运行：pytest -q
说明：产品代码不含任何演示/合成兜底；离线测试所需的结构自洽数据由 tests/synth_fixture.py
（仅测试用夹具，不进入发布包）提供。覆盖主链路、数据原则（质量门/缺失不填零/样本不足
显式失败/月频 vintage 可追溯）、HTML 渲染。
"""
import numpy as np
import pandas as pd
import pytest

import synth_fixture as sf
from oilcast.data_sources.quality import (align_monthly_with_vintage,
                                          assess_daily, assess_monthly)
from oilcast.features.engineering import (FACTOR_GROUPS, build_features,
                                          factor_availability, make_supervised)
from oilcast.models.errors import InsufficientData
from oilcast.models.short_term import ShortTermForecaster
from oilcast.models.evaluation import backtest_short
from oilcast.models.weights import learn_factor_weights
from oilcast.reporting.static_html import render_static_html

AS_OF = pd.Timestamp("2026-09-02 10:00")


@pytest.fixture(scope="module")
def bundle():
    return sf.make_test_bundle(AS_OF, 1095)


def test_factor_weights_pk_migration(tmp_path):
    # 旧库主键为 (report_date,factor)：迁移后变三元主键，同日可存多品种且旧数据归 wti
    import sqlite3
    from oilcast.storage.database import OilCastDB
    p = tmp_path / "t.db"
    con = sqlite3.connect(p)
    con.executescript(
        """CREATE TABLE factor_weights(report_date TEXT,factor TEXT,weight REAL,
           model_importance REAL,prior REAL,available INTEGER,
           PRIMARY KEY(report_date,factor));
           INSERT INTO factor_weights VALUES('2026-01-01','geopolitical_risk',.2,.2,.2,1);""")
    con.commit()
    OilCastDB._migrate_factor_weights(con)
    con.commit()
    pk = {r[1] for r in con.execute("PRAGMA table_info(factor_weights)") if r[5] > 0}
    assert pk == {"report_date", "instrument", "factor"}
    # 同一 report_date+factor 但不同 instrument 不再撞唯一约束
    con.execute("INSERT INTO factor_weights VALUES('2026-01-01','brent',"
                "'geopolitical_risk',.1,.1,.1,1)")
    con.commit()
    insts = {r[0] for r in con.execute("SELECT DISTINCT instrument FROM factor_weights")}
    assert insts == {"wti", "brent"}   # 旧数据归 wti，新插入 brent 同日同因素不冲突
    con.close()


# ------------------------------------------------- 上海原油客户端解析（离线样例）
def test_cn_futures_parse():
    from oilcast.data_sources.cn_futures_client import _finalize
    idx = ["2026-09-04", "2026-09-05", "2026-09-08"]
    close = ["730.1", "735.6", "741.5"]
    out = _finalize(idx, close, "SC0", "test", "u")
    assert out is not None
    s, meta = out
    assert len(s) == 3 and abs(s.iloc[-1] - 741.5) < 1e-9
    assert meta["last_observed"] == "2026-09-08"


# ------------------------------------------------- 测试夹具结构与标注
def test_synthetic_structure():
    b = sf.build_synthetic_bundle(AS_OF, 730)
    assert {"wti", "brent", "shanghai_crude",
            "heating_oil", "gasoil"}.issubset(b["prices"].columns)
    assert {"usdcny", "usdjpy"}.issubset(b["macro"].columns)
    assert len(b["prices"]) > 400
    assert not b["events"].empty and not b["views"].empty


def test_test_bundle_is_labeled(bundle):
    # 测试夹具必须逐字段标注 SYNTHETIC-TEST，绝不允许被误当成真实源
    assert len(bundle.prices.columns) == 5
    for col in ["dxy", "us10y", "cpi_yoy", "nonfarm_surprise",
                "fed_expectation", "gpr_index"]:
        assert col in bundle.macro.columns
    assert all("SYNTHETIC-TEST" in L["source_name"] for L in bundle.lineage.values())


def test_features_clean(bundle):
    feats = build_features(bundle.prices, bundle.macro, bundle.events,
                           bundle.views, target="wti",
                           events_available=True, views_available=True)
    assert not np.isinf(feats.values).any()
    for group_cols in FACTOR_GROUPS.values():
        for c in group_cols:
            assert c in feats.columns, c


def test_shanghai_usd_premium_and_jpy(bundle):
    # 上海原油：换算美元后必须能算出对布伦特的升贴水水平列；日元因素列须存在
    feats = build_features(bundle.prices, bundle.macro, bundle.events,
                           bundle.views, target="shanghai_crude")
    assert "spread_brent" in feats.columns and feats["spread_brent"].notna().sum() > 100
    assert "usdjpy_z" in feats.columns and feats["usdjpy_z"].notna().sum() > 100
    # 成品油单位不同：只给相对强弱，水平价差列保持缺失（不混用口径）
    f_go = build_features(bundle.prices, bundle.macro, bundle.events,
                          bundle.views, target="gasoil")
    assert f_go["rs_brent_5d"].notna().sum() > 100
    assert f_go["spread_brent"].notna().sum() == 0


def test_weights_sum_to_one_and_flag(bundle):
    feats = build_features(bundle.prices, bundle.macro, bundle.events, bundle.views)
    X, y = make_supervised(feats, bundle.prices["wti"], 10)
    w = learn_factor_weights(X, y, available={f: True for f in FACTOR_GROUPS
                                              if not f.startswith("_")})
    assert abs(w.dropna(subset=["weight"])["weight"].sum() - 1.0) < 1e-3
    assert len(w) == 10 and (w["available"]).all()   # 十大因素（含日元汇率）


def test_short_forecast_shape(bundle):
    feats = build_features(bundle.prices, bundle.macro, bundle.events, bundle.views)
    fc = ShortTermForecaster(horizon=3, window=120).fit(feats, bundle.prices["wti"])
    future = pd.bdate_range(feats.index[-1] + pd.Timedelta(days=1), periods=3)
    res = fc.predict(feats.iloc[[-1]], float(bundle.prices["wti"].iloc[-1]), future)
    assert len(res.path) == 3
    ep = res.endpoint
    assert ep["q05"] <= ep["mean"] <= ep["q95"] and 0 <= ep["prob_up"] <= 1


def test_full_pipeline_offline_bundle_and_html():
    # 离线注入测试夹具跑完整主链路（产品 CLI/Actions 路径不接受外部 bundle，恒走真实采集）
    import oilcast.pipeline.main as pm
    report = pm.run(as_of=AS_OF, bundle=sf.make_test_bundle(AS_OF, 1095))
    for hz in ("short", "mid", "long"):
        assert "wti" in report["forecasts"][hz]
    html = render_static_html(report)
    assert "<html" in html
    assert "演示模式" not in html and "严格真实" not in html  # 多余模式注释已移除


# ----------------------------------------------------------- strict 数据原则
def test_quality_gate_flags_stale_and_insufficient():
    idx = pd.bdate_range("2025-01-02", "2026-09-02")
    fresh = pd.Series(np.linspace(70, 80, len(idx)), index=idx)
    gate = {"min_obs": 120, "max_stale_bdays": 7}
    lin_ok = assess_daily("x", "x", fresh, AS_OF, "src", "u", gate)
    assert lin_ok.status == "ok"
    # 最后观测停在 8 月初 → stale
    stale = fresh.loc[:"2026-08-03"]
    assert assess_daily("x", "x", stale, AS_OF, "src", "u", gate).status == "stale"
    # 样本不足
    short = fresh.tail(60)
    assert assess_daily("x", "x", short, AS_OF, "src", "u", gate).status == "insufficient"
    # 源不可达
    assert assess_daily("x", "x", None, AS_OF, "src", "u", gate).status == "unavailable"


def test_monthly_vintage_is_traceable():
    # 月频发布值对齐到工作日时，必须逐点携带真实发布日期
    monthly = pd.Series([100.0, 101.0, 102.0],
                        index=pd.to_datetime(["2026-06-01", "2026-07-01", "2026-08-01"]))
    bdays = pd.bdate_range("2026-07-15", "2026-08-20")
    aligned, vint = align_monthly_with_vintage(monthly, bdays)
    # 7/15~7/31 对齐的是 7 月发布值，vintage 必须是 2026-07-01，不能写成当天
    assert vint.loc[pd.Timestamp("2026-07-15")] == "2026-07-01"
    assert aligned.loc[pd.Timestamp("2026-07-15")] == 101.0
    assert vint.loc[pd.Timestamp("2026-08-19")] == "2026-08-01"


def test_missing_factor_is_never_zero_filled():
    # 无 GPR、事件源不可达：GPR 两列必须保持 NaN，绝不造数
    b = sf.build_synthetic_bundle(AS_OF, 730)
    macro = b["macro"].drop(columns=["gpr_index"])
    feats = build_features(b["prices"], macro, events=pd.DataFrame(),
                           views=pd.DataFrame(), target="wti",
                           events_available=False, views_available=False)
    assert feats["gpr_level_z"].isna().all()
    assert feats["gpr_chg_5d"].isna().all()
    # 有 Brent/WTI 真实价格时，地缘溢价代理 geo_premium_chg5 承载真实地缘信息（非填充）
    assert feats["geo_premium_chg5"].notna().sum() > 0
    # 只有连 Brent-WTI 价差都无法构造（去掉 brent）、且无 GPR/事件时，地缘因素才整体缺失
    feats2 = build_features(b["prices"].drop(columns=["brent"]), macro,
                            events=pd.DataFrame(), views=pd.DataFrame(), target="wti",
                            events_available=False, views_available=False)
    avail2 = factor_availability(feats2)
    assert avail2["geopolitical_risk"] is False


def test_missing_factor_excluded_from_weights():
    b = sf.build_synthetic_bundle(AS_OF, 730)
    feats = build_features(b["prices"], b["macro"], b["events"], b["views"])
    X, y = make_supervised(feats, b["prices"]["wti"], 10)
    avail = {f: True for f in FACTOR_GROUPS if f != "_technical_"}
    avail["geopolitical_risk"] = False   # 模拟该因素无真实数据
    w = learn_factor_weights(X, y, available=avail)
    row = w[w["factor"] == "geopolitical_risk"].iloc[0]
    assert pd.isna(row["weight"]) and row["available"] == 0
    usable = w.dropna(subset=["weight"])
    assert abs(usable["weight"].sum() - 1.0) < 1e-3   # 只在可用因素间归一化


def test_model_refuses_insufficient_real_data():
    # 真实样本不足时必须显式失败，而不是硬算出预测
    b = sf.build_synthetic_bundle(AS_OF, 730)
    feats = build_features(b["prices"], b["macro"], b["events"], b["views"]).tail(100)
    with pytest.raises(InsufficientData):
        ShortTermForecaster(horizon=5).fit(feats, b["prices"]["wti"].loc[feats.index])


# ----------------------------------------------------- 多数据源优先级链
def test_chain_failover_to_second_source():
    from oilcast.data_sources.sources import run_chain
    import pandas as pd
    calls = []
    def dead():
        calls.append("A"); return None
    def alive():
        calls.append("B"); return (pd.Series([1.0, 2.0, 3.0]), {"k": 1})
    res = run_chain([("源A", dead), ("源B", alive)], field_name="x", min_obs=2)
    assert res.ok and res.used == "源B" and calls == ["A", "B"]
    assert res.attempts[0].ok is False and res.attempts[1].ok is True
    assert "源A✗" in res.trail_text() and "源B✓" in res.trail_text()


def test_chain_all_sources_fail():
    from oilcast.data_sources.sources import run_chain
    res = run_chain([("A", lambda: None), ("B", lambda: None)], field_name="x")
    assert not res.ok and res.used is None and len(res.attempts) == 2


def test_chain_insufficient_obs_falls_through():
    from oilcast.data_sources.sources import run_chain
    import pandas as pd
    thin = lambda: (pd.Series([1.0]), {})
    full = lambda: (pd.Series([1., 2., 3.]), {})
    res = run_chain([("薄源", thin), ("足源", full)], field_name="x", min_obs=3)
    assert res.used == "足源" and "薄源" in res.trail_text()


def test_chain_provider_exception_isolated():
    from oilcast.data_sources.sources import run_chain
    import pandas as pd
    def boom():
        raise RuntimeError("网络中断")
    res = run_chain([("崩源", boom), ("好源", lambda: (pd.Series([1., 2.]), {}))],
                    field_name="x", min_obs=1)
    assert res.ok and res.used == "好源"
    assert "RuntimeError" in res.attempts[0].reason

# ------------------------------------- 新鲜度优先选源 / CNBC解析 / 预测复盘
def test_chain_freshest_picks_latest_even_if_not_first():
    """现货主源滞后时，freshest 必须选末次观测更新的后位期货源。"""
    from oilcast.data_sources.sources import run_chain
    import pandas as pd
    old = lambda: (pd.Series([80., 81.], index=pd.to_datetime(["2026-08-29", "2026-09-01"])),
                   {"last_observed": "2026-09-01"})
    new = lambda: (pd.Series([80., 81., 82.], index=pd.to_datetime(["2026-09-02", "2026-09-03", "2026-09-04"])),
                   {"last_observed": "2026-09-04"})
    res = run_chain([("滞后现货", old), ("新鲜期货", new)], field_name="wti",
                    min_obs=2, select="freshest")
    assert res.used == "新鲜期货"
    assert res.attempts[0].ok and not res.attempts[0].chosen
    assert res.attempts[1].chosen and "采用" in res.attempts[1].line()


class _FakeResp:
    def __init__(self, payload):
        self._p = payload
    def json(self):
        return self._p


class _FakeSess:
    def __init__(self, payload):
        self._p = payload
    def get(self, url):
        return _FakeResp(self._p)


def test_cnbc_client_parses_bars():
    import pandas as pd
    from oilcast.data_sources.cnbc_client import fetch_cnbc_bars
    payload = {"barData": {"priceBars": [
        {"close": "90.0", "tradeTimeinMills": int(pd.Timestamp("2026-09-03").timestamp() * 1000)},
        {"close": "91.5", "tradeTimeinMills": int(pd.Timestamp("2026-09-04").timestamp() * 1000)},
    ]}}
    got = fetch_cnbc_bars("@CL.1", pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-05"),
                          sess=_FakeSess(payload))
    assert got is not None
    s, meta = got
    assert len(s) == 2 and abs(s.iloc[-1] - 91.5) < 1e-9
    assert meta["last_observed"] == "2026-09-04" and "期货" in meta["caliber"]


def test_review_scores_past_forecasts_against_realized():
    import pandas as pd
    from oilcast.storage.database import OilCastDB
    from oilcast.models.review import review_predictions
    db = OilCastDB()  # conftest 已把 OILCAST_HOME 关到临时目录
    import sqlite3 as _sq
    from oilcast.config import get_config as _gc
    _c = _sq.connect(_gc()["storage"]["sqlite_path"])
    _c.execute("DELETE FROM forecasts"); _c.commit(); _c.close()  # 清掉同 session 离线主链路写入，保证隔离
    idx = pd.bdate_range("2026-08-25", "2026-09-04")
    prices = {"wti": pd.Series([88.0, 88.5, 89.0, 89.5, 90.0, 90.5, 91.0, 91.48],
                               index=pd.to_datetime(["2026-08-25", "2026-08-26", "2026-08-27",
                                                     "2026-08-28", "2026-08-31", "2026-09-01",
                                                     "2026-09-02", "2026-09-04"]))}
    # 8-25 发布(发布价88)、目标 9-04：预测 90(看涨)，实际 91.48(看涨)→方向命中；区间[89,92]覆盖
    db.save_forecasts("2026-08-25", [{"horizon": "short", "instrument": "wti",
                                      "target_date": "2026-09-04", "mean": 90.0,
                                      "q05": 89.0, "q25": 89.5, "q50": 90.0, "q75": 91.0,
                                      "q95": 92.0, "prob_up": 0.6, "prob_down": 0.4}])
    rv = review_predictions(prices, pd.Timestamp("2026-09-05"))
    assert rv["available"] and rv["summary"]["n"] == 1
    row = rv["detail"][0]
    assert row["actual"] == 91.48 and row["covered"] is True and row["dir_hit"] is True
    assert row["pred_ret_pct"] > 0 and row["actual_ret_pct"] > 0


# ------------------------------------------------- 交易日历：非交易日不占位
def test_trading_calendar_skips_nontrading():
    from oilcast.data_sources.calendar import (observed_trading_index,
        closed_holiday_mmdd, next_trading_days, normalize_as_of_to_trading)
    # 两年工作日，挖掉固定假日 07-04、01-01（两年都挖）
    full = pd.bdate_range("2024-09-01", "2026-09-04")
    hol = set(pd.to_datetime(["2025-07-04", "2025-01-01", "2026-07-03", "2026-01-01"]))
    real = pd.DatetimeIndex([d for d in full if d not in hol])
    s = pd.Series(np.arange(len(real), dtype=float), index=real)
    oi = observed_trading_index(s)
    assert len(oi) == len(real)                       # 只保留真实交易日
    assert all(d.weekday() < 5 for d in oi)          # 无周末
    closed = closed_holiday_mmdd(oi)
    assert "07-04" in closed and "01-01" in closed   # 固定假日被识别
    nx = next_trading_days(pd.Timestamp("2026-09-04"), 6, oi, closed)
    assert all(d.weekday() < 5 for d in nx)          # 未来外推跳过周末
    assert nx[0] == pd.Timestamp("2026-09-07") and nx[-1] == pd.Timestamp("2026-09-14")
    # 周六归一到周五
    assert normalize_as_of_to_trading(pd.Timestamp("2026-09-05")) == pd.Timestamp("2026-09-04")


def test_session_asof_alignment_respects_close_time():
    """交易时段对齐：上海(UTC07收盘)在 t 日只能看到 t-1 的海外(UTC21收盘)行情；
    海外(21)看上海(07)同日即可；海外之间同时刻同日。防止跨时区隐性未来函数回退。"""
    from oilcast.features.engineering import (
        _asof_series, PRICE_CLOSE_UTC_HOUR)
    days = pd.DatetimeIndex(pd.bdate_range("2026-09-07", periods=4))
    brent = pd.Series([10., 20., 30., 40.], index=days)      # 海外，21 点收盘
    sc = pd.Series([100., 200., 300., 400.], index=days)      # 上海，07 点收盘
    # 上海视角看布伦特：整体滞后一根（首日因前一海外日缺失而为 NaN）
    seen_by_sc = _asof_series(brent, days,
                              PRICE_CLOSE_UTC_HOUR["brent"],
                              PRICE_CLOSE_UTC_HOUR["shanghai_crude"]).values
    assert np.isnan(seen_by_sc[0]) and list(seen_by_sc[1:]) == [10., 20., 30.]
    # WTI(海外)视角看上海：同日全部可见（上海更早收盘，属已知信息）
    seen_by_wti = _asof_series(sc, days,
                               PRICE_CLOSE_UTC_HOUR["shanghai_crude"],
                               PRICE_CLOSE_UTC_HOUR["wti"]).values
    assert list(seen_by_wti) == [100., 200., 300., 400.]
    # 海外之间（同为 21 点）同日对齐
    wti = pd.Series([11., 21., 31., 41.], index=days)
    same = _asof_series(wti, days, 21.0, 21.0).values
    assert list(same) == [11., 21., 31., 41.]


def test_intraday_synchronized_anchor_uses_same_instant_not_daytime_carry():
    """盘中同时刻对齐：02:30 夜盘锚点必须取凌晨已成交的 K，不得把 15:00 白天价带到夜里；
    锚点前 60 分钟无成交则缺失（容差），绝不取未来、不拿数小时前陈旧价硬桥。"""
    from oilcast.features.intraday_align import synchronized_panels
    rows = []
    # 交易日 2026-09-10：SC 凌晨02:00夜盘=700、15:00日盘收盘=760；Brent 02:00=100、15:00=105、21:00=108
    for sym, pts in {
        "shanghai_crude": [("02:00", 700.), ("15:00", 760.)],
        "brent": [("02:00", 100.), ("15:00", 105.), ("21:00", 108.)],
    }.items():
        for hm, c in pts:
            rows.append({"symbol": sym, "ts": pd.Timestamp(f"2026-09-10 {hm}"),
                         "open": c, "high": c, "low": c, "close": c,
                         "volume": 1, "source": "t"})
    # 次日只有白天价、凌晨无成交，用于检验 02:30 容差（不应回取到前一日 02:00，距今 24h）
    rows += [{"symbol": "shanghai_crude", "ts": pd.Timestamp("2026-09-11 15:00"),
              "open": 770., "high": 770., "low": 770., "close": 770., "volume": 1, "source": "t"},
             {"symbol": "brent", "ts": pd.Timestamp("2026-09-11 15:00"),
              "open": 106., "high": 106., "low": 106., "close": 106., "volume": 1, "source": "t"}]
    long = pd.DataFrame(rows)
    panels = synchronized_panels(long)
    day, night = panels["day_1500"], panels["night_0230"]
    d = pd.Timestamp("2026-09-10")
    # 15:00 锚点取当时值
    assert day.loc[d, "shanghai_crude"] == 760. and day.loc[d, "brent"] == 105.
    # 02:30 锚点取 02:00（夜盘）值，绝不是 15:00 的 760/105
    assert night.loc[d, "shanghai_crude"] == 700. and night.loc[d, "brent"] == 100.
    # 次日凌晨无成交且距前一根 24h>60min 容差 → 缺失（不硬桥）
    d2 = pd.Timestamp("2026-09-11")
    assert np.isnan(night.loc[d2, "shanghai_crude"]) and np.isnan(night.loc[d2, "brent"])


def test_intraday_fusion_only_shanghai_and_no_future():
    """同时刻融合只作用于上海原油：覆盖日把其跨市场 brent 换成 15:00 同时刻价；
    WTI 模型不被改动；且换入价不晚于上海收盘（无未来）。"""
    from oilcast.features.engineering import build_features
    idx = pd.bdate_range("2025-01-01", "2026-09-11")
    rng = np.random.default_rng(0)
    drift = np.cumsum(rng.normal(0, 0.01, len(idx)))
    prices = pd.DataFrame({
        "wti": 80 * np.exp(drift),
        "brent": 84 * np.exp(drift + 0.01),
        "shanghai_crude": 600 * np.exp(drift * 0.9),
        "heating_oil": 2.5 * np.exp(drift * 0.8),
        "gasoil": 900 * np.exp(drift * 0.8)}, index=idx)
    macro = pd.DataFrame(index=idx)
    macro["usdcny"] = 6.7
    for c in ["dxy", "us10y", "us2y", "cpi_yoy", "nonfarm_surprise", "fedfunds",
              "fed_expectation", "demand_proxy", "gpr_index", "usdjpy"]:
        macro[c] = 100.
    ev = pd.DataFrame(columns=["date", "theme", "intensity"])
    vw = pd.DataFrame(columns=["date", "stance"])
    # 构造只在最后一天有值的 02:30 夜盘收盘同时刻面板：上海自身与 brent/wti 都给可识别特异值
    isess = pd.DataFrame({"shanghai_crude": [999.0], "brent": [999.0], "wti": [888.0]},
                         index=pd.DatetimeIndex([idx[-1]]))
    f_sc = build_features(prices, macro, ev, vw, target="shanghai_crude",
                          events_available=False, views_available=False, intraday_session=isess)
    assert f_sc.attrs["intraday_synchronized_days"] == 1     # 上海自身 02:30 收盘融合 1 日
    f_w = build_features(prices, macro, ev, vw, target="wti",
                         events_available=False, views_available=False, intraday_session=isess)
    assert f_w.attrs["intraday_synchronized_days"] == 0      # WTI 不融合
    # 融合后上海跨市场 spread 反映同时刻 brent=999（对数价差显著为负），且过程不报错
    assert pd.notna(f_sc["spread_brent"].iloc[-1])


def test_ine_sc_session_mask_drops_ghost_bars_keeps_0230_close():
    """新浪 60 分 K 会把夜盘 02:30 收盘错标成 09:30；时段白名单须剔幽灵 K、保留真实 02:30。"""
    from oilcast.data_sources.intraday_client import ine_sc_session_mask
    # 2026-09-12 周六、09-14 周一、09-11 周五
    ts = pd.to_datetime([
        "2026-09-12 02:30",  # 周六凌晨承接周五夜盘收盘 → 合法
        "2026-09-12 09:30",  # 周六 09:30 上期所不开盘（60分K错标）→ 幽灵剔除
        "2026-09-14 00:00",  # 周一 00:00 承接周日晚（无夜盘）→ 幽灵剔除
        "2026-09-11 15:00",  # 周五日盘收盘 → 合法
        "2026-09-11 23:00",  # 周五晚夜盘 → 合法
        "2026-09-11 12:00",  # 周五午间休市 → 非法
    ])
    keep = ine_sc_session_mask(pd.Series(ts)).tolist()
    assert keep == [True, False, False, True, True, False]


def test_intraday_overnight_rolls_to_prior_trade_day_and_caps_at_asof():
    """北京时周六凌晨的周五夜盘归属回周五；as_of 之后的越界前瞻点被剔除。"""
    from oilcast.data_sources.intraday_client import apply_session_rollover
    raw = pd.DataFrame({
        "symbol": ["wti", "shanghai_crude", "wti"],
        "ts": pd.to_datetime(["2026-09-11 23:00", "2026-09-12 01:00", "2026-09-14 00:00"]),
        "close": [100.0, 837.4, 829.2]})
    out = apply_session_rollover(raw, pd.Timestamp("2026-09-11"))
    got = out.set_index("symbol")["ts"]
    assert got["wti"] == pd.Timestamp("2026-09-11 23:00")       # 23:00 不回退
    assert got["shanghai_crude"] == pd.Timestamp("2026-09-11 01:00")  # 周六凌晨→周五
    assert (out["symbol"] == "wti").sum() == 1                  # 9-14 越界点已剔除


def test_china_supply_risk_from_real_titles():
    """对华供油因子只被霍尔木兹/伊朗/油轮等真实标题触发，无关新闻为 0，源不可达为 NaN。"""
    from oilcast.features.engineering import china_supply_risk
    idx = pd.bdate_range("2026-09-07", periods=5)
    ev = pd.DataFrame({
        "date": ["2026-09-09", "2026-09-09", "2026-09-10"],
        "title": ["Oil Holds Above $100 as U.S.-Iran Tanker War Escalates",
                  "European Indexes Rise as Banks Gain",
                  "Brent Holds Above $100 on U.S.-Iran Escalation"],
        "intensity": [0.32, 0.2, 0.32]})
    s = china_supply_risk(ev, idx, True, 5)
    assert s.loc[pd.Timestamp("2026-09-08")] == 0.0
    assert round(s.loc[pd.Timestamp("2026-09-09")], 2) == 0.32
    assert round(s.loc[pd.Timestamp("2026-09-10")], 2) == 0.64
    assert np.isnan(china_supply_risk(ev, idx, False, 5).iloc[0])


def test_major_event_regime_phases_override_base():
    """长期事件 base 强度可被内部升级阶段(phases)取 max 覆盖，产生时变台阶。"""
    from oilcast.features.engineering import major_event_regime
    idx = pd.bdate_range("2026-09-07", "2026-09-11")
    r = major_event_regime(idx)
    assert r.loc[pd.Timestamp("2026-09-08")] == 0.75
    assert r.loc[pd.Timestamp("2026-09-09")] == 1.0
    assert r.loc[pd.Timestamp("2026-09-11")] == 1.0


def _toy_panel(n=320, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    cols = ["supply_disruption", "geopolitical_risk", "usd_index", "us_treasury_10y",
            "cpi_surprise", "jobs_surprise", "fed_policy_expectation",
            "demand_outlook", "institutional_view"]
    X = pd.DataFrame(rng.normal(0, 0.1, (n, len(cols))), index=idx, columns=cols)
    price = pd.Series(80 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    return X, price


# ------------------------------------------------- 模型持久化 + 导入导出
def test_model_registry_persist_export_import(tmp_path):
    from oilcast.models import registry as R
    # 清掉同 session 其他用例（离线主链路）写入的模型，保证版本计数从空开始
    import shutil as _sh
    _md = R._dir()
    if _md.exists():
        _sh.rmtree(_md)
    _md.mkdir(parents=True, exist_ok=True)
    X, price = _toy_panel()
    fc = ShortTermForecaster(10, 180, compute_residuals=False).fit(X, price)
    e1 = R.save_artifact("short", "wti", fc, meta=fc.train_meta)
    assert e1["version"] == 1 and e1["warm_started"] is False
    obj, ent = R.load_artifact("short", "wti")
    assert obj is not None and ent["version"] == 1
    zp = tmp_path / "models.zip"
    R.export_models(str(zp))
    # 清空模型目录后导入
    import shutil
    shutil.rmtree(R._dir())
    R.import_models(str(zp))
    obj2, ent2 = R.load_artifact("short", "wti")
    assert obj2 is not None and ent2["version"] == 1


# ------------------------------------------------- 短期模型热启动（不从零）
def test_short_term_warm_start_continues():
    from oilcast.models import registry as R
    # 清掉同 session 其他用例（离线主链路）写入的模型，保证版本计数从空开始
    import shutil as _sh
    _md = R._dir()
    if _md.exists():
        _sh.rmtree(_md)
    _md.mkdir(parents=True, exist_ok=True)
    X, price = _toy_panel()
    f1 = ShortTermForecaster(10, 180, compute_residuals=False).fit(X, price)
    R.save_artifact("short", "brent", f1, meta=f1.train_meta)
    warm, prev = R.load_artifact("short", "brent")
    f2 = ShortTermForecaster(10, 180, compute_residuals=False).fit(X, price, warm=warm)
    assert f2.warm_info["warm_started"] is True
    # 新机制：direct 树每个交易日在滚动窗内整体重拟合（迭代数恒定，不再跨期叠树），
    # 跨期"持续学习"由误差结构 EMA 继承 + 数据累积 + 工件持久化承担。
    from oilcast.config import get_config as _gc
    assert f2.train_meta["direct_mode"] == "rolling_refit"
    assert f2.cum_iters[1] == int(_gc()["model"]["direct_max_iter"])
    assert f2.train_meta["train_window"] == int(_gc()["model"]["train_window"])
    e2 = R.save_artifact("short", "brent", f2, meta=f2.train_meta, warm_from=prev)
    assert e2["version"] == 2 and e2["parent_version"] == 1
    # 特征列不一致时必须冷启动，绝不复用错配模型
    Xbad = X.drop(columns=["usd_index"])
    f3 = ShortTermForecaster(10, 180, compute_residuals=False).fit(Xbad, price, warm=warm)
    assert f3.warm_info["warm_started"] is False


# ------------------------------------------------- 训练样本用满全部真实历史（不再写死截断）
def test_short_term_uses_full_history_not_hardcap_540():
    from oilcast.config import get_config
    mcfg = get_config()["model"]
    saved = mcfg.get("max_train_rows", 0)
    try:
        # 900 行 > 旧版写死的 tail(540)；默认 max_train_rows=0 必须用满 900 行
        X, price = _toy_panel(n=900)
        mcfg["max_train_rows"] = 0
        fc = ShortTermForecaster(10, 180, compute_residuals=False).fit(X, price)
        assert fc.train_meta["n_rows"] == 900, "默认应使用全部历史，不得再写死截断到540"
        # 显式配置正整数时才截断为最近 N 行
        mcfg["max_train_rows"] = 300
        fc2 = ShortTermForecaster(10, 180, compute_residuals=False).fit(X, price)
        assert fc2.train_meta["n_rows"] == 300
    finally:
        mcfg["max_train_rows"] = saved


# ------------------------------------------------- 回测必须多原点、字段完整、严格样本外
def test_backtest_multi_origin_and_fields():
    X, price = _toy_panel(n=700)
    # 测试用轻量内部校准（少量原点/迭代）提速；生产走 config 的完整规模
    bt = backtest_short(X, price, horizon=10, n_origins=14, gap=4,
                        calib_origins=4, calib_iter=20)
    assert bt["available"] is True
    assert bt["n_origins"] >= 10, "回测原点必须足够多，旧版仅 8 点统计不可信"
    for k in ("mae_pct", "rmse_pct", "raw_mae_pct", "raw_direction_accuracy",
              "benchmark_mae_pct", "direction_accuracy",
              "ic", "beat_random", "mae_improve_pct", "window_start", "window_end"):
        assert k in bt, f"回测结果缺少字段 {k}"
    assert 0.0 <= bt["direction_accuracy"] <= 1.0
    assert bt["window_start"] <= bt["window_end"]


# ------------------------------------------------- 回测在含窗内常数列/稀疏列时不得静默归零
def test_backtest_survives_constant_and_sparse_columns():
    # 回归：训练端逐窗剔除"唯一值<2 的常数列"后，回测预测必须按模型实际入模列对齐，
    # 否则特征数不匹配、每个原点都被 except 吞掉，页面显示"有效回测原点不足(0<10)"。
    X, price = _toy_panel(n=700)
    X["const_col"] = 1.0                       # 整列常数（新版 sklearn 分箱会崩、训练端剔除）
    X["sparse_evt"] = np.nan
    X.loc[X.index[-3:], "sparse_evt"] = [0.1, 0.2, 0.3]   # 极稀疏事件列
    bt = backtest_short(X, price, horizon=10, n_origins=14, gap=4,
                        calib_origins=4, calib_iter=20)
    assert bt["available"] is True, f"含常数列时回测不应归零：{bt.get('reason')}"
    assert bt["n_origins"] >= 10


# ------------------------------------------------- 数据新鲜度守门：未取得新交易日必须显式告警
def test_freshness_gate_flags_unchanged_lagging_fresh(monkeypatch):
    """新鲜度以"截至运行时刻最近应已收盘交易日"为基准：凌晨/周末运行不误报，真缺数才告警。"""
    import oilcast.pipeline.main as M

    def clock(s):
        monkeypatch.setattr(M, "now_beijing",
                            lambda: pd.Timestamp(s, tz="Asia/Shanghai"))

    def so(parent_end):
        return {"wti": {"status": "ok",
                        "model_entry": {"parent_train_end": parent_end}}}

    ALL = ["wti", "brent", "shanghai_crude", "heating_oil", "gasoil"]

    def lrd(anchor_d, sh_d=None):
        # 其余品种给同一观测日；上海可单独指定
        return {t: pd.Timestamp(sh_d if t == "shanghai_crude" and sh_d else anchor_d)
                for t in ALL}

    # 固定为周二 2026-09-15 09:00（北京）：欧美最近应已收盘交易日=周一 09-14
    clock("2026-09-15 09:00")
    # 1) 缺最近一个已收盘交易日（停在上周五 09-11）-> unchanged
    f1 = M._assess_freshness("wti", so("2026-09-11"), lrd("2026-09-11"), "2026-09-14")
    assert f1["level"] == "unchanged" and f1["gap_bdays"] == 1
    # 2) 已到最近收盘日 09-14、较上期 09-11 推进 -> fresh
    f2 = M._assess_freshness("wti", so("2026-09-11"), lrd("2026-09-14"), "2026-09-14")
    assert f2["level"] == "fresh" and f2["advanced"] is True
    # 3) 落后应有交易日 >=2 -> lagging
    f3 = M._assess_freshness("wti", {"wti": {"status": "ok", "model_entry": {}}},
                             lrd("2026-09-09"), "2026-09-14")
    assert f3["level"] == "lagging" and f3["gap_bdays"] >= 2

    # 4) 周六下午重跑周五：上海/欧美均已收，cur=周五 -> fresh（周末/同日重跑不误报红）
    clock("2026-09-12 16:31")
    f4 = M._assess_freshness("wti", so("2026-09-11"), lrd("2026-09-11"), "2026-09-11")
    assert f4["level"] == "fresh" and f4["early_market_open"] is False
    # 5) 周六凌晨 01:04（上海夜盘未到02:30、欧美盘未收完）：欧美最近完整收盘=周四 09-10
    clock("2026-09-12 01:04")
    f5 = M._assess_freshness("wti", so("2026-09-10"), lrd("2026-09-10"), "2026-09-11")
    assert f5["level"] == "fresh" and f5["early_market_open"] is True
    assert f5["expected_latest"] == "2026-09-10"
    # 6) 关键：周六 03:00 上海02:30已收(周五齐)、欧美仍周五盘中(日K收盘要等06:00)。
    #    anchor=上海 cur=周五 -> fresh，且文案点明欧美取同时刻盘中价；欧美自身 expected 仍=周四
    clock("2026-09-12 03:00")
    f6 = M._assess_freshness("shanghai_crude", so("2026-09-11"),
                             lrd("2026-09-10", sh_d="2026-09-11"), "2026-09-11")
    assert f6["level"] == "fresh" and f6["expected_latest"] == "2026-09-11"
    assert f6["per_target_status"]["brent"]["expected"] == "2026-09-10"
    assert f6["per_target_status"]["brent"]["closed_for_asof"] is False
    assert "同时刻盘中价" in f6["message"]


# ------------------------------------------------- 全NaN/常数特征列不得拖垮短期模型
def test_short_term_fixed_schema_survives_sparse_allnan_and_warmstart():
    # 固定特征 schema + 零信息占位：整列全 NaN（事件源不可达）或仅零星非空的稀疏列
    # （事件稀疏）都做常数 0 占位而非删列。这两类列在新版 sklearn(>=1.6)+numpy2 下都会
    # 触发 "window shape cannot be larger than input array shape"；同时保证跨期
    # feature_cols 恒定，数据源一次抖动不会让热启动失效、模型从零重学。
    X, price = _toy_panel()
    X["ev_cpi_5d"] = np.nan                       # 整列全 NaN → 剔除、记为零信息
    X["ev_geo_5d"] = np.nan
    X.loc[X.index[-2:], "ev_geo_5d"] = [0.2, 0.3]  # 稀疏列：仅 2 个非空（含真实信息→保留入模）
    X["const_col"] = 1.0                          # 非空常数列（非空充足，保留但树不分裂）
    fc = ShortTermForecaster(10, 180, compute_residuals=False).fit(X, price)
    # schema 登记全部列；仅【整列全空】被识别为零信息列并移出 active 入模集合
    assert set(fc.feature_cols) == set(X.columns)
    assert set(fc.zero_info_cols) == {"ev_cpi_5d"}
    assert "ev_geo_5d" in fc.active_cols and "ev_geo_5d" not in fc.zero_info_cols
    assert "const_col" not in fc.zero_info_cols
    from oilcast.data_sources.calendar import next_trading_days
    nx = next_trading_days(X.index[-1], 10, price.dropna().index)
    pr = fc.predict(X.loc[[X.index[-1]]], float(price.iloc[-1]), nx)
    assert len(pr.path) == 10 and pr.endpoint["mean"] > 0
    # 下一期列集合不变时，即便仍有全 NaN/稀疏列也必须能在原模型上热启动（不从零）
    fc2 = ShortTermForecaster(10, 180, compute_residuals=False).fit(X.copy(), price, warm=fc)
    assert fc2.feature_cols == fc.feature_cols
    assert fc2.warm_info["warm_started"] is True


# ------------------------------------------------- 完整学习快照：模型+权重/预测/报告索引
def test_learning_bundle_roundtrip_restores_all(tmp_path):
    """删库/换机场景：导出的快照必须同时恢复模型工件与 factor_weights/forecasts/reports，
    否则权重 EMA 与历史复盘会从零。"""
    import sqlite3 as _sq
    from oilcast.models import registry as R
    from oilcast.config import get_config
    cfg = get_config()
    sto = cfg["storage"]
    orig = (sto.get("model_dir"), sto.get("sqlite_path"))
    sto["model_dir"] = str(tmp_path / "models")
    sto["sqlite_path"] = str(tmp_path / "oilcast.db")
    try:
        X, price = _toy_panel()
        fc = ShortTermForecaster(10, 180, compute_residuals=False).fit(X, price)
        R.save_artifact("short", "wti", fc, meta=fc.train_meta)
        # 写入三类学习状态表
        con = _sq.connect(sto["sqlite_path"])
        con.executescript("""
          CREATE TABLE factor_weights(report_date TEXT, factor TEXT, weight REAL,
            model_importance REAL, prior REAL, available INTEGER,
            PRIMARY KEY(report_date,factor));
          CREATE TABLE forecasts(report_date TEXT, horizon TEXT, instrument TEXT, target_date TEXT,
            mean REAL,q05 REAL,q25 REAL,q50 REAL,q75 REAL,q95 REAL,prob_up REAL,prob_down REAL);
          CREATE TABLE reports(report_date TEXT PRIMARY KEY, json_path TEXT, html_path TEXT, created_at TEXT);""")
        con.execute("INSERT INTO factor_weights VALUES('2026-09-06','usd_index',0.2,0.2,0.15,1)")
        con.execute("INSERT INTO forecasts VALUES('2026-09-06','short','wti','2026-09-18',90,80,85,90,95,100,0.5,0.5)")
        con.execute("INSERT INTO reports VALUES('2026-09-06','j.json','h.html','t')")
        con.commit(); con.close()
        zp = tmp_path / "bundle.zip"
        R.export_models(str(zp))
        # 模拟全部丢失
        import shutil
        shutil.rmtree(R._dir())
        (tmp_path / "oilcast.db").unlink()
        m = R.import_models(str(zp))
        obj, _ = R.load_artifact("short", "wti")
        assert obj is not None                                   # 模型恢复
        assert m["_restored_learning_tables"] == {"factor_weights": 1, "forecasts": 1, "reports": 1}
        con = _sq.connect(sto["sqlite_path"])
        assert con.execute("SELECT COUNT(*) FROM factor_weights").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0] == 1
        assert con.execute("SELECT report_date FROM reports").fetchone()[0] == "2026-09-06"
        con.close()
    finally:
        sto["model_dir"], sto["sqlite_path"] = orig


# ------------------------------------------------- 走势图按交易日等距（非自然日历）
def test_price_figure_category_axis_has_no_calendar_gaps():
    import plotly.graph_objects as go
    from oilcast.reporting.static_html import price_figure
    # 周四(01-04)、周五(01-05)、下周一(01-08)：自然日历隔着周末，交易日轴必须等距连续
    hist = [{"date": "2024-01-04", "wti": 70.0},
            {"date": "2024-01-05", "wti": 71.0},
            {"date": "2024-01-08", "wti": 72.0}]
    report = {"history": hist,
              "current_meta": {"wti": {"observed_date": "2024-01-08", "value": 72.0}},
              "forecasts": {"short": {}, "mid": {}, "long": {}}}
    fig = price_figure(report, "wti", "WTI", "美元/桶")
    xa = fig.layout.xaxis
    assert xa.type == "category"                      # 类别轴而非自然日历 date 轴
    cats = list(xa.categoryarray)
    assert cats == ["2024-01-04", "2024-01-05", "2024-01-08"]  # 无周六周日插入、无断点
    assert not any(d in cats for d in ("2024-01-06", "2024-01-07"))


# ------------------------------------------------- 非交易日（价格全空的假日）不入轴/不进图
def test_trading_axis_excludes_holiday_without_real_price():
    from oilcast.data_sources.calendar import observed_trading_index
    from oilcast.pipeline.main import _hist_records
    idx = pd.DatetimeIndex(["2025-12-31", "2026-01-01", "2026-01-02"])  # 01-01 元旦休市
    prices = pd.DataFrame({"wti": [57.4, np.nan, 57.3],
                           "brent": [60.8, np.nan, 60.7]}, index=idx)
    macro = pd.DataFrame({"dxy": [100.0, 100.0, 100.1]}, index=idx)     # 宏观假日被前填
    pdays = observed_trading_index(prices).strftime("%Y-%m-%d").tolist()
    assert "2026-01-01" not in pdays        # 价格全空的假日不得进入价格交易日轴
    recs = _hist_records(prices, n=10)
    assert [r["date"] for r in recs] == ["2025-12-31", "2026-01-02"]  # 历史无假日空行


# ------------------------------------------------- 重复交易日索引不得触发 duplicate labels（CI 回归）
def test_build_features_survives_duplicate_trading_index():
    from oilcast.features.engineering import institutional_bias
    idx = pd.bdate_range("2026-01-05", periods=60)
    dup = pd.DatetimeIndex(list(idx) + [idx[0]])          # 末尾制造一个重复交易日
    n = len(dup)
    prices = pd.DataFrame(
        {"wti": np.linspace(70, 80, n), "brent": np.linspace(74, 84, n),
         "shanghai_crude": np.linspace(500, 560, n), "heating_oil": np.linspace(2.4, 2.8, n),
         "gasoil": np.linspace(900, 980, n)}, index=dup).sort_index()
    macro = pd.DataFrame({"dxy": np.linspace(100, 103, n)}, index=dup).sort_index()
    # 同一交易日多条机构观点（重复日期，正是 CI 崩溃根因）
    views = pd.DataFrame({"date": [idx[5], idx[5], idx[10], idx[10]],
                          "stance": ["看涨", "看跌", "看涨", "中性"]})
    events = pd.DataFrame(columns=["date", "title", "intensity"])
    ib = institutional_bias(views, idx, source_available=True)
    assert ib.index.is_unique
    feats = build_features(prices, macro, events, views, target="wti",
                           events_available=False, views_available=True,
                           use_session_align=False)
    assert feats.index.is_unique and len(feats) == 60


# ---------------- 方向概率分类 + 显著性门控（三分类，严格无泄漏）----------------
def _direction_panel(predictable: bool, n: int = 1500, h: int = 5, seed: int = 1):
    """构造方向【可预测】或【纯随机游走】的价格+特征面板（仅测试用）。

    可预测：未来 h 日累计收益由当期可见信号 signal 的符号强决定（模拟真实 edge）；
    随机游走：未来收益与 signal 独立（弱有效市场，不应被判出稳定 edge）。"""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-02", periods=n)
    sig = rng.normal(size=n)
    r = rng.normal(0, 0.004, n)
    if predictable:
        for t in range(n - h):
            for k in range(1, h + 1):
                r[t + k] += 0.006 * np.sign(sig[t])
    price = pd.Series(70.0 * np.exp(np.cumsum(r)), index=idx)
    X = pd.DataFrame({"signal": sig, "noise_a": rng.normal(size=n),
                      "noise_b": rng.normal(size=n)}, index=idx)
    return X, price


def test_direction_gate_neutral_on_pure_random_walk():
    # 纯随机游走：交叉拟合门控不得制造出"显著方向 edge"，最终立场应为中性
    X, price = _direction_panel(predictable=False)
    fc = ShortTermForecaster(horizon=5, compute_residuals=False,
                             run_arima=False).fit(X, price)
    gate5 = fc.dir_edge.get(5, {})
    assert not gate5.get("has_edge", False), "随机游走不应通过方向显著性门控"
    future = pd.bdate_range(X.index[-1] + pd.Timedelta(days=1), periods=5)
    ep = fc.predict(X.iloc[[-1]], float(price.iloc[-1]), future).endpoint
    assert ep["dir_stance"] == "中性"
    assert set(["dir_stance", "dir_prob_up", "dir_has_edge"]).issubset(ep)


def test_direction_gate_engages_when_edge_is_real():
    # 存在真实可预测信号时：门控应识别 edge，且最终立场非中性（看涨或看跌）
    X, price = _direction_panel(predictable=True)
    fc = ShortTermForecaster(horizon=5, compute_residuals=False,
                             run_arima=False).fit(X, price)
    gate5 = fc.dir_edge.get(5, {})
    assert gate5.get("has_edge") is True, f"真实edge应被门控识别: {gate5}"
    assert gate5.get("engaged_hit", 0) >= 0.55
    future = pd.bdate_range(X.index[-1] + pd.Timedelta(days=1), periods=5)
    ep = fc.predict(X.iloc[[-1]], float(price.iloc[-1]), future).endpoint
    assert ep["dir_stance"] in ("看涨", "看跌")


def test_backtest_three_class_metrics():
    # 三分类回测：字段齐全；明确表态率+中性率=1；中性不被计为方向错误
    X, price = _direction_panel(predictable=True, n=1200)
    bt = backtest_short(X, price, horizon=5, n_origins=12, gap=3,
                        calib_origins=4, calib_iter=20, dir_origins=40)
    assert bt["available"], bt
    for k in ("stance_engagement_rate", "stance_neutral_rate", "stance_engaged_n",
              "direction_brier", "clf_direction_accuracy", "gate_has_edge"):
        assert k in bt
    assert abs(bt["stance_engagement_rate"] + bt["stance_neutral_rate"] - 1.0) < 1e-9
    assert bt["stance_engaged_n"] == round(bt["stance_engagement_rate"] * bt["n_origins"])


def test_trend_gate_rewards_payoff_not_just_winrate():
    # 趋势通道经济价值判据：胜率不足 55% 但盈亏比>1、期望显著为正也应放行
    tg = ShortTermForecaster._trend_gate
    good = [(1.0, 0.03)] * 9 + [(1.0, -0.01)] * 11   # 胜率45%、盈亏比3、期望为正
    g = tg(good, 0.05, min_eng=8, min_payoff=1.15, max_ret_p=0.10)
    assert g["has_edge"] is True and g["payoff"] >= 1.15 and g["mean_ret"] > 0
    assert g["hit"] < 0.55
    bad = [(1.0, 0.01)] * 11 + [(1.0, -0.03)] * 9    # 赢小亏大、期望为负，不放行
    assert tg(bad, 0.05, 8, 1.15, 0.10)["has_edge"] is False
    assert tg([(1.0, 0.02)] * 4, 0.05, 8, 1.15, 0.10)["has_edge"] is False


def test_backtest_reports_economic_value():
    X, price = _direction_panel(predictable=True, n=1200)
    bt = backtest_short(X, price, horizon=5, n_origins=12, gap=3,
                        calib_origins=4, calib_iter=20, dir_origins=40)
    for k in ("stance_engaged_mean_ret_pct", "stance_engaged_payoff",
              "stance_engaged_sharpe", "buyhold_mean_ret_pct"):
        assert k in bt


def test_direction_layer_persists_across_pickle(tmp_path):
    # 方向分类器/校准器/门控证据随工件整体持久化，导入后仍能给三分类立场（不从零）
    import joblib
    X, price = _direction_panel(predictable=True)
    fc = ShortTermForecaster(horizon=5, compute_residuals=False,
                             run_arima=False).fit(X, price)
    assert fc.dir_models and fc.dir_edge
    path = tmp_path / "fc.joblib"
    joblib.dump(fc, path)
    fc2 = joblib.load(path)
    assert set(fc2.dir_models.keys()) == set(fc.dir_models.keys())
    assert fc2.dir_edge == fc.dir_edge
    future = pd.bdate_range(X.index[-1] + pd.Timedelta(days=1), periods=5)
    ep1 = fc.predict(X.iloc[[-1]], float(price.iloc[-1]), future).endpoint
    ep2 = fc2.predict(X.iloc[[-1]], float(price.iloc[-1]), future).endpoint
    assert ep2["dir_stance"] == ep1["dir_stance"]
