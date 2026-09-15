"""SQLite 持久化层：原始数据、谱系、事件、预测、权重、报告索引。

可追溯设计：
- raw_prices/raw_macro 每行带 mode（固定 strict，真实观测）；
- data_lineage 记录每个字段每次报告的来源、抓取时刻、首末观测、质量门状态；
- 所有写入 upsert，重复执行同一天不产生重复行。
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from ..config import get_config, ensure_dirs

SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_prices (
    date TEXT, mode TEXT, wti REAL, brent REAL, shanghai_crude REAL,
    heating_oil REAL, gasoil REAL, PRIMARY KEY(date, mode));
CREATE TABLE IF NOT EXISTS raw_macro (
    date TEXT, mode TEXT, dxy REAL, us10y REAL, us2y REAL, cpi_yoy REAL,
    nonfarm_surprise REAL, fedfunds REAL, fed_expectation REAL,
    demand_proxy REAL, gpr_index REAL, usdcny REAL, usdjpy REAL,
    PRIMARY KEY(date, mode));
CREATE TABLE IF NOT EXISTS data_lineage (
    report_date TEXT, field TEXT, display TEXT, status TEXT, source_name TEXT,
    url TEXT, frequency TEXT, retrieved_at TEXT, n_obs INTEGER,
    first_observed TEXT, last_observed TEXT, stale INTEGER, mode TEXT, note TEXT,
    tried_sources TEXT, PRIMARY KEY(report_date, field));
CREATE TABLE IF NOT EXISTS events (
    date TEXT, title TEXT, source TEXT, theme TEXT, sentiment TEXT,
    intensity REAL, est_price_impact REAL, url TEXT, UNIQUE(date, title));
CREATE TABLE IF NOT EXISTS institutional_views (
    date TEXT, institution TEXT, target_wti REAL, stance TEXT, note TEXT,
    UNIQUE(date, note));
CREATE TABLE IF NOT EXISTS forecasts (
    report_date TEXT, horizon TEXT, instrument TEXT, target_date TEXT,
    mean REAL, q05 REAL, q25 REAL, q50 REAL, q75 REAL, q95 REAL,
    prob_up REAL, prob_down REAL, dir_stance TEXT);
CREATE TABLE IF NOT EXISTS factor_weights (
    report_date TEXT, instrument TEXT, factor TEXT, weight REAL, model_importance REAL,
    prior REAL, available INTEGER, PRIMARY KEY(report_date, instrument, factor));
CREATE TABLE IF NOT EXISTS reports (
    report_date TEXT PRIMARY KEY, json_path TEXT, html_path TEXT, created_at TEXT);
-- 分时（盘中）K 线：用于跨市场"同一真实时刻"对齐。ts 为北京时间字符串(YYYY-mm-dd HH:MM:SS)，
-- 每根带来源，upsert 主键(symbol,ts)；每日增量积累，同时刻面板随时间越来越长，绝不回填造数。
CREATE TABLE IF NOT EXISTS raw_intraday (
    symbol TEXT, ts TEXT, open REAL, high REAL, low REAL, close REAL,
    volume REAL, source TEXT, PRIMARY KEY(symbol, ts));
"""


class OilCastDB:
    def __init__(self, path: Optional[str] = None) -> None:
        ensure_dirs()
        cfg = get_config()
        self.path = path or cfg["storage"]["sqlite_path"]
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as con:
            con.executescript(SCHEMA)
            self._migrate_factor_weights(con)
            self._migrate_add_columns(con)

    @staticmethod
    def _migrate_add_columns(con) -> None:
        """旧宽表补新增列（不涉及主键，ADD COLUMN 即可）。"""
        add = {"raw_prices": [("shanghai_crude", "REAL")],
               "raw_macro": [("usdcny", "REAL"), ("usdjpy", "REAL")],
               "forecasts": [("dir_stance", "TEXT")]}
        for table, cols in add.items():
            existing = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
            for name, typ in cols:
                if name not in existing:
                    con.execute(f"ALTER TABLE {table} ADD COLUMN {name} {typ}")

    @staticmethod
    def _migrate_factor_weights(con) -> None:
        """把旧版 factor_weights 迁移到 (report_date,instrument,factor) 复合主键。

        SQLite 的 ALTER ADD COLUMN 改不了既有主键，因此检测到旧表（缺 instrument
        列，或建表 SQL 的主键未含 instrument）时，用"建新表→搬数据（历史单套权重
        归主锚 wti）→换表"重建，使分品种权重可同日多条而不撞唯一约束。"""
        info = con.execute("PRAGMA table_info(factor_weights)").fetchall()
        cols = [r[1] for r in info]
        # PRAGMA 第 6 列为主键序号(>0 即主键成员)，据此判断主键是否已含 instrument
        pk_cols = {r[1] for r in info if r[5] > 0}
        need_rebuild = (pk_cols != {"report_date", "instrument", "factor"})
        if not need_rebuild:
            return
        con.execute(
            """CREATE TABLE factor_weights_new (
                report_date TEXT, instrument TEXT, factor TEXT, weight REAL,
                model_importance REAL, prior REAL, available INTEGER,
                PRIMARY KEY(report_date, instrument, factor))""")
        if "instrument" in cols:
            con.execute("""INSERT OR REPLACE INTO factor_weights_new
                SELECT report_date, instrument, factor, weight, model_importance,
                       prior, available FROM factor_weights""")
        else:
            con.execute("""INSERT OR REPLACE INTO factor_weights_new
                SELECT report_date, 'wti', factor, weight, model_importance,
                       prior, available FROM factor_weights""")
        con.execute("DROP TABLE factor_weights")
        con.execute("ALTER TABLE factor_weights_new RENAME TO factor_weights")

    @contextmanager
    def _conn(self):
        con = sqlite3.connect(self.path)
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def _upsert_df(self, df: pd.DataFrame, table: str, mode: str, keep_cols) -> None:
        if df is None or df.empty:
            return
        cols = [c for c in keep_cols if c in df.columns]
        out = df[cols].copy()
        out.insert(0, "mode", mode)
        out.index.name = "date"
        out = out.reset_index()
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
        placeholders = ",".join("?" * len(out.columns))
        sql = f"INSERT OR REPLACE INTO {table} ({','.join(out.columns)}) VALUES ({placeholders})"
        with self._conn() as con:
            con.executemany(sql, out.where(pd.notna(out), None).values.tolist())

    def save_prices(self, prices: pd.DataFrame, mode: str = "strict",
                    lineage: Optional[dict] = None) -> None:
        self._upsert_df(prices, "raw_prices", mode,
                        ["wti", "brent", "shanghai_crude", "heating_oil", "gasoil"])

    def save_macro(self, macro: pd.DataFrame, mode: str = "strict",
                   lineage: Optional[dict] = None) -> None:
        self._upsert_df(macro, "raw_macro", mode,
                        ["dxy", "us10y", "us2y", "cpi_yoy", "nonfarm_surprise",
                         "fedfunds", "fed_expectation", "demand_proxy", "gpr_index",
                         "usdcny", "usdjpy"])

    def save_intraday(self, df: pd.DataFrame) -> int:
        """upsert 分时 K 线。df 需含列 symbol,ts(北京时),open/high/low/close[,volume,source]。
        返回写入行数；空表返回 0。重复 (symbol,ts) 以新值覆盖，重复采集不产生重复行。"""
        if df is None or len(df) == 0:
            return 0
        rows = []
        for _, r in df.iterrows():
            ts = pd.Timestamp(r["ts"]).strftime("%Y-%m-%d %H:%M:%S")
            def _num(k):
                v = r.get(k)
                return None if v is None or pd.isna(v) else float(v)
            rows.append((str(r["symbol"]), ts, _num("open"), _num("high"), _num("low"),
                         _num("close"), _num("volume"), str(r.get("source", ""))))
        with self._conn() as con:
            con.executemany(
                "INSERT OR REPLACE INTO raw_intraday VALUES (?,?,?,?,?,?,?,?)", rows)
        return len(rows)

    def read_intraday(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """读分时 K，返回按 (symbol, 北京时 ts) 排序的长表。"""
        q = "SELECT * FROM raw_intraday"
        params = ()
        if symbol:
            q += " WHERE symbol=?"
            params = (symbol,)
        q += " ORDER BY symbol, ts"
        with self._conn() as con:
            df = pd.read_sql_query(q, con, params=params)
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"])
        return df

    def save_lineage(self, report_date: str, lineage: Dict[str, dict], mode: str) -> None:
        with self._conn() as con:
            con.execute("DELETE FROM data_lineage WHERE report_date=?", (report_date,))
            con.executemany(
                "INSERT INTO data_lineage VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(report_date, f, L.get("display", ""), L.get("status", ""),
                  L.get("source_name", ""), L.get("url", ""), L.get("frequency", ""),
                  L.get("retrieved_at", ""), int(L.get("n_obs", 0) or 0),
                  L.get("first_observed"), L.get("last_observed"),
                  L.get("stale"), mode, L.get("note", ""), L.get("tried_sources", ""))
                 for f, L in lineage.items()])

    def save_events(self, events: pd.DataFrame) -> None:
        if events is None or events.empty:
            return
        with self._conn() as con:
            for _, r in events.iterrows():
                con.execute("INSERT OR IGNORE INTO events VALUES (?,?,?,?,?,?,?,?)",
                            (pd.Timestamp(r["date"]).strftime("%Y-%m-%d"), str(r.get("title", "")),
                             str(r.get("source", "")), str(r.get("theme", "")),
                             str(r.get("sentiment", "")), float(r.get("intensity", 0) or 0),
                             float(r.get("est_price_impact", 0) or 0), str(r.get("url", ""))))

    def save_views(self, views: pd.DataFrame) -> None:
        if views is None or views.empty:
            return
        with self._conn() as con:
            for _, r in views.iterrows():
                tgt = r.get("target_wti")
                con.execute("INSERT OR IGNORE INTO institutional_views VALUES (?,?,?,?,?)",
                            (pd.Timestamp(r["date"]).strftime("%Y-%m-%d"),
                             str(r.get("institution", "")),
                             None if pd.isna(tgt) else float(tgt),
                             str(r.get("stance", "")), str(r.get("note", ""))))

    def save_forecasts(self, report_date: str, records: list[dict]) -> None:
        with self._conn() as con:
            con.execute("DELETE FROM forecasts WHERE report_date=?", (report_date,))
            con.executemany(
                "INSERT INTO forecasts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(report_date, r["horizon"], r["instrument"], r["target_date"],
                  r["mean"], r["q05"], r["q25"], r["q50"], r["q75"], r["q95"],
                  r.get("prob_up"), r.get("prob_down"), r.get("dir_stance"))
                 for r in records])

    def save_weights(self, report_date: str, weights: pd.DataFrame,
                     instrument: str = "wti") -> None:
        with self._conn() as con:
            con.execute("DELETE FROM factor_weights WHERE report_date=? AND instrument=?",
                        (report_date, instrument))
            for _, r in weights.iterrows():
                w = r.get("weight")
                mi = r.get("model_importance")
                pr = r.get("prior")
                con.execute("INSERT INTO factor_weights VALUES (?,?,?,?,?,?,?)",
                            (report_date, instrument, r["factor"],
                             None if pd.isna(w) else float(w),
                             None if pd.isna(mi) else float(mi),
                             None if pd.isna(pr) else float(pr),
                             1 if r.get("available", False) else 0))

    def latest_weights(self, instrument: str, before_date: str) -> Optional[pd.DataFrame]:
        """取某品种在 before_date 之前最近一期的因素权重（供跨期 EMA 热启动）。"""
        with self._conn() as con:
            row = con.execute(
                "SELECT MAX(report_date) FROM factor_weights WHERE instrument=? AND report_date<?",
                (instrument, before_date)).fetchone()
        if not row or not row[0]:
            return None
        with self._conn() as con:
            df = pd.read_sql_query(
                "SELECT factor, weight FROM factor_weights WHERE instrument=? AND report_date=?",
                con, params=(instrument, row[0]))
        if df.empty:
            return None
        return df.set_index("factor")["weight"]

    def register_report(self, report_date: str, json_path: str, html_path: str) -> None:
        with self._conn() as con:
            con.execute("INSERT OR REPLACE INTO reports VALUES (?,?,?,?)",
                        (report_date, json_path, html_path,
                         datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    def read_table(self, table: str) -> pd.DataFrame:
        with self._conn() as con:
            df = pd.read_sql_query(f"SELECT * FROM {table} ORDER BY date", con)
        return df.set_index("date") if "date" in df.columns else df

    def list_report_dates(self) -> list[str]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT report_date FROM reports ORDER BY report_date DESC").fetchall()
        return [r[0] for r in rows]

    def weight_history(self, factor: str, instrument: str = "wti") -> pd.DataFrame:
        with self._conn() as con:
            return pd.read_sql_query(
                "SELECT report_date, weight FROM factor_weights "
                "WHERE factor=? AND instrument=? ORDER BY report_date",
                con, params=(factor, instrument))


def save_csv_snapshot(prices: pd.DataFrame, macro: pd.DataFrame, tag: str) -> Dict[str, str]:
    cfg = get_config()
    raw = Path(cfg["storage"]["raw_dir"])
    raw.mkdir(parents=True, exist_ok=True)
    paths = {}
    p1, p2 = raw / f"prices_{tag}.csv", raw / f"macro_{tag}.csv"
    prices.to_csv(p1, encoding="utf-8-sig")
    macro.to_csv(p2, encoding="utf-8-sig")
    paths["prices"], paths["macro"] = str(p1), str(p2)
    return paths
