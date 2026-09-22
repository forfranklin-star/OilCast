"""模型工件持久化、版本谱系与导入/导出。

目的：让预测模型**跨期、跨进程重启持续学习，而不是每次运行都从 0 开始**。
每期训练后把模型对象（短期 direct 梯度提升、样本外残差、ARIMA 参数等）以
joblib 落盘到 storage.model_dir，并在 manifest.json 记录版本链（parent_version）、
训练区间、样本行数、特征列、是否由上期热启动。下一期运行时 load_artifact 取回，
在其基础上增量训练（见 short_term.ShortTermForecaster.fit 的 warm 参数）。

只持久化由真实数据训练得到的模型；不缓存、不合成任何价格/宏观数据。

CLI::
    python -m oilcast.models.registry list
    python -m oilcast.models.registry export oilcast_models.zip
    python -m oilcast.models.registry import oilcast_models.zip
"""
from __future__ import annotations

import io
import json
import sqlite3
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import pandas as pd

from ..config import get_config
from ..utils import get_logger, now_beijing

LOG = get_logger(__name__)
MANIFEST_NAME = "manifest.json"
# 除模型工件外，构成"持续学习状态"的数据库表：
#   factor_weights —— 因素权重的跨期 EMA（缺它，导入后权重会退回人工先验，等于从零）
#   forecasts      —— 历史预测，用于到期复测/预测复盘（缺它，回测复盘样本清零）
#   reports        —— 报告日期索引，权重续用靠它定位“上一期”（缺它，EMA 找不到上期）
LEARNING_TABLES = ("factor_weights", "forecasts", "reports")


def _dir() -> Path:
    p = Path(get_config()["storage"]["model_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def artifact_key(horizon: str, instrument: str) -> str:
    return f"{horizon}__{instrument}"


def _manifest_path() -> Path:
    return _dir() / MANIFEST_NAME


def load_manifest() -> dict:
    p = _manifest_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            LOG.warning("模型 manifest 读取失败，按空清单处理：%s", exc)
    return {"models": {}}


def _save_manifest(m: dict) -> None:
    m["updated_at"] = now_beijing().strftime("%Y-%m-%d %H:%M:%S")
    _manifest_path().write_text(json.dumps(m, ensure_ascii=False, indent=2, default=str),
                                encoding="utf-8")


def save_artifact(horizon: str, instrument: str, obj, meta: Optional[dict] = None,
                  warm_from: Optional[dict] = None) -> dict:
    """落盘一个模型工件并推进版本号，返回该模型本次的 manifest 条目。

    warm_from: 本次热启动所依据的上期条目（可为 None=冷启动）。
    """
    key = artifact_key(horizon, instrument)
    d = _dir()
    manifest = load_manifest()
    prev = manifest["models"].get(key, {})
    parent_version = int(prev.get("version", 0))
    version = parent_version + 1
    fname = f"{key}.joblib"
    path = d / fname
    tmp = d / (fname + ".tmp")
    joblib.dump(obj, tmp, compress=3)
    tmp.replace(path)
    entry = {
        "key": key, "horizon": horizon, "instrument": instrument,
        "file": fname, "version": version,
        "parent_version": parent_version if warm_from is not None else 0,
        "warm_started": bool(warm_from is not None),
        "saved_at": now_beijing().strftime("%Y-%m-%d %H:%M:%S"),
        "bytes": int(path.stat().st_size),
    }
    if warm_from is not None:
        entry["parent_train_end"] = warm_from.get("train_end")
        entry["parent_saved_at"] = warm_from.get("saved_at")
    if meta:
        for k in ("train_start", "train_end", "n_rows", "n_valid", "window",
                  "n_steps", "fitted_at", "cum_iters"):
            if k in meta:
                entry[k] = meta[k]
        if "feature_cols" in meta:
            entry["feature_cols"] = list(meta["feature_cols"])
    manifest["models"][key] = entry
    _save_manifest(manifest)
    LOG.info("模型工件已保存 %s v%d（%s，%d 字节）", key, version,
             "热启动" if warm_from is not None else "冷启动", entry["bytes"])
    return entry


def load_artifact(horizon: str, instrument: str) -> Tuple[object, Optional[dict]]:
    """取回上期模型对象与其 manifest 条目；不存在或损坏返回 (None, None)。"""
    key = artifact_key(horizon, instrument)
    manifest = load_manifest()
    entry = manifest["models"].get(key)
    if not entry:
        return None, None
    path = _dir() / entry.get("file", f"{key}.joblib")
    if not path.exists():
        LOG.warning("manifest 记录了 %s 但工件文件缺失：%s", key, path)
        return None, None
    try:
        return joblib.load(path), entry
    except Exception as exc:
        LOG.warning("模型工件 %s 加载失败，将冷启动：%s", key, exc)
        return None, None


def list_models() -> dict:
    return load_manifest().get("models", {})


def _db_path() -> Optional[Path]:
    try:
        return Path(get_config()["storage"]["sqlite_path"])
    except Exception:
        return None


def _ensure_learning_tables(conn: sqlite3.Connection) -> None:
    """导入到一个空库时，先把承载学习状态的表建出来（结构与 database.SCHEMA 一致）。"""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS forecasts (
            report_date TEXT, horizon TEXT, instrument TEXT, target_date TEXT,
            mean REAL, q05 REAL, q25 REAL, q50 REAL, q75 REAL, q95 REAL,
            prob_up REAL, prob_down REAL);
        CREATE TABLE IF NOT EXISTS factor_weights (
            report_date TEXT, factor TEXT, weight REAL, model_importance REAL,
            prior REAL, available INTEGER, PRIMARY KEY(report_date, factor));
        CREATE TABLE IF NOT EXISTS reports (
            report_date TEXT PRIMARY KEY, json_path TEXT, html_path TEXT, created_at TEXT);
        """)


def _dump_learning_tables() -> Dict[str, str]:
    """把权重/历史预测表读成 {表名: CSV文本}；库不存在或为空则返回 {}（不报错、不造数）。"""
    p = _db_path()
    out: Dict[str, str] = {}
    if p is None or not p.exists():
        return out
    try:
        with sqlite3.connect(p) as conn:
            for t in LEARNING_TABLES:
                try:
                    df = pd.read_sql_query(f"SELECT * FROM {t}", conn)
                except Exception:
                    continue
                if not df.empty:
                    buf = io.StringIO()
                    df.to_csv(buf, index=False)
                    out[t] = buf.getvalue()
    except Exception as exc:
        LOG.warning("导出学习状态表失败（仅导出模型工件）：%s", exc)
    return out


def _restore_learning_tables(zf: zipfile.ZipFile) -> dict:
    """把 zip 内 learning/<table>.csv 幂等写回数据库，返回各表写入行数。"""
    p = _db_path()
    counts: Dict[str, int] = {}
    if p is None:
        return counts
    p.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(p) as conn:
        _ensure_learning_tables(conn)
        for t in LEARNING_TABLES:
            arc = f"learning/{t}.csv"
            if arc not in zf.namelist():
                continue
            try:
                df = pd.read_csv(io.BytesIO(zf.read(arc)))
                if df.empty:
                    continue
                # 幂等：先移除将被覆盖的报告日，再整体写入，避免重复堆叠
                dates = sorted(df["report_date"].dropna().astype(str).unique())
                if dates:
                    qmarks = ",".join("?" * len(dates))
                    conn.execute(f"DELETE FROM {t} WHERE report_date IN ({qmarks})", dates)
                df.to_sql(t, conn, if_exists="append", index=False)
                counts[t] = len(df)
            except Exception as exc:
                LOG.warning("学习状态表 %s 恢复失败：%s", t, exc)
    return counts


def export_models(zip_path: str) -> Path:
    """导出【完整持续学习状态快照】为 zip：
    模型工件 *.joblib + manifest 版本链 + learning/ 下的因素权重历史与历史预测。
    这样即使模型目录与数据库同时丢失，导入此包即可在原进度上继续热启动，不从零。"""
    d = _dir()
    out = Path(zip_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    files = [p for p in d.glob("*.joblib")]
    mp = d / MANIFEST_NAME
    learning = _dump_learning_tables()
    if not files and not mp.exists() and not learning:
        raise FileNotFoundError(f"模型目录与学习状态均为空，无可导出内容：{d}")
    manifest = load_manifest()
    bundle_meta = {
        "exported_at": now_beijing().strftime("%Y-%m-%d %H:%M:%S"),
        "model_count": len(files),
        "models": {k: {"version": e.get("version"), "train_end": e.get("train_end"),
                       "warm_started": e.get("warm_started")}
                   for k, e in manifest.get("models", {}).items()},
        "learning_tables": {t: len(csv.splitlines()) - 1 for t, csv in learning.items()},
        "format": "oilcast-learning-bundle/1",
    }
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, arcname=p.name)
        if mp.exists():
            z.write(mp, arcname=MANIFEST_NAME)
        for t, csv_text in learning.items():
            z.writestr(f"learning/{t}.csv", csv_text)
        z.writestr("bundle_meta.json", json.dumps(bundle_meta, ensure_ascii=False, indent=2))
    LOG.info("已导出学习快照：%d 个模型工件 + 学习表 %s 到 %s",
             len(files), bundle_meta["learning_tables"], out)
    return out


def import_models(zip_path: str, replace: bool = True) -> dict:
    """从 export 产出的 zip 恢复模型工件。replace=True 时整体替换当前模型目录，
    False 时按文件合并（同名以导入包为准）。返回导入清单。"""
    src = Path(zip_path)
    if not src.exists():
        raise FileNotFoundError(src)
    d = _dir()
    with zipfile.ZipFile(src) as z:
        names = z.namelist()
        if MANIFEST_NAME not in names and not any(n.endswith(".joblib") for n in names):
            raise ValueError("压缩包内既无 manifest.json 也无 .joblib，不是有效的模型包")
        if replace:
            for p in d.glob("*.joblib"):
                p.unlink()
            mp = d / MANIFEST_NAME
            if mp.exists():
                mp.unlink()
        # 只解压模型工件/manifest 到模型目录，learning/*.csv 与 bundle_meta 留在包内稍后入库
        for member in z.namelist():
            if member.endswith(".joblib") or member == MANIFEST_NAME:
                z.extract(member, d)
        restored = _restore_learning_tables(z)
    manifest = load_manifest()
    n = len(manifest.get("models", {}))
    LOG.info("已从 %s 导入：%d 个模型 + 学习状态表 %s", src, n, restored)
    manifest["_restored_learning_tables"] = restored
    return manifest


# --------------------------------------------------------------- CLI
def _main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="OilCast 模型工件导入/导出/清单")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    pe = sub.add_parser("export"); pe.add_argument("zip_path")
    pi = sub.add_parser("import"); pi.add_argument("zip_path")
    pi.add_argument("--merge", action="store_true", help="合并而非整体替换")
    args = ap.parse_args()
    if args.cmd == "list":
        models = list_models()
        if not models:
            print("（当前无已保存模型工件）")
        for k, e in models.items():
            print(f"{k}\tv{e['version']}\t训练截止 {e.get('train_end','—')}\t"
                  f"{'热启动' if e.get('warm_started') else '冷启动'}\t{e.get('saved_at','—')}")
    elif args.cmd == "export":
        out = export_models(args.zip_path)
        print(f"已导出：{out}")
    elif args.cmd == "import":
        m = import_models(args.zip_path, replace=not args.merge)
        print(f"已导入，共 {len(m.get('models', {}))} 个模型；"
              f"恢复学习状态表 {m.get('_restored_learning_tables', {})}")


if __name__ == "__main__":
    _main()
