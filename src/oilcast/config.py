"""配置加载与项目路径管理。

用法::

    from oilcast.config import get_config, PROJECT_ROOT
    cfg = get_config()
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

# src/oilcast/config.py -> 上溯三级即代码仓库根；
# OILCAST_HOME 可覆盖数据/报告的落盘根目录（测试隔离、容器部署时使用），
# 配置文件本身始终跟随代码仓库，不被覆盖。
_CODE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _CODE_ROOT / "config" / "config.yaml"


def _project_root() -> Path:
    return Path(os.environ["OILCAST_HOME"]) if os.environ.get("OILCAST_HOME") else _CODE_ROOT


# 向后兼容的模块级常量（首次 import 时的值）
PROJECT_ROOT = _project_root()


def _deep_freeze(obj: Any) -> Any:
    """把 dict/list 递归转成不可变结构，防止运行期误改全局配置。"""
    if isinstance(obj, dict):
        return DictProxy({k: _deep_freeze(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return tuple(_deep_freeze(v) for v in obj)
    return obj


class DictProxy(dict):
    """支持属性访问的只读字典（cfg.model.rolling_window）。"""

    def __getattr__(self, item):  # noqa: D401
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        raise TypeError("config is read-only")


@lru_cache(maxsize=4)
def get_config(path: str | os.PathLike | None = None) -> DictProxy:
    """加载并缓存全局配置。"""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    # 测试/冒烟轻量档：OILCAST_FAST=1 时缩小所有嵌套重拟合规模，仅用于离线测试提速，
    # 生产/Actions 不设置该变量，始终使用 config.yaml 的完整规模。
    if os.environ.get("OILCAST_FAST"):
        m = raw.setdefault("model", {})
        m["direct_max_iter"] = min(int(m.get("direct_max_iter", 300)), 40)
        m["calib_origins"] = 4
        m["calib_iter"] = 20
        m["backtest_origins"] = 12
        m["resid_train_window"] = 300
        m["dir_gate_lookback"] = 200
        m["edge_recent_span"] = 120
        m["dir_clf_max_iter"] = 30

    cfg = _deep_freeze(raw)

    # 把配置中的相对路径统一解析为落盘根目录下的绝对路径（可由 OILCAST_HOME 覆盖）
    root = _project_root()
    storage = cfg["storage"]
    for key in ("sqlite_path", "raw_dir", "processed_dir", "archive_dir", "latest_dir", "model_dir"):
        storage[key] = str(root / storage[key]) if not Path(storage[key]).is_absolute() else storage[key]
    return cfg


def ensure_dirs() -> None:
    """确保数据/报告目录存在。"""
    cfg = get_config()
    for key in ("raw_dir", "processed_dir", "archive_dir", "latest_dir", "model_dir"):
        Path(cfg["storage"][key]).mkdir(parents=True, exist_ok=True)
    Path(cfg["storage"]["sqlite_path"]).parent.mkdir(parents=True, exist_ok=True)
