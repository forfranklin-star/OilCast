"""Windows 非 ASCII（中文等）安装路径下的 CA 证书兼容修复。

现象：项目放在含中文的目录（如 ``E:\\OilCast-本地\\...``）时，yfinance 新版底层的
curl_cffi/libcurl 无法从该路径加载 ``certifi\\cacert.pem``，报
``curl: (77) error adding trust anchors ... CAfile: ...cacert.pem``。根因是 libcurl 的
C 层按系统 ANSI 代码页打开文件，遇到非 ASCII 路径失败（Python 原生 open 不受影响）。

处理：仅在 **Windows 且 certifi 路径含非 ASCII 字符**时，把 cacert.pem 复制到一个纯
ASCII、当前用户可写的稳定目录，并设置 ``CURL_CA_BUNDLE / REQUESTS_CA_BUNDLE /
SSL_CERT_FILE`` 指向它。其他平台、或路径本就是 ASCII 时完全不干预（云端/Linux 零影响）。
幂等，进程内只执行一次。
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

_FLAG = "OILCAST_CA_READY"


def _ascii_writable_dirs() -> list[Path]:
    """返回纯 ASCII 且可写的候选目录（按优先级）。"""
    cands: list[Path] = []
    program_data = os.environ.get("ProgramData")
    if program_data:
        cands.append(Path(program_data) / "OilCast")
    if sys.platform.startswith("win"):
        cands.append(Path("C:/Windows/Temp/OilCast"))
    cands.append(Path(tempfile.gettempdir()) / "oilcast")
    cands.append(Path.home() / ".oilcast")
    ok: list[Path] = []
    for d in cands:
        try:
            if not str(d).isascii():
                continue
            d.mkdir(parents=True, exist_ok=True)
            probe = d / "_write_probe"
            probe.write_text("x", encoding="utf-8")
            probe.unlink()
            ok.append(d)
        except Exception:
            continue
    return ok


def ensure_ca_bundle(ca_path: Optional[str] = None,
                     is_windows: Optional[bool] = None) -> Optional[str]:
    """必要时把 CA 束复制到纯 ASCII 路径并导出环境变量；返回该路径（无需处理时返回 None）。

    ca_path / is_windows 仅用于单元测试注入；生产默认自动探测 certifi 与当前平台。
    """
    if os.environ.get(_FLAG):
        return os.environ.get("CURL_CA_BUNDLE") or None

    win = sys.platform.startswith("win") if is_windows is None else is_windows
    try:
        if ca_path is None:
            import certifi  # 延迟导入：未装 yfinance/certifi 的环境不需要本模块
            ca_path = certifi.where()
        ca = Path(ca_path)
    except Exception:
        os.environ[_FLAG] = "1"
        return None

    # 非 Windows，或路径本身就是纯 ASCII（libcurl 能正常打开）→ 不干预。
    if not win or str(ca).isascii():
        os.environ[_FLAG] = "1"
        return None

    for d in _ascii_writable_dirs():
        try:
            dst = d / "cacert.pem"
            if (not dst.exists()) or dst.stat().st_size != ca.stat().st_size:
                shutil.copyfile(ca, dst)
            target = str(dst)
            for k in ("CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE"):
                os.environ[k] = target
            os.environ[_FLAG] = "1"
            return target
        except Exception:
            continue

    # 找不到合适的 ASCII 目录：放弃复制（调用方仍可改走 requests Session 绕开 libcurl）。
    os.environ[_FLAG] = "1"
    return None
