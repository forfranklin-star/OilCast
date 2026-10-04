"""跨进程单实例锁。

防止在 Streamlit 中重复点击"立即重新生成报告"、页面重连或脚本重跑时，
上一次 ``subprocess.Popen`` 启动的 pipeline 子进程（它脱离 Streamlit 脚本
生命周期、在后台继续运行）与新启动的子进程并发训练、并发替换同一个
``.joblib``，从而在 Windows 上触发 ``[WinError 32] 另一个程序正在使用此文件``。

实现采用绑定在打开文件描述符上的**咨询锁**：
- POSIX: ``fcntl.flock(LOCK_EX | LOCK_NB)``
- Windows: ``msvcrt.locking(LK_NBLCK, 1)``

``acquire_pipeline_lock()`` 成功后**故意保持该 fd 不关闭**，于是锁一直持有到
进程结束；进程（即使崩溃）退出时 OS 自动关闭 fd、释放锁——不会因崩溃而留下
永久死锁，也无需手工删除锁文件。
"""
from __future__ import annotations

import os
from pathlib import Path

from ..config import get_config
from ..utils import get_logger

LOG = get_logger(__name__)
LOCK_NAME = "pipeline.lock"


def _lock_path() -> Path:
    p = Path(get_config()["storage"]["model_dir"]) / LOCK_NAME
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _try_lock_fd(fd: int) -> bool:
    """对 fd 起始的 1 字节加非阻塞排他锁；已被占用返回 False。"""
    try:
        if os.name == "nt":
            import msvcrt
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def acquire_pipeline_lock() -> bool:
    """尝试获取单实例锁。

    返回 True 表示获得锁（调用方继续运行；fd 保持开启、锁持有到进程退出）；
    返回 False 表示已有另一个 pipeline 实例在运行，调用方应放弃本次执行。
    """
    path = _lock_path()
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        # msvcrt.locking 需要被锁区域至少有 1 字节
        os.lseek(fd, 0, os.SEEK_END)
        if os.fstat(fd).st_size < 1:
            os.write(fd, b"0")
        if not _try_lock_fd(fd):
            return False
        # 记录持有者 PID（仅用于排查；锁本身由 fd 持有）
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        os.fsync(fd)
        # 刻意不 os.close(fd)：保持锁到进程退出，由 OS 自动释放。
        return True
    except OSError:
        os.close(fd)
        return False
