"""代理服务器管理与"信源不可达暂停换代理"交互。

设计要点
--------
1. 代理来源优先级（高→低）：运行时 ``set_proxy`` / CLI ``--proxy`` / 网页侧栏
   > 环境变量 ``OILCAST_PROXY`` > 本机持久化文件 ``data/proxy.json``
   > ``config.yaml`` 的 ``network.proxy_url`` > 标准 ``HTTPS_PROXY`` 环境变量；
   都没有则直连。
2. 代理通过两种方式同时生效：写入进程环境变量（``HTTP_PROXY/HTTPS_PROXY/ALL_PROXY``，
   使 requests / urllib / pandas.read_csv / feedparser 等所有走环境的客户端统一使用），
   以及对 :class:`oilcast.utils.PoliteSession` 显式 ``session.proxies`` 双保险。
3. 信源在**网络层不可达**（连接被拒/超时，区别于 403/429 限频）且连续失败即将熔断时，
   交互终端会暂停并倒计时，等待用户更换代理；网页/CI 等无 stdin 环境**绝不阻塞**，
   直接走原有熔断与"缺失即 unavailable"降级。
4. 严格遵守"只用真实数据"：换代理只是换网络通道，拿不到的数据依旧标记不可达，不补齐。

代理地址形如 ``http://127.0.0.1:7890``、``http://user:pass@host:port``、
``socks5://127.0.0.1:1080``（内部自动改用 ``socks5h://``，让代理解析 DNS、规避本地
DNS 污染，需安装 PySocks，已列入 requirements）。
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import requests

from .config import get_config

PROXY_SCHEMES = ("http://", "https://", "socks5://", "socks5h://")
_DEFAULT_TEST_URL = "https://www.google.com/generate_204"


# ------------------------------------------------------------ 路径与归一化
def proxy_file() -> Path:
    """本机代理持久化文件（随数据目录，不打进发布包、不提交仓库）。"""
    return Path(get_config()["storage"]["sqlite_path"]).parent / "proxy.json"


def normalize_proxy(url: Optional[str]) -> str:
    """规范化代理地址；空值返回空串（直连）。socks5:// 升级为 socks5h://（远端解析 DNS）。"""
    u = (url or "").strip()
    if not u:
        return ""
    if not u.startswith(PROXY_SCHEMES):
        u = "http://" + u          # 只给 host:port 时默认 http 代理
    if u.startswith("socks5://"):
        u = "socks5h://" + u[len("socks5://"):]
    return u


def proxies_for(url: Optional[str]) -> dict:
    u = normalize_proxy(url)
    return {"http": u, "https": u} if u else {}


# ------------------------------------------------------------ 全局状态
# _url 三态：
#   None = 从未显式配置（透明模式）：完全不干预 requests 默认行为，保留 trust_env=True、
#          不清任何环境变量——云端/Streamlit 等形态与引入本模块前逐字节一致，互不影响；
#   ""   = 用户显式要求直连（--no-proxy）：proxies 清空、trust_env=False，绕过系统代理；
#   非空 = 用户显式代理：设置 proxies 并写入进程环境变量。
class _ProxyState:
    def __init__(self) -> None:
        self._url: Optional[str] = self._discover()
        self._paused_hosts: set[str] = set()

    def _discover(self) -> Optional[str]:
        # 1) 运行期环境变量（最高优先级，非空才生效）
        if os.environ.get("OILCAST_PROXY", "").strip():
            return normalize_proxy(os.environ["OILCAST_PROXY"])
        # 2) 本机持久化文件：只要文件存在就以其值为准（含空串=显式直连），不再向下回退
        try:
            f = proxy_file()
            if f.exists():
                return normalize_proxy(
                    json.loads(f.read_text(encoding="utf-8")).get("proxy_url"))
        except Exception:
            pass
        # 3) 配置文件显式代理
        cfg_url = get_config().get("network", {}).get("proxy_url", "")
        if cfg_url and str(cfg_url).strip():
            return normalize_proxy(cfg_url)
        # 4) 未配置 → 透明（None）。不主动读取/接管标准 HTTPS_PROXY，交给 requests 默认处理。
        return None

    @property
    def url(self) -> Optional[str]:
        return self._url

    def set_url(self, url: Optional[str], persist: bool = True) -> str:
        self._url = normalize_proxy(url)   # 传空串 → ""（显式直连）
        if persist:
            self._save()
        export_env_proxy(self._url)
        return self._url or ""

    def _save(self) -> None:
        try:
            f = proxy_file()
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(
                {"proxy_url": self._url or "",
                 "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def take_pause_token(self, host: str) -> bool:
        """每个 host 每次运行最多交互暂停一次，避免反复弹窗。"""
        if host in self._paused_hosts:
            return False
        self._paused_hosts.add(host)
        return True


_STATE = _ProxyState()


# ------------------------------------------------------------ 对外 API
def current_proxy() -> str:
    """当前显式代理；透明/直连均返回空串。"""
    return _STATE.url or ""


def is_configured() -> bool:
    """是否被显式配置过（含显式直连）。False=完全透明、不干预 requests 默认行为。"""
    return _STATE.url is not None


def set_proxy(url: Optional[str], persist: bool = True) -> str:
    """设置代理（传 URL）或显式直连（传空串）；默认持久化到 data/proxy.json 并注入进程环境。"""
    return _STATE.set_url(url, persist=persist)


def export_env_proxy(url: Optional[str] = None) -> str:
    """同步代理到进程环境变量。

    显式代理→写入 HTTP(S)_PROXY/ALL_PROXY；显式直连(空串)→清除这些变量；
    未配置(None)→**什么都不做**，保留宿主原有环境（云端透明）。
    """
    u = _STATE.url if url is None else url
    if u is None:
        return ""
    keys = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
            "http_proxy", "https_proxy", "all_proxy")
    if u:
        for k in keys:
            os.environ[k] = u
    else:
        for k in keys:
            os.environ.pop(k, None)
    return u


def apply_to_session(session: requests.Session) -> requests.Session:
    """给 requests.Session 应用代理策略。

    未配置(None)：完全不干预（保持 requests 默认 trust_env=True，云端行为不变）；
    显式直连(空串)：清空 proxies、trust_env=False，绕过任何系统代理；
    显式代理：设置 proxies、trust_env=True。
    """
    u = _STATE.url
    if u is None:
        return session
    if u == "":
        session.proxies.clear()
        session.trust_env = False
        return session
    session.proxies.update(proxies_for(u))
    session.trust_env = True
    return session


def test_proxy(url: Optional[str] = None, timeout: float = 8.0) -> dict:
    """测试当前通道（显式代理/显式直连/系统默认）能否访问探测地址，不抛异常。"""
    u = None if url is None else normalize_proxy(url)
    if u is None:
        u = _STATE.url          # None=透明（走系统默认），""=显式直连，URL=代理
    test_url = str(get_config().get("network", {})
                   .get("proxy_test_url", _DEFAULT_TEST_URL))
    t0 = time.time()
    via = u or ("直连" if u == "" else "系统默认通道")
    try:
        sess = requests.Session()
        if u == "":
            sess.trust_env = False                      # 显式直连：绕过系统代理
            resp = sess.get(test_url, timeout=timeout)
        elif u:
            resp = sess.get(test_url, proxies=proxies_for(u), timeout=timeout)
        else:
            resp = sess.get(test_url, timeout=timeout)  # 透明：requests 默认（含系统代理）
        ok = resp.status_code in (200, 204)
        return {"ok": ok, "status": resp.status_code,
                "elapsed": round(time.time() - t0, 2),
                "via": (u or via), "test_url": test_url, "error": None}
    except Exception as exc:  # noqa: BLE001 - 探测只汇报失败原因
        return {"ok": False, "status": None,
                "elapsed": round(time.time() - t0, 2),
                "via": (u or via), "test_url": test_url,
                "error": f"{type(exc).__name__}: {str(exc)[:160]}"}


# ------------------------------------------------------------ 不可达暂停
def pause_enabled() -> bool:
    """是否允许交互式暂停倒计时。

    on/true 强制开；off/false 关；auto（默认）仅在真实交互终端且非 CI 时开。
    Streamlit 以 subprocess（stdin 被捕获、非 tty）跑流水线时自动判定为关，绝不卡死网页。
    """
    mode = str(get_config().get("network", {})
               .get("proxy_pause_enabled", "auto")).strip().lower()
    if mode in ("off", "false", "0", "no"):
        return False
    if mode in ("on", "true", "1", "yes"):
        return True
    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
        return False
    try:
        return bool(sys.stdin and sys.stdin.isatty())
    except Exception:
        return False


def _read_line_with_timeout(seconds: int) -> Optional[str]:
    """后台线程读一行，最多等 seconds 秒；超时返回 None（无响应），读到返回文本（含空串）。"""
    box: dict = {}

    def _reader() -> None:
        try:
            box["line"] = sys.stdin.readline()
        except Exception:  # noqa: BLE001
            box["line"] = ""

    th = threading.Thread(target=_reader, daemon=True)
    th.start()
    left = seconds
    while left > 0 and "line" not in box:
        sys.stdout.write(
            f"\r  ⏳ 倒计时 {left:2d}s：更换好代理后按回车立即重试；"
            f"也可直接粘贴新代理地址后回车；不操作则倒计时结束继续下一步…")
        sys.stdout.flush()
        time.sleep(1)
        left -= 1
    sys.stdout.write("\n")
    sys.stdout.flush()
    return box.get("line") if "line" in box else None


def maybe_pause_for_proxy(host: str, exc: Optional[object] = None,
                          seconds: Optional[int] = None) -> bool:
    """信源网络层不可达、即将熔断前调用。

    返回 ``True``：用户已处理（换了代理，或空回车表示已在外部切换网络），调用方应清除该
    host 的失败计数/熔断并用新通道**重试一次**；
    返回 ``False``：非交互环境、该 host 已暂停过、或倒计时结束无响应——调用方继续下一步
    （按原逻辑熔断、该信源本次标记不可达，绝不阻塞流水线）。
    """
    if not pause_enabled():
        return False
    if not _STATE.take_pause_token(host):
        return False
    cfg_net = get_config().get("network", {})
    secs = int(seconds if seconds is not None
               else cfg_net.get("proxy_pause_seconds", 60))
    cur = _STATE.url or "（当前直连）"
    print("\n" + "=" * 72)
    print("⚠️  信源网络不可达，可能是代理失效或该站点需要特定网络通道")
    print(f"    主机：{host}")
    print(f"    错误：{str(exc)[:200]}")
    print(f"    当前代理：{cur}")
    print("    你可以：① 在代理软件里切换节点/线路后直接回车；"
          "② 粘贴新代理地址（如 http://127.0.0.1:7890）后回车；")
    print("             ③ 不做任何操作，倒计时结束后自动跳过该信源、继续后续步骤。")
    print("=" * 72)
    line = _read_line_with_timeout(secs)
    if line is None:
        print("  ⏱️  倒计时结束未收到操作，继续执行下一步（该信源本次标记为不可达）。")
        return False
    text = line.strip()
    if text.lower() in ("s", "n", "skip", "no", "跳过"):
        print("  已选择跳过该信源，继续执行下一步。")
        return False
    if text:
        new_u = set_proxy(text)
        print(f"  ✅ 已切换代理为：{new_u}，将用新通道重试。")
    else:
        print("  ✅ 收到回车（视为已在本机/代理软件中切换好网络），立即重试一次。")
    return True


def reset_circuit_for(host: str) -> None:
    """换通道后清除 PoliteSession 对该 host 的失败计数与熔断（延迟导入避免循环依赖）。"""
    from .utils import PoliteSession
    PoliteSession._fail_count.pop(host, None)
    PoliteSession._circuit.pop(host, None)
