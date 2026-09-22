"""代理管理与"信源不可达暂停换代理"测试（全部离线，不访问网络）。"""
import io
import os

import pytest
import requests

from oilcast import net
from oilcast.utils import PoliteSession


# ---------------------------------------------------------- 归一化 / 映射
def test_normalize_proxy():
    assert net.normalize_proxy("") == ""
    assert net.normalize_proxy(None) == ""
    assert net.normalize_proxy("127.0.0.1:7890") == "http://127.0.0.1:7890"
    assert net.normalize_proxy("http://127.0.0.1:7890") == "http://127.0.0.1:7890"
    # socks5 升级为 socks5h（远端解析 DNS）
    assert net.normalize_proxy("socks5://127.0.0.1:1080") == "socks5h://127.0.0.1:1080"
    assert net.normalize_proxy("socks5h://h:1") == "socks5h://h:1"
    assert net.proxies_for("http://h:8080") == {
        "http": "http://h:8080", "https": "http://h:8080"}
    assert net.proxies_for("") == {}


# ---------------------------------------------------------- 持久化 / 环境
def test_set_persist_and_env(monkeypatch):
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    u = net.set_proxy("127.0.0.1:7890")          # 自动补 http://
    assert u == "http://127.0.0.1:7890"
    assert net.current_proxy() == "http://127.0.0.1:7890"
    assert os.environ["HTTPS_PROXY"] == "http://127.0.0.1:7890"
    # 已落盘到本机 data/proxy.json
    assert net.proxy_file().exists()
    assert '"proxy_url": "http://127.0.0.1:7890"' in net.proxy_file().read_text("utf-8")
    # 清除 → 直连，环境变量被移除
    assert net.set_proxy("") == ""
    assert "HTTPS_PROXY" not in os.environ
    assert '"proxy_url": ""' in net.proxy_file().read_text("utf-8")


def test_transparent_when_unconfigured(monkeypatch):
    # 未显式配置代理（云端/Streamlit 形态）：必须完全透明，不干预 requests 默认、不清环境
    pf = net.proxy_file()
    if pf.exists():
        pf.unlink()
    monkeypatch.delenv("OILCAST_PROXY", raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://system-env:9999")  # 宿主原有代理
    net._STATE._url = None
    assert net.is_configured() is False
    s = requests.Session()
    net.apply_to_session(s)
    assert s.trust_env is True and not s.proxies        # 未被改动
    net.export_env_proxy()
    assert os.environ["HTTPS_PROXY"] == "http://system-env:9999"  # 宿主变量未被清除
    net.set_proxy("", persist=False)                   # 还原，避免污染后续用例


def test_apply_to_session_toggle():
    import requests as rq
    s = rq.Session()
    net.set_proxy("http://127.0.0.1:7890", persist=False)
    net.apply_to_session(s)
    assert s.proxies.get("https") == "http://127.0.0.1:7890" and s.trust_env is True
    net.set_proxy("", persist=False)
    net.apply_to_session(s)
    assert not s.proxies and s.trust_env is False


# ---------------------------------------------------------- 暂停开关
def test_pause_enabled_auto_off_in_test(monkeypatch):
    # pytest 下 stdin 非 tty、auto 模式必须判定为不暂停（绝不阻塞 CI/网页）
    monkeypatch.setenv("CI", "true")
    assert net.pause_enabled() is False


def test_pause_enabled_explicit(monkeypatch):
    fake = {"network": {"proxy_pause_enabled": "off"}}
    monkeypatch.setattr(net, "get_config", lambda *a, **k: fake)
    assert net.pause_enabled() is False
    fake2 = {"network": {"proxy_pause_enabled": "on"}}
    monkeypatch.setattr(net, "get_config", lambda *a, **k: fake2)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    assert net.pause_enabled() is True


# ---------------------------------------------------------- 交互暂停逻辑
def _force_interactive(monkeypatch):
    monkeypatch.setattr(net, "pause_enabled", lambda: True)


def test_pause_new_proxy_then_retry(monkeypatch):
    _force_interactive(monkeypatch)
    monkeypatch.setattr("sys.stdin", io.StringIO("127.0.0.1:1080\n"))
    host = "unit-new-proxy.example"
    net.set_proxy("", persist=False)
    handled = net.maybe_pause_for_proxy(host, RuntimeError("connect timeout"), seconds=3)
    assert handled is True                     # 用户给了新代理 → 重试
    assert net.current_proxy() == "http://127.0.0.1:1080"


def test_pause_blank_enter_means_retry(monkeypatch):
    _force_interactive(monkeypatch)
    monkeypatch.setattr("sys.stdin", io.StringIO("\n"))
    host = "unit-blank.example"
    assert net.maybe_pause_for_proxy(host, None, seconds=3) is True


def test_pause_timeout_returns_false(monkeypatch):
    _force_interactive(monkeypatch)
    monkeypatch.setattr(net, "_read_line_with_timeout", lambda secs: None)  # 无响应
    host = "unit-timeout.example"
    assert net.maybe_pause_for_proxy(host, None, seconds=1) is False


def test_pause_skip_keyword(monkeypatch):
    _force_interactive(monkeypatch)
    monkeypatch.setattr("sys.stdin", io.StringIO("skip\n"))
    assert net.maybe_pause_for_proxy("unit-skip.example", None, seconds=2) is False


def test_pause_once_per_host(monkeypatch):
    _force_interactive(monkeypatch)
    monkeypatch.setattr("sys.stdin", io.StringIO("\n\n"))
    host = "unit-once.example"
    assert net.maybe_pause_for_proxy(host, None, seconds=2) is True
    assert net.maybe_pause_for_proxy(host, None, seconds=2) is False  # 同 host 不再弹


def test_pause_non_interactive_never_blocks(monkeypatch):
    monkeypatch.setattr(net, "pause_enabled", lambda: False)
    called = {"n": 0}

    def _boom(secs):  # 若误调用读取则让测试失败
        called["n"] += 1
        raise AssertionError("非交互环境不得等待 stdin")

    monkeypatch.setattr(net, "_read_line_with_timeout", _boom)
    assert net.maybe_pause_for_proxy("unit-ci.example", None) is False
    assert called["n"] == 0


# --------------------------------------- PoliteSession：连续不可达→换代理→重试成功
def test_polite_session_retries_after_proxy_switch(monkeypatch):
    net.set_proxy("", persist=False)
    sess = PoliteSession()
    sess.sleep_sec = 0
    monkeypatch.setattr(net, "pause_enabled", lambda: True)

    switched = {"n": 0}

    def fake_pause(host, exc=None, seconds=None):
        # 模拟用户在暂停时换成了可用代理
        net.set_proxy("http://127.0.0.1:7890", persist=False)
        switched["n"] += 1
        return True

    monkeypatch.setattr(net, "maybe_pause_for_proxy", fake_pause)

    class _Resp:
        status_code = 200
        encoding = None

        def raise_for_status(self):
            return None

    calls = {"n": 0}

    def fake_get(url, params=None, timeout=None):
        calls["n"] += 1
        if calls["n"] <= PoliteSession.CIRCUIT_FAIL_THRESHOLD:
            raise requests.ConnectionError("simulated unreachable")
        return _Resp()

    monkeypatch.setattr(sess.session, "get", fake_get)
    resp = sess.get("https://unit-host.example/data")
    assert resp is not None and resp.status_code == 200
    assert switched["n"] == 1                  # 熔断前暂停了一次
    assert sess.session.proxies.get("https") == "http://127.0.0.1:7890"
    net.set_proxy("", persist=False)


def test_polite_session_circuit_breaks_when_no_switch(monkeypatch):
    net.set_proxy("", persist=False)
    sess = PoliteSession()
    sess.sleep_sec = 0
    monkeypatch.setattr(net, "pause_enabled", lambda: False)  # CI/网页：不暂停

    def fake_get(url, params=None, timeout=None):
        raise requests.ConnectionError("unreachable")

    monkeypatch.setattr(sess.session, "get", fake_get)
    assert sess.get("https://unit-host2.example/data") is None  # 熔断快速失败、不卡死
