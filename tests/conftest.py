"""pytest 全局隔离：所有测试的数据/报告/数据库必须落在临时目录，
严禁测试夹具数据写入项目真实 data/ 与 reports/（防止污染真实谱系）。"""
import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _isolate_oilcast_home(tmp_path_factory):
    home = tmp_path_factory.mktemp("oilcast_home")
    os.environ["OILCAST_HOME"] = str(home)
    from oilcast.config import get_config
    get_config.cache_clear()
    yield
    os.environ.pop("OILCAST_HOME", None)
    get_config.cache_clear()
