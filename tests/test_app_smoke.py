"""Streamlit AppTest 界面级冒烟：真实渲染页面并点击关键流程。

多页面架构（S0）：AppTest.from_file 入口是 app.py（仅导航），
对特定页面的测试直接 from_file 对应 pages/*.py 文件。
示例数据由运行时自愈生成（S1.1），任何全新检出都应通过。
"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ROOT / "pages"


def make_app(page: str = "") -> AppTest:
    target = PAGES / page if page else ROOT / "app.py"
    return AppTest.from_file(str(target), default_timeout=300)


def test_app_navigation_renders():
    at = make_app()
    at.run()
    assert not at.exception


def test_flow_page_renders():
    at = make_app("01_分析流程.py")
    at.run()
    assert not at.exception


def test_tracking_page_renders():
    at = make_app("03_实验追踪.py")
    at.run()
    assert not at.exception


def test_reports_page_renders():
    at = make_app("05_报告库.py")
    at.run()
    assert not at.exception


def test_ml_experiment_flow(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_DB", str(tmp_path / "app_ml.db"))
    at = make_app("02_ML实验室.py")
    at.run()
    assert not at.exception
    at.selectbox(key="ml_target").set_value("final_score")
    at.selectbox(key="ml_task").set_value("regression")
    at.button(key="run_ml").click()
    at.run()
    assert not at.exception
    result = at.session_state["ml_result"]
    assert result.status == "ok"
    assert result.task == "regression"
    assert result.tracked_uid


def test_agent_flow(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_DB", str(tmp_path / "app_agent.db"))
    at = make_app("04_自动研究.py")
    at.run()
    assert not at.exception
    at.text_input(key="agent_task").set_value("研究影响学生成绩的因素")
    at.slider(key="agent_max").set_value(2)
    at.checkbox(key="agent_lit").set_value(False)
    at.button(key="agent_run").click()
    at.run()
    assert not at.exception
    result = at.session_state["agent_result"]
    assert result.status == "ok"
    assert result.questions
    assert "AI 数据科学研究报告" in result.report_markdown


def test_intro_card_dismissed_after_click():
    at = make_app()
    at.run()
    assert not at.exception
    if at.button(key="intro_ok"):
        at.button(key="intro_ok").click()
        at.run()
        assert not at.exception
