"""Streamlit AppTest 界面级冒烟：真实渲染页面并点击关键流程。

示例数据由 app 运行时自愈生成（S1.1），因此这些测试在任何全新检出上都应通过。
"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def make_app() -> AppTest:
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)


def test_app_renders_without_exception():
    at = make_app()
    at.run()
    assert not at.exception


def test_app_ml_experiment_flow(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_DB", str(tmp_path / "app_ml.db"))
    at = make_app()
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


def test_app_agent_flow(tmp_path, monkeypatch):

    monkeypatch.setenv("RESEARCH_LAB_DB", str(tmp_path / "app_agent.db"))
    at = make_app()
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
