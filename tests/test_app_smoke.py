# -*- coding: utf-8 -*-
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "samples" / "student_performance.csv"

pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="示例数据未生成")


def make_app() -> AppTest:
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)


def test_app_renders_without_exception():
    at = make_app()
    at.run()
    assert not at.exception


def test_app_ml_experiment_flow():
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
