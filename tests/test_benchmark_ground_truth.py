"""Benchmark 真值守护：植入效应 → research-lab 全流程 → 结论与真值一致。

模块内只跑一遍 4 个任务的完整流程（fixture 共享结果），全量 5 任务走
`python benchmarks/runner.py`（CI 3.10 job 跑 --subset 3 --strict）。
测试直接注入 DataFrame，不写 benchmarks/data/。
"""
import pytest

from benchmarks.runner import load_tasks, run_pipeline_expectations
from benchmarks.synth import make_course_rating, make_exam_passed, make_tutoring_score

TASKS = {t["id"]: t for t in load_tasks()}


@pytest.fixture(scope="module")
def results():
    jobs = {
        "tutoring_group_gap": make_tutoring_score(),
        "tutoring_attention_null": make_tutoring_score(),
        "exam_school_assoc": make_exam_passed(),
        "rating_instructor_anova": make_course_rating(),
    }
    return {tid: run_pipeline_expectations(df, TASKS[tid]) for tid, df in jobs.items()}


def test_benchmark_group_gap_matches_truth(results):
    ok, detail = results["tutoring_group_gap"]
    assert ok, f"组间差异真值校验失败：{detail}"


def test_benchmark_null_control_stays_null(results):
    ok, detail = results["tutoring_attention_null"]
    assert ok, f"阴性对照被误报显著（管线在噪声上发现假效应）：{detail}"


def test_benchmark_chi2_assoc_matches_truth(results):
    ok, detail = results["exam_school_assoc"]
    assert ok, f"卡方关联真值校验失败：{detail}"


def test_benchmark_anova_direction_matches_truth(results):
    ok, detail = results["rating_instructor_anova"]
    assert ok, f"多组差异方向校验失败：{detail}"
