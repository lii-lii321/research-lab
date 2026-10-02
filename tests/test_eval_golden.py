"""golden 评测集：变量类型×方法组合 → 规则引擎必须命中期望方法。

这是规则计划引擎的回归测试（进 CI 主 job）；
LLM 路径的一致率评测见 scripts/eval_llm.py（需 Key，手动运行）。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from models.schemas import ResearchQuestion
from services.planner import generate_experiment_plan
from services.profiler import profile_dataset
from tests.eval.synth import build_df

GOLDEN_PATH = Path(__file__).resolve().parent / "eval" / "golden_cases.json"


def load_cases() -> list[dict]:
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    return data["cases"]


CASES = load_cases()
CASE_IDS = [c["id"] for c in CASES]


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_golden_method_matches(case: dict) -> None:
    df = build_df(case["columns"], seed=int(case.get("seed", 0)))
    report = profile_dataset(df)
    rq = ResearchQuestion(
        id=f"G-{case['id']}",
        question=f"{' 与 '.join(case['variables'])} 的关系",
        variables=case["variables"],
    )
    plan = generate_experiment_plan(rq, report, None)
    assert plan.method == case["expected_method"], (
        f"{case['id']}: 期望 {case['expected_method']}，实际 {plan.method}（规则引擎漂移）"
    )


def test_golden_cases_cover_all_type_combos() -> None:
    ids = set(CASE_IDS)
    assert {"num_num_pearson", "cat2_num_welch", "cat3_num_anova", "cat_cat_chi2"} <= ids
    assert len(CASES) >= 15
