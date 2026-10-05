import numpy as np
import pandas as pd
import pytest

from models.schemas import ExperimentPlan
from services.executor import run_experiment
from services.llm import LLMError


class FakeLLM:
    def chat(self, system: str, user: str, temperature: float = 0.2) -> str:
        return "基于给定统计量的解读。"


class BrokenLLM:
    def chat(self, system: str, user: str, temperature: float = 0.2) -> str:
        raise LLMError("连接失败")


def test_pearson_ci_brackets_r():
    rng = np.random.default_rng(12)
    df = pd.DataFrame({"x": np.arange(50.0), "y": np.arange(50.0) * 1.2 + rng.normal(0, 3, 50)})
    result = run_experiment(make_plan("pearson", ["x", "y"]), df)
    ci = result.extra["ci95"]
    assert ci[0] <= result.statistic <= ci[1]


def test_welch_mean_diff_ci_excludes_zero():
    rng = np.random.default_rng(13)
    df = pd.DataFrame(
        {
            "g": ["A"] * 50 + ["B"] * 50,
            "v": np.concatenate([rng.normal(50, 5, 50), rng.normal(58, 5, 50)]),
        }
    )
    result = run_experiment(make_plan("welch_ttest", ["g", "v"]), df)
    ci = result.extra["mean_diff_ci95"]
    assert ci[1] < 0  # meanA - meanB ≈ -8：CI 整体为负且不跨零


def test_regression_diagnostics():
    rng = np.random.default_rng(14)
    x = np.arange(40.0)
    df = pd.DataFrame({"x": x, "y": 3 * x + 5 + rng.normal(0, 2, 40)})
    result = run_experiment(make_plan("linear_regression", ["x", "y"]), df)
    assert result.extra["adj_r2"] > 0.9
    assert result.extra["f_pvalue"] is not None and result.extra["f_pvalue"] < 1e-6
    assert result.extra["resid_shapiro_p"] is not None
    assert result.extra["bp_p"] is not None
    lo, hi = result.extra["slope_ci95"]
    assert lo < 3 < hi  # 真实斜率 3 落在置信区间内


def make_plan(method: str, variables: list[str], alpha: float = 0.05) -> ExperimentPlan:
    return ExperimentPlan(
        experiment_id="EXP-RQ1",
        question_id="RQ1",
        hypothesis="h",
        method=method,
        h0="H0",
        h1="H1",
        alpha=alpha,
        variables=variables,
    )


def test_pearson_rejects_h0():
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {"x": np.arange(60.0), "y": np.arange(60.0) * 1.5 + rng.normal(0, 3, 60)}
    )
    result = run_experiment(make_plan("pearson", ["x", "y"]), df)
    assert result.status == "ok"
    assert result.p_value is not None and result.p_value < 0.05
    assert result.decision == "reject_h0"
    assert result.effect_name == "r²"
    assert result.statistic_name == "r"
    assert result.interpretation_source == "rule"
    assert "拒绝 H0" in result.interpretation


def test_pearson_constant_variable_fails():
    df = pd.DataFrame({"x": [5.0] * 30, "y": np.arange(30.0)})
    result = run_experiment(make_plan("pearson", ["x", "y"]), df)
    assert result.status == "failed"
    assert "变异" in result.reason


def test_insufficient_pairs_fails():
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0], "y": [np.nan, 2.0, 3.0]})
    result = run_experiment(make_plan("pearson", ["x", "y"]), df)
    assert result.status == "failed"
    assert "不足" in result.reason


def test_welch_detects_group_difference():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "g": ["A"] * 50 + ["B"] * 50,
            "v": np.concatenate([rng.normal(50, 5, 50), rng.normal(70, 5, 50)]),
        }
    )
    result = run_experiment(make_plan("welch_ttest", ["g", "v"]), df)
    assert result.status == "ok"
    assert result.decision == "reject_h0"
    assert result.statistic_name == "t"
    assert result.effect_name == "Cohen's d"
    assert result.effect_size is not None and abs(result.effect_size) > 2
    assert len(result.groups) == 2


def test_welch_no_difference_wording():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"g": ["A"] * 80 + ["B"] * 80, "v": rng.normal(50, 8, 160)})
    result = run_experiment(make_plan("welch_ttest", ["g", "v"]), df)
    assert result.decision == "fail_to_reject_h0"
    assert "未能拒绝" in result.interpretation
    assert "接受" not in result.interpretation


def test_ttest_with_three_groups_fails_with_hint():
    df = pd.DataFrame({"g": ["A"] * 30 + ["B"] * 30 + ["C"] * 30, "v": np.arange(90.0)})
    result = run_experiment(make_plan("independent_ttest", ["g", "v"]), df)
    assert result.status == "failed"
    assert "anova" in result.reason


def test_anova_three_groups():
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "g": np.repeat(["A", "B", "C"], 40),
            "v": np.concatenate(
                [rng.normal(10, 2, 40), rng.normal(15, 2, 40), rng.normal(22, 2, 40)]
            ),
        }
    )
    result = run_experiment(make_plan("anova", ["g", "v"]), df)
    assert result.status == "ok"
    assert result.decision == "reject_h0"
    assert result.statistic_name == "F"
    assert result.effect_name == "η²"
    assert result.effect_size is not None and result.effect_size > 0.3


def test_mannwhitney():
    df = pd.DataFrame(
        {"g": ["A"] * 40 + ["B"] * 40, "v": np.concatenate([np.arange(40.0), np.arange(100, 140.0)])}
    )
    result = run_experiment(make_plan("mannwhitney", ["g", "v"]), df)
    assert result.status == "ok"
    assert result.decision == "reject_h0"
    assert result.statistic_name == "U"
    assert result.effect_name == "rank-biserial r"
    assert result.effect_size is not None and abs(result.effect_size) > 0.9  # 完全分离


def test_kruskal():
    rng = np.random.default_rng(6)
    df = pd.DataFrame(
        {
            "g": np.repeat(["A", "B", "C"], 40),
            "v": np.concatenate(
                [rng.normal(10, 3, 40), rng.normal(14, 3, 40), rng.normal(20, 3, 40)]
            ),
        }
    )
    result = run_experiment(make_plan("kruskal", ["g", "v"]), df)
    assert result.status == "ok"
    assert result.statistic_name == "H"
    assert result.decision == "reject_h0"
    assert result.effect_name == "ε²"
    assert result.effect_size is not None and result.effect_size > 0.1


def test_chi2_dependent():
    rng = np.random.default_rng(4)
    a = rng.choice(["X", "Y"], 200, p=[0.5, 0.5])
    b = np.where(
        a == "X",
        rng.choice(["P", "Q"], 200, p=[0.9, 0.1]),
        rng.choice(["P", "Q"], 200, p=[0.2, 0.8]),
    )
    df = pd.DataFrame({"a": a, "b": b})
    result = run_experiment(make_plan("chi2", ["a", "b"]), df)
    assert result.status == "ok"
    assert result.decision == "reject_h0"
    assert result.statistic_name == "χ²"
    assert result.effect_name == "Cramér's V"
    assert result.contingency is not None and len(result.contingency) == 2


def test_chi2_independent():
    rng = np.random.default_rng(5)
    df = pd.DataFrame(
        {
            "a": rng.choice(["X", "Y"], 400, p=[0.5, 0.5]),
            "b": rng.choice(["P", "Q"], 400, p=[0.5, 0.5]),
        }
    )
    result = run_experiment(make_plan("chi2", ["a", "b"]), df)
    assert result.decision == "fail_to_reject_h0"


def test_chi2_small_table_fails():
    df = pd.DataFrame({"a": ["X"] * 10, "b": ["P"] * 10})
    result = run_experiment(make_plan("chi2", ["a", "b"]), df)
    assert result.status == "failed"
    assert "2×2" in result.reason


def test_fisher_auto_for_sparse_2x2():
    a = ["X"] * 9 + ["Y"] * 9
    b = ["P"] * 8 + ["Q"] + ["P"] + ["Q"] * 8
    df = pd.DataFrame({"a": a, "b": b})
    result = run_experiment(make_plan("chi2", ["a", "b"]), df)
    assert result.status == "ok"
    assert result.extra["test_used"] == "fisher_exact"
    assert result.statistic_name == "Fisher OR"
    assert result.effect_name == "Cramér's V"
    assert result.p_value is not None and 0 <= result.p_value <= 1


def test_dense_table_keeps_chi2():
    df = pd.DataFrame(
        {
            "a": ["X"] * 40 + ["Y"] * 40,
            "b": ["P"] * 30 + ["Q"] * 10 + ["P"] * 10 + ["Q"] * 30,
        }
    )
    result = run_experiment(make_plan("chi2", ["a", "b"]), df)
    assert result.status == "ok"
    assert result.extra["test_used"] == "chi2"
    assert result.statistic_name == "χ²"


def test_decision_uses_unrounded_p():
    """判定读取的是未舍入的 p_value_raw，展示值仍是 4 位舍入。"""
    df = pd.DataFrame({"x": np.arange(40.0), "y": np.arange(40.0) + np.random.default_rng(1).normal(0, 2, 40)})
    result = run_experiment(make_plan("pearson", ["x", "y"], alpha=0.05), df)
    assert result.p_value_raw is not None
    assert result.decision == ("reject_h0" if result.p_value_raw < 0.05 else "fail_to_reject_h0")
    assert result.p_value == round(result.p_value_raw, 4)


def test_linear_regression_extra():
    df = pd.DataFrame({"x": np.arange(30.0), "y": 3 * np.arange(30.0) + 2})
    result = run_experiment(make_plan("linear_regression", ["x", "y"]), df)
    assert result.status == "ok"
    assert abs(result.extra["slope"] - 3.0) < 1e-6
    assert abs(result.extra["intercept"] - 2.0) < 1e-6
    assert result.decision == "reject_h0"


def test_paired_ttest():
    rng = np.random.default_rng(8)
    pre = rng.normal(50, 6, 40)
    df = pd.DataFrame({"pre": pre, "post": pre + 8 + rng.normal(0, 2, 40)})
    result = run_experiment(make_plan("paired_ttest", ["pre", "post"]), df)
    assert result.status == "ok"
    assert result.decision == "reject_h0"
    assert result.effect_name == "Cohen's dz"


def test_numeric_group_column_coerced():
    rng = np.random.default_rng(9)
    df = pd.DataFrame(
        {
            "g": [1] * 40 + [2] * 40,
            "v": np.concatenate([rng.normal(10, 2, 40), rng.normal(16, 2, 40)]),
        }
    )
    result = run_experiment(make_plan("welch_ttest", ["g", "v"]), df)
    assert result.status == "ok"
    assert [g.group for g in result.groups] == ["1", "2"]


def test_unknown_method_fails():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    result = run_experiment(make_plan("black_magic", ["x", "y"]), df)
    assert result.status == "failed"
    assert "不支持" in result.reason


def test_logistic_regression_deferred_to_phase2():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    result = run_experiment(make_plan("logistic_regression", ["x", "y"]), df)
    assert result.status == "failed"
    assert "Phase 2" in result.reason


def test_missing_variable_raises():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        run_experiment(make_plan("pearson", ["ghost", "y"]), df)


def test_single_variable_raises():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        run_experiment(make_plan("pearson", ["x"]), df)


def test_llm_interpretation_used_and_fallback():
    rng = np.random.default_rng(10)
    df = pd.DataFrame(
        {"x": np.arange(50.0), "y": np.arange(50.0) * 2 + rng.normal(0, 5, 50)}
    )
    plan = make_plan("pearson", ["x", "y"])
    result = run_experiment(plan, df, FakeLLM())
    assert result.interpretation_source == "llm"
    assert result.interpretation == "基于给定统计量的解读。"
    result2 = run_experiment(plan, df, BrokenLLM())
    assert result2.interpretation_source == "rule"
    assert "拒绝 H0" in result2.interpretation


def test_rule_interpretation_contains_real_numbers():
    rng = np.random.default_rng(11)
    df = pd.DataFrame({"x": np.arange(40.0), "y": np.arange(40.0) + rng.normal(0, 2, 40)})
    result = run_experiment(make_plan("pearson", ["x", "y"]), df)
    assert f"{result.statistic:.3f}" in result.interpretation
    assert "p < 0.001" in result.interpretation or "p =" in result.interpretation


def test_kruskal_degenerate_single_obs_groups():
    # 每组仅 1 条：H/p 有效但 ε² 分母为 0 —— 不崩溃，效应量优雅置空
    df = pd.DataFrame({"g": ["A", "B"], "v": [1.0, 2.0]})
    result = run_experiment(make_plan("kruskal", ["g", "v"]), df)
    assert result.status == "ok"
    assert result.statistic_name == "H"
    assert result.effect_size is None


def test_kruskal_all_identical_returns_failed_not_crash():
    # scipy 对全同值样本抛 ValueError —— 转为干净 failed 而非 500
    df = pd.DataFrame({"g": ["A"] * 10 + ["B"] * 10, "v": [5.0] * 20})
    result = run_experiment(make_plan("kruskal", ["g", "v"]), df)
    assert result.status == "failed"
    assert "无法计算" in result.reason
    assert result.interpretation == result.reason


def test_two_group_with_nan_group_labels_runs_on_real_groups():
    # 分组列含 NaN：先剔除缺标签行，two-group 方法按真实组数执行而非误报 3 组
    rng = np.random.default_rng(15)
    df = pd.DataFrame(
        {
            "g": ["A"] * 40 + ["B"] * 40 + [None] * 3,
            "v": np.concatenate(
                [rng.normal(50, 5, 40), rng.normal(70, 5, 40), [1.0, 2.0, 3.0]]
            ),
        }
    )
    result = run_experiment(make_plan("welch_ttest", ["g", "v"]), df)
    assert result.status == "ok"
    assert [g.group for g in result.groups] == ["A", "B"]
    assert result.n_used == 80
    assert result.n_dropped == 3


def test_two_group_with_single_real_group_fails_with_true_count():
    # 剔除 NaN 标签后只剩 1 组：two-group 方法按真实组数报错，而非把 "None" 计入
    df = pd.DataFrame({"g": ["A"] * 10 + [None] * 5, "v": np.arange(15.0)})
    result = run_experiment(make_plan("independent_ttest", ["g", "v"]), df)
    assert result.status == "failed"
    assert "检测到 1 组" in result.reason
    assert "None" not in result.reason


def test_anova_nan_group_labels_no_ghost_group():
    # anova：剔除分组列 NaN 行，结果不含 "None" 幽灵组且 n_dropped 如实上报
    rng = np.random.default_rng(16)
    df = pd.DataFrame(
        {
            "g": np.repeat(["A", "B"], 30).tolist() + [None] * 2,
            "v": np.concatenate([rng.normal(10, 2, 30), rng.normal(20, 2, 30), [99.0, -99.0]]),
        }
    )
    result = run_experiment(make_plan("anova", ["g", "v"]), df)
    assert result.status == "ok"
    group_names = [g.group for g in result.groups]
    assert "None" not in group_names
    assert result.n_dropped == 2 and result.n_dropped > 0
    assert "None" not in result.interpretation
