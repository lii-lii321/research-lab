"""复现脚本生成器：产物必须是合法 Python 且包含真实统计调用。"""
from models.schemas import ExperimentPlan
from services.repro import build_repro_script


def make_plan(method: str, variables: list[str]) -> ExperimentPlan:
    return ExperimentPlan(
        experiment_id="EXP-RQ1",
        question_id="RQ1",
        hypothesis="出勤率与最终成绩存在线性相关",
        method=method,
        h0="H0",
        h1="H1",
        alpha=0.05,
        variables=variables,
    )


def assert_valid(src: str) -> None:
    compile(src, "repro.py", "exec")


def test_pearson_script_valid_and_complete():
    src = build_repro_script(make_plan("pearson", ["attendance_rate", "final_score"]))
    assert_valid(src)
    assert "stats.pearsonr" in src
    assert 'DATA_PATH = "your_data.csv"' in src
    assert "0.05" in src
    assert "拒绝 H0" in src


def test_welch_script_groups_by_column():
    src = build_repro_script(make_plan("welch_ttest", ["gender", "final_score"]))
    assert_valid(src)
    assert "stats.ttest_ind" in src
    assert "equal_var=False" in src


def test_mannwhitney_and_anova_and_kruskal():
    for method, needle in (
        ("mannwhitney", "stats.mannwhitneyu"),
        ("anova", "stats.f_oneway"),
        ("kruskal", "stats.kruskal"),
    ):
        src = build_repro_script(make_plan(method, ["g", "score"]))
        assert_valid(src)
        assert needle in src


def test_chi2_script_uses_contingency():
    src = build_repro_script(make_plan("chi2", ["gender", "school_type"]))
    assert_valid(src)
    assert "pd.crosstab" in src
    assert "stats.chi2_contingency" in src


def test_paired_and_regression_valid():
    src = build_repro_script(make_plan("paired_ttest", ["pre", "post"]))
    assert_valid(src)
    assert "stats.ttestrel" not in src  # 拼写守卫：应为 ttest_rel
    assert "stats.ttest_rel" in src
    src2 = build_repro_script(make_plan("linear_regression", ["x", "y"]))
    assert_valid(src2)
    assert "sm.OLS" in src2 and "statsmodels" in src2


def test_unknown_method_still_valid_python():
    src = build_repro_script(make_plan("black_magic", ["a", "b"]))
    assert_valid(src)
    assert "暂无自动复现代码" in src
