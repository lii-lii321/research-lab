import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from models.schemas import ExperimentPlan, ExperimentResult
from utils.charts import correlation_heatmap, result_figure


def make_plan(method: str, variables: list[str]) -> ExperimentPlan:
    return ExperimentPlan(
        experiment_id="EXP-RQ1",
        question_id="RQ1",
        hypothesis="h",
        method=method,
        h0="H0",
        h1="H1",
        alpha=0.05,
        variables=variables,
    )


def make_result(method: str, n_used: int = 50) -> ExperimentResult:
    return ExperimentResult(
        experiment_id="EXP-RQ1",
        question_id="RQ1",
        method=method,
        status="ok",
        alpha=0.05,
        n_used=n_used,
        statistic=0.5,
        statistic_name="r",
        p_value=0.001,
    )


def build_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "a": rng.normal(size=60),
            "b": rng.normal(size=60),
            "c": rng.normal(size=60),
            "g": rng.choice(["X", "Y"], 60),
            "label": rng.choice(["P", "Q"], 60),
        }
    )


def test_heatmap_returns_figure_for_numeric_cols():
    df = build_df()
    fig = correlation_heatmap(df, ["a", "b", "c"])
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_heatmap_none_when_fewer_than_two():
    df = build_df()
    assert correlation_heatmap(df, ["a"]) is None
    assert correlation_heatmap(df, []) is None


def test_result_figure_correlation():
    df = build_df()
    fig = result_figure(df, make_plan("pearson", ["a", "b"]), make_result("pearson"))
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_result_figure_group_compare():
    df = build_df()
    fig = result_figure(df, make_plan("welch_ttest", ["g", "a"]), make_result("welch_ttest"))
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_result_figure_chi2():
    df = build_df()
    fig = result_figure(df, make_plan("chi2", ["g", "label"]), make_result("chi2"))
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_result_figure_unknown_method_returns_none():
    df = build_df()
    assert result_figure(df, make_plan("mystery", ["a", "b"]), make_result("mystery")) is None
