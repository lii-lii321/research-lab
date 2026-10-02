import numpy as np
import pandas as pd
import pytest

from services.ml_lab import (
    infer_task,
    run_ml_experiment,
    select_features,
)
from services.profiler import profile_dataset


def build_regression_df(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "student_id": [f"S{i:04d}" for i in range(n)],
            "hours": rng.uniform(0, 10, n).round(2),
            "attendance": rng.uniform(50, 100, n).round(1),
        }
    )
    df["score"] = (3 * df["hours"] + 0.5 * df["attendance"] + rng.normal(0, 1, n)).round(2)
    return df


def build_classification_df(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    half = n // 2
    return pd.DataFrame(
        {
            "x1": np.concatenate([rng.normal(0, 1, half), rng.normal(4, 1, n - half)]),
            "x2": np.concatenate([rng.normal(0, 1, half), rng.normal(4, 1, n - half)]),
            "label": ["A"] * half + ["B"] * (n - half),
        }
    )


def test_tuning_persistence_and_importance(tmp_path):
    """非线性关系 → 树模型胜出并被调优；模型持久化可重载预测；产出特征重要性。"""
    import joblib

    from services.tracking import TrackingStore

    rng = np.random.default_rng(3)
    df = pd.DataFrame({"x1": rng.uniform(0, 10, 120), "x2": rng.uniform(0, 10, 120)})
    df["y"] = np.round(df["x1"] * df["x2"] / 10 + rng.normal(0, 0.5, 120), 2)
    profile = profile_dataset(df)
    store = TrackingStore(tmp_path / "ml.db")
    result = run_ml_experiment(
        df, profile, target="y", task="regression",
        store=store, dataset_name="nl.csv", persist_dir=tmp_path / "models",
    )
    assert result.status == "ok"
    best = next(m for m in result.models if m.model == result.best_model)
    assert "cv_best" in best.params  # 最佳为可调优树模型且已调优
    assert result.feature_importance and result.tuning_note
    model_files = list((tmp_path / "models").glob("*.joblib"))
    assert len(model_files) == 1
    loaded = joblib.load(model_files[0])
    pred = loaded["pipeline"].predict(df[["x1", "x2"]].head(3))
    assert len(pred) == 3


def test_linear_best_skips_tuning():
    df = build_regression_df()
    profile = profile_dataset(df)
    result = run_ml_experiment(df, profile, target="score", task="regression")
    assert result.best_model == "LinearRegression"
    assert "跳过调优" in result.tuning_note


def test_regression_baseline():
    df = build_regression_df()
    profile = profile_dataset(df)
    result = run_ml_experiment(df, profile, target="score", task="regression")
    assert result.status == "ok"
    assert result.task == "regression"
    linear = next(m for m in result.models if m.model == "LinearRegression")
    assert linear.metrics["R2"] > 0.95
    assert result.best_metric_name == "R2"
    assert len(result.models) >= 2
    assert result.n_train + result.n_test == len(df)
    assert result.dataset_fingerprint
    assert "student_id" in [e.column for e in result.excluded]


def test_classification_baseline_with_auc():
    df = build_classification_df()
    profile = profile_dataset(df)
    result = run_ml_experiment(df, profile, target="label", task="classification")
    assert result.status == "ok"
    best_value = result.best_metric_value
    assert best_value is not None and best_value > 0.9
    assert result.best_metric_name == "F1_macro"
    assert any("ROC_AUC" in m.metrics for m in result.models)


def test_clustering_finds_three_blobs():
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {
            "f1": np.concatenate([rng.normal(0, 0.5, 40), rng.normal(8, 0.5, 40), rng.normal(16, 0.5, 40)]),
            "f2": np.concatenate([rng.normal(0, 0.5, 40), rng.normal(8, 0.5, 40), rng.normal(16, 0.5, 40)]),
        }
    )
    profile = profile_dataset(df)
    result = run_ml_experiment(df, profile, target=None, task="clustering")
    assert result.status == "ok"
    assert result.n_clusters == 3
    assert result.models[0].metrics["silhouette"] > 0.5
    assert sum(result.cluster_sizes.values()) == 120


def test_auto_task_inference():
    df = build_regression_df()
    profile = profile_dataset(df)
    result = run_ml_experiment(df, profile, target="score", task="auto")
    assert result.task == "regression"
    result2 = run_ml_experiment(df, profile, target=None, task="auto")
    assert result2.task == "clustering"


def test_infer_task_rules():
    df = build_regression_df()
    profile = profile_dataset(df)
    assert infer_task(profile, "score") == "regression"
    with pytest.raises(ValueError):
        infer_task(profile, "student_id")  # 标识符列不能作为监督学习目标
    cls_df = build_classification_df()
    cls_profile = profile_dataset(cls_df)
    assert infer_task(cls_profile, "label") == "classification"
    with pytest.raises(ValueError):
        infer_task(profile, "ghost")


def test_select_features_excludes_identifier_and_high_missing():
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "row_id": [f"R{i}" for i in range(60)],
            "v": rng.normal(size=60),
            "sparse": [np.nan] * 40 + list(rng.normal(size=20)),
            "y": rng.normal(size=60),
        }
    )
    profile = profile_dataset(df)
    numeric, categorical, excluded = select_features(df, profile, target="y")
    assert numeric == ["v"]
    assert "row_id" in [e.column for e in excluded]
    assert "sparse" in [e.column for e in excluded]


def test_insufficient_rows_fails():
    df = build_regression_df(10)
    profile = profile_dataset(df)
    result = run_ml_experiment(df, profile, target="score", task="regression")
    assert result.status == "failed"
    assert "不足" in result.reason


def test_unknown_target_raises_for_router():
    df = build_regression_df()
    profile = profile_dataset(df)
    with pytest.raises(ValueError):
        run_ml_experiment(df, profile, target="ghost", task="regression")
    with pytest.raises(ValueError):
        run_ml_experiment(df, profile, target=None, task="regression")


def test_tracking_via_store(tmp_path):
    from services.tracking import TrackingStore

    df = build_regression_df()
    profile = profile_dataset(df)
    store = TrackingStore(tmp_path / "ml.db")
    result = run_ml_experiment(
        df, profile, target="score", task="regression", store=store, dataset_name="t.csv"
    )
    assert result.tracked_uid
    rows = store.list_experiments()
    assert len(rows) == 1
    assert rows[0].uid == result.tracked_uid
    assert rows[0].target == "score"
    assert rows[0].best_model == result.best_model
    detail = store.get_experiment(result.tracked_uid)
    assert detail is not None
    assert detail["dataset_name"] == "t.csv"
    assert isinstance(detail["models_results"], list)
    assert detail["models_results"][0]["model"]
