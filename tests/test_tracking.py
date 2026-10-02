import numpy as np
import pandas as pd

from services.tracking import TrackingStore, dataframe_fingerprint


def build_df(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"a": rng.normal(size=50), "b": rng.choice(["x", "y"], 50)})


def test_fingerprint_deterministic_and_sensitive():
    assert dataframe_fingerprint(build_df()) == dataframe_fingerprint(build_df())
    assert dataframe_fingerprint(build_df()) != dataframe_fingerprint(build_df(1))


def test_roundtrip_and_auto_schema(tmp_path):
    store = TrackingStore(tmp_path / "sub" / "track.db")
    df = build_df()
    uid = store.track(
        kind="ml",
        task="regression",
        dataset_name="demo.csv",
        df=df,
        feature_set={"numeric": ["a"], "categorical": ["b"]},
        models_results=[{"model": "LinearRegression", "metrics": {"R2": 0.9}}],
        target="a",
        best_model="LinearRegression",
        best_metric_name="R2",
        best_metric_value=0.9,
        notes="测试",
        runtime_seconds=1.23,
    )
    assert len(uid) == 12
    rows = store.list_experiments()
    assert len(rows) == 1
    row = rows[0]
    assert row.uid == uid and row.dataset_name == "demo.csv" and row.n_rows == 50
    assert row.best_metric_value == 0.9
    detail = store.get_experiment(uid)
    assert detail["feature_set"]["numeric"] == ["a"]
    assert detail["models_results"][0]["metrics"]["R2"] == 0.9
    assert store.get_experiment("missing") is None


def test_list_order_newest_first(tmp_path):
    import time as time_mod

    store = TrackingStore(tmp_path / "t.db")
    df = build_df()
    uid1 = store.track(kind="ml", task="regression", dataset_name="a.csv", df=df, feature_set={}, models_results=[])
    time_mod.sleep(1.1)
    uid2 = store.track(kind="ml", task="clustering", dataset_name="b.csv", df=df, feature_set={}, models_results=[])
    rows = store.list_experiments()
    assert [r.uid for r in rows] == [uid2, uid1]
