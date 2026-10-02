import numpy as np
import pandas as pd
import pytest

from services.profiler import infer_column_type, profile_dataset
from utils.sample_data import build_sample_dataframe


def test_basic_type_inference():
    df = pd.DataFrame(
        {
            "num": [1.0, 2.5, 3.7, 4.1, 5.9],
            "cat_str": ["a", "b", "a", "b", "a"],
            "flag": [True, False, True, False, True],
            "dt": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
            ),
        }
    )
    types = {name: infer_column_type(df[name]) for name in df.columns}
    assert types == {
        "num": "numeric",
        "cat_str": "categorical",
        "flag": "boolean",
        "dt": "datetime",
    }


def test_datetime_from_strings():
    s = pd.Series(["2024-01-01", "2024-01-02", "2024-06-15"] * 10, name="enroll_date")
    assert infer_column_type(s) == "datetime"


def test_identifier_detection():
    assert infer_column_type(pd.Series([f"S{i:04d}" for i in range(100)], name="student_id")) == "identifier"
    assert infer_column_type(pd.Series(range(100), name="row_id")) == "identifier"


def test_short_unique_numeric_is_numeric():
    assert infer_column_type(pd.Series([3.0, 1.0, 2.0], name="a")) == "numeric"


def test_long_unique_text_without_id_name_is_text():
    rng = np.random.default_rng(3)
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf"]
    s = pd.Series([" ".join(rng.choice(words, 8)) for _ in range(60)], name="note")
    assert infer_column_type(s) == "text"


def test_ordinal_codes_become_categorical():
    s = pd.Series([1] * 50 + [2] * 30 + [3] * 15 + [4] * 5, name="study_time")
    assert infer_column_type(s) == "categorical"


def test_outliers_detected():
    values = [float(v) for v in range(1, 31)] + [1000.0]
    report = profile_dataset(pd.DataFrame({"v": values}))
    col = report.columns[0]
    assert col.type == "numeric"
    assert col.outlier_count == 1
    assert col.outlier_values == [1000.0]


def test_high_correlation_pair_and_constant_excluded():
    rng = np.random.default_rng(1)
    x = rng.normal(size=200)
    df = pd.DataFrame(
        {
            "x": x,
            "y": x * 2 + rng.normal(scale=0.01, size=200),
            "z": rng.normal(size=200),
            "const": [7.0] * 200,
        }
    )
    report = profile_dataset(df)
    pairs = {(p.column_a, p.column_b) for p in report.correlations}
    assert ("x", "y") in pairs
    assert all("const" not in (a, b) for a, b in pairs)
    assert any(
        w.code == "CONSTANT_COLUMN" and w.columns == ["const"] for w in report.warnings
    )


def test_class_imbalance_warning():
    df = pd.DataFrame({"ok": ["Yes"] * 95 + ["No"] * 5, "v": np.arange(100.0)})
    report = profile_dataset(df)
    assert any(w.code == "CLASS_IMBALANCE" for w in report.warnings)


def test_high_missing_and_all_null_column():
    df = pd.DataFrame({"a": [np.nan] * 6 + [1.0, 2.0, 3.0, 4.0], "b": [np.nan] * 10})
    report = profile_dataset(df)
    levels = {(w.code, w.level) for w in report.warnings}
    assert ("HIGH_MISSING", "warning") in levels
    assert ("HIGH_MISSING", "critical") in levels
    assert [c.type for c in report.columns if c.name == "b"] == ["empty"]


def test_duplicate_rows_warning():
    df = pd.DataFrame({"v": [1.0, 2, 3, 1.0, 2, 3]})
    report = profile_dataset(df)
    assert report.dataset.duplicate_rows == 3
    assert any(w.code == "DUPLICATE_ROWS" for w in report.warnings)


def test_empty_dataframe_rejected():
    with pytest.raises(ValueError):
        profile_dataset(pd.DataFrame())
    with pytest.raises(ValueError):
        profile_dataset(pd.DataFrame({"a": []}))


def test_target_candidate_suggestion():
    df = pd.DataFrame({"hours": np.arange(50.0), "final_score": np.arange(50.0) * 2 + 10})
    report = profile_dataset(df)
    names = {t.column for t in report.target_candidates}
    assert "final_score" in names


def test_sample_dataset_end_to_end():
    df = build_sample_dataframe()
    report = profile_dataset(df)
    codes = {w.code for w in report.warnings}
    expected = {
        "HIGH_CORRELATION",
        "CLASS_IMBALANCE",
        "DUPLICATE_ROWS",
        "IDENTIFIER_COLUMN",
        "HIGH_MISSING",
        "SKEWED_DISTRIBUTION",
        "OUTLIERS_DETECTED",
    }
    assert expected <= codes
    assert any(t.column == "final_score" for t in report.target_candidates)
    assert report.dataset.missing_rate > 0
    types = {c.name: c.type for c in report.columns}
    assert types["student_id"] == "identifier"
    assert types["study_time"] == "categorical"
    assert types["gender"] == "categorical"
    assert types["final_score"] == "numeric"
