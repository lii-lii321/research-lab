"""大文件采样画像测试（S2.3）。"""
import numpy as np
import pandas as pd

from services.profiler import SAMPLE_ROWS, profile_dataset


def test_small_data_not_sampled():
    df = pd.DataFrame({"x": np.arange(100.0), "y": np.arange(100.0) * 2})
    report = profile_dataset(df)
    assert report.dataset.sampled is False
    assert report.dataset.sample_note == ""
    assert report.dataset.n_rows == 100


def test_large_data_sampled_with_note():
    n = SAMPLE_ROWS + 10_000
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=n), "y": rng.normal(size=n)})
    report = profile_dataset(df, sample=True)
    assert report.dataset.sampled is True
    assert "500,000" in report.dataset.sample_note
    assert report.dataset.n_rows == SAMPLE_ROWS  # overview 统计基于采样后数据


def test_large_data_sampling_disabled():
    n = SAMPLE_ROWS + 5_000
    df = pd.DataFrame({"x": np.arange(float(n))})
    report = profile_dataset(df, sample=False)
    assert report.dataset.sampled is False
    assert report.dataset.n_rows == n
