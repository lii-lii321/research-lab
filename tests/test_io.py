from io import BytesIO

import pandas as pd
import pytest

from utils.io import read_tabular


def test_read_utf8_csv():
    content = b"name,score\nAnn,90\nBob,80\n"
    df = read_tabular("t.csv", content)
    assert list(df.columns) == ["name", "score"]
    assert len(df) == 2


def test_read_gbk_csv():
    content = "姓名,成绩\n李雷,95\n韩梅,88\n".encode("gbk")
    df = read_tabular("t.csv", content)
    assert list(df.columns) == ["姓名", "成绩"]


def test_read_xlsx():
    buf = BytesIO()
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_excel(buf, index=False, engine="openpyxl")
    df = read_tabular("t.xlsx", buf.getvalue())
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


def test_rejects_unsupported_suffix():
    with pytest.raises(ValueError):
        read_tabular("t.parquet", b"1")


def test_rejects_empty_content():
    with pytest.raises(ValueError):
        read_tabular("t.csv", b"")
