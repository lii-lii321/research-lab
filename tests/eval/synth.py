"""golden 评测集的确定性合成数据构造器（无网络、无 LLM）。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_df(columns: dict[str, str], n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data: dict[str, list] = {}
    for name, kind in columns.items():
        if kind == "numeric":
            data[name] = np.round(rng.normal(60, 10, n), 2)
        elif kind == "skewed":
            data[name] = np.round(rng.lognormal(0.0, 1.6, n), 2)
        elif kind == "categorical2":
            data[name] = rng.choice(["A", "B"], n)
        elif kind == "categorical3":
            data[name] = rng.choice(["A", "B", "C"], n)
        elif kind == "boolean":
            data[name] = rng.choice([True, False], n)
        elif kind == "identifier":
            data[name] = [f"ID{i:05d}" for i in range(n)]
        else:
            raise ValueError(f"未知列类型：{kind}")
    return pd.DataFrame(data)
