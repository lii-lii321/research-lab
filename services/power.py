"""功效分析：为实验计划估算"检测中等效应所需的样本量"。

口径（报告与 notes 中标注）：
- 双侧检验、目标功效 0.8；
- 效应量未知时默认中等：d = 0.5（组间/配对）、f = 0.25（ANOVA）、r = 0.3（相关换算 d）；
- Mann-Whitney / Kruskal 用参数法近似（对秩效应偏保守）。
"""
from __future__ import annotations

import numpy as np
from statsmodels.stats.power import FTestAnovaPower, TTestIndPower

_GROUP = {"independent_ttest", "welch_ttest", "mannwhitney", "paired_ttest"}
_MULTI = {"anova", "kruskal"}
_CORR = {"pearson", "spearman"}


def required_n(method: str, alpha: float = 0.05, power: float = 0.8) -> int | None:
    """返回所需样本量观测数；功效分析不支持的方法返回 None。"""
    try:
        if method in _GROUP:
            return int(np.ceil(TTestIndPower().solve_power(effect_size=0.5, alpha=alpha, power=power, ratio=1.0)))
        if method in _MULTI:
            return int(np.ceil(FTestAnovaPower().solve_power(effect_size=0.25, alpha=alpha, power=power, k_groups=3)))
        if method in _CORR:
            r = 0.3
            d = 2 * r / (1 - r * r) ** 0.5
            return int(np.ceil(TTestIndPower().solve_power(effect_size=d, alpha=alpha, power=power, ratio=1.0)))
    except Exception:
        return None
    return None
