# -*- coding: utf-8 -*-
"""受控统计执行：真实运行 scipy，所有数字来自计算而非生成。"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sps

from models.schemas import ExperimentPlan, ExperimentResult, GroupStat
from services.llm import LLMClient, LLMError
from services.tracking import TrackingStore

CORRELATION_METHODS = {"pearson", "spearman"}
PAIRED_METHODS = {"paired_ttest"}
TWO_GROUP_METHODS = {"independent_ttest", "welch_ttest", "mannwhitney"}
MULTI_GROUP_METHODS = {"anova", "kruskal"}
GROUP_METHODS = TWO_GROUP_METHODS | MULTI_GROUP_METHODS
CHI2_METHODS = {"chi2"}
REGRESSION_METHODS = {"linear_regression"}
UNSUPPORTED_METHODS = {"logistic_regression"}

MIN_PAIRS = 3
MIN_GROUP_N = 2
SPARSE_EXPECTED_N = 5
SPARSE_RATE = 0.2

METHOD_CN = {
    "pearson": "Pearson 相关分析",
    "spearman": "Spearman 秩相关分析",
    "independent_ttest": "独立样本 t 检验",
    "welch_ttest": "Welch t 检验",
    "mannwhitney": "Mann-Whitney U 检验",
    "paired_ttest": "配对样本 t 检验",
    "anova": "单因素方差分析",
    "kruskal": "Kruskal-Wallis 检验",
    "chi2": "卡方独立性检验",
    "linear_regression": "简单线性回归",
}

STRENGTH_BANDS = ((0.7, "强"), (0.4, "中等"), (0.2, "弱"), (0.0, "极弱"))

INTERPRET_SYSTEM = (
    "你是严谨的统计结果解读者。只能基于给定的真实统计数字进行解读，"
    "禁止编造、修改或推测任何未提供的数字。用 2~4 句中文面向研究者解读，"
    "点明结论方向与统计证据强度。"
)


def _round(x, nd: int = 4):
    if x is None:
        return None
    x = float(x)
    if np.isnan(x) or np.isinf(x):
        return None
    return round(x, nd)


def format_p(p: float) -> str:
    return "p < 0.001" if p < 1e-3 else f"p = {p:.3f}"


def _strength_of(r: float) -> str:
    abs_r = abs(r)
    for threshold, label in STRENGTH_BANDS:
        if abs_r >= threshold:
            return label
    return "极弱"


def _failed(plan: ExperimentPlan, reason: str) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=plan.experiment_id,
        question_id=plan.question_id,
        method=plan.method,
        status="failed",
        reason=reason,
        alpha=plan.alpha,
    )


def _numeric_pair(df: pd.DataFrame, x: str, y: str) -> tuple[np.ndarray, np.ndarray, int]:
    sub = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    n_dropped = len(df) - len(sub)
    return sub[x].to_numpy(float), sub[y].to_numpy(float), n_dropped


def _group_arrays(df: pd.DataFrame, group_col: str, outcome_col: str):
    frame = pd.DataFrame(
        {
            "g": df[group_col].astype(str),
            "v": pd.to_numeric(df[outcome_col], errors="coerce"),
        }
    ).dropna()
    grouped: dict[str, np.ndarray] = {}
    for name, series in frame.groupby("g", sort=True)["v"]:
        grouped[str(name)] = series.to_numpy(float)
    n_used = int(sum(v.size for v in grouped.values()))
    return grouped, n_used, len(df) - n_used


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float | None:
    na, nb = a.size, b.size
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
    if pooled == 0 or np.isnan(pooled):
        return None
    return float((a.mean() - b.mean()) / pooled)


def _eta_squared(groups: dict[str, np.ndarray]) -> float | None:
    all_values = np.concatenate(list(groups.values()))
    grand = all_values.mean()
    ss_between = sum(v.size * (v.mean() - grand) ** 2 for v in groups.values())
    ss_total = float(((all_values - grand) ** 2).sum())
    if ss_total == 0:
        return None
    return float(ss_between / ss_total)


def _group_stats(groups: dict[str, np.ndarray]) -> list[GroupStat]:
    return [
        GroupStat(
            group=name,
            n=int(v.size),
            mean=_round(v.mean()),
            std=_round(v.std(ddof=1)),
            median=_round(np.median(v)),
        )
        for name, v in groups.items()
    ]


def _run_correlation(plan: ExperimentPlan, df: pd.DataFrame) -> ExperimentResult:
    x, y, n_dropped = _numeric_pair(df, plan.variables[0], plan.variables[1])
    if x.size < MIN_PAIRS:
        return _failed(plan, f"有效配对观测不足（{x.size} < {MIN_PAIRS}），无法计算相关")
    func = sps.pearsonr if plan.method == "pearson" else sps.spearmanr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = func(x, y)
    raw_p = float(res.pvalue)
    r, p = _round(res.statistic), _round(raw_p)
    if r is None or p is None:
        return _failed(plan, "变量无变异，无法计算相关系数")
    effect_name = "r²" if plan.method == "pearson" else "ρ²"
    return ExperimentResult(
        experiment_id=plan.experiment_id,
        question_id=plan.question_id,
        method=plan.method,
        status="ok",
        alpha=plan.alpha,
        n_used=int(x.size),
        n_dropped=n_dropped,
        statistic=r,
        statistic_name="r" if plan.method == "pearson" else "ρ",
        p_value=p,
        p_value_raw=raw_p,
        effect_size=_round(r * r),
        effect_name=effect_name,
    )


def _run_group_compare(plan: ExperimentPlan, df: pd.DataFrame) -> ExperimentResult:
    grouped, n_used, n_dropped = _group_arrays(df, plan.variables[0], plan.variables[1])
    two_group = plan.method in TWO_GROUP_METHODS
    if two_group and len(grouped) != 2:
        hint = "，可改用 anova / kruskal" if len(grouped) > 2 else ""
        return _failed(plan, f"该方法需要恰好两组，当前检测到 {len(grouped)} 组{hint}")
    if not two_group and len(grouped) < 2:
        return _failed(plan, f"分组后不足 2 组（当前 {len(grouped)} 组），无法进行组间比较")
    arrays = list(grouped.values())
    if plan.method == "independent_ttest":
        res = sps.ttest_ind(arrays[0], arrays[1], equal_var=True)
        stat_name, effect, effect_name = "t", _cohen_d(arrays[0], arrays[1]), "Cohen's d"
    elif plan.method == "welch_ttest":
        res = sps.ttest_ind(arrays[0], arrays[1], equal_var=False)
        stat_name, effect, effect_name = "t", _cohen_d(arrays[0], arrays[1]), "Cohen's d"
    elif plan.method == "mannwhitney":
        res = sps.mannwhitneyu(arrays[0], arrays[1], alternative="two-sided")
        rank_biserial = 1.0 - 2.0 * float(res.statistic) / (arrays[0].size * arrays[1].size)
        stat_name, effect, effect_name = "U", rank_biserial, "rank-biserial r"
    elif plan.method == "anova":
        res = sps.f_oneway(*arrays)
        stat_name, effect, effect_name = "F", _eta_squared(grouped), "η²"
    else:
        res = sps.kruskal(*arrays)
        k_groups = len(arrays)
        n_total = int(sum(a.size for a in arrays))
        epsilon_sq = (float(res.statistic) - k_groups + 1.0) / (n_total - k_groups)
        stat_name, effect, effect_name = "H", epsilon_sq, "ε²"
    raw_p = float(res.pvalue)
    stat, p = _round(res.statistic), _round(raw_p)
    if stat is None or p is None:
        return _failed(plan, "组内数据无变异，检验无法计算")
    return ExperimentResult(
        experiment_id=plan.experiment_id,
        question_id=plan.question_id,
        method=plan.method,
        status="ok",
        alpha=plan.alpha,
        n_used=n_used,
        n_dropped=n_dropped,
        statistic=stat,
        statistic_name=stat_name,
        p_value=p,
        p_value_raw=raw_p,
        effect_size=_round(effect) if effect is not None else None,
        effect_name=effect_name,
        groups=_group_stats(grouped),
    )


def _run_paired(plan: ExperimentPlan, df: pd.DataFrame) -> ExperimentResult:
    x, y, n_dropped = _numeric_pair(df, plan.variables[0], plan.variables[1])
    if x.size < MIN_PAIRS:
        return _failed(plan, f"有效配对观测不足（{x.size} < {MIN_PAIRS}）")
    res = sps.ttest_rel(x, y)
    diff = x - y
    std_diff = diff.std(ddof=1)
    dz = float(diff.mean() / std_diff) if std_diff and not np.isnan(std_diff) else None
    raw_p = float(res.pvalue)
    stat, p = _round(res.statistic), _round(raw_p)
    if stat is None or p is None:
        return _failed(plan, "差值无变异，配对检验无法计算")
    return ExperimentResult(
        experiment_id=plan.experiment_id,
        question_id=plan.question_id,
        method=plan.method,
        status="ok",
        alpha=plan.alpha,
        n_used=int(x.size),
        n_dropped=n_dropped,
        statistic=stat,
        statistic_name="t",
        p_value=p,
        p_value_raw=raw_p,
        effect_size=_round(dz) if dz is not None else None,
        effect_name="Cohen's dz" if dz is not None else "",
    )


def _run_chi2(plan: ExperimentPlan, df: pd.DataFrame) -> ExperimentResult:
    frame = df[[plan.variables[0], plan.variables[1]]].dropna().astype(str)
    if frame.empty:
        return _failed(plan, "两个变量没有同时非空的观测")
    ct = pd.crosstab(frame[plan.variables[0]], frame[plan.variables[1]])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return _failed(
            plan, f"列联表维度不足（当前 {ct.shape[0]}×{ct.shape[1]}，至少需要 2×2）"
        )
    chi2_stat, p_chi2, _dof, expected = sps.chi2_contingency(ct)
    n = int(ct.values.sum())
    cramers_v = float(np.sqrt(chi2_stat / (n * (min(ct.shape) - 1))))
    sparse_rate = float((expected < SPARSE_EXPECTED_N).mean())
    sparse = bool((expected < SPARSE_EXPECTED_N).any())
    if sparse and ct.shape == (2, 2):
        odds_ratio, raw_p = sps.fisher_exact(ct)
        stat, stat_name, test_used = float(odds_ratio), "Fisher OR", "fisher_exact"
    else:
        stat, stat_name, raw_p, test_used = float(chi2_stat), "χ²", float(p_chi2), "chi2"
    result = ExperimentResult(
        experiment_id=plan.experiment_id,
        question_id=plan.question_id,
        method=plan.method,
        status="ok",
        alpha=plan.alpha,
        n_used=n,
        n_dropped=len(df) - n,
        statistic=_round(stat),
        statistic_name=stat_name,
        p_value=_round(raw_p),
        p_value_raw=raw_p,
        effect_size=_round(cramers_v),
        effect_name="Cramér's V",
        contingency=ct.values.tolist(),
    )
    result.extra["test_used"] = test_used
    if sparse_rate > SPARSE_RATE:
        result.extra["sparse_expected_rate"] = _round(sparse_rate) or 0.0001
    return result


def _run_regression(plan: ExperimentPlan, df: pd.DataFrame) -> ExperimentResult:
    x, y, n_dropped = _numeric_pair(df, plan.variables[0], plan.variables[1])
    if x.size < MIN_PAIRS:
        return _failed(plan, f"有效配对观测不足（{x.size} < {MIN_PAIRS}）")
    res = sps.linregress(x, y)
    raw_p = float(res.pvalue)
    r, p = _round(res.rvalue), _round(raw_p)
    if r is None or p is None:
        return _failed(plan, "变量无变异，回归无法计算")
    return ExperimentResult(
        experiment_id=plan.experiment_id,
        question_id=plan.question_id,
        method=plan.method,
        status="ok",
        alpha=plan.alpha,
        n_used=int(x.size),
        n_dropped=n_dropped,
        statistic=r,
        statistic_name="r",
        p_value=p,
        p_value_raw=raw_p,
        effect_size=_round(r * r),
        effect_name="r²",
        extra={
            "slope": float(res.slope),
            "intercept": float(res.intercept),
            "stderr": float(res.stderr),
        },
    )


_RUNNERS = {
    **{m: _run_correlation for m in CORRELATION_METHODS},
    **{m: _run_group_compare for m in GROUP_METHODS},
    **{m: _run_paired for m in PAIRED_METHODS},
    **{m: _run_chi2 for m in CHI2_METHODS},
    **{m: _run_regression for m in REGRESSION_METHODS},
}

RUNNER_METHODS = frozenset(_RUNNERS)


def rule_interpretation(result: ExperimentResult, plan: ExperimentPlan) -> str:
    method_cn = METHOD_CN.get(plan.method, plan.method)
    p_txt = format_p(result.p_value)
    if result.decision == "reject_h0":
        decision = f"p 值小于 α = {result.alpha}，统计上显著，拒绝 H0。"
    else:
        decision = f"p 值不小于 α = {result.alpha}，统计上不显著，未能拒绝 H0（这不等于证明 H0 成立）。"
    if plan.method in CORRELATION_METHODS or plan.method in REGRESSION_METHODS:
        direction = "正" if (result.statistic or 0) > 0 else "负"
        strength = _strength_of(result.statistic or 0.0)
        return (
            f"{method_cn}基于 {result.n_used} 个有效观测：{result.statistic_name} = "
            f"{result.statistic:.3f}，{p_txt}。{decision}两变量呈{strength}{direction}相关。"
        )
    if plan.method in GROUP_METHODS:
        groups_desc = "、".join(
            f"{g.group}（n={g.n}，均值 {g.mean}）" for g in result.groups if g.mean is not None
        )
        effect = (
            f"效应量 {result.effect_name} = {result.effect_size:.3f}。"
            if result.effect_size is not None
            else ""
        )
        return (
            f"{method_cn}基于 {result.n_used} 个有效观测：{groups_desc}。"
            f"统计量 {result.statistic_name} = {result.statistic:.3f}，{p_txt}。{effect}{decision}"
        )
    if plan.method in PAIRED_METHODS:
        effect = (
            f"效应量 {result.effect_name} = {result.effect_size:.3f}。"
            if result.effect_size is not None
            else ""
        )
        return (
            f"{method_cn}基于 {result.n_used} 个配对观测：统计量 {result.statistic_name} = "
            f"{result.statistic:.3f}，{p_txt}。{effect}{decision}"
        )
    if plan.method in CHI2_METHODS:
        if result.extra.get("test_used") == "fisher_exact":
            method_cn = "Fisher 精确检验（2×2 稀疏表自动切换）"
        rows, cols = (
            (len(result.contingency), len(result.contingency[0])) if result.contingency else (0, 0)
        )
        text = (
            f"{method_cn}基于 {result.n_used} 个观测（{rows}×{cols} 列联表）：{result.statistic_name} = "
            f"{result.statistic:.3f}，{p_txt}。{decision}"
        )
        if (
            result.extra.get("sparse_expected_rate", 0) > SPARSE_RATE
            and result.extra.get("test_used") != "fisher_exact"
        ):
            text += "注意：超过 20% 的单元格期望频数小于 5，卡方近似可能不可靠。"
        return text
    return f"{method_cn}：统计量 {result.statistic}，{p_txt}。{decision}"


def llm_interpretation(result: ExperimentResult, plan: ExperimentPlan, client: LLMClient) -> str:
    payload = {
        "hypothesis": plan.hypothesis,
        "h0": plan.h0,
        "h1": plan.h1,
        "alpha": result.alpha,
        "method": METHOD_CN.get(plan.method, plan.method),
        "n_used": result.n_used,
        "statistic": {"name": result.statistic_name, "value": result.statistic},
        "p_value": result.p_value,
        "effect_size": (
            {"name": result.effect_name, "value": result.effect_size}
            if result.effect_size is not None
            else None
        ),
        "groups": [g.model_dump() for g in result.groups] or None,
        "decision": result.decision,
    }
    text = client.chat(
        INTERPRET_SYSTEM, json.dumps(payload, ensure_ascii=False), temperature=0.2
    ).strip()
    if not text:
        raise LLMError("LLM 返回空解读")
    return text


def run_experiment(
    plan: ExperimentPlan,
    df: pd.DataFrame,
    client: LLMClient | None = None,
    store: TrackingStore | None = None,
    dataset_name: str = "dataset",
) -> ExperimentResult:
    missing = [v for v in plan.variables if v not in df.columns]
    if missing:
        raise ValueError(f"变量不在数据集中：{'、'.join(missing)}")
    if len(plan.variables) < 2:
        raise ValueError("至少需要两个变量")
    if plan.method in UNSUPPORTED_METHODS:
        return _failed(plan, "逻辑回归属于 ML 实验范畴，将在 Phase 2 提供")
    runner = _RUNNERS.get(plan.method)
    if runner is None:
        return _failed(plan, f"暂不支持的方法：{plan.method}")
    result = runner(plan, df)
    if result.status != "ok":
        result.interpretation = result.reason
        result.interpretation_source = "none"
        return result
    if result.p_value_raw is not None or result.p_value is not None:
        raw = result.p_value_raw if result.p_value_raw is not None else result.p_value
        result.decision = "reject_h0" if raw < plan.alpha else "fail_to_reject_h0"
    if store is not None:
        try:
            store.track(
                kind="stats",
                task=plan.method,
                dataset_name=dataset_name,
                df=df,
                feature_set={
                    "variables": plan.variables,
                    "alpha": plan.alpha,
                    "question_id": plan.question_id,
                },
                models_results=[
                    {
                        "method": plan.method,
                        "statistic": {"name": result.statistic_name, "value": result.statistic},
                        "p_value": result.p_value,
                        "effect_size": (
                            {"name": result.effect_name, "value": result.effect_size}
                            if result.effect_size is not None
                            else None
                        ),
                        "decision": result.decision,
                        "plan_source": plan.source,
                    }
                ],
                target=None,
                best_model=plan.method,
                best_metric_name="p_value",
                best_metric_value=(
                    result.p_value_raw if result.p_value_raw is not None else result.p_value
                ),
                notes=plan.hypothesis,
                runtime_seconds=0.0,
            )
        except Exception:
            pass  # 追踪失败绝不影响实验结果本身
    if client is not None:
        try:
            result.interpretation = llm_interpretation(result, plan, client)
            result.interpretation_source = "llm"
        except LLMError:
            result.interpretation = rule_interpretation(result, plan)
            result.interpretation_source = "rule"
    else:
        result.interpretation = rule_interpretation(result, plan)
        result.interpretation_source = "rule"
    return result

