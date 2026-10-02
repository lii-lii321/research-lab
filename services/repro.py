"""由实验计划生成可独立运行的 scipy 复现脚本。

落实立项书承诺"每个分析结果可溯源到真实执行的代码"：
报告附录与单实验导出都携带这份脚本，替换 DATA_PATH 后可直接运行。
"""
from __future__ import annotations

from models.schemas import ExperimentPlan


def build_repro_script(plan: ExperimentPlan) -> str:
    v = plan.variables
    x, y = v[0], v[1]
    alpha = plan.alpha
    header = (
        '"""复现实验：'
        f"{plan.experiment_id}（{plan.question_id}）\n\n"
        f"假设：{plan.hypothesis}\n"
        f"方法：{plan.method}　变量：{', '.join(v)}　α：{alpha}\n"
        '依赖：pandas、scipy。把 DATA_PATH 替换为你的数据文件后可直接运行。\n'
        '"""\n'
        "import pandas as pd\n"
        "from scipy import stats\n\n"
        'DATA_PATH = "your_data.csv"  # ← 替换为你的数据文件\n'
        "df = pd.read_csv(DATA_PATH)\n"
    )
    if plan.method in ("pearson", "spearman"):
        fn = f"stats.{plan.method}r"
        body = (
            f"sub = df[[{x!r}, {y!r}]].apply(pd.to_numeric, errors=\"coerce\").dropna()\n"
            f"res = {fn}(sub[{x!r}], sub[{y!r}])\n"
        )
    elif plan.method == "paired_ttest":
        body = (
            f"sub = df[[{x!r}, {y!r}]].apply(pd.to_numeric, errors=\"coerce\").dropna()\n"
            f"res = stats.ttest_rel(sub[{x!r}], sub[{y!r}])\n"
        )
    elif plan.method == "linear_regression":
        body = (
            f"sub = df[[{x!r}, {y!r}]].apply(pd.to_numeric, errors=\"coerce\").dropna()\n"
            "import statsmodels.api as sm\n"
            f"X = sm.add_constant(sub[{x!r}])\n"
            f"ols = sm.OLS(sub[{y!r}], X).fit()\n"
            "print(ols.summary())\n"
        )
    elif plan.method in ("independent_ttest", "welch_ttest", "mannwhitney"):
        if plan.method == "mannwhitney":
            call = 'stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")'
        else:
            equal_var = "True" if plan.method == "independent_ttest" else "False"
            call = f"stats.ttest_ind(groups[0], groups[1], equal_var={equal_var})"
        body = (
            f"sub = df[[{x!r}, {y!r}]].dropna()\n"
            f"groups = [s.to_numpy(float) for _, s in sub.groupby(sub[{x!r}].astype(str))[{y!r}]]\n"
            f"res = {call}\n"
        )
    elif plan.method in ("anova", "kruskal"):
        fn = "f_oneway" if plan.method == "anova" else "kruskal"
        body = (
            f"sub = df[[{x!r}, {y!r}]].dropna()\n"
            f"groups = [s.to_numpy(float) for _, s in sub.groupby(sub[{x!r}].astype(str))[{y!r}]]\n"
            f"res = stats.{fn}(*groups)\n"
        )
    elif plan.method == "chi2":
        body = (
            f"ct = pd.crosstab(df[{x!r}].astype(str), df[{y!r}].astype(str))\n"
            "res = stats.chi2_contingency(ct)\n"
        )
    else:
        body = (
            f"sub = df[[{x!r}, {y!r}]].apply(pd.to_numeric, errors=\"coerce\").dropna()\n"
            f"raise SystemExit(f\"方法 {plan.method!r} 暂无自动复现代码\")\n"
        )
    tail = 'print(f"statistic={res.statistic:.6f} p={res.pvalue:.6g}")\n'
    tail += f'print("拒绝 H0" if res.pvalue < {alpha} else "未能拒绝 H0")\n'
    return header + body + tail
