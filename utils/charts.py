"""matplotlib 图表构建（白底、藏青标题、蓝色主色）。"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

MAX_HEATMAP_COLS = 12
MAX_HISTOGRAM_PLOTS = 6

NAVY = "#1a365d"
BLUE = "#2563eb"
SLATE = "#334155"


def numeric_histograms(df: pd.DataFrame, cols: list[str], max_plots: int = MAX_HISTOGRAM_PLOTS):
    """前 N 个数值列的分布直方图网格；无可绘列返回 None。"""
    cols = [c for c in cols if c in df.columns][:max_plots]
    if not cols:
        return None
    n = len(cols)
    ncols = 2 if n > 1 else 1
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.6 * ncols, 3.2 * nrows), dpi=120, squeeze=False
    )
    for ax, col in zip(axes.flat, cols, strict=False):
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        ax.hist(values, bins=24, color=BLUE, alpha=0.75)
        ax.set_title(col, fontsize=9, color=NAVY)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def missing_matrix(df: pd.DataFrame, max_rows: int = 80):
    """前 max_rows 行的缺失模式灰度矩阵（白=有值，黑=缺失）。"""
    sample = df.head(max_rows)
    if sample.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=120)
    ax.imshow(sample.isna().to_numpy(), aspect="auto", cmap="Greys", vmin=0, vmax=1)
    ax.set_xticks(
        range(len(sample.columns)),
        labels=[str(c) for c in sample.columns],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    ax.set_ylabel(f"行（前 {len(sample)}）", fontsize=8)
    fig.tight_layout()
    return fig


def correlation_heatmap(df: pd.DataFrame, numeric_cols: list[str]):
    cols = [c for c in numeric_cols if c in df.columns][:MAX_HEATMAP_COLS]
    if len(cols) < 2:
        return None
    corr = df[cols].apply(pd.to_numeric, errors="coerce").corr()
    size = 0.6 * len(cols) + 2.5
    fig, ax = plt.subplots(figsize=(size, size - 0.5), dpi=120)
    im = ax.imshow(corr.values, cmap="Blues", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), labels=cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)), labels=cols, fontsize=8)
    if len(cols) <= 8:
        for i in range(len(cols)):
            for j in range(len(cols)):
                v = corr.iloc[i, j]
                if pd.notna(v):
                    ax.text(
                        j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color=NAVY if abs(v) < 0.6 else "#ffffff",
                    )
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def result_figure(df: pd.DataFrame, plan, result):
    method, variables = plan.method, plan.variables
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    try:
        if method in ("pearson", "spearman", "linear_regression") and len(variables) >= 2:
            sub = df[variables[:2]].apply(pd.to_numeric, errors="coerce").dropna()
            x, y = sub.iloc[:, 0].to_numpy(float), sub.iloc[:, 1].to_numpy(float)
            ax.scatter(x, y, s=20, alpha=0.65, color=BLUE, edgecolors="none")
            if x.size >= 2:
                import numpy as np

                slope, intercept = np.polyfit(x, y, 1)
                xs = (x.min(), x.max())
                ax.plot(xs, [slope * v + intercept for v in xs], color=NAVY, linewidth=2)
            ax.set_xlabel(variables[0])
            ax.set_ylabel(variables[1])
        elif (
            method in ("independent_ttest", "welch_ttest", "mannwhitney", "anova", "kruskal")
            and len(variables) >= 2
        ):
            frame = pd.DataFrame(
                {
                    "g": df[variables[0]].astype(str),
                    "v": pd.to_numeric(df[variables[1]], errors="coerce"),
                }
            ).dropna()
            names, data = [], []
            for name, series in frame.groupby("g", sort=True)["v"]:
                names.append(str(name))
                data.append(series.to_numpy(float))
            ax.boxplot(
                data,
                tick_labels=names,
                patch_artist=True,
                boxprops={"facecolor": "#dbeafe", "color": NAVY},
                medianprops={"color": NAVY},
                whiskerprops={"color": SLATE},
                capprops={"color": SLATE},
            )
            ax.set_ylabel(variables[1])
        elif method == "chi2" and len(variables) >= 2:
            ct = pd.crosstab(df[variables[0]].astype(str), df[variables[1]].astype(str))
            ct.plot(
                kind="bar", ax=ax,
                color=[BLUE, "#60a5fa", "#93c5fd", "#bfdbfe", NAVY] * 4,
                width=0.75,
            )
            ax.set_xlabel(variables[0])
            ax.set_ylabel("计数")
            ax.legend(fontsize=8, title=variables[1], title_fontsize=8)
        else:
            plt.close(fig)
            return None
        ax.set_title(f"{method}（n = {result.n_used}）", fontsize=11, color=NAVY)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        return fig
    except Exception:
        plt.close(fig)
        return None
