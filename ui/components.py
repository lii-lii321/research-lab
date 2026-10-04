"""UI 共享组件：结果图渲染、Agent 时间线、报告下载、字段概览表。"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from models.schemas import TYPE_CN, ExperimentPlan, ProfileReport
from utils.charts import result_figure


def render_result_chart(df: pd.DataFrame, plan: ExperimentPlan, result) -> None:
    fig = result_figure(df, plan, result)
    if fig is not None:
        st.pyplot(fig)
    method, variables = plan.method, plan.variables
    pfig = None
    if method in ("pearson", "spearman", "linear_regression") and len(variables) >= 2:
        from utils.charts import plotly_scatter

        pfig = plotly_scatter(df, variables[0], variables[1])
    elif method in ("independent_ttest", "welch_ttest", "mannwhitney", "anova", "kruskal") and len(variables) >= 2:
        from utils.charts import plotly_box

        pfig = plotly_box(df, variables[0], variables[1])
    elif method == "chi2" and len(variables) >= 2:
        from utils.charts import plotly_bar

        pfig = plotly_bar(df, variables[0], variables[1])
    if pfig is not None:
        st.plotly_chart(pfig, use_container_width=True)


def render_agent_timeline(result) -> None:
    st.subheader("执行时间线")
    icon = {"ok": "✅", "failed": "⚠️"}
    for s in result.steps:
        st.markdown(f"- {icon[s.status]} **{s.name}**（{s.seconds}s）— {s.detail}")


def render_agent_report_downloads(result, key_prefix: str) -> None:
    dl_md, dl_html = st.columns(2)
    dl_md.download_button(
        "下载报告（.md）",
        data=result.report_markdown,
        file_name=f"{result.filename_base}_report.md",
        mime="text/markdown",
        key=f"{key_prefix}_dl_md",
    )
    dl_html.download_button(
        "下载报告（.html）",
        data=result.report_html,
        file_name=f"{result.filename_base}_report.html",
        mime="text/html",
        key=f"{key_prefix}_dl_html",
    )
    with st.expander("报告预览（Markdown）", expanded=False):
        st.markdown(result.report_markdown)


def column_table(report: ProfileReport) -> pd.DataFrame:
    rows = []
    for c in report.columns:
        if c.numeric is not None:
            summary = f"均值 {c.numeric.mean} · 标准差 {c.numeric.std} · 离群值 {c.outlier_count or 0}"
        elif c.type == "datetime" and c.description:
            summary = c.description
        elif c.top_values:
            summary = "、".join(f"{t.value}（{t.rate:.0%}）" for t in c.top_values[:3])
        else:
            summary = "—"
        rows.append(
            {
                "字段": c.name,
                "类型": TYPE_CN.get(c.type, c.type),
                "缺失率": f"{c.missing_rate:.1%}",
                "唯一值": c.n_unique,
                "概要": summary,
            }
        )
    return pd.DataFrame(rows)
