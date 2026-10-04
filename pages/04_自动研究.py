"""04_自动研究页面。"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from models.schemas import ProfileReport
from services.agent import run_research_agent
from services.llm import get_llm_client
from services.reports_store import save_report
from services.tracking import TrackingStore
from ui.components import render_agent_report_downloads, render_agent_timeline
from ui.errors import show_error_card
from ui.page_state import cached_profile, invalidate_on_data_change, load_source

LEVEL_ICON = {"critical": "🔴", "warning": "🟡", "info": "🔵"}
TASK_CN = {"regression": "回归", "classification": "分类", "clustering": "聚类"}


def render_agent(df: pd.DataFrame, report: ProfileReport) -> None:
    st.subheader("自动研究 Agent")
    st.caption(
        "一句话任务 → 数据画像 → 研究问题 → 统计实验 → ML 基线 → 研究报告，"
        "全自动串联；未配置 LLM 时以规则模式运行（每个环节的来源都会标注）。"
    )
    task = st.text_input("研究任务描述", value="研究影响学生成绩的关键因素", key="agent_task")
    max_q = st.slider("最多研究问题数", 1, 5, 3, key="agent_max")
    with_lit = st.checkbox("检索相关文献（arXiv，需联网）", value=True, key="agent_lit")
    client = get_llm_client()
    if st.button("启动自动研究", type="primary", key="agent_run"):
        with st.spinner("Agent 运行中：画像 → 研究问题 → 统计实验 → ML 基线 → 文献 → 报告…"):
            st.session_state["agent_result"] = run_research_agent(
                df,
                report,
                task_description=task,
                max_questions=max_q,
                client=client,
                store=TrackingStore(),
                dataset_name=st.session_state.get("dataset_name", "dataset"),
                enable_literature=with_lit,
            )
    result = st.session_state.get("agent_result")
    if result is None:
        return
    if result.status != "ok":
        show_error_card(result.reason)
    else:
        ml_desc = "—"
        if result.ml_result and result.ml_result.status == "ok":
            best = result.ml_result
            ml_desc = f"{best.best_model}（{best.best_metric_name} = {best.best_metric_value}）"
        st.success(
            f"完成：{len(result.questions)} 个研究问题 · {len(result.records)} 个统计实验 · "
            f"ML 最佳 {ml_desc} · 总耗时 {result.runtime_seconds} 秒"
        )
        render_agent_timeline(result)
        saved = save_report(f"{result.filename_base}_agent", result.report_markdown, result.report_html)
        st.caption(f"报告已沉淀到报告库：{saved['name']}")
    if result.references:
        with st.expander(f"引用文献（{len(result.references)} 篇）", expanded=False):
            for p in result.references:
                st.markdown(f"- **{p.title}**（{p.year}）— {p.url}")
    if result.report_markdown:
        render_agent_report_downloads(result, "agent")


def main() -> None:
    st.title("④ 自动研究")
    st.caption("一句话任务 → 画像 → 研究问题 → 统计实验 → ML 基线 → 文献 → 报告")

    df, err = load_source()
    if err:
        show_error_card(err)
    if df is None:
        st.info("在左侧选择数据源。")
        st.stop()

    try:
        report = cached_profile(df)
    except ValueError as exc:
        show_error_card(str(exc))
        st.stop()

    invalidate_on_data_change(report)
    render_agent(df, report)





main()

