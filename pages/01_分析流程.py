"""01_分析流程页面。"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from models.schemas import ExperimentRecord, ProfileReport
from services.executor import run_experiment
from services.llm import LLMError, get_llm_client
from services.planner import generate_experiment_plan
from services.report import build_report
from services.report_images import collect_profile_images
from services.reports_store import save_report
from services.repro import build_repro_script
from services.research_questions import generate_research_questions_auto
from services.tracking import TrackingStore
from ui.components import column_table, render_result_chart
from ui.errors import show_error_card
from ui.page_state import cached_profile, invalidate_on_data_change, load_source
from utils.charts import (
    plotly_heatmap,
)

LEVEL_ICON = {"critical": "🔴", "warning": "🟡", "info": "🔵"}
TASK_CN = {"regression": "回归", "classification": "分类", "clustering": "聚类"}


def main() -> None:
    st.title("① 分析流程")
    st.caption("从一份 CSV 到一份可复现的研究报告 —— 所有数字来自真实执行的代码")

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
    render_flow(df, report)


def render_flow(df: pd.DataFrame, report: ProfileReport) -> None:
    d = report.dataset
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("行数", f"{d.n_rows:,}")
    m2.metric("列数", d.n_cols)
    m3.metric("总体缺失率", f"{d.missing_rate:.1%}")
    m4.metric("重复行", f"{d.duplicate_rows:,}")
    m5.metric("内存占用", f"{d.memory_mb:.2f} MB")

    st.subheader("第一步 · 数据质量提示")
    if report.warnings:
        for w in report.warnings:
            st.markdown(f"{LEVEL_ICON[w.level]} `{w.code}` — {w.message}")
    else:
        st.success("未发现明显的质量问题。")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("字段概览")
        st.dataframe(column_table(report), use_container_width=True, hide_index=True)
    with right:
        st.subheader("目标变量候选")
        if report.target_candidates:
            for t in report.target_candidates:
                st.markdown(f"**{t.column}** — {t.reason}")
        else:
            st.caption("未识别到明显的目标变量。")
        st.divider()
        st.subheader("高相关字段对（|r| ≥ 0.8）")
        if report.correlations:
            st.dataframe(
                pd.DataFrame(
                    [
                        {"字段 A": p.column_a, "字段 B": p.column_b, "相关系数": p.coefficient}
                        for p in report.correlations
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("数值字段间未发现高相关。")

    missing = sorted(
        ((c.name, c.missing_rate) for c in report.columns if c.missing_rate > 0), key=lambda x: -x[1]
    )
    if missing:
        st.subheader("缺失率最高的字段")
        chart_df = pd.DataFrame({name: [rate] for name, rate in missing[:10]})
        st.bar_chart(chart_df.T.rename(columns={0: "缺失率"}))

    numeric_cols = [c.name for c in report.columns if c.type == "numeric"]
    heatmap = plotly_heatmap(df, numeric_cols)
    if heatmap is not None:
        st.subheader("数值字段相关矩阵")
        st.plotly_chart(heatmap, use_container_width=True)

    st.download_button(
        "下载画像报告（JSON）",
        data=json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        file_name="profile_report.json",
        mime="application/json",
    )

    st.divider()
    client = get_llm_client()
    st.subheader("第二步 · 研究问题")
    if client is None:
        st.caption("未配置 LLM：当前使用规则模式提出问题（可复制 `.env.example` 为 `.env` 填入 key 启用 LLM）。")
    if st.button("生成研究问题", type="primary"):
        with st.spinner("正在基于数据画像提出研究问题…"):
            try:
                st.session_state["rqs"] = generate_research_questions_auto(df, report, client)
                for key in ("plan", "result", "history", "report"):
                    st.session_state.pop(key, None)
            except (LLMError, ValueError) as exc:
                show_error_card(str(exc))
    rqs = st.session_state.get("rqs") or []
    rq = None
    if rqs:
        labels = [f"{q.id} · {q.question}" for q in rqs]
        choice = st.radio("选择研究问题", labels, label_visibility="collapsed")
        rq = rqs[labels.index(choice)]
        source_cn = "LLM 提出" if rq.source == "llm" else "规则生成"
        st.caption(
            f"{rq.rationale}　｜　变量：{'、'.join(rq.variables)}　｜　建议：{rq.suggested_method}　｜　{source_cn}"
        )

        st.divider()
        st.subheader("第三步 · 实验计划")
        if st.button("生成实验计划", type="primary"):
            with st.spinner("正在设计实验计划…"):
                try:
                    st.session_state["plan"] = generate_experiment_plan(rq, report, client)
                except (LLMError, ValueError) as exc:
                    show_error_card(str(exc))
        plan = st.session_state.get("plan")
        if plan and plan.question_id == rq.id:
            source_cn = "LLM 设计" if plan.source == "llm" else "规则生成"
            p1, p2, p3 = st.columns(3)
            p1.metric("统计方法", plan.method)
            p2.metric("显著性 α", plan.alpha)
            p3.metric("计划来源", source_cn)
            st.markdown(f"**假设**：{plan.hypothesis}")
            st.markdown(f"**H0**：{plan.h0}")
            st.markdown(f"**H1**：{plan.h1}")
            if plan.notes:
                st.caption(plan.notes)
            st.download_button(
                "下载实验计划（JSON）",
                data=json.dumps(plan.model_dump(mode="json"), ensure_ascii=False, indent=2),
                file_name=f"{plan.experiment_id or 'experiment'}.json",
                mime="application/json",
            )

            st.divider()
            st.subheader("第四步 · 运行实验（真实执行）")
            if st.button("执行统计检验", type="primary"):
                with st.spinner("正在真实执行统计检验…"):
                    try:
                        new_result = run_experiment(
                            plan,
                            df,
                            client,
                            store=TrackingStore(),
                            dataset_name=st.session_state.get("dataset_name", "dataset"),
                        )
                        st.session_state["result"] = new_result
                        if new_result.status == "ok":
                            history = [
                                h
                                for h in (st.session_state.get("history") or [])
                                if h.plan.experiment_id != plan.experiment_id
                            ]
                            history.append(ExperimentRecord(plan=plan, result=new_result))
                            st.session_state["history"] = history
                    except ValueError as exc:
                        show_error_card(str(exc))
                        st.session_state.pop("result", None)
            result = st.session_state.get("result")
            if result and result.experiment_id == plan.experiment_id:
                if result.status != "ok":
                    st.warning(f"执行未完成：{result.reason}")
                else:
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric(
                        "统计量",
                        "—" if result.statistic is None else f"{result.statistic_name} = {result.statistic:.3f}",
                    )
                    r2.metric(
                        "p 值",
                        "—"
                        if result.p_value is None
                        else ("< 0.001" if result.p_value < 1e-3 else f"{result.p_value:.4f}"),
                    )
                    r3.metric(
                        "效应量",
                        "—"
                        if result.effect_size is None
                        else f"{result.effect_name} = {result.effect_size:.3f}",
                    )
                    r4.metric("有效样本", f"{result.n_used} / {result.n_used + result.n_dropped}")
                    badge = "✅ 拒绝 H0" if result.decision == "reject_h0" else "➖ 未能拒绝 H0"
                    st.markdown(f"**统计结论**：{badge}（α = {result.alpha}）")
                    st.info(result.interpretation)
                    render_result_chart(df, plan, result)
                    repro_dl, result_dl = st.columns(2)
                    repro_dl.download_button(
                        "下载复现脚本（.py）",
                        data=build_repro_script(plan),
                        file_name=f"{plan.experiment_id}_repro.py",
                        mime="text/x-python",
                    )
                    result_dl.download_button(
                        "下载实验结果（JSON）",
                        data=json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
                        file_name=f"{plan.experiment_id}-result.json",
                        mime="application/json",
                    )

    st.divider()
    st.subheader("第五步 · 研究报告")
    history = st.session_state.get("history") or []
    st.caption(
        f"将汇总当前数据画像、研究问题与 {len(history)} 个已执行实验，"
        "生成 Markdown + HTML 双格式报告。"
    )
    if st.button("生成研究报告", type="primary"):
        with st.spinner("正在汇总生成研究报告…"):
            questions = st.session_state.get("rqs") or []
            images = collect_profile_images(df, report)
            markdown, html_doc = build_report(
                report,
                st.session_state.get("dataset_name", "dataset"),
                questions,
                history,
                st.session_state.get("ml_result"),
                images=images,
            )
            saved = save_report(
                st.session_state.get("dataset_name", "dataset").rsplit(".", 1)[0] or "report",
                markdown,
                html_doc,
                images=images,
            )
            st.session_state["report"] = {
                "md": markdown,
                "html": html_doc,
                "name": saved["name"],
            }
    rep = st.session_state.get("report")
    if rep:
        st.caption(f"报告已沉淀到报告库：{rep['name']}（可在 ⑤ 报告库 回看）")
        dl_md, dl_html = st.columns(2)
        dl_md.download_button(
            "下载报告（.md）",
            data=rep["md"],
            file_name=f"{rep['name']}_report.md",
            mime="text/markdown",
        )
        dl_html.download_button(
            "下载报告（.html）",
            data=rep["html"],
            file_name=f"{rep['name']}_report.html",
            mime="text/html",
        )
        with st.expander("报告预览（Markdown）", expanded=False):
            st.markdown(rep["md"])
