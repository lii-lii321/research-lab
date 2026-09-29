# -*- coding: utf-8 -*-
"""AI Data Research Lab — Streamlit 前端（分析流程 / ML 实验室 / 实验追踪）。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from models.schemas import ExperimentPlan, ExperimentRecord, MLExperimentResult, ProfileReport, TYPE_CN
from services.agent import run_research_agent
from services.executor import run_experiment
from services.llm import LLMError, get_llm_client
from services.ml_lab import run_ml_experiment
from services.planner import generate_experiment_plan
from services.profiler import profile_dataset
from services.report import build_report
from services.research_questions import generate_research_questions_auto
from services.tracking import TrackingStore
from utils.charts import correlation_heatmap, result_figure
from utils.io import read_tabular

st.set_page_config(page_title="AI Data Research Lab", page_icon="🗂", layout="wide")
st.markdown(
    """
    <style>
      .stApp { background: #ffffff; }
      h1, h2, h3 { color: #1a365d; font-weight: 600; }
      section[data-testid="stSidebar"] { background: #f8fafc; }
      .block-container { padding-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

SAMPLE_PATH = Path(__file__).resolve().parent / "data" / "samples" / "student_performance.csv"
LEVEL_ICON = {"critical": "🔴", "warning": "🟡", "info": "🔵"}
TASK_CN = {"regression": "回归", "classification": "分类", "clustering": "聚类"}


def load_source() -> tuple[pd.DataFrame | None, str | None]:
    with st.sidebar:
        st.header("数据源")
        uploaded = st.file_uploader("上传 CSV / Excel（≤50MB）", type=["csv", "xlsx"])
        use_sample = st.checkbox("使用示例数据（学生成绩）", value=uploaded is None)
        if uploaded is not None:
            try:
                st.session_state["dataset_name"] = uploaded.name
                return read_tabular(uploaded.name, uploaded.getvalue()), None
            except ValueError as exc:
                return None, str(exc)
        if use_sample:
            if not SAMPLE_PATH.exists():
                return None, "示例数据不存在：先运行 scripts/generate_sample.py"
            try:
                st.session_state["dataset_name"] = SAMPLE_PATH.name
                return read_tabular(SAMPLE_PATH.name, SAMPLE_PATH.read_bytes()), None
            except ValueError as exc:
                return None, str(exc)
    return None, None


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


def render_result_chart(df: pd.DataFrame, plan: ExperimentPlan, result) -> None:
    fig = result_figure(df, plan, result)
    if fig is not None:
        st.pyplot(fig)


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
    heatmap = correlation_heatmap(df, numeric_cols)
    if heatmap is not None:
        st.subheader("数值字段相关矩阵")
        st.pyplot(heatmap)

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
                st.error(f"生成失败：{exc}")
    rqs = st.session_state.get("rqs") or []
    rq = None
    if rqs:
        labels = [f"{q.id} · {q.question}" for q in rqs]
        choice = st.radio("选择研究问题", labels, label_visibility="collapsed")
        rq = rqs[labels.index(choice)]
        source_cn = "LLM 提出" if rq.source == "llm" else "规则生成"
        st.caption(f"{rq.rationale}　｜　变量：{'、'.join(rq.variables)}　｜　建议：{rq.suggested_method}　｜　{source_cn}")

        st.divider()
        st.subheader("第三步 · 实验计划")
        if st.button("生成实验计划", type="primary"):
            with st.spinner("正在设计实验计划…"):
                try:
                    st.session_state["plan"] = generate_experiment_plan(rq, report, client)
                except (LLMError, ValueError) as exc:
                    st.error(f"生成失败：{exc}")
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
                        new_result = run_experiment(plan, df, client)
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
                        st.error(f"执行失败：{exc}")
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
                    st.download_button(
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
            markdown, html_doc = build_report(
                report,
                st.session_state.get("dataset_name", "dataset"),
                questions,
                history,
                st.session_state.get("ml_result"),
            )
            st.session_state["report"] = {
                "md": markdown,
                "html": html_doc,
                "name": st.session_state.get("dataset_name", "dataset").rsplit(".", 1)[0]
                or "report",
            }
    rep = st.session_state.get("report")
    if rep:
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


def render_ml_lab(df: pd.DataFrame, report: ProfileReport) -> None:
    st.subheader("ML 实验室 · 多模型基线对比")
    st.caption(
        "自动选择特征（排除标识符与高缺失列），80/20 划分（随机种子 42，可复现），"
        "多模型真实训练；每次运行写入实验追踪库。不选目标即为聚类模式。"
    )
    supervised_cols = [
        c.name for c in report.columns if c.type in ("numeric", "categorical", "boolean")
    ]
    options = ["（无目标 · 聚类模式）"] + supervised_cols
    col_a, col_b = st.columns(2)
    sel_target = col_a.selectbox("目标变量", options, index=0, key="ml_target")
    sel_task = col_b.selectbox(
        "任务", ["auto", "regression", "classification", "clustering"], index=0, key="ml_task"
    )
    target = None if sel_target.startswith("（") else sel_target
    if st.button("运行 ML 基线实验", type="primary", key="run_ml"):
        with st.spinner("正在训练与评估模型…"):
            try:
                st.session_state["ml_result"] = run_ml_experiment(
                    df,
                    report,
                    target=target,
                    task=sel_task,
                    store=TrackingStore(),
                    dataset_name=st.session_state.get("dataset_name", "dataset"),
                )
            except ValueError as exc:
                st.error(f"无法运行：{exc}")
                st.session_state.pop("ml_result", None)
    result: MLExperimentResult | None = st.session_state.get("ml_result")
    if result is None:
        return
    if result.status != "ok":
        st.warning(f"实验未完成：{result.reason}")
        return
    st.success(
        f"完成：最佳模型 {result.best_model}（{result.best_metric_name} = "
        f"{result.best_metric_value}）· 追踪 uid：{result.tracked_uid}"
    )
    meta1, meta2, meta3, meta4 = st.columns(4)
    meta1.metric("任务", TASK_CN.get(result.task, result.task))
    if result.task == "clustering":
        meta2.metric("簇数 k", result.n_clusters)
        meta3.metric("数值特征", len(result.features_numeric))
        meta4.metric("数据指纹", result.dataset_fingerprint[:8])
    else:
        meta2.metric("训练 / 测试", f"{result.n_train} / {result.n_test}")
        meta3.metric(
            "特征",
            f"数值 {len(result.features_numeric)} + 类别 {len(result.features_categorical)}",
        )
        meta4.metric("数据指纹", result.dataset_fingerprint[:8])
    metrics_df = pd.DataFrame(
        [
            {"模型": m.model, **{k: round(v, 4) for k, v in m.metrics.items()}, "耗时(秒)": m.train_seconds}
            for m in result.models
        ]
    )
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    if result.task == "clustering" and result.cluster_sizes:
        st.caption("簇规模：" + "、".join(f"{k}：{v}" for k, v in result.cluster_sizes.items()))
    if result.excluded:
        st.caption("已排除：" + "；".join(f"{e.column}（{e.reason}）" for e in result.excluded))


def render_tracking() -> None:
    st.subheader("实验追踪")
    st.caption(
        "ML 实验自动入库（SQLite：`data/tracking/experiments.db`）——记录数据指纹、特征集、"
        "模型、超参、指标与时间戳。点击行查看详情；同指纹实验可看指标趋势。"
    )
    store = TrackingStore()
    rows = store.list_experiments(50)
    if not rows:
        st.info("暂无实验记录——到「ML 实验室」或「自动研究」运行一次实验即可入库。")
        return
    table = pd.DataFrame([t.model_dump() for t in rows]).rename(
        columns={
            "uid": "UID",
            "created_at": "时间",
            "kind": "类型",
            "task": "任务",
            "target": "目标",
            "dataset_name": "数据集",
            "n_rows": "行数",
            "best_model": "最佳模型",
            "best_metric_name": "最佳指标",
            "best_metric_value": "最佳值",
        }
    )
    selection = st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="track_table",
    )
    selected = selection.selection.rows if hasattr(selection, "selection") else []
    if selected:
        row = rows[selected[0]]
        detail = store.get_experiment(row.uid)
        if detail:
            with st.expander(f"实验详情 {row.uid}", expanded=True):
                st.json(detail)
        same = [
            t for t in store.list_experiments(100)
            if t.dataset_name == row.dataset_name and t.task == row.task and t.best_metric_value is not None
        ]
        if len(same) >= 2:
            same.sort(key=lambda t: t.created_at)
            trend = pd.DataFrame(
                {
                    "最佳值": [t.best_metric_value for t in same],
                },
                index=[t.created_at for t in same],
            )
            st.subheader("同数据集同任务 · 最佳指标趋势")
            st.line_chart(trend)


def render_agent(df: pd.DataFrame, report: ProfileReport) -> None:
    st.subheader("自动研究 Agent")
    st.caption(
        "一句话任务 → 数据画像 → 研究问题 → 统计实验 → ML 基线 → 研究报告，"
        "全自动串联；未配置 LLM 时以规则模式运行（每个环节的来源都会标注）。"
    )
    task = st.text_input("研究任务描述", value="研究影响学生成绩的关键因素", key="agent_task")
    max_q = st.slider("最多研究问题数", 1, 5, 3, key="agent_max")
    client = get_llm_client()
    if st.button("启动自动研究", type="primary", key="agent_run"):
        with st.spinner("Agent 运行中：画像 → 研究问题 → 统计实验 → ML 基线 → 报告…"):
            st.session_state["agent_result"] = run_research_agent(
                df,
                report,
                task_description=task,
                max_questions=max_q,
                client=client,
                store=TrackingStore(),
                dataset_name=st.session_state.get("dataset_name", "dataset"),
            )
    result = st.session_state.get("agent_result")
    if result is None:
        return
    if result.status != "ok":
        st.error(f"自动研究未完成：{result.reason}")
    else:
        ml_desc = "—"
        if result.ml_result and result.ml_result.status == "ok":
            ml_desc = f"{result.ml_result.best_model}（{result.ml_result.best_metric_name} = {result.ml_result.best_metric_value}）"
        st.success(
            f"完成：{len(result.questions)} 个研究问题 · {len(result.records)} 个统计实验 · "
            f"ML 最佳 {ml_desc} · 总耗时 {result.runtime_seconds} 秒"
        )
        st.subheader("执行时间线")
        icon = {"ok": "✅", "failed": "⚠️"}
        for s in result.steps:
            st.markdown(f"- {icon[s.status]} **{s.name}**（{s.seconds}s）— {s.detail}")
    if result.report_markdown:
        dl_md, dl_html = st.columns(2)
        dl_md.download_button(
            "下载报告（.md）",
            data=result.report_markdown,
            file_name=f"{result.filename_base}_agent_report.md",
            mime="text/markdown",
            key="agent_dl_md",
        )
        dl_html.download_button(
            "下载报告（.html）",
            data=result.report_html,
            file_name=f"{result.filename_base}_agent_report.html",
            mime="text/html",
            key="agent_dl_html",
        )
        with st.expander("报告预览（Markdown）", expanded=False):
            st.markdown(result.report_markdown)


def main() -> None:
    st.title("AI Data Research Lab")
    st.caption("从一份 CSV 到一份可复现的研究报告 —— 所有数字来自真实执行的代码")

    df, err = load_source()
    if err:
        st.error(err)
    if df is None:
        st.info("在左侧上传 CSV / Excel，或勾选示例数据开始。")
        st.stop()

    try:
        report = profile_dataset(df)
    except ValueError as exc:
        st.error(f"无法生成画像：{exc}")
        st.stop()

    data_sig = (report.dataset.n_rows, report.dataset.n_cols, report.dataset.missing_cells)
    if st.session_state.get("data_sig") != data_sig:
        st.session_state["data_sig"] = data_sig
        for key in ("rqs", "plan", "result", "history", "report", "ml_result"):
            st.session_state.pop(key, None)

    tab_flow, tab_ml, tab_track, tab_agent = st.tabs(
        ["① 分析流程", "② ML 实验室", "③ 实验追踪", "④ 自动研究"]
    )
    with tab_flow:
        render_flow(df, report)
    with tab_ml:
        render_ml_lab(df, report)
    with tab_track:
        render_tracking()
    with tab_agent:
        render_agent(df, report)


if __name__ == "__main__":
    main()
