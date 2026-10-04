"""02_ML实验室页面。"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from models.schemas import MLExperimentResult, ProfileReport
from services.ml_lab import run_ml_experiment
from services.tracking import TrackingStore
from ui.errors import show_error_card
from ui.page_state import cached_profile, invalidate_on_data_change, load_source

LEVEL_ICON = {"critical": "🔴", "warning": "🟡", "info": "🔵"}
TASK_CN = {"regression": "回归", "classification": "分类", "clustering": "聚类"}


def main() -> None:
    st.title("② ML 实验室")

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
    render_ml_lab(df, report)





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
        with st.status("正在训练与评估模型…", expanded=True) as training:
            progress_lines: list[str] = []

            def _ml_progress(msg: str) -> None:
                progress_lines.append(msg)
                training.write("· " + msg)

            try:
                st.session_state["ml_result"] = run_ml_experiment(
                    df,
                    report,
                    target=target,
                    task=sel_task,
                    store=TrackingStore(),
                    dataset_name=st.session_state.get("dataset_name", "dataset"),
                    persist_dir="data/models",
                    progress_cb=_ml_progress,
                )
            except ValueError as exc:
                show_error_card(str(exc))
                st.session_state.pop("ml_result", None)
                training.update(label="训练失败", state="error")
            else:
                training.update(label="训练完成", state="complete", expanded=False)
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
    if result.tuning_note:
        st.caption(f"调优：{result.tuning_note}")
    if result.feature_importance:
        st.subheader("特征重要性（permutation）")
        st.bar_chart(
            pd.DataFrame(
                [{"特征": i.feature, "重要性": i.importance} for i in result.feature_importance]
            ).set_index("特征")
        )
    if result.task == "clustering" and result.cluster_sizes:
        st.caption("簇规模：" + "、".join(f"{k}：{v}" for k, v in result.cluster_sizes.items()))
    if result.excluded:
        st.caption("已排除：" + "；".join(f"{e.column}（{e.reason}）" for e in result.excluded))

main()

