"""03_实验追踪页面。"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from services.tracking import TrackingStore

LEVEL_ICON = {"critical": "🔴", "warning": "🟡", "info": "🔵"}
TASK_CN = {"regression": "回归", "classification": "分类", "clustering": "聚类"}


def main() -> None:
    st.title("③ 实验追踪")
    render_tracking()





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

main()

