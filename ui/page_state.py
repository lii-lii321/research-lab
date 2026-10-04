"""页面共享状态：数据加载、画像缓存、失效逻辑，供 pages/ 各页面复用。"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from models.schemas import ProfileReport
from services.profiler import profile_dataset
from services.tracking import dataframe_fingerprint
from utils.io import read_tabular
from utils.sample_data import DATASET_GALLERY


def load_source() -> tuple[pd.DataFrame | None, str | None]:
    """数据源三模式：上传 / 粘贴 / 示例画廊（含运行时自愈）。"""
    with st.sidebar:
        st.header("数据源")
        mode = st.radio("方式", ["示例数据", "上传文件", "粘贴表格"], label_visibility="collapsed", key="source_mode")
        if mode == "上传文件":
            uploaded = st.file_uploader("上传 CSV / Excel（≤50MB）", type=["csv", "xlsx"])
            if uploaded is not None:
                try:
                    st.session_state["dataset_name"] = uploaded.name
                    return read_tabular(uploaded.name, uploaded.getvalue()), None
                except ValueError as exc:
                    return None, str(exc)
            return None, None
        if mode == "粘贴表格":
            st.caption("从 Excel 复制区域后，直接粘贴到下方表格（首行为表头）。")
            edited = st.data_editor(
                pd.DataFrame({"列1": [""] * 5}),
                num_rows="dynamic",
                use_container_width=True,
                key="paste_grid",
            )
            if st.button("使用粘贴的数据", type="primary", key="paste_use"):
                cleaned = edited.replace("", pd.NA).dropna(how="all").dropna(axis=1, how="all")
                if cleaned.empty:
                    st.warning("表格为空：粘贴数据或至少填入一行。")
                    return None, None
                st.session_state["dataset_name"] = "pasted_data"
                return cleaned.reset_index(drop=True), None
            return None, None
        gallery = st.selectbox("选择示例数据集", list(DATASET_GALLERY), key="sample_pick")
        names = {"学生成绩": "student_performance.csv", "门店销售": "store_sales.csv", "医疗随访": "cohort.csv"}
        st.session_state["dataset_name"] = names.get(gallery, gallery)
        try:
            st.query_params["data"] = "sample"
        except Exception:
            pass
        try:
            return DATASET_GALLERY[gallery](), None
        except Exception as exc:
            return None, f"示例数据生成失败：{exc}"
    return None, None


@st.cache_data(hash_funcs={pd.DataFrame: lambda df: dataframe_fingerprint(df)}, show_spinner="正在生成数据画像…")
def cached_profile(df: pd.DataFrame) -> ProfileReport:
    return profile_dataset(df)


def invalidate_on_data_change(report: ProfileReport) -> None:
    data_sig = (report.dataset.n_rows, report.dataset.n_cols, report.dataset.missing_cells)
    if st.session_state.get("data_sig") != data_sig:
        st.session_state["data_sig"] = data_sig
        for key in ("rqs", "plan", "result", "history", "report", "ml_result", "agent_result", "demo_result"):
            st.session_state.pop(key, None)
