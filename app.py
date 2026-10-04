"""AI Data Research Lab — Streamlit 多页面入口（st.navigation）。

页面本体在 pages/ 下，共享组件在 ui/ 下；本文件只负责导航注册与全局样式。
"""
from __future__ import annotations

import streamlit as st

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

if "seen_intro" not in st.session_state:
    with st.container(border=True):
        st.markdown(
            "**三步上手**：① 左侧选数据（示例已就绪）→ ② 生成研究问题 → ③ 产出可复现报告。"
        )
        if st.button("知道了", key="intro_ok"):
            st.session_state["seen_intro"] = True
            st.rerun()

nav = st.navigation(
    [
        st.Page("pages/01_分析流程.py", title="① 分析流程", default=True),
        st.Page("pages/02_ML实验室.py", title="② ML 实验室"),
        st.Page("pages/03_实验追踪.py", title="③ 实验追踪"),
        st.Page("pages/04_自动研究.py", title="④ 自动研究"),
        st.Page("pages/05_报告库.py", title="⑤ 报告库"),
    ],
    position="sidebar",
)
nav.run()
