"""AI Data Research Lab — Streamlit 多页面入口（st.navigation）。

页面本体在 pages/ 下，共享组件在 ui/ 下；本文件只负责导航注册与全局样式。
"""
from __future__ import annotations

import streamlit as st

from ui.i18n import t

st.set_page_config(page_title="AI Data Research Lab", page_icon="🗂", layout="wide")
st.markdown(
    """
    <style>
      .stApp { background: #ffffff; }
      h1, h2, h3 { color: #1a365d; font-weight: 600; }
      section[data-testid="stSidebar"] { background: #f8fafc; }
      .block-container { padding-top: 2rem; }
      img { max-width: 100%; }
    </style>
    """,
    unsafe_allow_html=True,
)

_lang = st.sidebar.radio(
    "🌐 Language / 语言", ["中文", "English"], key="ui_lang", horizontal=True,
) if st.sidebar else "中文"
_lang_code = "en" if _lang == "English" else "zh"
st.session_state["lang"] = _lang_code

if "seen_intro" not in st.session_state:
    with st.container(border=True):
        st.markdown(t("intro_body", _lang_code))
        if st.button(t("intro_btn", _lang_code), key="intro_ok"):
            st.session_state["seen_intro"] = True
            st.rerun()

nav = st.navigation(
    [
        st.Page("pages/01_分析流程.py", title=t("tab_flow", _lang_code), default=True),
        st.Page("pages/02_ML实验室.py", title=t("tab_ml", _lang_code)),
        st.Page("pages/03_实验追踪.py", title=t("tab_track", _lang_code)),
        st.Page("pages/04_自动研究.py", title=t("tab_agent", _lang_code)),
        st.Page("pages/05_报告库.py", title=t("tab_reports", _lang_code)),
    ],
    position="sidebar",
)
nav.run()
