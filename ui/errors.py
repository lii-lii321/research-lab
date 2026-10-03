"""错误卡片：把异常渲染为"发生了什么 / 为什么 / 你可以试什么"的人话卡。"""
from __future__ import annotations

import streamlit as st

from services.error_messages import explain


def show_error_card(message: str) -> None:
    """统一错误渲染：命中剧本显示人话 + 行动清单，否则回退原始信息。"""
    playbook = explain(message)
    actions = "".join(f"\n{i}. {a}" for i, a in enumerate(playbook["actions"], 1))
    generic = playbook["why"] == message  # 通用回退：why 承载原始信息
    if generic:
        body = f"**{playbook['title']}**\n\n{message}\n\n**你可以试：**{actions}"
    else:
        body = f"**{playbook['title']}**\n\n{playbook['why']}\n\n**你可以试：**{actions}"
    st.error(body)
