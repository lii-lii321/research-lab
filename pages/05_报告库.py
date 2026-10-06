"""05_报告库页面。"""
from __future__ import annotations

import streamlit as st

from services.reports_store import export_bundle, get_report, list_reports

LEVEL_ICON = {"critical": "🔴", "warning": "🟡", "info": "🔵"}
TASK_CN = {"regression": "回归", "classification": "分类", "clustering": "聚类"}


def main() -> None:
    st.title("⑤ 报告库")
    render_reports()


def render_reports() -> None:
    st.subheader("报告库")
    st.caption(
        "生成的研究报告自动沉淀在 `data/reports/`（跨会话可回看），"
        "分析流程、自动研究与命令行 cli.py 产出的报告都会汇集到这里。"
    )
    items = list_reports()
    if not items:
        st.info("还没有报告——在 ① 分析流程 或 ④ 自动研究 生成一份，或用命令行：`python cli.py analyze data.csv`")
        return

    # 按数据集前缀筛选（报告名通常以数据集名开头）
    prefixes = sorted({i["name"].split("_")[0] for i in items})
    sel_filter = st.selectbox("按数据集筛选", ["全部"] + prefixes, key="reports_filter")
    if sel_filter != "全部":
        items = [i for i in items if i["name"].startswith(sel_filter)]
        if not items:
            st.info(f"筛选「{sel_filter}」下暂无报告。")
            return

    names = [i["name"] for i in items]
    pick = st.selectbox("选择报告", names, key="reports_pick")
    detail = get_report(pick)
    if detail is None:
        st.warning("报告文件读取失败")
        return
    meta = next(i for i in items if i["name"] == pick)
    st.caption(f"修改于 {meta['modified']} · {meta['size_bytes']:,} 字符")
    dl_md, dl_html, dl_zip = st.columns(3)
    dl_md.download_button(
        "下载 .md", detail["markdown"], file_name=f"{pick}.md", mime="text/markdown", key="reports_dl_md"
    )
    if detail["html"]:
        dl_html.download_button(
            "下载 .html", detail["html"], file_name=f"{pick}.html", mime="text/html", key="reports_dl_html"
        )
    bundle_bytes = export_bundle(pick)
    if bundle_bytes:
        dl_zip.download_button(
            "下载研究包 (.zip)",
            data=bundle_bytes,
            file_name=f"{pick}_bundle.zip",
            mime="application/zip",
            key="reports_dl_zip",
        )
    with st.expander("报告预览", expanded=True):
        st.markdown(detail["markdown"])


main()
