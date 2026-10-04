"""UI 文案字典（中/英），通过 st.session_state["lang"] 切换。"""

ZH = {
    "app_title": "AI Data Research Lab",
    "app_caption": "从一份 CSV 到一份可复现的研究报告 —— 所有数字来自真实执行的代码",
    "intro_body": "**三步上手**：① 左侧选数据（示例已就绪）→ ② 生成研究问题 → ③ 产出可复现报告。",
    "intro_btn": "知道了",
    "tab_flow": "① 分析流程",
    "tab_ml": "② ML 实验室",
    "tab_track": "③ 实验追踪",
    "tab_agent": "④ 自动研究",
    "tab_reports": "⑤ 报告库",
    "source_header": "数据源",
    "source_mode": "方式",
    "source_sample": "示例数据",
    "source_upload": "上传文件",
    "source_paste": "粘贴表格",
    "upload_label": "上传 CSV / Excel（≤50MB）",
    "paste_caption": "从 Excel 复制区域后，直接粘贴到下方表格（首行为表头）。",
    "paste_btn": "使用粘贴的数据",
    "paste_empty": "表格为空：粘贴数据或至少填入一行。",
    "gallery_label": "选择示例数据集",
    "no_data": "在左侧选择数据源。",
    "demo_btn": "🚀 一键体验完整流程",
    "demo_running": "自动完成：画像 → 研究问题 → 统计实验 → ML → 文献 → 报告…",
    "lang_label": "Language",
}

EN = {
    "app_title": "AI Data Research Lab",
    "app_caption": "From one CSV to a reproducible research report — "
                   "every number comes from really executed code",
    "intro_body": "**Three steps**: 1) pick a dataset (sample ready) "
                  "→ 2) generate research questions → 3) produce a reproducible report.",
    "intro_btn": "Got it",
    "tab_flow": "① Analysis Flow",
    "tab_ml": "② ML Lab",
    "tab_track": "③ Tracking",
    "tab_agent": "④ Auto Research",
    "tab_reports": "⑤ Report Library",
    "source_header": "Data Source",
    "source_mode": "Mode",
    "source_sample": "Sample Data",
    "source_upload": "Upload File",
    "source_paste": "Paste Table",
    "upload_label": "Upload CSV / Excel (≤50MB)",
    "paste_caption": "Copy a range from Excel and paste it below (first row = header).",
    "paste_btn": "Use Pasted Data",
    "paste_empty": "Table is empty: paste data or fill at least one row.",
    "gallery_label": "Choose Sample Dataset",
    "no_data": "Pick a data source on the left.",
    "demo_btn": "🚀 One-Click Full Pipeline",
    "demo_running": "Auto: profile → questions → experiments → ML → literature → report…",
    "lang_label": "Language",
}

ALL = {"zh": ZH, "en": EN}


def t(key: str, lang: str = "zh") -> str:
    """取文案，缺 key 回退中文再回退 key 本身。"""
    return ALL.get(lang, ZH).get(key) or ALL["zh"].get(key) or key
