# -*- coding: utf-8 -*-
"""报告库：保存 / 列表 / 取回 / 名称净化 / 同名冲突。"""
from services.reports_store import get_report, list_reports, save_report, sanitize_name


def test_save_list_get_roundtrip(tmp_path):
    saved = save_report("students", "# 报告\n内容", "<html>报告</html>", base=tmp_path)
    assert saved["name"] == "students"
    items = list_reports(base=tmp_path)
    assert len(items) == 1 and items[0]["name"] == "students" and items[0]["has_html"]
    detail = get_report("students", base=tmp_path)
    assert detail is not None
    assert detail["markdown"].startswith("# 报告")
    assert "报告" in detail["html"]
    assert get_report("students.md", base=tmp_path)["name"] == "students"  # 带后缀也能取


def test_same_name_gets_timestamp_suffix(tmp_path):
    save_report("demo", "a", "a", base=tmp_path)
    saved2 = save_report("demo", "b", "b", base=tmp_path)
    assert saved2["name"] != "demo" and saved2["name"].startswith("demo_")
    assert len(list_reports(base=tmp_path)) == 2


def test_sanitize_name_strips_traversal_and_keeps_chinese():
    assert sanitize_name("../..//etc") == "etc"  # 分隔符替换后剥离首尾杂字符
    assert sanitize_name("学生成绩报告") == "学生成绩报告"
    assert sanitize_name("///") == "report"
    cleaned = sanitize_name("a" * 200)
    assert len(cleaned) <= 80


def test_get_missing_returns_none(tmp_path):
    assert get_report("ghost", base=tmp_path) is None
