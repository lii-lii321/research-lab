"""报告库：保存 / 列表 / 取回 / 名称净化 / 同名冲突 / 导出。"""
import io
import threading
import zipfile
from datetime import datetime
from pathlib import Path

from services.reports_store import (
    export_bundle,
    export_pdf,
    get_report,
    list_reports,
    sanitize_name,
    save_report,
)
from utils.version import APP_VERSION


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


def test_same_second_concurrent_saves_do_not_overwrite(tmp_path, monkeypatch):
    """冻结时钟到同一秒 + Barrier 双线程并发保存：互不覆盖，两份都留存。"""
    from services import reports_store as rs

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 1, 1, 12, 0, 0)

    monkeypatch.setattr(rs, "datetime", FrozenDatetime)
    barrier = threading.Barrier(2)
    outcomes: dict[str, str] = {}

    def worker(content: str):
        barrier.wait()
        saved = save_report("race", content, content, base=tmp_path)
        outcomes[saved["name"]] = content

    threads = [
        threading.Thread(target=worker, args=("content-T1",)),
        threading.Thread(target=worker, args=("content-T2",)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(outcomes) == 2  # 两个不同报告名，谁也没覆盖谁
    for name, content in outcomes.items():
        assert get_report(name, base=tmp_path)["markdown"] == content
    assert len(list_reports(base=tmp_path)) == 2


def test_list_reports_tolerates_vanishing_files(tmp_path, monkeypatch):
    """glob 与 stat 之间文件被并发删除：跳过该条而非抛 FileNotFoundError。"""
    save_report("keep", "a", "a", base=tmp_path)
    save_report("gone", "b", "b", base=tmp_path)

    real_stat = Path.stat

    def flaky_stat(self, *args, **kwargs):
        if self.name == "gone.md":
            raise FileNotFoundError(str(self))
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", flaky_stat)
    items = list_reports(base=tmp_path)
    monkeypatch.setattr(Path, "stat", real_stat)
    assert [i["name"] for i in items] == ["keep"]


def test_export_bundle_contains_manifest_md_html_assets(tmp_path):
    save_report(
        "full",
        "# 标题\n内容",
        "<html>full</html>",
        base=tmp_path,
        images=[("chart", b"\x89PNG-fake-bytes")],
    )
    bundle = export_bundle("full", base=tmp_path)
    assert bundle is not None
    with zipfile.ZipFile(io.BytesIO(bundle)) as zf:
        assert {"full.md", "full.html", "MANIFEST.txt", "assets/chart.png"} <= set(zf.namelist())
        manifest = zf.read("MANIFEST.txt").decode("utf-8")
        assert "报告：full" in manifest
        assert f"v{APP_VERSION}" in manifest  # 版本单一来源
        assert zf.read("assets/chart.png") == b"\x89PNG-fake-bytes"


def test_export_bundle_omits_empty_html(tmp_path):
    save_report("nhtml", "only md", "", base=tmp_path)
    bundle = export_bundle("nhtml", base=tmp_path)
    assert bundle is not None
    with zipfile.ZipFile(io.BytesIO(bundle)) as zf:
        assert "nhtml.md" in zf.namelist()
        assert "nhtml.html" not in zf.namelist()


def test_export_pdf_renders_bytes_and_missing_returns_none(tmp_path):
    # Windows 依赖系统中文字体（msyh/simhei）；fpdf2 为现有依赖
    save_report("pdfdoc", "# 标题\n\n正文段落", "<html/>", base=tmp_path)
    pdf = export_pdf("pdfdoc", base=tmp_path)
    assert pdf is not None
    assert pdf[:5] == b"%PDF-"
    assert export_pdf("ghost", base=tmp_path) is None
