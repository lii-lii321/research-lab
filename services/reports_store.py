"""报告库：把生成的研究报告沉淀到 data/reports/，可列表、取回、回看、打包。

场景：报告不该只活在浏览器会话里——课程作业、论文方法论、面试演示
都需要"事后还能找到这份报告"。
"""
from __future__ import annotations

import io
import os
import re
import zipfile
from datetime import datetime
from pathlib import Path

DEFAULT_DIR = Path("data/reports")
_SAFE_NAME = re.compile(r"[^0-9A-Za-z_.\-一-龥]+")


def export_bundle(name: str, base: Path | str | None = None) -> bytes | None:
    """打包一份报告为 zip（md + html + assets + 复现脚本引用清单）。"""
    detail = get_report(name, base)
    if detail is None:
        return None
    stem = detail["name"]
    directory = _dir(base)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{stem}.md", detail["markdown"])
        if detail["html"]:
            zf.writestr(f"{stem}.html", detail["html"])
        assets_dir = directory / "assets" / stem
        if assets_dir.exists():
            for f in sorted(assets_dir.iterdir()):
                if f.is_file():
                    zf.write(f, f"assets/{f.name}")
        manifest = (
            f"报告：{stem}\n导出：{datetime.now().isoformat(timespec='seconds')}\n"
            f"由 AI Data Research Lab v1.3.0 生成\n"
            f"包含：markdown + html + assets\n"
        )
        zf.writestr("MANIFEST.txt", manifest)
    return buf.getvalue()


def get_assets_dir(name: str, base: Path | str | None = None) -> Path | None:
    stem = sanitize_name(name)
    directory = _dir(base)
    assets = directory / "assets" / stem
    return assets if assets.exists() else None


def export_pdf(name: str, base: Path | str | None = None) -> bytes | None:
    """Markdown 报告转 PDF（fpdf2 纯 Python，无原生依赖）；失败返回 None。

    中文字体使用系统微软雅黑；文本按行渲染，标题加粗，表格降级为纯文本。
    """
    detail = get_report(name, base)
    if detail is None or not detail["markdown"]:
        return None
    try:
        from fpdf import FPDF

        CJK_FONT = "C:/Windows/Fonts/msyh.ttc"
        if not Path(CJK_FONT).exists():
            CJK_FONT = "C:/Windows/Fonts/simhei.ttf"
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        if Path(CJK_FONT).exists():
            pdf.add_font("cjk", style="", fname=CJK_FONT)
            pdf.add_font("cjk", style="B", fname=CJK_FONT)
            body_font, bold_font = "cjk", "cjk"
        else:  # 无中文字体时退化为 Helvetica（中文会缺失，但不至于报错）
            body_font = bold_font = "Helvetica"
        for raw_line in detail["markdown"].split("\n"):
            line = raw_line.rstrip()
            if not line:
                pdf.ln(3)
                continue
            if line.startswith("# "):
                pdf.set_font(bold_font, style="B", size=16)
                pdf.multi_cell(0, 8, line[2:])
                pdf.ln(2)
            elif line.startswith("## "):
                pdf.set_font(bold_font, style="B", size=14)
                pdf.multi_cell(0, 7, line[3:])
                pdf.ln(1)
            elif line.startswith("### "):
                pdf.set_font(bold_font, style="B", size=12)
                pdf.multi_cell(0, 6, line[4:])
            elif line.startswith("|"):
                pdf.set_font(body_font, size=9)
                cells = [c.strip() for c in line.strip("|").split("|")]
                if all(set(c) <= set("-: ") for c in cells):
                    continue  # 分隔行
                col_w = 180 / max(len(cells), 1)
                for c in cells:
                    pdf.cell(col_w, 5, c[:40], border=1)
                pdf.ln()
            else:
                pdf.set_font(body_font, size=10)
                pdf.multi_cell(0, 5, line.lstrip("- ").lstrip("*").rstrip("*"))
        return bytes(pdf.output())
    except Exception:
        return None


def _dir(base: Path | str | None = None) -> Path:
    if base is not None:
        path = Path(base)
    else:
        env = os.getenv("RESEARCH_LAB_REPORTS_DIR", "").strip()
        path = Path(env) if env else DEFAULT_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_name(name: str) -> str:
    cleaned = _SAFE_NAME.sub("_", name).strip("._") or "report"
    return cleaned[:80]


def save_report(
    name_base: str,
    markdown: str,
    html: str,
    base: Path | str | None = None,
    images: list[tuple[str, bytes]] | None = None,
) -> dict:
    """保存一份报告（同名自动加时间戳后缀），返回 {name, md_path, html_path}。

    images 为 (文件名, PNG 字节) 列表，落盘到 assets/<报告名>/ 供 MD 相对引用。
    """
    directory = _dir(base)
    stem = sanitize_name(name_base)
    md_path = directory / f"{stem}.md"
    html_path = directory / f"{stem}.html"
    if md_path.exists() or html_path.exists():
        stem = f"{stem}_{datetime.now().strftime('%H%M%S')}"
        md_path = directory / f"{stem}.md"
        html_path = directory / f"{stem}.html"
    md_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(html, encoding="utf-8")
    if images:
        assets = directory / "assets" / stem
        assets.mkdir(parents=True, exist_ok=True)
        for fname, data in images:
            (assets / f"{fname}.png").write_bytes(data)
    return {"name": stem, "md_path": str(md_path), "html_path": str(html_path)}


def list_reports(base: Path | str | None = None) -> list[dict]:
    directory = _dir(base)
    items: list[dict] = []
    for md_path in sorted(directory.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
        html_path = md_path.with_suffix(".html")
        stat = md_path.stat()
        items.append(
            {
                "name": md_path.stem,
                "md_path": str(md_path),
                "has_html": html_path.exists(),
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
            }
        )
    return items


def get_report(name: str, base: Path | str | None = None) -> dict | None:
    """按名称取回报告内容；name 带不带 .md / .html 后缀均可。"""
    stem = sanitize_name(name)
    for suffix in (".md", ".html"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    directory = _dir(base)
    md_path = directory / f"{stem}.md"
    html_path = directory / f"{stem}.html"
    if not md_path.exists():
        return None
    return {
        "name": md_path.stem,
        "markdown": md_path.read_text(encoding="utf-8"),
        "html": html_path.read_text(encoding="utf-8") if html_path.exists() else "",
    }
