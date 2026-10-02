"""为报告收集画像配图（PNG 字节流），供 build_report 嵌入。"""
from __future__ import annotations

import io

import matplotlib.pyplot as plt

from models.schemas import ProfileReport
from utils.charts import missing_matrix, numeric_histograms


def fig_to_png(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def collect_profile_images(df, report: ProfileReport) -> list[tuple[str, bytes]]:
    """直方图网格 + 缺失矩阵；收集失败静默降级为空列表。"""
    images: list[tuple[str, bytes]] = []
    numeric_cols = [c.name for c in report.columns if c.type == "numeric"]
    try:
        hist = numeric_histograms(df, numeric_cols)
        if hist is not None:
            images.append(("histograms", fig_to_png(hist)))
        matrix = missing_matrix(df)
        if matrix is not None:
            images.append(("missing_matrix", fig_to_png(matrix)))
    except Exception:
        pass  # 配图失败不阻断报告生成
    return images
