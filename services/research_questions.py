# -*- coding: utf-8 -*-
"""LLM 研究问题生成：数据画像 → 可检验的研究问题（RQ）列表。"""
from __future__ import annotations

import pandas as pd

from models.schemas import ProfileReport, ResearchQuestion
from services.llm import LLMClient
from utils.textjson import extract_json_array

SYSTEM = (
    "你是一位严谨的数据科学研究员。基于给出的数据画像提出有价值、"
    "可以用统计方法检验的研究问题。只输出 JSON 数组，不要输出任何其他内容。"
)

PROMPT_TEMPLATE = """以下是某数据集的画像与样例：

{profile_summary}

样例行：
{rows_preview}

请提出 3~5 个研究问题（RQ）。每个问题必须：
1. 可以用该数据集实际检验（涉及的列必须真实存在）；
2. 统计上可操作（相关分析 / 组间比较 / 关联分析 / 预测建模）；
3. 有研究价值，不要提出平凡陈述。

输出 JSON 数组，每个元素包含字段：
- id: "RQ1" 形式
- question: 问题陈述
- rationale: 为什么值得研究（一句话）
- variables: 涉及的列名（必须来自数据集，原样使用）
- suggested_method: "相关分析"、"组间比较"、"关联分析"、"预测建模" 之一
"""

RQ_MAX = 5


def summarize_profile(report: ProfileReport, max_columns: int = 40) -> str:
    lines = [
        f"行数 {report.dataset.n_rows}，列数 {report.dataset.n_cols}，"
        f"缺失率 {report.dataset.missing_rate:.1%}，重复行 {report.dataset.duplicate_rows}"
    ]
    for c in report.columns[:max_columns]:
        if c.numeric is not None:
            detail = f"数值，均值 {c.numeric.mean}，标准差 {c.numeric.std}，偏度 {c.numeric.skewness}"
        elif c.top_values:
            tops = "、".join(f"{t.value}({t.rate:.0%})" for t in c.top_values[:3])
            detail = f"取值：{tops}"
        else:
            detail = c.description or "—"
        lines.append(f"- {c.name}（{c.type}）：缺失 {c.missing_rate:.0%}；{detail}")
    if report.warnings:
        lines.append("质量提示：" + "；".join(w.message for w in report.warnings[:8]))
    return "\n".join(lines)


def rows_preview(df: pd.DataFrame, n: int = 3) -> str:
    return df.head(n).to_csv(index=False)


def generate_research_questions(
    df: pd.DataFrame, report: ProfileReport, client: LLMClient
) -> list[ResearchQuestion]:
    prompt = PROMPT_TEMPLATE.format(
        profile_summary=summarize_profile(report), rows_preview=rows_preview(df)
    )
    raw = client.chat(SYSTEM, prompt)
    items = extract_json_array(raw)[:RQ_MAX]
    known = set(map(str, df.columns))
    valid: list[ResearchQuestion] = []
    for item in items:
        try:
            q = ResearchQuestion.model_validate(item)
        except Exception:
            continue
        q.variables = [v for v in q.variables if v in known]
        if q.variables:
            valid.append(q)
    if not valid:
        raise ValueError("LLM 提出的问题均不含有效变量")
    return valid
