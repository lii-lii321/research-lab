"""LLM 研究问题生成：数据画像 → 可检验的研究问题（RQ）列表。"""
from __future__ import annotations

import pandas as pd

from models.schemas import ProfileReport, ResearchQuestion
from services.llm import LLMClient, LLMError
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
    df: pd.DataFrame, report: ProfileReport, client: LLMClient, task: str = ""
) -> list[ResearchQuestion]:
    prompt = PROMPT_TEMPLATE.format(
        profile_summary=summarize_profile(report), rows_preview=rows_preview(df)
    )
    if task:
        prompt = f"研究任务（提出的问题必须围绕它展开，不得偏题）：{task}\n\n" + prompt
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


def _matched_columns(report: ProfileReport, task: str) -> set[str]:
    """任务描述中点名列出的列（英文名或下划线转空格形式）。"""
    if not task:
        return set()
    t = task.lower()
    matched: set[str] = set()
    for c in report.columns:
        for variant in (c.name.lower(), c.name.lower().replace("_", " ")):
            if variant and variant in t:
                matched.add(c.name)
                break
    return matched


def generate_research_questions_rule(
    df: pd.DataFrame,
    report: ProfileReport,
    max_questions: int = 4,
    task: str = "",
) -> list[ResearchQuestion]:
    """确定性规则提出研究问题：目标候选 × 变量类型 → 可检验的 RQ。

    task 中点名的列会被排到最前（触达任务的问题优先），无点名时保持原序。
    """
    cols = {c.name: c for c in report.columns}
    candidates = [t.column for t in report.target_candidates]
    target = candidates[0] if candidates else next(
        (c.name for c in report.columns if c.type == "numeric"), None
    )
    questions: list[ResearchQuestion] = []

    def add(question: str, rationale: str, variables: list[str], method: str) -> None:
        if len(questions) >= max_questions:
            return
        questions.append(
            ResearchQuestion(
                id=f"RQ{len(questions) + 1}",
                question=question,
                rationale=rationale,
                variables=variables,
                suggested_method=method,
                source="rule",
            )
        )

    if target is not None and target in cols:
        numerics = [
            c for c in report.columns
            if c.type == "numeric" and c.name != target and c.missing_rate <= 0.3
        ][:2]
        for c in numerics:
            add(
                f"{c.name} 与 {target} 是否存在显著相关关系？",
                "数值目标配数值特征，可用相关分析直接检验",
                [c.name, target],
                "相关分析",
            )
        categorical = [
            c for c in report.columns
            if c.type in ("categorical", "boolean")
            and c.name != target and 2 <= c.n_unique <= 5
        ]
        if cols[target].type in ("categorical", "boolean"):
            for c in categorical[:2]:
                add(
                    f"{c.name} 与 {target} 之间是否存在关联？",
                    "两个类别变量，可用卡方独立性检验",
                    [c.name, target],
                    "关联分析",
                )
        else:
            for c in categorical[:2]:
                add(
                    f"不同 {c.name} 分组之间的 {target} 是否存在显著差异？",
                    f"{c.name} 有 {c.n_unique} 个取值，可做组间比较",
                    [c.name, target],
                    "组间比较",
                )
    if not questions:
        pair = [c.name for c in report.columns if c.type == "numeric"][:2]
        if len(pair) == 2:
            add(
                f"{pair[0]} 与 {pair[1]} 是否存在显著相关关系？",
                "未识别到目标变量，退而检验前两个数值列的相关性",
                pair,
                "相关分析",
            )
    matched = _matched_columns(report, task)
    if matched:
        touching = [q for q in questions if any(v in matched for v in q.variables)]
        questions = touching + [q for q in questions if q not in touching]
        questions = [
            ResearchQuestion(
                id=f"RQ{i + 1}",
                question=q.question,
                rationale=q.rationale,
                variables=q.variables,
                suggested_method=q.suggested_method,
                source=q.source,
            )
            for i, q in enumerate(questions)
        ]
    return questions[:max_questions]


def generate_research_questions_auto(
    df: pd.DataFrame, report: ProfileReport, client: LLMClient | None, task: str = ""
) -> list[ResearchQuestion]:
    """LLM 可用则用 LLM，失败或未配置时回退规则模式（source 标注来源）。"""
    if client is not None:
        try:
            return generate_research_questions(df, report, client, task)
        except (LLMError, ValueError):
            pass
    return generate_research_questions_rule(df, report, task=task)
