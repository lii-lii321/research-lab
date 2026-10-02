"""研究报告构建：Markdown + HTML，全部内容来自已执行的真实结果。"""
from __future__ import annotations

import base64
import html as html_lib
from datetime import datetime

from models.schemas import (
    TYPE_CN,
    ExperimentRecord,
    MLExperimentResult,
    PaperRef,
    ProfileReport,
    ResearchQuestion,
)
from services.executor import METHOD_CN, format_p
from services.literature import snippet
from services.reports_store import sanitize_name
from services.repro import build_repro_script

GENERATOR = "AI Data Research Lab v1.0.0"

LEVEL_ICON = {"critical": "🔴", "warning": "🟡", "info": "🔵"}

CN_NUM = ("一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "十一", "十二")

TASK_CN = {"regression": "回归", "classification": "分类", "clustering": "聚类"}

LIMITATIONS = [
    "本报告所有检验均为相关或组间差异分析，不构成因果推断。",
    "结论仅基于当前数据集，向其他人群或场景外推需谨慎。",
    "缺失值处理：相关与配对检验按有效配对剔除，组间与卡方检验按整行剔除。",
    "研究问题由同一数据画像生成并逐个检验，全部结论均为探索性分析；"
    "结论表所附 Benjamini–Hochberg 校正后 p 值供多重比较参考。",
    "标注来源为 LLM 的解读文字由模型生成，仅供参考；全部统计数字由本地 scipy 真实计算。",
]


def _bh_adjust(pvals: list[float]) -> list[float]:
    """Benjamini–Hochberg 校正，返回与输入顺序一致的校正后 p。"""
    n = len(pvals)
    if n <= 1:
        return list(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    adjusted = [0.0] * n
    running = 1.0
    for rank in range(n, 0, -1):
        idx = order[rank - 1]
        running = min(running, pvals[idx] * n / rank)
        adjusted[idx] = min(running, 1.0)
    return adjusted


def _adjusted_p_map(ok_records: list) -> dict[int, float]:
    pairs = [
        (
            i,
            r.result.p_value_raw if r.result.p_value_raw is not None else r.result.p_value,
        )
        for i, r in enumerate(ok_records)
    ]
    real = [p for _, p in pairs if p is not None]
    adjusted = _bh_adjust(real)
    mapping: dict[int, float] = {}
    cursor = 0
    for i, p in pairs:
        if p is not None:
            mapping[i] = adjusted[cursor]
            cursor += 1
    return mapping


def _sparse_marker(res) -> str:
    if res.extra.get("sparse_expected_rate", 0) > 0 and res.extra.get("test_used") != "fisher_exact":
        return "（稀疏）"
    return ""


def _md(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _summary_field(c) -> str:
    if c.numeric is not None:
        return f"均值 {c.numeric.mean} · 标准差 {c.numeric.std} · 离群值 {c.outlier_count or 0}"
    if c.type == "datetime" and c.description:
        return c.description
    if c.top_values:
        return "、".join(f"{t.value}（{t.rate:.0%}）" for t in c.top_values[:3])
    return "—"


def _experiment_lines_md(rec: ExperimentRecord) -> list[str]:
    plan, res = rec.plan, rec.result
    method_cn = METHOD_CN.get(plan.method, plan.method)
    source_cn = "LLM 设计" if plan.source == "llm" else "变量类型规则"
    lines = [
        f"### {_md(plan.experiment_id)} · {method_cn}",
        f"- 研究问题：{_md(plan.question_id)} — {_md(plan.hypothesis)}",
        f"- H0：{_md(plan.h0)} ｜ H1：{_md(plan.h1)} ｜ α = {plan.alpha}",
        f"- 变量：{_md('、'.join(plan.variables))}（计划来源：{source_cn}）",
    ]
    if plan.required_n:
        lines.append(
            f"- 所需样本量：约 {plan.required_n} 例观测（α = {plan.alpha}、power = 0.8、中等效应假设）"
        )
    if res.status != "ok":
        lines.append(f"- 执行结果：**未完成** — {_md(res.reason)}")
        return lines
    stat_txt = (
        f"{res.statistic_name} = {res.statistic:.3f}" if res.statistic is not None else "—"
    )
    p_txt = format_p(res.p_value) if res.p_value is not None else "—"
    effect = (
        f"｜ 效应量 {res.effect_name} = {res.effect_size:.3f}"
        if res.effect_size is not None
        else ""
    )
    decision = "拒绝 H0" if res.decision == "reject_h0" else "未能拒绝 H0"
    interp_source = "LLM" if res.interpretation_source == "llm" else "规则模板"
    lines.append(f"- 结果：**{stat_txt}**，{p_txt} → **{decision}**{effect}")
    lines.append(f"- 样本：有效 {res.n_used} / 总 {res.n_used + res.n_dropped}")
    ci = res.extra.get("ci95") or res.extra.get("mean_diff_ci95") or res.extra.get("slope_ci95")
    if ci:
        lines.append(f"- 95% CI：[{ci[0]}, {ci[1]}]")
    lines.append(f"- 解读（来源：{interp_source}）：{_md(res.interpretation)}")
    return lines


def build_markdown(
    profile: ProfileReport,
    dataset_name: str,
    questions: list[ResearchQuestion],
    records: list[ExperimentRecord],
    generated: str,
    ml_result: MLExperimentResult | None = None,
    references: list[PaperRef] | None = None,
    images: list[tuple[str, bytes]] | None = None,
) -> str:
    stem = sanitize_name(dataset_name.rsplit(".", 1)[0])
    section = {"n": 0}

    def heading(title: str) -> str:
        section["n"] += 1
        return f"## {CN_NUM[section['n'] - 1]}、{title}"

    d = profile.dataset
    lines = [
        "# AI 数据科学研究报告",
        "",
        f"- 数据集：{dataset_name}",
        f"- 生成时间：{generated}",
        "- 复现说明：全部统计量由 scipy 在本地对上传数据真实计算；"
        "LLM 仅参与研究问题提出与结果解读，且均已在正文标注来源。",
        "",
        heading("数据画像"),
        "",
        f"- 规模：{d.n_rows:,} 行 × {d.n_cols} 列",
        f"- 缺失：{d.missing_cells:,} 个单元格（{d.missing_rate:.1%}）",
        f"- 重复行：{d.duplicate_rows:,}（{d.duplicate_rate:.1%}）",
        f"- 内存占用：{d.memory_mb:.2f} MB",
    ]
    type_desc = " · ".join(f"{TYPE_CN.get(k, k)} {v}" for k, v in sorted(d.type_counts.items()))
    lines.append(f"- 字段类型：{type_desc or '—'}")
    lines += ["", "### 质量提示", ""]
    if profile.warnings:
        lines += [
            f"- {LEVEL_ICON[w.level]} `{w.code}` {_md(w.message)}"
            + (f"　→ 建议：{_md(w.suggestion)}" if w.suggestion else "")
            for w in profile.warnings
        ]
    else:
        lines.append("- 未发现明显的质量问题。")
    lines += [
        "",
        "### 字段概览",
        "",
        "| 字段 | 类型 | 缺失率 | 唯一值 | 概要 |",
        "|---|---|---|---|---|",
    ]
    for c in profile.columns:
        lines.append(
            f"| {_md(c.name)} | {TYPE_CN.get(c.type, c.type)} | {c.missing_rate:.1%} "
            f"| {c.n_unique} | {_md(_summary_field(c))} |"
        )
    if images:
        lines += ["", "### 画像配图", ""]
        for name, _png in images:
            lines += [f"![{name}](assets/{stem}/{name}.png)", ""]
    if profile.target_candidates:
        lines += ["", "### 目标变量候选", ""]
        lines += [f"- **{t.column}** — {_md(t.reason)}" for t in profile.target_candidates]
    if profile.correlations:
        lines += ["", "### 高相关字段对（|r| ≥ 0.8）", ""]
        lines += [
            f"- {p.column_a} × {p.column_b}：r = {p.coefficient:.2f}"
            for p in profile.correlations
        ]
    lines += ["", heading("研究问题"), ""]
    if questions:
        for q in questions:
            lines.append(
                f"- **{q.id}** {_md(q.question)}（变量：{_md('、'.join(q.variables))}；"
                f"建议：{_md(q.suggested_method or '—')}）"
            )
            if q.rationale:
                lines.append(f"  - 理由：{_md(q.rationale)}")
    else:
        lines.append("-（本次会话未生成研究问题）")
    lines += ["", heading("实验与结果"), ""]
    if records:
        for rec in records:
            lines += _experiment_lines_md(rec)
            lines.append("")
    else:
        lines.append("-（本次会话未执行实验）")
    sig_records = [r for r in records if r.result.status == "ok" and r.result.decision == "reject_h0"]
    if sig_records:
        lines += ["", heading("因果提示"), ""]
        for rec in sig_records:
            confounds = [
                p.column_b if p.column_a in rec.plan.variables else p.column_a
                for p in profile.correlations
                if rec.plan.variables
                and (p.column_a in rec.plan.variables) != (p.column_b in rec.plan.variables)
            ][:2]
            confound_txt = "、".join(confounds) if confounds else "共同原因与选择偏差"
            lines.append(
                f"- 「{_md(' → '.join(rec.plan.variables))}」显著不等于因果：需排除 {_md(confound_txt)} 等混杂，"
                "并进行随机化实验或匹配设计。"
            )
    if ml_result is not None:
        lines += ["", heading("ML 基线实验"), ""]
        if ml_result.status != "ok":
            lines.append(f"- ML 基线未完成：{_md(ml_result.reason)}")
        else:
            target_desc = ml_result.target or "（无目标 · 聚类）"
            fingerprint = ml_result.dataset_fingerprint[:8]
            lines.append(
                f"- 任务：{_md(TASK_CN.get(ml_result.task, ml_result.task))}"
                f"（目标：{_md(target_desc)}；数据指纹 {fingerprint}）"
            )
            lines.append(
                f"- 最佳模型：**{ml_result.best_model}**"
                f"（{ml_result.best_metric_name} = {ml_result.best_metric_value}）"
            )
            lines += [
                "| 模型 | 超参 | 指标 | 耗时(秒) |",
                "|---|---|---|---|",
            ]
            for m in ml_result.models:
                params = "、".join(f"{k}={v}" for k, v in m.params.items()) or "—"
                metrics = "，".join(f"{k}={v}" for k, v in m.metrics.items())
                lines.append(
                    f"| {m.model} | {_md(params)} | {_md(metrics)} | {m.train_seconds:.2f} |"
                )
            if ml_result.task == "clustering" and ml_result.cluster_sizes:
                sizes = "、".join(f"{k}：{v}" for k, v in ml_result.cluster_sizes.items())
                lines.append(f"- 簇规模：{_md(sizes)}")
            if ml_result.excluded:
                excluded_desc = "；".join(f"{e.column}（{e.reason}）" for e in ml_result.excluded)
                lines.append(f"- 已排除特征：{_md(excluded_desc)}")
            if ml_result.tuning_note:
                lines.append(f"- 调优：{_md(ml_result.tuning_note)}")
            if ml_result.feature_importance:
                lines += ["", "**特征重要性（permutation，top 10）**", ""]
                lines += [f"- {_md(i.feature)}：{i.importance}" for i in ml_result.feature_importance]
            lines += [
                "",
                "**模型卡片**",
                "",
                f"- 数据指纹：{ml_result.dataset_fingerprint[:8]}　"
                f"任务：{_md(TASK_CN.get(ml_result.task, ml_result.task))}　目标：{_md(target_desc)}",
                f"- 规模：训练/测试 {ml_result.n_train}/{ml_result.n_test}　"
                f"特征：数值 {len(ml_result.features_numeric)} + 类别 {len(ml_result.features_categorical)}",
                "- 局限：基线模型不构成因果解释；样本量与类别平衡情况见数据画像与质量提示。",
            ]
    if references:
        lines += [
            "",
            heading("相关工作（文献引用）"),
            "",
            "> 检索词来自数据字段与研究问题，由 arXiv 关键词检索得到，非系统性文献综述，仅作入口性参考。",
            "",
        ]
        for i, p in enumerate(references, 1):
            authors = "、".join(p.authors[:3]) + ("等" if len(p.authors) > 3 else "")
            lines.append(
                f"{i}. **{_md(p.title)}** — {_md(authors)}（{p.year}）。{_md(snippet(p.summary))} [链接]({p.url})"
            )
    lines += ["", heading("结论与局限"), ""]
    ok_records = [r for r in records if r.result.status == "ok"]
    if ok_records:
        adj_map = _adjusted_p_map(ok_records)
        lines += [
            "| 实验 | 方法 | 统计量 | p 值 | BH 校正后 | 结论 |",
            "|---|---|---|---|---|---|",
        ]
        for idx, rec in enumerate(ok_records):
            res = rec.result
            stat_txt = (
                f"{res.statistic_name} = {res.statistic:.3f}"
                if res.statistic is not None
                else "—"
            )
            p_txt = format_p(res.p_value) if res.p_value is not None else "—"
            adj = adj_map.get(idx)
            adj_txt = f"{adj:.4f}" if adj is not None else "—"
            decision = "拒绝 H0" if res.decision == "reject_h0" else "未能拒绝 H0"
            decision += _sparse_marker(res)
            method_cn = METHOD_CN.get(rec.plan.method, rec.plan.method)
            lines.append(
                f"| {_md(rec.plan.experiment_id)} | {method_cn} | {stat_txt} | {p_txt} | {adj_txt} | {decision} |"
            )
        lines.append("")
    lines += [f"- {item}" for item in LIMITATIONS]
    repro_ok = [r for r in records if r.result.status == "ok"]
    if repro_ok:
        lines += ["", heading("附录：复现脚本"), ""]
        for rec in repro_ok:
            lines += [
                f"**{rec.plan.experiment_id}**",
                "",
                "```python",
                build_repro_script(rec.plan).rstrip(),
                "```",
                "",
            ]
    lines += ["", "---", f"*本报告由 {GENERATOR} 生成。*"]
    return "\n".join(lines)


def _esc(text) -> str:
    return html_lib.escape(str(text))


def build_html(
    profile: ProfileReport,
    dataset_name: str,
    questions: list[ResearchQuestion],
    records: list[ExperimentRecord],
    generated: str,
    ml_result: MLExperimentResult | None = None,
    references: list[PaperRef] | None = None,
    images: list[tuple[str, bytes]] | None = None,
) -> str:
    section = {"n": 0}

    def heading(title: str) -> str:
        section["n"] += 1
        return f"<h2>{CN_NUM[section['n'] - 1]}、{_esc(title)}</h2>"

    d = profile.dataset
    parts: list[str] = []
    parts.append(
        "<!DOCTYPE html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        "<title>AI 数据科学研究报告</title><style>"
        "body{font-family:'Microsoft YaHei',system-ui,sans-serif;color:#334155;background:#f8fafc;margin:0;}"
        ".page{max-width:860px;margin:24px auto;background:#fff;padding:48px 56px;border:1px solid #e2e8f0;}"
        "h1{color:#1a365d;font-size:22px;}"
        "h2{color:#1a365d;font-size:17px;border-bottom:1px solid #e2e8f0;padding-bottom:6px;}"
        "h3{color:#1a365d;font-size:15px;}"
        "table{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0;}"
        "th,td{border:1px solid #e2e8f0;padding:6px 10px;text-align:left;}"
        "th{background:#f1f5f9;color:#1a365d;}"
        ".meta{color:#64748b;font-size:12px;}code{background:#f1f5f9;padding:1px 5px;}"
        "ul{margin:8px 0;}li{margin:4px 0;}hr{border:none;border-top:1px solid #e2e8f0;margin:24px 0;}"
        "</style></head><body><div class=\"page\">"
    )
    parts.append("<h1>AI 数据科学研究报告</h1>")
    parts.append(
        f"<p class=\"meta\">数据集：{_esc(dataset_name)} ｜ 生成时间：{_esc(generated)}<br>"
        "复现说明：全部统计量由 scipy 在本地对上传数据真实计算；"
        "LLM 仅参与研究问题提出与结果解读，均已在正文标注来源。</p>"
    )
    parts.append(heading("数据画像") + "<ul>")
    parts.append(
        f"<li>规模：{d.n_rows:,} 行 × {d.n_cols} 列；缺失 {d.missing_cells:,} 单元格"
        f"（{d.missing_rate:.1%}）；重复行 {d.duplicate_rows:,}；内存 {d.memory_mb:.2f} MB</li>"
    )
    type_desc = " · ".join(f"{TYPE_CN.get(k, k)} {v}" for k, v in sorted(d.type_counts.items()))
    parts.append(f"<li>字段类型：{_esc(type_desc) or '—'}</li></ul>")
    parts.append("<h3>质量提示</h3><ul>")
    if profile.warnings:
        for w in profile.warnings:
            icon = LEVEL_ICON.get(w.level, "")
            advice = f"　→ 建议：{_esc(w.suggestion)}" if w.suggestion else ""
            parts.append(f"<li>{icon} <code>{_esc(w.code)}</code> {_esc(w.message)}{advice}</li>")
    else:
        parts.append("<li>未发现明显的质量问题。</li>")
    parts.append("</ul>")
    parts.append(
        "<h3>字段概览</h3><table><tr><th>字段</th><th>类型</th><th>缺失率</th><th>唯一值</th><th>概要</th></tr>"
    )
    for c in profile.columns:
        parts.append(
            f"<tr><td>{_esc(c.name)}</td><td>{_esc(TYPE_CN.get(c.type, c.type))}</td>"
            f"<td>{c.missing_rate:.1%}</td><td>{c.n_unique}</td><td>{_esc(_summary_field(c))}</td></tr>"
        )
    parts.append("</table>")
    if images:
        parts.append("<h3>画像配图</h3>")
        for name, png in images:
            b64 = base64.b64encode(png).decode("ascii")
            parts.append(
                f'<img src="data:image/png;base64,{b64}" alt="{_esc(name)}" style="max-width:100%;">'
            )
    if profile.target_candidates:
        parts.append("<h3>目标变量候选</h3><ul>")
        parts += [
            f"<li><b>{_esc(t.column)}</b> — {_esc(t.reason)}</li>"
            for t in profile.target_candidates
        ]
        parts.append("</ul>")
    if profile.correlations:
        parts.append("<h3>高相关字段对（|r| ≥ 0.8）</h3><ul>")
        parts += [
            f"<li>{_esc(p.column_a)} × {_esc(p.column_b)}：r = {p.coefficient:.2f}</li>"
            for p in profile.correlations
        ]
        parts.append("</ul>")
    parts.append(heading("研究问题") + "<ul>")
    if questions:
        for q in questions:
            rationale = f"<br><span class=\"meta\">理由：{_esc(q.rationale)}</span>" if q.rationale else ""
            parts.append(
                f"<li><b>{_esc(q.id)}</b> {_esc(q.question)}（变量：{_esc('、'.join(q.variables))}；"
                f"建议：{_esc(q.suggested_method or '—')}）{rationale}</li>"
            )
    else:
        parts.append("<li>（本次会话未生成研究问题）</li>")
    parts.append("</ul>" + heading("实验与结果"))
    if records:
        for rec in records:
            plan, res = rec.plan, rec.result
            method_cn = METHOD_CN.get(plan.method, plan.method)
            source_cn = "LLM 设计" if plan.source == "llm" else "变量类型规则"
            parts.append(
                f"<h3>{_esc(plan.experiment_id)} · {_esc(method_cn)}</h3><ul>"
                f"<li>研究问题：{_esc(plan.question_id)} — {_esc(plan.hypothesis)}</li>"
                f"<li>H0：{_esc(plan.h0)} ｜ H1：{_esc(plan.h1)} ｜ α = {plan.alpha}</li>"
                f"<li>变量：{_esc('、'.join(plan.variables))}（计划来源：{source_cn}）</li>"
            )
            if res.status != "ok":
                parts.append(f"<li>执行结果：<b>未完成</b> — {_esc(res.reason)}</li></ul>")
                continue
            stat_txt = (
                f"{_esc(res.statistic_name)} = {res.statistic:.3f}"
                if res.statistic is not None
                else "—"
            )
            p_txt = format_p(res.p_value) if res.p_value is not None else "—"
            effect = (
                f" ｜ 效应量 {_esc(res.effect_name)} = {res.effect_size:.3f}"
                if res.effect_size is not None
                else ""
            )
            decision = "拒绝 H0" if res.decision == "reject_h0" else "未能拒绝 H0"
            interp_source = "LLM" if res.interpretation_source == "llm" else "规则模板"
            parts.append(
                f"<li>结果：<b>{stat_txt}</b>，{p_txt} → <b>{decision}</b>{effect}</li>"
                f"<li>样本：有效 {res.n_used} / 总 {res.n_used + res.n_dropped}</li>"
                f"<li>解读（来源：{interp_source}）：{_esc(res.interpretation)}</li></ul>"
            )
    else:
        parts.append("<ul><li>（本次会话未执行实验）</li></ul>")
    sig_records = [r for r in records if r.result.status == "ok" and r.result.decision == "reject_h0"]
    if sig_records:
        parts.append(heading("因果提示") + "<ul>")
        for rec in sig_records:
            confounds = [
                p.column_b if p.column_a in rec.plan.variables else p.column_a
                for p in profile.correlations
                if rec.plan.variables
                and (p.column_a in rec.plan.variables) != (p.column_b in rec.plan.variables)
            ][:2]
            confound_txt = "、".join(confounds) if confounds else "共同原因与选择偏差"
            parts.append(
                f"<li>「{_esc(' → '.join(rec.plan.variables))}」显著不等于因果：需排除 {_esc(confound_txt)} 等混杂，"
                "并进行随机化实验或匹配设计。</li>"
            )
        parts.append("</ul>")
    if ml_result is not None:
        parts.append(heading("ML 基线实验"))
        if ml_result.status != "ok":
            parts.append(f"<ul><li>ML 基线未完成：{_esc(ml_result.reason)}</li></ul>")
        else:
            target_desc = ml_result.target or "（无目标 · 聚类）"
            fingerprint = ml_result.dataset_fingerprint[:8]
            parts.append(
                f"<ul><li>任务：{_esc(TASK_CN.get(ml_result.task, ml_result.task))}"
                f"（目标：{_esc(target_desc)}；数据指纹 {fingerprint}）</li>"
                f"<li>最佳模型：<b>{_esc(ml_result.best_model)}</b>"
                f"（{_esc(ml_result.best_metric_name)} = {ml_result.best_metric_value}）</li></ul>"
            )
            parts.append(
                "<table><tr><th>模型</th><th>超参</th><th>指标</th><th>耗时(秒)</th></tr>"
            )
            for m in ml_result.models:
                params = "、".join(f"{k}={v}" for k, v in m.params.items()) or "—"
                metrics = "，".join(f"{k}={v}" for k, v in m.metrics.items())
                parts.append(
                    f"<tr><td>{_esc(m.model)}</td><td>{_esc(params)}</td>"
                    f"<td>{_esc(metrics)}</td><td>{m.train_seconds:.2f}</td></tr>"
                )
            parts.append("</table>")
            if ml_result.task == "clustering" and ml_result.cluster_sizes:
                sizes = "、".join(f"{k}：{v}" for k, v in ml_result.cluster_sizes.items())
                parts.append(f"<p class=\"meta\">簇规模：{_esc(sizes)}</p>")
            if ml_result.excluded:
                excluded_desc = "；".join(f"{e.column}（{e.reason}）" for e in ml_result.excluded)
                parts.append(f"<p class=\"meta\">已排除特征：{_esc(excluded_desc)}</p>")
            if ml_result.tuning_note:
                parts.append(f"<p class=\"meta\">调优：{_esc(ml_result.tuning_note)}</p>")
            if ml_result.feature_importance:
                parts.append(
                    "<p><b>特征重要性（permutation，top 10）</b></p>"
                    "<table><tr><th>特征</th><th>重要性</th></tr>"
                )
                for i in ml_result.feature_importance:
                    parts.append(f"<tr><td>{_esc(i.feature)}</td><td>{i.importance}</td></tr>")
                parts.append("</table>")
            parts.append("<h3>模型卡片</h3><ul>")
            parts.append(
                f"<li>数据指纹：{fingerprint}　任务：{_esc(TASK_CN.get(ml_result.task, ml_result.task))}　"
                f"目标：{_esc(target_desc)}</li>"
                f"<li>规模：训练/测试 {ml_result.n_train}/{ml_result.n_test}　"
                f"特征：数值 {len(ml_result.features_numeric)} + 类别 {len(ml_result.features_categorical)}</li>"
                "<li>局限：基线模型不构成因果解释；样本量与类别平衡情况见数据画像与质量提示。</li>"
            )
            parts.append("</ul>")
    if references:
        parts.append(heading("相关工作（文献引用）"))
        parts.append(
            "<p class=\"meta\">检索词来自数据字段与研究问题，由 arXiv 关键词检索得到，"
            "非系统性文献综述，仅作入口性参考。</p><ol>"
        )
        for p in references:
            authors = "、".join(p.authors[:3]) + ("等" if len(p.authors) > 3 else "")
            parts.append(
                f"<li><b>{_esc(p.title)}</b> — {_esc(authors)}（{_esc(p.year)}）。"
                f"{_esc(snippet(p.summary))} <a href=\"{_esc(p.url)}\">[abs]</a></li>"
            )
        parts.append("</ol>")
    parts.append(heading("结论与局限"))
    ok_records = [r for r in records if r.result.status == "ok"]
    if ok_records:
        adj_map = _adjusted_p_map(ok_records)
        parts.append(
            "<table><tr><th>实验</th><th>方法</th><th>统计量</th><th>p 值</th><th>BH 校正后</th><th>结论</th></tr>"
        )
        for idx, rec in enumerate(ok_records):
            res = rec.result
            stat_txt = (
                f"{_esc(res.statistic_name)} = {res.statistic:.3f}"
                if res.statistic is not None
                else "—"
            )
            p_txt = format_p(res.p_value) if res.p_value is not None else "—"
            adj = adj_map.get(idx)
            adj_txt = f"{adj:.4f}" if adj is not None else "—"
            decision = "拒绝 H0" if res.decision == "reject_h0" else "未能拒绝 H0"
            decision += _sparse_marker(res)
            method_cn = METHOD_CN.get(rec.plan.method, rec.plan.method)
            parts.append(
                f"<tr><td>{_esc(rec.plan.experiment_id)}</td><td>{_esc(method_cn)}</td>"
                f"<td>{stat_txt}</td><td>{p_txt}</td><td>{adj_txt}</td><td>{_esc(decision)}</td></tr>"
            )
        parts.append("</table>")
    parts.append("<ul>" + "".join(f"<li>{_esc(item)}</li>" for item in LIMITATIONS) + "</ul>")
    repro_ok = [r for r in records if r.result.status == "ok"]
    if repro_ok:
        parts.append(heading("附录：复现脚本"))
        for rec in repro_ok:
            parts.append(
                f"<p><b>{_esc(rec.plan.experiment_id)}</b></p>"
                f"<pre>{_esc(build_repro_script(rec.plan))}</pre>"
            )
    parts.append(f"<hr><p class=\"meta\">本报告由 {GENERATOR} 生成。</p></div></body></html>")
    return "".join(parts)


def build_report(
    profile: ProfileReport,
    dataset_name: str,
    questions: list[ResearchQuestion],
    records: list[ExperimentRecord],
    ml_result: MLExperimentResult | None = None,
    references: list[PaperRef] | None = None,
    images: list[tuple[str, bytes]] | None = None,
) -> tuple[str, str]:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    return (
        build_markdown(
            profile, dataset_name, questions, records, generated, ml_result, references, images
        ),
        build_html(
            profile, dataset_name, questions, records, generated, ml_result, references, images
        ),
    )
