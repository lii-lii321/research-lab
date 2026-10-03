"""Research Agent：一句话研究任务自动走完全流程（规则模式无需 LLM）。"""
from __future__ import annotations

import time
from typing import Literal

from models.schemas import (
    AgentRunResult,
    AgentStep,
    ExperimentRecord,
    MLExperimentResult,
    PaperRef,
    ProfileReport,
    ResearchQuestion,
)
from services.executor import run_experiment
from services.literature import search_related
from services.llm import LLMClient
from services.ml_lab import run_ml_experiment
from services.planner import generate_experiment_plan
from services.report import build_report
from services.report_images import collect_profile_images
from services.reports_store import save_report
from services.research_questions import generate_research_questions_auto
from services.tracking import TrackingStore


def run_research_agent(
    df,
    profile: ProfileReport,
    task_description: str = "",
    max_questions: int = 3,
    client: LLMClient | None = None,
    store: TrackingStore | None = None,
    dataset_name: str = "dataset",
    enable_literature: bool = True,
    enable_ml: bool = True,
    on_step=None,
) -> AgentRunResult:
    started = time.perf_counter()
    steps: list[AgentStep] = []
    questions: list[ResearchQuestion] = []
    records: list[ExperimentRecord] = []
    ml_result: MLExperimentResult | None = None
    references: list[PaperRef] = []
    report_md = report_html = ""
    filename_base = "report"

    def step(name: str, fn) -> bool:
        step_started = time.perf_counter()
        try:
            detail = fn()
            record = AgentStep(
                name=name,
                status="ok",
                detail=detail,
                seconds=round(time.perf_counter() - step_started, 3),
            )
            steps.append(record)
            if on_step is not None:
                try:
                    on_step(record)
                except Exception:
                    pass  # 进度回调失败不影响流程
            return True
        except Exception as exc:  # agent 边界：任何失败都进时间线而不是中断会话
            record = AgentStep(
                name=name,
                status="failed",
                detail=str(exc),
                seconds=round(time.perf_counter() - step_started, 3),
            )
            steps.append(record)
            if on_step is not None:
                try:
                    on_step(record)
                except Exception:
                    pass
            return False

    def do_questions() -> str:
        nonlocal questions
        questions = generate_research_questions_auto(
            df, profile, client, task=task_description
        )[: max(1, max_questions)]
        mode = "LLM" if client else "规则"
        return f"提出 {len(questions)} 个研究问题（{mode}模式{'，围绕任务关键词' if task_description else ''}）"

    def do_experiments() -> str:
        nonlocal records
        for q in questions:
            plan = generate_experiment_plan(q, profile, client)
            result = run_experiment(plan, df, client, store=store, dataset_name=dataset_name)
            records.append(ExperimentRecord(plan=plan, result=result))
        done = sum(1 for r in records if r.result.status == "ok")
        return f"执行 {len(records)} 个统计实验（{done} 个成功，已入库追踪）"

    def do_ml() -> str:
        nonlocal ml_result
        candidates = [t.column for t in profile.target_candidates]
        if candidates:
            ml_result = run_ml_experiment(
                df, profile, target=candidates[0], task="auto",
                store=store, dataset_name=dataset_name,
            )
        else:
            ml_result = run_ml_experiment(
                df, profile, target=None, task="clustering",
                store=store, dataset_name=dataset_name,
            )
        if ml_result.status != "ok":
            raise RuntimeError(f"ML 基线未完成：{ml_result.reason}")
        return f"ML 最佳模型 {ml_result.best_model}（{ml_result.best_metric_name} = {ml_result.best_metric_value}）"

    def do_literature() -> str:
        nonlocal references
        references = search_related(profile, questions)
        if not references:
            raise RuntimeError("arXiv 未返回相关文献（网络不可用或无匹配）")
        titles = "；".join(p.title[:40] for p in references)
        return f"检索到 {len(references)} 篇相关文献：{titles}…"

    def do_report() -> str:
        nonlocal report_md, report_html, filename_base
        images = collect_profile_images(df, profile)
        report_md, report_html = build_report(
            profile, dataset_name, questions, records, ml_result, references, images
        )
        filename_base = dataset_name.rsplit(".", 1)[0] or "report"
        if store is not None:
            try:
                saved = save_report(
                    f"{filename_base}_agent", report_md, report_html, images=images
                )
                filename_base = saved["name"]
            except Exception:
                pass  # 沉淀失败不影响 Agent 结果返回
        return f"报告生成（Markdown {len(report_md)} 字符 / HTML {len(report_html)} 字符）"

    step("数据画像", lambda: f"{profile.dataset.n_rows} 行 × {profile.dataset.n_cols} 列，"
         f"{len(profile.warnings)} 条质量提示")
    ok_questions = step("研究问题", do_questions)
    if ok_questions and questions:
        step("统计实验", do_experiments)
    if enable_ml:
        step("ML 基线", do_ml)
    if enable_literature:
        step("文献检索", do_literature)
    step("研究报告", do_report)

    status: Literal["ok", "failed"] = "ok" if (questions and report_md) else "failed"
    reason = "" if status == "ok" else "未能生成研究问题或报告，详见步骤时间线"
    return AgentRunResult(
        status=status,
        reason=reason,
        task_description=task_description or "自动研究",
        steps=steps,
        questions=questions,
        records=records,
        ml_result=ml_result,
        references=references,
        report_markdown=report_md,
        report_html=report_html,
        filename_base=filename_base,
        runtime_seconds=round(time.perf_counter() - started, 3),
    )
