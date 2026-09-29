# -*- coding: utf-8 -*-
import json
from pathlib import PurePosixPath, PureWindowsPath

from typing import Annotated, Optional

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from models.schemas import (
    ExperimentPlan,
    ExperimentRecord,
    ExperimentResult,
    ProfileReport,
    ReportBundle,
    ResearchQuestion,
)
from services.executor import run_experiment
from services.llm import LLMClient, LLMError, get_llm_client
from services.planner import generate_experiment_plan
from services.profiler import profile_dataset
from services.research_questions import generate_research_questions
from services.report import build_report
from utils.io import UnsupportedFileError, read_tabular

router = APIRouter(prefix="/api", tags=["research"])


async def _load_dataset(file: UploadFile) -> tuple[pd.DataFrame, ProfileReport]:
    content = await file.read()
    try:
        df = read_tabular(file.filename or "", content)
    except UnsupportedFileError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        return df, profile_dataset(df)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/research-questions", response_model=list[ResearchQuestion])
async def research_questions(
    file: UploadFile,
    client: Annotated[Optional[LLMClient], Depends(get_llm_client)],
) -> list[ResearchQuestion]:
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="LLM 未配置：复制 .env.example 为 .env 并填写 AI_API_KEY 后重启",
        )
    df, report = await _load_dataset(file)
    try:
        return generate_research_questions(df, report, client)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=f"LLM 输出无法解析：{exc}") from exc


@router.post("/experiment-plan", response_model=ExperimentPlan)
async def experiment_plan(
    file: UploadFile,
    question: str = Form(...),
    variables: str = Form(""),
    question_id: str = Form("RQ1"),
    client: Annotated[Optional[LLMClient], Depends(get_llm_client)] = None,
) -> ExperimentPlan:
    df, report = await _load_dataset(file)
    variables_list = [v.strip() for v in variables.split(",") if v.strip()]
    rq = ResearchQuestion(id=question_id, question=question, variables=variables_list)
    try:
        return generate_experiment_plan(rq, report, client)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/execute-experiment", response_model=ExperimentResult)
async def execute_experiment(
    file: UploadFile,
    method: str = Form(...),
    variables: str = Form(...),
    question: str = Form(""),
    question_id: str = Form("RQ1"),
    hypothesis: str = Form(""),
    h0: str = Form(""),
    h1: str = Form(""),
    alpha: float = Form(0.05),
    client: Annotated[Optional[LLMClient], Depends(get_llm_client)] = None,
) -> ExperimentResult:
    if not 0 < alpha < 0.5:
        raise HTTPException(status_code=422, detail="alpha 必须在 0 与 0.5 之间")
    df, _report = await _load_dataset(file)
    plan = ExperimentPlan(
        experiment_id=f"EXP-{question_id}",
        question_id=question_id,
        hypothesis=hypothesis or question or "（未提供假设）",
        method=method.strip().lower(),
        h0=h0,
        h1=h1,
        alpha=alpha,
        variables=[v.strip() for v in variables.split(",") if v.strip()],
    )
    try:
        return run_experiment(plan, df, client)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/report", response_model=ReportBundle)
async def report_endpoint(
    file: UploadFile,
    payload: str = Form("{}"),
    dataset_name: str = Form(""),
) -> ReportBundle:
    df, profile = await _load_dataset(file)
    try:
        data = json.loads(payload or "{}")
        questions = [ResearchQuestion.model_validate(q) for q in data.get("questions", [])]
        records = [ExperimentRecord.model_validate(r) for r in data.get("experiments", [])]
    except (json.JSONDecodeError, ValidationError) as exc:
        raise HTTPException(status_code=422, detail=f"payload 解析失败：{exc}") from exc
    name = dataset_name.strip() or file.filename or "dataset"
    markdown, html = build_report(profile, name, questions, records)
    stem = PureWindowsPath(name).stem or PurePosixPath(name).stem or "report"
    return ReportBundle(markdown=markdown, html=html, filename_base=stem)
