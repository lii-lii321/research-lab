# -*- coding: utf-8 -*-
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from models.schemas import AgentRunResult
from routers.deps import get_tracking_store
from services.agent import run_research_agent
from services.llm import LLMClient, get_llm_client
from services.tracking import TrackingStore
from utils.uploads import load_dataset

router = APIRouter(prefix="/api", tags=["agent"])


@router.post("/agent/run", response_model=AgentRunResult)
async def agent_run(
    file: UploadFile,
    task_description: str = Form("自动研究"),
    max_questions: int = Form(3),
    dataset_name: str = Form(""),
    store: Annotated[Optional[TrackingStore], Depends(get_tracking_store)] = None,
    client: Annotated[Optional[LLMClient], Depends(get_llm_client)] = None,
) -> AgentRunResult:
    df, profile = await load_dataset(file)
    name = dataset_name.strip() or file.filename or "dataset"
    return await run_in_threadpool(
        run_research_agent,
        df,
        profile,
        task_description.strip(),
        min(max(max_questions, 1), 5),
        client,
        store,
        name,
    )
