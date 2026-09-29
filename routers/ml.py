# -*- coding: utf-8 -*-
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from models.schemas import MLExperimentResult, TrackedExperiment
from routers.deps import get_tracking_store
from services.ml_lab import TASKS, run_ml_experiment
from services.tracking import TrackingStore
from utils.uploads import load_dataset

router = APIRouter(prefix="/api", tags=["ml"])


@router.post("/ml-experiment", response_model=MLExperimentResult)
async def ml_experiment(
    file: UploadFile,
    target: str = Form(""),
    task: str = Form("auto"),
    dataset_name: str = Form(""),
    store: Annotated[Optional[TrackingStore], Depends(get_tracking_store)] = None,
) -> MLExperimentResult:
    if task not in TASKS:
        raise HTTPException(status_code=422, detail=f"未知任务类型：{task}（可选 {' / '.join(TASKS)}）")
    df, profile = await load_dataset(file)
    name = dataset_name.strip() or file.filename or "dataset"
    result = await run_in_threadpool(
        run_ml_experiment, df, profile, target.strip() or None, task, store, name
    )
    return result


@router.get("/experiments", response_model=list[TrackedExperiment])
def experiments(
    limit: int = 20,
    store: Annotated[Optional[TrackingStore], Depends(get_tracking_store)] = None,
) -> list[TrackedExperiment]:
    return (store or TrackingStore()).list_experiments(limit=min(max(limit, 1), 100))


@router.get("/experiments/{uid}")
def experiment_detail(
    uid: str,
    store: Annotated[Optional[TrackingStore], Depends(get_tracking_store)] = None,
) -> dict:
    detail = (store or TrackingStore()).get_experiment(uid)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"实验 {uid} 不存在")
    return detail
