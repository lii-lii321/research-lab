# -*- coding: utf-8 -*-
from fastapi import APIRouter, HTTPException, UploadFile

from models.schemas import ProfileReport
from services.profiler import profile_dataset
from utils.io import UnsupportedFileError, read_tabular

router = APIRouter(prefix="/api", tags=["profile"])


@router.post("/profile", response_model=ProfileReport)
async def profile_upload(file: UploadFile) -> ProfileReport:
    content = await file.read()
    try:
        df = read_tabular(file.filename or "", content)
        return profile_dataset(df)
    except UnsupportedFileError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
