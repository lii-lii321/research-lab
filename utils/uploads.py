# -*- coding: utf-8 -*-
"""上传文件 → DataFrame + ProfileReport 的共享加载逻辑。"""
from __future__ import annotations

import pandas as pd
from fastapi import HTTPException, UploadFile

from models.schemas import ProfileReport
from services.profiler import profile_dataset
from utils.io import UnsupportedFileError, read_tabular


async def load_dataset(file: UploadFile) -> tuple[pd.DataFrame, ProfileReport]:
    content = await file.read()
    try:
        df = read_tabular(file.filename or "", content)
    except UnsupportedFileError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        return df, profile_dataset(df)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
