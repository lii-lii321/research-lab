"""上传文件 → DataFrame + ProfileReport 的共享加载逻辑。"""
from __future__ import annotations

import pandas as pd
from fastapi import HTTPException, UploadFile

from models.schemas import ProfileReport
from services.profiler import profile_dataset
from utils.io import MAX_UPLOAD_MB, UnsupportedFileError, read_tabular


async def load_dataset(file: UploadFile) -> tuple[pd.DataFrame, ProfileReport]:
    # 先查声明大小再读入内存，超限直接 413（否则 50MB 上限形同虚设）
    if file.size is not None and file.size > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"文件超过 {MAX_UPLOAD_MB}MB 上限")
    content = await file.read()
    try:
        df = read_tabular(file.filename or "", content)
    except UnsupportedFileError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        return df, profile_dataset(df)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
