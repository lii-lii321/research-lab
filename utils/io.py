# -*- coding: utf-8 -*-
"""表格文件读取：格式与编码自动回退。"""
from __future__ import annotations

from io import BytesIO

import pandas as pd

SUPPORTED_SUFFIXES = {".csv", ".txt", ".xlsx"}
MAX_UPLOAD_MB = 50

_ENCODINGS = ("utf-8-sig", "utf-8", "gbk", "gb18030", "latin-1")


class UnsupportedFileError(ValueError):
    pass


def read_tabular(filename: str, content: bytes) -> pd.DataFrame:
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedFileError(
            f"不支持的文件类型 {suffix or '(无后缀)'}，仅支持 {sorted(SUPPORTED_SUFFIXES)}"
        )
    if not content:
        raise UnsupportedFileError("文件内容为空")
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise UnsupportedFileError(f"文件超过 {MAX_UPLOAD_MB}MB 上限")
    if suffix == ".xlsx":
        try:
            return _normalize(pd.read_excel(BytesIO(content)))
        except Exception as exc:
            raise UnsupportedFileError(f"Excel 解析失败：{exc}") from exc
    last_exc: Exception | None = None
    for enc in _ENCODINGS:
        try:
            return _normalize(pd.read_csv(BytesIO(content), sep=None, engine="python", encoding=enc))
        except UnicodeDecodeError as exc:
            last_exc = exc
        except pd.errors.EmptyDataError as exc:
            raise UnsupportedFileError("文件内容为空") from exc
        except pd.errors.ParserError as exc:
            raise UnsupportedFileError(f"CSV 解析失败：{exc}") from exc
    raise UnsupportedFileError(f"无法识别文件编码，已尝试 {_ENCODINGS}") from last_exc


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    return df
