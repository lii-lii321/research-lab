# -*- coding: utf-8 -*-
from fastapi import APIRouter, HTTPException

from services.reports_store import get_report, list_reports

router = APIRouter(prefix="/api", tags=["reports"])


@router.get("/reports")
def reports() -> list[dict]:
    return list_reports()


@router.get("/reports/{name}")
def report_detail(name: str) -> dict:
    detail = get_report(name)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"报告 {name} 不存在")
    return detail
