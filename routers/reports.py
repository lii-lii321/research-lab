from fastapi import APIRouter

from services.reports_store import get_report, list_reports
from utils.problem import APIError

router = APIRouter(prefix="/api", tags=["reports"])


@router.get("/reports")
def reports() -> list[dict]:
    return list_reports()


@router.get("/reports/{name}")
def report_detail(name: str) -> dict:
    detail = get_report(name)
    if detail is None:
        raise APIError(404, "报告不存在", f"报告 {name} 不存在")
    return detail
