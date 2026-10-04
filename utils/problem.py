"""RFC 7807 problem+json 错误格式：只对自定义 APIError 生效，不碰 FastAPI 默认行为。"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class APIError(Exception):
    """业务层显式抛出的 API 错误（携带 RFC 7807 所需字段）。"""

    def __init__(self, status_code: int, title: str, detail: str):
        self.status_code = status_code
        self.title = title
        self.detail = detail
        super().__init__(detail)


def install_problem_handler(app: FastAPI) -> None:
    """注册 APIError 的 problem+json handler；FastAPI 默认 HTTPException 行为不变。"""

    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            media_type="application/problem+json",
            content={
                "type": f"https://research-lab.dev/errors/{exc.status_code}",
                "title": exc.title,
                "status": exc.status_code,
                "detail": exc.detail,
                "instance": str(request.url.path),
            },
        )
