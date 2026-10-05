"""RFC 7807 problem+json 与 APIError 的单元测试。"""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from utils.problem import APIError, install_problem_handler


def make_app() -> TestClient:
    app = FastAPI()
    install_problem_handler(app)

    @app.get("/boom")
    def boom() -> None:
        raise APIError(418, "示例错误", "这是一个测试错误")

    @app.get("/ok")
    def ok() -> dict:
        return {"status": "ok"}

    return TestClient(app)


def test_api_error_returns_problem_json():
    client = make_app()
    resp = client.get("/boom")
    assert resp.status_code == 418
    assert resp.headers["content-type"].startswith("application/problem+json")
    body = resp.json()
    assert body["title"] == "示例错误"
    assert body["status"] == 418
    assert body["detail"] == "这是一个测试错误"
    assert body["instance"] == "/boom"
    assert body["type"].endswith("/418")


def test_normal_endpoint_unaffected():
    client = make_app()
    resp = client.get("/ok")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_real_app_routes_raise_problem_json():
    """打真实 main.app：路由层 APIError 经已注册 handler 产出 problem+json。"""
    from main import app

    real = TestClient(app)
    resp = real.get("/api/reports/no-such-report")
    assert resp.status_code == 404
    assert resp.headers["content-type"].startswith("application/problem+json")
    body = resp.json()
    assert body["title"] == "报告不存在"
    assert body["status"] == 404
    assert body["instance"] == "/api/reports/no-such-report"
    assert body["type"].endswith("/404")
