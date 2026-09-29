# -*- coding: utf-8 -*-
import json

import pytest
from fastapi.testclient import TestClient

from main import app
from services.llm import get_llm_client

client = TestClient(app)

CSV = (
    "attendance_rate,final_score,gender\n80,70,M\n90,85,F\n70,60,M\n85,90,F\n"
).encode("utf-8")

RQ_JSON = json.dumps(
    [
        {
            "id": "RQ1",
            "question": "出勤率与最终成绩是否相关？",
            "variables": ["attendance_rate", "final_score"],
        }
    ],
    ensure_ascii=False,
)
PLAN_JSON = json.dumps(
    {"hypothesis": "h", "method": "pearson", "h0": "r=0", "h1": "r!=0", "alpha": 0.05},
    ensure_ascii=False,
)


class FakeLLM:
    def __init__(self, response: str):
        self.response = response

    def chat(self, system: str, user: str, temperature: float = 0.2) -> str:
        return self.response


@pytest.fixture
def override_llm():
    def _set(fake):
        app.dependency_overrides[get_llm_client] = lambda: fake

    yield _set
    app.dependency_overrides.pop(get_llm_client, None)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_profile_csv():
    content = "a,b\n1,x\n2,y\n3,z\n4,x\n".encode("utf-8")
    resp = client.post("/api/profile", files={"file": ("t.csv", content, "text/csv")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["dataset"]["n_rows"] == 4
    assert {c["name"] for c in body["columns"]} == {"a", "b"}


def test_profile_rejects_unsupported_suffix():
    resp = client.post(
        "/api/profile", files={"file": ("t.parquet", b"1", "application/octet-stream")}
    )
    assert resp.status_code == 400


def test_profile_rejects_empty():
    resp = client.post("/api/profile", files={"file": ("t.csv", b"", "text/csv")})
    assert resp.status_code == 400


def test_research_questions_rule_mode_without_llm(override_llm):
    override_llm(None)
    resp = client.post("/api/research-questions", files={"file": ("t.csv", CSV, "text/csv")})
    assert resp.status_code == 200
    body = resp.json()
    assert body
    assert all(q["source"] == "rule" for q in body)
    assert all(q["variables"] for q in body)


def test_research_questions_with_fake_llm(override_llm):
    override_llm(FakeLLM(f"```json\n{RQ_JSON}\n```"))
    resp = client.post("/api/research-questions", files={"file": ("t.csv", CSV, "text/csv")})
    assert resp.status_code == 200
    body = resp.json()
    assert body[0]["id"] == "RQ1"
    assert body[0]["variables"] == ["attendance_rate", "final_score"]
    assert body[0]["source"] == "llm"


def test_experiment_plan_rule_fallback_without_llm(override_llm):
    override_llm(None)
    resp = client.post(
        "/api/experiment-plan",
        files={"file": ("t.csv", CSV, "text/csv")},
        data={
            "question": "出勤率与成绩相关？",
            "variables": "attendance_rate,final_score",
            "question_id": "RQ1",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "rule"
    assert body["method"] == "pearson"


def test_experiment_plan_with_fake_llm(override_llm):
    override_llm(FakeLLM(PLAN_JSON))
    resp = client.post(
        "/api/experiment-plan",
        files={"file": ("t.csv", CSV, "text/csv")},
        data={"question": "q", "variables": "attendance_rate,final_score"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "llm"
    assert body["method"] == "pearson"


def test_execute_rule_mode(override_llm):
    override_llm(None)
    resp = client.post(
        "/api/execute-experiment",
        files={"file": ("t.csv", CSV, "text/csv")},
        data={
            "method": "pearson",
            "variables": "attendance_rate,final_score",
            "question_id": "RQ1",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["statistic_name"] == "r"
    assert body["p_value"] is not None
    assert body["decision"] in ("reject_h0", "fail_to_reject_h0")
    assert body["interpretation_source"] == "rule"
    assert body["interpretation"]


def test_execute_llm_interpretation(override_llm):
    override_llm(FakeLLM("这是一段基于真实数字的解读。"))
    resp = client.post(
        "/api/execute-experiment",
        files={"file": ("t.csv", CSV, "text/csv")},
        data={"method": "welch_ttest", "variables": "gender,final_score"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["interpretation_source"] == "llm"
    assert body["interpretation"] == "这是一段基于真实数字的解读。"


def test_execute_bad_alpha_rejected(override_llm):
    override_llm(None)
    resp = client.post(
        "/api/execute-experiment",
        files={"file": ("t.csv", CSV, "text/csv")},
        data={"method": "pearson", "variables": "attendance_rate,final_score", "alpha": "1.5"},
    )
    assert resp.status_code == 422


def test_execute_missing_variable_rejected(override_llm):
    override_llm(None)
    resp = client.post(
        "/api/execute-experiment",
        files={"file": ("t.csv", CSV, "text/csv")},
        data={"method": "pearson", "variables": "ghost,score"},
    )
    assert resp.status_code == 422


def test_execute_unknown_method_returns_failed(override_llm):
    override_llm(None)
    resp = client.post(
        "/api/execute-experiment",
        files={"file": ("t.csv", CSV, "text/csv")},
        data={"method": "magic_test", "variables": "attendance_rate,final_score"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "failed"
    assert "不支持" in body["reason"]


def test_report_endpoint(override_llm):
    override_llm(None)
    payload = json.dumps(
        {
            "questions": [
                {"id": "RQ1", "question": "出勤率与成绩相关？", "variables": ["attendance_rate", "final_score"]}
            ],
            "experiments": [],
        },
        ensure_ascii=False,
    )
    resp = client.post(
        "/api/report",
        files={"file": ("student.csv", CSV, "text/csv")},
        data={"payload": payload, "dataset_name": "student.csv"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["filename_base"] == "student"
    assert "AI 数据科学研究报告" in body["markdown"]
    assert "RQ1" in body["markdown"]
    assert "<!DOCTYPE html>" in body["html"]
    assert "未执行实验" in body["markdown"]


def test_report_endpoint_bad_payload(override_llm):
    override_llm(None)
    resp = client.post(
        "/api/report",
        files={"file": ("t.csv", CSV, "text/csv")},
        data={"payload": "{not json"},
    )
    assert resp.status_code == 422


def test_report_endpoint_defaults(override_llm):
    override_llm(None)
    resp = client.post("/api/report", files={"file": ("t.csv", CSV, "text/csv")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["filename_base"] == "t"


def test_ml_experiment_and_listing(override_llm, tmp_path):
    from routers.ml import get_tracking_store
    from services.tracking import TrackingStore

    override_llm(None)
    app.dependency_overrides[get_tracking_store] = lambda: TrackingStore(tmp_path / "t.db")
    try:
        rows = []
        for i in range(40):
            x = i / 10
            rows.append(f"{x:.2f},{2 * x + 1:.2f}")
        csv_data = ("x,y\n" + "\n".join(rows)).encode("utf-8")
        resp = client.post(
            "/api/ml-experiment",
            files={"file": ("lin.csv", csv_data, "text/csv")},
            data={"target": "y", "task": "regression", "dataset_name": "lin.csv"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["task"] == "regression"
        assert body["tracked_uid"]
        assert body["dataset_fingerprint"]
        listed = client.get("/api/experiments")
        assert listed.status_code == 200
        assert any(e["uid"] == body["tracked_uid"] for e in listed.json())
    finally:
        app.dependency_overrides.pop(get_tracking_store, None)


def test_ml_experiment_bad_task(override_llm):
    override_llm(None)
    resp = client.post(
        "/api/ml-experiment",
        files={"file": ("t.csv", CSV, "text/csv")},
        data={"task": "sorcery"},
    )
    assert resp.status_code == 422
    assert "未知任务" in resp.json()["detail"]


def test_agent_run_rule_mode(override_llm, tmp_path):
    from routers.deps import get_tracking_store
    from services.tracking import TrackingStore

    override_llm(None)
    app.dependency_overrides[get_tracking_store] = lambda: TrackingStore(tmp_path / "agent.db")
    try:
        rows = []
        for i, x in enumerate([v / 10 for v in range(40)]):
            rows.append(f"{x:.2f},{2 * x + 1:.2f},{'M' if i % 2 else 'F'}")
        csv_data = ("hours,score,gender\n" + "\n".join(rows)).encode("utf-8")
        resp = client.post(
            "/api/agent/run",
            files={"file": ("lin.csv", csv_data, "text/csv")},
            data={
                "task_description": "研究影响成绩的因素",
                "max_questions": "2",
                "dataset_name": "lin.csv",
                "with_literature": "false",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert [s["name"] for s in body["steps"]] == [
            "数据画像", "研究问题", "统计实验", "ML 基线", "研究报告",
        ]
        assert body["references"] == []
        assert body["questions"] and all(q["source"] == "rule" for q in body["questions"])
        assert "ML 基线实验" in body["report_markdown"]
        listed = client.get("/api/experiments")
        assert any(e["uid"] == body["ml_result"]["tracked_uid"] for e in listed.json())
        uid = body["ml_result"]["tracked_uid"]
        detail = client.get(f"/api/experiments/{uid}")
        assert detail.status_code == 200
        assert detail.json()["dataset_name"] == "lin.csv"
        missing = client.get("/api/experiments/nonexistent")
        assert missing.status_code == 404
    finally:
        app.dependency_overrides.pop(get_tracking_store, None)
