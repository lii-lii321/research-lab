# -*- coding: utf-8 -*-
"""冒烟验证：真实启动 uvicorn 与 streamlit，请求探活后清理进程。"""
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
SAMPLE = ROOT / "data" / "samples" / "student_performance.csv"


def check(url: str, expect_substr: str) -> str:
    with urllib.request.urlopen(url, timeout=10) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        status = resp.status
    assert status == 200, f"{url} -> {status}"
    assert expect_substr.lower() in body.lower(), f"{url} body missing {expect_substr!r}"
    return f"{url} -> {status}"


def post_csv(path: str, fields: dict) -> dict:
    boundary = "----smoketest"
    parts = []
    for name, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        )
    parts.append(
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
            f'filename="student_performance.csv"\r\nContent-Type: text/csv\r\n\r\n'
        ).encode()
        + SAMPLE.read_bytes()
        + f"\r\n--{boundary}--\r\n".encode()
    )
    req = urllib.request.Request(
        f"http://127.0.0.1:8011{path}",
        data=b"".join(parts),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise AssertionError(f"{path} -> {exc.code}: {exc.read().decode('utf-8', 'replace')}")


def main() -> None:
    results = []

    uvicorn_proc = subprocess.Popen(
        [str(PY), "-m", "uvicorn", "main:app", "--port", "8011"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(6)
        results.append(check("http://127.0.0.1:8011/health", '"status":"ok"'))
        body = post_csv("/api/profile", {})
        results.append(f"api profile rows={body['dataset']['n_rows']} warnings={len(body['warnings'])}")
        plan = post_csv(
            "/api/experiment-plan",
            {
                "question": "出勤率与最终成绩是否相关？",
                "variables": "attendance_rate,final_score",
                "question_id": "RQ1",
            },
        )
        results.append(f"api plan method={plan['method']} source={plan['source']}")
        result = post_csv(
            "/api/execute-experiment",
            {
                "method": "pearson",
                "variables": "attendance_rate,final_score",
                "question_id": "RQ1",
            },
        )
        assert result["status"] == "ok", result
        results.append(
            f"api execute {result['statistic_name']}={result['statistic']} "
            f"p={result['p_value']} decision={result['decision']}"
        )
        report = post_csv(
            "/api/report",
            {"payload": '{"questions": [], "experiments": []}', "dataset_name": "student_performance.csv"},
        )
        assert "AI 数据科学研究报告" in report["markdown"], report["filename_base"]
        assert "<!DOCTYPE html>" in report["html"]
        results.append(
            f"api report md={len(report['markdown'])} chars html={len(report['html'])} chars"
        )
    finally:
        uvicorn_proc.terminate()

    streamlit_proc = subprocess.Popen(
        [str(PY), "-m", "streamlit", "run", "app.py",
         "--server.port", "8511", "--server.headless", "true"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(12)
        results.append(check("http://127.0.0.1:8511/", "streamlit"))
    finally:
        streamlit_proc.terminate()
        time.sleep(1)
        if streamlit_proc.poll() is None:
            streamlit_proc.kill()

    for r in results:
        print("SMOKE OK:", r)


if __name__ == "__main__":
    main()
