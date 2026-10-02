"""LLM 客户端与降级链路的单元测试（真实 httpx 调用层，FakeLLM 之外的盲区）。"""
import httpx
import numpy as np
import pandas as pd
import pytest

import services.llm as llm_module
from services.llm import LLMClient, LLMConfig, LLMError, get_llm_client
from services.profiler import profile_dataset
from services.research_questions import generate_research_questions_auto


def make_client(monkeypatch, handler) -> LLMClient:
    def fake_post(url, json=None, headers=None, timeout=None):
        request = httpx.Request("POST", url, json=json)
        return handler(request)

    monkeypatch.setattr(llm_module.httpx, "post", fake_post)
    return LLMClient(LLMConfig("https://example.test/v1", "key-1", "test-model"))


def ok_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": "好的，解读如下"}}]})


def test_chat_success_returns_content(monkeypatch):
    monkeypatch.setattr(llm_module.time, "sleep", lambda _s: None)
    client = make_client(monkeypatch, ok_handler)
    assert client.chat("system", "user") == "好的，解读如下"


def test_retry_on_503_then_success(monkeypatch):
    calls = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="overloaded")
        return ok_handler(request)

    monkeypatch.setattr(llm_module.time, "sleep", lambda _s: None)
    client = make_client(monkeypatch, flaky)
    assert client.chat("s", "u") == "好的，解读如下"
    assert calls["n"] == 2


def test_retry_exhausted_raises(monkeypatch):
    calls = {"n": 0}

    def always_503(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, text="overloaded")

    monkeypatch.setattr(llm_module.time, "sleep", lambda _s: None)
    client = make_client(monkeypatch, always_503)
    with pytest.raises(LLMError, match="503"):
        client.chat("s", "u")
    assert calls["n"] == 3  # MAX_ATTEMPTS


def test_non_200_raises_llm_error(monkeypatch):
    client = make_client(monkeypatch, lambda request: httpx.Response(500, text="boom"))
    with pytest.raises(LLMError, match="500"):
        client.chat("system", "user")


def test_missing_choices_raises_llm_error(monkeypatch):
    client = make_client(monkeypatch, lambda request: httpx.Response(200, json={"nope": 1}))
    with pytest.raises(LLMError, match="格式异常"):
        client.chat("system", "user")


def test_connect_error_raises_llm_error(monkeypatch):
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("network down")

    client = make_client(monkeypatch, down)
    with pytest.raises(LLMError, match="请求失败"):
        client.chat("system", "user")


def test_from_env_without_key_returns_none(monkeypatch):
    monkeypatch.delenv("AI_API_KEY", raising=False)
    assert LLMConfig.from_env() is None
    assert get_llm_client() is None


def test_from_env_reads_all_three_vars(monkeypatch):
    monkeypatch.setenv("AI_API_KEY", "sk-test")
    monkeypatch.setenv("AI_BASE_URL", "https://api.example.com/v1/")
    monkeypatch.setenv("AI_MODEL", "test-model-9")
    config = LLMConfig.from_env()
    assert config is not None
    assert config.api_key == "sk-test"
    assert config.base_url == "https://api.example.com/v1"  # 尾部斜杠被去掉
    assert config.model == "test-model-9"


def test_research_questions_auto_falls_back_to_rule_on_llm_error(monkeypatch):
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("network down")

    client = make_client(monkeypatch, down)
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "attendance_rate": np.round(np.clip(rng.normal(85, 8, 60), 40, 100), 1),
            "final_score": np.round(rng.normal(70, 10, 60), 1),
        }
    )
    report = profile_dataset(df)
    questions = generate_research_questions_auto(df, report, client)
    assert questions
    assert all(q.source == "rule" for q in questions)
    assert all(q.variables for q in questions)
