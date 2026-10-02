"""OpenAI 兼容的 LLM 调用层（沿用 AI_BASE_URL / AI_API_KEY / AI_MODEL 约定）。"""
from __future__ import annotations

import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
REQUEST_TIMEOUT = 60.0


class LLMError(RuntimeError):
    pass


class LLMConfig:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    @classmethod
    def from_env(cls) -> LLMConfig | None:
        api_key = os.getenv("AI_API_KEY", "").strip()
        if not api_key:
            return None
        base_url = os.getenv("AI_BASE_URL", DEFAULT_BASE_URL).strip()
        model = os.getenv("AI_MODEL", DEFAULT_MODEL).strip()
        return cls(base_url=base_url, api_key=api_key, model=model)


class LLMClient:
    """OpenAI 兼容 chat/completions 客户端，httpx 直连不引 SDK。"""

    def __init__(self, config: LLMConfig, timeout: float = REQUEST_TIMEOUT):
        self.config = config
        self.timeout = timeout

    def chat(self, system: str, user: str, temperature: float = 0.2) -> str:
        url = f"{self.config.base_url}/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        try:
            resp = httpx.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM 请求失败：{exc}") from exc
        if resp.status_code != 200:
            raise LLMError(f"LLM 返回 HTTP {resp.status_code}：{resp.text[:200]}")
        try:
            return resp.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"LLM 响应格式异常：{resp.text[:200]}") from exc


def get_llm_client() -> LLMClient | None:
    config = LLMConfig.from_env()
    return LLMClient(config) if config else None
