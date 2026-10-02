"""从 LLM 输出中稳健提取 JSON（容忍代码围栏与前后缀文本）。"""
from __future__ import annotations

import json
import re

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$", flags=re.MULTILINE)


def strip_code_fences(text: str) -> str:
    return _FENCE_RE.sub("", text.strip()).strip()


def extract_json_array(text: str) -> list:
    body = strip_code_fences(text)
    start, end = body.find("["), body.rfind("]")
    if start == -1 or end <= start:
        raise ValueError("输出中未找到 JSON 数组")
    data = json.loads(body[start : end + 1])
    if not isinstance(data, list):
        raise ValueError("JSON 内容不是数组")
    return data


def extract_json_object(text: str) -> dict:
    body = strip_code_fences(text)
    start, end = body.find("{"), body.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("输出中未找到 JSON 对象")
    data = json.loads(body[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("JSON 内容不是对象")
    return data
