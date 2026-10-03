"""错误信息人性化：把异常文本映射为"发生了什么 / 为什么 / 你可以试什么"。

单一来源：UI 的错误卡与 API 的 detail 共用本模块的匹配规则。
未命中的消息回退为原始文本 + 通用三步指引（绝不丢弃原始信息）。
"""
from __future__ import annotations

from typing import TypedDict


class ErrorPlaybook(TypedDict):
    title: str
    why: str
    actions: list[str]


# 匹配子串（小写）→ 剧本。顺序即优先级，先命中先得。
_PLAYBOOKS: list[tuple[str, ErrorPlaybook]] = [
    (
        "不支持的文件类型",
        {
            "title": "文件格式暂不支持",
            "why": "目前仅支持 CSV（UTF-8 / GBK 自动识别）与 Excel（.xlsx）。",
            "actions": ["把数据导出为 CSV（UTF-8）或 .xlsx 后重新上传", "确认文件扩展名与真实格式一致"],
        },
    ),
    (
        "无法识别文件编码",
        {
            "title": "文件编码无法识别",
            "why": "已尝试 UTF-8 / GBK 等常见编码均解析失败。",
            "actions": ["用 Excel 打开后另存为 CSV UTF-8 再上传", "检查文件是否损坏或为二进制格式"],
        },
    ),
    (
        "csv 解析失败",
        {
            "title": "表格解析失败",
            "why": "文件能读取但行列结构无法解析。",
            "actions": ["用 Excel 打开检查表头与分隔符是否完整", "另存为标准 CSV（逗号分隔）再上传"],
        },
    ),
    (
        "文件超过",
        {
            "title": "文件超过大小上限",
            "why": "为保护交互流畅度，上传上限为 50MB。",
            "actions": ["删除无关列后重新导出", "先采样一部分行用于本轮分析"],
        },
    ),
    (
        "变量不在数据集中",
        {
            "title": "研究问题的变量与当前数据不匹配",
            "why": "研究问题引用的列在当前数据集中不存在（可能切换了数据）。",
            "actions": ["回到「研究问题」步骤重新生成", "检查列名是否被重命名或大小写不同"],
        },
    ),
    (
        "样本不足",
        {
            "title": "有效数据量不够",
            "why": "当前检验对最小样本量有要求（如配对 ≥3、组间每组 ≥2）。",
            "actions": ["补充数据行数", "减少缺失值以提高有效观测", "或改用对样本量要求更低的方法"],
        },
    ),
    (
        "llm 请求失败",
        {
            "title": "AI 服务暂时不可用",
            "why": "网络异常或服务端错误，已自动重试仍失败。",
            "actions": ["稍后重试", "检查 .env 的 AI_BASE_URL / AI_API_KEY", "规则模式功能不受影响，可继续使用"],
        },
    ),
    (
        "llm 返回 http",
        {
            "title": "AI 服务返回错误",
            "why": "服务端响应了非 200 状态（多为 Key 失效或余额不足）。",
            "actions": [
                "检查 AI_API_KEY 是否有效、余额是否充足",
                "确认 AI_MODEL 名称与服务商一致",
                "规则模式功能不受影响",
            ],
        },
    ),
    (
        "llm 未配置",
        {
            "title": "LLM 尚未配置",
            "why": "需要 API Key 才能使用 LLM 语义理解。",
            "actions": ["复制 .env.example 为 .env 并填入 AI_API_KEY 后重启", "规则模式功能不受影响"],
        },
    ),
    (
        "不存在",
        {
            "title": "找不到对应的记录",
            "why": "请求的报告 / 实验 uid 不存在或已被清理。",
            "actions": ["回到对应标签页确认列表中的名称", "刷新实验追踪页获取最新列表"],
        },
    ),
]

_GENERIC: ErrorPlaybook = {
    "title": "操作未成功",
    "why": "",
    "actions": ["检查输入与数据后重试", "若持续失败，先用 scripts/doctor.py 体检环境"],
}


def explain(message: str) -> ErrorPlaybook:
    """按子串匹配人性化剧本；未命中返回通用剧本（保留原始信息）。"""
    lowered = (message or "").lower()
    for needle, playbook in _PLAYBOOKS:
        if needle in lowered:
            return playbook
    return {
        "title": "操作未成功",
        "why": message or "未知原因",
        "actions": _GENERIC["actions"],
    }


def detail(message: str) -> str:
    """API detail 的人话前缀（UI 与 API 文案同源）。"""
    playbook = explain(message)
    if playbook is _GENERIC:
        return message
    return f"{playbook['title']}：{message}"
