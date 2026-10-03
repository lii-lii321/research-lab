"""错误信息人性化映射的单元测试。"""
import pytest

from services.error_messages import detail, explain


@pytest.mark.parametrize("msg,needle", [
    ("不支持的文件类型 ['.xlsx.enc']", "文件格式暂不支持"),
    ("无法识别文件编码，已尝试", "文件编码无法识别"),
    ("CSV 解析失败：预期 3 列实得 5 列", "表格解析失败"),
    ("文件超过 50MB 上限", "文件超过大小上限"),
    ("变量不在数据集中：ghost", "研究问题的变量与当前数据不匹配"),
    ("有效配对观测不足（2 < 3）", "有效数据量不够"),    ("LLM 请求失败：ConnectError", "AI 服务暂时不可用"),
    ("LLM 返回 HTTP 503", "AI 服务返回错误"),
    ("LLM 未配置", "LLM 尚未配置"),
    ("报告 ghost 不存在", "找不到对应的记录"),
])
def test_explain_hits_correct_playbook(msg: str, needle: str):
    result = explain(msg)
    assert needle in result["title"]
    assert result["actions"]


def test_explain_unknown_falls_back():
    result = explain("一段完全无关的错误描述")
    assert result["actions"]  # 至少有通用建议
    assert result["title"] == "操作未成功"


def test_detail_prefixes_known_errors():
    result = detail("不支持的文件类型 ['.parquet']")
    assert "文件格式暂不支持" in result


def test_detail_unknown_passthrough():
    assert detail("原始信息") == "原始信息"
