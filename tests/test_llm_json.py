# -*- coding: utf-8 -*-
import pytest

from utils.textjson import extract_json_array, extract_json_object


def test_extract_array_from_fenced():
    raw = "```json\n[{\"a\": 1}]\n```"
    assert extract_json_array(raw) == [{"a": 1}]


def test_extract_array_with_leading_text():
    raw = "好的，以下是研究问题：\n[{\"a\": 1}, {\"a\": 2}]"
    assert len(extract_json_array(raw)) == 2


def test_extract_array_missing_raises():
    with pytest.raises(ValueError):
        extract_json_array("没有任何 JSON")


def test_extract_array_not_array_raises():
    with pytest.raises(ValueError):
        extract_json_array("{\"a\": 1}")


def test_extract_object_from_fenced():
    raw = "```json\n{\"method\": \"pearson\"}\n```"
    assert extract_json_object(raw)["method"] == "pearson"


def test_extract_object_missing_raises():
    with pytest.raises(ValueError):
        extract_json_object("只是普通文字")
