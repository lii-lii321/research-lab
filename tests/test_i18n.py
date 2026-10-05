"""i18n 字典键一致性与 t() 三级回退链测试。"""
from ui.i18n import ALL, EN, ZH, t


def test_zh_en_key_sets_match():
    assert set(ZH) == set(EN)


def test_all_registry_covers_both_languages():
    assert set(ALL) == {"zh", "en"}
    assert ALL["zh"] is ZH
    assert ALL["en"] is EN


def test_translation_fallback_chain():
    # 第一级：语言与 key 均命中
    assert t("app_title", "zh") == ZH["app_title"]
    assert t("app_title", "en") == EN["app_title"]
    # 第二级：未知语言回退中文
    assert t("app_title", "fr") == ZH["app_title"]
    # 第三级：未知 key 回退 key 本身
    assert t("no_such_key", "zh") == "no_such_key"
    assert t("no_such_key", "xx") == "no_such_key"
