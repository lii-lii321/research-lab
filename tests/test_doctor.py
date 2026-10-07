"""doctor 体检脚本测试（mock 依赖与环境，不依赖真实系统状态）。"""

import scripts.doctor as doctor


def test_env_has_key_true_when_valid(tmp_path):
    env = tmp_path / ".env"
    env.write_text("AI_API_KEY=sk-1234567890abcdef\n", encoding="utf-8")
    assert doctor._env_has_key(env, "AI_API_KEY") is True


def test_env_has_key_false_when_missing_or_short(tmp_path):
    env = tmp_path / ".env"
    env.write_text("AI_API_KEY=\nOTHER=1\n", encoding="utf-8")
    assert doctor._env_has_key(env, "AI_API_KEY") is False
    assert doctor._env_has_key(tmp_path / "missing.env", "AI_API_KEY") is False


def test_main_runs_and_returns_int(tmp_path, monkeypatch):
    # 隔离：目录检查基于脚本自身 ROOT，不 monkeypatch；只验证可执行且退出码为 0/1
    code = doctor.main()
    assert code in (0, 1)
