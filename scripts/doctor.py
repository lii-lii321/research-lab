"""环境体检：一条命令回答"我能不能跑、缺什么"。

用法：python scripts/doctor.py
退出码：0 = 全部健康；1 = 存在缺失项（列出修复命令）。
"""
from __future__ import annotations

import importlib
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, fix: str = "") -> None:
    CHECKS.append((name, ok, fix))


def _env_has_key(env_path: Path, var_name: str) -> bool:
    """检查 env 文件中某变量是否已配置非占位值（只看长度，不读值内容）。"""
    if not env_path.exists():
        return False
    for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() == var_name and len(value.strip()) > 8:
            return True
    return False


def main() -> int:
    v = sys.version_info
    check(f"Python {v.major}.{v.minor}", v.major == 3 and v.minor >= 10, "安装 Python 3.10+")

    for mod in ("fastapi", "streamlit", "pandas", "scipy", "sklearn", "plotly"):
        try:
            importlib.import_module(mod)
            check(f"依赖 {mod}", True)
        except ImportError:
            check(f"依赖 {mod}", False, f"pip install {mod}")

    # fpdf2 的 PyPI 包名是 fpdf2，但 import 名是 fpdf
    for mod, import_name in (("fpdf2", "fpdf"),):
        try:
            importlib.import_module(import_name)
            check(f"依赖 {mod}", True)
        except ImportError:
            check(f"依赖 {mod}", False, f"pip install {mod}")

    try:
        importlib.import_module("statsmodels")
        check("依赖 statsmodels（功效分析）", True)
    except ImportError:
        check("依赖 statsmodels（功效分析）", False, "pip install statsmodels")

    check(
        "LLM API Key（.env）",
        _env_has_key(ROOT / ".env", "AI_API_KEY"),
        "复制 .env.example 为 .env 并填入 AI_API_KEY",
    )

    for d in ("data/samples", "data/tracking", "data/reports"):
        check(f"目录 {d}", (ROOT / d).exists(), f"mkdir {d}（或运行任一分析自动创建）")

    for port in (8000, 8501):
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", port))
            check(f"端口 {port} 空闲", True)
        except OSError:
            check(f"端口 {port} 空闲", False, f"关闭占用 {port} 的进程或换端口")
        finally:
            s.close()

    failed = [(n, f) for n, ok, f in CHECKS if not ok]
    for name, ok, _ in CHECKS:
        print(("✅" if ok else "❌") + " " + name)
    if failed:
        print("\n修复建议：")
        for name, fix in failed:
            if fix:
                print(f"  - {name}: {fix}")
        return 1
    print("\n全部健康，可以开始使用。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
