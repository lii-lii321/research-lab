# -*- coding: utf-8 -*-
"""命令行入口测试：直接调用 cli.main，不经过子进程。"""
import pytest

from cli import main
from services.reports_store import get_report, list_reports
from services.tracking import TrackingStore

CSV_ROWS = "\n".join(
    f"{i / 10:.2f},{2 * i / 10 + 1 + ((i * 7) % 5 - 2) * 0.2:.2f},{'M' if i % 2 else 'F'}"
    for i in range(40)
)
CSV = f"hours,score,gender\n{CSV_ROWS}".encode("utf-8")


@pytest.fixture
def csv_file(tmp_path):
    path = tmp_path / "students.csv"
    path.write_bytes(CSV)
    return str(path)


@pytest.fixture
def isolated_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_DB", str(tmp_path / "t.db"))
    monkeypatch.setenv("RESEARCH_LAB_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("AI_API_KEY", "")  # 强制规则模式，测试不碰网络
    monkeypatch.delenv("AI_API_KEY", raising=False)


def test_profile_prints_summary(csv_file, capsys, isolated_dirs):
    assert main(["profile", csv_file]) == 0
    out = capsys.readouterr().out
    assert "40 行 × 3 列" in out
    assert "hours" in out and "score" in out


def test_analyze_full_pipeline_and_saves_report(csv_file, capsys, isolated_dirs):
    code = main(
        [
            "analyze", csv_file,
            "--task", "hours 与 score 的关系",
            "--max-questions", "2",
            "--skip-ml",
            "--no-literature",
            "--name", "cli_demo",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "[OK  ] 数据画像" in out
    assert "[OK  ] 研究问题" in out
    assert "[OK  ] 研究报告" in out
    assert "ML 基线" not in out  # --skip-ml 生效
    saved = get_report("cli_demo")
    assert saved is not None
    assert "AI 数据科学研究报告" in saved["markdown"]
    assert list_reports()


def test_analyze_skip_ml_removes_step(csv_file, capsys, isolated_dirs):
    main(["analyze", csv_file, "--skip-ml", "--no-literature", "--name", "s2"])
    out = capsys.readouterr().out
    assert "ML 基线" not in out


def test_experiments_lists_after_analyze(csv_file, capsys, isolated_dirs):
    main(["analyze", csv_file, "--skip-ml", "--no-literature", "--name", "e1"])
    capsys.readouterr()
    assert main(["experiments", "--limit", "5"]) == 0
    out = capsys.readouterr().out
    assert "[stats/" in out


def test_reports_list_and_show(csv_file, capsys, isolated_dirs):
    main(["analyze", csv_file, "--skip-ml", "--no-literature", "--name", "r1"])
    capsys.readouterr()
    assert main(["reports"]) == 0
    assert "r1" in capsys.readouterr().out
    assert main(["reports", "r1"]) == 0
    assert "AI 数据科学研究报告" in capsys.readouterr().out
    assert main(["reports", "ghost"]) == 1


def test_analyze_missing_file_exits(csv_file, isolated_dirs):
    with pytest.raises(SystemExit):
        main(["analyze", str(csv_file) + ".missing"])

