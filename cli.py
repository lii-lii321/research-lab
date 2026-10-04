"""命令行入口：把 research-lab 接进终端与脚本管道。

用法（在仓库根目录）：
  python cli.py profile <file>
  python cli.py analyze <file> [--task T] [--max-questions N]
                        [--skip-ml] [--no-literature] [--no-llm] [--name NAME]
  python cli.py experiments [--limit N]
  python cli.py reports
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathlib import Path

from services.agent import run_research_agent  # noqa: E402
from services.llm import get_llm_client  # noqa: E402
from services.profiler import profile_dataset  # noqa: E402
from services.reports_store import export_bundle, get_report, list_reports, save_report  # noqa: F401
from services.tracking import TrackingStore  # noqa: E402
from utils.io import read_tabular  # noqa: E402


def _load(file_path: str):
    path = Path(file_path)
    if not path.exists():
        raise SystemExit(f"文件不存在：{path}")
    return read_tabular(path.name, path.read_bytes())


def cmd_profile(args: argparse.Namespace) -> int:
    df = _load(args.file)
    report = profile_dataset(df)
    d = report.dataset
    print(f"{d.n_rows} 行 × {d.n_cols} 列，缺失 {d.missing_rate:.1%}，重复行 {d.duplicate_rows}")
    for c in report.columns:
        print(f"  - {c.name} [{c.type}] 缺失 {c.missing_rate:.0%} 唯一值 {c.n_unique}")
    for w in report.warnings:
        print(f"  ! {w.code}: {w.message}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    df = _load(args.file)
    profile = profile_dataset(df)
    client = None if args.no_llm else get_llm_client()
    result = run_research_agent(
        df,
        profile,
        task_description=args.task or "自动研究",
        max_questions=args.max_questions,
        client=client,
        store=TrackingStore(),
        dataset_name=Path(args.file).name,
        enable_literature=not args.no_literature,
        enable_ml=not args.skip_ml,
    )
    for s in result.steps:
        mark = "OK  " if s.status == "ok" else "FAIL"
        print(f"[{mark}] {s.name}（{s.seconds}s）— {s.detail}")
    if result.status != "ok":
        print(f"分析未完成：{result.reason}")
        return 1
    saved = save_report(args.name or f"{result.filename_base}_cli", result.report_markdown, result.report_html)
    print(f"报告已沉淀：{saved['md_path']}")
    print(f"            {saved['html_path']}")
    print("下一步：python cli.py reports 查看全部报告；python cli.py experiments 回看实验")
    return 0


def cmd_experiments(args: argparse.Namespace) -> int:
    rows = TrackingStore().list_experiments(args.limit)
    if not rows:
        print("暂无实验记录")
        return 0
    for r in rows:
        metric = f"{r.best_metric_name}={r.best_metric_value}" if r.best_metric_name else "-"
        print(
            f"{r.created_at}  [{r.kind}/{r.task}] {r.dataset_name}（{r.n_rows} 行）"
            f"最佳 {r.best_model} {metric} uid={r.uid}"
        )
    return 0


def cmd_reports(args: argparse.Namespace) -> int:
    if args.name:
        detail = get_report(args.name)
        if detail is None:
            print(f"报告不存在：{args.name}")
            return 1
        if args.export:
            from services.reports_store import export_bundle

            bundle = export_bundle(args.name)
            if bundle is None:
                print(f"打包失败：{args.name}")
                return 1
            out = Path(f"{args.name}_bundle.zip")
            out.write_bytes(bundle)
            print(f"研究包已导出：{out}（{len(bundle):,} 字节）")
            return 0
        print(detail["markdown"])
        return 0
    items = list_reports()
    if not items:
        print("暂无报告——先运行 analyze 生成一份")
        return 0
    for i in items:
        print(f"{i['modified']}  {i['name']}  {i['size_bytes']:,} 字符")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="research-lab", description="AI Data Research Lab 命令行")
    parser.add_argument("--version", action="version", version="research-lab 1.1.0")
    sub = parser.add_subparsers(dest="command", required=True)

    p_profile = sub.add_parser("profile", help="数据画像摘要")
    p_profile.add_argument("file")

    p_analyze = sub.add_parser("analyze", help="一句话任务 → 全自动研究 → 报告沉淀")
    p_analyze.add_argument("file")
    p_analyze.add_argument("--task", default="", help="研究任务描述")
    p_analyze.add_argument("--max-questions", type=int, default=3)
    p_analyze.add_argument("--skip-ml", action="store_true", help="跳过 ML 基线")
    p_analyze.add_argument("--no-literature", action="store_true", help="跳过 arXiv 文献检索")
    p_analyze.add_argument("--no-llm", action="store_true", help="强制规则模式（不用 LLM）")
    p_analyze.add_argument("--name", default="", help="报告文件名（默认取数据文件名）")

    p_exp = sub.add_parser("experiments", help="列出已追踪的实验")
    p_exp.add_argument("--limit", type=int, default=10)

    p_reports = sub.add_parser("reports", help="列出或查看报告")
    p_reports.add_argument("name", nargs="?", default="", help="报告名（省略则列出全部）")
    p_reports.add_argument("--export", action="store_true", help="导出为 zip 研究包（需 --name）")

    args = parser.parse_args(argv)
    handlers = {
        "profile": cmd_profile,
        "analyze": cmd_analyze,
        "experiments": cmd_experiments,
        "reports": cmd_reports,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
