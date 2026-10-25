"""Benchmark：植入统计效应的合成数据 → research-lab 全流程 → 校验结论与真值一致。

思想移植自 DataAnalystAgent 的 benchmarks（效应即真值 / 检验族宽松匹配 / CI subset 冒烟），
针对本项目增强三点：
1. 断言用结构化 decision 字段（reject_h0 / fail_to_reject_h0），并含阴性对照任务；
2. 方向性断言：组间比较校验各组均值高低，相关分析校验 r/ρ 符号；
3. 全程零写入（store=None、无文献检索、无 ML），不污染实验追踪库与报告库。

用法（仓库根目录）：
  python benchmarks/runner.py                 # 全量任务
  python benchmarks/runner.py --subset 3      # 只跑前 N 个任务（CI 冒烟）
  python benchmarks/runner.py --strict        # 有失败退出码 1（CI 门禁）
  python benchmarks/runner.py --regen         # 强制重建 data/ 下数据集
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.synth import GENERATORS  # noqa: E402
from models.schemas import ExperimentPlan, ExperimentResult  # noqa: E402
from services.agent import run_research_agent  # noqa: E402
from services.profiler import profile_dataset  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"
TASKS_FILE = Path(__file__).resolve().parent / "tasks.jsonl"


def ensure_data(regen: bool = False) -> None:
    """数据集落盘：--regen 强制重建，否则惰性生成（保证本地可跑且确定性）。"""
    DATA_DIR.mkdir(exist_ok=True)
    for name, make in GENERATORS.items():
        path = DATA_DIR / name
        if regen or not path.exists():
            make().to_csv(path, index=False)


def load_tasks() -> list[dict]:
    return [
        json.loads(line)
        for line in TASKS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _matched_pairs(
    records: list[tuple[ExperimentPlan, ExperimentResult]], expect: dict
) -> list[tuple[ExperimentPlan, ExperimentResult]]:
    """按期望筛实验记录：先按检验族（子串匹配），再按变量组合精确过滤。"""
    test_any = expect.get("test_any")
    pairs = records
    if test_any:
        pairs = [(p, r) for p, r in pairs if any(name in r.method for name in test_any)]
    want_vars = set(expect.get("variables") or [])
    if want_vars:
        exact = [(p, r) for p, r in pairs if set(p.variables) == want_vars]
        pairs = exact or [(p, r) for p, r in pairs if want_vars.issubset(set(p.variables))]
    return pairs


def _check_direction(
    pairs: list[tuple[ExperimentPlan, ExperimentResult]], direction, problems: list[str]
) -> None:
    pool = [(p, r) for p, r in pairs if r.decision == "reject_h0"] or pairs
    if isinstance(direction, str):
        corr = [(p, r) for p, r in pool if r.statistic_name in ("r", "ρ")]
        if not corr:
            problems.append("direction check: 无相关分析记录")
        elif direction == "positive" and not any((r.statistic or 0) > 0 for _, r in corr):
            problems.append(f"期望正相关，实测 {[r.statistic for _, r in corr]}")
        elif direction == "negative" and not any((r.statistic or 0) < 0 for _, r in corr):
            problems.append(f"期望负相关，实测 {[r.statistic for _, r in corr]}")
        return
    highs = [g for _, r in pool for g in r.groups if g.group == direction.get("higher")]
    lows = [g for _, r in pool for g in r.groups if g.group == direction.get("lower")]
    if not highs or not lows:
        problems.append(f"direction check: 找不到组 {direction}")
    elif not any(
        h.mean is not None and lo.mean is not None and h.mean > lo.mean for h in highs for lo in lows
    ):
        problems.append(
            f"期望 {direction.get('higher')} > {direction.get('lower')}，实测 "
            f"{[(g.group, g.mean) for _, r in pool for g in r.groups]}"
        )


def run_pipeline_expectations(df: pd.DataFrame, task: dict) -> tuple[bool, str]:
    """跑完 research-lab 全流程并校验期望；与文件解耦，测试可直接注入 DataFrame。"""
    expect = task.get("expect", {})
    profile = profile_dataset(df)
    result = run_research_agent(
        df,
        profile,
        task_description=task["question"],
        max_questions=3,
        client=None,
        store=None,
        dataset_name=task.get("dataset", "benchmark"),
        enable_literature=False,
        enable_ml=False,
    )
    problems: list[str] = []
    if result.status != "ok":
        timeline = "; ".join(
            f"{s.name}:{s.status}({s.detail[:80]})" for s in result.steps
        )
        return False, f"agent 未完成：{result.reason}｜时间线 {timeline}"

    records = [(r.plan, r.result) for r in result.records if r.result.status == "ok"]

    for needle in expect.get("report_contains", []):
        if needle not in result.report_markdown:
            problems.append(f"report missing {needle!r}")

    pairs = _matched_pairs(records, expect)
    if expect.get("test_any") and not pairs:
        problems.append(
            f"未执行任何期望的检验 {expect['test_any']}（实际执行："
            f"{sorted({(r.plan.method, tuple(r.plan.variables)) for r in records})}）"
        )
        return False, "; ".join(problems)

    if expect.get("decision"):
        want = expect["decision"]
        if not any(r.decision == want for _, r in pairs):
            problems.append(
                f"期望 decision={want}，实测 "
                f"{[(r.method, r.decision, r.p_value) for _, r in pairs]}"
            )
    if expect.get("direction") and pairs:
        _check_direction(pairs, expect["direction"], problems)

    return (not problems), "; ".join(problems)


def run_task(task: dict) -> tuple[bool, str]:
    df = pd.read_csv(DATA_DIR / task["dataset"])
    return run_pipeline_expectations(df, task)


def main() -> int:
    parser = argparse.ArgumentParser(description="research-lab 真值基准")
    parser.add_argument("--subset", type=int, default=0, help="只跑前 N 个任务（0 = 全部）")
    parser.add_argument("--strict", action="store_true", help="有失败时退出码 1（CI 门禁）")
    parser.add_argument("--regen", action="store_true", help="强制重建 benchmarks/data/ 数据集")
    args = parser.parse_args()

    ensure_data(regen=args.regen)
    tasks = load_tasks()
    if args.subset:
        tasks = tasks[: args.subset]

    passed = 0
    failures: list[str] = []
    for task in tasks:
        ok, detail = run_task(task)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {task['id']}: {task['question']}" + (f" — {detail}" if detail else ""))
        if ok:
            passed += 1
        else:
            failures.append(task["id"])

    print(f"\nbenchmark: {passed}/{len(tasks)} tasks passed")
    if failures:
        print("failed:", ", ".join(failures))
    if args.strict and failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
