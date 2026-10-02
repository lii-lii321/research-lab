"""LLM 计划与规则基线的一致率评测（需 AI_API_KEY，手动运行）。

用法：python scripts/eval_llm.py [--out data/eval/report.json]
输出：逐案例 LLM 方法 vs 规则方法 vs 期望方法 + 总体一致率。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.schemas import ResearchQuestion  # noqa: E402
from services.llm import get_llm_client  # noqa: E402
from services.planner import generate_experiment_plan, suggest_method  # noqa: E402
from services.profiler import profile_dataset  # noqa: E402
from tests.eval.synth import build_df  # noqa: E402

GOLDEN = ROOT / "tests" / "eval" / "golden_cases.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM 计划与规则基线一致率评测")
    parser.add_argument("--out", default="", help="结果 JSON 输出路径")
    args = parser.parse_args()

    client = get_llm_client()
    if client is None:
        print("未配置 AI_API_KEY：请在 .env 填写后运行")
        return 1

    cases = json.loads(GOLDEN.read_text(encoding="utf-8"))["cases"]
    rows = []
    agree = 0
    for case in cases:
        df = build_df(case["columns"], seed=int(case.get("seed", 0)))
        report = profile_dataset(df)
        rq = ResearchQuestion(
            id=f"E-{case['id']}", question="q", variables=case["variables"]
        )
        try:
            llm_method = generate_experiment_plan(rq, report, client).method
        except Exception as exc:  # LLM 链路任何失败都记为不一致而不是中断评测
            llm_method = f"ERROR: {exc}"
        rule_method = suggest_method(report, case["variables"])
        hit = llm_method == case["expected_method"]
        agree += int(hit)
        rows.append(
            {"id": case["id"], "expected": case["expected_method"], "rule": rule_method, "llm": llm_method, "hit": hit}
        )
        print(f"{'✅' if hit else '❌'} {case['id']}: llm={llm_method} rule={rule_method} expected={case['expected_method']}")

    rate = agree / len(cases) if cases else 0.0
    print(f"\n一致率：{agree}/{len(cases)} = {rate:.0%}")
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {"generated_at": datetime.now().isoformat(timespec="seconds"), "agreement_rate": rate, "rows": rows},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"已写出 {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
