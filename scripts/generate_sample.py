# -*- coding: utf-8 -*-
"""生成示例数据集 data/samples/student_performance.csv。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.sample_data import build_sample_dataframe

OUT = ROOT / "data" / "samples" / "student_performance.csv"


def main() -> None:
    df = build_sample_dataframe()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"OK wrote {len(df)} rows -> {OUT}")


if __name__ == "__main__":
    main()
