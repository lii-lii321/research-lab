"""在 Python / Notebook 里直接使用 research-lab 的服务层。

在仓库根目录运行：python examples/api_demo.py
（需要先执行 scripts/generate_sample.py 生成示例数据）
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.executor import run_experiment
from services.planner import generate_experiment_plan
from services.profiler import profile_dataset
from services.report import build_report
from services.reports_store import save_report
from services.research_questions import generate_research_questions_auto
from utils.io import read_tabular

csv_path = ROOT / "data" / "samples" / "student_performance.csv"
df = read_tabular(csv_path.name, csv_path.read_bytes())

# 1) 数据画像
report = profile_dataset(df)
print(f"[画像] {report.dataset.n_rows} 行 × {report.dataset.n_cols} 列，{len(report.warnings)} 条质量提示")

# 2) 研究问题（规则模式；配置 .env 后传 client 即可切换 LLM）
questions = generate_research_questions_auto(df, report, None, task="attendance_rate 与 final_score")
q = questions[0]
print(f"[研究问题] {q.id}: {q.question}（变量：{', '.join(q.variables)}）")

# 3) 实验计划 + 真实执行
plan = generate_experiment_plan(q, report, None)
result = run_experiment(plan, df, None)
print(f"[实验] {result.statistic_name} = {result.statistic}, p = {result.p_value} → {result.decision}")
print(f"[解读] {result.interpretation}")

# 4) 报告沉淀
md, html = build_report(report, csv_path.name, questions, [], None, [])
saved = save_report(f"{csv_path.stem}_api", md, html)
print(f"[报告] {saved['md_path']}")
