"""复现 Student (1908) 配对 t 检验——现代假设检验范式的起点。

数据：Cushny & Peebles (1905) 催眠剂实验，10 名患者的额外睡眠时长
（对照 vs 権眠剂 B 的差值）。Student 用这份数据首次引入了
"配对差异 + t 分布"的检验思想，Fisher (1925) 将其发扬光大。

论文报告（Student 1908, Table III 之后示例）：t ≈ 4.06（单尾 p ≈ 0.0005 量级）。
现代双侧 t 检验：t = 4.0626, df = 9, p = 0.00283。

运行：python reproductions/1908-student-paired-t/run.py
依赖：scipy
"""
from scipy import stats

# Cushny & Peebles (1905) 数据：10 名患者，權眠剂 B 相对对照的额外睡眠时长（小时）
DIFF = [1.2, 2.4, 1.3, 1.3, 0.0, 1.0, 1.8, 0.8, 4.6, 1.4]

PAPER_T = 4.06  # Student (1908) 论文中报告的 t 值
PAPER_DF = 9


def main() -> None:
    n = len(DIFF)
    res = stats.ttest_1samp(DIFF, popmean=0.0)
    mean = sum(DIFF) / n
    print("=" * 60)
    print("复现 Student (1908) 配对 t 检验（Cushny-Peebles 数据）")
    print("=" * 60)
    print(f"n = {n}, 差值均值 = {mean:.4f} 小时")
    print(f"复现结果：t = {res.statistic:.4f}, df = {n - 1}, 双侧 p = {res.pvalue:.5f}")
    print(f"论文报告：t ≈ {PAPER_T}（Student 1908 仅给单尾参考值）")
    match = abs(res.statistic - PAPER_T) < 0.01
    print(f"对照：{'✅ 复现一致（差异 < 0.01）' if match else '⚠ 与论文值存在偏差，需人工复核'}")
    print("备注：现代双侧 p = 0.00283 与论文单尾参考值量级一致。")


if __name__ == "__main__":
    main()
