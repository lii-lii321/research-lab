# AI Data Research Lab

[![CI](https://github.com/lii-lii321/research-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/lii-lii321/research-lab/actions/workflows/ci.yml)

面向数据科学工作流的智能实验平台：从一份 CSV 到一份可复现的研究报告。
AI 只负责提出计划与解释结果 —— 所有数字来自真实执行的代码。

## 当前阶段

**v0.4.0：Research Agent 上线 —— 一句话任务自动产出研究报告**

*Phase 1 MVP（v0.2.0）：画像 → 研究问题 → 实验计划 → 真实执行 → 研究报告*
*Phase 2 核心（v0.3.0）：ML 实验室 + Experiment Tracking*

*Week 1 — Dataset Profiler*（纯确定性分析，不依赖 LLM）

- 表格文件读取：CSV（utf-8 / gbk 自动回退）与 Excel，≤50MB
- 类型推断：数值 / 类别 / 布尔 / 时间 / 标识符 / 文本
- 数据集总览：行列数、缺失、重复、内存、类型分布
- 数值画像：五数概括、偏度峰度、IQR 离群值
- 质量预警：高缺失、恒定列、类别不平衡、高相关字段对、明显偏斜、标识符提示
- 目标变量候选建议

*Week 2 — AI 研究问题 + 实验计划*

- LLM 基于画像提出 3~5 个可检验的研究问题（RQ），附理由、涉及变量、建议方法
- 每个问题生成结构化实验计划：假设 / 统计方法 / H0 / H1 / α
- 统计方法白名单校验：LLM 给出白名单外的方法时按变量类型规则自动校正
- 规则兜底：未配置 LLM 或 LLM 输出不可用时，按变量类型规则生成计划（source=rule，全程可溯源）

*Week 3 — 受控执行器 + 统计检验*

- 真实执行：实验计划用 scipy 真实运行——Pearson/Spearman 相关、独立/Welch/Mann-Whitney 组间比较、
  ANOVA/Kruskal-Wallis 多组比较、配对 t 检验、卡方独立性检验、简单线性回归
- 效应量：r²、Cohen's d / dz、η²、Cramér's V
- 统计结论自动判定（拒绝/未能拒绝 H0），措辞严格区分"不显著"与"H0 成立"
- LLM 结果解读：只允许基于执行产出的真实数字，禁止编造；失败或未配置时规则模板解读兜底
- 数据防护：配对观测数、分组样本量、列联表维度等前置校验，数据不足返回结构化失败原因
- 新增接口：`POST /api/execute-experiment`
- Streamlit 第四步：一键执行 → 统计量/p 值/效应量/有效样本指标卡 + 散点/箱线/柱状可视化 + 结果 JSON 下载

*Week 4 — 可视化打磨 + 研究报告导出*

- 数值字段相关矩阵热力图（画像页）
- 实验历史管理：同一数据集下多次执行自动汇总，换数据或重新生成问题时自动失效
- 一键生成研究报告（Markdown + HTML 双格式）：数据画像、质量提示、研究问题、
  每个实验的假设/方法/真实结果/解读、结论与局限、完整实验汇总表
- 报告内置复现说明与解读来源标注（LLM / 规则模板），HTML 对动态内容做转义
- 新增接口：`POST /api/report`
- 图表逻辑抽离 `utils/charts.py`（matplotlib Agg，微软雅黑，白底藏青）

已交付接口：`POST /api/profile`、`POST /api/research-questions`、`POST /api/experiment-plan`、`POST /api/execute-experiment`、`POST /api/report`

*Phase 2 — ML 基线 + Experiment Tracking（v0.3.0）*

- **ML 实验室**：回归 / 分类 / 聚类多模型自动对比——LinearRegression、LogisticRegression、RandomForest、XGBoost、LightGBM（后两者按可用性自动启用）
- 特征守门：自动排除标识符列、缺失率 >50% 列、时间列；类别列独热编码（低频归并为 other），数值列中位数填充 + 标准化
- 任务自动推断：数值目标 → 回归，类别目标 → 分类，无目标 → 聚类（KMeans，轮廓系数自动选 k∈[2,5]）
- 指标：MAE / RMSE / R²，Accuracy / Precision / Recall / F1（macro）/ 二分类 ROC-AUC，silhouette / inertia
- 固定随机种子 42、分层划分，结果可复现；训练耗时分模型记录
- **Experiment Tracking**：每次实验写入 SQLite（数据指纹 sha256 前 16 位、特征集、模型、超参、指标、耗时、时间戳），`GET /api/experiments` 列表对比
- Streamlit 改为三标签页：① 分析流程 ② ML 实验室 ③ 实验追踪
- 新增接口：`POST /api/ml-experiment`、`GET /api/experiments`

全部接口：`POST /api/profile`、`POST /api/research-questions`、`POST /api/experiment-plan`、`POST /api/execute-experiment`、`POST /api/report`、`POST /api/ml-experiment`、`GET /api/experiments`、`GET /api/experiments/{uid}`、`POST /api/agent/run`

*v0.4.0 — Research Agent*

- 一句话研究任务自动串联：数据画像 → 研究问题 → 统计实验（计划+真实执行）→ ML 基线 → 研究报告，每步带耗时与来源的执行时间线
- 规则化研究问题：未配置 LLM 时按"目标候选 × 变量类型"确定性提出 RQ——**零 Key 可跑通全链路**，所有环节标注 llm / rule 来源
- 报告新增 ML 基线章节（模型对比表 / 最佳模型 / 排除特征），章节号动态编排
- 实验追踪深化：实验详情接口与界面、同数据集同任务的最佳指标趋势图
- Docker 部署（api + ui 双服务，追踪库落卷持久化）

## Docker 部署

```bash
docker compose up --build   # API http://localhost:8000/docs · UI http://localhost:8501
```

## LLM 配置（可选）

```bash
copy .env.example .env    # 填入 AI_API_KEY，其余保持默认即可
```

未配置时研究问题生成不可用，实验计划自动走规则模式。

## 快速开始

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

.venv\Scripts\python scripts\generate_sample.py        # 生成示例数据
.venv\Scripts\python -m streamlit run app.py           # 界面 http://localhost:8501
.venv\Scripts\python -m uvicorn main:app --port 8000   # API 文档 http://localhost:8000/docs
.venv\Scripts\python -m pytest                         # 测试
```

## 目录

```
main.py / app.py          FastAPI 与 Streamlit 入口
routers/                  API 路由（profile / research）
models/                   Pydantic 模型
services/                 核心逻辑（profiler / llm / research_questions / planner / executor / report / ml_lab / tracking）
utils/                    文件读取、图表、上传加载、JSON 提取、示例数据
scripts/                  示例数据生成与冒烟脚本
tests/                    pytest 测试
PROJECT_BRIEF.md          立项书与三阶段路线
```

## 路线

Phase 1 统计分析流水线 → Phase 2 ML 基线 + 实验追踪 → Phase 3 Research Agent + Paper RAG
