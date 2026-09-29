# Changelog

## 0.5.0 - 2026-09-29

- Paper RAG：`services/literature.py` —— 从目标候选与研究问题自动构造检索词，arXiv API 关键词检索 + 词面重排（零嵌入、零 Key），返回 top-N 相关文献
- Agent 新增"文献检索"步骤（时间线可见、失败优雅降级不阻断报告），报告新增"相关工作（文献引用）"章节（MD+HTML，摘要截断、作者折叠、来源声明"非系统性综述"）
- `POST /api/agent/run` 新增 `with_literature` 参数；Streamlit ④ 自动研究页新增开关与引用文献列表
- 测试 102 → 114（fetch 可注入，全部离线可跑；真实 arXiv 由冒烟脚本覆盖）
- 版本口径：文献检索需联网，无网络时自动跳过该步骤并在时间线标注原因

## 0.4.0 - 2026-09-29

- Research Agent：一句话研究任务自动走完 画像 → 研究问题 → 统计实验 → ML 基线 → 研究报告 全流程，带执行时间线（`POST /api/agent/run`，Streamlit ④ 自动研究页）
- 规则化研究问题生成：未配置 LLM 时按目标候选 × 变量类型确定性提出 RQ，零 Key 可跑通全链路；RQ/实验计划/结果解读均带来源标注（llm / rule）
- 报告生成器新增 ML 基线章节（模型对比表、最佳模型、排除特征），章节号动态编排
- 实验追踪深化：`GET /api/experiments/{uid}` 详情、界面行选中查看 JSON、同数据集同任务最佳指标趋势图
- Docker 部署：Dockerfile + docker-compose（api + ui 双服务，data 卷持久化追踪库）
- 测试 97 → 102（含 Streamlit AppTest 界面级冒烟：ML 流程与 Agent 流程真实点击验证）

## 0.3.0 - 2026-09-29

- ML 实验室：回归 / 分类 / 聚类多模型基线（Linear / Logistic / RandomForest / XGBoost / LightGBM），特征守门（排除标识符与高缺失列），任务自动推断，种子 42 可复现
- Experiment Tracking：SQLite 存储数据指纹 / 特征集 / 超参 / 指标 / 耗时，`GET /api/experiments`
- Streamlit 三标签页结构（分析流程 / ML 实验室 / 实验追踪）
- 测试 81 → 97

## 0.2.0 - 2026-09-29

- Phase 1 MVP 闭环：受控统计执行器（10 种 scipy 检验 + 效应量 + H0 决策判定），LLM 结果解读只允许使用真实数字
- 研究报告导出（Markdown + HTML，含复现说明与局限章节），实验历史管理
- 相关矩阵热力图，图表逻辑抽离 `utils/charts.py`
- 测试 41 → 81

## 0.1.0 - 2026-09-28

- 项目启动：Dataset Profiler（类型推断 / 缺失重复 / 离群 / 高相关 / 类别不平衡 / 目标候选）
- FastAPI + Streamlit 双入口，pytest 测试与冒烟脚本
