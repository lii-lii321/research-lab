# Changelog

## 0.9.0 - 2026-10-02

落实手册 v0.9.0 批次（B1/B2/E2）：

- ML 三件套：基线最佳模型自动小网格调优（GridSearchCV，网格按模型硬编码；线性模型与 <50 训练样本自动跳过）、joblib 持久化到 data/models/（按追踪 uid 命名）、permutation importance 特征重要性 top10 进报告与界面
- 报告 ML 章节新增调优注记与模型卡片（数据指纹/任务/规模/局限）
- LLM 客户端指数退避重试：429/5xx 与传输异常最多 3 次（secrets 抖动），golden 测试覆盖 503-恢复与耗尽两路径
- 测试 152 → 157 全绿

## 0.8.0 - 2026-10-02

## 0.8.0 - 2026-10-02

落实手册 v0.8.0 批次（D1/D2/D3/A1）：

- 工程门禁：CI 新增 ruff、mypy（29 文件零错误）、pytest-cov 覆盖率门禁 85%（实测 94%）
- 依赖锁定：requirements.in + pip-compile 生成锁定版 requirements.txt
- 统计闭环（statsmodels）：相关系数 Fisher z 95% CI；Welch/t 均值差 95% CI；线性回归 OLS 诊断（调整 R²、F、斜率 CI、Shapiro、Breusch-Pagan）；实验计划自动附功效分析（所需样本量），报告展示 95% CI 与所需样本量
- 测试 145 → 152 全绿；README 用例数漂移修正

## 0.7.0 - 2026-09-30

## 0.7.1 - 2026-10-01

- **执行器退化样本防护**：kruskal 在每组仅 1 条观测时 H/p 仍有效但 ε² 分母为 0——效应量
  优雅置空而非 ZeroDivisionError；全同值样本 scipy 抛 ValueError——统一转为干净的
  `failed` 结果（附原因），API 不再 500；+2 退化场景测试（147 全绿）

## 0.7.0 - 2026-09-30

产品化一轮：从「目的 / 场景 / 方法」三个角度补齐使用体验。

- **场景 · 报告库**：报告自动沉淀到 `data/reports/`（跨会话可回看），新增 `GET /api/reports`、`GET /api/reports/{name}` 与 Streamlit ⑤ 报告库标签页；分析流程、自动研究、CLI 产出的报告统一汇集
- **场景 · 一键体验**：侧边栏「🚀 一键体验完整流程」——上传前即可用示例数据跑通全流程，内嵌展示执行时间线与报告
- **方法 · 命令行**：新增 `cli.py`——`profile`（30 秒看懂一份数据）/ `analyze`（一句话任务、可选 `--skip-ml` `--no-literature` `--no-llm`）/ `experiments` / `reports`，可接进终端与脚本管道
- **方法 · Python API**：`examples/api_demo.py` 演示在 Notebook 中组合服务层
- **文档**：README 新增「为谁设计 / 三个使用入口」与跨平台说明
- 测试 135 → 145 全绿

## 0.6.0 - 2026-09-30

- 监督学习基线升级：分层 k 折交叉验证（k=5/3 按样本量自适应，分类按 y 分层），每模型报告 CV 均值±标准差，最佳模型按 CV 均值选取而非单次 holdout；测试数据补噪声避免零噪声并列
- 统计实验入 tracking 库（kind=stats：方法/变量/α/原始 p/decision/计划来源），Agent 全流程自动入库；每个成功实验附可独立运行的 scipy 复现脚本（`services/repro.py`），嵌入报告"附录：复现脚本"并可单独下载
- task_description 真正驱动研究问题：LLM 模式作为硬约束写入 prompt 首行；规则模式按任务关键词对问题重排序
- 计划层加固：方法白名单与执行层单一来源（`RUNNER_METHODS`）、方法-变量类型组合校验（LLM 给 chi2 配两个数值变量会被规则降级并留痕）、偏度感知兜底（|skew|≥2 → 秩方法）
- 安全与部署加固：上传先查声明大小再读内存（413）、CORS 默认仅本地 Streamlit（`RESEARCH_LAB_CORS` 可覆盖）、compose 仅回环绑定、SQLite 连接超时
- CI/工具链：新增 smoke job（CI 真实起 uvicorn + Streamlit 做九项探活）；`scripts/smoke_check.py` 跨平台化（当前解释器 + 轮询就绪探测替代固定 sleep）；新增 `scripts/wait_for.py`（仅限回环地址的就绪等待）
- 文档：README 重构为"问题→机制→架构→上手"一页式；新增 ADR-0001 记录"白名单分派取代代码生成沙箱"的方法学决策；.env.example 与 PROJECT_BRIEF 过时口径修正
- 测试 124 → 135 全绿 0 跳过

## 0.5.1 - 2026-09-30

修复独立评审核实的 5 个问题：

- 测试真实有效性：CI 构建时先生成示例数据（UI 测试不再在全新检出时整体跳过）、`pytest -rs` 让 skip 透明、ML 流程测试隔离追踪库（不再污染本地实验记录）、删除两条恒真断言（`test_ml_lab.py` 的运算符优先级错误与 `test_agent.py` 的恒真状态断言）
- 统计严谨性：H0 判定改用未舍入的原始 p 值（真实 p=0.04996 不再被舍入成 0.05 而翻转结论，`p_value_raw` 入库）；2×2 稀疏列联表自动切换 Fisher 精确检验（`test_used` 标注实际方法）；Mann-Whitney/Kruskal 补齐 rank-biserial r 与 ε² 效应量；报告结论表新增 Benjamini–Hochberg 校正后 p 列，局限声明注明全部结论为探索性分析，稀疏卡方在结论表标注（稀疏）
- LLM 客户端补齐单测：非 200、畸形响应、断网、无 Key 四分支 + auto 模式 LLMError 规则回退（此前该层零覆盖）
- research 三个路由改 `run_in_threadpool`：LLM 调用最长 60 秒不再阻塞事件循环（对齐 agent/ml 路由的既有模式）

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


