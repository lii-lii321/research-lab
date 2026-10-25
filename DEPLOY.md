# 部署指南：Streamlit Cloud 公开 demo

## 为什么选 Streamlit Cloud

- 免费、GitHub 直连、推送 main 分支自动重新部署；
- 应用默认走**规则模式（零 API Key）**：示例数据 → 画像 → 研究问题 → 统计实验 → 报告全流程可完整体验；
- 数据目录全部按需自建（`reports_store` / `tracking` / `ml_lab` 均有 `mkdir(parents=True, exist_ok=True)`），全新检出可直接启动。

## 部署步骤（约 5 分钟）

1. 登录 [share.streamlit.io](https://share.streamlit.io)（用 GitHub 账号授权）；
2. **Workspaces → New app**；
3. Repository 选 `lii-lii321/research-lab`，Branch `main`；
4. **Main file path** 填 `app.py`；
5. （可选）Advanced settings 里把 Python 版本固定为 **3.10**（与 CI 矩阵、锁定依赖的编译基线一致；3.11–3.13 的依赖 wheel 本地已验证可用，不固定也能装）；
6. 点 **Deploy**。首次构建会从 `requirements.txt` 安装锁定依赖（文件头部为清华镜像源，公网可访问，安装约 3–5 分钟）。

## 云端能力边界（如实声明）

| 能力 | 云端状态 |
|---|---|
| 示例数据画廊（学生成绩 / 门店销售 / 医疗随访） | ✅ 默认数据源，打开即用 |
| 分析流程 / ML 实验室 / 自动研究 / 报告生成与下载 | ✅ 完整可用（规则模式） |
| arXiv 文献检索 | ⚠️ 依赖 Cloud 服务器对 arXiv API 的外网访问，不可达时自动跳过并标注 |
| LLM 模式（AI 研究问题与解读） | ⚠️ 默认关闭。服务端只读 `AI_API_KEY` / `AI_BASE_URL` / `AI_MODEL` 环境变量，而 Streamlit Cloud 的 secrets 不注入 `os.environ`——云端请使用规则模式；如需云端 LLM，建议后续给 `services/llm.py` 增加 `st.secrets` 回退（未实现） |
| 实验追踪库 / 报告库持久化 | ⚠️ SQLite 与报告文件写在容器本地盘，**应用重启或休眠唤醒后会清空**——演示语义，不是生产部署 |

## 替代方案：云主机 Docker

仓库自带 `Dockerfile` 与 `docker-compose.yml`（API + UI 双服务），需要数据持久化或多用户时改用自托管：

```bash
docker compose up -d        # API 8000 / UI 8501
```

配置要点与端口占用体检见 `scripts/doctor.py`（一条命令体检 Python 版本/依赖/Key/目录/端口）。

## 部署完成后

把 `https://<app-name>.streamlit.app` 加到 README 顶部与 portfolio 页，作为在线 demo 入口。
