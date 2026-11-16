# 电商 AI 智能客服

基于 FastAPI、LangGraph 和 LangChain 的电商客服个人项目。Supervisor 将用户咨询分发给 RAG、订单、投诉和闲聊专家流程。

本项目在 [智能客服多 Agent 系统](https://github.com/bcefghj/smart-cs-multi-agent) 的 Python 实现基础上进行电商场景二次开发，保留原项目 MIT 许可证。

## 当前功能

- 通过 `/api/chat` 接收咨询，支持会话 ID 续聊。
- Supervisor 使用规则和模型判断意图，并路由到四类专家流程。
- RAG 基于本地 Markdown 知识文档和 BM25 检索，回答附带来源引用；资料不足时转人工兜底。
- 订单和投诉使用 SQLite 演示数据，投诉提交前需要用户确认。
- Redis 保存短期聊天记录，并通过 LangGraph Redis checkpointer 持久化会话状态。
- Docker Compose 启动 FastAPI 和 Redis Stack。

## 技术栈

Python、FastAPI、LangGraph、LangChain、DeepSeek API、Redis Stack、SQLite、BM25。

## 本地运行

在 `python-impl/.env` 中设置模型配置和 Redis 地址：

```env
OPENAI_API_KEY=你的DeepSeek_API_Key
OPENAI_BASE_URL=https://api.deepseek.com/v1
MODEL_NAME=deepseek-chat
REDIS_URL=redis://localhost:6379/0
```

安装 `python-impl/requirements.txt` 中的依赖，并确保 Redis Stack 正在运行。进入 `python-impl` 目录后启动：

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

接口文档：<http://127.0.0.1:8000/docs>

## Docker 部署

将模型配置写入 `python-impl/.env`，然后在仓库根目录运行：

```bash
docker compose up -d --build
```

API 默认监听 8000 端口。Redis 仅在 Compose 内部网络使用，不对宿主机发布端口。

## 项目结构

```text
python-impl/
  agents/       Supervisor 与四类专家流程
  api/          FastAPI 接口
  core/         配置、模型和状态定义
  knowledge/    演示知识文档与检索评测问题
  memory/       Redis 短期聊天记录
  retrieval/    文档切块、BM25 检索与 RRF 融合
  storage/      SQLite 订单、投诉数据
docker-compose.yml
```

## 当前范围

当前检索主流程使用 BM25。Milvus、BGE-M3、BGE Reranker、长期用户记忆和知识审核回灌尚未接入；项目中的知识文档和订单数据用于本地演示。
