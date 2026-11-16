from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any
from core.config import settings
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from swagger_ui_bundle import swagger_ui_path
from agents.supervisor import create_supervisor_graph
from core.state import AgentState
from memory.short_term import ShortTermMemory
from storage.database import initialize_database

load_dotenv()
logger = logging.getLogger(__name__)

short_term_memory = ShortTermMemory(
    redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0")
)
graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph

    initialize_database()

    async with AsyncRedisSaver.from_conn_string(settings.redis_url) as checkpointer:
        graph = create_supervisor_graph(checkpointer=checkpointer)
        yield

app = FastAPI(
    title="电商 AI 智能客服",
    description="基于 LangGraph Supervisor 和领域专家子图的电商客服系统",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,
)
app.mount(
    "/static/swagger",
    StaticFiles(directory=swagger_ui_path),
    name="swagger-static",
)


@app.get("/docs", include_in_schema=False)
async def swagger_docs():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - API 文档",
        swagger_js_url="/static/swagger/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger/swagger-ui.css",
        swagger_favicon_url="/static/swagger/favicon-32x32.png",
    )

class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    user_id: str = "anonymous"
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    intent: str
    status: str
    citations: list[dict[str, Any]] = Field(default_factory=list)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if graph is None:
        raise HTTPException(status_code=503, detail="系统初始化中")

    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="message 不能为空")

    session_id = request.session_id or str(uuid.uuid4())

    from langchain_core.messages import HumanMessage

    initial_state: AgentState = {
        "messages": [HumanMessage(content=message)],
        "user_id": request.user_id,
        "session_id": session_id,
    }
    config = {"configurable": {"thread_id": session_id}}

    try:
        result = await graph.ainvoke(initial_state, config=config)
    except Exception as error:
        logger.exception("处理聊天请求失败")
        raise HTTPException(
            status_code=500,
            detail=f"处理失败: {error}",
        ) from error

    response = result.get("final_response", "暂时无法处理，请稍后再试。")
    await short_term_memory.add_message(session_id, "user", message)
    await short_term_memory.add_message(session_id, "assistant", response)

    agent_result = result.get("agent_result") or {}

    return ChatResponse(
        response=response,
        session_id=session_id,
        intent=result.get("intent", "unknown"),
        status=result.get("task_status", "failed"),
        citations=agent_result.get("citations", []),
    )


@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    history = await short_term_memory.get_history(session_id)
    return {"session_id": session_id, "messages": history}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )