"""AI Data Research Lab — FastAPI 入口。"""
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.agent import router as agent_router
from routers.ml import router as ml_router
from routers.profile import router as profile_router
from routers.reports import router as reports_router
from routers.research import router as research_router

app = FastAPI(title="AI Data Research Lab", version="0.10.0")

# CORS 默认只放行本地 Streamlit，可用 RESEARCH_LAB_CORS 覆盖（逗号分隔）
_CORS_ORIGINS = [
    o.strip()
    for o in os.getenv("RESEARCH_LAB_CORS", "http://localhost:8501,http://127.0.0.1:8501").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(profile_router)
app.include_router(research_router)
app.include_router(ml_router)
app.include_router(agent_router)
app.include_router(reports_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "ai-data-research-lab", "version": "0.10.0"}



