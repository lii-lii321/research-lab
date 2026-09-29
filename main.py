# -*- coding: utf-8 -*-
"""AI Data Research Lab — FastAPI 入口。"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.agent import router as agent_router
from routers.ml import router as ml_router
from routers.profile import router as profile_router
from routers.research import router as research_router

app = FastAPI(title="AI Data Research Lab", version="0.5.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(profile_router)
app.include_router(research_router)
app.include_router(ml_router)
app.include_router(agent_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "ai-data-research-lab", "version": "0.5.0"}
