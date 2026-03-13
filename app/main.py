from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings
from app.llm import OpenAIResponsesClient
from app.models import RouteRequest, RouteResponse
from app.router import PromptRouter

app = FastAPI(
    title="LLM-Powered Prompt Router",
    version="1.0.0",
    description="Intent-based routing service for specialized AI personas.",
)

STATIC_DIR = Path(__file__).resolve().parents[1] / "static"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()


@lru_cache(maxsize=1)
def get_router() -> PromptRouter:
    settings = get_settings()
    llm_client = OpenAIResponsesClient(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
    return PromptRouter(llm_client=llm_client, settings=settings)


@app.get("/", include_in_schema=False)
def chat_ui() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/route", response_model=RouteResponse)
def route_message(payload: RouteRequest) -> RouteResponse:
    try:
        return get_router().process_message(payload.message)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
