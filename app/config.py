from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

if Path(".env").exists():
    load_dotenv(".env")
elif Path(".env.example").exists():
    load_dotenv(".env.example")


def _parse_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _parse_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(slots=True)
class Settings:
    openai_api_key: str | None
    openai_base_url: str | None
    classifier_model: str
    generation_model: str
    confidence_threshold: float
    route_log_path: Path
    app_host: str
    app_port: int

    @classmethod
    def from_env(cls) -> "Settings":
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
        return cls(
            openai_api_key=api_key,
            openai_base_url=os.getenv("OPENAI_BASE_URL"),
            classifier_model=os.getenv("CLASSIFIER_MODEL", "gpt-4.1-nano"),
            generation_model=os.getenv("GENERATION_MODEL", "gpt-4.1-mini"),
            confidence_threshold=_parse_float(
                os.getenv("CONFIDENCE_THRESHOLD"), 0.70
            ),
            route_log_path=Path(os.getenv("ROUTE_LOG_PATH", "route_log.jsonl")),
            app_host=os.getenv("APP_HOST", "0.0.0.0"),
            app_port=_parse_int(os.getenv("APP_PORT"), 8000),
        )
