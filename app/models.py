from __future__ import annotations

from pydantic import BaseModel, Field


class IntentResult(BaseModel):
    intent: str = Field(default="unclear")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class RouteRequest(BaseModel):
    message: str = Field(min_length=1)


class RouteResponse(BaseModel):
    user_message: str
    normalized_message: str
    intent: str
    confidence: float
    routed_intent: str
    final_response: str
    manual_override: bool
    timestamp: str
