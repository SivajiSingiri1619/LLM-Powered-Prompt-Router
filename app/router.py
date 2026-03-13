from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

from app.config import Settings
from app.llm import LLMClient
from app.models import IntentResult, RouteResponse
from app.prompts import CLARIFICATION_QUESTION, CLASSIFIER_PROMPT, EXPERT_PROMPTS

INTENT_OVERRIDE_PATTERN = re.compile(
    r"^\s*@(?P<intent>code|data|writing|career|unclear)\b[\s:,-]*(?P<message>.*)$",
    re.IGNORECASE | re.DOTALL,
)


def clamp_confidence(value: object) -> float:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, numeric_value))


def default_intent_result() -> IntentResult:
    return IntentResult(intent="unclear", confidence=0.0)


def extract_json_object(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.DOTALL).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    return match.group(0) if match else cleaned


def safe_parse_intent_response(raw_response: str) -> IntentResult:
    try:
        parsed = json.loads(extract_json_object(raw_response))
    except (json.JSONDecodeError, TypeError):
        return default_intent_result()

    intent = str(parsed.get("intent", "unclear")).strip().lower()
    confidence = clamp_confidence(parsed.get("confidence", 0.0))

    if intent not in EXPERT_PROMPTS and intent != "unclear":
        return default_intent_result()

    return IntentResult(intent=intent, confidence=confidence)


class JsonLineLogger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def append(self, payload: dict[str, object]) -> None:
        with self._lock:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


class PromptRouter:
    def __init__(
        self,
        *,
        llm_client: LLMClient,
        settings: Settings | None = None,
        prompts: dict[str, str] | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.llm_client = llm_client
        self.prompts = prompts or EXPERT_PROMPTS
        self.logger = JsonLineLogger(self.settings.route_log_path)

    def classify_intent(self, message: str) -> IntentResult:
        raw_response = self.llm_client.complete(
            model=self.settings.classifier_model,
            instructions=CLASSIFIER_PROMPT,
            user_message=message,
            temperature=0.0,
            max_output_tokens=100,
        )
        return safe_parse_intent_response(raw_response)

    def route_and_respond(self, message: str, intent: IntentResult) -> str:
        effective_intent = self._effective_intent(intent)
        if effective_intent == "unclear":
            return self._clarification_question()

        prompt = self.prompts[effective_intent]
        return self.llm_client.complete(
            model=self.settings.generation_model,
            instructions=prompt,
            user_message=message,
            temperature=0.3,
            max_output_tokens=700,
        )

    def process_message(self, message: str) -> RouteResponse:
        normalized_message = message.strip()
        override_intent, override_message = self._parse_manual_override(normalized_message)

        if override_intent is not None:
            classified_intent = IntentResult(intent=override_intent, confidence=1.0)
            normalized_message = override_message or normalized_message
            manual_override = True
        else:
            classified_intent = self.classify_intent(normalized_message)
            manual_override = False

        final_response = self.route_and_respond(normalized_message, classified_intent)
        routed_intent = self._effective_intent(classified_intent)
        timestamp = datetime.now(UTC).isoformat()

        outcome = RouteResponse(
            user_message=message,
            normalized_message=normalized_message,
            intent=classified_intent.intent,
            confidence=classified_intent.confidence,
            routed_intent=routed_intent,
            final_response=final_response,
            manual_override=manual_override,
            timestamp=timestamp,
        )
        self.logger.append(outcome.model_dump())
        return outcome

    def _effective_intent(self, intent: IntentResult) -> str:
        if intent.intent == "unclear":
            return "unclear"
        if intent.confidence < self.settings.confidence_threshold:
            return "unclear"
        return intent.intent

    def _clarification_question(self) -> str:
        return CLARIFICATION_QUESTION

    def _parse_manual_override(self, message: str) -> tuple[str | None, str]:
        match = INTENT_OVERRIDE_PATTERN.match(message)
        if not match:
            return None, message
        intent = match.group("intent").lower()
        override_message = match.group("message").strip()
        return intent, override_message
