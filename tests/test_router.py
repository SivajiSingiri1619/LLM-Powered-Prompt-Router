from __future__ import annotations

import json
from pathlib import Path

from app.config import Settings
from app.prompts import CLASSIFIER_PROMPT, EXPERT_PROMPTS
from app.router import PromptRouter, safe_parse_intent_response


class FakeLLMClient:
    def __init__(self, classifier_outputs: dict[str, str] | None = None) -> None:
        self.classifier_outputs = classifier_outputs or {}
        self.calls: list[dict[str, object]] = []

    def complete(
        self,
        *,
        model: str,
        instructions: str,
        user_message: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        self.calls.append(
            {
                "model": model,
                "instructions": instructions,
                "user_message": user_message,
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            }
        )
        if instructions == CLASSIFIER_PROMPT:
            return self.classifier_outputs.get(
                user_message,
                json.dumps({"intent": "unclear", "confidence": 0.0}),
            )

        for intent, prompt in EXPERT_PROMPTS.items():
            if instructions == prompt:
                return f"{intent} expert response"

        return "unexpected response"


def build_router(
    tmp_path: Path,
    classifier_outputs: dict[str, str] | None = None,
    threshold: float = 0.70,
) -> tuple[PromptRouter, FakeLLMClient]:
    settings = Settings(
        openai_api_key="test-key",
        openai_base_url=None,
        classifier_model="classifier-test-model",
        generation_model="generator-test-model",
        confidence_threshold=threshold,
        route_log_path=tmp_path / "route_log.jsonl",
        app_host="127.0.0.1",
        app_port=8000,
    )
    fake_client = FakeLLMClient(classifier_outputs=classifier_outputs)
    return PromptRouter(llm_client=fake_client, settings=settings), fake_client


def test_safe_parse_intent_response_accepts_valid_json() -> None:
    result = safe_parse_intent_response('{"intent":"code","confidence":0.91}')
    assert result.intent == "code"
    assert result.confidence == 0.91


def test_safe_parse_intent_response_handles_code_fences() -> None:
    result = safe_parse_intent_response(
        '```json\n{"intent":"writing","confidence":"0.73"}\n```'
    )
    assert result.intent == "writing"
    assert result.confidence == 0.73


def test_safe_parse_intent_response_defaults_when_json_is_invalid() -> None:
    result = safe_parse_intent_response("definitely not json")
    assert result.intent == "unclear"
    assert result.confidence == 0.0


def test_route_and_respond_asks_for_clarification_for_unclear_intent(tmp_path: Path) -> None:
    router, _ = build_router(tmp_path)
    response = router.route_and_respond("hey", safe_parse_intent_response("{}"))
    assert response.endswith("?")
    assert "coding" in response


def test_process_message_logs_a_complete_json_line(tmp_path: Path) -> None:
    router, _ = build_router(
        tmp_path,
        classifier_outputs={
            "how do i sort a list of objects in python?": (
                '{"intent":"code","confidence":0.93}'
            )
        },
    )

    result = router.process_message("how do i sort a list of objects in python?")

    assert result.intent == "code"
    log_file = tmp_path / "route_log.jsonl"
    entries = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(entries) == 1
    payload = json.loads(entries[0])
    assert payload["intent"] == "code"
    assert payload["confidence"] == 0.93
    assert payload["user_message"] == "how do i sort a list of objects in python?"
    assert payload["final_response"] == "code expert response"


def test_manual_override_skips_classifier_and_routes_directly(tmp_path: Path) -> None:
    router, fake_client = build_router(tmp_path)
    result = router.process_message("@writing My sentence feels clunky.")

    assert result.intent == "writing"
    assert result.confidence == 1.0
    assert result.routed_intent == "writing"
    assert result.final_response == "writing expert response"
    assert all(call["instructions"] != CLASSIFIER_PROMPT for call in fake_client.calls)


def test_confidence_threshold_forces_unclear_routing(tmp_path: Path) -> None:
    router, _ = build_router(
        tmp_path,
        classifier_outputs={"Help me make this better.": '{"intent":"writing","confidence":0.41}'},
        threshold=0.70,
    )
    result = router.process_message("Help me make this better.")
    assert result.intent == "writing"
    assert result.routed_intent == "unclear"
    assert result.final_response.endswith("?")


def test_submission_sample_messages_route_cleanly(tmp_path: Path) -> None:
    sample_messages = {
        "how do i sort a list of objects in python?": '{"intent":"code","confidence":0.94}',
        "explain this sql query for me": '{"intent":"code","confidence":0.90}',
        "This paragraph sounds awkward, can you help me fix it?": '{"intent":"writing","confidence":0.92}',
        "I'm preparing for a job interview, any tips?": '{"intent":"career","confidence":0.95}',
        "what's the average of these numbers: 12, 45, 23, 67, 34": '{"intent":"data","confidence":0.96}',
        "Help me make this better.": '{"intent":"writing","confidence":0.44}',
        "I need to write a function that takes a user id and returns their profile, but also i need help with my resume.": '{"intent":"unclear","confidence":0.48}',
        "hey": '{"intent":"unclear","confidence":0.11}',
        "Can you write me a poem about clouds?": '{"intent":"unclear","confidence":0.98}',
        "Rewrite this sentence to be more professional.": '{"intent":"writing","confidence":0.83}',
        "I'm not sure what to do with my career.": '{"intent":"career","confidence":0.90}',
        "what is a pivot table": '{"intent":"data","confidence":0.85}',
        "fxi thsi bug pls: for i in range(10) print(i)": '{"intent":"code","confidence":0.88}',
        "How do I structure a cover letter?": '{"intent":"career","confidence":0.79}',
        "My boss says my writing is too verbose.": '{"intent":"writing","confidence":0.93}',
    }
    router, _ = build_router(tmp_path, classifier_outputs=sample_messages)

    unclear_messages = {
        "Help me make this better.",
        "I need to write a function that takes a user id and returns their profile, but also i need help with my resume.",
        "hey",
        "Can you write me a poem about clouds?",
    }

    for message in sample_messages:
        result = router.process_message(message)
        if message in unclear_messages:
            assert result.routed_intent == "unclear"
            assert result.final_response.endswith("?")
        else:
            assert result.routed_intent in {"code", "data", "writing", "career"}
            assert result.final_response.endswith("expert response")
