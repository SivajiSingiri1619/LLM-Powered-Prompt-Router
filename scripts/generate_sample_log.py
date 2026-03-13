from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import Settings
from app.prompts import CLASSIFIER_PROMPT, EXPERT_PROMPTS
from app.router import PromptRouter

SAMPLE_MESSAGES = {
    "how do i sort a list of objects in python?": {"intent": "code", "confidence": 0.94},
    "explain this sql query for me": {"intent": "code", "confidence": 0.90},
    "This paragraph sounds awkward, can you help me fix it?": {
        "intent": "writing",
        "confidence": 0.92,
    },
    "I'm preparing for a job interview, any tips?": {"intent": "career", "confidence": 0.95},
    "what's the average of these numbers: 12, 45, 23, 67, 34": {
        "intent": "data",
        "confidence": 0.96,
    },
    "Help me make this better.": {"intent": "writing", "confidence": 0.44},
    "I need to write a function that takes a user id and returns their profile, but also i need help with my resume.": {
        "intent": "unclear",
        "confidence": 0.48,
    },
    "hey": {"intent": "unclear", "confidence": 0.11},
    "Can you write me a poem about clouds?": {"intent": "unclear", "confidence": 0.98},
    "Rewrite this sentence to be more professional.": {
        "intent": "writing",
        "confidence": 0.83,
    },
    "I'm not sure what to do with my career.": {"intent": "career", "confidence": 0.90},
    "what is a pivot table": {"intent": "data", "confidence": 0.85},
    "fxi thsi bug pls: for i in range(10) print(i)": {"intent": "code", "confidence": 0.88},
    "How do I structure a cover letter?": {"intent": "career", "confidence": 0.79},
    "My boss says my writing is too verbose.": {"intent": "writing", "confidence": 0.93},
}


class DemoLLMClient:
    def complete(
        self,
        *,
        model: str,
        instructions: str,
        user_message: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        if instructions == CLASSIFIER_PROMPT:
            return json.dumps(SAMPLE_MESSAGES[user_message])

        for intent, prompt in EXPERT_PROMPTS.items():
            if instructions == prompt:
                return f"{intent.title()} specialist response for: {user_message}"

        return "I want to route this correctly. Are you asking for help with coding, data analysis, writing feedback, or career advice?"


def main() -> None:
    log_path = REPO_ROOT / "route_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    settings = Settings(
        openai_api_key="demo-key",
        openai_base_url=None,
        classifier_model="demo-classifier",
        generation_model="demo-generator",
        confidence_threshold=0.70,
        route_log_path=log_path,
        app_host="0.0.0.0",
        app_port=8000,
    )
    router = PromptRouter(llm_client=DemoLLMClient(), settings=settings)

    for message in SAMPLE_MESSAGES:
        router.process_message(message)


if __name__ == "__main__":
    main()
