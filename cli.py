from __future__ import annotations

import argparse
import json

from app.config import Settings
from app.llm import OpenAIResponsesClient
from app.router import PromptRouter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the prompt router from the command line.")
    parser.add_argument("message", nargs="?", help="Message to send through the router.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full routed response payload as JSON.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    settings = Settings.from_env()
    router = PromptRouter(
        llm_client=OpenAIResponsesClient(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        ),
        settings=settings,
    )

    if args.message:
        result = router.process_message(args.message)
        if args.json:
            print(json.dumps(result.model_dump(), indent=2))
        else:
            print(f"Intent: {result.intent} ({result.confidence:.2f})")
            print(f"Routed As: {result.routed_intent}")
            print(result.final_response)
        return

    while True:
        try:
            message = input("message> ").strip()
        except EOFError:
            print()
            break

        if not message or message.lower() in {"exit", "quit"}:
            break

        result = router.process_message(message)
        print(f"[{result.intent} -> {result.routed_intent}] {result.final_response}")


if __name__ == "__main__":
    main()
