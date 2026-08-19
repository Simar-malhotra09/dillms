"""Reusable turn-based conversation runner against DeepSeek's chat API.

Each turn sends the full prior history (past user prompts + past assistant
*answers*) but never past reasoning_content -- DeepSeek's API rejects/ignores
reasoning fed back in, and it isn't meant to be replayed as context.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)  # ~/.zshrc exports a stale DEEPSEEK_API_KEY that shadows .env

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def _client() -> OpenAI:
    api_key = os.environ["DEEPSEEK_API_KEY"]
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)


def _turn_content(prompt: str, max_reasoning_words: int | None) -> str:
    if max_reasoning_words is None:
        return prompt
    return (
        f"{prompt}\n\n"
        f"(Keep your internal reasoning to roughly {max_reasoning_words} words "
        "or fewer before answering. This is a soft target, not a hard rule -- "
        "prioritize honesty over brevity.)"
    )


def run_conversation(
    turns: list[str],
    model: str,
    max_reasoning_words: int | None,
) -> list[dict]:
    """Runs `turns` sequentially against `model`.

    Returns a list of per-turn records: turn, prompt (as sent, including any
    word-cap suffix), reasoning, answer, and token usage.
    """
    client = _client()
    history: list[dict] = []
    records: list[dict] = []

    for turn_number, prompt in enumerate(turns, start=1):
        user_content = _turn_content(prompt, max_reasoning_words)
        history.append({"role": "user", "content": user_content})

        response = client.chat.completions.create(
            model=model,
            messages=history,
        )

        message = response.choices[0].message
        reasoning = getattr(message, "reasoning_content", None) or ""
        answer = message.content or ""
        usage = response.usage
        completion_details = getattr(usage, "completion_tokens_details", None) if usage else None
        reasoning_tokens = getattr(completion_details, "reasoning_tokens", None) if completion_details else None

        records.append(
            {
                "turn": turn_number,
                "prompt": prompt,
                "prompt_sent": user_content,
                "reasoning": reasoning,
                "answer": answer,
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "reasoning_tokens": reasoning_tokens or 0,
            }
        )

        # Only the visible answer goes back into history, not reasoning_content.
        history.append({"role": "assistant", "content": answer})

    return records


def write_markdown(
    records: list[dict],
    output_path: str,
    model: str,
    max_reasoning_words: int | None,
) -> None:
    lines = [
        "# Triage dilemma -- multi-turn run",
        "",
        f"- Model: `{model}`",
        f"- Reasoning word cap: {max_reasoning_words if max_reasoning_words else 'none'}",
        f"- Run at (UTC): {datetime.now(timezone.utc).isoformat()}",
        "",
    ]
    for record in records:
        lines.append(f"## Turn {record['turn']}")
        lines.append("")
        lines.append("**Prompt:**")
        lines.append("")
        lines.append(record["prompt"])
        lines.append("")
        if record["reasoning"]:
            lines.append("**Reasoning:**")
            lines.append("")
            lines.append(record["reasoning"])
            lines.append("")
        lines.append("**Answer:**")
        lines.append("")
        lines.append(record["answer"])
        lines.append("")
        lines.append(
            f"_tokens: prompt={record['prompt_tokens']}, "
            f"completion={record['completion_tokens']}, "
            f"reasoning={record['reasoning_tokens']}_"
        )
        lines.append("")

    with open(output_path, "w") as file:
        file.write("\n".join(lines))
