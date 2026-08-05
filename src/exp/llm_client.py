"""Thin connector over the two OpenAI-compatible chat APIs used in the pilot."""

from __future__ import annotations

import json
import os
import time
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI

from exp.types import ActionOnlyDecision, ModelSpec, RoundDecision

load_dotenv(override=True)

DEEPSEEK = ModelSpec(
    model_id="deepseek-v4-flash",
    self_label="DeepSeek",
    api_key_env="DEEPSEEK_API_KEY",
    base_url="https://api.deepseek.com",
    reasoning_effort="low",
    max_completion_tokens=4000,
    include_reasoning=True,
)

# Kimi's hidden reasoning has been observed burning its entire completion-token
# budget before emitting visible JSON once discovery gives it real strategic
# content to reason about. Bumped budget so reasoning can finish, and
# include_reasoning=False drops the visible "reasoning" field from its schema
# to reduce output-token pressure. Flip include_reasoning back to True to
# restore Kimi's reasoning trace in the logs.
KIMI = ModelSpec(
    model_id="kimi-k2.6",
    self_label="Kimi K2.6",
    api_key_env="KIMI_API_KEY",
    base_url="https://api.moonshot.ai/v1",
    reasoning_effort="low",
    max_completion_tokens=8000,
    include_reasoning=False,
)

MAX_JSON_ATTEMPTS = 3


@lru_cache(maxsize=None)
def _client_for(base_url: str, api_key_env: str) -> OpenAI:
    api_key = os.environ[api_key_env]
    return OpenAI(api_key=api_key, base_url=base_url)


def call_model(
    model_spec: ModelSpec,
    messages: list[dict[str, str]],
    temperature: float,
) -> tuple[str, int, int, float]:
    """Returns (raw_text, prompt_tokens, completion_tokens, latency_ms)."""
    client = _client_for(model_spec.base_url, model_spec.api_key_env)
    start = time.monotonic()
    response = client.chat.completions.create(
        model=model_spec.model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=model_spec.max_completion_tokens,
        extra_body={"reasoning_effort": model_spec.reasoning_effort},
    )
    latency_ms = (time.monotonic() - start) * 1000
    raw_text = response.choices[0].message.content or ""
    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    return raw_text, prompt_tokens, completion_tokens, latency_ms


def _extract_json_object(raw_text: str) -> str:
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in model output: {raw_text!r}")
    return raw_text[start : end + 1]


def get_json_reply(
    model_spec: ModelSpec,
    messages: list[dict[str, str]],
    temperature: float,
    validate: "callable[[dict], None]",
    schema_hint: str,
) -> tuple[dict, str, int, int, float]:
    """Calls the model and parses a JSON object reply, validating it with
    `validate` (raises ValueError on an invalid parse). Retries with a
    corrective message up to MAX_JSON_ATTEMPTS times; raises on final
    failure (no silent fallback).
    """
    working_messages = list(messages)
    last_error: Exception | None = None
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_latency_ms = 0.0

    for _ in range(MAX_JSON_ATTEMPTS):
        raw_text, prompt_tokens, completion_tokens, latency_ms = call_model(
            model_spec, working_messages, temperature
        )
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        total_latency_ms += latency_ms
        try:
            parsed = json.loads(_extract_json_object(raw_text))
            validate(parsed)
            return (
                parsed,
                raw_text,
                total_prompt_tokens,
                total_completion_tokens,
                total_latency_ms,
            )
        except (json.JSONDecodeError, ValueError) as error:
            last_error = error
            retry_notice = {
                "role": "user",
                "content": (
                    f"Your previous reply was not valid JSON of the form {schema_hint}. "
                    "Reply again with only that JSON object."
                ),
            }
            # Some providers reject an empty assistant message in the history,
            # which can happen if the completion-token budget was exhausted
            # by hidden reasoning before any visible content was emitted.
            if raw_text.strip():
                working_messages = working_messages + [
                    {"role": "assistant", "content": raw_text},
                    retry_notice,
                ]
            else:
                working_messages = working_messages + [retry_notice]

    raise RuntimeError(
        f"Model {model_spec.model_id} failed to produce a valid reply after "
        f"{MAX_JSON_ATTEMPTS} attempts: {last_error}"
    )


def get_round_decision(
    model_spec: ModelSpec,
    messages: list[dict[str, str]],
    temperature: float,
    valid_actions: tuple[str, ...],
) -> tuple[RoundDecision, str, int, int, float]:
    if model_spec.include_reasoning:

        def validate(parsed: dict) -> None:
            decision = RoundDecision.model_validate(parsed)
            if decision.action not in valid_actions:
                raise ValueError(f"action {decision.action!r} not in {valid_actions}")

        parsed, raw_text, prompt_tokens, completion_tokens, latency_ms = get_json_reply(
            model_spec,
            messages,
            temperature,
            validate,
            f'{{"reasoning": "...", "action": "<one of {list(valid_actions)}>"}}',
        )
        decision = RoundDecision.model_validate(parsed)
    else:

        def validate(parsed: dict) -> None:
            decision = ActionOnlyDecision.model_validate(parsed)
            if decision.action not in valid_actions:
                raise ValueError(f"action {decision.action!r} not in {valid_actions}")

        parsed, raw_text, prompt_tokens, completion_tokens, latency_ms = get_json_reply(
            model_spec,
            messages,
            temperature,
            validate,
            f'{{"action": "<one of {list(valid_actions)}>"}}',
        )
        decision = RoundDecision(reasoning="", action=ActionOnlyDecision.model_validate(parsed).action)

    return decision, raw_text, prompt_tokens, completion_tokens, latency_ms
