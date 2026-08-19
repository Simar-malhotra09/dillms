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
    thinking_disabled=False,
    required_temperature=None,
)

# Kimi's hidden reasoning has been observed burning its entire completion-token
# budget before emitting visible JSON once discovery gives it real strategic
# content to reason about. Bumped budget so reasoning can finish, and
# include_reasoning=False drops the visible "reasoning" field from its schema
# to reduce output-token pressure. Flip include_reasoning back to True to
# restore Kimi's reasoning trace in the logs.
#
# Note: reasoning_effort is documented as a Kimi K3-only parameter. For K2.x
# models (this one), it's a no-op -- Kimi's actual reasoning depth here is
# provider-controlled, not "low" as the field name implies. Live-tested
# 2026-08-14: same prompt produced 1241 hidden reasoning tokens with
# reasoning_effort="low" set. Use KIMI_INSTANT below for a real cap.
KIMI = ModelSpec(
    model_id="kimi-k2.6",
    self_label="Kimi K2.6",
    api_key_env="KIMI_API_KEY",
    base_url="https://api.moonshot.ai/v1",
    reasoning_effort="low",
    max_completion_tokens=8000,
    include_reasoning=False,
    thinking_disabled=False,
    required_temperature=None,
)

# Kimi K2.x's real reasoning switch is thinking.type, not reasoning_effort.
# Disabling it forces Instant mode, which requires temperature=0.6 (Thinking
# mode requires 1.0; the API 400s if you mix them). Live-tested 2026-08-14:
# same prompt that produced 1241 reasoning tokens under KIMI dropped to 1
# reasoning token / 84 completion tokens total under this spec -- this is a
# different inference mode (shallow, no chain-of-thought), not a capped
# version of KIMI's reasoning, so it isn't reasoning-comparable to KIMI or
# to DEEPSEEK (which still reasons at low effort). Use when cost/reliability
# matters more than measuring deliberation depth, e.g. cheap self-play runs.
KIMI_INSTANT = ModelSpec(
    model_id="kimi-k2.6",
    self_label="Kimi K2.6 (instant)",
    api_key_env="KIMI_API_KEY",
    base_url="https://api.moonshot.ai/v1",
    reasoning_effort="low",
    max_completion_tokens=8000,
    include_reasoning=True,
    thinking_disabled=True,
    required_temperature=0.6,
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
) -> tuple[str, int, int, int, float]:
    """Returns (raw_text, prompt_tokens, completion_tokens, reasoning_tokens, latency_ms).

    reasoning_tokens is read from usage.completion_tokens_details.reasoning_tokens
    when the provider exposes it (OpenAI-compatible o1-style field); 0 if absent.
    It is a subset of completion_tokens, not additional to it.
    """
    client = _client_for(model_spec.base_url, model_spec.api_key_env)
    effective_temperature = (
        model_spec.required_temperature if model_spec.required_temperature is not None else temperature
    )
    extra_body = (
        {"thinking": {"type": "disabled"}}
        if model_spec.thinking_disabled
        else {"reasoning_effort": model_spec.reasoning_effort}
    )
    start = time.monotonic()
    response = client.chat.completions.create(
        model=model_spec.model_id,
        messages=messages,
        temperature=effective_temperature,
        max_tokens=model_spec.max_completion_tokens,
        extra_body=extra_body,
    )
    latency_ms = (time.monotonic() - start) * 1000
    raw_text = response.choices[0].message.content or ""
    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    completion_details = getattr(usage, "completion_tokens_details", None) if usage else None
    reasoning_tokens = getattr(completion_details, "reasoning_tokens", None) if completion_details else None
    return raw_text, prompt_tokens, completion_tokens, reasoning_tokens or 0, latency_ms


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
) -> tuple[dict, str, int, int, int, float]:
    """Calls the model and parses a JSON object reply, validating it with
    `validate` (raises ValueError on an invalid parse). Retries with a
    corrective message up to MAX_JSON_ATTEMPTS times; raises on final
    failure (no silent fallback).

    Returns (parsed, raw_text, prompt_tokens, completion_tokens,
    reasoning_tokens, latency_ms), summed across all retry attempts.
    """
    working_messages = list(messages)
    last_error: Exception | None = None
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_reasoning_tokens = 0
    total_latency_ms = 0.0

    for _ in range(MAX_JSON_ATTEMPTS):
        raw_text, prompt_tokens, completion_tokens, reasoning_tokens, latency_ms = call_model(
            model_spec, working_messages, temperature
        )
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        total_reasoning_tokens += reasoning_tokens
        total_latency_ms += latency_ms
        try:
            parsed = json.loads(_extract_json_object(raw_text))
            validate(parsed)
            return (
                parsed,
                raw_text,
                total_prompt_tokens,
                total_completion_tokens,
                total_reasoning_tokens,
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
) -> tuple[RoundDecision, str, int, int, int, float]:
    if model_spec.include_reasoning:

        def validate(parsed: dict) -> None:
            decision = RoundDecision.model_validate(parsed)
            if decision.action not in valid_actions:
                raise ValueError(f"action {decision.action!r} not in {valid_actions}")

        parsed, raw_text, prompt_tokens, completion_tokens, reasoning_tokens, latency_ms = get_json_reply(
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

        parsed, raw_text, prompt_tokens, completion_tokens, reasoning_tokens, latency_ms = get_json_reply(
            model_spec,
            messages,
            temperature,
            validate,
            f'{{"action": "<one of {list(valid_actions)}>"}}',
        )
        decision = RoundDecision(reasoning="", action=ActionOnlyDecision.model_validate(parsed).action)

    return decision, raw_text, prompt_tokens, completion_tokens, reasoning_tokens, latency_ms
