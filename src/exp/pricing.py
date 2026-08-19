"""Known per-model API pricing, USD per million tokens.

DeepSeek and Kimi entries below are sourced from web search on 2026-08-14
against third-party pricing aggregators, not the providers' own pricing
pages directly -- re-verify against DeepSeek's and Moonshot's official
pricing pages before using this table for a real budget decision.

Claude entries are Anthropic first-party API rates, verified 2026-08-17
against the claude-api skill's cached pricing table. GPT/Gemini entries are
user-supplied and not independently verified against the providers' own
pricing pages -- re-verify before using this table for a real budget
decision.

Pricing for any other model is intentionally not guessed here: supply it
via a JSON override file passed to load_pricing_overrides, keyed by
model_id, e.g.:

    {"gpt-5": {"input_price_per_million_usd": 1.25, "output_price_per_million_usd": 10.0}}
"""

from __future__ import annotations

import json
from pathlib import Path

from exp.types import ModelPricing

KNOWN_PRICING: dict[str, ModelPricing] = {
    "deepseek-v4-flash": ModelPricing(
        model_id="deepseek-v4-flash",
        input_price_per_million_usd=0.14,
        output_price_per_million_usd=0.28,
    ),
    "kimi-k2.6": ModelPricing(
        model_id="kimi-k2.6",
        input_price_per_million_usd=0.95,
        output_price_per_million_usd=4.00,
    ),
    # Anthropic first-party API rates, verified 2026-08-17.
    "claude-fable-5": ModelPricing(
        model_id="claude-fable-5",
        input_price_per_million_usd=10.00,
        output_price_per_million_usd=50.00,
    ),
    "claude-opus-5": ModelPricing(
        model_id="claude-opus-5",
        input_price_per_million_usd=5.00,
        output_price_per_million_usd=25.00,
    ),
    "claude-opus-4-7": ModelPricing(
        model_id="claude-opus-4-7",
        input_price_per_million_usd=5.00,
        output_price_per_million_usd=25.00,
    ),
    # Sonnet 5 is on introductory pricing ($2/$10) through 2026-08-31;
    # standard pricing ($3/$15) applies after that.
    "claude-sonnet-5": ModelPricing(
        model_id="claude-sonnet-5",
        input_price_per_million_usd=2.00,
        output_price_per_million_usd=10.00,
    ),
    # Not Anthropic models -- pricing supplied by the user, not independently
    # verified against the providers' own pricing pages.
    "gpt-5.6-sol": ModelPricing(
        model_id="gpt-5.6-sol",
        input_price_per_million_usd=5.00,
        output_price_per_million_usd=30.00,
    ),
    "gpt-5.6-terra": ModelPricing(
        model_id="gpt-5.6-terra",
        input_price_per_million_usd=2.00,
        output_price_per_million_usd=12.00,
    ),
    "gpt-5.6-luna": ModelPricing(
        model_id="gpt-5.6-luna",
        input_price_per_million_usd=0.20,
        output_price_per_million_usd=1.20,
    ),
    "gpt-5": ModelPricing(
        model_id="gpt-5",
        input_price_per_million_usd=1.25,
        output_price_per_million_usd=10.00,
    ),
    "gemini-2.5-pro": ModelPricing(
        model_id="gemini-2.5-pro",
        input_price_per_million_usd=1.25,
        output_price_per_million_usd=10.00,
    ),
}


def load_pricing_overrides(path: Path) -> dict[str, ModelPricing]:
    raw_entries: dict[str, dict[str, float]] = json.loads(path.read_text())
    overrides: dict[str, ModelPricing] = {
        model_id: ModelPricing(
            model_id=model_id,
            input_price_per_million_usd=entry["input_price_per_million_usd"],
            output_price_per_million_usd=entry["output_price_per_million_usd"],
        )
        for model_id, entry in raw_entries.items()
    }
    return {**KNOWN_PRICING, **overrides}
