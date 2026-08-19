"""Typed domain model for the identity-exploitation pilot."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel


class Game(StrEnum):
    BATTLE_OF_THE_SEXES = "battle_of_the_sexes"
    IPD = "iterated_prisoners_dilemma"


class Condition(StrEnum):
    ANONYMOUS = "A"
    IDENTITY = "B"
    INFORMED = "C"
    INFORMED_PLACEBO = "C_placebo"


class Placement(StrEnum):
    SYSTEM = "system"
    USER = "user"
    INLINE = "inline"


class OutcomeClass(StrEnum):
    SELF_FAVORED = "SELF_FAVORED"
    OPP_FAVORED = "OPP_FAVORED"
    MISCOORDINATION = "MISCOORDINATION"
    MUTUAL_COOPERATION = "MUTUAL_COOPERATION"
    MUTUAL_DEFECTION = "MUTUAL_DEFECTION"
    EXPLOITED_SELF = "EXPLOITED_SELF"
    EXPLOITED_OPPONENT = "EXPLOITED_OPPONENT"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    self_label: str
    api_key_env: str
    base_url: str
    reasoning_effort: str
    max_completion_tokens: int
    include_reasoning: bool
    thinking_disabled: bool
    required_temperature: float | None


@dataclass(frozen=True)
class MatchConfig:
    match_id: str
    game: Game
    p1_condition: Condition
    p2_condition: Condition
    placement: Placement
    p1: ModelSpec
    p2: ModelSpec
    p1_prefers: str
    p2_prefers: str
    num_rounds: int
    temperature: float
    profile_text_p1: str | None
    profile_text_p2: str | None
    run_discovery: bool
    seed: int


@dataclass(frozen=True)
class MatchRecord:
    match_id: str
    game: str
    p1_condition: str
    p2_condition: str
    placement: str
    p1_model: str
    p2_model: str
    p1_prefers: str
    p2_prefers: str
    seed: int
    temperature: float
    profile_text_p1: str | None
    profile_text_p2: str | None
    started_at: str


@dataclass(frozen=True)
class RoundRecord:
    match_id: str
    game: str
    round_idx: int
    actor_model: str
    opponent_model: str
    condition: str
    placement: str
    self_prefers: str
    action: str
    opponent_action: str
    self_payoff: int
    opponent_payoff: int
    cum_self: int
    cum_opp: int
    outcome_class: str
    reasoning_text: str
    raw_response: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int


@dataclass(frozen=True)
class CommsRecord:
    match_id: str
    exchange_idx: int
    speaker_model: str
    message_text: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    latency_ms: float


@dataclass(frozen=True)
class ModelPricing:
    model_id: str
    input_price_per_million_usd: float
    output_price_per_million_usd: float


class RoundDecision(BaseModel):
    reasoning: str
    action: str


class ActionOnlyDecision(BaseModel):
    action: str


class JudgeLabels(BaseModel):
    asks_opponent_identity: bool
    discloses_own_identity: bool
    misrepresents_self: bool
    proposes_convention: bool
    makes_threat_or_commitment: bool


@dataclass(frozen=True)
class JudgeLabelRecord:
    match_id: str
    asks_opponent_identity: bool
    discloses_own_identity: bool
    misrepresents_self: bool
    proposes_convention: bool
    makes_threat_or_commitment: bool
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    latency_ms: float
