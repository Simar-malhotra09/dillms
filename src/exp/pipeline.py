"""Orchestrates a single match: optional discovery phase, then N rounds."""

from __future__ import annotations

import random
from datetime import UTC, datetime
from pathlib import Path

from exp.discovery import run_discovery
from exp.games import classify_outcome, compute_payoffs
from exp.judge import run_judge
from exp.llm_client import get_round_decision
from exp.prompts import (
    VALID_ACTIONS,
    build_condition_block,
    build_history_text,
    build_round_messages,
    build_transcript_text,
)
from exp.storage import append_jsonl
from exp.types import CommsRecord, Game, JudgeLabelRecord, MatchConfig, MatchRecord, ModelSpec, RoundRecord


def run_match(config: MatchConfig, results_dir: Path, judge_spec: ModelSpec) -> None:
    matches_path = results_dir / "matches.jsonl"
    rounds_path = results_dir / "rounds.jsonl"
    comms_path = results_dir / "comms.jsonl"
    judge_path = results_dir / "judge_labels.jsonl"

    append_jsonl(
        matches_path,
        MatchRecord(
            match_id=config.match_id,
            game=config.game.value,
            condition=config.condition.value,
            placement=config.placement.value,
            p1_model=config.p1.model_id,
            p2_model=config.p2.model_id,
            p1_prefers=config.p1_prefers,
            p2_prefers=config.p2_prefers,
            seed=config.seed,
            temperature=config.temperature,
            profile_text_p1=config.profile_text_p1,
            profile_text_p2=config.profile_text_p2,
            started_at=datetime.now(UTC).isoformat(),
        ),
    )

    comms: list[CommsRecord] = []
    if config.run_discovery:
        comms = run_discovery(config)
        for entry in comms:
            append_jsonl(comms_path, entry)
        if comms:
            labels = run_judge(judge_spec, comms)
            append_jsonl(
                judge_path,
                JudgeLabelRecord(match_id=config.match_id, **labels.model_dump()),
            )

    p1_discovery_text = build_transcript_text(comms, config.p1.model_id) if comms else ""
    p2_discovery_text = build_transcript_text(comms, config.p2.model_id) if comms else ""

    rng = random.Random(config.seed)
    rounds_so_far_p1: list[RoundRecord] = []
    rounds_so_far_p2: list[RoundRecord] = []
    cum_p1 = 0
    cum_p2 = 0
    round_idx = 1
    continue_game = True

    while continue_game and round_idx <= config.num_rounds:
        p1_block = build_condition_block(config.condition, config.p1, config.p2, config.profile_text_p1)
        p2_block = build_condition_block(config.condition, config.p2, config.p1, config.profile_text_p2)

        p1_history = build_history_text(config.game, config.p1.model_id, rounds_so_far_p1)
        p2_history = build_history_text(config.game, config.p2.model_id, rounds_so_far_p2)

        p1_messages = build_round_messages(
            config.game,
            config.num_rounds,
            config.p1_prefers,
            p1_block,
            config.placement,
            round_idx,
            p1_history,
            p1_discovery_text,
            config.p1.include_reasoning,
        )
        p2_messages = build_round_messages(
            config.game,
            config.num_rounds,
            config.p2_prefers,
            p2_block,
            config.placement,
            round_idx,
            p2_history,
            p2_discovery_text,
            config.p2.include_reasoning,
        )

        valid_actions = VALID_ACTIONS[config.game]
        p1_decision, p1_raw, p1_pt, p1_ct, p1_lat = get_round_decision(
            config.p1, p1_messages, config.temperature, valid_actions
        )
        p2_decision, p2_raw, p2_pt, p2_ct, p2_lat = get_round_decision(
            config.p2, p2_messages, config.temperature, valid_actions
        )

        p1_payoff, p2_payoff = compute_payoffs(
            config.game, p1_decision.action, config.p1_prefers, p2_decision.action, config.p2_prefers
        )
        cum_p1 += p1_payoff
        cum_p2 += p2_payoff

        p1_outcome = classify_outcome(config.game, p1_decision.action, p2_decision.action, p1_payoff, p2_payoff)
        p2_outcome = classify_outcome(config.game, p2_decision.action, p1_decision.action, p2_payoff, p1_payoff)

        p1_record = RoundRecord(
            match_id=config.match_id,
            game=config.game.value,
            round_idx=round_idx,
            actor_model=config.p1.model_id,
            opponent_model=config.p2.model_id,
            condition=config.condition.value,
            placement=config.placement.value,
            self_prefers=config.p1_prefers,
            action=p1_decision.action,
            opponent_action=p2_decision.action,
            self_payoff=p1_payoff,
            opponent_payoff=p2_payoff,
            cum_self=cum_p1,
            cum_opp=cum_p2,
            outcome_class=p1_outcome.value,
            reasoning_text=p1_decision.reasoning,
            raw_response=p1_raw,
            latency_ms=p1_lat,
            prompt_tokens=p1_pt,
            completion_tokens=p1_ct,
        )
        p2_record = RoundRecord(
            match_id=config.match_id,
            game=config.game.value,
            round_idx=round_idx,
            actor_model=config.p2.model_id,
            opponent_model=config.p1.model_id,
            condition=config.condition.value,
            placement=config.placement.value,
            self_prefers=config.p2_prefers,
            action=p2_decision.action,
            opponent_action=p1_decision.action,
            self_payoff=p2_payoff,
            opponent_payoff=p1_payoff,
            cum_self=cum_p2,
            cum_opp=cum_p1,
            outcome_class=p2_outcome.value,
            reasoning_text=p2_decision.reasoning,
            raw_response=p2_raw,
            latency_ms=p2_lat,
            prompt_tokens=p2_pt,
            completion_tokens=p2_ct,
        )

        append_jsonl(rounds_path, p1_record)
        append_jsonl(rounds_path, p2_record)
        rounds_so_far_p1.append(p1_record)
        rounds_so_far_p2.append(p2_record)

        if config.game is Game.IPD:
            continue_game = round_idx < config.num_rounds and rng.random() < 0.9
        round_idx += 1
