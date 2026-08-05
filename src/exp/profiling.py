"""Builds an empirical behavioral profile of a model from its own
condition-A (anonymous) round logs, plus an inverted placebo version with
the same structure but false numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby

from exp.types import Game


@dataclass(frozen=True)
class ProfileStats:
    num_matches: int
    round1_rate: float | None
    round1_n: int
    switch_rate: float | None
    switch_opportunities: int


def compute_profile_stats(
    rounds: list[dict], target_model_id: str, game: Game
) -> ProfileStats:
    relevant = [
        r
        for r in rounds
        if r["actor_model"] == target_model_id
        and r["condition"] == "A"
        and r["game"] == game.value
    ]
    if not relevant:
        return ProfileStats(0, None, 0, None, 0)

    round1_rows = [r for r in relevant if r["round_idx"] == 1]
    round1_rate = (
        sum(1 for r in round1_rows if r["action"] == r["self_prefers"]) / len(round1_rows)
        if round1_rows
        else None
    )

    switches = 0
    opportunities = 0
    relevant_sorted = sorted(relevant, key=lambda r: r["match_id"])
    for _, group_iter in groupby(relevant_sorted, key=lambda r: r["match_id"]):
        match_rounds = sorted(group_iter, key=lambda r: r["round_idx"])
        for i in range(2, len(match_rounds)):
            prev2, prev1, curr = match_rounds[i - 2], match_rounds[i - 1], match_rounds[i]
            if prev2["outcome_class"] == "MISCOORDINATION" and prev1["outcome_class"] == "MISCOORDINATION":
                opportunities += 1
                if curr["action"] != prev1["action"]:
                    switches += 1
    switch_rate = switches / opportunities if opportunities else None

    num_matches = len({r["match_id"] for r in relevant})
    return ProfileStats(num_matches, round1_rate, len(round1_rows), switch_rate, opportunities)


def format_profile_text(label: str, stats: ProfileStats) -> str:
    if stats.round1_rate is None:
        raise ValueError(f"no condition-A round-1 data available for {label}")
    text = (
        f"In {stats.num_matches} previously observed matches ({stats.round1_n} round-1 "
        f"observations), {label} chose its own preferred option in round 1 in "
        f"{stats.round1_rate:.0%} of cases"
    )
    if stats.switch_rate is not None:
        text += (
            f", and switched its choice after 2 consecutive miscoordination rounds in "
            f"{stats.switch_rate:.0%} of {stats.switch_opportunities} such instances."
        )
    else:
        text += "."
    return text


def build_real_and_placebo_profiles(
    rounds: list[dict], target_model_id: str, game: Game, label: str
) -> tuple[str, str]:
    stats = compute_profile_stats(rounds, target_model_id, game)
    real_text = format_profile_text(label, stats)

    placebo_round1_rate = 1.0 - stats.round1_rate  # type: ignore[operator]
    placebo_switch_rate = 1.0 - stats.switch_rate if stats.switch_rate is not None else None
    placebo_stats = ProfileStats(
        stats.num_matches, placebo_round1_rate, stats.round1_n, placebo_switch_rate, stats.switch_opportunities
    )
    placebo_text = format_profile_text(label, placebo_stats)
    return real_text, placebo_text
