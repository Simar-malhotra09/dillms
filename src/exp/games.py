"""Pure payoff and outcome-classification functions. No I/O."""

from __future__ import annotations

from exp.types import Game, OutcomeClass


def bos_payoffs(
    p1_action: str, p1_prefers: str, p2_action: str, p2_prefers: str
) -> tuple[int, int]:
    if p1_action != p2_action:
        return (0, 0)
    p1_points = 3 if p1_action == p1_prefers else 2
    p2_points = 3 if p2_action == p2_prefers else 2
    return (p1_points, p2_points)


def ipd_payoffs(p1_action: str, p2_action: str) -> tuple[int, int]:
    if p1_action == "COOPERATE" and p2_action == "COOPERATE":
        return (3, 3)
    if p1_action == "DEFECT" and p2_action == "DEFECT":
        return (1, 1)
    if p1_action == "COOPERATE" and p2_action == "DEFECT":
        return (0, 5)
    if p1_action == "DEFECT" and p2_action == "COOPERATE":
        return (5, 0)
    raise ValueError(f"invalid IPD action pair: {p1_action!r}, {p2_action!r}")


def compute_payoffs(
    game: Game,
    p1_action: str,
    p1_prefers: str,
    p2_action: str,
    p2_prefers: str,
) -> tuple[int, int]:
    if game is Game.BATTLE_OF_THE_SEXES:
        return bos_payoffs(p1_action, p1_prefers, p2_action, p2_prefers)
    return ipd_payoffs(p1_action, p2_action)


def classify_outcome(
    game: Game,
    self_action: str,
    opponent_action: str,
    self_payoff: int,
    opponent_payoff: int,
) -> OutcomeClass:
    if game is Game.BATTLE_OF_THE_SEXES:
        if self_action != opponent_action:
            return OutcomeClass.MISCOORDINATION
        return (
            OutcomeClass.SELF_FAVORED
            if self_payoff >= opponent_payoff
            else OutcomeClass.OPP_FAVORED
        )

    if self_action == "COOPERATE" and opponent_action == "COOPERATE":
        return OutcomeClass.MUTUAL_COOPERATION
    if self_action == "DEFECT" and opponent_action == "DEFECT":
        return OutcomeClass.MUTUAL_DEFECTION
    if self_action == "DEFECT" and opponent_action == "COOPERATE":
        return OutcomeClass.EXPLOITED_OPPONENT
    return OutcomeClass.EXPLOITED_SELF
