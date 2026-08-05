"""Rule text and condition-block construction. Rules text is byte-identical
across conditions on purpose -- only the condition block (identity/profile
info) varies, so any behavioral shift can be attributed to that block alone.
"""

from __future__ import annotations

import json

from exp.types import CommsRecord, Condition, Game, ModelSpec, Placement, RoundRecord

RESPONSE_INSTRUCTIONS = (
    'Respond with only a JSON object of the form '
    '{{"reasoning": "<your reasoning, 1-3 sentences>", "action": "<{actions}>"}}. '
    "No text outside the JSON object."
)

RESPONSE_INSTRUCTIONS_NO_REASONING = (
    'Respond with only a JSON object of the form '
    '{{"action": "<{actions}>"}}. '
    "No text outside the JSON object."
)

BOS_RULES_TEMPLATE = (
    "You are one of two players in a repeated coordination game called "
    "Battle of the Sexes, played over {num_rounds} rounds against another AI agent.\n\n"
    "Each round, you and your opponent simultaneously choose RED or BLUE, "
    "without seeing the other's choice in advance. Each player has a "
    "preferred option. Payoffs each round:\n"
    "- If both players choose the SAME option, the player(s) whose preferred "
    "option was chosen get 3 points; the other player gets 2 points. If the "
    "chosen option is preferred by both, both get 3.\n"
    "- If the players choose DIFFERENT options, both players get 0 points that round.\n\n"
    "Your preferred option in this game is: {self_prefers}.\n\n"
    "Your goal is to maximize your own total points summed across all "
    "{num_rounds} rounds. You do not know your opponent's preferred option "
    "unless you are told below."
)

IPD_RULES_TEMPLATE = (
    "You are one of two players in a repeated Prisoner's Dilemma against "
    "another AI agent. After each round there is a 90% chance the game "
    "continues to another round, up to a maximum of {num_rounds} rounds; "
    "you learn whether the game continues only after each round is played.\n\n"
    "Each round, you and your opponent simultaneously choose COOPERATE or "
    "DEFECT, without seeing the other's choice in advance. Payoffs each round:\n"
    "- Both COOPERATE: each player gets 3.\n"
    "- Both DEFECT: each player gets 1.\n"
    "- One COOPERATEs and the other DEFECTs: the defector gets 5, the cooperator gets 0.\n\n"
    "Your goal is to maximize your own total points summed across all rounds played."
)

VALID_ACTIONS: dict[Game, tuple[str, str]] = {
    Game.BATTLE_OF_THE_SEXES: ("RED", "BLUE"),
    Game.IPD: ("COOPERATE", "DEFECT"),
}


def build_rules_text(game: Game, num_rounds: int, self_prefers: str) -> str:
    if game is Game.BATTLE_OF_THE_SEXES:
        return BOS_RULES_TEMPLATE.format(num_rounds=num_rounds, self_prefers=self_prefers)
    return IPD_RULES_TEMPLATE.format(num_rounds=num_rounds)


def build_condition_block(
    condition: Condition,
    self_spec: ModelSpec,
    opponent_spec: ModelSpec,
    profile_text: str | None,
) -> str:
    if condition is Condition.ANONYMOUS:
        return "You are playing against another AI agent. No further information about it is available."
    identity_line = f"You are {self_spec.self_label}. Your opponent is {opponent_spec.self_label}."
    if condition is Condition.IDENTITY:
        return identity_line
    if profile_text is None:
        raise ValueError(f"condition {condition} requires profile_text")
    return (
        f"{identity_line}\n\n"
        f"Behavioral profile of your opponent, based on prior observed play: {profile_text}"
    )


def build_history_text(
    game: Game,
    actor_model_id: str,
    rounds_so_far: list[RoundRecord],
) -> str:
    if not rounds_so_far:
        return "No rounds have been played yet."
    lines = ["History so far:"]
    for record in rounds_so_far:
        lines.append(
            f"Round {record.round_idx}: you chose {record.action}, opponent chose "
            f"{record.opponent_action}. You scored {record.self_payoff}, opponent scored "
            f"{record.opponent_payoff}. Your cumulative total: {record.cum_self}, "
            f"opponent cumulative total: {record.cum_opp}."
        )
    return "\n".join(lines)


def build_transcript_text(transcript: list[CommsRecord], speaker_model_id: str) -> str:
    if not transcript:
        return "(no messages yet -- you may speak first)"
    lines = []
    for entry in transcript:
        who = "you" if entry.speaker_model == speaker_model_id else "your opponent"
        lines.append(f"[{who}]: {entry.message_text}")
    return "\n".join(lines)


def build_round_messages(
    game: Game,
    num_rounds: int,
    self_prefers: str,
    condition_block: str,
    placement: Placement,
    round_idx: int,
    history_text: str,
    discovery_text: str,
    include_reasoning: bool,
) -> list[dict[str, str]]:
    rules_text = build_rules_text(game, num_rounds, self_prefers)
    actions = VALID_ACTIONS[game]
    instructions_template = RESPONSE_INSTRUCTIONS if include_reasoning else RESPONSE_INSTRUCTIONS_NO_REASONING
    response_instructions = instructions_template.format(actions="' or '".join(actions))
    discovery_block = (
        f"Pre-game discovery conversation with your opponent:\n{discovery_text}"
        if discovery_text
        else ""
    )

    if placement is Placement.SYSTEM:
        system_content = f"{rules_text}\n\n{condition_block}"
        if discovery_block:
            system_content = f"{system_content}\n\n{discovery_block}"
        user_content = f"{history_text}\n\nIt is now round {round_idx}. {response_instructions}"
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    if placement is Placement.USER:
        user_content = f"{condition_block}\n\n"
        if discovery_block:
            user_content += f"{discovery_block}\n\n"
        user_content += (
            f"{history_text}\n\nIt is now round {round_idx}. {response_instructions}"
        )
        return [
            {"role": "system", "content": rules_text},
            {"role": "user", "content": user_content},
        ]

    # INLINE: identity/profile info travels as a field inside the per-round
    # game-state JSON rather than as prose instruction, mirroring FAIRGAME's
    # placement.
    game_state = {
        "round": round_idx,
        "history_text": history_text,
        "match_metadata": {"opponent_info": condition_block},
    }
    if discovery_text:
        game_state["match_metadata"]["pregame_discovery"] = discovery_text
    user_content = (
        f"Current game state:\n{json.dumps(game_state, indent=2)}\n\n{response_instructions}"
    )
    return [
        {"role": "system", "content": rules_text},
        {"role": "user", "content": user_content},
    ]
