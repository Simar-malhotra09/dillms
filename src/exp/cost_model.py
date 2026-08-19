"""Cost projection for scaling the pilot up in rounds, reps, and model tier.

Input-token cost is computed analytically from the same prompt-building
functions the pipeline uses, with no API calls, because build_history_text
(prompts.py) never feeds a model's own reasoning/raw output back into later
rounds -- prompt length is a deterministic function of round index alone.

Output-token cost is model-tier-dependent (verbosity, hidden-reasoning
depth, JSON-retry rate all vary by model) and cannot be inferred from a
cheaper model's completion length. It is either read from empirical
RoundRecord logs for a model you've actually run, or must be supplied
explicitly via --assume-completion-tokens for a model you haven't.

Usage:
    uv run python -m exp.cost_model --game battle_of_the_sexes --rounds 30 \\
        --matches 20 --model-id deepseek-v4-flash --results-dir results/pilot

    uv run python -m exp.cost_model --game iterated_prisoners_dilemma --rounds 30 \\
        --matches 20 --model-id gpt-5 --assume-completion-tokens 180 \\
        --pricing-file my_pricing.json
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import tiktoken

from exp.discovery import MAX_WORDS
from exp.games import classify_outcome, compute_payoffs
from exp.pricing import KNOWN_PRICING, load_pricing_overrides
from exp.prompts import (
    build_condition_block,
    build_history_text,
    build_round_messages,
    build_transcript_text,
)
from exp.storage import read_jsonl
from exp.types import CommsRecord, Condition, Game, ModelPricing, ModelSpec, Placement, RoundRecord


def _encoding_for_model(model_id: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model_id)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def _synthetic_round_history(
    game: Game, num_rounds: int, self_prefers: str, opponent_prefers: str
) -> list[RoundRecord]:
    """A representative round sequence for token-length purposes only --
    steady mutual cooperation for IPD, alternating favored player for BoS.
    Digit widths of scores approximate real cumulative totals; the exact
    action sequence chosen barely affects history-line token count.
    """
    history: list[RoundRecord] = []
    cum_self = 0
    cum_opp = 0
    for round_idx in range(1, num_rounds + 1):
        if game is Game.IPD:
            self_action, opponent_action = "COOPERATE", "COOPERATE"
        else:
            self_action = self_prefers if round_idx % 2 == 1 else opponent_prefers
            opponent_action = self_action
        self_payoff, opponent_payoff = compute_payoffs(
            game, self_action, self_prefers, opponent_action, opponent_prefers
        )
        cum_self += self_payoff
        cum_opp += opponent_payoff
        history.append(
            RoundRecord(
                match_id="synthetic",
                game=game.value,
                round_idx=round_idx,
                actor_model="synthetic",
                opponent_model="synthetic",
                condition="",
                placement="",
                self_prefers=self_prefers,
                action=self_action,
                opponent_action=opponent_action,
                self_payoff=self_payoff,
                opponent_payoff=opponent_payoff,
                cum_self=cum_self,
                cum_opp=cum_opp,
                outcome_class=classify_outcome(
                    game, self_action, opponent_action, self_payoff, opponent_payoff
                ).value,
                reasoning_text="",
                raw_response="",
                latency_ms=0.0,
                prompt_tokens=0,
                completion_tokens=0,
                reasoning_tokens=0,
            )
        )
    return history


def worst_case_discovery_text() -> str:
    """A discovery transcript at the MAX_WORDS cap on all 4 exchanges --
    the largest discovery_block the real pipeline can ever produce.
    """
    synthetic = [
        CommsRecord(
            match_id="synthetic",
            exchange_idx=i,
            speaker_model="a" if i % 2 == 0 else "b",
            message_text=" ".join(["word"] * MAX_WORDS),
            prompt_tokens=0,
            completion_tokens=0,
            reasoning_tokens=0,
            latency_ms=0.0,
        )
        for i in range(4)
    ]
    return build_transcript_text(synthetic, "a")


def analytic_input_tokens_per_round(
    game: Game,
    num_rounds: int,
    placement: Placement,
    self_prefers: str,
    opponent_prefers: str,
    condition_block: str,
    discovery_text: str,
    include_reasoning: bool,
    encoding: tiktoken.Encoding,
) -> list[int]:
    """Input (prompt) token count for each round, computed deterministically
    from the actual prompt-building code -- no model calls involved.
    """
    full_history = _synthetic_round_history(game, num_rounds, self_prefers, opponent_prefers)
    token_counts: list[int] = []
    for round_idx in range(1, num_rounds + 1):
        history_text = build_history_text(game, "synthetic", full_history[: round_idx - 1])
        messages = build_round_messages(
            game,
            num_rounds,
            self_prefers,
            condition_block,
            placement,
            round_idx,
            history_text,
            discovery_text,
            include_reasoning,
        )
        content = "\n".join(message["content"] for message in messages)
        token_counts.append(len(encoding.encode(content)))
    return token_counts


@dataclass(frozen=True)
class EmpiricalOutputStats:
    model_id: str
    game: str
    num_observations: int
    mean_completion_tokens: float
    mean_reasoning_tokens: float
    mean_latency_ms: float


def compute_empirical_output_stats(rounds: list[dict], model_id: str, game: Game) -> EmpiricalOutputStats:
    relevant = [r for r in rounds if r["actor_model"] == model_id and r["game"] == game.value]
    if not relevant:
        raise ValueError(f"no logged rounds for model {model_id!r} in game {game.value!r}")
    num_observations = len(relevant)
    mean_completion_tokens = sum(r["completion_tokens"] for r in relevant) / num_observations
    mean_reasoning_tokens = sum(r.get("reasoning_tokens", 0) for r in relevant) / num_observations
    mean_latency_ms = sum(r["latency_ms"] for r in relevant) / num_observations
    return EmpiricalOutputStats(
        model_id, game.value, num_observations, mean_completion_tokens, mean_reasoning_tokens, mean_latency_ms
    )


@dataclass(frozen=True)
class CostProjection:
    model_id: str
    game: str
    num_rounds: int
    num_matches: int
    input_tokens_total: int
    output_tokens_total: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    output_basis: str


def project_cost(
    model_id: str,
    game: Game,
    num_rounds: int,
    num_matches: int,
    input_tokens_per_round: list[int],
    completion_tokens_per_round: float,
    output_basis: str,
    pricing: ModelPricing,
) -> CostProjection:
    input_tokens_total = sum(input_tokens_per_round) * num_matches
    output_tokens_total = round(completion_tokens_per_round * num_rounds * num_matches)
    input_cost_usd = input_tokens_total / 1_000_000 * pricing.input_price_per_million_usd
    output_cost_usd = output_tokens_total / 1_000_000 * pricing.output_price_per_million_usd
    return CostProjection(
        model_id=model_id,
        game=game.value,
        num_rounds=num_rounds,
        num_matches=num_matches,
        input_tokens_total=input_tokens_total,
        output_tokens_total=output_tokens_total,
        input_cost_usd=input_cost_usd,
        output_cost_usd=output_cost_usd,
        total_cost_usd=input_cost_usd + output_cost_usd,
        output_basis=output_basis,
    )


def _placeholder_model_spec(model_id: str, self_label: str, include_reasoning: bool) -> ModelSpec:
    return ModelSpec(
        model_id=model_id,
        self_label=self_label,
        api_key_env="",
        base_url="",
        reasoning_effort="low",
        max_completion_tokens=0,
        include_reasoning=include_reasoning,
        thinking_disabled=False,
        required_temperature=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", choices=[g.value for g in Game], required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--matches", type=int, required=True)
    parser.add_argument("--placement", choices=[p.value for p in Placement], default=Placement.USER.value)
    parser.add_argument("--condition", choices=[c.value for c in Condition], default=Condition.ANONYMOUS.value)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--self-label", default=None)
    parser.add_argument("--opponent-label", default="Opponent")
    parser.add_argument("--self-prefers", default=None)
    parser.add_argument("--opponent-prefers", default=None)
    parser.add_argument("--include-reasoning", action="store_true")
    parser.add_argument("--discovery", action="store_true")
    parser.add_argument("--profile-text", default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--assume-completion-tokens", type=float, default=None)
    parser.add_argument("--pricing-file", type=Path, default=None)
    args = parser.parse_args()

    game = Game(args.game)
    placement = Placement(args.placement)
    condition = Condition(args.condition)
    self_prefers = args.self_prefers or ("RED" if game is Game.BATTLE_OF_THE_SEXES else "COOPERATE")
    opponent_prefers = args.opponent_prefers or ("BLUE" if game is Game.BATTLE_OF_THE_SEXES else "COOPERATE")
    self_label = args.self_label or args.model_id
    encoding = _encoding_for_model(args.model_id)

    self_spec = _placeholder_model_spec(args.model_id, self_label, args.include_reasoning)
    opponent_spec = _placeholder_model_spec("opponent", args.opponent_label, args.include_reasoning)
    condition_block = build_condition_block(condition, self_spec, opponent_spec, args.profile_text)
    discovery_text = worst_case_discovery_text() if args.discovery else ""

    input_tokens_per_round = analytic_input_tokens_per_round(
        game,
        args.rounds,
        placement,
        self_prefers,
        opponent_prefers,
        condition_block,
        discovery_text,
        args.include_reasoning,
        encoding,
    )

    pricing_table = KNOWN_PRICING if args.pricing_file is None else load_pricing_overrides(args.pricing_file)
    if args.model_id not in pricing_table:
        raise ValueError(f"no pricing known for {args.model_id!r}; supply --pricing-file with an entry for it")
    pricing = pricing_table[args.model_id]

    completion_tokens_per_round: float
    output_basis: str
    empirical_stats: EmpiricalOutputStats | None = None
    if args.results_dir is not None:
        rounds = read_jsonl(args.results_dir / "rounds.jsonl")
        try:
            empirical_stats = compute_empirical_output_stats(rounds, args.model_id, game)
        except ValueError:
            empirical_stats = None
    if empirical_stats is not None:
        completion_tokens_per_round = empirical_stats.mean_completion_tokens
        output_basis = f"empirical:{empirical_stats.num_observations}_rounds"
    elif args.assume_completion_tokens is not None:
        completion_tokens_per_round = args.assume_completion_tokens
        output_basis = "assumed"
    else:
        raise ValueError(
            f"no logged rounds for {args.model_id!r} in {game.value!r} under {args.results_dir} "
            "and no --assume-completion-tokens given"
        )

    projection = project_cost(
        args.model_id,
        game,
        args.rounds,
        args.matches,
        input_tokens_per_round,
        completion_tokens_per_round,
        output_basis,
        pricing,
    )

    print(f"model:            {projection.model_id}")
    print(f"game:             {projection.game}")
    print(f"rounds per match: {projection.num_rounds}")
    print(f"matches:          {projection.num_matches}")
    print(f"input tokens:     {projection.input_tokens_total:,} (@ ${pricing.input_price_per_million_usd}/M)")
    print(
        f"output tokens:    {projection.output_tokens_total:,} "
        f"(@ ${pricing.output_price_per_million_usd}/M, basis={projection.output_basis})"
    )
    print(f"input cost:       ${projection.input_cost_usd:,.4f}")
    print(f"output cost:      ${projection.output_cost_usd:,.4f}")
    print(f"total cost:       ${projection.total_cost_usd:,.4f}")


if __name__ == "__main__":
    main()
