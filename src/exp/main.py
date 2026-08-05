"""CLI entry point.

`uv run python -m exp.main smoke`  -- tiny end-to-end correctness check.
`uv run python -m exp.main pilot [--reps N] [--bos-rounds N] [--ipd-max-rounds N] [--out DIR]`
    -- runs the staged pilot: BoS+IPD condition A (seeds profiles), then
    B (with BoS placement sweep), then C / C_placebo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from exp.llm_client import DEEPSEEK, KIMI
from exp.pipeline import run_match
from exp.profiling import build_real_and_placebo_profiles
from exp.storage import read_jsonl
from exp.types import Condition, Game, MatchConfig, ModelSpec, Placement

MODELS: list[ModelSpec] = [DEEPSEEK, KIMI]
JUDGE = KIMI


def _seatings() -> list[tuple[ModelSpec, ModelSpec]]:
    return [(DEEPSEEK, KIMI), (KIMI, DEEPSEEK)]


def _run_condition_a(
    game: Game, num_rounds: int, reps: int, results_dir: Path, run_discovery: bool
) -> None:
    match_index = 0
    for p1, p2 in _seatings():
        for rep in range(reps):
            match_index += 1
            config = MatchConfig(
                match_id=f"{game.value}_A_user_{p1.model_id}-vs-{p2.model_id}_rep{rep}",
                game=game,
                condition=Condition.ANONYMOUS,
                placement=Placement.USER,
                p1=p1,
                p2=p2,
                p1_prefers="RED" if game is Game.BATTLE_OF_THE_SEXES else "COOPERATE",
                p2_prefers="BLUE" if game is Game.BATTLE_OF_THE_SEXES else "COOPERATE",
                num_rounds=num_rounds,
                temperature=1.0,
                profile_text_p1=None,
                profile_text_p2=None,
                run_discovery=run_discovery,
                seed=1000 + match_index,
            )
            print(f"[{game.value}] A rep {rep}: {p1.model_id} vs {p2.model_id}", file=sys.stderr)
            run_match(config, results_dir, JUDGE)


def _run_condition_b(
    game: Game, num_rounds: int, reps: int, results_dir: Path, placements: list[Placement], run_discovery: bool
) -> None:
    match_index = 0
    for placement in placements:
        for p1, p2 in _seatings():
            for rep in range(reps):
                match_index += 1
                config = MatchConfig(
                    match_id=f"{game.value}_B_{placement.value}_{p1.model_id}-vs-{p2.model_id}_rep{rep}",
                    game=game,
                    condition=Condition.IDENTITY,
                    placement=placement,
                    p1=p1,
                    p2=p2,
                    p1_prefers="RED" if game is Game.BATTLE_OF_THE_SEXES else "COOPERATE",
                    p2_prefers="BLUE" if game is Game.BATTLE_OF_THE_SEXES else "COOPERATE",
                    num_rounds=num_rounds,
                    temperature=1.0,
                    profile_text_p1=None,
                    profile_text_p2=None,
                    run_discovery=run_discovery,
                    seed=2000 + match_index,
                )
                print(
                    f"[{game.value}] B rep {rep} placement={placement.value}: "
                    f"{p1.model_id} vs {p2.model_id}",
                    file=sys.stderr,
                )
                run_match(config, results_dir, JUDGE)


def _run_condition_c(
    game: Game,
    num_rounds: int,
    reps: int,
    results_dir: Path,
    condition: Condition,
    profile_by_model: dict[str, str],
) -> None:
    match_index = 0
    for p1, p2 in _seatings():
        for rep in range(reps):
            match_index += 1
            config = MatchConfig(
                match_id=f"{game.value}_{condition.value}_user_{p1.model_id}-vs-{p2.model_id}_rep{rep}",
                game=game,
                condition=condition,
                placement=Placement.USER,
                p1=p1,
                p2=p2,
                p1_prefers="RED" if game is Game.BATTLE_OF_THE_SEXES else "COOPERATE",
                p2_prefers="BLUE" if game is Game.BATTLE_OF_THE_SEXES else "COOPERATE",
                num_rounds=num_rounds,
                temperature=1.0,
                profile_text_p1=profile_by_model[p1.model_id],
                profile_text_p2=profile_by_model[p2.model_id],
                run_discovery=False,
                seed=3000 + match_index,
            )
            print(
                f"[{game.value}] {condition.value} rep {rep}: {p1.model_id} vs {p2.model_id}",
                file=sys.stderr,
            )
            run_match(config, results_dir, JUDGE)


def run_smoke(results_dir: Path) -> None:
    config = MatchConfig(
        match_id="smoke_bos_A_user",
        game=Game.BATTLE_OF_THE_SEXES,
        condition=Condition.ANONYMOUS,
        placement=Placement.USER,
        p1=DEEPSEEK,
        p2=KIMI,
        p1_prefers="RED",
        p2_prefers="BLUE",
        num_rounds=3,
        temperature=1.0,
        profile_text_p1=None,
        profile_text_p2=None,
        run_discovery=True,
        seed=1,
    )
    run_match(config, results_dir, JUDGE)
    print(f"Smoke test complete. Output in {results_dir}", file=sys.stderr)


def run_pilot(results_dir: Path, reps: int, bos_rounds: int, ipd_max_rounds: int) -> None:
    print("=== Stage 1: condition A (seeds behavioral profiles) ===", file=sys.stderr)
    _run_condition_a(Game.BATTLE_OF_THE_SEXES, bos_rounds, reps, results_dir, run_discovery=True)
    _run_condition_a(Game.IPD, ipd_max_rounds, reps, results_dir, run_discovery=False)

    rounds = read_jsonl(results_dir / "rounds.jsonl")
    bos_profiles: dict[str, str] = {}
    bos_placebo: dict[str, str] = {}
    ipd_profiles: dict[str, str] = {}
    ipd_placebo: dict[str, str] = {}
    for model in MODELS:
        real, placebo = build_real_and_placebo_profiles(
            rounds, model.model_id, Game.BATTLE_OF_THE_SEXES, model.self_label
        )
        bos_profiles[model.model_id] = real
        bos_placebo[model.model_id] = placebo
        real_ipd, placebo_ipd = build_real_and_placebo_profiles(
            rounds, model.model_id, Game.IPD, model.self_label
        )
        ipd_profiles[model.model_id] = real_ipd
        ipd_placebo[model.model_id] = placebo_ipd

    print("=== Stage 2: condition B (identity, BoS placement sweep) ===", file=sys.stderr)
    _run_condition_b(
        Game.BATTLE_OF_THE_SEXES,
        bos_rounds,
        reps,
        results_dir,
        [Placement.SYSTEM, Placement.USER, Placement.INLINE],
        run_discovery=True,
    )
    _run_condition_b(Game.IPD, ipd_max_rounds, reps, results_dir, [Placement.USER], run_discovery=False)

    print("=== Stage 3: condition C / C_placebo (informed) ===", file=sys.stderr)
    _run_condition_c(Game.BATTLE_OF_THE_SEXES, bos_rounds, reps, results_dir, Condition.INFORMED, bos_profiles)
    _run_condition_c(
        Game.BATTLE_OF_THE_SEXES, bos_rounds, reps, results_dir, Condition.INFORMED_PLACEBO, bos_placebo
    )
    _run_condition_c(Game.IPD, ipd_max_rounds, reps, results_dir, Condition.INFORMED, ipd_profiles)

    print(f"Pilot complete. Output in {results_dir}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--out", type=Path, default=Path("results/smoke"))

    pilot_parser = subparsers.add_parser("pilot")
    pilot_parser.add_argument("--out", type=Path, default=Path("results/pilot"))
    pilot_parser.add_argument("--reps", type=int, default=3)
    pilot_parser.add_argument("--bos-rounds", type=int, default=20)
    pilot_parser.add_argument("--ipd-max-rounds", type=int, default=20)

    args = parser.parse_args()

    if args.mode == "smoke":
        run_smoke(args.out)
    else:
        run_pilot(args.out, args.reps, args.bos_rounds, args.ipd_max_rounds)


if __name__ == "__main__":
    main()
