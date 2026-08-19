"""CLI entry point.

`uv run python -m exp.main smoke`  -- tiny end-to-end correctness check.
`uv run python -m exp.main pilot [--reps N] [--bos-rounds N] [--ipd-max-rounds N]
    [--ipd-reps N] [--skip-bos] [--out DIR]`
    -- runs the staged pilot: BoS+IPD condition A (seeds profiles), then
    B (with BoS placement sweep), then C / C_placebo for both games.
    --ipd-reps overrides --reps for IPD stages only (defaults to --reps).
    --skip-bos runs IPD-only, for a focused/cheaper condition-comparison run.
`uv run python -m exp.main identity-fixed [--reps N] [--bos-rounds N] [--out DIR]`
    -- holds one agent (DeepSeek) fixed as the main agent and only varies
    the opponent's revealed self_label ("Gandhi" vs "Hitler") across arms,
    with the same underlying opponent model and paired seeds per rep. Only
    the main agent's rows are meant to be analyzed; the opponent is the
    same policy in both arms, just labeled differently.
`uv run python -m exp.main self-play --model {deepseek,kimi,kimi-instant}
    [--reps N] [--bos-rounds N] [--ipd-max-rounds N] [--ipd-reps N] [--skip-bos] [--out DIR]`
    -- runs the same staged pilot (A/B/C/C_placebo) with one model playing
    against itself, so any behavioral shift is attributable purely to the
    condition and not to a cross-model confound. kimi-instant runs Kimi
    with hidden reasoning disabled (see llm_client.KIMI_INSTANT) -- cheap
    and fast, but not reasoning-comparable to kimi or deepseek.
`uv run python -m exp.main fixed-opponent --subject {deepseek,kimi,kimi-instant}
    --opponent {deepseek,kimi,kimi-instant} [--reps N] [--bos-rounds N]
    [--ipd-max-rounds N] [--ipd-reps N] [--skip-bos] [--out DIR]
    -- one subject's condition sweeps A -> B -> C -> C_placebo while the
    fixed opponent's policy and condition (always A) never move, so any
    behavioral shift in the subject is attributable purely to what the
    subject alone was told, with no confound from the opponent's own
    treatment moving too. Only the subject's rows are meant to be
    analyzed. See FUTURE_DESIGNS.md for the full rationale.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from exp.llm_client import DEEPSEEK, KIMI, KIMI_INSTANT
from exp.pipeline import run_match
from exp.profiling import build_real_and_placebo_profiles
from exp.storage import read_jsonl
from exp.types import Condition, Game, MatchConfig, ModelSpec, Placement

MODELS: list[ModelSpec] = [DEEPSEEK, KIMI]
JUDGE = KIMI

SELF_PLAY_MODELS: dict[str, ModelSpec] = {
    "deepseek": DEEPSEEK,
    "kimi": KIMI,
    "kimi-instant": KIMI_INSTANT,
}


def _seatings() -> list[tuple[ModelSpec, ModelSpec]]:
    return [(DEEPSEEK, KIMI), (KIMI, DEEPSEEK)]


def _run_condition_a(
    game: Game,
    num_rounds: int,
    reps: int,
    results_dir: Path,
    run_discovery: bool,
    seatings: list[tuple[ModelSpec, ModelSpec]],
) -> None:
    match_index = 0
    for p1, p2 in seatings:
        for rep in range(reps):
            match_index += 1
            config = MatchConfig(
                match_id=f"{game.value}_A_user_{p1.model_id}-vs-{p2.model_id}_rep{rep}",
                game=game,
                p1_condition=Condition.ANONYMOUS,
                p2_condition=Condition.ANONYMOUS,
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
    game: Game,
    num_rounds: int,
    reps: int,
    results_dir: Path,
    placements: list[Placement],
    run_discovery: bool,
    seatings: list[tuple[ModelSpec, ModelSpec]],
) -> None:
    match_index = 0
    for placement in placements:
        for p1, p2 in seatings:
            for rep in range(reps):
                match_index += 1
                config = MatchConfig(
                    match_id=f"{game.value}_B_{placement.value}_{p1.model_id}-vs-{p2.model_id}_rep{rep}",
                    game=game,
                    p1_condition=Condition.IDENTITY,
                    p2_condition=Condition.IDENTITY,
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
    seatings: list[tuple[ModelSpec, ModelSpec]],
) -> None:
    match_index = 0
    for p1, p2 in seatings:
        for rep in range(reps):
            match_index += 1
            config = MatchConfig(
                match_id=f"{game.value}_{condition.value}_user_{p1.model_id}-vs-{p2.model_id}_rep{rep}",
                game=game,
                p1_condition=condition,
                p2_condition=condition,
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
        p1_condition=Condition.ANONYMOUS,
        p2_condition=Condition.ANONYMOUS,
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


def _run_main_vs_labeled_opponent(
    game: Game,
    num_rounds: int,
    reps: int,
    results_dir: Path,
    main_agent: ModelSpec,
    opponent_base: ModelSpec,
    opponent_labels: list[str],
    run_discovery: bool,
) -> None:
    for label in opponent_labels:
        opponent = dataclasses.replace(opponent_base, self_label=label)
        for rep in range(reps):
            config = MatchConfig(
                match_id=f"{game.value}_B_fixedmain_{main_agent.model_id}-vs-{label.lower()}_rep{rep}",
                game=game,
                p1_condition=Condition.IDENTITY,
                p2_condition=Condition.IDENTITY,
                placement=Placement.USER,
                p1=main_agent,
                p2=opponent,
                p1_prefers="RED" if game is Game.BATTLE_OF_THE_SEXES else "COOPERATE",
                p2_prefers="BLUE" if game is Game.BATTLE_OF_THE_SEXES else "COOPERATE",
                num_rounds=num_rounds,
                temperature=1.0,
                profile_text_p1=None,
                profile_text_p2=None,
                run_discovery=run_discovery,
                seed=4000 + rep,
            )
            print(
                f"[{game.value}] fixed-main rep {rep}: {main_agent.model_id} vs "
                f"{opponent_base.model_id} labeled '{label}'",
                file=sys.stderr,
            )
            run_match(config, results_dir, JUDGE)


def run_identity_fixed(results_dir: Path, reps: int, bos_rounds: int) -> None:
    _run_main_vs_labeled_opponent(
        Game.BATTLE_OF_THE_SEXES,
        bos_rounds,
        reps,
        results_dir,
        main_agent=DEEPSEEK,
        opponent_base=KIMI,
        opponent_labels=["Gandhi", "Hitler"],
        run_discovery=True,
    )
    print(f"Fixed-main identity run complete. Output in {results_dir}", file=sys.stderr)


def _run_asymmetric_stage(
    game: Game,
    num_rounds: int,
    reps: int,
    results_dir: Path,
    subject_condition: Condition,
    subject: ModelSpec,
    opponent: ModelSpec,
    subject_profile_text: str | None,
    run_discovery: bool,
    seed_base: int,
) -> None:
    for rep in range(reps):
        config = MatchConfig(
            match_id=(
                f"{game.value}_{subject_condition.value}_fixedopp_"
                f"{subject.model_id}-vs-{opponent.model_id}_rep{rep}"
            ),
            game=game,
            p1_condition=subject_condition,
            p2_condition=Condition.ANONYMOUS,
            placement=Placement.USER,
            p1=subject,
            p2=opponent,
            p1_prefers="RED" if game is Game.BATTLE_OF_THE_SEXES else "COOPERATE",
            p2_prefers="BLUE" if game is Game.BATTLE_OF_THE_SEXES else "COOPERATE",
            num_rounds=num_rounds,
            temperature=1.0,
            profile_text_p1=subject_profile_text,
            profile_text_p2=None,
            run_discovery=run_discovery,
            seed=seed_base + rep,
        )
        print(
            f"[{game.value}] subject={subject_condition.value} rep {rep}: "
            f"{subject.model_id} (subject) vs {opponent.model_id} (fixed, always A)",
            file=sys.stderr,
        )
        run_match(config, results_dir, JUDGE)


def run_fixed_opponent(
    results_dir: Path,
    reps: int,
    bos_rounds: int,
    ipd_max_rounds: int,
    ipd_reps: int | None,
    skip_bos: bool,
    subject: ModelSpec,
    opponent: ModelSpec,
) -> None:
    """One subject sweeps A -> B -> C -> C_placebo; the opponent's policy and
    condition (always A) never move. Isolates whether behavior changes are a
    reaction to what the subject alone was told, with nothing on the
    opponent's side confounding the comparison -- see FUTURE_DESIGNS.md.
    """
    effective_ipd_reps = ipd_reps if ipd_reps is not None else reps

    print("=== Stage 1: subject condition A (also seeds fixed opponent's profile) ===", file=sys.stderr)
    if not skip_bos:
        _run_asymmetric_stage(
            Game.BATTLE_OF_THE_SEXES, bos_rounds, reps, results_dir, Condition.ANONYMOUS, subject, opponent,
            None, True, 5000,
        )
    _run_asymmetric_stage(
        Game.IPD, ipd_max_rounds, effective_ipd_reps, results_dir, Condition.ANONYMOUS, subject, opponent,
        None, False, 5100,
    )

    rounds = read_jsonl(results_dir / "rounds.jsonl")
    bos_profile: str | None = None
    bos_placebo: str | None = None
    if not skip_bos:
        bos_profile, bos_placebo = build_real_and_placebo_profiles(
            rounds, opponent.model_id, Game.BATTLE_OF_THE_SEXES, opponent.self_label
        )
    ipd_profile, ipd_placebo = build_real_and_placebo_profiles(
        rounds, opponent.model_id, Game.IPD, opponent.self_label
    )

    print("=== Stage 2: subject condition B (identity only) ===", file=sys.stderr)
    if not skip_bos:
        _run_asymmetric_stage(
            Game.BATTLE_OF_THE_SEXES, bos_rounds, reps, results_dir, Condition.IDENTITY, subject, opponent,
            None, True, 5200,
        )
    _run_asymmetric_stage(
        Game.IPD, ipd_max_rounds, effective_ipd_reps, results_dir, Condition.IDENTITY, subject, opponent,
        None, False, 5300,
    )

    print("=== Stage 3: subject condition C / C_placebo (informed about fixed opponent) ===", file=sys.stderr)
    if not skip_bos:
        _run_asymmetric_stage(
            Game.BATTLE_OF_THE_SEXES, bos_rounds, reps, results_dir, Condition.INFORMED, subject, opponent,
            bos_profile, False, 5400,
        )
        _run_asymmetric_stage(
            Game.BATTLE_OF_THE_SEXES, bos_rounds, reps, results_dir, Condition.INFORMED_PLACEBO, subject,
            opponent, bos_placebo, False, 5500,
        )
    _run_asymmetric_stage(
        Game.IPD, ipd_max_rounds, effective_ipd_reps, results_dir, Condition.INFORMED, subject, opponent,
        ipd_profile, False, 5600,
    )
    _run_asymmetric_stage(
        Game.IPD, ipd_max_rounds, effective_ipd_reps, results_dir, Condition.INFORMED_PLACEBO, subject,
        opponent, ipd_placebo, False, 5700,
    )

    print(f"Fixed-opponent run complete. Output in {results_dir}", file=sys.stderr)


def run_pilot(
    results_dir: Path,
    reps: int,
    bos_rounds: int,
    ipd_max_rounds: int,
    ipd_reps: int | None,
    skip_bos: bool,
    models: list[ModelSpec],
    seatings: list[tuple[ModelSpec, ModelSpec]],
) -> None:
    effective_ipd_reps = ipd_reps if ipd_reps is not None else reps

    print("=== Stage 1: condition A (seeds behavioral profiles) ===", file=sys.stderr)
    if not skip_bos:
        _run_condition_a(Game.BATTLE_OF_THE_SEXES, bos_rounds, reps, results_dir, True, seatings)
    _run_condition_a(Game.IPD, ipd_max_rounds, effective_ipd_reps, results_dir, False, seatings)

    rounds = read_jsonl(results_dir / "rounds.jsonl")
    bos_profiles: dict[str, str] = {}
    bos_placebo: dict[str, str] = {}
    ipd_profiles: dict[str, str] = {}
    ipd_placebo: dict[str, str] = {}
    for model in models:
        if not skip_bos:
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

    if not skip_bos:
        print("=== Stage 2: condition B (identity, BoS placement sweep) ===", file=sys.stderr)
        _run_condition_b(
            Game.BATTLE_OF_THE_SEXES,
            bos_rounds,
            reps,
            results_dir,
            [Placement.SYSTEM, Placement.USER, Placement.INLINE],
            True,
            seatings,
        )
    print("=== Stage 2b: condition B (identity, IPD) ===", file=sys.stderr)
    _run_condition_b(
        Game.IPD, ipd_max_rounds, effective_ipd_reps, results_dir, [Placement.USER], False, seatings
    )

    print("=== Stage 3: condition C / C_placebo (informed) ===", file=sys.stderr)
    if not skip_bos:
        _run_condition_c(
            Game.BATTLE_OF_THE_SEXES, bos_rounds, reps, results_dir, Condition.INFORMED, bos_profiles, seatings
        )
        _run_condition_c(
            Game.BATTLE_OF_THE_SEXES,
            bos_rounds,
            reps,
            results_dir,
            Condition.INFORMED_PLACEBO,
            bos_placebo,
            seatings,
        )
    _run_condition_c(
        Game.IPD, ipd_max_rounds, effective_ipd_reps, results_dir, Condition.INFORMED, ipd_profiles, seatings
    )
    _run_condition_c(
        Game.IPD,
        ipd_max_rounds,
        effective_ipd_reps,
        results_dir,
        Condition.INFORMED_PLACEBO,
        ipd_placebo,
        seatings,
    )

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
    pilot_parser.add_argument("--ipd-reps", type=int, default=None)
    pilot_parser.add_argument("--skip-bos", action="store_true")

    identity_fixed_parser = subparsers.add_parser("identity-fixed")
    identity_fixed_parser.add_argument("--out", type=Path, default=Path("results/identity_fixed"))
    identity_fixed_parser.add_argument("--reps", type=int, default=5)
    identity_fixed_parser.add_argument("--bos-rounds", type=int, default=10)

    self_play_parser = subparsers.add_parser("self-play")
    self_play_parser.add_argument("--model", choices=list(SELF_PLAY_MODELS), required=True)
    self_play_parser.add_argument("--out", type=Path, default=Path("results/self_play"))
    self_play_parser.add_argument("--reps", type=int, default=3)
    self_play_parser.add_argument("--bos-rounds", type=int, default=20)
    self_play_parser.add_argument("--ipd-max-rounds", type=int, default=20)
    self_play_parser.add_argument("--ipd-reps", type=int, default=None)
    self_play_parser.add_argument("--skip-bos", action="store_true")

    fixed_opponent_parser = subparsers.add_parser("fixed-opponent")
    fixed_opponent_parser.add_argument("--subject", choices=list(SELF_PLAY_MODELS), required=True)
    fixed_opponent_parser.add_argument("--opponent", choices=list(SELF_PLAY_MODELS), required=True)
    fixed_opponent_parser.add_argument("--out", type=Path, default=Path("results/fixed_opponent"))
    fixed_opponent_parser.add_argument("--reps", type=int, default=3)
    fixed_opponent_parser.add_argument("--bos-rounds", type=int, default=20)
    fixed_opponent_parser.add_argument("--ipd-max-rounds", type=int, default=20)
    fixed_opponent_parser.add_argument("--ipd-reps", type=int, default=None)
    fixed_opponent_parser.add_argument("--skip-bos", action="store_true")

    args = parser.parse_args()

    if args.mode == "smoke":
        run_smoke(args.out)
    elif args.mode == "identity-fixed":
        run_identity_fixed(args.out, args.reps, args.bos_rounds)
    elif args.mode == "fixed-opponent":
        if args.subject == args.opponent:
            raise ValueError("--subject and --opponent must be different models")
        run_fixed_opponent(
            args.out,
            args.reps,
            args.bos_rounds,
            args.ipd_max_rounds,
            args.ipd_reps,
            args.skip_bos,
            SELF_PLAY_MODELS[args.subject],
            SELF_PLAY_MODELS[args.opponent],
        )
    elif args.mode == "self-play":
        model = SELF_PLAY_MODELS[args.model]
        run_pilot(
            args.out,
            args.reps,
            args.bos_rounds,
            args.ipd_max_rounds,
            args.ipd_reps,
            args.skip_bos,
            [model],
            [(model, model)],
        )
    else:
        run_pilot(
            args.out,
            args.reps,
            args.bos_rounds,
            args.ipd_max_rounds,
            args.ipd_reps,
            args.skip_bos,
            MODELS,
            _seatings(),
        )


if __name__ == "__main__":
    main()
