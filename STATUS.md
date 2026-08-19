# Status (2026-08-14)

## The higher-order goal

This pilot exists to answer one question: **if you tell an LLM agent who
it's playing against, does it play differently?** Four conditions
(A anonymous → B identity-labeled → C real behavioral profile →
C_placebo same-shape-but-fake profile) isolate *how much the agent knows
about its opponent* as the only thing that varies, so any behavioral
shift is attributable to that alone. See `WHAT_THIS_IS.md` for the full
design rationale and `CLAUDE.md` for the code-level flow.

**Where we actually are on that goal: pre-data.** Nothing below is a
result about identity-exploitation behavior yet — everything so far is
infrastructure and correctness verification. No real pilot has been run
to completion; `results/` contains only tiny smoke/verification runs
(1-4 matches each, `smoke*`, `selfplay*`), not experiment data worth
analyzing. The interesting question — does B/C/C_placebo shift play
relative to A — is still unasked of the data.

## What's built and working

- Core pipeline: discovery → N rounds → JSONL logging, for both games
  (Battle of the Sexes, Iterated Prisoner's Dilemma), across all four
  conditions, with a BoS placement sweep (system/user/inline).
- `main.py` CLI: `smoke`, `pilot`, `identity-fixed` (loaded-label variant,
  e.g. "Gandhi" vs "Hitler" opponent labels), `self-play` (one model vs
  itself, removes cross-model confound).
- Token/cost accounting: `reasoning_tokens` now captured from provider
  usage data end-to-end (`llm_client.py`, `judge.py`, `types.py`); new
  `cost_model.py` (analytic pre-run cost projection) and `pricing.py`
  (per-model $/M token table) — not yet used to actually budget a real
  pilot run, just built and self-tested.
- Kimi mode split: `KIMI` (real hidden reasoning, ~1200+ reasoning
  tokens/call, live-tested 2026-08-14) vs `KIMI_INSTANT` (thinking
  disabled, temperature pinned to 0.6, ~1 reasoning token/call) — cheap
  self-play option when deliberation depth doesn't need to be measured.
- **Just implemented this session**: the asymmetric subject/fixed-opponent
  design from `FUTURE_DESIGNS.md` — `MatchConfig`/`MatchRecord` split
  `condition` into `p1_condition`/`p2_condition`, `RoundRecord.condition`
  now records each actor's own condition (previously always identical),
  and a new `fixed-opponent` CLI mode sweeps one subject through
  A→B→C→C_placebo while a fixed opponent stays at condition A the whole
  time — better isolates "does telling *this* agent about its opponent
  change *its* play" from the previous symmetric A/A, B/B, C/C design,
  which confounded that with "the opponent also being told." Verified
  against the real APIs (IPD, 1 rep) — `matches.jsonl`/`rounds.jsonl`
  confirmed the subject moves to B while the fixed opponent stays A on
  both the match record and each row's own condition field.

## What's blocking / not done

- **DeepSeek account is suspended for insufficient balance** (429
  `exceeded_current_quota_error`, hit live 2026-08-14 mid-verification).
  Needs a top-up before any DeepSeek-involving run — which is most of
  them, since DeepSeek is one of the two primary models — can proceed.
- No real `pilot` run has been taken to completion, so there are no
  numbers yet on round-1 preference rates, switch rates, cooperation
  rates, or condition-driven shifts in any of them.
- The new `fixed-opponent` mode has only been smoke-tested (partial IPD
  run, 1 rep, stopped by the billing error above) — never run at
  pilot scale (multiple reps, both games, full A→B→C→C_placebo sweep).
- `cost_model.py` hasn't been used to actually size a real run's budget
  yet — it's built and unit-verified but not yet load-bearing for a
  decision.
- `FUTURE_DESIGNS.md` still says the asymmetric design is "not
  implemented yet" — now stale, since this session implemented it. Not
  updated yet pending confirmation of what you want done with that file
  (mark done / delete / fold into `WHAT_THIS_IS.md`).
- Everything above is uncommitted (`git status`: 8 modified + 5 untracked
  files, all sitting on `main` since `4d7be4e`). Nothing has been
  committed this session or last.

## Suggested next step

Recharge the DeepSeek account, then either (a) run `pilot` at real scale
to get the first actual A/B/C/C_placebo numbers for the symmetric design,
or (b) run `fixed-opponent` at real scale to get the better-isolated
subject-only numbers — (b) is arguably the more decision-relevant run
now that it exists, since it removes the "opponent's treatment also
moved" confound the symmetric pilot has always had.
