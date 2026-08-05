# exp — identity-exploitation pilot

Studies whether telling an LLM agent identity/profile information about its
opponent (vs. staying anonymous) changes its behavior in repeated
coordination games (Battle of the Sexes, Iterated Prisoner's Dilemma).

## Flow

`main.py` (CLI: `smoke`, `pilot`) → `pipeline.run_match` orchestrates one
match: optional pre-game `discovery` (BoS only, conditions A/B — 2x2 short
message exchange, judged by `judge.run_judge` for identity-probing/threats/
convention-proposals), then N rounds via `llm_client.get_round_decision`.
`prompts.py` builds the rules text (byte-identical across conditions on
purpose) and the condition block (varies). `profiling.py` turns condition-A
logs into a behavioral profile string (+ an inverted placebo) used in
condition C. All output is append-only JSONL via `storage.py`.

Conditions: **A** anonymous, **B** identity (self-label shown), **C**
informed (real behavioral profile of opponent shown), **C_placebo** (same
profile shape, inverted numbers — controls for "any narrative" vs real info).

## What's shared vs. private between the two agents

- Shared: discovery transcript, each round's `history_text` (both players'
  *actions* and *payoffs* only), and in condition C the opponent's
  behavioral-profile string.
- Private (logged for analysis, never fed to the opponent):
  `reasoning_text`, `raw_response`.

## Per-model config gotcha (llm_client.py)

`ModelSpec` has explicit `max_completion_tokens` and `include_reasoning`
per model — not a shared global. Reason: Kimi's hidden reasoning
(`reasoning_effort`) can exhaust a shared token cap before emitting visible
JSON, especially once discovery gives it real strategic content to reason
about — this previously crashed matches with `RuntimeError: ... No JSON
object found`. Current settings: DeepSeek `max_completion_tokens=4000`,
`include_reasoning=True`; Kimi `max_completion_tokens=8000`,
`include_reasoning=False` (its round-decision schema drops the visible
`"reasoning"` field, so `RoundRecord.reasoning_text` is `""` for Kimi rows).
To restore Kimi's visible reasoning trace, flip `KIMI.include_reasoning`
back to `True` — everything else adapts automatically
(`get_round_decision` branches on this field; no other code needs to change).

## Running experiments

- `results/` is gitignored and JSONL files are **append-only** — rerunning
  writes into the same files unless you point at a fresh directory. Use a
  new `results/<name>/` per experiment variant so runs don't mix.
- Real API calls cost money and can take several minutes per match
  (reasoning models, plus up to `MAX_JSON_ATTEMPTS=3` retries per call). A
  quiet terminal isn't necessarily hung — check `results/<name>/comms.jsonl`
  / `rounds.jsonl` for incremental progress.
- One-off variants (e.g. custom self-labels instead of the real model
  identities) should use `dataclasses.replace(DEEPSEEK, self_label=...)`
  rather than editing the `DEEPSEEK`/`KIMI` constants, since those constants
  are reused across the whole pilot.
