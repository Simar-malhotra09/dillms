# dilemma_experiment

Standalone, throwaway-quality script for the chief-minister/AI-infrastructure
triage dilemma. Not part of the `src/exp` pipeline.

## Files

- `prompts.py` -- the ordered list of turn prompts (`TURNS`). Extend this
  list to go from 5 turns to any other count; nothing else needs to change.
- `run_conversation.py` -- reusable runner. `run_conversation(turns, model,
  max_reasoning_words)` sends turns one at a time, feeding back prior user
  prompts + prior *answers* (never `reasoning_content`) as history, per
  DeepSeek's multi-turn convention. `write_markdown(...)` dumps the full
  transcript (prompt/reasoning/answer/token counts per turn) to a file.
- `main.py` -- entry point. Edit `MODEL` / `MAX_REASONING_WORDS` here. Run
  with `uv run python dilemma_experiment/main.py` from the repo root.
- `results/run_<UTC timestamp>.md` -- one file per run.

## The reasoning word cap

`MAX_REASONING_WORDS` appends a soft instruction ("keep your internal
reasoning to roughly N words") to each turn's prompt. There is no hard API
parameter for this -- DeepSeek only exposes `reasoning_effort`
(low/medium/high), which is coarse and not word-counted.

In practice (see `results/run_20260815T071912Z.md`), the model mostly
ignores the soft cap -- reasoning blocks ran 500-1200+ words despite a
150-word target, and it just narrates about the instruction rather than
complying. Treat this as advisory, not enforced. Set `MAX_REASONING_WORDS =
None` to disable it entirely.

Caveat for this specific experiment: even if the cap worked, capping
reasoning length risks cutting exactly the kind of hedge/caveat (e.g.
self-interest introspection in turn 5) that this experiment is trying to
observe, rather than just trimming filler. Compare capped vs. uncapped runs
before drawing conclusions from a capped one.
