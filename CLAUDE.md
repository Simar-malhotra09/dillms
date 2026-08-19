# exp — briefing (paste this into a fresh Claude instance for context)

## What this is

Testing whether telling an LLM agent identity/behavioral information about
its opponent changes how it plays repeated coordination games (Battle of
the Sexes, Iterated Prisoner's Dilemma). Four conditions vary *only* how
much the agent knows about who it's playing: **A** anonymous, **B** bare
identity label, **C** identity + real behavioral profile from the
opponent's own past A-condition games, **C_placebo** same profile shape
with inverted numbers (controls for "any narrative" vs. real info).

## Current state (as of 2026-08-14)

- Pipeline is built and API-verified end-to-end: both games, all four
  conditions, an optional pre-game "discovery" chat phase (BoS only,
  judged by a third model for identity-probing/threats/convention
  proposals), and a "fixed-opponent" mode that sweeps one subject through
  A→B→C→C_placebo while the opponent stays fixed at A — the cleaner design
  for isolating "does telling *this* agent about its opponent change *its*
  play."
- **Zero real pilot data.** Every `results/` directory is a smoke test —
  1 to 4 matches, a few rounds each. The actual research question has no
  evidence either way yet.
- Token/cost instrumentation exists (`reasoning_tokens` captured from real
  provider usage) and a `cost_model.py` + `pricing.py` pair can project
  cost analytically — but has never been pointed at a real budget
  decision, only self-tested.
- DeepSeek billing was suspended (429 insufficient balance) mid-session;
  now topped up ($9.78 confirmed via `/user/balance`). Separately found
  and flagged (not yet removed by user) a stale `DEEPSEEK_API_KEY` in
  `~/.zshrc` that was shadowing the correct key in `.env` — caused an
  `invalid_request_error` that looked like a billing problem but wasn't.
- Goal in progress: a one-pager to pitch a dept head for funding. **Not
  ready** — see below.

## Open questions / concerns

From planning notes plus review of the current code:

1. **Prompt design** — system vs. user placement, how extensive, what are
   the limits. A placement sweep exists for condition B in BoS
   (system/user/inline-metadata) but hasn't been analyzed for effect.
2. **Discovery phase** — constraints (word/time caps) are currently picked,
   not literature-grounded; may introduce its own bias. Post-analysis is
   currently LLM-as-judge only; sentiment analysis as an alternative/
   complement hasn't been explored.
3. **Game + history** — untested hypothesis that attention over round
   history is U-shaped (more weight on first/last rounds); context length
   is a real constraint. Open question: how to correctly attribute an
   observed behavioral shift to its actual cause rather than a confound.
4. **Authority** — does *who* says something (not just *what*) change
   behavior; needs a literature pass on communication constraints in
   canonical game-theory experiments. Value case: using authority framing
   to direct/coordinate multi-agent systems in organizational settings.
5. **Discovery content** — what kinds of questions models spontaneously
   ask about their opponent during discovery is flagged as interesting
   and unexplored.
6. **Cost is the primary scope constraint.** Plan: a graph with cost on y,
   token count on x, binned by rounds-per-match, one line per model,
   pricing pulled from the web for 5-6 frontier models plus DeepSeek/Kimi.
   Caveats surfaced in review, not yet resolved:
   - Input-token cost is analytically real (deterministic from the actual
     prompt-builder code, no API calls needed).
   - Output-token cost is only empirically known for DeepSeek/Kimi, and
     even that's thin (2-24 logged rounds). Every other model's line would
     be an *assumed* completion-token count, not measured — the chart must
     visually distinguish measured vs. assumed or it overstates confidence.
   - Axis should read "rounds per match," not "iterations" — IPD's round
     count is nominal in the cost tool (continuation probability isn't
     modeled).
   - Discovery cost is a fixed one-time add-on per match, not part of the
     per-round slope — don't blend it into the curve.
7. **No statistical significance testing exists anywhere in the repo** —
   no scipy/stats dependency, no p-value/t-test code. `profiling.py`
   computes rates (BoS: round-1 preference rate, switch-rate after two
   miscoordinations; IPD: cooperation rate, retaliation rate) but nothing
   currently tests whether an A-vs-B difference is signal or noise.
8. **Condition C is circular as built** — the profile fed into C is built
   from the same condition-A logs that would later be used to evaluate
   whether C changed behavior. No train/eval split. Needs fixing before
   any C/C_placebo result can be trusted.
9. **Recommended smallest first real experiment** (to replace "a list of
   open questions" with one actual tested result): IPD only (no discovery
   phase exists for IPD, sidesteps #2), DeepSeek vs. Kimi, fixed-opponent
   mode (already implemented and live-API-verified once), condition A vs.
   B only (skips C's circularity problem in #8), using the already-coded
   IPD metrics (cooperation rate, retaliation rate). Remaining open
   decision: match count for adequate statistical power — depends on
   writing the significance test from #7; its cost is then read off the
   graph from #6 once N is chosen.

## What "done" looks like for the pending one-pager

Not fundable-pitch-ready today: zero behavioral results, zero real cost
projection ever actually run (the tool exists, unused), and the open
methodology questions above are unresolved. The one-pager should either
(a) honestly frame itself as a resourcing ask backed by de-risked infra
plus a real cost curve, or (b) wait until the #9 starting experiment
produces one real, attributed result to lead with.
