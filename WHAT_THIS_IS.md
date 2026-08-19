# What this code does

This is an experiment testing a simple question: **if you tell an AI agent
who it's playing against, does it play differently?**

Two AI models (DeepSeek and Kimi) are paired up and made to play repeated
two-player games against each other — the kind of games where the "right"
move depends on what you expect the other player to do. Across many
repeated matches, the only thing that changes between conditions is
*how much the agent is told about its opponent*. Everything else — the
rules, the payoffs, the number of rounds — is held identical. Any
difference in how the agent plays can then be attributed to that one
variable: what it knows, or thinks it knows, about who it's up against.

## The two games

- **Battle of the Sexes** — a coordination game. Both players want to land
  on the *same* choice (RED or RED, BLUE or BLUE), but each player has a
  different preferred color, and picking your own preferred color when it
  matches is worth more than picking the other player's. Picking different
  colors from each other is worthless to both. So there's a real tension:
  push for your own preference and risk miscoordinating, or defer to the
  other side and guarantee a smaller but safe payoff.
- **Iterated Prisoner's Dilemma** — the classic cooperate/defect game,
  played repeatedly with a random chance of ending after each round (so
  neither player knows exactly when the last round is, which matters for
  strategy).

## What varies: how much the agent knows about its opponent

Every match is run under one of four conditions:

- **A — Anonymous.** The agent is just told "you're playing against another
  AI agent," nothing else.
- **B — Identity.** The agent is told which model it is *and* which model
  its opponent is (e.g. "You are DeepSeek. Your opponent is Kimi K2.6").
  No track record, just a label.
- **C — Informed.** Same identity disclosure, plus a real behavioral
  profile of the opponent built from that opponent's own past games in
  condition A — e.g. "in prior matches, Kimi picked its own preferred
  option in round 1 in 80% of cases." This is genuine information the
  agent could use to anticipate the opponent's move.
- **C_placebo — Informed but fake.** Structurally identical profile text
  to condition C, but with the numbers inverted (so a real 80% becomes a
  fake 20%). This isolates whether behavior changes because the agent has
  *real predictive information* versus just because it was handed *any*
  narrative about its opponent — a control for the placebo effect of
  believing you know something.

Condition A's own game logs are what generate the profiles fed into
condition C — the pipeline plays A first, computes each model's actual
round-1 preference rate and how often it switched strategy after
miscoordinating, and only then runs B and C using those real numbers.

There's also a placement axis for condition B in Battle of the Sexes:
the same identity information is delivered either in the system prompt,
the user prompt, or buried as a metadata field inside a JSON blob
describing game state — testing whether *where* the information sits in
the prompt affects how much it's weighted, independent of its content.

A separate "identity-fixed" mode holds one agent's policy completely
constant and only changes the *label* given to its opponent (e.g. telling
it the opponent is "Gandhi" vs. "Hitler") — same underlying model, same
behavior, different name — to see whether the agent's own play shifts
purely as a reaction to a loaded label rather than any real information
about the opponent's strategy.

## How a single match runs

1. **(Optional) Discovery phase.** Before the game starts, the two agents
   get a very short, tightly capped chat (a few messages, 60 words each)
   where they can say whatever they want to each other — negotiate, ask
   who the other is, propose a convention like "let's alternate," threaten
   retaliation. This is deliberately tiny so it can't dominate the
   experiment; it exists to see whether agents *volunteer* identity-probing
   or strategic talk on their own. This transcript is then scored by a
   third model acting as judge, which labels it for things like "did
   either side ask who the other one is" or "did either side propose a
   coordination rule."
2. **N rounds of play.** Each round, both agents independently choose an
   action, told the same rules and the same running history of past
   rounds and scores. The only thing that differs between the two agents'
   prompts is their own condition-specific information about the
   opponent — the rules text itself is byte-for-byte identical across
   conditions, specifically so it can't be a confound.
3. Every round, match, discovery message, and judge label is appended to
   a JSONL log. Nothing is overwritten — rerunning the same output
   directory just appends more rows, so each experiment variant gets its
   own results folder.

## What's shared vs. kept private

The two agents only ever see: the discovery chat (if run), the public
history of actions and scores from prior rounds, and — in condition C —
the opponent's behavioral profile text. Each agent's private reasoning
(when the model is configured to expose it) and full raw API response are
logged for later analysis but never shown to the opponent — that's the
experimenter's view into *why* a model made a choice, not something either
player can see or exploit in-game.

## The underlying hypothesis being tested

Whether an AI agent, when it's given a legible identity for its opponent —
especially one backed by real behavioral history rather than just a name —
starts to treat the interaction less like an anonymous game-theoretic
exchange and more like a relationship with a specific, exploitable or
trustworthy counterpart. The anonymous condition is the baseline; identity
alone (B) tests whether a bare label is enough to shift behavior; the real
vs. placebo profile split (C vs. C_placebo) tests whether it's the
*information content* doing the work or just the presence of *any* story
about the other player; and the placement sweep and fixed-label variant
test whether the effect is about where information appears and how
strongly a loaded label alone (independent of real behavior) can move an
agent's play.
