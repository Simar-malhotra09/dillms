# exp

We're testing whether telling an AI agent *who* it's playing against
changes how it plays a game.

Two AI models play repeated simple games against each other (like
picking cooperate-or-defect, over and over). We vary only one thing
between runs: how much each agent is told about its opponent — nothing,
just a name, or a name plus a real track record of how that opponent has
played before. If the agent plays differently depending on what it was
told, that tells us AI agents can be steered just by giving them a
story about who they're dealing with.

Status: the code works, but no real experiment has been run yet — see
`STATUS.md` for what's built and what's next, and `WHAT_THIS_IS.md` for
the full design.
