"""Run the triage dilemma conversation end-to-end, write results/run_<ts>.md.

Usage (from repo root): uv run python dilemma_experiment/main.py

To change turn count, edit TURNS in prompts.py. To change the model or the
reasoning word cap, edit the constants below.
"""

import os
from datetime import datetime, timezone

from prompts import TURNS
from run_conversation import run_conversation, write_markdown

MODEL = "deepseek-v4-pro"
MAX_REASONING_WORDS: int | None = 150  # set to None to disable the cap

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    records = run_conversation(TURNS, MODEL, MAX_REASONING_WORDS)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = os.path.join(RESULTS_DIR, f"run_{timestamp}.md")
    write_markdown(records, output_path, MODEL, MAX_REASONING_WORDS)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
