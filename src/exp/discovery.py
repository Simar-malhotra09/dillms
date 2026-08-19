"""Tiny pre-game discovery phase: 2 exchanges of <=60 words per side.

Kept deliberately small (per pilot design decision) so it cannot dominate
or distort the game signal -- only used for Battle of the Sexes, conditions
A and B.
"""

from __future__ import annotations

from exp.llm_client import get_json_reply
from exp.prompts import build_condition_block, build_rules_text, build_transcript_text
from exp.types import CommsRecord, MatchConfig

MAX_WORDS = 60


def _validate_message(parsed: dict) -> None:
    if "message" not in parsed or not isinstance(parsed["message"], str):
        raise ValueError('expected {"message": "..."}')
    if len(parsed["message"].split()) > MAX_WORDS:
        raise ValueError(f"message exceeds {MAX_WORDS} words")


def run_discovery(config: MatchConfig) -> list[CommsRecord]:
    p1_block = build_condition_block(config.p1_condition, config.p1, config.p2, None)
    p2_block = build_condition_block(config.p2_condition, config.p2, config.p1, None)
    p1_rules = build_rules_text(config.game, config.num_rounds, config.p1_prefers)
    p2_rules = build_rules_text(config.game, config.num_rounds, config.p2_prefers)

    transcript: list[CommsRecord] = []
    exchange_idx = 0
    for _ in range(2):
        for speaker_spec, condition_block, rules_text in [
            (config.p1, p1_block, p1_rules),
            (config.p2, p2_block, p2_rules),
        ]:
            system_content = (
                f"{rules_text}\n\n{condition_block}\n\n"
                f"Before the game starts you may exchange short messages with your "
                f"opponent. Each message must be at most {MAX_WORDS} words. Respond "
                'with only a JSON object: {"message": "..."}.'
            )
            transcript_text = build_transcript_text(transcript, speaker_spec.model_id)
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"Conversation so far:\n{transcript_text}\n\nSend your message now."},
            ]
            parsed, _, prompt_tokens, completion_tokens, reasoning_tokens, latency_ms = get_json_reply(
                speaker_spec,
                messages,
                1.0,
                _validate_message,
                '{"message": "<at most 60 words>"}',
            )
            transcript.append(
                CommsRecord(
                    match_id=config.match_id,
                    exchange_idx=exchange_idx,
                    speaker_model=speaker_spec.model_id,
                    message_text=parsed["message"],
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    reasoning_tokens=reasoning_tokens,
                    latency_ms=latency_ms,
                )
            )
            exchange_idx += 1
    return transcript
