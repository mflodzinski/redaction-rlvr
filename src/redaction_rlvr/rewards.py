from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Any


MASK_TOKEN = "[PII]"
MASKED_OUTPUT_FIELD = "masked_output"

OPEN_TAG = f"<{MASKED_OUTPUT_FIELD}>"
CLOSE_TAG = f"</{MASKED_OUTPUT_FIELD}>"
MASKED_OUTPUT_RE = re.compile(
    rf"<{MASKED_OUTPUT_FIELD}>\s*(.*?)\s*</{MASKED_OUTPUT_FIELD}>",
    flags=re.DOTALL,
)


def extract_masked_output(completion: str) -> str:
    match = MASKED_OUTPUT_RE.search(completion)
    if not match:
        return ""
    return match.group(1).strip()


def mask_count(text: str) -> int:
    return text.count(MASK_TOKEN)


def output_format_reward(completion: str) -> float:
    text = completion.strip()
    score = 0.0
    if OPEN_TAG in text and CLOSE_TAG in text:
        score += 0.4
    if extract_masked_output(text):
        score += 0.2
    if text.startswith(OPEN_TAG):
        score += 0.2
    if text.endswith(CLOSE_TAG):
        score += 0.2
    return score


def redaction_accuracy_reward(completion: str, answer: str) -> float:
    return 1.0 if extract_masked_output(completion).strip() == answer.strip() else 0.0


def span_count_reward(completion: str, info: dict[str, Any] | None = None, answer: str | None = None) -> float:
    parsed = extract_masked_output(completion)
    if answer is not None:
        expected_count = mask_count(answer)
    else:
        expected_count = int((info or {}).get("pii_count", 0))
    return 1.0 if mask_count(parsed) == expected_count else 0.0


def _words(text: str) -> list[str]:
    table = str.maketrans("", "", string.punctuation)
    return [word.translate(table).lower() for word in text.split() if word.translate(table)]


def _non_pii_words(answer: str) -> set[str]:
    return {word for word in _words(answer) if word != "pii"}


def preservation_reward(completion: str, answer: str) -> float:
    """Reward preserving non-sensitive content while avoiding excess masks."""
    parsed = extract_masked_output(completion)
    expected_words = _non_pii_words(answer)
    if not expected_words:
        content_score = 1.0
    else:
        produced_words = set(_words(parsed))
        content_score = len(expected_words & produced_words) / len(expected_words)

    extra_masks = max(0, mask_count(parsed) - mask_count(answer))
    return max(0.0, content_score - 0.1 * extra_masks)


@dataclass(frozen=True)
class RewardBreakdown:
    redaction_accuracy: float
    output_format: float
    span_count: float
    preservation: float

    @property
    def total(self) -> float:
        return (
            1.0 * self.redaction_accuracy
            + 0.1 * self.output_format
            + 0.5 * self.span_count
            + 0.3 * self.preservation
        )


@dataclass(frozen=True)
class RedactionReward:
    """Composite reward for the open-pii-masking-en-us-30k task."""

    def breakdown(
        self,
        completion: str,
        answer: str,
        info: dict[str, Any] | None = None,
    ) -> RewardBreakdown:
        return RewardBreakdown(
            redaction_accuracy=redaction_accuracy_reward(completion, answer),
            output_format=output_format_reward(completion),
            span_count=span_count_reward(completion, info=info, answer=answer),
            preservation=preservation_reward(completion, answer),
        )

    def __call__(
        self,
        completion: str,
        answer: str,
        info: dict[str, Any] | None = None,
    ) -> float:
        return self.breakdown(completion, answer, info=info).total


def format_completion(masked_text: str) -> str:
    return f"{OPEN_TAG}\n{masked_text.strip()}\n{CLOSE_TAG}"
