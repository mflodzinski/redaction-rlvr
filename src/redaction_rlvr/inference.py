from __future__ import annotations

from redaction_rlvr.rewards import CLOSE_TAG, MASK_TOKEN, OPEN_TAG


SYSTEM_PROMPT = f"""Replace all personally identifiable information (PII) in the text with {MASK_TOKEN} tags.
PII includes names, dates, phone numbers, SSNs, account numbers, addresses, email addresses, and other identifying information.

Return only the masked text wrapped in masked_output XML tags:
{OPEN_TAG}
masked text here
{CLOSE_TAG}"""


def build_redaction_prompt(question: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nInput:\n{question}\n\nOutput:"
