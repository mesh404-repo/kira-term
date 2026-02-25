"""
KIRA-style LLM response parsing for message management.

Extracts Analysis and Plan from assistant text for structured message storage
(like harbor/KIRA: message_content = "Analysis: ...\\nPlan: ..." when present).
Does not depend on harbor; used for add-message logic only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ParsedResponse:
    """Parsed assistant response content (KIRA-style)."""
    analysis: str = ""
    plan: str = ""
    raw_content: str = ""
    has_structure: bool = False


def parse_response_content(response_text: str) -> ParsedResponse:
    """
    Extract Analysis and Plan from assistant response text (KIRA-style).

    Looks for common patterns: "Analysis:", "**Analysis**", "## Analysis",
    "Plan:", "**Plan**", "## Plan", etc. Used to build stored message content
    as "Analysis: ...\\nPlan: ..." when both are present.

    Args:
        response_text: Full assistant response string.

    Returns:
        ParsedResponse with analysis, plan, raw_content, and has_structure flag.
    """
    if not (response_text or response_text.strip()):
        return ParsedResponse(raw_content=response_text or "")

    text = response_text.strip()
    analysis = ""
    plan = ""

    # Patterns for Analysis section (case-insensitive; stop at Plan or end)
    analysis_patterns = [
        r"(?:^|\n)\s*#+\s*Analysis\s*:?\s*\n(.*?)(?=\n\s*#+\s*Plan|\n\s*Plan\s*:|\n\s*\*\*Plan\*\*|\Z)",
        r"(?:^|\n)\s*\*\*Analysis\*\*\s*:?\s*\n(.*?)(?=\n\s*#+\s*Plan|\n\s*Plan\s*:|\n\s*\*\*Plan\*\*|\Z)",
        r"(?:^|\n)\s*Analysis\s*:?\s*\n(.*?)(?=\n\s*#+\s*Plan|\n\s*Plan\s*:|\n\s*\*\*Plan\*\*|\Z)",
    ]
    for pat in analysis_patterns:
        m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
        if m:
            analysis = m.group(1).strip()
            break

    # Patterns for Plan section (stop at next ## or **Header or end)
    plan_patterns = [
        r"(?:^|\n)\s*#+\s*Plan\s*:?\s*\n(.*?)(?=\n\s*#+\s*[A-Za-z]|\n\s*\*\*[A-Za-z]|\Z)",
        r"(?:^|\n)\s*\*\*Plan\*\*\s*:?\s*\n(.*?)(?=\n\s*#+|\n\s*\*\*[A-Za-z]|\Z)",
        r"(?:^|\n)\s*Plan\s*:?\s*\n(.*?)(?=\n\s*#+|\n\s*\*\*[A-Za-z]|\Z)",
    ]
    for pat in plan_patterns:
        m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
        if m:
            plan = m.group(1).strip()
            break

    has_structure = bool(analysis or plan)
    return ParsedResponse(
        analysis=analysis,
        plan=plan,
        raw_content=text,
        has_structure=has_structure,
    )


def get_message_content_for_storage(parsed: ParsedResponse) -> str:
    """
    Build stored message content from parsed response (KIRA-style).

    When analysis/plan are present, format as "Analysis: ...\\nPlan: ...".
    Otherwise return raw content.

    Args:
        parsed: Result of parse_response_content().

    Returns:
        String to use as assistant message content in history.
    """
    if not parsed.has_structure or (not parsed.analysis and not parsed.plan):
        return parsed.raw_content
    parts = []
    if parsed.analysis:
        parts.append(f"Analysis: {parsed.analysis}")
    if parsed.plan:
        parts.append(f"Plan: {parsed.plan}")
    return "\n".join(parts)
