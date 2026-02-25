"""
Parses LLM response as JSON with: analysis, plan, commands (or image_read), task_complete.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ParsedCommand:
    keystrokes: str
    duration: float


@dataclass
class ImageReadRequest:
    file_path: str
    image_read_instruction: str


@dataclass
class ParseResult:
    """Result of parsing a JSON response."""
    commands: List[ParsedCommand]
    is_task_complete: bool
    error: str
    warning: str
    analysis: str = ""
    plan: str = ""
    image_read: Optional[ImageReadRequest] = None


# Single token markers like <|tool_call_begin|> — strip so we don't treat them as content
_TOOL_CALL_MARKER = re.compile(r"<\|[^|]+\|>")


def _strip_tool_call_markers(response: str) -> str:
    """Remove tool-call/section marker tokens so the rest can be parsed as JSON or text."""
    return _TOOL_CALL_MARKER.sub(" ", response)


def _find_json_object_bounds(text: str) -> List[Tuple[int, int]]:
    """Find all top-level JSON object spans (start, end) in text. Handles strings and escapes."""
    result: List[Tuple[int, int]] = []
    i = 0
    while i < len(text):
        if text[i] != "{":
            i += 1
            continue
        start = i
        brace_count = 0
        in_string = False
        escape_next = False
        quote_char = '"'
        j = i
        while j < len(text):
            c = text[j]
            if escape_next:
                escape_next = False
                j += 1
                continue
            if c == "\\" and in_string:
                escape_next = True
                j += 1
                continue
            if in_string:
                if c == quote_char:
                    in_string = False
                j += 1
                continue
            if c == '"' or c == "'":
                in_string = True
                quote_char = c
                j += 1
                continue
            if c == "{":
                brace_count += 1
                j += 1
                continue
            if c == "}":
                brace_count -= 1
                if brace_count == 0:
                    result.append((start, j + 1))
                    i = j + 1
                    break
                j += 1
                continue
            j += 1
        else:
            i += 1
    return result


def _extract_json_content(response: str) -> Tuple[str, List[str]]:
    """Extract best JSON object from response. Strips tool-call markers, prefers object with analysis+plan."""
    warnings: List[str] = []
    normalized = _strip_tool_call_markers(response).strip()

    bounds = _find_json_object_bounds(normalized)
    if not bounds:
        # Try plain-text fallback before giving up
        return "", ["No valid JSON object found"]

    # Try each candidate; prefer one that has both "analysis" and "plan"
    best_content = ""
    best_has_required = False
    for start, end in bounds:
        candidate = normalized[start:end]
        try:
            data = json.loads(candidate)
            if not isinstance(data, dict):
                continue
            has_required = "analysis" in data and "plan" in data
            if has_required and not best_has_required:
                best_content = candidate
                best_has_required = True
            elif has_required and best_has_required:
                # Prefer last full object (model often repeats at end)
                best_content = candidate
            elif not best_content:
                best_content = candidate
        except json.JSONDecodeError:
            continue

    if not best_content:
        return "", ["No valid JSON object found"]

    # Warn on extra text
    first_start = bounds[0][0]
    last_end = bounds[-1][1]
    if first_start > 0:
        warnings.append("Extra text detected before JSON object")
    if last_end < len(normalized):
        warnings.append("Extra text detected after JSON object")

    return best_content, warnings


def _parse_plain_text_fallback(response: str) -> Optional[ParseResult]:
    """
    When no JSON is found, try to extract Analysis/Plan/Commands/task_complete from plain text.
    Handles responses like "Analysis: ... Plan: ... Commands: []" or "Plan: ... task_complete: true".
    """
    normalized = _strip_tool_call_markers(response).strip()
    if not normalized:
        return None

    analysis = ""
    plan = ""
    commands: List[ParsedCommand] = []
    task_complete = False

    # Analysis: ... (until next major section or end)
    m_analysis = re.search(r"\bAnalysis:\s*(.+?)(?=\s+Plan:|\s+Requirements|\s+Commands:|\s+task_complete:|\s*$)", normalized, re.DOTALL | re.IGNORECASE)
    if m_analysis:
        analysis = m_analysis.group(1).strip()

    # Plan: ... (until next major section or end)
    m_plan = re.search(r"\bPlan:\s*(.+?)(?=\s+Requirements|\s+Commands:|\s+task_complete:|\s*\{\s*\"|\s*$)", normalized, re.DOTALL | re.IGNORECASE)
    if m_plan:
        plan = m_plan.group(1).strip()

    # task_complete: true/false
    if re.search(r"\btask_complete\s*:\s*true\b", normalized, re.IGNORECASE):
        task_complete = True
    # Heuristic: "task is complete" / "mark task complete" in plan or text when no commands
    if not task_complete and not commands and re.search(r"\b(?:task is complete|mark task complete|task complete\.?)\b", normalized, re.IGNORECASE):
        task_complete = True

    # Try to find a JSON array for commands after "commands" or "Commands:"
    m_commands = re.search(r"\bCommands?\s*:\s*(\[\s*\{.*?\}\s*\]|\[\s*\])", normalized, re.DOTALL)
    if m_commands:
        try:
            raw = m_commands.group(1).strip()
            arr = json.loads(raw)
            if isinstance(arr, list):
                for cmd in arr:
                    if isinstance(cmd, dict) and "keystrokes" in cmd:
                        duration = float(cmd.get("duration", 1.0)) if isinstance(cmd.get("duration"), (int, float)) else 1.0
                        duration = min(max(0.1, duration), 60.0)
                        commands.append(ParsedCommand(keystrokes=str(cmd["keystrokes"]), duration=duration))
        except json.JSONDecodeError:
            pass

    # If we have at least analysis or plan, or commands/task_complete, return a result
    if analysis or plan or commands or task_complete:
        return ParseResult(
            commands=commands,
            is_task_complete=task_complete,
            error="",
            warning="Response was plain text; parsed with fallback. Please respond with valid JSON next time.",
            analysis=analysis,
            plan=plan,
        )
    return None


def _parse_commands(commands_data: list, warnings: List[str]) -> Tuple[List[ParsedCommand], str]:
    """Parse commands array into ParsedCommand objects."""
    commands: List[ParsedCommand] = []
    for i, cmd_data in enumerate(commands_data):
        if not isinstance(cmd_data, dict):
            return [], f"Command {i + 1} must be an object"
        if "keystrokes" not in cmd_data:
            return [], f"Command {i + 1} missing required 'keystrokes' field"
        keystrokes = cmd_data["keystrokes"]
        if not isinstance(keystrokes, str):
            return [], f"Command {i + 1} 'keystrokes' must be a string"
        duration = 1.0
        if "duration" in cmd_data:
            d = cmd_data["duration"]
            if isinstance(d, (int, float)):
                duration = float(d)
            else:
                warnings.append(f"Command {i + 1}: Invalid duration value, using default 1.0")
        else:
            warnings.append(f"Command {i + 1}: Missing duration field, using default 1.0")
        duration = min(max(0.1, duration), 60.0)
        commands.append(ParsedCommand(keystrokes=keystrokes, duration=duration))
    return commands, ""


def parse_response(response: str) -> ParseResult:
    """
    Parse a JSON response (analysis, plan, commands or image_read, task_complete).

    Returns:
        ParseResult with commands, is_task_complete, error, warning, analysis, plan, image_read.
    """
    warnings: List[str] = []
    json_content, extra_warnings = _extract_json_content(response)
    warnings.extend(extra_warnings)

    if not json_content:
        fallback = _parse_plain_text_fallback(response)
        if fallback is not None:
            return fallback
        return ParseResult(
            commands=[],
            is_task_complete=False,
            error="No valid JSON found in response",
            warning="- " + "\n- ".join(warnings) if warnings else "",
        )

    try:
        data = json.loads(json_content)
    except json.JSONDecodeError as e:
        err = f"Invalid JSON: {str(e)}"
        if len(json_content) < 200:
            err += f" | Content: {repr(json_content)}"
        else:
            err += f" | Content preview: {repr(json_content[:100])}..."
        return ParseResult(
            commands=[],
            is_task_complete=False,
            error=err,
            warning="- " + "\n- ".join(warnings) if warnings else "",
        )

    # If JSON (e.g. from tool-call fragment) is missing analysis/plan, fill from preceding text
    normalized = _strip_tool_call_markers(response).strip()
    if not data.get("analysis") or not data.get("plan"):
        before_json = normalized
        idx = normalized.find(json_content)
        if idx >= 0:
            before_json = normalized[:idx]
        m_a = re.search(r"\bAnalysis:\s*(.+?)(?=\s+Plan:|\s+Requirements|\s+Commands:|\s*$)", before_json, re.DOTALL | re.IGNORECASE)
        m_p = re.search(r"\bPlan:\s*(.+?)(?=\s+Requirements|\s+Commands:|\s*\{\s*\"|\s*$)", before_json, re.DOTALL | re.IGNORECASE)
        if m_a:
            data["analysis"] = m_a.group(1).strip()
        if m_p:
            data["plan"] = m_p.group(1).strip()
        if data.get("analysis") or data.get("plan"):
            warnings.append("analysis/plan taken from text before JSON")

    if not isinstance(data, dict):
        return ParseResult(
            commands=[],
            is_task_complete=False,
            error="Response must be a JSON object",
            warning="- " + "\n- ".join(warnings) if warnings else "",
        )

    missing = [f for f in ("analysis", "plan") if f not in data]
    if missing:
        return ParseResult(
            commands=[],
            is_task_complete=False,
            error=f"Missing required fields: {', '.join(missing)}",
            warning="- " + "\n- ".join(warnings) if warnings else "",
        )

    if not isinstance(data.get("analysis", ""), str):
        warnings.append("Field 'analysis' should be a string")
    if not isinstance(data.get("plan", ""), str):
        warnings.append("Field 'plan' should be a string")

    has_commands = "commands" in data
    has_image_read = "image_read" in data

    if has_commands and has_image_read:
        return ParseResult(
            commands=[],
            is_task_complete=False,
            error="Fields 'commands' and 'image_read' are mutually exclusive. Provide exactly one per response.",
            warning="- " + "\n- ".join(warnings) if warnings else "",
            analysis=data.get("analysis", ""),
            plan=data.get("plan", ""),
        )

    if not has_commands and not has_image_read:
        task_complete = data.get("task_complete", False)
        if isinstance(task_complete, str):
            task_complete = task_complete.lower() in ("true", "1", "yes")
        if not task_complete:
            return ParseResult(
                commands=[],
                is_task_complete=False,
                error="Response must include either 'commands' or 'image_read'. Provide exactly one of them.",
                warning="- " + "\n- ".join(warnings) if warnings else "",
                analysis=data.get("analysis", ""),
                plan=data.get("plan", ""),
            )

    is_complete = data.get("task_complete", False)
    if isinstance(is_complete, str):
        is_complete = is_complete.lower() in ("true", "1", "yes")

    analysis = data.get("analysis", "") or ""
    plan = data.get("plan", "") or ""

    image_read: Optional[ImageReadRequest] = None
    if has_image_read:
        ir = data["image_read"]
        if not isinstance(ir, dict):
            return ParseResult(
                commands=[],
                is_task_complete=False,
                error="Field 'image_read' must be an object",
                warning="- " + "\n- ".join(warnings) if warnings else "",
                analysis=analysis,
                plan=plan,
            )
        if "file_path" not in ir or "image_read_instruction" not in ir:
            return ParseResult(
                commands=[],
                is_task_complete=False,
                error="Field 'image_read' missing required 'file_path' or 'image_read_instruction'",
                warning="- " + "\n- ".join(warnings) if warnings else "",
                analysis=analysis,
                plan=plan,
            )
        image_read = ImageReadRequest(
            file_path=str(ir["file_path"]),
            image_read_instruction=str(ir["image_read_instruction"]),
        )
        return ParseResult(
            commands=[],
            is_task_complete=is_complete,
            error="",
            warning="- " + "\n- ".join(warnings) if warnings else "",
            analysis=analysis,
            plan=plan,
            image_read=image_read,
        )

    # has_commands
    commands_list = data.get("commands", [])
    if not isinstance(commands_list, list):
        return ParseResult(
            commands=[],
            is_task_complete=False,
            error="Field 'commands' must be an array",
            warning="- " + "\n- ".join(warnings) if warnings else "",
            analysis=analysis,
            plan=plan,
        )

    commands, parse_error = _parse_commands(commands_list, warnings)
    if parse_error:
        if is_complete:
            warnings.append(parse_error)
            return ParseResult(
                commands=[],
                is_task_complete=True,
                error="",
                warning="- " + "\n- ".join(warnings) if warnings else "",
                analysis=analysis,
                plan=plan,
            )
        return ParseResult(
            commands=[],
            is_task_complete=False,
            error=parse_error,
            warning="- " + "\n- ".join(warnings) if warnings else "",
            analysis=analysis,
            plan=plan,
        )

    if is_complete and len(commands) > 0:
        is_complete = False
        warnings.append(
            "task_complete was set to true but commands are present. "
            "Overriding to false. Run your commands first, then mark task_complete on the next turn."
        )

    return ParseResult(
        commands=commands,
        is_task_complete=is_complete,
        error="",
        warning="- " + "\n- ".join(warnings) if warnings else "",
        analysis=analysis,
        plan=plan,
    )


def assistant_content_from_parse_result(parsed: ParseResult, raw_response: str) -> str:
    """Build assistant message content for history from parse result."""
    if parsed.error:
        return raw_response
    parts = []
    if parsed.analysis:
        parts.append(f"Analysis: {parsed.analysis}")
    if parsed.plan:
        parts.append(f"Plan: {parsed.plan}")
    return "\n".join(parts) if parts else raw_response
