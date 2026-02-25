"""
Main agent loop - no tool calling.

Implements the agentic loop that:
1. Receives instruction via --instruction argument
2. Calls LLM (no tools); expects JSON response with commands, analysis, plan, task_complete
3. Parses JSON and executes commands via shell
4. Loops until task is complete (with completion confirmation)
5. Emits JSONL events throughout

Context management: token-based overflow, pruning, AI compaction. Caching unchanged.
"""

from __future__ import annotations

import copy
import time
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from src.llm.client import LLMError, CostLimitExceeded

from src.output.jsonl import (
    emit,
    next_item_id,
    reset_item_counter,
    ThreadStartedEvent,
    TurnStartedEvent,
    TurnCompletedEvent,
    TurnFailedEvent,
    ItemStartedEvent,
    ItemCompletedEvent,
    make_agent_message_item,
    make_command_execution_item,
)
from src.prompts.system import (
    get_system_prompt,
    get_completion_confirmation,
    BEGIN_USER_MESSAGE,
)
from src.utils.truncate import middle_out_truncate
from src.core.compaction import (
    manage_context,
    estimate_total_tokens,
    unwind_messages_to_free_tokens,
)
from src.core.parser import (
    parse_response,
    assistant_content_from_parse_result,
    ImageReadRequest,
)

if TYPE_CHECKING:
    from src.llm.client import LiteLLMClient


@dataclass
class ShellRunResult:
    """Result of a shell run (registry-style run_shell)."""
    output: str
    exit_code: int


# Callable: (cwd, command, timeout_sec) -> ShellRunResult (like cute registry _run_shell)
RunShellCallable = Callable[[Path, str, int], ShellRunResult]

MAX_IMAGE_BYTES = 5 * 1024 * 1024

_IMAGE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def _execute_image_read(
    image_read: ImageReadRequest,
    llm: "LiteLLMClient",
    cwd: Path,
    run_shell: RunShellCallable,
    config: Dict[str, Any],
    model: Optional[str] = None,
) -> str:
    """
    Execute image_read: read image via shell (base64), send to LLM as multimodal, return analysis.
    """
    file_path = image_read.file_path
    # Use base64 -w 0 to avoid newlines if available
    result = run_shell(cwd, f"base64 -w 0 '{file_path}' 2>/dev/null || base64 '{file_path}' 2>/dev/null", 30)
    if result.exit_code != 0:
        err = result.output or "unknown error"
        return f"ERROR: Failed to read file '{file_path}': {err}"

    b64 = (result.output or "").replace("\n", "").strip()
    if not b64:
        return f"ERROR: Empty or invalid base64 for '{file_path}'"

    try:
        import base64 as b64_module
        size = len(b64_module.b64decode(b64, validate=True))
        if size > MAX_IMAGE_BYTES:
            return (
                f"ERROR: Image too large ({size} bytes, max {MAX_IMAGE_BYTES}). "
                f"Reduce size or use a smaller image."
            )
    except Exception:
        pass

    ext = Path(file_path).suffix.lower()
    mime = _IMAGE_MIME.get(ext)
    if mime is None:
        return (
            f"ERROR: Unsupported image format '{ext}'. "
            f"Convert to PNG first (e.g. convert image{ext} to image.png), then use image_read on the PNG file."
        )

    multimodal_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": image_read.image_read_instruction},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                },
            ],
        },
    ]

    try:
        response = llm.chat(
            multimodal_messages,
            tools=None,
            max_tokens=config.get("max_output_tokens", 4096),
            temperature=0.0,
            model=model,
        )
        response_text = response.text or ""
    except Exception as e:
        return f"ERROR: {e}"

    return f"File Read Result for '{file_path}':\n{response_text}"


def _log(msg: str) -> None:
    """Log to stderr."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [loop] {msg}", file=sys.stderr, flush=True)


def _add_cache_control_to_message(
    msg: Dict[str, Any],
    cache_control: Dict[str, str],
) -> Dict[str, Any]:
    """Add cache_control to a message, converting to multipart if needed."""
    content = msg.get("content")

    if isinstance(content, list):
        has_cache = any(
            isinstance(p, dict) and "cache_control" in p
            for p in content
        )
        if has_cache:
            return msg
        new_content = list(content)
        for i in range(len(new_content) - 1, -1, -1):
            part = new_content[i]
            if isinstance(part, dict) and part.get("type") == "text":
                new_content[i] = {**part, "cache_control": cache_control}
                break
        return {**msg, "content": new_content}

    if isinstance(content, str):
        return {
            **msg,
            "content": [
                {"type": "text", "text": content, "cache_control": cache_control},
            ],
        }
    return msg


def _apply_caching(
    messages: List[Dict[str, Any]],
    enabled: bool = True,
) -> List[Dict[str, Any]]:
    """Apply prompt caching (stable prefix + last messages). Do not modify caching logic."""
    if not enabled or not messages:
        return messages
    cache_control = {"type": "ephemeral"}
    system_indices = [i for i, m in enumerate(messages) if m.get("role") == "system"]
    non_system_indices = [i for i, m in enumerate(messages) if m.get("role") != "system"]
    indices_to_cache = set()
    for idx in system_indices[:2]:
        indices_to_cache.add(idx)
    for idx in non_system_indices[-2:]:
        indices_to_cache.add(idx)
    result = []
    for i, msg in enumerate(messages):
        if i in indices_to_cache:
            result.append(_add_cache_control_to_message(msg, cache_control))
        else:
            result.append(msg)
    if indices_to_cache:
        _log(f"Prompt caching: {len(indices_to_cache)} breakpoints")
    return result


def run_agent_loop(
    llm: "LiteLLMClient",
    ctx: Any,
    config: Dict[str, Any],
    run_shell: RunShellCallable,
) -> None:
    """
    Run the main agent loop

    Args:
        llm: LiteLLM client
        ctx: Agent context with instruction, cwd, done()
        config: Configuration dictionary
        run_shell: Registry-style shell runner: (cwd, command, timeout_sec) -> ShellRunResult
    """
    reset_item_counter()
    session_id = f"sess_{int(time.time() * 1000)}"
    emit(ThreadStartedEvent(thread_id=session_id))
    emit(TurnStartedEvent())

    cwd = Path(ctx.cwd)
    _log("Getting initial state...")
    initial_result = run_shell(cwd, "pwd && ls -la", 60)
    max_output_tokens = config.get("max_output_tokens", 2500)
    initial_state = middle_out_truncate(initial_result.output, max_tokens=max_output_tokens)
    system_prompt = get_system_prompt(instruction=ctx.instruction, terminal_state=initial_state)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": BEGIN_USER_MESSAGE},
    ]

    total_input_tokens = 0
    total_output_tokens = 0
    total_cached_tokens = 0
    pending_completion = False

    max_iterations = config.get("max_iterations", 200)
    cache_enabled = config.get("cache_enabled", True)
    iteration = 0
    total_cost = 0.0
    cost_limit = config.get("cost_limit", 100.0)
    prev_messages = copy.deepcopy(messages)

    # Model routing: round-robin through config["models"] on LLM errors
    models: List[str] = config.get("models") or [config.get("model", "zai-org/GLM-5-TEE")]
    if not models:
        models = [config.get("model", "zai-org/GLM-5-TEE")]
    main_model = models[0]

    while iteration < max_iterations:
        iteration += 1
        _log(f"Iteration {iteration}/{max_iterations} (model: {main_model})")

        temperature = 0.0
        try:
            context_messages = manage_context(
                messages=messages,
                system_prompt=system_prompt,
                llm=llm,
            )
            if len(context_messages) < len(messages):
                _log(f"Context compacted: {len(messages)} -> {len(context_messages)} messages")
                messages = context_messages

            cached_messages = _apply_caching(context_messages, enabled=cache_enabled)

            max_retries = 10
            response = None

            for attempt in range(1, max_retries + 1):
                try:
                    # No tool calling: LLM returns plain text (JSON)
                    response = llm.chat(
                        cached_messages,
                        tools=None,
                        max_tokens=config.get("max_tokens", 32768),
                        extra_body={
                            "reasoning": {"effort": config.get("reasoning_effort", "xhigh")},
                        },
                        temperature=temperature,
                        model=main_model,
                    )
                    prev_messages = copy.deepcopy(messages)
                    total_cost += response.cost
                    _log(f"current cost: ${response.cost:.4f} total cost: ${total_cost:.4f}")

                    if hasattr(response, "tokens") and response.tokens:
                        tokens = response.tokens
                        if isinstance(tokens, dict):
                            total_input_tokens += tokens.get("input", 0)
                            total_output_tokens += tokens.get("output", 0)
                            total_cached_tokens += tokens.get("cached", 0)
                    break

                except CostLimitExceeded:
                    raise
                except LLMError as e:
                    error_msg = str(e.message) if hasattr(e, "message") else str(e)
                    _log(f"LLM error (attempt {attempt}/{max_retries}): {e.code} - {error_msg}")
                    if e.code in ("authentication_error", "invalid_api_key"):
                        raise
                    msg_lower = error_msg.lower()
                    if "prompt is too long" in msg_lower or "context_length" in msg_lower or "context length" in msg_lower:
                        _log("Context length exceeded (retry), unwinding and compacting...")
                        messages = unwind_messages_to_free_tokens(messages)
                        messages = manage_context(
                            messages,
                            system_prompt=system_prompt,
                            llm=llm,
                            force_compaction=True,
                        )
                        prev_messages = copy.deepcopy(messages)
                        if attempt < max_retries:
                            continue
                        raise
                    # Model routing: every 2nd attempt switch to next model (round-robin)
                    if attempt % 2 == 0 and len(models) > 1:
                        idx = models.index(main_model) if main_model in models else -1
                        main_model = models[(idx + 1) % len(models)]
                        _log(f"Switching to model: {main_model}")
                    if "BadRequestError" in error_msg:
                        messages = copy.deepcopy(prev_messages)
                        cached_messages = _apply_caching(messages, enabled=cache_enabled)
                    if attempt < max_retries:
                        wait_time = min(10 * attempt, 120)
                        _log(f"Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        raise
                except Exception as e:
                    _log(f"Unexpected error (attempt {attempt}/{max_retries}): {type(e).__name__}: {e}")
                    if attempt % 2 == 0 and len(models) > 1:
                        idx = models.index(main_model) if main_model in models else -1
                        main_model = models[(idx + 1) % len(models)]
                        _log(f"Switching to model: {main_model}")
                    if attempt < max_retries:
                        wait_time = min(10 * attempt, 120)
                        time.sleep(wait_time)
                    else:
                        raise

        except CostLimitExceeded as e:
            _log(f"Cost limit exceeded: {e}")
            emit(TurnFailedEvent(error={"message": f"Cost limit exceeded: {e}"}))
            ctx.done()
            return
        except LLMError as e:
            _log(f"LLM error (fatal): {e.code} - {e.message}")
            emit(TurnFailedEvent(error={"message": str(e)}))
            msg_lower = (e.message or "").lower()
            if "prompt is too long" in msg_lower or "context_length" in msg_lower or "context length" in msg_lower:
                _log("Context length exceeded, unwinding and compacting...")
                messages = unwind_messages_to_free_tokens(messages)
                messages = manage_context(
                    messages,
                    system_prompt=system_prompt,
                    llm=llm,
                    force_compaction=True,
                )
                continue
            continue
        except Exception as e:
            _log(f"Unexpected error (fatal): {type(e).__name__}: {e}")
            emit(TurnFailedEvent(error={"message": str(e)}))
            continue

        response_text = response.text or ""

        if response_text:
            item_id = next_item_id()
            emit(ItemCompletedEvent(item=make_agent_message_item(item_id, response_text)))

        _log(f"response_text: {response_text}")

        parsed = parse_response(response_text)

        stored_content = assistant_content_from_parse_result(parsed, response_text)
        messages.append({"role": "assistant", "content": stored_content})

        # Parse error: ask for valid JSON and continue
        if parsed.error:
            _log(f"Parse error: {parsed.error}")
            prompt = (
                f"Previous response had parsing errors:\n{parsed.error}\n\n"
                "Please fix these issues and provide a valid JSON object with analysis, plan, and either commands or image_read."
            )
            if parsed.warning:
                prompt = f"{prompt}\n\nWarnings: {parsed.warning}"
            messages.append({"role": "user", "content": prompt})
            continue

        if parsed.image_read is not None:
            _log(f"image_read: {parsed.image_read.file_path}")
            try:
                raw_result = _execute_image_read(
                    parsed.image_read,
                    llm=llm,
                    cwd=cwd,
                    run_shell=run_shell,
                    config=config,
                    model=main_model,
                )
                observation = raw_result
            except Exception as e:
                observation = f"ERROR: image_read failed: {e}"
            messages.append({"role": "user", "content": observation})
            continue

        if parsed.is_task_complete:
            if total_cost >= cost_limit:
                break

            if pending_completion:
                if "task incomplete" in response_text.lower():
                    _log("Task incomplete – continue working")
                    pending_completion = False
                    messages.append({
                        "role": "user",
                        "content": "The task is incomplete. Please provide a JSON response with commands to address missing verifications or issues, then continue until the task is done.",
                    })
                    continue
                _log("Task completion confirmed after verification")
                break

            pending_completion = True
            term_result = run_shell(cwd, "pwd && ls -la", 60)
            terminal_output = middle_out_truncate(term_result.output or "", max_tokens=max_output_tokens)
            confirmation_msg = get_completion_confirmation(
                instruction=ctx.instruction,
                terminal_output=terminal_output,
            )
            messages.append({"role": "user", "content": confirmation_msg})
            _log("Requesting completion confirmation")
            continue

        # Execute commands via shell (no tool calling)
        output_parts: List[str] = []
        for idx, cmd in enumerate(parsed.commands):
            item_id = next_item_id()
            emit(ItemStartedEvent(
                item=make_command_execution_item(
                    item_id=item_id,
                    command=cmd.keystrokes.strip() or "(wait)",
                    status="in_progress",
                )
            ))
            timeout_sec = min(int(cmd.duration) + 5, 120) if cmd.duration else 30
            result = run_shell(cwd, cmd.keystrokes.strip() or "echo", timeout_sec)
            out = (result.output or "").strip()
            output_parts.append(out)
            emit(ItemCompletedEvent(
                item=make_command_execution_item(
                    item_id=item_id,
                    command=cmd.keystrokes.strip() or "(wait)",
                    status="completed" if result.exit_code == 0 else "failed",
                    aggregated_output=middle_out_truncate(out, max_tokens=max_output_tokens),
                    exit_code=result.exit_code,
                )
            ))

        raw_result = "\n".join(output_parts)
        limited_result = middle_out_truncate(raw_result or "no output", max_tokens=max_output_tokens)

        if parsed.warning:
            observation = f"Previous response had warnings:\n{parsed.warning}\n\n{limited_result}"
        else:
            observation = limited_result

        messages.append({"role": "user", "content": observation})

        if total_cost >= cost_limit:
            break

    emit(TurnCompletedEvent(usage={
        "input_tokens": total_input_tokens,
        "cached_input_tokens": total_cached_tokens,
        "output_tokens": total_output_tokens,
    }))
    _log(f"Loop complete after {iteration} iterations")
    _log(f"Tokens: {total_input_tokens} input, {total_cached_tokens} cached, {total_output_tokens} output")
    ctx.done()
