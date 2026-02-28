"""
Context compaction system for SuperAgent.

Implements intelligent context management like OpenCode/Codex:
1. Token-based overflow detection
2. Tool output pruning (clear old outputs, keep recent)
3. AI-powered conversation compaction (summarization)

This replaces naive sliding window truncation which breaks cache.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.llm.client import LiteLLMClient

# =============================================================================
# Constants (matching OpenCode)
# =============================================================================

# Token estimation
APPROX_CHARS_PER_TOKEN = 4

# Context limits
MODEL_CONTEXT_LIMIT = 200_000  # Claude Opus 4.5 context window
OUTPUT_TOKEN_MAX = 32_000  # Max output tokens to reserve
AUTO_COMPACT_THRESHOLD = 0.85  # Trigger compaction at 85% of usable context
# Harbor-style proactive summarization: when free tokens fall below this, compact before next request
PROACTIVE_SUMMARIZATION_THRESHOLD = 8_000
# Target free tokens when unwinding on context-length error (harbor: 4000)
UNWIND_TARGET_FREE_TOKENS = 4_000

# Pruning constants (from OpenCode)
PRUNE_PROTECT = 40_000  # Protect this many tokens of recent tool output
PRUNE_MINIMUM = 20_000  # Only prune if we can recover at least this many tokens
PRUNE_MARKER = "[Old tool result content cleared]"

MAX_IMAGES_PER_REQUEST = 100
IMAGE_PRUNE_TARGET = 10  # Keep only last N images (LLM has already seen older ones)

# Compaction prompts (from Codex)
COMPACTION_PROMPT = """You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM that will resume the task.

Include:
- Current progress and key decisions made
- Important context, constraints, or user preferences
- What remains to be done (clear next steps)
- Any critical data, examples, or references needed to continue
- Which files were modified and how
- Any errors encountered and how they were resolved

Be concise, structured, and focused on helping the next LLM seamlessly continue the work. Use bullet points and clear sections."""

SUMMARY_PREFIX = """Another language model started to solve this problem and produced a summary of its thinking process. You also have access to the state of the tools that were used. Use this to build on the work that has already been done and avoid duplicating work.

Here is the summary from the previous context:

"""


# =============================================================================
# Token Estimation
# =============================================================================

def estimate_tokens(text: str) -> int:
    """Estimate tokens from text length (4 chars per token heuristic)."""
    return max(0, len(text or "") // APPROX_CHARS_PER_TOKEN)


def estimate_message_tokens(msg: Dict[str, Any]) -> int:
    """Estimate tokens for a single message."""
    tokens = 0
    
    # Content tokens
    content = msg.get("content")
    if isinstance(content, str):
        tokens += estimate_tokens(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                tokens += estimate_tokens(part.get("text", ""))
                # Images count as ~1000 tokens roughly
                if part.get("type") == "image_url":
                    tokens += 1000
    
    # Tool calls tokens (function name + arguments)
    tool_calls = msg.get("tool_calls", [])
    for tc in tool_calls:
        func = tc.get("function", {})
        tokens += estimate_tokens(func.get("name", ""))
        tokens += estimate_tokens(func.get("arguments", ""))
    
    # Role overhead (~4 tokens)
    tokens += 4
    
    return tokens


def estimate_total_tokens(messages: List[Dict[str, Any]]) -> int:
    """Estimate total tokens for all messages."""
    return sum(estimate_message_tokens(m) for m in messages)


# =============================================================================
# Overflow Detection
# =============================================================================

def get_usable_context() -> int:
    """Get usable context window (total - reserved for output)."""
    return MODEL_CONTEXT_LIMIT - OUTPUT_TOKEN_MAX


def is_overflow(total_tokens: int, threshold: float = AUTO_COMPACT_THRESHOLD) -> bool:
    """Check if context is overflowing based on token count."""
    usable = get_usable_context()
    return total_tokens > usable * threshold


def needs_compaction(messages: List[Dict[str, Any]]) -> bool:
    """Check if messages need compaction."""
    total_tokens = estimate_total_tokens(messages)
    return is_overflow(total_tokens)


# =============================================================================
# Tool Output Pruning
# =============================================================================

def _log(msg: str) -> None:
    """Log to stderr."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [compaction] {msg}", file=sys.stderr, flush=True)


def prune_old_tool_outputs(
    messages: List[Dict[str, Any]],
    protect_last_turns: int = 2,
) -> List[Dict[str, Any]]:
    """
    Prune old tool outputs to save tokens.
    
    Strategy (exactly like OpenCode compaction.ts lines 49-89):
    1. Go backwards through messages
    2. Skip first 2 user turns (most recent)
    3. Accumulate tool output tokens
    4. Once we've accumulated PRUNE_PROTECT (40K) tokens, start marking for prune
    5. Only actually prune if we can recover > PRUNE_MINIMUM (20K) tokens
    
    Args:
        messages: List of messages
        protect_last_turns: Number of recent user turns to skip (default: 2)
        
    Returns:
        Messages with old tool outputs pruned (content replaced with PRUNE_MARKER)
    """
    if not messages:
        return messages
    
    total = 0  # Total tool output tokens seen (going backwards)
    pruned = 0  # Tokens that will be pruned
    to_prune: List[int] = []  # Indices to prune
    turns = 0  # User turn counter
    
    # Go backwards through messages (like OpenCode)
    for msg_index in range(len(messages) - 1, -1, -1):
        msg = messages[msg_index]
        
        # Count user turns
        if msg.get("role") == "user":
            turns += 1
        
        # Skip the first N user turns (most recent)
        if turns < protect_last_turns:
            continue
        
        # Process tool messages
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            
            # Skip already pruned
            if content == PRUNE_MARKER:
                # Already compacted, stop here (like OpenCode: break loop)
                break
            
            estimate = estimate_tokens(content)
            total += estimate
            
            # Once we've accumulated more than PRUNE_PROTECT tokens,
            # start marking older outputs for pruning
            if total > PRUNE_PROTECT:
                pruned += estimate
                to_prune.append(msg_index)
    
    _log(f"Prune scan: {total} total tokens, {pruned} prunable")
    
    # Only prune if we can recover enough tokens
    if pruned <= PRUNE_MINIMUM:
        _log(f"Prune skipped: only {pruned} tokens recoverable (min: {PRUNE_MINIMUM})")
        return messages
    
    _log(f"Pruning {len(to_prune)} tool outputs, recovering ~{pruned} tokens")
    
    # Create new messages with pruned content
    indices_to_prune = set(to_prune)
    result = []
    for i, msg in enumerate(messages):
        if i in indices_to_prune:
            result.append({
                **msg,
                "content": PRUNE_MARKER,
            })
        else:
            result.append(msg)
    
    return result

def count_images_in_message(msg: Dict[str, Any]) -> int:
    """Count number of images in a message."""
    count = 0
    content = msg.get("content")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "image_url" or part.get("type") == "image":
                    count += 1
    return count


def count_total_images(messages: List[Dict[str, Any]]) -> int:
    """Count total images across all messages."""
    return sum(count_images_in_message(m) for m in messages)


def is_image_analyzed(messages: List[Dict[str, Any]], image_msg_index: int) -> bool:
    """
    Check if an image has been analyzed by the LLM.
    
    An image is considered "analyzed" if there's an assistant message
    after the message containing the image.
    
    Args:
        messages: Full message list
        image_msg_index: Index of the message containing the image
        
    Returns:
        True if there's an assistant message after this image
    """
    for i in range(image_msg_index + 1, len(messages)):
        if messages[i].get("role") == "assistant":
            return True
    return False


def prune_old_images(
    messages: List[Dict[str, Any]],
    max_images: int = IMAGE_PRUNE_TARGET,
) -> List[Dict[str, Any]]:
    """
    Remove old images from context.
    
    Strategy:
    1. Prefer removing analyzed images first (have assistant response after)
    2. If still over HARD LIMIT (100), force remove oldest unanalyzed too
    3. Replace removed images with text placeholder
    
    Args:
        messages: List of messages
        max_images: Target maximum number of images
        
    Returns:
        Messages with old images pruned
    """
    total_images = count_total_images(messages)
    
    if total_images <= max_images:
        return messages
    
    # First, identify which messages have images and whether they've been analyzed
    image_msg_indices = []
    for i, msg in enumerate(messages):
        img_count = count_images_in_message(msg)
        if img_count > 0:
            analyzed = is_image_analyzed(messages, i)
            image_msg_indices.append((i, analyzed, img_count))
    
    # Count analyzed vs unanalyzed images
    analyzed_count = sum(count for _, analyzed, count in image_msg_indices if analyzed)
    unanalyzed_count = sum(count for _, analyzed, count in image_msg_indices if not analyzed)
    
    _log(f"Image status: {analyzed_count} analyzed, {unanalyzed_count} unanalyzed, {total_images} total")
    
    # Calculate how many we need to remove
    images_to_remove = total_images - max_images
    
    # Check if we're over the HARD API limit (100)
    over_hard_limit = total_images > MAX_IMAGES_PER_REQUEST
    
    if over_hard_limit:
        # MUST remove enough to get under 100, even if unanalyzed
        hard_limit_target = MAX_IMAGES_PER_REQUEST - 5  # Leave some buffer
        images_to_remove = max(images_to_remove, total_images - hard_limit_target)
        _log(f"Over hard limit ({MAX_IMAGES_PER_REQUEST}), forcing removal of {images_to_remove} images")
    
    # Build removal list: analyzed first, then unanalyzed if needed
    indices_to_prune = set()
    removed = 0
    
    # Pass 1: Remove analyzed images (oldest first)
    for msg_idx, analyzed, img_count in image_msg_indices:
        if removed >= images_to_remove:
            break
        if analyzed:
            indices_to_prune.add(msg_idx)
            removed += img_count
    
    # Pass 2: If still need to remove more (over hard limit), remove unanalyzed too
    if removed < images_to_remove and over_hard_limit:
        _log(f"Still need to remove {images_to_remove - removed} unanalyzed images")
        for msg_idx, analyzed, img_count in image_msg_indices:
            if removed >= images_to_remove:
                break
            if not analyzed and msg_idx not in indices_to_prune:
                indices_to_prune.add(msg_idx)
                removed += img_count
    
    if not indices_to_prune:
        _log(f"Image pruning skipped: no removable images")
        return messages
    
    _log(f"Image pruning: removing {removed} images from {len(indices_to_prune)} messages")
    
    # Build result with pruned images
    result = []
    for i, msg in enumerate(messages):
        if i not in indices_to_prune:
            result.append(msg)
            continue
        
        content = msg.get("content")
        if not isinstance(content, list):
            result.append(msg)
            continue
        
        # Remove images from this message
        new_content = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("image_url", "image"):
                # Replace with text placeholder
                new_content.append({
                    "type": "text",
                    "text": "[Image removed to stay within API limits]"
                })
            else:
                new_content.append(part)
        
        result.append({**msg, "content": new_content})
    
    _log(f"Image pruning complete: removed images from {len(indices_to_prune)} messages")
    return result


# =============================================================================
# AI Compaction
# =============================================================================

def run_compaction(
    llm: "LiteLLMClient",
    messages: List[Dict[str, Any]],
    system_prompt: str,
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Compact conversation history using AI summarization.
    
    Process (like Codex):
    1. Send all messages + compaction prompt to LLM
    2. Get summary response
    3. Create new message list:
       - Original system prompt
       - Summary as user message (with prefix)
       - Ready for continuation
    
    Args:
        llm: LLM client for summarization
        messages: Current message history
        system_prompt: Original system prompt to preserve
        model: Model to use (defaults to current)
        
    Returns:
        Compacted message list
    """
    _log("Starting AI compaction...")
    
    # Build compaction request
    compaction_messages = messages.copy()
    compaction_messages.append({
        "role": "user",
        "content": COMPACTION_PROMPT,
    })
    
    try:
        # Call LLM for summary (no tools, just text)
        response = llm.chat(
            compaction_messages,
            model=model,
            max_tokens=4096,  # Summary should be concise
        )
        
        summary = response.text or ""
        
        if not summary:
            _log("Compaction failed: empty response")
            return messages
        
        summary_tokens = estimate_tokens(summary)
        _log(f"Compaction complete: {summary_tokens} token summary")
        
        # Build new message list
        compacted = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": SUMMARY_PREFIX + summary},
        ]
        
        return compacted
        
    except Exception as e:
        _log(f"Compaction failed: {e}")
        # Return original messages if compaction fails
        return messages


# =============================================================================
# Unwind on context-length error (harbor-style)
# =============================================================================


def unwind_messages_to_free_tokens(
    messages: List[Dict[str, Any]],
    target_free_tokens: int = UNWIND_TARGET_FREE_TOKENS,
) -> List[Dict[str, Any]]:
    """
    Remove recent messages until we have at least target_free_tokens free (harbor-style).

    Removes the most recent pair of messages (user + assistant) repeatedly.
    Keeps at least the first two messages (system + first user).

    Args:
        messages: Current message list.
        target_free_tokens: Minimum free tokens to leave.

    Returns:
        New message list with trailing pairs removed.
    """
    usable = get_usable_context()
    result = list(messages)
    while len(result) > 2:
        current_tokens = estimate_total_tokens(result)
        free_tokens = usable - current_tokens
        if free_tokens >= target_free_tokens:
            break
        # Remove the most recent pair (user + assistant, or last two messages)
        result = result[:-2]
    if len(result) < len(messages):
        _log(
            f"Unwound messages: {len(messages)} -> {len(result)}, "
            f"free tokens ~{get_usable_context() - estimate_total_tokens(result)}"
        )
    return result


# =============================================================================
# Main Context Management
# =============================================================================

def manage_context(
    messages: List[Dict[str, Any]],
    system_prompt: str,
    llm: "LiteLLMClient",
    force_compaction: bool = False,
    proactive_summarization: bool = True,
) -> List[Dict[str, Any]]:
    """
    Main context management function (harbor-style: prune, proactive summarization, compaction).

    Called before each LLM request to ensure context fits.

    Strategy:
    1. Prune old images first
    2. Estimate current token usage
    3. If under threshold and free tokens above proactive threshold, return as-is
    4. Try pruning old tool outputs first
    5. If still over threshold or free tokens low, run AI compaction

    Args:
        messages: Current message history
        system_prompt: Original system prompt (preserved through compaction)
        llm: LLM client (for compaction)
        force_compaction: Force compaction even if under threshold
        proactive_summarization: When True, compact when free_tokens < PROACTIVE_SUMMARIZATION_THRESHOLD

    Returns:
        Managed message list (possibly compacted)
    """
    # Step 0: Always prune images first (hard API limit, not token-based)
    total_images = count_total_images(messages)
    if total_images > IMAGE_PRUNE_TARGET:
        _log(f"Image count: {total_images} (limit: {MAX_IMAGES_PER_REQUEST})")
        messages = prune_old_images(messages)

    total_tokens = estimate_total_tokens(messages)
    usable = get_usable_context()
    free_tokens = usable - total_tokens
    usage_pct = (total_tokens / usable) * 100

    _log(f"Context: {total_tokens} tokens ({usage_pct:.1f}% of {usable}), free ~{free_tokens}")

    # Check if we need to do anything: overflow or proactive (harbor-style)
    need_compaction = force_compaction or is_overflow(total_tokens)
    if proactive_summarization and free_tokens < PROACTIVE_SUMMARIZATION_THRESHOLD:
        _log(f"Proactive summarization: free tokens {free_tokens} < {PROACTIVE_SUMMARIZATION_THRESHOLD}")
        need_compaction = True
    if not need_compaction:
        return messages
    
    _log("Context management: pruning and/or compacting...")

    # Step 1: Try pruning old tool outputs
    pruned = prune_old_tool_outputs(messages)
    pruned_tokens = estimate_total_tokens(pruned)
    pruned_free = usable - pruned_tokens

    if not force_compaction and not is_overflow(pruned_tokens) and (
        not proactive_summarization or pruned_free >= PROACTIVE_SUMMARIZATION_THRESHOLD
    ):
        _log(f"Pruning sufficient: {total_tokens} -> {pruned_tokens} tokens")
        return pruned

    # Step 2: Run AI compaction
    _log(f"Pruning insufficient ({pruned_tokens} tokens), running AI compaction...")
    compacted = run_compaction(llm, pruned, system_prompt)
    compacted_tokens = estimate_total_tokens(compacted)
    
    _log(f"Compaction result: {total_tokens} -> {compacted_tokens} tokens")
    
    return compacted
