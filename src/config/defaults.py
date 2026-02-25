from __future__ import annotations

import os
from typing import Any, Dict, List

# Default model and fallback list for routing (round-robin on failure)
_DEFAULT_MODEL = os.environ.get("LLM_MODEL", "zai-org/GLM-5-TEE")
_DEFAULT_MODELS: List[str] = [    
    "zai-org/GLM-5-TEE",
    "zai-org/GLM-4.7-TEE",
    "moonshotai/Kimi-K2.5-TEE",
]

# Allow env to override the list as comma-separated (e.g. LLM_MODELS=model1,model2)
def _get_models() -> List[str]:
    raw = os.environ.get("LLM_MODELS", "").strip()
    if raw:
        return [m.strip() for m in raw.split(",") if m.strip()]
    # No LLM_MODELS: put LLM_MODEL (or default) first, then other defaults as fallbacks
    primary = _DEFAULT_MODEL
    rest = [m for m in _DEFAULT_MODELS if m != primary]
    return [primary] + rest if rest else [primary]


_models = _get_models()
_primary_model = _models[0] if _models else _DEFAULT_MODEL

# Main configuration
CONFIG: Dict[str, Any] = {
    # ==========================================================================
    # Model / API (OpenAI-compatible; no litellm, no OpenRouter)
    # ==========================================================================
    "model": _primary_model,
    "models": _models,
    "base_url": os.environ.get("CHUTES_BASE_URL", "https://llm.chutes.ai/v1"),
    "api_key": os.environ.get("CHUTES_API_KEY", ""),
    "reasoning_effort": "none",
    
    # Token limits
    "max_tokens": 16384,
    
    # Temperature (0 = deterministic)
    "temperature": 0.0,
    
    # ==========================================================================
    # Agent Execution Settings
    # ==========================================================================
    
    # Maximum iterations before stopping
    "max_iterations": 350,
    
    "cost_limit": 100.0,
    # Maximum tokens for tool output truncation (middle-out strategy)
    "max_output_tokens": 2500,  # ~10KB
    
    # Timeout for shell commands (seconds)
    "shell_timeout": 60,
    
    # ==========================================================================
    # Context Management (like OpenCode/Codex)
    # ==========================================================================
    
    # Model context window (Claude Opus 4.5 = 200K)
    "model_context_limit": 200_000,
    
    # Reserved tokens for output
    "output_token_max": 32_000,
    
    # Trigger compaction at this % of usable context (85%)
    "auto_compact_threshold": 0.85,
    
    # Tool output pruning constants (from OpenCode)
    "prune_protect": 40_000,   # Protect this many tokens of recent tool output
    "prune_minimum": 20_000,   # Only prune if we can recover at least this many
    
    # ==========================================================================
    # Prompt Caching (model-specific)
    # ==========================================================================
    
    # Enable prompt caching
    "cache_enabled": True,
    
    # OpenAI (gpt-5.1-codex-max) caching configuration:
    # - Caching is AUTOMATIC for OpenAI models (no markers needed)
    # - Minimum 1,024 tokens in prefix to start caching
    # - Prefix matching in increments of 128 tokens after first 1,024
    # - Extended retention: up to 24 hours (vs 5-10 min default)
    # - Best practice: Keep system prompt and tools first (stable prefix)
    "cache_extended_retention": True,  # Enable 24h retention for codex-max
    "cache_key": None,  # Optional: set for high-traffic (>15 req/min) scenarios
    
    # Anthropic caching notes (if switching to Claude):
    # - Uses cache_control breakpoints (max 4)
    # - Claude Opus 4.5 on Bedrock: 4096 tokens minimum per breakpoint
    # - Claude Sonnet/other: 1024 tokens minimum
    
    # ==========================================================================
    # Simulated Codex Flags (all enabled/bypassed for benchmark)
    # ==========================================================================
    
    # --dangerously-bypass-approvals-and-sandbox
    "bypass_approvals": True,
    "bypass_sandbox": True,
    
    # --skip-git-repo-check
    "skip_git_check": True,
    
    # --enable unified_exec
    "unified_exec": True,
    
    # --json (always JSONL output)
    "json_output": True,
    
    # ==========================================================================
    # Double Confirmation for Task Completion
    # ==========================================================================
    
    # Require double confirmation before marking task complete
    # Disabled for fully autonomous operation in evaluation mode
    "require_completion_confirmation": False,
}
