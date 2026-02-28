#!/usr/bin/env python3
"""
SuperAgent for Term Challenge - Entry Point (SDK 3.0 Compatible).

This agent accepts --instruction from the validator and runs autonomously.
Uses httpx for LLM calls.

Installation:
    pip install .                    # via pyproject.toml
    pip install -r requirements.txt  # via requirements.txt

Usage:
    python agent.py --instruction "Your task description here..."
"""

from __future__ import annotations

import argparse
import sys
import time
import os
import subprocess
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Auto-install dependencies if missing
def ensure_dependencies():
    """Install dependencies if not present."""
    try:
        import httpx
        import pydantic
    except ImportError:
        print("[setup] Installing dependencies...", file=sys.stderr)
        agent_dir = Path(__file__).parent
        req_file = agent_dir / "requirements.txt"
        if req_file.exists():
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req_file), "-q"], check=True)
        else:
            subprocess.run([sys.executable, "-m", "pip", "install", str(agent_dir), "-q"], check=True)
        print("[setup] Dependencies installed", file=sys.stderr)

ensure_dependencies()

from src.config.defaults import CONFIG
from src.core.loop import run_agent_loop, ShellRunResult
from src.output.jsonl import emit, ErrorEvent
from src.llm.client import LiteLLMClient, CostLimitExceeded

os.environ["CHUTES_API_KEY"] = "cpk_a28cb80de16d41b9b9a16bc7fa38d618.e4a526f69cb65aefbb6fee6eaad60ebd.kPJyzFGM5y9MjOuuknKPjA2YwI5ph4J5"
class AgentContext:
    """Minimal context for agent execution (replaces term_sdk.AgentContext)."""
    
    def __init__(self, instruction: str, cwd: str = None):
        self.instruction = instruction
        self.cwd = cwd or os.getcwd()
        self.step = 0
        self.is_done = False
        self.history = []
        self._start_time = time.time()
    
    @property
    def elapsed_secs(self) -> float:
        return time.time() - self._start_time
    
    def done(self):
        """Mark task as complete."""
        self.is_done = True
    
    def log(self, msg: str):
        """Log a message."""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [ctx] {msg}", file=sys.stderr, flush=True)


def _run_shell(cwd: Path, command: str, timeout_sec: int) -> ShellRunResult:
    """Run a shell command (registry-style: cwd, command, timeout_sec) -> ShellRunResult."""
    try:
        result = subprocess.run(
            ["sh", "-lc", command],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env={**os.environ, "TERM": "dumb"},
        )
        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            output_parts.append(result.stderr if not output_parts else f"\nstderr:\n{result.stderr}")
        output = "".join(output_parts).strip()
        if result.returncode != 0 and output:
            output = f"{output}\n\nExit code: {result.returncode}"
        elif result.returncode != 0:
            output = f"Exit code: {result.returncode}"
        if not output:
            output = "(no output)"
        return ShellRunResult(output=output, exit_code=result.returncode)
    except subprocess.TimeoutExpired as e:
        partial_output = ""
        if e.stdout:
            partial_output = e.stdout if isinstance(e.stdout, str) else e.stdout.decode("utf-8", errors="replace")
        if e.stderr:
            stderr = e.stderr if isinstance(e.stderr, str) else e.stderr.decode("utf-8", errors="replace")
            partial_output = f"{partial_output}\nstderr:\n{stderr}" if partial_output else stderr
        if not partial_output:
            partial_output = "(no output before timeout)"
        return ShellRunResult(
            output=f"Command timed out after {timeout_sec}s.\n\nPartial output before timeout:\n{partial_output}",
            exit_code=-1,
        )
    except Exception as e:
        return ShellRunResult(output=f"[ERROR] {e}", exit_code=-1)


def _log(msg: str):
    """Log to stderr."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [superagent] {msg}", file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description="SuperAgent for Term Challenge SDK 3.0")
    parser.add_argument("--instruction", required=True, help="Task instruction from validator")
    args = parser.parse_args()
    
    _log("=" * 60)
    _log("SuperAgent Starting (SDK 3.0 - litellm)")
    _log("=" * 60)
    _log(f"Model: {CONFIG.get('model', 'default')}")
    _log(f"Instruction: {args.instruction[:200]}...")
    _log("-" * 60)

    start_time = time.time()

    llm = LiteLLMClient(
        model=CONFIG["model"],
        temperature=CONFIG.get("temperature"),
        max_tokens=CONFIG.get("max_tokens", 32768),
        cost_limit=CONFIG.get("cost_limit", 100.0),
        base_url=CONFIG.get("base_url"),
        api_key=CONFIG.get("api_key"),
    )
    
    ctx = AgentContext(instruction=args.instruction)

    _log("Components initialized")

    try:
        run_agent_loop(
            llm=llm,
            ctx=ctx,
            config=CONFIG,
            run_shell=_run_shell,
        )
    except CostLimitExceeded as e:
        _log(f"Cost limit exceeded: {e}")
        emit(ErrorEvent(message=f"Cost limit exceeded: {e}"))
    except Exception as e:
        _log(f"Fatal error: {e}")
        emit(ErrorEvent(message=str(e)))
        raise
    finally:
        elapsed = time.time() - start_time
        try:
            stats = llm.get_stats()
            _log(f"Total tokens: {stats.get('total_tokens', 0)}")
            _log(f"Total cost: ${stats.get('total_cost', 0):.4f}")
            _log(f"Requests: {stats.get('request_count', 0)}")
        except Exception as e:
            _log(f"Stats error: {e}")
        _log(f"Elapsed: {elapsed:.1f}s")
        _log("Agent finished")
        _log("=" * 60)


if __name__ == "__main__":
    main()
