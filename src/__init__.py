"""
BaseAgent - An autonomous coding agent for Term Challenge.

Entry point: agent.py
"""

__version__ = "1.0.0"
__author__ = "Platform Network"

from src.config.defaults import CONFIG
from src.output.jsonl import emit

__all__ = [
    "CONFIG",
    "emit",
    "__version__",
]
