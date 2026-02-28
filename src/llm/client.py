"""LLM Client using httpx."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx


class CostLimitExceeded(Exception):
    """Raised when cost limit is exceeded."""

    def __init__(self, message: str, used: float = 0, limit: float = 0):
        super().__init__(message)
        self.used = used
        self.limit = limit


class LLMError(Exception):
    """LLM API error."""

    def __init__(self, message: str, code: str = "unknown"):
        super().__init__(message)
        self.message = message
        self.code = code


@dataclass
class FunctionCall:
    """Represents a function/tool call from the LLM."""

    id: str
    name: str
    arguments: Dict[str, Any]

@dataclass
class LLMResponse:
    """Response from the LLM."""

    text: str = ""
    function_calls: List[FunctionCall] = field(default_factory=list)
    tokens: Optional[Dict[str, int]] = None
    model: str = ""
    finish_reason: str = ""
    raw: Optional[Dict[str, Any]] = None
    cost: float = 0.0
    reasoning: Optional[str] = None
    reasoning_details: Optional[List[Dict[str, Any]]] = None

    def has_function_calls(self) -> bool:
        """Check if response contains function calls."""
        return len(self.function_calls) > 0


class LLMClient:
    """LLM Client using httpx."""

    DEFAULT_BASE_URL = "https://llm.chutes.ai/v1"

    def __init__(
        self,
        model: str,
        temperature: Optional[float] = None,
        max_tokens: int = 32768,
        cost_limit: Optional[float] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cost_limit = cost_limit or float(os.environ.get("LLM_COST_LIMIT", "100.0"))
        self.base_url = base_url or os.environ.get("CHUTES_BASE_URL", self.DEFAULT_BASE_URL)
        self.timeout = timeout

        self._api_key = api_key or os.environ.get("CHUTES_API_KEY", "")

        self._total_cost = 0.0
        self._total_tokens = 0
        self._request_count = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._cached_tokens = 0

        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout, connect=30.0),
        )

    def _supports_temperature(self, model: str) -> bool:
        model_lower = model.lower()
        if any(x in model_lower for x in ["o1", "o3", "deepseek-r1"]):
            return False
        return True

    def _build_tools(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        if not tools:
            return None
        result = []
        for tool in tools:
            result.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                },
            })
        return result

    def _prepare_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prepared = []
        for msg in messages:
            new_msg = dict(msg)
            content = new_msg.get("content")
            if isinstance(content, list):
                cleaned_parts = []
                for part in content:
                    if isinstance(part, dict):
                        cleaned_part = {k: v for k, v in part.items() if k != "cache_control"}
                        cleaned_parts.append(cleaned_part)
                    else:
                        cleaned_parts.append(part)
                new_msg["content"] = cleaned_parts
            prepared.append(new_msg)
        return prepared

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        temperature: float = 0.0,
        model: Optional[str] = None,
        tool_choice: Optional[str] = None,
    ) -> LLMResponse:
        """Send a streaming chat request; accumulate and return LLMResponse."""
        if self._total_cost >= self.cost_limit:
            raise CostLimitExceeded(
                f"Cost limit exceeded: ${self._total_cost:.4f} >= ${self.cost_limit:.4f}",
                used=self._total_cost,
                limit=self.cost_limit,
            )

        payload: Dict[str, Any] = {
            "model": model or self.model,
            "messages": self._prepare_messages(messages),
            "max_tokens": max_tokens or self.max_tokens,
            "stream": True,
        }
        if self._supports_temperature(payload["model"]):
            payload["temperature"] = self.temperature if self.temperature is not None else temperature
        if tools:
            payload["tools"] = self._build_tools(tools)
            payload["tool_choice"] = tool_choice if tool_choice is not None else "auto"
        if extra_body:
            payload.update(extra_body)

        try:
            with self._client.stream("POST", "/chat/completions", json=payload) as response:
                if response.status_code != 200:
                    error_body = response.read().decode("utf-8", errors="replace")
                    try:
                        error_json = json.loads(error_body)
                        error_msg = error_json.get("error", {}).get("message", error_body)
                    except (json.JSONDecodeError, KeyError):
                        error_msg = error_body
                    if response.status_code == 401:
                        raise LLMError(error_msg, code="authentication_error")
                    elif response.status_code == 429:
                        raise LLMError(error_msg, code="rate_limit")
                    elif response.status_code >= 500:
                        raise LLMError(error_msg, code="server_error")
                    else:
                        raise LLMError(
                            f"HTTP {response.status_code}: {error_msg}",
                            code="api_error",
                        )

                content_parts: List[str] = []
                reasoning_parts: List[str] = []
                tool_calls_acc: List[Dict[str, Any]] = []
                usage_data: Optional[Dict[str, int]] = None
                result_model = model or self.model
                finish_reason = ""
                reasoning_details_last: Optional[List[Dict[str, Any]]] = None

                for line in response.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    if line == "data: [DONE]":
                        break
                    try:
                        data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    choices = data.get("choices") or []
                    if not choices:
                        usage = data.get("usage")
                        if usage:
                            usage_data = usage
                        continue
                    delta = choices[0].get("delta") or {}
                    if not delta:
                        continue

                    if delta.get("content"):
                        content_parts.append(delta["content"])

                    reasoning_delta = delta.get("reasoning_content") or delta.get("reasoning")
                    if reasoning_delta:
                        reasoning_parts.append(reasoning_delta)
                    rd = delta.get("reasoning_details")
                    if isinstance(rd, list):
                        reasoning_details_last = rd

                    if delta.get("tool_calls"):
                        for tc in delta["tool_calls"]:
                            idx = tc.get("index", len(tool_calls_acc))
                            while len(tool_calls_acc) <= idx:
                                tool_calls_acc.append({"id": "", "name": "", "arguments": ""})
                            if "id" in tc and tc["id"]:
                                tool_calls_acc[idx]["id"] = tc["id"]
                            func = tc.get("function") or {}
                            if func.get("name"):
                                tool_calls_acc[idx]["name"] = func["name"]
                            if func.get("arguments"):
                                tool_calls_acc[idx]["arguments"] = (
                                    tool_calls_acc[idx]["arguments"] + func["arguments"]
                                )

                    if choices[0].get("finish_reason"):
                        finish_reason = choices[0]["finish_reason"] or ""
                    usage = data.get("usage")
                    if usage:
                        usage_data = usage

            self._request_count += 1
            result = LLMResponse(
                raw=None,
                model=result_model,
                finish_reason=finish_reason,
            )
            result.text = "".join(content_parts)
            if reasoning_parts:
                result.reasoning = "".join(reasoning_parts)
            if reasoning_details_last is not None:
                result.reasoning_details = reasoning_details_last

            for tc in tool_calls_acc:
                if not tc.get("id") and not tc.get("name"):
                    continue
                args_str = tc.get("arguments") or "{}"
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except json.JSONDecodeError:
                    args = {"raw": args_str}
                result.function_calls.append(
                    FunctionCall(
                        id=tc.get("id") or "",
                        name=tc.get("name") or "",
                        arguments=args if isinstance(args, dict) else {},
                    )
                )

            if usage_data:
                input_tokens = usage_data.get("prompt_tokens", 0) or 0
                output_tokens = usage_data.get("completion_tokens", 0) or 0
                cached_tokens = 0
                prompt_details = usage_data.get("prompt_tokens_details") or {}
                if prompt_details:
                    cached_tokens = prompt_details.get("cached_tokens", 0) or 0
                self._input_tokens += input_tokens
                self._output_tokens += output_tokens
                self._cached_tokens += cached_tokens
                self._total_tokens += input_tokens + output_tokens
                result.tokens = {
                    "input": input_tokens,
                    "output": output_tokens,
                    "cached": cached_tokens,
                }
                cost = (input_tokens * 3.0 / 1_000_000) + (output_tokens * 15.0 / 1_000_000)
                self._total_cost += cost
                result.cost = cost

            return result

        except httpx.TimeoutException as e:
            raise LLMError(f"Request timed out: {e}", code="timeout")
        except httpx.ConnectError as e:
            raise LLMError(f"Connection error: {e}", code="connection_error")
        except httpx.HTTPError as e:
            raise LLMError(f"HTTP error: {e}", code="api_error")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_tokens": self._total_tokens,
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "cached_tokens": self._cached_tokens,
            "total_cost": self._total_cost,
            "request_count": self._request_count,
        }

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# Alias for backward compatibility
LiteLLMClient = LLMClient
