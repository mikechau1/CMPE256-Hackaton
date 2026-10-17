"""Thin wrapper over the Anthropic SDK: structured outputs, tool loops, usage tracking."""
from __future__ import annotations

import json
import threading
from collections.abc import Callable
from typing import Any, TypeVar

import anthropic
from pydantic import BaseModel

from . import config

T = TypeVar("T", bound=BaseModel)

# Models that accept the server-side ``fallbacks: "default"`` parameter.
_FALLBACK_MODELS = {"claude-sonnet-5-5", "claude-opus-5-5", "claude-opus-5", "claude-fable-5-1"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

ToolHandler = Callable[[str, dict], str]


class LLM:
    def __init__(self, client: anthropic.Anthropic | None = None):
        self.client = client or anthropic.Anthropic()
        self._lock = threading.Lock()
        self.usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
                      "cache_creation_input_tokens": 0, "calls": 0}

    # -- internals ----------------------------------------------------------
    def _extra(self, model: str) -> dict:
        kw: dict[str, Any] = {"cache_control": {"type": "ephemeral"}}
        if config.USE_FALLBACKS and model in _FALLBACK_MODELS:
            kw["betas"] = [_FALLBACK_BETA]
            kw["fallbacks"] = "default"
        return kw

    def _track(self, resp) -> None:
        u = resp.usage
        with self._lock:
            self.usage["calls"] += 1
            self.usage["input_tokens"] += u.input_tokens or 0
            self.usage["output_tokens"] += u.output_tokens or 0
            self.usage["cache_read_input_tokens"] += getattr(u, "cache_read_input_tokens", 0) or 0
            self.usage["cache_creation_input_tokens"] += getattr(u, "cache_creation_input_tokens", 0) or 0

    @staticmethod
    def _check(resp) -> None:
        if resp.stop_reason == "refusal":
            cat = getattr(getattr(resp, "stop_details", None), "category", None)
            raise RuntimeError(f"Model declined the request (category={cat}).")

    # -- public API ---------------------------------------------------------
    def parse(
        self,
        output_format: type[T],
        prompt: str,
        *,
        system: str = "",
        model: str | None = None,
        effort: str = "medium",
        max_tokens: int = 8000,
    ) -> T:
        """Single call returning a validated pydantic instance."""
        model = model or config.AGENT_MODEL
        resp = self.client.beta.messages.parse(
            model=model,
            max_tokens=max_tokens,
            system=system or anthropic.NOT_GIVEN,
            messages=[{"role": "user", "content": prompt}],
            output_format=output_format,
            output_config={"effort": effort},
            **self._extra(model),
        )
        self._track(resp)
        self._check(resp)
        if resp.parsed_output is None:
            raise RuntimeError(f"No structured output (stop_reason={resp.stop_reason}).")
        return resp.parsed_output

    def text(self, prompt: str, *, system: str = "", model: str | None = None,
             effort: str = "low", max_tokens: int = 4000) -> str:
        model = model or config.AGENT_MODEL
        resp = self.client.beta.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system or anthropic.NOT_GIVEN,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": effort},
            **self._extra(model),
        )
        self._track(resp)
        self._check(resp)
        return "".join(b.text for b in resp.content if b.type == "text")

    def tool_loop(
        self,
        output_format: type[T],
        prompt: str,
        *,
        system: str,
        tools: list[dict],
        handler: ToolHandler,
        model: str | None = None,
        effort: str = "medium",
        max_turns: int = config.MAX_AGENT_TOOL_TURNS,
        max_tokens: int = 16000,
        on_tool: Callable[[str, dict], None] | None = None,
    ) -> T:
        """Manual agentic loop: run tools until Claude returns the structured answer.

        Tool calls in one assistant turn are executed in order and returned
        together in a single user message.
        """
        model = model or config.AGENT_MODEL
        messages: list[dict] = [{"role": "user", "content": prompt}]
        for turn in range(max_turns + 1):
            final_turn = turn == max_turns
            resp = self.client.beta.messages.parse(
                model=model,
                max_tokens=max_tokens,
                system=system,
                tools=tools,
                # On the last turn, withhold tools so the model must answer.
                tool_choice={"type": "none"} if final_turn else {"type": "auto"},
                messages=messages,
                output_format=output_format,
                output_config={"effort": effort},
                **self._extra(model),
            )
            self._track(resp)
            self._check(resp)
            tool_uses = [b for b in resp.content if b.type == "tool_use"]
            if not tool_uses:
                if resp.parsed_output is not None:
                    return resp.parsed_output
                if resp.stop_reason == "max_tokens":
                    raise RuntimeError("Agent answer truncated at max_tokens.")
                messages.append({"role": "assistant", "content": resp.content})
                messages.append({"role": "user", "content": "Return your final answer in the required JSON format."})
                continue
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for tu in tool_uses:
                args = tu.input if isinstance(tu.input, dict) else json.loads(tu.input)
                if on_tool:
                    on_tool(tu.name, args)
                try:
                    out = handler(tu.name, args)
                    results.append({"type": "tool_result", "tool_use_id": tu.id, "content": out})
                except Exception as e:  # report tool failures back to the model
                    results.append({"type": "tool_result", "tool_use_id": tu.id,
                                    "content": f"Error: {e}", "is_error": True})
            messages.append({"role": "user", "content": results})
        raise RuntimeError("Agent did not produce a final answer.")


_default: LLM | None = None


def get_llm() -> LLM:
    global _default
    if _default is None:
        _default = LLM()
    return _default
