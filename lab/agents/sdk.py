"""Thin sync wrapper over the Anthropic Agent SDK (claude-agent-sdk).

The harness is synchronous; the SDK is async — this bridges with asyncio.run. Each
run_agent() call uses query(), which starts a FRESH, independent session (no shared
context) — exactly what we need to keep the generator and the evaluator isolated. We
never pass resume/continue_conversation, and setting_sources=[] avoids pulling in
project/user settings, keeping each call reproducible.

Token usage is read from the terminal ResultMessage (input_tokens/output_tokens), which
is how the harness budgets in tokens — not turns. The actual SDK import is lazy so this
module (and the Planner/Evaluator that import it) load even where the SDK is absent;
they are tested with an injected fake run_fn."""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from ..models import Usage

DEFAULT_MODEL = "claude-sonnet-4-6"


@dataclass
class AgentResult:
    data: Any            # parsed dict when a schema is requested, else the text
    text: str
    usage: Usage
    cost_usd: float = 0.0
    is_error: bool = False
    num_turns: int = 0


def _parse_json(text: str) -> dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    i, j = text.find("{"), text.rfind("}")
    if 0 <= i < j:
        try:
            return json.loads(text[i:j + 1])
        except Exception:
            pass
    raise ValueError(f"could not parse JSON from agent response: {text[:200]!r}")


async def _run_async(prompt: str, *, system_prompt: str | None, schema: dict | None,
                     model: str, allowed_tools: list[str] | None, cwd: str | None,
                     max_turns: int, max_budget_usd: float | None) -> AgentResult:
    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

    kwargs: dict[str, Any] = dict(
        model=model,
        max_turns=max_turns,
        permission_mode="bypassPermissions",
        setting_sources=[],                 # isolation: no project/user settings loaded
        allowed_tools=allowed_tools or [],  # pure reasoning by default (no tools)
    )
    if system_prompt is not None:
        kwargs["system_prompt"] = system_prompt
    if cwd is not None:
        kwargs["cwd"] = cwd
    if max_budget_usd is not None:
        kwargs["max_budget_usd"] = max_budget_usd
    if schema is not None:
        kwargs["output_format"] = {"type": "json_schema", "schema": schema}

    options = ClaudeAgentOptions(**kwargs)

    in_tok = out_tok = num_turns = 0
    cost = 0.0
    text = ""
    structured: Any = None
    is_error = False
    async for msg in query(prompt=prompt, options=options):
        if isinstance(msg, ResultMessage):
            u = msg.usage or {}
            # Include cache tokens: the SDK reports most input under cache_* fields, so
            # plain input_tokens undercounts. Budget on the true input cost.
            in_tok = (u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
                      + u.get("cache_creation_input_tokens", 0))
            out_tok = u.get("output_tokens", out_tok)
            cost = msg.total_cost_usd or 0.0
            text = msg.result or ""
            structured = msg.structured_output
            is_error = bool(msg.is_error)
            num_turns = msg.num_turns or 0

    if schema is None:
        data: Any = text
    else:
        data = structured if structured is not None else _parse_json(text)

    return AgentResult(data=data, text=text, usage=Usage(in_tok, out_tok),
                       cost_usd=cost, is_error=is_error, num_turns=num_turns)


def run_agent(prompt: str, *, system_prompt: str | None = None, schema: dict | None = None,
              model: str = DEFAULT_MODEL, allowed_tools: list[str] | None = None,
              cwd: str | None = None, max_turns: int = 6,
              max_budget_usd: float | None = None) -> AgentResult:
    """Run one isolated agent session synchronously and return its result + token usage."""
    return asyncio.run(_run_async(
        prompt, system_prompt=system_prompt, schema=schema, model=model,
        allowed_tools=allowed_tools, cwd=cwd, max_turns=max_turns,
        max_budget_usd=max_budget_usd,
    ))


# The injectable boundary: Planner/Evaluator take run_fn=RunFn so tests pass a fake.
RunFn = Callable[..., AgentResult]
