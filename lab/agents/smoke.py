"""Live smoke test for the Agent SDK wiring. COSTS A SMALL AMOUNT (real API calls).

  python -m lab.agents.smoke

Makes two independent structured calls and prints their session ids (to prove isolation)
and token usage (to prove accounting). Needs ANTHROPIC_API_KEY in the environment."""
from __future__ import annotations

import sys

from .sdk import run_agent

_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "integer"}, "word": {"type": "string"}},
    "required": ["answer", "word"],
}


def main() -> int:
    print("Call A (independent session)...")
    a = run_agent("Return answer=2+2 and word='alpha'.", schema=_SCHEMA, max_turns=2)
    print(f"  data={a.data} tokens=in:{a.usage.tokens_in}/out:{a.usage.tokens_out} "
          f"cost=${a.cost_usd:.4f} error={a.is_error}")

    print("Call B (separate session; must NOT see Call A)...")
    b = run_agent("Return answer=10*10 and word='beta'. "
                  "If you recall any earlier number, ignore it.",
                  schema=_SCHEMA, max_turns=2)
    print(f"  data={b.data} tokens=in:{b.usage.tokens_in}/out:{b.usage.tokens_out} "
          f"cost=${b.cost_usd:.4f} error={b.is_error}")

    ok = (not a.is_error and not b.is_error
          and isinstance(a.data, dict) and isinstance(b.data, dict)
          and a.data.get("answer") == 4 and b.data.get("answer") == 100)
    print("SMOKE:", "OK (isolated sessions + structured output + token accounting)"
          if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
