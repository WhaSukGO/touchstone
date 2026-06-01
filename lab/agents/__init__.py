"""Agent SDK wiring: real Claude sessions behind the harness's Planner/Evaluator seams.

  SdkPlanner   — proposes ExperimentContracts (the reasoning/generator seat)
  SdkEvaluator — independent, separate-context, skeptical (composes ScriptEvaluator)
  run_agent    — sync wrapper over claude-agent-sdk query() with token accounting

The SDK call is injectable (run_fn) so all glue is testable without API spend."""
from __future__ import annotations

from .committee import Committee, Expert
from .evaluator import SdkEvaluator
from .planner import SdkPlanner
from .sdk import AgentResult, DEFAULT_MODEL, RunFn, run_agent

__all__ = ["Committee", "Expert", "SdkEvaluator", "SdkPlanner", "AgentResult",
           "DEFAULT_MODEL", "RunFn", "run_agent"]
