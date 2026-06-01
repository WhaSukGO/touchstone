"""JSON schemas for structured agent output (SDK output_format).

The planner returns a contract matching CONTRACT_SCHEMA, which maps 1:1 onto our
ExperimentContract dataclass via serde.from_dict. The evaluator returns a JUDGMENT."""
from __future__ import annotations

_CRITERION = {
    "type": "object",
    "properties": {
        "metric": {"type": "string"},
        "op": {"type": "string", "enum": [">=", "<=", ">", "<", "==", "~="]},
        "value": {"type": "number"},
        "tolerance": {"type": "number"},
    },
    "required": ["metric", "op", "value"],
}

CONTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "success_definition": {"type": "string"},
        "gradable_criteria": {"type": "array", "items": _CRITERION},
        "framework": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "version": {"type": "string"},
                           "cuda": {"type": "string"}},
            "required": ["name", "version", "cuda"],
        },
        "datasets": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "source": {"type": "string"},
                               "held_out": {"type": "boolean"}},
                "required": ["name", "source"],
            },
        },
        "command": {"type": "string"},
        "eval_command": {"type": "string"},
        "code_dir": {"type": "string"},
        "seed": {"type": "integer"},
        "oracle": {
            "type": "object",
            "properties": {"criterion": _CRITERION, "source": {"type": "string"}},
            "required": ["criterion", "source"],
        },
        "budget": {
            "type": "object",
            "properties": {"max_tokens": {"type": "integer"},
                           "max_wall_s": {"type": "number"},
                           "max_retries": {"type": "integer"}},
        },
    },
    "required": ["success_definition", "gradable_criteria", "framework", "datasets",
                 "command", "eval_command"],
}

# Skeptical judgment from the independent evaluator (used only AFTER the oracle passes;
# the evaluator can downgrade a passed measurement to FAIL but never the reverse).
JUDGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["PASS", "FAIL"]},
        "rationale": {"type": "string"},
        "concerns": {"type": "array", "items": {"type": "string"}},
        "leakage_suspected": {"type": "boolean"},
    },
    "required": ["verdict", "rationale"],
}

# Planner's follow-up decision.
DECIDE_SCHEMA = {
    "type": "object",
    "properties": {
        "propose_followup": {"type": "boolean"},
        "next_id": {"type": "string"},
        "hypothesis": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": ["propose_followup"],
}
