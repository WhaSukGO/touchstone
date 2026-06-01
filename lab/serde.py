"""Minimal typed (de)serialization for dataclasses, enums, Optional, list, dict.

Keeps the data models DRY: every model is a plain dataclass, and the registry stores
them as a single JSON blob. Reconstruction is type-driven so nested models survive
round-trips through SQLite and across process restarts (context-reset survival)."""
from __future__ import annotations

import dataclasses
import enum
import types
import typing
from typing import Any, Union, get_args, get_origin, get_type_hints


def to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    return obj


def from_dict(cls: Any, data: Any) -> Any:
    if data is None:
        return None

    origin = get_origin(cls)

    # Optional[X] / Union[...] / X | None
    if origin is Union or origin is types.UnionType:
        non_none = [a for a in get_args(cls) if a is not type(None)]  # noqa: E721
        last_err: Exception | None = None
        for arm in non_none:
            try:
                return from_dict(arm, data)
            except Exception as e:  # try the next arm
                last_err = e
        if last_err:
            raise last_err
        return data

    if origin in (list, tuple):
        args = get_args(cls)
        elem = args[0] if args else Any
        return [from_dict(elem, x) for x in data]

    if origin is dict:
        args = get_args(cls)
        vt = args[1] if len(args) == 2 else Any
        return {k: from_dict(vt, v) for k, v in data.items()}

    if origin is typing.Literal:
        return data

    if isinstance(cls, type) and issubclass(cls, enum.Enum):
        return cls(data)

    if dataclasses.is_dataclass(cls):
        hints = get_type_hints(cls)
        kwargs = {}
        for f in dataclasses.fields(cls):
            if f.name in data:
                kwargs[f.name] = from_dict(hints[f.name], data[f.name])
        return cls(**kwargs)

    return data
