from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Param:
    name: str
    prompt: str | None = None
    default: Any = None
    display_default: Any = None
    required: bool = True


@dataclass(frozen=True)
class CompoundParam(Param):
    role: str = "reference"
    multi: bool = False


@dataclass(frozen=True)
class AtomLabelsParam(Param):
    role: str = "reference"
    compound: str | None = None
    allow_empty: bool = False


@dataclass(frozen=True)
class IntParam(Param):
    minval: int | None = None
    maxval: int | None = None


@dataclass(frozen=True)
class FloatParam(Param):
    minval: float | None = None
    maxval: float | None = None


@dataclass(frozen=True)
class BoolParam(Param):
    pass


@dataclass(frozen=True)
class ChoiceParam(Param):
    choices: list[str] = field(default_factory=list)
