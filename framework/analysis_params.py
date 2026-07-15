"""Declarative prompt-schema primitives for migrated analyses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Param:
    """Base parameter spec for one prompted config field."""

    name: str
    prompt: str | None = None
    default: Any = None
    display_default: Any = None


@dataclass(frozen=True)
class CompoundParam(Param):
    """Prompt for one or more compound-type selections."""

    role: str = "reference"
    multi: bool = False


@dataclass(frozen=True)
class AtomLabelsParam(Param):
    """Prompt for one or more atom-label selections."""

    role: str = "reference"
    compound: str | None = None
    allow_empty: bool = False


@dataclass(frozen=True)
class IntParam(Param):
    minval: int | None = None
    maxval: int | None = None


@dataclass(frozen=True)
class IntListParam(Param):
    """Prompt for a comma-separated list of integers."""

    minval: int | None = None
    maxval: int | None = None
    min_items: int = 1


@dataclass(frozen=True)
class FloatParam(Param):
    minval: float | None = None
    maxval: float | None = None
    allow_none: bool = False


@dataclass(frozen=True)
class BoolParam(Param):
    pass


@dataclass(frozen=True)
class ChoiceParam(Param):
    choices: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ForEach:
    """Repeat a prompt sub-schema for each item from a prior value."""

    source: str
    item_name: str
    steps: list[object]
    collect_as: str
    collect_mode: str = "dict"
    config_class: type | None = None
    include_item_as: str | None = None


@dataclass(frozen=True)
class Group:
    """Collect a prompt sub-schema into one nested structured value."""

    name: str
    steps: list[object]
    config_class: type | None = None


@dataclass(frozen=True)
class Repeat:
    """Repeat a prompt sub-schema until the user declines another item."""

    name: str
    item_name: str
    steps: list[object]
    add_prompt: str
    min_items: int = 1
    config_class: type | None = None


@dataclass(frozen=True)
class Variant:
    """Run one config branch selected by an earlier prompted value."""

    name: str
    selector: str
    cases: dict[str, list[object]]


@dataclass(frozen=True)
class When:
    """Conditionally run a prompt sub-schema."""

    source: str
    op: str = "=="
    value: object = True
    value_source: str | None = None
    steps: list[object] = field(default_factory=list)
