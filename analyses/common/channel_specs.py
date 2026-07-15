from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analyses.common.reference_channels import AngleChannel, DistanceChannel
from framework.analysis_params import AtomLabelsParam, BoolParam, ChoiceParam, CompoundParam, FloatParam, Group, IntParam, Variant, When


@dataclass(frozen=True)
class DistanceSpec:
    """Shared configuration for one distance-valued channel."""

    obs_compound_index: int
    ref_labels: list[str]
    obs_labels: list[str]
    include_intramolecular: bool = False
    max_distance: float = 10.0
    bin_count: int = 1000

    def __post_init__(self):
        if self.obs_compound_index < 0:
            raise ValueError("obs_compound_index must be >= 0.")
        if not self.ref_labels:
            raise ValueError("ref_labels must not be empty.")
        if not self.obs_labels:
            raise ValueError("obs_labels must not be empty.")
        if self.max_distance <= 0:
            raise ValueError("max_distance must be positive.")
        if self.bin_count < 1:
            raise ValueError("bin_count must be >= 1.")


@dataclass(frozen=True)
class AngleSpec:
    """Shared configuration for one angle-valued channel."""

    obs_compound_index: int
    ref_base_source: str
    ref_tip_source: str
    ref_base_labels: list[str]
    ref_tip_labels: list[str]
    obs_base_source: str
    obs_tip_source: str
    obs_base_labels: list[str]
    obs_tip_labels: list[str]
    enforce_shared_atom: bool = False
    bin_count: int = 180
    v1_cutoff: float | None = None
    v2_cutoff: float | None = None

    def __post_init__(self):
        if self.obs_compound_index < 0:
            raise ValueError("obs_compound_index must be >= 0.")
        for field_name in ("ref_base_source", "ref_tip_source", "obs_base_source", "obs_tip_source"):
            if getattr(self, field_name) not in {"r", "o"}:
                raise ValueError(f"{field_name} must be 'r' or 'o'.")
        for field_name in ("ref_base_labels", "ref_tip_labels", "obs_base_labels", "obs_tip_labels"):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must not be empty.")
        if self.bin_count < 1:
            raise ValueError("bin_count must be >= 1.")
        if self.v1_cutoff is not None and self.v1_cutoff < 0:
            raise ValueError("v1_cutoff must be >= 0 or None.")
        if self.v2_cutoff is not None and self.v2_cutoff < 0:
            raise ValueError("v2_cutoff must be >= 0 or None.")


def distance_spec_schema(name: str, label: str) -> Group:
    """Prompt schema for one shared distance channel spec."""
    return Group(name=name, config_class=DistanceSpec, steps=_distance_spec_steps(label))


def angle_spec_schema(name: str, label: str) -> Group:
    """Prompt schema for one shared angle channel spec."""
    return Group(name=name, config_class=AngleSpec, steps=_angle_spec_steps(label))


def axis_spec_schema(name: str, label: str) -> Group:
    """Prompt schema for one shared scalar-axis spec selectable by kind."""
    return Group(
        name=name,
        config_class=_build_axis_spec,
        steps=[
            ChoiceParam(
                name="kind",
                prompt=f"Channel kind for {label}?",
                choices=["distance", "angle"],
                default="distance",
            ),
            Variant(
                name=f"{name}_variant",
                selector="kind",
                cases={
                    "distance": _distance_spec_steps(label),
                    "angle": _angle_spec_steps(label),
                },
            ),
        ],
    )


def build_distance_channel(owner, ref_type, ref_key, spec: DistanceSpec, output_name="r/Angstrom"):
    """Resolve one shared distance spec into selections and a DistanceChannel."""
    topology_frame = owner.traj.topology_frame
    (obs_type, obs_key), = owner.resolve_compound_types([spec.obs_compound_index])
    ref_selection = topology_frame.resolve_selection(ref_type, spec.ref_labels)
    obs_selection = topology_frame.resolve_selection(obs_type, spec.obs_labels)
    if len(ref_selection.local_indices) == 0 or len(obs_selection.local_indices) == 0:
        raise ValueError("Distance-axis label selection matched no atoms in the initial frame.")

    channel = DistanceChannel(
        ref_key=ref_key,
        obs_key=obs_key,
        ref_local_indices=ref_selection.local_indices,
        obs_local_indices=obs_selection.local_indices,
        include_intramolecular=spec.include_intramolecular,
        max_distance=spec.max_distance,
        bin_edges=np.linspace(0.0, spec.max_distance, spec.bin_count + 1),
        output_name=output_name,
    )
    return obs_type, obs_key, ref_selection, obs_selection, channel


def build_angle_channel(owner, ref_type, ref_key, spec: AngleSpec, output_name="angle/deg"):
    """Resolve one shared angle spec into selections and an AngleChannel."""
    topology_frame = owner.traj.topology_frame
    (obs_type, obs_key), = owner.resolve_compound_types([spec.obs_compound_index])
    ref_base_selection = topology_frame.resolve_selection(
        ref_type if spec.ref_base_source == "r" else obs_type,
        spec.ref_base_labels,
    )
    ref_tip_selection = topology_frame.resolve_selection(
        ref_type if spec.ref_tip_source == "r" else obs_type,
        spec.ref_tip_labels,
    )
    obs_base_selection = topology_frame.resolve_selection(
        obs_type if spec.obs_base_source == "o" else ref_type,
        spec.obs_base_labels,
    )
    obs_tip_selection = topology_frame.resolve_selection(
        obs_type if spec.obs_tip_source == "o" else ref_type,
        spec.obs_tip_labels,
    )
    if any(
        len(selection.local_indices) == 0
        for selection in (ref_base_selection, ref_tip_selection, obs_base_selection, obs_tip_selection)
    ):
        raise ValueError("Angle-axis label selection matched no atoms in the initial frame.")

    channel = AngleChannel(
        ref_key=ref_key,
        obs_key=obs_key,
        ref_base_source=spec.ref_base_source,
        ref_tip_source=spec.ref_tip_source,
        obs_base_source=spec.obs_base_source,
        obs_tip_source=spec.obs_tip_source,
        ref_base_local_indices=ref_base_selection.local_indices,
        ref_tip_local_indices=ref_tip_selection.local_indices,
        obs_base_local_indices=obs_base_selection.local_indices,
        obs_tip_local_indices=obs_tip_selection.local_indices,
        bin_edges=np.linspace(0.0, 180.0, spec.bin_count + 1),
        output_name=output_name,
        enforce_shared_atom=spec.enforce_shared_atom,
        v1_cutoff=spec.v1_cutoff,
        v2_cutoff=spec.v2_cutoff,
    )
    return (
        obs_type,
        obs_key,
        ref_base_selection,
        ref_tip_selection,
        obs_base_selection,
        obs_tip_selection,
        channel,
    )


def _distance_spec_steps(label: str) -> list[object]:
    return [
        CompoundParam(
            name="obs_compound_index",
            role="observed",
            prompt=f"Choose the observed compound for {label} (number): ",
        ),
        AtomLabelsParam(
            name="ref_labels",
            role="reference",
            compound="ref_compound_index",
            prompt=f"Which reference atom(s) define {label}? (comma-separated) ",
        ),
        AtomLabelsParam(
            name="obs_labels",
            role="observed",
            compound="obs_compound_index",
            prompt=f"Which observed atom(s) define {label}? (comma-separated) ",
        ),
        When(
            source="obs_compound_index",
            value_source="ref_compound_index",
            steps=[
                BoolParam(
                    name="include_intramolecular",
                    prompt=f"Include intramolecular distances in {label}?",
                    default=False,
                ),
            ],
        ),
        FloatParam(
            name="max_distance",
            prompt=f"Enter the maximum distance for {label} (in Angstrom): ",
            default=10.0,
            minval=0.1,
        ),
        IntParam(
            name="bin_count",
            prompt=f"Enter the number of bins for {label}: ",
            default=1000,
            minval=1,
        ),
    ]


def _angle_spec_steps(label: str) -> list[object]:
    return [
        CompoundParam(
            name="obs_compound_index",
            role="observed",
            prompt=f"Choose the observed compound for {label} (number): ",
        ),
        ChoiceParam(
            name="ref_base_source",
            prompt=f"Base atom of the first vector for {label}?",
            choices=["r", "o"],
            default="r",
        ),
        ChoiceParam(
            name="ref_tip_source",
            prompt=f"Tip atom of the first vector for {label}?",
            choices=["r", "o"],
            default="r",
        ),
        AtomLabelsParam(
            name="ref_base_labels",
            prompt=f"Which atom(s) are at the base of the first vector for {label}? ",
        ),
        AtomLabelsParam(
            name="ref_tip_labels",
            prompt=f"Which atom(s) are at the tip of the first vector for {label}? ",
        ),
        ChoiceParam(
            name="obs_base_source",
            prompt=f"Base atom of the second vector for {label}?",
            choices=["r", "o"],
            default="o",
        ),
        ChoiceParam(
            name="obs_tip_source",
            prompt=f"Tip atom of the second vector for {label}?",
            choices=["r", "o"],
            default="o",
        ),
        AtomLabelsParam(
            name="obs_base_labels",
            prompt=f"Which atom(s) are at the base of the second vector for {label}? ",
        ),
        AtomLabelsParam(
            name="obs_tip_labels",
            prompt=f"Which atom(s) are at the tip of the second vector for {label}? ",
        ),
        When(
            source="ref_tip_labels",
            op="unordered==",
            value_source="obs_base_labels",
            steps=[
                BoolParam(
                    name="enforce_shared_atom",
                    prompt="Should the tip atom of the reference vector and the base atom of the observed vector be the same atom?",
                    default=True,
                ),
            ],
        ),
        IntParam(
            name="bin_count",
            prompt=f"Enter the number of bins for {label}: ",
            default=180,
            minval=1,
        ),
        FloatParam(
            name="v1_cutoff",
            prompt=f"Enter maximum length for the first vector of {label}: ",
            default=None,
            display_default="None",
            minval=0.0,
            allow_none=True,
        ),
        FloatParam(
            name="v2_cutoff",
            prompt=f"Enter maximum length for the second vector of {label}: ",
            default=None,
            display_default="None",
            minval=0.0,
            allow_none=True,
        ),
    ]


def _build_axis_spec(**kwargs):
    kind = kwargs.pop("kind")
    if kind == "distance":
        return DistanceSpec(**kwargs)
    if kind == "angle":
        return AngleSpec(**kwargs)
    raise ValueError(f"Unsupported axis kind: {kind!r}")
