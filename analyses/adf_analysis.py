from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from framework.analysis_params import AtomLabelsParam, BoolParam, ChoiceParam, CompoundParam, FloatParam, IntParam, When
from analyses.common.base_analysis import BaseAnalysis
from analyses.common.histogram import HistogramND
from analyses.common.reference_channels import AngleChannel, angular_inverse_sin_weights
from io_support.console import console
from io_support.output_writer import build_output_filename, format_selection, write_histogram_1d


@dataclass(frozen=True)
class ADFConfig:
    """Configuration for ADF analysis."""

    ref_compound_index: int
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
        if self.ref_compound_index < 0:
            raise ValueError("ref_compound_index must be >= 0.")
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


class ADF(BaseAnalysis):
    """Angular distribution function analysis."""

    CONFIG_CLASS = ADFConfig
    CONFIG_SCHEMA = [
        CompoundParam(name="ref_compound_index", role="reference"),
        CompoundParam(name="obs_compound_index", role="observed"),
        ChoiceParam(name="ref_base_source", prompt="Base atom of first vector?", choices=["r", "o"], default="r"),
        ChoiceParam(name="ref_tip_source", prompt="Tip atom of first vector?", choices=["r", "o"], default="r"),
        AtomLabelsParam(name="ref_base_labels", prompt="Which atom(s) are at the base of the first vector? "),
        AtomLabelsParam(name="ref_tip_labels", prompt="Which atom(s) are at the tip of the first vector? "),
        ChoiceParam(name="obs_base_source", prompt="Base atom of second vector?", choices=["r", "o"], default="o"),
        ChoiceParam(name="obs_tip_source", prompt="Tip atom of second vector?", choices=["r", "o"], default="o"),
        AtomLabelsParam(name="obs_base_labels", prompt="Which atom(s) are at the base of the second vector? "),
        AtomLabelsParam(name="obs_tip_labels", prompt="Which atom(s) are at the tip of the second vector? "),
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
        IntParam(name="bin_count", prompt="Enter the number of bins for ADF calculation: ", default=180, minval=1),
        FloatParam(name="v1_cutoff", prompt="Enter maximum length for the first vector: ", default=None, display_default="None", minval=0.0, allow_none=True),
        FloatParam(name="v2_cutoff", prompt="Enter maximum length for the second vector: ", default=None, display_default="None", minval=0.0, allow_none=True),
    ]

    def configure(self, config: ADFConfig):
        self.bind_config(config)
        (self.ref_type, self.ref_key), = self.resolve_compound_types([self.ref_compound_index])
        (self.obs_type, self.obs_key), = self.resolve_compound_types([self.obs_compound_index])
        topology_frame = self.traj.topology_frame

        self.ref_base_selection = topology_frame.resolve_selection(
            self.ref_type if self.ref_base_source == "r" else self.obs_type,
            self.ref_base_labels,
        )
        self.ref_tip_selection = topology_frame.resolve_selection(
            self.ref_type if self.ref_tip_source == "r" else self.obs_type,
            self.ref_tip_labels,
        )
        self.obs_base_selection = topology_frame.resolve_selection(
            self.obs_type if self.obs_base_source == "o" else self.ref_type,
            self.obs_base_labels,
        )
        self.obs_tip_selection = topology_frame.resolve_selection(
            self.obs_type if self.obs_tip_source == "o" else self.ref_type,
            self.obs_tip_labels,
        )

        self.angle_edges = np.linspace(0, 180, self.bin_count + 1)
        self.channel = AngleChannel(
            ref_key=self.ref_key,
            obs_key=self.obs_key,
            ref_base_source=self.ref_base_source,
            ref_tip_source=self.ref_tip_source,
            obs_base_source=self.obs_base_source,
            obs_tip_source=self.obs_tip_source,
            ref_base_local_indices=self.ref_base_selection.local_indices,
            ref_tip_local_indices=self.ref_tip_selection.local_indices,
            obs_base_local_indices=self.obs_base_selection.local_indices,
            obs_tip_local_indices=self.obs_tip_selection.local_indices,
            bin_edges=self.angle_edges,
            output_name="angle/deg",
            enforce_shared_atom=self.enforce_shared_atom,
            v1_cutoff=self.v1_cutoff,
            v2_cutoff=self.v2_cutoff,
        )
        self.rebuild_runtime_state()
        if any(arr.size == 0 for arr in (self.ref_base_ids, self.ref_tip_ids, self.obs_base_ids, self.obs_tip_ids)):
            raise ValueError("No angle vectors matched the given labels in the initial frame.")

        topology_frame = self.traj.topology_frame
        self.n_ref = topology_frame.get_molecule_count(self.ref_type)
        self.n_obs = topology_frame.get_molecule_count(self.obs_type)
        self.hist = HistogramND([self.angle_edges])
        self.mark_configured()

    def rebuild_runtime_state(self):
        self.channel.rebuild_runtime_state(self.traj)
        self.ref_base_ids = self.channel.ref_base_ids
        self.ref_tip_ids = self.channel.ref_tip_ids
        self.obs_base_ids = self.channel.obs_base_ids
        self.obs_tip_ids = self.channel.obs_tip_ids

    def post_compound_update(self):
        topology_frame = self.traj.topology_frame
        if not topology_frame.has_compound_type_key(self.ref_key) or not topology_frame.has_compound_type_key(self.obs_key):
            return False
        self.ref_type = topology_frame.get_compound_type_by_key(self.ref_key)
        self.obs_type = topology_frame.get_compound_type_by_key(self.obs_key)

        self.rebuild_runtime_state()
        if any(arr.size == 0 for arr in (self.ref_base_ids, self.ref_tip_ids, self.obs_base_ids, self.obs_tip_ids)):
            return False

        self.n_ref = (
            self.n_ref * self.processed_frames + topology_frame.get_molecule_count(self.ref_type)
        ) / (self.processed_frames + 1)
        self.n_obs = (
            self.n_obs * self.processed_frames + topology_frame.get_molecule_count(self.obs_type)
        ) / (self.processed_frames + 1)
        return True

    def process_frame(self):
        batch = self.channel.build_batch(self.traj)
        all_angles = []
        for ref_molecule_index in range(batch.n_references):
            angles = self.channel.samples_for_reference(batch, ref_molecule_index).values
            if angles.size:
                all_angles.append(angles)
        if all_angles:
            self.hist.add(np.concatenate(all_angles))

    def postprocess(self):
        has_data = self.hist.counts.sum() > 0
        if not has_data:
            console.warn("No ADF values were accumulated.")
            return

        sin_weights = angular_inverse_sin_weights(self.angle_edges)

        self.hist.counts = self.hist.counts.astype(np.float64)
        self.hist.counts *= sin_weights
        self.hist.counts /= (self.processed_frames * self.n_ref * self.n_obs)

        self.hist.normalize(method="total", total=self.bin_count * 100)
        fname = build_output_filename(
            "adf",
            [
                format_selection(self.ref_base_labels, self.ref_type.formula if self.ref_base_source == "r" else self.obs_type.formula),
                format_selection(self.ref_tip_labels, self.ref_type.formula if self.ref_tip_source == "r" else self.obs_type.formula),
                format_selection(self.obs_base_labels, self.obs_type.formula if self.obs_base_source == "o" else self.ref_type.formula),
                format_selection(self.obs_tip_labels, self.obs_type.formula if self.obs_tip_source == "o" else self.ref_type.formula),
            ],
        )
        write_histogram_1d(fname, self.hist, headers=["angle/deg", "ADF"])
        console.success(f"Saved ADF results to {fname}")
