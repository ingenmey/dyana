from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analyses.common.base_analysis import BaseAnalysis
from analyses.common.channel_specs import AngleSpec, angle_spec_schema, build_angle_channel
from analyses.common.histogram import HistogramND
from analyses.common.reference_channels import angular_inverse_sin_weights
from framework.analysis_params import CompoundParam
from io_support.console import console
from io_support.output_writer import build_output_filename, format_selection, write_histogram_1d


@dataclass(frozen=True)
class ADFConfig:
    """Configuration for ADF analysis."""

    ref_compound_index: int
    axis: AngleSpec

    def __post_init__(self):
        if self.ref_compound_index < 0:
            raise ValueError("ref_compound_index must be >= 0.")


class ADF(BaseAnalysis):
    """Angular distribution function analysis."""

    CONFIG_CLASS = ADFConfig
    CONFIG_SCHEMA = [
        CompoundParam(name="ref_compound_index", role="reference"),
        angle_spec_schema(name="axis", label="the ADF angle channel"),
    ]

    def configure(self, config: ADFConfig):
        self.bind_config(config)
        (self.ref_type, self.ref_key), = self.resolve_compound_types([self.ref_compound_index])
        (
            self.obs_type,
            self.obs_key,
            self.ref_base_selection,
            self.ref_tip_selection,
            self.obs_base_selection,
            self.obs_tip_selection,
            self.channel,
        ) = build_angle_channel(self, self.ref_type, self.ref_key, self.axis, output_name="angle/deg")
        self.angle_edges = self.channel.bin_edges
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
            angles = self.channel.values_for_reference(batch, ref_molecule_index)
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

        self.hist.normalize(method="total", total=self.axis.bin_count * 100)
        fname = build_output_filename(
            "adf",
            [
                format_selection(self.axis.ref_base_labels, self.ref_type.formula if self.axis.ref_base_source == "r" else self.obs_type.formula),
                format_selection(self.axis.ref_tip_labels, self.ref_type.formula if self.axis.ref_tip_source == "r" else self.obs_type.formula),
                format_selection(self.axis.obs_base_labels, self.obs_type.formula if self.axis.obs_base_source == "o" else self.ref_type.formula),
                format_selection(self.axis.obs_tip_labels, self.obs_type.formula if self.axis.obs_tip_source == "o" else self.ref_type.formula),
            ],
        )
        write_histogram_1d(fname, self.hist, headers=["angle/deg", "ADF"])
        console.success(f"Saved ADF results to {fname}")
