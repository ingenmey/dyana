from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analyses.common.base_analysis import BaseAnalysis
from analyses.common.channel_specs import AngleSpec, DistanceSpec, axis_spec_schema, build_angle_channel, build_distance_channel
from analyses.common.histogram import HistogramND
from analyses.common.reference_channels import AngleChannel, DistanceChannel, ReferenceSamples
from framework.analysis_params import BoolParam, ChoiceParam, CompoundParam, When
from io_support.console import console
from io_support.output_writer import build_output_filename, format_selection, write_table

DistanceAxisConfig = DistanceSpec
AngleAxisConfig = AngleSpec


@dataclass(frozen=True)
class CDFConfig:
    """Configuration for a 2D combined distribution function."""

    ref_compound_index: int
    x_axis: DistanceSpec | AngleSpec
    y_axis: DistanceSpec | AngleSpec
    tuple_mode: str = "same_context"
    exclude_identical_contexts: bool = False

    def __post_init__(self):
        if self.ref_compound_index < 0:
            raise ValueError("ref_compound_index must be >= 0.")
        if self.tuple_mode not in {"same_context", "second_context"}:
            raise ValueError("tuple_mode must be 'same_context' or 'second_context'.")
        if self.tuple_mode != "second_context" and self.exclude_identical_contexts:
            raise ValueError("exclude_identical_contexts is only valid with tuple_mode='second_context'.")


def build_2d_tuples(
    x_samples: ReferenceSamples,
    y_samples: ReferenceSamples,
    mode: str,
    exclude_identical_contexts: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build 2D tuples for one reference molecule from grouped channel samples."""
    if mode not in {"same_context", "second_context"}:
        raise ValueError(f"Unsupported CDF tuple mode: {mode}")

    x_values = []
    y_values = []

    if mode == "same_context":
        y_by_context = {context.context_id: context for context in y_samples.contexts}
        for x_context in x_samples.contexts:
            y_context = y_by_context.get(x_context.context_id)
            if y_context is None:
                continue
            x_context_values = x_context.filtered_values()
            y_context_values = y_context.filtered_values()
            if x_context_values.size == 0 or y_context_values.size == 0:
                continue
            x_values.append(np.repeat(x_context_values, y_context_values.size))
            y_values.append(np.tile(y_context_values, x_context_values.size))
    else:
        for x_context in x_samples.contexts:
            x_context_values = x_context.filtered_values()
            if x_context_values.size == 0:
                continue
            for y_context in y_samples.contexts:
                if exclude_identical_contexts and x_context.context_id == y_context.context_id:
                    continue
                y_context_values = y_context.filtered_values()
                if y_context_values.size == 0:
                    continue
                x_values.append(np.repeat(x_context_values, y_context_values.size))
                y_values.append(np.tile(y_context_values, x_context_values.size))

    if not x_values:
        empty = np.array([], dtype=np.float64)
        return empty, empty

    return np.concatenate(x_values), np.concatenate(y_values)


class CDFAnalysis(BaseAnalysis):
    """Two-dimensional combined distribution function over two scalar axes."""

    CONFIG_CLASS = CDFConfig
    CONFIG_SCHEMA = [
        CompoundParam(name="ref_compound_index", role="reference"),
        axis_spec_schema(name="x_axis", label="the X axis"),
        axis_spec_schema(name="y_axis", label="the Y axis"),
        ChoiceParam(
            name="tuple_mode",
            prompt="How should both axes be combined?",
            choices=["same_context", "second_context"],
            default="same_context",
        ),
        When(
            source="tuple_mode",
            value="second_context",
            steps=[
                BoolParam(
                    name="exclude_identical_contexts",
                    prompt="Exclude identical observed contexts when both axes use the same observed family?",
                    default=True,
                ),
            ],
        ),
    ]

    def configure(self, config: CDFConfig):
        self.bind_config(config)
        (self.ref_type, self.ref_key), = self.resolve_compound_types([self.ref_compound_index])
        self.x_channel = self._build_channel(self.x_axis)
        self.y_channel = self.x_channel if self.x_axis == self.y_axis else self._build_channel(self.y_axis)
        if self.tuple_mode == "same_context" and self.x_channel.obs_key != self.y_channel.obs_key:
            raise ValueError("tuple_mode='same_context' requires both axes to use the same observed compound type.")
        self.exclude_same_contexts = (
            self.exclude_identical_contexts
            and self.x_channel.obs_key == self.y_channel.obs_key
        )

        self.rebuild_runtime_state()
        if not self._channel_is_ready(self.x_channel) or not self._channel_is_ready(self.y_channel):
            raise ValueError("One or both CDF axes matched no values in the initial frame.")

        self.hist = HistogramND([self.x_channel.bin_edges, self.y_channel.bin_edges], mode="simple")
        self.mark_configured()

    def _build_channel(self, axis_config: DistanceSpec | AngleSpec):
        if isinstance(axis_config, DistanceSpec):
            return build_distance_channel(self, self.ref_type, self.ref_key, axis_config, output_name="r/Angstrom")[-1]
        return build_angle_channel(self, self.ref_type, self.ref_key, axis_config, output_name="angle/deg")[-1]

    def _channel_is_ready(self, channel):
        if isinstance(channel, DistanceChannel):
            return channel.ref_atom_ids.size > 0 and channel.obs_atom_ids.size > 0
        return all(
            arr.size > 0
            for arr in (channel.ref_base_ids, channel.ref_tip_ids, channel.obs_base_ids, channel.obs_tip_ids)
        )

    def rebuild_runtime_state(self):
        for channel in self._channels():
            channel.rebuild_runtime_state(self.traj)

    def post_compound_update(self):
        topology_frame = self.traj.topology_frame
        if not topology_frame.has_compound_type_key(self.ref_key):
            return False
        if any(
            not topology_frame.has_compound_type_key(channel.obs_key)
            for channel in self._channels()
        ):
            return False

        self.ref_type = topology_frame.get_compound_type_by_key(self.ref_key)
        self.rebuild_runtime_state()
        return self._channel_is_ready(self.x_channel) and self._channel_is_ready(self.y_channel)

    def process_frame(self):
        batch = self.x_channel.build_batch(self.traj)
        for channel in self._channels():
            begin_batch = getattr(channel, "begin_batch", None)
            if begin_batch is not None:
                begin_batch(batch)

        x_values = []
        y_values = []
        for ref_molecule_index in range(batch.n_references):
            ref_x = self.x_channel.samples_for_reference(batch, ref_molecule_index)
            if self.x_channel is self.y_channel:
                ref_y = ref_x
            else:
                ref_y = self.y_channel.samples_for_reference(batch, ref_molecule_index)
            pair_x, pair_y = build_2d_tuples(
                ref_x,
                ref_y,
                mode=self.tuple_mode,
                exclude_identical_contexts=self.exclude_same_contexts,
            )
            if pair_x.size == 0:
                continue
            x_values.append(pair_x)
            y_values.append(pair_y)

        if x_values:
            self.hist.add(np.concatenate(x_values), np.concatenate(y_values))

    def postprocess(self):
        if self.hist.counts.sum() == 0:
            console.warn("No CDF values were accumulated.")
            return

        x_factors = self.x_channel.axis_normalization_factors()
        if x_factors is None:
            x_factors = np.ones(self.hist.counts.shape[0], dtype=np.float64)
        y_factors = self.y_channel.axis_normalization_factors()
        if y_factors is None:
            y_factors = np.ones(self.hist.counts.shape[1], dtype=np.float64)

        norm_grid = np.outer(np.asarray(x_factors, dtype=np.float64), np.asarray(y_factors, dtype=np.float64))
        corrected = np.divide(
            self.hist.counts.astype(np.float64),
            norm_grid,
            out=np.zeros_like(self.hist.counts, dtype=np.float64),
            where=norm_grid != 0,
        )
        total = corrected.sum()
        if total > 0:
            corrected /= total
        self.hist.counts = corrected * 10000

        x_centers = 0.5 * (self.hist.bin_edges[0][1:] + self.hist.bin_edges[0][:-1])
        y_centers = 0.5 * (self.hist.bin_edges[1][1:] + self.hist.bin_edges[1][:-1])
        joint_rows = [
            [x_center, y_center, self.hist.counts[ix, iy]]
            for ix, x_center in enumerate(x_centers)
            for iy, y_center in enumerate(y_centers)
        ]

        joint_filename = build_output_filename(
            "cdf_joint",
            [self._axis_filename_part("x", self.x_axis, self.x_channel), self._axis_filename_part("y", self.y_axis, self.y_channel)],
        )
        write_table(
            joint_filename,
            headers=[self.x_channel.output_name, self.y_channel.output_name, "P"],
            data=joint_rows,
            comment_lines=[f"tuple_mode={self.tuple_mode}"],
        )
        console.success(f"Saved joint CDF results to {joint_filename}")

        x_marginal = self.hist.counts.sum(axis=1)
        y_marginal = self.hist.counts.sum(axis=0)
        x_filename = build_output_filename("cdf_x", [self._axis_filename_part("x", self.x_axis, self.x_channel)])
        y_filename = build_output_filename("cdf_y", [self._axis_filename_part("y", self.y_axis, self.y_channel)])
        write_table(
            x_filename,
            headers=[self.x_channel.output_name, "P"],
            data=[[x_center, x_marginal[index]] for index, x_center in enumerate(x_centers)],
        )
        console.success(f"Saved X-axis CDF marginal to {x_filename}")
        write_table(
            y_filename,
            headers=[self.y_channel.output_name, "P"],
            data=[[y_center, y_marginal[index]] for index, y_center in enumerate(y_centers)],
        )
        console.success(f"Saved Y-axis CDF marginal to {y_filename}")

    def _axis_filename_part(self, axis_name, axis_config, channel):
        if isinstance(axis_config, DistanceSpec):
            return "_".join(
                [
                    axis_name,
                    "dist",
                    format_selection(axis_config.ref_labels, self.ref_type.formula),
                    format_selection(axis_config.obs_labels, channel.obs_type.formula),
                ]
            )
        return "_".join(
            [
                axis_name,
                "angle",
                format_selection(
                    axis_config.ref_base_labels,
                    self.ref_type.formula if axis_config.ref_base_source == "r" else channel.obs_type.formula,
                ),
                format_selection(
                    axis_config.ref_tip_labels,
                    self.ref_type.formula if axis_config.ref_tip_source == "r" else channel.obs_type.formula,
                ),
                format_selection(
                    axis_config.obs_base_labels,
                    channel.obs_type.formula if axis_config.obs_base_source == "o" else self.ref_type.formula,
                ),
                format_selection(
                    axis_config.obs_tip_labels,
                    channel.obs_type.formula if axis_config.obs_tip_source == "o" else self.ref_type.formula,
                ),
            ]
        )

    def _channels(self):
        if self.x_channel is self.y_channel:
            return (self.x_channel,)
        return (self.x_channel, self.y_channel)
