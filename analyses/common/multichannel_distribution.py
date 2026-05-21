"""Shared multichannel histogram engine for reference-compound-aligned analyses."""

from __future__ import annotations

from typing import Literal

import numpy as np

from analyses.common.histogram import HistogramND
from analyses.common.reference_channels import ChannelSamples
from io_support.output_writer import write_table


CombinationMode = Literal["cartesian", "matched"]


def combine_channel_samples(
    sample_sets: list[ChannelSamples],
    mode: CombinationMode = "cartesian",
) -> np.ndarray:
    """Combine one sample list per channel into tuples for one reference molecule."""
    if not sample_sets:
        return np.empty((0, 0), dtype=np.float64)

    if mode == "cartesian":
        return _combine_cartesian(sample_sets)
    if mode == "matched":
        return _combine_matched(sample_sets)
    raise ValueError(f"Unsupported combination mode: {mode}")


class MultichannelDistributionEngine:
    """Accumulate and export multichannel joint distributions."""

    def __init__(self, channels, combination_mode: CombinationMode = "cartesian"):
        if not channels:
            raise ValueError("At least one channel is required.")

        self.channels = list(channels)
        self.combination_mode = combination_mode
        self.hist = HistogramND([np.asarray(channel.bin_edges) for channel in self.channels], mode="simple")
        self.reference_count = 0
        self.tuple_count = 0

    def process_batch(self, batch) -> None:
        for channel in self.channels:
            begin_batch = getattr(channel, "begin_batch", None)
            if callable(begin_batch):
                begin_batch(batch)

        for ref_molecule_index in range(batch.n_references):
            sample_sets = [channel.samples_for_reference(batch, ref_molecule_index) for channel in self.channels]
            self.add_reference_samples(sample_sets)
            self.reference_count += 1

    def add_reference_samples(self, sample_sets: list[ChannelSamples]) -> int:
        combined = combine_channel_samples(sample_sets, mode=self.combination_mode)
        if combined.size == 0:
            return 0

        self.hist.add(*(combined[:, axis] for axis in range(combined.shape[1])))
        self.tuple_count += int(combined.shape[0])
        return int(combined.shape[0])

    def has_data(self) -> bool:
        return bool(self.tuple_count > 0 and self.hist.counts.sum() > 0)

    def axis_bin_centers(self, axis: int) -> np.ndarray:
        edges = self.hist.bin_edges[axis]
        return 0.5 * (edges[1:] + edges[:-1])

    def joint_headers(self, value_header: str = "count") -> list[str]:
        return [channel.output_name for channel in self.channels] + [value_header]

    def joint_rows(self, normalize: bool = False) -> list[list[float]]:
        if not self.has_data():
            return []

        centers = [self.axis_bin_centers(axis) for axis in range(len(self.channels))]
        nonzero = np.argwhere(self.hist.counts > 0)
        total = float(self.hist.counts.sum())

        rows = []
        for index_tuple in nonzero:
            value = float(self.hist.counts[tuple(index_tuple)])
            if normalize and total > 0:
                value /= total
            row = [float(centers[axis][index_tuple[axis]]) for axis in range(len(self.channels))]
            row.append(value)
            rows.append(row)
        return rows

    def marginal_headers(self, axis: int, normalize: bool = True) -> list[str]:
        axis_name = self.channels[axis].output_name
        return [axis_name, f"P({axis_name})" if normalize else "count"]

    def marginal_rows(self, axis: int, normalize: bool = True) -> list[list[float]]:
        counts = self._marginal_counts(axis)
        centers = self.axis_bin_centers(axis)
        total = float(counts.sum())

        rows = []
        for center, count in zip(centers, counts):
            value = float(count)
            if normalize and total > 0:
                value /= total
            rows.append([float(center), value])
        return rows

    def apply_axis_normalization(self, axis_factors: list[np.ndarray | None]) -> None:
        if len(axis_factors) != len(self.channels):
            raise ValueError("axis_factors must match the number of channels.")

        scale = np.ones_like(self.hist.counts, dtype=np.float64)
        for axis, factors in enumerate(axis_factors):
            if factors is None:
                continue
            factors = np.asarray(factors, dtype=np.float64).reshape(-1)
            expected = self.hist.counts.shape[axis]
            if len(factors) != expected:
                raise ValueError(
                    f"Normalization factors for axis {axis} must have length {expected}, got {len(factors)}."
                )
            reshape = [1] * self.hist.counts.ndim
            reshape[axis] = expected
            scale *= factors.reshape(reshape)

        self.hist.counts = np.divide(
            self.hist.counts,
            scale,
            out=np.zeros_like(self.hist.counts, dtype=np.float64),
            where=scale != 0,
        )

    def apply_channel_axis_normalization(self) -> None:
        axis_factors = [channel.axis_normalization_factors() for channel in self.channels]
        self.apply_axis_normalization(axis_factors)

    def write_joint_table(self, filename: str, normalize: bool = False) -> None:
        write_table(
            filename,
            headers=self.joint_headers(value_header="P" if normalize else "count"),
            data=self.joint_rows(normalize=normalize),
        )

    def write_marginal_table(self, axis: int, filename: str, normalize: bool = True) -> None:
        write_table(
            filename,
            headers=self.marginal_headers(axis, normalize=normalize),
            data=self.marginal_rows(axis, normalize=normalize),
        )

    def _marginal_counts(self, axis: int) -> np.ndarray:
        axes = tuple(idx for idx in range(self.hist.counts.ndim) if idx != axis)
        if not axes:
            return np.asarray(self.hist.counts, dtype=np.float64)
        return self.hist.counts.sum(axis=axes)


def _combine_cartesian(sample_sets: list[ChannelSamples]) -> np.ndarray:
    if any(sample_set.is_empty for sample_set in sample_sets):
        return np.empty((0, len(sample_sets)), dtype=np.float64)

    index_grids = np.meshgrid(*[np.arange(sample_set.size) for sample_set in sample_sets], indexing="ij")
    columns = []
    for axis, sample_set in enumerate(sample_sets):
        columns.append(np.asarray(sample_set.values[index_grids[axis].reshape(-1)], dtype=np.float64))
    return np.column_stack(columns)


def _combine_matched(sample_sets: list[ChannelSamples]) -> np.ndarray:
    if any(sample_set.is_empty for sample_set in sample_sets):
        return np.empty((0, len(sample_sets)), dtype=np.float64)
    if any(sample_set.sample_ids is None for sample_set in sample_sets):
        raise ValueError("Matched combination requires sample_ids for every channel.")

    id_maps = [_build_unique_id_map(sample_set) for sample_set in sample_sets]
    shared_ids = [sample_id for sample_id in id_maps[0] if all(sample_id in id_map for id_map in id_maps[1:])]
    if not shared_ids:
        return np.empty((0, len(sample_sets)), dtype=np.float64)

    rows = []
    for sample_id in shared_ids:
        rows.append([float(id_map[sample_id]) for id_map in id_maps])
    return np.asarray(rows, dtype=np.float64)


def _build_unique_id_map(sample_set: ChannelSamples) -> dict[object, float]:
    id_map = {}
    for sample_id, value in zip(sample_set.sample_ids, sample_set.values):
        if sample_id in id_map:
            raise ValueError("Matched combination requires unique sample_ids within each channel.")
        id_map[sample_id] = value
    return id_map
