"""Shared histogram container used by the supported analyses."""

import numpy as np


class HistogramND:
    """N-dimensional histogram with optional per-field storage."""

    def __init__(self, bin_edges: list[np.ndarray], mode="simple"):
        self.bin_edges = bin_edges
        self.mode = mode
        shape = [len(edges) - 1 for edges in bin_edges]
        self.data = {"count": np.zeros(shape, dtype=np.float64)}

        if mode == "linear" and len(bin_edges) == 1:
            diffs = np.diff(bin_edges[0])
            if not np.allclose(diffs, diffs[0]):
                raise ValueError("Linear interpolation requires uniform bin spacing")
            self._bin_width = diffs[0]
            self._bin_min = bin_edges[0][0]

    @property
    def counts(self):
        return self.data["count"]

    @counts.setter
    def counts(self, value):
        self.data["count"] = value

    def add_data_field(self, field, values=None):
        shape = self.data["count"].shape
        if values is not None:
            values = np.asarray(values)
            if values.shape != shape:
                raise ValueError(f"Provided array shape {values.shape} does not match histogram shape {shape}.")
            self.data[field] = values
        else:
            self.data[field] = np.zeros(shape, dtype=np.float64)

    def add(self, *values: np.ndarray, field="count"):
        if len(values) != len(self.bin_edges):
            raise ValueError(f"Expected {len(self.bin_edges)} value arrays, got {len(values)}")

        if self.mode == "simple":
            self._add_simple(*values, field=field)
        elif self.mode == "linear":
            self._add_linear(*values, field=field)
        else:
            raise ValueError(f"Unknown histogram mode: {self.mode}")

    def _add_simple(self, *values, field):
        data = np.stack(values, axis=1)
        hist, _ = np.histogramdd(data, bins=self.bin_edges)
        self.data[field] += hist

    def _add_linear(self, *values, field):
        if len(values) != 1:
            raise NotImplementedError("Linear binning is currently supported only for 1D histograms")

        x = values[0]
        bin_idx = (x - self._bin_min) / self._bin_width

        lower = np.floor(bin_idx).astype(int)
        upper = lower + 1

        w_upper = bin_idx - lower
        w_lower = 1.0 - w_upper

        valid_lower = (lower >= 0) & (lower < len(self.data[field]))
        valid_upper = (upper >= 0) & (upper < len(self.data[field]))

        np.add.at(self.data[field], lower[valid_lower], w_lower[valid_lower])
        np.add.at(self.data[field], upper[valid_upper], w_upper[valid_upper])

    def normalize(self, field="count", method="total", box_volume=None, total=1):
        if method == "volume":
            bin_widths = [np.diff(edges) for edges in self.bin_edges]
            mesh = np.meshgrid(*bin_widths, indexing="ij")
            volumes = np.ones_like(self.data[field], dtype=np.float64)
            for bw in mesh:
                volumes *= bw
            norm = volumes * (box_volume if box_volume else 1)
            self.data[field] = self.data[field] / norm

        elif method == "total":
            count_sum = self.data[field].sum()
            if count_sum > 0:
                self.data[field] = self.data[field] / count_sum * total
