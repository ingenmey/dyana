# histogram.py

import numpy as np

class HistogramND:
    """
    N-dimensional histogram supporting multiple data fields, with simple and weighted binning modes.

    Parameters
    ----------
    bin_edges : list of np.ndarray
        List of 1D arrays specifying bin edges for each axis/dimension. The number
        of arrays determines the histogram dimension (1D, 2D, etc).

    mode : {"simple", "linear"}, optional
        Histogram binning mode:
            - "simple": Standard binning (default, supports any dimension).
            - "linear": Linear/weighted binning (currently supports only 1D histograms).
              Each value is distributed between two nearest bins according to proximity.

    Attributes
    ----------
    bin_edges : list of np.ndarray
        List of 1D arrays specifying bin edges for each axis.
    mode : str
        Binning mode.
    data : dict
        Dictionary mapping field names (str) to data arrays of shape determined by bin_edges.
        By default, a single field "count" is present and used for most operations.
    counts : np.ndarray (property)
        Shortcut property for self.data["count"], i.e., the primary/default data field.

    Methods
    -------
    add(*values, field="count")
        Add data points to the histogram for the specified field, using the chosen binning mode.
        *values should be arrays, one for each dimension, all length N.
    add_data_field(field, values=None)
        Add an additional data field (e.g., for auxiliary or post-processed values).
        If values is None, initializes the field to zeros.
    normalize(field="count", method="total", box_volume=None, total=1)
        Normalize the specified data field by total counts, bin volume, or no normalization.
    save_txt(filename, headers=None, fields=None)
        Save the histogram (one or more fields) as a plain-text file, with customizable headers.
    save_npy(filename)
        Save the default ("count") field to a .npy file.
    save_all(basename)
        Save both .npy and .dat versions (default field only).

    Example
    -------
    >>> hist = HistogramND([np.linspace(0, 1, 11)], mode="linear")
    >>> hist.add(np.random.rand(1000))
    >>> hist.normalize()
    >>> hist.save_txt("myhist.dat")
    """

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
        """np.ndarray: The primary data field, same as self.data['count'] (read/write property)."""
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
        """
        Add values to the histogram for a specified field.

        Parameters
        ----------
        *values : np.ndarray
            One array per dimension (each of shape (N,)), specifying data points to bin.
        field : str, optional
            Name of the data field to update (default is "count").
        """

        if len(values) != len(self.bin_edges):
            raise ValueError(f"Expected {len(self.bin_edges)} value arrays, got {len(values)}")

        if self.mode == "simple":
            self._add_simple(*values, field=field)
        elif self.mode == "linear":
            self._add_linear(*values, field=field)
        else:
            raise ValueError(f"Unknown histogram mode: {self.mode}")

    def _add_simple(self, *values, field):
        data = np.stack(values, axis=1)  # shape (N, D)
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
        """
        Normalize the specified histogram field.

        Parameters
        ----------
        field : str, optional
            Field to normalize (default: "count").
        method : {"volume", "total", None}, optional
            - "volume": Normalize by per-bin volume (density).
            - "total":  Normalize so that sum of field equals `total` (default: 1).
            - None:     No normalization applied.
        box_volume : float, optional
            If specified, scales densities by box volume (for "volume" normalization).
        total : float, optional
            The value to normalize the total to (default: 1).
        """
        if method == "volume":
            bin_widths = [np.diff(edges) for edges in self.bin_edges]
            mesh = np.meshgrid(*bin_widths, indexing='ij')
            volumes = np.ones_like(self.data[field], dtype=np.float64)
            for bw in mesh:
                volumes *= bw
            norm = volumes * (box_volume if box_volume else 1)
            self.data[field] = self.data[field] / norm

        elif method == "total":
            count_sum = self.data[field].sum()
            if count_sum > 0:
                self.data[field] = self.data[field] / count_sum * total

    def save_txt(self, filename: str, headers=None, fields=None):
        """
        Save histogram bin centers and specified fields to a text file.

        Parameters
        ----------
        filename : str
            Output file path.
        headers : list of str, optional
            Column headers (one per dimension plus one per field). If None, defaults
            to ['bin_0', ..., 'bin_N', ...fields].
        fields : list of str, optional
            List of field names to write. If None, all fields in self.data are saved.

        Notes
        -----
        The first column is left-aligned; all others are decimal-aligned for easier reading.
        """
        dims = len(self.bin_edges)
        bin_centers = [0.5 * (edges[1:] + edges[:-1]) for edges in self.bin_edges]
        mesh = np.meshgrid(*bin_centers, indexing="ij")
        flat_centers = [m.flatten() for m in mesh]

        # Choose which fields to write (defaults to all in self.data, in insertion order)
        if fields is None:
            fields = list(self.data.keys())
        flat_data = [self.data[field].flatten() for field in fields]

        if headers is None:
            headers = [f"bin_{i}" for i in range(dims)] + list(fields)
        elif len(headers) != dims + len(fields):
            raise ValueError(f"headers must have {dims+len(fields)} entries, got {len(headers)}.")

        with open(filename, "w") as f:
            f.write("# " + f"{headers[0]:<12}" + " ".join(f"{h:>12}" for h in headers[1:]) + "\n")
            for row in zip(*flat_centers, *flat_data):
                center_str = f"{row[0]:<12.6f}" + "".join(f" {v:>12.6f}" for v in row[1:])
                f.write(center_str + "\n")

