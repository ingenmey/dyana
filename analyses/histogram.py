# histogram.py

import numpy as np

class HistogramND:
    def __init__(self, bin_edges: list[np.ndarray]):
        """
        Initialize an N-dimensional histogram.

        Parameters:
            bin_edges: A list of 1D arrays specifying bin edges for each axis.
                       For example, for a 2D histogram:
                       [np.linspace(0, 10, 101), np.linspace(0, 180, 181)]
        """
        self.bin_edges = bin_edges
        self.counts = np.zeros([len(edges) - 1 for edges in bin_edges], dtype=np.int64)

    def add(self, *values: np.ndarray):
        """
        Add metric values to the histogram.

        Parameters:
            *values: Each value is a 1D array of shape (N,), where N is number of samples.
                     The number of arrays must match the dimensionality.
        """
        if len(values) != len(self.bin_edges):
            raise ValueError(f"Expected {len(self.bin_edges)} value arrays, got {len(values)}")

        # Combine the individual arrays into a single (N, D) array
        data = np.stack(values, axis=1)  # shape (N, D)
        hist, _ = np.histogramdd(data, bins=self.bin_edges)
        self.counts += hist.astype(np.int64)

    def normalize(self, method="total", box_volume=None, total=1):
        """
        Normalize the histogram by bin volume or total count.

        method:
            - 'volume': normalize by the hypervolume of each bin (default)
            - 'total': normalize to sum to total
            - None: no normalization

        For RDFs, supply `box_volume` for physical normalization.
        """
        if method == "volume":
            bin_widths = [np.diff(edges) for edges in self.bin_edges]
            volume_grid = np.meshgrid(*bin_widths, indexing='ij')
            volumes = np.ones_like(self.counts, dtype=np.float64)
            for width in volume_grid:
                volumes *= width
            norm = volumes * (box_volume if box_volume else 1)
            self.counts = self.counts / norm

        elif method == "total":
            count_sum = self.counts.sum()
            if count_sum > 0:
                self.counts = self.counts / count_sum * total

    def save_txt(self, filename: str):
        """
        Save the histogram to a tabular text file.

        For 1D: columns are bin_center, count
        For 2D: columns are x_center, y_center, count
        """
        dims = len(self.bin_edges)
        if dims == 1:
            centers = 0.5 * (self.bin_edges[0][1:] + self.bin_edges[0][:-1])
            with open(filename, "w") as f:
                f.write("# bin_center    count\n")
                for x, c in zip(centers, self.counts):
                    f.write(f"{x:.6f} {c:.6f}\n")

        elif dims == 2:
            x_edges, y_edges = self.bin_edges
            x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
            y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
            X, Y = np.meshgrid(x_centers, y_centers, indexing='ij')

            with open(filename, "w") as f:
                f.write("# x_center    y_center    count\n")
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        f.write(f"{X[i,j]:.6f} {Y[i,j]:.6f} {self.counts[i,j]:.6f}\n")

        else:
            raise NotImplementedError("save_txt only implemented for 1D and 2D histograms")

    def save_npy(self, filename: str):
        """Save the histogram counts as a .npy file."""
        np.save(filename, self.counts)

    def save_all(self, basename: str):
        """
        Convenience method to save both .npy and .txt versions.
        """
        self.save_npy(f"{basename}.npy")
        self.save_txt(f"{basename}.dat")

