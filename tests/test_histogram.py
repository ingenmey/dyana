import unittest

import numpy as np

from analyses.histogram import HistogramND


class HistogramTests(unittest.TestCase):
    def test_simple_histogram_counts_1d_bins(self):
        hist = HistogramND([np.array([0.0, 1.0, 2.0, 3.0])])

        hist.add(np.array([0.2, 0.8, 1.5, 2.9]))

        np.testing.assert_allclose(hist.counts, [2.0, 1.0, 1.0])

    def test_linear_histogram_splits_weight_between_neighbor_bins(self):
        hist = HistogramND([np.array([0.0, 1.0, 2.0])], mode="linear")

        hist.add(np.array([0.25, 0.75]))

        np.testing.assert_allclose(hist.counts, [1.0, 1.0])

    def test_total_normalization_scales_to_requested_total(self):
        hist = HistogramND([np.array([0.0, 1.0, 2.0])])
        hist.counts = np.array([1.0, 3.0])

        hist.normalize(method="total", total=100.0)

        np.testing.assert_allclose(hist.counts, [25.0, 75.0])


if __name__ == "__main__":
    unittest.main()

