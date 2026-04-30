import unittest

import numpy as np

from geometry import distance_squared, minimum_image, periodic_center, wrap


class GeometryTests(unittest.TestCase):
    def test_minimum_image_wraps_large_displacement(self):
        box = np.array([10.0, 10.0, 10.0])
        delta = np.array([8.0, -6.0, 1.0])

        np.testing.assert_allclose(minimum_image(delta, box), [-2.0, 4.0, 1.0])

    def test_distance_squared_uses_periodic_boundary(self):
        box = np.array([10.0, 10.0, 10.0])

        self.assertAlmostEqual(distance_squared([9.5, 0.0, 0.0], [0.5, 0.0, 0.0], box), 1.0)

    def test_wrap_keeps_coordinates_inside_box(self):
        box = np.array([10.0, 10.0, 10.0])

        np.testing.assert_allclose(wrap(np.array([-1.0, 10.5, 5.0]), box), [9.0, 0.5, 5.0])

    def test_periodic_center_unwraps_around_first_point(self):
        box = np.array([10.0, 10.0, 10.0])
        coords = np.array([[9.8, 0.0, 0.0], [0.2, 0.0, 0.0]])

        np.testing.assert_allclose(periodic_center(coords, box), [0.0, 0.0, 0.0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()

