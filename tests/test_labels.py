import unittest

from utils import label_matches


class LabelMatchingTests(unittest.TestCase):
    def test_exact_label_with_number_requires_exact_match(self):
        self.assertTrue(label_matches("O1", "O1"))
        self.assertFalse(label_matches("O1", "O2"))
        self.assertFalse(label_matches("O1", "O10"))

    def test_element_prefix_matches_numbered_labels(self):
        self.assertTrue(label_matches("O", "O1"))
        self.assertTrue(label_matches("Na", "Na1"))
        self.assertFalse(label_matches("O", "Na1"))


if __name__ == "__main__":
    unittest.main()

