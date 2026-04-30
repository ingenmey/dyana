import unittest

import numpy as np

from input_providers import FileInputProvider, NullInputProvider
from workflow_prompts import WorkflowPrompts


class WorkflowPromptsTests(unittest.TestCase):
    def test_prompt_cell_vectors_uses_shared_schema_engine(self):
        provider = FileInputProvider(lines=["10.0", "11.0", "12.0"], fallback=NullInputProvider())
        prompts = WorkflowPrompts(input_provider=provider)

        cell_vectors = prompts.prompt_cell_vectors("xyz")

        np.testing.assert_allclose(cell_vectors, [10.0, 11.0, 12.0])

    def test_prompt_cell_vectors_for_lammps_returns_zero_box_placeholder(self):
        prompts = WorkflowPrompts(input_provider=NullInputProvider())

        cell_vectors = prompts.prompt_cell_vectors("lammps")

        np.testing.assert_allclose(cell_vectors, [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
