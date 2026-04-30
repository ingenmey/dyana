import unittest
from dataclasses import dataclass

from analysis_params import AtomLabelsParam, BoolParam, ChoiceParam, CompoundParam, FloatParam, IntParam
from config_builder import prompt_config_from_schema
from input_providers import FileInputProvider, NullInputProvider


@dataclass(frozen=True)
class DummyConfig:
    ref_compound_index: int
    ref_labels: list[str]
    cutoff: float
    n_bins: int
    enabled: bool
    axis: str


class DummyCompound:
    def __init__(self, rep, comp_id):
        self.rep = rep
        self.comp_id = comp_id


class DummyAnalysis:
    def __init__(self, provider=None):
        self.input_provider = provider
        self.compounds = [DummyCompound("H2O", 0), DummyCompound("Na+", 1)]

    def compound_selection(self, role="reference", multi=False, prompt_text=None, provider=None):
        input_provider = provider or self.input_provider
        prompt = prompt_text or f"Choose the {role} compound (number): "
        idx = input_provider.ask_int(prompt, 1, minval=1) - 1
        return idx, self.compounds[idx]

    def compound_by_index(self, index):
        return self.compounds[index]

    def atom_selection(self, role="reference", compound=None, prompt_text=None, allow_empty=False, provider=None):
        input_provider = provider or self.input_provider
        prompt = prompt_text or f"Which atom(s) in {role} compound {compound.comp_id + 1} ({compound.rep})? (comma-separated) "
        answer = input_provider.ask_str(prompt, default="" if allow_empty else None)
        return [s.strip() for s in answer.split(",") if s.strip()]


class ConfigBuilderTests(unittest.TestCase):
    def test_prompt_config_from_schema_builds_typed_config(self):
        provider = FileInputProvider(
            lines=["1", "O,H", "3.5", "12", "y", "z"],
            fallback=NullInputProvider(),
        )
        analysis = DummyAnalysis(provider=provider)
        schema = [
            CompoundParam(name="ref_compound_index", role="reference"),
            AtomLabelsParam(name="ref_labels", role="reference", compound="ref_compound_index"),
            FloatParam(name="cutoff", prompt="Cutoff?", default=5.0, minval=0.1),
            IntParam(name="n_bins", prompt="Bins?", default=100, minval=1),
            BoolParam(name="enabled", prompt="Enable option?", default=False),
            ChoiceParam(name="axis", prompt="Axis?", choices=["x", "y", "z"], default="x"),
        ]

        config = prompt_config_from_schema(analysis, schema, DummyConfig, provider=provider)

        self.assertEqual(
            config,
            DummyConfig(
                ref_compound_index=0,
                ref_labels=["O", "H"],
                cutoff=3.5,
                n_bins=12,
                enabled=True,
                axis="z",
            ),
        )


if __name__ == "__main__":
    unittest.main()
