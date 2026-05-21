import unittest
from dataclasses import dataclass

from framework.analysis_params import (
    AtomLabelsParam,
    BoolParam,
    ChoiceParam,
    CompoundParam,
    FloatParam,
    ForEach,
    Group,
    IntParam,
    Repeat,
    Variant,
    When,
)
from framework.config_builder import prompt_config_from_schema
from io_support.input_providers import FileInputProvider, NullInputProvider


@dataclass(frozen=True)
class DummyConfig:
    ref_compound_index: int
    ref_labels: list[str]
    cutoff: float
    n_bins: int
    enabled: bool
    axis: str


@dataclass(frozen=True)
class DynamicConfig:
    obs_compound_indices: list[int]
    obs_labels_per_compound: dict[int, list[str]]
    enabled: bool
    note: str | None = None


@dataclass(frozen=True)
class SelectionConfig:
    ref_compound_index: int
    ref_labels: list[str]


@dataclass(frozen=True)
class GroupedConfig:
    selection: SelectionConfig
    enabled: bool


@dataclass(frozen=True)
class ObservedSiteConfig:
    compound_index: int
    labels: list[str]
    cutoff: float


@dataclass(frozen=True)
class ForEachConfig:
    obs_compound_indices: list[int]
    observed_sites: list[ObservedSiteConfig]


@dataclass(frozen=True)
class ChannelConfig:
    kind: str
    cutoff: float | None = None
    bins: int | None = None


@dataclass(frozen=True)
class RepeatedConfig:
    channels: list[ChannelConfig]


class DummyCompoundType:
    def __init__(self, formula, type_id, canonical_labels):
        self.formula = formula
        self.type_id = type_id
        self.canonical_labels = canonical_labels


class DummyTopologyFrame:
    def __init__(self, compound_types):
        self.compound_types = compound_types

    def get_compound_type_by_index(self, index):
        return self.compound_types[index]


class DummyTrajectory:
    def __init__(self, compound_types):
        self.topology_frame = DummyTopologyFrame(compound_types)


class DummyAnalysis:
    def __init__(self, provider=None):
        self.input_provider = provider
        self.compound_types = [
            DummyCompoundType("H2O", 0, ("O1", "H1", "H2")),
            DummyCompoundType("Na+", 1, ("Na1",)),
        ]
        self.traj = DummyTrajectory(self.compound_types)

    def compound_selection(self, role="reference", multi=False, prompt_text=None, provider=None):
        input_provider = provider or self.input_provider
        if multi:
            prompt = prompt_text or f"Choose the {role} compounds (comma-separated numbers): "
            choices = input_provider.ask_str(prompt).strip()
            idxs = [int(x.strip()) - 1 for x in choices.split(",") if x.strip()]
            return [(idx, self.compound_types[idx]) for idx in idxs]
        prompt = prompt_text or f"Choose the {role} compound (number): "
        idx = input_provider.ask_int(prompt, 1, minval=1) - 1
        return idx, self.compound_types[idx]

    def atom_selection(self, role="reference", compound=None, prompt_text=None, allow_empty=False, provider=None):
        input_provider = provider or self.input_provider
        prompt = prompt_text or f"Which atom(s) in {role} compound {compound.type_id + 1} ({compound.formula})? (comma-separated) "
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

    def test_prompt_config_from_schema_supports_for_each_and_when(self):
        provider = FileInputProvider(
            lines=["1,2", "O,H", "Na1", "y", "hello"],
            fallback=NullInputProvider(),
        )
        analysis = DummyAnalysis(provider=provider)
        schema = [
            CompoundParam(name="obs_compound_indices", role="observed", multi=True),
            ForEach(
                source="obs_compound_indices",
                item_name="obs_compound_index",
                steps=[
                    AtomLabelsParam(name="obs_labels", role="observed", compound="obs_compound_index"),
                ],
                collect_as="obs_labels_per_compound",
                collect_mode="dict",
            ),
            BoolParam(name="enabled", prompt="Enable option?", default=False),
            When(
                source="enabled",
                op="==",
                value=True,
                steps=[
                    ChoiceParam(name="note", prompt="Note?", choices=["hello", "bye"], default="hello"),
                ],
            ),
        ]

        config = prompt_config_from_schema(analysis, schema, DynamicConfig, provider=provider)

        self.assertEqual(
            config,
            DynamicConfig(
                obs_compound_indices=[0, 1],
                obs_labels_per_compound={0: ["O", "H"], 1: ["Na1"]},
                enabled=True,
                note="hello",
            ),
        )

    def test_prompt_config_from_schema_supports_group(self):
        provider = FileInputProvider(
            lines=["1", "O,H", "y"],
            fallback=NullInputProvider(),
        )
        analysis = DummyAnalysis(provider=provider)
        schema = [
            Group(
                name="selection",
                config_class=SelectionConfig,
                steps=[
                    CompoundParam(name="ref_compound_index", role="reference"),
                    AtomLabelsParam(name="ref_labels", role="reference", compound="ref_compound_index"),
                ],
            ),
            BoolParam(name="enabled", prompt="Enable option?", default=False),
        ]

        config = prompt_config_from_schema(analysis, schema, GroupedConfig, provider=provider)

        self.assertEqual(
            config,
            GroupedConfig(
                selection=SelectionConfig(ref_compound_index=0, ref_labels=["O", "H"]),
                enabled=True,
            ),
        )

    def test_prompt_config_from_schema_supports_typed_for_each_with_item_injection(self):
        provider = FileInputProvider(
            lines=["1,2", "O,H", "3.5", "Na1", "2.4"],
            fallback=NullInputProvider(),
        )
        analysis = DummyAnalysis(provider=provider)
        schema = [
            CompoundParam(name="obs_compound_indices", role="observed", multi=True),
            ForEach(
                source="obs_compound_indices",
                item_name="obs_compound_index",
                steps=[
                    AtomLabelsParam(name="labels", role="observed", compound="obs_compound_index"),
                    FloatParam(name="cutoff", prompt="Cutoff?", default=3.5, minval=0.1),
                ],
                collect_as="observed_sites",
                collect_mode="list",
                config_class=ObservedSiteConfig,
                include_item_as="compound_index",
            ),
        ]

        config = prompt_config_from_schema(analysis, schema, ForEachConfig, provider=provider)

        self.assertEqual(
            config,
            ForEachConfig(
                obs_compound_indices=[0, 1],
                observed_sites=[
                    ObservedSiteConfig(compound_index=0, labels=["O", "H"], cutoff=3.5),
                    ObservedSiteConfig(compound_index=1, labels=["Na1"], cutoff=2.4),
                ],
            ),
        )

    def test_prompt_config_from_schema_supports_repeat_and_variant(self):
        provider = FileInputProvider(
            lines=["distance", "3.5", "y", "angle", "24", "n"],
            fallback=NullInputProvider(),
        )
        analysis = DummyAnalysis(provider=provider)
        schema = [
            Repeat(
                name="channels",
                item_name="channel_index",
                add_prompt="Add another channel?",
                config_class=ChannelConfig,
                steps=[
                    ChoiceParam(
                        name="kind",
                        prompt="Channel kind?",
                        choices=["distance", "angle"],
                        default="distance",
                    ),
                    Variant(
                        name="channel_kind",
                        selector="kind",
                        cases={
                            "distance": [
                                FloatParam(name="cutoff", prompt="Cutoff?", default=3.5, minval=0.1),
                            ],
                            "angle": [
                                IntParam(name="bins", prompt="Bins?", default=18, minval=1),
                            ],
                        },
                    ),
                ],
            ),
        ]

        config = prompt_config_from_schema(analysis, schema, RepeatedConfig, provider=provider)

        self.assertEqual(
            config,
            RepeatedConfig(
                channels=[
                    ChannelConfig(kind="distance", cutoff=3.5, bins=None),
                    ChannelConfig(kind="angle", cutoff=None, bins=24),
                ],
            ),
        )


if __name__ == "__main__":
    unittest.main()
