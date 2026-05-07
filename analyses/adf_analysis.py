from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analysis_params import AtomLabelsParam, BoolParam, ChoiceParam, CompoundParam, FloatParam, IntParam, When
from analyses.base_analysis import BaseAnalysis
from analyses.histogram import HistogramND
from analyses.metrics import AngleMetric, Selector
from analyses.selection import find_matching_labels
from output_writer import build_output_filename, format_selection, write_histogram_1d


@dataclass(frozen=True)
class ADFConfig:
    ref_compound_index: int
    obs_compound_index: int
    ref_base_source: str
    ref_tip_source: str
    ref_base_labels: list[str]
    ref_tip_labels: list[str]
    obs_base_source: str
    obs_tip_source: str
    obs_base_labels: list[str]
    obs_tip_labels: list[str]
    enforce_shared_atom: bool = False
    bin_count: int = 180
    v1_cutoff: float | None = None
    v2_cutoff: float | None = None

    def __post_init__(self):
        if self.ref_compound_index < 0:
            raise ValueError("ref_compound_index must be >= 0.")
        if self.obs_compound_index < 0:
            raise ValueError("obs_compound_index must be >= 0.")
        for field_name in ("ref_base_source", "ref_tip_source", "obs_base_source", "obs_tip_source"):
            if getattr(self, field_name) not in {"r", "o"}:
                raise ValueError(f"{field_name} must be 'r' or 'o'.")
        for field_name in ("ref_base_labels", "ref_tip_labels", "obs_base_labels", "obs_tip_labels"):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must not be empty.")
        if self.bin_count < 1:
            raise ValueError("bin_count must be >= 1.")
        if self.v1_cutoff is not None and self.v1_cutoff < 0:
            raise ValueError("v1_cutoff must be >= 0 or None.")
        if self.v2_cutoff is not None and self.v2_cutoff < 0:
            raise ValueError("v2_cutoff must be >= 0 or None.")


class ADF(BaseAnalysis):
    CONFIG_CLASS = ADFConfig
    CONFIG_SCHEMA = [
        CompoundParam(name="ref_compound_index", role="reference"),
        CompoundParam(name="obs_compound_index", role="observed"),
        ChoiceParam(name="ref_base_source", prompt="Base atom of first vector?", choices=["r", "o"], default="r"),
        ChoiceParam(name="ref_tip_source", prompt="Tip atom of first vector?", choices=["r", "o"], default="r"),
        AtomLabelsParam(name="ref_base_labels", prompt="Which atom(s) are at the base of the first vector? "),
        AtomLabelsParam(name="ref_tip_labels", prompt="Which atom(s) are at the tip of the first vector? "),
        ChoiceParam(name="obs_base_source", prompt="Base atom of second vector?", choices=["r", "o"], default="o"),
        ChoiceParam(name="obs_tip_source", prompt="Tip atom of second vector?", choices=["r", "o"], default="o"),
        AtomLabelsParam(name="obs_base_labels", prompt="Which atom(s) are at the base of the second vector? "),
        AtomLabelsParam(name="obs_tip_labels", prompt="Which atom(s) are at the tip of the second vector? "),
        When(
            source="ref_tip_labels",
            op="unordered==",
            value_source="obs_base_labels",
            steps=[
                BoolParam(
                    name="enforce_shared_atom",
                    prompt="Should the tip atom of the reference vector and the base atom of the observed vector be the same atom?",
                    default=True,
                ),
            ],
        ),
        IntParam(name="bin_count", prompt="Enter the number of bins for ADF calculation: ", default=180, minval=1),
        FloatParam(name="v1_cutoff", prompt="Enter maximum length for the first vector: ", default=None, display_default="None", minval=0.0, allow_none=True),
        FloatParam(name="v2_cutoff", prompt="Enter maximum length for the second vector: ", default=None, display_default="None", minval=0.0, allow_none=True),
    ]

    def configure(self, config: ADFConfig):
        self.config = config
        (self.ref_comp, self.ref_key), = self.resolve_compounds([config.ref_compound_index])
        (self.obs_comp, self.obs_key), = self.resolve_compounds([config.obs_compound_index])

        self.ref_base_source = config.ref_base_source
        self.ref_tip_source = config.ref_tip_source
        self.ref_base_labels = list(config.ref_base_labels)
        self.ref_tip_labels = list(config.ref_tip_labels)
        self.obs_base_source = config.obs_base_source
        self.obs_tip_source = config.obs_tip_source
        self.obs_base_labels = list(config.obs_base_labels)
        self.obs_tip_labels = list(config.obs_tip_labels)
        self.enforce_shared_atom = config.enforce_shared_atom
        self.bin_count = config.bin_count
        self.v1_cutoff = config.v1_cutoff
        self.v2_cutoff = config.v2_cutoff

        self.rebuild_runtime_state()
        if not all([self.ref_base_ids, self.ref_tip_ids, self.obs_base_ids, self.obs_tip_ids]):
            raise ValueError("No angle vectors matched the given labels in the initial frame.")

        self.n_ref = len(self.ref_comp.members)
        self.n_obs = len(self.obs_comp.members)
        self.angle_edges = np.linspace(0, 180, self.bin_count + 1)
        self.hist = HistogramND([self.angle_edges])
        self.mark_configured()

    def rebuild_runtime_state(self):
        self.ref_base_ids, self.ref_tip_ids, self.obs_base_ids, self.obs_tip_ids = build_vector_lists(
            self.ref_comp,
            self.obs_comp,
            self.ref_base_source,
            self.ref_tip_source,
            self.obs_base_source,
            self.obs_tip_source,
            self.ref_base_labels,
            self.ref_tip_labels,
            self.obs_base_labels,
            self.obs_tip_labels,
            self.enforce_shared_atom,
        )

        self.metric = AngleMetric(
            selector_ref_base=Selector(np.array(self.ref_base_ids)),
            selector_ref_tip=Selector(np.array(self.ref_tip_ids)),
            selector_obs_base=Selector(np.array(self.obs_base_ids)),
            selector_obs_tip=Selector(np.array(self.obs_tip_ids)),
            box=self.traj.box_size,
            enforce_shared_atom=self.enforce_shared_atom,
            v1_cutoff=self.v1_cutoff,
            v2_cutoff=self.v2_cutoff,
        )

    def post_compound_update(self):
        try:
            self.ref_comp, self.obs_comp = self.reattach_compounds([self.ref_key, self.obs_key])
        except KeyError:
            return False

        self.rebuild_runtime_state()
        if not all([self.ref_base_ids, self.ref_tip_ids, self.obs_base_ids, self.obs_tip_ids]):
            return False

        self.n_ref = (self.n_ref * self.processed_frames + len(self.ref_comp.members)) / (self.processed_frames + 1)
        self.n_obs = (self.n_obs * self.processed_frames + len(self.obs_comp.members)) / (self.processed_frames + 1)
        return True

    def process_frame(self):
        angles = self.metric(self.traj.coords)
        self.hist.add(angles)

    def postprocess(self):
        bin_centers = 0.5 * (self.angle_edges[1:] + self.angle_edges[:-1])
        radians = np.deg2rad(bin_centers)
        sin_weights = 1.0 / np.sin(radians)

        self.hist.counts = self.hist.counts.astype(np.float64)
        self.hist.counts *= sin_weights

        if self.processed_frames > 0:
            self.hist.counts /= (self.processed_frames * self.n_ref * self.n_obs)

        self.hist.normalize(method="total", total=self.bin_count * 100)
        fname = build_output_filename(
            "adf",
            [
                format_selection(self.ref_base_labels, self.ref_comp.rep),
                format_selection(self.ref_tip_labels, self.ref_comp.rep),
                format_selection(self.obs_base_labels, self.obs_comp.rep),
                format_selection(self.obs_tip_labels, self.obs_comp.rep),
            ],
        )
        write_histogram_1d(fname, self.hist)
        print(f"Saved ADF results to {fname}")

def build_vector_lists(
    ref_comp,
    obs_comp,
    ref_base_source,
    ref_tip_source,
    obs_base_source,
    obs_tip_source,
    ref_base_labels,
    ref_tip_labels,
    obs_base_labels,
    obs_tip_labels,
    enforce_shared_atom,
):
    ref_base_ids, ref_tip_ids = [], []
    obs_base_ids, obs_tip_ids = [], []

    for ref_mol in ref_comp.members:
        for obs_mol in obs_comp.members:
            if ref_mol == obs_mol:
                continue

            rb_mol = ref_mol if ref_base_source == "r" else obs_mol
            rt_mol = ref_mol if ref_tip_source == "r" else obs_mol
            ob_mol = obs_mol if obs_base_source == "o" else ref_mol
            ot_mol = obs_mol if obs_tip_source == "o" else ref_mol

            rb_ids = find_matching_labels(rb_mol, ref_base_labels)
            rt_ids = find_matching_labels(rt_mol, ref_tip_labels)
            ob_ids = find_matching_labels(ob_mol, obs_base_labels)
            ot_ids = find_matching_labels(ot_mol, obs_tip_labels)

            for rb in rb_ids:
                for rt in rt_ids:
                    for ob in ob_ids:
                        if enforce_shared_atom and rt != ob:
                            continue
                        for ot in ot_ids:
                            ref_base_ids.append(rb)
                            ref_tip_ids.append(rt)
                            obs_base_ids.append(ob)
                            obs_tip_ids.append(ot)

    return ref_base_ids, ref_tip_ids, obs_base_ids, obs_tip_ids
