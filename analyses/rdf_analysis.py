# analyses/rdf_analysis.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analysis_params import AtomLabelsParam, CompoundParam, FloatParam, IntParam
from analyses.base_analysis import BaseAnalysis
from analyses.histogram import HistogramND
from analyses.metrics import DistanceMetric, Selector
from analyses.selection import collect_atom_indices


@dataclass(frozen=True)
class RDFConfig:
    """Configuration for RDF analysis setup."""

    ref_compound_index: int
    obs_compound_index: int
    ref_labels: list[str]
    obs_labels: list[str]
    max_distance: float = 10.0
    bin_count: int = 1000

    def __post_init__(self):
        if self.ref_compound_index < 0:
            raise ValueError("ref_compound_index must be >= 0.")
        if self.obs_compound_index < 0:
            raise ValueError("obs_compound_index must be >= 0.")
        if not self.ref_labels:
            raise ValueError("ref_labels must not be empty.")
        if not self.obs_labels:
            raise ValueError("obs_labels must not be empty.")
        if self.max_distance <= 0:
            raise ValueError("max_distance must be positive.")
        if self.bin_count < 1:
            raise ValueError("bin_count must be >= 1.")


class RDF(BaseAnalysis):
    CONFIG_CLASS = RDFConfig
    CONFIG_SCHEMA = [
        CompoundParam(name="ref_compound_index", role="reference"),
        CompoundParam(name="obs_compound_index", role="observed"),
        AtomLabelsParam(name="ref_labels", role="reference", compound="ref_compound_index"),
        AtomLabelsParam(name="obs_labels", role="observed", compound="obs_compound_index"),
        FloatParam(
            name="max_distance",
            prompt="Enter the maximum distance for RDF calculation (in Angstrom): ",
            default=10.0,
            minval=0.1,
        ),
        IntParam(
            name="bin_count",
            prompt="Enter the number of bins for RDF calculation: ",
            default=1000,
            minval=1,
        ),
    ]

    def configure(self, config: RDFConfig):
        self.config = config
        compounds = self.get_compounds()
        keys = list(self.traj.compounds.keys())

        try:
            self.ref_comp = compounds[config.ref_compound_index]
            self.obs_comp = compounds[config.obs_compound_index]
            self.ref_key = keys[config.ref_compound_index]
            self.obs_key = keys[config.obs_compound_index]
        except IndexError as exc:
            raise ValueError("RDF compound index is out of range.") from exc

        self.ref_labels = list(config.ref_labels)
        self.obs_labels = list(config.obs_labels)
        self.max_distance = config.max_distance
        self.bin_count = config.bin_count

        self.update_selectors()
        self.n_ref = len(self.ref_indices)
        self.n_obs = len(self.obs_indices)
        edges = np.linspace(0.0, self.max_distance, self.bin_count + 1)
        self.hist = HistogramND([edges], "linear")
        self.box_volume = np.prod(self.traj.box_size)
        self.mark_configured()

    def update_selectors(self):
        self.ref_indices = collect_atom_indices(self.ref_comp, self.ref_labels)
        self.obs_indices = collect_atom_indices(self.obs_comp, self.obs_labels)

        self.ref_sel = Selector(np.array(self.ref_indices))
        self.obs_sel = Selector(np.array(self.obs_indices))

        self.metric = DistanceMetric(
            self.ref_sel,
            self.obs_sel,
            self.traj.box_size,
            cutoff=self.max_distance,
        )

    def post_compound_update(self):
        try:
            self.ref_comp = self.traj.compounds[self.ref_key]
            self.obs_comp = self.traj.compounds[self.obs_key]
        except KeyError:
            return False

        self.update_selectors()
        self.n_ref = (
            (self.n_ref * self.processed_frames + len(self.ref_indices))
            / (self.processed_frames + 1)
        )
        self.n_obs = (
            (self.n_obs * self.processed_frames + len(self.obs_indices))
            / (self.processed_frames + 1)
        )
        return True

    def process_frame(self):
        values = self.metric(self.traj.coords)
        self.hist.add(values)

    def postprocess(self):
        bin_edges = self.hist.bin_edges[0]
        shell_volumes = (4.0 / 3.0) * np.pi * (bin_edges[1:] ** 3 - bin_edges[:-1] ** 3)

        if self.processed_frames > 0 and self.n_ref > 0 and self.n_obs > 0:
            norm_factor = self.n_ref * self.n_obs * self.processed_frames
            self.hist.counts = self.hist.counts / (shell_volumes * norm_factor / self.box_volume)
        else:
            self.hist.counts = np.zeros_like(self.hist.counts)

        obs_density = self.n_obs / self.box_volume if self.box_volume else 0.0
        number_integral = obs_density * np.cumsum(self.hist.counts * shell_volumes)
        self.hist.data["number_integral"] = number_integral

        fname = f"rdf_{_label_str(self.ref_labels)}_{_label_str(self.obs_labels)}.dat"
        self.hist.save_txt(fname, ["r/Angstrom", "g(r)", "N(r)"])

        print(f"RDF and number-integral saved to {fname}")


def _label_str(labels):
    return "_".join(label.replace(" ", "") for label in labels)
