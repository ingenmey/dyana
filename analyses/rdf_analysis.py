# analyses/rdf_analysis.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from framework.analysis_params import AtomLabelsParam, CompoundParam, FloatParam, IntParam
from analyses.common.base_analysis import BaseAnalysis
from analyses.common.histogram import HistogramND
from analyses.common.reference_channels import DistanceChannel, radial_shell_volumes
from io_support.console import console
from io_support.output_writer import build_output_filename, format_selection, write_histogram_1d


@dataclass(frozen=True)
class RDFConfig:
    """Configuration for RDF analysis."""

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
    """Radial distribution function analysis."""

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
        self.bind_config(config)
        (self.ref_type, self.ref_key), = self.resolve_compound_types([self.ref_compound_index])
        (self.obs_type, self.obs_key), = self.resolve_compound_types([self.obs_compound_index])
        topology_frame = self.traj.topology_frame

        self.ref_selection = topology_frame.resolve_selection(self.ref_type, self.ref_labels)
        self.obs_selection = topology_frame.resolve_selection(self.obs_type, self.obs_labels)

        edges = np.linspace(0.0, self.max_distance, self.bin_count + 1)
        self.channel = DistanceChannel(
            ref_key=self.ref_key,
            obs_key=self.obs_key,
            ref_local_indices=self.ref_selection.local_indices,
            obs_local_indices=self.obs_selection.local_indices,
            max_distance=self.max_distance,
            bin_edges=edges,
            output_name="r/Angstrom",
        )
        self.rebuild_runtime_state()
        self.n_ref = int(self.ref_indices.size)
        self.n_obs = int(self.obs_indices.size)
        self.hist = HistogramND([edges], "linear")
        self.box_volume = np.prod(self.traj.box_size)
        self.mark_configured()

    def rebuild_runtime_state(self):
        self.channel.rebuild_runtime_state(self.traj)
        self.ref_indices = self.channel.ref_atom_ids
        self.obs_indices = self.channel.obs_atom_ids

    def post_compound_update(self):
        topology_frame = self.traj.topology_frame
        if not topology_frame.has_compound_type_key(self.ref_key) or not topology_frame.has_compound_type_key(self.obs_key):
            return False
        self.ref_type = topology_frame.get_compound_type_by_key(self.ref_key)
        self.obs_type = topology_frame.get_compound_type_by_key(self.obs_key)

        self.rebuild_runtime_state()
        self.n_ref = (
            (self.n_ref * self.processed_frames + int(self.ref_indices.size))
            / (self.processed_frames + 1)
        )
        self.n_obs = (
            (self.n_obs * self.processed_frames + int(self.obs_indices.size))
            / (self.processed_frames + 1)
        )
        return True

    def process_frame(self):
        batch = self.channel.build_batch(self.traj)
        self.channel.begin_batch(batch)
        all_values = []
        for ref_molecule_index in range(batch.n_references):
            values = self.channel.samples_for_reference(batch, ref_molecule_index).values
            if values.size:
                all_values.append(values)
        if all_values:
            self.hist.add(np.concatenate(all_values))

    def postprocess(self):
        has_data = self.hist.counts.sum() > 0
        if not has_data:
            console.warn("No RDF values were accumulated.")
            return

        bin_edges = self.hist.bin_edges[0]
        shell_volumes = radial_shell_volumes(bin_edges)
        norm_factor = self.n_ref * self.n_obs * self.processed_frames
        self.hist.counts = self.hist.counts / (shell_volumes * norm_factor / self.box_volume)

        obs_density = self.n_obs / self.box_volume if self.box_volume else 0.0
        number_integral = obs_density * np.cumsum(self.hist.counts * shell_volumes)
        self.hist.data["number_integral"] = number_integral

        fname = build_output_filename(
            "rdf",
            [
                format_selection(self.ref_labels, self.ref_type.formula),
                format_selection(self.obs_labels, self.obs_type.formula),
            ],
        )
        write_histogram_1d(fname, self.hist, headers=["r/Angstrom", "g(r)", "N(r)"], fields=["count", "number_integral"])

        console.success(f"Saved RDF results to {fname}")
