from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from framework.analysis_params import BoolParam, ChoiceParam, FloatParam
from analyses.common.base_analysis import BaseAnalysis
from analyses.common.histogram import HistogramND
from io_support.output_writer import build_output_filename, write_histogram_1d


@dataclass(frozen=True)
class DensityConfig:
    """Configuration for one-dimensional density analysis."""

    axis: str = "z"
    step_size: float = 0.1
    per_compound_normalization: bool = False

    def __post_init__(self):
        if self.axis not in {"x", "y", "z"}:
            raise ValueError("axis must be one of 'x', 'y', or 'z'.")
        if self.step_size <= 0:
            raise ValueError("step_size must be positive.")


class DensityAnalysis(BaseAnalysis):
    """One-dimensional density profile over molecule centers of mass."""

    CONFIG_CLASS = DensityConfig
    CONFIG_SCHEMA = [
        ChoiceParam(
            name="axis",
            prompt="Choose the axis for density analysis",
            choices=["x", "y", "z"],
            default="z",
        ),
        FloatParam(
            name="step_size",
            prompt="Enter the step size for density calculation (in Angstrom): ",
            default=0.1,
            minval=1e-12,
        ),
        BoolParam(
            name="per_compound_normalization",
            prompt="Normalize each compound using only the frames in which it appeared?",
            default=False,
        ),
    ]

    def configure(self, config: DensityConfig):
        self.bind_config(config)
        self.axis_index = {"x": 0, "y": 1, "z": 2}[self.axis]

        self.box_length = self.traj.box_size[self.axis_index]
        self.num_bins = int(np.ceil(self.box_length / self.step_size))
        self.edges = np.arange(self.num_bins + 1) * self.step_size

        self.hist = HistogramND([self.edges], mode="linear")
        self.all_compounds = {}
        if self.per_compound_normalization:
            self.compound_frame_counts = {}

        for compound_type in self.get_compound_types():
            self.hist.add_data_field(field=compound_type.formula)
            self.all_compounds[compound_type.key] = compound_type.formula
            if self.per_compound_normalization:
                self.compound_frame_counts[compound_type.key] = 0

        self.mark_configured()

    def post_compound_update(self):
        for compound_type in self.get_compound_types():
            if compound_type.formula not in self.hist.data:
                self.hist.add_data_field(field=compound_type.formula)
                self.all_compounds[compound_type.key] = compound_type.formula
            if self.per_compound_normalization:
                if compound_type.key not in self.compound_frame_counts:
                    self.compound_frame_counts[compound_type.key] = 0
        return True

    def process_frame(self):
        topology_frame = self.traj.topology_frame
        for compound_type in self.get_compound_types():
            coms = topology_frame.get_molecule_coms(
                compound_type,
                coords=self.traj.coords,
                box_size=self.traj.box_size,
            )[:, self.axis_index]
            if len(coms) > 0:
                self.hist.add(coms, field=compound_type.formula)

            if self.per_compound_normalization:
                self.compound_frame_counts[compound_type.key] = self.compound_frame_counts.get(compound_type.key, 0) + 1

    def postprocess(self):
        for comp_key, formula in self.all_compounds.items():
            if self.per_compound_normalization:
                frames = self.compound_frame_counts.get(comp_key, 0)
                if frames == 0:
                    continue
                self.hist.data[formula] /= frames
            else:
                self.hist.data[formula] /= self.processed_frames

        sorted_formulas = [self.all_compounds[k] for k in sorted(self.all_compounds)]
        headers = ["r/Angstrom"] + sorted_formulas
        fname = build_output_filename("density")
        write_histogram_1d(fname, self.hist, headers=headers, fields=sorted_formulas)
        print(f"Saved density results to {fname}")
