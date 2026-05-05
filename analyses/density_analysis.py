from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analysis_params import BoolParam, ChoiceParam, FloatParam
from analyses.base_analysis import BaseAnalysis
from analyses.histogram import HistogramND


@dataclass(frozen=True)
class DensityConfig:
    axis: str = "z"
    step_size: float = 0.1
    per_compound_normalization: bool = False

    def __post_init__(self):
        if self.axis not in {"x", "y", "z"}:
            raise ValueError("axis must be one of 'x', 'y', or 'z'.")
        if self.step_size <= 0:
            raise ValueError("step_size must be positive.")


class DensityAnalysis(BaseAnalysis):
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
        self.config = config
        self.axis = config.axis
        self.axis_index = {"x": 0, "y": 1, "z": 2}[config.axis]
        self.step_size = config.step_size
        self.per_compound_normalization = config.per_compound_normalization

        self.box_length = self.traj.box_size[self.axis_index]
        self.num_bins = int(np.ceil(self.box_length / self.step_size))
        self.edges = np.arange(self.num_bins + 1) * self.step_size

        self.hist = HistogramND([self.edges], mode="linear")
        self.all_compounds = {}
        if self.per_compound_normalization:
            self.compound_frame_counts = {}

        for comp_key, comp in self.traj.compounds.items():
            self.hist.add_data_field(field=comp.rep)
            self.all_compounds[comp_key] = comp.rep
            if self.per_compound_normalization:
                self.compound_frame_counts[comp_key] = 0

        self.mark_configured()

    def post_compound_update(self):
        for comp_key, comp in self.traj.compounds.items():
            if comp.rep not in self.hist.data:
                self.hist.add_data_field(field=comp.rep)
                self.all_compounds[comp_key] = comp.rep
            if self.per_compound_normalization:
                if comp_key not in self.compound_frame_counts:
                    self.compound_frame_counts[comp_key] = 0
                self.compound_frame_counts[comp_key] += 1
        return True

    def process_frame(self):
        for compound in self.traj.compounds.values():
            coms = np.array([mol.com[self.axis_index] for mol in compound.members])
            if len(coms) > 0:
                self.hist.add(coms, field=compound.rep)

    def postprocess(self):
        for comp_key, rep in self.all_compounds.items():
            if self.per_compound_normalization:
                frames = self.compound_frame_counts.get(comp_key, 1)
                self.hist.data[rep] /= frames
            else:
                self.hist.data[rep] /= self.processed_frames

        sorted_reps = [self.all_compounds[k] for k in sorted(self.all_compounds)]
        headers = ["r/Angstrom"] + sorted_reps
        self.hist.save_txt("density.dat", headers=headers, fields=sorted_reps)
        print("\nDensity data saved to 'density.dat'.")
