# analyses/density_analysis.py

import numpy as np
from utils import prompt, prompt_int, prompt_float, prompt_yn, prompt_choice
from analyses.histogram import HistogramND
from analyses.base_analysis import BaseAnalysis

class DensityAnalysis(BaseAnalysis):
    def setup(self):
        axis = prompt_choice("Choose the axis for density analysis", ["x", "y", "z"], "z")
        self.axis_index = {"x": 0, "y": 1, "z": 2}[axis]
        self.step_size = prompt_float("Enter the step size for density calculation (in Å): ", 0.1)

        self.box_length = self.traj.box_size[self.axis_index]
        self.num_bins = int(np.ceil(self.box_length / self.step_size))
        self.bin_centers = (np.arange(self.num_bins) + 0.5) * self.step_size
        self.edges = np.arange(self.num_bins + 1) * self.step_size

        self.per_compound_normalization = prompt_yn(
            "Normalize each compound using only the frames in which it appeared?", False
        )
        if self.per_compound_normalization:
            self.compound_frame_counts = {comp_key: 0 for comp_key in self.traj.compounds.keys()}

        self.all_compounds = {}  # comp_key -> rep

        # Create a single multi-field histogram for all compounds (fields named by compound rep)
        self.hist = HistogramND([self.edges], mode="linear")
        for comp_key, comp in self.traj.compounds.items():
            self.hist.add_data_field(field=comp.rep)
            self.all_compounds[comp_key] = comp.rep

    def post_compound_update(self):
        # Ensure any new compounds are accounted for
        for comp_key, comp in self.traj.compounds.items():
            if comp.rep not in self.hist.data:
                self.hist.add_data_field(field=comp.rep)
                self.all_compounds[comp_key] = comp.rep
            if self.per_compound_normalization and comp_key not in self.compound_frame_counts:
                self.compound_frame_counts[comp_key] = 0
            if self.per_compound_normalization:
                self.compound_frame_counts[comp_key] += 1
        return True

    def process_frame(self):
        for comp_key, compound in self.traj.compounds.items():
            coms = np.array([mol.com[self.axis_index] for mol in compound.members])
            if len(coms) > 0:
                self.hist.add(coms, field=compound.rep)

    def postprocess(self):
        # Normalize using all compounds seen in any frame
        for comp_key, rep in self.all_compounds.items():
            if self.per_compound_normalization:
                frames = self.compound_frame_counts.get(comp_key, 1)
                self.hist.data[rep] /= frames
            else:
                self.hist.data[rep] /= self.processed_frames

        # Output order by comp_key for reproducibility
        sorted_reps = [self.all_compounds[k] for k in sorted(self.all_compounds)]
        headers = ["r/Å"] + sorted_reps
        fields = sorted_reps
        self.hist.save_txt("density.dat", headers=headers, fields=fields)
        print("\nDensity data saved to 'density.dat'.")
