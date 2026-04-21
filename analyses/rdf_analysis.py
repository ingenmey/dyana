# analyses/rdf_analysis.py

import numpy as np
from analyses.base_analysis import BaseAnalysis
from utils import prompt, prompt_int, prompt_float, prompt_yn, label_matches
from analyses.metrics import Selector, DistanceMetric
from analyses.histogram import HistogramND

class RDF(BaseAnalysis):
    def setup(self):
        # --- User setup ---
        ref_idx, self.ref_comp = self.compound_selection("reference")
        obs_idx, self.obs_comp = self.compound_selection("observed")

        self.ref_labels = self.atom_selection("reference")
        self.obs_labels = self.atom_selection("observed")

        self.max_distance = prompt_float("Enter the maximum distance for RDF calculation (in Å): ", 10.0, minval=0.1)
        self.bin_count = prompt_int("Enter the number of bins for RDF calculation: ", 500, minval=1)

        self.ref_key = list(self.traj.compounds.keys())[ref_idx]
        self.obs_key = list(self.traj.compounds.keys())[obs_idx]

        # --- Index precomputation ---
        self.update_selectors()

        self.n_ref = len(self.ref_indices)
        self.n_obs = len(self.obs_indices)
        edges = np.linspace(0, self.max_distance, self.bin_count + 1)
        self.hist = HistogramND([edges], "linear")
        self.box_volume = np.prod(self.traj.box_size)

    def update_selectors(self):
        self.ref_indices = self._get_indices(self.ref_comp, self.ref_labels)
        self.obs_indices = self._get_indices(self.obs_comp, self.obs_labels)

        self.ref_sel = Selector(np.array(self.ref_indices))
        self.obs_sel = Selector(np.array(self.obs_indices))

        self.metric = DistanceMetric(
            self.ref_sel, self.obs_sel,
            self.traj.box_size, cutoff=self.max_distance
        )

    def _get_indices(self, compound, labels):
        return [
            idx for mol in compound.members
            for label, idx in mol.label_to_global_id.items()
            if any(label_matches(lab, label) for lab in labels)
        ]

    def post_compound_update(self):
        try:
            self.ref_comp = self.traj.compounds[self.ref_key]
            self.obs_comp = self.traj.compounds[self.obs_key]
        except KeyError:
            # compound disappeared this frame – skip
            return False

        self.update_selectors()
        self.n_ref = (self.n_ref * self.processed_frames + len(self.ref_indices))/(self.processed_frames + 1)
        self.n_obs = (self.n_obs * self.processed_frames + len(self.obs_indices))/(self.processed_frames + 1)
        return True

    def process_frame(self):
        values = self.metric(self.traj.coords)
        self.hist.add(values)

    def postprocess(self):
        # Normalize RDF
        # Shell volumes per bin
        bin_edges = self.hist.bin_edges[0]
        bin_widths = np.diff(bin_edges)
        shell_volumes = (4/3) * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)

        # Normalize RDF manually
        norm_factor = self.n_ref * self.n_obs * self.processed_frames
        self.hist.counts = self.hist.counts / (shell_volumes * norm_factor / self.box_volume)

        # Number Integral
        obs_density = self.n_obs / self.box_volume
        g_of_r = self.hist.counts
        number_integral = obs_density * np.cumsum(g_of_r * shell_volumes)
        self.hist.data["number_integral"] = number_integral

        label_str = lambda labels: "_".join(l.replace(" ", "") for l in labels)
        fname = f"rdf_{label_str(self.ref_labels)}_{label_str(self.obs_labels)}.dat"
        self.hist.save_txt(fname, ["r/Å", "g(r)", "N(r)"])

        print(f"RDF and number-integral saved to {fname}")

