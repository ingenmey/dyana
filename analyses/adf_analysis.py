# analyses/adf_analysis.py

import numpy as np
from analyses.base_analysis import BaseAnalysis
from utils import (
    prompt, prompt_int, prompt_float, prompt_yn, prompt_choice, label_matches
)
from analyses.metrics import Selector, AngleMetric
from analyses.histogram import HistogramND


class ADF(BaseAnalysis):
    def setup(self):
        # --- User Setup ---
        self.ref_idx, self.ref_comp = self.compound_selection("reference")
        self.obs_idx, self.obs_comp = self.compound_selection("observed")

        self.ref_base_source = prompt_choice("Base atom of first vector?", ["r", "o"], "r")
        self.ref_tip_source = prompt_choice("Tip atom of first vector?", ["r", "o"], "r")
        self.ref_base_label = prompt("Which atom is at the base of the first vector? ")
        self.ref_tip_label = prompt("Which atom is at the tip of the first vector? ")

        self.obs_base_source = prompt_choice("Base atom of second vector?", ["r", "o"], "o")
        self.obs_tip_source = prompt_choice("Tip atom of second vector?", ["r", "o"], "o")
        self.obs_base_label = prompt("Which atom is at the base of the second vector? ")
        self.obs_tip_label = prompt("Which atom is at the tip of the second vector? ")

        self.enforce_shared_atom = (
            self.ref_tip_label == self.obs_base_label
            and prompt_yn("Should the tip atom of the reference vector and the base atom of the observed vector be the same atom?", True)
        )

        self.bin_count = prompt_int("Enter the number of bins for ADF calculation: ", 180, minval=1)
        self.v1_cutoff = prompt_float("Enter maximum length for the first vector: ", None, "None", minval=0.0)
        self.v2_cutoff = prompt_float("Enter maximum length for the second vector: ", None, "None", minval=0.0)

        self.ref_key = list(self.traj.compounds.keys())[self.ref_idx]
        self.obs_key = list(self.traj.compounds.keys())[self.obs_idx]

        self._update_vectors()
        self._create_metric()

        self.n_ref = len(self.ref_comp.members)
        self.n_obs = len(self.obs_comp.members)
        self.angle_edges = np.linspace(0, 180, self.bin_count + 1)
        self.hist = HistogramND([self.angle_edges])

    def _update_vectors(self):
        self.ref_base_ids, self.ref_tip_ids, self.obs_base_ids, self.obs_tip_ids = build_vector_lists(
            self.ref_comp, self.obs_comp,
            self.ref_base_source, self.ref_tip_source,
            self.obs_base_source, self.obs_tip_source,
            self.ref_base_label, self.ref_tip_label,
            self.obs_base_label, self.obs_tip_label,
            self.enforce_shared_atom
        )

    def _create_metric(self):
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
            self.ref_comp = self.traj.compounds[self.ref_key]
            self.obs_comp = self.traj.compounds[self.obs_key]
        except KeyError:
            return False

        self._update_vectors()
        if not all([self.ref_base_ids, self.ref_tip_ids, self.obs_base_ids, self.obs_tip_ids]):
            return False

        self._create_metric()
        self.n_ref = (self.n_ref * self.processed_frames + len(self.ref_comp.members))/(self.processed_frames + 1)
        self.n_obs = (self.n_obs * self.processed_frames + len(self.obs_comp.members))/(self.processed_frames + 1)
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
        self.hist.save_txt("adf.dat")

        print("ADF results saved to adf.dat")


def find_matching_labels(mol, user_label):
    return [
        idx for label, idx in mol.label_to_global_id.items()
        if label_matches(user_label, label)
    ]


def build_vector_lists(ref_comp, obs_comp, ref_base_source, ref_tip_source,
                       obs_base_source, obs_tip_source, ref_base_label, ref_tip_label,
                       obs_base_label, obs_tip_label, enforce_shared_atom):

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

            rb_ids = find_matching_labels(rb_mol, ref_base_label)
            rt_ids = find_matching_labels(rt_mol, ref_tip_label)
            ob_ids = find_matching_labels(ob_mol, obs_base_label)
            ot_ids = find_matching_labels(ot_mol, obs_tip_label)

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

