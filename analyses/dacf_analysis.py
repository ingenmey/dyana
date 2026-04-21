# analyses/dacf_analysis.py

import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree

from analyses.base_analysis import BaseAnalysis
from utils import (
    prompt,
    prompt_int,
    prompt_float,
    prompt_yn,
    label_matches,
)


class DACFAnalysis(BaseAnalysis):
    """
    Dimer existence autocorrelation function (DACF) analysis.

    Supports two modes:
      - 'global' (default): all frames in the trajectory are part of the time axis.
        Frames where the selected compounds/atoms are missing contribute β = 0 for
        all pairs (i.e. dimers are absent).
      - 'conditional': only frames where both selected compounds AND atoms exist
        are included. Frames where they are missing are skipped (like the original
        implementation), i.e. the time axis is 'compressed' to existence windows.
    """

    def setup(self):
        print("\n--- Dimer existence auto-correlation function analysis ---")

        # --- Compound + atom selection ---
        self.ref_idx, self.ref_comp = self.compound_selection("reference")
        self.obs_idx, self.obs_comp = self.compound_selection("observed")

        self.ref_labels = self.atom_selection("reference", compound=self.ref_comp)
        self.obs_labels = self.atom_selection("observed", compound=self.obs_comp)

        # --- Parameters ---
        self.cutoff = prompt_float("Cutoff distance for dimer (Å): ", 3.5)
        self.use_continuous = prompt_yn("Use continuous autocorrelation function?", False)
        self.corr_depth = prompt_int("Maximum correlation depth (number of frames): ", 100, minval=1)
        self.apply_correction = prompt_yn("Perform finite-size ensemble equilibrium correction?", True)
        self.frame_time = prompt_float("Time per frame (fs):", 1.0)

        # DACF mode: global vs conditional-on-existence
        self.global_mode = prompt_yn("Count frames where compound is missing as β=0 [y] or skip them [n]?", True)

        # Stable keys for per-frame compound updates
        keys = list(self.traj.compounds.keys())
        self.ref_key = keys[self.ref_idx]
        self.obs_key = keys[self.obs_idx]

        # Box (use first-frame box; lammps case updates it inside read_frame anyway)
        self.box = np.asarray(self.traj.box_size, dtype=float)

        # Initial index lists from first frame
        self._update_indices()

        if not self.ref_indices or not self.obs_indices:
            raise ValueError("No atoms matched the given labels in reference or observed compound.")

        # N and M: number of ref/obs atoms (will be averaged over frames if compounds change)
        self.n_ref = float(len(self.ref_indices))
        self.n_obs = float(len(self.obs_indices))

        # Tracking β(t) for each dimer pair (ref_idx, obs_idx)
        self.beta_tracker = defaultdict(list)  # { (ref_idx, obs_idx): [0/1, ...] }
        self.all_pairs = set()                 # set of all (ref_idx, obs_idx) ever observed

        # Flag for current frame when update_compounds is enabled
        self.frame_active = True

    # ---------- helpers for index management ----------

    def _update_indices(self):
        """Rebuild global index lists for the selected labels in current ref/obs compounds."""
        self.ref_indices = [
            idx for mol in self.ref_comp.members
            for label, idx in mol.label_to_global_id.items()
            if any(label_matches(lab, label) for lab in self.ref_labels)
        ]
        self.obs_indices = [
            idx for mol in self.obs_comp.members
            for label, idx in mol.label_to_global_id.items()
            if any(label_matches(lab, label) for lab in self.obs_labels)
        ]

    # ---------- BaseAnalysis interface ----------

    def post_compound_update(self):
        """
        Called when self.update_compounds is True (per BaseAnalysis).

        - In GLOBAL mode:
            Frames where compounds/indices are missing are treated as "no dimers":
            we set frame_active = False but return True so the frame is processed
            and β=0 is appended for all pairs.

        - In CONDITIONAL mode:
            Frames where compounds/indices are missing are skipped entirely (like
            the original function): we return False so BaseAnalysis skips the frame.
        """
        try:
            self.ref_comp = self.traj.compounds[self.ref_key]
            self.obs_comp = self.traj.compounds[self.obs_key]
        except KeyError:
            # Selected compound types are not present this frame
            if self.global_mode:
                self.frame_active = False
                self.ref_indices = []
                self.obs_indices = []
                return True   # process frame as "no dimers"
            else:
                # conditional: skip this frame entirely
                return False

        # Compounds present: rebuild indices
        self._update_indices()

        if not self.ref_indices or not self.obs_indices:
            # Labels exist but no atoms match in this frame
            if self.global_mode:
                self.frame_active = False
                return True   # process as "no dimers"
            else:
                # conditional: skip this frame
                return False

        # Compounds and indices present
        self.frame_active = True

        # Update average N, M over processed frames (only makes sense for frames we actually count)
        if self.processed_frames >= 0:
            self.n_ref = (self.n_ref * self.processed_frames + len(self.ref_indices)) / (self.processed_frames + 1)
            self.n_obs = (self.n_obs * self.processed_frames + len(self.obs_indices)) / (self.processed_frames + 1)

        return True

    def _pad_absent_for_all_pairs(self):
        """Append 0 for all known pairs (used in GLOBAL mode when no dimers exist in this frame)."""
        for pair in self.all_pairs:
            self.beta_tracker[pair].append(0)

    def process_frame(self):
        # CONDITIONAL mode + update_compounds:
        #   post_compound_update() only returns True for frames where compounds & indices exist.
        #   So if we are here in conditional mode, we can assume indices are valid.
        #
        # GLOBAL mode:
        #   post_compound_update() returns True for every frame, and sets frame_active
        #   to False when compounds/indices are missing -> treat as "no dimers present".
        if self.global_mode:
            # If update_compounds is enabled and this frame is "inactive", treat as no dimers
            if self.update_compounds and not self.frame_active:
                self._pad_absent_for_all_pairs()
                return

            # Extra safety: if somehow indices are empty, treat as no dimers
            if not self.ref_indices or not self.obs_indices:
                self._pad_absent_for_all_pairs()
                return
        else:
            # conditional mode:
            #   - if update_compounds=True, we only get here for active frames
            #   - if update_compounds=False, indices are static from setup()
            if not self.ref_indices or not self.obs_indices:
                # nothing meaningful to do this frame; equivalent to skipping it
                return

        # --- Normal per-frame DACF accumulation ---
        coords = self.traj.coords
        ref_coords = coords[self.ref_indices]
        obs_coords = coords[self.obs_indices]

        tree = cKDTree(obs_coords, boxsize=self.box)
        results = tree.query_ball_point(ref_coords, self.cutoff)

        active_pairs = set()

        for i, obs_list in enumerate(results):
            ref_idx = self.ref_indices[i]
            for j in obs_list:
                obs_idx = self.obs_indices[j]
                if ref_idx == obs_idx:
                    continue  # skip self
                pair = (ref_idx, obs_idx)
                active_pairs.add(pair)
                self.beta_tracker[pair].append(1)
                self.all_pairs.add(pair)

        # Pairs that have existed at least once but are not present this frame: append 0
        for pair in self.all_pairs:
            if pair not in active_pairs:
                self.beta_tracker[pair].append(0)

    def postprocess(self):
        print("\n")

        T = self.processed_frames
        if T <= 0 or not self.beta_tracker:
            print("No frames processed or no dimer pairs observed; skipping DACF.")
            return

        max_tau = min(self.corr_depth, T)
        dacf = np.zeros(max_tau, dtype=float)

        N = self.n_ref
        M = self.n_obs

        # Ensure each β_p(t) has length T (in GLOBAL mode, some early frames may be implicit zeros)
        for pair in self.beta_tracker:
            b = self.beta_tracker[pair]
            if len(b) < T:
                b.extend([0] * (T - len(b)))
            self.beta_tracker[pair] = np.array(b, dtype=np.uint8)

        # --- Continuous autocorrelation ---
        if self.use_continuous:
            segments = []
            for b in self.beta_tracker.values():
                mask = b == 1
                if not np.any(mask):
                    continue

                changes = np.diff(mask.astype(int))
                starts = np.where(changes == 1)[0] + 1
                ends = np.where(changes == -1)[0] + 1

                if mask[0]:
                    starts = np.insert(starts, 0, 0)
                if mask[-1]:
                    ends = np.append(ends, T)

                segments.extend(zip(starts, ends))

            for start, end in segments:
                length = end - start
                if length <= 0:
                    continue
                local_max = min(max_tau, length)
                for tau in range(local_max):
                    dacf[tau] += length - tau

            dacf /= (N * M * T)
            if dacf[0] != 0:
                dacf /= dacf[0]

        # --- Intermittent autocorrelation ---
        else:
            for b in self.beta_tracker.values():
                for tau in range(max_tau):
                    dacf[tau] += np.sum(b[:T - tau] * b[tau:])

            normalization = N * M * np.arange(T, T - max_tau, -1)
            dacf /= normalization

            if self.apply_correction:
                total_ones = sum(np.sum(b) for b in self.beta_tracker.values())
                avg_beta = total_ones / (N * M * T)
                dacf -= avg_beta ** 2

            if dacf[0] != 0:
                dacf /= dacf[0]

        # --- Save Results ---
        with open("dacf.dat", "w") as f:
            f.write("tau/ps DACF\n")
            for tau, val in enumerate(dacf):
                t_ps = tau * self.frame_time / 1000.0
                f.write(f"{t_ps:.6f} {val:.12f}\n")

        print("Dimer autocorrelation results saved to dacf.dat")


# Backwards-compatible alias for main.py
# (AVAILABLE_ANALYSES expects a class callable as analysis_func(traj))
dacf = DACFAnalysis

