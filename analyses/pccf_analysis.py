# analyses/pccf_analysis.py

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


class PCCFAnalysis(BaseAnalysis):
    """
    Proton transfer chain / proton coupling analysis.

    IMPORTANT: This analysis assumes a *static* molecular topology:
      - Protons have been cut off from acceptors during the initial molecule
        recognition (in the interactive preprocessing step).
      - Molecules and their atoms (traj.compounds, traj.atoms) are not re-built
        per frame.

    Consequently, per-frame molecule recognition is *disabled* for this analysis.
    We only update coordinates each frame and track proton–acceptor distances.
    """

    def setup(self):
        print("\n--- Proton Coupling Correlation Function Analysis ---")
        print("NOTE: This analysis assumes that transferable protons have been")
        print("      cut off from acceptor molecules during the initial")
        print("      compound recognition. Molecules are treated as static;")
        print("      per-frame molecule recognition is disabled.\n")


        # === Protons ===
        proton_selection = self.compound_selection(
            role="proton",
            multi=True,
            prompt_text="Choose the compounds with transferable protons (comma-separated): "
        )
        # proton_selection: list of (idx, Compound)

        protons_per_compound = {}
        for idx, comp in proton_selection:
            labels = self.atom_selection(
                role="proton",
                compound=comp,
                prompt_text=("Enter transferable proton labels for {role} compound "
                             "{compound_num} ({compound_name}) (comma-separated): ")
            )
            protons_per_compound[comp] = labels

        # === Acceptors ===
        acceptor_selection = self.compound_selection(
            role="acceptor",
            multi=True,
            prompt_text="Choose the compounds that may accept protons (comma-separated): "
        )

        acceptors_per_compound = {}
        for idx, comp in acceptor_selection:
            labels = self.atom_selection(
                role="acceptor",
                compound=comp,
                prompt_text=("Enter acceptor atom labels for {role} compound "
                             "{compound_num} ({compound_name}) (comma-separated): ")
            )
            acceptors_per_compound[comp] = labels

        # === Mode: molecule vs atom ===
        self.is_molecule_mode = prompt_yn(
            "Track proton transfers by molecule [y] or atom [n]?",
            True
        )

        # --- Build proton & acceptor global index lists ---
        protons = []
        acceptors = []

        # protons: global indices
        for comp, labels in protons_per_compound.items():
            for mol in comp.members:
                for label, gid in mol.label_to_global_id.items():
                    if any(label_matches(lab, label) for lab in labels):
                        protons.append(gid)

        # acceptors: global indices
        for comp, labels in acceptors_per_compound.items():
            for mol in comp.members:
                for label, gid in mol.label_to_global_id.items():
                    if any(label_matches(lab, label) for lab in labels):
                        acceptors.append(gid)

        if not protons or not acceptors:
            raise ValueError("No protons or acceptors found with the given labels.")

        # Store indices
        self.proton_indices = np.array(protons, dtype=int)
        self.acceptor_indices = np.array(acceptors, dtype=int)
        self.n_protons = len(self.proton_indices)

        # --- Map acceptor atoms to acceptor molecules (for molecule mode) ---
        self.atom_to_mol = {}       # maps index in acceptors-array -> acceptor-molecule ID
        seen_molecules = {}
        acceptor_molecules = []

        for i, gid in enumerate(acceptors):  # acceptors are global atom indices
            atom = self.traj.atoms[gid]
            mol = atom.parent_molecule
            if mol not in seen_molecules:
                col_idx = len(acceptor_molecules)
                acceptor_molecules.append(mol)
                seen_molecules[mol] = col_idx
            self.atom_to_mol[i] = seen_molecules[mol]

        if self.is_molecule_mode:
            self.n_acceptors = len(seen_molecules)  # number of acceptor molecules
        else:
            self.n_acceptors = len(self.acceptor_indices)  # number of acceptor atoms

        # === Parameters ===
        self.bond_cutoff = prompt_float(
            "Enter maximum proton-acceptor bond distance (in Å): ",
            1.2,
            minval=0.1
        )
        self.dwell_threshold = prompt_int(
            "Enter minimum dwell time on an acceptor (in frames): ",
            2,
            minval=0
        )
        max_chain_gaps = prompt(
            "Enter one or more max chain gaps (in frames, comma-separated): ",
            "100"
        ).strip()
        self.max_chain_gaps = [
            int(x.strip())
            for x in max_chain_gaps.split(',')
            if x.strip().isdigit()
        ]
        if not self.max_chain_gaps:
            self.max_chain_gaps = [100]

        # === Transfer tracking state ===
        self.transfer_events = []  # list of (frame_idx, proton_id, donor_acceptor, new_acceptor)
        self.last_acceptor = np.full(self.n_protons, fill_value=-1, dtype=int)
        self.dwell_counters = np.zeros(self.n_protons, dtype=int)

        # Internal box reference for KDTree
        self.box = np.asarray(self.traj.box_size, dtype=float)

    def setup_frame_loop(self):
        """
        Override BaseAnalysis.setup_frame_loop to *disable* per-frame molecule updates.

        We only ask for start_frame / nframes / frame_stride.
        """
        print("\nPer-frame molecule recognition is disabled for proton coupling analysis.")
        print("Molecules & proton/acceptor definitions are taken from the initial\n"
              "compound recognition and assumed to be static.\n")

        self.update_compounds = False  # important: do NOT re-run guess_molecules()
        self.start_frame = prompt_int(
            "In which trajectory frame to start processing the trajectory?",
            1,
            minval=1
        )
        self.nframes = prompt_int(
            "How many trajectory frames to read (from this position on)?",
            -1,
            "all"
        )
        self.frame_stride = prompt_int(
            "Use every n-th read trajectory frame for the analysis:",
            1,
            minval=1
        )

    def post_compound_update(self):
        """
        Required abstract method, but never used because update_compounds is False.
        """
        return True

    def process_frame(self):
        """
        For each frame:
          - build KDTree over current acceptor positions,
          - find nearest acceptor within cutoff for each proton,
          - build a binary bond matrix,
          - detect transfers based on dwell-time threshold.
        """
        coords = self.traj.coords

        # Build KDTree over acceptor coordinates
        acceptor_coords = coords[self.acceptor_indices]
        tree = cKDTree(acceptor_coords, boxsize=self.traj.box_size)

        # Binary matrix: n_protons x n_acceptors
        bond_matrix = np.zeros((self.n_protons, self.n_acceptors), dtype=bool)

        # Assign each proton to its nearest acceptor within cutoff
        for i, p_idx in enumerate(self.proton_indices):
            p_coord = coords[p_idx]
            neighbors = tree.query_ball_point(p_coord, self.bond_cutoff)

            if neighbors:
                # pick nearest acceptor (by distance)
                dists = np.linalg.norm(acceptor_coords[neighbors] - p_coord, axis=1)
                nearest = neighbors[np.argmin(dists)]
                if self.is_molecule_mode:
                    acceptor_id = self.atom_to_mol[nearest]
                else:
                    acceptor_id = nearest
                bond_matrix[i, acceptor_id] = True

        # Dwell tracking and transfer event identification
        for p_idx in range(self.n_protons):
            bonded = np.where(bond_matrix[p_idx])[0]
            if bonded.size == 0:
                continue
            new_acceptor = bonded[0]
            if self.last_acceptor[p_idx] == -1:
                self.last_acceptor[p_idx] = new_acceptor
                self.dwell_counters[p_idx] = 1
            elif new_acceptor == self.last_acceptor[p_idx]:
                self.dwell_counters[p_idx] += 1
            else:
                # proton changed acceptor; check dwell threshold
                if self.dwell_counters[p_idx] >= self.dwell_threshold:
                    donor = self.last_acceptor[p_idx]
                    self.transfer_events.append(
                        (self.frame_idx, p_idx, donor, new_acceptor)
                    )
                self.last_acceptor[p_idx] = new_acceptor
                self.dwell_counters[p_idx] = 1

    def postprocess(self):
        print("\nAnalyzing transfer events and building chains...")

        if not self.transfer_events:
            print("No transfer events detected. Nothing to analyze.")
            return

        # === Cancel rapid back-and-forth oscillations ===
        oscillation_cutoff = 10  # frames
        by_proton = defaultdict(list)

        # Group events by proton
        for f, pid, d, a in sorted(self.transfer_events):
            by_proton[pid].append((f, d, a))

        filtered_events = []
        for pid, evs in by_proton.items():
            filtered_events.extend(squash(evs, oscillation_cutoff))

        filtered_events.sort()
        self.transfer_events = filtered_events

        # Analyze chains for each max_chain_gap
        for gap in self.max_chain_gaps:
            print(f"\nAnalyzing chains with max_chain_gap = {gap}...")

            # Rebuild donor -> list of (frame, acceptor)
            acceptor_transfer_map = defaultdict(list)
            for frame, donor, acceptor in self.transfer_events:
                acceptor_transfer_map[donor].append((frame, acceptor))

            max_depth = 20
            chain_counts = np.zeros(max_depth, dtype=int)

            # Walk through chains
            for frame, donor, acceptor in self.transfer_events:
                visited = set()
                current = acceptor
                depth = 1
                visited.add(donor)
                current_frame = frame

                while depth < max_depth:
                    next_links = [
                        (f, a) for (f, a) in acceptor_transfer_map.get(current, [])
                        if a not in visited and 0 < f - current_frame <= gap
                    ]
                    if not next_links:
                        break

                    # choose earliest valid transfer
                    next_frame, next_acc = min(next_links, key=lambda t: t[0])
                    visited.add(current)
                    current = next_acc
                    current_frame = next_frame
                    depth += 1

                chain_counts[depth - 1] += 1

            total_chains = np.sum(chain_counts)
            if total_chains > 0:
                Cn = chain_counts / total_chains
            else:
                Cn = chain_counts

            fname = f"proton_chains_gap{gap}.dat"
            with open(fname, "w") as f:
                f.write(f"# Total number of chains: {total_chains}\n")
                f.write("# n   C(n)\n")
                for i, val in enumerate(Cn):
                    if val > 0:
                        f.write(f"{i+1} {val:.6f}\n")

            print(f"Wrote C(n) to {fname}")


def squash(events, tau):
    """
    Remove back-and-forth hops within time window tau.

    events: list of (frame, donor, acceptor) for a single proton.
    """
    stack = []
    for f, d, a in events:
        if stack and stack[-1][1] == a and stack[-1][2] == d and (f - stack[-1][0]) <= tau:
            # cancel A→B followed by B→A within tau
            stack.pop()
        else:
            stack.append((f, d, a))
    return stack

