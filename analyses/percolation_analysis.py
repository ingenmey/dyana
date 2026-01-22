import numpy as np
from collections import deque, defaultdict
from scipy.spatial import cKDTree

from analyses.base_analysis import BaseAnalysis
#from analyses.histogram import HistogramND
from utils import prompt, prompt_int, prompt_float, prompt_yn, label_matches


class PercolationAnalysis(BaseAnalysis):
    def setup(self):
        print("\n--- Hydrogen Bond Percolation Pathway Analysis ---")

        # Select compounds (multi) and remember stable keys for later frame updates
        selection = self.compound_selection(
            multi=True,
            prompt_text="Enter compound numbers to include in the percolation analysis (comma-separated): "
        )
        if not selection:
            raise ValueError("No compounds selected.")

        self.comp_idxs = [idx for idx, _ in selection]
        self.comp_objs = [comp for _, comp in selection]
        all_keys = list(self.traj.compounds.keys())
        self.comp_keys = [all_keys[i] for i in self.comp_idxs]

        # Per-compound labels (acceptors & protons), kept by compound key
        self.acc_labels = {}
        self.prot_labels = {}
        for key, comp in zip(self.comp_keys, self.comp_objs):
            tmpl = "Enter {role} atom labels for compound {compound_num} ({compound_name}): "
            acc = self.atom_selection(role="acceptor", compound=comp, prompt_text=tmpl)
            prot = self.atom_selection(role="proton", compound=comp, prompt_text=tmpl, allow_empty=True)
            self.acc_labels[key] = acc
            self.prot_labels[key] = prot

        # Parameters
        self.cutoff = prompt_float("Enter hydrogen bond cutoff distance (in Å): ", 2.5, minval=0.5)
        self.max_depth = prompt_int("Enter the maximum number of H-bond steps to consider: ", 5, minval=1)
        self.use_alternating = prompt_yn(
            "Limit percolation pathways to alternating chains of accepted and donated hydrogen bonds?", False
        )

        # Accumulators
        self.depth_counts = np.zeros(self.max_depth, dtype=np.float64)
        self.total_seeds = 0
#        edges = np.arange(0.5, self.max_depth + 0.5 + 1)  # centers: 1..max_depth
#        self.depth_hist = HistogramND([edges])  # mode="simple" is fine

        # Resolve selected compounds for current frame
        self._update_selected_compounds()

        # Precompute atom lists if compounds are static (faster)
        # If update_compounds is True, we'll rebuild them per post_compound_update().
        if not self.update_compounds:
            self._build_and_cache_atom_lists()

    def post_compound_update(self):
        """
        Re-attach selected compounds for the current frame and rebuild atom lists
        when per-frame compound recognition is enabled.
        """
        try:
            self._update_selected_compounds()
        except KeyError:
            return False

        # When compounds can change per frame, rebuild atom lists to stay consistent.
        self._build_and_cache_atom_lists()
        return True

#    def _update_selected_compounds(self):
#        self.comp_objs = [self.traj.compounds[k] for k in self.comp_keys]

    def _update_selected_compounds(self):
        self.comp_objs = []
        self.active_keys = []
        for k in self.comp_keys:
            if k in self.traj.compounds:
                self.comp_objs.append(self.traj.compounds[k])
                self.active_keys.append(k)
        # If none of the selected compounds are present, bail
        if not self.comp_objs:
            raise KeyError("No selected compounds present in this frame.")

    # ---------- Atom list building / caching ----------

    def _build_and_cache_atom_lists(self):
        """
        Build and cache:
          - self.acc_indices : np.ndarray of global atom indices (acceptors)
          - self.prot_indices: np.ndarray of global atom indices (protons)
          - self.acc_mols    : list of Molecule refs (same length as acc_indices)
          - self.prot_mols   : list of Molecule refs (same length as prot_indices)
        """
        acc_indices, prot_indices = [], []
        acc_mols, prot_mols = [], []

        #for key, comp in zip(self.comp_keys, self.comp_objs):
        for key, comp in zip(self.active_keys, self.comp_objs):
            acc_lbls = self.acc_labels.get(key, [])
            prot_lbls = self.prot_labels.get(key, [])

            for mol in comp.members:
                # collect global indices by matching labels with wildcard support
                for label, gid in mol.label_to_global_id.items():
                    if any(label_matches(lab, label) for lab in acc_lbls):
                        acc_indices.append(gid)
                        acc_mols.append(mol)
                    if any(label_matches(lab, label) for lab in prot_lbls):
                        prot_indices.append(gid)
                        prot_mols.append(mol)

        self.acc_indices = np.array(acc_indices, dtype=int)
        self.prot_indices = np.array(prot_indices, dtype=int)
        self.acc_mols = acc_mols
        self.prot_mols = prot_mols

    def process_frame(self):
        # If compounds are static and we haven't built lists yet (edge case), build once.
        if not hasattr(self, "acc_indices") or not hasattr(self, "prot_indices"):
            self._build_and_cache_atom_lists()

        if len(self.acc_indices) == 0 or len(self.prot_indices) == 0:
            return

        # Build KD-trees from current-frame coordinates via indices (periodic)
        acc_coords = self.traj.coords[self.acc_indices]
        prot_coords = self.traj.coords[self.prot_indices]
        acc_tree = cKDTree(acc_coords, boxsize=self.traj.box_size)
        prot_tree = cKDTree(prot_coords, boxsize=self.traj.box_size)

        # Option A: alternating chains
        if self.use_alternating:
            # per-molecule lists of prot/acc atoms by index position
            mol_protons = defaultdict(list)
            for idx_in_list, mol in enumerate(self.prot_mols):
                mol_protons[mol].append(idx_in_list)

            seed_molecules = set(self.acc_mols)

            for seed in seed_molecules:
                visited = {seed}
                frontier = deque([(seed, 0)])  # start at acceptor molecule
                while frontier:
                    current_mol, depth = frontier.popleft()
                    if depth >= self.max_depth:
                        continue
                    self.depth_counts[depth] += 1

                    # donate from each proton in current_mol to nearby acceptors
                    for p_idx in mol_protons[current_mol]:
                        nearby = acc_tree.query_ball_point(prot_coords[p_idx], r=self.cutoff)
                        for ai in nearby:
                            neighbor_mol = self.acc_mols[ai]
                            if neighbor_mol not in visited:
                                visited.add(neighbor_mol)
                                frontier.append((neighbor_mol, depth + 1))
                self.total_seeds += 1

        # Option B: undirected neighbor graph
        else:
            neighbors = defaultdict(set)

            # acceptor -> proton
            for ai, mol_a in enumerate(self.acc_mols):
                nearby = prot_tree.query_ball_point(acc_coords[ai], r=self.cutoff)
                for pj in nearby:
                    mol_p = self.prot_mols[pj]
                    if mol_p != mol_a:
                        neighbors[mol_a].add(mol_p)
                        neighbors[mol_p].add(mol_a)

            # proton -> acceptor (redundant but robust)
            for pj, mol_p in enumerate(self.prot_mols):
                nearby = acc_tree.query_ball_point(prot_coords[pj], r=self.cutoff)
                for ai in nearby:
                    mol_a = self.acc_mols[ai]
                    if mol_p != mol_a:
                        neighbors[mol_p].add(mol_a)
                        neighbors[mol_a].add(mol_p)

            seed_molecules = set(self.acc_mols)

            for seed in seed_molecules:
                visited = {seed}
                frontier = deque([(seed, 0)])
                while frontier:
                    current_mol, depth = frontier.popleft()
                    if depth >= self.max_depth:
                        continue
                    self.depth_counts[depth] += 1
                    for nbr in neighbors[current_mol]:
                        if nbr not in visited:
                            visited.add(nbr)
                            frontier.append((nbr, depth + 1))
                self.total_seeds += 1

    def postprocess(self):
        if self.total_seeds > 0:
            avg_counts = self.depth_counts / self.total_seeds
        else:
            avg_counts = np.zeros_like(self.depth_counts)

        print("\nAverage number of reachable molecules by H-bond depth:")
        for d, count in enumerate(avg_counts, 1):
            print(f"Depth {d}: {count:.4f}")

#        self.depth_hist.counts = avg_counts
#        self.depth_hist.save_txt("percolation_depths.dat", headers=["Depth", "AverageReachableMolecules"])

        with open("percolation_depths.dat", "w") as f:
            f.write("# Depth    AverageReachableMolecules\n")
            for d, count in enumerate(avg_counts, 1):
                f.write(f"{d} {count:.6f}\n")
        print("Results written to percolation_depths.dat")

