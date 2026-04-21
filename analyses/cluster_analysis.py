# analyses/cluster_analysis.py

import os
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from scipy.spatial import cKDTree
from collections import deque

import networkx as nx
import hashlib
import matplotlib.pyplot as plt

from analyses.base_analysis import BaseAnalysis
from utils import (
    label_matches, prompt, prompt_int, prompt_float, prompt_yn, prompt_choice
)


class ClusterAnalysis(BaseAnalysis):
    """
    Refactored cluster analysis using BaseAnalysis framework.
    """

    def setup(self):
        # ----- Per-compound atom labels to include in clustering -----
        self.compound_atom_labels = {}  # {compound_key: [user_label1, user_label2, ...]}

        self._all_keys = list(self.traj.compounds.keys())
        self._all_comps = list(self.traj.compounds.values())

        print("\n--- Cluster composition histogram ---")
        print("For each compound, specify which atom labels/types should be considered for clustering.")
        print("Leave empty to exclude that compound from clustering.\n")

        for i, (key, comp) in enumerate(zip(self._all_keys, self._all_comps), start=1):
            atom_labels = prompt(
                f"Enter atom types in compound {i} ({comp.rep}) to consider (comma-separated): ",
                ""
            ).strip()
            if atom_labels:
                self.compound_atom_labels[key] = [lab.strip() for lab in atom_labels.split(",") if lab.strip()]
            else:
                self.compound_atom_labels[key] = []

        # Only keep compounds that the user actually wants to use
        self.selected_keys = [k for k, labs in self.compound_atom_labels.items() if len(labs) > 0]
        if not self.selected_keys:
            raise ValueError("No compounds selected for clustering (all label lists empty).")

        # Store a stable, user-friendly display index for compositions ("1-O", "2-H", ...)
        # based on the order at setup time.
        self.key_to_disp = {}
        for i, k in enumerate(self.selected_keys, start=1):
            self.key_to_disp[k] = i

        # ----- Cutoffs between (compound, label) pairs across compounds -----
        # cutoff_distances[((key1,label1),(key2,label2))] = cutoff
        self.cutoff_distances = {}
        self._build_cutoff_table()

        # ----- Options -----
        self.should_hash = prompt_yn(
            "Count clusters by composition (n) or by composition and graph hash (y)?",
            True
        )

        self.vis_format = prompt_yn("Visualize cluster graphs?", False)
        if self.vis_format:
            self.vis_format = prompt_choice("Save cluster graphs in which format?", ["svg", "png"], "svg")
        else:
            self.vis_format = False

        self.is_save_xyz = prompt_yn("Save cluster coordinates as XYZ files?", False)
        self.is_save_whole = False
        if self.is_save_xyz:
            self.is_save_whole = prompt_yn("Save whole molecules (Y) or only specified atom types (N)?", False)
            os.makedirs("xyz", exist_ok=True)

        self.compute_cacf = prompt_yn("Compute cluster autocorrelation functions?", True)
        if self.compute_cacf:
            self.corr_depth = prompt_int("Maximum correlation depth (number of frames): ", 100, minval=1)

        # Standard Deviations
        self.compute_errors = prompt_yn(
            "Compute per-frame standard deviations for cluster counts?", False
        )
        if self.compute_errors:
            # (composition, graph_id) -> [count_in_frame0, count_in_frame1, ...]
            self.frame_cluster_counts = {}
        else:
            self.frame_cluster_counts = None

        # ----- Accumulators -----
        self.cluster_histogram = Counter()
        self.graph_list = []
        self.seen_graphs = set()  # (composition, graph_id)
        self.cluster_beta = defaultdict(lambda: defaultdict(list))  # {(composition, graph_id): {atom_ids: [0/1...]}}
        self.all_cluster_ids = set()  # (composition, graph_id, atom_ids)

        # Resolve selected compound objects for current frame
        self._update_selected_compounds()

    def _build_cutoff_table(self):
        """
        Prompt user for cutoff distances for each unique atom pair across different selected compounds.
        Mirrors old behavior: only pairs across different compounds (combinations of compounds).
        """
        # Build all (key,label) entries
        compound_atom_entries = []
        for k in self.selected_keys:
            for lab in self.compound_atom_labels[k]:
                compound_atom_entries.append((k, lab))

        # Ask cutoffs for each unique compound-pair (k1 != k2) and label-pair.
        # This matches the old `combinations(compound_atom_labels.items(), 2)` logic.
        for (k1, labs1), (k2, labs2) in combinations(
            [(k, self.compound_atom_labels[k]) for k in self.selected_keys], 2
        ):
            for a1 in labs1:
                for a2 in labs2:
                    c1 = self.key_to_disp[k1]
                    c2 = self.key_to_disp[k2]
                    cutoff = prompt_float(
                        f"Cutoff for {a1} in compound {c1} and {a2} in compound {c2} (Å): ",
                        0.0,
                        minval=0.0
                    )
                    self.cutoff_distances[((k1, a1), (k2, a2))] = cutoff
                    self.cutoff_distances[((k2, a2), (k1, a1))] = cutoff  # symmetry

    def post_compound_update(self):
        """
        Called only when BaseAnalysis.update_compounds==True.
        We re-attach selected compounds that still exist.
        If none exist, skip the frame.
        """
        try:
            self._update_selected_compounds()
        except KeyError:
            return False
        return True

    def _update_selected_compounds(self):
        """
        Keep only selected compounds that are present in this frame.
        """
        self.active_keys = []
        self.active_comps = []
        for k in self.selected_keys:
            if k in self.traj.compounds:
                self.active_keys.append(k)
                self.active_comps.append(self.traj.compounds[k])

        if not self.active_comps:
            raise KeyError("No selected compounds are present in this frame.")

    def process_frame(self):
        # 1) Gather atoms per (compound_key, user_label)
        atom_groups = self._build_atom_groups()

        if not atom_groups:
            # Nothing to cluster this frame
            self._pad_absent_clusters(seen_this_frame=set())
            if self.compute_errors:
                self._update_frame_counts_for_empty_frame()
            return

        # 2) Identify clusters (graphs)
        clusters = identify_clusters(
            atom_groups=atom_groups,
            cutoff_distances=self.cutoff_distances,
            box_size=self.traj.box_size,
            key_to_disp=self.key_to_disp
        )

        seen_this_frame = set()  # (composition, graph_id, atom_ids)
        frame_counts = Counter()

        # 3) Count clusters + optionally write xyz
        for composition, graph, cluster_atoms in clusters:
            graph_id = get_graph_id(graph) if self.should_hash else 0
            key = (composition, graph_id)
            self.cluster_histogram[key] += 1

            frame_counts[key] += 1

            atom_ids = frozenset(atom.idx for atom in cluster_atoms)
            seen_this_frame.add((composition, graph_id, atom_ids))
            self.all_cluster_ids.add((composition, graph_id, atom_ids))
            self.cluster_beta[(composition, graph_id)][atom_ids].append(1)

            if (composition, graph_id) not in self.seen_graphs:
                self.seen_graphs.add((composition, graph_id))
                self.graph_list.append((composition, graph_id, graph))

            if self.is_save_xyz and len(cluster_atoms) > 1:
                write_xyz(
                    f"{composition}_{graph_id}.xyz",
                    cluster_atoms,
                    self.is_save_whole,
                    self.traj.box_size
                )

        if self.compute_errors:
            self._update_frame_counts(frame_counts)

        # 4) Pad 0s for clusters not seen in this frame (for intermittent CACF)
        self._pad_absent_clusters(seen_this_frame)

    def _build_atom_groups(self):
        """
        Returns:
            atom_groups: {(compound_key, user_label): [Atom, Atom, ...]}
        """
        atom_groups = {}

        for k, comp in zip(self.active_keys, self.active_comps):
            user_labels = self.compound_atom_labels.get(k, [])
            if not user_labels:
                continue

            for mol in comp.members:
                # mol.atoms are Atom objects created during guess_molecules(),
                # with atom.coord updated each frame in mol.update_coords()
                for label, local_idx in mol.label_to_id.items():
                    for user_label in user_labels:
                        if label_matches(user_label, label):
                            atom_groups.setdefault((k, user_label), []).append(mol.atoms[local_idx])

        return atom_groups

    def _pad_absent_clusters(self, seen_this_frame):
        """
        Append 0 to the beta-list for any cluster instance not present in this frame.
        """
        for (composition, graph_id), instance_dict in self.cluster_beta.items():
            for atom_ids in instance_dict:
                if (composition, graph_id, atom_ids) not in seen_this_frame:
                    self.cluster_beta[(composition, graph_id)][atom_ids].append(0)

    def _update_frame_counts(self, frame_counts):
        """
        Maintain a per-frame time series of cluster counts for each
        (composition, graph_id) key.

        self.processed_frames is the number of *already* processed frames
        when process_frame() is called, so it is the index of the current frame.
        """
        t = self.processed_frames  # 0-based index for "current" frame

        # 1) For existing keys: append 0 or the current count
        for key, series in self.frame_cluster_counts.items():
            series.append(frame_counts.get(key, 0))

        # 2) For new keys: create a series full of zeros for past frames, plus current count
        for key, count in frame_counts.items():
            if key not in self.frame_cluster_counts:
                self.frame_cluster_counts[key] = [0] * t + [count]

    def _update_frame_counts_for_empty_frame(self):
        if not self.frame_cluster_counts:
            return
        for series in self.frame_cluster_counts.values():
            series.append(0)


    def postprocess(self):
        # ----- Print histogram -----
        sorted_compositions = sorted(self.cluster_histogram.items(), key=lambda item: item[0])

        print("\nCluster Composition Histogram:")
        for (composition, graph_id), count in sorted_compositions:
            print(f"Composition {composition}: {count} occurrences")

        # ----- Post-process clusters (populations, images, occurrences files) -----
        post_process_clusters(self.cluster_histogram, self.graph_list, self.vis_format, self.frame_cluster_counts)

        # ----- CACF -----
        if self.compute_cacf:
            print("\nComputing intermittent cluster autocorrelation functions...")
            T = self.processed_frames
            if T <= 0:
                print("No frames processed; skipping CACF.")
                return

            max_tau = min(self.corr_depth, T)

            for (composition, graph_id), instances in self.cluster_beta.items():
                cacf = np.zeros(max_tau, dtype=np.float64)

                for beta in instances.values():
                    b = np.array(beta, dtype=np.float64)
                    if len(b) < T:
                        b = np.pad(b, (0, T - len(b)), constant_values=0.0)

                    for tau in range(max_tau):
                        cacf[tau] += np.sum(b[:T - tau] * b[tau:])

                normalization = len(instances) * np.arange(T, T - max_tau, -1)
                cacf /= normalization

                if cacf[0] != 0:
                    cacf /= cacf[0]

                with open(f"cacf_{composition}_{graph_id}.dat", "w") as f:
                    f.write("tau CACF\n")
                    for tau, val in enumerate(cacf):
                        f.write(f"{tau} {val:.12f}\n")

                print(f"Saved CACF for cluster type: {composition}, graph ID: {graph_id}")


# -------------------------- helpers (mostly unchanged) --------------------------

def write_xyz(filename, atoms, is_save_whole, box_size):
    def unwrap_coords(coords, box_size):
        """Unwrap coordinates to avoid broken bonds due to PBC."""
        unwrapped = np.copy(coords)
        for i in range(1, len(unwrapped)):
            delta = unwrapped[i] - unwrapped[i - 1]
            delta -= box_size * np.round(delta / box_size)
            unwrapped[i] = unwrapped[i - 1] + delta
        return unwrapped

    symbols = []
    coords = []

    if is_save_whole:
        written_molecules = set()
        atoms_sorted = sorted(atoms, key=lambda atom: (atom.parent_molecule.comp_id, atom.parent_molecule.mol_id))

        for atom in atoms_sorted:
            mol = atom.parent_molecule
            if mol not in written_molecules:
                written_molecules.add(mol)
                symbols.extend(mol.symbols)
                coords.extend(mol.coords)
    else:
        atoms_sorted = sorted(atoms, key=lambda atom: (atom.parent_molecule.comp_id, atom.parent_molecule.mol_id))
        for atom in atoms_sorted:
            # Find atom symbol by locating the atom in mol.atoms
            mol = atom.parent_molecule
            local_idx = mol.atoms.index(atom)
            symbols.append(mol.symbols[local_idx])
            coords.append(atom.coord)

    coords = np.array(coords, dtype=np.float64)
    coords = unwrap_coords(coords, box_size)
    cog = np.mean(coords, axis=0)
    coords -= cog

    with open(os.path.join("xyz", filename), "a") as f:
        f.write(f"{len(symbols)}\n")
        f.write("Generated by cluster analysis\n")
        for symbol, coord in zip(symbols, coords):
            f.write(f"{symbol} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")


def identify_clusters(atom_groups, cutoff_distances, box_size, key_to_disp):
    """
    Build clusters as graphs using KDTree neighbor queries, then extract connected components
    by DFS growth (same spirit as legacy code, but compound IDs are stable keys).
    """
    # KDTree per (compound_key, user_label)
    kdtrees = {
        (key, atom_label): cKDTree([atom.coord for atom in atoms], boxsize=box_size)
        for (key, atom_label), atoms in atom_groups.items()
    }

    # Precompute valid neighbor lookups
    valid_neighbors = {
        (key, atom_label): [
            (other_key, other_atom_label)
            for (other_key, other_atom_label) in atom_groups.keys()
            if ((key, atom_label), (other_key, other_atom_label)) in cutoff_distances
        ]
        for (key, atom_label) in atom_groups.keys()
    }

    visited = set()
    clusters = []

    def grow_cluster(atom, key, atom_label, cluster, graph, atom_counts):
        if atom in visited:
            return
        visited.add(atom)

        graph.add_node(atom, element=atom_label)
        cluster.append(atom)

        # Use stable display index for composition naming
        disp = key_to_disp.get(key, 0)
        count_key = f"{disp}-{atom_label}"
        atom_counts[count_key] = atom_counts.get(count_key, 0) + 1

        for (other_key, other_label) in valid_neighbors[(key, atom_label)]:
            cutoff = cutoff_distances[((key, atom_label), (other_key, other_label))]
            neighbors = kdtrees[(other_key, other_label)].query_ball_point(atom.coord, cutoff)

            for neighbor_idx in neighbors:
                neighbor_atom = atom_groups[(other_key, other_label)][neighbor_idx]
                if neighbor_atom not in visited:
                    grow_cluster(neighbor_atom, other_key, other_label, cluster, graph, atom_counts)

                graph.add_edge(atom, neighbor_atom)

    # Loop atoms and grow clusters
    for (key, atom_label), atoms in atom_groups.items():
        for atom in atoms:
            if atom in visited:
                continue
            cluster = []
            atom_counts = {}
            graph = nx.Graph()
            grow_cluster(atom, key, atom_label, cluster, graph, atom_counts)

            composition = "_".join(f"{k}-{v}" for k, v in sorted(atom_counts.items()))
            clusters.append((composition, graph, cluster))

    return clusters


def get_graph_id(graph):
    """
    Generate a unique identifier for a graph using WL hash + md5 for compactness.
    """
    graph_string = nx.weisfeiler_lehman_graph_hash(graph)
    return hashlib.md5(graph_string.encode()).hexdigest()


#def post_process_clusters(cluster_histogram, graph_list, vis_format, frame_cluster_counts):
#    graph_dict = {(composition, graph_id): graph for composition, graph_id, graph in graph_list}
#
#    sorted_clusters = sorted(
#        cluster_histogram.items(),
#        key=lambda item: (-item[1], item[0][0], item[0][1])
#    )
#
#    # Save images for visualization if requested
#    if vis_format:
#        for i, ((composition, graph_id), count) in enumerate(sorted_clusters):
#            if i > 200:
#                break
#            if (composition, graph_id) in graph_dict:
#                filename = f"graph{i}_{composition}.{vis_format}"
#                draw_graph(graph_dict[(composition, graph_id)], filename)
#
#    # Save cluster occurrences
#    with open("cluster_occurrences.dat", "w") as f_occ:
#        if frame_cluster_counts is not None:
#            f_occ.write(f"{'Cluster':<30} {'Occurrences':>12} {'GraphID':<32} {'mean_per_frame':>16} {'std_per_frame':>16}\n")
#
#            for (composition, graph_id), count in sorted_clusters:
#                series = frame_cluster_counts.get((composition, graph_id), [])
#                if series:
#                    arr = np.asarray(series, dtype=float)
#                    mean = arr.mean()
#                    std = arr.std(ddof=1) if len(arr) > 1 else 0.0
#                else:
#                    mean = std = 0.0
#
#                f_occ.write(f"{composition:<30} {count:>12d} {graph_id:<32} {mean:>16.6f} {std:>16.6f}\n")
#
#        else:
#            f_occ.write(f"{'Cluster':<30} {'Occurrences':>12} {'GraphID':<32}\n")
#            for (composition, graph_id), count in sorted_clusters:
#                f_occ.write(f"{composition:<30} {count:>12d} {graph_id:<32}\n")
#
#
#    # Extract unique atom types from compositions: entries look like "1-O-2"
#    atom_types = {
#        entry.split("-")[1]
#        for (composition, _), _ in cluster_histogram.items()
#        for entry in composition.split("_")
#        if len(entry.split("-")) >= 3
#    }
#
#    # Compute populations I(P) for each atom type
#    atom_populations = {atom: {} for atom in atom_types}
#    for atom in atom_types:
#        weighted_occurrences = {}
#        total_weighted = 0
#
#        for (composition, graph_id), occ in cluster_histogram.items():
#            atom_count = 0
#            for entry in composition.split("_"):
#                parts = entry.split("-")
#                if len(parts) < 3:
#                    continue
#                _, atom_label, count = parts[0], parts[1], parts[2]
#                if atom_label == atom:
#                    atom_count += int(count)
#
#            w_occ = occ * atom_count
#            weighted_occurrences[(composition, graph_id)] = w_occ
#            total_weighted += w_occ
#
#        for (composition, graph_id), w_occ in weighted_occurrences.items():
#            atom_populations[atom][(composition, graph_id)] = (w_occ / total_weighted) if total_weighted > 0 else 0.0
#
#    # Save cluster populations I(P)
#    with open("cluster_populations.dat", "w") as f_pop:
#        header = f"{'Cluster':<30}" + " ".join(f"{f'I({atom})':>15}" for atom in atom_types)
#        f_pop.write(f"{header}\n")
#
#        sorted_comps = sorted(
#            cluster_histogram.keys(),
#            key=lambda comp: sum(atom_populations[a].get((comp[0], comp[1]), 0.0) for a in atom_types),
#            reverse=True
#        )
#
#        for (composition, graph_id) in sorted_comps:
#            line = f"{composition:<30}" + " ".join(
#                f"{atom_populations[atom].get((composition, graph_id), 0.0):>15.10f}"
#                for atom in atom_types
#            )
#            f_pop.write(f"{line}\n")
#
#    # Summarized populations by cluster size (per atom type)
#    cluster_size_populations = {atom: {} for atom in atom_types}
#    for atom in atom_types:
#        size_pop = {}
#        for (composition, graph_id), _occ in cluster_histogram.items():
#            atom_count = 0
#            for entry in composition.split("_"):
#                parts = entry.split("-")
#                if len(parts) < 3:
#                    continue
#                _, atom_label, count = parts[0], parts[1], parts[2]
#                if atom_label == atom:
#                    atom_count += int(count)
#
#            if atom_count > 0:
#                size_pop[atom_count] = size_pop.get(atom_count, 0.0) + atom_populations[atom].get((composition, graph_id), 0.0)
#
#        cluster_size_populations[atom] = size_pop
#
#    with open("cluster_size.dat", "w") as f_size:
#        header = f"{'Cluster Size':<15}" + " ".join(f"{f'I({atom})':>15}" for atom in atom_types)
#        f_size.write(f"{header}\n")
#
#        max_size = max(
#            (max(sp.keys(), default=0) for sp in cluster_size_populations.values()),
#            default=0
#        )
#
#        for size in range(1, max_size + 1):
#            line = f"{size:<15}" + " ".join(
#                f"{cluster_size_populations[atom].get(size, 0.0):>15.10f}"
#                for atom in atom_types
#            )
#            f_size.write(f"{line}\n")


def draw_graph(graph, filename="graph.png"):
    from atomic_properties import elem_vdW, elem_color

    def get_element(atom_label):
        return "".join([c for c in atom_label if not c.isdigit()])

    plt.figure(figsize=(6, 6))

    node_sizes = [elem_vdW.get(get_element(graph.nodes[n]["element"]), 1) * 1000 for n in graph.nodes]
    node_colors = [elem_color.get(get_element(graph.nodes[n]["element"]), "gray") for n in graph.nodes]
    node_labels = {n: graph.nodes[n]["element"] for n in graph.nodes}

    pos = nx.spring_layout(graph, seed=42)
    nx.draw(
        graph, pos,
        with_labels=True, labels=node_labels,
        node_size=node_sizes, node_color=node_colors,
        edge_color="black",
        font_weight="bold", font_size=10, width=2.0
    )

    plt.savefig(filename, dpi=300)
    plt.close()



def post_process_clusters(cluster_histogram, graph_list, vis_format, frame_cluster_counts):
    graph_dict = {(composition, graph_id): graph for composition, graph_id, graph in graph_list}

    sorted_clusters = sorted(
        cluster_histogram.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1])
    )

    if vis_format:
        for i, ((composition, graph_id), count) in enumerate(sorted_clusters):
            if i > 200:
                break
            if (composition, graph_id) in graph_dict:
                filename = f"graph{i}_{composition}.{vis_format}"
                draw_graph(graph_dict[(composition, graph_id)], filename)

    # -------------------- Build per-frame count series --------------------
    # counts_per_frame[(composition, graph_id)] -> np.array shape (T,)
    counts_per_frame = {}

    if frame_cluster_counts:
        # Use the recorded per-frame series
        for key, series in frame_cluster_counts.items():
            counts_per_frame[key] = np.asarray(series, dtype=float)

        # Determine T and pad defensively
        T = max(len(s) for s in counts_per_frame.values())

        # Ensure all histogram keys are present
        for key in cluster_histogram.keys():
            if key not in counts_per_frame:
                counts_per_frame[key] = np.zeros(T, dtype=float)

        # Pad any shorter series
        for key, s in counts_per_frame.items():
            if len(s) < T:
                counts_per_frame[key] = np.pad(s, (0, T - len(s)), constant_values=0.0)
    else:
        # No per-frame info: treat each cluster count as a single-frame series
        # This recovers the old behaviour for I(P) etc, with std = 0
        for key, occ in cluster_histogram.items():
            counts_per_frame[key] = np.array([float(occ)], dtype=float)
        T = 1

    # -------------------- cluster_occurrences.dat (with mean/std) --------------------
    with open("cluster_occurrences.dat", "w") as f_occ:
        f_occ.write(f"{'Cluster':<30} {'Occurrences':>12} {'GraphID':<32} {'mean_per_frame':>16} {'std_per_frame':>16}\n")

        for (composition, graph_id), total_occ in sorted_clusters:
            series = counts_per_frame.get((composition, graph_id), np.zeros(T, dtype=float))
            mean = series.mean()
            std = series.std(ddof=1) if len(series) > 1 else 0.0

            f_occ.write(f"{composition:<30} {total_occ:>12d} {graph_id:<32} {mean:>16.6f} {std:>16.6f}\n")

    # -------------------- Atom types & atom counts per cluster --------------------
    # entries look like "1-O-2"
    atom_types = {
        entry.split("-")[1]
        for (composition, _), _ in cluster_histogram.items()
        for entry in composition.split("_")
        if len(entry.split("-")) >= 3
    }

    # atom_counts_per_cluster[atom][(composition,graph_id)] = N_atom(P)
    atom_counts_per_cluster = {atom: {} for atom in atom_types}
    for atom in atom_types:
        for (composition, graph_id) in cluster_histogram.keys():
            atom_count = 0
            for entry in composition.split("_"):
                parts = entry.split("-")
                if len(parts) < 3:
                    continue
                _, atom_label, count = parts[0], parts[1], parts[2]
                if atom_label == atom:
                    atom_count += int(count)
            if atom_count > 0:
                atom_counts_per_cluster[atom][(composition, graph_id)] = atom_count

    # -------------------- Per-atom cluster & size populations with mean/std --------------------
    atom_populations_mean = {atom: {} for atom in atom_types}
    atom_populations_std  = {atom: {} for atom in atom_types}
    cluster_size_populations_mean = {atom: {} for atom in atom_types}
    cluster_size_populations_std  = {atom: {} for atom in atom_types}

    for atom in atom_types:
        # Denominator per frame: sum_P n_t(P)*N_atom(P)
        denom = np.zeros(T, dtype=float)

        for key, series in counts_per_frame.items():
            N_atom = atom_counts_per_cluster[atom].get(key, 0)
            if N_atom == 0:
                continue
            denom += series * N_atom

        valid_mask = denom > 0
        if not np.any(valid_mask):
            # No frames where this atom type is present
            continue

        # Temporary storage for size-resolved series
        size_series = defaultdict(lambda: np.zeros(T, dtype=float))

        # Per-cluster I_t and stats
        for key in cluster_histogram.keys():
            N_atom = atom_counts_per_cluster[atom].get(key, 0)
            if N_atom == 0:
                atom_populations_mean[atom][key] = 0.0
                atom_populations_std[atom][key] = 0.0
                continue

            series = counts_per_frame.get(key, np.zeros(T, dtype=float))
            w = series * N_atom

            I_t = np.zeros(T, dtype=float)
            I_t[valid_mask] = w[valid_mask] / denom[valid_mask]

            vals = I_t[valid_mask]
            mean = vals.mean()
            std = vals.std(ddof=1) if len(vals) > 1 else 0.0

            atom_populations_mean[atom][key] = mean
            atom_populations_std[atom][key] = std

            size = N_atom
            size_series[size][valid_mask] += I_t[valid_mask]

        # Size-resolved means/stds
        for size, s_series in size_series.items():
            vals = s_series[valid_mask]
            mean = vals.mean()
            std = vals.std(ddof=1) if len(vals) > 1 else 0.0
            cluster_size_populations_mean[atom][size] = mean
            cluster_size_populations_std[atom][size] = std

    # -------------------- cluster_populations.dat (mean & std for I(P)) --------------------
    with open("cluster_populations.dat", "w") as f_pop:
        header = f"{'Cluster':<30}"
        for atom in atom_types:
            header += f"{f'I({atom})':>15}{f'std(I({atom}))':>15}"
        f_pop.write(header + "\n")

        sorted_comps = sorted(
            cluster_histogram.keys(),
            key=lambda comp: sum(
                atom_populations_mean[a].get((comp[0], comp[1]), 0.0) for a in atom_types
            ),
            reverse=True
        )

        for (composition, graph_id) in sorted_comps:
            key = (composition, graph_id)
            line = f"{composition:<30}"
            for atom in atom_types:
                mean = atom_populations_mean[atom].get(key, 0.0)
                std  = atom_populations_std[atom].get(key, 0.0)
                line += f"{mean:>15.10f}{std:>15.10f}"
            f_pop.write(line + "\n")

    # -------------------- cluster_size.dat (mean & std by cluster size) --------------------
    with open("cluster_size.dat", "w") as f_size:
        header = f"{'Cluster Size':<15}"
        for atom in atom_types:
            header += f"{f'I({atom})':>15}{f'std(I({atom}))':>15}"
        f_size.write(header + "\n")

        max_size = 0
        for atom in atom_types:
            if cluster_size_populations_mean[atom]:
                max_size = max(max_size, max(cluster_size_populations_mean[atom].keys()))

        for size in range(1, max_size + 1):
            line = f"{size:<15}"
            for atom in atom_types:
                mean = cluster_size_populations_mean[atom].get(size, 0.0)
                std  = cluster_size_populations_std[atom].get(size, 0.0)
                line += f"{mean:>15.10f}{std:>15.10f}"
            f_size.write(line + "\n")

