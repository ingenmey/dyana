import os
import numpy as np
import re
from scipy.spatial import cKDTree
from collections import Counter
from itertools import combinations
from utils import label_matches, prompt, prompt_int, prompt_float, prompt_yn, prompt_choice

import networkx as nx
import hashlib
import matplotlib.pyplot as plt

def cluster(traj):
    # Prompt user for clustering parameters per compound
    compound_atom_labels = {}
    for i, compound in enumerate(traj.compounds.values(), start=1):
        atom_labels = prompt(f"Enter the atom types in compound {i} to be considered for clustering (comma-separated): ", "").strip()
        if atom_labels:
            compound_atom_labels[i] = [label.strip() for label in atom_labels.split(',') if label.strip()]
        else:
            compound_atom_labels[i] = []  # Explicitly set as an empty list

    # Prompt user for cutoff distances for each unique atom pair across compounds
    cutoff_distances = {}
    compound_atom_pairs = []
    for (comp1, atoms1), (comp2, atoms2) in combinations(compound_atom_labels.items(), 2):
        for atom1 in atoms1:
            for atom2 in atoms2:
                cutoff = prompt_float(f"Enter the cut-off distance for {atom1} in compound {comp1} and {atom2} in compound {comp2} (in Å): ", 0.0, minval=0.0)
                cutoff_distances[((comp1, atom1), (comp2, atom2))] = cutoff
                cutoff_distances[((comp2, atom2), (comp1, atom1))] = cutoff  # Ensure symmetry
                compound_atom_pairs.append(((comp1, atom1), (comp2, atom2)))

    # Prompt user if cluster graphs should be drawn
    shouldHash = prompt_yn("Count clusters by composition (n) or by composition and graph hash (y)?", True)
    visFormat = prompt_yn("Visualize cluster graphs?", False)
    if visFormat:
        visFormat = prompt_choice("Save cluster in which format?", ["svg", "png"], "svg")
    isSaveXYZ = prompt_yn("Save cluster coordinates as XYZ files?", False)
    isSaveWhole = False
    if isSaveXYZ:
        isSaveWhole = prompt_yn("Save whole molecules (Y) or only specified atom types (N)?", False)

    start_frame =  prompt_int("In which trajectory frame to start processing the trajectory?", 1, minval=1)
    nframes =      prompt_int("How many trajectory frames to read (from this position on)?", -1, "all")
    frame_stride = prompt_int("Use every n-th read trajectory frame for the analysis:", 1, minval=1)
    frame_idx = 0
    processed_frames = 0

    # Initialize histogram
    cluster_histogram = Counter()

    # Loop through all frames
    graph_list = []
    seen_graphs = set()  # Track unique graphs

    if (start_frame > 1):
        print(f"Skipping forward to frame {start_frame}.")
        while (frame_idx < start_frame - 1):
            traj.read_frame()
            frame_idx += 1


    while (nframes != 0):
        try:
            # Update coordinates for compounds
            for compound in traj.compounds.values():
                for molecule in compound.members:
                    molecule.update_coords(traj.coords)
                compound.update_coms(traj.box_size)


            # Gather Atoms per compound/label
            atom_groups = {}
            for comp_id, compound in enumerate(traj.compounds.values(), start=1):
                for molecule in compound.members:
                    for label, idx in molecule.label_to_id.items():
                        for user_label in compound_atom_labels[comp_id]:
                            if label_matches(user_label, label):
                                atom_groups.setdefault((comp_id, user_label), []).append(molecule.atoms[idx])



            # Identify clusters as graphs
            clusters = identify_clusters(atom_groups, compound_atom_labels, cutoff_distances, traj.box_size)

            if isSaveXYZ and not os.path.exists("xyz"):
                os.makedirs("xyz")

            # Store graphs and their occurrences
            for composition, graph, cluster_atoms in clusters:
                if shouldHash:
                    graph_id = get_graph_id(graph)
                else:
                    graph_id = 0
                cluster_histogram[(composition, graph_id)] += 1
                if (composition, graph_id) not in seen_graphs:
                    seen_graphs.add((composition, graph_id))
                    graph_list.append((composition, graph_id, graph))
                if isSaveXYZ:
                    if len(cluster_atoms) > 1:
                        write_xyz(f"{composition}_{graph_id}.xyz", cluster_atoms, isSaveWhole, traj.box_size)


            processed_frames += 1
            print(f"\rProcessed {processed_frames} frames (current frame {frame_idx+1})", end="")

            for _ in range(frame_stride):
                frame_idx += 1
                nframes -= 1
                traj.read_frame()

        except ValueError:
            # End of trajectory file
            break

        except KeyboardInterrupt:
            # Graceful exit when user presses Ctrl+C
            print("\nInterrupt received! Exiting main loop and post-processing data...")
            break

    # Sort clusters by atom type counts
    sorted_compositions = sorted(cluster_histogram.items(), key=lambda item: item[0])

    # Output cluster histogram
    print("\nCluster Composition Histogram:")
    for (composition, graph_id), count in sorted_compositions:
        print(f"Composition {composition}: {count} occurrences")

    # Post-process cluster to compute populations
    post_process_clusters(cluster_histogram, graph_list, visFormat)


def write_xyz(filename, atoms, isSaveWhole, box_size):
    def unwrap_coords(coords, box_size):
        """Unwrap molecule coordinates to avoid broken bonds due to periodic boundary conditions."""
        unwrapped = np.copy(coords)
        for i in range(1, len(unwrapped)):
            delta = unwrapped[i] - unwrapped[i - 1]
            delta -= box_size * np.round(delta / box_size)
            unwrapped[i] = unwrapped[i - 1] + delta
        return unwrapped

    symbols = []
    coords = []

    if isSaveWhole:
        # Collect entire molecules but avoid duplicates
        written_molecules = set()

        # Sort atoms by compound ID of their molecule
        atoms_sorted = sorted(atoms, key=lambda atom: (atom.parent_molecule.comp_id, atom.parent_molecule.mol_id))

        for atom in atoms_sorted:
            mol = atom.parent_molecule
            if mol not in written_molecules:
                written_molecules.add(mol)
                symbols.extend(mol.symbols)
                coords.extend(mol.coords)
    else:
        # Only save exact atoms participating
        # Sort atoms by compound ID of their molecule
        atoms_sorted = sorted(atoms, key=lambda atom: (atom.parent_molecule.comp_id, atom.parent_molecule.mol_id))

        for atom in atoms_sorted:
            symbols.append(atom.parent_molecule.symbols[atom.parent_molecule.atoms.index(atom)])
            coords.append(atom.coord)

    coords = np.array(coords)
    coords = unwrap_coords(coords, box_size)
    cog = np.mean(coords, axis=0)
    coords -= cog

    with open("xyz/" + filename, 'a') as f:
        f.write(f"{len(symbols)}\n")
        f.write("Generated by cluster analysis\n")
        for symbol, coord in zip(symbols, coords):
            f.write(f"{symbol} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")


def identify_clusters(atom_groups, compound_atom_labels, cutoff_distances, box_size):
    # Build KDTree per (comp_id, label)
    kdtrees = {
        (comp_id, atom_label): cKDTree([atom.coord for atom in atoms], boxsize=box_size)
        for (comp_id, atom_label), atoms in atom_groups.items()
    }

    # Precompute valid neighbor lookups
    valid_neighbors = {
    (comp_id, atom_label): [
        (other_comp, other_atom_label)
        for (other_comp, other_atom_label) in atom_groups.keys()
        if ((comp_id, atom_label), (other_comp, other_atom_label)) in cutoff_distances
        ]
        for (comp_id, atom_label) in atom_groups.keys()
    }


    visited = set()
    clusters = []

    def grow_cluster(atom, comp_id, atom_label, atom_idx, cluster, graph, atom_counts):
        if atom in visited:
            return
        visited.add(atom)

        graph.add_node(atom, element=atom_label)  # Directly use Atom as node
        cluster.append(atom)

        key = f"{comp_id}-{atom_label}"
        atom_counts[key] = atom_counts.get(key, 0) + 1

        for (other_comp, other_label) in valid_neighbors[(comp_id, atom_label)]:
            neighbors = kdtrees[(other_comp, other_label)].query_ball_point(
                atom.coord,
                cutoff_distances[((comp_id, atom_label), (other_comp, other_label))]
            )
            for neighbor_idx in neighbors:
                neighbor_atom = atom_groups[(other_comp, other_label)][neighbor_idx]
                if neighbor_atom not in visited:
                    grow_cluster(neighbor_atom, other_comp, other_label, neighbor_idx, cluster, graph, atom_counts)
                # Always add the edge
                graph.add_edge(atom, neighbor_atom)

    # Loop over atoms and grow clusters
    for (comp_id, atom_label), atoms in atom_groups.items():
        for idx, atom in enumerate(atoms):
            if atom not in visited:
                cluster = []
                atom_counts = {}
                graph = nx.Graph()
                grow_cluster(atom, comp_id, atom_label, idx, cluster, graph, atom_counts)

                composition = "_".join(f"{key}-{count}" for key, count in sorted(atom_counts.items()))
                clusters.append((composition, graph, cluster))


    return clusters


def get_graph_id(graph):
    """
    Generate a unique identifier for a graph using a canonical form.
    """
    # Convert graph to canonical string (sorted adjacency list)
    graph_string = nx.weisfeiler_lehman_graph_hash(graph)

    # Hash the string for compact storage
    return hashlib.md5(graph_string.encode()).hexdigest()


def post_process_clusters(cluster_histogram, graph_list, visFormat):
    # Create a mapping from (composition, graph_id) to graph
    graph_dict = {(composition, graph_id): graph for composition, graph_id, graph in graph_list}

    # Sort clusters once (by occurrences descending, then by composition and graph_id ascending)
    sorted_clusters = sorted(cluster_histogram.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))

    # Save images for visualization if requested
    if visFormat:
        for i, ((composition, graph_id), count) in enumerate(sorted_clusters):
            if i > 200:
                break
            if (composition, graph_id) in graph_dict:
                filename = f"graph{i}_{composition}.{visFormat}"
                draw_graph(graph_dict[(composition, graph_id)], filename)

    # Save cluster occurrences
    with open("cluster_occurrences.dat", "w") as f_occ:
        f_occ.write("Graph ID    Occurrences\n")
        # Sort by occurrences (descending), then by composition and graph_id (ascending)
        for (composition, graph_id), count in sorted_clusters:
            f_occ.write(f"{composition:<30} {count:<10} {graph_id}\n")

    # Extract unique atom types from cluster compositions
    atom_types = {atom for (composition, _), _ in cluster_histogram.items()
                  for entry in composition.split('_')
                  for _, atom, _ in [entry.split('-')]}

    # Compute populations I(P) for each atom type
    atom_populations = {atom: {} for atom in atom_types}
    for atom in atom_types:
        weighted_occurrences = {}
        total_weighted = 0

        # Calculate weighted occurrences for each cluster
        for (composition, graph_id), occ in cluster_histogram.items():
            atom_count = sum(int(count) for _, atom_label, count in (entry.split('-') for entry in composition.split('_')) if atom_label == atom)
            wOcc_P = occ * atom_count
            weighted_occurrences[(composition, graph_id)] = wOcc_P
            total_weighted += wOcc_P

        # Normalize to get I(P) and handle zero division
        for (composition, graph_id), wOcc_P in weighted_occurrences.items():
            atom_populations[atom][(composition, graph_id)] = wOcc_P / total_weighted if total_weighted > 0 else 0.0

    # Save cluster populations I(P) for each atom type
    with open("cluster_populations.dat", "w") as f_pop:
        header = f"{'Cluster':<30}" + " ".join(f"{f'I({atom})':>15}" for atom in atom_types)
        f_pop.write(f"{header}\n")

        # Sort by the sum of populations for each cluster across all atom types
        sorted_compositions = sorted(cluster_histogram.keys(), key=lambda comp: sum(atom_populations[atom].get((comp[0], comp[1]), 0.0) for atom in atom_types), reverse=True)

        for (composition, graph_id) in sorted_compositions:
            line = f"{composition:<30}" + ' '.join(f"{atom_populations[atom].get((composition, graph_id), 0.0):>15.10f}" for atom in atom_types)
            f_pop.write(f"{line}\n")

    # Compute summarized cluster populations for each atom type by cluster size
    cluster_size_populations = {atom: {} for atom in atom_types}
    for atom in atom_types:
        size_populations = {}

        # Sum populations by cluster size
        for (composition, graph_id), count in cluster_histogram.items():
            atom_count = sum(int(count) for _, atom_label, count in (entry.split('-') for entry in composition.split('_')) if atom_label == atom)
            if atom_count > 0:
                size_populations[atom_count] = size_populations.get(atom_count, 0) + atom_populations[atom].get((composition, graph_id), 0.0)

        cluster_size_populations[atom] = size_populations

    # Save summarized cluster populations by cluster size
    with open("cluster_size.dat", "w") as f_size:
        header = f"{'Cluster Size':<15}" + " ".join(f"{f'I({atom})':>15}" for atom in atom_types)
        f_size.write(f"{header}\n")

        max_size = max((max(size_populations.keys(), default=0) for size_populations in cluster_size_populations.values()), default=0)
        for size in range(1, max_size + 1):
            line = f"{size:<15}" + " ".join(f"{cluster_size_populations[atom].get(size, 0.0):>15.10f}" for atom in atom_types)
            f_size.write(f"{line}\n")


def draw_graph(graph, filename="graph.png"):
    from atomic_properties import elem_masses, elem_vdW, elem_covalent, elem_number, elem_color  # Import atomic properties
    """
    Draws a given graph and saves it as an image.
    """
    def get_element(atom_label):
        return ''.join([c for c in atom_label if not c.isdigit()])

    plt.figure(figsize=(6,6))

    node_sizes = [elem_vdW.get(get_element(graph.nodes[n]["element"]), 1) * 1000 for n in graph.nodes]
    node_colors = [elem_color.get(get_element(graph.nodes[n]["element"]), "gray") for n in graph.nodes]

    node_labels = {n: graph.nodes[n]["element"] for n in graph.nodes}

    # Draw the graph
    pos = nx.spring_layout(graph, seed=42)  # Layout for visualization
    nx.draw(graph, pos, with_labels=True, labels=node_labels, node_size=node_sizes, node_color=node_colors, edge_color="black", font_weight="bold", font_size=10, width=2.0)

    # Save file
    plt.savefig(filename, dpi=300)
    plt.close()





