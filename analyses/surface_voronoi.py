# analyses/surface_voronoi.py

import os
import pyvoro
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import cKDTree
from collections import defaultdict
from utils import prompt_float, prompt_int, prompt_yn


def get_surface_atoms(traj, z0, z1):
    atoms = []
    for compound in traj.compounds.values():
        for mol in compound.members:
            for i, (x, y, z) in enumerate(mol.coords):
                if z0 <= z <= z1:
                    atoms.append({
                        'coord': [x, y, z],
                        'radius': mol.atomic_radii[i],
                        'symbol': mol.symbols[i],
                        'compound': compound.rep,
                        'molecule': mol,
                        'atom_id': mol.atom_ids[i]
                    })
    return atoms


def remove_close_points_enhanced(atoms, min_distance, box_size):
    dimx, dimy = box_size
    box = np.array([dimx, dimy])

    coords = np.array([a['coord'][:2] for a in atoms])
    radii = np.array([a['radius'] for a in atoms])
    tree = cKDTree(coords, boxsize=box)
    to_remove = set()

    for i in range(len(coords)):
        if i in to_remove:
            continue
        neighbors = tree.query_ball_point(coords[i], r=min_distance)
        for j in neighbors:
            if j == i or j in to_remove:
                continue
            delta = coords[i] - coords[j]
            delta -= box * np.round(delta / box)
            dist = np.linalg.norm(delta)
            if dist < min_distance:
                if radii[i] >= radii[j]:
                    to_remove.add(j)
                else:
                    to_remove.add(i)
                    break

    atoms_filtered = [a for i, a in enumerate(atoms) if i not in to_remove]
    coords_filtered = np.array([a['coord'][:2] for a in atoms_filtered])
    radii_filtered = np.array([a['radius'] for a in atoms_filtered])
    return atoms_filtered, coords_filtered, radii_filtered


def compute_voronoi(S, R, dimx, dimy):
    try:
        return pyvoro.compute_2d_voronoi(
            S, [[0, dimx], [0, dimy]],
            0.8, radii=R, periodic=[True, True]
        )
    except pyvoro.voroplusplus.VoronoiPlusPlusError as e:
        print(f"Voronoi error: {str(e)}")
        return None


def build_compound_graph(voronoi_cells, atoms):
    G = nx.Graph()
    for i, cell in enumerate(voronoi_cells):
        G.add_node(i, compound=atoms[i]['compound'])
        for face in cell['faces']:
            j = face['adjacent_cell']
            if j >= 0 and atoms[i]['compound'] == atoms[j]['compound']:
                G.add_edge(i, j)
    return G


def accumulate_histograms(hist_dict, values_by_domain, bin_count, area_min, area_max):
    for key, areas in values_by_domain.items():
        hist, _ = np.histogram(areas, bins=bin_count, range=(area_min, area_max))
        hist_dict[key] += hist


def compute_compactness_for_sets(voronoi_cells, atoms, sets):
    """Compute compactness (isoperimetric quotient) for arbitrary groups of Voronoi cells."""
    compactness_by_label = {}
    for label, indices in sets.items():
        area = sum(voronoi_cells[i]['volume'] for i in indices)
        perimeter = 0.0
        seen_edges = set()

        for i in indices:
            for face in voronoi_cells[i]['faces']:
                j = face['adjacent_cell']
                if j < 0 or j not in indices:
                    verts = face['vertices']
                    pts = [voronoi_cells[i]['vertices'][v] for v in verts]
                    for k in range(len(pts)):
                        a = tuple(pts[k])
                        b = tuple(pts[(k + 1) % len(pts)])
                        edge = tuple(sorted((a, b)))
                        if edge not in seen_edges:
                            seen_edges.add(edge)
                            perimeter += np.linalg.norm(np.array(a) - np.array(b))

        if perimeter > 0:
            Q = (4 * math.pi * area) / (perimeter ** 2)
            compactness_by_label[label] = Q
    return compactness_by_label

def visualize_voronoi(result, atoms, dimx, dimy, filename="voronoi_vis.png"):
    from matplotlib import cm
    from atomic_properties import elem_vdW, elem_color

    def get_element(symbol):
        return ''.join(c for c in symbol if not c.isdigit())

    fig, ax = plt.subplots()
    compounds = sorted(set(a["compound"] for a in atoms))
    compound_to_color = {comp: i for i, comp in enumerate(compounds)}
    base_colors = cm.get_cmap("Blues")(np.linspace(0.3, 0.9, len(compounds)))
    legend_handles = []

    for i, cell in enumerate(result):
        comp = atoms[i]["compound"]
        color = base_colors[compound_to_color[comp]]
        polygon = patches.Polygon(cell['vertices'], closed=True, fill=True,
                                  edgecolor='black', facecolor=color, linewidth=0.7)
        ax.add_patch(polygon)

    for atom in atoms:
        x, y = atom["coord"][:2]
        element = get_element(atom["symbol"])
        color = elem_color.get(element, "gray")
        radius = elem_vdW.get(element, 1.0) * 0.15
        ax.add_patch(plt.Circle((x, y), radius, color=color, zorder=10, linewidth=0.3))

    ax.set_xlim(0, dimx)
    ax.set_ylim(0, dimy)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    for comp in compounds:
        color = base_colors[compound_to_color[comp]]
        legend_handles.append(patches.Patch(color=color, label=comp))
    ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0, frameon=False, fontsize=8)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def process_frame(traj, z0, z1, min_distance):
    dimx, dimy = traj.box_size[:2]
    atoms = get_surface_atoms(traj, z0, z1)
    atoms, coords, radii = remove_close_points_enhanced(atoms, min_distance, (dimx, dimy))
    voronoi_cells = compute_voronoi(coords, radii, dimx, dimy)
    return voronoi_cells, atoms if voronoi_cells else (None, atoms)


def surface_voronoi(traj):
    z0 = prompt_float("Enter minimum z value in Å: ", 0.0)
    z1 = prompt_float("Enter maximum z value in Å: ", 10.0)
    min_distance = prompt_float("Minimum distance between surface atoms (Å):", 0.8, minval=0.0)
    do_visualize = prompt_yn("Visualize Voronoi diagram?", False)

    output_filename = "voronoi_areas.dat"
    area_dist_filename = "voronoi_area_distributions.dat"
    mol_dist_filename = "voronoi_molecular_distributions.dat"
    cluster_dist_filename = "voronoi_cluster_distributions.dat"
    cluster_count_filename = "voronoi_cluster_counts.dat"
    compactness_filename = "voronoi_cluster_compactness.dat"
    global_compactness_filename = "voronoi_global_compactness.dat"

    domain_areas = {compound.rep: 0.0 for compound in traj.compounds.values()}

    dimx, dimy = traj.box_size[:2]
    area_min = 0.0
    area_max = int(np.ceil(dimx * dimy / 100.0) * 100.0)

    bin_count = prompt_int("Enter number of bins for area distribution histogram:", area_max * 2, minval=1)
    area_histograms = defaultdict(lambda: np.zeros(bin_count, dtype=np.float64))
    molecular_histograms = defaultdict(lambda: np.zeros(bin_count, dtype=np.float64))
    cluster_histograms = defaultdict(lambda: np.zeros(bin_count, dtype=np.float64))
    cluster_count_bins = defaultdict(lambda: defaultdict(int))
    compactness_by_domain = defaultdict(list)
    global_compactness_histograms = defaultdict(lambda: np.zeros(compactness_bin_count, dtype=np.float64))

    compactness_bin_count = 100
    compactness_bins = np.linspace(0, 1, compactness_bin_count + 1)
    compactness_centers = 0.5 * (compactness_bins[:-1] + compactness_bins[1:])
    compactness_histograms = defaultdict(lambda: np.zeros(compactness_bin_count, dtype=np.float64))

    start_frame = prompt_int("In which trajectory frame to start processing the trajectory?", 1, minval=1)
    nframes = prompt_int("How many trajectory frames to read (from this position on)?", -1, "all")
    frame_stride = prompt_int("Use every n-th read trajectory frame for the analysis:", 1, minval=1)

    frame_idx = 0
    processed_frames = 0

    if start_frame > 1:
        print(f"Skipping forward to frame {start_frame}.")
        while frame_idx < start_frame - 1:
            traj.read_frame()
            frame_idx += 1

    with open(output_filename, "w") as fout:
        fout.write("# Frame Domain Areas (Å²)\n")

        while nframes != 0:
            try:
                traj.update_molecule_coords()
                voronoi_cells, atoms = process_frame(traj, z0, z1, min_distance)

                if voronoi_cells is not None:
                    processed_frames += 1
                    domains_red = [a['compound'] for a in atoms]

                    # Atomic cell areas
                    frame_areas = defaultdict(float)
                    per_domain_areas = defaultdict(list)
                    for i, cell in enumerate(voronoi_cells):
                        domain = domains_red[i]
                        frame_areas[domain] += cell['volume']
                        per_domain_areas[domain].append(cell["volume"])

                    for domain in domain_areas:
                        domain_areas[domain] = (
                            (1.0 - 1.0 / processed_frames) * domain_areas[domain]
                            + frame_areas.get(domain, 0.0) / processed_frames
                        )
                    accumulate_histograms(area_histograms, per_domain_areas, bin_count, area_min, area_max)

                    # Molecular cell areas
                    mol_areas = defaultdict(float)
                    for atom, cell in zip(atoms, voronoi_cells):
                        key = (atom['compound'], atom['molecule'])
                        mol_areas[key] += cell['volume']
                    mol_by_domain = defaultdict(list)
                    for (comp, _), area in mol_areas.items():
                        mol_by_domain[comp].append(area)
                    accumulate_histograms(molecular_histograms, mol_by_domain, bin_count, area_min, area_max)

                    # Cluster areas
                    cluster_by_domain = defaultdict(list)
                    cluster_counts_this_frame = defaultdict(int)
                    G = build_compound_graph(voronoi_cells, atoms)
                    for component in nx.connected_components(G):
                        comp = atoms[next(iter(component))]['compound']
                        area = sum(voronoi_cells[i]['volume'] for i in component)
                        cluster_by_domain[comp].append(area)
                        cluster_counts_this_frame[comp] += 1
                    accumulate_histograms(cluster_histograms, cluster_by_domain, bin_count, area_min, area_max)

                    # Update cluster count histogram
                    for comp, count in cluster_counts_this_frame.items():
                        cluster_count_bins[comp][count] += 1

                    # Cluster compactness analysis
                    # Build {compound: [list of cluster index sets]}
                    clusters_by_compound = defaultdict(list)
                    for component in nx.connected_components(G):
                        comp = atoms[next(iter(component))]['compound']
                        clusters_by_compound[comp].append(component)

                    # Flatten and compute compactness
                    for comp, cluster_list in clusters_by_compound.items():
                        for cluster in cluster_list:
                            q = compute_compactness_for_sets(voronoi_cells, atoms, {comp: cluster}).get(comp)
                            if q is not None:
                                compactness_by_domain[comp].append(q)


                    # --- Global compactness (all Voronoi cells of compound) ---
                    compound_cells = defaultdict(list)
                    for i, atom in enumerate(atoms):
                        compound_cells[atom['compound']].append(i)

                    global_q = compute_compactness_for_sets(voronoi_cells, atoms, compound_cells)
                    for comp, q in global_q.items():
                        hist, _ = np.histogram([q], bins=compactness_bins)
                        global_compactness_histograms[comp] += hist


                    fout.write(f"{frame_idx + 1} " + " ".join(f"{domain_areas[d]:.4f}" for d in domain_areas) + "\n")
                    print(f"\rProcessed {processed_frames} frames (current frame {frame_idx+1})", end="")

                for _ in range(frame_stride):
                    frame_idx += 1
                    nframes -= 1
                    traj.read_frame()

            except ValueError:
                print("\nReached end of trajectory.")
                break
            except KeyboardInterrupt:
                print("\nAnalysis interrupted by user.")
                break

    print(f"\nAverage domain areas written to {output_filename}")

    # Output histograms
    bin_edges = np.linspace(area_min, area_max, bin_count + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    def write_distribution_file(filename, histograms, title):
        with open(filename, "w") as f:
            f.write(f"# {title}\n")
            f.write("# Columns: Bin_Center " + " ".join(sorted(histograms.keys())) + "\n")
            for i in range(bin_count):
                row = [f"{bin_centers[i]:.4f}"]
                for domain in sorted(histograms.keys()):
                    hist = histograms[domain]
                    total = np.sum(hist)
                    prob = hist[i] / total if total > 0 else 0.0
                    row.append(f"{prob:.6f}")
                f.write(" ".join(row) + "\n")

    write_distribution_file(area_dist_filename, area_histograms, "Atomic Voronoi cell area distributions")
    write_distribution_file(mol_dist_filename, molecular_histograms, "Molecular Voronoi area distributions")
    write_distribution_file(cluster_dist_filename, cluster_histograms, "Cluster Voronoi area distributions")

    print(f"Atomic distributions saved to {area_dist_filename}")
    print(f"Molecular distributions saved to {mol_dist_filename}")
    print(f"Cluster distributions saved to {cluster_dist_filename}")

    # --- Write cluster count histogram with full range and aligned formatting ---
    with open(cluster_count_filename, "w") as fcount:
        fcount.write("# Cluster count histogram per compound (percentage of frames)\n")

        compounds_sorted = sorted(cluster_count_bins.keys())
        max_cluster_count = max((max(d.keys(), default=0) for d in cluster_count_bins.values()), default=0)

        header = "{:<4}".format("Count") + "".join(f"{comp:>8}" for comp in compounds_sorted) + "\n"
        fcount.write(header)

        # Compute and write weighted average per compound
        averages = []
        for comp in compounds_sorted:
            total = sum(cluster_count_bins[comp].values())
            weighted_sum = sum(c * n for c, n in cluster_count_bins[comp].items())
            avg = (weighted_sum / total) if total > 0 else 0.0
            averages.append(avg)

        fcount.write("# Avg  " + "".join(f"{avg:8.2f}" for avg in averages) + "\n")

        for count in range(1, max_cluster_count + 1):
            row = f"{count:<4}"
            for comp in compounds_sorted:
                total = sum(cluster_count_bins[comp].values())
                percent = (cluster_count_bins[comp].get(count, 0) / total * 100) if total > 0 else 0.0
                row += f"{percent:8.2f}"
            fcount.write(row + "\n")

    print(f"Cluster count histogram saved to {cluster_count_filename}")

    for comp, values in compactness_by_domain.items():
        hist, _ = np.histogram(values, bins=compactness_bins)
        compactness_histograms[comp] += hist

    with open(compactness_filename, "w") as f:
        f.write("# Compactness ratio (isoperimetric quotient) distributions per compound\n")
        f.write("# Columns: Bin_Center " + " ".join(sorted(compactness_histograms.keys())) + "\n")
        for i in range(compactness_bin_count):
            row = [f"{compactness_centers[i]:.4f}"]
            for comp in sorted(compactness_histograms.keys()):
                h = compactness_histograms[comp]
                total = np.sum(h)
                prob = h[i] / total if total > 0 else 0.0
                row.append(f"{prob:.6f}")
            f.write(" ".join(row) + "\n")

    print(f"Cluster compactness distributions saved to {compactness_filename}")


    with open(global_compactness_filename, "w") as f:
        f.write("# Global compactness ratio distributions per compound (entire compound shape per frame)\n")
        f.write("# Columns: Bin_Center " + " ".join(sorted(global_compactness_histograms.keys())) + "\n")
        for i in range(compactness_bin_count):
            row = [f"{compactness_centers[i]:.4f}"]
            for comp in sorted(global_compactness_histograms.keys()):
                h = global_compactness_histograms[comp]
                total = np.sum(h)
                prob = h[i] / total if total > 0 else 0.0
                row.append(f"{prob:.6f}")
            f.write(" ".join(row) + "\n")
    print(f"Global compactness distributions saved to {global_compactness_filename}")



    if do_visualize:
        visualize_voronoi(voronoi_cells, atoms, dimx, dimy)

