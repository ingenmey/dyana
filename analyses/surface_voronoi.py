#!/usr/bin/env python3

import pyvoro
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils import prompt

class Layer:
    """
    Represents a surface layer composed of multiple atoms.
    """
    def __init__(self, names: list, coords: np.ndarray, radii: np.ndarray):
        self.names = names
        self.coords = coords
        self.radii = radii

def get_surface_atoms(traj, z0, z1):
    names = []
    coords = []
    extras = []
    radii = []
    domains_red = []

    for compound in traj.compounds.values():
        for mol in compound.members:
            for i, (x, y, z) in enumerate(mol.coords):
                if (z < z0) or (z > z1):
                    continue
                coords.append([x, y, z])
                names.append(mol.symbols[i])
                radii.append(mol.atomic_radii[i])
                domains_red.append(compound.rep)
    return Layer(names, np.array(coords), np.array(radii)), domains_red

def remove_close_points(S, R, domains_red, min_distance=0.5):
    to_remove = set()
    for i in range(len(S)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(S)):
            if j in to_remove:
                continue
            if np.linalg.norm(S[i] - S[j]) < min_distance:
                if R[i] > R[j]:
                    to_remove.add(j)
                else:
                    to_remove.add(i)
                    break
    S = np.array([S[i] for i in range(len(S)) if i not in to_remove])
    R = np.array([R[i] for i in range(len(R)) if i not in to_remove])
    domains_red = [domains_red[i] for i in range(len(domains_red)) if i not in to_remove]
    return S, R, domains_red

def calculate_domain_volumes(result, domains_red, S):
    domain_volumes = {dom_id: 0.0 for dom_id in set(domains_red)}

    for i,cell in enumerate(result):
        dom_id = domains_red[i]
        domain_volumes[dom_id] += cell['volume']

    return domain_volumes

def compute_voronoi(S, R, dimx, dimy):
    try:
        result = pyvoro.compute_2d_voronoi(
            S,
            [[0, dimx], [0, dimy]],
            0.8,  # Approximate cell size, adjust if necessary
            radii=R,
            periodic=[True, True]
        )
        return result
    except pyvoro.voroplusplus.VoronoiPlusPlusError as e:
        print(str(e))
        return None

def visualize_voronoi(result, S, domains_red, dimx, dimy):
    fig, ax = plt.subplots()

    unique_domains = list(set(domains_red))
    domain_to_color = {dom: i for i, dom in enumerate(unique_domains)}
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_domains)))

    legend_handles = []

    # Draw Voronoi cells
    for i, cell in enumerate(result):
        dom_id = domains_red[i]
        color = colors[domain_to_color[dom_id] % len(colors)]
        polygon = patches.Polygon(cell['vertices'], closed=True, fill=True, edgecolor='black', facecolor=color)
        ax.add_patch(polygon)

    # Create legend handles
    for dom_id in unique_domains:
        color = colors[domain_to_color[dom_id] % len(colors)]
        legend_handles.append(patches.Patch(color=color, label=dom_id))

    # Draw the original points
    for point in S:
        ax.plot(point[0], point[1], 'ro')

    # Set the limits and aspect ratio
    ax.set_xlim(0, dimx)
    ax.set_ylim(0, dimy)
    ax.set_aspect('equal')

    plt.title('2D Radical Voronoi Diagram with Periodic Boundaries')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)

    # Add the legend to the plot
    ax.legend(handles=legend_handles, loc='upper right')

    plt.show()

# --- Main entry point --------------------------------------------------------

def process_frame(traj, z0, z1):
    for compound in traj.compounds.values():
        for molecule in compound.members:
            molecule.update_coords(traj.coords)

    dimx = traj.box_size[0]
    dimy = traj.box_size[1]

    adlayer, domains_red = get_surface_atoms(traj, z0, z1)

    S = np.array(adlayer.coords[:, 0:2])
    R = np.array(adlayer.radii)

    # Remove points that are too close to each other
    S, R, domains_red = remove_close_points(S, R, domains_red, 0.8)

    result = compute_voronoi(S, R, dimx, dimy)

    if result is None:
        return None

    # Calculate and print domain volumes
    domain_volumes = calculate_domain_volumes(result, domains_red, S)

    # Visualization using matplotlib
#    visualize_voronoi(result, S, domains_red, dimx, dimy)

    return domain_volumes

def surface_voronoi(traj):
    z0 = float(prompt("Enter minimum z value in Å: "))
    z1 = float(prompt("Enter maximum z value in Å: "))

    domain_areas = {compound.rep: 0.0 for compound in traj.compounds.values()}
    num_frames = 0

    while True:
        try:
            # Update the coordinates and COMs for each compound
            for compound in traj.compounds.values():
                for molecule in compound.members:
                    molecule.update_coords(traj.coords)

            # Perform surface voronoi analysis frame by frame
            result = process_frame(traj, z0, z1)
            if result is not None:
                num_frames += 1
                for domain in domain_areas:
                    domain_areas[domain] = (1.0-1.0/num_frames) * domain_areas[domain] + result.get(domain, 0.0)/num_frames
                print(num_frames, domain_areas)

            # Read the next frame
            traj.read_frame()
        except:
            # End of the trajectory file
            break

