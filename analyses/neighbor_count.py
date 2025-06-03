# analyses/neighbor_count.py

import numpy as np
from collections import Counter
from scipy.spatial import cKDTree
from utils import label_matches, prompt, prompt_int, prompt_float, prompt_yn

def neighbor_count(traj):
    # Prompt user for RDF parameters
    print("\nAvailable Compounds:")
    for i, compound in enumerate(traj.compounds.values(), start=1):
        print(f"{i}: {compound.rep} (Number: {len(compound.members)})")

    ref_index = prompt_int("Choose the reference compound (number): ", 1, minval=1) - 1
    obs_index = prompt_int("Choose the observed compound (number): ", 1, minval=1) - 1
    ref_labels = [s.strip() for s in prompt("Which atom(s) in reference compound? (comma-separated) ").split(',')]
    obs_labels = [s.strip() for s in prompt("Which atom(s) in observed compound? (comma-separated) ").split(',')]

    r_cut = prompt_float("Neighbour cutoff distance Å: ", 3.5, minval=0.1)

    update_compounds = prompt_yn("Perform molecule recognition and update compound list in each frame?", False)

    start_frame = prompt_int("In which trajectory frame to start processing the trajectory?", 1, minval=1)
    nframes = prompt_int("How many trajectory frames to read (from this position on)?", -1, "all")
    frame_stride = prompt_int("Use every n-th read trajectory frame for the analysis:", 1, minval=1)

    # ------------------- helper to collect indices --------------------
    def get_indices(comp, labels):
        return [
            idx for mol in comp.members
            for lab, idx in mol.label_to_global_id.items()
            if any(label_matches(lab_in, lab) for lab_in in labels)
        ]

    # --- Precompute atom indices ---
    ref_compound = list(traj.compounds.values())[ref_index]
    obs_compound = list(traj.compounds.values())[obs_index]
    ref_key = next(k for k, c in traj.compounds.items() if c is ref_compound)
    obs_key = next(k for k, c in traj.compounds.items() if c is obs_compound)

    ref_indices = get_indices(ref_compound, ref_labels)
    obs_indices = get_indices(obs_compound, obs_labels)


    if not ref_indices or not obs_indices:
        print("No atoms matched the given labels.")
        return

    # --- Setup histogram ---
    n_hist = Counter()      # n -> occurrences
    total_ref_atoms = 0


    # --- Frame loop ---
    frame_idx = 0
    processed_frames = 0
    if start_frame > 1:
        print(f"Skipping forward to frame {start_frame}.")
        while frame_idx < start_frame - 1:
            traj.read_frame()
            frame_idx += 1

    while nframes != 0:
        try:
            if update_compounds:
                traj.guess_molecules()
                traj.update_molecule_coords()
                try:
                    ref_compound = traj.compounds[ref_key]
                    obs_compound = traj.compounds[obs_key]
                except KeyError:
                    # compound disappeared this frame – skip
                    frame_idx += 1
                    nframes -= 1
                    traj.read_frame()
                    continue

                ref_indices = get_indices(ref_compound, ref_labels)
                obs_indices = get_indices(obs_compound, obs_labels)

                if not ref_indices or not obs_indices:
                    frame_idx += 1
                    nframes -= 1
                    traj.read_frame()
                    continue

            else:
                traj.update_molecule_coords()


            total_ref_atoms += len(ref_indices)

            # KD-tree on observed atoms
            obs_coords = traj.coords[obs_indices]
            tree = cKDTree(obs_coords, boxsize=traj.box_size)

            # for every reference atom: neighbour count
            ref_coords = traj.coords[ref_indices]
            neighbours = tree.query_ball_point(ref_coords, r_cut)

            for nb_list in neighbours:
                n_hist[len(nb_list)] += 1


            processed_frames += 1
            if processed_frames % 100 == 0:
                print(f"Processed {processed_frames} frames (current frame {frame_idx+1})")
#            print(f"\rProcessed {processed_frames} frames (current frame {frame_idx+1})", end="")

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

    print()

    # ------------------- probability P(n) -----------------------------
    if total_ref_atoms == 0:
        print("No reference atoms found — nothing to write.")
        return

    max_n = max(n_hist) if n_hist else 0
    probs = {n: n_hist[n] / total_ref_atoms for n in range(max_n+1)}

    # ------------------- write output ---------------------------------
    fname = "ncount.dat"
    with open(fname, "w") as f:
        f.write(f"# P(n)   cutoff = {r_cut:.2f} Å\n")
        f.write("#  n   P(n)\n")
        for n in range(max_n+1):
            f.write(f"{n:3d}  {probs.get(n,0):.6f}\n")

    print(f"Neighbour-count distribution written to {fname}")

