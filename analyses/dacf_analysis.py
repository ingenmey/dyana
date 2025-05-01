import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree
from utils import prompt, prompt_int, prompt_float, prompt_yn, label_matches

def dacf(traj):
    print("\nAvailable Compounds:")
    for i, compound in enumerate(traj.compounds.values(), start=1):
        print(f"{i}: {compound.rep} (Number: {len(compound.members)})")

    ref_index = prompt_int("Choose the reference compound (number): ", 1, minval=1) - 1
    obs_index = prompt_int("Choose the observed compound (number): ", 1, minval=1) - 1

    ref_label = prompt("Label of reference atom: ")
    obs_label = prompt("Label of observed atom: ")

    cutoff = prompt_float("Cutoff distance for dimer (Å): ", 3.5)
    use_continuous = prompt_yn("Use continuous autocorrelation function?", False)
    corr_depth = prompt_int("Maximum correlation depth (number of frames): ", 100, minval=1)
    apply_correction = prompt_yn("Perform finite-size ensemble equilibrium correction?", True)
    frame_time = prompt_float("Time per frame (fs):", 1.0)

    start_frame = prompt_int("Start from which trajectory frame?", 1, minval=1)
    nframes = prompt_int("How many trajectory frames to read?", -1, "all")
    frame_stride = prompt_int("Use every n-th trajectory frame:", 1, minval=1)

    ref_compound = list(traj.compounds.values())[ref_index]
    obs_compound = list(traj.compounds.values())[obs_index]
    box = np.asarray(traj.box_size)

    ref_atoms = [atom for mol in ref_compound.members
                 for label, idx in mol.label_to_id.items()
                 if label_matches(ref_label, label)
                 for atom in [mol.atoms[idx]]]

    obs_atoms = [atom for mol in obs_compound.members
                 for label, idx in mol.label_to_id.items()
                 if label_matches(obs_label, label)
                 for atom in [mol.atoms[idx]]]

    ref_ids = np.array([a.idx for a in ref_atoms])
    obs_ids = np.array([a.idx for a in obs_atoms])
    beta_tracker = defaultdict(list)
    all_pairs = set()

    frame_idx = 0
    processed_frames = 0

    if start_frame > 1:
        print(f"Skipping forward to frame {start_frame}...")
        while frame_idx < start_frame - 1:
            traj.read_frame()
            frame_idx += 1

    while nframes != 0:
        try:
            traj.update_molecule_coords()
            coords = traj.coords

            ref_coords = coords[ref_ids]
            obs_coords = coords[obs_ids]

            tree = cKDTree(obs_coords, boxsize=box)
            results = tree.query_ball_point(ref_coords, cutoff)

            active_pairs = set()

            for i, obs_list in enumerate(results):
                ref_idx = ref_ids[i]
                for j in obs_list:
                    obs_idx = obs_ids[j]
                    if ref_idx == obs_idx:
                        continue  # skip self
                    pair = (ref_idx, obs_idx)
                    active_pairs.add(pair)
                    beta_tracker[pair].append(1)
                    all_pairs.add(pair)

            for pair in all_pairs:
                if pair not in active_pairs:
                    beta_tracker[pair].append(0)

            processed_frames += 1
            print(f"\rProcessed {processed_frames} frames (current frame {frame_idx+1})", end="")

            for _ in range(frame_stride):
                frame_idx += 1
                nframes -= 1
                traj.read_frame()
        except ValueError:
            break
        except KeyboardInterrupt:
            print("\nInterrupted! Finishing early.")
            break

    print("\n")

    # --- Prepare ---
    T = processed_frames
    max_tau = min(corr_depth, T)
    dacf = np.zeros(max_tau)
    N = len(ref_ids)
    M = len(obs_ids)

    # Pad and convert beta arrays once
    for pair in beta_tracker:
        b = beta_tracker[pair]
        if len(b) < T:
            b += [0] * (T - len(b))
        beta_tracker[pair] = np.array(b, dtype=np.uint8)

    # --- Continuous autocorrelation ---
    if use_continuous:
        segments = []
        for b in beta_tracker.values():
            mask = b == 1
            changes = np.diff(mask.astype(int))
            starts = np.where(changes == 1)[0] + 1
            ends = np.where(changes == -1)[0] + 1

            if mask[0]:
                starts = np.insert(starts, 0, 0)
            if mask[-1]:
                ends = np.append(ends, T)

            segments.extend(zip(starts, ends))

        for start, end in segments:
            for tau in range(min(max_tau, end - start)):
                dacf[tau] += end - start - tau


        dacf /= (N * M * T)
        dacf /= dacf[0]



    # --- Intermittent autocorrelation ---
    else:
        for b in beta_tracker.values():
            for tau in range(max_tau):
                dacf[tau] += np.sum(b[:T - tau] * b[tau:])

        normalization = N * M * np.arange(T, T - max_tau, -1)
        dacf /= normalization

        if apply_correction:
            total_ones = sum(np.sum(b) for b in beta_tracker.values())
            avg_beta = total_ones / (N * M * T)
            dacf -= avg_beta ** 2

        dacf /= dacf[0]

    # --- Save Results ---
    with open("dacf.dat", "w") as f:
        f.write("tau/ps DACF\n")
        for tau, val in enumerate(dacf):
            f.write(f"{tau * frame_time/1000:.6f} {val:.12f}\n")

    print("Dimer autocorrelation results saved to dacf.dat")

