# analyses/rdf_analysis.py

import numpy as np
from utils import label_matches, prompt, prompt_int, prompt_float, prompt_yn
from analyses.metrics import Selector, DistanceMetric
from analyses.histogram import HistogramND

def rdf(traj):
    # Prompt user for RDF parameters
    print("\nAvailable Compounds:")
    for i, compound in enumerate(traj.compounds.values(), start=1):
        print(f"{i}: {compound.rep} (Number: {len(compound.members)})")

    ref_index = prompt_int("Choose the reference compound (number): ", 1, minval=1) - 1
    obs_index = prompt_int("Choose the observed compound (number): ", 1, minval=1) - 1
    ref_labels = [s.strip() for s in prompt("Which atom(s) in reference compound? (comma-separated) ").split(',')]
    obs_labels = [s.strip() for s in prompt("Which atom(s) in observed compound? (comma-separated) ").split(',')]
    max_distance = prompt_float("Enter the maximum distance for RDF calculation (in Å): ", 10.0, minval=0.1)
    bin_count = prompt_int("Enter the number of bins for RDF calculation: ", 500, minval=1)

    update_compounds = prompt_yn("Perform molecule recognition and update compound list in each frame?", False)

    start_frame = prompt_int("In which trajectory frame to start processing the trajectory?", 1, minval=1)
    nframes = prompt_int("How many trajectory frames to read (from this position on)?", -1, "all")
    frame_stride = prompt_int("Use every n-th read trajectory frame for the analysis:", 1, minval=1)

    # --- Precompute atom indices ---
    ref_compound = list(traj.compounds.values())[ref_index]
    obs_compound = list(traj.compounds.values())[obs_index]
    ref_key = next(k for k, c in traj.compounds.items() if c is ref_compound)
    obs_key = next(k for k, c in traj.compounds.items() if c is obs_compound)

    ref_indices = [
        idx for mol in ref_compound.members
        for label, idx in mol.label_to_global_id.items()
        if any(label_matches(lab, label) for lab in ref_labels)
    ]
    obs_indices = [
        idx for mol in obs_compound.members
        for label, idx in mol.label_to_global_id.items()
        if any(label_matches(lab, label) for lab in obs_labels)
    ]

    if not ref_indices or not obs_indices:
        print("No atoms matched the given labels.")
        return

    # --- Setup metric and histogram ---
    ref_sel = Selector(np.array(ref_indices))
    obs_sel = Selector(np.array(obs_indices))
    metric = DistanceMetric(ref_sel, obs_sel, traj.box_size, cutoff=max_distance)

    n_ref = len(ref_indices)
    n_obs = len(obs_indices)
    edges = np.linspace(0, max_distance, bin_count + 1)
    hist = HistogramND([edges])

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

                ref_indices = [
                    idx for mol in ref_compound.members
                    for label, idx in mol.label_to_global_id.items()
                    if any(label_matches(lab, label) for lab in ref_labels)
                ]
                obs_indices = [
                    idx for mol in obs_compound.members
                    for label, idx in mol.label_to_global_id.items()
                    if any(label_matches(lab, label) for lab in obs_labels)
                    ]

                if not ref_indices or not obs_indices:
                    frame_idx += 1
                    nframes -= 1
                    traj.read_frame()
                    continue

                ref_sel = Selector(np.array(ref_indices))
                obs_sel = Selector(np.array(obs_indices))
                metric = DistanceMetric(ref_sel, obs_sel, traj.box_size, cutoff=max_distance)

                n_ref = (n_ref * processed_frames + len(ref_indices))/(processed_frames + 1)
                n_obs = (n_obs * processed_frames + len(obs_indices))/(processed_frames + 1)

            else:
                traj.update_molecule_coords()

            values = metric(traj.coords)  # distance array
            hist.add(values)

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

    # --- Normalize and save ---
    box_volume = np.prod(traj.box_size)
#    hist.normalize(method="volume", box_volume=box_volume)
    # Shell volumes per bin
    bin_edges = hist.bin_edges[0]
    bin_widths = np.diff(bin_edges)
    shell_volumes = (4/3) * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)

    # Normalize RDF manually
    norm_factor = n_ref * n_obs * processed_frames
    hist.counts = hist.counts / (shell_volumes * norm_factor / box_volume)

    # Number Integral
    obs_density = n_obs / box_volume                     # ρ  (atoms Å⁻³)
    g_of_r = hist.counts                             # normalised RDF values
    number_integral = obs_density * np.cumsum(g_of_r * shell_volumes)

    # ---- write combined output -----------------------------------------
    r_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    label_str = lambda labels: "_".join(l.replace(" ", "") for l in labels)
    fname = f"rdf_{label_str(ref_labels)}_{label_str(obs_labels)}.dat"
    with open(fname, "w") as f:
        f.write("#  r/Å     g(r)        N(r)\n")
        for r, g, n in zip(r_centers, g_of_r, number_integral):
            f.write(f"{r:8.4f}  {g:10.6f}  {n:12.6f}\n")

    # still keep the binary counts array for post-processing
#    hist.save_npy(f"rdf_{ref_label}_{obs_label}.npy")

    print(f"RDF and number-integral saved to {fname} and .npy")


#    hist.save_all(f"rdf_{ref_label}_{obs_label}")
#    print(f"RDF results saved to rdf_{ref_label}_{obs_label}.dat / .npy")

