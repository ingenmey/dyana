# analyses/adf_analysis.py

import numpy as np
from utils import prompt, prompt_int, prompt_float, prompt_yn, prompt_choice, label_matches
from analyses.metrics import Selector, AngleMetric
from analyses.histogram import HistogramND

def adf(traj):
    print("\nAvailable Compounds:")
    for i, compound in enumerate(traj.compounds.values(), start=1):
        print(f"{i}: {compound.rep} (Number: {len(compound.members)})")

    ref_index = prompt_int("Choose the reference compound (number): ", 1, minval=1) - 1
    obs_index = prompt_int("Choose the observed compound (number): ", 1, minval=1) - 1

    ref_base_source = prompt_choice("Base atom of first vector?", ["r", "o"], "r")
    ref_tip_source = prompt_choice("Tip atom of first vector?", ["r", "o"], "r")
    ref_base_label = prompt("Which atom is at the base of the first vector? ")
    ref_tip_label = prompt("Which atom is at the tip of the first vector? ")

    obs_base_source = prompt_choice("Base atom of second vector?", ["r", "o"], "o")
    obs_tip_source = prompt_choice("Tip atom of second vector?", ["r", "o"], "o")
    obs_base_label = prompt("Which atom is at the base of the second vector? ")
    obs_tip_label = prompt("Which atom is at the tip of the second vector? ")

    enforce_shared_atom = False
    if ref_tip_label == obs_base_label:
        enforce_shared_atom = prompt_yn("Should the tip atom of the reference vector and the base atom of the observed vector be the same atom?", True)

    bin_count = prompt_int("Enter the number of bins for ADF calculation: ", 180, minval=1)
    v1_cutoff = prompt_float("Enter maximum length for the first vector: ", None, "None", minval=0.0)
    v2_cutoff = prompt_float("Enter maximum length for the second vector: ", None, "None", minval=0.0)

    update_compounds = prompt_yn("Perform molecule recognition and update compound list in each frame?", False)

    start_frame = prompt_int("In which trajectory frame to start processing the trajectory?", 1, minval=1)
    nframes = prompt_int("How many trajectory frames to read (from this position on)?", -1, "all")
    frame_stride = prompt_int("Use every n-th read trajectory frame for the analysis:", 1, minval=1)

    ref_comp = list(traj.compounds.values())[ref_index]
    obs_comp = list(traj.compounds.values())[obs_index]
    ref_key = next(k for k, c in traj.compounds.items() if c is ref_comp)
    obs_key = next(k for k, c in traj.compounds.items() if c is obs_comp)
    box = traj.box_size


    # Build global atom indices
    ref_base_ids, ref_tip_ids, obs_base_ids, obs_tip_ids = build_vector_lists(
        ref_comp, obs_comp,
        ref_base_source, ref_tip_source,
        obs_base_source, obs_tip_source,
        ref_base_label, ref_tip_label,
        obs_base_label, obs_tip_label,
        enforce_shared_atom
    )


    # Create metric and histogram
    metric = AngleMetric(
        selector_ref_base=Selector(np.array(ref_base_ids)),
        selector_ref_tip=Selector(np.array(ref_tip_ids)),
        selector_obs_base=Selector(np.array(obs_base_ids)),
        selector_obs_tip=Selector(np.array(obs_tip_ids)),
        box=box,
        enforce_shared_atom=enforce_shared_atom,
        v1_cutoff=v1_cutoff,
        v2_cutoff=v2_cutoff
    )

    angle_edges = np.linspace(0, 180, bin_count + 1)
    hist = HistogramND([angle_edges])

    # Frame loop
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
                    ref_comp = traj.compounds[ref_key]
                    obs_comp = traj.compounds[obs_key]
                except KeyError:
                    # compound disappeared this frame – skip
                    frame_idx += 1
                    nframes -= 1
                    traj.read_frame()
                    continue

                ref_base_ids, ref_tip_ids, obs_base_ids, obs_tip_ids = build_vector_lists(
                    ref_comp, obs_comp,
                    ref_base_source, ref_tip_source,
                    obs_base_source, obs_tip_source,
                    ref_base_label, ref_tip_label,
                    obs_base_label, obs_tip_label,
                    enforce_shared_atom
                )


                if not all([ref_base_ids, ref_tip_ids, obs_base_ids, obs_tip_ids]):
                # no valid triplets this frame
                    frame_idx += 1
                    nframes -= 1
                    traj.read_frame()
                    continue

                metric = AngleMetric(
                    selector_ref_base=Selector(np.array(ref_base_ids)),
                    selector_ref_tip=Selector(np.array(ref_tip_ids)),
                    selector_obs_base=Selector(np.array(obs_base_ids)),
                    selector_obs_tip=Selector(np.array(obs_tip_ids)),
                    box=box,
                    enforce_shared_atom=enforce_shared_atom,
                    v1_cutoff=v1_cutoff,
                    v2_cutoff=v2_cutoff
                )

            else:
                traj.update_molecule_coords()

            angles = metric(traj.coords)
            hist.add(angles)

            processed_frames += 1
            if processed_frames % 100 == 0:
                print(f"Processed {processed_frames} frames (current frame {frame_idx+1})")
#            print(f"\rProcessed {processed_frames} frames (current frame {frame_idx+1})", end="")

            for _ in range(frame_stride):
                frame_idx += 1
                nframes -= 1
                traj.read_frame()

        except (ValueError, KeyboardInterrupt):
            print("\nTerminating early.")
            break

    print()

    # Normalize ADF: sin(θ) correction + per-frame + per-pair
    bin_centers = 0.5 * (angle_edges[1:] + angle_edges[:-1])
    radians = np.deg2rad(bin_centers)
    sin_weights = 1.0 / np.sin(radians)

    hist.counts = hist.counts.astype(np.float64)
    hist.counts *= sin_weights
    if processed_frames > 0:
        hist.counts /= (processed_frames * len(ref_comp.members) * len(obs_comp.members))
    hist.normalize("total", total=bin_count*100)

    hist.save_all("adf")
    print("ADF results saved to adf.dat / adf.npy")


def find_matching_labels(mol, user_label):
    return [idx for label, idx in mol.label_to_global_id.items() if label_matches(user_label, label)]


def build_vector_lists(ref_comp, obs_comp, ref_base_source, ref_tip_source,
        obs_base_source, obs_tip_source, ref_base_label, ref_tip_label,
        obs_base_label, obs_tip_label, enforce_shared_atom):

    ref_base_ids, ref_tip_ids = [], []
    obs_base_ids, obs_tip_ids = [], []

    for ref_mol in ref_comp.members:
        for obs_mol in obs_comp.members:
            if ref_mol == obs_mol:
                continue

            rb_mol = ref_mol if ref_base_source == "r" else obs_mol
            rt_mol = ref_mol if ref_tip_source == "r" else obs_mol
            ob_mol = obs_mol if obs_base_source == "o" else ref_mol
            ot_mol = obs_mol if obs_tip_source == "o" else ref_mol

            rb_ids = find_matching_labels(rb_mol, ref_base_label)
            rt_ids = find_matching_labels(rt_mol, ref_tip_label)
            ob_ids = find_matching_labels(ob_mol, obs_base_label)
            ot_ids = find_matching_labels(ot_mol, obs_tip_label)

            for rb in rb_ids:
                for rt in rt_ids:
                    for ob in ob_ids:
                        if enforce_shared_atom and rt != ob:
                            continue
                        for ot in ot_ids:
                            ref_base_ids.append(rb)
                            ref_tip_ids.append(rt)
                            obs_base_ids.append(ob)
                            obs_tip_ids.append(ot)

    return ref_base_ids, ref_tip_ids, obs_base_ids, obs_tip_ids


