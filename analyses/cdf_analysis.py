# analyses/cdf_analysis.py

import numpy as np
from utils import prompt, prompt_int, prompt_float, prompt_choice, prompt_yn, label_matches
from analyses.metrics import Selector, DistanceMetric, AngleMetric
from analyses.histogram import HistogramND


def find_matching_labels(mol, user_label):
    return [idx for label, idx in mol.label_to_global_id.items() if label_matches(user_label, label)]

def select_atoms(traj, comp_idx):
    compound = list(traj.compounds.values())[comp_idx]
    labels = prompt("Enter atom labels (comma-separated): ").strip()
    labels = [l.strip() for l in labels.split(',') if l.strip()]

    indices = []
    for mol in compound.members:
        for label, idx in mol.label_to_global_id.items():
            if any(label_matches(lab, label) for lab in labels):
                indices.append(idx)

    return Selector(np.array(indices, dtype=int))


def cdf(traj):
    print("\n--- Combined Distribution Function (CDF) Analysis ---")

    # Select metric types
    print("Select two metrics to combine:")
    metric1_type = prompt_choice("First metric", ["rdf", "angle"], "rdf")
    metric2_type = prompt_choice("Second metric", ["rdf", "angle"], "angle")

    ref_index = prompt_int("Choose the reference compound (number): ", 1, minval=1) - 1
    obs_index = prompt_int("Choose the observed compound (number): ", 1, minval=1) - 1

    # Bin settings

    # Selectors and metric constructors
    box = traj.box_size

    def make_metric(metric_type, suffix, ref_index, obs_index):
        if metric_type == "rdf":
            print(f"\n-- RDF {suffix} setup --")
            sel_a  = select_atoms(traj, ref_index)
            sel_b  = select_atoms(traj, obs_index)
            rmin   = prompt_float(f"Minimum value for {metric1_type}", 0.0)
            rmax   = prompt_float(f"Maximum value for {metric1_type}", 10.0)
            n_bins = prompt_int(f"Number of bins for {metric1_type}", 200, minval=1)
            bins = np.linspace(rmin, rmax, n_bins + 1)
            return DistanceMetric(sel_a, sel_b, box, rmax), bins

        elif metric_type == "angle":
            print(f"\n-- Angle {suffix} setup --")
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


            amin = prompt_float(f"Minimum value for {metric2_type}", 0.0)
            amax = prompt_float(f"Maximum value for {metric2_type}", 180.0)
            n_bins = prompt_int(f"Number of bins for {metric2_type}", 180, minval=1)
            bins = np.linspace(amin, amax, n_bins + 1)

            v1_cutoff = prompt_float("Enter maximum length for the first vector: ", None, "None", minval=0.0)
            v2_cutoff = prompt_float("Enter maximum length for the second vector: ", None, "None", minval=0.0)

            ref_comp = list(traj.compounds.values())[ref_index]
            obs_comp = list(traj.compounds.values())[obs_index]

            # Build global atom indices
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

            return AngleMetric(
                selector_ref_base=Selector(np.array(ref_base_ids)),
                selector_ref_tip=Selector(np.array(ref_tip_ids)),
                selector_obs_base=Selector(np.array(obs_base_ids)),
                selector_obs_tip=Selector(np.array(obs_tip_ids)),
                box=box,
                enforce_shared_atom=enforce_shared_atom,
                v1_cutoff=v1_cutoff,
                v2_cutoff=v2_cutoff
                ), bins

        else:
            raise ValueError(f"Unsupported metric type: {metric_type}")

    metric1, bins1 = make_metric(metric1_type, "Metric 1", ref_index, obs_index)
    metric2, bins2 = make_metric(metric2_type, "Metric 2", ref_index, obs_index)

    histogram = HistogramND([bins1, bins2])

    # Frame loop setup
    start_frame = prompt_int("Start frame index", 1, minval=1)
    nframes = prompt_int("Number of frames to process", -1, "all")
    stride = prompt_int("Frame stride", 1, minval=1)

    frame_idx = 0
    processed = 0

    if start_frame > 1:
        print(f"Skipping to frame {start_frame}...")
        while frame_idx < start_frame - 1:
            traj.read_frame()
            frame_idx += 1

    print("\nStarting CDF analysis...")
    while nframes != 0:
        try:
            traj.update_molecule_coords()
            coords = traj.coords

            vals1 = metric1(coords)
            vals2 = metric2(coords)

            if len(vals1) != len(vals2):
                print("Warning")
                input()
                min_len = min(len(vals1), len(vals2))
                vals1 = vals1[:min_len]
                vals2 = vals2[:min_len]

            histogram.add(vals1, vals2)

            processed += 1
            print(f"\rProcessed frame {frame_idx + 1} ({processed} total)", end="")

            for _ in range(stride):
                frame_idx += 1
                nframes -= 1
                traj.read_frame()

        except ValueError:
            print("\nEnd of trajectory reached.")
            break
        except KeyboardInterrupt:
            print("\nAnalysis interrupted by user.")
            break

    print("\n\nCDF collection complete.")

    # Normalization and saving
#    method = prompt_choice("Normalization method", ["total", "volume", "none"], "total")
#    if method == "volume":
#        histogram.normalize("volume", box_volume=np.prod(box))
#    elif method == "total":
#        histogram.normalize("total")

    bin_edges_1 = histogram.bin_edges[0]
    bin_edges_2 = histogram.bin_edges[1]

    shell_volumes = (4/3) * np.pi * (bin_edges_1[1:]**3 - bin_edges_1[:-1]**3)

    centers = 0.5 * (bin_edges_2[1:] + bin_edges_2[:-1])
    sin_weights = 1.0 / np.sin(np.deg2rad(centers))

    norm_grid = np.outer(shell_volumes, 1/sin_weights)
    histogram.counts = histogram.counts / (norm_grid)

    base = "cdf"
    histogram.save_all(base)
    print(f"Histogram saved to {base}.npy and {base}.dat")

