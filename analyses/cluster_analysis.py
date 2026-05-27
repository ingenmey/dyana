from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from analyses.common.base_analysis import BaseAnalysis
from core.atomic_properties import elem_color, elem_vdW
from core.geometry import unwrap_around_reference
from core.topology import CompoundType
from io_support.console import console
from io_support.output_writer import build_output_filename, resolve_output_path, write_table
from utils import label_matches


@dataclass(frozen=True)
class ClusterCompoundSpec:
    """One compound selection contributing atoms to cluster detection."""

    compound_index: int
    labels: list[str]

    def __post_init__(self):
        if self.compound_index < 0:
            raise ValueError("compound_index must be >= 0.")
        if not self.labels:
            raise ValueError("labels must not be empty.")


@dataclass(frozen=True)
class ClusterCutoffSpec:
    """One cross-compound cutoff between two selected user-label groups."""

    left_compound_index: int
    left_label: str
    right_compound_index: int
    right_label: str
    cutoff: float

    def __post_init__(self):
        if self.left_compound_index < 0 or self.right_compound_index < 0:
            raise ValueError("cutoff compound indices must be >= 0.")
        if self.left_compound_index == self.right_compound_index:
            raise ValueError("Cluster cutoffs must connect two different compounds.")
        if not self.left_label or not self.right_label:
            raise ValueError("cutoff labels must not be empty.")
        if self.cutoff < 0:
            raise ValueError("cutoff must be >= 0.")


@dataclass(frozen=True)
class ClusterConfig:
    """Configuration for supported cluster-composition analysis."""

    selected_compounds: list[ClusterCompoundSpec]
    cutoffs: list[ClusterCutoffSpec]
    hash_graphs: bool = True
    graph_format: str | None = None
    save_xyz: bool = False
    save_whole_molecules: bool = False
    compute_cacf: bool = True
    corr_depth: int = 100
    compute_errors: bool = False

    def __post_init__(self):
        if not self.selected_compounds:
            raise ValueError("selected_compounds must not be empty.")
        compound_indices = [spec.compound_index for spec in self.selected_compounds]
        if len(set(compound_indices)) != len(compound_indices):
            raise ValueError("selected_compounds must not repeat the same compound index.")
        if self.graph_format not in {None, "svg", "png"}:
            raise ValueError("graph_format must be None, 'svg', or 'png'.")
        if self.save_whole_molecules and not self.save_xyz:
            raise ValueError("save_whole_molecules requires save_xyz=True.")
        if self.corr_depth < 1:
            raise ValueError("corr_depth must be >= 1.")

        expected_pairs = {
            _normalized_cutoff_key(left.compound_index, left_label, right.compound_index, right_label)
            for left, right in combinations(self.selected_compounds, 2)
            for left_label in left.labels
            for right_label in right.labels
        }
        actual_pairs = {
            _normalized_cutoff_key(
                cutoff.left_compound_index,
                cutoff.left_label,
                cutoff.right_compound_index,
                cutoff.right_label,
            )
            for cutoff in self.cutoffs
        }
        missing = expected_pairs - actual_pairs
        extras = actual_pairs - expected_pairs
        if missing:
            raise ValueError(f"cutoffs is missing {len(missing)} required compound-label cutoff pairs.")
        if extras:
            raise ValueError("cutoffs contains compound-label cutoff pairs outside the selected compounds.")


@dataclass
class _ResolvedClusterCompound:
    compound_key: tuple
    formula: str
    display_index: int
    labels: list[str]
    local_indices_by_label: dict[str, tuple[int, ...]]
    n_local_atoms: int
    compound_type: CompoundType | None = None
    molecule_atom_ids: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.int32))


class ClusterAnalysis(BaseAnalysis):
    """Cluster-composition histogram analysis on the supported framework path."""

    CONFIG_CLASS = ClusterConfig

    def prompt_config(self, provider=None):
        input_provider = self.get_input_provider(provider)
        selected_compounds: list[ClusterCompoundSpec] = []

        console.info("Cluster composition histogram")
        console.plain("For each compound, choose which atom labels should be considered for clustering.")
        console.plain("Leave the answer blank to exclude that compound from clustering.")

        for compound_index, compound_type in enumerate(self.get_compound_types()):
            labels = self.atom_selection(
                compound=compound_type,
                prompt_text=(
                    f"Which atom(s) in compound {compound_index + 1} ({compound_type.formula}) "
                    "should be considered for clustering? (comma-separated) "
                ),
                allow_empty=True,
                provider=input_provider,
            )
            if labels:
                selected_compounds.append(
                    ClusterCompoundSpec(
                        compound_index=compound_index,
                        labels=labels,
                    )
                )

        if not selected_compounds:
            raise ValueError("No compounds selected for clustering (all label lists empty).")

        cutoffs: list[ClusterCutoffSpec] = []
        for left_display_index, left_spec in enumerate(selected_compounds, start=1):
            for right_display_index, right_spec in enumerate(selected_compounds[left_display_index:], start=left_display_index + 1):
                for left_label in left_spec.labels:
                    for right_label in right_spec.labels:
                        cutoff = input_provider.ask_float(
                            (
                                f"Cutoff for {left_label} in compound {left_display_index} "
                                f"and {right_label} in compound {right_display_index} (Angstrom): "
                            ),
                            default=0.0,
                            minval=0.0,
                        )
                        cutoffs.append(
                            ClusterCutoffSpec(
                                left_compound_index=left_spec.compound_index,
                                left_label=left_label,
                                right_compound_index=right_spec.compound_index,
                                right_label=right_label,
                                cutoff=cutoff,
                            )
                        )

        hash_graphs = input_provider.ask_bool(
            "Count clusters by composition and graph hash?",
            True,
        )
        graph_format = None
        if input_provider.ask_bool("Visualize cluster graphs?", False):
            graph_format = input_provider.ask_choice(
                "Save cluster graphs in which format?",
                ["svg", "png"],
                default="svg",
            )

        save_xyz = input_provider.ask_bool("Save cluster coordinates as XYZ files?", False)
        save_whole_molecules = False
        if save_xyz:
            save_whole_molecules = input_provider.ask_bool(
                "Save whole molecules instead of only the selected atom types?",
                False,
            )

        compute_cacf = input_provider.ask_bool(
            "Compute cluster autocorrelation functions?",
            True,
        )
        corr_depth = 100
        if compute_cacf:
            corr_depth = input_provider.ask_int(
                "Maximum correlation depth (number of frames): ",
                100,
                minval=1,
            )

        compute_errors = input_provider.ask_bool(
            "Compute per-frame standard deviations for cluster counts?",
            False,
        )

        return ClusterConfig(
            selected_compounds=selected_compounds,
            cutoffs=cutoffs,
            hash_graphs=hash_graphs,
            graph_format=graph_format,
            save_xyz=save_xyz,
            save_whole_molecules=save_whole_molecules,
            compute_cacf=compute_cacf,
            corr_depth=corr_depth,
            compute_errors=compute_errors,
        )

    def configure(self, config: ClusterConfig):
        self.bind_config(config)
        topology_frame = self.traj.topology_frame

        self.selected_compounds = []
        self.selected_keys = []
        self.display_index_by_key = {}
        self._selected_spec_by_index = {}
        for display_index, compound_spec in enumerate(config.selected_compounds, start=1):
            self._selected_spec_by_index[compound_spec.compound_index] = compound_spec
            (compound_type, compound_key), = self.resolve_compound_types([compound_spec.compound_index])

            local_indices_by_label = {}
            for label in compound_spec.labels:
                local_indices = topology_frame.resolve_selection(compound_type, [label]).local_indices
                if not local_indices:
                    raise ValueError(f"Cluster label {label!r} matched no atoms in the initial frame.")
                local_indices_by_label[label] = local_indices

            resolved = _ResolvedClusterCompound(
                compound_key=compound_key,
                formula=compound_type.formula,
                display_index=display_index,
                labels=list(compound_spec.labels),
                local_indices_by_label=local_indices_by_label,
                n_local_atoms=compound_type.n_local_atoms,
                compound_type=compound_type,
                molecule_atom_ids=np.empty((0, compound_type.n_local_atoms), dtype=np.int32),
            )
            self.selected_compounds.append(resolved)
            self.selected_keys.append(compound_key)
            self.display_index_by_key[compound_key] = display_index

        key_by_index = {
            spec.compound_index: resolved.compound_key
            for spec, resolved in zip(config.selected_compounds, self.selected_compounds)
        }
        self.cutoff_distances = {}
        for cutoff in config.cutoffs:
            left_key = key_by_index[cutoff.left_compound_index]
            right_key = key_by_index[cutoff.right_compound_index]
            left_group = (left_key, cutoff.left_label)
            right_group = (right_key, cutoff.right_label)
            self.cutoff_distances[(left_group, right_group)] = cutoff.cutoff
            self.cutoff_distances[(right_group, left_group)] = cutoff.cutoff

        self.graph_format = config.graph_format
        self.hash_graphs = config.hash_graphs
        self.save_xyz = config.save_xyz
        self.save_whole_molecules = config.save_whole_molecules
        self.compute_cacf = config.compute_cacf
        self.corr_depth = config.corr_depth
        self.compute_errors = config.compute_errors

        self.cluster_histogram = Counter()
        self.graph_list: list[tuple[str, str | int, nx.Graph]] = []
        self.seen_graphs: set[tuple[str, str | int]] = set()
        self.cluster_beta = defaultdict(lambda: defaultdict(list))
        self.frame_cluster_counts = {} if self.compute_errors else None
        self._prepared_xyz_paths: dict[str, Path] = {}

        self.rebuild_runtime_state()
        if not self.active_compounds:
            raise ValueError("No selected compounds are present in the initial frame.")

        self.mark_configured()

    def rebuild_runtime_state(self):
        topology_frame = self.traj.topology_frame
        self.active_compounds = []
        for compound in self.selected_compounds:
            if topology_frame.has_compound_type_key(compound.compound_key):
                compound.compound_type = topology_frame.get_compound_type_by_key(compound.compound_key)
                compound.molecule_atom_ids = topology_frame.get_molecule_atom_ids(compound.compound_type)
                self.active_compounds.append(compound)
            else:
                compound.compound_type = None
                compound.molecule_atom_ids = np.empty((0, compound.n_local_atoms), dtype=np.int32)

    def post_compound_update(self):
        self.rebuild_runtime_state()
        return bool(self.active_compounds)

    def process_frame(self):
        atom_groups = self._build_atom_groups()
        if not atom_groups:
            self._pad_absent_clusters(set())
            if self.compute_errors:
                self._update_frame_counts(Counter())
            return

        clusters = identify_clusters(
            atom_groups=atom_groups,
            coords=self.traj.coords,
            cutoff_distances=self.cutoff_distances,
            box_size=self.traj.box_size,
            display_index_by_key=self.display_index_by_key,
        )

        seen_this_frame = set()
        frame_counts = Counter()

        for composition, graph, cluster_atom_ids in clusters:
            graph_id = get_graph_id(graph) if self.hash_graphs else 0
            key = (composition, graph_id)
            self.cluster_histogram[key] += 1
            frame_counts[key] += 1

            atom_ids = frozenset(cluster_atom_ids)
            seen_this_frame.add((composition, graph_id, atom_ids))
            beta_series = self.cluster_beta[key][atom_ids]
            if not beta_series:
                beta_series.extend([0] * self.processed_frames)
            beta_series.append(1)

            if key not in self.seen_graphs:
                self.seen_graphs.add(key)
                self.graph_list.append((composition, graph_id, graph))

            if self.save_xyz and len(cluster_atom_ids) > 1:
                self._write_cluster_xyz(composition, graph_id, cluster_atom_ids)

        if self.compute_errors:
            self._update_frame_counts(frame_counts)

        self._pad_absent_clusters(seen_this_frame)

    def _build_atom_groups(self) -> dict[tuple[tuple, str], np.ndarray]:
        atom_groups: dict[tuple[tuple, str], np.ndarray] = {}
        for compound in self.active_compounds:
            if compound.molecule_atom_ids.size == 0:
                continue
            for label, local_indices in compound.local_indices_by_label.items():
                atom_ids = compound.molecule_atom_ids[:, list(local_indices)].reshape(-1)
                if atom_ids.size:
                    atom_groups[(compound.compound_key, label)] = atom_ids
        return atom_groups

    def _write_cluster_xyz(self, composition: str, graph_id: str | int, cluster_atom_ids: list[int]):
        filename = f"cluster_{composition}_{graph_id}.xyz"
        if filename not in self._prepared_xyz_paths:
            self._prepared_xyz_paths[filename] = resolve_output_path(Path("xyz") / filename, rotate=True)
        output_path = self._prepared_xyz_paths[filename]

        atom_ids, symbols, coords = self._cluster_xyz_payload(cluster_atom_ids)
        if not atom_ids:
            return

        centered = unwrap_around_reference(np.asarray(coords, dtype=float), self.traj.box_size)
        centered -= centered.mean(axis=0)

        with open(output_path, "a", encoding="utf-8") as fout:
            fout.write(f"{len(symbols)}\n")
            fout.write("Generated by cluster analysis\n")
            for symbol, coord in zip(symbols, centered):
                fout.write(f"{symbol} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")

    def _cluster_xyz_payload(self, cluster_atom_ids: list[int]) -> tuple[list[int], list[str], list[np.ndarray]]:
        topology_frame = self.traj.topology_frame
        coords = self.traj.coords

        if self.save_whole_molecules:
            molecule_keys = []
            seen_molecules = set()
            for atom_id in sorted(cluster_atom_ids):
                type_id = int(topology_frame.atom_to_type_id[atom_id])
                molecule_index = int(topology_frame.atom_to_molecule_index[atom_id])
                molecule_key = (type_id, molecule_index)
                if molecule_key not in seen_molecules:
                    seen_molecules.add(molecule_key)
                    molecule_keys.append(molecule_key)

            atom_ids_out: list[int] = []
            symbols: list[str] = []
            coords_out: list[np.ndarray] = []
            for type_id, molecule_index in molecule_keys:
                compound_type = topology_frame.get_compound_type_by_index(type_id)
                molecule_atom_ids = topology_frame.get_molecule_atom_ids(compound_type)[molecule_index]
                atom_ids_out.extend(int(atom_id) for atom_id in molecule_atom_ids)
                symbols.extend(compound_type.local_elements)
                coords_out.extend(coords[molecule_atom_ids])
            return atom_ids_out, symbols, coords_out

        atom_ids_out = sorted(int(atom_id) for atom_id in cluster_atom_ids)
        symbols = []
        coords_out = []
        for atom_id in atom_ids_out:
            type_id = int(topology_frame.atom_to_type_id[atom_id])
            local_index = int(topology_frame.atom_to_local_index[atom_id])
            compound_type = topology_frame.get_compound_type_by_index(type_id)
            symbols.append(compound_type.local_elements[local_index])
            coords_out.append(coords[atom_id])
        return atom_ids_out, symbols, coords_out

    def _pad_absent_clusters(self, seen_this_frame):
        for (composition, graph_id), instances in self.cluster_beta.items():
            for atom_ids, beta in instances.items():
                if (composition, graph_id, atom_ids) not in seen_this_frame:
                    beta.append(0)

    def _update_frame_counts(self, frame_counts: Counter):
        frame_index = self.processed_frames
        for key, series in self.frame_cluster_counts.items():
            series.append(frame_counts.get(key, 0))

        for key, count in frame_counts.items():
            if key not in self.frame_cluster_counts:
                self.frame_cluster_counts[key] = [0] * frame_index + [count]

    def postprocess(self):
        if not self.cluster_histogram:
            console.warn("No clusters were accumulated.")
            return

        sorted_clusters = sorted(
            self.cluster_histogram.items(),
            key=lambda item: (-item[1], item[0][0], str(item[0][1])),
        )

        console.info("Cluster composition histogram:")
        for (composition, graph_id), count in sorted_clusters:
            console.key_value(f"{composition} [{graph_id}]", f"{count} occurrences", indent=2)

        counts_per_frame = self._counts_per_frame()
        self._write_cluster_occurrences(sorted_clusters, counts_per_frame)
        self._write_cluster_populations(sorted_clusters, counts_per_frame)
        self._write_cluster_graphs(sorted_clusters)
        self._write_cluster_cacf()

    def _counts_per_frame(self) -> dict[tuple[str, str | int], np.ndarray]:
        if self.frame_cluster_counts is None:
            return {
                key: np.array([float(total_occ)], dtype=float)
                for key, total_occ in self.cluster_histogram.items()
            }

        counts_per_frame = {
            key: np.asarray(series, dtype=float)
            for key, series in self.frame_cluster_counts.items()
        }
        total_frames = self.processed_frames
        for key in self.cluster_histogram:
            if key not in counts_per_frame:
                counts_per_frame[key] = np.zeros(total_frames, dtype=float)
        for key, series in counts_per_frame.items():
            if len(series) < total_frames:
                counts_per_frame[key] = np.pad(series, (0, total_frames - len(series)), constant_values=0.0)
        return counts_per_frame

    def _write_cluster_occurrences(self, sorted_clusters, counts_per_frame):
        rows = []
        for key, total_occ in sorted_clusters:
            series = counts_per_frame[key]
            if self.frame_cluster_counts is None:
                mean = total_occ / self.processed_frames if self.processed_frames else 0.0
                std = 0.0
            else:
                mean = float(series.mean())
                std = float(series.std(ddof=1)) if len(series) > 1 else 0.0
            composition, graph_id = key
            rows.append([composition, total_occ, graph_id, mean, std])

        filename = build_output_filename("cluster_occurrences")
        write_table(
            filename,
            headers=["Cluster", "Occurrences", "GraphID", "mean_per_frame", "std_per_frame"],
            data=rows,
        )
        console.success(f"Saved cluster occurrences to {filename}")

    def _write_cluster_populations(self, sorted_clusters, counts_per_frame):
        atom_counts_by_cluster = {
            key: _parse_composition_counts(key[0])
            for key in self.cluster_histogram
        }
        atom_labels = sorted({label for counts in atom_counts_by_cluster.values() for label in counts})
        if not atom_labels:
            return

        cluster_population_mean = {label: {} for label in atom_labels}
        cluster_population_std = {label: {} for label in atom_labels}
        size_population_mean = {label: {} for label in atom_labels}
        size_population_std = {label: {} for label in atom_labels}

        for label in atom_labels:
            total_weight = None
            for key, series in counts_per_frame.items():
                label_count = atom_counts_by_cluster[key].get(label, 0)
                if label_count == 0:
                    continue
                weighted = series * label_count
                total_weight = weighted.copy() if total_weight is None else total_weight + weighted

            if total_weight is None:
                continue

            valid_mask = total_weight > 0
            if not np.any(valid_mask):
                continue

            size_series = defaultdict(lambda: np.zeros_like(total_weight))
            for key, series in counts_per_frame.items():
                label_count = atom_counts_by_cluster[key].get(label, 0)
                if label_count == 0:
                    cluster_population_mean[label][key] = 0.0
                    cluster_population_std[label][key] = 0.0
                    continue

                weighted = series * label_count
                population = np.zeros_like(total_weight)
                population[valid_mask] = weighted[valid_mask] / total_weight[valid_mask]
                values = population[valid_mask]
                cluster_population_mean[label][key] = float(values.mean())
                cluster_population_std[label][key] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                size_series[label_count][valid_mask] += population[valid_mask]

            for size, series in size_series.items():
                values = series[valid_mask]
                size_population_mean[label][size] = float(values.mean())
                size_population_std[label][size] = float(values.std(ddof=1)) if len(values) > 1 else 0.0

        rows = []
        for (composition, graph_id), _ in sorted_clusters:
            row = [composition, graph_id]
            key = (composition, graph_id)
            for label in atom_labels:
                row.extend(
                    [
                        cluster_population_mean[label].get(key, 0.0),
                        cluster_population_std[label].get(key, 0.0),
                    ]
                )
            rows.append(row)

        headers = ["Cluster", "GraphID"]
        for label in atom_labels:
            headers.extend([f"I({label})", f"std(I({label}))"])

        filename = build_output_filename("cluster_populations")
        write_table(filename, headers=headers, data=rows)
        console.success(f"Saved cluster populations to {filename}")

        max_size = max(
            (max(size_population_mean[label], default=0) for label in atom_labels),
            default=0,
        )
        if max_size == 0:
            return

        size_rows = []
        for size in range(1, max_size + 1):
            row = [size]
            for label in atom_labels:
                row.extend(
                    [
                        size_population_mean[label].get(size, 0.0),
                        size_population_std[label].get(size, 0.0),
                    ]
                )
            size_rows.append(row)

        size_headers = ["ClusterSize"]
        for label in atom_labels:
            size_headers.extend([f"I({label})", f"std(I({label}))"])

        filename = build_output_filename("cluster_size")
        write_table(filename, headers=size_headers, data=size_rows)
        console.success(f"Saved cluster-size populations to {filename}")

    def _write_cluster_graphs(self, sorted_clusters):
        if self.graph_format is None:
            return

        graph_by_key = {
            (composition, graph_id): graph
            for composition, graph_id, graph in self.graph_list
        }
        for index, ((composition, graph_id), _count) in enumerate(sorted_clusters[:200]):
            graph = graph_by_key.get((composition, graph_id))
            if graph is None:
                continue
            filename = f"graph{index}_{composition}_{graph_id}.{self.graph_format}"
            draw_graph(graph, resolve_output_path(filename, rotate=True))
        console.success("Saved cluster graph visualizations.")

    def _write_cluster_cacf(self):
        if not self.compute_cacf:
            return
        if self.processed_frames <= 0:
            console.warn("No frames processed; skipping cluster autocorrelation output.")
            return

        max_tau = min(self.corr_depth, self.processed_frames)
        for (composition, graph_id), instances in self.cluster_beta.items():
            if not instances:
                continue
            cacf = np.zeros(max_tau, dtype=np.float64)
            n_instances = len(instances)

            for beta in instances.values():
                values = np.asarray(beta, dtype=np.float64)
                if len(values) < self.processed_frames:
                    values = np.pad(values, (0, self.processed_frames - len(values)), constant_values=0.0)
                for tau in range(max_tau):
                    cacf[tau] += np.sum(values[: self.processed_frames - tau] * values[tau:self.processed_frames])

            normalization = n_instances * np.arange(self.processed_frames, self.processed_frames - max_tau, -1)
            cacf /= normalization
            if cacf[0] != 0:
                cacf /= cacf[0]

            filename = build_output_filename("cacf", [composition, str(graph_id)])
            write_table(
                filename,
                headers=["tau", "CACF"],
                data=[[tau, value] for tau, value in enumerate(cacf)],
            )
            console.success(f"Saved CACF for cluster type {composition} [{graph_id}] to {filename}")


def identify_clusters(atom_groups, coords, cutoff_distances, box_size, display_index_by_key):
    """Build cluster graphs by growing connected components over cutoff-linked atom groups."""
    kdtrees = {
        group_key: cKDTree(coords[atom_ids], boxsize=box_size)
        for group_key, atom_ids in atom_groups.items()
        if len(atom_ids) > 0
    }
    valid_neighbors = {
        group_key: [
            other_group
            for other_group in atom_groups
            if (group_key, other_group) in cutoff_distances
        ]
        for group_key in atom_groups
    }

    visited = set()
    clusters = []

    def grow_cluster(atom_id, group_key, cluster_atom_ids, graph, atom_counts):
        if atom_id in visited:
            return
        visited.add(atom_id)

        compound_key, label = group_key
        graph.add_node(int(atom_id), label=label)
        cluster_atom_ids.append(int(atom_id))

        count_key = f"{display_index_by_key[compound_key]}-{label}"
        atom_counts[count_key] = atom_counts.get(count_key, 0) + 1

        for other_group in valid_neighbors[group_key]:
            cutoff = cutoff_distances[(group_key, other_group)]
            other_atom_ids = atom_groups[other_group]
            tree = kdtrees.get(other_group)
            if tree is None:
                continue
            for neighbor_index in tree.query_ball_point(coords[atom_id], cutoff):
                neighbor_atom_id = int(other_atom_ids[neighbor_index])
                if neighbor_atom_id not in visited:
                    grow_cluster(neighbor_atom_id, other_group, cluster_atom_ids, graph, atom_counts)
                graph.add_edge(int(atom_id), neighbor_atom_id)

    for group_key, atom_ids in atom_groups.items():
        for atom_id in atom_ids:
            atom_id = int(atom_id)
            if atom_id in visited:
                continue
            graph = nx.Graph()
            cluster_atom_ids: list[int] = []
            atom_counts = {}
            grow_cluster(atom_id, group_key, cluster_atom_ids, graph, atom_counts)
            composition = "_".join(f"{key}-{count}" for key, count in sorted(atom_counts.items()))
            clusters.append((composition, graph, cluster_atom_ids))

    return clusters


def get_graph_id(graph: nx.Graph) -> str:
    """Return a stable WL hash for one cluster graph."""
    return nx.weisfeiler_lehman_graph_hash(graph, node_attr="label")


def draw_graph(graph: nx.Graph, output_path: Path):
    """Render one cluster graph to an image file."""
    import matplotlib.pyplot as plt

    def element_from_label(label: str) -> str:
        return "".join(ch for ch in label if not ch.isdigit()) or label

    plt.figure(figsize=(6, 6))
    node_labels = {node: graph.nodes[node]["label"] for node in graph.nodes}
    node_elements = [element_from_label(graph.nodes[node]["label"]) for node in graph.nodes]
    node_sizes = [elem_vdW.get(element, 1.0) * 1000 for element in node_elements]
    node_colors = [elem_color.get(element, "gray") for element in node_elements]

    positions = nx.spring_layout(graph, seed=42)
    nx.draw(
        graph,
        positions,
        with_labels=True,
        labels=node_labels,
        node_size=node_sizes,
        node_color=node_colors,
        edge_color="black",
        font_weight="bold",
        font_size=10,
        width=2.0,
    )
    plt.savefig(output_path, dpi=300)
    plt.close()


def _normalized_cutoff_key(left_compound_index, left_label, right_compound_index, right_label):
    if (left_compound_index, left_label) <= (right_compound_index, right_label):
        return (left_compound_index, left_label, right_compound_index, right_label)
    return (right_compound_index, right_label, left_compound_index, left_label)


def _parse_composition_counts(composition: str) -> dict[str, int]:
    counts = {}
    for entry in composition.split("_"):
        parts = entry.split("-")
        if len(parts) < 3:
            continue
        label = parts[1]
        count = int(parts[2])
        counts[label] = counts.get(label, 0) + count
    return counts
