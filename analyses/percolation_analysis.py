from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from analyses.common.base_analysis import BaseAnalysis
from core.geometry import minimum_image
from framework.analysis_params import AtomLabelsParam, BoolParam, CompoundParam, FloatParam, ForEach, Group, IntParam
from io_support.console import console
from io_support.output_writer import build_output_filename, write_table


@dataclass(frozen=True)
class PercolationCompoundSelection:
    compound_index: int
    acceptor_labels: list[str]
    proton_labels: list[str]

    def __post_init__(self):
        if self.compound_index < 0:
            raise ValueError("compound_index must be >= 0.")
        acceptor_labels = [str(label) for label in self.acceptor_labels]
        proton_labels = [str(label) for label in self.proton_labels]
        if not acceptor_labels and not proton_labels:
            raise ValueError("Each included compound must define acceptors, donor protons, or both.")
        object.__setattr__(self, "acceptor_labels", acceptor_labels)
        object.__setattr__(self, "proton_labels", proton_labels)


@dataclass(frozen=True)
class PercolationConfig:
    compounds: list[PercolationCompoundSelection]
    cutoff: float = 2.5
    max_depth: int = 5
    use_alternating: bool = False
    fit_min_depth: int = 2
    fit_max_depth: int = -1

    def __post_init__(self):
        object.__setattr__(self, "compounds", _normalize_compounds(self.compounds))
        if self.cutoff <= 0:
            raise ValueError("cutoff must be > 0.")
        if self.max_depth < 1:
            raise ValueError("max_depth must be >= 1.")
        if self.fit_min_depth < 1:
            raise ValueError("fit_min_depth must be >= 1.")
        if self.fit_max_depth != -1 and self.fit_max_depth < self.fit_min_depth:
            raise ValueError("fit_max_depth must be -1 or >= fit_min_depth.")


def _normalize_compounds(compounds: list[PercolationCompoundSelection]) -> list[PercolationCompoundSelection]:
    if not compounds:
        raise ValueError("compounds must not be empty.")
    indices = [compound.compound_index for compound in compounds]
    if len(indices) != len(set(indices)):
        raise ValueError("compounds must not repeat the same compound index.")
    return list(compounds)


def _build_compounds(compound_indices, compounds):
    if not compound_indices:
        raise ValueError("At least one compound must be selected.")
    compounds = _normalize_compounds(compounds)
    if {int(index) for index in compound_indices} != {compound.compound_index for compound in compounds}:
        raise ValueError("Compound selections do not match the chosen compound indices.")
    return compounds


def _fit_power_law(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < 2:
        return np.nan, np.nan
    return tuple(np.polyfit(np.log(x[valid]), np.log(y[valid]), 1))


def _count_components(neighbours):
    n_nodes = len(neighbours)
    visited = np.zeros(n_nodes, dtype=bool)
    count = 0
    for seed in range(n_nodes):
        if visited[seed]:
            continue
        count += 1
        stack = [seed]
        visited[seed] = True
        while stack:
            node = stack.pop()
            for neighbour in neighbours[node]:
                if not visited[neighbour]:
                    visited[neighbour] = True
                    stack.append(neighbour)
    return count


class PercolationAnalysis(BaseAnalysis):
    """Hydrogen-bond percolation pathway analysis."""

    CONFIG_CLASS = PercolationConfig
    CONFIG_SCHEMA = [
        Group(
            name="compounds",
            config_class=_build_compounds,
            steps=[
                CompoundParam(
                    name="compound_indices",
                    role="included",
                    multi=True,
                    prompt="Choose the compounds to include in the percolation analysis (comma-separated numbers): ",
                ),
                ForEach(
                    source="compound_indices",
                    item_name="compound_index",
                    steps=[
                        AtomLabelsParam(
                            name="acceptor_labels",
                            role="acceptor",
                            compound="compound_index",
                            prompt="Which atom(s) in compound {compound_num} ({compound_name}) are acceptors? (comma-separated, blank for none) ",
                            allow_empty=True,
                        ),
                        AtomLabelsParam(
                            name="proton_labels",
                            role="proton",
                            compound="compound_index",
                            prompt="Which hydrogen atom(s) in compound {compound_num} ({compound_name}) can be donated? (comma-separated, blank for none) ",
                            allow_empty=True,
                        ),
                    ],
                    collect_as="compounds",
                    collect_mode="list",
                    config_class=PercolationCompoundSelection,
                    include_item_as="compound_index",
                ),
            ],
        ),
        FloatParam(
            name="cutoff",
            prompt="Enter the hydrogen-bond cutoff distance (Angstrom): ",
            default=2.5,
            minval=1e-12,
        ),
        IntParam(
            name="max_depth",
            prompt="Enter the maximum number of hydrogen-bond steps to consider: ",
            default=5,
            minval=1,
        ),
        BoolParam(
            name="use_alternating",
            prompt="Limit pathways to alternating donor-to-acceptor chains?",
            default=False,
        ),
        IntParam(
            name="fit_min_depth",
            prompt="Enter the minimum fit depth: ",
            default=2,
            minval=1,
        ),
        IntParam(
            name="fit_max_depth",
            prompt="Enter the maximum fit depth (-1 for no limit): ",
            default=-1,
            display_default="all",
            minval=-1,
        ),
    ]

    def configure(self, config: PercolationConfig):
        self.bind_config(config)
        self.compound_specs = self._resolve_compound_specs(self.compounds)
        self.rebuild_runtime_state()

        if self.n_nodes == 0:
            raise ValueError("No molecules matched the selected compounds in the initial frame.")
        if self.acceptor_indices.size == 0:
            raise ValueError("No acceptor atoms matched the selected compounds and labels in the initial frame.")
        if self.proton_indices.size == 0:
            raise ValueError("No proton atoms matched the selected compounds and labels in the initial frame.")

        self.depth_counts = np.zeros(self.max_depth, dtype=np.float64)
        self.shell_r2_sum = np.zeros(self.max_depth, dtype=np.float64)
        self.shell_r2_count = np.zeros(self.max_depth, dtype=np.int64)
        self.total_seeds = 0
        self.out_degree_sum = 0.0
        self.out_degree_count = 0
        self.out_degree2_count = 0
        self.out_branch_count = 0
        self.in_degree_sum = 0.0
        self.in_degree_count = 0
        self.in_degree2_count = 0
        self.in_branch_count = 0
        self.degree_sum = 0.0
        self.degree_count = 0
        self.degree2_count = 0
        self.branch_count = 0
        self.loop_density_sum = 0.0
        self.loop_density_count = 0
        self.mode_label = "alternating" if self.use_alternating else "network"
        self.mark_configured()

    def _resolve_compound_specs(self, compounds):
        topology_frame = self.traj.topology_frame
        compound_indices = [compound.compound_index for compound in compounds]
        resolved = {
            index: pair
            for index, pair in zip(compound_indices, self.resolve_compound_types(compound_indices))
        }

        specs = []
        for compound in compounds:
            compound_type, key = resolved[compound.compound_index]
            acceptor_local_indices = tuple()
            if compound.acceptor_labels:
                acceptor_local_indices = tuple(
                    topology_frame.resolve_selection(compound_type, compound.acceptor_labels).local_indices
                )
                if not acceptor_local_indices:
                    raise ValueError(
                        f"No acceptor atoms matched labels {compound.acceptor_labels} in compound type {compound_type.formula}."
                    )

            proton_local_indices = tuple()
            if compound.proton_labels:
                proton_local_indices = tuple(
                    topology_frame.resolve_selection(compound_type, compound.proton_labels).local_indices
                )
                if not proton_local_indices:
                    raise ValueError(
                        f"No proton atoms matched labels {compound.proton_labels} in compound type {compound_type.formula}."
                    )

            specs.append((key, acceptor_local_indices, proton_local_indices))
        return tuple(specs)

    def rebuild_runtime_state(self):
        topology_frame = self.traj.topology_frame
        acceptor_parts = []
        acceptor_owner_parts = []
        proton_parts = []
        proton_owner_parts = []
        representative_parts = []
        next_node = 0

        for compound_key, acceptor_local_indices, proton_local_indices in self.compound_specs:
            if not topology_frame.has_compound_type_key(compound_key):
                continue

            compound_type = topology_frame.get_compound_type_by_key(compound_key)
            molecule_atom_ids = topology_frame.get_molecule_atom_ids(compound_type)
            molecule_count = len(molecule_atom_ids)
            if molecule_count == 0:
                continue

            owner_ids = np.arange(next_node, next_node + molecule_count, dtype=np.int32)
            if acceptor_local_indices:
                acceptor_atom_ids = (
                    molecule_atom_ids[:, list(acceptor_local_indices)].reshape(-1).astype(np.int32, copy=False)
                )
                acceptor_parts.append(acceptor_atom_ids)
                acceptor_owner_parts.append(np.repeat(owner_ids, len(acceptor_local_indices)))
                representative_parts.append(molecule_atom_ids[:, acceptor_local_indices[0]].astype(np.int32, copy=False))
            else:
                representative_parts.append(molecule_atom_ids[:, proton_local_indices[0]].astype(np.int32, copy=False))

            if proton_local_indices:
                proton_atom_ids = (
                    molecule_atom_ids[:, list(proton_local_indices)].reshape(-1).astype(np.int32, copy=False)
                )
                proton_parts.append(proton_atom_ids)
                proton_owner_parts.append(np.repeat(owner_ids, len(proton_local_indices)))

            next_node += molecule_count

        self.n_nodes = next_node
        self.acceptor_indices = (
            np.concatenate(acceptor_parts) if acceptor_parts else np.empty(0, dtype=np.int32)
        )
        self.acceptor_owner_ids = (
            np.concatenate(acceptor_owner_parts) if acceptor_owner_parts else np.empty(0, dtype=np.int32)
        )
        self.proton_indices = (
            np.concatenate(proton_parts) if proton_parts else np.empty(0, dtype=np.int32)
        )
        self.proton_owner_ids = (
            np.concatenate(proton_owner_parts) if proton_owner_parts else np.empty(0, dtype=np.int32)
        )
        self.representative_atom_ids = (
            np.concatenate(representative_parts) if representative_parts else np.empty(0, dtype=np.int32)
        )

    def post_compound_update(self):
        self.rebuild_runtime_state()
        return self.n_nodes > 0 and self.acceptor_indices.size > 0 and self.proton_indices.size > 0

    def process_frame(self):
        if self.n_nodes == 0 or self.acceptor_indices.size == 0 or self.proton_indices.size == 0:
            return

        tree = cKDTree(self.traj.coords[self.acceptor_indices], boxsize=self.traj.box_size)
        nearby_acceptors = tree.query_ball_point(self.traj.coords[self.proton_indices], r=self.cutoff)
        neighbours = [set() for _ in range(self.n_nodes)]
        in_degrees = np.zeros(self.n_nodes, dtype=np.int32) if self.use_alternating else None

        for proton_position, acceptor_positions in enumerate(nearby_acceptors):
            source = int(self.proton_owner_ids[proton_position])
            for acceptor_position in acceptor_positions:
                target = int(self.acceptor_owner_ids[acceptor_position])
                if target == source:
                    continue
                if target in neighbours[source]:
                    continue
                neighbours[source].add(target)
                if self.use_alternating:
                    in_degrees[target] += 1
                else:
                    neighbours[target].add(source)

        out_degrees = np.array([len(nbrs) for nbrs in neighbours], dtype=np.int32)
        self.out_degree_sum += float(out_degrees.sum())
        self.out_degree_count += int(len(out_degrees))
        self.out_degree2_count += int(np.count_nonzero(out_degrees == 2))
        self.out_branch_count += int(np.count_nonzero(out_degrees >= 3))

        if self.use_alternating:
            self.in_degree_sum += float(in_degrees.sum())
            self.in_degree_count += int(len(in_degrees))
            self.in_degree2_count += int(np.count_nonzero(in_degrees == 2))
            self.in_branch_count += int(np.count_nonzero(in_degrees >= 3))
        else:
            degrees = out_degrees
            self.degree_sum += float(degrees.sum())
            self.degree_count += int(len(degrees))
            self.degree2_count += int(np.count_nonzero(degrees == 2))
            self.branch_count += int(np.count_nonzero(degrees >= 3))
            edge_count = int(degrees.sum() // 2)
            component_count = _count_components(neighbours)
            self.loop_density_sum += (edge_count - self.n_nodes + component_count) / self.n_nodes
            self.loop_density_count += 1

        self.total_seeds += self.n_nodes
        representative_coords = self.traj.coords[self.representative_atom_ids]
        box = self.traj.box_size
        for seed in range(self.n_nodes):
            visited = np.zeros(self.n_nodes, dtype=bool)
            visited[seed] = True
            frontier = np.array([seed], dtype=np.int32)
            seed_position = representative_coords[seed]
            for depth in range(self.max_depth):
                next_frontier = set()
                for node in frontier:
                    next_frontier.update(neighbours[node])
                next_nodes = np.fromiter(
                    (node for node in next_frontier if not visited[node]),
                    dtype=np.int32,
                )
                if next_nodes.size == 0:
                    break
                self.depth_counts[depth] += next_nodes.size
                deltas = minimum_image(representative_coords[next_nodes] - seed_position, box)
                self.shell_r2_sum[depth] += float(np.sum(deltas * deltas))
                self.shell_r2_count[depth] += next_nodes.size
                visited[next_nodes] = True
                frontier = next_nodes

    def postprocess(self):
        if self.total_seeds == 0:
            console.warn("No eligible hydrogen-bond seeds were accumulated.")
            return

        shell_counts = self.depth_counts / self.total_seeds
        a_l = 1.0 + np.cumsum(shell_counts)
        r_l = np.full(self.max_depth, np.nan, dtype=np.float64)
        valid_r = self.shell_r2_count > 0
        r_l[valid_r] = np.sqrt(self.shell_r2_sum[valid_r] / self.shell_r2_count[valid_r])

        console.info("Percolation shell observables:")
        for depth, shell_count in enumerate(shell_counts, start=1):
            summary = f"S(l)={shell_count:.6f}, A(l)={a_l[depth - 1]:.6f}"
            r_value = r_l[depth - 1]
            if np.isfinite(r_value):
                summary += f", R(l)={r_value:.6f}"
            else:
                summary += ", R(l)=nan"
            console.key_value(f"Depth {depth}", summary, indent=2)

        filename = build_output_filename("percolation", [self.mode_label])
        rows = [(0, 0.0, 1.0, 0.0)]
        rows.extend(
            (depth, shell_count, a_l[depth - 1], r_l[depth - 1])
            for depth, shell_count in enumerate(shell_counts, start=1)
        )
        write_table(
            filename,
            headers=["Depth", "ShellPopulation", "A_l", "R_l"],
            data=rows,
        )

        depths = np.arange(1, self.max_depth + 1, dtype=float)
        fit_mask = depths >= self.fit_min_depth
        if self.fit_max_depth != -1:
            fit_mask &= depths <= self.fit_max_depth
        d_l, _ = _fit_power_law(depths[fit_mask], a_l[fit_mask])
        r_slope, _ = _fit_power_law(depths[fit_mask], r_l[fit_mask])
        d_min = np.nan
        if np.isfinite(r_slope) and r_slope != 0:
            d_min = 1.0 / r_slope
        d_f = d_l * d_min if np.isfinite(d_l) and np.isfinite(d_min) else np.nan

        console.key_value("d_l", f"{d_l:.6f}" if np.isfinite(d_l) else "nan")
        console.key_value("d_min", f"{d_min:.6f}" if np.isfinite(d_min) else "nan")
        console.key_value("d_f", f"{d_f:.6f}" if np.isfinite(d_f) else "nan")

        if self.use_alternating:
            mean_out_degree = self.out_degree_sum / self.out_degree_count if self.out_degree_count else np.nan
            mean_in_degree = self.in_degree_sum / self.in_degree_count if self.in_degree_count else np.nan
            f_out_degree2 = self.out_degree2_count / self.out_degree_count if self.out_degree_count else np.nan
            f_in_degree2 = self.in_degree2_count / self.in_degree_count if self.in_degree_count else np.nan
            f_out_branch = self.out_branch_count / self.out_degree_count if self.out_degree_count else np.nan
            f_in_branch = self.in_branch_count / self.in_degree_count if self.in_degree_count else np.nan

            console.key_value("mean out-degree", f"{mean_out_degree:.6f}" if np.isfinite(mean_out_degree) else "nan")
            console.key_value("mean in-degree", f"{mean_in_degree:.6f}" if np.isfinite(mean_in_degree) else "nan")
            console.key_value("f_out_degree2", f"{f_out_degree2:.6f}" if np.isfinite(f_out_degree2) else "nan")
            console.key_value("f_in_degree2", f"{f_in_degree2:.6f}" if np.isfinite(f_in_degree2) else "nan")
            console.key_value("f_out_branch", f"{f_out_branch:.6f}" if np.isfinite(f_out_branch) else "nan")
            console.key_value("f_in_branch", f"{f_in_branch:.6f}" if np.isfinite(f_in_branch) else "nan")
        else:
            mean_degree = self.degree_sum / self.degree_count if self.degree_count else np.nan
            f_degree2 = self.degree2_count / self.degree_count if self.degree_count else np.nan
            f_branch = self.branch_count / self.degree_count if self.degree_count else np.nan
            loop_density = self.loop_density_sum / self.loop_density_count if self.loop_density_count else np.nan

            console.key_value("mean degree", f"{mean_degree:.6f}" if np.isfinite(mean_degree) else "nan")
            console.key_value("f_degree2", f"{f_degree2:.6f}" if np.isfinite(f_degree2) else "nan")
            console.key_value("f_branch", f"{f_branch:.6f}" if np.isfinite(f_branch) else "nan")
            console.key_value("loop density", f"{loop_density:.6f}" if np.isfinite(loop_density) else "nan")

        console.success(f"Saved percolation results to {filename}")
