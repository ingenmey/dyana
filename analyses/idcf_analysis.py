from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import partial

import numpy as np
from scipy.spatial import cKDTree

from analyses.common.base_analysis import BaseAnalysis
from framework.analysis_params import (
    AtomLabelsParam,
    BoolParam,
    ChoiceParam,
    CompoundParam,
    FloatParam,
    ForEach,
    Group,
    IntParam,
)
from io_support.console import console
from io_support.output_writer import build_output_filename, format_selection_group, write_table


@dataclass(frozen=True)
class IDCFCompoundSelection:
    compound_index: int
    labels: list[str]

    def __post_init__(self):
        if self.compound_index < 0:
            raise ValueError("compound_index must be >= 0.")
        labels = [str(label) for label in self.labels]
        if not labels:
            raise ValueError("labels must not be empty.")
        object.__setattr__(self, "labels", labels)


@dataclass(frozen=True)
class IDCFConfig:
    proton_sites: list[IDCFCompoundSelection]
    acceptor_sites: list[IDCFCompoundSelection]
    acceptor_identity: str = "molecule"
    bond_cutoff: float | None = None
    use_continuous: bool = False
    corr_depth: int = 100
    frame_time_fs: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "proton_sites", _normalize_sites("proton_sites", self.proton_sites))
        object.__setattr__(self, "acceptor_sites", _normalize_sites("acceptor_sites", self.acceptor_sites))
        if self.acceptor_identity not in {"molecule", "site"}:
            raise ValueError("acceptor_identity must be 'molecule' or 'site'.")
        if self.bond_cutoff is not None and self.bond_cutoff <= 0:
            raise ValueError("bond_cutoff must be > 0 when set.")
        if self.corr_depth < 1:
            raise ValueError("corr_depth must be >= 1.")
        if self.frame_time_fs <= 0:
            raise ValueError("frame_time_fs must be positive.")


def _build_sites(role: str, compound_indices, sites):
    if not compound_indices:
        raise ValueError(f"{role} compound selection must not be empty.")
    sites = _normalize_sites(f"{role}_sites", sites)
    if {int(index) for index in compound_indices} != {site.compound_index for site in sites}:
        raise ValueError(f"{role} sites do not match the chosen {role} compound indices.")
    return sites


def _normalize_sites(name: str, sites: list[IDCFCompoundSelection]) -> list[IDCFCompoundSelection]:
    if not sites:
        raise ValueError(f"{name} must not be empty.")
    indices = [site.compound_index for site in sites]
    if len(indices) != len(set(indices)):
        raise ValueError(f"{name} must not repeat the same compound index.")
    return list(sites)


def _state_description(state: int) -> str:
    if state == 0:
        return "acceptor context with 0 selected protons bound (unprotonated)"
    if state == 1:
        return "acceptor context with 1 selected proton bound"
    return f"acceptor context with {state} selected protons bound"


def _compute_idcf(interval_groups, origin_counts, total_frames: int, max_tau: int, use_continuous: bool) -> np.ndarray:
    idcf = np.zeros(max_tau, dtype=np.float64)

    if use_continuous:
        for intervals in interval_groups:
            for start, end in intervals:
                length = end - start
                if length > 0:
                    local_max = min(max_tau, length)
                    idcf[:local_max] += length - np.arange(local_max, dtype=np.float64)
    else:
        for intervals in interval_groups:
            occupancy = np.zeros(total_frames, dtype=np.float64)
            for start, end in intervals:
                occupancy[start:end] = 1.0
            idcf += np.correlate(
                occupancy,
                occupancy,
                mode="full",
            )[total_frames - 1:total_frames - 1 + max_tau]

    valid = origin_counts > 0
    idcf[valid] /= origin_counts[valid]
    idcf[~valid] = 0.0
    return idcf


class IDCFAnalysis(BaseAnalysis):
    """Identity autocorrelation of acceptor contexts."""

    CONFIG_CLASS = IDCFConfig
    CONFIG_SCHEMA = [
        Group(
            name="proton_sites",
            config_class=partial(_build_sites, "proton"),
            steps=[
                CompoundParam(
                    name="compound_indices",
                    role="proton",
                    multi=True,
                    prompt="Choose the compounds with transferable protons (comma-separated numbers): ",
                ),
                ForEach(
                    source="compound_indices",
                    item_name="compound_index",
                    steps=[
                        AtomLabelsParam(
                            name="labels",
                            role="proton",
                            compound="compound_index",
                        ),
                    ],
                    collect_as="sites",
                    collect_mode="list",
                    config_class=IDCFCompoundSelection,
                    include_item_as="compound_index",
                ),
            ],
        ),
        Group(
            name="acceptor_sites",
            config_class=partial(_build_sites, "acceptor"),
            steps=[
                CompoundParam(
                    name="compound_indices",
                    role="acceptor",
                    multi=True,
                    prompt="Choose the compounds that may accept protons (comma-separated numbers): ",
                ),
                ForEach(
                    source="compound_indices",
                    item_name="compound_index",
                    steps=[
                        AtomLabelsParam(
                            name="labels",
                            role="acceptor",
                            compound="compound_index",
                        ),
                    ],
                    collect_as="sites",
                    collect_mode="list",
                    config_class=IDCFCompoundSelection,
                    include_item_as="compound_index",
                ),
            ],
        ),
        ChoiceParam(
            name="acceptor_identity",
            prompt="Track acceptor identities by molecule or by individual site?",
            choices=["molecule", "site"],
            default="molecule",
        ),
        FloatParam(
            name="bond_cutoff",
            prompt="Enter maximum proton-acceptor bond distance (Angstrom), or leave blank for no cutoff: ",
            default=None,
            allow_none=True,
            minval=1e-12,
        ),
        BoolParam(
            name="use_continuous",
            prompt="Use the continuous autocorrelation function?",
            default=False,
        ),
        IntParam(
            name="corr_depth",
            prompt="Enter the maximum correlation depth (frames): ",
            default=100,
            minval=1,
        ),
        FloatParam(
            name="frame_time_fs",
            prompt="Enter the time per frame (fs): ",
            default=1.0,
            minval=1e-12,
        ),
    ]

    def setup(self):
        console.info(
            "IDCF assumes a static topology with transferable protons already separated during the initial topology setup."
        )
        console.info("Per-frame molecule recognition is disabled; only coordinates are updated during the frame loop.")
        super().setup()

    def configure(self, config: IDCFConfig):
        self.bind_config(config)
        self.allow_compound_update = False
        self.track_mode_label = self.acceptor_identity

        topology_frame = self.traj.topology_frame
        self.proton_specs, self.proton_selection_entries = self._resolve_named_sites(self.proton_sites, topology_frame)
        self.acceptor_specs, self.acceptor_selection_entries = self._resolve_named_sites(
            self.acceptor_sites,
            topology_frame,
        )
        self.rebuild_runtime_state()

        if self.proton_indices.size == 0:
            raise ValueError("No proton atoms matched the selected compounds and labels in the initial frame.")
        if self.acceptor_indices.size == 0:
            raise ValueError("No acceptor sites matched the selected compounds and labels in the initial frame.")

        self.open_intervals = {}
        self.identity_intervals = defaultdict(list)
        self.frame_state_counts = []
        self.mark_configured()

    def _resolve_named_sites(self, sites, topology_frame):
        compound_indices = sorted({site.compound_index for site in sites})
        resolved = {
            index: pair
            for index, pair in zip(compound_indices, self.resolve_compound_types(compound_indices))
        }

        runtime_specs = []
        selection_entries = []
        for site in sites:
            compound_type, key = resolved[site.compound_index]
            local_indices = tuple(topology_frame.resolve_selection(compound_type, site.labels).local_indices)
            if not local_indices:
                raise ValueError(
                    f"No atoms matched labels {site.labels} in compound type {compound_type.formula}."
                )
            runtime_specs.append((key, local_indices))
            selection_entries.append((site.labels, compound_type.formula))
        return tuple(runtime_specs), selection_entries

    def rebuild_runtime_state(self):
        topology_frame = self.traj.topology_frame

        proton_parts = []
        for compound_key, local_indices in self.proton_specs:
            compound_type = topology_frame.get_compound_type_by_key(compound_key)
            atom_ids = topology_frame.get_atom_ids_for_local_indices(compound_type, local_indices)
            if atom_ids.size:
                proton_parts.append(atom_ids.astype(np.int32, copy=False))
        self.proton_indices = (
            np.concatenate(proton_parts) if proton_parts else np.empty(0, dtype=np.int32)
        )

        acceptor_parts = []
        context_parts = []
        next_context = 0
        for compound_key, local_indices in self.acceptor_specs:
            compound_type = topology_frame.get_compound_type_by_key(compound_key)
            molecule_atom_ids = topology_frame.get_molecule_atom_ids(compound_type)
            if len(molecule_atom_ids) == 0:
                continue

            atom_ids = molecule_atom_ids[:, list(local_indices)].reshape(-1).astype(np.int32, copy=False)
            acceptor_parts.append(atom_ids)
            if self.acceptor_identity == "molecule":
                context_parts.append(
                    np.repeat(
                        np.arange(next_context, next_context + len(molecule_atom_ids), dtype=np.int32),
                        len(local_indices),
                    )
                )
                next_context += len(molecule_atom_ids)
            else:
                context_parts.append(np.arange(next_context, next_context + atom_ids.size, dtype=np.int32))
                next_context += atom_ids.size

        self.acceptor_indices = (
            np.concatenate(acceptor_parts) if acceptor_parts else np.empty(0, dtype=np.int32)
        )
        self.acceptor_context_ids = (
            np.concatenate(context_parts) if context_parts else np.empty(0, dtype=np.int32)
        )
        self.n_acceptors = int(np.unique(self.acceptor_context_ids).size)

    def post_compound_update(self):
        return True

    def process_frame(self):
        frame_number = self.processed_frames
        frame_entities = self._collect_frame_entities()
        active_entities = set(frame_entities)
        state_counts = defaultdict(int)
        for _, proton_ids in frame_entities:
            state_counts[len(proton_ids)] += 1
        self.frame_state_counts.append(dict(state_counts))

        for entity, start in list(self.open_intervals.items()):
            if entity not in active_entities:
                self.identity_intervals[entity].append((start, frame_number))
                del self.open_intervals[entity]

        for entity in active_entities:
            if entity not in self.open_intervals:
                self.open_intervals[entity] = frame_number

    def _collect_frame_entities(self) -> list[tuple[int, tuple[int, ...]]]:
        if self.proton_indices.size == 0 or self.acceptor_indices.size == 0:
            return []

        tree = cKDTree(self.traj.coords[self.acceptor_indices], boxsize=self.traj.box_size)
        if self.bond_cutoff is None:
            _, nearest = tree.query(self.traj.coords[self.proton_indices], k=1)
            valid = np.ones(self.proton_indices.size, dtype=bool)
        else:
            distances, nearest = tree.query(
                self.traj.coords[self.proton_indices],
                k=1,
                distance_upper_bound=self.bond_cutoff,
            )
            valid = np.isfinite(distances) & (nearest < self.acceptor_indices.size)

        bound_protons = [[] for _ in range(self.n_acceptors)]
        for proton_id, is_valid in enumerate(valid):
            if not is_valid:
                continue
            context_id = int(self.acceptor_context_ids[int(nearest[proton_id])])
            bound_protons[context_id].append(proton_id)

        return [
            (context_id, tuple(proton_ids))
            for context_id, proton_ids in enumerate(bound_protons)
        ]

    def postprocess(self):
        total_frames = self.processed_frames
        if total_frames <= 0:
            console.warn("No frames were processed.")
            return

        for entity, start in self.open_intervals.items():
            self.identity_intervals[entity].append((start, total_frames))
        self.open_intervals.clear()

        if not self.identity_intervals:
            console.warn("No acceptor identities were observed.")
            return

        max_tau = min(self.corr_depth, total_frames)
        origin_counts_total = np.full(max_tau, self.n_acceptors, dtype=np.float64) * np.arange(
            total_frames,
            total_frames - max_tau,
            -1,
            dtype=np.float64,
        )
        intervals_by_state = defaultdict(list)
        for entity, intervals in self.identity_intervals.items():
            intervals_by_state[len(entity[1])].append(intervals)

        console.info("IDCF state labels:")
        for state in sorted(intervals_by_state):
            console.key_value(f"n{state}", _state_description(state), indent=2)
        console.info("Each state-resolved file still correlates exact acceptor-plus-proton-set identities within that state.")

        summary_idcf = _compute_idcf(
            self.identity_intervals.values(),
            origin_counts_total,
            total_frames,
            max_tau,
            self.use_continuous,
        )
        summary_filename = build_output_filename(
            "idcf",
            [
                self.track_mode_label,
                format_selection_group(self.proton_selection_entries),
                format_selection_group(self.acceptor_selection_entries),
            ],
        )
        summary_rows = [(tau * self.frame_time_fs / 1000.0, value) for tau, value in enumerate(summary_idcf)]
        write_table(summary_filename, headers=["tau/ps", "IDCF"], data=summary_rows)
        console.success(f"Saved summarized IDCF results to {summary_filename}")

        for state in sorted(intervals_by_state):
            state_counts = np.array(
                [frame_counts.get(state, 0) for frame_counts in self.frame_state_counts],
                dtype=np.float64,
            )
            if state_counts.sum() <= 0:
                continue

            origin_counts = np.cumsum(state_counts, dtype=np.float64)
            origin_counts = origin_counts[[total_frames - tau - 1 for tau in range(max_tau)]]
            idcf = _compute_idcf(
                intervals_by_state[state],
                origin_counts,
                total_frames,
                max_tau,
                self.use_continuous,
            )

            filename = build_output_filename(
                "idcf",
                [
                    self.track_mode_label,
                    f"n{state}",
                    format_selection_group(self.proton_selection_entries),
                    format_selection_group(self.acceptor_selection_entries),
                ],
            )
            rows = [(tau * self.frame_time_fs / 1000.0, value) for tau, value in enumerate(idcf)]
            write_table(filename, headers=["tau/ps", "IDCF"], data=rows)
            console.success(f"Saved IDCF results for state n={state} to {filename}")
