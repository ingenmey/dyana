from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import partial

import numpy as np
from scipy.spatial import cKDTree

from analyses.common.base_analysis import BaseAnalysis
from framework.analysis_params import (
    AtomLabelsParam,
    ChoiceParam,
    CompoundParam,
    FloatParam,
    ForEach,
    Group,
    IntListParam,
    IntParam,
)
from io_support.console import console
from io_support.output_writer import build_output_filename, write_table


@dataclass(frozen=True)
class PCCFCompoundSelection:
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
class PCCFConfig:
    proton_sites: list[PCCFCompoundSelection]
    acceptor_sites: list[PCCFCompoundSelection]
    acceptor_identity: str = "molecule"
    bond_cutoff: float = 1.2
    dwell_threshold: int = 2
    max_unassigned_gap: int = 2
    max_chain_gaps: list[int] = field(default_factory=lambda: [100])

    def __post_init__(self):
        object.__setattr__(self, "proton_sites", _normalize_sites("proton_sites", self.proton_sites))
        object.__setattr__(self, "acceptor_sites", _normalize_sites("acceptor_sites", self.acceptor_sites))

        if self.acceptor_identity not in {"molecule", "site"}:
            raise ValueError("acceptor_identity must be 'molecule' or 'site'.")
        if self.bond_cutoff <= 0:
            raise ValueError("bond_cutoff must be > 0.")
        if self.dwell_threshold < 0:
            raise ValueError("dwell_threshold must be >= 0.")
        if self.max_unassigned_gap < 0:
            raise ValueError("max_unassigned_gap must be >= 0.")

        object.__setattr__(
            self,
            "max_chain_gaps",
            _normalize_nonnegative_ints("max_chain_gaps", self.max_chain_gaps),
        )


@dataclass(frozen=True)
class TransferEvent:
    frame: int
    proton_id: int
    donor_context: int
    acceptor_context: int


def _build_sites(role: str, compound_indices, sites):
    if not compound_indices:
        raise ValueError(f"{role} compound selection must not be empty.")
    sites = _normalize_sites(f"{role}_sites", sites)
    if {int(index) for index in compound_indices} != {site.compound_index for site in sites}:
        raise ValueError(f"{role} sites do not match the chosen {role} compound indices.")
    return sites


def _normalize_sites(name: str, sites: list[PCCFCompoundSelection]) -> list[PCCFCompoundSelection]:
    if not sites:
        raise ValueError(f"{name} must not be empty.")
    indices = [site.compound_index for site in sites]
    if len(indices) != len(set(indices)):
        raise ValueError(f"{name} must not repeat the same compound index.")
    return list(sites)


def _normalize_nonnegative_ints(name: str, values) -> list[int]:
    normalized: list[int] = []
    seen: set[int] = set()
    for value in values:
        value = int(value)
        if value < 0:
            raise ValueError(f"{name} must contain only values >= 0.")
        if value not in seen:
            normalized.append(value)
            seen.add(value)
    if not normalized:
        raise ValueError(f"{name} must not be empty.")
    return normalized


class PCCFAnalysis(BaseAnalysis):
    """Proton coupling correlation / transfer-chain analysis."""

    CONFIG_CLASS = PCCFConfig
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
                    config_class=PCCFCompoundSelection,
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
                    config_class=PCCFCompoundSelection,
                    include_item_as="compound_index",
                ),
            ],
        ),
        ChoiceParam(
            name="acceptor_identity",
            prompt="Track acceptor contexts by molecule or by individual site?",
            choices=["molecule", "site"],
            default="molecule",
        ),
        FloatParam(
            name="bond_cutoff",
            prompt="Enter maximum proton-acceptor bond distance (in Angstrom): ",
            default=1.2,
            minval=1e-12,
        ),
        IntParam(
            name="dwell_threshold",
            prompt="Enter minimum dwell time on an acceptor (in assigned frames): ",
            default=2,
            minval=0,
        ),
        IntParam(
            name="max_unassigned_gap",
            prompt="Enter the maximum number of unassigned frames to bridge between stable residences: ",
            default=2,
            minval=0,
        ),
        IntListParam(
            name="max_chain_gaps",
            prompt="Enter one or more maximum chain gaps (in frames, comma-separated): ",
            default=[100],
            minval=0,
            min_items=1,
        ),
    ]

    def setup(self):
        console.info(
            "PCCF assumes a static topology with transferable protons already separated during the initial topology setup."
        )
        console.info("Per-frame molecule recognition is disabled; only coordinates are updated during the frame loop.")
        super().setup()

    def configure(self, config: PCCFConfig):
        self.bind_config(config)
        self.allow_compound_update = False
        self.track_mode_label = self.acceptor_identity
        self.required_residence_frames = max(self.dwell_threshold, 1)

        topology_frame = self.traj.topology_frame
        self.proton_specs = self._resolve_named_sites(self.proton_sites, topology_frame)
        self.acceptor_specs = self._resolve_named_sites(self.acceptor_sites, topology_frame)
        self.rebuild_runtime_state()

        if self.proton_indices.size == 0:
            raise ValueError("No proton atoms matched the selected compounds and labels in the initial frame.")
        if self.acceptor_indices.size == 0:
            raise ValueError("No acceptor sites matched the selected compounds and labels in the initial frame.")

        self.transfer_events: list[TransferEvent] = []
        self.confirmed_acceptors = np.full(self.proton_indices.size, -1, dtype=np.int32)
        self.confirmed_gaps = np.zeros(self.proton_indices.size, dtype=np.int32)
        self.candidate_acceptors = np.full(self.proton_indices.size, -1, dtype=np.int32)
        self.candidate_starts = np.full(self.proton_indices.size, -1, dtype=np.int32)
        self.candidate_counts = np.zeros(self.proton_indices.size, dtype=np.int32)
        self.candidate_gaps = np.zeros(self.proton_indices.size, dtype=np.int32)
        self.mark_configured()

    def _resolve_named_sites(self, sites, topology_frame):
        compound_indices = sorted({site.compound_index for site in sites})
        resolved = {
            index: pair
            for index, pair in zip(compound_indices, self.resolve_compound_types(compound_indices))
        }

        runtime_specs = []
        for site in sites:
            compound_type, key = resolved[site.compound_index]
            local_indices = tuple(topology_frame.resolve_selection(compound_type, site.labels).local_indices)
            if not local_indices:
                raise ValueError(
                    f"No atoms matched labels {site.labels} in compound type {compound_type.formula}."
                )
            runtime_specs.append((key, local_indices))
        return tuple(runtime_specs)

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
        self.n_protons = int(self.proton_indices.size)
        self.n_acceptors = int(np.unique(self.acceptor_context_ids).size)

    def post_compound_update(self):
        return True

    def process_frame(self):
        if self.proton_indices.size == 0 or self.acceptor_indices.size == 0:
            return

        tree = cKDTree(self.traj.coords[self.acceptor_indices], boxsize=self.traj.box_size)
        distances, nearest = tree.query(
            self.traj.coords[self.proton_indices],
            k=1,
            distance_upper_bound=self.bond_cutoff,
        )

        assignments = np.full(self.proton_indices.size, -1, dtype=np.int32)
        valid = np.isfinite(distances) & (nearest < self.acceptor_indices.size)
        assignments[valid] = self.acceptor_context_ids[np.asarray(nearest[valid], dtype=np.intp)]

        for proton_id, assignment in enumerate(assignments):
            self._update_residence(proton_id, int(assignment))

    def _update_residence(self, proton_id: int, assignment: int):
        confirmed = int(self.confirmed_acceptors[proton_id])
        candidate = int(self.candidate_acceptors[proton_id])

        if candidate >= 0 and assignment == candidate:
            self.candidate_counts[proton_id] += 1
            self.candidate_gaps[proton_id] = 0
            if self.candidate_counts[proton_id] >= self.required_residence_frames:
                self._confirm_candidate(proton_id)
            return

        if assignment < 0:
            if candidate >= 0:
                self.candidate_gaps[proton_id] += 1
                if self.candidate_gaps[proton_id] > self.max_unassigned_gap:
                    self.confirmed_acceptors[proton_id] = -1
                    self.confirmed_gaps[proton_id] = 0
                    self._reset_candidate(proton_id)
                return
            if confirmed >= 0:
                self.confirmed_gaps[proton_id] += 1
                if self.confirmed_gaps[proton_id] > self.max_unassigned_gap:
                    self.confirmed_acceptors[proton_id] = -1
                    self.confirmed_gaps[proton_id] = 0
            return

        if assignment == confirmed:
            self.confirmed_gaps[proton_id] = 0
            self._reset_candidate(proton_id)
            return

        self.candidate_acceptors[proton_id] = assignment
        self.candidate_starts[proton_id] = self.frame_idx
        self.candidate_counts[proton_id] = 1
        self.candidate_gaps[proton_id] = 0
        if self.required_residence_frames == 1:
            self._confirm_candidate(proton_id)

    def _confirm_candidate(self, proton_id: int):
        confirmed = int(self.confirmed_acceptors[proton_id])
        candidate = int(self.candidate_acceptors[proton_id])
        if confirmed >= 0 and confirmed != candidate:
            self.transfer_events.append(
                TransferEvent(
                    frame=int(self.candidate_starts[proton_id]),
                    proton_id=proton_id,
                    donor_context=confirmed,
                    acceptor_context=candidate,
                )
            )
        self.confirmed_acceptors[proton_id] = candidate
        self.confirmed_gaps[proton_id] = 0
        self._reset_candidate(proton_id)

    def _reset_candidate(self, proton_id: int):
        self.candidate_acceptors[proton_id] = -1
        self.candidate_starts[proton_id] = -1
        self.candidate_counts[proton_id] = 0
        self.candidate_gaps[proton_id] = 0

    def postprocess(self):
        if not self.transfer_events:
            console.warn("No confirmed proton transfer events were detected.")
            return

        console.key_value("Stable transfer events", len(self.transfer_events))

        for gap in self.max_chain_gaps:
            chain_counts = _count_transfer_chains(self.transfer_events, max_chain_gap=gap)
            total_chains = int(sum(chain_counts.values()))
            if total_chains == 0:
                continue

            running = 0
            rows = []
            for length in range(max(chain_counts), 0, -1):
                count = int(chain_counts.get(length, 0))
                running += count
                if running == 0:
                    continue
                rows.append((length, count, count / total_chains, running / total_chains))
            rows.reverse()

            filename = build_output_filename("pccf", [self.track_mode_label, f"gap{gap}"])
            write_table(filename, headers=["n", "count", "C(n)", "P(n)"], data=rows)
            console.success(f"Saved PCCF chain distribution for gap {gap} to {filename}")


def _count_transfer_chains(
    transfer_events: list[TransferEvent],
    max_chain_gap: int,
) -> Counter:
    events = sorted(
        transfer_events,
        key=lambda event: (event.frame, event.proton_id, event.donor_context, event.acceptor_context),
    )
    successors = np.full(len(events), -1, dtype=np.int32)
    pending_by_context: dict[int, list[int]] = defaultdict(list)

    for index, event in enumerate(events):
        pending = pending_by_context[event.donor_context]
        while pending and event.frame - events[pending[0]].frame > max_chain_gap:
            pending.pop(0)

        if pending:
            match_index = next(
                (
                    position
                    for position in range(len(pending) - 1, -1, -1)
                    if events[pending[position]].proton_id == event.proton_id
                ),
                None,
            )
            if match_index is None:
                parent_index = pending.pop(0)
            else:
                parent_index = pending.pop(match_index)
            successors[parent_index] = index

        pending_by_context[event.acceptor_context].append(index)

    chain_counts: Counter = Counter()

    for index, event in enumerate(events):
        visited = {event.donor_context}
        current_index = index
        length = 1

        while True:
            next_index = int(successors[current_index])
            if next_index < 0:
                break
            next_event = events[next_index]
            if next_event.acceptor_context in visited:
                break
            visited.add(events[current_index].acceptor_context)
            current_index = next_index
            length += 1

        chain_counts[length] += 1

    return chain_counts
