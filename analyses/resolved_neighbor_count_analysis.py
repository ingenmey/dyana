from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from analyses.common.base_analysis import BaseAnalysis
from io_support.console import console
from io_support.output_writer import build_output_filename, format_selection, format_selection_group, write_table


@dataclass(frozen=True)
class ObservedSiteConfig:
    """One observed-site selection contributing to one collapsed observable."""

    compound_index: int
    labels: list[str]
    cutoff: float
    exclude_same_molecule: bool = True

    def __post_init__(self):
        if self.compound_index < 0:
            raise ValueError("compound_index must be >= 0.")
        if not self.labels:
            raise ValueError("labels must not be empty.")
        if self.cutoff <= 0:
            raise ValueError("cutoff must be positive.")


@dataclass(frozen=True)
class ObservableConfig:
    """One resolved observable around one reference site."""

    ref_labels: list[str]
    observed_sites: list[ObservedSiteConfig]

    def __post_init__(self):
        if len(self.ref_labels) != 1:
            raise ValueError("Each observable must define exactly one reference-site label.")
        if not self.observed_sites:
            raise ValueError("Each observable must contain at least one observed-site selection.")


@dataclass(frozen=True)
class ResolvedNeighborCountConfig:
    """Configuration for fully resolved joint neighbour-count analysis."""

    ref_compound_index: int
    observables: list[ObservableConfig]

    def __post_init__(self):
        if self.ref_compound_index < 0:
            raise ValueError("ref_compound_index must be >= 0.")
        if not self.observables:
            raise ValueError("observables must not be empty.")


@dataclass
class _ResolvedObservedSite:
    compound_key: tuple
    formula: str
    labels: list[str]
    cutoff: float
    exclude_same_molecule: bool
    local_indices: tuple[int, ...]
    compound_type: object | None = None
    atom_ids: np.ndarray | None = None


@dataclass
class _ResolvedObservable:
    ref_labels: list[str]
    ref_local_index: int
    ref_atom_ids: np.ndarray
    observed_sites: list[_ResolvedObservedSite]


class ResolvedNeighborCountAnalysis(BaseAnalysis):
    """Fully resolved joint neighbour-count analysis with collapsed observables."""

    CONFIG_CLASS = ResolvedNeighborCountConfig

    def prompt_config(self, provider=None):
        input_provider = self.get_input_provider(provider)
        ref_compound_index, ref_compound = self.compound_selection(role="reference", provider=input_provider)

        observables = []
        observable_index = 1
        while True:
            ref_labels = self._prompt_reference_site(observable_index, ref_compound, input_provider)
            observed_sites = []
            observed_selection = self.compound_selection(
                role="observed",
                multi=True,
                prompt_text=f"Choose the observed compounds for observable {observable_index} (comma-separated numbers): ",
                provider=input_provider,
            )
            for obs_compound_index, obs_compound in observed_selection:
                obs_labels = self.atom_selection(
                    role="observed",
                    compound=obs_compound,
                    prompt_text=(
                        f"Which atom(s) in observed compound {obs_compound.type_id + 1} ({obs_compound.formula}) "
                        f"belong to observable {observable_index}? (comma-separated) "
                    ),
                    provider=input_provider,
                )
                cutoff = input_provider.ask_float(
                    "Neighbour cutoff distance Angstrom: ",
                    default=3.5,
                    minval=0.1,
                )
                exclude_same_molecule = input_provider.ask_bool(
                    "Exclude observed atoms that belong to the same molecule as the reference atom?",
                    True,
                )
                observed_sites.append(
                    ObservedSiteConfig(
                        compound_index=obs_compound_index,
                        labels=obs_labels,
                        cutoff=cutoff,
                        exclude_same_molecule=exclude_same_molecule,
                    )
                )

            observables.append(ObservableConfig(ref_labels=ref_labels, observed_sites=observed_sites))
            if not input_provider.ask_bool("Add another observable?", False):
                break
            observable_index += 1

        return ResolvedNeighborCountConfig(
            ref_compound_index=ref_compound_index,
            observables=observables,
        )

    def _prompt_reference_site(self, observable_index, ref_compound, input_provider):
        while True:
            ref_labels = self.atom_selection(
                role="reference",
                compound=ref_compound,
                prompt_text=f"Which atom defines reference site {observable_index}? ",
                provider=input_provider,
            )
            if len(ref_labels) == 1:
                return ref_labels
            console.warn("Please choose exactly one reference-site label for each observable.")

    def configure(self, config: ResolvedNeighborCountConfig):
        self.config = config
        self.ref_compound_index = config.ref_compound_index
        (self.ref_type, self.ref_key), = self.resolve_compound_types([self.ref_compound_index])
        topology_frame = self.traj.topology_frame

        self.observables = []
        for observable_config in config.observables:
            ref_selection = topology_frame.resolve_selection(self.ref_type, observable_config.ref_labels)
            if len(ref_selection.local_indices) != 1:
                raise ValueError("Each observable must resolve to exactly one reference site atom.")

            observed_sites = []
            for observed_site_config in observable_config.observed_sites:
                (obs_type, obs_key), = self.resolve_compound_types([observed_site_config.compound_index])
                obs_selection = topology_frame.resolve_selection(obs_type, observed_site_config.labels)
                if len(obs_selection.local_indices) == 0:
                    raise ValueError("Observed-site selection matched no atoms in the initial frame.")
                observed_sites.append(
                    _ResolvedObservedSite(
                        compound_key=obs_key,
                        formula=obs_type.formula,
                        labels=list(observed_site_config.labels),
                        cutoff=observed_site_config.cutoff,
                        exclude_same_molecule=observed_site_config.exclude_same_molecule,
                        local_indices=obs_selection.local_indices,
                        compound_type=obs_type,
                        atom_ids=np.empty(0, dtype=np.int32),
                    )
                )

            self.observables.append(
                _ResolvedObservable(
                    ref_labels=list(observable_config.ref_labels),
                    ref_local_index=ref_selection.local_indices[0],
                    ref_atom_ids=np.empty(0, dtype=np.int32),
                    observed_sites=observed_sites,
                )
            )

        self.rebuild_runtime_state()
        if not self.observables or self.observables[0].ref_atom_ids.size == 0:
            raise ValueError("No reference sites matched in the initial frame.")

        self.joint_hist = Counter()
        self.total_reference_sites = 0
        self.mark_configured()
        self._report_observables()

    def _report_observables(self):
        console.info("Resolved observables:")
        for observable_index, observable in enumerate(self.observables, start=1):
            observed_summary = " + ".join(
                f"{'+'.join(site.labels)} in {site.formula} (r<{site.cutoff:g})"
                for site in observable.observed_sites
            )
            console.key_value(
                f"N{observable_index}",
                f"ref {'+'.join(observable.ref_labels)} in {self.ref_type.formula} ; obs {observed_summary}",
                indent=2,
            )

    def rebuild_runtime_state(self):
        topology_frame = self.traj.topology_frame
        for observable in self.observables:
            observable.ref_atom_ids = topology_frame.get_atom_ids_for_local_indices(
                self.ref_type,
                (observable.ref_local_index,),
            )
            for observed_site in observable.observed_sites:
                if topology_frame.has_compound_type_key(observed_site.compound_key):
                    observed_site.compound_type = topology_frame.get_compound_type_by_key(observed_site.compound_key)
                    observed_site.atom_ids = topology_frame.get_atom_ids_for_local_indices(
                        observed_site.compound_type,
                        observed_site.local_indices,
                    )
                else:
                    observed_site.compound_type = None
                    observed_site.atom_ids = np.empty(0, dtype=np.int32)

    def post_compound_update(self):
        topology_frame = self.traj.topology_frame
        if not topology_frame.has_compound_type_key(self.ref_key):
            return False

        self.ref_type = topology_frame.get_compound_type_by_key(self.ref_key)
        self.rebuild_runtime_state()
        return bool(self.observables) and self.observables[0].ref_atom_ids.size > 0

    def process_frame(self):
        if not self.observables or self.observables[0].ref_atom_ids.size == 0:
            return

        coords = self.traj.coords
        box = self.traj.box_size
        atom_to_type_id = self.traj.topology_frame.atom_to_type_id
        atom_to_molecule_index = self.traj.topology_frame.atom_to_molecule_index
        n_reference_sites = self.observables[0].ref_atom_ids.size
        observable_counts = []

        for observable in self.observables:
            ref_atom_ids = observable.ref_atom_ids
            ref_coords = coords[ref_atom_ids]
            per_site_neighbor_lists = []

            for observed_site in observable.observed_sites:
                if observed_site.atom_ids.size == 0:
                    per_site_neighbor_lists.append((observed_site, [[] for _ in range(n_reference_sites)]))
                    continue
                tree = cKDTree(coords[observed_site.atom_ids], boxsize=box)
                neighbors = tree.query_ball_point(ref_coords, observed_site.cutoff)
                per_site_neighbor_lists.append((observed_site, neighbors))

            counts = np.zeros(n_reference_sites, dtype=np.int32)
            for ref_position, ref_atom_id in enumerate(ref_atom_ids):
                matched = set()
                ref_type_id = int(atom_to_type_id[ref_atom_id])
                ref_molecule_index = int(atom_to_molecule_index[ref_atom_id])

                for observed_site, neighbors in per_site_neighbor_lists:
                    for neighbor_position in neighbors[ref_position]:
                        obs_atom_id = int(observed_site.atom_ids[neighbor_position])
                        if obs_atom_id == ref_atom_id:
                            continue
                        if observed_site.exclude_same_molecule:
                            if (
                                int(atom_to_type_id[obs_atom_id]) == ref_type_id
                                and int(atom_to_molecule_index[obs_atom_id]) == ref_molecule_index
                            ):
                                continue
                        matched.add(obs_atom_id)

                counts[ref_position] = len(matched)

            observable_counts.append(counts)

        self.total_reference_sites += n_reference_sites
        for ref_position in range(n_reference_sites):
            key = tuple(int(counts[ref_position]) for counts in observable_counts)
            self.joint_hist[key] += 1

    def postprocess(self):
        if self.total_reference_sites == 0:
            console.warn("No resolved neighbour-count values were accumulated.")
            return

        observable_parts = [self._format_observable_filename_part(observable) for observable in self.observables]
        joint_filename = build_output_filename("rncount_joint", observable_parts)
        joint_rows = [
            [*counts, occurrences / self.total_reference_sites]
            for counts, occurrences in sorted(self.joint_hist.items())
        ]
        joint_headers = [f"N{index}" for index in range(1, len(self.observables) + 1)] + ["P"]
        write_table(joint_filename, headers=joint_headers, data=joint_rows)
        console.success(f"Saved resolved neighbour-count results to {joint_filename}")

        for observable_index in range(len(self.observables)):
            marginal = Counter()
            for counts, occurrences in self.joint_hist.items():
                marginal[counts[observable_index]] += occurrences

            max_count = max(marginal)
            marginal_filename = build_output_filename(
                f"rncount_obs{observable_index + 1}",
                [observable_parts[observable_index]],
            )
            marginal_rows = [
                [count, marginal.get(count, 0) / self.total_reference_sites]
                for count in range(max_count + 1)
            ]
            axis_name = f"N{observable_index + 1}"
            write_table(
                marginal_filename,
                headers=[axis_name, f"P({axis_name})"],
                data=marginal_rows,
            )
            console.success(f"Saved marginal neighbour-count results to {marginal_filename}")

    def _format_observable_filename_part(self, observable: _ResolvedObservable) -> str:
        observed_entries = [
            (observed_site.labels, observed_site.formula)
            for observed_site in observable.observed_sites
        ]
        return "_".join(
            [
                format_selection(observable.ref_labels, self.ref_type.formula),
                format_selection_group(observed_entries),
            ]
        )
