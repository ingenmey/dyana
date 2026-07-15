from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from analyses.common.base_analysis import BaseAnalysis
from analyses.common.pair_selectors import PairSelector, PairSelectorSpec, pair_selector_schema
from framework.analysis_params import BoolParam, CompoundParam, FloatParam, IntParam, When
from io_support.console import console
from io_support.output_writer import build_output_filename, format_selection, format_selection_group, write_table


@dataclass(frozen=True)
class DACFConfig:
    """Configuration for dimer existence autocorrelation analysis."""

    ref_compound_index: int
    selector: PairSelectorSpec
    use_continuous: bool = False
    corr_depth: int = 100
    apply_correction: bool = True
    frame_time_fs: float = 1.0
    count_missing_as_zero: bool = True

    def __post_init__(self):
        if self.ref_compound_index < 0:
            raise ValueError("ref_compound_index must be >= 0.")
        if self.corr_depth < 1:
            raise ValueError("corr_depth must be >= 1.")
        if self.frame_time_fs <= 0:
            raise ValueError("frame_time_fs must be positive.")


class DACFAnalysis(BaseAnalysis):
    """Dimer existence autocorrelation function analysis."""

    CONFIG_CLASS = DACFConfig
    CONFIG_SCHEMA = [
        CompoundParam(name="ref_compound_index", role="reference"),
        pair_selector_schema(
            name="selector",
            label="the dimer definition",
            ref_compound_field="ref_compound_index",
            default_use_distance=True,
            default_distance_max=3.5,
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
        When(
            source="use_continuous",
            value=False,
            steps=[
                BoolParam(
                    name="apply_correction",
                    prompt="Apply the finite-size equilibrium correction?",
                    default=True,
                ),
            ],
        ),
        FloatParam(
            name="frame_time_fs",
            prompt="Enter the time per frame (fs): ",
            default=1.0,
            minval=1e-12,
        ),
        BoolParam(
            name="count_missing_as_zero",
            prompt="Count frames with missing selected compounds as beta=0 instead of skipping them?",
            default=True,
        ),
    ]

    def configure(self, config: DACFConfig):
        self.bind_config(config, exclude=("selector",))
        self.selector_spec = config.selector
        (self.ref_type, self.ref_key), = self.resolve_compound_types([self.ref_compound_index])
        self.dimer_selector = PairSelector(self, self.ref_type, self.ref_key, self.selector_spec)
        self.rebuild_runtime_state()

        self.frame_active = True
        self.sampled_ref_count_sum = 0.0
        self.sampled_obs_count_sum = 0.0
        self.sampled_frame_count = 0
        self.total_pair_occupancy = 0
        self.open_intervals = {}
        self.pair_intervals = defaultdict(list)
        self.mark_configured()

    def rebuild_runtime_state(self):
        self.dimer_selector.rebuild_runtime_state()
        self.ref_type = self.dimer_selector.ref_type
        self.ref_indices = self.dimer_selector.ref_indices
        self.obs_indices = self.dimer_selector.obs_indices
        self.frame_active = self.ref_indices.size > 0 and self.obs_indices.size > 0

    def post_compound_update(self):
        if not self.dimer_selector.reattach_and_rebuild():
            self.frame_active = False
            self.ref_indices = np.empty(0, dtype=np.int32)
            self.obs_indices = np.empty(0, dtype=np.int32)
            return self.count_missing_as_zero

        self.ref_type = self.dimer_selector.ref_type
        self.ref_indices = self.dimer_selector.ref_indices
        self.obs_indices = self.dimer_selector.obs_indices
        self.frame_active = self.ref_indices.size > 0 and self.obs_indices.size > 0
        return self.frame_active or self.count_missing_as_zero

    def process_frame(self):
        frame_number = self.processed_frames
        active_pairs = set()

        if self.frame_active and self.ref_indices.size > 0 and self.obs_indices.size > 0:
            self.sampled_ref_count_sum += float(self.ref_indices.size)
            self.sampled_obs_count_sum += float(self.obs_indices.size)
            self.sampled_frame_count += 1

            obs_ids_by_ref, _, _ = self.dimer_selector.select_frame()
            for ref_atom_id, obs_ids in zip(self.ref_indices, obs_ids_by_ref):
                for obs_atom_id in obs_ids:
                    active_pairs.add((int(ref_atom_id), int(obs_atom_id)))

        self.total_pair_occupancy += len(active_pairs)

        for pair, start in list(self.open_intervals.items()):
            if pair not in active_pairs:
                self.pair_intervals[pair].append((start, frame_number))
                del self.open_intervals[pair]

        for pair in active_pairs:
            if pair not in self.open_intervals:
                self.open_intervals[pair] = frame_number

    def postprocess(self):
        total_frames = self.processed_frames
        if total_frames <= 0:
            console.warn("No frames were processed.")
            return

        for pair, start in self.open_intervals.items():
            self.pair_intervals[pair].append((start, total_frames))
        self.open_intervals.clear()

        if not self.pair_intervals or self.sampled_frame_count == 0:
            console.warn("No dimer pairs were observed.")
            return

        max_tau = min(self.corr_depth, total_frames)
        dacf = np.zeros(max_tau, dtype=np.float64)
        n_ref = self.sampled_ref_count_sum / self.sampled_frame_count
        n_obs = self.sampled_obs_count_sum / self.sampled_frame_count

        if self.use_continuous:
            for intervals in self.pair_intervals.values():
                for start, end in intervals:
                    length = end - start
                    if length > 0:
                        local_max = min(max_tau, length)
                        dacf[:local_max] += length - np.arange(local_max, dtype=np.float64)
            dacf /= n_ref * n_obs * total_frames
        else:
            for intervals in self.pair_intervals.values():
                occupancy = np.zeros(total_frames, dtype=np.float64)
                for start, end in intervals:
                    occupancy[start:end] = 1.0
                dacf += np.correlate(occupancy, occupancy, mode="full")[total_frames - 1:total_frames - 1 + max_tau]

            dacf /= n_ref * n_obs * np.arange(total_frames, total_frames - max_tau, -1, dtype=np.float64)
            if self.apply_correction:
                avg_beta = self.total_pair_occupancy / (n_ref * n_obs * total_frames)
                dacf -= avg_beta**2

        if dacf[0] != 0:
            dacf /= dacf[0]

        filename = build_output_filename(
            "dacf",
            [
                format_selection(self.selector_spec.ref_labels, self.ref_type.formula),
                format_selection_group(self.dimer_selector.observed_selection_entries),
            ],
        )
        rows = [(tau * self.frame_time_fs / 1000.0, value) for tau, value in enumerate(dacf)]
        write_table(filename, headers=["tau/ps", "DACF"], data=rows)
        console.success(f"Saved DACF results to {filename}")
