#!/usr/bin/env python3
"""
Cognitive Neuroscience Knapsack + 3D Brain Simulator
====================================================

Modes:
    python app.py --mode knapsack
    python app.py --mode brain3d
    python app.py --mode both

Install:
    pip install numpy plotly nibabel scikit-image scipy
"""

from __future__ import annotations

import argparse
import csv
import heapq
import math
import os
import queue
import shutil
import threading
import time
import tkinter as tk
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from tkinter import filedialog, messagebox, ttk

# Optional visualization / MRI stack
try:
    import nibabel as nib
    from scipy import ndimage
    from scipy.spatial import cKDTree
    from skimage import measure
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
    MRI_AVAILABLE = True
except ImportError:
    nib = None
    ndimage = None
    cKDTree = None
    measure = None
    go = None
    PLOTLY_AVAILABLE = False
    MRI_AVAILABLE = False


# =============================================================================
# Knapsack / Subset-Sum
# =============================================================================

@dataclass
class KnapsackItem:
    index: int
    value: float
    weight: float


@dataclass
class KnapsackInstance:
    items: List[KnapsackItem]
    capacity: float
    target_sum: Optional[float] = None


@dataclass
class SearchState:
    cost: float
    value: float
    total_weight: float
    item_index: int
    path: Tuple[int, ...]
    conflict: float = 0.0


@dataclass
class SolverResult:
    best_path: Tuple[int, ...]
    best_value: float
    best_weight: float
    best_cost: float
    history: List[dict] = field(default_factory=list)


def generate_knapsack_instance(
    n_items: int = 20,
    capacity: float = 50.0,
    value_range: Tuple[float, float] = (1.0, 10.0),
    weight_range: Tuple[float, float] = (1.0, 10.0),
    target_ratio: Optional[float] = None,
    seed: int = 1,
) -> KnapsackInstance:
    rng = np.random.default_rng(seed)

    values = rng.uniform(
        value_range[0],
        value_range[1],
        size=n_items,
    )

    weights = rng.uniform(
        weight_range[0],
        weight_range[1],
        size=n_items,
    )

    items = [
        KnapsackItem(
            index=i,
            value=float(values[i]),
            weight=float(weights[i]),
        )
        for i in range(n_items)
    ]

    target_sum = None
    if target_ratio is not None:
        target_sum = float(target_ratio) * capacity

    return KnapsackInstance(
        items=items,
        capacity=float(capacity),
        target_sum=target_sum,
    )


def bounded_beam_search(
    instance: KnapsackInstance,
    beam_width: int = 8,
    max_depth: Optional[int] = None,
    seed: int = 2,
) -> SolverResult:
    items = instance.items
    capacity = instance.capacity
    target_sum = instance.target_sum

    if max_depth is None:
        max_depth = len(items)

    initial_state = SearchState(
        cost=(
            0.0
            if target_sum is None
            else abs(target_sum)
        ),
        value=0.0,
        total_weight=0.0,
        item_index=0,
        path=(),
    )

    beam = [initial_state]
    history = []
    best_state = initial_state

    def state_key(s: SearchState) -> float:
        return (
            s.cost
            - 0.01 * s.value
            + 1e-6 * s.total_weight
        )

    for depth in range(max_depth):
        if depth >= len(items):
            break

        next_beam = []

        for state in beam:
            if state.item_index >= len(items):
                continue

            item = items[state.item_index]

            # Skip item
            skip_cost = (
                0.0
                if target_sum is None
                else abs(target_sum - state.total_weight)
            )
            next_beam.append(
                SearchState(
                    cost=skip_cost,
                    value=state.value,
                    total_weight=state.total_weight,
                    item_index=state.item_index + 1,
                    path=state.path,
                )
            )

            # Take item
            new_weight = state.total_weight + item.weight
            if new_weight <= capacity + 1e-12:
                take_cost = (
                    0.0
                    if target_sum is None
                    else abs(target_sum - new_weight)
                )
                next_beam.append(
                    SearchState(
                        cost=take_cost,
                        value=state.value + item.value,
                        total_weight=new_weight,
                        item_index=state.item_index + 1,
                        path=state.path + (item.index,),
                    )
                )

        next_beam.sort(key=state_key)
        beam = next_beam[:beam_width]

        for s in beam:
            if target_sum is None:
                better = (
                    s.value > best_state.value
                    or (
                        abs(s.value - best_state.value) < 1e-12
                        and s.total_weight < best_state.total_weight
                    )
                )
            else:
                better = (
                    s.cost < best_state.cost
                    or (
                        abs(s.cost - best_state.cost) < 1e-12
                        and s.value > best_state.value
                    )
                )
            if better:
                best_state = s

        best = beam[0] if beam else best_state
        remaining_items = len(items) - depth - 1

        history.append(
            {
                "depth": depth,
                "beam_width_actual": len(beam),
                "best_cost": best.cost,
                "best_value": best.value,
                "best_weight": best.total_weight,
                "remaining_items": remaining_items,
                "beam_states": [
                    {
                        "cost": s.cost,
                        "value": s.value,
                        "weight": s.total_weight,
                        "path_len": len(s.path),
                    }
                    for s in beam
                ],
            }
        )

    return SolverResult(
        best_path=best_state.path,
        best_value=best_state.value,
        best_weight=best_state.total_weight,
        best_cost=(
            best_state.cost
            if target_sum is not None
            else 0.0
        ),
        history=history,
    )


# =============================================================================
# Continuous Cognitive Drive
# =============================================================================

def cognitive_drive_from_solver_state(
    *,
    target_sum: Optional[float],
    current_sum: float,
    current_value: float,
    max_possible_value: float,
    remaining_items: int,
    beam_width: int,
    beam_states: Sequence[dict],
    exact_match: bool = False,
) -> np.ndarray:
    if target_sum is None:
        value_gap = 1.0 - (
            current_value /
            max(max_possible_value, 1e-12)
        )

        difficulty = np.clip(
            0.40 * np.log1p(remaining_items)
            + 0.40 * np.log1p(beam_width)
            + 0.20 * value_gap,
            0.0,
            1.0,
        )

        conflict = np.clip(
            0.5 * value_gap
            + 0.5 * (beam_width / 32.0),
            0.0,
            1.0,
        )
    else:
        error = abs(target_sum - current_sum)

        normalized_error = (
            error /
            max(abs(target_sum), 1e-6)
        )

        difficulty = np.clip(
            0.35 * np.log1p(remaining_items)
            + 0.40 * np.log1p(beam_width)
            + 0.25 * np.log1p(error),
            0.0,
            1.0,
        )

        conflict = np.clip(
            normalized_error,
            0.0,
            1.0,
        )

    working_memory_load = np.clip(
        beam_width / 32.0,
        0.0,
        1.0,
    )

    value_signal = np.clip(
        current_value /
        max(max_possible_value, 1e-12),
        0.0,
        1.0,
    )

    selection_signal = (
        1.0
        if exact_match
        else 0.0
    )

    if len(beam_states) > 1:
        costs = np.array(
            [s["cost"] for s in beam_states],
            dtype=float,
        )

        if target_sum is not None:
            neg_costs = -costs
            exp_neg = np.exp(
                neg_costs - neg_costs.max()
            )
            probs = (
                exp_neg /
                max(exp_neg.sum(), 1e-12)
            )
        else:
            values_arr = np.array(
                [s["value"] for s in beam_states],
                dtype=float,
            )
            exp_v = np.exp(
                values_arr - values_arr.max()
            )
            probs = (
                exp_v /
                max(exp_v.sum(), 1e-12)
            )

        entropy = -np.sum(
            probs *
            np.log(probs + 1e-12)
        )
        max_entropy = np.log(
            len(beam_states)
        )
        beam_entropy_norm = (
            entropy /
            max(max_entropy, 1e-12)
        )
    else:
        beam_entropy_norm = 0.0

    drive = np.array(
        [
            0.70 * difficulty
            + 0.30 * conflict
            + 0.20 * beam_entropy_norm,

            0.80 * difficulty
            + 0.20 * conflict,

            0.60 * working_memory_load
            + 0.40 * difficulty,

            0.70 * working_memory_load
            + 0.30 * conflict,

            0.60 * value_signal
            + 0.20 * (1.0 - conflict),

            0.90 * selection_signal,
        ],
        dtype=float,
    )

    return np.clip(
        drive,
        0.0,
        1.0,
    )


# =============================================================================
# Wilson-Cowan Network
# =============================================================================

@dataclass
class WilsonCowanParams:
    tau_e: float = 0.040
    tau_i: float = 0.080

    w_ee: float = 12.0
    w_ei: float = 10.0
    w_ie: float = 10.0
    w_ii: float = 2.0

    gain_e: float = 1.3
    gain_i: float = 1.0

    threshold_e: float = 2.5
    threshold_i: float = 2.0


def sigmoid(
    x: np.ndarray,
    gain: float,
    threshold: float,
) -> np.ndarray:
    x = np.clip(
        x,
        -60.0,
        60.0,
    )
    return 1.0 / (
        1.0
        + np.exp(
            -gain * (x - threshold)
        )
    )


class CognitiveElectricalNetwork:
    def __init__(
        self,
        connectivity: Optional[np.ndarray] = None,
        params: Optional[WilsonCowanParams] = None,
        seed: int = 1,
    ):
        self.rng = np.random.default_rng(seed)
        self.params = params or WilsonCowanParams()

        self.names = [
            "dACC",
            "Anterior Insula",
            "Intraparietal Sulcus",
            "DLPFC",
            "OFC/vmPFC",
            "Motor/Report",
        ]

        n = len(self.names)

        if connectivity is None:
            connectivity = np.array(
                [
                    [0.00, 0.30, 0.40, 0.50, 0.20, 0.10],
                    [0.30, 0.00, 0.40, 0.30, 0.20, 0.10],
                    [0.40, 0.40, 0.00, 0.60, 0.30, 0.10],
                    [0.50, 0.30, 0.60, 0.00, 0.50, 0.20],
                    [0.20, 0.20, 0.30, 0.50, 0.00, 0.30],
                    [0.10, 0.10, 0.10, 0.20, 0.30, 0.00],
                ],
                dtype=float,
            )

        self.C = np.asarray(
            connectivity,
            dtype=float,
        )
        self.C /= max(
            self.C.max(),
            1e-12,
        )

        self.E = np.full(
            n,
            0.05,
            dtype=float,
        )
        self.I = np.full(
            n,
            0.05,
            dtype=float,
        )

    def reset(self):
        self.E.fill(0.05)
        self.I.fill(0.05)

    def step(
        self,
        external_input: np.ndarray,
        dt: float = 0.001,
        coupling: float = 0.8,
    ) -> Tuple[np.ndarray, np.ndarray]:
        p = self.params

        external_input = np.asarray(
            external_input,
            dtype=float,
        )

        net_excitation = (
            coupling *
            (self.C @ self.E)
        )

        e_drive = (
            p.w_ee * self.E
            - p.w_ei * self.I
            + net_excitation
            + external_input
        )

        i_drive = (
            p.w_ie * self.E
            - p.w_ii * self.I
        )

        target_e = sigmoid(
            e_drive,
            p.gain_e,
            p.threshold_e,
        )

        target_i = sigmoid(
            i_drive,
            p.gain_i,
            p.threshold_i,
        )

        self.E += (
            dt /
            p.tau_e
            * (-self.E + target_e)
        )

        self.I += (
            dt /
            p.tau_i
            * (-self.I + target_i)
        )

        self.E = np.clip(
            self.E,
            0.0,
            1.0,
        )

        self.I = np.clip(
            self.I,
            0.0,
            1.0,
        )

        return (
            self.E.copy(),
            self.I.copy(),
        )


# =============================================================================
# Persistent oscillatory simulation
# =============================================================================

def simulate_continuous(
    network: CognitiveElectricalNetwork,
    drive_time_series: np.ndarray,
    sample_rate: int = 500,
    coupling: float = 0.8,
    oscillation_strength: float = 0.32,
    noise_strength: float = 0.025,
    seed: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    drive_time_series = np.asarray(
        drive_time_series,
        dtype=float,
    )

    total_steps = drive_time_series.shape[0]
    n_regions = len(network.names)

    e_trace = np.zeros(
        (total_steps, n_regions),
        dtype=float,
    )
    i_trace = np.zeros_like(e_trace)

    network.reset()
    rng = np.random.default_rng(seed)
    dt = 1.0 / sample_rate

    frequencies = np.array(
        [
            6.0,  # dACC
            8.0,  # Anterior Insula
            10.0,  # IPS
            10.0,  # DLPFC
            7.0,  # OFC/vmPFC
            20.0,  # Motor/Report
        ],
        dtype=float,
    )

    phases = rng.uniform(
        0.0,
        2.0 * np.pi,
        size=n_regions,
    )

    secondary_frequencies = frequencies * 1.63
    secondary_phases = rng.uniform(
        0.0,
        2.0 * np.pi,
        size=n_regions,
    )

    envelope_state = np.zeros(
        n_regions,
        dtype=float,
    )

    noise_state = np.zeros(
        n_regions,
        dtype=float,
    )

    for t in range(total_steps):
        current_time = t / sample_rate
        solver_drive = drive_time_series[t]

        cognitive_load = np.clip(
            0.45 * solver_drive
            + 0.55 * np.mean(solver_drive),
            0.0,
            1.0,
        )

        target_envelope = 0.45 + 0.70 * cognitive_load
        envelope_state += (
            dt / 0.35
            * (target_envelope - envelope_state)
        )

        primary = np.sin(
            2.0 * np.pi * frequencies * current_time + phases
        )

        secondary = np.sin(
            2.0 * np.pi * secondary_frequencies * current_time + secondary_phases
        )

        slow_modulation = (
            0.5
            + 0.5 * np.sin(2.0 * np.pi * 0.35 * current_time)
        )

        white_noise = rng.normal(
            0.0,
            1.0,
            size=n_regions,
        )

        noise_state += (
            dt / 0.035
            * (white_noise - noise_state)
        )

        network_wave = network.C @ primary
        network_norm = (
            network_wave /
            max(np.max(np.abs(network_wave)), 1e-12)
        )

        oscillation = (
            oscillation_strength
            * envelope_state
            * (
                0.72 * primary
                + 0.18 * secondary
                + 0.10 * network_norm
            )
            * (0.70 + 0.30 * slow_modulation)
        )

        stochastic = noise_strength * noise_state

        neural_input = (
            2.25 * solver_drive
            + oscillation
            + stochastic
        )

        tonic_bias = np.array(
            [0.10, 0.08, 0.07, 0.09, 0.06, 0.05],
            dtype=float,
        )
        neural_input += tonic_bias

        e, i = network.step(
            external_input=neural_input,
            dt=dt,
            coupling=coupling,
        )

        e_trace[t] = e
        i_trace[t] = i

    return e_trace, i_trace


# =============================================================================
# Continuous Task + Network Simulation
# =============================================================================

def simulate_task_with_network_continuous(
    instance: KnapsackInstance,
    network: CognitiveElectricalNetwork,
    beam_width: int = 8,
    max_depth: Optional[int] = None,
    trial_duration: float = 20.0,
    sample_rate: int = 500,
    seed: int = 3,
) -> Tuple[SolverResult, dict]:
    result = bounded_beam_search(
        instance=instance,
        beam_width=beam_width,
        max_depth=max_depth,
        seed=seed,
    )

    max_possible_value = sum(
        item.value
        for item in instance.items
    )

    n_steps = len(result.history)
    if n_steps == 0:
        raise RuntimeError("Solver produced no history.")

    drive_matrix = np.zeros(
        (n_steps, 6),
        dtype=float,
    )

    for k, step_data in enumerate(result.history):
        beam_states = step_data["beam_states"]
        remaining_items = step_data["remaining_items"]
        best_step = beam_states[0] if beam_states else {
            "cost": 0.0,
            "value": 0.0,
            "weight": 0.0,
            "path_len": 0,
        }

        exact_match = (
            instance.target_sum is not None
            and abs(best_step["cost"]) < 1e-6
        )

        drive = cognitive_drive_from_solver_state(
            target_sum=instance.target_sum,
            current_sum=best_step["weight"],
            current_value=best_step["value"],
            max_possible_value=max_possible_value,
            remaining_items=remaining_items,
            beam_width=beam_width,
            beam_states=beam_states,
            exact_match=exact_match,
        )

        drive_matrix[k] = drive

    total_steps = max(
        2,
        int(trial_duration * sample_rate),
    )

    time_discrete = np.linspace(
        0.0,
        trial_duration,
        n_steps,
    )
    time_continuous = np.arange(
        total_steps,
        dtype=float,
    ) / sample_rate
    time_continuous = np.clip(
        time_continuous,
        0.0,
        trial_duration,
    )

    drive_continuous = np.zeros(
        (total_steps, 6),
        dtype=float,
    )

    for i in range(6):
        drive_continuous[:, i] = np.interp(
            time_continuous,
            time_discrete,
            drive_matrix[:, i],
        )

    network.reset()
    e_trace, i_trace = simulate_continuous(
        network=network,
        drive_time_series=drive_continuous,
        sample_rate=sample_rate,
        coupling=0.8,
        oscillation_strength=0.32,
        noise_strength=0.025,
        seed=seed + 100,
    )

    eeg_weights = np.array(
        [0.20, 0.15, 0.20, 0.25, 0.15, 0.05],
        dtype=float,
    )
    eeg_weights /= eeg_weights.sum()

    virtual_eeg_raw = e_trace @ eeg_weights
    virtual_eeg = virtual_eeg_raw - np.mean(virtual_eeg_raw)
    eeg_std = np.std(virtual_eeg)

    if eeg_std > 1e-12:
        virtual_eeg_normalized = virtual_eeg / eeg_std
    else:
        virtual_eeg_normalized = virtual_eeg.copy()

    neural_data = {
        "e_trace": e_trace,
        "i_trace": i_trace,
        "virtual_eeg": virtual_eeg,
        "virtual_eeg_normalized": virtual_eeg_normalized,
        "drive_continuous": drive_continuous,
        "sample_rate": sample_rate,
        "region_names": network.names,
        "solver_history": result.history,
        "trial_duration": trial_duration,
        "oscillation_strength": 0.32,
        "noise_strength": 0.025,
    }

    return result, neural_data


# =============================================================================
# CSV and NumPy outputs
# =============================================================================

def save_experiment_csv(
    filename: str,
    instance: KnapsackInstance,
    result: SolverResult,
    neural_data: dict,
) -> None:
    e_trace = neural_data["e_trace"]
    virtual_eeg = neural_data["virtual_eeg"]
    sample_rate = neural_data["sample_rate"]
    history = neural_data["solver_history"]
    trial_duration = neural_data["trial_duration"]

    n_steps = e_trace.shape[0]
    time_vec = np.arange(n_steps) / sample_rate
    n_depths = len(history)

    depth_per_step = np.clip(
        (time_vec / max(trial_duration, 1e-12) * n_depths).astype(int),
        0,
        n_depths - 1,
    )

    fieldnames = [
        "time_s",
        "depth",
        "best_cost",
        "best_value",
        "best_weight",
        "beam_width_actual",
        "dACC_E",
        "insula_E",
        "ips_E",
        "dlpfc_E",
        "ofc_E",
        "motor_E",
        "virtual_eeg",
    ]

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for t_idx in range(n_steps):
            depth = int(depth_per_step[t_idx])
            step_data = history[depth]

            writer.writerow(
                {
                    "time_s": time_vec[t_idx],
                    "depth": depth,
                    "best_cost": step_data["best_cost"],
                    "best_value": step_data["best_value"],
                    "best_weight": step_data["best_weight"],
                    "beam_width_actual": step_data["beam_width_actual"],
                    "dACC_E": e_trace[t_idx, 0],
                    "insula_E": e_trace[t_idx, 1],
                    "ips_E": e_trace[t_idx, 2],
                    "dlpfc_E": e_trace[t_idx, 3],
                    "ofc_E": e_trace[t_idx, 4],
                    "motor_E": e_trace[t_idx, 5],
                    "virtual_eeg": virtual_eeg[t_idx],
                }
            )


def save_numpy_arrays(
    base_filename: str,
    neural_data: dict,
) -> None:
    np.save(
        base_filename + "_e_trace.npy",
        neural_data["e_trace"],
    )
    np.save(
        base_filename + "_i_trace.npy",
        neural_data["i_trace"],
    )
    np.save(
        base_filename + "_virtual_eeg.npy",
        neural_data["virtual_eeg"],
    )
    np.save(
        base_filename + "_virtual_eeg_normalized.npy",
        neural_data["virtual_eeg_normalized"],
    )
    np.save(
        base_filename + "_drive_continuous.npy",
        neural_data["drive_continuous"],
    )


# =============================================================================
# Plotly EEG
# =============================================================================

def plot_neural_activity(
    neural_data: dict,
    output_html: str = "neural_activity.html",
) -> None:
    if not PLOTLY_AVAILABLE:
        raise RuntimeError("Plotly is not installed.")

    e_trace = neural_data["e_trace"]
    virtual_eeg = neural_data["virtual_eeg"]
    sample_rate = neural_data["sample_rate"]
    region_names = neural_data["region_names"]
    trial_duration = neural_data["trial_duration"]

    n_steps = e_trace.shape[0]
    time_vec = np.arange(n_steps) / sample_rate

    traces = []
    for i, name in enumerate(region_names):
        traces.append(
            go.Scatter(
                x=time_vec,
                y=e_trace[:, i],
                mode="lines",
                name=f"{name} (E)",
            )
        )

    traces.append(
        go.Scatter(
            x=time_vec,
            y=virtual_eeg,
            mode="lines",
            name="Virtual EEG",
            line=dict(
                width=2,
                color="black",
            ),
        )
    )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Continuous Simulated Brain Activity + Virtual EEG",
        xaxis_title="Time (s)",
        yaxis_title="Activity",
        legend_title="Signal",
        height=650,
        xaxis=dict(
            range=[0, trial_duration],
        ),
        hovermode="x unified",
    )

    fig.write_html(
        output_html,
        include_plotlyjs=True,
    )
    webbrowser.open(
        "file://"
        + os.path.realpath(output_html)
    )


# =============================================================================
# File manifest with descriptive labels
# =============================================================================

def required_file_manifest(base_dir: Path):
    """
    Return a list of (descriptive_label, filename, purpose, required_from_user).
    """
    return [
        (
            "Beam-search weight data",
            base_dir / "weights.bin",
            "Binary float32 weights used by the subset-sum beam search.",
            False,
        ),
        (
            "T1-weighted MRI volume",
            base_dir / "real_brain_mri_t1.nii.gz",
            "MRI volume used to generate the anatomical brain surface.",
            True,
        ),
        (
            "Cached brain-surface vertices",
            base_dir / "brain_surface_verts.npy",
            "Generated 3D mesh vertex coordinates.",
            False,
        ),
        (
            "Cached brain-surface triangles",
            base_dir / "brain_surface_faces.npy",
            "Generated 3D mesh triangle indices.",
            False,
        ),
        (
            "Cached internal brain points",
            base_dir / "brain_inner_points.npy",
            "Generated points used for internal neural circuit paths.",
            False,
        ),
    ]


# =============================================================================
# Knapsack GUI
# =============================================================================

class KnapsackNeuroApp:
    def __init__(
        self,
        root: tk.Tk,
        master: Optional[tk.Widget] = None,
    ):
        self.root = root
        self.master = master if master is not None else root
        self.master.title("Knapsack Cognitive-Neuro Simulator")
        self.master.geometry("900x700")
        self.master.minsize(700, 550)

        self.is_running = False
        self.result_queue = queue.Queue()

        self.latest_result = None
        self.latest_instance = None
        self.latest_neural_data = None

        self._create_widgets()

    def _create_widgets(self):
        main_frame = ttk.Frame(self.master, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            main_frame,
            text="Knapsack Cognitive-Neuro Simulator — Continuous EEG",
            font=("Arial", 14, "bold"),
        ).pack(pady=(0, 10))

        task_frame = ttk.LabelFrame(
            main_frame,
            text="Task Parameters",
            padding=10,
        )
        task_frame.pack(fill=tk.X, pady=5)

        ttk.Label(
            task_frame,
            text="Number of items:",
        ).grid(row=0, column=0, sticky="w", pady=5)
        self.n_items_entry = ttk.Entry(task_frame, width=12)
        self.n_items_entry.insert(0, "20")
        self.n_items_entry.grid(row=0, column=1, padx=10, pady=5)

        ttk.Label(
            task_frame,
            text="Capacity:",
        ).grid(row=1, column=0, sticky="w", pady=5)
        self.capacity_entry = ttk.Entry(task_frame, width=12)
        self.capacity_entry.insert(0, "50.0")
        self.capacity_entry.grid(row=1, column=1, padx=10, pady=5)

        ttk.Label(
            task_frame,
            text="Target ratio:",
        ).grid(row=2, column=0, sticky="w", pady=5)
        self.target_ratio_entry = ttk.Entry(task_frame, width=12)
        self.target_ratio_entry.insert(0, "0.7")
        self.target_ratio_entry.grid(row=2, column=1, padx=10, pady=5)

        solver_frame = ttk.LabelFrame(
            main_frame,
            text="Solver + Neural Parameters",
            padding=10,
        )
        solver_frame.pack(fill=tk.X, pady=5)

        ttk.Label(
            solver_frame,
            text="Beam width:",
        ).grid(row=0, column=0, sticky="w", pady=5)
        self.beam_width_entry = ttk.Entry(solver_frame, width=12)
        self.beam_width_entry.insert(0, "8")
        self.beam_width_entry.grid(row=0, column=1, padx=10, pady=5)

        ttk.Label(
            solver_frame,
            text="Trial duration (s):",
        ).grid(row=1, column=0, sticky="w", pady=5)
        self.trial_duration_entry = ttk.Entry(solver_frame, width=12)
        self.trial_duration_entry.insert(0, "20.0")
        self.trial_duration_entry.grid(row=1, column=1, padx=10, pady=5)

        ttk.Label(
            solver_frame,
            text="Sample rate (Hz):",
        ).grid(row=2, column=0, sticky="w", pady=5)
        self.sample_rate_entry = ttk.Entry(solver_frame, width=12)
        self.sample_rate_entry.insert(0, "500")
        self.sample_rate_entry.grid(row=2, column=1, padx=10, pady=5)

        self.run_button = ttk.Button(
            solver_frame,
            text="Run Simulation",
            command=self.start_simulation,
        )
        self.run_button.grid(
            row=0,
            column=2,
            rowspan=3,
            padx=15,
            ipadx=5,
            ipady=10,
        )

        self.plot_button = ttk.Button(
            solver_frame,
            text="Plot Neural Activity",
            command=self.show_plot,
            state=tk.DISABLED,
        )
        self.plot_button.grid(
            row=0,
            column=3,
            rowspan=3,
            padx=10,
            ipadx=5,
            ipady=10,
        )

        self.status_var = tk.StringVar(value="Status: Ready")
        ttk.Label(
            main_frame,
            textvariable=self.status_var,
            font=("Arial", 10, "italic"),
        ).pack(anchor="w", pady=(10, 5))

        results_frame = ttk.LabelFrame(
            main_frame,
            text="Solver + Neural Summary",
            padding=10,
        )
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.summary_text = tk.Text(
            results_frame,
            height=12,
            wrap=tk.WORD,
            font=("Consolas", 10),
        )
        self.summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(
            results_frame,
            orient=tk.VERTICAL,
            command=self.summary_text.yview,
        )
        self.summary_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _worker_loop(self):
        try:
            n_items = int(self.n_items_entry.get())
            capacity = float(self.capacity_entry.get())
            target_ratio_str = self.target_ratio_entry.get().strip()
            beam_width = int(self.beam_width_entry.get())
            trial_duration = float(self.trial_duration_entry.get())
            sample_rate = int(self.sample_rate_entry.get())

            target_ratio = float(target_ratio_str) if target_ratio_str else None

            instance = generate_knapsack_instance(
                n_items=n_items,
                capacity=capacity,
                target_ratio=target_ratio,
                seed=7,
            )

            network = CognitiveElectricalNetwork(seed=11)

            result, neural_data = simulate_task_with_network_continuous(
                instance=instance,
                network=network,
                beam_width=beam_width,
                trial_duration=trial_duration,
                sample_rate=sample_rate,
                seed=13,
            )

            base_name = f"experiment_n{n_items}_bw{beam_width}"

            save_experiment_csv(
                base_name + ".csv",
                instance,
                result,
                neural_data,
            )
            save_numpy_arrays(
                base_name,
                neural_data,
            )

            self.result_queue.put(("SUCCESS", (instance, result, neural_data)))
        except Exception as e:
            self.result_queue.put(("ERROR", str(e)))

    def start_simulation(self):
        if self.is_running:
            return

        try:
            n_items = int(self.n_items_entry.get())
            capacity = float(self.capacity_entry.get())
            beam_width = int(self.beam_width_entry.get())
            trial_duration = float(self.trial_duration_entry.get())
            sample_rate = int(self.sample_rate_entry.get())

            if n_items <= 0 or capacity <= 0 or beam_width <= 0:
                raise ValueError("Parameters must be positive.")
            if trial_duration <= 0 or sample_rate <= 0:
                raise ValueError("Duration and sample rate must be positive.")
        except ValueError as err:
            messagebox.showerror(
                "Invalid Input",
                "Please enter valid numeric parameters.\n\n"
                f"Details: {err}",
            )
            return

        self.is_running = True
        self.run_button.config(state=tk.DISABLED)
        self.plot_button.config(state=tk.DISABLED)
        self.status_var.set(
            "Status: Running continuous solver + neural simulation..."
        )

        worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
        )
        worker_thread.start()

        self.root.after(100, self._check_queue)

    def _check_queue(self):
        try:
            status, data = self.result_queue.get_nowait()
        except queue.Empty:
            if self.is_running:
                self.root.after(100, self._check_queue)
            return

        self.is_running = False
        self.run_button.config(state=tk.NORMAL)

        if status == "SUCCESS":
            instance, result, neural_data = data
            self.latest_instance = instance
            self.latest_result = (result, neural_data)
            self.latest_neural_data = neural_data

            self.status_var.set(
                "Status: Simulation complete — persistent oscillations generated."
            )
            self._update_summary(instance, result, neural_data)

            if PLOTLY_AVAILABLE:
                self.plot_button.config(state=tk.NORMAL)
            else:
                self.status_var.set(
                    "Status: Complete. Install Plotly for visualization."
                )
        else:
            self.status_var.set("Status: Simulation failed.")
            messagebox.showerror("Execution Error", data)

    def _update_summary(self, instance, result, neural_data):
        self.summary_text.delete("1.0", tk.END)

        lines = [
            "Task type: "
            + ("Subset-sum knapsack" if instance.target_sum is not None else "0-1 knapsack"),
            f"Number of items: {len(instance.items)}",
            f"Capacity: {instance.capacity:.3f}",
            f"Target sum: {instance.target_sum}",
            "",
            "Best solution:",
            f" Selected indices: {result.best_path}",
            f" Total value: {result.best_value:.4f}",
            f" Total weight: {result.best_weight:.4f}",
            f" Cost: {result.best_cost:.4f}",
            "",
            "Continuous neural simulation:",
            f" Regions: {', '.join(neural_data['region_names'])}",
            f" E-trace shape: {neural_data['e_trace'].shape}",
            f" EEG samples: {len(neural_data['virtual_eeg'])}",
            f" Trial duration: {neural_data['trial_duration']:.2f} s",
            "",
            "Persistent oscillation:",
            f" Strength: {neural_data['oscillation_strength']:.3f}",
            f" Noise: {neural_data['noise_strength']:.3f}",
            " Status: ACTIVE",
            "",
            "Outputs:",
            " - CSV and .npy files written to the current directory",
            "   (see experiment_n*_bw*.csv and *_e_trace.npy, etc.)",
        ]

        self.summary_text.insert(tk.END, "\n".join(lines))

    def show_plot(self):
        if self.latest_neural_data is None:
            messagebox.showinfo("No Results", "Run the simulation first.")
            return

        try:
            plot_neural_activity(
                self.latest_neural_data,
                "neural_activity.html",
            )
            self.status_var.set("Status: Opened neural_activity.html")
        except Exception as e:
            messagebox.showerror("Plot Error", str(e))


# =============================================================================
# 3D Brain / Circuit Visualizer
# =============================================================================

class NeuralBeamSimulationApp:
    def __init__(
        self,
        root: tk.Tk,
        master: Optional[tk.Widget] = None,
    ):
        self.root = root
        self.master = master if master is not None else root

        if master is None:
            self.master.title("Neural Beam Engine + 3D Brain")
            self.master.geometry("800x650")
            self.master.minsize(700, 550)

        self.base_dir = Path.cwd()

        self.data_path = self.base_dir / "weights.bin"
        self.mri_path = self.base_dir / "real_brain_mri_t1.nii.gz"
        self.surface_verts_path = self.base_dir / "brain_surface_verts.npy"
        self.surface_faces_path = self.base_dir / "brain_surface_faces.npy"
        self.inner_points_path = self.base_dir / "brain_inner_points.npy"

        self.is_running = False
        self.result_queue = queue.Queue()
        self.latest_results = []

        self._ensure_data_files()
        self._create_widgets()

    def _create_widgets(self):
        main_frame = ttk.Frame(self.master, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            main_frame,
            text="Neural Beam Engine with 3D Brain Circuits",
            font=("Arial", 14, "bold"),
        ).pack(pady=(0, 10))

        # File manifest panel
        file_frame = ttk.LabelFrame(
            main_frame,
            text="Input and Generated Files",
            padding=10,
        )
        file_frame.pack(fill=tk.X, pady=5)

        columns = ("label", "filename", "status", "purpose")
        tree = ttk.Treeview(
            file_frame,
            columns=columns,
            show="headings",
            height=5,
        )

        headings = {
            "label": "Descriptive label",
            "filename": "Filename",
            "status": "Status",
            "purpose": "Purpose",
        }
        widths = {
            "label": 190,
            "filename": 230,
            "status": 140,
            "purpose": 380,
        }

        for column in columns:
            tree.heading(column, text=headings[column])
            tree.column(column, width=widths[column], anchor="w")

        for label, path, purpose, user_required in required_file_manifest(self.base_dir):
            exists = path.exists()
            if exists:
                status = "Available"
            elif user_required:
                status = "MISSING"
            else:
                status = "Generated when needed"

            tree.insert(
                "",
                tk.END,
                values=(label, path.name, status, purpose),
            )

        tree.pack(fill=tk.X, expand=True)
        self.file_status_tree = tree

        ttk.Button(
            file_frame,
            text="Select T1 MRI File...",
            command=self.select_mri,
        ).pack(anchor="e", pady=(6, 0))

        # Simulation controls
        control_frame = ttk.LabelFrame(
            main_frame,
            text="Simulation Parameters",
            padding=10,
        )
        control_frame.pack(fill=tk.X, pady=5)

        ttk.Label(
            control_frame,
            text="Target Sum:",
        ).grid(row=0, column=0, sticky="w", pady=5)
        self.target_entry = ttk.Entry(control_frame, width=12)
        self.target_entry.insert(0, "7.0")
        self.target_entry.grid(row=0, column=1, padx=10, pady=5)

        ttk.Label(
            control_frame,
            text="Beam Width:",
        ).grid(row=1, column=0, sticky="w", pady=5)
        self.beam_entry = ttk.Entry(control_frame, width=12)
        self.beam_entry.insert(0, "4")
        self.beam_entry.grid(row=1, column=1, padx=10, pady=5)

        self.run_button = ttk.Button(
            control_frame,
            text="Run Simulation",
            command=self.start_simulation,
        )
        self.run_button.grid(
            row=0,
            column=2,
            rowspan=2,
            padx=15,
            ipadx=5,
            ipady=10,
        )

        self.mri_button = ttk.Button(
            control_frame,
            text="Render Circuits + Labels",
            command=self.show_cortical_overlay,
            state=tk.DISABLED,
        )
        self.mri_button.grid(
            row=0,
            column=3,
            rowspan=2,
            padx=10,
            ipadx=5,
            ipady=10,
        )

        self.status_var = tk.StringVar(value="Status: Ready")
        ttk.Label(
            main_frame,
            textvariable=self.status_var,
            font=("Arial", 10, "italic"),
        ).pack(anchor="w", pady=(10, 5))

        results_frame = ttk.LabelFrame(
            main_frame,
            text="Top Beam Paths",
            padding=10,
        )
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        columns = ("rank", "cost", "sum", "path")
        self.tree = ttk.Treeview(
            results_frame,
            columns=columns,
            show="headings",
            height=8,
        )

        self.tree.heading("rank", text="Rank")
        self.tree.heading("cost", text="Cost Error")
        self.tree.heading("sum", text="Achieved Sum")
        self.tree.heading("path", text="Selected Path Weights")

        self.tree.column("rank", width=50, anchor="center")
        self.tree.column("cost", width=90, anchor="center")
        self.tree.column("sum", width=100, anchor="center")
        self.tree.column("path", width=400, anchor="w")

        scrollbar = ttk.Scrollbar(
            results_frame,
            orient=tk.VERTICAL,
            command=self.tree.yview,
        )
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _ensure_data_files(self):
        if not self.data_path.exists():
            raw_weights = np.array(
                [1.5, 2.2, 3.8, 4.1, 5.0, 6.3, 1.1],
                dtype=np.float32,
            )
            raw_weights.tofile(self.data_path)

    def _load_data(self) -> np.ndarray:
        if not self.data_path.exists():
            raise FileNotFoundError(
                "Beam-search weight data is missing:\n"
                f"{self.data_path}\n\n"
                "Expected file label: Beam-search weight data"
            )
        weights = np.fromfile(self.data_path, dtype=np.float32)
        if weights.size == 0:
            raise ValueError(
                "Beam-search weight data is empty:\n"
                f"{self.data_path}"
            )
        return weights

    def bounded_beam_search(
        self,
        target_sum: float,
        beam_width: int,
        weights: np.ndarray,
    ):
        beam = [(0.0, 0.0, ())]
        for w in weights:
            if not self.is_running:
                break
            w_float = float(w)
            next_beam = []
            for cost, current_sum, path in beam:
                next_beam.append((cost, current_sum, path))
                new_sum = current_sum + w_float
                new_cost = abs(float(target_sum) - new_sum)
                next_beam.append((new_cost, new_sum, path + (w_float,)))
            beam = heapq.nsmallest(beam_width, next_beam, key=lambda x: x[0])
            time.sleep(0.001)
        return beam

    def _worker_loop(self, target_sum: float, beam_width: int):
        try:
            weights = self._load_data()
            results = self.bounded_beam_search(target_sum, beam_width, weights)
            self.result_queue.put(("SUCCESS", results))
        except Exception as e:
            self.result_queue.put(("ERROR", str(e)))

    def start_simulation(self):
        if self.is_running:
            return

        try:
            target_sum = float(self.target_entry.get())
            beam_width = int(self.beam_entry.get())
            if beam_width <= 0:
                raise ValueError("Beam width must be greater than zero.")
        except ValueError as err:
            messagebox.showerror("Invalid Input", str(err))
            return

        for item in self.tree.get_children():
            self.tree.delete(item)

        self.is_running = True
        self.run_button.config(state=tk.DISABLED)
        self.mri_button.config(state=tk.DISABLED)
        self.status_var.set("Status: Running subset-sum beam search...")

        threading.Thread(
            target=self._worker_loop,
            args=(target_sum, beam_width),
            daemon=True,
        ).start()

        self.root.after(100, self._check_queue)

    def _check_queue(self):
        try:
            status, data = self.result_queue.get_nowait()
        except queue.Empty:
            if self.is_running:
                self.root.after(100, self._check_queue)
            return

        self.is_running = False
        self.run_button.config(state=tk.NORMAL)

        if status == "SUCCESS":
            self.latest_results = data
            self.status_var.set("Status: Simulation complete.")

            for idx, res in enumerate(data, start=1):
                self.tree.insert(
                    "",
                    tk.END,
                    values=(
                        idx,
                        f"{float(res[0]):.4f}",
                        f"{float(res[1]):.4f}",
                        str(res[2]),
                    ),
                )

            if PLOTLY_AVAILABLE:
                self.mri_button.config(state=tk.NORMAL)
        else:
            self.status_var.set("Status: Simulation failed.")
            messagebox.showerror("Execution Error", data)

    def select_mri(self):
        source = filedialog.askopenfilename(
            title="Select T1-weighted MRI volume",
            filetypes=[("NIfTI MRI", "*.nii *.nii.gz"), ("All files", "*")],
        )
        if not source:
            return
        dest = self.mri_path
        try:
            shutil.copy2(source, dest)
            messagebox.showinfo("MRI selected", f"Saved as:\n{dest}")
            # Refresh file status
            self.master.destroy()
            NeuralBeamSimulationApp(self.root, self.master)
        except Exception as exc:
            messagebox.showerror("MRI copy failed", str(exc))

    # =========================================================================
    # Brain region naming
    # =========================================================================

    def _get_scientific_region_name(self, x: float, y: float, z: float) -> str:
        hemisphere = "Right" if x >= 0 else "Left"
        ax, ay, az = abs(x), y, z

        if ay > 30:
            if az > 20:
                return f"{hemisphere} Sup. Frontal"
            elif az > 0:
                return f"{hemisphere} Mid. Frontal"
            return f"{hemisphere} Orbital Frontal"
        elif 0 <= ay <= 30:
            if az > 35:
                return f"{hemisphere} Precentral / Motor"
            elif az > 10:
                return f"{hemisphere} Supp. Motor"
            elif az < -5:
                return f"{hemisphere} Insular"
            return f"{hemisphere} Ant. Cingulate"
        elif -50 <= ay < 0:
            if az > 40:
                return f"{hemisphere} Postcentral"
            elif az > 20:
                return f"{hemisphere} Inf. Parietal"
            elif az < 0:
                return f"{hemisphere} Sup. Temporal"
            return f"{hemisphere} Post. Cingulate"
        else:
            if az > 10:
                return f"{hemisphere} Precuneus"
            elif az < -10:
                return f"{hemisphere} Fusiform"
            return f"{hemisphere} Occipital Pole"

    def _get_inner_structure_name(self, x: float, y: float, z: float) -> str:
        hemisphere = "Right" if x >= 0 else "Left"
        ax, ay, az = abs(x), y, z

        if ax < 10:
            if ay > 15:
                return "Corpus Callosum (Genu)"
            elif ay > -5:
                return "Thalamus"
            elif ay > -25:
                return "Corpus Callosum (Splenium)"
            return "Brainstem"
        else:
            if az > 10:
                if ay > 0:
                    return f"{hemisphere} Caudate Nucleus"
                return f"{hemisphere} Putamen"
            elif az > -8:
                if ay > -5:
                    return f"{hemisphere} Putamen"
                elif ay > -25:
                    return f"{hemisphere} Amygdala"
                return f"{hemisphere} Hippocampus"
            else:
                if ay < -20:
                    return f"{hemisphere} Cerebellum"
                return f"{hemisphere} Hippocampus"

    # =========================================================================
    # MRI data helpers
    # =========================================================================

    def _build_or_load_inner_points(self) -> np.ndarray:
        if self.inner_points_path.exists():
            return np.load(self.inner_points_path)

        if not self.mri_path.exists():
            raise FileNotFoundError(
                "The T1-weighted MRI volume is missing.\n\n"
                f"Expected file: {self.mri_path.name}\n"
                "Descriptive label: T1-weighted MRI volume\n\n"
                "Place the MRI file beside app.py and retry."
            )

        if not MRI_AVAILABLE:
            raise RuntimeError(
                "MRI processing dependencies are unavailable.\n\n"
                "Install them with:\n"
                "pip install nibabel scipy scikit-image plotly"
            )

        img = nib.load(self.mri_path)
        data = img.get_fdata().astype(np.float32)
        positive = data[data > 0]
        if len(positive) == 0:
            raise RuntimeError("MRI contains no positive voxels.")

        robust_max = np.percentile(positive, 99)
        volume = np.clip(
            data / max(robust_max, 1e-12),
            0,
            1,
        )
        mask = volume > 0.2
        interior_mask = ndimage.binary_erosion(mask, iterations=15)
        coords = np.argwhere(interior_mask).astype(np.float64)
        if len(coords) == 0:
            raise RuntimeError("Could not generate interior points.")

        rng = np.random.default_rng(0)
        n_sample = min(60000, len(coords))
        sample = coords[rng.choice(len(coords), size=n_sample, replace=False)]
        sample -= np.array(data.shape) / 2.0
        np.save(self.inner_points_path, sample)
        return sample

    def _build_or_load_brain_surface(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.surface_verts_path.exists() and self.surface_faces_path.exists():
            return (
                np.load(self.surface_verts_path),
                np.load(self.surface_faces_path),
            )

        if not self.mri_path.exists():
            raise FileNotFoundError(
                "The T1-weighted MRI volume is missing.\n\n"
                f"Expected file: {self.mri_path.name}\n"
                "Descriptive label: T1-weighted MRI volume\n\n"
                "Place the MRI file beside app.py and retry."
            )

        if not MRI_AVAILABLE:
            raise RuntimeError(
                "MRI processing dependencies are unavailable.\n\n"
                "Install them with:\n"
                "pip install nibabel scipy scikit-image plotly"
            )

        img = nib.load(self.mri_path)
        data = img.get_fdata().astype(np.float32)
        positive = data[data > 0]
        if len(positive) == 0:
            raise RuntimeError("MRI contains no positive voxels.")

        robust_max = np.percentile(positive, 99)
        volume = np.clip(
            data / max(robust_max, 1e-12),
            0,
            1,
        )

        verts, faces, _, _ = measure.marching_cubes(
            volume,
            level=0.2,
            step_size=2,
        )
        verts -= np.array(data.shape) / 2.0
        np.save(self.surface_verts_path, verts)
        np.save(self.surface_faces_path, faces)
        return verts, faces

    # =========================================================================
    # 3D Pipe generation
    # =========================================================================

    @staticmethod
    def create_pipe_mesh(
        polyline,
        radius=4.5,
        n_segs=16,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pts = np.asarray(polyline, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 2:
            return np.empty((0, 3)), np.empty((0, 3), dtype=int)

        cleaned = [pts[0]]
        for pnt in pts[1:]:
            if np.linalg.norm(pnt - cleaned[-1]) > 1e-8:
                cleaned.append(pnt)
        pts = np.asarray(cleaned, dtype=float)
        if len(pts) < 2:
            return np.empty((0, 3)), np.empty((0, 3), dtype=int)

        dense = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            dist = float(np.linalg.norm(b - a))
            steps = max(2, int(np.ceil(dist / max(radius * 0.55, 1e-6))))
            for k in range(1, steps + 1):
                u = k / steps
                dense.append(a * (1.0 - u) + b * u)
        pts = np.asarray(dense, dtype=float)

        tangents = np.empty_like(pts)
        for i in range(len(pts)):
            if i == 0:
                d = pts[1] - pts[0]
            elif i == len(pts) - 1:
                d = pts[-1] - pts[-2]
            else:
                d = pts[i + 1] - pts[i - 1]
            nrm = np.linalg.norm(d)
            tangents[i] = d / (nrm if nrm > 1e-12 else 1.0)

        refs = [
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        ]
        ref = min(refs, key=lambda r: abs(np.dot(r, tangents[0])))
        normal = np.cross(tangents[0], ref)
        normal /= max(np.linalg.norm(normal), 1e-12)
        binormal = np.cross(tangents[0], normal)
        binormal /= max(np.linalg.norm(binormal), 1e-12)

        normals = [normal]
        binormals = [binormal]

        for i in range(1, len(pts)):
            t0 = tangents[i - 1]
            t1 = tangents[i]
            n0 = normals[-1]

            axis = np.cross(t0, t1)
            axis_len = np.linalg.norm(axis)
            dot = np.clip(np.dot(t0, t1), -1.0, 1.0)

            if axis_len < 1e-10:
                n1 = n0 - t1 * np.dot(n0, t1)
            else:
                axis /= axis_len
                angle = np.arctan2(axis_len, dot)
                ca = np.cos(angle)
                sa = np.sin(angle)
                n1 = (
                    n0 * ca
                    + np.cross(axis, n0) * sa
                    + axis * np.dot(axis, n0) * (1.0 - ca)
                )
                n1 -= t1 * np.dot(n1, t1)
                n1 /= max(np.linalg.norm(n1), 1e-12)

            b1 = np.cross(t1, n1)
            b1 /= max(np.linalg.norm(b1), 1e-12)
            normals.append(n1)
            binormals.append(b1)

        theta = np.linspace(0.0, 2.0 * np.pi, n_segs, endpoint=False)
        ct = np.cos(theta)
        st = np.sin(theta)

        verts = []
        for i, center in enumerate(pts):
            ring = (
                center
                + radius * (
                    ct[:, None] * normals[i][None, :]
                    + st[:, None] * binormals[i][None, :]
                )
            )
            verts.append(ring)
        verts = np.vstack(verts)

        faces = []
        ring_count = len(pts)
        for i in range(ring_count - 1):
            a = i * n_segs
            b = (i + 1) * n_segs
            for j in range(n_segs):
                j2 = (j + 1) % n_segs
                faces.append([a + j, a + j2, b + j])
                faces.append([a + j2, b + j2, b + j])

        start_center = len(verts)
        end_center = start_center + 1
        verts = np.vstack([verts, pts[0], pts[-1]])

        first = 0
        last = (ring_count - 1) * n_segs
        for j in range(n_segs):
            j2 = (j + 1) % n_segs
            faces.append([start_center, first + j2, first + j])
            faces.append([end_center, last + j, last + j2])

        return verts, np.asarray(faces, dtype=np.int32)

    # =========================================================================
    # 3D visualization
    # =========================================================================

    def show_cortical_overlay(self):
        if not PLOTLY_AVAILABLE:
            messagebox.showerror(
                "Missing Dependency",
                "Install:\n\n"
                "pip install plotly nibabel scikit-image scipy",
            )
            return

        if not self.latest_results:
            messagebox.showinfo("Empty Results", "Run the simulation first.")
            return

        self.status_var.set("Status: Building brain surface...")
        self.root.update_idletasks()

        try:
            points, edges = self._build_or_load_brain_surface()
        except Exception as e:
            messagebox.showerror("Surface Build Error", str(e))
            return

        traces = []

        # Brain mesh
        traces.append(
            go.Mesh3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                i=edges[:, 0],
                j=edges[:, 1],
                k=edges[:, 2],
                color="lightskyblue",
                opacity=0.16,
                lighting=dict(
                    ambient=0.7,
                    diffuse=0.6,
                    specular=0.2,
                ),
                name="Anatomical Brain Mesh",
                hoverinfo="skip",
            )
        )

        # Outer cortical labels
        all_labels = np.array(
            [
                self._get_scientific_region_name(pt[0], pt[1], pt[2])
                for pt in points
            ]
        )
        mesh_centroid = points.mean(axis=0)

        label_points = []
        label_text = []
        for label in np.unique(all_labels):
            group_idx = np.where(all_labels == label)[0]
            group_pts = points[group_idx]
            centroid = group_pts.mean(axis=0)
            nearest = group_idx[np.argmin(np.linalg.norm(group_pts - centroid, axis=1))]
            anchor = points[nearest]
            direction = anchor - mesh_centroid
            direction /= max(np.linalg.norm(direction), 1e-8)
            label_points.append(anchor + direction * 18.0)
            label_text.append(label)

        label_points = np.asarray(label_points)
        traces.append(
            go.Scatter3d(
                x=label_points[:, 0],
                y=label_points[:, 1],
                z=label_points[:, 2],
                mode="text",
                text=label_text,
                textfont=dict(
                    color="Black",
                    size=10,
                    family="Arial Bold",
                ),
                name="Scientific Region Labels",
            )
        )

        # Inner structures
        try:
            inner_points = self._build_or_load_inner_points()
        except Exception:
            inner_points = None

        if inner_points is not None and len(inner_points) > 0:
            inner_all_labels = np.array(
                [
                    self._get_inner_structure_name(pt[0], pt[1], pt[2])
                    for pt in inner_points
                ]
            )

            inner_label_points = []
            inner_label_text = []
            for label in np.unique(inner_all_labels):
                group_idx = np.where(inner_all_labels == label)[0]
                group_pts = inner_points[group_idx]
                centroid = group_pts.mean(axis=0)
                nearest = group_idx[np.argmin(np.linalg.norm(group_pts - centroid, axis=1))]
                inner_label_points.append(inner_points[nearest])
                inner_label_text.append(label)

            inner_label_points = np.asarray(inner_label_points)
            traces.append(
                go.Scatter3d(
                    x=inner_label_points[:, 0],
                    y=inner_label_points[:, 1],
                    z=inner_label_points[:, 2],
                    mode="text",
                    text=inner_label_text,
                    textfont=dict(
                        color="darkred",
                        size=10,
                        family="Arial Bold",
                    ),
                    name="Inner Structure Labels",
                )
            )

        # Circuit paths
        try:
            inner_points_for_pipes = self._build_or_load_inner_points()
        except Exception:
            inner_points_for_pipes = np.empty((0, 3), dtype=float)

        colorscales = ["Turbo", "Viridis", "Plasma", "Inferno", "Magma", "Cividis"]

        if len(inner_points_for_pipes) >= 2:
            inner_tree = cKDTree(inner_points_for_pipes)
            inner_count = len(inner_points_for_pipes)
            INNER_NEIGHBORS_K = 80

            for rank_idx, res in enumerate(self.latest_results):
                target_path = res[2]
                if not target_path or len(target_path) < 2:
                    continue

                rng = np.random.default_rng(1000 + rank_idx)
                current_idx = int(rng.integers(0, inner_count))
                direction = rng.normal(size=3)
                direction /= max(np.linalg.norm(direction), 1e-12)

                inner_coords = []
                path_weights = []
                visited = {current_idx}

                for weight in target_path:
                    raw_point = inner_points_for_pipes[current_idx]
                    inner_coords.append(raw_point.copy())
                    path_weights.append(float(weight))

                    k = min(INNER_NEIGHBORS_K + 1, inner_count)
                    _, neighbor_idx = inner_tree.query(raw_point, k=k)
                    candidates = np.atleast_1d(neighbor_idx).astype(int)
                    candidates = candidates[candidates != current_idx]

                    if visited:
                        unvisited = candidates[~np.isin(candidates, list(visited))]
                        if len(unvisited):
                            candidates = unvisited

                    if len(candidates) == 0:
                        continue

                    vecs = inner_points_for_pipes[candidates] - raw_point
                    lengths = np.linalg.norm(vecs, axis=1)
                    valid = lengths > 1e-6
                    if not np.any(valid):
                        continue

                    candidates = candidates[valid]
                    vecs = vecs[valid]
                    lengths = lengths[valid]
                    vecs_unit = vecs / lengths[:, None]
                    alignment = vecs_unit @ direction
                    score = alignment + 0.018 * np.minimum(lengths, 12.0)
                    best = int(candidates[int(np.argmax(score))])

                    direction = inner_points_for_pipes[best] - raw_point
                    direction /= max(np.linalg.norm(direction), 1e-12)
                    current_idx = best
                    visited.add(current_idx)

                if len(inner_coords) < 2:
                    continue

                pipe_radius = max(1.7, 2.8 - rank_idx * 0.12)
                v_mesh, f_mesh = self.create_pipe_mesh(
                    inner_coords,
                    radius=pipe_radius,
                    n_segs=14,
                )
                if len(v_mesh) == 0 or len(f_mesh) == 0:
                    continue

                source_w = np.asarray(path_weights, dtype=float)
                ring_count = max(1, len(v_mesh) // 14)

                if len(source_w) == 1:
                    dense_w = np.full(ring_count, source_w[0])
                else:
                    dense_w = np.interp(
                        np.linspace(0.0, 1.0, ring_count),
                        np.linspace(0.0, 1.0, len(source_w)),
                        source_w,
                    )

                vertex_values = np.repeat(dense_w, 14)
                if len(vertex_values) < len(v_mesh):
                    vertex_values = np.pad(
                        vertex_values,
                        (0, len(v_mesh) - len(vertex_values)),
                        mode="edge",
                    )
                elif len(vertex_values) > len(v_mesh):
                    vertex_values = vertex_values[:len(v_mesh)]

                vmin = float(np.min(vertex_values))
                vmax = float(np.max(vertex_values))
                if vmax <= vmin:
                    vmax = vmin + 1.0

                traces.append(
                    go.Mesh3d(
                        x=v_mesh[:, 0],
                        y=v_mesh[:, 1],
                        z=v_mesh[:, 2],
                        i=f_mesh[:, 0],
                        j=f_mesh[:, 1],
                        k=f_mesh[:, 2],
                        intensity=vertex_values,
                        colorscale=colorscales[rank_idx % len(colorscales)],
                        cmin=vmin,
                        cmax=vmax,
                        opacity=1.0,
                        flatshading=False,
                        lighting=dict(
                            ambient=0.75,
                            diffuse=1.0,
                            specular=0.9,
                            roughness=0.18,
                        ),
                        name=f"INNER Path Rank #{rank_idx + 1}",
                        hoverinfo="skip",
                        showscale=False,
                    )
                )

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=dict(
                text="Anatomical Brain Surface with Scientific Labels & Neural Circuits"
            ),
            scene=dict(
                xaxis=dict(visible=True, autorange="reversed"),
                yaxis=dict(visible=True),
                zaxis=dict(visible=True),
                camera=dict(
                    eye=dict(x=1.6, y=1.6, z=1.3),
                ),
            ),
        )

        output_file = "labeled_clean_brain.html"
        fig.write_html(output_file, include_plotlyjs=True)
        webbrowser.open("file://" + os.path.realpath(output_file))
        self.status_var.set("Status: Opened " + output_file)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cognitive-Neuro + 3D Brain Simulator"
    )
    parser.add_argument(
        "--mode",
        choices=["knapsack", "brain3d", "both"],
        default="brain3d",
        help="GUI mode: knapsack, brain3d, or both",
    )
    args = parser.parse_args()

    root = tk.Tk()

    if args.mode == "knapsack":
        KnapsackNeuroApp(root)
    elif args.mode == "brain3d":
        NeuralBeamSimulationApp(root)
    else:
        root.title("Cognitive-Neuro + 3D Brain Lab")
        root.geometry("1000x750")

        notebook = ttk.Notebook(root)
        notebook.pack(fill=tk.BOTH, expand=True)

        frame1 = ttk.Frame(notebook)
        frame2 = ttk.Frame(notebook)

        notebook.add(frame1, text="Knapsack Neuro")
        notebook.add(frame2, text="3D Brain Circuits")

        KnapsackNeuroApp(root, master=frame1)
        NeuralBeamSimulationApp(root, master=frame2)

    root.mainloop()


if __name__ == "__main__":
    main()
