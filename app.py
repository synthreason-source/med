"""
Cognitive Neuroscience Knapsack + 3D Brain Simulator (Continuous Drive)
=======================================================================

Single-file application that combines:

1. Knapsack/subset-sum task generation and beam-search solver.
2. Six-node Wilson–Cowan cognitive electrical network.
3. Continuous-time mapping from solver state to neural drive (interpolated).
4. Single-trial simulation of virtual EEG and regional activity.
5. CSV and NumPy output.
6. Optional Plotly EEG visualization.
7. Original 3D brain/circuit visualizer with MRI surface, labels, and pipes.
8. Tkinter GUI with tabs for both modes.

Run:
    python app.py --mode knapsack   # knapsack + neural GUI
    python app.py --mode brain3d    # 3D brain circuits GUI
    python app.py --mode both       # both in one window with tabs

Install:
    pip install numpy plotly nibabel scikit-image scipy
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import queue
import threading
import time
import tkinter as tk
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Any
import heapq

import numpy as np

try:
    import nibabel as nib
    from skimage import measure
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from tkinter import ttk, messagebox


# =============================================================================
# Knapsack / Subset-Sum Task
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
    target_sum: Optional[float] = None  # For subset-sum mode


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

    values = rng.uniform(value_range[0], value_range[1], size=n_items)
    weights = rng.uniform(weight_range[0], weight_range[1], size=n_items)

    items = [
        KnapsackItem(index=i, value=float(values[i]), weight=float(weights[i]))
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
        cost=0.0 if target_sum is None else abs(target_sum - 0.0),
        value=0.0,
        total_weight=0.0,
        item_index=0,
        path=(),
    )

    beam: List[SearchState] = [initial_state]
    history: List[dict] = []

    best_state = initial_state

    def state_key(s: SearchState) -> float:
        return s.cost - 0.01 * s.value + 1e-6 * s.total_weight

    for depth in range(max_depth):
        if depth >= len(items):
            break

        next_beam: List[SearchState] = []

        for state in beam:
            item = items[state.item_index]

            # Skip item
            new_state_skip = SearchState(
                cost=(
                    0.0
                    if target_sum is None
                    else abs(target_sum - state.total_weight)
                ),
                value=state.value,
                total_weight=state.total_weight,
                item_index=state.item_index + 1,
                path=state.path,
            )
            next_beam.append(new_state_skip)

            # Take item if capacity allows
            new_weight = state.total_weight + item.weight
            if new_weight <= capacity + 1e-12:
                new_cost = (
                    0.0
                    if target_sum is None
                    else abs(target_sum - new_weight)
                )
                new_state_take = SearchState(
                    cost=new_cost,
                    value=state.value + item.value,
                    total_weight=new_weight,
                    item_index=state.item_index + 1,
                    path=state.path + (item.index,),
                )
                next_beam.append(new_state_take)

        next_beam.sort(key=state_key)
        beam = next_beam[:beam_width]

        for s in beam:
            if target_sum is None:
                better = s.value > best_state.value or (
                    abs(s.value - best_state.value) < 1e-12
                    and s.total_weight < best_state.total_weight
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

        history.append({
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
        })

    return SolverResult(
        best_path=best_state.path,
        best_value=best_state.value,
        best_weight=best_state.total_weight,
        best_cost=best_state.cost if target_sum is not None else 0.0,
        history=history,
    )


# =============================================================================
# Cognitive Drive Mapping (Continuous)
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
            current_value / max(max_possible_value, 1e-12)
        )
        difficulty = np.clip(
            0.40 * np.log1p(remaining_items)
            + 0.40 * np.log1p(beam_width)
            + 0.20 * value_gap,
            0.0,
            1.0,
        )
        conflict = np.clip(
            0.5 * value_gap + 0.5 * (beam_width / 32.0),
            0.0,
            1.0,
        )
    else:
        error = abs(target_sum - current_sum)
        normalized_error = error / max(abs(target_sum), 1e-6)

        difficulty = np.clip(
            0.35 * np.log1p(remaining_items)
            + 0.40 * np.log1p(beam_width)
            + 0.25 * np.log1p(error),
            0.0,
            1.0,
        )

        conflict = np.clip(normalized_error, 0.0, 1.0)

    working_memory_load = np.clip(beam_width / 32.0, 0.0, 1.0)

    value_signal = np.clip(
        current_value / max(max_possible_value, 1e-12),
        0.0,
        1.0,
    )

    selection_signal = 1.0 if exact_match else 0.0

    if len(beam_states) > 1:
        costs = np.array([s["cost"] for s in beam_states], dtype=float)
        if target_sum is not None:
            neg_costs = -costs
            exp_neg = np.exp(neg_costs - neg_costs.max())
            probs = exp_neg / exp_neg.sum()
        else:
            values_arr = np.array([s["value"] for s in beam_states], dtype=float)
            exp_v = np.exp(values_arr - values_arr.max())
            probs = exp_v / exp_v.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-12))
        max_entropy = np.log(len(beam_states))
        beam_entropy_norm = entropy / max(max_entropy, 1e-12)
    else:
        beam_entropy_norm = 0.0

    drive = np.array([
        0.70 * difficulty + 0.30 * conflict + 0.20 * beam_entropy_norm,
        0.80 * difficulty + 0.20 * conflict,
        0.60 * working_memory_load + 0.40 * difficulty,
        0.70 * working_memory_load + 0.30 * conflict,
        0.60 * value_signal + 0.20 * (1.0 - conflict),
        0.90 * selection_signal,
    ])

    drive = np.clip(drive, 0.0, 1.0)
    return drive


# =============================================================================
# Wilson–Cowan Neural Network
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


def sigmoid(x: np.ndarray, gain: float, threshold: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-gain * (x - threshold)))


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
            connectivity = np.array([
                [0.00, 0.30, 0.40, 0.50, 0.20, 0.10],
                [0.30, 0.00, 0.40, 0.30, 0.20, 0.10],
                [0.40, 0.40, 0.00, 0.60, 0.30, 0.10],
                [0.50, 0.30, 0.60, 0.00, 0.50, 0.20],
                [0.20, 0.20, 0.30, 0.50, 0.00, 0.30],
                [0.10, 0.10, 0.10, 0.20, 0.30, 0.00],
            ], dtype=float)

        self.C = np.asarray(connectivity, dtype=float)
        self.C /= max(self.C.max(), 1e-12)

        self.E = np.full(n, 0.05, dtype=float)
        self.I = np.full(n, 0.05, dtype=float)

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
        external_input = np.asarray(external_input, dtype=float)

        net_excitation = coupling * (self.C @ self.E)

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

        target_e = sigmoid(e_drive, p.gain_e, p.threshold_e)
        target_i = sigmoid(i_drive, p.gain_i, p.threshold_i)

        self.E += dt / p.tau_e * (-self.E + target_e)
        self.I += dt / p.tau_i * (-self.I + target_i)

        self.E = np.clip(self.E, 0.0, 1.0)
        self.I = np.clip(self.I, 0.0, 1.0)

        return self.E.copy(), self.I.copy()

    def simulate_continuous(
        self,
        drive_time_series: np.ndarray,
        sample_rate: int = 500,
        coupling: float = 0.8,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate the network over a full trial with a continuous drive signal.

        drive_time_series: shape (T, 6)
        sample_rate: Hz
        """
        drive_time_series = np.asarray(drive_time_series, dtype=float)
        total_steps = drive_time_series.shape[0]

        e_trace = np.zeros((total_steps, len(self.names)), dtype=float)
        i_trace = np.zeros_like(e_trace)

        dt = 1.0 / sample_rate

        self.reset()

        for t in range(total_steps):
            e, i = self.step(
                external_input=drive_time_series[t],
                dt=dt,
                coupling=coupling,
            )
            e_trace[t] = e
            i_trace[t] = i

        return e_trace, i_trace


# =============================================================================
# Task + Network Simulation (Continuous Drive)
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
    """
    Run knapsack solver and generate a continuous neural drive over the whole trial.
    """

    result = bounded_beam_search(
        instance=instance,
        beam_width=beam_width,
        max_depth=max_depth,
        seed=seed,
    )

    max_possible_value = sum(
        item.value for item in instance.items
    )

    n_steps = len(result.history)
    drive_matrix = np.zeros((n_steps, 6), dtype=float)

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

    # Interpolate drive to continuous time
    total_steps = int(trial_duration * sample_rate)
    time_discrete = np.linspace(0, trial_duration, n_steps)
    time_continuous = np.linspace(0, trial_duration, total_steps)

    drive_continuous = np.zeros((total_steps, 6), dtype=float)
    for i in range(6):
        drive_continuous[:, i] = np.interp(
            time_continuous, time_discrete, drive_matrix[:, i]
        )

    network.reset()

    e_trace, i_trace = network.simulate_continuous(
        drive_time_series=drive_continuous,
        sample_rate=sample_rate,
    )

    eeg_weights = np.array([
        0.20,  # dACC
        0.15,  # Insula
        0.20,  # IPS
        0.25,  # DLPFC
        0.15,  # OFC
        0.05,  # Motor
    ])

    virtual_eeg = e_trace @ eeg_weights

    neural_data = {
        "e_trace": e_trace,
        "i_trace": i_trace,
        "virtual_eeg": virtual_eeg,
        "drive_continuous": drive_continuous,
        "sample_rate": sample_rate,
        "region_names": network.names,
        "solver_history": result.history,
        "trial_duration": trial_duration,
    }

    return result, neural_data


# =============================================================================
# CSV / File I/O
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

    # Map each time point to a depth index
    n_depths = len(history)
    depth_per_step = np.clip(
        (time_vec / trial_duration * n_depths).astype(int),
        0,
        n_depths - 1,
    )

    rows = []
    for t_idx in range(n_steps):
        depth = int(depth_per_step[t_idx])
        step_data = history[depth]

        rows.append({
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
        })

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

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_numpy_arrays(
    base_filename: str,
    neural_data: dict,
) -> None:
    np.save(base_filename + "_e_trace.npy", neural_data["e_trace"])
    np.save(base_filename + "_i_trace.npy", neural_data["i_trace"])
    np.save(base_filename + "_virtual_eeg.npy", neural_data["virtual_eeg"])
    np.save(
        base_filename + "_drive_continuous.npy",
        neural_data["drive_continuous"],
    )


# =============================================================================
# Plotly EEG Visualization
# =============================================================================


def plot_neural_activity(
    neural_data: dict,
    output_html: str = "neural_activity.html",
) -> None:
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
            line=dict(width=2, color="black"),
        )
    )

    fig = go.Figure(data=traces)

    fig.update_layout(
        title="Simulated Regional Excitatory Activity and Virtual EEG (Continuous)",
        xaxis_title="Time (s)",
        yaxis_title="Activity",
        legend_title="Region",
        height=500,
        xaxis=dict(range=[0, trial_duration]),
    )

    fig.write_html(output_html, include_plotlyjs=True)

    import webbrowser
    webbrowser.open("file://" + os.path.realpath(output_html))


# =============================================================================
# Tkinter GUI Application - Knapsack Neuro (Continuous)
# =============================================================================

class KnapsackNeuroApp:
    def __init__(self, root: tk.Tk, master: Optional[tk.Widget] = None):
        self.root = root
        self.master = master if master is not None else root

        self.master.title("Knapsack Cognitive-Neuro Simulator (Continuous)")
        self.master.geometry("900x700")
        self.master.minsize(700, 550)

        self.is_running = False
        self.result_queue: queue.Queue = queue.Queue()
        self.latest_result = None
        self.latest_instance = None
        self.latest_neural_data = None

        self._create_widgets()

    def _create_widgets(self) -> None:
        main_frame = ttk.Frame(self.master, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        title_label = ttk.Label(
            main_frame,
            text="Knapsack Cognitive-Neuro Simulator (Continuous)",
            font=("Arial", 14, "bold"),
        )
        title_label.pack(pady=(0, 10))

        # Task parameters
        task_frame = ttk.LabelFrame(
            main_frame, text="Task Parameters", padding=10
        )
        task_frame.pack(fill=tk.X, pady=5)

        ttk.Label(task_frame, text="Number of items:").grid(
            row=0, column=0, sticky="w", pady=5
        )
        self.n_items_entry = ttk.Entry(task_frame, width=12)
        self.n_items_entry.insert(0, "20")
        self.n_items_entry.grid(
            row=0, column=1, sticky="w", padx=10, pady=5
        )

        ttk.Label(task_frame, text="Capacity:").grid(
            row=1, column=0, sticky="w", pady=5
        )
        self.capacity_entry = ttk.Entry(task_frame, width=12)
        self.capacity_entry.insert(0, "50.0")
        self.capacity_entry.grid(
            row=1, column=1, sticky="w", padx=10, pady=5
        )

        ttk.Label(task_frame, text="Target ratio (optional):").grid(
            row=2, column=0, sticky="w", pady=5
        )
        self.target_ratio_entry = ttk.Entry(task_frame, width=12)
        self.target_ratio_entry.insert(0, "0.7")
        self.target_ratio_entry.grid(
            row=2, column=1, sticky="w", padx=10, pady=5
        )

        # Solver & neural parameters
        solver_frame = ttk.LabelFrame(
            main_frame, text="Solver & Neural Parameters", padding=10
        )
        solver_frame.pack(fill=tk.X, pady=5)

        ttk.Label(solver_frame, text="Beam width:").grid(
            row=0, column=0, sticky="w", pady=5
        )
        self.beam_width_entry = ttk.Entry(solver_frame, width=12)
        self.beam_width_entry.insert(0, "8")
        self.beam_width_entry.grid(
            row=0, column=1, sticky="w", padx=10, pady=5
        )

        ttk.Label(solver_frame, text="Trial duration (s):").grid(
            row=1, column=0, sticky="w", pady=5
        )
        self.trial_duration_entry = ttk.Entry(solver_frame, width=12)
        self.trial_duration_entry.insert(0, "20.0")
        self.trial_duration_entry.grid(
            row=1, column=1, sticky="w", padx=10, pady=5
        )

        ttk.Label(solver_frame, text="Sample rate (Hz):").grid(
            row=2, column=0, sticky="w", pady=5
        )
        self.sample_rate_entry = ttk.Entry(solver_frame, width=12)
        self.sample_rate_entry.insert(0, "500")
        self.sample_rate_entry.grid(
            row=2, column=1, sticky="w", padx=10, pady=5
        )

        # Run / visualize buttons
        self.run_button = ttk.Button(
            solver_frame, text="Run Simulation", command=self.start_simulation
        )
        self.run_button.grid(
            row=0, column=2, rowspan=3, padx=15, ipadx=5, ipady=10
        )

        self.plot_button = ttk.Button(
            solver_frame,
            text="Plot Neural Activity",
            command=self.show_plot,
            state=tk.DISABLED,
        )
        self.plot_button.grid(
            row=0, column=3, rowspan=3, padx=10, ipadx=5, ipady=10
        )

        self.status_var = tk.StringVar(value="Status: Ready")
        status_label = ttk.Label(
            main_frame,
            textvariable=self.status_var,
            font=("Arial", 10, "italic"),
        )
        status_label.pack(anchor="w", pady=(10, 5))

        # Results area
        results_frame = ttk.LabelFrame(
            main_frame, text="Solver Summary", padding=10
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
            results_frame, orient=tk.VERTICAL, command=self.summary_text.yview
        )
        self.summary_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _worker_loop(self) -> None:
        try:
            n_items = int(self.n_items_entry.get())
            capacity = float(self.capacity_entry.get())
            target_ratio_str = self.target_ratio_entry.get().strip()
            beam_width = int(self.beam_width_entry.get())
            trial_duration = float(self.trial_duration_entry.get())
            sample_rate = int(self.sample_rate_entry.get())

            if target_ratio_str:
                target_ratio = float(target_ratio_str)
            else:
                target_ratio = None

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
            save_numpy_arrays(base_name, neural_data)

            self.result_queue.put(("SUCCESS", (instance, result, neural_data)))

        except Exception as e:
            self.result_queue.put(("ERROR", str(e)))

    def start_simulation(self) -> None:
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
                f"Please enter valid numeric parameters.\nDetails: {err}",
            )
            return

        self.is_running = True
        self.run_button.config(state=tk.DISABLED)
        self.plot_button.config(state=tk.DISABLED)
        self.status_var.set("Status: Running knapsack + neural simulation...")

        worker_thread = threading.Thread(target=self._worker_loop)
        worker_thread.daemon = True
        worker_thread.start()

        self.root.after(100, self._check_queue)

    def _check_queue(self) -> None:
        try:
            status, data = self.result_queue.get_nowait()
            self.is_running = False
            self.run_button.config(state=tk.NORMAL)

            if status == "SUCCESS":
                instance, result, neural_data = data
                self.latest_instance = instance
                self.latest_result = (result, neural_data)
                self.latest_neural_data = neural_data

                self.status_var.set("Status: Simulation complete.")
                self._update_summary(instance, result, neural_data)

                if PLOTLY_AVAILABLE:
                    self.plot_button.config(state=tk.NORMAL)
                else:
                    self.status_var.set(
                        "Status: Complete. Install 'plotly' to enable visualization."
                    )
            else:
                self.status_var.set("Status: Simulation failed.")
                messagebox.showerror("Execution Error", data)

        except queue.Empty:
            if self.is_running:
                self.root.after(100, self._check_queue)

    def _update_summary(
        self,
        instance,
        result,
        neural_data,
    ) -> None:
        self.summary_text.delete("1.0", tk.END)

        lines = [
            "Task type: " + (
                "Subset-sum knapsack"
                if instance.target_sum is not None
                else "0–1 knapsack"
            ),
            f"Number of items: {len(instance.items)}",
            f"Capacity: {instance.capacity:.3f}",
            f"Target sum: {instance.target_sum}",
            "",
            "Best solution:",
            f"  Selected indices: {result.best_path}",
            f"  Total value: {result.best_value:.4f}",
            f"  Total weight: {result.best_weight:.4f}",
            f"  Cost (target error): {result.best_cost:.4f}",
            "",
            "Neural simulation (continuous):",
            f"  Regions: {', '.join(neural_data['region_names'])}",
            f"  E-trace shape: {neural_data['e_trace'].shape}",
            f"  Virtual EEG length: {len(neural_data['virtual_eeg'])}",
            f"  Trial duration: {neural_data['trial_duration']:.2f} s",
            "",
            "Outputs saved:",
            "  - <base>_e_trace.npy",
            "  - <base>_i_trace.npy",
            "  - <base>_virtual_eeg.npy",
            "  - <base>_drive_continuous.npy",
            "  - <base>.csv",
        ]

        self.summary_text.insert(tk.END, "\n".join(lines))

    def show_plot(self) -> None:
        if self.latest_neural_data is None:
            messagebox.showinfo(
                "No Results",
                "Please run the simulation first to generate data.",
            )
            return

        output_html = "neural_activity.html"
        plot_neural_activity(self.latest_neural_data, output_html=output_html)

        self.status_var.set(f"Status: Opened {output_html}")


# =============================================================================
# 3D Brain / Circuit Visualization (unchanged logic, only class name)
# =============================================================================


class NeuralBeamSimulationApp:
    def __init__(self, root: tk.Tk, master: Optional[tk.Widget] = None):
        self.root = root
        self.master = master if master is not None else root

        if master is None:
            # Only set title/geometry when used as a top-level app
            self.master.title("Neural Beam Engine & Custom Connectivity Visualizer")
            self.master.geometry("800x650")
            self.master.minsize(700, 550)

        self.data_path = "weights.bin"
        self.mri_path = "real_brain_mri_t1.nii.gz"
        self.surface_verts_path = "brain_surface_verts.npy"
        self.surface_faces_path = "brain_surface_faces.npy"
        self.inner_points_path = "brain_inner_points.npy"

        self.is_running = False
        self.result_queue = queue.Queue()
        self.latest_results = []

        self._ensure_data_files()
        self._create_widgets()

    def _create_widgets(self):
        main_frame = ttk.Frame(self.master, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        title_label = ttk.Label(
            main_frame,
            text="Neural Beam Engine with Custom Connectivity Files",
            font=("Arial", 14, "bold"),
        )
        title_label.pack(pady=(0, 10))

        control_frame = ttk.LabelFrame(
            main_frame, text="Simulation Parameters", padding=10
        )
        control_frame.pack(fill=tk.X, pady=5)

        ttk.Label(control_frame, text="Target Sum:").grid(
            row=0, column=0, sticky="w", pady=5
        )
        self.target_entry = ttk.Entry(control_frame, width=12)
        self.target_entry.insert(0, "7.0")
        self.target_entry.grid(
            row=0, column=1, sticky="w", padx=10, pady=5
        )

        ttk.Label(control_frame, text="Beam Width:").grid(
            row=1, column=0, sticky="w", pady=5
        )
        self.beam_entry = ttk.Entry(control_frame, width=12)
        self.beam_entry.insert(0, "4")
        self.beam_entry.grid(
            row=1, column=1, sticky="w", padx=10, pady=5
        )

        self.run_button = ttk.Button(
            control_frame, text="Run Simulation", command=self.start_simulation
        )
        self.run_button.grid(
            row=0, column=2, rowspan=2, padx=15, ipadx=5, ipady=10
        )

        self.mri_button = ttk.Button(
            control_frame,
            text="Render Circuits & Labels (3D)",
            command=self.show_cortical_overlay,
            state=tk.DISABLED,
        )
        self.mri_button.grid(
            row=0, column=3, rowspan=2, padx=10, ipadx=5, ipady=10
        )

        self.status_var = tk.StringVar(value="Status: Ready")
        status_label = ttk.Label(
            main_frame,
            textvariable=self.status_var,
            font=("Arial", 10, "italic"),
        )
        status_label.pack(anchor="w", pady=(10, 5))

        results_frame = ttk.LabelFrame(
            main_frame, text="Top Optimal Beam Paths", padding=10
        )
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        columns = ("rank", "cost", "sum", "path")
        self.tree = ttk.Treeview(
            results_frame, columns=columns, show="headings", height=8
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
            results_frame, orient=tk.VERTICAL, command=self.tree.yview
        )
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    def _ensure_data_files(self):
        if not os.path.exists(self.data_path):
            raw_weights = np.array(
                [1.5, 2.2, 3.8, 4.1, 5.0, 6.3, 1.1], dtype=np.float32
            )
            raw_weights.tofile(self.data_path)

    def _load_data(self) -> np.ndarray:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at {self.data_path}")

        with open(self.data_path, "rb") as f:
            return np.fromfile(f, dtype=np.float32)

    def bounded_beam_search(
        self, target_sum: float, beam_width: int, weights: np.ndarray
    ):
        beam = [(0.0, 0.0, ())]

        for w in weights:
            if not self.is_running:
                break

            w_float = float(w)
            next_beam = []

            for cost, current_sum, path in beam:
                c_float = float(cost)
                s_float = float(current_sum)

                next_beam.append((c_float, s_float, path))

                new_sum = s_float + w_float
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
            messagebox.showerror(
                "Invalid Input",
                f"Please enter valid numeric parameters.\nDetails: {err}",
            )
            return

        for item in self.tree.get_children():
            self.tree.delete(item)

        self.is_running = True
        self.run_button.config(state=tk.DISABLED)
        self.mri_button.config(state=tk.DISABLED)
        self.status_var.set("Status: Running subset-sum beam search...")

        worker_thread = threading.Thread(
            target=self._worker_loop, args=(target_sum, beam_width)
        )
        worker_thread.daemon = True
        worker_thread.start()

        self.root.after(100, self._check_queue)

    def _check_queue(self):
        try:
            status, data = self.result_queue.get_nowait()
            self.is_running = False
            self.run_button.config(state=tk.NORMAL)

            if status == "SUCCESS":
                self.latest_results = data
                self.status_var.set("Status: Simulation complete.")
                for idx, res in enumerate(data, start=1):
                    cost_val = float(res[0])
                    sum_val = float(res[1])
                    path_val = str(res[2])
                    self.tree.insert(
                        "",
                        tk.END,
                        values=(
                            idx,
                            f"{cost_val:.4f}",
                            f"{sum_val:.4f}",
                            path_val,
                        ),
                    )

                if PLOTLY_AVAILABLE:
                    self.mri_button.config(state=tk.NORMAL)
                else:
                    self.status_var.set(
                        "Status: Complete. Install 'plotly' to enable 3D visualizer."
                    )
            else:
                self.status_var.set("Status: Simulation failed.")
                messagebox.showerror("Execution Error", data)

        except queue.Empty:
            if self.is_running:
                self.root.after(100, self._check_queue)

    def _get_scientific_region_name(self, x: float, y: float, z: float) -> str:
        hemisphere = "Right" if x >= 0 else "Left"
        ax, ay, az = abs(x), y, z

        if ay > 30:
            if az > 20:
                return f"{hemisphere} Sup. Frontal"
            elif az > 0:
                return f"{hemisphere} Mid. Frontal"
            else:
                return f"{hemisphere} Orbital Frontal"
        elif 0 <= ay <= 30:
            if az > 35:
                return f"{hemisphere} Precentral / Motor"
            elif az > 10:
                return f"{hemisphere} Supp. Motor"
            elif az < -5:
                return f"{hemisphere} Insular"
            else:
                return f"{hemisphere} Ant. Cingulate"
        elif -50 <= ay < 0:
            if az > 40:
                return f"{hemisphere} Postcentral"
            elif az > 20:
                return f"{hemisphere} Inf. Parietal"
            elif az < 0:
                return f"{hemisphere} Sup. Temporal"
            else:
                return f"{hemisphere} Post. Cingulate"
        else:
            if az > 10:
                return f"{hemisphere} Precuneus"
            elif az < -10:
                return f"{hemisphere} Fusiform"
            else:
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
            else:
                return "Brainstem"
        else:
            if az > 10:
                if ay > 0:
                    return f"{hemisphere} Caudate Nucleus"
                else:
                    return f"{hemisphere} Putamen"
            elif az > -8:
                if ay > -5:
                    return f"{hemisphere} Putamen"
                elif ay > -25:
                    return f"{hemisphere} Amygdala"
                else:
                    return f"{hemisphere} Hippocampus"
            else:
                if ay < -20:
                    return f"{hemisphere} Cerebellum"
                else:
                    return f"{hemisphere} Hippocampus"

    def _build_or_load_inner_points(self):
        if os.path.exists(self.inner_points_path):
            return np.load(self.inner_points_path)

        if not os.path.exists(self.mri_path):
            raise FileNotFoundError(f"MRI file not found at {self.mri_path}")

        from scipy import ndimage

        img = nib.load(self.mri_path)
        data = img.get_fdata().astype(np.float32)
        robust_max = np.percentile(data[data > 0], 99)
        volume = np.clip(data / robust_max, 0, 1)

        mask = volume > 0.2
        interior_mask = ndimage.binary_erosion(mask, iterations=15)

        coords = np.argwhere(interior_mask).astype(np.float64)
        rng = np.random.default_rng(0)
        n_sample = min(60000, len(coords))
        sample = coords[rng.choice(len(coords), size=n_sample, replace=False)]

        sample = sample - np.array(data.shape) / 2.0

        np.save(self.inner_points_path, sample)
        return sample

    def _build_or_load_brain_surface(self):
        if os.path.exists(self.surface_verts_path) and os.path.exists(
            self.surface_faces_path
        ):
            verts = np.load(self.surface_verts_path)
            faces = np.load(self.surface_faces_path)
            return verts, faces

        if not os.path.exists(self.mri_path):
            raise FileNotFoundError(f"MRI file not found at {self.mri_path}")

        img = nib.load(self.mri_path)
        data = img.get_fdata().astype(np.float32)

        robust_max = np.percentile(data[data > 0], 99)
        volume = np.clip(data / robust_max, 0, 1)

        verts, faces, _normals, _values = measure.marching_cubes(
            volume, level=0.2, step_size=2
        )

        verts = verts - np.array(data.shape) / 2.0

        np.save(self.surface_verts_path, verts)
        np.save(self.surface_faces_path, faces)
        return verts, faces

    def show_cortical_overlay(self):
        if not PLOTLY_AVAILABLE:
            messagebox.showerror(
                "Missing Dependency",
                "Plotly and scikit-image are required. Run: pip install plotly scikit-image",
            )
            return

        if not self.latest_results:
            messagebox.showinfo(
                "Empty Results", "Please run the simulation first to generate paths."
            )
            return

        self.status_var.set(
            "Status: Building brain surface from MRI (first run only)..."
        )
        self.root.update_idletasks()

        try:
            points, edges = self._build_or_load_brain_surface()
        except Exception as e:
            messagebox.showerror(
                "Surface Build Error", f"Failed to build brain surface:\n{e}"
            )
            return

        self.status_var.set("Status: Loading labels and circuit paths...")
        self.root.update_idletasks()

        traces = []

        brain_mesh = go.Mesh3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            i=edges[:, 0],
            j=edges[:, 1],
            k=edges[:, 2],
            color="lightskyblue",
            opacity=0.16,
            lighting=dict(ambient=0.7, diffuse=0.6, specular=0.2),
            name="Anatomical Brain Mesh",
            hoverinfo="skip",
        )
        traces.append(brain_mesh)

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
            nearest = group_idx[
                np.argmin(np.linalg.norm(group_pts - centroid, axis=1))
            ]
            anchor = points[nearest]
            direction = anchor - mesh_centroid
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            label_points.append(anchor + direction * 18.0)
            label_text.append(label)
        label_points = np.array(label_points)

        labels_trace = go.Scatter3d(
            x=label_points[:, 0],
            y=label_points[:, 1],
            z=label_points[:, 2],
            mode="text",
            text=label_text,
            textfont=dict(color="Black", size=10, family="Arial Bold"),
            name="Scientific Region Labels",
        )
        traces.append(labels_trace)

        try:
            inner_points = self._build_or_load_inner_points()
        except Exception as e:
            inner_points = None
            self.status_var.set(f"Status: Inner labels skipped ({e})")

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
                nearest = group_idx[
                    np.argmin(np.linalg.norm(group_pts - centroid, axis=1))
                ]
                inner_label_points.append(inner_points[nearest])
                inner_label_text.append(label)
            inner_label_points = np.array(inner_label_points)

            inner_labels_trace = go.Scatter3d(
                x=inner_label_points[:, 0],
                y=inner_label_points[:, 1],
                z=inner_label_points[:, 2],
                mode="text",
                text=inner_label_text,
                textfont=dict(color="darkred", size=10, family="Arial Bold"),
                name="Inner Structure Labels",
            )
            traces.append(inner_labels_trace)

        def create_pipe_mesh(polyline, radius=4.5, n_segs=16, subdivisions=8):
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

            ref_candidates = (
                np.array([0.0, 0.0, 1.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([1.0, 0.0, 0.0]),
            )
            ref = min(ref_candidates, key=lambda r: abs(np.dot(r, tangents[0])))
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
                    ca, sa = np.cos(angle), np.sin(angle)
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
                ring = center + radius * (
                    ct[:, None] * normals[i][None, :]
                    + st[:, None] * binormals[i][None, :]
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
                    faces.append((a + j, a + j2, b + j))
                    faces.append((a + j2, b + j2, b + j))

            start_center = len(verts)
            end_center = start_center + 1
            verts = np.vstack((verts, pts[0], pts[-1]))
            first = 0
            last = (ring_count - 1) * n_segs
            for j in range(n_segs):
                j2 = (j + 1) % n_segs
                faces.append((start_center, first + j2, first + j))
                faces.append((end_center, last + j, last + j2))

            return verts, np.asarray(faces, dtype=np.int32)

        colorscales = ["Turbo", "Viridis", "Plasma", "Inferno", "Magma", "Cividis"]

        from scipy.spatial import cKDTree

        try:
            inner_points_for_pipes = self._build_or_load_inner_points()
        except Exception:
            inner_points_for_pipes = np.empty((0, 3), dtype=float)

        if len(inner_points_for_pipes) >= 2:
            inner_tree = cKDTree(inner_points_for_pipes)
            inner_count = len(inner_points_for_pipes)
            INNER_NEIGHBORS_K = 80

            for rank_idx, res in enumerate(self.latest_results):
                target_path = res[2]
                if not target_path or len(target_path) < 2:
                    continue

                seed_rng = np.random.default_rng(1000 + rank_idx)
                current_idx = int(seed_rng.integers(0, inner_count))
                direction = seed_rng.normal(size=3)
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
                        unvisited = candidates[
                            ~np.isin(candidates, list(visited))
                        ]
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
                v_mesh, f_mesh = create_pipe_mesh(
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
                    vertex_values = vertex_values[: len(v_mesh)]

                cscale = colorscales[rank_idx % len(colorscales)]
                pipe_trace = go.Mesh3d(
                    x=v_mesh[:, 0],
                    y=v_mesh[:, 1],
                    z=v_mesh[:, 2],
                    i=f_mesh[:, 0],
                    j=f_mesh[:, 1],
                    k=f_mesh[:, 2],
                    intensity=vertex_values,
                    colorscale=cscale,
                    cmin=float(np.min(vertex_values)),
                    cmax=(
                        float(np.max(vertex_values))
                        if np.max(vertex_values) > np.min(vertex_values)
                        else float(np.min(vertex_values) + 1.0)
                    ),
                    opacity=1.0,
                    flatshading=False,
                    lighting=dict(
                        ambient=0.75, diffuse=1.0, specular=0.9, roughness=0.18
                    ),
                    name=f"INNER Path Rank #{rank_idx+1}",
                    hoverinfo="skip",
                    showscale=False,
                )
                traces.append(pipe_trace)

        fig = go.Figure(data=traces)

        fig.update_layout(
            title=dict(text="Anatomical Brain Surface with Scientific Labels & Circuits"),
            scene=dict(
                xaxis=dict(visible=True, autorange="reversed"),
                yaxis=dict(visible=True),
                zaxis=dict(visible=True),
                camera=dict(eye=dict(x=1.6, y=1.6, z=1.3)),
            ),
        )

        output_file = "labeled_clean_brain.html"
        fig.write_html(output_file, include_plotlyjs=True)

        import webbrowser

        webbrowser.open("file://" + os.path.realpath(output_file))

        self.status_var.set(f"Status: Opened labeled clean view ({output_file}).")


# =============================================================================
# Main entry point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Knapsack Cognitive-Neuro + 3D Brain Simulator"
    )
    parser.add_argument(
        "--mode",
        choices=["knapsack", "brain3d"],
        default="brain3d",
        help="Which GUI to launch.",
    )
    args = parser.parse_args()

    root = tk.Tk()

    if args.mode == "knapsack":
        app = KnapsackNeuroApp(root)
    elif args.mode == "brain3d":
        app = NeuralBeamSimulationApp(root)
    else:
        root.title("Cognitive-Neuro + 3D Brain Lab")
        notebook = ttk.Notebook(root)
        notebook.pack(fill=tk.BOTH, expand=True)

        frame1 = ttk.Frame(notebook)
        frame2 = ttk.Frame(notebook)
        notebook.add(frame1, text="Knapsack Neuro (Continuous)")
        notebook.add(frame2, text="3D Brain Circuits")

        app1 = KnapsackNeuroApp(root, master=frame1)
        app2 = NeuralBeamSimulationApp(root, master=frame2)

    root.mainloop()

if __name__ == "__main__":
    main()
