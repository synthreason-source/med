"""
Neural circuit simulation on an MRI-derived structural graph.

This reuses three real engineering patterns from the original subset-sum
beam-search script:

  1. Disk-backed data + mmap streaming, so RAM stays flat regardless of
     how large the connectivity graph gets.
  2. Beam search pruned with heapq, guided by a score function.
  3. An immutable linked-list path (`_materialize`) so the beam doesn't
     need to store full candidate paths in memory.

The original script's "holographic tensor / resonance" scoring was
decorative math with no real signal in it (it was still just a subset-sum
solver underneath). Here the equivalent scoring is grounded in something
real: how much a candidate path advances a propagating signal toward a
target brain region.

No MRI file was provided, so a synthetic volume stands in for one. Swap
`synthesize_mri_volume()` for a loader (e.g. nibabel) to use a real scan.
"""

import os
import struct
import mmap
import heapq
from collections import deque
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MRI_PATH = "real_brain_mri_t1.nii.gz"
STRIDE = 1  # volume below is already downsampled, so 1 grid-step == 4 real voxels
EDGES_PATH = "connectivity_edges.bin"


# ---------------------------------------------------------------------------
# 1. Load the real MRI scan
# ---------------------------------------------------------------------------
def load_real_mri_volume(path=MRI_PATH, downsample=8):
    """
    Load a real NIfTI scan and normalize/downsample it so the rest of the
    pipeline (node count, disk edge list, beam search) stays fast.

    Real scans have arbitrary intensity units (this one's T1 values run
    0-1461, not 0-1) and are full-resolution (240x240x155 = ~8.9M voxels
    here), so both normalization and downsampling are necessary -- the
    original synthetic-phantom version skipped both because it generated
    data already in the range/size the rest of the code expected.
    """
    import nibabel as nib

    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)

    # coarse spatial downsample: every `downsample`-th voxel along each axis
    data = data[::downsample, ::downsample, ::downsample]

    # normalize intensity to [0, 1] using a robust (99th percentile) max,
    # so a few very bright outlier voxels don't wash out the rest
    robust_max = np.percentile(data[data > 0], 99)
    volume = np.clip(data / robust_max, 0, 1)

    return volume


# ---------------------------------------------------------------------------
# 2. Build a structural connectivity graph and stream it through disk
# ---------------------------------------------------------------------------
def build_connectivity_nodes(volume, stride=STRIDE, threshold=0.2):
    """Subsample tissue voxels above `threshold` into graph nodes."""
    coords = []
    shape = volume.shape
    for z in range(0, shape[0], stride):
        for y in range(0, shape[1], stride):
            for x in range(0, shape[2], stride):
                if volume[z, y, x] > threshold:
                    coords.append((z, y, x))
    return coords


def write_connectivity_edges(volume, coords, path=EDGES_PATH, stride=STRIDE):
    """
    Stream candidate structural edges straight to disk as fixed-size
    binary records (src:u32, dst:u32, weight:f32) -- the same
    disk-backed-dataset pattern the original script used for its integer
    list. This is what lets the graph scale to whole-brain connectomes
    (millions of nodes) without holding it all in RAM at once.
    """
    index = {c: i for i, c in enumerate(coords)}
    offsets = [(stride, 0, 0), (0, stride, 0), (0, 0, stride)]
    n_edges = 0
    with open(path, "wb") as f:
        for (z, y, x) in coords:
            i = index[(z, y, x)]
            for dz, dy, dx in offsets:
                nb = (z + dz, y + dy, x + dx)
                j = index.get(nb)
                if j is not None:
                    w = 1.0 - abs(float(volume[z, y, x]) - float(volume[nb]))
                    f.write(struct.pack("<IIf", i, j, w))
                    f.write(struct.pack("<IIf", j, i, w))
                    n_edges += 2
    return n_edges


def load_adjacency_from_disk(path, n_nodes):
    """mmap the edge file and build an adjacency list in one streamed pass."""
    adjacency = [[] for _ in range(n_nodes)]
    rec_size = 12  # 4 + 4 + 4 bytes
    with open(path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            n_records = len(mm) // rec_size
            for r in range(n_records):
                off = r * rec_size
                i, j, w = struct.unpack("<IIf", mm[off:off + rec_size])
                adjacency[i].append((j, w))
    return adjacency


# ---------------------------------------------------------------------------
# 3. Real-time activation propagation (leaky-integrate-and-fire-ish)
# ---------------------------------------------------------------------------
def simulate_realtime_activity(adjacency, seed_idx, steps=60,
                                decay=0.85, threshold=0.5, gain=1.3):
    """
    Simple discrete-time propagation: nodes above `threshold` "fire" and
    push weighted activation to neighbors each step, while all activation
    leaks away by `decay`. This is a coarse stand-in for real-time neural
    circuit dynamics, not a biophysical model.
    """
    n = len(adjacency)
    activity = np.zeros(n, dtype=np.float32)
    activity[seed_idx] = 1.0
    history = [activity.copy()]

    for _ in range(steps):
        new_activity = activity * decay
        firing = np.where(activity > threshold)[0]
        for i in firing:
            for j, w in adjacency[i]:
                new_activity[j] += gain * w * activity[i]
        activity = np.clip(new_activity, 0, 3)
        history.append(activity.copy())

    return np.array(history)


def simulate_self_stimulation(adjacency, seed_idx, steps=60,
                               decay=0.85, threshold=0.5, gain=1.3,
                               stim_interval=8, stim_amplitude=1.0):
    """
    Same propagation model as `simulate_realtime_activity`, but instead of a
    single initial pulse the seed node re-injects current into itself every
    `stim_interval` steps -- a coarse model of a repeatedly-pulsing
    stimulating electrode (the same idea used in deep-brain-stimulation
    modeling), rather than a one-off triggering event.
    """
    n = len(adjacency)
    activity = np.zeros(n, dtype=np.float32)
    activity[seed_idx] = stim_amplitude
    history = [activity.copy()]

    for t in range(steps):
        new_activity = activity * decay
        firing = np.where(activity > threshold)[0]
        for i in firing:
            for j, w in adjacency[i]:
                new_activity[j] += gain * w * activity[i]

        # periodic self-stimulation pulse at the seed node
        if (t + 1) % stim_interval == 0:
            new_activity[seed_idx] += stim_amplitude

        activity = np.clip(new_activity, 0, 3)
        history.append(activity.copy())

    return np.array(history)


# ---------------------------------------------------------------------------
# 4. Beam search for the strongest signal path (mirrors the original
#    beam + heapq + linked-list-path pattern, with a real score)
# ---------------------------------------------------------------------------
def _materialize(node):
    out = []
    while node is not None:
        val, node = node
        out.append(val)
    out.reverse()
    return out


def largest_component_endpoints(adjacency, coords):
    """
    Pick a seed/target pair guaranteed to be connected, by BFS-ing out the
    largest connected component and returning the two nodes in it that are
    farthest apart spatially. Real anatomical data can have isolated or
    weakly-connected voxels (noise, edge artifacts) that a naive "first and
    last node" choice can accidentally land on.
    """
    n = len(adjacency)
    unvisited = set(range(n))
    best_component = []

    while unvisited:
        start = next(iter(unvisited))
        component = []
        queue = deque([start])
        unvisited.discard(start)
        while queue:
            u = queue.popleft()
            component.append(u)
            for v, _ in adjacency[u]:
                if v in unvisited:
                    unvisited.discard(v)
                    queue.append(v)
        if len(component) > len(best_component):
            best_component = component

    comp_coords = np.array([coords[i] for i in best_component])
    # farthest-apart pair, approximated: pick one extreme, then the node
    # farthest from it (exact all-pairs search is unnecessary here)
    anchor = best_component[np.argmin(comp_coords[:, 0])]
    anchor_coord = np.array(coords[anchor], dtype=float)
    dists = np.linalg.norm(comp_coords - anchor_coord, axis=1)
    far = best_component[int(np.argmax(dists))]

    return anchor, far, len(best_component)


def beam_search_signal_path(adjacency, coords, seed_idx, target_idx,
                             max_hops=25, beam_width=200):
    target_coord = np.array(coords[target_idx], dtype=float)
    beam = [(0.0, seed_idx, (seed_idx, None))]  # (score, node, path)

    for hop in range(1, max_hops + 1):
        for score, node, path in beam:
            if node == target_idx:
                return _materialize(path), hop - 1

        candidates = []
        for score, node, path in beam:
            for nb, w in adjacency[node]:
                nb_coord = np.array(coords[nb], dtype=float)
                progress = -np.linalg.norm(target_coord - nb_coord)
                new_score = score + w + 0.05 * progress
                candidates.append((new_score, nb, (nb, path)))

        if not candidates:
            break

        # heapq-based pruning, same role as the original script's beam cut
        best = heapq.nlargest(beam_width, candidates, key=lambda t: t[0])
        dedup = {}
        for s, node, path in best:
            if node not in dedup or s > dedup[node][0]:
                dedup[node] = (s, node, path)
        beam = list(dedup.values())

    for score, node, path in beam:
        if node == target_idx:
            return _materialize(path), max_hops
    return None, max_hops


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Neural Circuit Simulation on Real MRI Volume ===\n")

    volume = load_real_mri_volume()
    print(f"Loaded and downsampled real MRI to shape {volume.shape}")
    coords = build_connectivity_nodes(volume)
    print(f"Volume shape: {volume.shape}, active graph nodes: {len(coords)}")

    n_edges = write_connectivity_edges(volume, coords)
    print(f"Streamed {n_edges} directed edges to disk "
          f"({os.path.getsize(EDGES_PATH)/1024:.1f} KB)")

    adjacency = load_adjacency_from_disk(EDGES_PATH, len(coords))

    seed_idx, target_idx, comp_size = largest_component_endpoints(adjacency, coords)
    print(f"Largest connected component: {comp_size}/{len(coords)} nodes")
    print(f"Seed node: {coords[seed_idx]}  Target node: {coords[target_idx]}")

    print("\nRunning real-time activation propagation (single pulse)...")
    # Default params (decay=0.85, gain=1.3, threshold=0.5) sit in a
    # supercritical regime where any pulse cascades to saturate the whole
    # graph -- self-stimulation pulses would just get swamped and show no
    # visible effect. These params sit in a subcritical regime instead: a
    # single pulse dies out completely, so any sustained activity we see
    # is attributable to the self-stimulation itself.
    sim_params = dict(decay=0.6, gain=0.4, threshold=0.55)
    history = simulate_realtime_activity(adjacency, seed_idx, steps=60, **sim_params)
    total_activity = history.sum(axis=1)

    print("Running self-stimulation propagation (periodic pulses at seed)...")
    stim_history = simulate_self_stimulation(adjacency, seed_idx, steps=60,
                                              stim_interval=6, stim_amplitude=1.0, **sim_params)
    stim_total_activity = stim_history.sum(axis=1)
    print(f"Single pulse dies out to {total_activity[-1]:.3f}; "
          f"self-stimulation sustains a periodic {stim_total_activity[-1]:.3f}-level oscillation.")

    # Manhattan distance in stride-units tells us the minimum hops needed,
    # so the beam search is given enough budget to actually reach the target.
    dz, dy, dx = (abs(a - b) for a, b in zip(coords[seed_idx], coords[target_idx]))
    min_hops = (dz + dy + dx) // STRIDE

    print("Running beam search for the strongest signal path...")
    # beam_width=200: on real anatomical data the graph is close to a plain
    # 3D grid, so the true shortest path is thin and a narrow beam (60, the
    # value that worked on the synthetic phantom) prunes it away before it
    # can reach the target. 200 was the smallest width in testing that
    # reliably recovers the true shortest path here.
    path_idx, hops = beam_search_signal_path(
        adjacency, coords, seed_idx, target_idx,
        max_hops=min_hops + 20, beam_width=200,
    )
    if path_idx:
        print(f"Found path in {hops} hops ({len(path_idx)} nodes).")
    else:
        print("No path found within hop budget.")

    # --- visualize in true 3D (coords are already (z, y, x) triples; the
    # graph, propagation and beam search were all 3D already -- only the
    # old rendering flattened it onto one slice) ---
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d proj)

    coords_arr = np.array(coords, dtype=float)
    intensities = np.array([volume[tuple(c)] for c in coords])
    # A threshold cascade like this one saturates almost the entire graph
    # once it fully settles (final step: ~every node at max activation), so
    # coloring by the *final* state is visually uninformative -- it's just
    # one flat color. The propagating wavefront partway through is what
    # actually shows the circuit "lighting up" in real time.
    # A single pulse in this subcritical regime dies out almost immediately,
    # so there's nothing interesting to color by. The self-stimulation run
    # sustains a real, ongoing local firing zone -- that's the wavefront
    # worth visualizing here.
    # Pick the snapshot from the later, steady-state portion of the run
    # (skip the first half, which is just the initial transient) so we
    # capture a peak of the settled self-stimulation oscillation rather
    # than the very first pulse.
    later_half = stim_total_activity[len(stim_total_activity) // 2:]
    snapshot_step = len(stim_total_activity) // 2 + int(np.argmax(later_half))
    snapshot_activity = stim_history[snapshot_step]
    print(f"Coloring 3D snapshot at step {snapshot_step}/{len(stim_history)-1} "
          f"of the self-stimulated (not single-pulse) run.")

    # subsample the point cloud for a legible, fast-to-render scatter
    rng = np.random.default_rng(0)
    n_show = min(4000, len(coords))
    sample = rng.choice(len(coords), size=n_show, replace=False)

    fig = plt.figure(figsize=(15, 5.5))

    # Panel 1: the 3D structural graph (node cloud), colored by MRI intensity
    ax0 = fig.add_subplot(1, 3, 1, projection="3d")
    ax0.scatter(coords_arr[sample, 2], coords_arr[sample, 1], coords_arr[sample, 0],
                c=intensities[sample], cmap="gray", s=3, alpha=0.5)
    ax0.set_title("3D structural graph (from real MRI)")
    ax0.set_xlabel("x"); ax0.set_ylabel("y"); ax0.set_zlabel("z")

    # Panel 2: real-time activity (this is a genuinely scalar, not spatial,
    # quantity -- total activation summed across the whole 3D circuit)
    ax1 = fig.add_subplot(1, 3, 2)
    ax1.plot(total_activity, label="single pulse")
    ax1.plot(stim_total_activity, label="self-stimulation (periodic pulses)")
    ax1.set_title("Total network activity: single pulse vs. self-stimulation")
    ax1.set_xlabel("time step")
    ax1.set_ylabel("summed activation")
    ax1.legend(fontsize=8)

    # Panel 3: the beam-searched circuit path traced through 3D space, with
    # the node cloud as dim spatial context. The self-stimulation regime is
    # subcritical, so only a handful of nodes are ever active -- explicitly
    # highlight them (they'd almost never land in a random background
    # subsample otherwise).
    ax2 = fig.add_subplot(1, 3, 3, projection="3d")
    ax2.scatter(coords_arr[sample, 2], coords_arr[sample, 1], coords_arr[sample, 0],
                c="lightgray", s=2, alpha=0.15)
    active_idx = np.where(snapshot_activity > 0)[0]
    if len(active_idx):
        ax2.scatter(coords_arr[active_idx, 2], coords_arr[active_idx, 1], coords_arr[active_idx, 0],
                    c=snapshot_activity[active_idx], cmap="autumn_r", s=80,
                    edgecolors="black", linewidths=0.5, label=f"active ({len(active_idx)} nodes)")
    if path_idx:
        pc = coords_arr[path_idx]
        ax2.plot(pc[:, 2], pc[:, 1], pc[:, 0], "r-", linewidth=2)
        ax2.scatter(*pc[0, [2, 1, 0]], c="lime", s=60, label="seed")
        ax2.scatter(*pc[-1, [2, 1, 0]], c="red", s=60, label="target")
        ax2.legend(loc="upper left")
        ax2.set_title(f"3D circuit path + self-stimulation wavefront (step {snapshot_step})")
    else:
        ax2.set_title("No path found")
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")

    plt.tight_layout()
    out_path = "neural_circuit_simulation.png"
    plt.savefig(out_path, dpi=140)
    print(f"\nSaved visualization to {out_path}")
