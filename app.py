import heapq
import threading
import queue
import time
import os
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

try:
    import nibabel as nib
    from skimage import measure
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class NeuralBeamSimulationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Beam Engine & Custom Connectivity Visualizer")
        self.root.geometry("800x650")
        self.root.minsize(700, 550)

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

    def _ensure_data_files(self):
        """Creates default files if weights.bin doesn't exist."""
        if not os.path.exists(self.data_path):
            raw_weights = np.array([1.5, 2.2, 3.8, 4.1, 5.0, 6.3, 1.1], dtype=np.float32)
            raw_weights.tofile(self.data_path)

    def _create_widgets(self):
        main_frame = ttk.Frame(self.root, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        title_label = ttk.Label(
            main_frame, text="Neural Beam Engine with Custom Connectivity Files", font=("Arial", 14, "bold")
        )
        title_label.pack(pady=(0, 10))

        control_frame = ttk.LabelFrame(main_frame, text="Simulation Parameters", padding=10)
        control_frame.pack(fill=tk.X, pady=5)

        ttk.Label(control_frame, text="Target Sum:").grid(row=0, column=0, sticky="w", pady=5)
        self.target_entry = ttk.Entry(control_frame, width=12)
        self.target_entry.insert(0, "7.0")
        self.target_entry.grid(row=0, column=1, sticky="w", padx=10, pady=5)

        ttk.Label(control_frame, text="Beam Width:").grid(row=1, column=0, sticky="w", pady=5)
        self.beam_entry = ttk.Entry(control_frame, width=12)
        self.beam_entry.insert(0, "4")
        self.beam_entry.grid(row=1, column=1, sticky="w", padx=10, pady=5)

        self.run_button = ttk.Button(control_frame, text="Run Simulation", command=self.start_simulation)
        self.run_button.grid(row=0, column=2, rowspan=2, padx=15, ipadx=5, ipady=10)

        self.mri_button = ttk.Button(
            control_frame, 
            text="Render Circuits & Labels (3D)", 
            command=self.show_cortical_overlay,
            state=tk.DISABLED
        )
        self.mri_button.grid(row=0, column=3, rowspan=2, padx=10, ipadx=5, ipady=10)

        self.status_var = tk.StringVar(value="Status: Ready")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, font=("Arial", 10, "italic"))
        status_label.pack(anchor="w", pady=(10, 5))

        results_frame = ttk.LabelFrame(main_frame, text="Top Optimal Beam Paths", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        columns = ("rank", "cost", "sum", "path")
        self.tree = ttk.Treeview(results_frame, columns=columns, show="headings", height=8)
        self.tree.heading("rank", text="Rank")
        self.tree.heading("cost", text="Cost Error")
        self.tree.heading("sum", text="Achieved Sum")
        self.tree.heading("path", text="Selected Path Weights")

        self.tree.column("rank", width=50, anchor="center")
        self.tree.column("cost", width=90, anchor="center")
        self.tree.column("sum", width=100, anchor="center")
        self.tree.column("path", width=400, anchor="w")

        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _load_data(self) -> np.ndarray:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
            
        with open(self.data_path, "rb") as f:
            return np.fromfile(f, dtype=np.float32)

    def bounded_beam_search(self, target_sum: float, beam_width: int, weights: np.ndarray):
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
            messagebox.showerror("Invalid Input", f"Please enter valid numeric parameters.\nDetails: {err}")
            return

        for item in self.tree.get_children():
            self.tree.delete(item)

        self.is_running = True
        self.run_button.config(state=tk.DISABLED)
        self.mri_button.config(state=tk.DISABLED)
        self.status_var.set("Status: Running subset-sum beam search...")

        worker_thread = threading.Thread(target=self._worker_loop, args=(target_sum, beam_width))
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
                    self.tree.insert("", tk.END, values=(idx, f"{cost_val:.4f}", f"{sum_val:.4f}", path_val))
                
                if PLOTLY_AVAILABLE:
                    self.mri_button.config(state=tk.NORMAL)
                else:
                    self.status_var.set("Status: Complete. Install 'plotly' to enable 3D visualizer.")
            else:
                self.status_var.set("Status: Simulation failed.")
                messagebox.showerror("Execution Error", data)

        except queue.Empty:
            if self.is_running:
                self.root.after(100, self._check_queue)

    def _get_scientific_region_name(self, x: float, y: float, z: float) -> str:
        """Maps MNI coordinate space to standard neuroanatomical scientific names."""
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
        """
        Same kind of coordinate-bucket heuristic as _get_scientific_region_name,
        just tuned for the deeper structures instead of outer cortex -- not a
        real anatomical segmentation, just a plausible-looking label for a
        given position relative to the brain's center.
        """
        hemisphere = "Right" if x >= 0 else "Left"
        ax, ay, az = abs(x), y, z

        if ax < 10:  # near the midline
            if ay > 15:
                return "Corpus Callosum (Genu)"
            elif ay > -5:
                return "Thalamus"
            elif ay > -25:
                return "Corpus Callosum (Splenium)"
            else:
                return "Brainstem"
        else:  # lateral structures, mirrored left/right
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
        """
        Samples points from well *inside* the tissue mask (not on its outer
        boundary) so labels can be placed for interior/subcortical-style
        structures, not just the outer surface. Erodes the same tissue mask
        used for the surface by ~25 voxels so sampled points sit clearly
        underneath the surface, then randomly subsamples for a manageable
        point count.
        """
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

        sample = sample - np.array(data.shape) / 2.0  # center, same as the surface

        np.save(self.inner_points_path, sample)
        return sample

    def _build_or_load_brain_surface(self):
        """
        Builds a real triangulated brain surface straight from the MRI volume
        (via marching cubes) and caches it to disk. The previous version read
        `connectivity_edges.bin`, but that file stores graph edges
        (src:int32, dst:int32, weight:float32) written by the simulation
        script -- not triangle indices -- so reinterpreting its bytes as
        (i, j, k) mesh faces produced zero valid triangles. This builds an
        actual surface instead.
        """
        if os.path.exists(self.surface_verts_path) and os.path.exists(self.surface_faces_path):
            verts = np.load(self.surface_verts_path)
            faces = np.load(self.surface_faces_path)
            return verts, faces

        if not os.path.exists(self.mri_path):
            raise FileNotFoundError(f"MRI file not found at {self.mri_path}")

        img = nib.load(self.mri_path)
        data = img.get_fdata().astype(np.float32)

        robust_max = np.percentile(data[data > 0], 99)
        volume = np.clip(data / robust_max, 0, 1)

        # step_size=2 keeps the surface at ~30k vertices/60k faces: detailed
        # enough to look like a brain, light enough to render smoothly.
        verts, faces, _normals, _values = measure.marching_cubes(
            volume, level=0.2, step_size=2
        )

        # Center on the volume's midpoint so coordinates run roughly
        # -80..+100 per axis, matching the ranges _get_scientific_region_name
        # expects (it was written assuming roughly MNI-centered coordinates).
        verts = verts - np.array(data.shape) / 2.0

        np.save(self.surface_verts_path, verts)
        np.save(self.surface_faces_path, faces)
        return verts, faces

    def show_cortical_overlay(self):
        """Renders brain mesh, circuit paths, and scientific text labels without marker balls."""
        if not PLOTLY_AVAILABLE:
            messagebox.showerror("Missing Dependency", "Plotly and scikit-image are required. Run: pip install plotly scikit-image")
            return

        if not self.latest_results:
            messagebox.showinfo("Empty Results", "Please run the simulation first to generate paths.")
            return

        self.status_var.set("Status: Building brain surface from MRI (first run only)...")
        self.root.update_idletasks()

        try:
            points, edges = self._build_or_load_brain_surface()
        except Exception as e:
            messagebox.showerror("Surface Build Error", f"Failed to build brain surface:\n{e}")
            return

        self.status_var.set("Status: Loading labels and circuit paths...")
        self.root.update_idletasks()

        traces = []

        # 1. Real Anatomical Brain Surface Mesh
        # Note: the color string previously had alpha baked in
        # ('rgba(...,0.2)') *and* a separate opacity=0.25 on top of it.
        # Mesh3d doesn't reliably honor an alpha channel inside a solid
        # `color` string, and stacking that with `opacity` compounded down
        # to ~0.05 effective opacity -- invisible against a white
        # background, even though the geometry itself was rendering fine
        # (labels and pipes, which don't go through `color`, still showed).
        # A solid color name plus a single opacity value fixes it.
        brain_mesh = go.Mesh3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            i=edges[:, 0], j=edges[:, 1], k=edges[:, 2],
            color='lightskyblue',
            opacity=0.16,
            lighting=dict(ambient=0.7, diffuse=0.6, specular=0.2),
            name='Anatomical Brain Mesh',
            hoverinfo='skip'
        )
        traces.append(brain_mesh)

        # 2. Add Scientific Region Labels. _get_scientific_region_name only
        # ever produces ~28 distinct strings (2 hemispheres x 14 named
        # zones) -- with ~30k mesh vertices, labeling "all of them" would
        # mean stamping the same ~28 words tens of thousands of times on
        # top of each other and would also make the browser choke trying to
        # render that many text objects. Instead this labels every one of
        # those distinct regions exactly once, at a representative point
        # (the vertex closest to that region's own centroid).
        all_labels = np.array([
            self._get_scientific_region_name(pt[0], pt[1], pt[2]) for pt in points
        ])

        mesh_centroid = points.mean(axis=0)
        label_points = []
        label_text = []
        for label in np.unique(all_labels):
            group_idx = np.where(all_labels == label)[0]
            group_pts = points[group_idx]
            centroid = group_pts.mean(axis=0)
            nearest = group_idx[np.argmin(np.linalg.norm(group_pts - centroid, axis=1))]
            anchor = points[nearest]
            # Push the label outward along the mesh's own radial direction so
            # it floats just outside the surface instead of sitting exactly
            # on it -- with 28 labels around one small object, text anchored
            # right at the surface reads as one dense, overlapping cloud.
            # Pushing radially outward spreads near neighbors further apart
            # in screen space and stops the mesh from crowding the text.
            direction = anchor - mesh_centroid
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            label_points.append(anchor + direction * 18.0)
            label_text.append(label)
        label_points = np.array(label_points)

        labels_trace = go.Scatter3d(
            x=label_points[:, 0], y=label_points[:, 1], z=label_points[:, 2],
            mode='text',
            text=label_text,
            textfont=dict(color='Black', size=10, family='Arial Bold'),
            name='Scientific Region Labels'
        )
        traces.append(labels_trace)

        # 2b. Inner/subcortical labels. The outer surface trace only has
        # vertices ON the boundary of the head, so it can only ever label
        # outer-cortex-style regions -- there was nothing to anchor a label
        # like "Thalamus" or "Hippocampus" to, since those sit underneath
        # the surface, not on it. This samples points from well inside the
        # tissue mask instead (see _build_or_load_inner_points) and labels
        # those the same "one representative point per distinct name" way,
        # so labels for deep structures actually sit inside the brain
        # rather than being pushed outward like the surface ones are.
        try:
            inner_points = self._build_or_load_inner_points()
        except Exception as e:
            inner_points = None
            self.status_var.set(f"Status: Inner labels skipped ({e})")

        if inner_points is not None and len(inner_points) > 0:
            inner_all_labels = np.array([
                self._get_inner_structure_name(pt[0], pt[1], pt[2]) for pt in inner_points
            ])
            inner_label_points = []
            inner_label_text = []
            for label in np.unique(inner_all_labels):
                group_idx = np.where(inner_all_labels == label)[0]
                group_pts = inner_points[group_idx]
                centroid = group_pts.mean(axis=0)
                nearest = group_idx[np.argmin(np.linalg.norm(group_pts - centroid, axis=1))]
                inner_label_points.append(inner_points[nearest])
                inner_label_text.append(label)
            inner_label_points = np.array(inner_label_points)

            inner_labels_trace = go.Scatter3d(
                x=inner_label_points[:, 0], y=inner_label_points[:, 1], z=inner_label_points[:, 2],
                mode='text',
                text=inner_label_text,
                textfont=dict(color='darkred', size=10, family='Arial Bold'),
                name='Inner Structure Labels'
            )
            traces.append(inner_labels_trace)

        def create_pipe_mesh(polyline, radius=4.5, n_segs=16, subdivisions=8):
            """Create a genuinely continuous tube around a dense centerline.

            The important part is that the tube is built from a *densified*
            centerline.  The old renderer put a ring only at the sparse solver
            points, so long solver hops could look like broken/segmented pipes
            after perspective projection.  Here every hop is subdivided before
            rings are generated and all adjacent rings share vertices/faces.
            """
            pts = np.asarray(polyline, dtype=float)
            if pts.ndim != 2 or pts.shape[0] < 2:
                return np.empty((0, 3)), np.empty((0, 3), dtype=int)

            # Remove duplicate points.
            cleaned = [pts[0]]
            for pnt in pts[1:]:
                if np.linalg.norm(pnt - cleaned[-1]) > 1e-8:
                    cleaned.append(pnt)
            pts = np.asarray(cleaned, dtype=float)
            if len(pts) < 2:
                return np.empty((0, 3)), np.empty((0, 3), dtype=int)

            # Densify every solver hop.  This is what prevents apparent gaps
            # when the solver path contains relatively distant vertices.
            dense = [pts[0]]
            for a, b in zip(pts[:-1], pts[1:]):
                dist = float(np.linalg.norm(b - a))
                steps = max(2, int(np.ceil(dist / max(radius * 0.55, 1e-6))))
                for k in range(1, steps + 1):
                    u = k / steps
                    dense.append(a * (1.0 - u) + b * u)
            pts = np.asarray(dense, dtype=float)

            # Tangents from neighboring points.
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

            # Stable initial frame.
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

            # Parallel transport keeps neighboring rings aligned without the
            # sudden frame flips that can create visual pinches.
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
                    n1 = n0 * ca + np.cross(axis, n0) * sa + axis * np.dot(axis, n0) * (1.0 - ca)
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
                    ct[:, None] * normals[i][None, :] +
                    st[:, None] * binormals[i][None, :]
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

            # Caps.
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

        colorscales = ['Turbo', 'Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis']
        num_points = len(points)

        from scipy.spatial import cKDTree

        # 3. Render the circuit paths INSIDE the brain tissue.
        # The circuit paths must use the eroded interior point cloud, not the
        # cortical surface.  Using surface vertices here puts every pipe on
        # the outside of the brain and makes the inner network appear absent.
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

                # Deterministic interior seed.  Start away from the outer
                # boundary by selecting from the middle of the eroded cloud.
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

                    # Prefer continuation in the current direction, but also
                    # prefer a meaningful step so the pipe actually travels
                    # through the interior instead of collapsing into one spot.
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
                        mode='edge',
                    )
                elif len(vertex_values) > len(v_mesh):
                    vertex_values = vertex_values[:len(v_mesh)]

                cscale = colorscales[rank_idx % len(colorscales)]
                pipe_trace = go.Mesh3d(
                    x=v_mesh[:, 0], y=v_mesh[:, 1], z=v_mesh[:, 2],
                    i=f_mesh[:, 0], j=f_mesh[:, 1], k=f_mesh[:, 2],
                    intensity=vertex_values,
                    colorscale=cscale,
                    cmin=float(np.min(vertex_values)),
                    cmax=float(np.max(vertex_values)) if np.max(vertex_values) > np.min(vertex_values) else float(np.min(vertex_values) + 1.0),
                    opacity=1.0,
                    flatshading=False,
                    lighting=dict(ambient=0.75, diffuse=1.0, specular=0.9, roughness=0.18),
                    name=f'INNER Path Rank #{rank_idx+1}',
                    hoverinfo='skip',
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
                camera=dict(
                    eye=dict(x=1.6, y=1.6, z=1.3)
                )
            )
        )

        output_file = "labeled_clean_brain.html"
        fig.write_html(output_file, include_plotlyjs=True)
        
        import webbrowser
        webbrowser.open("file://" + os.path.realpath(output_file))

        self.status_var.set(f"Status: Opened labeled clean view ({output_file}).")


if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralBeamSimulationApp(root)
    root.mainloop()
