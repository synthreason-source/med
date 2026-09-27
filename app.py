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
    from nilearn import plotting, datasets
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
        self.points_path = "connectivity_points.npy"
        self.edges_path = "connectivity_edges.bin"

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

    def show_cortical_overlay(self):
        """Renders brain mesh, circuit paths, and scientific text labels without marker balls."""
        if not PLOTLY_AVAILABLE:
            messagebox.showerror("Missing Dependency", "Plotly is required. Run: pip install plotly")
            return

        if not os.path.exists(self.points_path) or not os.path.exists(self.edges_path):
            messagebox.showerror(
                "Files Missing", 
                f"Could not find required surface files:\n- {self.points_path}\n- {self.edges_path}"
            )
            return

        if not self.latest_results:
            messagebox.showinfo("Empty Results", "Please run the simulation first to generate paths.")
            return

        self.status_var.set("Status: Loading brain surface, labels, and circuit paths...")
        self.root.update_idletasks()

        try:
            points = np.load(self.points_path).astype(np.float64)
            raw_edges = np.fromfile(self.edges_path, dtype=np.int32)
            
            num_triangles = raw_edges.size // 3
            if num_triangles == 0:
                raise ValueError("connectivity_edges.bin contains no valid triangle data.")
            
            edges = raw_edges[:num_triangles * 3].reshape((-1, 3))
            
            if edges.min() >= 1:
                edges -= 1
                
        except Exception as e:
            messagebox.showerror("File Parsing Error", f"Failed to load binary mesh files:\n{e}")
            return

        if np.max(np.abs(points)) < 5.0:
            points *= 1000.0

        max_idx = len(points) - 1
        valid_mask = (
            (edges[:, 0] >= 0) & (edges[:, 0] <= max_idx) &
            (edges[:, 1] >= 0) & (edges[:, 1] <= max_idx) &
            (edges[:, 2] >= 0) & (edges[:, 2] <= max_idx)
        )
        edges = edges[valid_mask]

        traces = []

        # 1. Real Anatomical Brain Surface Mesh
        brain_mesh = go.Mesh3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            i=edges[:, 0], j=edges[:, 1], k=edges[:, 2],
            color='rgba(135, 206, 250, 0.2)',  
            opacity=0.25,
            lighting=dict(ambient=0.7, diffuse=0.6, specular=0.2),
            name='Anatomical Brain Mesh',
            hoverinfo='skip'
        )
        traces.append(brain_mesh)

        # 2. Add Scientific Region Labels (Sampled for clean readability without marker balls)
        sample_step = max(1, len(points) // 60)
        sampled_points = points[::sample_step]
        scientific_labels = [
            self._get_scientific_region_name(pt[0], pt[1], pt[2]) for pt in sampled_points
        ]

        labels_trace = go.Scatter3d(
            x=sampled_points[:, 0], y=sampled_points[:, 1], z=sampled_points[:, 2],
            mode='text',
            text=scientific_labels,
            textfont=dict(color='Black', size=10, family='Arial Bold'),
            name='Scientific Region Labels'
        )
        traces.append(labels_trace)

        def create_pipe_mesh(p1, p2, radius=4.5, n_segs=10):
            p1, p2 = np.array(p1), np.array(p2)
            v = p2 - p1
            length = np.linalg.norm(v)
            if length < 1e-6:
                return np.empty((0, 3)), np.empty((0, 3), dtype=int)
            v = v / length
            
            n1 = np.array([0, 1, 0]) if abs(v[0]) > 0.9 else np.array([1, 0, 0])
            n1 = np.cross(v, n1)
            n1 /= np.linalg.norm(n1)
            n2 = np.cross(v, n1)
            
            theta = np.linspace(0, 2 * np.pi, n_segs, endpoint=False)
            circle = [radius * (np.cos(th) * n1 + np.sin(th) * n2) for th in theta]
            
            verts = [p1 + pt for pt in circle] + [p2 + pt for pt in circle]
            verts = np.array(verts)
            
            faces = []
            for i in range(n_segs):
                nxt = (i + 1) % n_segs
                faces.append([i, nxt, i + n_segs])
                faces.append([nxt, nxt + n_segs, i + n_segs])
                
            return verts, np.array(faces, dtype=int)

        colorscales = ['Turbo', 'Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis']
        num_points = len(points)

        # 3. Render all optimal circuit paths
        for rank_idx, res in enumerate(self.latest_results):
            target_path = res[2]
            if not target_path or len(target_path) < 2:
                continue

            circuit_coords = []
            path_weights = []
            for idx, weight in enumerate(target_path):
                mapped_idx = (idx * 19 + rank_idx * 7) % num_points
                circuit_coords.append(points[mapped_idx])
                path_weights.append(float(weight))

            all_verts = []
            all_faces = []
            all_intensities = []
            vertex_offset = 0

            for i in range(len(circuit_coords) - 1):
                p1 = circuit_coords[i]
                p2 = circuit_coords[i + 1]
                seg_weight = (path_weights[i] + path_weights[i+1]) / 2.0
                
                v_mesh, f_mesh = create_pipe_mesh(p1, p2, radius=5.0 - (rank_idx * 0.2), n_segs=10)
                if len(v_mesh) > 0:
                    all_verts.append(v_mesh)
                    all_faces.append(f_mesh + vertex_offset)
                    all_intensities.extend([seg_weight] * len(v_mesh))
                    vertex_offset += len(v_mesh)

            if all_verts:
                combined_verts = np.vstack(all_verts)
                combined_faces = np.vstack(all_faces)
                combined_intensities = np.array(all_intensities)

                cscale = colorscales[rank_idx % len(colorscales)]
                pipe_trace = go.Mesh3d(
                    x=combined_verts[:, 0], y=combined_verts[:, 1], z=combined_verts[:, 2],
                    i=combined_faces[:, 0], j=combined_faces[:, 1], k=combined_faces[:, 2],
                    intensity=combined_intensities,
                    colorscale=cscale,
                    opacity=0.95,
                    lighting=dict(ambient=0.5, diffuse=0.9, specular=0.6),
                    name=f'Path Rank #{rank_idx+1}'
                )
                traces.append(pipe_trace)

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=dict(text="Anatomical Brain Surface with Scientific Labels & Circuits"),
            scene=dict(
                xaxis=dict(visible=True),
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