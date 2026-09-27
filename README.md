# Connectivity-driven electrical neural simulator

The app loads `connectivity_points.npy` and `connectivity_edges.bin` before starting the electrical simulation.

Binary edge format: repeated little-endian records `(uint32 source, uint32 target, float32 weight)`.
Point format: NumPy `.npy` array shaped `(N, 3)`.

Run: `python app.py`. If files are absent, a clearly synthetic demo graph is used.
