# make_connectivity_points.py
from pathlib import Path
import mmap
import struct
import numpy as np

EDGE = struct.Struct("<IIf")
edge_path = Path("connectivity_edges.bin")
output_path = Path("connectivity_points.npy")

with edge_path.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
    n_records = len(mm) // EDGE.size
    records = [
        EDGE.unpack_from(mm, offset * EDGE.size)
        for offset in range(n_records)
    ]

n_nodes = 1 + max(
    max(source, target)
    for source, target, weight in records
)

rng = np.random.default_rng(4)
points = rng.normal(size=(n_nodes, 3)).astype(np.float32)
points /= np.linalg.norm(points, axis=1, keepdims=True) + 1e-8
points *= 180.0

np.save(output_path, points)

print(f"created {output_path}")
print(f"nodes: {n_nodes}")
