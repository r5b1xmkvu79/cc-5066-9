"""Triangle mesh container with connectivity and OBJ I/O.

The geometry stored here is the *input* geometry. Every later stage
(direction field, UV, rendering) reads these vertices but never moves them,
so texture direction is achieved by solving the field rather than deforming
the surface.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TriangleMesh:
    vertices: np.ndarray   # (V, 3) float64, world-space positions, never mutated
    faces: np.ndarray      # (F, 3) int64, vertex indices, CCW outward orientation

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        self.faces = np.asarray(self.faces, dtype=np.int64)
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("only triangle meshes are supported")
        self._build_connectivity()

    # ------------------------------------------------------------------ #
    @property
    def n_vertices(self) -> int:
        return self.vertices.shape[0]

    @property
    def n_faces(self) -> int:
        return self.faces.shape[0]

    def _build_connectivity(self) -> None:
        """Edges, face adjacency, per-face vertex lists and boundary info."""
        f = self.faces
        # Half-edges are concatenated face by face, slot by slot, so the
        # flattened index of (face, slot) is exactly 3*face + slot.  Each
        # half-edge leaves f[face, slot] toward the next vertex in the CCW
        # triangle.
        a = np.empty(3 * self.n_faces, dtype=np.int64)
        b = np.empty_like(a)
        opp = np.empty_like(a)
        for slot in range(3):
            idx = np.arange(self.n_faces) * 3 + slot
            a[idx] = f[:, slot]
            b[idx] = f[:, (slot + 1) % 3]
            opp[idx] = f[:, (slot + 2) % 3]
        keys = np.minimum(a, b) * self.n_vertices + np.maximum(a, b)

        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        # Start index of each distinct undirected edge in the sorted order.
        starts = np.concatenate(([0],
                                 np.flatnonzero(np.diff(sorted_keys)) + 1,
                                 [len(sorted_keys)]))
        n_edges = len(starts) - 1
        edges = np.empty((n_edges, 2), dtype=np.int64)
        edge_id_of_half = np.empty(len(a), dtype=np.int64)
        ef = np.full((n_edges, 2), -1, dtype=np.int64)
        eopp = np.full((n_edges, 2), -1, dtype=np.int64)
        boundary = np.zeros(n_edges, dtype=bool)
        for eid in range(n_edges):
            grp = order[starts[eid]:starts[eid + 1]]
            if grp.size > 2:
                raise ValueError("non-manifold edge with >2 incident faces")
            h0 = grp[0]
            edges[eid] = (min(a[h0], b[h0]), max(a[h0], b[h0]))
            for slot_i, hi in enumerate(grp):
                edge_id_of_half[hi] = eid
                ef[eid, slot_i] = hi // 3
                eopp[eid, slot_i] = opp[hi]
            if grp.size == 1:
                boundary[eid] = True
        self.edges = edges
        self.edge_id_of_halfedge = edge_id_of_half
        self.edge_faces = ef
        self.edge_opposite = eopp
        self.boundary_edges = np.flatnonzero(boundary)
        # Vertex -> incident faces
        self.vertex_faces: list[list[int]] = [[] for _ in range(self.n_vertices)]
        for fi, tri in enumerate(f):
            for vi in tri:
                self.vertex_faces[int(vi)].append(fi)
        self.vertex_edges: list[list[int]] = [[] for _ in range(self.n_vertices)]
        for eid, (u, v) in enumerate(edges):
            self.vertex_edges[u].append(eid)
            self.vertex_edges[v].append(eid)

    def halfedge_edge(self, face: int, slot: int) -> int:
        """Edge id of the half-edge leaving local vertex ``slot``."""
        return int(self.edge_id_of_halfedge[3 * face + slot])

    def face_edges(self, face: int) -> tuple[int, int, int]:
        """The three edge ids bordering a face, in half-edge slot order."""
        return tuple(int(self.edge_id_of_halfedge[3 * face + s])
                     for s in range(3))

    def shared_edge(self, f0: int, f1: int) -> int:
        """Edge id shared by two adjacent faces (-1 if not adjacent)."""
        v1 = set(self.faces[f1].tolist())
        for eid in self.face_edges(f0):
            a, b = self.edges[eid]
            if a in v1 and b in v1:
                return eid
        return -1

    def edge_between_faces(self, f0: int, f1: int) -> int:
        """Look up the interior edge whose two incident faces are f0, f1."""
        table = getattr(self, "_face_pair_edge", None)
        if table is None:
            table = {}
            for eid, (a, b) in enumerate(self.edge_faces):
                if b >= 0:
                    table[(int(a), int(b))] = eid
                    table[(int(b), int(a))] = eid
            self._face_pair_edge = table
        return table.get((int(f0), int(f1)), -1)

    def face_centers(self) -> np.ndarray:
        return self.vertices[self.faces].mean(axis=1)

    def is_boundary_vertex(self, vi: int) -> bool:
        return bool(any(e in set(self.boundary_edges.tolist())
                        for e in self.vertex_edges[vi]))

    def boundary_vertex_mask(self) -> np.ndarray:
        be = set(self.boundary_edges.tolist())
        mask = np.zeros(self.n_vertices, dtype=bool)
        for vi in range(self.n_vertices):
            if any(e in be for e in self.vertex_edges[vi]):
                mask[vi] = True
        return mask

    def edge_vector(self, eid: int) -> np.ndarray:
        u, v = self.edges[eid]
        return self.vertices[v] - self.vertices[u]

    # ------------------------------------------------------------------ #
    @classmethod
    def from_obj(cls, path: str) -> "TriangleMesh":
        verts: list[list[float]] = []
        faces: list[list[int]] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if not parts or parts[0] == "#":
                    continue
                if parts[0] == "v":
                    verts.append([float(x) for x in parts[1:4]])
                elif parts[0] == "f":
                    idx = []
                    for token in parts[1:]:
                        token = token.split("/", 1)[0]
                        k = int(token)
                        idx.append(k - 1 if k > 0 else k + len(verts))
                    if len(idx) != 3:
                        raise ValueError("OBJ face must be a triangle; "
                                         "triangulate the input first")
                    faces.append(idx)
        if not faces:
            raise ValueError(f"no triangle faces found in {path}")
        return cls(np.array(verts), np.array(faces))

    def to_obj(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("# written by texorient; vertex geometry is unchanged "
                     "from the input mesh\n")
            for v in self.vertices:
                fh.write(f"v {v[0]:.9f} {v[1]:.9f} {v[2]:.9f}\n")
            for tri in self.faces:
                fh.write("f {} {} {}\n".format(*(tri + 1)))
