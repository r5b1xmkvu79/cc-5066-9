"""Local texture coordinates constructed from the solved direction field.

Two declared construction methods are provided:

``seam``
    A similarity (rotation + uniform scale) per triangle, propagated over a
    spanning tree of the dual graph.  The texture direction is satisfied
    exactly on every face and there is no non-uniform scale; the price is a
    discontinuous seam on each edge omitted from the tree.  Omitted edges are
    chosen where the direction field itself jumps (singularities / high
    mismatch), so seams land exactly where a continuous direction cannot
    exist.

``global``
    One continuous (u, v) pair per vertex obtained by a tangent-plane least
    squares fit.  Texture is globally continuous with no seams, but triangles
    must shear/scale and locally deviate from the designed direction.

The mesh vertices are never moved: only (u, v) coordinates are produced.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import lsqr, splu
from scipy.sparse.csgraph import minimum_spanning_tree

from .field import DirectionField


@dataclass
class UVResult:
    method: str
    uv: np.ndarray            # (V, 2) for global; (F, 3, 2) per-corner for seam
    per_corner: bool         # True => uv indexed [face, corner], False [vertex]
    seams: np.ndarray         # edge ids that are cut (only meaningful for seam)
    face_rotation: np.ndarray # rotation of the texture frame vs face frame, rad
    face_scale: np.ndarray    # uniform scale (seam) or mean scale (global)
    notes: str


def _local_coords(mesh, dfield, fi: int, vi: int) -> np.ndarray:
    d = mesh.vertices[vi] - mesh.face_centers()[fi]
    return np.array([np.dot(d, dfield.e1[fi]), np.dot(d, dfield.e2[fi])])


def _face_xy(mesh, dfield, fi, vi, beta, s, t):
    x, y = _local_coords(mesh, dfield, fi, vi)
    c, sn = np.cos(beta), np.sin(beta)
    return s * np.array([c * x - sn * y, sn * x + c * y]) + t


# ---------------------------------------------------------------------- #
# Method 1: seam-cut similarity coordinates
# ---------------------------------------------------------------------- #
def build_seam_uv(dfield: DirectionField, world_scale: float = 1.0) -> UVResult:
    mesh = dfield.mesh
    F = mesh.n_faces
    # Cost of keeping an edge continuous: the transported direction jump.
    # Singular neighbourhoods have large jumps and get cut by the MST.
    interior = np.array([eid for eid in range(mesh.edges.shape[0])
                         if mesh.edge_faces[eid, 1] >= 0], dtype=np.int64)
    period = dfield.period
    f0_all = mesh.edge_faces[interior, 0]
    f1_all = mesh.edge_faces[interior, 1]
    jump = np.abs((dfield.angles[f1_all] - dfield.angles[f0_all]
                   - dfield.rho[interior] + period / 2) % period
                  - period / 2) + 1e-6
    # One undirected dual-graph row per interior edge, deduplicated by the
    # unordered face pair (csr_matrix silently overwrites duplicate keys).
    pair_cost: dict[tuple[int, int], float] = {}
    for a, b, c in zip(f0_all, f1_all, jump):
        pair_cost[(int(a), int(b))] = float(c)
    rows = np.array([p[0] for p in pair_cost], dtype=np.int64)
    cols = np.array([p[1] for p in pair_cost], dtype=np.int64)
    vals = np.array(list(pair_cost.values()), dtype=np.float64)
    dual = csr_matrix((vals, (rows, cols)), shape=(F, F))
    dual = dual.maximum(dual.T)   # undirected dual graph
    tree = minimum_spanning_tree(dual).tocoo()
    tree_edges: set[int] = set()
    for i, j in zip(tree.row, tree.col):
        eid = mesh.edge_between_faces(int(i), int(j))
        if eid >= 0:
            tree_edges.add(eid)
    if len(tree_edges) != F - 1:
        raise RuntimeError("dual spanning tree failed; mesh must be a "
                           "connected manifold without disconnected parts")
    seams = np.array(sorted(set(interior.tolist()) - tree_edges), dtype=np.int64)

    # Per-face 2-D similarity: p2d = s * R(beta) * (local coords) + t.
    # The texture x-axis must follow the solved direction on every face, so
    # beta[fi] = -angle[fi].  The solved angles already live on one real
    # cover that includes parallel transport, so no extra rotation is
    # propagated; only the translation t is stitched across tree edges.
    beta = -dfield.angles.copy()
    scale = np.full(F, float(world_scale))
    txy = np.zeros((F, 2))
    parent = np.full(F, -1, dtype=np.int64)
    root = 0
    parent[root] = root

    adjacency: dict[int, list[tuple[int, int]]] = {i: [] for i in range(F)}
    for eid in tree_edges:
        f0, f1 = mesh.edge_faces[eid]
        adjacency[f0].append((f1, eid))
        adjacency[f1].append((f0, eid))

    stack = [root]
    while stack:
        f0 = stack.pop()
        for f1, eid in adjacency[f0]:
            if parent[f1] != -1:
                continue
            # Match one shared endpoint's 2-D position to fix translation.
            u, _ = mesh.edges[eid]
            p0 = _face_xy(mesh, dfield, f0, u, beta[f0], scale[f0], txy[f0])
            p1 = _face_xy(mesh, dfield, f1, u, beta[f1], scale[f1],
                          np.zeros(2))
            txy[f1] = p0 - p1
            parent[f1] = f0
            stack.append(f1)
    if np.any(parent < 0):
        raise RuntimeError("dual graph disconnected; mesh must be connected")

    uv_corner = np.zeros((F, 3, 2))
    for fi in range(F):
        for slot in range(3):
            uv_corner[fi, slot] = _face_xy(mesh, dfield, fi,
                                           mesh.faces[fi, slot],
                                           beta[fi], scale[fi], txy[fi])
    return UVResult("seam", uv_corner, True, seams, beta, scale,
                    "similarity coordinates over a dual spanning tree; "
                    "direction exact, cut edges carry the seam")


# ---------------------------------------------------------------------- #
# Method 2: global continuous least-squares UV
# ---------------------------------------------------------------------- #
def build_global_uv(dfield: DirectionField) -> UVResult:
    """Continuous per-vertex UV via edge-vector (gradient) least squares.

    For each triangle we know exactly where its two edge vectors should land
    in the texture plane from the solved frame and true 3-D edge lengths.
    Constraining *differences* (uv_j - uv_i) rather than absolute positions
    lets shared vertices agree; only the global translation is free, and we
    pin one vertex to remove it.  The continuous solution necessarily shears
    and deviates in direction near singularities -- that is the documented
    trade-off versus the seam method.
    """
    mesh = dfield.mesh
    V, F = mesh.n_vertices, mesh.n_faces
    # Per-face 2x2 map from face-frame coordinates to texture coordinates:
    # texture +u follows the solved direction, preserving edge lengths.
    face_rot = np.zeros(F)
    rows: list[float] = []
    cols: list[int] = []
    data: list[float] = []
    b_u: list[float] = []
    b_v: list[float] = []
    r = 0

    for fi in range(F):
        beta = -dfield.angles[fi]
        face_rot[fi] = beta
        c_, s_ = np.cos(beta), np.sin(beta)
        frame = np.column_stack([dfield.e1[fi], dfield.e2[fi]])
        tri = mesh.faces[fi]
        # Two independent edge vectors leaving local vertex 0.
        for k in (1, 2):
            e3 = mesh.vertices[tri[k]] - mesh.vertices[tri[0]]
            local = frame.T @ e3                 # coords in face frame
            tu = c_ * local[0] - s_ * local[1]
            tv = s_ * local[0] + c_ * local[1]
            # uv[tri[k]] - uv[tri[0]] = (tu, tv)
            rows.append(r); cols.append(tri[k]); data.append(1.0)
            rows.append(r); cols.append(tri[0]); data.append(-1.0)
            b_u.append(tu); b_v.append(tv)
            r += 1

    a_mat = csr_matrix((data, (rows, cols)), shape=(r, V))
    ata = (a_mat.T @ a_mat).tolil()
    pinned = int(mesh.faces[0, 0])
    pin_w = float(ata[pinned, pinned]) * 1e6 + 1.0
    ata[pinned, pinned] += pin_w
    ata = ata.tocsc()
    rhs_u = a_mat.T @ np.array(b_u)
    rhs_v = a_mat.T @ np.array(b_v)
    rhs_u[pinned] += pin_w * 0.0
    rhs_v[pinned] += pin_w * 0.0
    factor = splu(ata)
    u = factor.solve(rhs_u)
    v = factor.solve(rhs_v)
    uv = np.column_stack([u, v])

    scale = _mean_scale(mesh, uv)
    return UVResult("global", uv, False, np.array([], dtype=np.int64),
                    face_rot, scale,
                    "continuous per-vertex least-squares UV; no seams, "
                    "local direction deviates near singularities")


def _mean_scale(mesh, uv) -> np.ndarray:
    """Per-triangle mean metric scale, 1.0 means no length change."""
    out = np.ones(mesh.n_faces)
    for fi, tri in enumerate(mesh.faces):
        p3 = mesh.vertices[tri]
        p2 = uv[tri]
        l3 = sum(np.linalg.norm(p3[(i + 1) % 3] - p3[i]) ** 2 for i in range(3))
        l2 = sum(np.linalg.norm(p2[(i + 1) % 3] - p2[i]) ** 2 for i in range(3))
        if l3 > 1e-20:
            out[fi] = np.sqrt(l2 / l3)
    return out
