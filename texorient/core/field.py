"""Tangent direction-field solver and singularity (holonomy) analysis.

The unknown is one angle per face, measured in that face's own tangent frame.
Neighbouring faces are coupled through the parallel-transport rotation
``rho`` (see :mod:`frames`), so everything is intrinsic to the tangent bundle
-- no 3-D vector averaging is used anywhere.

Periodic cover treatment
------------------------
A stripe (line) field is defined modulo pi and a vector field modulo 2*pi.
On a surface with non-trivial topology or an imposed winding guide, no single
real-valued angle function can represent the field: a real function always
has zero closed-loop integral.  We therefore

1. build a spanning tree of the dual graph and lift angles onto one real
   cover along tree edges (zero period jump there);
2. determine the integer period jump ``k_e`` on every *non-tree* edge by
   snapping the transported, guide-consistent mismatch to the nearest
   multiple of the period;
3. solve a continuous least-squares system whose edge targets already
   contain those integer jumps.

The integer jumps are real topological data.  Summing them around a vertex
fan gives the holonomy and hence the Poincare index; singularities are
reported, never smoothed away.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
from scipy.sparse.csgraph import minimum_spanning_tree

from .frames import angles_to_vectors, tangent_frames, transport_angles
from .mesh import TriangleMesh


@dataclass
class DirectionConstraint:
    face: int
    angle: float          # radians, in the face's local (e1, e2) frame
    weight: float = 10.0
    label: str = ""


@dataclass
class DirectionField:
    mesh: TriangleMesh
    mode: str                      # "vector" (2*pi) or "line" (pi, stripes)
    angles: np.ndarray             # solved lifted local angles, per face
    n: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    rho: np.ndarray                # transport angle per edge (NaN boundary)
    edge_residual: np.ndarray      # wrapped mismatch per interior edge
    edge_jump: np.ndarray          # unwrapped mismatch incl. integer periods
    period_jump: np.ndarray        # integer k on each edge, edge jump = k*p
    constraints: list[DirectionConstraint]
    holonomy: "HolonomyReport"
    vectors: np.ndarray = dc_field(default=None)
    reference_angles: np.ndarray = dc_field(default=None)

    @property
    def period(self) -> float:
        return np.pi if self.mode == "line" else 2.0 * np.pi

    def direction_vectors(self) -> np.ndarray:
        if self.vectors is None:
            self.vectors = angles_to_vectors(self.angles, self.e1, self.e2)
        return self.vectors


@dataclass
class HolonomyReport:
    vertex: np.ndarray
    is_closed_loop: np.ndarray
    total_rotation: np.ndarray     # summed unwrapped mismatch, radians
    index_times2: np.ndarray       # 2 * Poincare index (integer for both)
    residual_rad: np.ndarray       # non-quantised leftover rotation
    singular: np.ndarray           # nonzero index on a closed loop

    def table(self) -> str:
        lines = [f"{'vertex':>7} {'loop':>6} {'holonomy(turn)':>15} "
                 f"{'index':>8} {'residual(deg)':>14}"]
        for v, closed, tot, k2, res in zip(self.vertex, self.is_closed_loop,
                                           self.total_rotation,
                                           self.index_times2,
                                           self.residual_rad):
            tag = "closed" if closed else "open"
            lines.append(
                f"{v:>7d} {tag:>6} {tot / (2 * np.pi):>15.4f} "
                f"{k2 / 2.0:>8.2f} {np.degrees(res):>14.3f}")
        return "\n".join(lines)


def _wrap(x, period: float):
    return (x + period / 2.0) % period - period / 2.0


def solve_direction_field(mesh: TriangleMesh,
                          constraints: list[DirectionConstraint],
                          mode: str = "line") -> DirectionField:
    if mode not in ("line", "vector"):
        raise ValueError("mode must be 'line' (stripes, mod pi) or "
                         "'vector' (mod 2*pi)")
    if not constraints:
        raise ValueError("at least one direction constraint is required; "
                         "a field without any guide has no preferred direction")
    for c in constraints:
        if not (0 <= c.face < mesh.n_faces):
            raise ValueError(f"constraint face {c.face} out of range")
    if len({c.face for c in constraints}) != len(constraints):
        raise ValueError("duplicate constraint on a face")

    n, e1, e2 = tangent_frames(mesh)
    rho = transport_angles(mesh, n, e1, e2)
    period = np.pi if mode == "line" else 2.0 * np.pi

    interior = np.array([eid for eid in range(mesh.edges.shape[0])
                         if mesh.edge_faces[eid, 1] >= 0], dtype=np.int64)
    # ----- dual spanning tree -------------------------------------------
    f0a = mesh.edge_faces[interior, 0]
    f1a = mesh.edge_faces[interior, 1]
    dual = csr_matrix((np.ones(len(interior)), (f0a, f1a)),
                      shape=(mesh.n_faces, mesh.n_faces)).maximum(
        csr_matrix((np.ones(len(interior)), (f1a, f0a)),
                   shape=(mesh.n_faces, mesh.n_faces)))
    mst = minimum_spanning_tree(dual).tocoo()
    tree_edges: set[int] = set()
    adj: dict[int, list[tuple[int, int]]] = {i: [] for i in
                                              range(mesh.n_faces)}
    for i, j in zip(mst.row, mst.col):
        eid = mesh.edge_between_faces(int(i), int(j))
        tree_edges.add(eid)
        adj[int(i)].append((int(j), eid))
        adj[int(j)].append((int(i), eid))

    # ----- transport-only tree lift --------------------------------------
    root = constraints[0].face
    lift = np.zeros(mesh.n_faces)
    parent = np.full(mesh.n_faces, -1, dtype=np.int64)
    parent[root] = root
    stack = [root]
    while stack:
        f0 = stack.pop()
        for f1, eid in adj[f0]:
            if parent[f1] != -1:
                continue
            parent[f1] = f0
            lift[f1] = lift[f0] + (rho[eid]
                                   if mesh.edge_faces[eid, 0] == f0
                                   else -rho[eid])
            stack.append(f1)
    if np.any(parent < 0):
        raise RuntimeError("mesh dual graph is disconnected")

    # ----- reference lifted field over the tree --------------------------
    # The tree lift ``lift`` transports a zero angle from the root, so the
    # frame of face f sits at real angle lift[f] on one cover.  Each guide is
    # first expressed on this cover nearest its own value.  The BFS reference
    # field then walks the tree, transporting between faces and snapping only
    # when a constrained face's guide differs from the transported value by
    # at least half a period.  On a smooth (non-winding) guide every snap
    # difference is below that threshold, so no spurious jump appears; a
    # genuinely winding guide forces the threshold to be crossed, producing
    # exactly its topological integer jumps.
    cw = {c.face: c.angle for c in constraints}
    ref = np.zeros(mesh.n_faces)
    tree_parent = {root: None}
    ref[root] = cw[root]
    queue = [root]
    head = 0
    while head < len(queue):
        f0 = queue[head]
        head += 1
        for f1, eid in adj[f0]:
            if f1 in tree_parent:
                continue
            tree_parent[f1] = f0
            step = (rho[eid] if mesh.edge_faces[eid, 0] == f0 else -rho[eid])
            base = ref[f0] + step
            if f1 in cw:
                diff = cw[f1] - base
                # Only a difference beyond one full period (i.e. the guide is
                # on a different cover sheet) is a real topological jump.
                if abs(diff) > period * 0.5:
                    k = int(round(diff / period))
                    base = cw[f1] - k * period
                else:
                    base = cw[f1]
            ref[f1] = base
            queue.append(f1)

    # ----- integer period jumps read from the reference field ------------
    # Tree edges have zero jump by construction; the non-tree integer jumps
    # are the discrete carriers of topology and winding.
    period_jump = np.zeros(mesh.edges.shape[0], dtype=np.int64)
    edge_target = np.full(mesh.edges.shape[0], np.nan)
    for eid in interior:
        f0, f1 = mesh.edge_faces[eid]
        raw = ref[f1] - ref[f0] - rho[eid]
        k = 0 if eid in tree_edges else int(round(raw / period))
        period_jump[eid] = k
        edge_target[eid] = rho[eid] + k * period

    # ----- continuous relaxation with the integer jumps fixed -----------
    rows: list[float] = []
    cols: list[int] = []
    data: list[float] = []
    rhs: list[float] = []
    r = 0

    def add(cells, value):
        nonlocal r
        for c, d in cells:
            rows.append(r); cols.append(c); data.append(d)
        rhs.append(value); r += 1

    for eid in interior:
        f0, f1 = mesh.edge_faces[eid]
        add([(f0, -1.0), (f1, 1.0)], float(edge_target[eid]))
    # Guide Dirichlet rows use the reference (already cover-consistent) value
    # on constrained faces, so the relaxed field stays anchored to the guides.
    for c in constraints:
        add([(c.face, c.weight)], c.weight * float(ref[c.face]))
    a_mat = csr_matrix((data, (rows, cols)),
                       shape=(r, mesh.n_faces), dtype=np.float64)
    angles = lsqr(a_mat, np.array(rhs, dtype=np.float64),
                  atol=1e-12, btol=1e-12, iter_lim=20000)[0]

    edge_res = np.full(mesh.edges.shape[0], np.nan)
    edge_jump = np.full(mesh.edges.shape[0], np.nan)
    for eid in interior:
        f0, f1 = mesh.edge_faces[eid]
        raw = angles[f1] - angles[f0] - rho[eid]
        edge_jump[eid] = raw
        edge_res[eid] = _wrap(raw, period)

    # Holonomy is computed intrinsically from the *reference* lifted field:
    # it follows parallel transport exactly and only changes cover sheet at a
    # genuine winding, so its fan unwrap carries the true Poincare index
    # without smoothing noise.  (The relaxed ``angles`` are used for UV and
    # rendering; ``ref`` is the topological ground truth.)
    report = _holonomy(mesh, ref, rho, mode)
    return DirectionField(mesh, mode, angles, n, e1, e2, rho, edge_res,
                          edge_jump, period_jump, list(constraints), report,
                          reference_angles=ref)


def _face_fan(mesh: TriangleMesh, v: int):
    """Ordered fan of faces around v, crossed edges, oriented legs and flag.

    The walk crosses the half-edge leaving v; the next face is located and v
    re-found inside it directly (irregular fans have no fixed slot offset).
    Interior fans return to the start; boundary fans end at a boundary edge.
    """
    start_face = mesh.vertex_faces[v][0]
    faces: list[int] = []
    crossed: list[int] = []
    legs: list[tuple[int, int]] = []
    face = start_face
    visited: set[int] = set()
    closed = False
    guard = len(mesh.vertex_faces[v]) + 1
    while len(visited) < guard:
        if face in visited:
            closed = (face == start_face and len(faces) ==
                      len(mesh.vertex_faces[v]))
            break
        visited.add(face)
        faces.append(face)
        slot = int(np.flatnonzero(mesh.faces[face] == v)[0])
        eid = mesh.halfedge_edge(face, slot)
        f0, f1 = mesh.edge_faces[eid]
        nxt = f1 if f0 == face else f0
        if nxt < 0:
            break
        crossed.append(eid)
        legs.append((face, nxt))
        face = nxt
    return faces, crossed, legs, closed


def _holonomy(mesh: TriangleMesh, angles: np.ndarray, rho: np.ndarray,
              mode: str) -> HolonomyReport:
    """Gauge-invariant holonomy by parallel-transporting face angles along
    each vertex fan and unwrapping on the field's periodic cover.

    For every closed fan we express each face angle in a common transported
    frame, take the periodic unwrap, and close the loop back to the first
    face.  The geometric total rotation is always ``index * 2*pi`` (the
    Poincare index may be a half-integer for line fields, an integer for
    vector fields), so twice the index is simply ``round(total / pi)`` and is
    an integer in both cases.  Transport and unwrap are intrinsic, so the
    result is independent of any spanning-tree gauge.  Open (boundary) fans
    cannot carry a quantised index and are reported with their raw rotation
    instead.
    """
    period = np.pi if mode == "line" else 2.0 * np.pi
    verts, closeds, totals, k2s, residuals = [], [], [], [], []
    for v in range(mesh.n_vertices):
        if not mesh.vertex_faces[v]:
            continue
        faces, crossed, legs, closed = _face_fan(mesh, v)
        faces = np.asarray(faces, dtype=np.int64)

        def leg_sign(eid, ff, tt):
            f0, f1 = mesh.edge_faces[eid]
            return 1.0 if (ff == f0 and tt == f1) else -1.0

        # The fan walk runs clockwise around v.  Compute the transported
        # angle accumulation along it, then negate the closed-loop total so
        # the Poincare index follows the standard counter-clockwise sign.
        transported = np.empty(len(faces))
        transported[0] = angles[faces[0]]
        for i in range(1, len(faces)):
            eid = crossed[i - 1]
            sgn = leg_sign(eid, *legs[i - 1])
            transported[i] = angles[faces[i]] - sgn * rho[eid]
        unwrapped = np.unwrap(transported, period=period)

        if closed:
            # Closing leg returns from the last fan face to the first.
            eid = crossed[-1]
            sgn = leg_sign(eid, *legs[-1])
            closing = angles[faces[0]] - sgn * rho[eid]
            chain = np.unwrap(np.append(unwrapped, closing), period=period)
            total = -float(chain[-1] - chain[0])   # CCW sign convention
            # Geometric holonomy is index*2*pi for both field types, hence
            # 2*index = total/pi (integer: vector indices are whole, line
            # indices may be half-integers giving an odd doubled value).
            k2 = int(round(total / np.pi))
            residual = total - k2 * np.pi
        else:
            k2 = 0
            total = -float(unwrapped[-1] - unwrapped[0])
            residual = total
        verts.append(v)
        closeds.append(closed)
        totals.append(total)
        k2s.append(k2)
        residuals.append(residual)
    return HolonomyReport(
        vertex=np.array(verts, dtype=np.int64),
        is_closed_loop=np.array(closeds, dtype=bool),
        total_rotation=np.array(totals, dtype=np.float64),
        index_times2=np.array(k2s, dtype=np.int64),
        residual_rad=np.array(residuals, dtype=np.float64),
        singular=np.array([c and k != 0 for c, k in
                           zip(closeds, k2s)], dtype=bool))
