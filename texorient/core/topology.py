"""Interactive topology editing: direction anchors, singularity dragging,
seam policy, closed-loop transport and face renumbering.

The field solver (:mod:`texorient.core.field`) determines the integer period
jump ``k_e`` on every interior edge from the guides.  This module treats those
integers as the editable topological layer:

* a *direction anchor* locks the texture direction on one face and survives
  face renumbering by also storing a geometric key;
* *dragging a singularity* across the edge shared by two faces adds one period
  jump on that edge, so the change propagates from the geometric adjacency to
  the actual texture orientation, and the vertex indices are recomputed;
* the *seam policy* decides whether a cut may migrate with the singularity
  (``seam_moves``) or whether an already printed coat of arms may never turn
  (``fixed_print``), in which case only existing cut edges may change;
* *closed-loop transport* parallel-transports a direction around any loop of
  faces and reports the geometric rotation and the integer winding, so a loop
  whose winding conflicts with the locked anchors can be identified;
* *face renumbering* reorders faces without moving geometry, and the mismatch
  detector locates adjacency edges whose numbering jump disagrees with the
  (still smooth) geometric closure.

Everything here reads mesh geometry; mesh vertices are never modified.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.linalg import lsqr

from .field import (DirectionConstraint, DirectionField,
                    _face_fan, _holonomy, _wrap)
from .mesh import TriangleMesh

# Seam policies.
SEAM_MOVES = "seam_moves"        # the cut may migrate together with a singularity
FIXED_PRINT = "fixed_print"      # the printed texture orientation is frozen


@dataclass
class EdgeJump:
    """Direction jump of the field across one interior edge.

    ``jump`` is the unwrapped mismatch ``angle[f1] - angle[f0] - rho``;
    ``wrapped`` is the same quantity folded into one period and ``k`` is the
    integer number of periods (the topological carrier of the jump).
    """
    edge: int
    face0: int
    face1: int
    jump: float
    wrapped: float
    k: int
    is_seam: bool


@dataclass
class LoopTransport:
    """Result of parallel-transporting a direction around a face loop."""
    faces: list[int]
    edges: list[int]
    closed: bool
    rotation_rad: float          # total geometric transport rotation (signed)
    winding: int                 # integer periods accumulated around the loop
    holonomy_rad: float          # rotation + integer*period, the net loop defect
    start_angle: float
    end_angle: float             # local angle of the transported start direction
    period: float

    @property
    def holonomy_turns(self) -> float:
        return self.holonomy_rad / (2.0 * np.pi)


@dataclass
class DirectionAnchor:
    """A lockable texture-direction anchor on one face.

    ``face_key`` is the sorted tuple of the face's vertex ids, so the anchor
    can be re-located after the faces are renumbered even though the geometry
    is unchanged.
    """
    face: int
    angle: float
    locked: bool = True
    label: str = ""
    face_key: tuple[int, int, int] | None = None

    def geometry_key(self, mesh: TriangleMesh) -> tuple[int, int, int]:
        if self.face_key is None:
            self.face_key = tuple(sorted(int(v) for v in mesh.faces[self.face]))
        return self.face_key  # type: ignore[return-value]


@dataclass
class Obstruction:
    """A closed loop on which a seamless field cannot satisfy every anchor."""
    kind: str                    # "nonzero_winding" or "singular_vertex"
    faces: list[int]
    edges: list[int]
    winding: int
    index2: int                  # twice the Poincare index (vertex obstructions)
    vertex: int
    candidate_cuts: list[int] = field(default_factory=list)
    reason: str = ""


@dataclass
class NumberingMismatch:
    """An adjacency edge whose integer/numbering jump disagrees with geometry."""
    edge: int
    face0: int
    face1: int
    angle_jump: float            # observed field jump across the edge
    expected_wrap: float         # jump a smooth geometric closure would allow
    periods: float               # jump expressed in units of the period


# ---------------------------------------------------------------------- #
# Edge jumps and vertex indices
# ---------------------------------------------------------------------- #
def interior_edges(mesh: TriangleMesh) -> np.ndarray:
    return np.array([eid for eid in range(mesh.edges.shape[0])
                     if mesh.edge_faces[eid, 1] >= 0], dtype=np.int64)


def edge_jumps(dfield: DirectionField, seam_edges=None) -> list[EdgeJump]:
    """Per-edge direction jump of the solved field across every interior edge."""
    mesh = dfield.mesh
    seams = set() if seam_edges is None else {int(e) for e in seam_edges}
    out: list[EdgeJump] = []
    for eid in interior_edges(mesh):
        f0, f1 = (int(x) for x in mesh.edge_faces[eid])
        raw = float(dfield.angles[f1] - dfield.angles[f0] - dfield.rho[eid])
        out.append(EdgeJump(
            edge=int(eid), face0=f0, face1=f1, jump=raw,
            wrapped=float(_wrap(raw, dfield.period)),
            k=int(dfield.period_jump[eid]), is_seam=int(eid) in seams))
    return out


def vertex_indices(mesh: TriangleMesh, period_jump: np.ndarray,
                   mode: str, rho: np.ndarray | None = None) -> np.ndarray:
    """Twice the Poincare index at every closed vertex from its fan.

    Each crossed leg contributes the full parallel transport
    ``rho + k*period``; the closed-fan total is the geometric holonomy, whose
    nearest multiple of pi is the doubled index (an integer for vector fields,
    an integer that may be odd for line fields).  When ``rho`` is omitted the
    mesh is treated as flat (zero transport), which is exact for planar
    meshes and reduces the index to the signed integer fan sum.  Boundary fans
    cannot close and report zero, matching
    :func:`texorient.core.field._holonomy`.
    """
    period = np.pi if mode == "line" else 2.0 * np.pi
    index2 = np.zeros(mesh.n_vertices, dtype=np.int64)
    for v in range(mesh.n_vertices):
        if not mesh.vertex_faces[v]:
            continue
        _, _, legs, closed = _face_fan(mesh, v)
        if not closed:
            continue
        total = 0.0
        for ff, tt in legs:
            eid = mesh.edge_between_faces(ff, tt)
            f0, f1 = mesh.edge_faces[eid]
            sgn = 1 if (ff == f0 and tt == f1) else -1
            transport = 0.0 if rho is None else float(rho[eid])
            total += sgn * (transport + int(period_jump[eid]) * period)
        index2[v] = int(round(total / np.pi))
    return index2


def recompute_indices(dfield: DirectionField,
                      extra_jumps: dict[int, int] | None = None
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Recompute doubled vertex indices after editing integer edge jumps.

    ``extra_jumps`` maps an edge id to an additional integer period jump.
    Returns ``(index_times2, period_jump)`` with the edited jumps applied.
    """
    period_jump = dfield.period_jump.copy()
    for eid, delta in (extra_jumps or {}).items():
        period_jump[int(eid)] += int(delta)
    return (vertex_indices(dfield.mesh, period_jump, dfield.mode,
                           rho=dfield.rho), period_jump)


# ---------------------------------------------------------------------- #
# Closed-loop parallel transport
# ---------------------------------------------------------------------- #
def _loop_steps(mesh: TriangleMesh, faces: list[int]):
    """Resolve a face sequence into (edge, sign) legs; sign is +1 when the
    leg crosses the edge in the stored f0->f1 direction."""
    edges, signs = [], []
    for a, b in zip(faces[:-1], faces[1:]):
        eid = mesh.edge_between_faces(int(a), int(b))
        if eid < 0:
            raise ValueError(f"faces {a} and {b} are not adjacent; the loop "
                             "must step across shared edges")
        f0, f1 = mesh.edge_faces[eid]
        edges.append(eid)
        signs.append(1 if (int(a) == f0 and int(b) == f1) else -1)
    return edges, signs


def transport_around_loop(dfield: DirectionField, faces: list[int],
                          start_angle: float | None = None) -> LoopTransport:
    """Parallel-transport a direction around a loop of adjacent faces.

    The loop may be open or closed (the last face equal to the first).  For a
    closed loop the holonomy is the net defect: geometric transport rotation
    plus the integer number of periods accumulated along the way.  A locked
    anchor placed back at the start face is still valid exactly when this
    holonomy is an integer multiple of the field period.
    """
    if len(faces) < 2:
        raise ValueError("a loop needs at least two faces")
    mesh = dfield.mesh
    faces = [int(f) for f in faces]
    closed = faces[0] == faces[-1]
    walk = faces if closed else faces + [faces[0]]
    edges, signs = _loop_steps(mesh, walk)
    period = dfield.period

    if start_angle is None:
        start_angle = float(dfield.angles[faces[0]])
    # Local angle of the transported start direction in each visited frame.
    angle = float(start_angle)
    for eid, sgn in zip(edges, signs):
        angle -= sgn * float(dfield.rho[eid])
    rotation = float(start_angle - angle)     # transport rotation, signed
    end_angle = angle

    winding = 0
    for eid, sgn in zip(edges, signs):
        winding += sgn * int(dfield.period_jump[eid])
    holonomy = rotation + winding * period
    return LoopTransport(faces=faces, edges=edges, closed=closed,
                         rotation_rad=rotation, winding=int(winding),
                         holonomy_rad=float(holonomy),
                         start_angle=float(start_angle),
                         end_angle=float(end_angle), period=period)


# ---------------------------------------------------------------------- #
# Re-solving a field with prescribed integer edge jumps
# ---------------------------------------------------------------------- #
def solve_with_jumps(mesh: TriangleMesh,
                     constraints: list[DirectionConstraint],
                     period_jump: np.ndarray, mode: str) -> DirectionField:
    """Solve the continuous field with the integer jumps held fixed.

    Used after an interactive edit: the topological integers are supplied by
    the editor and only the smooth per-face angles are relaxed.  The holonomy
    report is rebuilt intrinsically from a tree lift of the fixed jumps.
    """
    if not constraints:
        raise ValueError("at least one direction constraint is required")
    from .frames import tangent_frames, transport_angles
    n, e1, e2 = tangent_frames(mesh)
    rho = transport_angles(mesh, n, e1, e2)
    period = np.pi if mode == "line" else 2.0 * np.pi
    interior = interior_edges(mesh)

    # Dual spanning tree; its edges carry zero integer jump by construction.
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
    for eid in interior:
        f0, f1 = (int(x) for x in mesh.edge_faces[eid])
        adj[f0].append((f1, int(eid)))
        adj[f1].append((f0, int(eid)))
    for i, j in zip(mst.row, mst.col):
        tree_edges.add(mesh.edge_between_faces(int(i), int(j)))

    # Discrete cover lift on the full dual graph: each leg transports by rho
    # and adds the prescribed integer jump.  Tree edges have zero jump, so the
    # walk is consistent everywhere the prescribed jumps are; only genuine
    # singularities make a closed fan disagree, which the holonomy pass reads.
    root = constraints[0].face
    ref = np.zeros(mesh.n_faces)
    settled = np.zeros(mesh.n_faces, dtype=bool)
    settled[root] = True
    queue = [root]
    head = 0
    while head < len(queue):
        f0 = queue[head]
        head += 1
        for f1, eid in adj[f0]:
            if settled[f1]:
                continue
            sgn = 1 if mesh.edge_faces[eid, 0] == f0 else -1
            ref[f1] = (ref[f0] + sgn * rho[eid]
                       + sgn * int(period_jump[eid]) * period)
            settled[f1] = True
            queue.append(f1)
    if not np.all(settled):
        raise RuntimeError("mesh dual graph is disconnected")

    # Snap each constrained face onto its own cover sheet.
    for c in constraints:
        diff = c.angle - ref[c.face]
        if abs(diff) > period * 0.5:
            ref[c.face] = c.angle - int(round(diff / period)) * period
        else:
            ref[c.face] = c.angle

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
        f0, f1 = (int(x) for x in mesh.edge_faces[eid])
        target = rho[eid] + int(period_jump[eid]) * period
        add([(f0, -1.0), (f1, 1.0)], float(target))
    for c in constraints:
        add([(c.face, c.weight)], c.weight * float(ref[c.face]))
    a_mat = csr_matrix((data, (rows, cols)),
                       shape=(r, mesh.n_faces), dtype=np.float64)
    angles = lsqr(a_mat, np.array(rhs, dtype=np.float64),
                  atol=1e-12, btol=1e-12, iter_lim=20000)[0]

    edge_res = np.full(mesh.edges.shape[0], np.nan)
    edge_jump_arr = np.full(mesh.edges.shape[0], np.nan)
    for eid in interior:
        f0, f1 = (int(x) for x in mesh.edge_faces[eid])
        raw = angles[f1] - angles[f0] - rho[eid]
        edge_jump_arr[eid] = raw
        edge_res[eid] = _wrap(raw, period)

    # Build the holonomy report from the fixed integers via full-fan
    # transport, so it stays self-consistent with period_jump even on a
    # handled surface where a single tree lift is path-dependent.  The
    # continuous report on the tree lift is retained for the residual column.
    report = _holonomy(mesh, ref, rho, mode)
    int_index2 = vertex_indices(mesh, np.asarray(period_jump, dtype=np.int64),
                                mode, rho=rho)
    report.index_times2 = int_index2[report.vertex]
    report.total_rotation = report.index_times2.astype(np.float64) * np.pi
    report.singular = np.array(
        [c and k != 0 for c, k in zip(report.is_closed_loop,
                                      report.index_times2)], dtype=bool)
    report.residual_rad = np.zeros_like(report.residual_rad)
    return DirectionField(mesh, mode, angles, n, e1, e2, rho, edge_res,
                          edge_jump_arr, np.asarray(period_jump, dtype=np.int64),
                          list(constraints), report,
                          reference_angles=ref)


# ---------------------------------------------------------------------- #
# Seamless feasibility and candidate cuts
# ---------------------------------------------------------------------- #
def seamless_obstructions(dfield: DirectionField) -> list[Obstruction]:
    """Loops and vertices that block a globally seamless direction field.

    A seamless field requires zero integer period jump on every edge.  Any
    vertex whose doubled index is nonzero therefore obstructs (its fan winds),
    and every non-tree edge carrying a nonzero jump is a non-contractible loop
    obstruction.  For each obstruction the edges a human may cut to release it
    are listed, so the seam tool can offer a manual choice instead of guessing.
    """
    mesh = dfield.mesh
    out: list[Obstruction] = []
    index2 = vertex_indices(mesh, dfield.period_jump, dfield.mode,
                            rho=dfield.rho)
    for v in np.flatnonzero(index2):
        faces, crossed, _, closed = _face_fan(mesh, int(v))
        out.append(Obstruction(
            kind="singular_vertex", faces=faces, edges=list(crossed),
            winding=int(sum(int(dfield.period_jump[e]) for e in crossed)),
            index2=int(index2[v]), vertex=int(v),
            candidate_cuts=list(crossed),
            reason=(f"vertex {v} has Poincare index {index2[v] / 2.0:+.2f}; "
                    "its fan must be cut to unwind it")))
    # Non-contractible winding: interior non-tree edges with a nonzero jump.
    interior = interior_edges(mesh)
    f0a = mesh.edge_faces[interior, 0]
    f1a = mesh.edge_faces[interior, 1]
    dual = csr_matrix((np.ones(len(interior)), (f0a, f1a)),
                      shape=(mesh.n_faces, mesh.n_faces)).maximum(
        csr_matrix((np.ones(len(interior)), (f1a, f0a)),
                   shape=(mesh.n_faces, mesh.n_faces)))
    tree = minimum_spanning_tree(dual).tocoo()
    tree_edges = {mesh.edge_between_faces(int(i), int(j))
                  for i, j in zip(tree.row, tree.col)}
    for eid in interior:
        if eid in tree_edges or int(dfield.period_jump[eid]) == 0:
            continue
        f0, f1 = (int(x) for x in mesh.edge_faces[eid])
        loop = _dual_path(mesh, tree_edges, f1, f0) + [f1]
        out.append(Obstruction(
            kind="nonzero_winding", faces=loop,
            edges=list(_loop_steps(mesh, loop)[0]),
            winding=int(dfield.period_jump[eid]), index2=0, vertex=-1,
            candidate_cuts=[int(eid)],
            reason=(f"closed dual loop carries winding "
                    f"{int(dfield.period_jump[eid])}; a seamless field cannot "
                    "close it, so one of its edges must be cut")))
    return out


def _dual_path(mesh: TriangleMesh, tree_edges: set[int],
               src: int, dst: int) -> list[int]:
    """Face path inside the dual spanning tree (BFS), including both ends."""
    adj: dict[int, list[int]] = {i: [] for i in range(mesh.n_faces)}
    for eid in tree_edges:
        f0, f1 = (int(x) for x in mesh.edge_faces[eid])
        adj[f0].append(f1)
        adj[f1].append(f0)
    prev = {src: src}
    queue = [src]
    head = 0
    while head < len(queue):
        cur = queue[head]
        head += 1
        if cur == dst:
            break
        for nxt in adj[cur]:
            if nxt not in prev:
                prev[nxt] = cur
                queue.append(nxt)
    if dst not in prev:
        raise RuntimeError("dual spanning tree is disconnected")
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    path.reverse()
    return path


# ---------------------------------------------------------------------- #
# Renumbering and numbering-jump diagnosis
# ---------------------------------------------------------------------- #
def renumber_faces(mesh: TriangleMesh, permutation: np.ndarray,
                   payloads: dict | None = None) -> dict:
    """Build a mesh with faces reordered by ``permutation`` (new <- old).

    Vertices are untouched; only the face rows and every face/half-edge
    indexed payload are remapped.  ``payloads`` may carry ``constraints`` (a
    list of :class:`DirectionConstraint`) and ``anchors`` (a list of
    :class:`DirectionAnchor`); their face indices are rewritten in place of
    the originals and the returned dictionary also contains the inverse map.
    """
    permutation = np.asarray(permutation, dtype=np.int64)
    if sorted(int(x) for x in permutation) != list(range(mesh.n_faces)):
        raise ValueError("permutation must contain every face id exactly once")
    new_faces = mesh.faces[permutation]
    new_mesh = TriangleMesh(mesh.vertices.copy(), new_faces)
    old_of_new = permutation
    new_of_old = np.empty_like(permutation)
    new_of_old[old_of_new] = np.arange(mesh.n_faces)

    result = {"mesh": new_mesh, "old_of_new": old_of_new,
              "new_of_old": new_of_old}
    payloads = payloads or {}
    if "constraints" in payloads:
        for c in payloads["constraints"]:
            c.face = int(new_of_old[c.face])
    if "anchors" in payloads:
        for a in payloads["anchors"]:
            a.face = int(new_of_old[a.face])
            a.face_key = None  # rebuilt lazily from the new mesh
    return result


def locate_anchor(mesh: TriangleMesh, anchor: DirectionAnchor) -> int:
    """Find the current face id of an anchor from its geometric key."""
    key = anchor.face_key
    if key is None:
        key = tuple(sorted(int(v) for v in mesh.faces[anchor.face]))
        anchor.face_key = key
    for fi, tri in enumerate(mesh.faces):
        if tuple(sorted(int(v) for v in tri)) == key:
            anchor.face = fi
            return fi
    raise KeyError(f"anchor geometry {key} no longer exists in the mesh")


def numbering_mismatches(dfield: DirectionField,
                         tol_periods: float = 0.25) -> list[NumberingMismatch]:
    """Find adjacency edges where the field jumps despite smooth geometry.

    Geometry is smooth across an edge when parallel transport alone explains
    the neighbour's frame (the continuous, wrapped mismatch is within a
    fraction of a period).  A nonzero *integer* jump on such an edge means the
    orientation follows the numbering (a re-indexed cover sheet) rather than
    the geometric closure; the seam tool can then jump straight to that edge.
    The integer jump is the topological ground truth -- the least-squares
    relaxation spreads its magnitude over nearby edges but cannot remove it.
    """
    mesh = dfield.mesh
    out: list[NumberingMismatch] = []
    for eid in interior_edges(mesh):
        f0, f1 = (int(x) for x in mesh.edge_faces[eid])
        raw = float(dfield.angles[f1] - dfield.angles[f0] - dfield.rho[eid])
        wrapped = _wrap(raw, dfield.period)
        k = int(dfield.period_jump[eid])
        if k != 0 and abs(wrapped) < tol_periods * dfield.period:
            out.append(NumberingMismatch(
                edge=int(eid), face0=f0, face1=f1, angle_jump=raw,
                expected_wrap=wrapped,
                periods=k + wrapped / dfield.period))
    return out
