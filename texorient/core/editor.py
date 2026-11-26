"""Interactive editor state on top of a solved :class:`DirectionField`.

The editor keeps the topological layer editable independently of the smooth
least-squares layer:

* :class:`DirectionAnchor` objects lock texture directions on chosen faces;
* singularities are dragged *across the shared edge of two faces* by editing
  one integer period jump, after which every vertex index is recomputed and
  the field is re-solved with the new integers held fixed;
* two seam policies are supported -- the cut may migrate with the singularity
  (:data:`~texorient.core.topology.SEAM_MOVES`), or an already printed coat of
  arms may never turn (:data:`~texorient.core.topology.FIXED_PRINT`), in which
  case an edit that would rotate an uncut face is refused;
* when a closed loop's winding conflicts with the locked anchors, the editor
  reports exactly which constraint (seam migration or frozen print) must be
  released, since both cannot hold at once.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .field import DirectionConstraint, DirectionField, solve_direction_field
from .mesh import TriangleMesh
from .topology import (FIXED_PRINT, SEAM_MOVES, DirectionAnchor,
                       LoopTransport, Obstruction, edge_jumps,
                       recompute_indices, seamless_obstructions,
                       solve_with_jumps, transport_around_loop,
                       vertex_indices)


class TopologyConflict(RuntimeError):
    """Raised when an edit would violate the active seam/anchor policy."""


@dataclass
class DragResult:
    edge: int
    delta: int
    moved_vertices: list[int]
    index_times2: np.ndarray
    field: DirectionField
    seam_edges: np.ndarray
    policy: str


class FieldEditor:
    """Holds a field, its locked anchors, cut edges and the active policy."""

    def __init__(self, mesh: TriangleMesh,
                 constraints: list[DirectionConstraint], mode: str = "line",
                 policy: str = SEAM_MOVES):
        if policy not in (SEAM_MOVES, FIXED_PRINT):
            raise ValueError(f"unknown seam policy {policy!r}")
        self.mesh = mesh
        self.mode = mode
        self.policy = policy
        self.constraints = list(constraints)
        self.anchors: list[DirectionAnchor] = []
        self.field: DirectionField = solve_direction_field(
            mesh, self.constraints, mode=mode)
        self.seam_edges: set[int] = set()

    # ------------------------------------------------------------------ #
    def set_seams(self, edges) -> None:
        """Declare the currently cut interior edges."""
        interior = {int(e) for e in range(self.mesh.edges.shape[0])
                    if self.mesh.edge_faces[e, 1] >= 0}
        picked = {int(e) for e in edges}
        unknown = picked - interior
        if unknown:
            raise ValueError(f"edges {sorted(unknown)} are not interior edges")
        self.seam_edges = picked

    def add_anchor(self, face: int, angle: float | None = None,
                   locked: bool = True, label: str = "") -> DirectionAnchor:
        """Lock the texture direction on a face (defaults to the solved one)."""
        if not (0 <= face < self.mesh.n_faces):
            raise ValueError(f"face {face} out of range")
        if angle is None:
            angle = float(self.field.angles[face])
        anchor = DirectionAnchor(int(face), float(angle), locked=locked,
                                 label=label)
        anchor.geometry_key(self.mesh)
        self.anchors = [a for a in self.anchors if a.face != face]
        self.anchors.append(anchor)
        return anchor

    def locked_anchor_faces(self) -> list[int]:
        return [a.face for a in self.anchors if a.locked]

    # ------------------------------------------------------------------ #
    def drag_singularity(self, edge: int, delta: int = 1) -> DragResult:
        """Drag a singularity across the edge shared by two adjacent faces.

        The integer jump on the crossed edge changes by ``delta``.  Under the
        fixed-print policy the edge must already be a seam: otherwise an
        uncut, already printed face would have to rotate, so the edit is
        refused and the caller must either cut the edge first or switch to the
        migrating-seam policy.
        """
        edge = int(edge)
        f0, f1 = (int(x) for x in self.mesh.edge_faces[edge])
        if f1 < 0:
            raise ValueError("a singularity can only cross an interior edge")
        if int(delta) == 0:
            raise ValueError("drag delta must be nonzero")
        if self.policy == FIXED_PRINT and edge not in self.seam_edges:
            raise TopologyConflict(
                f"fixed-print policy: edge {edge} (faces {f0},{f1}) is not a "
                "seam, so dragging across it would rotate the printed texture; "
                "cut it first or release the fixed-print constraint")

        before = vertex_indices(self.mesh, self.field.period_jump, self.mode)
        index2, period_jump = recompute_indices(self.field, {edge: delta})
        moved = [int(v) for v in np.flatnonzero(index2 != before)]

        self.field = solve_with_jumps(self.mesh, self.constraints,
                                      period_jump, self.mode)
        if self.policy == SEAM_MOVES:
            # The cut follows the charge: the crossed edge itself becomes the
            # seam that absorbs the moved integer jump.
            self.seam_edges.add(edge)
        return DragResult(edge=edge, delta=int(delta), moved_vertices=moved,
                          index_times2=index2, field=self.field,
                          seam_edges=np.array(sorted(self.seam_edges),
                                              dtype=np.int64),
                          policy=self.policy)

    def move_singularity_to_vertex(self, vertex: int) -> DragResult:
        """Convenience: drag across the first available edge of a vertex fan.

        Models a singularity pulled onto ``vertex`` from a neighbouring one;
        the same fixed-print refusal applies.
        """
        from .field import _face_fan
        _, crossed, _, closed = _face_fan(self.mesh, int(vertex))
        if not crossed:
            raise ValueError(f"vertex {vertex} has no crossable fan edge")
        return self.drag_singularity(crossed[0], 1)

    # ------------------------------------------------------------------ #
    def closed_loop(self, faces: list[int],
                    start_angle: float | None = None) -> LoopTransport:
        """Transport a direction around a closed loop and report its holonomy."""
        return transport_around_loop(self.field, faces, start_angle)

    def loop_winding_conflicts(self, faces: list[int]) -> list[Obstruction]:
        """Return the winding obstruction of a closed loop, if any.

        A nonzero winding means a single seamless cover cannot satisfy every
        locked anchor around the loop: either a seam must cross it or one
        anchor must be released.
        """
        loop = self.closed_loop(faces)
        if not loop.closed or loop.winding == 0:
            return []
        return [Obstruction(
            kind="nonzero_winding", faces=list(faces),
            edges=list(loop.edges), winding=loop.winding, index2=0,
            vertex=-1, candidate_cuts=list(loop.edges),
            reason=(f"loop winding {loop.winding} conflicts with the locked "
                    "anchors; cut one loop edge (seam moves) or release an "
                    "anchor (fixed print) -- only one constraint can give"))]

    def seamless_blocks(self) -> list[Obstruction]:
        """Every loop/vertex that blocks a fully seamless, all-anchor field."""
        return seamless_obstructions(self.field)

    # ------------------------------------------------------------------ #
    def readback_rotation(self, faces: list[int],
                          observed_angles: np.ndarray) -> dict:
        """Read the actual rotation of strokes not used in the fit.

        ``observed_angles`` are local-frame stroke angles on each loop face,
        measured independently of the constraints that fitted the field.  Two
        totals are reported:

        * ``rotation_rad`` -- geometric parallel transport alone, which is
          near zero on a smooth flat loop;
        * ``actual_rotation_rad`` -- transport plus the integer period jumps,
          i.e. the rotation a printed stroke really accumulates, including the
          jump when the loop crosses a seam bank.

        The locked anchor on the start face is still valid exactly when the
        transported observed direction agrees with it modulo one period.
        """
        if len(observed_angles) != len(faces):
            raise ValueError("need one observed angle per loop face")
        mesh = self.mesh
        walk = list(faces)
        if walk[0] != walk[-1]:
            walk = walk + [walk[0]]
        edges, signs = [], []
        for a, b in zip(walk[:-1], walk[1:]):
            eid = mesh.edge_between_faces(int(a), int(b))
            if eid < 0:
                raise ValueError("loop faces must be adjacent")
            f0, f1 = mesh.edge_faces[eid]
            sgn = 1 if (int(a) == f0 and int(b) == f1) else -1
            edges.append(eid); signs.append(sgn)
        period = self.field.period

        geo_angle = float(observed_angles[0])
        full_angle = float(observed_angles[0])
        for eid, sgn in zip(edges, signs):
            geo_angle -= sgn * float(self.field.rho[eid])
            full_angle -= (sgn * float(self.field.rho[eid])
                           + sgn * int(self.field.period_jump[eid]) * period)
        geo_net = float(observed_angles[0] - geo_angle)
        actual_net = float(observed_angles[0] - full_angle)

        anchor_face = int(faces[0])
        anchor = next((a for a in self.anchors
                       if a.face == anchor_face and a.locked), None)
        anchor_valid = None
        if anchor is not None:
            diff = ((full_angle - anchor.angle + period / 2) % period
                    - period / 2)
            anchor_valid = bool(abs(diff) < 1e-6)
        cross_seam = [e for e in edges if e in self.seam_edges]
        return {
            "rotation_rad": geo_net,
            "rotation_deg": float(np.degrees(geo_net)),
            "actual_rotation_rad": actual_net,
            "actual_rotation_deg": float(np.degrees(actual_net)),
            "winding": int(round(actual_net / period)),
            "period_residual_rad": float(
                abs(actual_net - round(actual_net / period) * period)),
            "crossed_seam_edges": cross_seam,
            "locked_anchor_face": anchor_face if anchor else None,
            "locked_anchor_still_valid": anchor_valid,
        }

    # ------------------------------------------------------------------ #
    def edge_jump_table(self):
        """The brush-facing list of per-edge direction jumps."""
        return edge_jumps(self.field, seam_edges=self.seam_edges)

    def current_indices(self) -> np.ndarray:
        return vertex_indices(self.mesh, self.field.period_jump, self.mode,
                              rho=self.field.rho)

    def anchor_direction_world(self, anchor: DirectionAnchor) -> np.ndarray:
        """The still-queryable locked reference direction as a 3-D vector."""
        from .frames import angles_to_vectors
        return angles_to_vectors(
            np.array([anchor.angle]),
            self.field.e1[anchor.face:anchor.face + 1],
            self.field.e2[anchor.face:anchor.face + 1])[0]
