"""High-level orchestration shared by the GUI and the auxiliary CLI.

Nothing here moves mesh vertices.  A run takes the immutable input mesh plus a
list of direction constraints and produces, in order,

1. a tangent-bundle direction :class:`~texorient.core.field.DirectionField`
   (with its intrinsic holonomy / singularity report);
2. texture coordinates by one of the two declared methods
   (``seam`` = direction-exact with cuts, ``global`` = continuous, sacrificing
   local direction);
3. quantitative :class:`~texorient.core.metrics.TextureMetrics`;
4. renderable sub-triangles with colours derived from the *constructed* UVs.

The GUI calls :func:`run_pipeline` and only handles display and interaction;
the CLI calls the same function, so both paths run identical mathematics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.field import (DirectionConstraint, DirectionField,
                          solve_direction_field)
from ..core.frames import (angles_to_vectors, local_angles_of_3d_vectors,
                           tangent_frames)
from ..core.mesh import TriangleMesh
from ..core.metrics import TextureMetrics, evaluate
from ..core.models import REGISTRY
from ..core.render import textured_tris
from ..core.uv import UVResult, build_global_uv, build_seam_uv


@dataclass
class PipelineResult:
    mesh: TriangleMesh
    field: DirectionField
    uv: UVResult
    metrics: TextureMetrics
    tri_world: np.ndarray
    tri_colors: np.ndarray
    spacing: float
    pattern: str

    def singularity_vertices(self) -> np.ndarray:
        rep = self.field.holonomy
        return rep.vertex[rep.singular]

    def singularity_points(self) -> np.ndarray:
        return self.mesh.vertices[self.singularity_vertices()]

    def metrics_summary(self) -> dict:
        return self.metrics.summary()


def make_mesh(name: str, **kwargs) -> TriangleMesh:
    """Build one of the registered procedural test meshes by name."""
    if name not in REGISTRY:
        raise KeyError(f"unknown mesh '{name}'; choose from "
                       f"{sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)


def constraints_from_guide(mesh: TriangleMesh, guide_vectors: np.ndarray,
                           faces=None, weight: float = 20.0
                           ) -> list[DirectionConstraint]:
    """Convert per-face 3-D guide vectors into local-angle constraints.

    The vectors are projected into each face's own tangent plane and
    re-normalised (see :func:`local_angles_of_3d_vectors`); after this step the
    solver works purely on angles with parallel transport.  ``faces`` selects
    a subset (default: every face), which is how sparse user guides are built.
    """
    n, e1, e2 = tangent_frames(mesh)
    angles = local_angles_of_3d_vectors(mesh, guide_vectors, e1, e2, n)
    if faces is None:
        faces = range(mesh.n_faces)
    return [DirectionConstraint(int(f), float(angles[int(f)]),
                                weight=float(weight)) for f in faces]


def run_pipeline(mesh: TriangleMesh,
                 constraints: list[DirectionConstraint],
                 mode: str = "line",
                 method: str = "seam",
                 spacing: float = 0.25,
                 subdiv: int = 5,
                 pattern: str = "stripe",
                 world_scale: float = 1.0,
                 dfield: DirectionField | None = None) -> PipelineResult:
    """Solve the field, build UVs, measure and render in one call.

    When ``dfield`` is supplied (for example by the interactive topology
    editor after a singularity drag), its integer jumps and solved angles are
    used directly instead of re-solving from the constraints.
    """
    if not constraints:
        raise ValueError("add at least one direction constraint before laying "
                         "the texture; a field with no guide has no direction")
    field = dfield if dfield is not None else solve_direction_field(
        mesh, constraints, mode=mode)
    if method == "seam":
        uv = build_seam_uv(field, world_scale=world_scale)
    elif method == "global":
        uv = build_global_uv(field)
    else:
        raise ValueError("method must be 'seam' (cuts, direction exact) or "
                         "'global' (continuous, direction sacrificed)")
    metrics = evaluate(field, uv, stripe_spacing=spacing)
    tri_world, tri_colors, _ = textured_tris(
        mesh, uv, spacing=spacing, subdiv=subdiv, pattern=pattern)
    return PipelineResult(mesh, field, uv, metrics, tri_world, tri_colors,
                          spacing, pattern)


def face_direction_arrow(mesh: TriangleMesh, field: DirectionField,
                         face: int, length: float | None = None
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Return (start, end) world points of the solved direction arrow on a
    face, used for the optional field overlay."""
    center = mesh.face_centers()[face]
    vec = angles_to_vectors(
        np.array([field.angles[face]]),
        field.e1[face:face + 1], field.e2[face:face + 1])[0]
    if length is None:
        tri = mesh.vertices[mesh.faces[face]]
        length = 0.6 * float(np.mean(
            np.linalg.norm(tri - center, axis=1)))
    return center, center + length * vec


def nearest_face(mesh: TriangleMesh, point: np.ndarray) -> int:
    """Index of the face whose centre is nearest ``point`` (GUI picking)."""
    centers = mesh.face_centers()
    return int(np.argmin(np.linalg.norm(centers - np.asarray(point), axis=1)))


def tangent_angle_constraint(mesh: TriangleMesh, face: int,
                             world_direction: np.ndarray,
                             weight: float = 20.0) -> DirectionConstraint:
    """Build a local-angle constraint from a direction drawn in 3-D.

    The drawn vector is projected onto the picked face's tangent plane, so an
    imprecise mouse drag still yields a well-defined in-plane direction.
    Only the picked face carries a vector; every other entry is zero and is
    discarded before the projection, so a single-face guide can never trip the
    "normal to the face" check meant for a full per-face guide.
    """
    n, e1, e2 = tangent_frames(mesh)
    vec = np.asarray(world_direction, dtype=np.float64)
    tang = vec - np.dot(vec, n[face]) * n[face]
    if np.linalg.norm(tang) < 1e-10:
        raise ValueError("the drawn direction is normal to the picked face; "
                         "a tangent-plane direction cannot be defined there")
    tang /= np.linalg.norm(tang)
    angle = float(np.arctan2(np.dot(tang, e2[face]),
                             np.dot(tang, e1[face])))
    return DirectionConstraint(int(face), angle, weight=float(weight))
