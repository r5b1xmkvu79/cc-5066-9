"""Matplotlib 3-D drawing helpers shared by the GUI and headless rendering.

Keeping all figure construction in one module guarantees that the image files
written for the report show exactly what the GUI displays: textured
sub-triangles, the optional direction-field arrow overlay, seam cuts and the
computed singularity markers.
"""
from __future__ import annotations

import numpy as np

from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from ..core.field import DirectionField
from ..core.mesh import TriangleMesh
from ..core.uv import UVResult
from .pipeline import PipelineResult, face_direction_arrow


def equal_axes(ax, vertices: np.ndarray) -> None:
    mid = (vertices.max(axis=0) + vertices.min(axis=0)) / 2.0
    span = float((vertices.max(axis=0) - vertices.min(axis=0)).max()) / 2.0
    span = max(span, 1e-6)
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:  # pragma: no cover - older matplotlib
        pass


def draw_texture(ax, result: PipelineResult, alpha: float = 1.0) -> Poly3DCollection:
    coll = Poly3DCollection(result.tri_world, facecolors=result.tri_colors,
                            edgecolors="none", linewidths=0, alpha=alpha)
    ax.add_collection3d(coll)
    equal_axes(ax, result.mesh.vertices)
    return coll


def draw_wireframe(ax, mesh: TriangleMesh, color: str = "0.6",
                   lw: float = 0.25, alpha: float = 0.5) -> Line3DCollection:
    segs = []
    for u, v in mesh.edges:
        segs.append([mesh.vertices[u], mesh.vertices[v]])
    coll = Line3DCollection(segs, colors=color, linewidths=lw, alpha=alpha)
    ax.add_collection3d(coll)
    return coll


def draw_seams(ax, result: PipelineResult) -> Line3DCollection | None:
    seams = result.uv.seams
    if seams is None or len(seams) == 0:
        return None
    mesh = result.mesh
    segs = [[mesh.vertices[u], mesh.vertices[v]]
            for u, v in mesh.edges[seams]]
    coll = Line3DCollection(segs, colors="red", linewidths=1.4, alpha=0.9)
    ax.add_collection3d(coll)
    return coll


def draw_singularities(ax, result: PipelineResult,
                       color: str = "magenta") -> object:
    rep = result.field.holonomy
    sing = np.where(rep.singular)[0]
    if len(sing) == 0:
        return None
    verts = rep.vertex[sing]
    pts = result.mesh.vertices[verts]
    # Offset the marker slightly along the local normal so it sits above the
    # surface and the seam lines cannot hide it.
    normals = result.field.n[verts]
    lift = pts + 0.05 * normals
    sc = ax.scatter(lift[:, 0], lift[:, 1], lift[:, 2], c=color, s=260,
                    marker="*", edgecolors="black", linewidths=1.0,
                    depthshade=False, zorder=100,
                    label=f"singularity x{len(sing)}")
    for i, (p, idx_v) in enumerate(zip(lift, rep.index_times2[sing])):
        ax.text(p[0], p[1], p[2], f" {idx_v / 2.0:+.1f}",
                color="darkmagenta", fontsize=9, fontweight="bold",
                zorder=101)
    return sc


def draw_field_overlay(ax, mesh: TriangleMesh, field: DirectionField,
                       step: int = 6, color: str = "deepskyblue") -> None:
    starts, ends = [], []
    for face in range(0, mesh.n_faces, step):
        s, e = face_direction_arrow(mesh, field, face)
        starts.append(s)
        ends.append(e)
    starts = np.array(starts)
    ends = np.array(ends)
    ax.quiver(starts[:, 0], starts[:, 1], starts[:, 2],
              ends[:, 0] - starts[:, 0],
              ends[:, 1] - starts[:, 1],
              ends[:, 2] - starts[:, 2],
              color=color, linewidth=1.0, arrow_length_ratio=0.25,
              length=1.0, normalize=False)


def draw_constraints(ax, mesh: TriangleMesh, constraints,
                     color: str = "orange") -> None:
    """Draw user-drawn constraint arrows before solving (uses raw local angles
    and the canonical frames)."""
    from ..core.frames import angles_to_vectors, tangent_frames
    n, e1, e2 = tangent_frames(mesh)
    centers = mesh.face_centers()
    angles = np.array([c.angle for c in constraints])
    faces = np.array([c.face for c in constraints])
    vecs = angles_to_vectors(angles, e1[faces], e2[faces])
    starts = centers[faces]
    ax.quiver(starts[:, 0], starts[:, 1], starts[:, 2],
              vecs[:, 0], vecs[:, 1], vecs[:, 2],
              color=color, linewidth=1.6, arrow_length_ratio=0.3,
              length=0.18, normalize=False)


def draw_anchors(ax, mesh: TriangleMesh, anchors, field=None,
                 color: str = "lime") -> None:
    """Draw lockable direction anchors: a square marker plus the locked arrow.

    Anchors use the field's solved frames when available, so a locked
    reference direction stays queryable and visible after topology edits.
    """
    if not anchors:
        return
    from ..core.frames import angles_to_vectors, tangent_frames
    if field is None:
        n, e1, e2 = tangent_frames(mesh)
    else:
        e1, e2 = field.e1, field.e2
    centers = mesh.face_centers()
    faces = np.array([a.face for a in anchors])
    angles = np.array([a.angle for a in anchors])
    vecs = angles_to_vectors(angles, e1[faces], e2[faces])
    starts = centers[faces]
    ax.scatter(starts[:, 0], starts[:, 1], starts[:, 2],
               c=color, marker="s", s=90, edgecolors="black",
               depthshade=False, zorder=98, label=f"anchor x{len(anchors)}")
    ax.quiver(starts[:, 0], starts[:, 1], starts[:, 2],
              vecs[:, 0], vecs[:, 1], vecs[:, 2],
              color=color, linewidth=2.0, arrow_length_ratio=0.3,
              length=0.22, normalize=False, zorder=99)


def draw_transport_loop(ax, mesh: TriangleMesh, faces: list[int],
                        color: str = "gold") -> None:
    """Draw the dual face-centre path of a closed-loop transport demo."""
    if len(faces) < 2:
        return
    centers = mesh.face_centers()
    pts = centers[np.asarray(faces, dtype=np.int64)]
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=2.2,
            zorder=97, label="transport loop")


def draw_edges(ax, mesh: TriangleMesh, edge_ids, color: str = "red",
               lw: float = 1.6, label: str | None = None) -> None:
    """Highlight arbitrary interior edges (manual seam cuts / mismatches)."""
    edge_ids = list(edge_ids)
    if not edge_ids:
        return
    segs = [[mesh.vertices[u], mesh.vertices[v]]
            for u, v in mesh.edges[np.asarray(edge_ids, dtype=np.int64)]]
    coll = Line3DCollection(segs, colors=color, linewidths=lw, alpha=0.95)
    ax.add_collection3d(coll)
    if label:
        coll.set_label(label)


def style_axes(ax, title: str) -> None:
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    try:
        ax.xaxis.pane.set_alpha(0.0)
        ax.yaxis.pane.set_alpha(0.0)
        ax.zaxis.pane.set_alpha(0.0)
    except Exception:  # pragma: no cover
        pass
