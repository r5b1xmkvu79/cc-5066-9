"""Per-face tangent frames and parallel transport between adjacent faces.

A direction field lives in the *tangent bundle*, not in R^3.  Two directions
on neighbouring faces can only be compared after parallel transporting one
tangent plane into the other.  We encode every direction by its angle inside
a canonical per-face frame (n, e1, e2); the transport between two frames is a
rotation angle, never a raw 3-D projection followed by re-normalisation.
"""
from __future__ import annotations

import numpy as np

from .mesh import TriangleMesh


def face_normals(mesh: TriangleMesh) -> np.ndarray:
    """Unit face normals from the (CCW) cross product; geometry is read only."""
    p = mesh.vertices[mesh.faces]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    lengths = np.linalg.norm(n, axis=1)
    if np.any(lengths < 1e-14):
        raise ValueError("degenerate (zero-area) triangle present")
    return n / lengths[:, None]


def tangent_frames(mesh: TriangleMesh):
    """Return normals and orthonormal frame vectors (e1, e2) per face.

    The frame is built deterministically: e1 is the in-plane projection of a
    fixed world axis (a fallback axis if the face is nearly parallel to it),
    then e2 = n x e1.  A fixed rule keeps neighbouring frames related by
    smooth rotations instead of random per-face flips.
    """
    n = face_normals(mesh)
    axis = np.zeros((mesh.n_faces, 3))
    axis[:] = (0.0, 0.0, 1.0)
    # Faces nearly parallel to z use x instead.
    flip = np.abs(n[:, 2]) > 0.92
    axis[flip] = (1.0, 0.0, 0.0)
    e1 = axis - np.einsum("ij,ij->i", axis, n)[:, None] * n
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(n, e1)
    return n, e1, e2


def transport_angles(mesh: TriangleMesh, n: np.ndarray,
                     e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    """Parallel-transport rotation angle across every interior edge.

    For an interior edge shared by faces (f0, f1), rotating a tangent vector
    of f0 about the shared edge direction by the angle between the normals
    brings it into the tangent plane of f1.  We return the signed angle
    ``rho`` such that a direction with local angle ``a0`` in f0 maps to local
    angle ``a1 = a0 + rho`` in f1 (the shared-edge half-edge orientation
    defines the sign consistently).

    Boundary edges receive NaN.
    """
    rho = np.full(mesh.edges.shape[0], np.nan)
    for eid, (u, v) in enumerate(mesh.edges):
        f0, f1 = mesh.edge_faces[eid]
        if f1 < 0:
            continue
        edge_dir = mesh.vertices[v] - mesh.vertices[u]
        edge_dir /= np.linalg.norm(edge_dir)
        # Signed angle between normals about the edge axis.
        cos_t = float(np.clip(np.dot(n[f0], n[f1]), -1.0, 1.0))
        sin_t = float(np.dot(edge_dir, np.cross(n[f0], n[f1])))
        theta = np.arctan2(sin_t, cos_t)
        # Rotation matrix of the transport, expressed in frame f1.
        rot = _rodrigues(edge_dir, theta)
        te1 = rot @ e1[f0]
        # Local coordinates of the transported e1[f0] in frame f1.
        x = float(np.dot(te1, e1[f1]))
        y = float(np.dot(te1, e2[f1]))
        rho[eid] = np.arctan2(y, x)
    return rho


def _rodrigues(axis: np.ndarray, theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    x, y, z = axis
    return np.array([
        [c + x * x * (1 - c),     x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c),     y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ])


def local_angles_of_3d_vectors(mesh: TriangleMesh, vectors: np.ndarray,
                               e1: np.ndarray, e2: np.ndarray,
                               n: np.ndarray) -> np.ndarray:
    """Project 3-D vectors into *their own face* tangent planes and angle them.

    This is used only to convert user-supplied guide vectors (which are
    approximately tangent) into local-angle constraints.  The field itself is
    never formed by averaging such projections: after this conversion all
    solving happens on angles with transport, and the projected vector is
    re-orthogonalised so a slightly off-plane guide cannot silently tilt the
    field out of the surface.
    """
    if vectors.shape != (mesh.n_faces, 3):
        raise ValueError("need one 3-D guide vector per face")
    tang = vectors - np.einsum("ij,ij->i", vectors, n)[:, None] * n
    norm = np.linalg.norm(tang, axis=1)
    if np.any(norm < 1e-10):
        raise ValueError("a guide vector is normal to its face, "
                         "no tangent direction is defined")
    tang /= norm[:, None]
    return np.arctan2(np.einsum("ij,ij->i", tang, e2),
                      np.einsum("ij,ij->i", tang, e1))


def angles_to_vectors(angles: np.ndarray, e1: np.ndarray,
                      e2: np.ndarray) -> np.ndarray:
    """Reconstruct unit tangent 3-D vectors from per-face local angles."""
    return (np.cos(angles)[:, None] * e1 + np.sin(angles)[:, None] * e2)
