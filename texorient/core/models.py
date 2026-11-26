"""Procedural test meshes: cylinder, holed annulus and a curved band.

All generators return :class:`TriangleMesh` with consistently oriented faces
and leave geometry generation separate from every texture computation.
"""
from __future__ import annotations

import numpy as np

from .mesh import TriangleMesh


def cylinder(radius: float = 1.0, height: float = 2.0,
             n_theta: int = 48, n_z: int = 12) -> TriangleMesh:
    """Open cylindrical tube (two boundary rings), no end caps."""
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    z = np.linspace(-height / 2, height / 2, n_z)
    th, zz = np.meshgrid(theta, z)
    verts = np.column_stack([
        (radius * np.cos(th)).ravel(),
        (radius * np.sin(th)).ravel(),
        zz.ravel()])
    def vid(i, j):  # row i (z), col j (theta)
        return i * n_theta + (j % n_theta)
    faces = []
    for i in range(n_z - 1):
        for j in range(n_theta):
            a, b = vid(i, j), vid(i, j + 1)
            c, d = vid(i + 1, j + 1), vid(i + 1, j)
            faces.append((a, b, c))
            faces.append((a, c, d))
    return TriangleMesh(verts, np.array(faces, dtype=np.int64))


def annulus(r_inner: float = 0.6, r_outer: float = 1.6,
            n_theta: int = 56, n_rad: int = 10) -> TriangleMesh:
    """Flat disk with a hole: two boundary loops, exercises topology."""
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    radii = np.linspace(r_inner, r_outer, n_rad)
    th, rr = np.meshgrid(theta, radii)
    verts = np.column_stack([(rr * np.cos(th)).ravel(),
                             (rr * np.sin(th)).ravel(),
                             np.zeros(th.size)])
    def vid(i, j):
        return i * n_theta + (j % n_theta)
    faces = []
    for i in range(n_rad - 1):
        for j in range(n_theta):
            a, b = vid(i, j), vid(i, j + 1)
            c, d = vid(i + 1, j + 1), vid(i + 1, j)
            faces.append((a, b, c))
            faces.append((a, c, d))
    return TriangleMesh(verts, np.array(faces, dtype=np.int64))


def curved_band(length: float = 4.0, width: float = 1.2,
                n_x: int = 40, n_w: int = 10,
                curve: float = 0.8) -> TriangleMesh:
    """A ribbon following y = curve * sin(pi x / L) at z ~ 0.

    The ribbon bends in the plane and is tilted up at the edges, so it has
    genuinely varying tangent planes along its length.
    """
    xs = np.linspace(-length / 2, length / 2, n_x)
    ws = np.linspace(-width / 2, width / 2, n_w)
    center_y = curve * np.sin(np.pi * xs / length)
    dy = curve * (np.pi / length) * np.cos(np.pi * xs / length)
    verts = []
    for x, y, slope in zip(xs, center_y, dy):
        # in-plane unit tangent and normal to the centreline:
        # tangent t = (1, slope, 0)/norm; normal n2d = (-slope, 1, 0)/norm
        norm = np.hypot(1.0, slope)
        tx, ty = 1.0 / norm, slope / norm
        nx, ny = -slope / norm, 1.0 / norm
        for w in ws:
            lift = 0.25 * (w / (width / 2)) ** 2
            verts.append((x + nx * w, y + ny * w, lift))
    verts = np.array(verts)
    def vid(i, j):
        return i * n_w + j
    faces = []
    for i in range(n_x - 1):
        for j in range(n_w - 1):
            a, b = vid(i, j), vid(i, j + 1)
            c, d = vid(i + 1, j + 1), vid(i + 1, j)
            faces.append((a, d, c))
            faces.append((a, c, b))
    return TriangleMesh(verts, np.array(faces, dtype=np.int64))


def disk_with_interior(radius: float = 1.5, n_theta: int = 48,
                       n_rad: int = 9) -> TriangleMesh:
    """Filled polar disk including a centre vertex, so interior vertices and
    their closed-loop holonomy exist (used to expose genuine singularities)."""
    verts = [np.array([0.0, 0.0, 0.0])]
    rings = []
    for ring in range(1, n_rad + 1):
        rr = radius * ring / n_rad
        for j in range(n_theta):
            t = 2 * np.pi * j / n_theta
            verts.append(np.array([rr * np.cos(t), rr * np.sin(t), 0.0]))
        rings.append(ring)
    verts = np.array(verts)

    def vid(ring, j):  # ring 0 is the single centre vertex
        if ring == 0:
            return 0
        return 1 + (ring - 1) * n_theta + (j % n_theta)

    faces = []
    for j in range(n_theta):  # centre fan, CCW about +z
        faces.append((0, vid(1, j), vid(1, j + 1)))
    for ring in range(1, n_rad):
        for j in range(n_theta):
            a, b = vid(ring, j), vid(ring, j + 1)
            c, d = vid(ring + 1, j + 1), vid(ring + 1, j)
            faces.append((a, d, c))
            faces.append((a, c, b))
    return TriangleMesh(verts, np.array(faces, dtype=np.int64))


def cone(radius: float = 1.2, height: float = 1.6,
         n_theta: int = 48, n_slant: int = 12) -> TriangleMesh:
    """Open cone with the apex removed (small hole): concentrated curvature."""
    r_tip = 0.08
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    t = np.linspace(0.0, 1.0, n_slant)
    radii = r_tip + (radius - r_tip) * t
    zs = height * (1.0 - (radii - r_tip) / (radius - r_tip))
    th, rr = np.meshgrid(theta, radii)
    zz = np.broadcast_to(zs[:, None], th.shape)
    verts = np.column_stack([(rr * np.cos(th)).ravel(),
                             (rr * np.sin(th)).ravel(),
                             zz.ravel()])
    def vid(i, j):
        return i * n_theta + (j % n_theta)
    faces = []
    for i in range(n_slant - 1):
        for j in range(n_theta):
            a, b = vid(i, j), vid(i, j + 1)
            c, d = vid(i + 1, j + 1), vid(i + 1, j)
            faces.append((a, d, c))
            faces.append((a, c, b))
    return TriangleMesh(verts, np.array(faces, dtype=np.int64))


def torus(major: float = 1.6, minor: float = 0.55,
          n_theta: int = 48, n_phi: int = 24) -> TriangleMesh:
    """Closed genus-1 torus: a handled surface with two non-contractible loops.

    ``theta`` runs around the major ring, ``phi`` around the tube.  Both
    directions wrap periodically, so an editor must cut at least one loop
    before any globally seamless direction with winding can be laid.
    """
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    th, ph = np.meshgrid(theta, phi)
    rr = major + minor * np.cos(ph)
    verts = np.column_stack([
        (rr * np.cos(th)).ravel(),
        (rr * np.sin(th)).ravel(),
        (minor * np.sin(ph)).ravel()])

    def vid(i, j):  # tube row i, major col j
        return i * n_theta + (j % n_theta)

    faces = []
    for i in range(n_phi):
        i2 = (i + 1) % n_phi
        for j in range(n_theta):
            j2 = (j + 1) % n_theta
            faces.append((vid(i, j), vid(i, j2), vid(i2, j2)))
            faces.append((vid(i, j), vid(i2, j2), vid(i2, j)))
    return TriangleMesh(verts, np.array(faces, dtype=np.int64))


REGISTRY = {
    "cylinder": cylinder,
    "annulus": annulus,
    "disk": disk_with_interior,
    "band": curved_band,
    "cone": cone,
    "torus": torus,
}
