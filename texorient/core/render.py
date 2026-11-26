"""Render the actual stripe/fibre texture on the surface.

The texture is evaluated from the constructed UV coordinates, not from the
input mesh's original UVs.  Each triangle is tessellated, every sub-vertex is
mapped to (u, v) by barycentric interpolation, and a procedural pattern
returns a colour; the coloured sub-triangles are drawn in 3-D.  Therefore
changing a direction guide changes the solved field, the UVs and the visible
texture -- arrows are only an optional overlay.
"""
from __future__ import annotations

import numpy as np

from .mesh import TriangleMesh
from .uv import UVResult


def _corner_uvs(uvres: UVResult, mesh: TriangleMesh):
    """Expand either UV representation to per-corner (F, 3, 2) UVs."""
    if uvres.per_corner:
        return uvres.uv
    return uvres.uv[mesh.faces]


def stripe_colors(uv_corners: np.ndarray, spacing: float,
                  bary: np.ndarray, pattern: str = "stripe") -> np.ndarray:
    """Pattern colour at barycentric sample points of every triangle.

    The texture +u axis follows the solved direction, so stripes that *lay
    along* the field are level sets of v (constant-v lines run in u).  The
    band phase therefore uses the v coordinate.  ``stripe`` gives periodic
    bands with phase v / spacing; ``fibre`` adds a fine longitudinal thread
    modulation so fibres look like individual threads running along u.
    Colours are greyscale RGB in [0, 1].
    """
    # uv_corners (F,3,2), bary (F, S, 3) -> sampled uv (F, S, 2)
    sampled = np.einsum("fsc,fck->fsk", bary, uv_corners)
    phase = sampled[..., 1] / spacing
    bands = 0.5 + 0.5 * np.cos(2 * np.pi * phase)
    if pattern == "fibre":
        # Dense cross-thread shading across the fibre (v), with a slow
        # longitudinal variation along u to separate parallel threads.
        thread = 0.5 + 0.5 * np.cos(
            2 * np.pi * sampled[..., 1] / (spacing * 0.2))
        along = 0.5 + 0.5 * np.cos(
            2 * np.pi * sampled[..., 0] / (spacing * 4.0))
        shade = 0.3 + 0.5 * bands * (0.65 + 0.35 * thread) + 0.12 * along
    else:
        shade = 0.25 + 0.7 * bands
    shade = np.clip(shade, 0, 1)
    return np.repeat(shade[..., None], 3, axis=-1)


def tessellate(mesh: TriangleMesh, uvres: UVResult, subdiv: int = 6):
    """Return world points, UVs and face id per sub-triangle vertex."""
    uv_corners = _corner_uvs(uvres, mesh)
    # barycentric lattice over the standard triangle
    weights = []
    for i in range(subdiv + 1):
        for j in range(subdiv + 1 - i):
            k = subdiv - i - j
            weights.append((i / subdiv, j / subdiv, k / subdiv))
    weights = np.array(weights)
    # triangulate the lattice once (index topology is shared by all faces)
    small = _lattice_triangles(subdiv)
    wpts = mesh.vertices[mesh.faces]          # (F,3,3)
    world = np.einsum("sc,fck->fsk", weights, wpts)
    uvs = np.einsum("sc,fck->fsk", weights, uv_corners)
    face_id = np.repeat(np.arange(mesh.n_faces), small.shape[0])
    tri_world = world[:, small, :].reshape(-1, 3, 3)
    tri_uv = uvs[:, small, :].reshape(-1, 3, 2)
    return tri_world, tri_uv, face_id


def _lattice_triangles(n: int) -> np.ndarray:
    idx = {}
    grid = []
    k = 0
    for i in range(n + 1):
        for j in range(n + 1 - i):
            idx[(i, j)] = k
            grid.append((i, j))
            k += 1
    tris = []
    for (i, j), a in idx.items():
        if (i + 1, j) in idx and (i, j + 1) in idx:
            tris.append((a, idx[(i + 1, j)], idx[(i, j + 1)]))
        if (i + 1, j) in idx and (i + 1, j - 1) in idx:
            tris.append((a, idx[(i + 1, j)], idx[(i + 1, j - 1)]))
    return np.array(tris, dtype=np.int64)


def textured_tris(mesh: TriangleMesh, uvres: UVResult,
                  spacing: float = 1.0, subdiv: int = 6,
                  pattern: str = "stripe"):
    """Convenience: sub-triangle world coords + mean face colour per tri."""
    tri_world, tri_uv, face_id = tessellate(mesh, uvres, subdiv)
    # colour from the centroid UV of each sub-triangle; bands are level sets
    # of v so the stripes run along the field-following u axis
    centroid_uv = tri_uv.mean(axis=1)
    phase = centroid_uv[:, 1] / spacing
    shade = 0.25 + 0.7 * (0.5 + 0.5 * np.cos(2 * np.pi * phase))
    if pattern == "fibre":
        tphase = centroid_uv[:, 1] / (spacing * 0.2)
        aphase = centroid_uv[:, 0] / (spacing * 4.0)
        shade = (0.3
                 + 0.5 * (0.5 + 0.5 * np.cos(2 * np.pi * phase))
                 * (0.65 + 0.35 * (0.5 + 0.5 * np.cos(2 * np.pi * tphase)))
                 + 0.12 * (0.5 + 0.5 * np.cos(2 * np.pi * aphase)))
    colors = np.clip(np.repeat(shade[:, None], 3, axis=1), 0, 1)
    return tri_world, colors, face_id
