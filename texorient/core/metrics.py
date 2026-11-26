"""Quantitative measurements of the constructed texture.

Three quantities are reported against the *solved* direction field:

* direction deviation (degrees): angle between the actual texture gradient
  direction on a face and the designed field direction, wrapped to the field
  period;
* scale stretch: per-face singular values of the 3-D -> UV linear map
  (sigma1, sigma2), their ratio (anisotropy) and the mean scale;
* seam mismatch: on cut edges, the UV gap expressed as a fraction of the
  local stripe spacing, showing what the seam choice actually costs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .field import DirectionField
from .uv import UVResult


@dataclass
class TextureMetrics:
    deviation_deg: np.ndarray       # per face, wrapped to [0, period/2]
    stretch_sigma1: np.ndarray
    stretch_sigma2: np.ndarray
    anisotropy: np.ndarray          # max/min singular value
    mean_scale: np.ndarray
    seam_gap_fraction: np.ndarray   # per cut edge (NaN otherwise)
    seam_edges: np.ndarray

    @property
    def collapsed_triangle_count(self) -> int:
        """Triangles where a continuous UV collapses (infinite anisotropy)."""
        return int(np.isinf(self.anisotropy).sum())

    def summary(self) -> dict:
        finite = self.deviation_deg[np.isfinite(self.deviation_deg)]
        gaps = self.seam_gap_fraction[np.isfinite(self.seam_gap_fraction)]
        finite_aniso = self.anisotropy[np.isfinite(self.anisotropy)]
        return {
            "direction_deviation_deg_max": float(np.max(finite)),
            "direction_deviation_deg_mean": float(np.mean(finite)),
            "direction_deviation_deg_rms": float(
                np.sqrt(np.mean(finite ** 2))),
            "anisotropy_max": float(np.max(finite_aniso))
            if finite_aniso.size else float("inf"),
            "anisotropy_mean": float(np.mean(finite_aniso))
            if finite_aniso.size else float("inf"),
            "collapsed_triangle_count": int(np.isinf(self.anisotropy).sum()),
            "mean_scale_min": float(np.min(self.mean_scale)),
            "mean_scale_max": float(np.max(self.mean_scale)),
            "seam_count": int(len(self.seam_edges)),
            "seam_gap_fraction_max": float(np.max(gaps)) if gaps.size else 0.0,
        }


def evaluate(dfield: DirectionField, uvres: UVResult,
             stripe_spacing: float = 1.0) -> TextureMetrics:
    mesh = dfield.mesh
    F = mesh.n_faces
    period = dfield.period
    deviation = np.zeros(F)
    s1 = np.ones(F); s2 = np.ones(F); aniso = np.ones(F); mscale = np.ones(F)

    for fi in range(F):
        uv_face = (uvres.uv[fi] if uvres.per_corner
                  else uvres.uv[mesh.faces[fi]])
        p = mesh.vertices[mesh.faces[fi]]
        e31, e32 = p[1] - p[0], p[2] - p[0]
        e21, e22 = uv_face[1] - uv_face[0], uv_face[2] - uv_face[0]
        # Face-frame Jacobian: E3 (3x2 world edge vectors) = F C with F the
        # orthonormal face frame and C the edge-vector coefficients in it.
        # Then J_face = E2 C^{-1} maps face-frame vectors to UV vectors.
        frame = np.column_stack([dfield.e1[fi], dfield.e2[fi]])
        c_mat = frame.T @ np.column_stack([e31, e32])
        e2_mat = np.column_stack([e21, e22])
        j_face = e2_mat @ np.linalg.inv(c_mat)
        sv = np.linalg.svd(j_face, compute_uv=False)
        s1[fi], s2[fi] = sv[0], max(sv[1], 1e-12)
        # A collapsed triangle (sigma2 at the numerical floor) means a single
        # continuous UV cannot cover this periodic surface without a seam.
        # Report a finite, clearly-flagged anisotropy instead of an enormous
        # ratio that hides the actual diagnosis.
        aniso[fi] = s1[fi] / s2[fi] if sv[1] > 1e-9 else np.inf
        mscale[fi] = 0.5 * (s1[fi] + s2[fi])
        # Texture u-axis direction in face frame = first row of J (du).
        du = j_face[0]
        tex_angle = np.arctan2(du[1], du[0])
        diff = (tex_angle - dfield.angles[fi] + period / 2) % period \
            - period / 2
        deviation[fi] = np.degrees(abs(diff))

    seam_edges = uvres.seams
    gap = np.full(mesh.edges.shape[0], np.nan)
    for eid in seam_edges:
        f0, f1 = mesh.edge_faces[eid]
        u, v = mesh.edges[eid]
        uv0 = _corner_uv(uvres, mesh, f0, u)
        uv1 = _corner_uv(uvres, mesh, f1, u)
        gap[eid] = np.linalg.norm(uv0 - uv1) / max(stripe_spacing, 1e-12)

    return TextureMetrics(deviation, s1, s2, aniso, mscale, gap, seam_edges)


def _corner_uv(uvres: UVResult, mesh, fi: int, vi: int) -> np.ndarray:
    if uvres.per_corner:
        slot = int(np.flatnonzero(mesh.faces[fi] == vi)[0])
        return uvres.uv[fi, slot]
    return uvres.uv[vi]
