"""Generate the reported result images and metrics for the three test models.

Run from the repository root::

    python3 examples/generate_results.py

Outputs are written to ``results/``:

* one texture image per (model, guide, UV method) with singularity markers,
  seam cuts and the direction-field overlay;
* a side-by-side ``guide_change_*`` pair proving that changing the guide
  actually rotates the local texture direction (no reused/old UV);
* a ``swirl_singularity`` image showing a genuine +2 index singularity;
* one ``*_metrics.json`` per run with deviation, stretch and holonomy numbers.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from texorient.app import plotting
from texorient.app.pipeline import (constraints_from_guide, make_mesh,
                                    run_pipeline)
from texorient.core.frames import tangent_frames, local_angles_of_3d_vectors
from texorient.core.field import DirectionConstraint

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results")
SPACING = 0.25


def _save(fig, name):
    path = os.path.join(OUT, name + ".png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


def _single(mesh, guide, name, method="seam", pattern="stripe", stride=1):
    faces = list(range(0, mesh.n_faces, stride))
    cons = constraints_from_guide(mesh, guide, faces=faces)
    result = run_pipeline(mesh, cons, mode="line", method=method,
                          spacing=SPACING, subdiv=6, pattern=pattern)
    fig = plt.figure(figsize=(8, 7), tight_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    plotting.draw_texture(ax, result)
    plotting.draw_seams(ax, result)
    plotting.draw_singularities(ax, result)
    plotting.draw_field_overlay(ax, result.mesh, result.field)
    plotting.style_axes(ax, name.replace("_", " "))
    _save(fig, name)
    with open(os.path.join(OUT, name + "_metrics.json"), "w",
              encoding="utf-8") as fh:
        json.dump(result.metrics_summary(), fh, indent=2)
    return result


def guide_change_pair():
    """The same mesh with two orthogonal guides must rotate the texture."""
    mesh = make_mesh("band")
    n, e1, e2 = tangent_frames(mesh)
    fig = plt.figure(figsize=(13, 6), tight_layout=True)
    angles = []
    for k, axis in enumerate((0, 1)):
        g = np.zeros((mesh.n_faces, 3)); g[:, axis] = 1.0
        ang = local_angles_of_3d_vectors(mesh, g, e1, e2, n)
        cons = [DirectionConstraint(f, float(ang[f]))
                for f in range(0, mesh.n_faces, 6)]
        result = run_pipeline(mesh, cons, mode="line", method="seam",
                              spacing=SPACING, subdiv=6)
        angles.append(result.field.angles.copy())
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        plotting.draw_texture(ax, result)
        plotting.style_axes(
            ax, f"guide = world {'X' if axis == 0 else 'Y'}")
    _save(fig, "guide_change_band")
    d = np.angle(np.exp(1j * (angles[0] - angles[1])))
    print("guide-change median angle delta (deg):",
          round(float(np.degrees(np.median(np.abs(d)))), 2))


def swirl_singularity():
    """A 2*theta winding guide forces a real +2 Poincare-index singularity."""
    mesh = make_mesh("disk")
    n, e1, e2 = tangent_frames(mesh)
    c = mesh.face_centers()
    theta = np.arctan2(c[:, 1], c[:, 0])
    g = np.stack([np.cos(2 * theta), np.sin(2 * theta),
                  np.zeros_like(theta)], axis=1)
    ang = local_angles_of_3d_vectors(mesh, g, e1, e2, n)
    cons = [DirectionConstraint(f, float(ang[f]), weight=40.0)
            for f in range(mesh.n_faces)]
    result = run_pipeline(mesh, cons, mode="line", method="seam",
                          spacing=SPACING, subdiv=6)
    fig = plt.figure(figsize=(8, 7), tight_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    plotting.draw_texture(ax, result)
    plotting.draw_seams(ax, result)
    plotting.draw_singularities(ax, result)
    plotting.style_axes(ax, "2-theta guide -> +2 index singularity (marked)")
    _save(fig, "swirl_singularity_disk")
    rep = result.field.holonomy
    i0 = int(np.where(rep.vertex == 0)[0][0])
    print("centre index:", rep.index_times2[i0] / 2.0,
          "holonomy turns:", rep.total_rotation[i0] / (2 * np.pi),
          "residual deg:", np.degrees(rep.residual_rad[i0]))


def main():
    os.makedirs(OUT, exist_ok=True)
    # Required models, each with a seam-built texture.
    cyl = make_mesh("cylinder")
    gz = np.zeros((cyl.n_faces, 3)); gz[:, 2] = 1.0
    _single(cyl, gz, "cylinder_seam_z")

    ann = make_mesh("annulus")
    gx = np.zeros((ann.n_faces, 3)); gx[:, 0] = 1.0
    _single(ann, gx, "annulus_seam_x")
    # Annulus with the continuous method: no cuts, slight trade-off visible.
    _single(ann, gx, "annulus_global_x", method="global")

    band = make_mesh("band")
    gbx = np.zeros((band.n_faces, 3)); gbx[:, 0] = 1.0
    _single(band, gbx, "band_seam_x")
    _single(band, gbx, "band_global_x", method="global")
    _single(band, gbx, "band_fibre_x", pattern="fibre")

    guide_change_pair()
    swirl_singularity()
    print("all results written to", OUT)


if __name__ == "__main__":
    main()
