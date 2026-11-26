"""Auxiliary command-line interface.

The GUI (``python3 -m texorient.gui.main``) is the primary entry point; this
CLI only mirrors the same :func:`~texorient.app.pipeline.run_pipeline` for
scripting and batch image generation.  It writes a texture image, a metrics
JSON file and prints the singularity / one-loop rotation-residual table.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from ..app.pipeline import (constraints_from_guide, make_mesh, run_pipeline)
from ..core.mesh import TriangleMesh
from ..core.models import REGISTRY

# A small library of analytic guides addressable by name from the CLI.
GUIDES = ["x", "y", "z", "radial", "swirl", "swirl2"]


def _guide_vectors(name: str, mesh: TriangleMesh) -> np.ndarray:
    centers = mesh.face_centers()
    g = np.zeros((mesh.n_faces, 3))
    if name == "x":
        g[:, 0] = 1.0
    elif name == "y":
        g[:, 1] = 1.0
    elif name == "z":
        g[:, 2] = 1.0
    elif name == "radial":
        g[:, 0] = centers[:, 0]
        g[:, 1] = centers[:, 1]
    elif name == "swirl":
        g[:, 0] = -centers[:, 1]
        g[:, 1] = centers[:, 0]
    elif name == "swirl2":
        theta = np.arctan2(centers[:, 1], centers[:, 0])
        g[:, 0] = np.cos(2 * theta)
        g[:, 1] = np.sin(2 * theta)
    else:  # pragma: no cover - argparse restricts choices
        raise ValueError(f"unknown guide {name}")
    return g


def _mesh_from_args(args) -> TriangleMesh:
    if args.obj:
        return TriangleMesh.from_obj(args.obj)
    return make_mesh(args.mesh)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="texorient",
        description="Surface texture direction solver (CLI companion to the "
                    "GUI). Solves a tangent direction field, builds texture "
                    "coordinates, measures deviation/stretch and renders the "
                    "actual stripe/fibre texture with singularity markers.")
    p.add_argument("--mesh", choices=sorted(REGISTRY),
                   default="cylinder", help="procedural test mesh (default: cylinder)")
    p.add_argument("--obj", help="optional triangulated OBJ file to use "
                                 "instead of a procedural mesh")
    p.add_argument("--guide", choices=GUIDES, default="z",
                   help="analytic direction guide")
    p.add_argument("--mode", choices=["line", "vector"], default="line",
                   help="line=stripes mod pi, vector=mod 2pi")
    p.add_argument("--method", choices=["seam", "global"], default="seam",
                   help="seam=direction exact with cuts, global=continuous")
    p.add_argument("--pattern", choices=["stripe", "fibre"], default="stripe")
    p.add_argument("--spacing", type=float, default=0.25)
    p.add_argument("--stride", type=int, default=1,
                   help="use one constraint every N faces (sparse guide)")
    p.add_argument("--weight", type=float, default=20.0)
    p.add_argument("--subdiv", type=int, default=5)
    p.add_argument("--outdir", default="results",
                   help="directory for the PNG image and metrics JSON")
    p.add_argument("--basename", default=None,
                   help="output file base name (default: mesh_method_guide)")
    p.add_argument("--no-overlay", action="store_true",
                   help="omit direction arrows, seams and singularity markers")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..app import plotting

    try:
        mesh = _mesh_from_args(args)
        guide = _guide_vectors(args.guide, mesh)
        faces = list(range(0, mesh.n_faces, max(1, args.stride)))
        constraints = constraints_from_guide(
            mesh, guide, faces=faces, weight=args.weight)
        result = run_pipeline(
            mesh, constraints, mode=args.mode, method=args.method,
            spacing=args.spacing, subdiv=args.subdiv, pattern=args.pattern)
    except Exception as exc:
        print(f"texorient: computation failed: {exc}", file=sys.stderr)
        return 2

    os.makedirs(args.outdir, exist_ok=True)
    base = args.basename or f"{args.obj or args.mesh}_{args.method}_{args.guide}"
    base = os.path.join(args.outdir, os.path.basename(base))

    fig = plt.figure(figsize=(8, 7), tight_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    plotting.draw_texture(ax, result)
    if not args.no_overlay:
        plotting.draw_seams(ax, result)
        plotting.draw_singularities(ax, result)
        plotting.draw_field_overlay(ax, result.mesh, result.field)
    plotting.style_axes(
        ax, f"{args.mesh or args.obj} | {args.method} UV | {args.guide} guide")
    png = base + ".png"
    fig.savefig(png, dpi=150)
    plt.close(fig)

    rep = result.field.holonomy
    sing = np.where(rep.singular)[0]
    metrics = result.metrics_summary()
    payload = {
        "mesh": args.obj or args.mesh,
        "n_faces": int(mesh.n_faces),
        "n_vertices": int(mesh.n_vertices),
        "mode": args.mode,
        "method": args.method,
        "guide": args.guide,
        "pattern": args.pattern,
        "stripe_spacing": args.spacing,
        "n_constraints": len(constraints),
        "metrics": metrics,
        "singularities": [
            {"vertex": int(rep.vertex[i]),
             "index": rep.index_times2[i] / 2.0,
             "holonomy_turns": float(rep.total_rotation[i] / (2 * np.pi)),
             "residual_deg": float(np.degrees(rep.residual_rad[i]))}
            for i in sing],
    }
    json_path = base + "_metrics.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"wrote {png}")
    print(f"wrote {json_path}")
    print(f"singularities: {len(sing)}")
    if len(sing):
        print(rep.table())
    print("direction deviation max (deg): "
          f"{metrics['direction_deviation_deg_max']:.4f}")
    print(f"anisotropy max: {metrics['anisotropy_max']:.4f}  "
          f"collapsed triangles: {metrics['collapsed_triangle_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
