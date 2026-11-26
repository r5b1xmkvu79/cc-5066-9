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
from ..core.editor import FieldEditor, TopologyConflict
from ..core.field import _face_fan
from ..core.mesh import TriangleMesh
from ..core.models import REGISTRY
from ..core.topology import (FIXED_PRINT, SEAM_MOVES, numbering_mismatches,
                             seamless_obstructions, transport_around_loop)

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
    p.add_argument("--seam-policy", choices=[SEAM_MOVES, FIXED_PRINT],
                   default=SEAM_MOVES,
                   help="whether the cut may migrate with a dragged "
                        "singularity or the printed orientation is frozen")
    p.add_argument("--drag-edge", type=int, action="append", default=[],
                   help="interior edge id to drag a singularity across "
                        "(repeatable); pair with --drag-delta")
    p.add_argument("--drag-delta", type=int, default=1,
                   help="integer period jump added per dragged edge")
    p.add_argument("--seam-edge", type=int, action="append", default=[],
                   help="interior edge id to mark as a manual seam cut")
    p.add_argument("--loop-vertex", type=int, default=None,
                   help="print the closed-loop transport around this vertex")
    p.add_argument("--check-seamless", action="store_true",
                   help="report loops/vertices that block a seamless field "
                        "and the edges a human may cut")
    p.add_argument("--check-numbering", action="store_true",
                   help="locate adjacency edges whose integer jump disagrees "
                        "with the smooth geometric closure")
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

        # Optional interactive-topology layer.  The editor holds the integer
        # edge jumps; after every drag the field is re-solved and the (edited)
        # constraints/field feed the same render pipeline as an unedited run.
        editor = FieldEditor(mesh, constraints, mode=args.mode,
                             policy=args.seam_policy)
        if args.seam_edge:
            editor.set_seams(args.seam_edge)
        for eid in args.drag_edge:
            editor.drag_singularity(eid, args.drag_delta)
        constraints = editor.field.constraints
        field_for_report = editor.field

        if args.loop_vertex is not None:
            v = args.loop_vertex
            loop_faces, _, _, closed = _face_fan(mesh, v)
            if not closed:
                raise ValueError(
                    f"vertex {v} is on a boundary; its fan cannot close")
            loop = transport_around_loop(field_for_report,
                                         loop_faces + [loop_faces[0]])
            print(f"loop around vertex {v}: transport "
                  f"{np.degrees(loop.rotation_rad):+.4f} deg, winding "
                  f"{loop.winding}, holonomy {loop.holonomy_turns:+.4f} turn")
        if args.check_seamless:
            blocks = seamless_obstructions(field_for_report)
            if not blocks:
                print("seamless check: no obstruction; all anchors satisfiable "
                      "without a cut")
            for i, o in enumerate(blocks, 1):
                if o.kind == "singular_vertex":
                    print(f"obstruction {i}: singular vertex {o.vertex} index "
                          f"{o.index2 / 2.0:+.2f}; cut candidates "
                          f"{o.candidate_cuts}")
                else:
                    print(f"obstruction {i}: loop winding {o.winding:+d}; "
                          f"cut candidates {o.candidate_cuts}")
        if args.check_numbering:
            mismatches = numbering_mismatches(field_for_report)
            if not mismatches:
                print("numbering check: no mismatch; geometric closure is "
                      "smooth on every adjacency edge")
            for m in mismatches:
                print(f"numbering jump: edge {m.edge} faces "
                      f"({m.face0},{m.face1}) {m.periods:+.2f} period(s), "
                      f"wrap {np.degrees(m.expected_wrap):+.3f} deg")

        result = run_pipeline(
            mesh, constraints, mode=args.mode, method=args.method,
            spacing=args.spacing, subdiv=args.subdiv, pattern=args.pattern,
            dfield=field_for_report if args.drag_edge else None)
    except TopologyConflict as exc:
        print(f"texorient: seam policy conflict: {exc}", file=sys.stderr)
        return 3
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
