# TexOrient

Tangent-bundle surface **texture direction** tool: draw local direction
constraints, propagate a real direction field across the surface's tangent
planes, build renderable texture coordinates, and mark every singularity with
its one-loop rotation residual.

**Graphical entry point (primary):**

```bash
python3 -m texorient.gui.main
```

**Command line (auxiliary):**

```bash
python3 -m texorient.app.cli --mesh cylinder --guide z --method seam
python3 -m texorient.app.cli --mesh torus --guide z --check-seamless
python3 examples/generate_results.py     # writes figures into results/
```

The studio also supports interactive topology editing: lockable direction
anchors, singularities dragged across shared edges, migrating versus
frozen-print seam policies, closed-loop parallel transport with winding
conflicts, cross-seam stroke readback, seamless-feasibility obstruction
reports and face renumbering with numbering-jump localisation.

See [`docs/README.md`](docs/README.md) for the full method, the seam-vs-global
trade-off, the topology-editing workflow, quantitative results and headless
graphics setup.

## Install

```bash
python3 -m pip install -r requirements.txt   # numpy scipy matplotlib PySide6
```

## Test

```bash
python3 -m pytest tests/ -q
```

The input mesh vertices are never modified; stripes/fibres follow the solved
direction, and changing a guide changes the rendered texture.
