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
python3 examples/generate_results.py     # writes figures into results/
```

See [`docs/README.md`](docs/README.md) for the full method, the seam-vs-global
trade-off, quantitative results and headless graphics setup.

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
