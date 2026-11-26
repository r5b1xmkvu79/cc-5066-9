# TexOrient — surface texture direction tool

TexOrient lets you specify a **texture direction (走向)** on local regions of a
triangle mesh, propagates a **direction field through the tangent planes** of
the surface (not by averaging 3-D vectors), and builds renderable local texture
coordinates so that **stripe or fibre patterns lay along the designed
direction**. Where curvature, fixed boundaries or topology make a globally
consistent direction impossible, the singularities are **computed and marked**,
together with the **one-loop rotation residual (一周的旋转残差)**, and you choose
between adding **seam cuts (切缝)** or **sacrificing local direction**.

The input mesh geometry is never moved: only direction angles and `(u, v)`
coordinates are produced.

---

## 1. Graphical entry point (primary)

The texture-making studio is the main way to use the tool:

```bash
python3 -m texorient.gui.main
```

> On a minimal/headless system where Qt reports a missing `libEGL.so.1` (or
> `libGL.so.1`, `libxkbcommon.so.0`, `libdbus-1.so.3`), install those runtime
> libraries through your package manager (`libegl1 libgl1 libxkbcommon0
> libdbus-1-3` on Debian/Ubuntu). See **Appendix A** for a rootless setup.

### Workflow

1. **Choose a surface** — cylinder, holed annulus, curved band, filled disk or
   open cone (top-left *Surface* box).
2. **Draw direction constraints** in the 3-D view:
   * left **click a face** to select it, then **drag** in the direction you want
     the texture to run. The drag is projected into that face's tangent plane;
   * or pick a **preset guide** (world X/Y/Z, radial, tangential swirl, 2×
     swirl). The *stride* controls how sparse the guide is.
3. **Choose the field type** — `line / stripes (mod π)` for stripes and fibres,
   or `vector (mod 2π)`.
4. **Choose the UV construction method** (see §3):
   * `seam` — direction exact, cuts where continuity is impossible;
   * `global` — continuous, sacrifices local direction near singularities.
5. Press **“Solve field and lay texture”**.

### Result area

The right-hand view renders the **actual texture computed from the constructed
UVs** and overlays:

* a magenta `★` with the **Poincaré index label** at every singularity;
* red **seam cut** edges (seam method);
* optional blue **direction-field arrows**.

The report panel lists, per run:

* direction deviation (max / mean / RMS degrees) between the texture and the
  solved field;
* scale stretch — singular values, anisotropy, mean scale and any collapsed
  triangles;
* the seam count and seam gap as a fraction of stripe spacing;
* the **singularity table** with each vertex's index, holonomy in turns and the
  **one-loop rotation residual in degrees**.

Any failure (no constraint, a direction normal to a face, a disconnected mesh,
…) is raised by the solver and shown in a **modal error dialog** with details;
it is never silently smoothed over.

Use **“Save result image…”** to export the current view as PNG.

---

## 2. Why this is a tangent-bundle field, not vector averaging

A direction lives in the surface's **tangent bundle**. Two directions on
neighbouring faces sit in different tangent planes and can only be compared
after **parallel transport**. TexOrient therefore:

1. builds a deterministic orthonormal frame `(n, e1, e2)` on every face;
2. represents each direction by a **local angle** in that frame;
3. computes a per-edge **transport rotation `ρ`** by rotating about the shared
   edge (Rodrigues) and reading the result in the neighbour's frame;
4. solves one angle per face with the transport angle built into every edge
   coupling.

There is no “project a 3-D vector, normalize and assume continuity” step. The
only projection happens when converting an *input guide* to a local angle, and
that vector is re-orthogonalised so an off-plane guide cannot tilt the field.

### Periodic cover and singularities

A line field is defined modulo `π`, a vector field modulo `2π`. A single
real-valued angle function always has zero closed-loop integral, so it cannot
represent winding. The solver:

* lifts angles onto one real cover along a dual-graph spanning tree;
* determines the integer period jump on every non-tree edge by snapping the
   transported mismatch to the nearest multiple of the period;
* runs a continuous least-squares relaxation with those integers fixed.

Holonomy is then measured **intrinsically** by walking each vertex fan,
parallel-transporting the angles and unwrapping them. For a closed fan the
total rotation is `index × 2π`; the doubled index `2·index = round(total/π)` is
an integer for both field types (line indices may be half-integers). These
singularities are reported, never hidden. The sum of indices is conserved
(Poincaré–Hopf) even when a sparse guide spreads one singularity into several.

---

## 3. The two declared UV construction methods

| | `seam` (切缝) | `global` (牺牲局部方向) |
|---|---|---|
| continuity | per-face similarity over a dual spanning tree; cut edges are the seams | one continuous `(u,v)` per vertex from edge-vector least squares |
| direction | **exact on every face** (0° deviation, isotropic scale 1) | deviates near singularities / curvature |
| scale | uniform, anisotropy = 1 | shears/stretches; can **collapse** on a periodic surface |
| topology | cuts are placed where the field itself jumps | cannot wrap a closed periodic direction without a seam |

On a periodic cylinder the global method necessarily collapses triangles; the
metrics flag this explicitly (`collapsed_triangle_count`) instead of hiding it,
while the seam method covers the same cylinder exactly.

---

## 4. Command line (auxiliary)

The CLI mirrors the same pipeline for scripting and batch rendering; the GUI is
the primary interface.

```bash
# cylinder, vertical guide, seam-built stripes
python3 -m texorient.app.cli --mesh cylinder --guide z --method seam

# holed annulus, continuous UV (no cuts)
python3 -m texorient.app.cli --mesh annulus --guide x --method global

# curved band with a fibre pattern
python3 -m texorient.app.cli --mesh band --guide x --pattern fibre

# 2-theta swirl on the disk to expose the +2 singularity
python3 -m texorient.app.cli --mesh disk --guide swirl2 --method seam

# your own triangulated OBJ (geometry is read, never modified)
python3 -m texorient.app.cli --obj surface.obj --guide x
```

Each run writes `<name>.png` and `<name>_metrics.json` into `--outdir`
(default `results/`) and prints the singularity table. Guides:
`x y z radial swirl swirl2`; modes: `line vector`; methods: `seam global`.

### Regenerate the reported figures

```bash
python3 examples/generate_results.py
```

This produces, in `results/`, the cylinder / annulus / band textures for both
methods, the fibre pattern, a side-by-side **guide-change** pair (proving the
texture rotates when the guide changes), and the swirl singularity figure with
its metrics JSON.

---

## 5. Quantitative results

Representative numbers from `results/*_metrics.json` (seam method, stripe
spacing 0.25):

| model / guide | max direction deviation | max anisotropy | mean scale | collapsed |
|---|---|---|---|---|
| cylinder / z | 0.0000° | 1.0000 | 1.000 | 0 |
| annulus / x | 0.0000° | 1.0000 | 1.000 | 0 |
| band / x (seam) | ~0.005° | 1.0000 | 1.000 | 0 |
| band / x (global) | 0.1435° | 1.0597 | 0.99–1.01 | 0 |
| annulus / x (global) | 0.0000° | 1.0000 | 1.000 | 0 |
| cylinder (global) | up to 90° | ∞ | — | **1056** |

The dense `2θ` swirl on the disk yields exactly one centre singularity with
index **+2**, holonomy **2 turns**, and one-loop residual **0.000°**. A sparse
guide (stride 8) splits it into one `−1` and six `+½` vertices whose indices
still sum to `+2`.

### Changing the guide changes the texture

`results/guide_change_band.png` renders the same band with a world-X versus
world-Y guide; the measured median field-angle difference is **87.6°** (≈ 90°),
and the stripes switch from longitudinal to transverse. No pre-existing UV is
reused — every UV is rebuilt from the freshly solved field.

---

## 6. Project layout

```
texorient/
  core/        mesh connectivity, frames/transport, field solver, holonomy,
               UV construction, metrics, procedural models, stripe renderer
  app/         pipeline.py (shared orchestration), plotting.py, cli.py
  gui/main.py  the graphical texture studio (primary entry point)
examples/      generate_results.py
tests/         27 tests: connectivity, transport, field, holonomy, UV,
               metrics, end-to-end rendering and guide response
results/       generated PNG images and metrics JSON
docs/          this documentation
```

### Run the tests

```bash
python3 -m pytest tests/ -q
```

---

## Appendix A. Rootless graphics libraries (headless container)

If you cannot install system packages, download the Debian `.deb` files for
`libegl1 libgl1 libglvnd0 libglx0 libegl-mesa0 libgbm1 libdrm2
libwayland-client0 libwayland-server0 libxkbcommon0 libdbus-1-3`, extract them
with `dpkg-deb -x` into a user prefix, and export:

```bash
export LD_LIBRARY_PATH="$HOME/.local/qtlibs/lib/aarch64-linux-gnu:\
$HOME/.local/qtlibs/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH"
export QT_QPA_PLATFORM=offscreen   # only for headless GUI smoke tests
```

The CLI and image generation use the headless `Agg` backend and need no display.
