"""Texture-direction studio (the primary graphical entry point).

Run with::

    python3 -m texorient.gui.main

The studio lets the user pick a test mesh, *draw direction constraints*
directly on the surface (click a face, drag the desired texture direction in
its tangent plane), choose line/vector mode and the seam-vs-global UV method,
and lay the texture.  The result area renders the actual computed texture and
marks every singularity; a report panel lists the one-loop rotation residual,
the direction deviation and the scale stretch.  All numbers and the texture
come from the real solver -- any failure is raised in the pipeline and shown
to the user as a dialog instead of being hidden.
"""
from __future__ import annotations

import sys
import traceback

import numpy as np

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
                               QLabel, QMainWindow, QMessageBox, QPlainTextEdit,
                               QPushButton, QSpinBox, QVBoxLayout, QWidget)

from ..app import plotting
from ..app.pipeline import (PipelineResult, constraints_from_guide, make_mesh,
                            nearest_face, run_pipeline, tangent_angle_constraint)
from ..core.frames import tangent_frames
from ..core.mesh import TriangleMesh

MESHES = [("cylinder", "cylinder"), ("holed annulus", "annulus"),
          ("curved band", "band"), ("filled disk", "disk"), ("open cone", "cone")]
PRESETS = ["(draw manually)", "world X", "world Y", "world Z",
           "radial (outward)", "tangential swirl", "2x swirl (signed)"]


class View3D(FigureCanvasQTAgg):
    """Interactive 3-D canvas: rotate/zoom with the matplotlib toolbar, pick
    a face with a left click and drag a direction with a left drag."""

    def __init__(self, on_pick):
        self.fig = Figure(figsize=(6.4, 5.6), tight_layout=True)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.on_pick = on_pick
        self._down = None
        self.mpl_connect("button_press_event", self._press)
        self.mpl_connect("button_release_event", self._release)

    def reset(self):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.draw_idle()

    def _event_ray(self, event):
        """Unproject a 2-D mouse event to a 3-D point on the camera's focal
        plane; good enough for nearest-face picking on these small meshes."""
        if event.inaxes != self.ax or event.xdata is None:
            return None
        ax = self.ax
        # Display coordinates -> normalized device coordinates.
        w, h = self.get_width_height()
        ndx = (2.0 * event.x / max(w, 1)) - 1.0
        ndy = (2.0 * event.y / max(h, 1)) - 1.0
        # Camera basis in data coordinates.
        elev = np.radians(ax.elev)
        azim = np.radians(ax.azim)
        view = np.array([np.cos(elev) * np.cos(azim),
                         np.cos(elev) * np.sin(azim), np.sin(elev)])
        right = np.array([-np.sin(azim), np.cos(azim), 0.0])
        up = np.cross(view, right)
        center = np.array([*(ax.get_xlim()[1] + ax.get_xlim()[0]) / 2.0,
                           (ax.get_ylim()[1] + ax.get_ylim()[0]) / 2.0,
                           (ax.get_zlim()[1] + ax.get_zlim()[0]) / 2.0])
        span = (ax.get_xlim()[1] - ax.get_xlim()[0])
        return center + span * 0.5 * (ndx * right + ndy * up)

    def _press(self, event):
        if event.button != 1 or event.inaxes != self.ax:
            return
        self._down = self._event_ray(event)

    def _release(self, event):
        if event.button != 1 or self._down is None:
            self._down = None
            return
        up = self._event_ray(event)
        start, self._down = self._down, None
        if up is None or start is None:
            return
        self.on_pick(start, up)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TexOrient — surface texture direction studio")
        self.resize(1280, 820)
        self.mesh: TriangleMesh | None = None
        self.constraints = []
        self.result: PipelineResult | None = None

        self.view = View3D(self.handle_pick)
        self.toolbar = NavigationToolbar2QT(self.view, self)
        self.report = QPlainTextEdit(); self.report.setReadOnly(True)
        self.report.setMinimumHeight(180)

        central = QWidget()
        root = QHBoxLayout(central)
        root.addWidget(self._view_panel(), 3)
        root.addWidget(self._control_panel(), 2)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Load a mesh, then click-drag on the "
                                     "surface to draw a texture direction.")
        self.load_mesh()

    # ------------------------------------------------------------------ #
    def _view_panel(self) -> QWidget:
        box = QWidget(); lay = QVBoxLayout(box)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.view, 1)
        lay.addWidget(QLabel("Left click-drag on a face: click sets the face, "
                             "drag sets the texture direction. Use the toolbar "
                             "to orbit / zoom."))
        return box

    def _control_panel(self) -> QWidget:
        box = QWidget(); lay = QVBoxLayout(box)

        mesh_box = QGroupBox("Surface")
        mform = QFormLayout(mesh_box)
        self.mesh_combo = QComboBox()
        for label, key in MESHES:
            self.mesh_combo.addItem(label, key)
        self.mesh_combo.currentIndexChanged.connect(self.load_mesh)
        mform.addRow("model", self.mesh_combo)
        lay.addWidget(mesh_box)

        guide_box = QGroupBox("Direction constraints")
        gform = QFormLayout(guide_box)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(PRESETS)
        self.preset_combo.currentIndexChanged.connect(self.apply_preset)
        gform.addRow("preset guide", self.preset_combo)
        self.weight_spin = QDoubleSpinBox()
        self.weight_spin.setRange(0.1, 1000.0); self.weight_spin.setValue(20.0)
        gform.addRow("constraint weight", self.weight_spin)
        self.stride_spin = QSpinBox()
        self.stride_spin.setRange(1, 200); self.stride_spin.setValue(8)
        gform.addRow("preset stride (faces)", self.stride_spin)
        clear = QPushButton("Clear constraints")
        clear.clicked.connect(self.clear_constraints)
        gform.addRow(clear)
        self.constraint_count = QLabel("0 constraints")
        gform.addRow(self.constraint_count)
        lay.addWidget(guide_box)

        solve_box = QGroupBox("Field and texture coordinates")
        sform = QFormLayout(solve_box)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("line / stripes (mod π)", "line")
        self.mode_combo.addItem("vector (mod 2π)", "vector")
        sform.addRow("field type", self.mode_combo)
        self.method_combo = QComboBox()
        self.method_combo.addItem("seam: cuts, direction exact", "seam")
        self.method_combo.addItem("global: continuous, sacrifice direction",
                                  "global")
        sform.addRow("UV method", self.method_combo)
        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems(["stripe", "fibre"])
        sform.addRow("pattern", self.pattern_combo)
        self.spacing_spin = QDoubleSpinBox()
        self.spacing_spin.setRange(0.02, 5.0)
        self.spacing_spin.setDecimals(3); self.spacing_spin.setValue(0.25)
        self.spacing_spin.setSingleStep(0.05)
        sform.addRow("stripe spacing", self.spacing_spin)
        self.subdiv_spin = QSpinBox()
        self.subdiv_spin.setRange(1, 12); self.subdiv_spin.setValue(5)
        sform.addRow("render subdiv", self.subdiv_spin)
        lay.addWidget(solve_box)

        self.show_field = QCheckBox("show direction arrows")
        self.show_seams = QCheckBox("show seam cuts")
        self.show_sing = QCheckBox("mark singularities")
        self.show_sing.setChecked(True)
        lay.addWidget(self.show_field)
        lay.addWidget(self.show_seams)
        lay.addWidget(self.show_sing)

        run = QPushButton("Solve field and lay texture")
        run.setStyleSheet("font-weight:bold;")
        run.clicked.connect(self.safe_run)
        lay.addWidget(run)

        save = QPushButton("Save result image…")
        save.clicked.connect(self.save_image)
        lay.addWidget(save)
        lay.addWidget(QLabel("Report"))
        lay.addWidget(self.report, 1)
        return box

    # ------------------------------------------------------------------ #
    def load_mesh(self):
        key = self.mesh_combo.currentData() or "cylinder"
        try:
            self.mesh = make_mesh(key)
        except Exception as exc:  # pragma: no cover - defensive
            self._error("Could not build mesh", exc)
            return
        self.constraints = []
        self.result = None
        self._update_count()
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentIndex(0)
        self.preset_combo.blockSignals(False)
        self.draw_base()
        self.statusBar().showMessage(
            f"Loaded {key}: {self.mesh.n_faces} faces, "
            f"{self.mesh.n_vertices} vertices. Draw a direction or pick a preset.")

    def clear_constraints(self):
        self.constraints = []
        self._update_count()
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentIndex(0)
        self.preset_combo.blockSignals(False)
        self.draw_base()

    def _update_count(self):
        self.constraint_count.setText(f"{len(self.constraints)} constraints")

    # ------------------------------------------------------------------ #
    def handle_pick(self, start: np.ndarray, end: np.ndarray):
        if self.mesh is None:
            return
        try:
            face = nearest_face(self.mesh, start)
            direction = end - start
            if np.linalg.norm(direction) < 1e-9:
                # A plain click: default to the face's local e1 direction.
                n, e1, e2 = tangent_frames(self.mesh)
                direction = e1[face]
            cons = tangent_angle_constraint(
                self.mesh, face, direction, self.weight_spin.value())
            # Replace any earlier constraint on the same face.
            self.constraints = [c for c in self.constraints
                                if c.face != cons.face]
            self.constraints.append(cons)
            self._update_count()
            self.draw_base()
            self.statusBar().showMessage(
                f"Constraint on face {face} (local angle "
                f"{np.degrees(cons.angle):.1f}°). {len(self.constraints)} total.")
        except Exception as exc:
            self._error("Could not add constraint", exc)

    def apply_preset(self):
        if self.mesh is None:
            return
        choice = self.preset_combo.currentText()
        if choice == PRESETS[0]:
            return
        try:
            centers = self.mesh.face_centers()
            g = np.zeros((self.mesh.n_faces, 3))
            if choice == "world X":
                g[:, 0] = 1.0
            elif choice == "world Y":
                g[:, 1] = 1.0
            elif choice == "world Z":
                g[:, 2] = 1.0
            elif choice == "radial (outward)":
                g[:, 0] = centers[:, 0]
                g[:, 1] = centers[:, 1]
            elif choice == "tangential swirl":
                g[:, 0] = -centers[:, 1]
                g[:, 1] = centers[:, 0]
            elif choice == "2x swirl (signed)":
                theta = np.arctan2(centers[:, 1], centers[:, 0])
                g[:, 0] = np.cos(2 * theta)
                g[:, 1] = np.sin(2 * theta)
            stride = max(1, self.stride_spin.value())
            faces = list(range(0, self.mesh.n_faces, stride))
            self.constraints = constraints_from_guide(
                self.mesh, g, faces=faces, weight=self.weight_spin.value())
            self._update_count()
            self.draw_base()
            self.statusBar().showMessage(
                f"Applied '{choice}' guide to {len(faces)} faces.")
        except Exception as exc:
            self._error("Could not apply preset", exc)

    # ------------------------------------------------------------------ #
    def safe_run(self):
        try:
            if self.mesh is None:
                raise ValueError("load a mesh first")
            self.result = run_pipeline(
                self.mesh, self.constraints,
                mode=self.mode_combo.currentData(),
                method=self.method_combo.currentData(),
                spacing=self.spacing_spin.value(),
                subdiv=self.subdiv_spin.value(),
                pattern=self.pattern_combo.currentText())
            self.draw_result()
            self.fill_report()
            n_sing = len(self.result.singularity_vertices())
            self.statusBar().showMessage(
                f"Texture laid with method '{self.result.uv.method}'; "
                f"{n_sing} singularity point(s).")
        except Exception as exc:
            self._error("Direction/texture computation failed", exc,
                        details=traceback.format_exc())

    # ------------------------------------------------------------------ #
    def draw_base(self):
        self.view.reset()
        ax = self.view.ax
        plotting.draw_wireframe(ax, self.mesh)
        if self.constraints:
            plotting.draw_constraints(ax, self.mesh, self.constraints)
        plotting.style_axes(ax, "input surface + direction constraints")
        self.view.draw_idle()

    def draw_result(self):
        self.view.reset()
        ax = self.view.ax
        r = self.result
        plotting.draw_texture(ax, r)
        if self.show_seams.isChecked():
            plotting.draw_seams(ax, r)
        if self.show_sing.isChecked():
            plotting.draw_singularities(ax, r)
        if self.show_field.isChecked():
            plotting.draw_field_overlay(ax, r.mesh, r.field)
        plotting.style_axes(
            ax, f"{r.uv.method} UV — {r.pattern} texture "
                f"({len(r.singularity_vertices())} singularity)")
        self.view.draw_idle()

    def fill_report(self):
        r = self.result
        s = r.metrics_summary()
        rep = r.field.holonomy
        lines = []
        lines.append(f"UV method          : {r.uv.method}")
        lines.append(f"field type         : {r.field.mode} "
                     f"(period {r.field.period:.4f} rad)")
        lines.append(f"constraints        : {len(r.field.constraints)}")
        lines.append("")
        lines.append("Direction deviation (texture vs solved field):")
        lines.append(f"  max  {s['direction_deviation_deg_max']:.4f} deg")
        lines.append(f"  mean {s['direction_deviation_deg_mean']:.4f} deg")
        lines.append(f"  rms  {s['direction_deviation_deg_rms']:.4f} deg")
        lines.append("")
        lines.append("Scale stretch:")
        lines.append(f"  anisotropy max  {s['anisotropy_max']:.4f}")
        lines.append(f"  anisotropy mean {s['anisotropy_mean']:.4f}")
        lines.append(f"  mean scale      [{s['mean_scale_min']:.4f}, "
                     f"{s['mean_scale_max']:.4f}]")
        lines.append(f"  collapsed tris  {s['collapsed_triangle_count']}")
        lines.append("")
        lines.append(f"Seam cuts          : {s['seam_count']} edges")
        lines.append(f"seam gap (spacing) : {s['seam_gap_fraction_max']:.4f}")
        lines.append("")
        sing = rep.vertex[rep.singular]
        lines.append(f"Singularities      : {len(sing)} "
                     f"(marked on the surface)")
        if len(sing):
            lines.append("  one-loop rotation residual report:")
            idx = np.where(rep.singular)[0]
            for i in idx:
                lines.append(
                    f"    vertex {int(rep.vertex[i]):>5d}  "
                    f"index {rep.index_times2[i] / 2.0:+.2f}  "
                    f"holonomy {rep.total_rotation[i] / (2 * np.pi):+.4f} turn "
                    f"residual {np.degrees(rep.residual_rad[i]):+.3f} deg")
        else:
            lines.append("  no closed-loop index detected")
        lines.append("")
        lines.append("Full per-vertex holonomy table:")
        lines.append(rep.table())
        self.report.setPlainText("\n".join(lines))

    def save_image(self):
        if self.result is None:
            self._error("Nothing to save", ValueError(
                "solve and lay a texture before saving an image"))
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save result image", "texture_result.png",
            "PNG image (*.png)")
        if path:
            self.view.fig.savefig(path, dpi=150)
            self.statusBar().showMessage(f"Saved {path}")

    def _error(self, title: str, exc: Exception, details: str | None = None):
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle(title)
        box.setText(f"{title}:\n{exc}")
        if details:
            box.setDetailedText(details)
        box.exec()
        self.statusBar().showMessage(f"{title}: {exc}")


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
