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

from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
                               QLabel, QMainWindow, QMessageBox, QPlainTextEdit,
                               QPushButton, QSpinBox, QVBoxLayout, QWidget)

from ..app import plotting
from ..app.pipeline import (PipelineResult, constraints_from_guide, make_mesh,
                            nearest_face, run_pipeline, tangent_angle_constraint)
from ..core.editor import FieldEditor, TopologyConflict
from ..core.field import _face_fan
from ..core.frames import tangent_frames
from ..core.mesh import TriangleMesh
from ..core.topology import FIXED_PRINT, SEAM_MOVES, numbering_mismatches

MESHES = [("cylinder", "cylinder"), ("holed annulus", "annulus"),
          ("curved band", "band"), ("filled disk", "disk"), ("open cone", "cone"),
          ("handled torus", "torus")]
PRESETS = ["(draw manually)", "world X", "world Y", "world Z",
           "radial (outward)", "tangential swirl", "2x swirl (signed)"]
# What a left click-drag does in the 3-D view.
PICK_CONSTRAINT = "draw direction constraint"
PICK_ANCHOR = "lock direction anchor"
PICK_SINGULARITY = "drag singularity across edge"
PICK_SEAM = "mark / clear seam edge"


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
        self.editor: FieldEditor | None = None
        self.manual_seams: set[int] = set()
        self.last_loop_faces: list[int] = []

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

        topo_box = QGroupBox("Direction anchors, singularities and seams")
        tform = QFormLayout(topo_box)
        self.pick_combo = QComboBox()
        self.pick_combo.addItems([PICK_CONSTRAINT, PICK_ANCHOR,
                                  PICK_SINGULARITY, PICK_SEAM])
        tform.addRow("click-drag mode", self.pick_combo)
        self.policy_combo = QComboBox()
        self.policy_combo.addItem("seam migrates with singularity", SEAM_MOVES)
        self.policy_combo.addItem("printed coat of arms never turns",
                                  FIXED_PRINT)
        tform.addRow("seam policy", self.policy_combo)
        self.drag_spin = QSpinBox()
        self.drag_spin.setRange(-3, 3); self.drag_spin.setValue(1)
        tform.addRow("singularity drag (periods)", self.drag_spin)
        loop_row = QHBoxLayout()
        self.loop_vertex_spin = QSpinBox()
        self.loop_vertex_spin.setRange(0, 10_000_000); self.loop_vertex_spin.setValue(0)
        loop_row.addWidget(self.loop_vertex_spin)
        loop_btn = QPushButton("transport around vertex loop")
        loop_btn.clicked.connect(self.demo_loop_transport)
        loop_row.addWidget(loop_btn)
        tform.addRow("loop at vertex", loop_row)
        obs_btn = QPushButton("check seamless feasibility…")
        obs_btn.clicked.connect(self.show_obstructions)
        tform.addRow(obs_btn)
        mism_btn = QPushButton("locate numbering-jump edges…")
        mism_btn.clicked.connect(self.show_numbering_mismatches)
        tform.addRow(mism_btn)
        self.anchor_count = QLabel("0 anchors")
        tform.addRow(self.anchor_count)
        lay.addWidget(topo_box)

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
        self.show_anchors = QCheckBox("show locked direction anchors")
        self.show_anchors.setChecked(True)
        self.show_loop = QCheckBox("show transported closed loop")
        lay.addWidget(self.show_field)
        lay.addWidget(self.show_seams)
        lay.addWidget(self.show_sing)
        lay.addWidget(self.show_anchors)
        lay.addWidget(self.show_loop)

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
        self.editor = None
        self.manual_seams = set()
        self.last_loop_faces = []
        self._update_count()
        self._update_anchor_count()
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentIndex(0)
        self.preset_combo.blockSignals(False)
        self.draw_base()
        self.statusBar().showMessage(
            f"Loaded {key}: {self.mesh.n_faces} faces, "
            f"{self.mesh.n_vertices} vertices. Draw a direction or pick a preset.")

    def clear_constraints(self):
        self.constraints = []
        self.editor = None
        self.manual_seams = set()
        self.last_loop_faces = []
        self._update_count()
        self._update_anchor_count()
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentIndex(0)
        self.preset_combo.blockSignals(False)
        self.draw_base()

    def _update_count(self):
        self.constraint_count.setText(f"{len(self.constraints)} constraints")

    # ------------------------------------------------------------------ #
    def _ensure_editor(self) -> FieldEditor:
        if self.editor is None:
            self.editor = FieldEditor(
                self.mesh, self.constraints,
                mode=self.mode_combo.currentData(),
                policy=self.policy_combo.currentData())
        self.editor.set_seams(self.manual_seams)
        return self.editor

    def handle_pick(self, start: np.ndarray, end: np.ndarray):
        if self.mesh is None:
            return
        mode = self.pick_combo.currentText()
        try:
            if mode == PICK_ANCHOR:
                self._pick_anchor(start, end)
            elif mode == PICK_SINGULARITY:
                self._pick_singularity_drag(start)
            elif mode == PICK_SEAM:
                self._pick_seam(start)
            else:
                self._pick_constraint(start, end)
        except TopologyConflict as exc:
            self._error("Seam policy conflict", exc)
        except Exception as exc:
            self._error("Could not edit field", exc,
                        details=traceback.format_exc())

    def _pick_constraint(self, start: np.ndarray, end: np.ndarray):
        face = nearest_face(self.mesh, start)
        direction = end - start
        if np.linalg.norm(direction) < 1e-9:
            n, e1, e2 = tangent_frames(self.mesh)
            direction = e1[face]
        cons = tangent_angle_constraint(
            self.mesh, face, direction, self.weight_spin.value())
        self.constraints = [c for c in self.constraints
                            if c.face != cons.face]
        self.constraints.append(cons)
        self.editor = None       # constraints changed: rebuild the editor
        self._update_count()
        self.draw_base()
        self.statusBar().showMessage(
            f"Constraint on face {face} (local angle "
            f"{np.degrees(cons.angle):.1f}°). {len(self.constraints)} total.")

    def _pick_anchor(self, start: np.ndarray, end: np.ndarray):
        face = nearest_face(self.mesh, start)
        direction = end - start
        if np.linalg.norm(direction) >= 1e-9:
            cons = tangent_angle_constraint(self.mesh, face, direction,
                                            self.weight_spin.value())
            angle = cons.angle
        else:
            angle = None
        editor = self._ensure_editor()
        anchor = editor.add_anchor(face, angle=angle)
        self._update_anchor_count()
        self.draw_base()
        self.statusBar().showMessage(
            f"Locked direction anchor on face {anchor.face} "
            f"(angle {np.degrees(anchor.angle):.1f}°).")

    def _pick_singularity_drag(self, start: np.ndarray):
        editor = self._ensure_editor()
        face = nearest_face(self.mesh, start)
        # The edge bordering the picked face nearest the click point is the
        # shared edge the singularity is dragged across.
        candidates = [e for e in self.mesh.face_edges(face)
                      if self.mesh.edge_faces[e, 1] >= 0]
        if not candidates:
            raise ValueError("picked face has no interior edge to cross")
        edge = min(candidates, key=lambda e: float(np.linalg.norm(
            self.mesh.vertices[self.mesh.edges[e]].mean(axis=0) - start)))
        result = editor.drag_singularity(edge, self.drag_spin.value())
        self.manual_seams = set(editor.seam_edges)
        self.draw_base()
        self.statusBar().showMessage(
            f"Dragged singularity across edge {edge} by {result.delta} "
            f"period(s); indices changed at vertices "
            f"{result.moved_vertices}; total index conserved.")

    def _pick_seam(self, start: np.ndarray):
        face = nearest_face(self.mesh, start)
        candidates = [e for e in self.mesh.face_edges(face)
                      if self.mesh.edge_faces[e, 1] >= 0]
        edge = min(candidates, key=lambda e: float(np.linalg.norm(
            self.mesh.vertices[self.mesh.edges[e]].mean(axis=0) - start)))
        if edge in self.manual_seams:
            self.manual_seams.discard(edge)
            action = "cleared seam on"
        else:
            self.manual_seams.add(edge)
            action = "marked seam on"
        if self.editor is not None:
            self.editor.set_seams(self.manual_seams)
        self.draw_base()
        self.statusBar().showMessage(f"{action} edge {edge}.")

    def demo_loop_transport(self):
        if self.mesh is None:
            return
        try:
            editor = self._ensure_editor()
            v = self.loop_vertex_spin.value()
            if v >= self.mesh.n_vertices or not self.mesh.vertex_faces[v]:
                raise ValueError(f"vertex {v} does not exist on this mesh")
            faces, _, _, closed = _face_fan(self.mesh, v)
            if not closed:
                raise ValueError(
                    f"vertex {v} is on a boundary; its fan cannot close, so no "
                    "quantised loop holonomy exists there")
            loop_faces = faces + [faces[0]]
            self.last_loop_faces = loop_faces
            loop = editor.closed_loop(loop_faces)
            conflicts = editor.loop_winding_conflicts(loop_faces)
            msg = (f"Loop around vertex {v}: transport "
                   f"{np.degrees(loop.rotation_rad):+.3f}°, winding "
                   f"{loop.winding}, holonomy "
                   f"{loop.holonomy_turns:+.4f} turn(s).")
            if conflicts:
                cut = conflicts[0].candidate_cuts
                msg += (f" Winding conflicts with the locked anchors; cut one "
                        f"of {len(cut)} loop edges or release an anchor.")
            self.draw_base()
            self.statusBar().showMessage(msg)
        except Exception as exc:
            self._error("Loop transport failed", exc,
                        details=traceback.format_exc())

    def show_obstructions(self):
        if self.mesh is None:
            return
        try:
            editor = self._ensure_editor()
            blocks = editor.seamless_blocks()
            if not blocks:
                QMessageBox.information(
                    self, "Seamless feasibility",
                    "No obstruction: a seamless field can satisfy every "
                    "current anchor.")
                return
            lines = [f"{len(blocks)} obstruction(s) to a seamless field:"]
            for i, o in enumerate(blocks[:12], 1):
                if o.kind == "singular_vertex":
                    lines.append(
                        f"{i}. singular vertex {o.vertex}, index "
                        f"{o.index2 / 2.0:+.2f}; choose one of "
                        f"{len(o.candidate_cuts)} fan edges to cut")
                else:
                    lines.append(
                        f"{i}. closed loop winding {o.winding:+d}; choose one "
                        f"of {len(o.candidate_cuts)} loop edges to cut")
            QMessageBox.information(
                self, "Seamless feasibility — manual cut choices",
                "\n".join(lines))
        except Exception as exc:
            self._error("Feasibility check failed", exc,
                        details=traceback.format_exc())

    def show_numbering_mismatches(self):
        if self.editor is None:
            QMessageBox.information(
                self, "Numbering jumps",
                "Edit or solve the field before checking numbering jumps.")
            return
        mismatches = numbering_mismatches(self.editor.field)
        if not mismatches:
            QMessageBox.information(
                self, "Numbering jumps",
                "No mismatch: geometry closes smoothly on every adjacency "
                "edge despite the current face numbering.")
            return
        lines = [f"{len(mismatches)} edge(s) jump by a whole period while the "
                 "geometry stays smooth:"]
        for m in mismatches[:12]:
            lines.append(
                f"edge {m.edge} (faces {m.face0},{m.face1}): "
                f"{m.periods:+.2f} period(s), geometric wrap "
                f"{np.degrees(m.expected_wrap):+.2f}°")
        QMessageBox.warning(self, "Numbering-jump edges", "\n".join(lines))

    def _update_anchor_count(self):
        n_anchor = 0 if self.editor is None else len(self.editor.anchors)
        self.anchor_count.setText(f"{n_anchor} anchors")

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
            self.editor = None
            self.manual_seams = set()
            self.last_loop_faces = []
            self._update_count()
            self._update_anchor_count()
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
            edited = self.editor.field if self.editor is not None else None
            self.result = run_pipeline(
                self.mesh, self.constraints,
                mode=self.mode_combo.currentData(),
                method=self.method_combo.currentData(),
                spacing=self.spacing_spin.value(),
                subdiv=self.subdiv_spin.value(),
                pattern=self.pattern_combo.currentText(),
                dfield=edited)
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
        if self.editor is not None:
            if self.manual_seams:
                plotting.draw_edges(ax, self.mesh, self.manual_seams,
                                    color="red", label="manual seam")
            if self.show_anchors.isChecked() and self.editor.anchors:
                plotting.draw_anchors(ax, self.mesh, self.editor.anchors,
                                      field=self.editor.field)
        if self.show_loop.isChecked() and self.last_loop_faces:
            plotting.draw_transport_loop(ax, self.mesh, self.last_loop_faces)
        plotting.style_axes(ax, "input surface + direction constraints")
        self.view.draw_idle()

    def draw_result(self):
        self.view.reset()
        ax = self.view.ax
        r = self.result
        plotting.draw_texture(ax, r)
        if self.show_seams.isChecked():
            plotting.draw_seams(ax, r)
        if self.manual_seams:
            plotting.draw_edges(ax, self.mesh, self.manual_seams,
                                color="orange", label="editor seam")
        if self.show_sing.isChecked():
            plotting.draw_singularities(ax, r)
        if self.show_field.isChecked():
            plotting.draw_field_overlay(ax, r.mesh, r.field)
        if self.editor is not None and self.show_anchors.isChecked() \
                and self.editor.anchors:
            plotting.draw_anchors(ax, r.mesh, self.editor.anchors,
                                  field=r.field)
        if self.show_loop.isChecked() and self.last_loop_faces:
            plotting.draw_transport_loop(ax, r.mesh, self.last_loop_faces)
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
