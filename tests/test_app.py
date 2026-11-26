"""Tests for the shared application pipeline and the rendered texture.

These cover the end-to-end behaviour the task cares about: the rendered stripe
long-axis follows the solved direction, changing a guide rotates the texture,
the geometry is untouched, and singularities are reported with their
one-loop rotation residual.
"""
import numpy as np
import pytest

from texorient.app.pipeline import (constraints_from_guide, make_mesh,
                                    nearest_face, run_pipeline,
                                    tangent_angle_constraint)
from texorient.core.frames import tangent_frames


def _axis_guide(mesh, axis):
    g = np.zeros((mesh.n_faces, 3))
    g[:, axis] = 1.0
    return g


def _stripe_long_axis_angle_to_guide(result, guide):
    """Angle between the texture u-axis (stripe long axis) in the world and
    the guide direction, evaluated through each face's Jacobian."""
    mesh = result.mesh
    out = []
    for fi in range(mesh.n_faces):
        tri = mesh.faces[fi]
        p = mesh.vertices[tri]
        frame = np.column_stack([result.field.e1[fi], result.field.e2[fi]])
        c_mat = frame.T @ np.column_stack([p[1] - p[0], p[2] - p[0]])
        uv_face = (result.uv.uv[fi] if result.uv.per_corner
                   else result.uv.uv[tri])
        e2 = np.column_stack([uv_face[1] - uv_face[0],
                              uv_face[2] - uv_face[0]])
        j = e2 @ np.linalg.inv(c_mat)
        along = frame @ np.linalg.inv(j) @ np.array([1.0, 0.0])
        along /= np.linalg.norm(along)
        g = guide[fi]
        g = g - g @ result.field.n[fi] * result.field.n[fi]
        g /= np.linalg.norm(g)
        out.append(np.degrees(np.arccos(np.clip(abs(along @ g), -1, 1))))
    return np.array(out)


def test_rendered_stripes_follow_guide_on_all_required_models():
    # The seam construction is direction-exact up to least-squares relaxation;
    # tolerance stays well below 0.1 degree so any real misalignment fails.
    for name, axis in (("cylinder", 2), ("annulus", 0), ("band", 0)):
        mesh = make_mesh(name)
        guide = _axis_guide(mesh, axis)
        cons = constraints_from_guide(mesh, guide)
        result = run_pipeline(mesh, cons, method="seam", subdiv=2)
        ang = _stripe_long_axis_angle_to_guide(result, guide)
        assert ang.max() < 1e-2, name


def test_changing_guide_rotates_rendered_texture():
    mesh = make_mesh("band")
    fields = []
    for axis in (0, 1):
        guide = _axis_guide(mesh, axis)
        result = run_pipeline(mesh, constraints_from_guide(mesh, guide),
                              method="seam", subdiv=2)
        fields.append(result.field.angles.copy())
    d = np.angle(np.exp(1j * (fields[0] - fields[1])))
    assert np.all(np.abs(np.abs(d) - np.pi / 2) < 0.2)


def test_pipeline_requires_constraint():
    mesh = make_mesh("annulus")
    with pytest.raises(ValueError):
        run_pipeline(mesh, [], method="seam")


def test_pipeline_rejects_unknown_method():
    mesh = make_mesh("annulus")
    with pytest.raises(ValueError):
        run_pipeline(mesh, constraints_from_guide(mesh, _axis_guide(mesh, 0)),
                     method="bogus")


def test_pipeline_geometry_not_mutated():
    mesh = make_mesh("band")
    before = mesh.vertices.copy()
    run_pipeline(mesh, constraints_from_guide(mesh, _axis_guide(mesh, 0)),
                 method="global", subdiv=2)
    assert np.array_equal(mesh.vertices, before)


def test_render_triangles_cover_every_face():
    mesh = make_mesh("cylinder", n_theta=12, n_z=4)
    result = run_pipeline(
        mesh, constraints_from_guide(mesh, _axis_guide(mesh, 2)),
        method="seam", subdiv=3)
    # subdiv=3 lattice yields 9 sub-triangles per face
    assert result.tri_world.shape[0] == mesh.n_faces * 9
    assert result.tri_colors.shape == (result.tri_world.shape[0], 3)
    assert result.tri_colors.min() >= 0.0 and result.tri_colors.max() <= 1.0


def test_dense_swirl_single_singularity_with_zero_residual():
    mesh = make_mesh("disk")
    n, e1, e2 = tangent_frames(mesh)
    c = mesh.face_centers()
    theta = np.arctan2(c[:, 1], c[:, 0])
    g = np.stack([np.cos(2 * theta), np.sin(2 * theta),
                  np.zeros_like(theta)], axis=1)
    result = run_pipeline(mesh, constraints_from_guide(mesh, g, weight=40.0),
                          method="seam", subdiv=2)
    rep = result.field.holonomy
    assert list(result.singularity_vertices()) == [0]
    i0 = int(np.where(rep.vertex == 0)[0][0])
    assert rep.index_times2[i0] == 4
    assert abs(rep.residual_rad[i0]) < 1e-8


def test_sparse_guide_preserves_total_index():
    # Poincare-Hopf: splitting the guide into sparse constraints can spread the
    # index over several vertices but the sum must stay +2.
    mesh = make_mesh("disk")
    c = mesh.face_centers()
    theta = np.arctan2(c[:, 1], c[:, 0])
    g = np.stack([np.cos(2 * theta), np.sin(2 * theta),
                  np.zeros_like(theta)], axis=1)
    cons = constraints_from_guide(
        mesh, g, faces=range(0, mesh.n_faces, 8), weight=20.0)
    result = run_pipeline(mesh, cons, method="seam", subdiv=2)
    rep = result.field.holonomy
    total_index = rep.index_times2[rep.singular].sum() / 2.0
    assert abs(total_index - 2.0) < 1e-9


def test_nearest_face_and_drawn_constraint():
    mesh = make_mesh("annulus")
    centers = mesh.face_centers()
    fi = nearest_face(mesh, centers[5])
    assert fi == 5
    cons = tangent_angle_constraint(mesh, fi, np.array([1.0, 0.0, 0.0]))
    assert cons.face == 5
    result = run_pipeline(mesh, [cons], method="seam", subdiv=2)
    assert result.tri_world.shape[0] > 0


def test_fibre_pattern_runs():
    mesh = make_mesh("band")
    result = run_pipeline(
        mesh, constraints_from_guide(mesh, _axis_guide(mesh, 0)),
        method="seam", pattern="fibre", subdiv=2)
    assert np.isfinite(result.tri_colors).all()


# --------------------------------------------------------------------- #
# Editing-aware CLI surface (headless Agg backend; no Qt involved)
# --------------------------------------------------------------------- #
def _cli_argv(tmp_path, **extra):
    argv = ["--mesh", "disk", "--guide", "swirl2", "--method", "seam",
            "--outdir", str(tmp_path)]
    for key, value in extra.items():
        if value == "":
            argv.append(key)
        elif isinstance(value, (list, tuple)):
            for item in value:
                argv += [key, str(item)]
        else:
            argv += [key, str(value)]
    return argv


def test_cli_loop_and_seamless_checks_run(tmp_path):
    from texorient.app.cli import main
    rc = main(_cli_argv(tmp_path, **{
        "--loop-vertex": 0, "--check-seamless": "",
        "--check-numbering": ""}))
    assert rc == 0
    assert (tmp_path / "disk_seam_swirl2.png").exists()


def test_cli_fixed_print_conflict_returns_distinct_exit(tmp_path):
    from texorient.app.cli import main
    edge = next(e for e in range(make_mesh("disk").edges.shape[0])
                if make_mesh("disk").edge_faces[e, 1] >= 0)
    rc = main(_cli_argv(tmp_path, **{
        "--seam-policy": "fixed_print", "--drag-edge": edge}))
    assert rc == 3
    # allowing the seam to migrate makes the same drag succeed
    rc_ok = main(_cli_argv(tmp_path, **{
        "--seam-policy": "seam_moves", "--drag-edge": edge}))
    assert rc_ok == 0


def test_run_pipeline_accepts_editor_field():
    from texorient.core.editor import FieldEditor
    from texorient.core.topology import SEAM_MOVES
    mesh = make_mesh("disk")
    guide = np.zeros((mesh.n_faces, 3))
    c = mesh.face_centers()
    theta = np.arctan2(c[:, 1], c[:, 0])
    guide[:, 0], guide[:, 1] = np.cos(2 * theta), np.sin(2 * theta)
    cons = constraints_from_guide(mesh, guide, weight=40.0)
    editor = FieldEditor(mesh, cons, mode="line", policy=SEAM_MOVES)
    edge = next(e for e in range(mesh.edges.shape[0])
                if mesh.edge_faces[e, 1] >= 0)
    editor.drag_singularity(edge, 1)
    result = run_pipeline(mesh, cons, method="seam", subdiv=2,
                          dfield=editor.field)
    # the rendered result carries the edited integers
    assert result.field.period_jump[edge] == editor.field.period_jump[edge]
    assert np.isfinite(result.tri_colors).all()
