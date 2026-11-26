"""Tests for the direction field, holonomy, UV construction and metrics."""
import numpy as np
import pytest

from texorient.core.models import (annulus, cylinder, curved_band,
                                   disk_with_interior)
from texorient.core.field import (solve_direction_field, DirectionConstraint)
from texorient.core.frames import local_angles_of_3d_vectors, tangent_frames
from texorient.core.uv import build_seam_uv, build_global_uv
from texorient.core.metrics import evaluate


def _constant_x_constraints(mesh, n, e1, e2, step=15):
    g = np.zeros((mesh.n_faces, 3))
    g[:, 0] = 1.0
    ang = local_angles_of_3d_vectors(mesh, g, e1, e2, n)
    return [DirectionConstraint(f, float(ang[f]))
            for f in range(0, mesh.n_faces, step)]


def test_constant_field_has_no_singularity_on_disk():
    mesh = disk_with_interior(n_theta=32, n_rad=7)
    n, e1, e2 = tangent_frames(mesh)
    df = solve_direction_field(
        mesh, _constant_x_constraints(mesh, n, e1, e2), mode="line")
    assert not np.any(df.holonomy.singular)


def test_swirl_guide_produces_center_singularity():
    mesh = disk_with_interior(n_theta=40, n_rad=8)
    n, e1, e2 = tangent_frames(mesh)
    c = mesh.face_centers()
    theta = np.arctan2(c[:, 1], c[:, 0])
    g = np.stack([np.cos(2 * theta), np.sin(2 * theta),
                  np.zeros_like(theta)], axis=1)
    ang = local_angles_of_3d_vectors(mesh, g, e1, e2, n)
    cons = [DirectionConstraint(f, float(ang[f]), weight=40.0)
            for f in range(mesh.n_faces)]
    df = solve_direction_field(mesh, cons, mode="line")
    rep = df.holonomy
    assert rep.singular[np.where(rep.vertex == 0)[0][0]]
    i0 = np.where(rep.vertex == 0)[0][0]
    # 2*theta winds by 4*pi -> line-field Poincare index +2 (index_times2=4)
    assert rep.index_times2[i0] == 4
    assert abs(rep.total_rotation[i0] - 4 * np.pi) < 1e-8


def test_seam_uv_preserves_direction_exactly():
    mesh = curved_band(n_x=30, n_w=8)
    n, e1, e2 = tangent_frames(mesh)
    df = solve_direction_field(
        mesh, _constant_x_constraints(mesh, n, e1, e2), mode="line")
    uv = build_seam_uv(df)
    m = evaluate(df, uv)
    assert m.deviation_deg.max() < 1e-6
    assert np.allclose(m.anisotropy, 1.0, atol=1e-8)
    assert np.allclose(m.mean_scale, 1.0, atol=1e-8)


def test_seam_uv_is_continuous_across_tree_edges():
    mesh = annulus(n_theta=28, n_rad=6)
    n, e1, e2 = tangent_frames(mesh)
    df = solve_direction_field(
        mesh, _constant_x_constraints(mesh, n, e1, e2), mode="line")
    uv = build_seam_uv(df)
    # zero gap on a smooth field: every cut edge still matches in phase
    m = evaluate(df, uv)
    assert m.seam_gap_fraction[np.isfinite(m.seam_gap_fraction)].max() < 1e-6


def test_global_uv_continuous_and_isometric_on_flat_disk():
    mesh = disk_with_interior(n_theta=28, n_rad=6)
    n, e1, e2 = tangent_frames(mesh)
    df = solve_direction_field(
        mesh, _constant_x_constraints(mesh, n, e1, e2), mode="line")
    uv = build_global_uv(df)
    m = evaluate(df, uv)
    assert m.deviation_deg.max() < 1e-6
    assert np.allclose(m.anisotropy, 1.0, atol=1e-6)
    assert m.collapsed_triangle_count == 0


def test_periodic_cylinder_global_uv_flags_collapse():
    # A single continuous UV cannot wrap a closed periodic direction without
    # a seam; the global method must flag the collapse rather than hide it.
    mesh = cylinder(n_theta=28, n_z=8)
    n, e1, e2 = tangent_frames(mesh)
    g = np.zeros((mesh.n_faces, 3)); g[:, 2] = 1.0
    ang = local_angles_of_3d_vectors(mesh, g, e1, e2, n)
    cons = [DirectionConstraint(f, float(ang[f]))
            for f in range(0, mesh.n_faces, 8)]
    df = solve_direction_field(mesh, cons, mode="line")
    m = evaluate(df, build_global_uv(df))
    assert m.collapsed_triangle_count > 0
    # the seam method handles the same surface without any collapse
    ms = evaluate(df, build_seam_uv(df))
    assert ms.collapsed_triangle_count == 0
    assert ms.deviation_deg.max() < 1e-6


def test_geometry_never_mutated():
    mesh = disk_with_interior(n_theta=20, n_rad=5)
    before = mesh.vertices.copy()
    n, e1, e2 = tangent_frames(mesh)
    df = solve_direction_field(
        mesh, _constant_x_constraints(mesh, n, e1, e2), mode="line")
    build_seam_uv(df)
    build_global_uv(df)
    assert np.array_equal(mesh.vertices, before)


def test_changing_guide_changes_texture_direction():
    # The texture must respond to the guide, not reuse an existing UV.
    mesh = curved_band(n_x=24, n_w=6)
    n, e1, e2 = tangent_frames(mesh)
    angles_out = []
    for axis in (0, 1):
        g = np.zeros((mesh.n_faces, 3)); g[:, axis] = 1.0
        ang = local_angles_of_3d_vectors(mesh, g, e1, e2, n)
        cons = [DirectionConstraint(f, float(ang[f]))
                for f in range(0, mesh.n_faces, 6)]
        df = solve_direction_field(mesh, cons, mode="line")
        uv = build_seam_uv(df)
        m = evaluate(df, uv)
        assert m.deviation_deg.max() < 1e-6
        angles_out.append(df.angles.copy())
    # the two orthogonal guides produce fields differing by ~pi/2
    d = np.angle(np.exp(1j * (angles_out[0] - angles_out[1])))
    assert np.all(np.abs(np.abs(d) - np.pi / 2) < 0.2)


def test_requires_at_least_one_constraint():
    mesh = annulus(n_theta=12, n_rad=3)
    with pytest.raises(ValueError):
        solve_direction_field(mesh, [], mode="line")


def test_vector_field_uses_two_pi_period():
    mesh = disk_with_interior(n_theta=24, n_rad=5)
    n, e1, e2 = tangent_frames(mesh)
    df = solve_direction_field(
        mesh, _constant_x_constraints(mesh, n, e1, e2), mode="vector")
    assert df.period == 2 * np.pi
    assert not np.any(df.holonomy.singular)
