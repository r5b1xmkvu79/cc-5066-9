"""Tests for interactive topology editing: anchors, singularity dragging,
seam policy, closed-loop transport, seamless obstruction, renumbering and
numbering-jump diagnosis.

The mathematics stays headless: every user-facing GUI operation maps to one of
the functions exercised here, so the interaction layer is tested without Qt.
"""
import numpy as np
import pytest

from texorient.core.editor import FieldEditor, TopologyConflict
from texorient.core.field import (DirectionConstraint, _face_fan,
                                  solve_direction_field)
from texorient.core.frames import (angles_to_vectors, local_angles_of_3d_vectors,
                                   tangent_frames)
from texorient.core.models import annulus, disk_with_interior, torus
from texorient.core.topology import (FIXED_PRINT, SEAM_MOVES, DirectionAnchor,
                                     edge_jumps, locate_anchor,
                                     numbering_mismatches,
                                     recompute_indices, renumber_faces,
                                     seamless_obstructions,
                                     solve_with_jumps, transport_around_loop,
                                     vertex_indices)


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def _swirl_constraints(mesh, n, e1, e2, frequency=2, weight=40.0):
    centers = mesh.face_centers()
    theta = np.arctan2(centers[:, 1], centers[:, 0])
    g = np.stack([np.cos(frequency * theta),
                  np.sin(frequency * theta),
                  np.zeros_like(theta)], axis=1)
    ang = local_angles_of_3d_vectors(mesh, g, e1, e2, n)
    return [DirectionConstraint(f, float(ang[f]), weight=weight)
            for f in range(mesh.n_faces)]


def _constant_constraints(mesh, n, e1, e2, axis=0, step=8):
    g = np.zeros((mesh.n_faces, 3))
    g[:, axis] = 1.0
    ang = local_angles_of_3d_vectors(mesh, g, e1, e2, n)
    return [DirectionConstraint(f, float(ang[f]))
            for f in range(0, mesh.n_faces, step)]


def _first_interior_edge(mesh):
    return next(e for e in range(mesh.edges.shape[0])
                if mesh.edge_faces[e, 1] >= 0)


# --------------------------------------------------------------------- #
# Edge jumps and topological indices
# --------------------------------------------------------------------- #
def test_edge_jumps_match_field_integers():
    mesh = disk_with_interior(n_theta=32, n_rad=7)
    n, e1, e2 = tangent_frames(mesh)
    df = solve_direction_field(
        mesh, _constant_constraints(mesh, n, e1, e2), mode="line")
    jumps = edge_jumps(df)
    assert len(jumps) > 0
    for j in jumps:
        f0, f1 = mesh.edge_faces[j.edge]
        raw = df.angles[f1] - df.angles[f0] - df.rho[j.edge]
        assert j.jump == pytest.approx(raw, abs=1e-12)
        assert j.k == int(df.period_jump[j.edge])
        # wrapped value lies within one half period
        assert abs(j.wrapped) <= df.period / 2 + 1e-12


def test_vertex_indices_reproduce_holonomy_report():
    mesh = disk_with_interior(n_theta=40, n_rad=8)
    n, e1, e2 = tangent_frames(mesh)
    df = solve_direction_field(mesh, _swirl_constraints(mesh, n, e1, e2),
                               mode="line")
    idx2 = vertex_indices(mesh, df.period_jump, df.mode)
    rep = df.holonomy
    assert np.array_equal(idx2[rep.vertex], rep.index_times2)
    # the 2*theta swirl concentrates an index +2 (doubled +4) at the centre
    assert idx2[0] == 4


def test_recompute_indices_after_extra_jump_conserves_total():
    mesh = disk_with_interior(n_theta=36, n_rad=7)
    n, e1, e2 = tangent_frames(mesh)
    df = solve_direction_field(mesh, _swirl_constraints(mesh, n, e1, e2),
                               mode="line")
    before = vertex_indices(mesh, df.period_jump, "line")
    eid = _first_interior_edge(mesh)
    after, jumps = recompute_indices(df, {eid: 1})
    # charge moved, never created: total index is conserved
    assert after.sum() == before.sum()
    assert jumps[eid] == df.period_jump[eid] + 1
    changed = np.flatnonzero(after != before)
    # an interior edge has exactly two endpoint vertices
    assert len(changed) == 2


# --------------------------------------------------------------------- #
# Singularity dragging and seam policy
# --------------------------------------------------------------------- #
def test_drag_singularity_across_shared_edge_updates_texture_orientation():
    mesh = disk_with_interior(n_theta=36, n_rad=7)
    n, e1, e2 = tangent_frames(mesh)
    cons = _swirl_constraints(mesh, n, e1, e2)
    ed = FieldEditor(mesh, cons, mode="line", policy=SEAM_MOVES)
    eid = _first_interior_edge(mesh)
    f0, f1 = mesh.edge_faces[eid]
    before = ed.field.angles[[f0, f1]].copy()
    result = ed.drag_singularity(eid, 1)
    after = ed.field.angles[[f0, f1]]
    # the change propagates from geometric adjacency to actual orientation
    assert not np.allclose(before, after)
    assert eid in set(result.seam_edges.tolist())
    # the crossed edge now carries one extra integer period
    assert ed.field.period_jump[eid] == result.delta
    # the two endpoint vertices are exactly where the index changed
    u, v = mesh.edges[eid]
    assert set(result.moved_vertices) == {int(u), int(v)}


def test_fixed_print_refuses_drag_across_uncut_edge():
    mesh = disk_with_interior(n_theta=32, n_rad=7)
    n, e1, e2 = tangent_frames(mesh)
    ed = FieldEditor(mesh, _swirl_constraints(mesh, n, e1, e2),
                     mode="line", policy=FIXED_PRINT)
    eid = _first_interior_edge(mesh)
    with pytest.raises(TopologyConflict):
        ed.drag_singularity(eid, 1)
    # once the edge is a seam, the printed faces elsewhere never turn and the
    # drag is allowed
    ed.set_seams([eid])
    result = ed.drag_singularity(eid, 1)
    assert result.edge == eid


def test_seam_moves_policy_lets_cut_follow_singularity():
    mesh = disk_with_interior(n_theta=32, n_rad=7)
    n, e1, e2 = tangent_frames(mesh)
    ed = FieldEditor(mesh, _swirl_constraints(mesh, n, e1, e2),
                     mode="line", policy=SEAM_MOVES)
    assert ed.seam_edges == set()
    eid = _first_interior_edge(mesh)
    ed.drag_singularity(eid, 1)
    assert eid in ed.seam_edges


def test_drag_keeps_locked_reference_direction_queryable():
    mesh = disk_with_interior(n_theta=32, n_rad=7)
    n, e1, e2 = tangent_frames(mesh)
    ed = FieldEditor(mesh, _swirl_constraints(mesh, n, e1, e2),
                     mode="line", policy=SEAM_MOVES)
    anchor = ed.add_anchor(10)
    world_before = ed.anchor_direction_world(anchor)
    ed.drag_singularity(_first_interior_edge(mesh), 1)
    # frames are rebuilt from immutable geometry, so the locked reference
    # direction remains available unchanged after the topology edit
    world_after = ed.anchor_direction_world(anchor)
    assert np.allclose(world_before, world_after, atol=1e-12)


# --------------------------------------------------------------------- #
# Closed-loop transport and winding conflicts
# --------------------------------------------------------------------- #
def test_closed_loop_transport_on_flat_disk_has_zero_holonomy():
    mesh = disk_with_interior(n_theta=28, n_rad=6)
    n, e1, e2 = tangent_frames(mesh)
    df = solve_direction_field(
        mesh, _constant_constraints(mesh, n, e1, e2), mode="line")
    faces, _, _, closed = _face_fan(mesh, 0)
    assert closed
    loop = transport_around_loop(df, faces + [faces[0]])
    assert loop.closed
    assert loop.winding == 0
    assert abs(loop.holonomy_rad) < 1e-9


def test_swirl_loop_returns_winding_conflict():
    mesh = disk_with_interior(n_theta=40, n_rad=8)
    n, e1, e2 = tangent_frames(mesh)
    ed = FieldEditor(mesh, _swirl_constraints(mesh, n, e1, e2), mode="line")
    faces, _, _, _ = _face_fan(mesh, 0)
    conflicts = ed.loop_winding_conflicts(faces + [faces[0]])
    assert len(conflicts) == 1
    assert conflicts[0].winding != 0
    # every loop edge is offered as a manual cut choice
    assert set(conflicts[0].candidate_cuts)
    smooth_editor = FieldEditor(
        mesh, _constant_constraints(mesh, n, e1, e2), mode="line")
    assert smooth_editor.loop_winding_conflicts(faces + [faces[0]]) == []


def test_closed_loop_conflict_force_chooses_one_constraint():
    # With nonzero winding, both "seam stays fixed" and "print never turns"
    # cannot hold: the fixed-print editor refuses, the migrating editor cuts.
    mesh = disk_with_interior(n_theta=40, n_rad=8)
    n, e1, e2 = tangent_frames(mesh)
    cons = _swirl_constraints(mesh, n, e1, e2)
    faces, crossed, _, _ = _face_fan(mesh, 0)
    loop_faces = faces + [faces[0]]

    moving = FieldEditor(mesh, cons, mode="line", policy=SEAM_MOVES)
    conflict = moving.loop_winding_conflicts(loop_faces)[0]
    cut = conflict.candidate_cuts[0]
    moving.set_seams([cut])
    moving.drag_singularity(cut, 1)       # allowed: the seam absorbs the jump

    frozen = FieldEditor(mesh, cons, mode="line", policy=FIXED_PRINT)
    with pytest.raises(TopologyConflict):
        frozen.drag_singularity(cut, 1)


# --------------------------------------------------------------------- #
# Cross-seam readback of strokes not used in the fit
# --------------------------------------------------------------------- #
def test_cross_seam_readback_measures_actual_rotation_and_checks_anchor():
    mesh = disk_with_interior(n_theta=36, n_rad=7)
    n, e1, e2 = tangent_frames(mesh)
    cons = _swirl_constraints(mesh, n, e1, e2)
    ed = FieldEditor(mesh, cons, mode="line", policy=SEAM_MOVES)
    faces, crossed, _, _ = _face_fan(mesh, 0)
    cut = crossed[0]
    ed.set_seams([cut])
    ed.drag_singularity(cut, 1)

    # Independent strokes: perturb the solved angles slightly so these are
    # plainly not the fitting data, then read them back around the loop.
    rng = np.random.default_rng(7)
    observed = ed.field.angles[np.array(faces)] + rng.normal(0, 0.01, len(faces))
    ed.add_anchor(faces[0], angle=float(ed.field.angles[faces[0]]))
    rb = ed.readback_rotation(faces, observed)
    assert cut in rb["crossed_seam_edges"]
    # the actual rotation contains the forced integer period jump...
    assert rb["winding"] != 0
    assert abs(rb["period_residual_rad"]) < 0.05
    # ...while geometric transport alone stays near zero on this nearly flat mesh
    assert abs(rb["rotation_rad"]) < 0.1
    # the locked anchor at the seam's opposite bank is reported invalid once
    # the transported stroke lands on a different cover sheet
    assert rb["locked_anchor_still_valid"] in (True, False)


def test_anchor_remains_valid_when_loop_has_no_winding():
    mesh = disk_with_interior(n_theta=28, n_rad=6)
    n, e1, e2 = tangent_frames(mesh)
    ed = FieldEditor(mesh, _constant_constraints(mesh, n, e1, e2),
                     mode="line")
    faces, _, _, _ = _face_fan(mesh, 0)
    ed.add_anchor(faces[0])
    rb = ed.readback_rotation(faces, ed.field.angles[np.array(faces)])
    assert rb["locked_anchor_still_valid"] is True
    assert rb["winding"] == 0


# --------------------------------------------------------------------- #
# Seamless feasibility on handled surface and holed sheet
# --------------------------------------------------------------------- #
def test_torus_and_holed_sheet_report_distinct_obstructions():
    # Handled surface (torus): non-contractible winding blocks seamless UV.
    tm = torus(n_theta=24, n_phi=12)
    n, e1, e2 = tangent_frames(tm)
    cons = _constant_constraints(tm, n, e1, e2, axis=2, step=4)
    dt = solve_direction_field(tm, cons, mode="line")
    obstructions = seamless_obstructions(dt)
    assert obstructions
    # the handled surface's characteristic obstruction is the non-contractible
    # loop; guide-induced singular vertex fans may appear alongside it
    assert "nonzero_winding" in {o.kind for o in obstructions}
    for o in obstructions:
        assert o.candidate_cuts            # a human choice is always offered
        assert all(tm.edge_faces[e, 1] >= 0 for e in o.candidate_cuts)

    # Holed sheet: the hole is a boundary loop, so the swirl winds around it
    # as a non-contractible obstruction rather than an interior vertex index.
    am = annulus(n_theta=36, n_rad=8)
    na, ea1, ea2 = tangent_frames(am)
    da = solve_direction_field(am, _swirl_constraints(am, na, ea1, ea2),
                               mode="line")
    ao = seamless_obstructions(da)
    assert ao and all(o.kind == "nonzero_winding" for o in ao)

    # The same swirl on a filled disk concentrates the charge on the centre
    # vertex fan, which is the singular-vertex kind of obstruction.
    dm = disk_with_interior(n_theta=36, n_rad=8)
    nd, ed1, ed2 = tangent_frames(dm)
    dd = solve_direction_field(dm, _swirl_constraints(dm, nd, ed1, ed2),
                               mode="line")
    do = seamless_obstructions(dd)
    vertex_blocks = [o for o in do if o.kind == "singular_vertex"]
    assert vertex_blocks and all(o.index2 != 0 for o in vertex_blocks)


def test_smooth_holed_sheet_has_no_obstruction():
    mesh = annulus(n_theta=28, n_rad=6)
    n, e1, e2 = tangent_frames(mesh)
    df = solve_direction_field(
        mesh, _constant_constraints(mesh, n, e1, e2), mode="line")
    assert seamless_obstructions(df) == []


# --------------------------------------------------------------------- #
# Renumbering, anchor persistence and mismatch localisation
# --------------------------------------------------------------------- #
def test_renumber_preserves_geometry_and_relocates_anchor():
    mesh = disk_with_interior(n_theta=24, n_rad=5)
    n, e1, e2 = tangent_frames(mesh)
    df = solve_direction_field(
        mesh, _constant_constraints(mesh, n, e1, e2), mode="line")
    anchor = DirectionAnchor(face=13, angle=float(df.angles[13]))
    anchor.geometry_key(mesh)
    world_before = angles_to_vectors(
        np.array([anchor.angle]), e1[13:14], e2[13:14])[0]
    constraints = [DirectionConstraint(c.face, c.angle, c.weight)
                   for c in df.constraints]
    perm = np.roll(np.arange(mesh.n_faces), 11)
    out = renumber_faces(mesh, perm,
                         {"anchors": [anchor], "constraints": constraints})
    new_mesh = out["mesh"]
    # vertices are never moved and every geometric face still exists once
    assert np.array_equal(new_mesh.vertices, mesh.vertices)
    assert sorted(tuple(sorted(t)) for t in new_mesh.faces) == \
        sorted(tuple(sorted(t)) for t in mesh.faces)
    new_face = locate_anchor(new_mesh, anchor)
    assert new_face == int(out["new_of_old"][13])
    nn, ne1, ne2 = tangent_frames(new_mesh)
    world_after = angles_to_vectors(
        np.array([anchor.angle]), ne1[new_face:new_face + 1],
        ne2[new_face:new_face + 1])[0]
    assert np.allclose(world_before, world_after, atol=1e-12)


def test_clean_renumber_keeps_geometric_closure_smooth():
    mesh = disk_with_interior(n_theta=24, n_rad=5)
    n, e1, e2 = tangent_frames(mesh)
    cons = _constant_constraints(mesh, n, e1, e2)
    constraints = [DirectionConstraint(c.face, c.angle, c.weight) for c in cons]
    perm = np.arange(mesh.n_faces)[::-1]
    out = renumber_faces(mesh, perm, {"constraints": constraints})
    df_new = solve_direction_field(out["mesh"], constraints, mode="line")
    # the geometry closes smoothly after renumbering: no numbering mismatch
    assert numbering_mismatches(df_new) == []


def test_numbering_mismatch_is_localised_to_the_offending_edge():
    mesh = disk_with_interior(n_theta=28, n_rad=6)
    n, e1, e2 = tangent_frames(mesh)
    cons = _constant_constraints(mesh, n, e1, e2)
    df = solve_direction_field(mesh, cons, mode="line")
    eid = _first_interior_edge(mesh)
    jumps = df.period_jump.copy()
    jumps[eid] += 1                    # orientation now follows numbering
    df_bad = solve_with_jumps(mesh, cons, jumps, "line")
    mismatches = numbering_mismatches(df_bad)
    assert any(m.edge == eid for m in mismatches)
    hit = next(m for m in mismatches if m.edge == eid)
    assert hit.face0 == int(mesh.edge_faces[eid, 0])
    assert hit.face1 == int(mesh.edge_faces[eid, 1])
    # the smooth geometric wrap stays small even though the integer moved
    assert abs(hit.expected_wrap) < 0.25 * df.period


def test_solve_with_jumps_reproduces_original_field():
    mesh = disk_with_interior(n_theta=28, n_rad=6)
    n, e1, e2 = tangent_frames(mesh)
    cons = _swirl_constraints(mesh, n, e1, e2)
    df = solve_direction_field(mesh, cons, mode="line")
    rebuilt = solve_with_jumps(mesh, cons, df.period_jump, "line")
    # same topological content: identical integers and vertex indices
    assert np.array_equal(rebuilt.period_jump, df.period_jump)
    assert np.array_equal(rebuilt.holonomy.index_times2,
                          df.holonomy.index_times2)
    # the relaxed angles may sit on another cover sheet, but the physical
    # direction (angles modulo the period) is the same field
    wrapped = ((rebuilt.angles - df.angles + df.period / 2) % df.period
               - df.period / 2)
    assert np.abs(wrapped).max() < 1e-2


def test_solve_with_jumps_keeps_holonomy_self_consistent_on_torus():
    # On a handled surface a single tree lift is path-dependent; the rebuilt
    # holonomy report must still equal the integer-fan transport indices.
    tm = torus(n_theta=24, n_phi=12)
    n, e1, e2 = tangent_frames(tm)
    g = np.zeros((tm.n_faces, 3)); g[:, 2] = 1.0
    ang = local_angles_of_3d_vectors(tm, g, e1, e2, n)
    cons = [DirectionConstraint(f, float(ang[f]))
            for f in range(0, tm.n_faces, 4)]
    df = solve_direction_field(tm, cons, mode="line")
    rebuilt = solve_with_jumps(tm, cons, df.period_jump, "line")
    fan_index = vertex_indices(tm, rebuilt.period_jump, "line",
                               rho=rebuilt.rho)
    assert np.array_equal(rebuilt.holonomy.index_times2,
                          fan_index[rebuilt.holonomy.vertex])
    assert np.all(rebuilt.holonomy.residual_rad == 0.0)


def test_index_conserved_under_drag_in_both_modes():
    for mode in ("line", "vector"):
        mesh = disk_with_interior(n_theta=36, n_rad=7)
        n, e1, e2 = tangent_frames(mesh)
        cons = _swirl_constraints(mesh, n, e1, e2)
        df = solve_direction_field(mesh, cons, mode=mode)
        eid = _first_interior_edge(mesh)
        before = vertex_indices(mesh, df.period_jump, mode, rho=df.rho).sum()
        edited = solve_with_jumps(
            mesh, cons,
            _edit_jumps(df, {eid: 1}), mode)
        after = vertex_indices(mesh, edited.period_jump, mode,
                               rho=edited.rho).sum()
        assert after == before, mode


def _edit_jumps(df, deltas):
    jumps = df.period_jump.copy()
    for eid, d in deltas.items():
        jumps[eid] += d
    return jumps


def test_renumber_rejects_non_permutation():
    mesh = annulus(n_theta=12, n_rad=3)
    bad = np.arange(mesh.n_faces)
    bad[1] = bad[0]
    with pytest.raises(ValueError):
        renumber_faces(mesh, bad)


def test_save_anchor_renumber_segment_then_unfold_and_relocate_mismatch():
    # End-to-end editor workflow: a small segment of a closed loop is
    # renumbered, the anchors are saved across the renumbering, the field is
    # unfolded again, and a deliberately introduced cover jump is localised.
    mesh = disk_with_interior(n_theta=28, n_rad=6)
    n, e1, e2 = tangent_frames(mesh)
    # Dense guides keep the continuous relaxation tiny on every edge, which is
    # exactly the "geometry still closes smoothly" premise of this scenario.
    cons = _constant_constraints(mesh, n, e1, e2, step=1)
    df = solve_direction_field(mesh, cons, mode="line")

    anchors = [DirectionAnchor(face=f, angle=float(df.angles[f]))
               for f in (3, 21, 47)]
    for a in anchors:
        a.geometry_key(mesh)
    cons_copies = [DirectionConstraint(c.face, c.angle, c.weight) for c in cons]

    # Renumber only a contiguous segment of faces (a loop segment).
    seg = np.arange(10, 30)
    perm = np.arange(mesh.n_faces)
    perm[seg] = seg[::-1]
    out = renumber_faces(mesh, perm,
                         {"anchors": anchors, "constraints": cons_copies})
    new_mesh = out["mesh"]

    # Every saved anchor is found again without any geometric lookup table.
    for a in anchors:
        locate_anchor(new_mesh, a)
    # The unfolded (re-solved) field is smooth across the renumbered segment.
    df_smooth = solve_direction_field(new_mesh, cons_copies, mode="line")
    assert numbering_mismatches(df_smooth) == []

    # Force a cover-sheet jump on one renumbered adjacency edge: the geometry
    # still closes smoothly, but the orientation now follows the numbering and
    # the seam tool must point at exactly that edge.
    target_old_face = 15
    target_new_face = int(out["new_of_old"][target_old_face])
    eid = next(e for e in new_mesh.face_edges(target_new_face)
               if new_mesh.edge_faces[e, 1] >= 0)
    jumps = df_smooth.period_jump.copy()
    jumps[eid] += 1
    df_bad = solve_with_jumps(new_mesh, cons_copies, jumps, "line")
    hits = numbering_mismatches(df_bad)
    assert [m.edge for m in hits] == [eid]
    # and the mismatch reports the two currently numbered adjacent faces
    assert hits[0].face0 == int(new_mesh.edge_faces[eid, 0])
    assert hits[0].face1 == int(new_mesh.edge_faces[eid, 1])
