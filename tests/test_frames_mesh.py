"""Unit tests for mesh connectivity and tangent-frame transport."""
import numpy as np

from texorient.core.models import (annulus, cylinder, curved_band,
                                   disk_with_interior, cone)
from texorient.core.frames import (face_normals, tangent_frames,
                                   transport_angles, angles_to_vectors,
                                   local_angles_of_3d_vectors)
from texorient.core.field import _face_fan


def _all_models():
    return {
        "annulus": annulus(n_theta=20, n_rad=5),
        "cylinder": cylinder(n_theta=18, n_z=6),
        "band": curved_band(n_x=16, n_w=5),
        "disk": disk_with_interior(n_theta=18, n_rad=5),
        "cone": cone(n_theta=18, n_slant=5),
    }


def test_halfedge_indexing_matches_face_slots():
    mesh = annulus(n_theta=12, n_rad=4)
    for fi, tri in enumerate(mesh.faces):
        for slot in range(3):
            eid = mesh.halfedge_edge(fi, slot)
            u, v = mesh.edges[eid]
            # edges are stored as an unordered sorted pair; the half-edge
            # joins the same two vertices as the face's directed slot edge
            assert {u, v} == {tri[slot], tri[(slot + 1) % 3]}


def test_interior_edges_have_two_oppositely_wound_faces():
    for name, mesh in _all_models().items():
        for eid in range(mesh.edges.shape[0]):
            f0, f1 = mesh.edge_faces[eid]
            if f1 < 0:
                continue
            u, v = mesh.edges[eid]

            def directed(fi, a, b):
                tri = mesh.faces[fi]
                return any(tri[s] == a and tri[(s + 1) % 3] == b
                           for s in range(3))
            # one face traverses u->v, the other v->u
            assert directed(f0, u, v) != directed(f1, u, v), name


def test_face_normals_outward_and_unit():
    mesh = cylinder(n_theta=20, n_z=5)
    nrm = face_normals(mesh)
    assert np.allclose(np.linalg.norm(nrm, axis=1), 1.0)
    centers = mesh.face_centers()
    radial = np.column_stack([centers[:, 0], centers[:, 1], np.zeros(len(centers))])
    assert np.all(np.sum(nrm * radial, axis=1) > 0.9)


def test_frames_orthonormal():
    for mesh in _all_models().values():
        n, e1, e2 = tangent_frames(mesh)
        assert np.allclose(np.einsum("ij,ij->i", e1, e2), 0.0, atol=1e-10)
        assert np.allclose(np.einsum("ij,ij->i", n, e1), 0.0, atol=1e-10)
        assert np.allclose(np.linalg.norm(e1, axis=1), 1.0)


def test_transport_zero_on_flat_planar_mesh():
    # All faces coplanar with parallel frames -> transport angle is 0.
    mesh = disk_with_interior(n_theta=20, n_rad=5)
    n, e1, e2 = tangent_frames(mesh)
    rho = transport_angles(mesh, n, e1, e2)
    interior = [e for e in range(mesh.edges.shape[0])
                if mesh.edge_faces[e, 1] >= 0]
    assert np.allclose(rho[interior], 0.0, atol=1e-10)


def test_face_fan_partitions_incident_faces():
    for name, mesh in _all_models().items():
        for v in range(mesh.n_vertices):
            faces, crossed, legs, closed = _face_fan(mesh, v)
            incident = set(mesh.vertex_faces[v])
            assert len(faces) == len(set(faces))        # no repeated face
            assert set(faces).issubset(incident)
            if closed:
                assert set(faces) == incident          # full loop covers all
                assert len(crossed) == len(faces)


def test_angle_vector_roundtrip():
    mesh = annulus(n_theta=16, n_rad=4)
    n, e1, e2 = tangent_frames(mesh)
    rng = np.random.default_rng(0)
    ang = rng.uniform(-np.pi, np.pi, mesh.n_faces)
    vec = angles_to_vectors(ang, e1, e2)
    back = local_angles_of_3d_vectors(mesh, vec, e1, e2, n)
    # angles recovered modulo 2*pi
    assert np.allclose(np.exp(1j * ang), np.exp(1j * back), atol=1e-10)
