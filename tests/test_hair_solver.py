"""Pure-Python (no bpy) tests for Tessera Projection Hair:
    python3 tests/test_hair_solver.py
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'arc_flow'))

from core import hair_solver as H
from core import primitives as MS
from core import sdf as SDF


def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        ok.fail += 1
ok.fail = 0


def _sphere_field(r=0.12, c=(0, 0, 1.6), res=28, level=1):
    v, t = MS.uvsphere(n=24, m=12, r=r, center=c)
    return H.make_collider(v, t, resolution=res, level=level, mode='SOLID'), v, t


def test_pack_and_groom():
    P, off = H.make_hemisphere_groom(n_strands=20, n_pts=8, radius=0.12, length=0.25, center=(0, 0, 1.6), seed=3)
    ok(len(off) == 21 and len(P) == 20 * 8, "packed hemisphere groom 20 x 8")
    strands = H.unpack_strands(P, off)
    ok(len(strands) == 20 and all(len(s) == 8 for s in strands), "unpack restores 20 strands")
    P2, off2 = H.pack_strands(strands)
    ok(np.allclose(P, P2) and np.array_equal(off, off2), "pack/unpack round-trip")
    Pr, offr = H.apply_segment_budget(P, off, max_segments=5)
    ok(int(np.diff(offr).max()) == 5 and len(offr) == 21, "segment budget resamples to 5 points/strand")


def test_root_pinning():
    P, off = H.make_hemisphere_groom(16, 8, 0.12, 0.25, center=(0, 0, 1.6), seed=1)
    S = H.HairSolver(P, off, H.HairParams(dt=1 / 24, substeps=4, gravity=(0, 0, -9.81)))
    S.bind_roots()
    rest_roots = S.positions[S.offsets[:-1]].copy()
    for _ in range(8):
        S.step()
    err = np.linalg.norm(S.positions[S.offsets[:-1]] - rest_roots, axis=1).max()
    ok(err < 1e-12, "roots stay pinned at rest under gravity (err %.2e)" % err)
    M = S.root_M.copy()
    M[:, 0, 3] += 0.25
    S.set_root_transforms(M)
    S.step()
    tgt = H.xform_per_strand(S.rest_local, S.root_M, S.strand_id)
    err = np.linalg.norm(S.positions[S.offsets[:-1]] - tgt[S.offsets[:-1]], axis=1).max()
    ok(err < 1e-12, "roots follow a translated attachment exactly (err %.2e)" % err)
    ok(S.length_error() < 0.05, "inextensibility holds after the jump (max rel err %.4f)" % S.length_error())


def test_gravity_and_inertia():
    # a few straight hanging strands, no style memory, gravity down
    roots = np.array([[0.0, 0.0, 1.0], [0.05, 0.0, 1.0]])
    strands = [H.make_strand(r, (0, 0, -1), 0.4, 10, sag=1.0, curl=0.0) for r in roots]
    # start almost horizontal so gravity has something to do
    strands = [H.make_strand(r, (1, 0, 0), 0.4, 10, sag=0.0, curl=0.0) for r in roots]
    P, off = H.pack_strands(strands)
    S = H.HairSolver(P, off, H.HairParams(dt=1 / 24, substeps=6, gravity=(0, 0, -9.81),
                                          rest_shape=0.0, damping=0.98, wind_drag=0.0))
    S.bind_roots()
    z_tip0 = float(S.positions[off[1:] - 1, 2].mean())
    for _ in range(24):
        S.step()
    z_tip1 = float(S.positions[off[1:] - 1, 2].mean())
    ok(z_tip1 < z_tip0 - 0.05, "tips fall under gravity (%.3f -> %.3f)" % (z_tip0, z_tip1))

    # inertia: snap the head sideways; tips must lag the roots on the first frame
    S.reset()
    x_root0 = S.positions[S.offsets[:-1], 0].copy()
    x_tip0 = S.positions[S.offsets[1:] - 1, 0].copy()
    M = S.root_M.copy()
    M[:, 0, 3] += 0.30
    S.set_root_transforms(M)
    S.step()
    dx_root = float((S.positions[S.offsets[:-1], 0] - x_root0).mean())
    dx_tip = float((S.positions[S.offsets[1:] - 1, 0] - x_tip0).mean())
    ok(dx_root > 0.29 and dx_tip < dx_root - 0.08,
       "tips lag a 0.30 m root jump (root dx %.3f, tip dx %.3f)" % (dx_root, dx_tip))


def test_angular_impulse():
    P, off = H.make_hemisphere_groom(24, 10, 0.12, 0.30, center=(0, 0, 1.6), seed=4)
    S = H.HairSolver(P, off, H.HairParams(dt=1 / 24, substeps=4, gravity=(0, 0, 0),
                                          rest_shape=0.15, damping=0.96, wind_drag=0.0))
    # bind at a common head frame centred on the scalp
    M0 = np.eye(4)
    M0[:3, 3] = (0, 0, 1.6)
    S.bind_roots(np.repeat(M0[None], S.n_strands, axis=0))
    rest_tips = S.positions[S.offsets[1:] - 1].copy()
    # 60-degree yaw in one tick
    ang = np.deg2rad(60.0)
    c, s_ = np.cos(ang), np.sin(ang)
    M1 = M0.copy()
    M1[:3, :3] = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
    S.set_root_transforms(np.repeat(M1[None], S.n_strands, axis=0))
    S.step()
    tips = S.positions[S.offsets[1:] - 1]
    moved = np.linalg.norm(tips - rest_tips, axis=1).mean()
    ok(moved > 0.02, "rotating the head swings the tips (mean move %.3f m)" % moved)
    # after many steps with style memory the groom should settle near the rotated rest pose
    for _ in range(36):
        S.step()
    target = H.xform_per_strand(S.rest_local, S.root_M, S.strand_id)
    tip_err = np.linalg.norm(S.positions[S.offsets[1:] - 1] - target[S.offsets[1:] - 1], axis=1).mean()
    rest_span = np.linalg.norm(S.rest[S.offsets[1:] - 1] - S.rest[S.offsets[:-1]], axis=1).mean()
    ok(tip_err < 0.45 * rest_span, "curl memory settles tips toward the rotated rest (err %.3f, span %.3f)" % (tip_err, rest_span))


def test_sdf_collision():
    c, r = (0.0, 0.0, 1.6), 0.12
    field, _v, _t = _sphere_field(r, c, res=32, level=2)
    P, off = H.make_hemisphere_groom(32, 10, r, 0.32, center=c, seed=5, sag=0.7)
    S = H.HairSolver(P, off, H.HairParams(dt=1 / 24, substeps=4, gravity=(0, 0, -9.81),
                                          rest_shape=0.1, collision_radius=0.01,
                                          collision_iterations=2, sdf_refine=True, wind_drag=0.0))
    S.bind_roots()
    S.add_collider(field, matrix=np.eye(4))
    for _ in range(16):
        S.step()
    free = S.inv_mass > 0
    sd = field.sample_sd(S.positions)
    margin = 0.01 - 3e-3
    n_in = int((sd[free] < margin).sum())
    ok(n_in == 0 and float(sd[free].min()) >= margin,
       "zero free-point penetration into the head (min sd %.4f, radius 0.01, %d inside)" % (float(sd[free].min()), n_in))
    # moving the collider with the head: keep the same relative clearance
    M = np.eye(4)
    M[0, 3] = 0.2
    # translate both the attachment and the collider together
    Mb = S.bind_M.copy()
    Mb[:, 0, 3] += 0.2
    S.set_root_transforms(Mb)
    S.set_collider_matrix(0, M)
    for _ in range(8):
        S.step()
    Xl = H.inv_xform_points(S.positions, M)
    sd2 = field.sample_sd(Xl)
    n_in2 = int((sd2[free] < margin).sum())
    ok(n_in2 == 0, "still no penetration after the head+collider translate together (%d inside, min sd %.4f)" % (n_in2, float(sd2[free].min())))


def test_mesh_bind_inertia():
    """A deforming scalp (translated / rotated triangle soup) pulls roots and leaves tips lagging."""
    v, t = MS.uvsphere(n=20, m=10, r=0.12, center=(0, 0, 1.6))
    P, off = H.make_hemisphere_groom(16, 8, 0.12, 0.25, center=(0, 0, 1.6), seed=6)
    S = H.HairSolver(P, off, H.HairParams(dt=1 / 24, substeps=4, gravity=(0, 0, 0),
                                          rest_shape=0.05, damping=0.95, wind_drag=0.0))
    S.bind_to_mesh(v, t)
    roots0 = S.positions[S.offsets[:-1]].copy()
    tips0 = S.positions[S.offsets[1:] - 1].copy()
    v2 = v + np.array((0.18, 0.0, 0.0))
    S.update_from_mesh(v2, t)
    S.step()
    roots = S.positions[S.offsets[:-1]]
    tips = S.positions[S.offsets[1:] - 1]
    ok(abs(float((roots - roots0)[:, 0].mean()) - 0.18) < 0.02,
       "scalp bind moves roots with the mesh (dx %.3f)" % float((roots - roots0)[:, 0].mean()))
    ok(float((tips - tips0)[:, 0].mean()) < float((roots - roots0)[:, 0].mean()) - 0.02,
       "tips lag the deforming scalp (tips dx %.3f, roots dx %.3f)" % (
           float((tips - tips0)[:, 0].mean()), float((roots - roots0)[:, 0].mean())))


def test_wind_coupling():
    strands = [H.make_strand((0.05 * i, 0.0, 1.2), (0, 0, -1), 0.35, 10, sag=1.0, curl=0.0) for i in range(6)]
    P, off = H.pack_strands(strands)
    S = H.HairSolver(P, off, H.HairParams(dt=1 / 24, substeps=4, gravity=(0, 0, 0),
                                          rest_shape=0.0, damping=0.9, wind_drag=2.0, wind_tangent=0.1,
                                          fluid_coupling=1.0))
    S.bind_roots()
    S.set_wind((4.0, 0.0, 0.0))
    x0 = S.positions[S.offsets[1:] - 1, 0].copy()
    for _ in range(20):
        S.step()
    x1 = S.positions[S.offsets[1:] - 1, 0]
    ok(float(x1.mean() - x0.mean()) > 0.04, "wind in +X deflects hanging tips (+%.3f m)" % float(x1.mean() - x0.mean()))

    # a callable fluid field (e.g. Tessera wind tunnel / fluid particles) should also couple
    S.reset()
    S.set_wind((0, 0, 0))
    S.set_fluid_velocity(lambda P: np.broadcast_to(np.array((0.0, 3.0, 0.0)), P.shape).copy())
    y0 = S.positions[S.offsets[1:] - 1, 1].copy()
    for _ in range(20):
        S.step()
    y1 = S.positions[S.offsets[1:] - 1, 1]
    ok(float(y1.mean() - y0.mean()) > 0.03, "fluid-velocity callable in +Y deflects tips (+%.3f m)" % float(y1.mean() - y0.mean()))


def test_guides_and_children():
    P, off = H.make_hemisphere_groom(40, 8, 0.12, 0.28, center=(0, 0, 1.6), seed=7)
    gidx = H.select_guides(40, ratio=0.25, seed=0)
    ok(len(gidx) >= 10 and len(np.unique(gidx)) == len(gidx), "guide selection is unique and ~25 %% (%d)" % len(gidx))
    # same seed is deterministic
    ok(np.array_equal(gidx, H.select_guides(40, ratio=0.25, seed=0)), "guide selection is deterministic")
    cidx = np.setdiff1d(np.arange(40), gidx)
    gP, gOff = H.pack_strands([P[off[i]:off[i + 1]] for i in gidx])
    cP, cOff = H.pack_strands([P[off[i]:off[i + 1]] for i in cidx])
    # at rest, interpolation must reproduce the child rest pose
    out = H.interpolate_children(gP, gOff, cP, cOff, guide_rest=gP)
    ok(np.allclose(out, cP, atol=1e-9), "child interpolation is exact at rest (max err %.2e)" % np.max(np.abs(out - cP)))
    S = H.HairSolver(gP, gOff, H.HairParams(dt=1 / 24, substeps=4, gravity=(0, 0, -9.81), rest_shape=0.2))
    S.bind_roots()
    S.attach_children(cP, cOff)
    ch0 = S.child_positions()
    ok(np.allclose(ch0, cP, atol=1e-9), "attached children match rest before any step")
    for _ in range(10):
        S.step()
    ch = S.child_positions()
    # children should have moved with the guides (tips dropped)
    ok(float(ch[cOff[1:] - 1, 2].mean()) < float(cP[cOff[1:] - 1, 2].mean()) - 0.002,
       "children follow simulated guides under gravity")
    # weights are a partition of unity
    ok(np.allclose(S.child_w.sum(axis=1), 1.0, atol=1e-12), "child skin weights sum to 1")


def test_curl_memory_without_gravity():
    P, off = H.make_hemisphere_groom(12, 8, 0.12, 0.25, center=(0, 0, 1.6), seed=8)
    S = H.HairSolver(P, off, H.HairParams(dt=1 / 24, substeps=4, gravity=(0, 0, 0),
                                          rest_shape=0.8, damping=0.85, wind_drag=0.0))
    S.bind_roots()
    # perturb mid-strand points then let style memory pull them back
    mid = S.inv_mass > 0
    S.positions[mid] += (0.05, 0.0, 0.0)
    S.prev[:] = S.positions
    for _ in range(30):
        S.step()
    err = np.linalg.norm(S.positions - S.rest, axis=1).mean()
    ok(err < 0.02, "curl / style memory restores a perturbed rest pose (mean err %.4f m)" % err)


def test_determinism():
    P, off = H.make_hemisphere_groom(24, 8, 0.12, 0.25, center=(0, 0, 1.6), seed=9)
    field, _v, _t = _sphere_field()
    def run():
        S = H.HairSolver(P, off, H.HairParams(dt=1 / 24, substeps=4, gravity=(0, 0, -9.81),
                                              rest_shape=0.3, collision_radius=0.01, sdf_refine=False,
                                              wind_drag=0.5))
        S.bind_roots()
        S.add_collider(field)
        S.set_wind((1.0, 0.2, 0.0))
        for _ in range(8):
            S.step()
        return S.positions.copy()
    a, b = run(), run()
    ok(np.array_equal(a, b), "two identical solvers produce bit-identical positions")


def test_package_and_cache(tmp=None):
    import tempfile
    P, off = H.make_hemisphere_groom(12, 6, 0.12, 0.2, center=(0, 0, 1.6), seed=10)
    S = H.HairSolver(P, off, H.HairParams())
    S.bind_roots()
    frames = []
    for i in range(5):
        S.step()
        frames.append(S.frame_record())
    folder = tmp or tempfile.mkdtemp(prefix="tessera_hair_")
    path = H.write_package(folder, S, frames=frames, collider_meta=[dict(name="head", type="mesh_sdf", resolution=32)])
    ok(os.path.isfile(path) and os.path.isfile(os.path.join(folder, "rest_curves.npz")),
       "game-engine package writes hair_physics.json + rest_curves.npz")
    pkg, rest, cache = H.read_package(folder)
    ok(pkg.get("format") == H.PACKAGE_FORMAT and pkg.get("n_strands") == 12 and "per_strand" in pkg,
       "package metadata has format, strand counts and per-strand stiffness")
    ok(cache is not None and len(cache["frames"]) == 5, "package cache contains 5 frames")
    rec = cache["frames"][-1]
    ok(np.allclose(rec["positions"], S.positions, atol=1e-5), "cached last frame matches the solver")
    # standalone npz cache
    cpath = os.path.join(folder, "solo.npz")
    H.write_npz_cache(cpath, frames, off, rest=P)
    back = H.read_npz_cache(cpath)
    ok(len(back["frames"]) == 5 and np.allclose(back["frames"][0]["positions"], frames[0]["positions"], atol=1e-5),
       "npz point-cache round-trip")


def test_speed():
    field, _v, _t = _sphere_field(res=24, level=1)
    timings = {}
    for label, n, pts, sub, iters in (
        ("64x10 realtime", 64, 10, 4, 6),
        ("256x16 dense", 256, 16, 4, 8),
    ):
        P, off = H.make_hemisphere_groom(n, pts, 0.12, 0.28, center=(0, 0, 1.6), seed=0)
        S = H.HairSolver(P, off, H.HairParams(dt=1 / 24, substeps=sub, stretch_iterations=iters,
                                              collision_iterations=1, sdf_refine=False,
                                              bend_iterations=4, gravity=(0, 0, -9.81)))
        S.bind_roots()
        S.add_collider(field)
        # warmup
        S.step()
        t0 = time.perf_counter()
        nsteps = 20
        for _ in range(nsteps):
            S.step()
        ms = (time.perf_counter() - t0) / nsteps * 1000.0
        timings[label] = ms
        print("  BENCH %s: %.2f ms/step  (%d strands x %d pts, len-err %.4f)" % (label, ms, n, pts, S.length_error()))
    ok(timings["64x10 realtime"] < 16.5, "64x10 groom steps under 16.5 ms (%.2f ms) — 60 fps budget" % timings["64x10 realtime"])
    ok(timings["256x16 dense"] < 120.0, "256x16 dense groom stays interactive (%.2f ms/step)" % timings["256x16 dense"])
    test_speed.timings = timings


def main():
    test_pack_and_groom()
    test_root_pinning()
    test_gravity_and_inertia()
    test_angular_impulse()
    test_sdf_collision()
    test_mesh_bind_inertia()
    test_wind_coupling()
    test_guides_and_children()
    test_curl_memory_without_gravity()
    test_determinism()
    test_package_and_cache()
    test_speed()
    print("ALL PASS" if not ok.fail else "FAILED %d" % ok.fail)
    return ok.fail


if __name__ == "__main__":
    sys.exit(main())
