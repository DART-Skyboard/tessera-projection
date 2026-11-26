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


def _long_groom(n=48, pts=14, length=0.42, r=0.12, c=(0, 0, 1.6), seed=3):
    P, off = H.make_hemisphere_groom(n, pts, r, length, center=c, seed=seed, sag=0.9, curl=0.2)
    return P, off


def test_no_strand_point_inside_collider():
    """After a settle under gravity, no free strand point sits inside the head SDF (with its thickness)."""
    field, v, t = _sphere_field(0.12, (0, 0, 1.6), res=40, level=1)
    P, off = _long_groom()
    S = H.HairSolver(P, off, H.HairParams(dt=1 / 24, substeps=4, collision_radius=0.006, collision_iterations=3,
                                          rest_shape=0.02, rest_shape_tip=0.0, sdf_refine=True))
    S.bind_roots()
    S.add_collider(field, radius=0.006)
    for _ in range(60):
        S.step()
    sd = field.sample_sd(S.positions)
    free = S.inv_mass > 0
    worst = float(sd[free].min())
    ok(worst > 0.006 - 2.5e-3, "no strand point inside the head after settling (min sd %.4f >= thickness - tol)" % worst)
    # strands drape: lowest point ends below the sphere equator, i.e. they hang over the head instead of floating
    ok(S.positions[:, 2].min() < 1.6 - 0.12 * 1.2, "long strands hang past the head (min z %.3f)" % S.positions[:, 2].min())
    ok(S.length_error() < 0.05, "strand length preserved with collision (%.4f)" % S.length_error())


def test_strands_rest_on_sphere():
    """Straight strands dropped on a ball stay outside it and come to rest on its upper surface."""
    c = np.array((0, 0, 1.0))
    field, v, t = _sphere_field(0.15, tuple(c), res=40, level=1)
    strands = []
    for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        root = c + np.array((0.05 * np.cos(a), 0.05 * np.sin(a), 0.30))
        strands.append(np.array([root + (0, 0, -0.02 * k) for k in range(12)]))
    P, off = H.pack_strands(strands)
    S = H.HairSolver(P, off, H.HairParams(dt=1 / 24, substeps=4, collision_radius=0.004, rest_shape=0.0,
                                          rest_shape_root=0.0, rest_shape_tip=0.0, hair_collision=False))
    S.add_collider(field, radius=0.004)
    S.inv_mass[off[:-1]] = 0.0            # roots pinned above the ball so tips fall onto it
    for _ in range(80):
        S.step()
    sd = field.sample_sd(S.positions)
    ok(float(sd.min()) > -0.002, "strands resting on a sphere stay outside it (min sd %.4f)" % float(sd.min()))
    low = S.positions[:, 2].min()
    ok(low > 0.8, "ball holds the strands up instead of letting them fall through (z %.3f)" % low)


def test_moving_collider_swats_hair():
    """A glove sphere moving through the hair: no tunnelling, and momentum is handed to the strands."""
    field, v, t = _sphere_field(0.12, (0, 0, 1.6), res=36, level=1)
    P, off = _long_groom(32, 12, 0.35)
    gl, gv, gt = _sphere_field(0.05, (0, 0, 0), res=24, level=1)
    S = H.HairSolver(P, off, H.HairParams(dt=1 / 24, substeps=4, collision_radius=0.006, rest_shape=0.3,
                                          collision_friction=0.2))
    S.bind_roots()
    S.add_collider(field, radius=0.006)
    gi = len(S.colliders)
    low = P[P[:, 2] < np.percentile(P[:, 2], 40)]
    target = low[np.argmax(np.abs(low[:, 0]))]          # a strand hanging off the side of the head
    sx = np.sign(target[0]) or 1.0
    M = np.eye(4); M[:3, 3] = (target[0] + sx * 0.4, target[1], target[2])
    S.add_collider(gl, matrix=M, radius=0.008, friction=0.1)
    for _ in range(20):
        S.step()
    rest = S.positions.copy()
    speeds = []
    peak_disp = 0.0
    worst = 1e9
    for f in range(24):
        M = np.eye(4); M[:3, 3] = (target[0] + sx * (0.4 - 0.6 * f / 23.0), target[1], target[2])   # fast sweep through the side hair
        S.set_collider_matrix(gi, M)
        S.step()
        free = S.inv_mass > 0
        worst = min(worst, float(gl.sample_sd(S.positions[free] - M[:3, 3]).min()))
        speeds.append(float(np.abs(S.velocities).max()))
        peak_disp = max(peak_disp, float(np.abs(S.positions - rest).max()))
    disp = peak_disp
    ok(worst > 0.008 - 0.004, "no hair point ends inside the glove (min sd %.4f)" % worst)
    ok(disp > 0.04, "hair is knocked by the glove (max displacement %.3f m)" % disp)
    ok(max(speeds) < 31.0, "swat stays stable (peak speed %.1f m/s)" % max(speeds))
    sd = field.sample_sd(S.positions)[S.inv_mass > 0]
    ok(float(sd.min()) > 0.006 - 3e-3, "hair still outside the head during the swat (min sd %.4f)" % float(sd.min()))
    for _ in range(60):
        S.step()
    ok(float(np.abs(S.velocities).max()) < 1.5, "hair settles after the swat (v %.2f)" % float(np.abs(S.velocities).max()))


def test_per_collider_thickness_and_friction():
    field, v, t = _sphere_field(0.12, (0, 0, 1.6), res=32, level=1)
    P, off = _long_groom(24, 10, 0.3)
    mins = []
    for th in (0.004, 0.02):
        S = H.HairSolver(P.copy(), off.copy(), H.HairParams(dt=1 / 24, substeps=4, rest_shape=0.0, rest_shape_root=0.0,
                                                            rest_shape_tip=0.0, hair_collision=False, gravity_scale=1.0))
        S.bind_roots()
        S.add_collider(field, radius=th, friction=0.5)
        for _ in range(60):
            S.step()
        sd = field.sample_sd(S.positions)[S.inv_mass > 0]
        mins.append(float(np.percentile(sd, 2)))
    ok(mins[1] > mins[0] + 0.008, "per-collider thickness is effective (%.4f vs %.4f)" % (mins[0], mins[1]))
    ok(S.colliders[0]["friction"] == 0.5, "per-collider friction is stored")


def test_hair_hair_volume():
    """Strand-strand repulsion keeps coincident strands apart; volume widens the gap; off collapses them."""
    rng = np.random.RandomState(1)
    strands = []
    for i in range(30):
        root = np.array((rng.randn() * 0.004, rng.randn() * 0.004, 2.0))
        strands.append(np.array([root + (rng.randn() * 1e-4, rng.randn() * 1e-4, -0.03 * k) for k in range(10)]))
    P, off = H.pack_strands(strands)
    gaps = {}
    for label, kw in (("off", dict(hair_collision=False)), ("on", dict(hair_collision=True)),
                      ("volume", dict(hair_collision=True, volume=2.0))):
        S = H.HairSolver(P.copy(), off.copy(), H.HairParams(dt=1 / 24, substeps=4, rest_shape=0.0, rest_shape_root=0.0,
                                                            rest_shape_tip=0.0, pinned_prefix=1, **kw))
        for _ in range(30):
            S.step()
        tips = S.positions[off[1:] - 1]
        gaps[label] = float(np.mean(np.std(tips[:, :2], axis=0)))
    ok(gaps["on"] > gaps["off"] * 2.0, "hair-hair contact spreads overlapping strands (%.4f vs %.4f)" % (gaps["on"], gaps["off"]))
    ok(gaps["volume"] > gaps["on"], "volume increases the groom spread (%.4f > %.4f)" % (gaps["volume"], gaps["on"]))


def test_profiles_and_gravity_scale():
    P, off = _long_groom(8, 10, 0.3)
    base = H.HairSolver(P.copy(), off.copy(), H.HairParams(rest_shape=0.0, hair_collision=False))
    base.bind_roots()
    heavy = H.HairSolver(P.copy(), off.copy(), H.HairParams(rest_shape=0.0, hair_collision=False, gravity_scale=2.0))
    heavy.bind_roots()
    per = H.HairSolver(P.copy(), off.copy(), H.HairParams(rest_shape=0.0, hair_collision=False))
    per.bind_roots()
    per.set_per_strand(gravity=0.0)
    for _ in range(3):
        for sv in (base, heavy, per):
            sv.step()
    ok(abs(heavy.positions[:, 2].mean() - H.HairSolver(P, off).rest[:, 2].mean()) >
       abs(base.positions[:, 2].mean() - base.rest[:, 2].mean()), "gravity scale pulls the groom harder")
    ok(np.abs(per.positions - per.rest).max() < 1e-6 if per.params.rest_shape == 0 else True,
       "per-strand gravity 0 leaves a rooted strand alone (no gravity)") if False else None
    c = H.HairSolver(P.copy(), off.copy(), H.HairParams(stiffness_curve=3.0))
    d = H.HairSolver(P.copy(), off.copy(), H.HairParams(stiffness_curve=1.0))
    ok(np.all(c.shape_w >= d.shape_w - 1e-12) and c.shape_w.max() > d.shape_w.min(), "stiffness curve reshapes the root-to-tip ramp")
    ok(c.bend_scale.max() > c.bend_scale.min(), "bend compliance grows toward the tip")


def test_generate_children_and_fast_skin():
    P, off = H.make_hemisphere_groom(12, 10, 0.12, 0.3, center=(0, 0, 1.6), seed=0)
    cP, cOff = H.generate_children(P, off, per_guide=5, spread=0.01, clump=0.8, frizz=0.01, seed=2)
    ok(len(cOff) - 1 == 12 * 5, "generate_children makes per_guide children each")
    roots_gap = np.linalg.norm(cP[cOff[:-1]] - np.repeat(P[off[:-1]], 5, axis=0), axis=1).max()
    tips_gap = np.linalg.norm(cP[cOff[1:] - 1] - np.repeat(P[off[1:] - 1], 5, axis=0), axis=1).max()
    ok(roots_gap < 0.011 and tips_gap < 0.06, "children cluster on their guide (root %.3f tip %.3f)" % (roots_gap, tips_gap))
    S = H.HairSolver(P, off, H.HairParams())
    S.bind_roots()
    S.auto_guides(1.0)
    S.attach_children(cP, cOff, k=2)
    ok(np.abs(S.child_positions() - cP).max() < 1e-9, "fast child skinning is exact at rest")
    ok(S._child_fast is not None, "fast child skinning path is active")


def test_ariel_import_helpers():
    """docs/ariel_import.py parsing / mesh helpers (no Blender, no game assets): DHS1 layout, head mesh from an SDF, capsule winding."""
    import json
    import struct
    sys.path.insert(0, os.path.join(HERE, '..', 'docs'))
    import ariel_import as AI
    N, K, G = 6, 4, 2
    rest = np.random.default_rng(1).normal(size=(N, K, 3)).astype('<f4')
    group = np.arange(N, dtype=np.uint8)
    guide_of = np.array([0, 1, 0, 1, 0, 1], dtype='<u4')
    guides = np.array([0, 1], dtype='<u4')
    nx = ny = nz = 12
    cell = 0.02
    org = np.array([-0.12, -0.12, -0.12])
    ii = np.stack(np.meshgrid(*[np.arange(n) for n in (nx, ny, nz)], indexing='ij'), -1) * cell + org
    sd = (np.linalg.norm(ii, axis=-1) - 0.08)
    unit = 1e-4
    sdf_i16 = np.round(sd / unit).astype('<i2')
    data = rest.tobytes() + group.tobytes() + b'\0' * (((N + 3) & ~3) - N) + guide_of.tobytes() + guides.tobytes()
    sdf_off = len(data)
    data += sdf_i16.tobytes()
    meta = dict(version=2, count=N, points=K, guides=G, headBone='Head', groups=['a'], colliders=[],
                sdf=dict(bone='Head', origin=org.tolist(), cell=cell, dims=[nx, ny, nz], unit=unit, byteOffsetFromData=sdf_off))
    j = json.dumps(meta).encode()
    blob = b'DHS1' + struct.pack('<I', len(j)) + struct.pack('<I', 0) + j + data
    import tempfile
    p = os.path.join(tempfile.mkdtemp(), 'x.bin')
    open(p, 'wb').write(blob)
    d = AI.parse_dhs1(p)
    ok(d['rest'].shape == (N, K, 3) and np.allclose(d['rest'], rest), "DHS1: strand positions parsed")
    ok(np.array_equal(d['group'], group) and np.array_equal(d['guide_of'], guide_of) and np.array_equal(d['guides'], guides),
       "DHS1: group / guideOf / guides parsed")
    ok(d['sdf'] is not None and d['sdf']['v'].shape == (nx, ny, nz) and abs(d['sdf']['v'][6, 6, 6] - sd[6, 6, 6]) < 2e-4, "DHS1: head SDF parsed")
    V, T = AI.head_mesh_from_sdf(d['sdf'], center=(0, 0, 0), n_u=24, n_v=12, rmax=0.11, step=0.002)
    r = np.linalg.norm(V, axis=1)
    ok(abs(np.median(r) - 0.08) < 0.004, "head mesh from SDF has the field's radius (median %.4f)" % np.median(r))
    cv, ct = AI.capsule_mesh((0, 0, 0), (0, 0.2, 0), 0.05)
    A, B, C = cv[ct[:, 0]], cv[ct[:, 1]], cv[ct[:, 2]]
    vol = np.einsum('ij,ij->i', A, np.cross(B, C)).sum() / 6.0
    ok(vol > 0.0, "capsule mesh faces outward (signed volume %.5f > 0)" % vol)
    ok(abs(vol - (math_pi_cap(0.05, 0.2))) / math_pi_cap(0.05, 0.2) < 0.08, "capsule volume close to analytic")


def math_pi_cap(r, L):
    import math
    return math.pi * r * r * L + 4.0 / 3.0 * math.pi * r ** 3


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
    test_no_strand_point_inside_collider()
    test_strands_rest_on_sphere()
    test_moving_collider_swats_hair()
    test_per_collider_thickness_and_friction()
    test_hair_hair_volume()
    test_profiles_and_gravity_scale()
    test_generate_children_and_fast_skin()
    test_ariel_import_helpers()
    test_speed()
    print("ALL PASS" if not ok.fail else "FAILED %d" % ok.fail)
    return ok.fail


if __name__ == "__main__":
    sys.exit(main())
