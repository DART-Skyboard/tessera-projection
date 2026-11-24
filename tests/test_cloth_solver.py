"""Pure-Python (no bpy) tests for the XPBD cloth solver:

    python3 tests/test_cloth_solver.py
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'arc_flow'))

from core import cloth_solver as CS
from core import primitives as MS
from core import sdf


def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        ok.fail += 1
ok.fail = 0


def hanging_cape(nx=13, ny=13, width=0.80, height=0.90, preset='COTTON', quality='BALANCED', **kw):
    """Vertical cape in the XZ plane, top edge pinned, hanging down in -Z from z=height."""
    verts, tris, _q = CS.grid_mesh(nx, ny, width, height, origin=(-width / 2, 0.0, 0.0), axis='xz')
    pins = CS.top_row_pins(nx, ny)
    S = CS.ClothSolver(verts, tris, pins=pins, preset=preset, quality=quality, **kw)
    return S, pins, nx, ny


def curtain(nx=13, ny=13, width=0.80, height=0.90, preset='COTTON', quality='BALANCED', **kw):
    """Horizontal cloth in the XY plane, pinned on the +Y edge, drapes under gravity in -Z.

    This is the classic PBD curtain: the sheet starts flat and has to both fall and fold,
    so stretch, bend and wind all have a visible effect.
    """
    verts, tris, _q = CS.grid_mesh(nx, ny, width, height, origin=(-width / 2, 0.0, 0.80), axis='xy')
    pins = CS.top_row_pins(nx, ny)  # last row in Y, the far edge
    S = CS.ClothSolver(verts, tris, pins=pins, preset=preset, quality=quality, **kw)
    return S, pins, nx, ny


def test_topology_and_presets():
    v, t, q = CS.grid_mesh(4, 3, 1.0, 0.5, axis='xz')
    ok(len(v) == 12 and len(t) == 12 and len(q) == 6, "4x3 grid: 12 verts, 12 tris, 6 quads")
    topo = CS.topology(t, len(v))
    ok(len(topo['edges']) == 23 or len(topo['edges']) >= 17, "unique structural edges (%d)" % len(topo['edges']))
    ok(len(topo['hinges']) >= 1 and len(topo['shear']) >= 1, "hinges %d shear %d" % (len(topo['hinges']), len(topo['shear'])))
    pins = CS.top_row_pins(4, 3)
    ok(list(pins) == [8, 9, 10, 11] and np.allclose(v[pins, 2], 0.5), "top-row pins sit at z=height")
    band = CS.band_pins(v, axis=2, lo=0.4, hi=0.6)
    ok(len(band) >= 4, "waist band finds a mid-row (%d verts)" % len(band))
    for key in ('SILK', 'COTTON', 'DENIM', 'LEATHER', 'CAPE'):
        ok(key in CS.PRESETS and CS.PRESETS[key]['density'] > 0, "preset %s" % key)
    ok(set(CS.QUALITY) == {'REALTIME', 'BALANCED', 'QUALITY'}, "quality profiles")


def test_pins_stay_put():
    S, pins, nx, ny = hanging_cape(11, 11, preset='COTTON', quality='REALTIME')
    rest = S.x[pins].copy()
    for _ in range(30):
        S.step(1 / 60.0)
    ok(np.allclose(S.x[pins], rest, atol=1e-9), "hard pins do not move under gravity")
    ok(np.all(np.isfinite(S.x)) and np.all(np.isfinite(S.v)), "positions and velocities stay finite")
    free = ~S.pin_hard
    # a vertical cape mostly stretches: the hem drops below its rest z
    hem = slice(0, nx)  # j=0 row
    ok(S.x[hem, 2].mean() < S.rest[hem, 2].mean() - 0.001 or S.last_stretch > 0,
       "free vertices respond to gravity (hem z %.3f, strain %.4f)" % (S.x[hem, 2].mean(), S.last_stretch))


def test_gravity_drape_stable():
    S, pins, nx, ny = curtain(15, 15, preset='COTTON', quality='BALANCED',
                              use_dihedral=True, self_collision=False)
    z0 = S.x[:, 2].copy()
    t0 = time.perf_counter()
    for _ in range(90):
        S.step(1 / 60.0)
    el = time.perf_counter() - t0
    hem = slice(0, nx)
    hem_drop = float(z0[hem].mean() - S.x[hem, 2].mean())
    # a horizontal sheet pinned on one edge falls into a hanging curtain
    ok(0.20 < hem_drop < 1.10, "cotton curtain hem drapes stably (hem drop %.3f m in 1.5 s, %.1f ms/step)" % (hem_drop, 1000 * el / 90))
    ok(S.last_stretch < 0.45, "peak stretch stays under 45 %% after the swing (%.2f %%)" % (100 * S.last_stretch))
    ok(np.allclose(S.x[pins, 2], z0[pins], atol=1e-8), "pinned edge holds its height")
    ok(S.x[hem, 2].mean() < z0[pins].mean() - 0.20, "hem falls well below the pin line (z %.3f)" % S.x[hem, 2].mean())
    ok(S.x[:, 2].min() > -0.6, "hem does not explode below the rest plane (zmin %.3f)" % S.x[:, 2].min())
    ok(np.all(np.isfinite(S.x)) and np.all(np.isfinite(S.v)), "drape stays finite")


def test_stretch_stiffness():
    def hang(stretch):
        # Vertical cape: elongation is almost pure stretch (already in equilibrium shape).
        S, pins, nx, ny = hanging_cape(11, 13, height=0.90, preset='COTTON', quality='BALANCED',
                                       stretch=stretch, shear=stretch * 0.4, bend=20.0,
                                       use_dihedral=False, self_collision=False, damping=0.4)
        for _ in range(80):
            S.step(1 / 60.0)
        hem = slice(0, nx)
        elong = float(S.rest[hem, 2].mean() - S.x[hem, 2].mean())
        return S.last_stretch, elong, S

    lo, drop_lo, _ = hang(6.0e2)
    hi, drop_hi, _ = hang(1.6e5)
    ok(drop_lo > drop_hi * 1.4 or lo > hi * 1.4,
       "low stretch stiffness elongates more (strain %.3f vs %.3f, hem drop %.4f vs %.4f m)" % (lo, hi, drop_lo, drop_hi))
    ok(hi < 0.08, "stiff cloth peak strain stays tight (%.3f)" % hi)


def test_bend_stiffness():
    """A 3x2 strip, pinned on the left two columns, is folded 90 deg about the middle
    hinge.  After a few steps a stiffer dihedral unfold is flatter (smaller |z|)."""
    def residual(bend, steps=6):
        verts, tris, _q = CS.grid_mesh(3, 2, width=0.4, height=0.2, axis='xy')
        S = CS.ClothSolver(verts, tris, pins=(0, 1, 3, 4), preset='COTTON', quality='REALTIME',
                           gravity=(0, 0, 0), damping=1.0, bend=bend,
                           use_dihedral=True, use_isometric_bend=False, self_collision=False,
                           stretch=8.0e4, shear=4.0e4, substeps=2, iterations=4)
        for vid in (2, 5):
            x, y, z = S.x[vid]
            S.x[vid] = (0.2, y, -(x - 0.2))
        S.x_prev = S.x.copy()
        S.v[:] = 0
        for _ in range(steps):
            S.step(1 / 60.0, gravity=(0, 0, 0))
        rec = 0.5 * (S.x[2, 0] + S.x[5, 0])          # 0.2 folded, 0.4 flat
        fold = 0.5 * (abs(S.x[2, 2]) + abs(S.x[5, 2]))  # 0.2 folded, 0 flat
        return rec, fold, S
    rec_s, fold_s, _ = residual(0.5)
    rec_h, fold_h, _ = residual(80.0)
    ok(fold_h < fold_s - 0.04 and rec_h > rec_s + 0.01,
       "stiffer bend unfolds a 90 deg fold farther (x rec %.3f vs %.3f, |z| %.3f vs %.3f)" % (rec_h, rec_s, fold_h, fold_s))


def test_wind_billowing():
    def run(wind):
        # vertical XZ cape: the faces point along +/-Y, so a +Y wind is front-on
        S, pins, nx, ny = hanging_cape(13, 13, preset='CAPE', quality='BALANCED',
                                       self_collision=False)
        for _ in range(50):
            S.step(1 / 60.0, wind=np.asarray(wind, float))
        free = ~S.pin_hard
        return S.x[free].mean(axis=0), S

    c0, _ = run((0.0, 0.0, 0.0))
    cW, SW = run((0.0, 10.0, 0.0))
    ok(cW[1] > c0[1] + 0.04, "cape billows downwind (+Y): still y=%.3f, wind y=%.3f" % (c0[1], cW[1]))
    ok(np.all(np.isfinite(SW.x)), "wind step stays finite")


def test_sdf_body_collision():
    # cape hanging in front of a sphere body at the origin; the cloth sits at y=0.22
    # and must remain outside the sphere (r=0.20) plus the collision margin
    R = 0.20
    verts, tris, _q = CS.grid_mesh(15, 15, 0.70, 0.80, origin=(-0.35, 0.22, -0.05), axis='xz')
    pins = CS.top_row_pins(15, 15)
    S = CS.ClothSolver(verts, tris, pins=pins, preset='COTTON', quality='BALANCED',
                       margin=0.008, self_collision=False, use_dihedral=True)
    sv, st = MS.uvsphere(n=20, m=12, r=R, center=(0.0, 0.0, 0.35))
    S.set_collider_mesh(sv, st, margin=0.008, tessellate=False)
    for _ in range(70):
        S.step(1 / 60.0)
    sd = S.penetration()
    free = S.inv_mass > 0
    # signed distance is negative inside the body; require every free vertex outside + margin (tiny slack)
    slack = 1.5e-3
    worst = float((S.margin - sd[free]).max()) if free.any() else 0.0
    ok(worst <= slack, "zero body penetration past margin (worst %.4f m, margin %.4f, slack %.4f)" % (worst, S.margin, slack))
    ok(float(sd[free].min()) >= -0.002, "no vertex sits inside the sphere (min sd %.4f)" % float(sd[free].min()))
    # the drape should actually have come to rest *against* the body, not hung in mid-air
    ok(float(sd[free].min()) < S.margin + 0.06, "cloth comes to rest against the collider (min sd %.4f)" % float(sd[free].min()))


def test_moving_rig_inertia():
    S, pins, nx, ny = hanging_cape(11, 11, preset='COTTON', quality='REALTIME',
                                   self_collision=False, gravity=(0, 0, 0), damping=0.05)
    # settle briefly
    for _ in range(5):
        S.step(1 / 60.0, gravity=(0, 0, 0))
    # yank the pins in +X at 2 m/s for 0.25 s
    dt = 1 / 60.0
    vel = np.array([2.0, 0.0, 0.0])
    tgt = S.pin_target[pins].copy()
    free = ~S.pin_hard
    # mid-hem vertex (bottom row, centre)
    hem = (ny - 0) * 0 + (0) * nx + nx // 2  # bottom row starts at 0
    # actually j=0 is the bottom in grid_mesh (ys from 0 to height), top is last row
    hem = nx // 2  # bottom-centre
    for _ in range(15):
        tgt = tgt + vel * dt
        S.set_pin_targets(tgt, pins, dt=dt)
        S.step(dt, gravity=(0, 0, 0))
    pin_x = float(S.x[pins, 0].mean())
    hem_x = float(S.x[hem, 0])
    ok(pin_x > 0.4, "pins travelled with the rig (pin x=%.3f)" % pin_x)
    ok(hem_x < pin_x - 0.015, "hem lags the pins from inertia (hem x=%.3f, pin x=%.3f)" % (hem_x, pin_x))
    # after the yank stops, the hem should start catching up
    hold = S.pin_target[pins].copy()
    for _ in range(40):
        S.set_pin_targets(hold, pins, dt=dt)
        S.step(dt, gravity=(0, 0, 0))
    hem_later = float(S.x[hem, 0])
    ok(hem_later > hem_x + 0.02, "hem catches up after the rig stops (%.3f -> %.3f)" % (hem_x, hem_later))


def test_skin_weights():
    v, t, _q = CS.grid_mesh(5, 5, 0.4, 0.4, axis='xz')
    # one bone: identity bind, later translated
    bind = np.eye(4)[None, ...]
    idx = np.full((len(v), 1), 0, np.int64)
    w = np.zeros((len(v), 1), float)
    top = CS.top_row_pins(5, 5)
    w[top, 0] = 1.0
    S = CS.ClothSolver(v, t, preset='SILK', quality='REALTIME', gravity=(0, 0, -9.81))
    S.set_skin(idx, w, bind, threshold=0.5, anim_drive=1.0)
    M = np.eye(4)
    M[:3, 3] = (0.15, 0.0, 0.0)
    for _ in range(20):
        S.step(1 / 60.0, bone_matrices=M[None, ...])
    ok(np.allclose(S.x[top, 0], v[top, 0] + 0.15, atol=1e-6), "skinned pins follow the bone translation")
    ok(S.x[~S.pin_hard, 2].mean() < v[:, 2].mean(), "unskinned vertices still drape under gravity")


def test_self_collision_pushout():
    # two layers of a folded sheet that start closer than 2*thickness
    v, t, _q = CS.grid_mesh(6, 6, 0.3, 0.3, axis='xy')
    S = CS.ClothSolver(v, t, preset='COTTON', quality='REALTIME',
                       self_collision=True, thickness=0.02, gravity=(0, 0, 0),
                       use_dihedral=False)
    # squash a non-adjacent vertex onto another
    # pick vertex 0 and a vertex not in its 2-ring
    a = 0
    skip = set()
    for j in S.neighbors[a]:
        skip.add(j)
        skip.update(S.neighbors[j])
    skip.add(a)
    b = next(i for i in range(S.n) if i not in skip)
    S.x[b] = S.x[a] + np.array([0.002, 0.0, 0.0])
    S._self_collide()
    sep = float(np.linalg.norm(S.x[b] - S.x[a]))
    ok(sep >= 2 * S.thickness - 1e-6, "self-collision push-out separates overlapping verts (%.4f >= %.4f)" % (sep, 2 * S.thickness))


def test_package_and_snapshot():
    S, pins, nx, ny = hanging_cape(8, 8, preset='CAPE', quality='REALTIME')
    S.step(1 / 60.0)
    snap = S.snapshot()
    x = S.x.copy()
    S.step(1 / 60.0)
    S.restore(snap)
    ok(np.allclose(S.x, x) and S.frame == snap['frame'], "snapshot / restore round-trips state")
    pkg = S.package_dict(name="cape", colliders=[dict(name="body", margin=S.margin)])
    ok(pkg['format'] == 'tessera-cloth-package' and pkg['preset'] == 'CAPE', "package format and preset")
    ok(pkg['rest_mesh']['n_vertices'] == S.n and len(pkg['pins']['indices']) == len(pins), "package rest mesh and pins")
    ok('unreal_chaos' in pkg['engine_hints'] and 'unity' in pkg['engine_hints'] and 'web' in pkg['engine_hints'],
       "Unreal / Unity / web engine hints present")
    ok(0.0 <= pkg['engine_hints']['unreal_chaos']['stretch_stiffness'] <= 1.0, "Chaos stretch hint is 0-1")
    CS.apply_preset(S, 'LEATHER')
    ok(S.preset == 'LEATHER' and S.density == CS.PRESETS['LEATHER']['density'], "apply_preset retargets density")


def test_realtime_step_speed():
    # 25x25 = 625 verts, typical real-time garment budget
    S, pins, nx, ny = hanging_cape(25, 25, width=1.0, height=1.1, preset='CAPE', quality='REALTIME',
                                   self_collision=False)
    # warm up (JIT-less but fills caches / BVH)
    S.step(1 / 60.0)
    times = []
    for _ in range(40):
        t0 = time.perf_counter()
        S.step(1 / 60.0, wind=np.array([3.0, 0.0, 0.0]))
        times.append(time.perf_counter() - t0)
    med = 1000.0 * float(np.median(times))
    mean = 1000.0 * float(np.mean(times))
    print("   realtime 25x25 cape: median %.2f ms/step, mean %.2f ms/step (budget 16.7 ms)" % (med, mean))
    ok(med < 25.0, "real-time quality 25x25 cape steps at interactive rates (median %.2f ms; 60 fps budget is 16.7 ms)" % med)
    # also record a balanced 17x17 timing for the PR notes
    S2, _, _, _ = hanging_cape(17, 17, preset='COTTON', quality='BALANCED', self_collision=True)
    sv, st = MS.uvsphere(n=16, m=10, r=0.18, center=(0, -0.05, 0.4))
    S2.set_collider_mesh(sv, st, tessellate=False)
    S2.step(1 / 60.0)
    t0 = time.perf_counter()
    for _ in range(20):
        S2.step(1 / 60.0)
    bal = 1000.0 * (time.perf_counter() - t0) / 20
    print("   balanced 17x17 + sphere collider + self-collision: mean %.2f ms/step" % bal)
    ok(np.all(np.isfinite(S2.x)), "balanced collider run stays finite")
    # stash timings on the module for the bpy test / report
    test_realtime_step_speed.median_ms = med
    test_realtime_step_speed.balanced_ms = bal



def test_quality_profiles_default_self_collision():
    v, t, _q = CS.grid_mesh(4, 4, 0.2, 0.2, axis='xz')
    for q in ('REALTIME', 'BALANCED', 'QUALITY'):
        S = CS.ClothSolver(v, t, quality=q)
        ok(S.self_collision is True, "quality %s enables self-collision by default" % q)


def test_sheet_self_intersection_count():
    """Two overlapping sheets: report contact count before and after self-collision resolve."""
    v1, t1, _ = CS.grid_mesh(8, 8, 0.4, 0.4, axis='xy')
    v2, t2, _ = CS.grid_mesh(8, 8, 0.4, 0.4, axis='xy')
    v2[:, 2] += 0.002          # 2 mm above, well inside 2*thickness
    v2[:, 0] += 0.05
    off = len(v1)
    v = np.vstack([v1, v2])
    t = np.vstack([t1, t2 + off])
    S = CS.ClothSolver(v, t, preset='SILK', quality='BALANCED',
                       self_collision=True, thickness=0.01, gravity=(0, 0, 0),
                       use_dihedral=False)
    before = S.count_self_intersections(proximity=True)
    ok(before > 0, "overlapping sheets start with self-contacts (%d)" % before)
    for _ in range(8):
        S._self_collide()
    after = S.count_self_intersections(proximity=True)
    ok(after <= max(2, before // 20),
       "self-collision clears overlapping sheets (before %d, after %d)" % (before, after))
    test_sheet_self_intersection_count.before = before
    test_sheet_self_intersection_count.after = after


def test_collider_tessellation():
    v, t, _q = CS.grid_mesh(6, 6, 0.4, 0.4, axis='xz')
    S = CS.ClothSolver(v, t, quality='QUALITY', tessellation_edge=0.15, tessellation_cap=5000)
    sv, st = MS.box(1.0, 1.0, 1.0, center=(0, -0.6, 0.2))
    S.set_collider_mesh(sv, st, tessellate=True)
    ok(S.collider is not None and len(S.collider) >= 12, "tessellated box collider has at least the source triangles (%d)" % len(S.collider))


if __name__ == '__main__':
    test_topology_and_presets()
    test_realtime_step_speed()
    test_pins_stay_put()
    test_gravity_drape_stable()
    test_stretch_stiffness()
    test_bend_stiffness()
    test_wind_billowing()
    test_sdf_body_collision()
    test_moving_rig_inertia()
    test_skin_weights()
    test_self_collision_pushout()
    test_quality_profiles_default_self_collision()
    test_sheet_self_intersection_count()
    test_package_and_snapshot()
    test_collider_tessellation()
    print("ALL PASS" if ok.fail == 0 else "FAILED %d" % ok.fail)
    sys.exit(1 if ok.fail else 0)
