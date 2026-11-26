"""Pure-Python (no bpy) tests for GARMENT GROUPS: several meshes merged into one XPBD system.

    python3 tests/test_cloth_group.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'arc_flow'))
from core import cloth_solver as CS


def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        ok.fail += 1
ok.fail = 0


def merge(parts):
    vs, ts, mem, off = [], [], [], 0
    for k, (v, t) in enumerate(parts):
        vs.append(v); ts.append(t + off); mem.append(np.full(len(v), k)); off += len(v)
    return np.concatenate(vs), np.concatenate(ts), np.concatenate(mem)


def sheet(n, w, z, ox=0.0, oy=0.0):
    v, t, _q = CS.grid_mesh(n, n, w, w, origin=(ox, oy, z), axis='xy')
    return v, t


def gap_report(S, lo_ids, hi_ids):
    """Smallest signed (hi - lo) height over hi vertices vs lo sheet (sheets are horizontal, hi starts above)."""
    zlo = S.x[lo_ids, 2]
    xy = S.x[lo_ids, :2]
    worst = 1e9
    for i in hi_ids:
        d = np.linalg.norm(xy - S.x[i, :2], axis=1)
        j = np.argsort(d)[:4]
        worst = min(worst, float(S.x[i, 2] - zlo[j].mean()))
    return worst


def run_drop(sc, steps=70, members=True, **kw):
    a = sheet(14, 0.5, 0.0)
    b = sheet(11, 0.45, 0.06, 0.02, 0.02)
    v, t, mem = merge([a, b])
    pins = np.arange(len(a[0]))                 # the lower sheet is pinned (a table cloth), the upper one falls on it
    S = CS.ClothSolver(v, t, pins=pins, preset='COTTON', quality='BALANCED', self_collision=sc,
                       self_collision_mode='LAYERED', self_thickness=0.004, substeps=4, iterations=8,
                       members=mem if members else None, member_params=[dict(thickness=0.003), dict(thickness=0.003)])
    for _ in range(steps):
        S.step(1.0 / 60.0)
        S.v[len(a[0]):] += np.array([0, 0, -0.4])     # shove the upper sheet down hard each step
    return S, np.arange(len(a[0])), np.arange(len(a[0]), len(v))


def test_two_members_collide():
    S0, lo, hi = run_drop(False)
    S1, lo, hi = run_drop(True)
    g0, g1 = gap_report(S0, lo, hi), gap_report(S1, lo, hi)
    ok(g0 < -0.05, "group members WITHOUT self-collision pass through each other (gap %.3f)" % g0)
    ok(g1 > 0.0, "two separate meshes solved as one group: upper never passes through the lower (gap %.4f)" % g1)
    ok(S1.members is not None and S1.members.max() == 1, "solver keeps one member id per vertex")


def test_instances_stack():
    base = sheet(9, 0.3, 0.0)
    parts = [(base[0] + np.array([0, 0, 0.05 * k]) , base[1]) for k in range(4)]   # four instances of one part, stacked
    v, t, mem = merge(parts)
    n = len(base[0])
    S = CS.ClothSolver(v, t, pins=np.arange(n), preset='SILK', quality='BALANCED', self_collision=True,
                       self_collision_mode='LAYERED', self_thickness=0.004, members=mem)
    for _ in range(60):
        S.step(1.0 / 60.0)
        S.v[n:] += np.array([0, 0, -0.3])
    zs = [float(S.x[k * n:(k + 1) * n, 2].mean()) for k in range(4)]
    ok(all(zs[k + 1] > zs[k] for k in range(3)), "four instances of one part keep their stacking order (mean z %s)" % np.round(zs, 4))
    ws = S.self_collider().wrong_side(S.x, labels=mem)
    bad = sum(c for (a, b), (c, d) in ws.items())
    ok(bad == 0, "instances: zero vertices on the wrong side (%d)" % bad)


def test_pocket_stays_attached():
    # a pinned cloth hanging in XZ with a small pocket sewn on the front, welded along its border
    cv, ct, _q = CS.grid_mesh(15, 15, 0.7, 0.7, origin=(-0.35, 0.0, 0.0), axis='xz')
    pv, pt, _q = CS.grid_mesh(5, 5, 0.18, 0.18, origin=(-0.09, -0.004, 0.25), axis='xz')
    v, t, mem = merge([(cv, ct), (pv, pt)])
    pins = CS.top_row_pins(15, 15)
    S = CS.ClothSolver(v, t, pins=pins, preset='COTTON', quality='BALANCED', self_collision=True,
                       self_collision_mode='LAYERED', self_thickness=0.003, members=mem,
                       member_params=[dict(), dict(preset=None) if False else dict(density=0.5, bend=200.0)])
    # weld the pocket border vertices to the nearest cloth surface point
    n0 = len(cv)
    border = [i for i in range(len(pv)) if (pv[i, 0] in (pv[:, 0].min(), pv[:, 0].max()) or pv[i, 2] in (pv[:, 2].min(), pv[:, 2].max()))]
    from core import cloth_selfcol as SC
    A, B, C = cv[ct[:, 0]], cv[ct[:, 1]], cv[ct[:, 2]]
    vs, trs, bs = [], [], []
    for i in border:
        d = np.linalg.norm((A + B + C) / 3 - pv[i], axis=1)
        k = int(np.argmin(d))
        q, bary, _in = SC.closest_point_triangle(pv[i:i + 1], A[k:k + 1], B[k:k + 1], C[k:k + 1])
        vs.append(n0 + i); trs.append(ct[k]); bs.append(bary[0])
    S.set_stitches(vs, trs, bs, stiffness=5e4)
    wind = np.array([0.0, 6.0, 0.0])
    for k in range(120):
        S.step(1.0 / 60.0, wind=wind * (1 + 0.5 * np.sin(k * 0.2)))
    st = S.stitch
    q = (st['bary'][:, :, None] * S.x[st['tri']]).sum(1)
    dist = np.linalg.norm(S.x[st['a']] - q, axis=1)
    ok(dist.max() < 0.02, "pocket stays attached: worst weld distance %.4f m after 120 steps in wind" % dist.max())
    ws = S.self_collider().wrong_side(S.x, labels=mem)
    ok(ws[(1, 0)][0] == 0 or ws[(1, 0)][1] < 0.01, "pocket does not poke through the body cloth %s" % (ws[(1, 0)],))
    S2 = CS.ClothSolver(v, t, pins=pins, preset='COTTON', quality='BALANCED', self_collision=True, members=mem)
    for k in range(120):
        S2.step(1.0 / 60.0, wind=wind * (1 + 0.5 * np.sin(k * 0.2)))
    d2 = np.linalg.norm(S2.x[n0:].mean(0) - S.x[n0:].mean(0))
    ok(d2 > 0.0, "unwelded pocket moves differently (sanity %.3f)" % d2)


def test_member_fabric():
    a = sheet(10, 0.4, 0.0)
    b = sheet(10, 0.4, 0.0, 1.0, 0.0)
    v, t, mem = merge([a, b])
    S = CS.ClothSolver(v, t, pins=None, preset='COTTON', members=mem,
                       member_params=[dict(density=0.7, stretch=1.4e5, bend=520.0, thickness=0.005),
                                      dict(density=0.08, stretch=2.4e4, bend=12.0, thickness=0.0015)])
    ok(S.inv_mass[0] < S.inv_mass[len(a[0])], "leather member is heavier than the silk member")
    ok(S._ce['stretch'][0] < S._ce['stretch'][-1], "leather stretch compliance is lower than silk")
    ok(abs(S.vert_thickness[0] - 0.005) < 1e-9 and abs(S.vert_thickness[-1] - 0.0015) < 1e-9, "per-member thickness")


def test_external_collider():
    # a group of two sheets resting on a box: the collider applies to the whole group
    a = sheet(11, 0.4, 0.30, -0.2, -0.2)
    b = sheet(11, 0.4, 0.34, -0.2, -0.2)
    v, t, mem = merge([a, b])
    from core import primitives as MS
    S = CS.ClothSolver(v, t, preset='COTTON', members=mem, self_collision=True, self_collision_mode='LAYERED',
                       self_thickness=0.004, ground_z=0.0)
    bv, bt = np.array([(-1, -1, 0.0), (1, -1, 0.0), (1, 1, 0.0), (-1, 1, 0.0), (-1, -1, 0.2), (1, -1, 0.2), (1, 1, 0.2), (-1, 1, 0.2)], float), \
        np.array([(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)])
    S.set_collider_mesh(bv * 0.6, bt)
    for _ in range(80):
        S.step(1.0 / 60.0)
    zmin_a = float(S.x[:len(a[0]), 2].min())
    ok(zmin_a > 0.119, "whole group rests on the external collider (lower layer min z %.3f)" % zmin_a)
    ws = S.self_collider().wrong_side(S.x, labels=mem)
    ok(sum(c for c, d in ws.values()) == 0, "and the layers did not swap on the way down")


if __name__ == '__main__':
    for f in (test_member_fabric, test_two_members_collide, test_instances_stack, test_pocket_stays_attached, test_external_collider):
        f()
    print("ALL PASS" if not ok.fail else "%d FAILED" % ok.fail)
    sys.exit(1 if ok.fail else 0)
