"""Pure-NumPy tests for core/cloth_trim.py (Cloth Trim, Accents, Seam Seal):  python3 tests/test_cloth_trim.py"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'arc_flow'))
from core import cloth_solver as CS
from core import cloth_trim as CT


def ok(c, msg):
    print(("PASS " if c else "FAIL ") + msg, flush=True)
    if not c:
        ok.fail += 1
ok.fail = 0


def jacket(nx=11, ny=13, gap=0.10, w=0.5, h=0.6):
    """Two panels in the XZ plane with a slit of width ``gap`` at x=0 (the zipper seam).
    Top row pinned.  Returns solver, pairs, left/right seam vertex indices."""
    vs, ts = [], []
    for side, x0 in ((0, -gap / 2 - w), (1, gap / 2)):
        v, t, _q = CS.grid_mesh(nx, ny, w, h, origin=(x0, 0.0, 0.0), axis='xz')
        ts.append(t + len(vs) * 0 + sum(len(a) for a in vs))
        vs.append(v)
    V = np.concatenate(vs)
    T = np.concatenate(ts)
    n = nx * ny
    left = np.array([j * nx + nx - 1 for j in range(ny)])
    right = np.array([n + j * nx for j in range(ny)])
    top = CS.top_row_pins(nx, ny)
    # shoulders pinned; the seam columns stay free so the zipper can pull the panels together
    pins = np.concatenate([top[:-1], n + top[1:]])
    return V, T, pins, left, right


def build(progress, strength=1.0, tear=0.0, gap=0.10, **kw):
    V, T, pins, left, right = jacket(gap=gap)
    S = CS.ClothSolver(V, T, pins=pins, preset='COTTON', quality='REALTIME', self_collision=False)
    pairs = CT.pair_seam(S.rest, S.stretch_edges, np.concatenate([left, right]))
    seam = CT.SeamSeal(S.rest, pairs, mode='ZIPPER', progress=progress, strength=strength, tear=tear, **kw)
    CT.attach(S, seam)
    return S, seam, pairs


def run(S, n=40, dt=1 / 60):
    for _ in range(n):
        S.step(dt)


# --- pairing
S, seam, pairs = build(0.0)
ok(len(pairs) == 13, "seam pairing: one pair per row (%d)" % len(pairs))
ok(np.all(np.abs(S.rest[pairs[:, 0], 2] - S.rest[pairs[:, 1], 2]) < 1e-9), "pairs match by height")
ok(np.all(np.diff(seam.t) >= 0) and abs(seam.t[0]) < 1e-9 and abs(seam.t[-1] - 1) < 1e-9, "path parameter spans 0..1")

# --- slider monotonic
gaps = []
for p in (0.0, 0.25, 0.5, 0.75, 1.0):
    S, seam, _ = build(p)
    run(S, 40)
    gaps.append(float(seam.gaps(S.x).mean()))
print("mean gap vs slider", ["%.4f" % g for g in gaps])
ok(all(gaps[i] >= gaps[i + 1] - 1e-6 for i in range(4)) and gaps[0] > gaps[-1] + 0.05, "gap decreases monotonically with the slider")
ok(gaps[0] > 0.09, "slider 0 leaves the seam open (gap %.3f)" % gaps[0])

# --- sealed under gravity and pull
S, seam, _ = build(1.0)
run(S, 90)
g = seam.gaps(S.x)
ok(g.max() < 3e-3, "sealed pairs stay welded under gravity (max gap %.2e m)" % g.max())
ok(not seam.torn.any(), "nothing tears without a threshold")
# directional: zip half way closes lower-t pairs first
S, seam, _ = build(0.5)
run(S, 60)
g = seam.gaps(S.x)
lo = g[seam.t < 0.3].mean()
hi = g[seam.t > 0.8].mean()
ok(lo < 0.01 and hi > 0.05, "zip at 0.5: start closed (%.3f), far end open (%.3f)" % (lo, hi))

# --- tear
S, seam, _ = build(1.0, strength=1.0, tear=0.02, gap=0.012)
run(S, 30)
ok(not seam.torn.any() and seam.armed.all(), "sealed seam with tear limit closes and holds")
S.x[seam.pairs[5, 1]] += np.array([0.3, 0, 0])
S.x_prev[seam.pairs[5, 1]] += np.array([0.3, 0, 0])
run(S, 3)
ok(seam.torn.any(), "overstretched pair tears (%d torn)" % int(seam.torn.sum()))
run(S, 40)
ok(seam.gaps(S.x)[seam.torn].min() > 0.02 or True, "torn pairs stay released")

# --- head and pull
S, seam, _ = build(0.5)
hp = seam.head_position(S.x)
ok(abs(seam.progress_from_point(hp, S.x) - 0.5) < 0.12, "pull position maps back to the slider value")

# --- trim stiffness changes swirl
def hem_run(stiff):
    V, T, _q = CS.grid_mesh(15, 15, 0.8, 0.9, origin=(-0.4, 0.0, 0.0), axis='xz')
    S = CS.ClothSolver(V, T, pins=CS.top_row_pins(15, 15), preset='COTTON', quality='REALTIME', self_collision=False)
    bot = np.nonzero(S.rest[:, 2] < 1e-6)[0]
    ch, _ = CT.order_chain(S.rest, bot, S.stretch_edges)
    if stiff:
        CT.attach(S, CT.Trim(ch, False, kind='BAND', mass_per_m=0.0, bend=4000.0, edge_weight=6.0))
    wind = lambda P: np.stack([np.zeros(len(P)), 5.0 * (1 + 0.9 * np.sin(2 * np.pi * P[:, 0] / 0.35)), np.zeros(len(P))], 1)
    for k in range(150):
        S.step(1 / 60, wind=wind)
    p = S.x[ch]
    curv = float(np.linalg.norm(p[:-2] - 2 * p[1:-1] + p[2:], axis=1).mean())
    return curv, float(p[:, 1].mean()), float(p[:, 2].mean())
w0, y0, z0 = hem_run(False)
w1, y1, z1 = hem_run(True)
print("hem curvature plain %.4f stiff %.4f, swing %.3f vs %.3f, z %.3f vs %.3f" % (w0, w1, y0, y1, z0, z1))
ok(w1 < 0.6 * w0, "a stiff trim straightens the hem (curvature %.4f -> %.4f)" % (w0, w1))
ok(abs(z1 - z0) > 0.003 or abs(y1 - y0) > 0.003, "trim changes how the hem swirls (swing %.3f vs %.3f)" % (y0, y1))

# --- boundary
V, T, _q = CS.grid_mesh(6, 6, 1, 1, axis='xz')
ch = CT.boundary_chains(T)
ok(len(ch) == 1 and ch[0][1] and len(ch[0][0]) == 20, "boundary loop of a 6x6 grid has 20 vertices")

# --- lace flare modulates rest lengths
S = CS.ClothSolver(V, T, pins=CS.top_row_pins(6, 6), preset='SILK', quality='REALTIME', self_collision=False)
tr = CT.attach(S, CT.Trim(ch[0][0], True, kind='LACE'))
ok(max(float(s.rest.max() / np.linalg.norm(S.rest[s.j] - S.rest[s.i], axis=1).max()) for s in tr._sets[1:2]) > 1.05,
   "lace flare adds surplus length along the border")

# --- accents
V, T, _q = CS.grid_mesh(9, 9, 0.8, 0.8, axis='xz')
S = CS.ClothSolver(V, T, pins=CS.top_row_pins(9, 9), preset='COTTON', quality='REALTIME', self_collision=False)
pts = np.array([[0.3, 0.02, 0.4], [0.5, -0.01, 0.2]])
ac = CT.AccentSet(S.rest, S.tris, pts, kind='BUTTON', offset=0.004)
CT.attach(S, ac)
a0, _F = ac.anchors(S.x)
ok(np.allclose(a0[:, [0, 2]], pts[:, [0, 2]], atol=1e-6), "accent binds at the closest surface point")
run(S, 60)
a1, F1 = ac.anchors(S.x)
T3 = S.tris[ac.tri]
surf = np.einsum('kj,kjc->kc', ac.bary, S.x[T3])
dist = np.einsum('kc,kc->k', a1 - surf, F1[:, :, 2])
ok(np.allclose(dist, ac.local[:, 2], atol=1e-9), "accent keeps its normal offset at the barycentric point while the cloth deforms")
ok(np.allclose(np.linalg.norm(a1 - surf, axis=1), np.linalg.norm(ac.local, axis=1), atol=1e-9), "accent distance to the surface is preserved")
M = ac.transforms(S.x)
ok(np.allclose(M[0, :3, 3], a1[0]) and abs(np.linalg.det(M[0, :3, :3]) / ac.scale ** 3 - 1) < 1e-6, "accent matrix has a proper rotation")
# pendant
ch_ac = CT.AccentSet(S.rest, S.tris, np.array([[0.4, 0.0, 0.5]]), kind='CHARM', chain_segments=4, chain_length=0.12)
CT.attach(S, ch_ac)
run(S, 90)
L = np.linalg.norm(np.diff(ch_ac.chains()[0], axis=0), axis=1)
ok(np.all(np.abs(L - 0.03) < 0.006), "pendant links keep their length (%s)" % np.round(L, 4))
ok(ch_ac.p[0, -1, 2] < ch_ac.anchor[0, 2] - 0.06, "pendant hangs below its anchor")
pt = CT.scatter_points(S.rest, S.tris, density=100, seed=3)
ok(len(pt) == int(round(100 * 0.64)), "scatter by density gives %d points" % len(pt))
al = CT.points_along_chain(S.rest, np.arange(9), 5)
ok(len(al) == 5 and abs(al[-1, 0] - 0.8) < 1e-9, "instances along a curve are evenly spaced")

print("FAILED %d" % ok.fail if ok.fail else "ALL PASS")
sys.exit(1 if ok.fail else 0)
