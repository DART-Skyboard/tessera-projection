"""Pure-Python tests of the SDF spatial index, narrow-band bricks and the error-probe loop: python3 tests/test_sdf_index.py"""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'arc_flow'))
from core import sdf
from core import primitives as MS


def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        ok.fail += 1
ok.fail = 0


def test_bvh_matches_brute():
    rng = np.random.default_rng(5)
    for name, (v, t) in (('sphere', MS.uvsphere()), ('tube', MS.tube())):
        s = sdf.soup_from_mesh(v, t)
        s = sdf.virtual_subdivide(s, 0.12 if name == 'sphere' else 0.5, cap=3000)
        P = rng.uniform(v.min(0) - 0.6, v.max(0) + 0.6, (1500, 3))
        a = sdf.signed_distance_brute(s, P)
        b = sdf.signed_distance(s, P)
        ok(len(s) >= sdf.BVH_MIN_TRIS and np.abs(a - b).max() < 1e-8 and np.all(np.sign(a) == np.sign(b)),
           "BVH signed distance == brute force on %s (%d tris, max diff %.1e)" % (name, len(s), np.abs(a - b).max()))
        cp, d, ti = sdf.closest_surface(s, P[:200])
        ok(np.abs(np.abs(d) - np.linalg.norm(P[:200] - cp, axis=1)).max() < 1e-9, "closest_surface point is at |sd| from the query on %s" % name)


def test_open_shell_sign():
    v, t = MS.tube()                       # open ends: pseudo-normal sign must stay robust
    s = sdf.soup_from_mesh(v, t)
    P = np.array([[0, 0, 0.0], [3, 0, 0], [0, 0, 5.0]])
    a, b = sdf.signed_distance_brute(s, P), sdf.signed_distance(s, P)
    ok(np.allclose(a, b, atol=1e-9), "open shell sign identical (%s)" % np.round(b, 3))


def test_bricks_add_remove():
    v, t = MS.uvsphere()
    s = sdf.virtual_subdivide(sdf.soup_from_mesh(v, t), 0.2, cap=3000)
    h = 2.4 / 128
    g = sdf.BrickGrid(s, np.array([-1.2] * 3), h, np.array([129] * 3), band=3.0)
    n0 = len(g)
    rng = np.random.default_rng(2)
    P = rng.normal(size=(2000, 3)); P /= np.linalg.norm(P, axis=1, keepdims=True); P *= 1 + rng.uniform(-0.04, 0.04, (2000, 1))
    ex = sdf.signed_distance(s, P)
    ok(np.abs(g.sample(P) - ex).max() < 0.5 * h, "brick grid reproduces the field inside the band (err %.4f < %.4f)" % (np.abs(g.sample(P) - ex).max(), 0.5 * h))
    far = np.array([[0, 0, 0.0], [1.1, 1.1, 1.1]])
    sg = g.sample(far)
    ok(sg[0] < 0 < sg[1], "background cells keep the correct sign")
    ok(g.nbytes() < 0.3 * g.dense_nbytes(), "narrow band stores %.1f%% of a dense float64 grid (%d bricks)" % (100.0 * g.nbytes() / g.dense_nbytes(), n0))
    ids = g.keys[:10].copy()
    ok(g.remove_bricks(ids) == 10 and len(g) == n0 - 10, "bricks removed")
    ok(g.add_bricks(ids) == 10 and len(g) == n0 and np.abs(g.sample(P) - ex).max() < 0.5 * h, "bricks added back (slots recycled, pool %d)" % len(g.pool))
    ok(g.trim(n0 // 2) > 0 and len(g) == n0 // 2, "trim evicts the oldest bricks")


def test_partfield_sparse_refine():
    v, t = MS.uvsphere()
    f = sdf.PartField(v, t, 'AUTO', 24, 2)
    P = np.random.default_rng(1).normal(size=(400, 3)); P /= np.linalg.norm(P, axis=1, keepdims=True)
    n = f.request_refine(P * 1.01, max_new=64)
    ok(n > 0 and f.n_refined == n and not hasattr(f, 'brick_index'), "sparse brick store: %d refined, no dense cell index" % n)
    before = f.sample_sd(P * 1.01).copy()
    k = f.remove_bricks(f._bkey[:n // 2])
    ok(k == n // 2 and f.n_refined == n - k, "PartField.remove_bricks coarsens (%d removed)" % k)
    ok(np.abs(f.sample_sd(P * 1.01) - before).max() < 0.05, "field still continuous after coarsening")
    kept = f.coarsen(np.zeros((0, 3)))
    ok(f.n_refined == 0 and f.sample_sd(P).shape == (400,), "coarsen with no particles frees every brick (%d)" % kept)


def test_adaptive_probe():
    # the website's thresholds (refine above 0.75 voxel, coarsen below 0.15) rarely trigger refinement because the field is exact
    # near the surface; a stricter P90_HI exercises the refine path, a smooth sphere exercises coarsening
    v, t = MS.tube()
    old = sdf.PartField.P90_HI
    sdf.PartField.P90_HI = 0.2
    try:
        a = sdf.PartField(v, t, 'AUTO', 12, 2, adaptive=True)
    finally:
        sdf.PartField.P90_HI = old
    ok(a.resolution_chosen > 12 and a.resolution_chosen <= 24 and len(a.probe_history) > 1,
       "adaptive refine: tube 12 -> %d (history %s)" % (a.resolution_chosen, [(r, round(e, 2)) for r, e in a.probe_history]))
    v, t = MS.uvsphere()
    b = sdf.PartField(v, t, 'AUTO', 48, 2, adaptive=True)
    ok(0.6 * 48 - 1 <= b.resolution_chosen < 48 and b.probe_p90 < 0.75,
       "adaptive coarsen within bounds: sphere 48 -> %d (p90 %.2f voxel)" % (b.resolution_chosen, b.probe_p90))
    off = sdf.PartField(v, t, 'AUTO', 24, 2)
    ok(off.resolution_chosen == 24 and off.probe_p90 is None, "adaptive off: one build at the requested resolution")
    ok(np.isfinite(off.probe_error()), "probe_error callable on demand (p90 %.3f voxel)" % off.probe_error())


if __name__ == '__main__':
    t0 = time.time()
    for fn in (test_bvh_matches_brute, test_open_shell_sign, test_bricks_add_remove, test_partfield_sparse_refine, test_adaptive_probe):
        fn()
    print("%d failures (%.1fs)" % (ok.fail, time.time() - t0))
    sys.exit(1 if ok.fail else 0)
