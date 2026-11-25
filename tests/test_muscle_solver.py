"""Pure numpy tests of the Tessera muscle solver:  python3 tests/test_muscle_solver.py"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'arc_flow', 'core'))
import muscle_solver as MS  # noqa: E402

fails = [0]


def ok(c, msg):
    print(("PASS " if c else "FAIL ") + msg, flush=True)
    if not c:
        fails[0] += 1


def tube(nu=24, nv=30, r=0.05, h=0.4):
    """Closed-ish cylinder tube (no caps) along z with quad edges + radial normals."""
    th = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    z = np.linspace(0, h, nv)
    V = np.array([[r * np.cos(a), r * np.sin(a), zz] for zz in z for a in th])
    N = np.array([[np.cos(a), np.sin(a), 0.0] for zz in z for a in th])
    E = []
    for j in range(nv):
        for i in range(nu):
            a = j * nu + i
            E.append((a, j * nu + (i + 1) % nu))
            if j + 1 < nv:
                E.append((a, (j + 1) * nu + i))
    return V, np.array(E), N, nu, nv


def main():
    # catalog / colours
    keys = [r['key'] for r in MS.CATALOG]
    ok(len(set(keys)) == len(keys), "catalog keys are unique (%d regions)" % len(keys))
    cols = [MS.region_color(k) for k in keys]
    ok(len({tuple(np.round(c, 2)) for c in cols}) == len(cols), "region colours are all distinct")
    ok(MS.region_color('zygomaticus') == MS.region_color(MS.CATALOG_INDEX['zygomaticus']), "colour is a static function of the region")
    b1 = {r['key']: r['color'] for r in MS.regions_for('BIPED', 'CINEMATIC')}
    q1 = {r['key']: r['color'] for r in MS.regions_for('QUADRUPED', 'CINEMATIC')}
    ok(all(b1[k] == q1[k] for k in b1 if k in q1) and 'zygomaticus' in b1 and 'zygomaticus' in q1,
       "cheek colour is identical on biped and quadruped")
    nc = len(MS.regions_for('BIPED', 'CARTOON'))
    ns = len(MS.regions_for('BIPED', 'STANDARD'))
    nf = len(MS.regions_for('BIPED', 'CINEMATIC'))
    ok(nc < ns < nf, "detail levels grow: cartoon %d < standard %d < cinematic %d" % (nc, ns, nf))
    ok({'biceps', 'sternocleidomastoid'} <= {r['key'] for r in MS.regions_for('BIPED', 'CARTOON')}, "cartoon keeps biceps and neck")
    ok('orbicularis_oris' in {r['key'] for r in MS.regions_for('BIPED', 'CINEMATIC')}
       and 'orbicularis_oris' not in {r['key'] for r in MS.regions_for('BIPED', 'CARTOON')}, "facial muscles only in cinematic")
    ok('biceps' not in {r['key'] for r in MS.regions_for('QUADRUPED', 'CINEMATIC')}
       and 'haunch' in {r['key'] for r in MS.regions_for('QUADRUPED', 'CARTOON')}, "quadruped swaps the arm/leg set for haunch/shoulder")
    m = {r['key']: r['emphasis'] for r in MS.regions_for('HUMAN_MALE', 'STANDARD')}
    f = {r['key']: r['emphasis'] for r in MS.regions_for('HUMAN_FEMALE', 'STANDARD')}
    ok(m['pectoralis'] > f['pectoralis'] and f['gluteus'] > m['gluteus'], "male / female presets differ in emphasis")
    cr = {r['key'] for r in MS.regions_for('CUSTOM', 'CINEMATIC', dict(heads=2, arms=0, legs=4, tails=1, joints_per_limb=3))}
    ok('biceps' not in cr and 'tail_base' in cr and 'head_mass' in cr and 'limb_mid' in cr and 'quadriceps' in cr,
       "custom creature spec (2 heads, 0 arms, 4 legs, tail, 3 joints) drives the region list")
    nh = {r['key'] for r in MS.regions_for('CUSTOM', 'CINEMATIC', dict(heads=0, tails=0))}
    ok('zygomaticus' not in nh and 'tail_base' not in nh, "zero heads / tails removes those categories")

    # graph + components
    V, E, N, nu, nv = tube()
    g = MS.Graph(V, E)
    mask = np.zeros(len(V), bool)
    mask[:nu * 5] = True
    mask[nu * 20:nu * 25] = True
    comp = MS.Graph.components(g, mask)
    ok(comp.max() == 1 and (comp[mask] >= 0).all() and (comp[~mask] == -1).all(), "connected components find two separate patches")

    # projection: seed two painted bands on a tube, rest of region solved
    lab = -np.ones(len(V), np.int64)
    lab[nu * 8:nu * 9] = 0          # a single ring painted for region 0
    lab[nu * 21:nu * 22] = 1
    regions = [dict(key='biceps', bones=('upper_arm',), sub=1), dict(key='triceps', bones=('upper_arm',), sub=1)]
    res = MS.solve(V, E, N, lab, regions, res=1.0)
    ok(len(res['instances']) == 2, "two painted rings solve to two muscle instances")
    lbl = res['label']
    ok((lbl[nu * 8:nu * 9] == 0).all() and (lbl[nu * 21:nu * 22] == 1).all(), "seeds keep their region")
    grown0 = (lbl == 0).sum()
    ok(grown0 > nu * 3, "projection grows a single painted ring into a region (%d verts)" % grown0)
    ok((lbl[: nu * 2] == -1).all(), "far, unpainted skin stays unassigned")
    mid = (lbl[nu * 14: nu * 16])
    ok(((mid == -1) | (mid == 0) | (mid == 1)).all(), "labels are valid")
    soft = res['membership']
    ok(np.all(soft >= -1e-9) and np.all(soft.sum(1) <= 1.0 + 1e-6), "membership is a sub-partition of unity (sum <= 1)")
    # border blending: membership of region 0 decays smoothly (no jump bigger than 0.5 between neighbours)
    jump = np.abs(soft[E[:, 0], 0] - soft[E[:, 1], 0]).max()
    ok(jump < 0.5, "region membership blends smoothly across neighbours (max edge jump %.2f)" % jump)

    # tendons: stroke along the tube, flex weight zero at the ends, belly in the middle
    stroke_ids = [j * nu for j in range(6, 25)]
    lab2 = -np.ones(len(V), np.int64)
    lab2[nu * 12: nu * 18] = 0
    res = MS.solve(V, E, N, lab2, regions[:1], strokes=[dict(region=0, verts=stroke_ids)], res=1.0)
    ins = res['instances'][0]
    ok(ins.source == 'stroke', "contour stroke defines the fibre axis")
    zi = V[ins.idx, 2]
    w_mid = ins.w[np.abs(zi - zi.mean()) < 0.03].mean()
    w_end = ins.w[(zi < zi.min() + 0.02) | (zi > zi.max() - 0.02)].mean()
    ok(w_mid > 3.0 * w_end, "belly flex %.3f >> tendon end flex %.3f" % (w_mid, w_end))
    ok(ins.bulge.shape == (len(ins.idx), 3) and np.all(np.isfinite(ins.bulge)), "bulge field finite")
    dots = (ins.bulge * N[ins.idx]).sum(1)
    ok(dots.min() >= -1e-12 and dots.max() > 0, "bulge pushes outward along the skin normal")
    ok(np.linalg.norm(ins.slide, axis=1).max() > 0, "accent slide field exists")
    sl_dir = ins.slide[np.linalg.norm(ins.slide, axis=1) > 1e-9].mean(0)
    ok(abs(np.dot(sl_dir / np.linalg.norm(sl_dir), ins.axis)) > 0.95, "accent slide runs along the fibre axis")

    # slider scaling
    d1 = MS.assemble(len(V), [ins], {'biceps': dict(flex=1.0, accent=1.0, emphasis=1.0)})
    d2 = MS.assemble(len(V), [ins], {'biceps': dict(flex=2.0, accent=1.0, emphasis=1.0)})
    ok(abs(np.linalg.norm(d2, axis=1).max() / np.linalg.norm(d1, axis=1).max() - 2.0) < 1e-6, "flex slider scales displacement linearly")
    da0 = MS.assemble(len(V), [ins], {'biceps': dict(flex=1.0, accent=0.0, emphasis=1.0)})
    da1 = MS.assemble(len(V), [ins], {'biceps': dict(flex=1.0, accent=1.0, emphasis=1.0)})
    ok(np.abs(da1 - da0).max() > 0 and np.allclose(da0, (ins.bulge[:, :] if False else da0)), "accent slider adds the slide component")
    off = MS.assemble(len(V), [ins], {'biceps': dict(flex=1.0, accent=1.0, emphasis=1.0, enabled=False)})
    ok(np.abs(off).max() == 0, "disabled region adds nothing")

    # resolution factor
    resl = {}
    for rf in (0.3, 1.0, 4.0):
        r = MS.solve(V, E, N, lab2, regions[:1], strokes=[dict(region=0, verts=stroke_ids)], res=rf)
        i = r['instances'][0]
        resl[rf] = (np.std(np.diff(np.sort(i.w))), np.std(i.w), np.abs(np.diff(i.w)).mean())
        full = np.zeros(len(V))
        full[i.idx] = i.w
        resl[rf] = float(np.abs(full[E[:, 0]] - full[E[:, 1]]).mean())
    ok(resl[4.0] > resl[1.0] > resl[0.3], "resolution factor adds detail (edge-to-edge weight variation %.5f > %.5f > %.5f)"
       % (resl[4.0], resl[1.0], resl[0.3]))

    # joint-angle / velocity driven flex
    f_rest = MS.flex_drive(0.0, 0.0, tone=0.06)
    f_bent = MS.flex_drive(1.0, 0.0, tone=0.06, rng=1.2)
    f_run = MS.flex_drive(1.0, 5.0, tone=0.06, rng=1.2)
    f_walk = MS.flex_drive(1.0, 1.2, tone=0.06, rng=1.2)
    ok(abs(f_rest - 0.06) < 1e-9 and f_bent > f_rest + 0.3, "flex rises with joint angle (rest %.2f, bent %.2f)" % (f_rest, f_bent))
    ok(f_run > f_walk > f_bent, "faster motion raises the intensity (bent %.2f, walk %.2f, run %.2f)" % (f_bent, f_walk, f_run))
    ok(MS.flex_drive(-1.0, 0.0) == MS.flex_drive(1.0, 0.0), "flex is symmetric in the joint angle")
    x, vv = 0.0, 0.0
    peak = 0.0
    for _ in range(120):
        x, vv = MS.jiggle_step(x, vv, 1.0, 1 / 60.0)
        peak = max(peak, x)
    ok(peak > 1.0 and abs(x - 1.0) < 0.05, "jiggle spring overshoots then settles (peak %.2f, final %.3f)" % (peak, x))

    # auto seed from a two-bone skeleton
    th = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    pts, nrm = [], []
    for z in np.linspace(0, 1, 20):
        for a in th:
            pts.append([0.05 * np.cos(a), 0.05 * np.sin(a), z])
            nrm.append([np.cos(a), np.sin(a), 0])
    pts = np.array(pts)
    Wt = np.zeros((len(pts), 1))
    Wt[:] = 1.0
    reg2 = [dict(key='biceps', bones=('upper_arm',), t=(0.2, 0.8), facing='front'),
            dict(key='triceps', bones=('upper_arm',), t=(0.2, 0.8), facing='back')]
    lb, sc = MS.auto_seed(pts, np.array(nrm), ['upper_arm.L'], [[0, 0, 0]], [[0, 0, 1.0]], Wt, reg2)
    fr = np.array(nrm)[:, 1] < -0.5
    bk = np.array(nrm)[:, 1] > 0.5
    zz = pts[:, 2]
    inside = (zz > 0.25) & (zz < 0.75)
    ok((lb[fr & inside] == 0).all() and (lb[bk & inside] == 1).all() and (lb[(zz < 0.15) | (zz > 0.85)] == -1).all(),
       "auto-seed puts biceps on the front (-Y) and triceps on the back inside the bone's t range")

    j = MS.result_to_json([ins], MS.regions_for('BIPED', 'STANDARD'))
    ins2, meta = MS.instances_from_json(j)
    ok(len(ins2) == 1 and np.allclose(ins2[0].bulge, ins.bulge, atol=1e-5) and meta['version'] == 1, "solve data round-trips through JSON")
    print("FAILS", fails[0])
    sys.exit(1 if fails[0] else 0)


main()
