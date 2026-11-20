"""Fluid mode solver tests (pure NumPy): python3 blender/tests/test_fluidmode.py"""
import sys
import time
import numpy as np
from fm_helpers import FM, SDF, box, container, lathe, body_of
from core import primitives as MS
from core import atmosphere as ATM

fails = [0]


def ok(c, msg):
    print(("PASS " if c else "FAIL ") + msg, flush=True)
    if not c:
        fails[0] += 1


def run(S, seconds, dt=1.0 / 24.0, cb=None):
    n = int(round(seconds / dt))
    for k in range(n):
        if cb:
            cb(k, S)
        S.step(dt)
    return S


def lattice_in(lo, hi, dx):
    I = FM.lattice_in_box(lo, hi, dx)
    return FM.lat_pos(I, dx)


def sphere_pts(c, r, dx):
    P = lattice_in(np.array(c) - r, np.array(c) + r, dx)
    return P[np.linalg.norm(P - np.asarray(c), axis=1) < r]


def test_units():
    d = FM.derive_properties(mass=2.0, volume=0.002)
    ok(abs(d['density'] - 1000.0) < 1e-9 and abs(d['weight'] - 2.0 * 9.80665) < 1e-9, "mass + volume -> density, weight = m g")
    d = FM.derive_properties(density=800.0, volume=0.5)
    ok(abs(d['mass'] - 400.0) < 1e-9, "density + volume -> mass")
    d = FM.derive_properties(density=800.0, mass=400.0)
    ok(abs(d['volume'] - 0.5) < 1e-12, "density + mass -> volume")
    d = FM.derive_properties(weight=98.0665, volume=0.1)
    ok(abs(d['mass'] - 10.0) < 1e-9 and abs(d['density'] - 100.0) < 1e-9, "weight -> mass (g) -> density")
    d = FM.derive_properties(weight=24.79 * 5.0, volume=1.0, g=24.79)
    ok(abs(d['mass'] - 5.0) < 1e-9, "weight uses the environment gravity")


def test_environment():
    e = FM.FEnv()
    H = e.scale_height()
    z = np.array([0.0, 1000.0, 5000.0, 8000.0])
    P = e.pressure_at(z)
    exp = 101325.0 * np.exp(-z * e.molar * 9.80665 / (FM.R_GAS * 288.15))
    ok(np.allclose(P, exp, rtol=1e-9) and np.all(np.diff(P) < 0), "ambient pressure follows the barometric formula (decreases with altitude)")
    ok(7000 < H < 9000, "Earth scale height about 8.4 km (%.0f m)" % H)
    es = FM.FEnv.from_atmosphere(ATM.preset_state('EARTH'))
    ok(abs(es.density_at(0.0) - ATM.preset_state('EARTH')['rho']) < 1e-6, "ambient density agrees with the Environment card atmosphere data")


def test_boussinesq():
    b = container((-.08, -.08, 0), (.08, .08, .2))
    P = sphere_pts((0, 0, 0.1), 0.03, 0.008)
    res = {}
    for name, dT in (('hot', +40.0), ('cold', -40.0)):
        car = FM.fluid_def('WATER', name='carrier', body=0, fill=0.85, beta=2e-3, k=0.01)
        blob = FM.fluid_def('WATER', name='blob', body=0, shape=dict(kind='POINTS', pts=P), temperature=293.15 + dT, tref=293.15, beta=2e-3, k=0.01, miscible=False, group=1)
        S = FM.FluidSolver.build([blob, car], [b], FM.FEnv(), dx=0.008)
        z0 = S.x[S.lo[0]:S.hi[0], 2].mean()
        run(S, 1.0)
        res[name] = S.x[S.lo[0]:S.hi[0], 2].mean() - z0
    ok(res['hot'] > 0.01, "hot (lighter, Boussinesq) blob rises in a carrier (+%.3f m)" % res['hot'])
    ok(res['cold'] < -0.005 or res['cold'] < res['hot'] - 0.02, "cold (denser) blob sinks relative to the hot one (%.3f m)" % res['cold'])


def test_stratify_and_mass():
    b = container((-.07, -.07, 0), (.07, .07, .2))
    dx = 0.008
    P = FM.lat_pos(FM.lattice_in_box((-.06, -.06, 0.004), (.06, .06, .11), dx), dx)
    # inverted start: water (heavy) on top of oil with a perturbed interface (Rayleigh-Taylor); a grid pressure solve cannot
    # separate fluids that are mixed below the cell size, so the start is layered, like a poured test
    zb = 0.055 + 0.008 * np.cos(2 * np.pi * P[:, 0] / 0.07) * np.cos(2 * np.pi * P[:, 1] / 0.07)
    sel = P[:, 2] > zb
    water = FM.fluid_def('WATER', name='water', body=0, shape=dict(kind='POINTS', pts=P[sel]), miscible=False)
    oil = FM.fluid_def('MINERAL_OIL', name='oil', body=0, shape=dict(kind='POINTS', pts=P[~sel]), miscible=False, group=7)
    S = FM.FluidSolver.build([water, oil], [b], FM.FEnv(), dx=dx)
    m0 = [t['mass'] for t in S.totals()]
    n0 = S.n
    zw0, zo0 = S.x[S.lo[0]:S.hi[0], 2].mean(), S.x[S.lo[1]:S.hi[1], 2].mean()
    run(S, 3.0)
    zw, zo = S.x[S.lo[0]:S.hi[0], 2].mean(), S.x[S.lo[1]:S.hi[1], 2].mean()
    ok(zo0 - zw0 < -0.03 and zo - zw > 0.02, "density stratification: inverted water-over-oil start swaps (oil above water: dz %.3f -> %.3f m)" % (zo0 - zw0, zo - zw))
    zl = np.sort(S.x[S.lo[0]:S.hi[0], 2]); zu = np.sort(S.x[S.lo[1]:S.hi[1], 2])
    ok(np.percentile(zl, 25) < np.percentile(zu, 25), "water layer sits below the oil layer (lower quartiles)")
    m1 = [t['mass'] for t in S.totals()]
    ok(S.n == n0 and np.allclose(m0, m1) and np.isfinite(S.x).all(), "mass conserved (%.3f kg water, %.3f kg oil), no particle lost" % (m1[0], m1[1]))
    d = b.distance(S.x, np.eye(4))
    ok(d.min() > -0.5 * dx, "all particles stay inside the container (min wall distance %.4f m)" % d.min())


def test_gas_wind():
    dx = 0.02
    P = sphere_pts((0, 0, 1.0), 0.15, dx)
    env = FM.FEnv(wind=(2.0, 0.0, 0.0), temperature=293.15)
    g = FM.fluid_def('AIR', name='air', shape=dict(kind='POINTS', pts=P), volume=len(P) * dx ** 3, count=len(P), tref=293.15)
    S = FM.FluidSolver.build([g], [], env, dx=dx)
    x0 = S.x[:, 0].mean(); z0 = S.x[:, 2].mean()
    run(S, 4.0)
    dxm, dzm = S.x[:, 0].mean() - x0, S.x[:, 2].mean() - z0
    tau = 1.0 / 0.6
    expect = 2.0 * (4.0 - tau * (1 - np.exp(-4.0 / tau)))
    ok(abs(dxm - expect) < 0.2 * expect, "free gas cloud drifts with the wind (%.2f m, drag model predicts %.2f m)" % (dxm, expect))
    ok(abs(dzm) < 0.15, "neutrally buoyant air in air does not rise (dz %.3f m)" % dzm)


def gas_dz(fluid, env, seconds=2.0, z=100.0, ground=None):
    dx = 0.02
    P = sphere_pts((0, 0, z), 0.12, dx)
    g = FM.fluid_def(fluid, name=fluid, shape=dict(kind='POINTS', pts=P), volume=len(P) * dx ** 3, count=len(P), drag=0.05, diffusion=0.0, tref=293.15, temperature=293.15)
    S = FM.FluidSolver.build([g], [], env, dx=dx)
    z0 = S.x[:, 2].mean()
    run(S, seconds)
    return S.x[:, 2].mean() - z0


def test_gas_buoyancy():
    earth = FM.FEnv(temperature=293.15)
    up = gas_dz('HELIUM', earth)
    ok(up > 3.0, "helium rises in Earth air (+%.1f m in 2 s)" % up)
    he_env = FM.FEnv(temperature=293.15, molar=0.004003)
    flat = gas_dz('HELIUM', he_env)
    ok(abs(flat) < 0.3, "helium does not rise in a helium atmosphere (dz %.2f m)" % flat)
    down = gas_dz('CO2', earth)
    ok(down < -2.5, "CO2 (heavier than air) sinks (%.1f m in 2 s)" % down)
    air_up = gas_dz('AIR', earth)
    ok(abs(air_up) < 0.6, "air in air floats (dz %.2f m)" % air_up)


def test_gravity_scaling():
    dx = 0.01
    P = sphere_pts((0, 0, 5.0), 0.04, dx)
    out = {}
    for name in ('EARTH', 'MARS'):
        st = ATM.preset_state(name)
        env = FM.FEnv.from_atmosphere(st)
        d = FM.fluid_def('WATER', name='drop', shape=dict(kind='POINTS', pts=P), body=-1)
        S = FM.FluidSolver.build([d], [], env, dx=dx)
        z0 = S.x[:, 2].mean()
        run(S, 0.8, dt=1.0 / 48.0)
        out[name] = z0 - S.x[:, 2].mean()
    ratio = out['EARTH'] / out['MARS']
    expect = 9.80665 / 3.721
    ok(abs(ratio - expect) < 0.06 * expect, "free fall distance Earth/Mars = %.3f (expected g ratio %.3f)" % (ratio, expect))


def test_heat():
    dx = 0.01
    b = container((-.06, -.06, 0), (.06, .06, .15))
    w = FM.fluid_def('WATER', name='w', body=0, fill=0.8, k=0.6)
    S = FM.FluidSolver.build([w], [b], FM.FEnv(temperature=293.15), dx=dx, thermal_speedup=200.0)
    t0 = S.T.mean()
    run(S, 2.0, cb=lambda k, s: setattr(s, 'sources', [((0.0, 0.0, 0.02), 400.0, 0.05)]))
    t1 = S.T.mean()
    ok(t1 > t0 + 0.3, "heat source warms the fluid (%.2f -> %.2f K)" % (t0, t1))
    hot = S.T.max()
    S.sources = []
    run(S, 6.0)
    ok(S.T.max() < hot and S.T.mean() < t1 + 0.1 + 0.0, "after the source is removed the fluid cools / equalises (max %.2f -> %.2f K)" % (hot, S.T.max()))
    # conduction conserves energy between a hot and a cold half (no ambient exchange)
    w2 = FM.fluid_def('WATER', name='w2', body=0, fill=0.8, cool=1e-9)
    S2 = FM.FluidSolver.build([w2], [b], FM.FEnv(), dx=dx, thermal_speedup=500.0)
    S2.T[:] = np.where(S2.x[:, 0] > 0, 320.0, 290.0)
    e0 = (S2.mass_ref * S2.cp[0] * S2.T).sum()
    run(S2, 3.0)
    e1 = (S2.mass_ref * S2.cp[0] * S2.T).sum()
    ok(abs(e1 - e0) / e0 < 2e-3 and np.ptp(S2.T) < 30.0, "conduction conserves thermal energy (%.2e rel) and evens out (range %.1f K)" % (abs(e1 - e0) / e0, np.ptp(S2.T)))


def test_moving_wall():
    dx = 0.01
    b = container((-.1, -.06, 0), (.1, .06, .15))
    w = FM.fluid_def('WATER', name='w', body=0, fill=0.5)
    S = FM.FluidSolver.build([w], [b], FM.FEnv(), dx=dx)
    run(S, 0.6)
    x0 = S.x[:, 0].mean()
    nfr = 24
    for k in range(nfr):
        M = np.eye(4); M[0, 3] = 0.12 * (k + 1) / nfr
        b.set_matrix(M)
        S.step(1.0 / 24.0)
    dxm = S.x[:, 0].mean() - x0
    ok(dxm > 0.07, "moving container wall pushes the fluid along (mean x +%.3f m for a 0.12 m move)" % dxm)
    d = b.distance(S.x, b.M1)
    ok(d.min() > -0.5 * dx, "fluid stays inside the moving container (%.4f)" % d.min())


def test_solid_displaces():
    dx = 0.01
    b = container((-.08, -.08, 0), (.08, .08, .25))
    w = FM.fluid_def('WATER', name='w', body=0, fill=0.45)
    S = FM.FluidSolver.build([w], [b], FM.FEnv(), dx=dx)
    run(S, 0.6)
    top0 = np.percentile(S.x[:, 2], 98)
    V, T = MS.uvsphere(24, 12, 0.045)
    col = FM.Body(SDF.PartField(V, T, 'AUTO', resolution=32, level=2), np.eye(4), 'COLLIDER', None, 0.0)
    S.bodies.append(col)
    z_start, z_end = 0.22, 0.045
    nfr = 24
    for k in range(nfr):
        M = np.eye(4); M[2, 3] = z_start + (z_end - z_start) * (k + 1) / nfr
        col.set_matrix(M)
        S.step(1.0 / 24.0)
    run(S, 0.5)
    dsolid = col.distance(S.x, col.M1)
    top1 = np.percentile(S.x[:, 2], 98)
    ok(dsolid.min() > -0.7 * dx, "no fluid inside the dropped solid (min distance %.4f m)" % dsolid.min())
    ok(top1 > top0 + 0.003, "the solid displaces fluid: surface rises %.3f -> %.3f m" % (top0, top1))


def gear(n_teeth=10, r0=0.1, r1=0.12, h=0.03):
    ang = np.linspace(0, 2 * np.pi, 4 * n_teeth, endpoint=False)
    rr = np.tile([r0, r1, r1, r0], n_teeth)
    pts = np.column_stack([rr * np.cos(ang), rr * np.sin(ang)])
    n = len(pts)
    V = np.vstack([np.column_stack([pts, np.zeros(n)]), np.column_stack([pts, np.full(n, h)]), [[0, 0, 0], [0, 0, h]]])
    T = []
    for i in range(n):
        j = (i + 1) % n
        T += [(i, j, n + j), (i, n + j, n + i), (2 * n, j, i), (2 * n + 1, n + i, n + j)]
    return V, np.array(T)


def spin_test(omega):
    V, T = gear()
    body = FM.Body(SDF.PartField(V, T, 'AUTO', resolution=48, level=2), np.eye(4), 'COLLIDER', None, 1.0)
    dx = 0.01
    P = FM.lat_pos(FM.lattice_in_box((0.05, -0.02, 0.05), (0.09, 0.02, 0.08), dx), dx)
    d = FM.fluid_def('WATER', name='drop', shape=dict(kind='POINTS', pts=P), body=-1, drag=0.0, cohesion=0.0)
    S = FM.FluidSolver.build([d], [body], FM.FEnv(), dx=dx)
    th = 0.0
    dt = 1.0 / 48.0
    vts = []
    for k in range(48):
        th += omega * dt
        M = np.eye(4); c, s = np.cos(th), np.sin(th)
        M[:2, :2] = [[c, -s], [s, c]]
        body.set_matrix(M)
        S.step(dt)
        if k > 24:
            r = S.x.copy(); r[:, 2] = 0
            rn = np.linalg.norm(r, axis=1)
            tang = np.column_stack([-r[:, 1], r[:, 0], np.zeros(len(r))]) / np.maximum(rn, 1e-9)[:, None]
            low = S.x[:, 2] < 0.05
            if low.any():
                vts.append(float(np.mean(np.einsum('ij,ij->i', S.v[low], tang[low]))))
    r_mean = float(np.mean(np.linalg.norm(S.x[:, :2], axis=1)))
    return float(np.mean(vts)) if vts else 0.0, r_mean


def test_spinning_gear():
    om = 2.0
    vt, r = spin_test(om)
    ok(vt > 0.5 * om * r and vt < 1.3 * om * r, "fluid on a spinning gear gets tangential momentum in the spin direction (%.3f m/s, omega r = %.3f)" % (vt, om * r))
    vt0, _r = spin_test(0.0)
    ok(abs(vt0) < 0.03, "a stationary gear gives no tangential momentum (%.3f m/s)" % vt0)
    vt2, _r = spin_test(-om)
    ok(vt2 < -0.5 * om * r, "reverse spin gives reverse momentum (%.3f m/s)" % vt2)


def test_barometric_expansion():
    dx = 0.05
    env = FM.FEnv()
    rs = {}
    for z in (0.0, 5000.0):
        P = sphere_pts((0, 0, z), 0.2, dx)
        g = FM.fluid_def('AIR', name='a', shape=dict(kind='POINTS', pts=P), volume=len(P) * dx ** 3, count=len(P))
        S = FM.FluidSolver.build([g], [], env, dx=dx)
        rs[z] = float(S.parcel_radius().mean())
    expect = (1.0 / np.exp(-5000.0 / env.scale_height())) ** (1.0 / 3.0)
    ok(abs(rs[5000.0] / rs[0.0] - expect) < 0.03 * expect, "free gas parcels expand with altitude (radius ratio %.3f, barometric %.3f)" % (rs[5000.0] / rs[0.0], expect))


CONE = [(0, 0), (0.075, 0), (0.04, 0.36), (0, 0.36)]
VASE = [(0, 0), (0.05, 0), (0.07, 0.06), (0.045, 0.13), (0.025, 0.2), (0.025, 0.26), (0.045, 0.3), (0, 0.3)]   # bottle: bulge, neck, flare
SPHERE = [(0, -0.1)] + [(0.1 * np.sin(a), -0.1 * np.cos(a)) for a in np.linspace(0.15, np.pi - 0.15, 15)] + [(0, 0.1)]


def radius_at(profile, z):
    p = np.asarray(profile, float)
    return np.interp(z, p[:, 1], p[:, 0])


def test_cavity_fill():
    # cone and non-convex vase: the fill is the cavity's real contoured shape (sampled through the collision-body projection)
    for name, prof, level, dx in (("cone", CONE, 0.5, 0.008), ("vase", VASE, 0.5, 0.008)):
        V, T = lathe(prof)
        b = body_of(V, T)
        cav = FM.mesh_volume(V, T)
        w = FM.fluid_def('WATER', name='w', body=0, volume=level * cav)
        S = FM.FluidSolver.build([w], [b], FM.FEnv(), dx=dx)
        vol = S.n * S.dx ** 3
        ok(abs(vol / (level * cav) - 1.0) < 0.06, "%s: fill volume %.3f L vs requested %.3f L (%.1f%%)" % (name, vol * 1e3, level * cav * 1e3, 100 * (vol / (level * cav) - 1)))
        ok(b.distance(S.x, np.eye(4)).min() > 0.0 and S.outside_count() == 0, "%s: no seeded particle outside the cavity" % name)
        r = np.hypot(S.x[:, 0], S.x[:, 1])
        bad = 0
        for z0 in np.linspace(0.02, S.x[:, 2].max() - 0.02, 6):
            m = np.abs(S.x[:, 2] - z0) < 0.5 * S.dx
            if m.any():
                bad += abs(r[m].max() - radius_at(prof, z0)) > 1.6 * S.dx
        ok(bad == 0, "%s: seeded cross-section follows the cavity contour at 6 levels" % name)
        if name == "cone":
            lo = np.abs(S.x[:, 2] - 0.03) < S.dx
            hi = np.abs(S.x[:, 2] - (S.x[:, 2].max() - 0.03)) < S.dx
            zl, zh = S.x[:, 2][lo].mean(), S.x[:, 2][hi].mean()
            want = radius_at(prof, zl) - radius_at(prof, zh)
            ok(r[lo].max() - r[hi].max() > 0.6 * want, "cone: column is narrower at the top (radius %.3f m at z=%.2f, %.3f m at z=%.2f; the cavity narrows by %.3f)" % (r[lo].max(), zl, r[hi].max(), zh, want))
    # the level of a finite fill follows the contour: the free surface height matches the frustum volume at 3 fill levels
    V, T = lathe(CONE)
    b = body_of(V, T)
    for fr in (0.25, 0.4, 0.5):
        w = FM.fluid_def('WATER', name='w', body=0, volume=fr * FM.mesh_volume(V, T))
        S = FM.FluidSolver.build([w], [b], FM.FEnv(), dx=0.008)
        zs = np.linspace(0, 0.36, 3601)
        rr = radius_at(CONE, zs)
        vz = np.cumsum(np.pi * rr ** 2) * (zs[1] - zs[0])
        z_expect = zs[np.searchsorted(vz, fr * FM.mesh_volume(V, T))]
        d_top = S.x[:, 2].max() + 0.5 * S.dx - z_expect
        ok(abs(d_top) < 1.2 * S.dx, "cone fill %d%%: level-set seeded surface at %.3f m, exact level %.3f m" % (fr * 100, S.x[:, 2].max(), z_expect))


def test_confinement():
    import math
    rot = np.eye(4)
    a = math.radians(35.0)
    rot[:3, :3] = [[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]]
    rot[:3, 3] = (0.0, 0.0, 0.15)
    cyl = [(0, 0), (0.06, 0), (0.06, 0.25), (0, 0.25)]
    for name, prof, M in (("cone", CONE, None), ("sphere", SPHERE, np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.12], [0, 0, 0, 1.0]])), ("tilted cylinder", cyl, rot)):
        V, T = lathe(prof)
        b = body_of(V, T, matrix=M)
        w = FM.fluid_def('WATER', name='w', body=0, volume=0.45 * FM.mesh_volume(V, T) * (1.0 if M is None else abs(np.linalg.det(M[:3, :3]))))
        S = FM.FluidSolver.build([w], [b], FM.FEnv(), dx=0.01)
        run(S, 2.5)
        ok(S.outside_count() == 0 and np.isfinite(S.x).all(), "%s: zero particles outside the cavity after 60 steps (%d particles, %d escapes projected back)" % (name, S.n, S.escapes))


def test_fill_settle():
    from core import fillsettle as FS
    # settled level matches volume / cross-section (cone and cylinder), dynamic fill under gravity
    cyl = [(0, 0), (0.06, 0), (0.06, 0.25), (0, 0.25)]
    for name, prof, fr in (("cone", CONE, 0.5), ("cylinder", cyl, 0.5)):
        V, T = lathe(prof)
        b = body_of(V, T)
        w = FM.fluid_def('WATER', name='w', body=0, volume=fr * FM.mesh_volume(V, T))
        S = FM.FluidSolver.build([w], [b], FM.FEnv(), dx=0.008)
        t_s = time.time()
        rep = FS.settle(S, mode='LEVEL')
        t_lv = time.time() - t_s
        zs = np.linspace(0, prof[-1][1], 3601)
        vz = np.cumsum(np.pi * radius_at(prof, zs) ** 2) * (zs[1] - zs[0])
        z_expect = zs[np.searchsorted(vz, fr * FM.mesh_volume(V, T))]
        top = np.percentile(S.x[:, 2], 99) + 0.5 * S.dx
        ok(abs(top - z_expect) < 2.0 * S.dx, "%s: level-set fill + settle %.3f m vs volume/cross-section %.3f m (%d steps, %.1f s, %d particles)" % (name, top, z_expect, rep['steps'], t_lv, S.n))
        ok(S.outside_count() == 0, "%s: settled fill stays in the cavity" % name)
        if name == "cone":
            hs = FS.hydrostatic_report(S)
            ok(hs is not None and 0.75 < hs['ratio'] < 1.3, "cone hydrostatic pressure at the bottom %.0f Pa vs rho g h %.0f Pa (ratio %.2f)" % (hs['measured'], hs['expected'], hs['ratio']))
            # determinism: the same build and settle gives the same level
            S2 = FM.FluidSolver.build([w], [b], FM.FEnv(), dx=0.008)
            Sp = FM.FluidSolver.build([w], [b], FM.FEnv(), dx=0.008, fill_seed='POUR')
            t_p = time.time(); rp = FS.settle(Sp, max_steps=120, mode='POUR'); t_p = time.time() - t_p
            ok(abs(np.percentile(Sp.x[:, 2], 99) + 0.5 * Sp.dx - z_expect) < 2.0 * S.dx, "pour fill (full dynamic): level %.3f m in %d steps, %.1f s (level-set fill: %d steps, %.1f s)" % (np.percentile(Sp.x[:, 2], 99) + 0.5 * Sp.dx, rp['steps'], t_p, rep['steps'], t_lv))
            FS.settle(S2, mode='LEVEL')
            ok(abs(np.percentile(S2.x[:, 2], 99) - np.percentile(S.x[:, 2], 99)) < 1e-6, "fill / settle is deterministic")
    # quick fill keeps the level-set seeding
    V, T = lathe(CONE)
    b = body_of(V, T)
    w = FM.fluid_def('WATER', name='w', body=0, volume=0.5 * FM.mesh_volume(V, T))
    S = FM.FluidSolver.build([w], [b], FM.FEnv(), dx=0.008)
    rep = FS.settle(S, mode='QUICK')
    ok(rep['steps'] == 3 and S.outside_count() == 0, "quick fill: level-set seeding plus %d relaxing steps" % rep['steps'])
    # overflow: sealed and open vessels
    for opened in (False, True):
        b = body_of(V, T)
        b.open = opened
        w = FM.fluid_def('WATER', name='w', body=0, volume=1.3 * FM.mesh_volume(V, T))
        S = FM.FluidSolver.build([w], [b], FM.FEnv(), dx=0.01)
        ov = FS.overflow_report(S)[0]
        ok(abs(ov["overflow"] / FM.mesh_volume(V, T) - 0.32) < 0.08 and (('spilled' in ov['handling']) if opened else ('over-pressure' in ov['handling'])),
           "%s cone filled to 130%%: %s" % ("open" if opened else "sealed", ov['handling']))
    # two fluids in one container: the denser one settles below (poured in the wrong order: lighter fluid seeded first by index is irrelevant, the fill sorts by density)
    b = body_of(*lathe(CONE))
    oil = FM.fluid_def('MINERAL_OIL', name='oil', body=0, volume=0.25 * FM.mesh_volume(V, T), miscible=False)
    wat = FM.fluid_def('WATER', name='water', body=0, volume=0.25 * FM.mesh_volume(V, T), miscible=False)
    S = FM.FluidSolver.build([oil, wat], [b], FM.FEnv(), dx=0.008)
    rep = FS.settle(S, mode='LEVEL')
    lay = {n: (lo, hi) for n, lo, hi in rep['layers']}
    ok(lay['water'][1] <= lay['oil'][1] and np.percentile(S.x[S.lo[1]:S.hi[1], 2], 50) < np.percentile(S.x[S.lo[0]:S.hi[0], 2], 50),
       "two fluids: water layer %.3f-%.3f m below the oil layer %.3f-%.3f m" % (lay['water'][0], lay['water'][1], lay['oil'][0], lay['oil'][1]))


def components(P, r):
    """Connected components of a point set (link distance r): (count, sizes descending).  Pure NumPy union-find."""
    n = len(P)
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    I, J = FM.neighbor_pairs(P, r)
    for a, b in zip(I, J):
        if np.linalg.norm(P[a] - P[b]) < r:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    roots = np.array([find(a) for a in range(n)])
    sz = np.sort(np.bincount(np.unique(roots, return_inverse=True)[1]))[::-1]
    return len(sz), sz


def wax_scene(centers, T_wax, dx=0.007, coh=2.0, z_ref=293.15):
    V, T = lathe([(0, 0), (0.06, 0), (0.06, 0.20), (0, 0.20)])
    b = body_of(V, T)
    cav = FM.mesh_volume(V, T)
    car = FM.fluid_def('WATER', name='carrier', body=0, volume=0.75 * cav, density=985.0, beta=4e-4, tref=z_ref, temperature=z_ref, miscible=False, cohesion=0.04)
    wx = []
    for k, c in enumerate(centers):
        P = FM.lat_pos(FM.lattice_in_box(np.array(c) - 0.03, np.array(c) + 0.03, dx), dx)
        P = P[np.linalg.norm(P - np.array(c), axis=1) < 0.028]
        wx.append(FM.fluid_def('LAVA_WAX', name='wax%d' % k, body=0, shape=dict(kind='POINTS', pts=P), density=1034.0, beta=3.5e-3, tref=z_ref,
                               temperature=T_wax, miscible=False, cohesion=coh, group=10 + k))
    return FM.FluidSolver.build([car] + wx, [b], FM.FEnv(), dx=dx)


def interface_mix(S, fw, dx):
    """Fraction of fluid fw's particles that sit mostly among other fluids (numerical mixing across the interface)."""
    wax = np.flatnonzero(S.fid == fw)
    I, J = FM.neighbor_pairs(S.x, 1.6 * dx)
    same = np.zeros(S.n); tot = np.zeros(S.n)
    m = S.fid[I] == S.fid[J]
    for a, b in ((I, J), (J, I)):
        tot += np.bincount(a, minlength=S.n)
        same += np.bincount(a, weights=m.astype(float), minlength=S.n)
    frac = (same[wax] + 1.0) / (tot[wax] + 1.0)
    return float(np.mean(frac < 0.3))


def test_wax_blobs():
    dx = 0.007
    # hot blob: rises, stays ONE connected cluster, little mixing into the carrier
    S = wax_scene([(0, 0, 0.04)], 330.0)
    w0 = S.x[S.lo[1]:S.hi[1], 2].mean()
    zmax, ncomp, mixm = w0, [], []
    for k in range(72):
        S.step(1 / 24.0)
        if k % 12 == 11:
            w = S.x[S.lo[1]:S.hi[1]]
            nc, sz = components(w, 1.5 * dx); ncomp.append((nc, round(sz[0] / len(w), 3))); mixm.append(interface_mix(S, 1, dx)); zmax = max(zmax, w[:, 2].mean())
    ok(zmax - w0 > 0.06, "hot wax blob (%d particles, 4 spacings radius) rises %.0f mm in the carrier" % (S.hi[1] - S.lo[1], 1e3 * (zmax - w0)))
    ok(min(c[1] for c in ncomp) >= 0.95, "the rising blob stays one connected cluster (components, largest share over time: %s)" % ncomp)
    ok(max(mixm) < 0.05, "interface mixing metric stays low (max %.3f)" % max(mixm))
    # cold wax is denser than the carrier: sinks; the same blob cooled while rising turns around
    S = wax_scene([(0, 0, 0.13)], 293.15)
    w0 = S.x[S.lo[1]:S.hi[1], 2].mean()
    for k in range(48):
        S.step(1 / 24.0)
    ok(S.x[S.lo[1]:S.hi[1], 2].mean() < w0 - 0.03, "cold wax blob sinks (%.0f mm)" % (1e3 * (w0 - S.x[S.lo[1]:S.hi[1], 2].mean())))
    # two blobs rising side by side do not merge unless they touch
    S = wax_scene([(-0.033, 0, 0.04), (0.033, 0, 0.04)], 330.0)
    for k in range(48):
        S.step(1 / 24.0)
    wa, wb = S.x[S.lo[1]:S.hi[1]], S.x[S.lo[2]:S.hi[2]]
    allw = np.vstack([wa, wb])
    sa, sb, sall = components(wa, 1.5 * dx)[1], components(wb, 1.5 * dx)[1], components(allw, 1.5 * dx)[1]
    ok(sa[0] >= 0.95 * len(wa) and sb[0] >= 0.95 * len(wb) and sall[0] < 1.1 * max(len(wa), len(wb)),
       "two blobs side by side stay two separate clusters (largest clusters %d / %d of %d / %d, union %d)" % (sa[0], sb[0], len(wa), len(wb), sall[0]))


def test_species():
    from core import species as SP
    dx = 0.008
    # --- yellow water + blue water: composition, colour and temperature mix; mass and heat are conserved
    b = container((-.05, -.05, 0), (.05, .05, .1))
    P = FM.lat_pos(FM.lattice_in_box((-.045, -.045, 0.004), (.045, .045, .08), dx), dx)
    left = P[:, 0] < 0
    yel = FM.fluid_def('WATER', name='yellow water', body=0, shape=dict(kind='POINTS', pts=P[left]), color=(1.0, 0.9, 0.1), temperature=330.0, miscible=True, mix_rate=16.0, cool=1e-12)
    blu = FM.fluid_def('WATER', name='blue water', body=0, shape=dict(kind='POINTS', pts=P[~left]), color=(0.1, 0.3, 1.0), temperature=290.0, miscible=True, mix_rate=16.0, cool=1e-12)
    S = FM.FluidSolver.build([yel, blu], [b], FM.FEnv(), dx=dx)
    m = S.mass_ref
    sp0 = (S.C * m[:, None]).sum(0)
    E0 = (m * S.cp_p * S.T).sum()
    col0 = S.colours()
    spread0 = float(col0.std(0).max())
    run(S, 12.0)
    col = S.colours()
    green = np.exp(0.5 * (np.log(np.clip(np.array([1.0, 0.9, 0.1]), 0.02, 1)) + np.log(np.clip(np.array([0.1, 0.3, 1.0]), 0.02, 1))))
    ok(col.std(0).max() < 0.3 * spread0 and np.abs(col.mean(0) - green).max() < 0.08 and col.mean(0)[1] > col.mean(0)[0] and col.mean(0)[1] > col.mean(0)[2],
       "yellow + blue water converge to green: mean colour %s (subtractive mix %s), colour spread %.3f -> %.3f" % (np.round(col.mean(0), 2), np.round(green, 2), spread0, col.std(0).max()))
    sp1 = (S.C * m[:, None]).sum(0)
    ok(np.allclose(sp0, sp1, rtol=1e-9) and np.allclose(S.C.sum(1), 1.0), "species mass conserved by the mixing (%.4f kg yellow, %.4f kg blue)" % (sp1[0], sp1[1]))
    Tm = float((m * S.cp_p * S.T).sum() / (m * S.cp_p).sum())
    ok(abs((m * S.cp_p * S.T).sum() - E0) < 1e-6 * E0 and S.T.std() < 8.0 and abs(Tm - 310.0) < 1.0,
       "temperature equilibrates to the mass-weighted value: %.1f K (expected 310), spread %.1f K, enthalpy conserved" % (Tm, S.T.std()))
    # a small amount of a colour merges into the larger body and fades into it
    big = FM.fluid_def('WATER', name='body', body=0, fill=0.5, color=(0.1, 0.3, 1.0), miscible=True, mix_rate=16.0)
    dab = FM.fluid_def('WATER', name='dab', body=0, shape=dict(kind='POINTS', pts=FM.lat_pos(FM.lattice_in_box((-.01, -.01, 0.03), (.01, .01, 0.05), dx), dx)),
                       color=(1.0, 0.1, 0.1), miscible=True, mix_rate=16.0)
    S2 = FM.FluidSolver.build([big, dab], [container((-.05, -.05, 0), (.05, .05, .1))], FM.FEnv(), dx=dx)
    c_dab0 = S2.colours()[S2.lo[1]:S2.hi[1]].copy()
    run(S2, 10.0)
    cd = S2.colours()[S2.lo[1]:S2.hi[1]]
    ok(np.abs(cd.mean(0) - S2.colours()[S2.lo[0]:S2.hi[0]].mean(0)).max() < 0.1 and np.abs(cd - c_dab0).max() > 0.3,
       "a small red dab fades into the larger blue body (dab colour %s -> %s)" % (np.round(c_dab0.mean(0), 2), np.round(cd.mean(0), 2)))
    # --- ABSORB: the higher priority species absorbs the other, cluster ids propagate from the root
    g0 = FM.FEnv(gravity=(0.0, 0.0, 0.0))
    big_sp = FM.fluid_def('WATER', name='dominant', body=0, shape=dict(kind='POINTS', pts=P[left]), color=(0.9, 0.2, 0.2), priority=5, miscible=False)
    low_sp = FM.fluid_def('WATER', name='lesser', body=0, shape=dict(kind='POINTS', pts=P[~left]), color=(0.2, 0.2, 0.9), priority=1, miscible=False)
    S3 = FM.FluidSolver.build([big_sp, low_sp], [b], g0, dx=dx, rules=[dict(a='dominant', b='lesser', outcome='ABSORB', rate=3.0)])
    S3.mixing = True
    run(S3, 8.0)
    dom = S3.SS.dominant(S3.C)
    ok(np.all(dom == 0) and np.allclose(S3.colours(), np.array([0.9, 0.2, 0.2]), atol=1e-6),
       "ABSORB: all lower-priority particles adopt the dominant species and colour (%d merges)" % S3.mix_stats['merges'])
    lo1 = S3.cluster[S3.lo[1]:S3.hi[1]]
    ok(len(np.unique(S3.cluster)) == 1 and S3.mix_stats['merges'] >= S3.hi[1] - S3.lo[1], "ABSORB: cluster ids propagate from the root (%d cluster id)" % len(np.unique(S3.cluster)))
    # --- REACT: A + B -> C only inside the temperature / relative speed / density window, with the stated heat
    def react_run(Tc):
        wa = FM.fluid_def('WATER', name='A', body=0, shape=dict(kind='POINTS', pts=P[left]), velocity=(0.4, 0, 0), temperature=Tc, miscible=False, color=(1, 0.2, 0.2))
        wb = FM.fluid_def('WATER', name='B', body=0, shape=dict(kind='POINTS', pts=P[~left]), velocity=(-0.4, 0, 0), temperature=Tc, miscible=False, color=(0.2, 0.2, 1))
        rule = dict(a='A', b='B', outcome='REACT', product='C', tmin=300.0, tmax=400.0, vmin=0.2, q=2.0e5, cmin=0.6)
        S4 = FM.FluidSolver.build([wa, wb], [b], g0, dx=dx, rules=[rule], species=[dict(name='C', color=(0.2, 0.8, 0.2), density=1100.0, cp=4186.0, beta=2e-4)])
        S4.mixing = True
        T0 = S4.T.mean()
        for k in range(6):
            S4.step(1 / 24.0)
        return S4, T0
    Sa, Ta = react_run(350.0)
    Sb, Tb = react_run(250.0)
    nC = int(np.count_nonzero(Sa.SS.dominant(Sa.C) == 2))
    ok(Sa.mix_stats['reactions'] > 0 and nC > 0 and Sa.T[Sa.SS.dominant(Sa.C) == 2].mean() > Ta + 40.0 and Sb.mix_stats['reactions'] == 0,
       "REACT: inside the window %d reactions, %d product particles at +%.0f K (stated heat 2e5 J/kg = +%.0f K); outside it (250 K) %d reactions"
       % (Sa.mix_stats['reactions'], nC, Sa.T[Sa.SS.dominant(Sa.C) == 2].mean() - Ta, 2e5 / 4186.0, Sb.mix_stats['reactions']))
    # density window: the same pair with an impossible density ratio does not react
    wa = FM.fluid_def('WATER', name='A', body=0, shape=dict(kind='POINTS', pts=P[left]), velocity=(0.4, 0, 0), temperature=350.0, miscible=False)
    wb = FM.fluid_def('WATER', name='B', body=0, shape=dict(kind='POINTS', pts=P[~left]), velocity=(-0.4, 0, 0), temperature=350.0, miscible=False)
    S5 = FM.FluidSolver.build([wa, wb], [b], g0, dx=dx, rules=[dict(a='A', b='B', outcome='REACT', product='A', rmin=5.0)])
    S5.mixing = True
    for k in range(6):
        S5.step(1 / 24.0)
    ok(S5.mix_stats['reactions'] == 0, "REACT: a density ratio outside its window does not react")
    # JSON preset round trip
    js = SP.SpeciesSystem([SP.species_def('A'), SP.species_def('B')], [SP.rule_def('A', 'B', 'ABSORB')]).to_json()
    ok(SP.SpeciesSystem.from_json(js).rules[0]['outcome'] == 'ABSORB', "species / rule table round-trips through JSON")


def test_surface_merge():
    from core import fluidsurface as FS
    dx = 0.008

    def ball(c, r):
        P = FM.lat_pos(FM.lattice_in_box((c[0] - r, c[1] - r, c[2] - r), (c[0] + r, c[1] + r, c[2] + r), dx), dx)
        return P[np.linalg.norm(P - np.array(c), axis=1) < r]
    r = 0.03
    far = np.vstack([ball((-0.05, 0, 0), r), ball((0.05, 0, 0), r)])
    near = np.vstack([ball((-0.025, 0, 0), r), ball((0.025, 0, 0), r)])
    v, f, _ = FS.fluid_surface(far, None, dx)
    nf = FS.mesh_components(len(v), f)
    v, f, _ = FS.fluid_surface(near, None, dx)
    nn = FS.mesh_components(len(v), f)
    ok(nf == 2, "separated blobs give 2 surface components (%d)" % nf)
    ok(nn == 1, "touching blobs merge into 1 surface component (%d)" % nn)


def test_surface_builder():
    """Merged, frame-coherent liquid surface: colours blend across the seam, a merging pair joins without flicker."""
    from core import fluidsurface as FS
    dx = 0.008
    r = 0.03

    def ball(c):
        P = FM.lat_pos(FM.lattice_in_box((c[0] - r, c[1] - r, c[2] - r), (c[0] + r, c[1] + r, c[2] + r), dx), dx)
        return P[np.linalg.norm(P - np.array(c), axis=1) < r]
    sb = FS.SurfaceBuilder(dx)
    counts, nv = [], []
    red, blue = np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
    mid_col = None
    for k, d in enumerate(np.linspace(0.10, 0.03, 8)):
        A, B = ball((-d / 2, 0, 0)), ball((d / 2, 0, 0))
        x = np.vstack([A, B])
        col = np.vstack([np.tile(red, (len(A), 1)), np.tile(blue, (len(B), 1))])
        out = sb.update(x, col, frame=k)
        v, f, c = out[0], out[1], out[2]
        counts.append(FS.mesh_components(len(v), f))
        nv.append(len(v))
        if k == 7 and c is not None:
            mid_col = np.asarray(c)[:, :3]
    ok(counts[0] == 2 and counts[-1] == 1, "pair merges from 2 to 1 component (%s)" % counts)
    ok(all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1)), "component count is monotonic while merging (no flicker)")
    ok(max(abs(nv[i + 1] - nv[i]) / nv[i] for i in range(len(nv) - 1)) < 0.6, "vertex count changes smoothly (%s)" % nv)
    if mid_col is not None:
        purple = np.sum((mid_col[:, 0] > 0.2) & (mid_col[:, 2] > 0.2))
        ok(purple > 0, "colours blend across the seam (%d mixed vertices)" % purple)


def test_gas_pressure():
    """Ideal-gas pressure force: gas heated in place (fixed volume) expands and pushes neighbours (another asset's particles, a free block,
    a piston); the pair forces conserve momentum.  `tref` is the temperature the gas was filled at, `temperature` the heated state."""
    env0 = FM.FEnv(gravity=(0.0, 0.0, 0.0))
    dx = 0.03
    # --- 1. a heated blob next to a cold blob of another gas: it expands, the neighbour's particles are pushed away, momentum is conserved
    Ph = sphere_pts((0, 0, 0), 0.12, dx)
    Pc = sphere_pts((0.22, 0, 0), 0.10, dx)
    hot = FM.fluid_def('AIR', name='hot', shape=dict(kind='POINTS', pts=Ph), count=len(Ph), temperature=900.0, tref=293.15)
    cold = FM.fluid_def('CO2', name='cold', shape=dict(kind='POINTS', pts=Pc), count=len(Pc), temperature=293.15)
    S = FM.FluidSolver.build([hot, cold], [], env0, dx=dx)
    hi = slice(S.lo[0], S.hi[0]); ci = slice(S.lo[1], S.hi[1])
    r0 = float(np.linalg.norm(S.x[hi], axis=1).mean())
    cx0 = float(S.x[ci, 0].mean())
    pk, cxm, fs, fa = 0.0, cx0, 0.0, 0.0
    for k in range(12):
        S.step(1 / 24.0)
        pk = max(pk, float(S.pressure[hi].max()))
        cxm = max(cxm, float(S.x[ci, 0].mean()))
        fs, fa = max(fs, float(np.linalg.norm(S.gas_force_sum))), max(fa, S.gas_force_abs)
    r1 = float(np.linalg.norm(S.x[hi], axis=1).mean())
    ok(r1 > 1.2 * r0, "heated gas expands: mean radius %.3f -> %.3f m" % (r0, r1))
    ok(cxm > cx0 + 0.005, "the neighbouring asset's particles are pushed outward: centre x %.3f -> %.3f m (peak)" % (cx0, cxm))
    ok(pk > 2.0 * FM.P_ATM and np.all(np.isfinite(S.pressure)), "pressure is a solver output: the heated parcels peak at %.0f Pa (ambient %.0f)" % (pk, FM.P_ATM))
    ok(fs < 0.02 * max(fa, 1e-12), "pair forces conserve momentum: net %.2e N against %.2e N of pair force" % (fs, fa))
    # --- 2. heated gas in a closed chamber pushes a piston, much more than the same chamber unheated
    def chamber(T):
        b3 = container((-.2, -.1, 0), (.4, .1, .2))
        Vp, Tp = box((0.05, -.105, -.005), (.15, .105, .205))
        pis = FM.Body(SDF.PartField(Vp, Tp, 'AUTO', resolution=24, level=2), np.eye(4), 'COLLIDER', None, 0.0)
        pis.mass = 1.0
        pis.constraint, pis.axis, pis.limits = 'SLIDER', np.array([1.0, 0, 0]), (0.0, 0.3)
        P = FM.lat_pos(FM.lattice_in_box(np.array([-.18, -.085, .015]), np.array([0.04, .085, .185]), 0.02), 0.02)
        gas = FM.fluid_def('AIR', name='chamber', body=0, shape=dict(kind='POINTS', pts=P), count=len(P), temperature=T, tref=293.15)
        S = FM.FluidSolver.build([gas], [b3, pis], env0, dx=0.02)
        run(S, 0.17)
        return pis, S
    pis, S = chamber(1500.0)
    pis2, S2 = chamber(293.15)
    ok(pis.travel > 0.05, "compressed hot gas pushes the piston: moved %.3f m (speed %.2f m/s)" % (pis.travel, float(pis.vel[0])))
    ok(pis.travel > 3.0 * max(pis2.travel, 1e-4), "heated gas moves the piston more than unheated: %.3f vs %.3f m" % (pis.travel, pis2.travel))
    # --- 3. a free block next to expanding hot gas is pushed outward
    Vb, Tb = box((0.12, -.03, -.03), (0.18, .03, .03))
    blk = FM.Body(SDF.PartField(Vb, Tb, 'AUTO', resolution=24, level=2), np.eye(4), 'COLLIDER', None, 0.0)
    blk.mass, blk.constraint = 0.5, 'FREE'
    Pg = sphere_pts((0, 0, 0), 0.1, 0.03)
    g3 = FM.fluid_def('AIR', name='hot', shape=dict(kind='POINTS', pts=Pg), count=len(Pg), temperature=1500.0, tref=293.15)
    S3 = FM.FluidSolver.build([g3], [blk], env0, dx=0.03)
    x0 = float(blk.com_world()[0])
    xm = x0
    for k in range(14):
        S3.step(1 / 24.0)
        xm = max(xm, float(blk.com_world()[0]))
    ok(xm > x0 + 0.02, "expanding hot gas pushes a free block outward: x %.3f -> %.3f m" % (x0, xm))
    # --- 4. the heated gas also pushes a neighbouring liquid asset's particles (water droplet, free)
    Pw = sphere_pts((0.17, 0, 0), 0.05, 0.02)
    gw = FM.fluid_def('AIR', name='hot', shape=dict(kind='POINTS', pts=sphere_pts((0, 0, 0), 0.1, 0.03)), count=len(sphere_pts((0, 0, 0), 0.1, 0.03)), temperature=1500.0, tref=293.15)
    dw = FM.fluid_def('WATER', name='drop', shape=dict(kind='POINTS', pts=Pw), drag=0.0, cohesion=0.0)
    S4 = FM.FluidSolver.build([gw, dw], [], env0, dx=0.03)
    wi = slice(S4.lo[1], S4.hi[1])
    vx = 0.0
    for k in range(4):
        S4.step(1 / 24.0)
        vx = max(vx, float(S4.v[wi, 0].mean()))
    ok(vx > 0.05, "neighbouring liquid asset is pushed by the gas pressure: drop speed %.2f m/s" % vx)



def test_surface_coupling():
    """A viscosity surface is two-way: it drags the fluid and the fluid's drag moves the body the surface belongs to (momentum exchanged)."""
    from core import fsurf as FS
    env0 = FM.FEnv(gravity=(0.0, 0.0, 0.0))
    Pw, Nw, Aw, _ = FS.sample_faces(np.array([[-0.5, -0.5, 0.05], [1.0, -0.5, 0.05], [1.0, 1.0, 0.05], [-0.5, 1.0, 0.05]]), np.array([[0, 1, 2], [0, 2, 3]]), 0.04)
    Vb, Tb = box((0.0, 0.0, 5.0), (0.05, 0.05, 5.05))                         # the body's own geometry sits far from the fluid
    blk = FM.Body(SDF.PartField(Vb, Tb, 'AUTO', resolution=16, level=2), np.eye(4), 'COLLIDER', None, 0.0)
    blk.mass, blk.constraint, blk.axis = 2.0, 'SLIDER', np.array([1.0, 0, 0])
    P = sphere_pts((0.2, 0.2, 0.05), 0.1, 0.04)
    d = FM.fluid_def('WATER', name='w', shape=dict(kind='POINTS', pts=P), velocity=(1.0, 0.0, 0.0), drag=0.0, cohesion=0.0)
    S = FM.FluidSolver.build([d], [blk], env0, dx=0.04)
    vs = FS.ViscositySurface("plate", Pw, np.array([[0, 0, 1.0]] * len(Pw)), Aw, coefficient=200.0)
    vs.body = 0
    FS.install(S, viscosity=[vs])
    m = S.mass_ref
    p0 = float((m * S.v[:, 0]).sum())
    for _ in range(12):
        S.step(1 / 24.0)
    p1 = float((m * S.v[:, 0]).sum())
    gain = blk.mass * float(blk.vel[0])
    ok(blk.vel[0] > 0.0 and blk.travel > 0.0, "the fluid's drag moves the body: speed %.3f m/s, travel %.3f m" % (blk.vel[0], blk.travel))
    ok(abs(blk.vel[1]) < 1e-9 and abs(blk.vel[2]) < 1e-9, "the slider keeps to its axis")
    loss = p0 - p1
    ok(loss > 0 and 0.4 * loss < gain < 1.5 * loss, "momentum exchanged: body gained %.3f, fluid lost %.3f kg m/s" % (gain, loss))
    # the same surface without a body link only drags the fluid (previous behaviour)
    S2 = FM.FluidSolver.build([dict(d)], [], env0, dx=0.04)
    vs2 = FS.ViscositySurface("plate", Pw, np.array([[0, 0, 1.0]] * len(Pw)), Aw, coefficient=200.0)
    FS.install(S2, viscosity=[vs2])
    for _ in range(12):
        S2.step(1 / 24.0)
    ok(float(S2.v[:, 0].mean()) < 0.95 and float(np.linalg.norm(vs2.force)) > 0, "unlinked surface: still drags the fluid, reports its reaction force")
    # excess gas pressure pushes the faces along their normal
    Pg = sphere_pts((0.2, 0.2, 0.15), 0.08, 0.03)
    gg = FM.fluid_def('AIR', name='g', shape=dict(kind='POINTS', pts=Pg), count=len(Pg), temperature=1500.0, tref=293.15)
    S3 = FM.FluidSolver.build([gg], [], env0, dx=0.03)
    vs3 = FS.ViscositySurface("plate", Pw, np.array([[0, 0, 1.0]] * len(Pw)), Aw, coefficient=1.0, noslip=0.0, reach=0.08)
    FS.install(S3, viscosity=[vs3])
    fz = 0.0
    for _ in range(6):
        S3.step(1 / 24.0)
        fz = max(fz, float(vs3.pressure_force[2]))
    ok(abs(fz) > 0.0 and np.isfinite(fz), "gas pressure acts on the surface faces: peak normal force %.3g N" % fz)


def test_phase():
    """Transmutation: phase / chemistry rules with animated blending, mass and enthalpy bookkeeping, crystals grown from nuclei."""
    from core import phase as PH
    env0 = FM.FEnv()
    # --- 1. superheated water -> steam: gas parcels rise, mass conserved, enthalpy changes by q x converted mass
    pr = PH.preset('water_steam')
    for sp in pr['species']:
        sp['cp'] = 4000.0                                   # equal heat capacities: the enthalpy ledger is then exact
    P = FM.lat_pos(FM.lattice_in_box(np.array([0.0, 0.0, 1.0]), np.array([0.16, 0.16, 1.08]), 0.02), 0.02)
    w = FM.fluid_def('WATER', name='Water', shape=dict(kind='POINTS', pts=P), temperature=900.0, cp=4000.0, cool=1e-9)
    S = FM.FluidSolver.build([w], [], env0, dx=0.02, species=pr['species'][1:], phase=pr['phase'])
    m0 = float(S.mass_ref.sum())
    H0 = float((S.mass_ref * S.cp_p * S.T).sum())
    z0 = float(S.x[:, 2].mean())
    fr = []
    for k in range(40):
        S.step(1 / 24.0)
        fr.append(float(S.C[:, 1].mean()))
    gz = float(S.x[S.gasp, 2].mean()) if S.gasp.any() else z0
    ok(abs(float(S.mass_ref.sum()) - m0) < 1e-12 and np.allclose(S.C.sum(1), 1.0), "mass conserved, composition sums to 1")
    ok(S.gasp.mean() > 0.8, "liquid -> gas: %.0f %% of the particles became gas parcels" % (100 * S.gasp.mean()))
    d = np.diff([0.0] + fr)
    ok(d.min() > -1e-9 and d.max() < 0.35, "the steam fraction rises smoothly (largest frame step %.3f, none backwards)" % d.max())
    H1 = float((S.mass_ref * S.cp_p * S.T).sum())
    ok(abs((H1 - H0) - S.phase_stats['heat']) < 0.03 * abs(S.phase_stats['heat']) + 1.0, "enthalpy changes by the latent heat: dH %.3g J, q x dm %.3g J" % (H1 - H0, S.phase_stats['heat']))
    ok(gz > z0 + 0.3, "steam rises: gas parcels +%.2f m" % (gz - z0))
    ok(float(S.T.mean()) < 600.0, "latent heat cools the vapour (%.0f K)" % S.T.mean())
    # --- 2. wax solidifies into frozen crystals growing outward from 3 nuclei
    pw = PH.preset('wax')
    V3, T3 = box((0, 0, 0), (0.2, 0.2, 0.2))
    P = FM.lat_pos(FM.lattice_in_box(np.array([0.02, 0.02, 0.02]), np.array([0.18, 0.18, 0.12]), 0.02), 0.02)
    wx = FM.fluid_def('LAVA_WAX', name='Wax melt', shape=dict(kind='POINTS', pts=P), temperature=300.0, cool=3.0)
    cont = FM.Body(SDF.PartField(V3, T3, 'SHELL', resolution=24, level=2), np.eye(4), 'CONTAINER', None, 0.2)
    S = FM.FluidSolver.build([wx], [cont], env0, dx=0.02, species=pw['species'][1:], phase=pw['phase'])
    cnt, xs_prev, still = [], None, True
    for k in range(48):
        S.step(1 / 24.0)
        cnt.append(int(S.frozen.sum()))
        if xs_prev is not None:
            fz = xs_prev[1] & S.frozen
            still &= bool(np.abs(S.x[fz] - xs_prev[0][fz]).max() < 1e-12) if fz.any() else True
        xs_prev = (S.x.copy(), S.frozen.copy())
    ok(cnt[0] <= 6 and cnt[-1] > 0.6 * S.n, "crystal growth from nuclei: %d -> %d of %d particles frozen" % (cnt[0], cnt[-1], S.n))
    ok(all(cnt[i + 1] >= cnt[i] for i in range(len(cnt) - 1)), "the frozen set only grows while cooling")
    ok(still and (not S.frozen.any() or float(np.abs(S.v[S.frozen]).max()) == 0.0), "frozen crystal particles stay in place with zero velocity")
    solid_frac = S.C[:, 1]
    ok(((solid_frac > 0.05) & (solid_frac < 0.95)).any() or cnt[-1] == S.n, "the transition is animated (intermediate composition exists)")
    # reheating melts it back
    for k in range(48):
        S.T[:] = 360.0
        S.step(1 / 24.0)
    ok(S.frozen.sum() < 0.2 * S.n, "heated above the melting window the crystal melts (%d frozen)" % S.frozen.sum())
    # --- 3. rust tint: pure composition change, colour moves continuously
    pg = PH.preset('rust')
    wr = FM.fluid_def('CUSTOM', name='Iron', shape=dict(kind='POINTS', pts=P[:200]), temperature=300.0, color=(0.55, 0.57, 0.6), cool=1e-9)
    S = FM.FluidSolver.build([wr], [], FM.FEnv(gravity=(0, 0, 0)), dx=0.02, species=pg['species'][1:], phase=pg['phase'])
    c0 = S.colours().mean(0)
    run(S, 4.0)
    c1 = S.colours().mean(0)
    fr_ = float(S.C[:, 1].mean())
    ok(abs(fr_ - (1 - np.exp(-0.15 * 4.0))) < 0.05, "rust fraction follows the rate: %.3f (exp %.3f)" % (fr_, 1 - np.exp(-0.15 * 4.0)))
    ok(c1[0] > c1[2] + 0.1 and c0[2] >= c0[0] - 0.1, "the colour moves from grey toward rust (r-b %.2f -> %.2f)" % (c0[0] - c0[2], c1[0] - c1[2]))
    # --- 4. fuel -> flame -> smoke by temperature; JSON round trip of the rule table
    pf = PH.preset('fuel_flame_smoke')
    wf = FM.fluid_def('CUSTOM', name='Fuel', shape=dict(kind='POINTS', pts=P[:200]), temperature=1100.0, color=(0.9, 0.7, 0.2), cool=1e-9)
    S = FM.FluidSolver.build([wf], [], FM.FEnv(gravity=(0, 0, 0)), dx=0.02, species=pf['species'][1:], phase=pf['phase'])
    run(S, 2.0)
    flame_peak = float(S.C[:, 1].max())
    S.T[:] = 700.0
    run(S, 3.0)
    ok(flame_peak > 0.3 and float(S.C[:, 2].mean()) > 0.2, "fuel burns to flame (peak %.2f) which turns to smoke when it cools (%.2f)" % (flame_peak, S.C[:, 2].mean()))
    sp_, ph_ = PH.from_json(PH.to_json('salt_crystal'))
    ok(len(sp_) == 2 and ph_[0]['kind'] == 'CRYSTAL' and PH.rule_def(**ph_[0])['b'] == 'Salt crystal', "JSON rule table round trip")


def test_coupling():
    """Two-way fluid-solid coupling: the fluid pushes bodies."""
    # --- 1. hydrostatics: the force of the water on a fixed tank equals its weight
    dx = 0.012
    V, T = lathe([(0, 0), (0.09, 0), (0.09, 0.3), (0, 0.3)])
    tank = body_of(V, T)
    w = FM.fluid_def('WATER', name='w', body=0, fill=0.6)
    S = FM.FluidSolver.build([w], [tank], FM.FEnv(), dx=dx)
    run(S, 1.0)
    Fs = []
    for _ in range(12):
        S.step(1 / 24.0)
        Fs.append(S.bodies[0].F.copy())
    F = np.mean(Fs, 0)
    weight = float(S.mass_ref.sum()) * 9.80665
    ok(abs(-F[2] - weight) < 0.2 * weight and abs(F[0]) < 0.1 * weight, "water on a fixed tank: force %.1f N down against weight %.1f N (side force %.1f N)" % (-F[2], weight, abs(F[0])))
    # --- 2. buoyancy: a block lighter than water released at the waterline stays afloat, the fluid force carries its weight; a heavy one sinks
    def block(rho_b, z0):
        b2 = container((-.1, -.1, 0), (.1, .1, .35))
        side = 0.1
        Vb, Tb = box((-side / 2,) * 3, (side / 2,) * 3)
        blk = FM.Body(SDF.PartField(Vb, Tb, 'AUTO', resolution=24, level=2), np.eye(4), 'COLLIDER', None, 0.0)
        blk.mass, blk.kfric = rho_b * side ** 3, 5.0
        M = np.eye(4); M[2, 3] = z0
        blk.M0 = blk.M1 = M
        S = FM.FluidSolver.build([FM.fluid_def('WATER', name='w', body=0, fill=0.5)], [b2, blk], FM.FEnv(), dx=0.01)
        zs, Fz = [], []
        for k in range(36):
            S.step(1 / 24.0)
            zs.append(blk.com_world()[2])
            Fz.append(blk.F[2])
        return blk, np.array(zs), np.array(Fz)
    blk, zs, Fz = block(500.0, 0.17)
    ok(np.all(np.abs(zs[12:] - 0.18) < 0.035) and np.all(np.isfinite(zs)), "a block of half the density of water stays afloat at the waterline (centre %.3f to %.3f m over the last 1.5 s)" % (zs[12:].min(), zs[12:].max()))
    ok(abs(np.mean(Fz[12:]) - blk.mass * 9.80665) < 0.3 * blk.mass * 9.80665, "the fluid carries its weight: mean buoyancy force %.1f N, weight %.1f N" % (np.mean(Fz[12:]), blk.mass * 9.80665))
    blk, zs, Fz = block(2000.0, 0.17)
    ok(zs[-1] < 0.1, "a block twice as dense as water sinks to the bottom (centre %.3f m)" % zs[-1])
    # --- 3. a slider (piston) pushed by a fluid slug: momentum goes into the piston
    env0 = FM.FEnv(gravity=(0.0, 0.0, 0.0))
    b3 = container((-.4, -.1, 0), (.4, .1, .2))
    Vp, Tp = box((-.015, -.09, .01), (.015, .09, .19))
    pis = FM.Body(SDF.PartField(Vp, Tp, 'AUTO', resolution=24, level=2), np.eye(4), 'COLLIDER', None, 0.0)
    pis.mass = 1.0
    pis.constraint, pis.axis, pis.limits = 'SLIDER', np.array([1.0, 0, 0]), (0.0, 0.3)
    P = FM.lat_pos(FM.lattice_in_box(np.array([-.35, -.085, .015]), np.array([-.2, .085, .185]), 0.012), 0.012)
    slug = FM.fluid_def('WATER', name='slug', body=0, shape=dict(kind='POINTS', pts=P), velocity=(2.0, 0.0, 0.0))
    S = FM.FluidSolver.build([slug], [b3, pis], env0, dx=0.012)
    m_slug = float(S.mass_ref.sum())
    px0 = float((S.mass_ref * S.v[:, 0]).sum())
    v_p, px1, vs_ = 0.0, px0, []
    for k in range(36):
        S.step(1 / 24.0)
        vs_.append((float(pis.vel[0]), float((S.mass_ref * S.v[:, 0]).sum())))
    kmax = int(np.argmax([a for a, _ in vs_]))
    v_p, px1 = vs_[kmax]
    ok(pis.travel > 0.02 and v_p > 0.3, "fluid slug pushes the piston along its rail: moved %.3f m, speed %.2f m/s" % (pis.travel, v_p))
    ok(abs(pis.vel[1]) < 1e-9 and abs(pis.vel[2]) < 1e-9, "the piston only moves along its allowed axis")
    gain = pis.mass * v_p
    loss = px0 - px1
    ok(0.15 * loss < gain < 1.1 * loss, "momentum: piston gained %.2f kg m/s, fluid lost %.2f kg m/s (slug %.2f kg at 2 m/s)" % (gain, loss, m_slug))
    # --- 4. toggle off: the same piston stays put
    pis2 = FM.Body(SDF.PartField(Vp, Tp, 'AUTO', resolution=24, level=2), np.eye(4), 'COLLIDER', None, 0.0)
    pis2.mass, pis2.two_way = 1.0, False
    pis2.constraint, pis2.axis = 'SLIDER', np.array([1.0, 0, 0])
    S = FM.FluidSolver.build([FM.fluid_def('WATER', name='slug', body=0, shape=dict(kind='POINTS', pts=P), velocity=(2.0, 0.0, 0.0))], [container((-.4, -.1, 0), (.4, .1, .2)), pis2], env0, dx=0.012)
    for k in range(24):
        S.step(1 / 24.0)
    ok(abs(pis2.travel) < 1e-12 and not pis2.dynamic, "per-body toggle off: the body does not move")
    # --- 5. a stuck body: friction holds it until the force exceeds it
    pis3 = FM.Body(SDF.PartField(Vp, Tp, 'AUTO', resolution=24, level=2), np.eye(4), 'COLLIDER', None, 0.0)
    pis3.mass, pis3.stick = 1.0, 1e4
    pis3.constraint, pis3.axis = 'SLIDER', np.array([1.0, 0, 0])
    S = FM.FluidSolver.build([FM.fluid_def('WATER', name='slug', body=0, shape=dict(kind='POINTS', pts=P), velocity=(2.0, 0.0, 0.0))], [container((-.4, -.1, 0), (.4, .1, .2)), pis3], env0, dx=0.012)
    for k in range(24):
        S.step(1 / 24.0)
    ok(abs(pis3.travel) < 1e-9, "static friction (10 kN) holds the piston: the fluid force stayed below it")


def test_speed():
    b = container((-.15, -.15, 0), (.15, .15, .4))
    w = FM.fluid_def('WATER', name='w', body=0, fill=0.5)
    for N in (5000, 20000):
        S = FM.FluidSolver.build([w], [b], FM.FEnv(), max_particles=N)
        S.step(1 / 24.0)
        t = time.time()
        for _ in range(3):
            S.step(1 / 24.0)
        dtm = (time.time() - t) / 3
        print("SPEED %d particles: %.2f s per frame (%d drift iterations, up to %d substeps, %d CG iterations last)" % (S.n, dtm, S.iters, S.max_sub, S.cg_last), flush=True)


if __name__ == '__main__':
    t0 = time.time()
    only = sys.argv[1:]
    for fn in (test_units, test_environment, test_boussinesq, test_stratify_and_mass, test_gas_wind, test_gas_buoyancy, test_gravity_scaling,
               test_heat, test_moving_wall, test_solid_displaces, test_spinning_gear, test_barometric_expansion, test_cavity_fill, test_confinement, test_fill_settle, test_wax_blobs, test_species, test_surface_merge, test_surface_builder, test_gas_pressure, test_surface_coupling, test_phase, test_coupling, test_speed):
        if only and fn.__name__ not in only:
            continue
        t = time.time()
        try:
            fn()
        except Exception as ex:
            import traceback
            traceback.print_exc()
            ok(False, "%s raised %r" % (fn.__name__, ex))
        print("  (%s %.1fs)" % (fn.__name__, time.time() - t), flush=True)
    print("FAILURES:", fails[0], "(%.0fs)" % (time.time() - t0))
    raise SystemExit(1 if fails[0] else 0)
