"""Fill / Settle phase (no bpy): the settled state of a container-filled fluid is the INITIAL STATE of the main simulation.

FluidSolver.build seeds every container fluid from its quantity (volume, or mass / density) by sampling the REAL cavity through the
collision-body projection, lowest point along gravity first, so each finite fill takes the cavity's contour and its free surface is at the
level that the volume needs.  settle() then runs the dynamic fill: the solver is stepped under the Environment gravity with every heat
source off until the kinetic energy has dropped (or max_steps), unless `quick` is set, which keeps the level-set seeding and only
runs a few relaxing steps.  Reports: capacity / overflow, layer heights along gravity, hydrostatic pressure at the bottom.

Overflow: an OPEN vessel spills the excess (it is not seeded, the report gives the spilled volume and mass); a SEALED vessel is filled to the
brim and the report gives the excess volume and the compression over-pressure dp = K dV / V (K = bulk modulus, water 2.2 GPa), which is
reported, not simulated."""
import numpy as np

BULK = 2.2e9


def gravity_dir(S):
    g = np.asarray(S.env.g, float)
    n = np.linalg.norm(g)
    return g / n if n > 0 else np.array([0.0, 0.0, -1.0])


def layer_heights(S):
    """Per fluid: (name, lowest, highest) particle height above the lowest particle of the system, measured along gravity."""
    up = -gravity_dir(S)
    liq = [f for f in range(S.nf) if not S.gas[f] and S.hi[f] > S.lo[f]]
    if not liq:
        return []
    h0 = min(float((S.x[S.lo[f]:S.hi[f]] @ up).min()) for f in liq)
    out = []
    for f in liq:
        h = S.x[S.lo[f]:S.hi[f]] @ up - h0
        out.append((S.defs[f]['name'], float(h.min()), float(h.max())))
    return out


def overflow_report(S):
    """List of dicts per container: capacity, requested, overflow volume / mass, open or sealed handling, over-pressure when sealed."""
    out = []
    for r in S.fill_report:
        ov = r['overflow']
        rho = np.mean([S.defs[f]['density'] for f, _v, _n in r['fluids']]) if r['fluids'] else 0.0
        d = dict(capacity=r['capacity'], requested=r['requested'], overflow=ov, overflow_mass=ov * rho, open=r['open'])
        if ov > 0:
            if r['open']:
                d['handling'] = "open vessel: %.3f L spilled to the environment (not seeded)" % (ov * 1e3)
            else:
                d['over_pressure'] = BULK * ov / max(r['free_capacity'], 1e-12)
                d['handling'] = "sealed vessel: filled to the brim, %.3f L excess, compression over-pressure about %.2e Pa (reported, not simulated)" % (ov * 1e3, d['over_pressure'])
        else:
            d['handling'] = "fits (%.0f%% of the cavity)" % (100.0 * r['requested'] / max(r['capacity'], 1e-12))
        out.append(d)
    return out


def hydrostatic_report(S, n_layers=3):
    """Pressure at the bottom of the central column from the solver's last pressure solve vs the expected integral of rho g dz.
    Returns dict(measured, expected, ratio) or None (no pressure solve yet / gravity not along Z)."""
    g = gravity_dir(S)
    if abs(g[2]) < 0.999 or S._p_prev is None or S._p_origin is None:
        return None
    liq = np.flatnonzero(~S.gas[S.fid])
    if len(liq) == 0:
        return None
    X = S.x[liq]
    c = X[:, :2].mean(0)
    R = 1.5 * S.hg
    col = np.flatnonzero(np.hypot(X[:, 0] - c[0], X[:, 1] - c[1]) < R)
    if len(col) < 4:
        return None
    rho = S.rho_particles()[liq[col]]
    z = X[col, 2]
    area = np.pi * R * R
    gmag = float(np.linalg.norm(S.env.g))
    p, org, hg = S._p_prev, S._p_origin, S.hg
    i = int(np.floor((c[0] - org[0]) / hg)); j = int(np.floor((c[1] - org[1]) / hg))
    k0 = int(np.floor((z.min() - org[2]) / hg))
    meas, exp = [], []
    for k in range(k0 + 1, k0 + 1 + n_layers):
        if not (0 <= i < p.shape[0] and 0 <= j < p.shape[1] and 0 <= k < p.shape[2]):
            continue
        zc = org[2] + (k + 0.5) * hg
        above = z > zc
        e = gmag * float((rho[above] * S.dx ** 3).sum()) / area
        if e > 0:
            meas.append(float(p[i, j, k])); exp.append(e)
    if not meas:
        return None
    return dict(measured=float(np.mean(meas)), expected=float(np.mean(exp)), ratio=float(np.mean(meas) / np.mean(exp)))


def settle(S, dt=1.0 / 24.0, max_steps=120, tol=None, quick=False, progress=None):
    """Dynamic fill: step with heat sources off until the mean liquid speed is below tol (m/s, default 0.2 sqrt(g dx)) three steps in a row, or max_steps.
    quick: level-set seeding is kept as it is, with 6 relaxing steps.  progress(step, max_steps, mean_speed) is called every step.
    Returns the report dict.  Gas parcels do not take part (they are not container-filled liquids)."""
    liq = np.flatnonzero(~S.gas[S.fid])
    if tol is None:                 # FLIP keeps a particle-noise floor of ~0.2 sqrt(g dx); below that the fluid counts as at rest
        tol = 0.2 * float(np.sqrt(np.linalg.norm(S.env.g) * S.dx))
    n_max = 6 if quick else int(max_steps)
    calm, steps, ms = 0, 0, 0.0
    lv = []
    up = -gravity_dir(S)
    it0 = S.iters
    S.iters = max(it0, 8)           # more drift-correction passes while filling: closes the half-cell wall gaps faster
    for steps in range(1, n_max + 1):
        S.step(dt, [])
        if len(liq):
            ms = float(np.sqrt((S.v[liq] ** 2).sum(1)).mean())
        if progress is not None:
            progress(steps, n_max, ms)
        lv.append(float(np.percentile(S.x[liq] @ up, 99)) if len(liq) else 0.0)
        still = len(lv) > 12 and abs(lv[-1] - lv[-12]) < 0.2 * S.dx      # the free surface has stopped moving
        calm = calm + 1 if (ms < tol and still) else 0
        if calm >= 3 and not quick:
            break
    S.iters = it0
    S.t = 0.0
    out = dict(steps=steps, converged=bool(calm >= 3) or quick, mean_speed=ms, quick=bool(quick), layers=layer_heights(S),
               overflow=overflow_report(S), pressure=hydrostatic_report(S))
    return out
