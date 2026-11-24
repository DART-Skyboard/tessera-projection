"""Thermal readback of collision-body surfaces (pure NumPy, no bpy).

Every surface sample (a mesh vertex or a face centre) of a selected body carries a wall temperature T_s.  Each step the
solver's particles around the sample are probed with a smooth kernel; the wall then relaxes toward the local
*recovery temperature* of the fluid with a convective heat-transfer coefficient, and absorbs radiation from heat sources:

    probe      T_f, v_f, cp, k, rho, mu = kernel-weighted means of the particles within h of the sample (w = (1 - d/h)^2)
    coverage   c = min(1, sum(w) / w_ref)   (w_ref = weight of a fully packed half space; sparse gas parcels give c < 1)
    stagnation windward faces add r |v_inf|^2 cos^2 / 2cp (free stream v_inf = the Environment wind)
    recovery   T_r = T_f + r |v_f - v_wall|^2 / (2 cp)                       r = 0.85 (turbulent boundary layer, Pr^(1/3))
    target     T_eq = c T_r + (1 - c) T_ambient
    h_conv     flat plate: Nu = 0.664 Re^(1/2) Pr^(1/3) (Re < 5e5), 0.037 Re^(4/5) Pr^(1/3) above, Re = rho |v_rel| L / mu;
               floor: natural convection 5 W/(m2 K) scaled by k / k_air.  h = max(Nu k / L, floor) * c
    radiation  q = alpha P / (4 pi d^2) for every heat source P at distance d (cooling sources give negative q)
    update     C_a dT/dt = h (T_eq - T) + q               C_a = areal heat capacity of the wall (J/(m2 K)), speed-up applied to dt
               integrated exactly for one step: T += (1 - exp(-h dt / C_a)) (T_eq + q / h - T)

A coarse simplification, not conjugate heat transfer: the wall is one lumped layer, lateral conduction along the wall is ignored.
`patch_sources` converts the exchanged heat into heat sources for the fluid (the heat the body gives to / takes from the air)."""
import numpy as np

K_AIR = 0.0257
SIGMA = 5.670374e-8
H_NATURAL = 5.0

RAMPS = {
    'THERMAL': [(0.00, (0.02, 0.05, 0.55)), (0.17, (0.00, 0.35, 0.95)), (0.33, (0.00, 0.80, 0.90)), (0.48, (0.10, 0.80, 0.20)),
                (0.62, (0.95, 0.90, 0.05)), (0.76, (1.00, 0.50, 0.00)), (0.90, (0.90, 0.07, 0.03)), (1.00, (1.00, 0.95, 0.90))],
    'INFERNO': [(0.00, (0.001, 0.000, 0.014)), (0.20, (0.258, 0.039, 0.406)), (0.40, (0.578, 0.148, 0.404)),
                (0.60, (0.865, 0.317, 0.226)), (0.80, (0.988, 0.645, 0.040)), (1.00, (0.988, 1.000, 0.645))],
    'TURBO': [(0.00, (0.190, 0.072, 0.232)), (0.17, (0.275, 0.440, 0.900)), (0.35, (0.100, 0.800, 0.800)),
              (0.55, (0.400, 0.950, 0.300)), (0.75, (0.980, 0.760, 0.150)), (1.00, (0.640, 0.050, 0.020))],
    'COLD_HOT': [(0.00, (0.050, 0.200, 0.800)), (0.35, (0.300, 0.650, 0.950)), (0.50, (0.900, 0.900, 0.880)),
                 (0.70, (0.960, 0.600, 0.200)), (1.00, (0.800, 0.050, 0.030))],
    'GRAY': [(0.00, (0.02, 0.02, 0.02)), (1.00, (1.0, 1.0, 1.0))],
}
RAMP_ITEMS = [('THERMAL', "Thermal", "Deep blue, cyan, green, yellow, orange, red, white-hot"), ('INFERNO', "Inferno", "Black, purple, orange, pale yellow"), ('TURBO', "Turbo", "Blue, green, yellow, red"),
              ('COLD_HOT', "Cold to hot", "Blue, white, red"), ('GRAY', "Grayscale", "")]


def ramp_color(t, ramp='THERMAL'):
    """Colour (n, 3) for normalised values t in [0, 1] (piecewise linear, same stops as the shader ramp)."""
    t = np.clip(np.asarray(t, float), 0.0, 1.0)
    st = RAMPS[ramp]
    xs = np.array([s[0] for s in st])
    return np.stack([np.interp(t, xs, np.array([s[1][c] for s in st])) for c in range(3)], axis=-1)


def normalise(T, lo, hi):
    return np.clip((np.asarray(T, float) - lo) / max(hi - lo, 1e-9), 0.0, 1.0)


# ----------------------------------------------------------------------------- neighbour query (surface samples against particles)
def cross_pairs(P, X, h):
    """All (i, j) with |P_i - X_j| < h (uniform grid of X, 27 cell offsets)."""
    P, X = np.asarray(P, float), np.asarray(X, float)
    e = np.zeros(0, np.int64)
    if len(P) == 0 or len(X) == 0:
        return e, e
    lo = np.minimum(P.min(0), X.min(0)) - h
    cx = np.floor((X - lo) / h).astype(np.int64) + 1
    cp = np.floor((P - lo) / h).astype(np.int64) + 1
    dims = np.maximum(cx.max(0), cp.max(0)) + 3
    kx = (cx[:, 0] * dims[1] + cx[:, 1]) * dims[2] + cx[:, 2]
    kp = (cp[:, 0] * dims[1] + cp[:, 1]) * dims[2] + cp[:, 2]
    order = np.argsort(kx, kind='stable')
    sk, sx = kx[order], X[order]
    ip = np.arange(len(P))
    I, J = [], []
    h2 = h * h
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            for c in (-1, 0, 1):
                nk = kp + (a * dims[1] + b) * dims[2] + c
                lo_ = np.searchsorted(sk, nk, 'left')
                hi_ = np.searchsorted(sk, nk, 'right')
                cnt = hi_ - lo_
                tot = int(cnt.sum())
                if tot == 0:
                    continue
                ia = np.repeat(ip, cnt)
                st = np.cumsum(cnt) - cnt
                ja = np.arange(tot) - np.repeat(st, cnt) + np.repeat(lo_, cnt)
                d = sx[ja] - P[ia]
                m = np.einsum('ij,ij->i', d, d) < h2
                I.append(ia[m]); J.append(order[ja[m]])
    if not I:
        return e, e
    return np.concatenate(I), np.concatenate(J)


_REF = {}


def reference_weight(h, dx):
    """Kernel weight sum of a fully packed lattice (spacing dx) seen from a sample in the first particle layer next to a flat wall."""
    key = round(h / dx, 4)
    if key not in _REF:
        m = int(np.ceil(h / dx)) + 1
        g = np.arange(-m, m + 1) * dx
        L = np.array([[a, b, c] for a in g for b in g for c in g if c >= 0.0])
        d = np.linalg.norm(L, axis=1)
        _REF[key] = max(float(np.where(d < h, (1.0 - d / h) ** 2, 0.0).sum()), 1.0)
    return _REF[key]


def probe(P, X, V, T, cp, k, rho, mu, h, dx):
    """Kernel-weighted fluid state at the sample points P.  Returns dict(coverage, T, v, cp, k, rho, mu, wsum)."""
    n = len(P)
    I, J = cross_pairs(P, X, h)
    out = dict(coverage=np.zeros(n), T=np.zeros(n), v=np.zeros((n, 3)), cp=np.full(n, 1005.0), k=np.full(n, K_AIR),
               rho=np.zeros(n), mu=np.full(n, 1.8e-5), wsum=np.zeros(n))
    if len(I) == 0:
        return out
    d = np.linalg.norm(P[I] - X[J], axis=1)
    w = (1.0 - d / h) ** 2
    ws = np.bincount(I, w, n)
    inv = 1.0 / np.maximum(ws, 1e-12)
    has = ws > 0
    out['wsum'] = ws
    out['coverage'] = np.clip(ws / reference_weight(h, dx), 0.0, 1.0)
    out['T'] = np.where(has, np.bincount(I, w * T[J], n) * inv, 0.0)
    for c in range(3):
        out['v'][:, c] = np.where(has, np.bincount(I, w * V[J, c], n) * inv, 0.0)
    for key, arr in (('cp', cp), ('k', k), ('rho', rho), ('mu', mu)):
        out[key] = np.where(has, np.bincount(I, w * arr[J], n) * inv, out[key])
    return out


def convection_coefficient(rho, vrel, mu, cp, k, L):
    """Flat-plate forced convection with a natural-convection floor (W/(m2 K))."""
    Re = rho * np.abs(vrel) * L / np.maximum(mu, 1e-12)
    Pr = cp * mu / np.maximum(k, 1e-9)
    pr3 = np.cbrt(np.clip(Pr, 1e-3, 1e3))
    Nu = np.where(Re < 5e5, 0.664 * np.sqrt(Re) * pr3, 0.037 * Re ** 0.8 * pr3)
    return np.maximum(Nu * k / max(L, 1e-6), H_NATURAL * np.maximum(k, 1e-6) / K_AIR)


def radiative_flux(P, sources, alpha=0.7, n=None, d_min=0.02):
    """W/m2 absorbed at the points P from point heat sources [(pos, power W, radius)]; the source radius softens the singularity.
    n (optional unit normals): only the side facing the source receives (cosine factor)."""
    q = np.zeros(len(P))
    for pos, power, rad in sources:
        r = P - np.asarray(pos, float)
        d = np.maximum(np.linalg.norm(r, axis=1), max(d_min, float(rad)))
        f = alpha * power / (4.0 * np.pi * d * d)
        if n is not None:
            dirs = -r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-9)
            f = f * np.clip(np.einsum('ij,ij->i', n, dirs), 0.0, 1.0)
        q += f
    return q


def surface_step(Ts, P, N, X, V, T, cp, k, rho, mu, dt, *, h, dx, ambient, L, C_a=4900.0, recovery=0.85, sources=(), alpha=0.7,
                 v_wall=None, speedup=1.0, emissivity=0.0, v_inf=None, aero=False, roughness=0.0):
    """One thermal step for the sample temperatures Ts (n,), positions P (n,3) already offset off the wall, wall normals N.
    Returns (Ts_new, info) with info = dict(coverage, T_fluid, h_conv, T_eq, q_conv W/m2, q_rad W/m2, q_frict W/m2).
    `aero`: also apply boundary-layer friction / leading-edge stagnation heating from core.friction_heat (free-stream
    v_inf plus any local particle flow); q_frict is the aerodynamic heat flux into the wall."""
    pr = probe(P, X, V, T, cp, k, rho, mu, h, dx)
    vw = np.zeros((len(P), 3)) if v_wall is None else v_wall
    vrel = np.linalg.norm(pr['v'] - vw, axis=1)
    c = pr['coverage']
    Tr = pr['T'] + recovery * vrel ** 2 / (2.0 * np.maximum(pr['cp'], 1.0))
    Teq = np.where(pr['wsum'] > 0, c * Tr + (1.0 - c) * ambient, ambient)
    hc = convection_coefficient(np.where(pr['rho'] > 0, pr['rho'], 1.2), vrel, pr['mu'], pr['cp'], pr['k'], L)
    hc = np.where(pr['wsum'] > 0, hc * np.maximum(c, 0.05), H_NATURAL * 0.1)
    if v_inf is not None and np.linalg.norm(v_inf) > 0:
        # stagnation heating of the windward surfaces: the free stream is brought to rest at the wall, cos^2 of the incidence angle
        vi = np.asarray(v_inf, float)
        face = np.clip(-(N @ (vi / np.linalg.norm(vi))), 0.0, 1.0) ** 2
        Teq = Teq + face * recovery * float(vi @ vi) / (2.0 * np.maximum(pr['cp'], 1.0))
    q_frict = hc * (Teq - Ts)
    if aero and v_inf is not None and np.linalg.norm(v_inf) > 0:
        from . import friction_heat as FH
        rho_a = np.where(pr['rho'] > 0, pr['rho'], 1.225)
        ae = FH.aero_evaluate(P, N, v_inf=v_inf, T_inf=ambient, rho=rho_a, mu=pr['mu'], cp=pr['cp'], k=pr['k'], L=L,
                              roughness=roughness, recovery=recovery,
                              T_local=np.where(pr['wsum'] > 0, pr['T'], ambient), T_wall=Ts)
        Teq = np.maximum(Teq, ae['T_aw'])
        hc = np.maximum(hc, ae['h'])
        q_frict = ae['q']
    qr = radiative_flux(P, sources, alpha, N)
    dts = dt * speedup
    # the wall re-radiates: linearised  h_rad = 4 eps sigma T_m^3  toward the ambient temperature (caps the heating by close sources)
    hr = 4.0 * emissivity * SIGMA * (0.5 * (Ts + ambient)) ** 3
    ht = hc + hr
    T_inf = (hc * Teq + qr + hr * ambient) / np.maximum(ht, 1e-9)
    a = 1.0 - np.exp(-ht * dts / max(C_a, 1.0))
    Tnew = Ts + a * (T_inf - Ts)
    return Tnew, dict(coverage=c, T_fluid=pr['T'], h_conv=hc, T_eq=Teq, q_conv=hc * (Teq - Ts), q_rad=qr, q_frict=q_frict)


# ----------------------------------------------------------------------------- body -> fluid heat
def patch_sources(P, area, Ts, info, cell, min_power=1e-3, max_patches=24):
    """Heat the body exchanges with the fluid, gathered into patches of size `cell`: [(centre, power W, radius)].
    Power is positive where the wall heats the fluid (wall hotter than the local fluid), negative where it cools it."""
    if len(P) == 0:
        return []
    cov = info['coverage'] > 0.05
    if not cov.any():
        return []
    q = info['h_conv'] * (Ts - info['T_fluid']) * area
    q = np.where(cov, q, 0.0)
    key = np.floor(P / cell).astype(np.int64)
    _u, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    pw = np.bincount(inv, q)
    ctr = np.stack([np.bincount(inv, np.abs(q) * P[:, c]) for c in range(3)], axis=1) / np.maximum(np.bincount(inv, np.abs(q)), 1e-12)[:, None]
    order = np.argsort(-np.abs(pw))[:max_patches]
    return [(tuple(ctr[i]), float(pw[i]), float(cell)) for i in order if abs(pw[i]) > min_power]
