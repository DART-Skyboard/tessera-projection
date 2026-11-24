"""Aerodynamic boundary-layer friction and stagnation heating (pure NumPy, no bpy).

A surface sample with outward unit normal N in a flow of relative velocity v sees two contributions:

    stagnation   the normal (impinging) part of v is brought to rest: isentropic / recovery temperature
                 T0 = T + r |v|^2 / (2 cp) = T (1 + r (gamma-1)/2 M^2), weighted by cos^2 of the incidence
    skin friction the tangential (grazing) part shears in the boundary layer: adiabatic wall temperature
                 T_aw = T + r |v_t|^2 / (2 cp), heat flux q = h (T_aw - T_w)

Skin friction Cf is the local Blasius / 1/7-power plate law, with a Schlichting sand-grain roughness
correction and a van-Driest compressibility factor.  Stanton number follows the Chilton-Colburn
analogy St = (Cf/2) / Pr^(2/3).  At a blunt leading edge, Fay-Riddell / cylinder stagnation
Nu = 0.57 Re^{1/2} Pr^{0.37} takes over so Cf does not blow up as u_e -> 0.

    h          W/(m2 K)   = St rho |u_e| cp, floored by the stagnation correlation and natural convection
    q_frict    W/m2       = h (T_aw - T_w)          (positive into the wall when the air is hotter)
    q_diss     W/m2       = r * tau_w * |u_e|      (viscous dissipation available to the wall)
    tau_w      Pa         = Cf * 1/2 rho u_e^2

A lumped wall of areal heat capacity C_a is stepped exactly:

    C_a dT/dt = h (T_aw - T) + q_rad - eps sigma (T^4 - T_amb^4)
    T += (1 - exp(-h_tot dt / C_a)) (T_inf - T)

This is the same integrator as core/thermal.py; the aerodynamic terms only change T_aw and h.
Lateral conduction along the wall is an optional neighbour average (`smooth_field`) so the written
attributes stay smooth.  Downstream of a leading edge the heat flux falls as Re_x grows (the boundary
layer thickens) and leeward faces receive only the residual recovery of any recirculating speed.
"""
import math
import numpy as np

K_AIR = 0.0257
PR_AIR = 0.71
GAMMA_AIR = 1.4
R_AIR = 287.05287
SIGMA = 5.670374e-8
H_NATURAL = 5.0
RE_CRIT = 5.0e5


def sound_speed(T, gamma=GAMMA_AIR, cp=1005.0):
    """Isentropic speed of sound (m/s).  a = sqrt((gamma-1) cp T) = sqrt(gamma R T)."""
    T = np.asarray(T, float)
    return np.sqrt(np.maximum((gamma - 1.0) * np.maximum(cp, 1.0) * np.maximum(T, 1.0), 1e-12))


def mach_number(v, T, gamma=GAMMA_AIR, cp=1005.0):
    """Mach number |v| / a(T).  v is a speed (n,) or a vector (n,3)."""
    v = np.asarray(v, float)
    speed = np.linalg.norm(v, axis=-1) if v.ndim > 1 and v.shape[-1] == 3 else np.abs(v)
    return speed / np.maximum(sound_speed(T, gamma, cp), 1e-9)


def recovery_factor(Pr=PR_AIR, turbulent=None, Re=None):
    """Recovery factor r: Pr^{1/2} laminar, Pr^{1/3} turbulent (defaults to 0.85 when unspecified)."""
    Pr = np.asarray(Pr, float)
    Pr = np.clip(Pr, 1e-3, 1e3)
    if turbulent is None and Re is not None:
        turbulent = np.asarray(Re, float) >= RE_CRIT
    if turbulent is None:
        return np.full(np.shape(Pr), 0.85) if np.shape(Pr) else 0.85
    turb = np.asarray(turbulent, bool)
    r = np.where(turb, np.cbrt(Pr), np.sqrt(Pr))
    return r


def skin_friction_cf(Re, roughness=0.0, L=1.0, M=0.0, recovery=0.85, gamma=GAMMA_AIR):
    """Local skin-friction coefficient Cf (dimensionless).

    Smooth: Blasius 0.664/sqrt(Re) below Re_crit, 0.0592/Re^{1/5} above, floored so a stagnating
    edge does not explode.  Rough: Schlichting fully-rough plate blended in by the roughness
    Reynolds number.  Compressible: van Driest / Eckert factor (1 + r (gamma-1)/2 M^2)^{-0.35}.
    `roughness` is an equivalent sand-grain height ks in metres; 0 is hydraulically smooth.
    """
    Re = np.maximum(np.asarray(Re, float), 1.0)
    L = max(float(L), 1e-6)
    Cf_lam = 0.664 / np.sqrt(Re)
    Cf_turb = 0.0592 / Re ** 0.2
    Cf = np.where(Re < RE_CRIT, Cf_lam, Cf_turb)
    ks = max(float(roughness), 0.0)
    if ks > 0.0:
        # Schlichting fully-rough mean Cf; blend from hydraulically smooth (ks+ < 5) to rough (ks+ > 70)
        ratio = max(L / max(ks, 1e-9), 1.01)
        Cf_r = (2.87 + 1.58 * math.log10(ratio)) ** -2.5
        Cf_r = max(float(Cf_r), 1e-5)
        ks_plus = (ks / L) * Re * np.sqrt(np.maximum(Cf, 1e-12) / 2.0)
        blend = np.clip((ks_plus - 5.0) / 65.0, 0.0, 1.0)
        Cf = Cf * (1.0 - blend) + Cf_r * blend
        Cf = np.maximum(Cf, Cf_r * blend)  # roughness cannot lower Cf below the rough law once fully rough
    # van Driest compressibility (incompressible Cf is larger than the compressible one)
    M = np.asarray(M, float)
    f = (1.0 + recovery * (gamma - 1.0) / 2.0 * np.maximum(M, 0.0) ** 2) ** 0.35
    Cf = Cf / np.maximum(f, 1e-6)
    return np.clip(Cf, 1e-6, 0.5)


def stanton(Cf, Pr=PR_AIR):
    """Chilton-Colburn / Reynolds analogy: St = (Cf/2) / Pr^{2/3}."""
    Pr = np.clip(np.asarray(Pr, float), 1e-3, 1e3)
    return 0.5 * np.asarray(Cf, float) / np.power(Pr, 2.0 / 3.0)


def recovery_temperature(T, v, cp=1005.0, recovery=0.85, gamma=GAMMA_AIR):
    """Adiabatic-wall / recovery temperature (K): T + r v^2 / (2 cp) = T (1 + r (gamma-1)/2 M^2)."""
    T = np.asarray(T, float)
    v = np.asarray(v, float)
    speed2 = np.einsum('...i,...i->...', v, v) if v.ndim > 1 and v.shape[-1] == 3 else np.asarray(v, float) ** 2
    return T + np.asarray(recovery, float) * speed2 / (2.0 * np.maximum(cp, 1.0))


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True) if np.ndim(v) > 1 else abs(float(np.asarray(v)))
    return np.asarray(v, float) / np.maximum(n, 1e-12)


def decompose_flow(N, v):
    """Split relative velocity v (n,3) on outward normals N into impinging speed v_n (>=0) and tangential v_t (n,3)."""
    N = np.asarray(N, float)
    v = np.asarray(v, float)
    vn_signed = -np.einsum('ij,ij->i', N, v)          # >0 when the flow hits the face
    v_n = np.maximum(vn_signed, 0.0)
    v_t = v + N * vn_signed[:, None]                  # remove the inward-normal part
    # on a purely leeward face vn_signed < 0; leave the full velocity as residual (wake / recirculation)
    lee = vn_signed < 0.0
    v_t = np.where(lee[:, None], v, v_t)
    return v_n, v_t, vn_signed


def aero_evaluate(P, N, *, v_inf, T_inf, rho, mu, cp, k, L, roughness=0.0, recovery=None,
                  gamma=GAMMA_AIR, v_local=None, T_local=None, T_wall=None, Pr=None):
    """Adiabatic-wall temperature, heat-transfer coefficient and friction heat flux at surface samples.

    Free-stream (`v_inf`, `T_inf`, `rho`, `mu`, `cp`, `k`) is always used.  Optional `v_local` / `T_local`
    (n,3) / (n,) override the edge velocity and static temperature when a particle probe is available.
    `T_wall` (n,), when given, yields the convective flux q = h (T_aw - T_wall).

    Returns a dict of (n,) arrays: T_aw, h, q, q_diss, Cf, St, Mach, Re, v_t, v_n, impinge, graze, recovery, Pr.
    """
    P = np.asarray(P, float)
    N = np.asarray(N, float)
    n = len(P)
    if n == 0:
        z = np.zeros(0)
        return dict(T_aw=z, h=z, q=z, q_diss=z, Cf=z, St=z, Mach=z, Re=z, v_t=z, v_n=z,
                    impinge=z, graze=z, recovery=z, Pr=z, tau=z, speed=z)
    vi = np.asarray(v_inf, float).reshape(3)
    v = np.broadcast_to(np.asarray(v_local, float), (n, 3)) if v_local is not None else np.broadcast_to(vi, (n, 3))
    T = np.broadcast_to(np.asarray(T_local if T_local is not None else T_inf, float), (n,)).copy()
    rho = np.broadcast_to(np.asarray(rho, float), (n,)).copy()
    mu = np.broadcast_to(np.asarray(mu, float), (n,)).copy()
    cp = np.broadcast_to(np.asarray(cp, float), (n,)).copy()
    k = np.broadcast_to(np.asarray(k, float), (n,)).copy()
    Pr = np.broadcast_to(np.asarray(Pr if Pr is not None else cp * mu / np.maximum(k, 1e-9), float), (n,))
    Pr = np.clip(Pr, 1e-3, 1e3)
    v_n, v_tang, vn_signed = decompose_flow(N, v)
    speed_t = np.linalg.norm(v_tang, axis=1)
    speed = np.linalg.norm(v, axis=1)
    # incidence weights: 1 at a normal leading edge, 0 for pure graze; leeward faces have impinge = 0
    impinge = np.clip(v_n / np.maximum(speed, 1e-9), 0.0, 1.0) ** 2
    graze = np.clip(speed_t / np.maximum(speed, 1e-9), 0.0, 1.0)
    lee = vn_signed < 0.0
    # characteristic lengths: object size L, a sharp nose / leading-edge radius, and the streamwise
    # run from the most windward sample (so q falls as the boundary layer thickens downstream)
    L = max(float(L), 1e-4)
    R_n = max(0.02 * L, 0.008)
    if n and float(np.linalg.norm(vi)) > 0:
        s = (P - P[np.argmin(P @ vi)]) @ (vi / max(float(np.linalg.norm(vi)), 1e-9))
    else:
        s = np.zeros(n)
    # leading edges (high impinge) keep a short effective x even mid-span (swept wing, inlet lip)
    Lx = np.clip(s, 0.25 * R_n, L) * (1.0 - 0.85 * impinge) + 0.25 * R_n * impinge
    Lx = np.clip(Lx, 0.25 * R_n, L)
    u_ref = np.maximum(speed_t, 0.05 * np.maximum(speed, float(np.linalg.norm(vi))))
    Re = rho * u_ref * Lx / np.maximum(mu, 1e-12)
    r = recovery_factor(Pr, Re=Re) if recovery is None else np.broadcast_to(np.asarray(recovery, float), (n,))
    M = mach_number(v, T, gamma, cp)
    Cf = skin_friction_cf(Re, roughness=roughness, L=L, M=M, recovery=r, gamma=gamma)
    St = stanton(Cf, Pr)
    # adiabatic wall: full kinetic temperature at a stagnation point, tangential only when grazing,
    # a small residual on the leeward face (recirculation / base flow, not a second stagnation)
    kinetic = r / (2.0 * np.maximum(cp, 1.0))
    T_aw = T + kinetic * (impinge * speed ** 2 + (1.0 - impinge) * speed_t ** 2)
    T_aw = np.where(lee, T + kinetic * 0.15 * speed ** 2, T_aw)
    # convection: plate analogy (h ~ x^{-1/2} / x^{-1/5}) plus Fay-Riddell / cylinder stagnation at the nose
    h_plate = St * rho * u_ref * cp
    Re_n = rho * np.maximum(speed, float(np.linalg.norm(vi))) * (2.0 * R_n) / np.maximum(mu, 1e-12)
    Nu_stag = 0.763 * np.sqrt(np.maximum(Re_n, 1.0)) * np.power(Pr, -0.6)
    h_stag = Nu_stag * k / max(R_n, 1e-6)
    h = h_plate + impinge * np.maximum(h_stag - h_plate, 0.0)
    h = np.maximum(h, H_NATURAL * np.maximum(k, 1e-6) / K_AIR)
    h = np.where(lee, h * 0.25, h)
    tau = Cf * 0.5 * rho * u_ref ** 2
    q_diss = r * tau * u_ref
    if T_wall is None:
        q = h * (T_aw - T)          # flux into a wall already at the static temperature (upper-bound heating)
    else:
        q = h * (T_aw - np.asarray(T_wall, float))
    return dict(T_aw=T_aw, h=h, q=q, q_diss=q_diss, Cf=Cf, St=St, Mach=M, Re=Re, v_t=speed_t, v_n=v_n,
                impinge=impinge, graze=graze, recovery=r, Pr=Pr, tau=tau, speed=speed)


def surface_step(Ts, P, N, dt, *, v_inf, T_inf, rho, mu, cp, k, L, C_a=4900.0, roughness=0.0,
                 recovery=None, sources=(), alpha=0.7, emissivity=0.0, speedup=1.0, ambient=None,
                 v_local=None, T_local=None, gamma=GAMMA_AIR, q_rad=None):
    """One lumped-wall step under aerodynamic friction / stagnation heating.

    `sources` is the same list of (pos, power W, radius) used by core.thermal.radiative_flux; pass
    `q_rad` directly to skip that lookup.  Returns (Ts_new, info) where info contains aero_evaluate
    plus q_conv, q_rad, T_eq, h_conv, coverage (1 when a free-stream or local flow is present).
    """
    from . import thermal as TH
    ambient = float(T_inf if ambient is None else ambient)
    info = aero_evaluate(P, N, v_inf=v_inf, T_inf=T_inf, rho=rho, mu=mu, cp=cp, k=k, L=L,
                         roughness=roughness, recovery=recovery, gamma=gamma, v_local=v_local,
                         T_local=T_local, T_wall=Ts)
    T_aw, hc = info['T_aw'], info['h']
    if q_rad is None:
        q_rad = TH.radiative_flux(P, sources, alpha, N) if sources else np.zeros(len(P))
    dts = dt * speedup
    hr = 4.0 * emissivity * SIGMA * (0.5 * (Ts + ambient)) ** 3
    ht = hc + hr
    T_inf_w = (hc * T_aw + q_rad + hr * ambient) / np.maximum(ht, 1e-9)
    a = 1.0 - np.exp(-ht * dts / max(float(C_a), 1.0))
    Tnew = Ts + a * (T_inf_w - Ts)
    info.update(dict(coverage=np.ones(len(P)), T_fluid=np.broadcast_to(np.asarray(T_local if T_local is not None else T_inf, float), (len(P),)),
                     h_conv=hc, T_eq=T_aw, q_conv=hc * (T_aw - Ts), q_rad=np.asarray(q_rad, float), q_frict=info['q']))
    return Tnew, info


def smooth_field(P, values, radius, iters=3, lam=0.45):
    """Neighbour-weighted Laplacian smoothing of a per-sample field (keeps attributes visually continuous)."""
    P = np.asarray(P, float)
    out = np.asarray(values, float).copy()
    if len(P) < 2 or radius <= 0.0 or iters <= 0:
        return out
    # uniform grid neighbour query (reuse thermal.cross_pairs when available)
    try:
        from . import thermal as TH
        I, J = TH.cross_pairs(P, P, radius)
    except Exception:
        D = np.linalg.norm(P[:, None] - P[None], axis=2)
        I, J = np.nonzero((D > 0) & (D < radius))
    if len(I) == 0:
        return out
    d = np.linalg.norm(P[I] - P[J], axis=1)
    w = np.clip(1.0 - d / radius, 0.0, None)
    # drop self pairs
    keep = I != J
    I, J, w = I[keep], J[keep], w[keep]
    if len(I) == 0:
        return out
    for _ in range(int(iters)):
        acc = np.bincount(I, w * out[J], len(P))
        wt = np.bincount(I, w, len(P))
        avg = np.where(wt > 0, acc / np.maximum(wt, 1e-12), out)
        out = (1.0 - lam) * out + lam * avg
    return out


def blackbody_rgb(T, lo=400.0, hi=3000.0):
    """sRGB-ish glow colour for a temperature in K (cold = deep red ember, hot = white).  Values in 0..1."""
    # piecewise incandescent ramp (not a CIE blackbody, but stable and shader-matched)
    t = np.clip((np.asarray(T, float) - lo) / max(hi - lo, 1.0), 0.0, 1.0)
    stops = np.array([0.0, 0.15, 0.35, 0.55, 0.75, 1.0])
    cols = np.array([
        (0.02, 0.00, 0.00),
        (0.55, 0.04, 0.00),
        (0.95, 0.22, 0.02),
        (1.00, 0.55, 0.08),
        (1.00, 0.85, 0.45),
        (1.00, 0.97, 0.92),
    ])
    return np.stack([np.interp(t, stops, cols[:, c]) for c in range(3)], axis=-1)
