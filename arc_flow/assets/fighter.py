"""Procedural tailed fighter airframe (scale model, ~1.6 m long) as a clean closed mesh, pure NumPy (no bpy).

A stand-in for the fighter wind-tunnel stills: pointed ogive nose facing -X (the tunnel wind blows toward +X),
canopy, swept wings with sharp leading edges, wing-root gloves, a single vertical tail and a low tailplane,
circular exhaust.  Origin at mid-fuselage.  build() -> dict(verts, faces, origin).
"""
import math
import numpy as np


def _volume(v, f):
    p = v[f]
    return float(np.einsum('ij,ij->', p[:, 0], np.cross(p[:, 1], p[:, 2])) / 6.0)


def _tris(rings, cap0=True, cap1=True):
    n = len(rings[0])
    V = [np.asarray(r, float) for r in rings]
    verts = np.concatenate(V)
    F = []
    for i in range(len(V) - 1):
        for j in range(n):
            a, b = i * n + j, i * n + (j + 1) % n
            c, d = (i + 1) * n + (j + 1) % n, (i + 1) * n + j
            F += [(a, b, c), (a, c, d)]
    ex = []
    for use, ring_i in ((cap0, 0), (cap1, len(V) - 1)):
        if not use:
            continue
        ctr = V[ring_i].mean(0)
        ci = len(verts) + len(ex)
        ex.append(ctr)
        for j in range(n):
            a, b = ring_i * n + j, ring_i * n + (j + 1) % n
            F.append((a, b, ci) if ring_i == 0 else (b, a, ci))
    if ex:
        verts = np.concatenate([verts, np.array(ex)])
    F = np.array(F, np.int64)
    if _volume(verts, F) < 0:
        F = F[:, ::-1]
    return verts, F


def _merge(parts):
    vs, fs, off = [], [], 0
    for v, f in parts:
        vs.append(v)
        fs.append(f + off)
        off += len(v)
    return np.concatenate(vs), np.concatenate(fs)


def _ellipse(a, b, n=24, z=0.0, x=0.0, twist=0.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False) + twist
    return np.stack([np.full(n, x), a * np.cos(t), b * np.sin(t) + z], axis=1)


def _naca(c, t=0.10, n=14, x=0.0, y0=0.0, z=0.0, sweep_x=0.0, scale=1.0):
    """Sharp-nosed NACA-like section in the XZ plane, extruded toward +Y (then mirrored)."""
    u = np.linspace(0, 1, n)
    yt = 5 * t * (0.2969 * np.sqrt(u) - 0.1260 * u - 0.3516 * u * u + 0.2843 * u ** 3 - 0.1015 * u ** 4)
    # leading edge at smaller x (into the wind), trailing edge downstream
    xu = -(u * c) + sweep_x + x
    # close the section: upper then lower reversed
    yu = yt * c * scale
    yl = -yt * c * scale
    upper = np.stack([xu, np.full(n, y0), yu + z], axis=1)
    lower = np.stack([xu[::-1][1:-1], np.full(n - 2, y0), yl[::-1][1:-1] + z], axis=1)
    return np.concatenate([upper, lower], axis=0)


def fuselage(n=20):
    # stations from nose (-0.85) to exhaust (+0.75)
    xs = np.array([-0.85, -0.78, -0.68, -0.52, -0.35, -0.18, 0.00, 0.18, 0.38, 0.55, 0.68, 0.75])
    # radius y, z at each station
    ry = np.array([0.008, 0.030, 0.055, 0.075, 0.090, 0.095, 0.095, 0.090, 0.082, 0.070, 0.055, 0.040])
    rz = np.array([0.008, 0.028, 0.050, 0.070, 0.082, 0.088, 0.090, 0.085, 0.075, 0.062, 0.048, 0.036])
    cz = np.array([0.000, 0.000, 0.000, 0.000, 0.002, 0.004, 0.004, 0.002, 0.000, 0.000, 0.000, 0.000])
    rings = [_ellipse(a, b, n=n, z=z, x=x) for x, a, b, z in zip(xs, ry, rz, cz)]
    return _tris(rings, cap0=True, cap1=True)


def canopy(n=16):
    # blister on top of the forward fuselage
    xs = np.array([-0.42, -0.32, -0.18, -0.04])
    ry = np.array([0.020, 0.042, 0.048, 0.022])
    rz = np.array([0.010, 0.032, 0.040, 0.016])
    z0 = 0.072
    rings = [_ellipse(a, b, n=n, z=z0 + b * 0.15, x=x) for x, a, b in zip(xs, ry, rz)]
    return _tris(rings, cap0=True, cap1=True)


def _wing_half(sign=1.0, root_c=0.42, tip_c=0.16, span=0.72, sweep=0.38, z=0.0, t=0.10, nsec=8, nchord=12):
    stations = []
    for i in range(nsec):
        u = i / (nsec - 1)
        y = sign * (0.08 + u * span)
        c = root_c * (1 - u) + tip_c * u
        sw = sweep * u
        z_ = z - 0.01 * u
        stations.append(_naca(c, t=t * (1 - 0.3 * u), n=nchord, x=-0.02, y0=y, z=z_, sweep_x=sw, scale=1.0))
    return _tris(stations, cap0=False, cap1=True)


def wings():
    return _merge([_wing_half(1.0), _wing_half(-1.0)])


def tailplane():
    return _merge([
        _wing_half(1.0, root_c=0.18, tip_c=0.08, span=0.22, sweep=0.12, z=0.02, t=0.08, nsec=5, nchord=10),
        _wing_half(-1.0, root_c=0.18, tip_c=0.08, span=0.22, sweep=0.12, z=0.02, t=0.08, nsec=5, nchord=10),
    ])


def _shift(v, f, dx=0, dy=0, dz=0):
    return v + np.array([dx, dy, dz]), f


def vertical_tail():
    # fin in the XZ plane, standing on the rear fuselage
    nsec, nchord = 6, 10
    stations = []
    for i in range(nsec):
        u = i / (nsec - 1)
        z = 0.06 + u * 0.28
        c = 0.28 * (1 - 0.55 * u)
        sw = 0.22 * u
        # rotate a NACA section from XZ-horizontal to a vertical fin (section in XY, standing in Z)
        sec = _naca(c, t=0.10, n=nchord, x=0.42, y0=0.0, z=0.0, sweep_x=sw, scale=0.85)
        # currently y is constant (span), z is thickness; swap y-thickness / z-span
        # _naca: (x, y=span, z=thickness).  For a fin we want (x, y=thickness, z=span)
        p = sec.copy()
        p[:, 1] = sec[:, 2]          # thickness -> Y
        p[:, 2] = z                  # span -> Z
        stations.append(p)
    return _tris(stations, cap0=False, cap1=True)


def intake_and_exhaust():
    # slight belly intake bump and a short exhaust lip (already in fuselage cap)
    xs = np.array([-0.15, 0.05, 0.22])
    rings = [_ellipse(0.04 + 0.02 * (i == 1), 0.018, n=16, z=-0.085, x=x) for i, x in enumerate(xs)]
    return _tris(rings, cap0=True, cap1=True)


def build():
    """Closed fighter airframe.  Returns dict(verts, faces, origin=(0,0,0))."""
    parts = [fuselage(), canopy(), wings()]
    # shift tailplane to the rear
    tv, tf = tailplane()
    tv = tv + np.array([0.52, 0.0, 0.01])
    parts.append((tv, tf))
    parts.append(vertical_tail())
    parts.append(intake_and_exhaust())
    v, f = _merge(parts)
    if _volume(v, f) < 0:
        f = f[:, ::-1]
    return dict(verts=v, faces=f, origin=np.zeros(3))
