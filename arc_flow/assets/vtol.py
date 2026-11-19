"""Procedural tilt-rotor VTOL aircraft (scale model, 1.2 m long) as clean closed meshes, pure NumPy (no bpy).

This is a stand-in built from the layout of a tilt-rotor VTOL: the DART Skyboard game's own VTOL model was not reachable from the
repositories available to the extension (only the unrelated Skyboard.glb part exists), so the aircraft here is generated from a few
dozen numbers: a lofted fuselage, a high wing (NACA 0012 sections), twin wingtip nacelles, twin fins and a tailplane, and two
three-bladed rotors.  Nose toward -X (the wind in the tunnel blows toward +X), Z up, origin at the centre of gravity.

build() -> dict name -> dict(verts (n,3), faces (m,3) triangles, origin (3,) for parts that rotate about it), EXHAUSTS = nozzle positions."""
import math

import numpy as np

NACELLE_Y = 0.62
NACELLE_X = -0.02
NACELLE_Z = 0.075
EXHAUSTS = [(NACELLE_X + 0.20, NACELLE_Y, NACELLE_Z), (NACELLE_X + 0.20, -NACELLE_Y, NACELLE_Z)]
ROTOR_HUB = [(NACELLE_X - 0.15, NACELLE_Y, NACELLE_Z), (NACELLE_X - 0.15, -NACELLE_Y, NACELLE_Z)]


def _tris(rings, cap0=True, cap1=True):
    """Loft closed rings (list of (N,3)) into a triangle mesh with fan caps; faces oriented outward."""
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


def _volume(v, f):
    p = v[f]
    return float(np.einsum('ij,ij->', p[:, 0], np.cross(p[:, 1], p[:, 2])) / 6.0)


def _merge(parts):
    vs, fs, off = [], [], 0
    for v, f in parts:
        vs.append(v); fs.append(f + off); off += len(v)
    return np.concatenate(vs), np.concatenate(fs)


def _superellipse(a, b, n=20, p=2.4):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    return np.stack([a * np.sign(c) * np.abs(c) ** (2.0 / p), b * np.sign(s) * np.abs(s) ** (2.0 / p)], axis=1)


def fuselage():
    L = 1.2
    ts = np.concatenate([np.linspace(0, 0.2, 9), np.linspace(0.2, 0.62, 9)[1:], np.linspace(0.62, 1.0, 9)[1:]])
    rings = []
    for t in ts:
        if t < 0.2:
            u = t / 0.2
            s = math.sqrt(max(1 - (1 - u) ** 2, 0.0))
        elif t < 0.62:
            s = 1.0
        else:
            u = (t - 0.62) / 0.38
            s = 1.0 - 0.72 * (u ** 1.5)
        s = max(s, 0.02)
        a, b = 0.085 * s, 0.075 * s
        zc = 0.0 if t < 0.62 else 0.07 * ((t - 0.62) / 0.38) ** 2
        zc += -0.012 * (1 - min(t / 0.2, 1.0)) if t < 0.2 else 0.0
        ring = _superellipse(a, b)
        x = -L / 2 + t * L
        rings.append(np.stack([np.full(len(ring), x), ring[:, 0], ring[:, 1] + zc], axis=1))
    return _tris(rings)


def _naca(m=9, t=0.12):
    x = np.linspace(0, 1, m + 1) ** 1.6
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4)
    up = np.stack([x[::-1], yt[::-1]], axis=1)
    lo = np.stack([x[1:-1], -yt[1:-1]], axis=1)
    return np.concatenate([up, lo])


def _surface(root_le, tip_le, root_c, tip_c, span_dir, n=8, t=0.12, twist=0.0, round_tip=True):
    """Lofted lifting surface from root to tip.  Sections lie in the (x, z) plane (span along y) or (x, y) (span along z) as span_dir says."""
    af = _naca(9, t)
    rings = []
    for k in range(n + 1):
        u = k / n
        le = np.asarray(root_le, float) * (1 - u) + np.asarray(tip_le, float) * u
        c = root_c * (1 - u) + tip_c * u
        if round_tip and k == n:
            c *= 0.45
        sx = -(af[:, 0] - 0.25) * c
        cz = af[:, 1] * c
        ang = twist * u
        xx = sx * math.cos(ang) - cz * math.sin(ang)
        zz = sx * math.sin(ang) + cz * math.cos(ang)
        if span_dir == 'Y':
            ring = np.stack([le[0] - xx, np.full(len(xx), le[1]), le[2] + zz], axis=1)
        else:
            ring = np.stack([le[0] - xx, le[1] + zz, np.full(len(xx), le[2])], axis=1)
        rings.append(ring)
    return _tris(rings)


def wing():
    parts = []
    for sgn in (1, -1):
        root = (0.0 + 0.06, sgn * 0.04, 0.075)
        tip = (0.0 + 0.10, sgn * (NACELLE_Y + 0.02), 0.085)
        parts.append(_surface(root, tip, 0.34, 0.20, 'Y', n=10, t=0.12))
    return _merge(parts)


def nacelles():
    parts = []
    for sgn in (1, -1):
        rings = []
        xs = np.linspace(-0.14, 0.17, 12)
        for x in xs:
            u = (x - xs[0]) / (xs[-1] - xs[0])
            r = 0.042 * (math.sqrt(max(1 - ((0.12 - u) / 0.12) ** 2, 0.0)) if u < 0.12 else 1.0) * (1.0 - 0.45 * max(u - 0.7, 0) / 0.3)
            r = max(r, 0.004)
            ring = _superellipse(r, r, 16, 2.0)
            rings.append(np.stack([np.full(len(ring), NACELLE_X + x), sgn * NACELLE_Y + ring[:, 0], NACELLE_Z + ring[:, 1]], axis=1))
        parts.append(_tris(rings))
    return _merge(parts)


def tail():
    parts = []
    x0 = 0.47
    for sgn in (1, -1):
        parts.append(_surface((x0 - 0.02, sgn * 0.17, 0.09), (x0 + 0.08, sgn * 0.17, 0.30), 0.19, 0.10, 'Z', n=8, t=0.10))
    # tailplane between the fins (and a little beyond)
    parts.append(_surface((x0 + 0.04, -0.20, 0.20), (x0 + 0.04, 0.20, 0.20), 0.13, 0.13, 'Y', n=6, t=0.09, round_tip=False))
    return _merge(parts)


def canopy():
    rings = []
    xs = np.linspace(-0.43, -0.17, 9)
    for x in xs:
        u = (x - xs[0]) / (xs[-1] - xs[0])
        s = math.sqrt(max(1 - (2 * u - 1) ** 2 * 0.98, 0.0))
        ring = _superellipse(0.06 * s + 0.002, 0.050 * s + 0.002, 14, 2.2)
        rings.append(np.stack([np.full(len(ring), x), ring[:, 0], 0.045 + ring[:, 1]], axis=1))
    return _tris(rings)


def airframe():
    return _merge([fuselage(), wing(), nacelles(), tail(), canopy()])


def rotor(side):
    """Three tapered, twisted blades and a spinner around the hub; origin returned is the hub (rotation axis +X)."""
    hub = np.array(ROTOR_HUB[0 if side > 0 else 1], float)
    parts = []
    for k in range(3):
        phi = 2 * math.pi * k / 3 + (0.0 if side > 0 else math.pi / 3)
        rings = []
        for u in np.linspace(0.12, 1.0, 9):
            r = 0.20 * u
            chord = 0.034 * (1 - 0.55 * u) + 0.006
            thick = 0.004 + 0.0045 * (1 - u)
            tw = math.radians(34 - 24 * u)
            pts = np.array([[-chord / 2, -thick], [chord / 2, -thick * 0.6], [chord / 2, thick * 0.6], [-chord / 2, thick]])
            pts = pts @ np.array([[math.cos(tw), -math.sin(tw)], [math.sin(tw), math.cos(tw)]]).T
            ring = np.stack([pts[:, 1] * 1.0, np.full(4, 0.0) + 0 * pts[:, 0], np.zeros(4)], axis=1)
            # blade: chord across y (tangential), thickness along x (axial); spans radially in the (y, z) plane at angle phi
            tang = pts[:, 0]; ax = pts[:, 1]
            y = r * math.cos(phi) - tang * math.sin(phi)
            z = r * math.sin(phi) + tang * math.cos(phi)
            rings.append(np.stack([ax, y, z], axis=1))
        parts.append(_tris(rings))
    sp = []
    xs = np.linspace(-0.045, 0.01, 8)
    for x in xs:
        u = (x - xs[0]) / (xs[-1] - xs[0])
        r = 0.026 * (math.sqrt(max(1 - ((0.9 - u) / 0.9) ** 2, 0.0)) if u < 0.9 else 1.0)
        r = max(r, 0.003)
        ring = _superellipse(r, r, 12, 2.0)
        sp.append(np.stack([np.full(len(ring), x), ring[:, 0], ring[:, 1]], axis=1))
    parts.append(_tris(sp))
    v, f = _merge(parts)
    return v, f, hub


def build():
    out = {}
    v, f = airframe()
    out['Airframe'] = dict(verts=v, faces=f, origin=np.zeros(3))
    for side, name in ((1, 'Rotor R'), (-1, 'Rotor L')):
        v, f, hub = rotor(side)
        out[name] = dict(verts=v + hub, faces=f, origin=hub)
    return out
