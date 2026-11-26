"""Layered dress regression: three dress layers (lining, skirt, overskirt) solved as ONE cloth on a mannequin
(torso / hips / legs colliders) with a gusting fan jet and a swaying, turning waist, for 300+ frames.

Counts per frame, with metrics that do not use the solver's own side bookkeeping (tests/cloth_layer_metrics.py):
  * layer crossings     - edges of one layer piercing a triangle of another layer (segment / triangle test)
  * self-intersections  - edges of a layer piercing a non-adjacent triangle of the same layer
  * wrong-side vertices - vertices of an outer layer on the inner side of the nearest face of the layer below
and reports them per frame range (1-100, 101-200, 201-300).

usage:  python3 tests/test_cloth_layered_300.py [--frames 300] [--mode LAYERED] [--res 48x20] [--arc PATH]
                                                [--json out.json] [--quick]
--quick runs a lighter 120-frame variant and asserts zero crossings (used by the regular test run).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _setup(arc):
    sys.path.insert(0, arc)
    sys.path.insert(0, HERE)


def ring_grid(nt, nz, r, z0, length, flare=0.0, bodice=0.0):
    """Open tube (rows top -> hem), outward normals.  ``bodice`` narrows the top 25 % toward a waist."""
    th = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    V = []
    for k, f in enumerate(np.linspace(0, 1, nz)):
        rr = r * (1 + flare * max(f - 0.15, 0.0) / 0.85)
        if bodice > 0 and f < 0.25:
            rr *= 1.0 - bodice * np.sin(np.pi * f / 0.25)
        for t in th:
            V.append((rr * np.cos(t), rr * np.sin(t), z0 - f * length))
    T = []
    for k in range(nz - 1):
        for i in range(nt):
            a = k * nt + i
            b = k * nt + (i + 1) % nt
            c, d = a + nt, b + nt
            T += [(a, d, b), (a, c, d)]          # outward normals
    return np.array(V, float), np.array(T, np.int64)


def mannequin():
    from core import primitives as PR
    parts = [PR.tube(n=40, r0=0.150, r1=0.135, L=0.30, center=(0, 0, 1.22), axis='z'),   # chest / waist
             PR.tube(n=40, r0=0.180, r1=0.150, L=0.22, center=(0, 0, 0.97), axis='z'),   # hips
             PR.tube(n=24, r0=0.075, r1=0.060, L=0.85, center=(0.085, 0, 0.43), axis='z'),
             PR.tube(n=24, r0=0.075, r1=0.060, L=0.85, center=(-0.085, 0, 0.43), axis='z')]
    V, T, off = [], [], 0
    for v, t in parts:
        V.append(np.asarray(v, float)); T.append(np.asarray(t) + off); off += len(v)
    return np.concatenate(V), np.concatenate(T)


def build(nt=48, nz=20, mode='LAYERED', thickness=0.004):
    from core import cloth_solver as CS
    top = 1.36
    parts = [ring_grid(nt, nz, 0.160, top, 0.95, 0.75, 0.10),    # lining (inner), longest
             ring_grid(nt, nz, 0.169, top, 0.85, 0.95, 0.10),    # skirt
             ring_grid(nt, nz, 0.178, top, 0.70, 1.20, 0.10)]    # overskirt (outer), shortest, fullest
    V, T, lay, off = [], [], [], 0
    for li, (v, t) in enumerate(parts):
        V.append(v); T.append(t + off); lay.append(np.full(len(v), li)); off += len(v)
    V = np.concatenate(V); T = np.concatenate(T); lay = np.concatenate(lay)
    pins = np.nonzero(np.isclose(V[:, 2], top))[0]
    S = CS.ClothSolver(V, T, pins=pins, preset='SILK', quality='BALANCED', self_collision=True,
                       self_collision_mode=mode, self_thickness=thickness)
    S.self_thickness = thickness
    bv, bt = mannequin()
    S.set_collider_mesh(bv, bt, margin=0.006, tessellate=False)
    return S, V, T, lay, pins


def fan(t):
    """Gusting fan jet from -X (with a slow sideways drift), stronger near hip height; callable wind field."""
    gust = 1.0 + 0.6 * np.sin(2 * np.pi * 0.31 * t) + 0.3 * np.sin(2 * np.pi * 1.7 * t)
    side = 0.8 * np.cos(2 * np.pi * 0.23 * t)

    def field(P):
        h = np.exp(-((P[:, 2] - 0.8) / 0.45) ** 2)
        w = np.zeros_like(P)
        w[:, 0] = 3.2 * gust * h
        w[:, 1] = side * h
        w[:, 2] = 0.4 * gust * h
        return w
    return field


def run(frames=300, mode='LAYERED', nt=48, nz=20, verbose=True, every=1):
    import cloth_layer_metrics as M
    S, V, T, lay, pins = build(nt, nz, mode)
    pairs = [(0, 1), (1, 2)]
    ref = M.side_signs(S.x, T, lay, pairs)
    edges = M.mesh_edges(T)
    rest_pins = V[pins].copy()
    dt = 1.0 / 24.0
    cross, selfx, wrong = {}, {}, {}
    t0 = time.time()
    for f in range(1, frames + 1):
        t = f * dt
        ang = 0.9 * np.sin(2 * np.pi * 0.35 * t)                 # +-52 deg turn (mannequin turntable / walk)
        sway = np.array([0.05 * np.sin(2 * np.pi * 0.45 * t), 0.035 * np.sin(2 * np.pi * 0.9 * t), 0.0])
        c, s = np.cos(ang), np.sin(ang)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        S.set_pin_targets(rest_pins @ R.T + sway, pins, dt=dt)
        S.step(dt, wind=fan(t))
        if not np.isfinite(S.x).all():
            raise RuntimeError("non-finite positions at frame %d" % f)
        if f % every == 0:
            a, b = M.edge_triangle_intersections(S.x, T, lay, edges)
            cross[f], selfx[f] = a, b
            wrong[f] = sum(M.wrong_side_vertices(S.x, T, lay, pairs, ref_sign=ref).values())
            if verbose and (f % 10 == 0 or a or b or wrong[f]):
                print("frame %3d  layer-crossing edges %3d  self-intersecting edges %3d  wrong-side verts %3d  "
                      "(%.0f ms/step, %.0f s)" % (f, a, b, wrong[f], S.last_ms, time.time() - t0), flush=True)
    return dict(cross=cross, selfx=selfx, wrong=wrong, n=int(S.n), secs=time.time() - t0)


def report(res, frames):
    import cloth_layer_metrics as M
    hi = max(frames, 300)
    ranges = [(1, 100), (101, 200), (201, 300)] + ([(301, hi)] if hi > 300 else [])
    print("\nrange      | layer crossings (frames>0 / max / sum) | self-intersections | wrong-side verts")
    for (lo, hi_), a, b, c in zip(ranges, M.range_table(res['cross'], ranges), M.range_table(res['selfx'], ranges),
                                  M.range_table(res['wrong'], ranges)):
        print("%4d-%-4d  | %3d / %3d / %5d                     | %3d / %3d / %5d  | %3d / %3d / %5d"
              % (lo, hi_, a[2], a[3], a[4], b[2], b[3], b[4], c[2], c[3], c[4]))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=300)
    ap.add_argument('--mode', default='LAYERED')
    ap.add_argument('--res', default='48x20')
    ap.add_argument('--arc', default=os.path.join(os.path.dirname(HERE), 'arc_flow'))
    ap.add_argument('--json', default='')
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    _setup(a.arc)
    if a.quick:
        a.frames, a.res = 120, '32x14'
    nt, nz = map(int, a.res.split('x'))
    res = run(a.frames, a.mode, nt, nz)
    report(res, a.frames)
    if a.json:
        with open(a.json, 'w') as fh:
            json.dump({k: ({str(f): v for f, v in d.items()} if isinstance(d, dict) else d) for k, d in res.items()}, fh)
    total = sum(res['cross'].values()) + sum(res['selfx'].values()) + sum(res['wrong'].values())
    print("%s: %d crossing / intersection events over %d frames (%d verts, %.0f s)"
          % ("PASS" if total == 0 else "FAIL", total, a.frames, res['n'], res['secs']))
    if a.quick and total:
        sys.exit(1)
