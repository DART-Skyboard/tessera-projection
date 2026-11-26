"""Independent layer-crossing / self-intersection metrics for cloth tests (NumPy only, no bpy).

These do NOT use the solver's own side bookkeeping:

* ``edge_triangle_intersections`` - every mesh edge that geometrically pierces a triangle it does not share a vertex
  with (segment / triangle test).  Split into *cross-piece* hits (one garment layer through another = layer crossing)
  and *same-piece* hits (a layer through itself = self-intersection).
* ``wrong_side_vertices`` - for garments whose layers are built with outward normals, vertices of an outer layer that
  sit on the inner side of the nearest face of the layer directly below (closest-point sign), farther than ``tol``.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'arc_flow'))
from core import cloth_selfcol as SC  # noqa: E402


def mesh_edges(tris):
    T = np.asarray(tris, np.int64)
    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    e.sort(1)
    return np.unique(e, axis=0)


def edge_triangle_intersections(x, tris, piece, edges=None, detail=False):
    """Return (n_cross_piece_edges, n_same_piece_edges): unique edges piercing a non-adjacent triangle.
    With ``detail`` also return the (edge index, triangle index) hit arrays and the edge list."""
    x = np.asarray(x, float)
    T = np.asarray(tris, np.int64)
    E = mesh_edges(T) if edges is None else np.asarray(edges, np.int64)
    piece = np.asarray(piece)
    elen = np.linalg.norm(x[E[:, 0]] - x[E[:, 1]], axis=1)
    L = float(elen.max()) if len(elen) else 0.0
    if L <= 0:
        return (0, 0, np.zeros(0, np.int64), np.zeros(0, np.int64), E) if detail else (0, 0)
    # broad phase: an edge that pierces triangle t has both end points within its length of t
    sc = SC.SelfCollider(T, E, len(x), thickness=L)
    sc.max_pairs = 40_000_000
    v, t = sc._candidates(x, x, 0.0)
    if 'overflow' in sc.stats:
        raise RuntimeError("metric broad phase overflow")
    # expand vertex -> incident edges (edge listed from its lower end point only, to keep unique)
    order = np.argsort(E[:, 0], kind='stable')
    Es = E[order]
    start = np.searchsorted(Es[:, 0], np.arange(len(x) + 1))
    cnt = start[1:] - start[:-1]
    rep = cnt[v]
    vv = np.repeat(v, rep)
    tt = np.repeat(t, rep)
    off = np.repeat(start[v], rep) + (np.arange(rep.sum()) - np.repeat(np.cumsum(rep) - rep, rep))
    ei = order[off]
    tr = T[tt]
    a0, a1 = E[ei, 0], E[ei, 1]
    share = (tr == a0[:, None]).any(1) | (tr == a1[:, None]).any(1)
    ei, tt, tr = ei[~share], tt[~share], tr[~share]
    if len(ei) == 0:
        return (0, 0, ei, tt, E) if detail else (0, 0)
    p, q = x[E[ei, 0]], x[E[ei, 1]]
    A, B, C = x[tr[:, 0]], x[tr[:, 1]], x[tr[:, 2]]
    n = np.cross(B - A, C - A)
    dp = ((p - A) * n).sum(1)
    dq = ((q - A) * n).sum(1)
    s = (dp * dq) < 0.0
    ei, tt, p, q, A, B, C, dp, dq, n = ei[s], tt[s], p[s], q[s], A[s], B[s], C[s], dp[s], dq[s], n[s]
    if len(ei) == 0:
        return (0, 0, ei, tt, E) if detail else (0, 0)
    X = p + (dp / (dp - dq))[:, None] * (q - p)
    c0 = (np.cross(B - A, X - A) * n).sum(1)
    c1 = (np.cross(C - B, X - B) * n).sum(1)
    c2 = (np.cross(A - C, X - C) * n).sum(1)
    hit = ((c0 >= 0) & (c1 >= 0) & (c2 >= 0)) | ((c0 <= 0) & (c1 <= 0) & (c2 <= 0))
    ei, tt = ei[hit], tt[hit]
    cross = piece[E[ei, 0]] != piece[T[tt, 0]]
    res = (int(len(np.unique(ei[cross]))), int(len(np.unique(ei[~cross]))))
    if detail:
        return res + (ei, tt, E)
    return res


def wrong_side_vertices(x, tris, layer, pairs, tol=5e-4, k=8, ref_sign=None):
    """{(lower, upper): count} of upper-layer vertices on the wrong side of the nearest lower-layer face (farther than
    ``tol``).  ``ref_sign`` {(lower, upper): +1/-1} is the correct side (sign of the closest-point offset along the face
    normal); measure it on the rest pose with ``side_signs``.  Default +1 (outward normals)."""
    x = np.asarray(x, float)
    T = np.asarray(tris, np.int64)
    layer = np.asarray(layer)
    out = {}
    for lo, up in pairs:
        P = np.nonzero(layer == up)[0]
        Tl = T[np.all(layer[T] == lo, axis=1)]
        A, B, C = x[Tl[:, 0]], x[Tl[:, 1]], x[Tl[:, 2]]
        cen = (A + B + C) / 3.0
        cnt = 0
        for s in range(0, len(P), 256):
            p = x[P[s:s + 256]]
            d2 = ((p[:, None, :] - cen[None]) ** 2).sum(-1)
            near = np.argsort(d2, axis=1)[:, :k]
            best = np.full(len(p), 1e9)
            sgn = np.ones(len(p))
            for j in range(near.shape[1]):
                tt = near[:, j]
                q, _b, _i = SC.closest_point_triangle(p, A[tt], B[tt], C[tt])
                dd = np.linalg.norm(p - q, axis=1)
                nrm = np.cross(B[tt] - A[tt], C[tt] - A[tt])
                nrm /= np.maximum(np.linalg.norm(nrm, axis=1), 1e-14)[:, None]
                sd = ((p - q) * nrm).sum(1)
                m = dd < best
                best = np.where(m, dd, best)
                sgn = np.where(m, np.sign(sd + 1e-12), sgn)
            good = (ref_sign or {}).get((lo, up), 1.0)
            cnt += int(((sgn * good < 0) & (best > tol)).sum())
        out[(lo, up)] = cnt
    return out


def side_signs(x, tris, layer, pairs, k=8):
    """Majority side sign of each upper layer relative to its lower layer (call on the rest pose)."""
    out = {}
    for pr in pairs:
        pos = wrong_side_vertices(x, tris, layer, [pr], tol=0.0, k=k, ref_sign={pr: -1.0})[pr]   # count of +side
        neg = wrong_side_vertices(x, tris, layer, [pr], tol=0.0, k=k, ref_sign={pr: 1.0})[pr]    # count of -side
        out[pr] = 1.0 if pos >= neg else -1.0
    return out


def range_table(per_frame, ranges=((1, 100), (101, 200), (201, 300))):
    """per_frame: {frame: int} -> [(lo, hi, frames_with_any, max, total)]"""
    rows = []
    for lo, hi in ranges:
        vals = [per_frame[f] for f in per_frame if lo <= f <= hi]
        rows.append((lo, hi, sum(1 for v in vals if v > 0), max(vals) if vals else 0, sum(vals)))
    return rows
