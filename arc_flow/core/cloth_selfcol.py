"""Vectorised NumPy cloth self-collision (no bpy).

Vertex-vs-triangle contact with thickness, built for XPBD position solvers:

* broad phase  - uniform grid, triangles hashed into every cell their *swept* bounding box
  touches (current + previous positions, plus thickness), vertices looked up by sorted
  key / ``searchsorted``.  Exact integer cell keys, so there are no hash collisions.
* narrow phase - Ericson closest point on triangle, all pairs at once.
* topology     - triangles incident to the vertex, or sharing an edge with any vertex in its
  1-ring (a 2-hop neighbourhood), are skipped; so are pairs whose layers are marked to
  ignore each other (``set_layers``).
* continuity   - the side of the triangle a vertex *should* be on is read from the previous
  (collision-free) positions, so a vertex that tunnelled through a face in one step is pushed
  back out the way it came, not further through.
* response     - position corrections along the contact normal (mass weighted, barycentric
  on the triangle side), Coulomb friction on the tangential relative motion, averaged
  Jacobi style with ``np.bincount`` (no ``np.add.at``).

Two modes: ``'ACCURATE'`` re-detects every substep and runs two response passes;
``'FAST'`` re-detects every ``fast_period`` substeps (cached candidate pairs, slightly fatter
swept boxes) with a single response pass, which is the real-time variant.

Edge-edge contact is not modelled: with a thickness much smaller than the edge length,
two crossing edges are caught one step later as a vertex-vs-face contact.
"""
import numpy as np


# ----------------------------------------------------------------------------- geometry

def closest_point_triangle(p, a, b, c):
    """Closest point on triangle (a,b,c) to p, vectorised.  Returns (q (N,3), bary (N,3), inside (N,))
    where ``inside`` is True when the closest point is in the triangle interior."""
    ab, ac, ap = b - a, c - a, p - a
    d1 = (ab * ap).sum(1)
    d2 = (ac * ap).sum(1)
    bp = p - b
    d3 = (ab * bp).sum(1)
    d4 = (ac * bp).sum(1)
    cp = p - c
    d5 = (ab * cp).sum(1)
    d6 = (ac * cp).sum(1)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    den = va + vb + vc
    den = np.where(np.abs(den) < 1e-30, 1e-30, den)
    inv = 1.0 / den
    v = vb * inv
    w = vc * inv
    u = 1.0 - v - w
    inside = np.ones(len(p), bool)

    def over(mask, uu, vv, ww):
        nonlocal u, v, w, inside
        u = np.where(mask, uu, u)
        v = np.where(mask, vv, v)
        w = np.where(mask, ww, w)
        inside &= ~mask

    one, zero = np.ones_like(d1), np.zeros_like(d1)
    with np.errstate(divide='ignore', invalid='ignore'):
        # reverse priority of Ericson's tests (later override earlier)
        t_bc = np.clip((d4 - d3) / np.where((d4 - d3) + (d5 - d6) == 0, 1e-30, (d4 - d3) + (d5 - d6)), 0, 1)
        over((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), zero, 1 - t_bc, t_bc)
        t_ac = np.clip(d2 / np.where(d2 - d6 == 0, 1e-30, d2 - d6), 0, 1)
        over((vb <= 0) & (d2 >= 0) & (d6 <= 0), 1 - t_ac, zero, t_ac)
        over((d6 >= 0) & (d5 <= d6), zero, zero, one)
        t_ab = np.clip(d1 / np.where(d1 - d3 == 0, 1e-30, d1 - d3), 0, 1)
        over((vc <= 0) & (d1 >= 0) & (d3 <= 0), 1 - t_ab, t_ab, zero)
        over((d3 >= 0) & (d4 <= d3), zero, one, zero)
        over((d1 <= 0) & (d2 <= 0), one, zero, zero)
    q = a * u[:, None] + b * v[:, None] + c * w[:, None]
    return q, np.stack([u, v, w], 1), inside


def _ring_keys(n, edges, hops=1):
    """Sorted int64 keys ``v*n+u`` of every ordered pair (v,u) with u within ``hops`` edges of v (incl. v)."""
    ids = np.arange(n, dtype=np.int64)
    e = np.asarray(edges, np.int64).reshape(-1, 2)
    a = np.concatenate([ids, e[:, 0], e[:, 1]])
    b = np.concatenate([ids, e[:, 1], e[:, 0]])
    keys = np.unique(a * n + b)
    for _ in range(hops - 1):
        src, dst = keys // n, keys % n
        # expand by composing with the 1-ring
        order = np.argsort(a * n + b, kind='stable')
        a1, b1 = a[order], b[order]
        start = np.searchsorted(a1, np.arange(n + 1))
        cnt = start[1:] - start[:-1]
        rep = cnt[dst]
        s = np.repeat(src, rep)
        off = np.repeat(start[dst], rep)
        loc = np.arange(len(s)) - np.repeat(np.cumsum(rep) - rep, rep)
        keys = np.unique(np.concatenate([keys, s * n + b1[off + loc]]))
    return keys


class SelfCollider:
    """Self-collision state for one triangle mesh."""

    def __init__(self, tris, edges, nverts, thickness=0.003, friction=0.2, mode='ACCURATE',
                 fast_period=3, iterations=2):
        self.tris = np.asarray(tris, np.int64).reshape(-1, 3)
        self.n = int(nverts)
        self.thickness = float(thickness)
        self.friction = float(friction)
        self.mode = mode
        self.fast_period = int(fast_period)
        self.iterations = int(iterations)
        self.ring = _ring_keys(self.n, edges, 1)       # (v,u) with u in 1-ring(v) or u == v
        self.layer = np.zeros(self.n, np.int32)
        self.allow = np.ones((1, 1), bool)
        self.vert_on = np.ones(self.n, bool)
        self._cache = None
        self.touched_idx = np.zeros(0, np.int64)
        self.max_cells_per_tri = 64
        self.max_pairs = 6_000_000
        self.stats = dict(pairs=0, contacts=0, ms=0.0, last_depth=0.0)

    # ------------------------------------------------------------------ configuration
    def set_layers(self, layer_ids, ignore_pairs=()):
        """Assign each vertex a layer id (>= 0); ``ignore_pairs`` [(a, b), ...] never collide.
        A layer id of -1 removes the vertex (and every triangle that uses it) from self-collision."""
        lid = np.asarray(layer_ids, np.int64).reshape(-1)
        if len(lid) != self.n:
            raise ValueError("one layer id per vertex")
        self.vert_on = lid >= 0
        k = int(max(lid.max(), 0)) + 1
        allow = np.ones((k, k), bool)
        for a, b in ignore_pairs:
            if 0 <= a < k and 0 <= b < k:
                allow[a, b] = allow[b, a] = False
        self.layer = np.maximum(lid, 0).astype(np.int32)
        self.allow = allow
        self._cache = None

    def invalidate(self):
        self._cache = None

    # ------------------------------------------------------------------ broad phase
    def _candidates(self, x, xp, slack):
        tris = self.tris
        th = self.thickness
        if not (np.isfinite(x).all() and np.isfinite(xp).all()):
            self.stats['nonfinite'] = True
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        a, b, c = x[tris[:, 0]], x[tris[:, 1]], x[tris[:, 2]]
        pa, pb, pc = xp[tris[:, 0]], xp[tris[:, 1]], xp[tris[:, 2]]
        lo = np.minimum(np.minimum(np.minimum(a, b), np.minimum(c, pa)), np.minimum(pb, pc))
        hi = np.maximum(np.maximum(np.maximum(a, b), np.maximum(c, pa)), np.maximum(pb, pc))
        grow = th + slack
        lo = lo - grow
        hi = hi + grow
        # cell size from the mean triangle bbox extent
        ext = (hi - lo).max(1)
        cs = float(np.clip(np.median(ext - 2.0 * grow) * 1.2 + 2.0 * grow, 2.0 * th, 1e3))
        origin = np.minimum(lo.min(0), x.min(0) - grow)
        # keep the grid addressable by an int64 key
        dims_f = (np.maximum(hi.max(0), x.max(0)) - origin) / cs + 3.0
        while float(np.prod(dims_f)) > 4.0e18:
            cs *= 2.0
            dims_f = (np.maximum(hi.max(0), x.max(0)) - origin) / cs + 3.0
        dims = np.floor(dims_f).astype(np.int64) + 1
        clo = np.floor((lo - origin) / cs).astype(np.int64)
        chi = np.floor((hi - origin) / cs).astype(np.int64)
        span = chi - clo + 1
        cnt = span[:, 0] * span[:, 1] * span[:, 2]
        # a runaway (stretched / non-finite) triangle must not explode the table: leave it out of the broad phase
        bad = (cnt > self.max_cells_per_tri) | ~np.isfinite(lo).all(1) | ~np.isfinite(hi).all(1)
        if bad.any():
            cnt = np.where(bad, 0, cnt)
            self.stats['dropped_tris'] = int(bad.sum())
        tot = int(cnt.sum())
        t_idx = np.repeat(np.arange(len(tris), dtype=np.int64), cnt)
        base = np.cumsum(cnt) - cnt
        loc = np.arange(tot, dtype=np.int64) - np.repeat(base, cnt)
        sx, sy = np.repeat(span[:, 0], cnt), np.repeat(span[:, 1], cnt)
        ix = loc % sx
        iy = (loc // sx) % sy
        iz = loc // (sx * sy)
        cx = np.repeat(clo[:, 0], cnt) + ix
        cy = np.repeat(clo[:, 1], cnt) + iy
        cz = np.repeat(clo[:, 2], cnt) + iz
        tkey = (cx * dims[1] + cy) * dims[2] + cz
        order = np.argsort(tkey, kind='stable')
        tkey = tkey[order]
        t_sorted = t_idx[order]
        vc = np.floor((x - origin) / cs).astype(np.int64)
        vkey = (vc[:, 0] * dims[1] + vc[:, 1]) * dims[2] + vc[:, 2]
        if tot == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        left = np.searchsorted(tkey, vkey, 'left')
        right = np.searchsorted(tkey, vkey, 'right')
        vcnt = right - left
        vcnt = np.where(self.vert_on, vcnt, 0)
        tot2 = int(vcnt.sum())
        if tot2 > self.max_pairs:
            self.stats['overflow'] = tot2
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        v_idx = np.repeat(np.arange(self.n, dtype=np.int64), vcnt)
        base2 = np.cumsum(vcnt) - vcnt
        loc2 = np.arange(tot2, dtype=np.int64) - np.repeat(base2, vcnt)
        t_of = t_sorted[np.repeat(left, vcnt) + loc2]
        return v_idx, t_of

    def _filter_topology(self, v, t):
        if len(v) == 0:
            return v, t
        tr = self.tris[t]
        keep = np.ones(len(v), bool)
        n = self.n
        ring = self.ring
        for k in range(3):
            key = v * n + tr[:, k]
            pos = np.searchsorted(ring, key)
            pos = np.minimum(pos, len(ring) - 1)
            keep &= ring[pos] != key
        keep &= self.vert_on[tr[:, 0]] & self.vert_on[tr[:, 1]] & self.vert_on[tr[:, 2]]
        if self.allow.shape[0] > 1:
            keep &= self.allow[self.layer[v], self.layer[tr[:, 0]]]
        return v[keep], t[keep]

    # ------------------------------------------------------------------ one response pass
    def _contacts(self, x, xp, v, t, w):
        """Geometry for candidate pairs -> (mask, normal*sign, depth, bary, tri verts) of true contacts."""
        th = self.thickness
        tr = self.tris[t]
        a, b, c = x[tr[:, 0]], x[tr[:, 1]], x[tr[:, 2]]
        p = x[v]
        nrm = np.cross(b - a, c - a)
        nl = np.linalg.norm(nrm, axis=1)
        good = nl > 1e-14
        nrm = nrm / np.where(good, nl, 1.0)[:, None]
        q, bary, inside = closest_point_triangle(p, a, b, c)
        d = ((p - a) * nrm).sum(1)
        # side the vertex was on before this step
        pa, pb, pc = xp[tr[:, 0]], xp[tr[:, 1]], xp[tr[:, 2]]
        n0 = np.cross(pb - pa, pc - pa)
        n0l = np.linalg.norm(n0, axis=1)
        n0 = n0 / np.where(n0l > 1e-14, n0l, 1.0)[:, None]
        d0 = ((xp[v] - pa) * n0).sum(1)
        s0 = np.where(d0 < 0.0, -1.0, 1.0)
        sd = s0 * d
        # projection test for the face region: point projects inside (closest point interior)
        proj_in = inside
        crossed = sd < 0.0
        # slack for tunnelled vertices: they must still project inside and be within a bound
        move = np.maximum(np.linalg.norm(x[v] - xp[v], axis=1), 0.0)
        near_plane = np.abs(d) < (th + 2.0 * move + 1e-6)
        face = proj_in & good & (sd < th) & (near_plane | ~crossed)
        # edge / vertex region: plain distance
        diff = p - q
        dist = np.linalg.norm(diff, axis=1)
        edge = (~proj_in) & (dist < th) & good & (sd >= 0.0)
        direction = np.where(face[:, None], s0[:, None] * nrm,
                             diff / np.maximum(dist, 1e-12)[:, None])
        depth = np.where(face, th - sd, th - dist)
        m = (face | edge) & (depth > 1e-9)
        # at least one side must be free to move
        wsum_v = w[v]
        wsum_t = (w[tr[:, 0]] * bary[:, 0] ** 2 + w[tr[:, 1]] * bary[:, 1] ** 2 + w[tr[:, 2]] * bary[:, 2] ** 2)
        m &= (wsum_v + wsum_t) > 0.0
        return m, direction, depth, bary, tr, wsum_v, wsum_t

    def _respond(self, x, xp, w, v, t):
        m, direction, depth, bary, tr, wv, wt = self._contacts(x, xp, v, t, w)
        k = int(m.sum())
        if k == 0:
            self.stats['contacts'] = 0
            return 0
        vi, ti = v[m], t[m]
        dr, dp, br, trm, wv, wt = direction[m], depth[m], bary[m], tr[m], wv[m], wt[m]
        den = wv + wt
        lam = dp / den
        # friction: tangential relative displacement over the step
        mu = self.friction
        corr_v = (wv * lam)[:, None] * dr
        if mu > 0.0:
            disp_t = (x[trm[:, 0]] - xp[trm[:, 0]]) * br[:, 0:1] + (x[trm[:, 1]] - xp[trm[:, 1]]) * br[:, 1:2] \
                + (x[trm[:, 2]] - xp[trm[:, 2]]) * br[:, 2:3]
            rel = (x[vi] - xp[vi]) - disp_t
            reln = (rel * dr).sum(1)
            rel_t = rel - reln[:, None] * dr
            rl = np.linalg.norm(rel_t, axis=1)
            fac = np.minimum(1.0, mu * dp / np.maximum(rl, 1e-12))
            fvec = -rel_t * fac[:, None]            # friction displacement applied to the pair (relative)
            fv = (wv / den)[:, None] * fvec
        else:
            fv = 0.0
        # accumulate (Jacobi average over contacts per vertex)
        n = self.n
        dx = np.zeros((n, 3))
        cn = np.zeros(n)
        dv = corr_v + (fv if mu > 0.0 else 0.0)
        for j in range(3):
            dx[:, j] += np.bincount(vi, weights=dv[:, j], minlength=n)
        cn += np.bincount(vi, minlength=n)
        for kk in range(3):
            wk = w[trm[:, kk]] * br[:, kk]
            cvec = (-(wk * lam))[:, None] * dr
            if mu > 0.0:
                cvec = cvec + (-(wk / den))[:, None] * fvec
            for j in range(3):
                dx[:, j] += np.bincount(trm[:, kk], weights=cvec[:, j], minlength=n)
            cn += np.bincount(trm[:, kk], weights=(br[:, kk] > 1e-3).astype(float), minlength=n)
        touched = cn > 0
        x[touched] += dx[touched] / np.maximum(cn[touched], 1.0)[:, None] * np.minimum(cn[touched], 2.0)[:, None]
        self.touched_idx = np.unique(np.concatenate([vi, trm.ravel()]))
        self.stats['contacts'] = k
        self.stats['last_depth'] = float(dp.max())
        return k

    # ------------------------------------------------------------------ public
    def solve(self, x, xp, inv_mass, substep_index=0, speed_slack=None):
        """Resolve contacts in place on ``x`` (positions after constraint projection).
        ``xp`` are the positions at the start of the substep.  Returns number of contacts."""
        import time as _t
        t0 = _t.perf_counter()
        fast = self.mode == 'FAST'
        refresh = (not fast) or self._cache is None or (substep_index % max(self.fast_period, 1) == 0)
        if refresh:
            if speed_slack is None:
                disp = np.linalg.norm(x - xp, axis=1)
                disp = disp[inv_mass > 0] if (inv_mass > 0).sum() > 100 else disp
                # robust to a few runaway vertices: the 99th percentile bounds the swept box
                speed_slack = float(np.percentile(disp, 99.0)) if len(disp) > 100 else float(disp.max())
            slack = min(speed_slack * (2.0 if fast else 1.5), 0.04)
            v, t = self._candidates(x, xp, slack)
            v, t = self._filter_topology(v, t)
            # prefilter: plane distance first (cheap, a lower bound of the true distance), then closest point
            if len(v):
                lim = self.thickness + slack * (3.0 if fast else 2.0) + 1e-6
                T = self.tris
                A0, B0, C0 = x[T[:, 0]], x[T[:, 1]], x[T[:, 2]]
                N = np.cross(B0 - A0, C0 - A0)
                N /= np.maximum(np.linalg.norm(N, axis=1), 1e-14)[:, None]
                pd = np.abs(((x[v] - A0[t]) * N[t]).sum(1))
                k = pd < lim
                v, t = v[k], t[k]
                if len(v):
                    tr = T[t]
                    q, _bary, _in = closest_point_triangle(x[v], x[tr[:, 0]], x[tr[:, 1]], x[tr[:, 2]])
                    k = np.linalg.norm(x[v] - q, axis=1) < lim
                    v, t = v[k], t[k]
            self._cache = (v, t)
        else:
            v, t = self._cache
        self.stats['pairs'] = len(v)
        total = 0
        if len(v):
            for _ in range(1 if fast else max(1, self.iterations)):
                total += self._respond(x, xp, inv_mass, v, t)
        self.stats['ms'] = 1000.0 * (_t.perf_counter() - t0)
        return total

    def min_distance(self, x, limit=None):
        """Smallest vertex-to-triangle distance over all non-excluded pairs (diagnostic, unsigned)."""
        lim = float(limit if limit is not None else self.thickness * 4.0)
        v, t = self._candidates(x, x, lim)
        v, t = self._filter_topology(v, t)
        if len(v) == 0:
            return np.inf
        tr = self.tris[t]
        q, _b, _i = closest_point_triangle(x[v], x[tr[:, 0]], x[tr[:, 1]], x[tr[:, 2]])
        return float(np.linalg.norm(x[v] - q, axis=1).min())
