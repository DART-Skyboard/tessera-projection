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


def mesh_components(n, edges):
    """Connected-component label per vertex (min-label propagation over the edges, vectorised)."""
    lab = np.arange(n, dtype=np.int64)
    e = np.asarray(edges, np.int64).reshape(-1, 2)
    if len(e) == 0:
        return lab
    a, b = e[:, 0], e[:, 1]
    for _ in range(4 * n + 8):
        m = np.minimum(lab[a], lab[b])
        new = lab.copy()
        np.minimum.at(new, a, m)
        np.minimum.at(new, b, m)
        new = new[new]                       # pointer jumping
        if np.array_equal(new, lab):
            break
        lab = new
    return lab


class SelfCollider:
    """Self-collision state for one triangle mesh."""

    def __init__(self, tris, edges, nverts, thickness=0.003, friction=0.2, mode='ACCURATE',
                 fast_period=3, iterations=2, ring_extra=None, vert_th=None):
        self.tris = np.asarray(tris, np.int64).reshape(-1, 3)
        self.n = int(nverts)
        self.thickness = float(thickness)
        self.friction = float(friction)
        self.mode = mode
        self.fast_period = int(fast_period)
        self.iterations = int(iterations)
        # welded / stitched vertex pairs are topological neighbours for contact purposes (ring), but they do
        # not merge the pieces: ``comp`` (layer identity) comes from the mesh edges only
        ring_edges = edges
        if ring_extra is not None and len(ring_extra):
            ring_edges = np.concatenate([np.asarray(edges, np.int64).reshape(-1, 2),
                                         np.asarray(ring_extra, np.int64).reshape(-1, 2)])
        self.ring = _ring_keys(self.n, ring_edges, 1)
        self.vert_th = None if vert_th is None else np.asarray(vert_th, float).reshape(-1)
        self.rest_amb = 0.002       # rest offsets smaller than this are ambiguous: the piece-pair majority side decides
        self.pair_sign = None
        self.layer_reach = 0.6      # rest-side ordering applies to pairs that were within this distance at rest
        self.comp = mesh_components(self.n, edges)   # disconnected pieces (garment layers) keep their rest-pose order
        self.rest = None
        self.layered = False       # 'LAYERED' mode: rest-side corrector for ordered pieces
        self.layer_radius = 0.025
        self.layer_period = 1      # run the rest-side corrector every N-th substep (the last of each step)
        self.rest_nrm = None       # (v,u) with u in 1-ring(v) or u == v
        self.layer = np.zeros(self.n, np.int32)
        self.allow = np.ones((1, 1), bool)
        self.vert_on = np.ones(self.n, bool)
        self._cache = None
        self.touched_idx = np.zeros(0, np.int64)
        self.max_cells_per_tri = 64
        self.max_pairs = 6_000_000
        self.stats = dict(pairs=0, contacts=0, ms=0.0, last_depth=0.0)
        self.edge_contacts = False     # edge-edge contact (open hems sliding through each other); on for ACCURATE / LAYERED
        self.edge_every = 1
        self._edges = None
        self.max_edge_pairs = 3_000_000

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

    def set_rest(self, rest):
        """Rest-pose positions: pairs from different connected pieces keep the side they had at rest."""
        r = np.asarray(rest, float)
        T = self.tris
        n = np.cross(r[T[:, 1]] - r[T[:, 0]], r[T[:, 2]] - r[T[:, 0]])
        self.rest_nrm = n / np.maximum(np.linalg.norm(n, axis=1), 1e-14)[:, None]
        self.rest = r.copy()
        self.pair_sign = None

    def _rest_side(self, v, t, dr):
        """Rest side (+1 / -1) of vertex ``v`` against triangle ``t`` from the rest offset ``dr``.  Pieces that were
        laid ON each other at rest (offset below ``rest_amb``, e.g. a sash on a dress) take the side that the piece
        pair has on average, so the unavoidable noise of a near-zero offset cannot pull a vertex through the cloth."""
        sgn = np.where(dr < 0.0, -1.0, 1.0)
        if self.pair_sign is None:
            self._compute_pair_signs()
        amb = np.abs(dr) < self.rest_amb
        if amb.any() and self.pair_sign:
            ps = np.array([self.pair_sign.get((int(a), int(b)), 1.0) for a, b in
                           zip(self.comp[v[amb]], self.comp[self.tris[t[amb], 0]])])
            sgn[amb] = ps
        return sgn

    def _compute_pair_signs(self):
        """Mean rest offset sign per ordered pair of pieces (vertices of A against faces of B within a few mm)."""
        self.pair_sign = {}
        if self.rest is None or len(np.unique(self.comp)) < 2:
            return
        r = self.rest
        save = self.thickness
        try:
            self.thickness = max(self.rest_amb * 2.0, 0.004)
            v, t = self._candidates(r, r, 0.0)
        finally:
            self.thickness = save
        if len(v) == 0:
            return
        tr = self.tris[t]
        k = self.comp[v] != self.comp[tr[:, 0]]
        v, t, tr = v[k], t[k], tr[k]
        if len(v) == 0:
            return
        q, _b, inside = closest_point_triangle(r[v], r[tr[:, 0]], r[tr[:, 1]], r[tr[:, 2]])
        dr = ((r[v] - r[tr[:, 0]]) * self.rest_nrm[t]).sum(1)
        ok = inside & (np.abs(dr) < self.rest_amb * 2.0)
        acc = {}
        for a, b, d in zip(self.comp[v[ok]], self.comp[tr[ok, 0]], dr[ok]):
            acc[(int(a), int(b))] = acc.get((int(a), int(b)), 0.0) + float(d)
        self.pair_sign = {k_: (-1.0 if d < 0 else 1.0) for k_, d in acc.items()}

    def invalidate(self):
        self._cache = None

    # ------------------------------------------------------------------ broad phase
    def _cells(self, lo, hi, origin, cs, dims, ids):
        """Expand per-object boxes (lo, hi) into (object id, int64 cell key) rows."""
        clo = np.floor((lo - origin) / cs).astype(np.int64)
        chi = np.floor((hi - origin) / cs).astype(np.int64)
        span = np.maximum(chi - clo + 1, 1)
        cnt = span[:, 0] * span[:, 1] * span[:, 2]
        bad = (cnt > self.max_cells_per_tri) | ~np.isfinite(lo).all(1) | ~np.isfinite(hi).all(1)
        if bad.any():
            cnt = np.where(bad, 0, cnt)
            self.stats['dropped'] = int(bad.sum())
        tot = int(cnt.sum())
        oid = np.repeat(ids, cnt)
        base = np.cumsum(cnt) - cnt
        loc = np.arange(tot, dtype=np.int64) - np.repeat(base, cnt)
        sx, sy = np.repeat(span[:, 0], cnt), np.repeat(span[:, 1], cnt)
        cx = np.repeat(clo[:, 0], cnt) + loc % sx
        cy = np.repeat(clo[:, 1], cnt) + (loc // sx) % sy
        cz = np.repeat(clo[:, 2], cnt) + loc // (sx * sy)
        return oid, (cx * dims[1] + cy) * dims[2] + cz

    def _candidates(self, x, xp, slack):
        """Swept broad phase: triangles and vertices both hashed over the box they sweep this substep
        (current + previous positions) grown by thickness (+ ``slack`` for cached FAST candidates)."""
        tris = self.tris
        th = self.thickness
        if not (np.isfinite(x).all() and np.isfinite(xp).all()):
            self.stats['nonfinite'] = True
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        if self.vert_th is not None:
            th = max(th, float(self.vert_th.max()))
        grow = th + slack
        P = np.stack([x[tris[:, 0]], x[tris[:, 1]], x[tris[:, 2]], xp[tris[:, 0]], xp[tris[:, 1]], xp[tris[:, 2]]])
        tlo = P.min(0) - grow
        thi = P.max(0) + grow
        vlo = np.minimum(x, xp) - grow
        vhi = np.maximum(x, xp) + grow
        ext = (thi - tlo).max(1)
        cs = float(np.clip(np.median(ext) * 1.1, 2.0 * th, 1e3))
        origin = np.minimum(tlo.min(0), vlo.min(0)) - cs
        top = np.maximum(thi.max(0), vhi.max(0))
        dims_f = (top - origin) / cs + 3.0
        while float(np.prod(dims_f)) > 4.0e18:
            cs *= 2.0
            dims_f = (top - origin) / cs + 3.0
        dims = np.floor(dims_f).astype(np.int64) + 1
        t_id, tkey = self._cells(tlo, thi, origin, cs, dims, np.arange(len(tris), dtype=np.int64))
        if len(t_id) == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        order = np.argsort(tkey, kind='stable')
        tkey, t_sorted = tkey[order], t_id[order]
        vids = np.nonzero(self.vert_on)[0].astype(np.int64)
        v_id, vkey = self._cells(vlo[vids], vhi[vids], origin, cs, dims, vids)
        left = np.searchsorted(tkey, vkey, 'left')
        vcnt = np.searchsorted(tkey, vkey, 'right') - left
        tot2 = int(vcnt.sum())
        if tot2 > self.max_pairs:
            self.stats['overflow'] = tot2
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        v_idx = np.repeat(v_id, vcnt)
        base2 = np.cumsum(vcnt) - vcnt
        loc2 = np.arange(tot2, dtype=np.int64) - np.repeat(base2, vcnt)
        t_of = t_sorted[np.repeat(left, vcnt) + loc2]
        if len(v_id) > len(vids):           # vertices that sweep several cells can pair twice
            key = np.unique(v_idx * len(tris) + t_of)
            v_idx, t_of = key // len(tris), key % len(tris)
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
    def _thmax(self):
        return self.thickness if self.vert_th is None else max(self.thickness, float(self.vert_th.max()))

    def _pair_th(self, v, tr):
        """Contact thickness of vertex/triangle pairs: the thicker of the two fabrics (scalar without members)."""
        if self.vert_th is None:
            return self.thickness
        vt = self.vert_th
        return np.maximum(self.thickness, np.maximum(vt[v], np.maximum(vt[tr[:, 0]], np.maximum(vt[tr[:, 1]], vt[tr[:, 2]]))))

    def _contacts(self, x, xp, v, t, w):
        """Geometry for candidate pairs -> (mask, normal*sign, depth, bary, tri verts) of true contacts."""
        tr = self.tris[t]
        th = self._pair_th(v, tr)
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
        cross = np.zeros(len(v), bool)
        if self.rest is not None:
            cross = self.comp[v] != self.comp[tr[:, 0]]
            if cross.any():
                dr = ((self.rest[v] - self.rest[tr[:, 0]]) * self.rest_nrm[t]).sum(1)
                # rest-side ordering only for pieces that really were stacked at rest (within reach)
                cross &= np.abs(dr) < self.layer_reach
                if cross.any():
                    s0 = np.where(cross, self._rest_side(v, t, dr), s0)
        sd = s0 * d
        # projection test for the face region: point projects inside (closest point interior)
        proj_in = inside
        crossed = sd < 0.0
        # slack for tunnelled vertices: they must still project inside and be within a bound
        dtri = np.maximum(np.maximum(np.linalg.norm(a - pa, axis=1), np.linalg.norm(b - pb, axis=1)),
                          np.linalg.norm(c - pc, axis=1))
        move = np.linalg.norm(x[v] - xp[v], axis=1) + dtri
        near_plane = (np.abs(d) < (th + 2.0 * move + 1e-6)) | (cross & (np.abs(d) < 0.25))
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
    def solve(self, x, xp, inv_mass, substep_index=0, speed_slack=None, reuse=False, passes=None):
        """Resolve contacts in place on ``x`` (positions after constraint projection).
        ``xp`` are the positions at the start of the substep.  Returns number of contacts."""
        import time as _t
        t0 = _t.perf_counter()
        fast = self.mode == 'FAST'
        refresh = (not reuse) and ((not fast) or self._cache is None or (substep_index % max(self.fast_period, 1) == 0))
        if reuse and self._cache is None:
            refresh = True
        if fast and not refresh and not reuse:
            # a cached candidate list is only valid while nothing moves much: fast movers force a re-detect
            d99 = float(np.percentile(np.linalg.norm(x - xp, axis=1), 99.0))
            if d99 > 0.5 * self.thickness:
                refresh = True
        if refresh:
            if speed_slack is None:
                disp = np.linalg.norm(x - xp, axis=1)
                disp = disp[inv_mass > 0] if (inv_mass > 0).sum() > 100 else disp
                # robust to a few runaway vertices: the 99th percentile bounds the swept box
                speed_slack = float(np.percentile(disp, 99.0)) if len(disp) > 100 else float(disp.max())
            slack = min(speed_slack * 3.0, 0.15) if fast else 0.0
            v, t = self._candidates(x, xp, slack)
            v, t = self._filter_topology(v, t)
            # prefilter: plane distance first (cheap, a lower bound of the true distance), then closest point
            if len(v):
                T = self.tris
                dall = np.linalg.norm(x - xp, axis=1)
                dtri = np.maximum(np.maximum(dall[T[:, 0]], dall[T[:, 1]]), dall[T[:, 2]])
                # a pair can only touch during the step if it is within thickness + both parties' travel
                lim = self._thmax() + (2.0 if fast else 1.0) * (dall[v] + dtri[t]) + 1e-6
                A0, B0, C0 = x[T[:, 0]], x[T[:, 1]], x[T[:, 2]]
                N = np.cross(B0 - A0, C0 - A0)
                N /= np.maximum(np.linalg.norm(N, axis=1), 1e-14)[:, None]
                pd = np.abs(((x[v] - A0[t]) * N[t]).sum(1))
                k = pd < lim
                v, t, lim = v[k], t[k], lim[k]
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
            for _ in range(int(passes) if passes else (1 if fast else max(1, self.iterations))):
                total += self._respond(x, xp, inv_mass, v, t)
        if self.edge_contacts and substep_index % max(int(self.edge_every), 1) == 0:
            total += self.edge_correct(x, xp, inv_mass)
        if self.layered and (substep_index + 1) % max(int(self.layer_period), 1) == 0:
            total += self.layered_correct(x, xp, inv_mass)
        self.stats['ms'] = 1000.0 * (_t.perf_counter() - t0)
        return total

    def layered_correct(self, x, xp, w):
        """Rest-side corrector for ordered pieces (garment layers / separate objects of a group).

        A vertex of one piece must stay on the side of the nearest face of every OTHER piece that it had at rest,
        at >= contact thickness, wherever it got across (vertex-face, edge-edge, tunnelling).  Both directions are
        corrected (upper vertex vs lower face and lower vertex vs upper face).  Only pairs that were stacked at
        rest (within ``layer_reach``) are ordered this way; pieces that started apart use the continuity (previous
        side) logic of the normal contacts, so a flag may swing round the edge of a plate.  The candidate search
        box is the swept box of the vertex and the face grown by ``layer_radius``."""
        if self.rest is None:
            return 0
        if len(np.unique(self.comp)) < 2:
            return 0
        R = max(float(self.layer_radius), 2.0 * self._thmax())
        save_th = self.thickness
        try:
            for _try in range(4):             # a large radius can overflow the pair budget: shrink it
                self.thickness = R
                self.stats.pop('overflow', None)
                v, t = self._candidates(x, xp, 0.0)
                if 'overflow' not in self.stats or R <= 2.0 * self._thmax():
                    break
                R = max(0.5 * R, 2.0 * self._thmax())
        finally:
            self.thickness = save_th
        if len(v) == 0:
            return 0
        tr = self.tris[t]
        k = (self.comp[v] != self.comp[tr[:, 0]]) & self.vert_on[v]
        v, t, tr = v[k], t[k], tr[k]
        if len(v) == 0:
            return 0
        th = self._pair_th(v, tr)
        a, b, c = x[tr[:, 0]], x[tr[:, 1]], x[tr[:, 2]]
        p = x[v]
        nrm = np.cross(b - a, c - a)
        nl = np.linalg.norm(nrm, axis=1)
        good = nl > 1e-14
        nrm = nrm / np.where(good, nl, 1.0)[:, None]
        dr = ((self.rest[v] - self.rest[tr[:, 0]]) * self.rest_nrm[t]).sum(1)
        k = good & (np.abs(dr) < self.layer_reach)
        if not k.any():
            return 0
        v, t, tr, a, b, c, p, nrm, dr = v[k], t[k], tr[k], a[k], b[k], c[k], p[k], nrm[k], dr[k]
        th = th[k] if np.ndim(th) else th
        q, bary, inside = closest_point_triangle(p, a, b, c)
        d = ((p - a) * nrm).sum(1)
        s0 = self._rest_side(v, t, dr)
        sd = s0 * d
        diff = p - q
        dist = np.linalg.norm(diff, axis=1)
        face = inside & (np.abs(d) < R) & (sd < th)
        edge = (~inside) & (dist < th) & (sd >= 0.0)
        m = face | edge
        if not m.any():
            return 0
        v, t, tr, nrm, s0, sd, diff, dist, face = v[m], t[m], tr[m], nrm[m], s0[m], sd[m], diff[m], dist[m], face[m]
        th = th[m] if np.ndim(th) else th
        direction = np.where(face[:, None], s0[:, None] * nrm, diff / np.maximum(dist, 1e-12)[:, None])
        need = np.where(face, th - sd, th - dist)
        order = np.argsort(-need, kind='stable')
        v, tr, direction, need = v[order], tr[order], direction[order], need[order]
        _u, first = np.unique(v, return_index=True)
        v, tr, direction, need = v[first], tr[first], direction[first], need[first]
        push = need[:, None] * direction
        free = w[v] > 0.0
        x[v[free]] += push[free]
        pin = ~free
        if pin.any():                      # the vertex cannot move: carry the lower face away instead
            for kk in range(3):
                x[tr[pin, kk]] -= push[pin] * (w[tr[pin, kk]] > 0.0)[:, None]
        self.stats['layered_fixed'] = int(len(v))
        return int(len(v))

    def wrong_side(self, x, reach=None, chunk=160, labels=None, detail=None):
        """Diagnostic: per ordered piece pair (P, Q) the number of P vertices that sit on the wrong (non-rest) side of
        the nearest Q face they project onto, and the deepest such vertex.  Independent of the solver (brute force
        matrix test, no hashing, no pair budget).  Returns {(p, q): (count, worst_depth_m)} over piece labels."""
        reach = self.layer_reach if reach is None else float(reach)
        T = self.tris
        A, B, C = x[T[:, 0]], x[T[:, 1]], x[T[:, 2]]
        v0, v1 = B - A, C - A
        N = np.cross(v0, v1)
        nl = np.linalg.norm(N, axis=1)
        okT = nl > 1e-14
        N = N / np.where(okT, nl, 1.0)[:, None]
        d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
        den = d00 * d11 - d01 * d01
        den = np.where(np.abs(den) < 1e-18, 1e-18, den)
        rest, RN = self.rest, self.rest_nrm
        lab = self.comp if labels is None else np.asarray(labels, np.int64)
        labels = np.unique(lab)
        tcomp = lab[T[:, 0]]
        out = {}
        for p in labels:
            vp = np.nonzero((lab == p) & self.vert_on)[0]
            cnt = {int(q): 0 for q in labels if q != p}
            worst = {int(q): 0.0 for q in labels if q != p}
            for q in labels:
                if q == p:
                    continue
                tq = np.nonzero((tcomp == q) & okT)[0]
                if len(tq) == 0 or len(vp) == 0:
                    continue
                for s in range(0, len(vp), chunk):
                    ids = vp[s:s + chunk]
                    P = x[ids][:, None, :]
                    rel = P - A[tq][None]
                    d = (rel * N[tq][None]).sum(-1)
                    d02 = (rel * v0[tq][None]).sum(-1)
                    d12 = (rel * v1[tq][None]).sum(-1)
                    u = (d11[tq] * d02 - d01[tq] * d12) / den[tq]
                    w_ = (d00[tq] * d12 - d01[tq] * d02) / den[tq]
                    inside = (u >= -1e-6) & (w_ >= -1e-6) & (u + w_ <= 1.0 + 1e-6)
                    drr = ((rest[ids][:, None, :] - rest[T[tq, 0]][None]) * RN[tq][None]).sum(-1)
                    ok = inside & (np.abs(drr) < reach)
                    ad = np.where(ok, np.abs(d), np.inf)
                    j = np.argmin(ad, axis=1)
                    r_ = np.arange(len(ids))
                    has = np.isfinite(ad[r_, j])
                    sd = self._rest_side(ids, tq[j], drr[r_, j]) * d[r_, j]
                    bad = has & (sd < 0.0)
                    cnt[int(q)] += int(bad.sum())
                    if detail is not None and bad.any():
                        detail.setdefault((int(p), int(q)), []).extend(zip(ids[bad].tolist(), (-sd[bad]).tolist(), tq[j[bad]].tolist()))
                    if bad.any():
                        worst[int(q)] = max(worst[int(q)], float((-sd[bad]).max()))
            for q in cnt:
                out[(int(p), q)] = (cnt[q], worst[q])
        return out

    # ------------------------------------------------------------------ edge-edge contact
    def _edge_list(self):
        if self._edges is None:
            T = self.tris
            self._edges = np.unique(np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1), axis=0)
        return self._edges

    def _edge_candidates(self, x, xp, thick):
        E = self._edge_list()
        if not (np.isfinite(x).all() and np.isfinite(xp).all()):
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        P = np.stack([x[E[:, 0]], x[E[:, 1]], xp[E[:, 0]], xp[E[:, 1]]])
        lo = P.min(0) - thick
        hi = P.max(0) + thick
        ext = (hi - lo).max(1)
        cs = float(np.clip(np.median(ext) * 1.2, 2.0 * thick, 1e3))
        origin = lo.min(0) - cs
        top = hi.max(0)
        dims_f = (top - origin) / cs + 3.0
        while float(np.prod(dims_f)) > 4.0e18:
            cs *= 2.0
            dims_f = (top - origin) / cs + 3.0
        dims = np.floor(dims_f).astype(np.int64) + 1
        eid, key = self._cells(lo, hi, origin, cs, dims, np.arange(len(E), dtype=np.int64))
        if len(eid) == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        order = np.argsort(key, kind='stable')
        key_s, id_s = key[order], eid[order]
        left = np.searchsorted(key_s, key, 'left')
        cnt = np.searchsorted(key_s, key, 'right') - left
        tot = int(cnt.sum())
        if tot > 6 * self.max_edge_pairs:
            self.stats['edge_overflow'] = tot
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        ii = np.repeat(eid, cnt)
        base = np.cumsum(cnt) - cnt
        loc = np.arange(tot, dtype=np.int64) - np.repeat(base, cnt)
        jj = id_s[np.repeat(left, cnt) + loc]
        k = ii < jj
        ii, jj = ii[k], jj[k]
        pk = np.unique(ii * len(E) + jj)
        return pk // len(E), pk % len(E)

    @staticmethod
    def _seg_seg(a, b, c, d):
        """Closest points between segments ab and cd (vectorised, Ericson).  Returns s, t in [0,1]."""
        d1, d2, r = b - a, d - c, a - c
        A_ = (d1 * d1).sum(1)
        E_ = (d2 * d2).sum(1)
        F_ = (d2 * r).sum(1)
        C_ = (d1 * r).sum(1)
        B_ = (d1 * d2).sum(1)
        den = A_ * E_ - B_ * B_
        with np.errstate(divide='ignore', invalid='ignore'):
            s = np.where(den > 1e-18, np.clip((B_ * F_ - C_ * E_) / np.where(den > 1e-18, den, 1.0), 0.0, 1.0), 0.0)
            t = (B_ * s + F_) / np.where(E_ > 1e-18, E_, 1.0)
            lo_ = t < 0.0
            hi_ = t > 1.0
            s = np.where(lo_, np.clip(-C_ / np.where(A_ > 1e-18, A_, 1.0), 0.0, 1.0), s)
            s = np.where(hi_, np.clip((B_ - C_) / np.where(A_ > 1e-18, A_, 1.0), 0.0, 1.0), s)
            t = np.clip(np.where(lo_, 0.0, np.where(hi_, 1.0, t)), 0.0, 1.0)
        return s, t

    def edge_correct(self, x, xp, w):
        """Edge-edge contact with continuity: two edges of the cloth (open hems sliding past each other, a fold
        edge through a fold edge) are kept ``thickness`` apart, and pushed back to the side they were on at the
        start of the substep when they crossed.  Complements the vertex-vs-face contacts, which cannot see a hem
        edge cutting through a face while all four end points are still outside."""
        E = self._edge_list()
        th = self._thmax()
        i, j = self._edge_candidates(x, xp, th)
        if len(i) == 0:
            return 0
        n = self.n
        ring = self.ring
        keep = np.ones(len(i), bool)
        for ka in (0, 1):
            for kb in (0, 1):
                key = E[i, ka] * n + E[j, kb]
                pos = np.minimum(np.searchsorted(ring, key), len(ring) - 1)
                keep &= ring[pos] != key
        keep &= self.vert_on[E[i, 0]] & self.vert_on[E[i, 1]] & self.vert_on[E[j, 0]] & self.vert_on[E[j, 1]]
        if self.allow.shape[0] > 1:
            keep &= self.allow[self.layer[E[i, 0]], self.layer[E[j, 0]]]
        i, j = i[keep], j[keep]
        if len(i) == 0:
            return 0
        a, b, c, d = E[i, 0], E[i, 1], E[j, 0], E[j, 1]
        s, t = self._seg_seg(x[a], x[b], x[c], x[d])
        P = x[a] + (x[b] - x[a]) * s[:, None]
        Q = x[c] + (x[d] - x[c]) * t[:, None]
        diff = P - Q
        dist = np.linalg.norm(diff, axis=1)
        P0 = xp[a] + (xp[b] - xp[a]) * s[:, None]
        Q0 = xp[c] + (xp[d] - xp[c]) * t[:, None]
        ref = P0 - Q0
        rl = np.linalg.norm(ref, axis=1)
        # only edges that cross at an angle are handled: nearly parallel / coincident edges (a sash laid on a dress, two
        # hems lying together) have no stable edge-edge normal and are left to the vertex-face contacts
        d1 = x[b] - x[a]
        d2 = x[d] - x[c]
        sin = np.linalg.norm(np.cross(d1, d2), axis=1) / np.maximum(np.linalg.norm(d1, axis=1) * np.linalg.norm(d2, axis=1), 1e-12)
        angled = sin > 0.35
        have = (rl > 0.5 * th) & angled
        flipped = have & ((diff * ref).sum(1) < 0.0) & (rl > 0.5 * th)
        near = (dist < th) & have
        m = near | flipped
        if not m.any():
            return 0
        a, b, c, d, s, t, diff, dist, ref, rl, have, flipped = (v[m] for v in (a, b, c, d, s, t, diff, dist, ref, rl, have, flipped))
        nrm = np.where((dist > 1e-9)[:, None], diff / np.maximum(dist, 1e-12)[:, None], ref / np.maximum(rl, 1e-12)[:, None])
        nrm = np.where(flipped[:, None], -nrm, nrm)
        depth = np.where(flipped, th + dist, th - dist)
        w1 = (1.0 - s) ** 2 * w[a] + s ** 2 * w[b]
        w2 = (1.0 - t) ** 2 * w[c] + t ** 2 * w[d]
        den = w1 + w2
        ok = (den > 0.0) & (depth > 1e-9)
        if not ok.any():
            return 0
        a, b, c, d, s, t, nrm, depth, den = (v[ok] for v in (a, b, c, d, s, t, nrm, depth, den))
        lam = depth / den
        dx = np.zeros((n, 3))
        cn = np.zeros(n)
        for idx, coef, sg in ((a, 1.0 - s, 1.0), (b, s, 1.0), (c, 1.0 - t, -1.0), (d, t, -1.0)):
            vec = (sg * w[idx] * coef * lam)[:, None] * nrm
            for k in range(3):
                dx[:, k] += np.bincount(idx, weights=vec[:, k], minlength=n)
            cn += np.bincount(idx, weights=(coef > 1e-3).astype(float), minlength=n)
        tch = cn > 0
        x[tch] += dx[tch] / np.maximum(cn[tch], 1.0)[:, None] * np.minimum(cn[tch], 2.0)[:, None]
        self.stats['edge_contacts'] = int(len(a))
        return int(len(a))

    def crossings(self, x, labels=None, chunk=400, eps=1e-9, same=True, extra=None, depth=0.001):
        """Ground-truth interpenetration: number of mesh EDGES of piece P that pierce a TRIANGLE of piece Q
        (segment-triangle test, Moeller-Trumbore, AABB prefiltered).  Needs no rest pose and no ordering assumption.
        Returns {(p, q): count} for every ordered pair; with ``same`` also (p, p) = self-intersections of one piece
        (triangles that share a vertex with the edge are ignored).  A pierce only counts when both end points are
        more than ``depth`` (default 1 mm) from the triangle plane, so surfaces that merely touch / lie within a
        millimetre of each other (a sash laid on a dress) are not reported as crossings.  ``extra`` = (verts, tris) of an outside body
        (e.g. the dress form): edges of every piece against it, reported as (p, 'ext')."""
        T = self.tris
        lab = self.comp if labels is None else np.asarray(labels, np.int64)
        E = np.unique(np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1), axis=0)
        A, B, C = x[T[:, 0]], x[T[:, 1]], x[T[:, 2]]
        tlo = np.minimum(np.minimum(A, B), C)
        thi = np.maximum(np.maximum(A, B), C)
        e0, e1 = x[E[:, 0]], x[E[:, 1]]
        elo, ehi = np.minimum(e0, e1), np.maximum(e0, e1)
        out = {}
        labels_u = np.unique(lab)
        tl = lab[T[:, 0]]
        el = lab[E[:, 0]]

        def count(ei, tsel, TA, TB, TC, tlo_, thi_, tris_idx=None):
            cnt = 0
            for s0 in range(0, len(ei), chunk):
                ids = ei[s0:s0 + chunk]
                ov = ((elo[ids][:, None, :] <= thi_[None]) & (ehi[ids][:, None, :] >= tlo_[None])).all(-1)
                a_, b_ = np.nonzero(ov)
                if len(a_) == 0:
                    continue
                ee, tt = ids[a_], b_
                if tris_idx is not None:                      # same piece: skip triangles sharing a vertex with the edge
                    tv = tris_idx[tt]
                    share = ((tv == E[ee, 0][:, None]) | (tv == E[ee, 1][:, None])).any(1)
                    ee, tt = ee[~share], tt[~share]
                    if len(ee) == 0:
                        continue
                P0, D = e0[ee], e1[ee] - e0[ee]
                V0, V1, V2 = TA[tt], TB[tt], TC[tt]
                E1, E2 = V1 - V0, V2 - V0
                h = np.cross(D, E2)
                det = (E1 * h).sum(1)
                ok = np.abs(det) > eps
                inv = 1.0 / np.where(ok, det, 1.0)
                sv = P0 - V0
                u = (sv * h).sum(1) * inv
                qv = np.cross(sv, E1)
                v = (D * qv).sum(1) * inv
                t = (E2 * qv).sum(1) * inv
                hit = ok & (u > 1e-7) & (v > 1e-7) & (u + v < 1 - 1e-7) & (t > 1e-7) & (t < 1 - 1e-7)
                if depth > 0.0 and hit.any():
                    nn = np.cross(E1, E2)
                    nn /= np.maximum(np.linalg.norm(nn, axis=1), 1e-14)[:, None]
                    d0 = (sv * nn).sum(1)
                    d1 = ((P0 + D - V0) * nn).sum(1)
                    hit &= (np.abs(d0) > depth) & (np.abs(d1) > depth)
                cnt += int(hit.sum())
            return cnt

        for p in labels_u:
            ei = np.nonzero(el == p)[0]
            for q in labels_u:
                if q == p and not same:
                    continue
                ti = np.nonzero(tl == q)[0]
                out[(int(p), int(q))] = count(ei, ti, A[ti], B[ti], C[ti], tlo[ti], thi[ti], T[ti] if q == p else None)
            if extra is not None:
                ev, et = extra
                EA, EB, EC = ev[et[:, 0]], ev[et[:, 1]], ev[et[:, 2]]
                out[(int(p), 'ext')] = count(ei, None, EA, EB, EC, np.minimum(np.minimum(EA, EB), EC),
                                             np.maximum(np.maximum(EA, EB), EC))
        return out

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
