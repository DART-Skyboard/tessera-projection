"""LOTBCBP collision fields (NumPy).

TriSoup        triangles with angle-weighted pseudo-normals (vertex, edge, face) for a robust sign test
virtual_subdivide   on-demand "behind the scenes" tessellation of long triangles with optional Phong smoothing
signed_distance     exact signed distance of many points to a soup (vectorised BVH traversal; chunk-pruned brute force for tiny soups)
TriBVH / closest_surface   Morton-sorted binary BVH over the triangles, shared by sign, projection and refinement queries
BrickGrid      narrow-band sparse 8^3 brick SDF (bricks within the band only; bricks can be added and removed)
PartField      per-part coarse grid + dynamically requested refined bricks (the LOTBCBP adaptive refinement)
"""
import numpy as np

# ----------------------------------------------------------------------------- triangle soup


class TriSoup:
    __slots__ = ('A', 'B', 'C', 'fn', 'vn', 'en', 'bvh')

    def __init__(self, A, B, C, fn, vn, en):
        self.A, self.B, self.C, self.fn, self.vn, self.en = A, B, C, fn, vn, en
        self.bvh = None                     # lazily built TriBVH (see TriBVH / get_bvh)

    def __len__(self):
        return len(self.A)

    def take(self, idx):
        return TriSoup(self.A[idx], self.B[idx], self.C[idx], self.fn[idx], self.vn[idx], self.en[idx])

    @staticmethod
    def concat(a, b):
        return TriSoup(*[np.concatenate([getattr(a, k), getattr(b, k)]) for k in ('A', 'B', 'C', 'fn', 'vn', 'en')])

    def bounds(self):
        allv = np.concatenate([self.A, self.B, self.C])
        return allv.min(axis=0), allv.max(axis=0)


def _unit(v, fallback=None):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    out = v / np.maximum(n, 1e-30)
    if fallback is not None:
        bad = (n[..., 0] < 1e-12)
        out[bad] = fallback[bad]
    return out


def soup_from_mesh(verts, tris):
    """verts (V,3) float, tris (F,3) int -> TriSoup with pseudo-normals.  Degenerate faces dropped."""
    verts = np.asarray(verts, dtype=np.float64)
    tris = np.asarray(tris, dtype=np.int64)
    P = verts[tris]
    cr = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    area2 = np.linalg.norm(cr, axis=1)
    good = area2 > 1e-14
    tris, P, cr, area2 = tris[good], P[good], cr[good], area2[good]
    F = len(tris)
    if F == 0:
        z = np.zeros((0, 3))
        return TriSoup(z, z, z, z, np.zeros((0, 3, 3)), np.zeros((0, 3, 3)))
    fn = cr / area2[:, None]

    def angle(p, q, r):
        u, w = _unit(q - p), _unit(r - p)
        return np.arccos(np.clip(np.sum(u * w, axis=1), -1.0, 1.0))
    ang = np.stack([angle(P[:, 0], P[:, 1], P[:, 2]), angle(P[:, 1], P[:, 2], P[:, 0]),
                    angle(P[:, 2], P[:, 0], P[:, 1])], axis=1)
    vacc = np.zeros((len(verts), 3))
    for k in range(3):
        np.add.at(vacc, tris[:, k], fn * ang[:, k:k + 1])
    vn_all = _unit(vacc)
    vn = vn_all[tris]                       # (F,3,3)
    V = len(verts)
    e = np.stack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]], axis=1)   # (F,3,2)
    lo, hi = e.min(axis=2), e.max(axis=2)
    key = (lo * V + hi).reshape(-1)
    _u, inv = np.unique(key, return_inverse=True)
    eacc = np.zeros((len(_u), 3))
    np.add.at(eacc, inv, np.repeat(fn, 3, axis=0))
    en = _unit(eacc[inv].reshape(F, 3, 3), fallback=np.repeat(fn[:, None, :], 3, axis=1))
    return TriSoup(P[:, 0].copy(), P[:, 1].copy(), P[:, 2].copy(), fn, vn, en)


def boundary_loops(verts, tris):
    """Open-edge loops of a mesh -> list of dict(centroid, normal, radius, count).  Normal is
    the loop plane normal oriented away from the mesh centre (points out of the opening)."""
    verts = np.asarray(verts, dtype=np.float64)
    tris = np.asarray(tris, dtype=np.int64)
    if len(tris) == 0:
        return []
    V = len(verts)
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    lo, hi = e.min(axis=1), e.max(axis=1)
    key = lo * V + hi
    u, idx, cnt = np.unique(key, return_index=True, return_counts=True)
    bnd = u[cnt == 1]
    if len(bnd) == 0:
        return []
    a, b = bnd // V, bnd % V
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, j in zip(a.tolist(), b.tolist()):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    groups = {}
    for v in list(parent.keys()):
        groups.setdefault(find(v), []).append(v)
    centre = verts.mean(axis=0)
    out = []
    for g in groups.values():
        pts = verts[g]
        if len(pts) < 3:
            continue
        c = pts.mean(axis=0)
        _u, _s, vt = np.linalg.svd(pts - c, full_matrices=False)
        n = vt[-1]
        if np.dot(n, c - centre) < 0:
            n = -n
        r = float(np.mean(np.linalg.norm(pts - c, axis=1)))
        out.append(dict(centroid=c, normal=n / max(np.linalg.norm(n), 1e-12), radius=r, count=len(g)))
    return out

# ----------------------------------------------------------------------------- virtual tessellation


def virtual_subdivide(soup, max_edge, cap=200000, phong=0.0, passes=14):
    """Longest-edge bisection of every triangle with an edge > max_edge ("virtual tessellation").

    Children inherit the parent's feature normals, so signed distance stays bit-compatible for flat
    triangles.  With phong>0, midpoints of smooth edges (both endpoint normals agree with the edge normal)
    are displaced toward the Phong tangent-plane projection: low-poly curved surfaces gain real
    curvature, i.e. the collision body exceeds its modelled resolution.  `cap` bounds triangles ADDED."""
    added = 0
    cos_smooth = np.cos(np.radians(40.0))
    for _ in range(passes):
        if len(soup) == 0:
            break
        L = np.stack([np.linalg.norm(soup.B - soup.A, axis=1), np.linalg.norm(soup.C - soup.B, axis=1),
                      np.linalg.norm(soup.A - soup.C, axis=1)], axis=1)
        k = L.argmax(axis=1)
        need = L.max(axis=1) > max_edge
        n_need = int(need.sum())
        if n_need == 0 or added + n_need > cap:
            if n_need and added < cap:
                # split only as many as the cap allows (longest first)
                order = np.argsort(-L.max(axis=1))[:cap - added]
                need = np.zeros(len(soup), bool)
                need[order] = True
                need &= L.max(axis=1) > max_edge
                n_need = int(need.sum())
            if n_need == 0:
                break
        keep = soup.take(np.flatnonzero(~need))
        s = soup.take(np.flatnonzero(need))
        kk = k[need]
        perm = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]])[kk]          # (n,3) vertex rotation
        ar = np.arange(len(kk))[:, None]
        verts = np.stack([s.A, s.B, s.C], axis=1)[ar, perm]            # (n,3,3)
        vn = s.vn[ar, perm]
        en = s.en[ar, perm]                                            # edge e' = (k + e) % 3
        # en rotation: edge index e' corresponds to original edge (kk+e')%3
        eidx = (kk[:, None] + np.arange(3)[None, :]) % 3
        en = s.en[ar, eidx]
        A, B, Cc = verts[:, 0], verts[:, 1], verts[:, 2]
        nA, nB, nC = vn[:, 0], vn[:, 1], vn[:, 2]
        M = 0.5 * (A + B)
        nM = en[:, 0].copy()
        if phong > 0.0:
            e0 = _unit(en[:, 0])
            smooth = (np.sum(nA * e0, axis=1) > cos_smooth) & (np.sum(nB * e0, axis=1) > cos_smooth)
            pa = M - np.sum((M - A) * nA, axis=1, keepdims=True) * nA
            pb = M - np.sum((M - B) * nB, axis=1, keepdims=True) * nB
            Mp = 0.5 * (pa + pb)
            M = np.where(smooth[:, None], M + phong * (Mp - M), M)
            nM = np.where(smooth[:, None], _unit(nA + nB), nM)
        fn_old = s.fn

        def make(a, b, c, vna, vnb, vnc, e_ab, e_bc, e_ca):
            cr = np.cross(b - a, c - a)
            nl = np.linalg.norm(cr, axis=1, keepdims=True)
            f = np.where(nl > 1e-14, cr / np.maximum(nl, 1e-30), fn_old)
            flip = np.sum(f * fn_old, axis=1) < 0
            f = np.where(flip[:, None], -f, f)
            return TriSoup(a, b, c, f, np.stack([vna, vnb, vnc], axis=1), np.stack([e_ab, e_bc, e_ca], axis=1))
        c1 = make(A, M, Cc, nA, nM, nC, en[:, 0], fn_old, en[:, 2])
        c2 = make(M, B, Cc, nM, nB, nC, en[:, 0], en[:, 1], fn_old)
        soup = TriSoup.concat(TriSoup.concat(keep, c1), c2)
        added += n_need
    return soup

# ----------------------------------------------------------------------------- exact signed distance


def closest_points(P, A, B, C):
    """Vectorised Ericson closest point on triangle.  Returns (cp, region) with
    region 0=face, 1/2/3=vertex A/B/C, 4=edge AB, 5=edge BC, 6=edge CA."""
    ab, ac, ap = B - A, C - A, P - A
    d1, d2 = np.sum(ab * ap, 1), np.sum(ac * ap, 1)
    bp = P - B
    d3, d4 = np.sum(ab * bp, 1), np.sum(ac * bp, 1)
    cp_ = P - C
    d5, d6 = np.sum(ab * cp_, 1), np.sum(ac * cp_, 1)
    vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4
    # face
    den = va + vb + vc
    den = np.where(np.abs(den) < 1e-300, 1e-300, den)
    v, w = vb / den, vc / den
    cp = A + ab * v[:, None] + ac * w[:, None]
    reg = np.zeros(len(P), np.int8)
    # edge BC
    m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    t = (d4 - d3) / np.where(((d4 - d3) + (d5 - d6)) == 0, 1e-300, (d4 - d3) + (d5 - d6))
    cp = np.where(m[:, None], B + (C - B) * t[:, None], cp); reg = np.where(m, 5, reg)
    # edge AC
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    t = d2 / np.where((d2 - d6) == 0, 1e-300, d2 - d6)
    cp = np.where(m[:, None], A + ac * t[:, None], cp); reg = np.where(m, 6, reg)
    # vertex C
    m = (d6 >= 0) & (d5 <= d6)
    cp = np.where(m[:, None], C, cp); reg = np.where(m, 3, reg)
    # edge AB
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    t = d1 / np.where((d1 - d3) == 0, 1e-300, d1 - d3)
    cp = np.where(m[:, None], A + ab * t[:, None], cp); reg = np.where(m, 4, reg)
    # vertex B
    m = (d3 >= 0) & (d4 <= d3)
    cp = np.where(m[:, None], B, cp); reg = np.where(m, 2, reg)
    # vertex A
    m = (d1 <= 0) & (d2 <= 0)
    cp = np.where(m[:, None], A, cp); reg = np.where(m, 1, reg)
    return cp, reg


def signed_distance_brute(soup, points, chunk=96):
    """Reference implementation (chunk-pruned brute force, O(chunks x tris)); kept for tiny soups and as the test oracle.

    Points are processed in small spatial chunks (callers pass grid-ordered or brick-ordered points):
    a chunk bound (nearest centroid + chunk radius) discards every triangle that cannot be nearest,
    then per-point candidates are pruned again with a centroid upper bound and AABB lower bound."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n, m = len(points), len(soup)
    if m == 0:
        return np.full(n, 1e9)
    cen = (soup.A + soup.B + soup.C) / 3.0
    tmin = np.minimum(np.minimum(soup.A, soup.B), soup.C)
    tmax = np.maximum(np.maximum(soup.A, soup.B), soup.C)
    out = np.empty(n)
    for s in range(0, n, chunk):
        P = points[s:s + chunk]
        k = len(P)
        pmin, pmax = P.min(0), P.max(0)
        ctr = 0.5 * (pmin + pmax)
        rad = 0.5 * float(np.linalg.norm(pmax - pmin))
        Umax = float(np.sqrt(np.min(np.sum((cen - ctr) ** 2, axis=1)))) + rad
        lbc = np.maximum(np.maximum(tmin - pmax, pmin - tmax), 0.0)
        cand = np.flatnonzero(np.sum(lbc * lbc, axis=1) <= Umax * Umax * (1.0 + 1e-9))
        c_cen, c_min, c_max = cen[cand], tmin[cand], tmax[cand]
        dc = P[:, None, :] - c_cen[None]
        U2 = np.min(np.sum(dc * dc, axis=2), axis=1)
        lb = np.maximum(np.maximum(c_min[None] - P[:, None, :], P[:, None, :] - c_max[None]), 0.0)
        mask = np.sum(lb * lb, axis=2) <= (U2[:, None] * (1.0 + 1e-9) + 1e-18)
        pi, ti = np.nonzero(mask)
        ti = cand[ti]
        cp, reg = closest_points(P[pi], soup.A[ti], soup.B[ti], soup.C[ti])
        diff = P[pi] - cp
        d2 = np.sum(diff * diff, axis=1)
        starts = np.flatnonzero(np.r_[True, pi[1:] != pi[:-1]])
        dmin = np.minimum.reduceat(d2, starts)
        rowmin = np.empty(k); rowmin[pi[starts]] = dmin
        hit = d2 <= rowmin[pi] * (1.0 + 1e-9) + 1e-18
        hi = np.flatnonzero(hit)
        first = hi[np.r_[True, pi[hi][1:] != pi[hi][:-1]]]
        sel_t, sel_r, sel_p = ti[first], reg[first], pi[first]
        pn = np.where((sel_r == 0)[:, None], soup.fn[sel_t], 0.0)
        for rcode, which, kind in ((1, 0, 'v'), (2, 1, 'v'), (3, 2, 'v'), (4, 0, 'e'), (5, 1, 'e'), (6, 2, 'e')):
            mm = sel_r == rcode
            if mm.any():
                pn[mm] = (soup.vn if kind == 'v' else soup.en)[sel_t[mm], which]
        sign = np.where(np.sum(diff[first] * pn, axis=1) >= 0.0, 1.0, -1.0)
        res = np.empty(k)
        res[sel_p] = np.sqrt(d2[first]) * sign
        out[s:s + chunk] = res
    return out



# ----------------------------------------------------------------------------- triangle BVH


def _spread21(x):
    x = x.astype(np.uint64) & np.uint64(0x1FFFFF)
    x = (x | (x << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
    x = (x | (x << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
    x = (x | (x << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
    x = (x | (x << np.uint64(4))) & np.uint64(0x10C30C30C30C30C3)
    x = (x | (x << np.uint64(2))) & np.uint64(0x1249249249249249)
    return x


def morton_order(cen):
    lo, hi = cen.min(0), cen.max(0)
    q = np.clip((cen - lo) / np.maximum(hi - lo, 1e-30) * 2097151.0, 0, 2097151.0)
    code = _spread21(q[:, 0]) | (_spread21(q[:, 1]) << np.uint64(1)) | (_spread21(q[:, 2]) << np.uint64(2))
    return np.argsort(code, kind='stable')


class TriBVH:
    """Complete binary BVH over Morton-sorted triangle leaves (LEAF triangles each).  Level k has 2^k nodes, node j has the
    children 2j, 2j+1 on level k+1; padding leaves are empty boxes (+inf/-inf).  Queries descend a whole batch of points
    level by level (breadth first), pruned by an exact upper bound found with one greedy dive, so a query costs O(log n + local)
    instead of O(n).  Memory: about 56 bytes per leaf box plus 4 bytes per triangle."""
    LEAF = 4
    TIGHTEN_FROM = 3      # levels (from the root) below which a representative triangle per node tightens the bound

    def __init__(self, soup):
        m = len(soup)
        L = self.LEAF
        tmin = np.minimum(np.minimum(soup.A, soup.B), soup.C)
        tmax = np.maximum(np.maximum(soup.A, soup.B), soup.C)
        order = morton_order((soup.A + soup.B + soup.C) / 3.0)
        nleaf = max(1, -(-m // L))
        K = int(np.ceil(np.log2(nleaf))) if nleaf > 1 else 0
        P2 = 1 << K
        tri = -np.ones(P2 * L, np.int64)
        tri[:m] = order
        self.leaf_tri = tri.reshape(P2, L)
        valid = self.leaf_tri >= 0
        ti = np.where(valid, self.leaf_tri, 0)
        lo = np.where(valid[..., None], tmin[ti], np.inf).min(axis=1)
        hi = np.where(valid[..., None], tmax[ti], -np.inf).max(axis=1)
        los, his = [lo], [hi]
        for _ in range(K):
            lo = np.minimum(lo[0::2], lo[1::2])
            hi = np.maximum(hi[0::2], hi[1::2])
            los.append(lo); his.append(hi)
        self.lo = los[::-1]                     # lo[0] = root ... lo[K] = leaves
        self.hi = his[::-1]
        self.K = K
        self.m = m

    def nbytes(self):
        return sum(a.nbytes for a in self.lo) + sum(a.nbytes for a in self.hi) + self.leaf_tri.nbytes

    @staticmethod
    def _lb2(P, lo, hi):
        d = np.maximum(np.maximum(lo - P, P - hi), 0.0)
        return np.einsum('ij,ij->i', d, d)

    def _dive_key(self, P, k, node):
        """Greedy-dive preference: distance to the box, ties (point inside the box) broken by the distance to its centre."""
        lo, hi = self.lo[k][node], self.hi[k][node]
        with np.errstate(invalid='ignore'):
            c = 0.5 * (lo + hi)
        c = np.where(np.isfinite(c), c, 1e30)
        dc = P - c
        return self._lb2(P, lo, hi) * 1e3 + np.einsum('ij,ij->i', dc, dc)

    def query(self, soup, P, chunk=2048):
        """Nearest triangle of every point.  Returns (d2, tri, cp, reg, diff) (all per point)."""
        n = len(P)
        d2o = np.empty(n); tio = np.empty(n, np.int64); cpo = np.empty((n, 3)); rgo = np.empty(n, np.int8)
        for s in range(0, n, chunk):
            r = self._query_chunk(soup, P[s:s + chunk])
            d2o[s:s + chunk], tio[s:s + chunk], cpo[s:s + chunk], rgo[s:s + chunk] = r
        return d2o, tio, cpo, rgo

    def _leaf_pairs(self, soup, P, pt, leaf):
        """Exact closest point of every (point, leaf) pair's triangles.  Returns pi (point), ti (tri), cp, reg, d2 per triangle pair."""
        tl = self.leaf_tri[leaf]                                    # (np, L)
        ok = tl >= 0
        a, b = np.nonzero(ok)
        ti = tl[a, b]
        pi = pt[a]
        cp, reg = closest_points(P[pi], soup.A[ti], soup.B[ti], soup.C[ti])
        diff = P[pi] - cp
        return pi, ti, cp, reg, np.einsum('ij,ij->i', diff, diff)

    def _query_chunk(self, soup, P):
        n, K = len(P), self.K
        # 1. greedy dive: a real leaf near each point gives a tight exact upper bound
        node = np.zeros(n, np.int64)
        for k in range(K):
            c0 = node * 2
            node = c0 + (self._dive_key(P, k + 1, c0 + 1) < self._dive_key(P, k + 1, c0))
        pi, ti, cp, reg, d2 = self._leaf_pairs(soup, P, np.arange(n), node)
        ub = np.full(n, np.inf)
        if len(pi):
            st = np.flatnonzero(np.r_[True, pi[1:] != pi[:-1]])
            ub[pi[st]] = np.minimum.reduceat(d2, st)
        tol = 1.0 + 1e-9
        # 2. breadth-first traversal of everything that can still beat the bound
        fp = np.arange(n); fn = np.zeros(n, np.int64)
        for k in range(K):
            fp = np.repeat(fp, 2)
            fn = (np.repeat(fn, 2) * 2) + np.tile(np.array([0, 1], np.int64), len(fn))
            keep = self._lb2(P[fp], self.lo[k + 1][fn], self.hi[k + 1][fn]) <= ub[fp] * tol + 1e-18
            fp, fn = fp[keep], fn[keep]
            if k + 1 > self.TIGHTEN_FROM and k + 1 < K and len(fp):
                # tighten the bound with one real triangle (the first of the subtree) per surviving node
                rep = self.leaf_tri[fn << (K - k - 1), 0]
                okr = rep >= 0
                if okr.any():
                    pr, tr = fp[okr], rep[okr]
                    cpr, _r = closest_points(P[pr], soup.A[tr], soup.B[tr], soup.C[tr])
                    dd = P[pr] - cpr
                    d2r = np.einsum('ij,ij->i', dd, dd)
                    st = np.flatnonzero(np.r_[True, pr[1:] != pr[:-1]])
                    ub[pr[st]] = np.minimum(ub[pr[st]], np.minimum.reduceat(d2r, st))
                    keep = self._lb2(P[fp], self.lo[k + 1][fn], self.hi[k + 1][fn]) <= ub[fp] * tol + 1e-18
                    fp, fn = fp[keep], fn[keep]
        pi, ti, cp, reg, d2 = self._leaf_pairs(soup, P, fp, fn)
        keep = d2 <= ub[pi] * tol + 1e-18
        pi, ti, cp, reg, d2 = pi[keep], ti[keep], cp[keep], reg[keep], d2[keep]
        return _reduce_pairs(P, pi, ti, cp, reg, d2, n)


def _reduce_pairs(P, pi, ti, cp, reg, d2, n):
    """Per point nearest (first within a 1e-9 tie band) of sorted candidate pairs.  Returns (d2, tri, cp, reg)."""
    starts = np.flatnonzero(np.r_[True, pi[1:] != pi[:-1]])
    dmin = np.minimum.reduceat(d2, starts)
    rowmin = np.empty(n); rowmin[pi[starts]] = dmin
    hit = d2 <= rowmin[pi] * (1.0 + 1e-9) + 1e-18
    hi = np.flatnonzero(hit)
    first = hi[np.r_[True, pi[hi][1:] != pi[hi][:-1]]]
    out_p = pi[first]
    d2o = np.empty(n); tio = np.empty(n, np.int64); cpo = np.empty((n, 3)); rgo = np.empty(n, np.int8)
    d2o[out_p], tio[out_p], cpo[out_p], rgo[out_p] = d2[first], ti[first], cp[first], reg[first]
    return d2o, tio, cpo, rgo


BVH_MIN_TRIS = 192          # below this the chunk-pruned brute force is as fast and needs no build


def get_bvh(soup):
    if soup.bvh is None or soup.bvh.m != len(soup):
        soup.bvh = TriBVH(soup)
    return soup.bvh


def _sign_of(soup, P, d2, ti, cp, reg):
    pn = np.where((reg == 0)[:, None], soup.fn[ti], 0.0)
    for rcode, which, kind in ((1, 0, 'v'), (2, 1, 'v'), (3, 2, 'v'), (4, 0, 'e'), (5, 1, 'e'), (6, 2, 'e')):
        mm = reg == rcode
        if mm.any():
            pn[mm] = (soup.vn if kind == 'v' else soup.en)[ti[mm], which]
    return np.where(np.einsum('ij,ij->i', P - cp, pn) >= 0.0, 1.0, -1.0)


def closest_surface(soup, points):
    """Nearest surface point of every query point: (cp (n,3), signed distance (n,), triangle index (n,)).  Sign from the
    angle-weighted pseudo-normal of the nearest feature (robust on open shells).  O(log n) per point through the BVH."""
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(soup) == 0 or len(P) == 0:
        return P.copy(), np.full(len(P), 1e9), np.zeros(len(P), np.int64)
    d2, ti, cp, reg = get_bvh(soup).query(soup, P)
    return cp, np.sqrt(d2) * _sign_of(soup, P, d2, ti, cp, reg), ti


def signed_distance(soup, points, chunk=96):
    """Exact signed distance (negative on the side opposite the pseudo-normal, i.e. 'inside').

    Soups of BVH_MIN_TRIS or more triangles are answered by a BVH traversal (O(log n + local) per point, the index is built once and
    cached on the soup); smaller ones by the chunk-pruned brute force.  Both return the same values."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(soup) < BVH_MIN_TRIS:
        return signed_distance_brute(soup, points, chunk)
    if len(soup) == 0 or len(points) == 0:
        return np.full(len(points), 1e9)
    d2, ti, cp, reg = get_bvh(soup).query(soup, points)
    return np.sqrt(d2) * _sign_of(soup, points, d2, ti, cp, reg)


# ----------------------------------------------------------------------------- narrow-band brick grid


class BrickGrid:
    """Sparse narrow-band SDF: 8^3-cell bricks (9^3 nodes with a shared border, float32) stored only where the surface is within
    `band` voxels; everything else is answered from one coarse value per brick (sign and a lower bound |d| >= band*h).
    Bricks are kept in a sorted key array + slot pool, so they can be ADDED (add_bricks, exact values from the BVH) and REMOVED
    (remove_bricks, slots are recycled) at any time.  Memory is proportional to the surface area, not the bounding box volume."""
    B = 8

    def __init__(self, soup, origin, h, dims, band=3.0):
        self.soup, self.origin, self.h, self.band = soup, np.asarray(origin, float), float(h), float(band)
        self.dims = np.asarray(dims, int)
        self.nb = np.maximum(-(-(self.dims - 1) // self.B), 1)
        ii = np.stack(np.meshgrid(*[np.arange(n) for n in self.nb], indexing='ij'), -1).reshape(-1, 3)
        ctr = self.origin + (ii + 0.5) * (self.B * self.h)
        self.coarse = signed_distance(soup, ctr).astype(np.float32).reshape(self.nb)
        self.keys = np.zeros(0, np.int64)
        self.slots = np.zeros(0, np.int64)
        self.pool = np.zeros((0, self.B + 1, self.B + 1, self.B + 1), np.float32)
        self._free = []
        self.stamp = np.zeros(0, np.int64)
        self._clock = 0
        reach = self.band * self.h + 0.5 * np.sqrt(3.0) * self.B * self.h
        self.add_bricks(np.flatnonzero(np.abs(self.coarse.reshape(-1)) <= reach))

    # -- bookkeeping
    def __len__(self):
        return len(self.keys)

    def nbytes(self):
        return self.pool.nbytes + self.keys.nbytes + self.slots.nbytes + self.coarse.nbytes

    def dense_nbytes(self):
        """What a dense float64 node grid of the same extent would cost."""
        return int(np.prod(self.dims)) * 8

    def _find(self, ids):
        pos = np.searchsorted(self.keys, ids)
        pos_c = np.minimum(pos, max(len(self.keys) - 1, 0))
        ok = (self.keys[pos_c] == ids) if len(self.keys) else np.zeros(len(ids), bool)
        return np.where(ok, self.slots[pos_c] if len(self.keys) else -1, -1)

    def _unravel(self, ids):
        nz, ny = self.nb[2], self.nb[1]
        return np.stack([ids // (ny * nz), (ids // nz) % ny, ids % nz], 1)

    # -- add / remove
    def add_bricks(self, ids):
        """Compute and store the bricks `ids` (linear brick ids) that are not yet present; returns the number added."""
        ids = np.unique(np.asarray(ids, np.int64))
        ids = ids[self._find(ids) < 0] if len(self.keys) else ids
        if len(ids) == 0:
            return 0
        B1 = self.B + 1
        lin = np.arange(B1) * self.h
        off = np.stack(np.meshgrid(lin, lin, lin, indexing='ij'), -1).reshape(-1, 3)
        corner = self.origin + self._unravel(ids) * (self.B * self.h)
        for s in range(0, len(ids), 64):
            c = corner[s:s + 64]
            d = signed_distance(self.soup, (c[:, None, :] + off[None]).reshape(-1, 3)).reshape(len(c), B1, B1, B1)
            self._store(ids[s:s + 64], d.astype(np.float32))
        return len(ids)

    def _store(self, ids, d):
        n = len(ids)
        take = [self._free.pop() for _ in range(min(n, len(self._free)))]
        need = n - len(take)
        if need:
            start = len(self.pool)
            self.pool = np.concatenate([self.pool, np.zeros((max(need, start // 2),) + self.pool.shape[1:], np.float32)])
            self.stamp = np.concatenate([self.stamp, np.zeros(len(self.pool) - len(self.stamp), np.int64)])
            take += list(range(start, start + need))
            self._free += list(range(start + need, len(self.pool)))[::-1]
        slot = np.array(take, np.int64)
        self.pool[slot] = d
        self._clock += 1
        self.stamp[slot] = self._clock
        keys = np.concatenate([self.keys, ids]); slots = np.concatenate([self.slots, slot])
        o = np.argsort(keys, kind='stable')
        self.keys, self.slots = keys[o], slots[o]

    def remove_bricks(self, ids):
        """Drop the bricks `ids` (absent ones are ignored); their slots are recycled.  Returns the number removed."""
        ids = np.unique(np.asarray(ids, np.int64))
        if len(self.keys) == 0 or len(ids) == 0:
            return 0
        pos = np.searchsorted(self.keys, ids)
        pos = pos[pos < len(self.keys)]
        pos = pos[np.isin(self.keys[pos], ids)]
        if len(pos) == 0:
            return 0
        self._free += self.slots[pos].tolist()
        keep = np.ones(len(self.keys), bool); keep[pos] = False
        self.keys, self.slots = self.keys[keep], self.slots[keep]
        return len(pos)

    def trim(self, max_bricks):
        """Evict the oldest bricks until at most `max_bricks` remain."""
        extra = len(self.keys) - int(max_bricks)
        if extra <= 0:
            return 0
        old = np.argsort(self.stamp[self.slots], kind='stable')[:extra]
        return self.remove_bricks(self.keys[old])

    # -- sampling
    def sample(self, P):
        P = np.asarray(P, float).reshape(-1, 3)
        g = (P - self.origin) / (self.B * self.h)
        bi = np.clip(np.floor(g).astype(np.int64), 0, self.nb - 1)
        ids = (bi[:, 0] * self.nb[1] + bi[:, 1]) * self.nb[2] + bi[:, 2]
        c = self.coarse[bi[:, 0], bi[:, 1], bi[:, 2]].astype(np.float64)
        out = np.where(c >= 0, 1.0, -1.0) * np.maximum(np.abs(c), self.band * self.h)
        slot = self._find(ids) if len(self.keys) else -np.ones(len(P), np.int64)
        sel = np.flatnonzero(slot >= 0)
        if len(sel):
            f = np.clip(g[sel] - bi[sel], 0.0, 1.0) * self.B
            j0 = np.minimum(np.floor(f).astype(np.int64), self.B - 1)
            fr = f - j0
            b = self.pool[slot[sel]]
            ar = np.arange(len(sel))

            def q(dx, dy, dz):
                return b[ar, j0[:, 0] + dx, j0[:, 1] + dy, j0[:, 2] + dz].astype(np.float64)
            fx, fy, fz = fr[:, 0], fr[:, 1], fr[:, 2]
            c00 = q(0, 0, 0) * (1 - fx) + q(1, 0, 0) * fx
            c10 = q(0, 1, 0) * (1 - fx) + q(1, 1, 0) * fx
            c01 = q(0, 0, 1) * (1 - fx) + q(1, 0, 1) * fx
            c11 = q(0, 1, 1) * (1 - fx) + q(1, 1, 1) * fx
            out[sel] = (c00 * (1 - fy) + c10 * fy) * (1 - fz) + (c01 * (1 - fy) + c11 * fy) * fz
        return out


# ----------------------------------------------------------------------------- per-part field


def _trilerp(vals, i0, f):
    x, y, z = i0[:, 0], i0[:, 1], i0[:, 2]
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    c000, c100 = vals[x, y, z], vals[x + 1, y, z]
    c010, c110 = vals[x, y + 1, z], vals[x + 1, y + 1, z]
    c001, c101 = vals[x, y, z + 1], vals[x + 1, y, z + 1]
    c011, c111 = vals[x, y + 1, z + 1], vals[x + 1, y + 1, z + 1]
    c00 = c000 * (1 - fx) + c100 * fx
    c10 = c010 * (1 - fx) + c110 * fx
    c01 = c001 * (1 - fx) + c101 * fx
    c11 = c011 * (1 - fx) + c111 * fx
    return (c00 * (1 - fy) + c10 * fy) * (1 - fz) + (c01 * (1 - fy) + c11 * fy) * fz


def _thin_barriers(sd, ax, h):
    """Walls thinner than one voxel fall between two nodes that are both outside the material.  Detect them
    from the V-shape of the distance field along the axis: approaching, receding, and the two distances add up
    to less than the node spacing.  Returns B (bool, length n-1 along `ax`): barrier between node k and k+1."""
    m = np.moveaxis(sd, ax, 0)
    if m.shape[0] < 4:
        return np.zeros_like(m[:-1], bool)
    d0, d1, d2, d3 = m[:-3], m[1:-2], m[2:-1], m[3:]
    inner = (d1 > 0) & (d2 > 0) & ((d1 - d0) < -0.25 * h) & ((d3 - d2) > 0.25 * h) & ((d1 + d2) < 0.95 * h)
    B = np.zeros_like(m[:-1], bool)
    B[1:-1] = inner
    return B


def _enclosure(material, sd=None, h=None):
    """Number (0..6) of axis directions in which each node is blocked by material (or a thin wall) further along."""
    cnt = np.zeros(material.shape, np.int8)
    blocked = {}
    for ax in range(3):
        m = np.moveaxis(material, ax, 0)
        B = np.moveaxis(_thin_barriers(sd, ax, h), 0, 0) if sd is not None else np.zeros_like(m[:-1], bool)
        # forward: material strictly beyond node i, or a barrier between k,k+1 with k >= i
        fwd_m = np.flip(np.maximum.accumulate(np.flip(m, 0), 0), 0)
        fwd_ex = np.concatenate([fwd_m[1:], np.zeros_like(fwd_m[:1])], 0)
        fwd_b = np.flip(np.maximum.accumulate(np.flip(B, 0), 0), 0)               # barrier at k >= i
        fwd_ex = fwd_ex | np.concatenate([fwd_b, np.zeros_like(fwd_b[:1])], 0)
        bwd_m = np.maximum.accumulate(m, 0)
        bwd_ex = np.concatenate([np.zeros_like(bwd_m[:1]), bwd_m[:-1]], 0)
        bwd_b = np.maximum.accumulate(B, 0)                                          # barrier at k <= i-1
        bwd_ex = bwd_ex | np.concatenate([np.zeros_like(bwd_b[:1]), bwd_b], 0)
        blocked[(ax, 1)] = np.moveaxis(fwd_ex, 0, ax)
        blocked[(ax, -1)] = np.moveaxis(bwd_ex, 0, ax)
        cnt += np.moveaxis(fwd_ex.astype(np.int8), 0, ax) + np.moveaxis(bwd_ex.astype(np.int8), 0, ax)
    _enclosure.last_blocked = blocked
    return cnt


class PartField:
    """Local-space collision field of one part with dynamic (LOTBCBP) brick refinement.

    mode: 'SHELL' (the surface encloses the fluid: inside = cavity), 'SOLID' (mesh is a body with
    modelled cavities: fluid only in enclosed cavities) or 'AUTO'."""

    # LTCBP error probe (the website's live watch, run once per build): refine x1.25 above P90_HI voxels of error, coarsen /1.15
    # below P90_LO, resolution scale kept within [SCALE_MIN, SCALE_MAX] of the requested one.
    P90_HI, P90_LO, SCALE_MIN, SCALE_MAX, N_PROBES, MAX_BUILDS = 0.75, 0.15, 0.6, 2.0, 96, 4

    def __init__(self, verts, tris, mode='AUTO', resolution=40, level=2, smooth=True, adaptive=False):
        """adaptive=True runs the error-probe loop: build, probe the grid's distance error in voxel units on sampled surface
        points, and rebuild at a finer / coarser resolution (bounded) until the error sits in the target band.
        `resolution_chosen`, `probe_p90` and `probe_history` report what happened (adaptive=False: one build, probe not run)."""
        self.soup = soup_from_mesh(verts, tris)
        self.mode_req = mode
        self.level = int(level)
        self.smooth = smooth
        self.openings_loops = boundary_loops(verts, tris) if len(tris) else []
        self.resolution_requested = int(resolution)
        self.resolution_chosen = int(resolution)
        self.probe_p90 = None
        self.probe_history = []
        self._build(resolution)
        if adaptive and len(self.soup):
            self._adapt(resolution)

    def _adapt(self, base):
        cache = {}
        res = base
        err = self.probe_error()
        self.probe_history.append((res, err))
        cache[res] = (err, None)
        for _ in range(self.MAX_BUILDS - 1):
            if err > self.P90_HI and res * 1.25 <= base * self.SCALE_MAX + 1e-9:
                nres = int(round(res * 1.25))
            elif err < self.P90_LO and res / 1.15 >= base * self.SCALE_MIN - 1e-9:
                nres = int(round(res / 1.15))
            else:
                break
            if nres == res or nres in cache or nres < 8:
                break
            keep_state = self.__dict__.copy()
            self._build(nres)
            nerr = self.probe_error()
            self.probe_history.append((nres, nerr))
            cache[nres] = (nerr, None)
            if nres < res and nerr > self.P90_HI:        # coarsening overshot: back to the previous grid
                self.__dict__.update(keep_state)
                break
            res, err = nres, nerr
        self.resolution_chosen = int(res)
        self.probe_p90 = float(err)

    def probe_error(self, n=None, seed=7):
        """p90 of |grid sd - exact sd| / h over `n` probes placed within one voxel beside the surface (bricks excluded)."""
        n = n or self.N_PROBES
        s = self.soup
        rng = np.random.default_rng(seed)
        area = 0.5 * np.linalg.norm(np.cross(s.B - s.A, s.C - s.A), axis=1)
        t = rng.choice(len(s), n, p=area / area.sum())
        u, v = rng.random(n), rng.random(n)
        flip = u + v > 1.0
        u, v = np.where(flip, 1 - u, u), np.where(flip, 1 - v, v)
        P = s.A[t] + (s.B[t] - s.A[t]) * u[:, None] + (s.C[t] - s.A[t]) * v[:, None]
        P = P + s.fn[t] * rng.uniform(-1.0, 1.0, (n, 1)) * self.h
        err = np.abs(self.sample_sd(P, bricks=False) - signed_distance(s, P)) / self.h
        return float(np.percentile(err, 90))

    def _build(self, resolution):
        mode = self.mode_req
        lo, hi = self.soup.bounds() if len(self.soup) else (np.zeros(3), np.ones(3) * 1e-3)
        self.bmin, self.bmax = lo, hi
        ext = np.maximum(hi - lo, 1e-6)
        self.h = float(ext.max() / max(resolution, 8))
        pad = 2.0 * self.h
        self.origin = lo - pad
        self.dims = np.maximum(np.ceil((hi + pad - self.origin) / self.h).astype(int) + 1, 4)
        sd = self._build_sd()
        self.sd = sd
        # --- cavity classification
        material = sd < 0.0
        enc = _enclosure(material, sd, self.h)
        self._blocked = _enclosure.last_blocked
        solid_cav = (sd > 0.0) & (enc >= 4)
        auto_solid = solid_cav.sum() >= max(20, int(0.005 * sd.size))
        self.mode = mode if mode in ('SHELL', 'SOLID') else ('SOLID' if auto_solid else 'SHELL')
        if self.mode == 'SOLID':
            self.cavm = solid_cav.astype(np.float64)
            # keep only the biggest-agreement enclosed region by dropping speckles smaller than 8 nodes
        else:
            self.cavm = (sd < 0.0).astype(np.float64)
        # a cavity cannot extend past the modelled bounds (pad nodes beyond open ends are ambiguous)
        node_pos = None
        inb = np.ones(self.dims, bool)
        for ax in range(3):
            coord = self.origin[ax] + np.arange(self.dims[ax]) * self.h
            ok = (coord >= lo[ax] - 1e-9) & (coord <= hi[ax] + 1e-9)
            shape = [1, 1, 1]; shape[ax] = -1
            inb &= ok.reshape(shape)
        self.cavm = self.cavm * inb
        self.wall_t = 1.0 * self.h
        # --- brick storage
        self.cells = self.dims - 1
        # refined bricks: sorted cell-id keys -> slots of a recycled pool (sparse; no dense cell-sized index)
        self.F = 2 ** max(self.level, 1)
        self._bkey = np.zeros(0, np.int64)
        self._bslot = np.zeros(0, np.int64)
        self._bstamp = np.zeros(0, np.int64)
        self.bricks = np.zeros((0, self.F + 1, self.F + 1, self.F + 1))
        self._bfree = []
        self._clock = 0
        self._band_max = 0.0
        self.n_refined = 0
        self._compute_openings()

    def _build_sd(self):
        """Exact signed distance on a half-resolution lattice, interpolated to full resolution, then recomputed
        exactly wherever the surface is near (the only place accuracy matters)."""
        dims, h = self.dims, self.h
        cidx = [np.arange(0, d, 2) for d in dims]
        cd = np.array([len(c) for c in cidx])
        gi = np.stack(np.meshgrid(*cidx, indexing='ij'), -1).reshape(-1, 3)
        coarse = signed_distance(self.soup, self.origin + gi * h).reshape(cd)
        fi = np.stack(np.meshgrid(*[np.arange(d) for d in dims], indexing='ij'), -1).reshape(-1, 3)
        g = np.clip(fi / 2.0, 0, cd - 1 - 1e-9)
        i0 = np.floor(g).astype(np.int64)
        sd = _trilerp(coarse, i0, g - i0)
        near = np.flatnonzero(np.abs(sd) < 3.0 * h)
        if len(near):
            sd[near] = signed_distance(self.soup, self.origin + fi[near] * h)
        return sd.reshape(dims)

    # ------------------------------------------------------------------ sampling
    def _locate(self, P):
        g = (P - self.origin) / self.h
        gc = np.clip(g, 0.0, self.dims - 1 - 1e-9)
        i0 = np.floor(gc).astype(np.int64)
        f = gc - i0
        out = np.linalg.norm(g - gc, axis=1) * self.h
        return i0, f, out

    def sample_sd(self, P, bricks=True):
        i0, f, out = self._locate(P)
        val = _trilerp(self.sd, i0, f)
        if bricks and self.n_refined:
            # bricks only exist within (band + one cell diagonal) of the surface: test only those points
            cand = np.flatnonzero(np.abs(val) < self._band_max + 1.8 * self.h)
            cid = (i0[cand, 0] * self.cells[1] + i0[cand, 1]) * self.cells[2] + i0[cand, 2]
            bid_c = self._brick_slots(cid)
            hit = bid_c >= 0
            sel = cand[hit]
            bid = np.full(len(P), -1, np.int64)
            bid[sel] = bid_c[hit]
            if len(sel):
                ff = f[sel] * self.F
                j0 = np.minimum(np.floor(ff).astype(np.int64), self.F - 1)
                fr = ff - j0
                b = self.bricks[bid[sel]]                       # (k,F+1,F+1,F+1)
                ar = np.arange(len(sel))
                jx, jy, jz = j0[:, 0], j0[:, 1], j0[:, 2]
                fx, fy, fz = fr[:, 0], fr[:, 1], fr[:, 2]

                def g(dx, dy, dz):
                    return b[ar, jx + dx, jy + dy, jz + dz]
                c00 = g(0, 0, 0) * (1 - fx) + g(1, 0, 0) * fx
                c10 = g(0, 1, 0) * (1 - fx) + g(1, 1, 0) * fx
                c01 = g(0, 0, 1) * (1 - fx) + g(1, 0, 1) * fx
                c11 = g(0, 1, 1) * (1 - fx) + g(1, 1, 1) * fx
                val[sel] = (c00 * (1 - fy) + c10 * fy) * (1 - fz) + (c01 * (1 - fy) + c11 * fy) * fz
        return val + out

    def sample_cav_mask(self, P):
        i0, f, _o = self._locate(P)
        return _trilerp(self.cavm, i0, f)

    def fields(self, P, bricks=True):
        """Local-space (cav_sdf, mat_sdf): fluid-legal side negative / wall-material side negative."""
        sd = self.sample_sd(P, bricks)
        if self.mode == 'SHELL':
            return sd, np.abs(sd) - 0.5 * self.wall_t
        inside = self.sample_cav_mask(P) > 0.5
        a = np.abs(sd)
        return np.where(inside, -np.maximum(a, 1e-6), a + 1e-6), sd

    # ------------------------------------------------------------------ LOTBCBP dynamic refinement
    def request_refine(self, P, band=None, max_new=96):
        """Refine the bricks around local points that are near the surface.  Returns number added."""
        if self.level <= 0 or len(P) == 0:
            return 0
        band = band if band is not None else 1.5 * self.h
        i0, _f, out = self._locate(P)
        val = _trilerp(self.sd, i0, np.zeros((len(P), 3)))
        near = (np.abs(val) < band) & (out < self.h)
        if not near.any():
            return 0
        cid = (i0[near, 0] * self.cells[1] + i0[near, 1]) * self.cells[2] + i0[near, 2]
        cid = np.unique(cid)
        self._band_max = max(self._band_max, float(band))
        cid = cid[self._brick_slots(cid) < 0][:max_new]
        if len(cid) == 0:
            return 0
        cz = cid % self.cells[2]
        cy = (cid // self.cells[2]) % self.cells[1]
        cx = cid // (self.cells[2] * self.cells[1])
        corner = self.origin + np.stack([cx, cy, cz], 1) * self.h               # (nb,3)
        lin = np.arange(self.F + 1) / self.F * self.h
        gx, gy, gz = np.meshgrid(lin, lin, lin, indexing='ij')
        off = np.stack([gx, gy, gz], -1).reshape(-1, 3)                            # ((F+1)^3,3)
        nodes = (corner[:, None, :] + off[None]).reshape(-1, 3)
        # local soup: triangles near the requested cells, virtually tessellated
        lo, hi = corner.min(0) - band * 2, corner.max(0) + self.h + band * 2
        s = self.soup
        tmin = np.minimum(np.minimum(s.A, s.B), s.C)
        tmax = np.maximum(np.maximum(s.A, s.B), s.C)
        sel = np.flatnonzero(np.all(tmax >= lo, 1) & np.all(tmin <= hi, 1))
        sub = s.take(sel) if len(sel) else s
        sub = virtual_subdivide(sub, max_edge=self.h * 4.0, cap=6000, phong=0.6 if self.smooth else 0.0, passes=6)
        d = signed_distance(sub, nodes).reshape(len(cid), self.F + 1, self.F + 1, self.F + 1)
        self._add_bricks(cid, d)
        return len(cid)

    # ------------------------------------------------------------------ sparse brick store (add / remove)
    def _brick_slots(self, cid):
        """Pool slot of each refined cell id (-1 = not refined)."""
        if len(self._bkey) == 0:
            return -np.ones(len(cid), np.int64)
        pos = np.minimum(np.searchsorted(self._bkey, cid), len(self._bkey) - 1)
        return np.where(self._bkey[pos] == cid, self._bslot[pos], -1)

    def _add_bricks(self, cid, d):
        n = len(cid)
        slot = [self._bfree.pop() for _ in range(min(n, len(self._bfree)))]
        need = n - len(slot)
        if need:
            start = len(self.bricks)
            grow = max(need, start // 2)
            self.bricks = np.concatenate([self.bricks, np.zeros((grow,) + self.bricks.shape[1:])]) if start else np.zeros((grow,) + d.shape[1:])
            self._bstamp = np.concatenate([self._bstamp, np.zeros(grow, np.int64)])
            slot += list(range(start, start + need))
            self._bfree += list(range(start + need, start + grow))[::-1]
        slot = np.array(slot, np.int64)
        self.bricks[slot] = d
        self._clock += 1
        self._bstamp[slot] = self._clock
        keys = np.concatenate([self._bkey, cid]); slots = np.concatenate([self._bslot, slot])
        o = np.argsort(keys, kind='stable')
        self._bkey, self._bslot = keys[o], slots[o]
        self.n_refined = len(self._bkey)

    def remove_bricks(self, cid):
        """Coarsen: drop the refined bricks of the given cell ids (absent ones ignored).  Returns the number removed."""
        cid = np.unique(np.asarray(cid, np.int64))
        if len(self._bkey) == 0 or len(cid) == 0:
            return 0
        pos = np.flatnonzero(np.isin(self._bkey, cid))
        if len(pos) == 0:
            return 0
        self._bfree += self._bslot[pos].tolist()
        keep = np.ones(len(self._bkey), bool); keep[pos] = False
        self._bkey, self._bslot = self._bkey[keep], self._bslot[keep]
        self.n_refined = len(self._bkey)
        return len(pos)

    def coarsen(self, P_local, margin=1.0):
        """Drop every refined brick that no point of `P_local` is within `margin` cells of (simulation moved on).  Returns removed count."""
        if len(self._bkey) == 0:
            return 0
        if len(P_local) == 0:
            return self.remove_bricks(self._bkey.copy())
        i0, _f, _o = self._locate(P_local)
        r = int(np.ceil(margin))
        want = set()
        base = np.unique(i0, axis=0)
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    c = np.clip(base + [dx, dy, dz], 0, self.cells - 1)
                    want.update(((c[:, 0] * self.cells[1] + c[:, 1]) * self.cells[2] + c[:, 2]).tolist())
        stale = self._bkey[~np.isin(self._bkey, np.fromiter(want, np.int64, len(want)))]
        return self.remove_bricks(stale)

    def trim_bricks(self, max_bricks):
        """Evict the oldest refined bricks until at most `max_bricks` remain."""
        extra = len(self._bkey) - int(max_bricks)
        if extra <= 0:
            return 0
        old = np.argsort(self._bstamp[self._bslot], kind='stable')[:extra]
        return self.remove_bricks(self._bkey[old])

    def memory_bytes(self):
        """Bytes held by the grids and the (live part of the) brick pool."""
        return int(self.sd.nbytes + self.cavm.nbytes + self._bkey.nbytes + self._bslot.nbytes
                   + len(self._bkey) * (self.F + 1) ** 3 * 8)

    # ------------------------------------------------------------------ openings (local space)
    def _compute_openings(self):
        ops = []
        if self.mode == 'SHELL':
            for lp in self.openings_loops:
                ops.append(dict(centre=np.asarray(lp['centroid'], float), normal=np.asarray(lp['normal'], float),
                                radius=float(lp['radius'])))
        else:
            cav = self.cavm > 0.5
            air = (self.sd > 0.0) & ~cav
            pts_c, pts_n = [], []
            for ax in range(3):
                for sgn in (1, -1):
                    sh = np.roll(air, -sgn, axis=ax)               # neighbour in direction sgn
                    edge = [slice(None)] * 3
                    edge[ax] = slice(-1, None) if sgn == 1 else slice(0, 1)
                    sh[tuple(edge)] = False
                    mouth = cav & sh & ~self._blocked[(ax, sgn)]
                    ii = np.argwhere(mouth)
                    if len(ii):
                        d = np.zeros(3); d[ax] = sgn
                        pos = self.origin + (ii + 0.5 * d) * self.h
                        pts_c.append(pos); pts_n.append(np.tile(d, (len(ii), 1)))
            if pts_c:
                C = np.concatenate(pts_c); N = np.concatenate(pts_n)
                # cluster by proximity (union of voxels within 1.9 h) using a coarse python flood fill
                lab = -np.ones(len(C), int)
                cur = 0
                key = np.floor(C / (self.h * 2.0)).astype(np.int64)
                from collections import defaultdict
                buckets = defaultdict(list)
                for i, k in enumerate(map(tuple, key)):
                    buckets[k].append(i)
                for s0 in range(len(C)):
                    if lab[s0] >= 0:
                        continue
                    stack = [s0]; lab[s0] = cur
                    while stack:
                        i = stack.pop()
                        ki = key[i]
                        for dx in (-1, 0, 1):
                            for dy in (-1, 0, 1):
                                for dz in (-1, 0, 1):
                                    for j in buckets.get((ki[0] + dx, ki[1] + dy, ki[2] + dz), ()):
                                        if lab[j] < 0 and np.linalg.norm(C[i] - C[j]) < 1.9 * self.h:
                                            lab[j] = cur; stack.append(j)
                    cur += 1
                for L in range(cur):
                    m = lab == L
                    if m.sum() < 3:
                        continue
                    n = N[m].mean(0)
                    nn = np.linalg.norm(n)
                    if nn < 1e-6:
                        continue
                    area = m.sum() * self.h * self.h
                    ops.append(dict(centre=C[m].mean(0), normal=n / nn, radius=float(np.sqrt(area / np.pi))))
        self.openings = ops

    def cavity_volume(self):
        """Estimated cavity volume (local units^3) from the coarse mask."""
        return float(self.cavm.sum() * self.h ** 3)
