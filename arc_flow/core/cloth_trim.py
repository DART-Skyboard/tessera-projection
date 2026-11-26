"""Cloth Trim, Accents and Seam Seal Control Solving (pure NumPy, no bpy).

Three add-ons for ``core.cloth_solver.ClothSolver`` that plug into its ``extra_constraints``
hook (``begin_step`` / ``solve`` / ``post_step``):

* ``Trim``      edge finishing bound to a boundary loop or a vertex chain: hems, piping, lace
                and ruffle borders, belts, collar / cuff bands.  The strip has its own
                stiffness, bend, mass and rest-shape memory, can weight the stiffness of the
                cloth edges next to it, and can flare / ruffle along its length.
* ``AccentSet`` decorative objects (buttons, bows, jewels, sequins, chains, charms, tassels)
                bound to the cloth by closest surface point with barycentric coordinates, an
                orientation frame from the surface, optional pendant sub-particle dynamics.
* ``SeamSeal``  seams that open and close with a 0..1 progress value (zipper, button placket,
                Velcro, magnetic strip, laces): distance constraints with compliance that weld
                when sealed, directional zip progress, an optional tear threshold and a
                gather / fold-over ruffle at the zipper head.

Nothing here touches bpy; the Blender side lives in ``cloth_trim_ops``.
"""
import numpy as np

from . import cloth_solver as CS

# ----------------------------------------------------------------------------- geometry helpers


def vertex_normals(x, tris):
    x = np.asarray(x, float)
    tris = np.asarray(tris, np.int64)
    n = np.zeros_like(x)
    p = x[tris]
    fn = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    for k in range(3):
        np.add.at(n, tris[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(ln, 1e-12)


def boundary_chains(tris, n=None):
    """Ordered boundary loops / chains of a triangle mesh.

    Returns a list of ``(indices, closed)``; the loops are ordered by length (longest first).
    """
    tris = np.asarray(tris, np.int64)
    if len(tris) == 0:
        return []
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    key = np.sort(e, axis=1)
    uniq, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = np.asarray(inv).reshape(-1)
    bnd = uniq[cnt == 1]
    adj = {}
    for a, b in bnd:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))
    seen_e = set()
    out = []
    # start chains at degree-1 vertices first (open chains), then loops
    starts = [v for v, nb in adj.items() if len(nb) == 1] + sorted(adj)
    for s in starts:
        for first in adj[s]:
            ek = (min(s, first), max(s, first))
            if ek in seen_e:
                continue
            chain = [s]
            prev, cur = s, first
            seen_e.add(ek)
            closed = False
            while True:
                chain.append(cur)
                if cur == s:
                    chain.pop()
                    closed = True
                    break
                nxt = [w for w in adj[cur] if w != prev and (min(cur, w), max(cur, w)) not in seen_e]
                if not nxt:
                    break
                seen_e.add((min(cur, nxt[0]), max(cur, nxt[0])))
                prev, cur = cur, nxt[0]
            out.append((np.asarray(chain, np.int64), closed))
    out.sort(key=lambda c: -len(c[0]))
    return out


def edge_band_chain(rest, tris, edges, which='LOWEST', axis=2, band=0.35):
    """Boundary chain along the lowest (or highest) edge of a garment: the boundary vertices
    within ``band`` mesh-edge lengths of the extreme coordinate along ``axis``.  Works for a
    skirt hem loop and for the hem of a slit jacket tube (where the whole boundary is one loop)."""
    rest = np.asarray(rest, float)
    tris = np.asarray(tris, np.int64)
    bnd = np.unique(np.concatenate([c for c, _cl in boundary_chains(tris)])) if len(tris) else np.zeros(0, np.int64)
    if len(bnd) == 0:
        return np.zeros(0, np.int64), False
    e = np.asarray(edges, np.int64)
    med = float(np.median(np.linalg.norm(rest[e[:, 1]] - rest[e[:, 0]], axis=1)))
    z = rest[bnd, axis]
    if which == 'HIGHEST':
        sel = bnd[z >= z.max() - band * med]
    else:
        sel = bnd[z <= z.min() + band * med]
    return order_chain(rest, sel, edges)


def order_chain(rest, idx, edges=None):
    """Order an unordered vertex set into a polyline: mesh-edge walk when ``edges`` link them,
    otherwise a greedy nearest-neighbour walk from an extreme end."""
    idx = np.unique(np.asarray(idx, np.int64))
    if len(idx) < 2:
        return idx, False
    sset = set(int(i) for i in idx)
    if edges is not None:
        adj = {int(i): [] for i in idx}
        for a, b in np.asarray(edges, np.int64):
            if int(a) in sset and int(b) in sset:
                adj[int(a)].append(int(b))
                adj[int(b)].append(int(a))
        if all(len(v) <= 2 for v in adj.values()) and all(len(v) >= 1 for v in adj.values()):
            ends = [v for v, nb in adj.items() if len(nb) == 1]
            start = ends[0] if ends else int(idx[0])
            chain, prev, cur = [start], None, start
            while True:
                nxt = [w for w in adj[cur] if w != prev and w not in chain]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
                chain.append(cur)
            if len(chain) == len(idx):
                return np.asarray(chain, np.int64), (not ends)
    pts = rest[idx]
    c = pts - pts.mean(0)
    ax = np.linalg.svd(c, full_matrices=False)[2][0]
    cur = int(np.argmin(c @ ax))
    left = set(range(len(idx)))
    order = [cur]
    left.discard(cur)
    while left:
        lst = np.fromiter(left, np.int64)
        d = np.linalg.norm(pts[lst] - pts[cur], axis=1)
        cur = int(lst[np.argmin(d)])
        order.append(cur)
        left.discard(cur)
    chain = idx[np.asarray(order)]
    closed = bool(np.linalg.norm(rest[chain[0]] - rest[chain[-1]]) < 1.5 * np.median(
        np.linalg.norm(np.diff(rest[chain], axis=0), axis=1)))
    return chain, closed


def closest_on_triangles(P, V, T, chunk=64):
    """Closest point on a triangle mesh for each point of ``P`` (K,3).

    Returns ``(tri, bary, cp, dist)``: the triangle index, barycentric weights (K,3) of the
    closest point, the point itself and the distance.
    """
    P = np.asarray(P, float).reshape(-1, 3)
    V = np.asarray(V, float)
    T = np.asarray(T, np.int64)
    A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    tri = np.zeros(len(P), np.int64)
    bary = np.zeros((len(P), 3), float)
    cp = np.zeros((len(P), 3), float)
    dist = np.zeros(len(P), float)
    for s in range(0, len(P), chunk):
        p = P[s:s + chunk][:, None, :]            # (k,1,3)
        a, b, c = A[None], B[None], C[None]       # (1,F,3)
        w = _tri_bary(p, a, b, c)                 # (k,F,3)
        q = w[..., 0:1] * a + w[..., 1:2] * b + w[..., 2:3] * c
        d = np.linalg.norm(q - p, axis=2)
        j = np.argmin(d, axis=1)
        r = np.arange(len(j))
        tri[s:s + chunk] = j
        bary[s:s + chunk] = w[r, j]
        cp[s:s + chunk] = q[r, j]
        dist[s:s + chunk] = d[r, j]
    return tri, bary, cp, dist


def _tri_bary(p, a, b, c):
    """Barycentric weights of the point of triangle abc closest to p (Ericson, vectorised)."""
    ab, ac, ap = b - a, c - a, p - a
    d1 = (ab * ap).sum(-1)
    d2 = (ac * ap).sum(-1)
    bp = p - b
    d3 = (ab * bp).sum(-1)
    d4 = (ac * bp).sum(-1)
    cpv = p - c
    d5 = (ab * cpv).sum(-1)
    d6 = (ac * cpv).sum(-1)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    shape = np.broadcast_shapes(d1.shape, d3.shape)
    w = np.zeros(shape + (3,), float)
    done = np.zeros(shape, bool)

    def setw(mask, u, v, ww):
        m = mask & ~done
        w[..., 0] = np.where(m, u, w[..., 0])
        w[..., 1] = np.where(m, v, w[..., 1])
        w[..., 2] = np.where(m, ww, w[..., 2])
        done[:] = done | m

    one, zero = np.ones(shape), np.zeros(shape)
    setw((d1 <= 0) & (d2 <= 0), one, zero, zero)
    setw((d3 >= 0) & (d4 <= d3), zero, one, zero)
    t = d1 / np.where(np.abs(d1 - d3) < 1e-30, 1e-30, d1 - d3)
    setw((vc <= 0) & (d1 >= 0) & (d3 <= 0), 1 - t, t, zero)
    setw((d6 >= 0) & (d5 <= d6), zero, zero, one)
    t = d2 / np.where(np.abs(d2 - d6) < 1e-30, 1e-30, d2 - d6)
    setw((vb <= 0) & (d2 >= 0) & (d6 <= 0), 1 - t, zero, t)
    t = (d4 - d3) / np.where(np.abs((d4 - d3) + (d5 - d6)) < 1e-30, 1e-30, (d4 - d3) + (d5 - d6))
    setw((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), zero, 1 - t, t)
    den = va + vb + vc
    den = np.where(np.abs(den) < 1e-30, 1e-30, den)
    v = vb / den
    ww = vc / den
    setw(np.ones(shape, bool), 1 - v - ww, v, ww)
    return w


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


class _Pairs:
    """A set of distance constraints solved Jacobi-style (corrections averaged per vertex)."""

    def __init__(self, i, j, rest, compliance):
        self.i = np.asarray(i, np.int64)
        self.j = np.asarray(j, np.int64)
        self.rest = np.asarray(rest, float)
        self.compliance = np.broadcast_to(np.asarray(compliance, float), self.rest.shape).copy()
        self.lam = np.zeros(len(self.rest), float)

    def reset(self):
        self.lam[:] = 0.0

    def solve(self, S, h, rest=None, mask=None, omega=1.0):
        if len(self.rest) == 0:
            return
        r = self.rest if rest is None else rest
        i, j, w = self.i, self.j, S.inv_mass
        d = S.x[j] - S.x[i]
        L = np.linalg.norm(d, axis=1)
        n = d / np.maximum(L, 1e-12)[:, None]
        alpha = self.compliance / (h * h)
        wsum = w[i] + w[j] + alpha
        dl = (-(L - r) - alpha * self.lam) / np.maximum(wsum, 1e-30)
        live = (L > 1e-12) & (wsum > 0)
        if mask is not None:
            live &= mask
        dl = np.where(live, dl, 0.0)
        self.lam += dl
        corr = dl[:, None] * n
        dx = np.zeros_like(S.x)
        cnt = np.zeros(len(S.x), float)
        np.add.at(dx, i, -w[i, None] * corr)
        np.add.at(dx, j, w[j, None] * corr)
        np.add.at(cnt, i, live.astype(float))
        np.add.at(cnt, j, live.astype(float))
        S.x += omega * dx / np.maximum(cnt, 1.0)[:, None]


class _Curv:
    """Rod bending with rest-shape memory: |x_i - (x_{i-1} + x_{i+1}) / 2| keeps its rest value
    (zero for a straight hem, the rest curvature for a curved band).  Jacobi XPBD."""

    def __init__(self, chain, closed, rest, compliance):
        n = len(chain)
        if closed:
            ii = np.arange(n)
            self.a, self.b, self.c = chain[(ii - 1) % n], chain, chain[(ii + 1) % n]
        else:
            self.a, self.b, self.c = chain[:-2], chain[1:-1], chain[2:]
        e0 = rest[self.b] - 0.5 * (rest[self.a] + rest[self.c])
        self.rest = np.linalg.norm(e0, axis=1)
        self.compliance = float(compliance)
        self.lam = np.zeros(len(self.rest), float)

    def reset(self):
        self.lam[:] = 0.0

    def solve(self, S, h, omega=1.0):
        if len(self.rest) == 0:
            return
        a, b, c, w = self.a, self.b, self.c, S.inv_mass
        e = S.x[b] - 0.5 * (S.x[a] + S.x[c])
        L = np.linalg.norm(e, axis=1)
        n = e / np.maximum(L, 1e-12)[:, None]
        alpha = self.compliance / (h * h)
        wsum = w[b] + 0.25 * (w[a] + w[c]) + alpha
        dl = (-(L - self.rest) - alpha * self.lam) / np.maximum(wsum, 1e-30)
        dl = np.where((L > 1e-12) | (self.rest > 0), dl, 0.0)
        dl = np.where(L > 1e-12, dl, 0.0)
        self.lam += dl
        dx = np.zeros_like(S.x)
        cnt = np.zeros(len(S.x), float)
        np.add.at(dx, b, w[b, None] * dl[:, None] * n)
        np.add.at(dx, a, -0.5 * w[a, None] * dl[:, None] * n)
        np.add.at(dx, c, -0.5 * w[c, None] * dl[:, None] * n)
        for v in (a, b, c):
            np.add.at(cnt, v, 1.0)
        S.x += omega * dx / np.maximum(cnt, 1.0)[:, None]


# ----------------------------------------------------------------------------- Trim

TRIM_KINDS = {
    'HEM': dict(label="Hem", width=0.025, thickness=0.004, mass_per_m=0.05, stiffness=6.0e4, bend=250.0, edge_weight=2.0, flare=0.0, flare_count=0),
    'PIPING': dict(label="Piping", width=0.008, thickness=0.006, mass_per_m=0.03, stiffness=1.2e5, bend=900.0, edge_weight=1.0, flare=0.0, flare_count=0),
    'LACE': dict(label="Lace / ruffle border", width=0.05, thickness=0.002, mass_per_m=0.025, stiffness=2.0e4, bend=8.0, edge_weight=0.0, flare=0.45, flare_count=14),
    'BAND': dict(label="Belt / band", width=0.06, thickness=0.006, mass_per_m=0.18, stiffness=1.4e5, bend=2500.0, edge_weight=3.0, flare=0.0, flare_count=0),
    'COLLAR': dict(label="Collar / cuff band", width=0.04, thickness=0.005, mass_per_m=0.10, stiffness=1.0e5, bend=1200.0, edge_weight=2.5, flare=0.0, flare_count=0),
}


class Trim:
    """A strip following a chain of cloth vertices (usually a boundary loop)."""

    def __init__(self, chain, closed=False, kind='HEM', **kw):
        pr = dict(TRIM_KINDS.get(kind, TRIM_KINDS['HEM']))
        pr.update(kw)
        self.kind = kind
        self.chain = np.asarray(chain, np.int64)
        self.closed = bool(closed)
        self.width = float(pr['width'])
        self.thickness = float(pr['thickness'])
        self.mass_per_m = float(pr['mass_per_m'])
        self.stiffness = float(pr['stiffness'])
        self.bend = float(pr['bend'])
        self.edge_weight = float(pr['edge_weight'])
        self.flare = float(pr['flare'])
        self.flare_count = int(pr['flare_count'])
        self.memory = float(pr.get('memory', 1.0))
        self.material_slot = int(pr.get('material_slot', 0))
        self.name = pr.get('name', kind.title())
        self.enabled = True
        self._attached = False

    # ---- construction
    def attach(self, S):
        n = len(self.chain)
        ch = self.chain
        self._S = S
        # arc-length parameter
        rest = S.rest
        seg = np.linalg.norm(np.diff(rest[ch], axis=0), axis=1) if n > 1 else np.zeros(0)
        if self.closed and n > 2:
            seg = np.append(seg, np.linalg.norm(rest[ch[0]] - rest[ch[-1]]))
        self.length = float(seg.sum())
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        self.t = cum[:n] / max(self.length, 1e-9)
        # mass: lineal density x segment length shared by its two vertices
        add = np.zeros(S.n, float)
        for k, s in enumerate(seg):
            a = ch[k]
            b = ch[(k + 1) % n]
            add[a] += 0.5 * s * self.mass_per_m
            add[b] += 0.5 * s * self.mass_per_m
        self._mass_added = add
        S.mass = S.mass + add
        free = ~S.pin_hard
        S.inv_mass = np.where(free, 1.0 / S.mass, 0.0)
        # chain constraints: k = 1 (stretch), 2, 3 (rod-like bend with shape memory)
        sets = []
        for k, comp in ((1, 1.0 / max(self.stiffness, 1.0)), (2, 1.0 / max(self.bend, 1e-3)),
                        (3, 2.0 / max(self.bend, 1e-3))):
            if n <= k:
                continue
            if self.closed:
                ii = np.arange(n)
                jj = (ii + k) % n
            else:
                ii = np.arange(n - k)
                jj = ii + k
            r0 = np.linalg.norm(rest[ch[jj]] - rest[ch[ii]], axis=1)
            tm = 0.5 * (self.t[ii] + self.t[jj])
            if self.flare > 0 and k >= 2:
                # surplus length along the strip buckles into ruffles
                mod = 1.0 + self.flare * (0.5 + 0.5 * np.sin(2.0 * np.pi * self.flare_count * tm))
                r0 = r0 * mod
            sets.append(_Pairs(ch[ii], ch[jj], r0, comp))
        if n >= 3:
            sets.append(_Curv(ch, self.closed, rest, 1.0 / max(self.bend, 1e-3)))
        self._sets = sets
        # stiffen the cloth edges next to the trim (hem weight)
        self._ring = []
        if self.edge_weight > 0:
            inchain = np.zeros(S.n, bool)
            inchain[ch] = True
            e = S.stretch_edges
            m = inchain[e[:, 0]] | inchain[e[:, 1]]
            if m.any():
                self._ring.append(_Pairs(e[m, 0], e[m, 1], S.stretch_rest[m],
                                         S.stretch_compliance / self.edge_weight))
            e2 = S.bend_edges
            m2 = inchain[e2[:, 0]] | inchain[e2[:, 1]]
            if m2.any():
                self._ring.append(_Pairs(e2[m2, 0], e2[m2, 1], S.bend_rest[m2],
                                         S.bend_compliance / self.edge_weight))
        if self.flare > 0 and S.frame == 0:
            nrm = vertex_normals(S.x, S.tris)
            seed = 0.004 + 0.02 * self.flare * self.width
            off = seed * np.sin(2.0 * np.pi * self.flare_count * self.t + 0.7)[:, None] * nrm[ch]
            S.x[ch] += off
            S.x_prev[ch] += off
        self._attached = True
        return self

    def detach(self, S):
        if self._attached:
            S.mass = S.mass - self._mass_added
            S.inv_mass = np.where(S.pin_hard, 0.0, 1.0 / S.mass)
            self._attached = False

    # ---- hook
    def begin_step(self, S, dt):
        for s in self._sets:
            s.reset()
        for s in self._ring:
            s.reset()

    def solve(self, S, h):
        if not self.enabled:
            return
        for s in self._sets:
            s.solve(S, h, omega=0.9)
        for s in self._ring:
            s.solve(S, h, omega=0.9)

    # ---- visual strip
    def strip(self, x, tris, inward=None, lift=0.0):
        """Ribbon of (2n) vertices: the chain and a copy offset ``width`` along the surface,
        away from the boundary.  Returns ``(verts, faces)``."""
        ch = self.chain
        n = len(ch)
        nrm = vertex_normals(x, tris)[ch]
        p = x[ch]
        tang = np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)
        if not self.closed:
            tang[0] = p[1] - p[0]
            tang[-1] = p[-1] - p[-2]
        tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-12)
        side = np.cross(nrm, tang)
        side /= np.maximum(np.linalg.norm(side, axis=1, keepdims=True), 1e-12)
        if inward is None:
            inward = x.mean(0)
        # choose the side that points into the cloth (toward the centroid of the patch)
        cen = x[np.unique(S_tris_flat(tris))].mean(0) if len(tris) else inward
        sgn = np.sign(((cen - p) * side).sum(axis=1))
        sgn[sgn == 0] = 1.0
        if n > 2:
            sgn = np.full(n, np.sign(np.median(sgn)) or 1.0)
        p = p + nrm * lift
        q = p + side * sgn[:, None] * self.width
        verts = np.concatenate([p, q])
        faces = []
        rng = range(n) if self.closed else range(n - 1)
        for k in rng:
            a, b = k, (k + 1) % n
            faces.append((a, b, n + b, n + a))
        return verts, np.asarray(faces, np.int64)


def S_tris_flat(tris):
    return np.asarray(tris).ravel()


# ----------------------------------------------------------------------------- Seam seal

SEAM_MODES = {
    'ZIPPER': dict(label="Zipper", directional=True, window=0.10, stride=1, soft=1.0, gather=0.30, ruffle=0.010),
    'BUTTONS': dict(label="Button placket", directional=True, window=0.02, stride=4, soft=1.0, gather=0.0, ruffle=0.0),
    'VELCRO': dict(label="Velcro", directional=True, window=0.45, stride=1, soft=8.0, gather=0.05, ruffle=0.002),
    'MAGNETIC': dict(label="Magnetic strip", directional=False, window=0.0, stride=3, soft=4.0, gather=0.0, ruffle=0.0),
    'LACES': dict(label="Laces", directional=True, window=0.30, stride=2, soft=12.0, gather=0.08, ruffle=0.0),
}


def pair_seam(rest, edges, idx, side_a=None, side_b=None):
    """Pair the vertices of a seam: ``(P, 2)`` indices, side A first.

    With explicit sides every A vertex takes its nearest B vertex.  Otherwise the seam vertex
    set is split into connected components (the two lips of an open slit) and each vertex is
    matched with the nearest vertex of another component (greedy, one to one).
    """
    rest = np.asarray(rest, float)
    if side_a is not None and side_b is not None and len(side_a) and len(side_b):
        a = np.unique(np.asarray(side_a, np.int64))
        b = np.unique(np.asarray(side_b, np.int64))
        comp = {int(v): 0 for v in a}
        comp.update({int(v): 1 for v in b})
        idx = np.concatenate([a, b])
    else:
        idx = np.unique(np.asarray(idx, np.int64))
        parent = {int(v): int(v) for v in idx}

        def find(v):
            while parent[v] != v:
                parent[v] = parent[parent[v]]
                v = parent[v]
            return v
        for p, q in np.asarray(edges, np.int64):
            if int(p) in parent and int(q) in parent:
                parent[find(int(p))] = find(int(q))
        comp = {int(v): find(int(v)) for v in idx}
    if len(set(comp.values())) < 2:
        return np.zeros((0, 2), np.int64)
    # candidate pairs sorted by distance
    ids = np.asarray([int(v) for v in idx])
    cc = np.asarray([comp[int(v)] for v in ids])
    cand = []
    for k, v in enumerate(ids):
        other = np.nonzero(cc != cc[k])[0]
        if len(other) == 0:
            continue
        d = np.linalg.norm(rest[ids[other]] - rest[v], axis=1)
        for o in other[np.argsort(d)[:3]]:
            cand.append((float(np.linalg.norm(rest[ids[o]] - rest[v])), int(v), int(ids[o])))
    cand.sort()
    used, pairs = set(), []
    for _d, u, v in cand:
        if u in used or v in used:
            continue
        used.add(u)
        used.add(v)
        pairs.append((u, v) if comp[u] <= comp[v] else (v, u))
    if not pairs:
        return np.zeros((0, 2), np.int64)
    # side A first (smaller component label first is arbitrary: keep the larger component as A)
    return np.asarray(pairs, np.int64)


class SeamSeal:
    """Seam that closes with ``progress`` 0 (open) .. 1 (sealed)."""

    def __init__(self, rest, pairs, mode='ZIPPER', progress=0.0, strength=1.0, tear=0.0,
                 gather=None, ruffle=None, reverse=False, window=None, stride=None,
                 weld_tol=2e-3, tris=None):
        cfg = dict(SEAM_MODES.get(mode, SEAM_MODES['ZIPPER']))
        self.mode = mode
        self.directional = cfg['directional']
        self.window = float(cfg['window'] if window is None else window)
        self.stride = max(1, int(cfg['stride'] if stride is None else stride))
        self.soft = float(cfg['soft'])
        self.gather = float(cfg['gather'] if gather is None else gather)
        self.ruffle = float(cfg['ruffle'] if ruffle is None else ruffle)
        self.progress = float(progress)
        self.strength = float(strength)
        self.tear = float(tear)
        self.reverse = bool(reverse)
        self.weld_tol = float(weld_tol)
        rest = np.asarray(rest, float)
        pairs = np.asarray(pairs, np.int64).reshape(-1, 2)
        mid = 0.5 * (rest[pairs[:, 0]] + rest[pairs[:, 1]]) if len(pairs) else np.zeros((0, 3))
        if len(pairs) > 1:
            c = mid - mid.mean(0)
            ax = np.linalg.svd(c, full_matrices=False)[2][0]
            if ax[int(np.argmax(np.abs(ax)))] < 0:     # t grows toward +X / +Y / +Z
                ax = -ax
            tt = c @ ax
        else:
            ax = np.array([0.0, 0.0, 1.0])
            tt = np.zeros(len(pairs))
        order = np.argsort(tt)
        pairs, mid, tt = pairs[order], mid[order], tt[order]
        span = max(float(tt.max() - tt.min()), 1e-9) if len(tt) else 1.0
        t = (tt - (tt.min() if len(tt) else 0.0)) / span
        if self.reverse:
            t = 1.0 - t
            order2 = np.argsort(t)
            pairs, mid, t = pairs[order2], mid[order2], t[order2]
        self.pairs = pairs
        self.t = t
        self.axis = ax
        self.gap0 = np.linalg.norm(rest[pairs[:, 1]] - rest[pairs[:, 0]], axis=1) if len(pairs) else np.zeros(0)
        self.active = (np.arange(len(pairs)) % self.stride) == 0
        if len(pairs):
            self.active[0] = True
            self.active[-1] = True
        self.torn = np.zeros(len(pairs), bool)
        self.armed = np.zeros(len(pairs), bool)
        self.lam = np.zeros(len(pairs), float)
        self._it = 0
        self.rest_mid = mid
        self.tris = tris
        self._edges_sets = None
        self.s = np.zeros(len(pairs), float)
        self.pre_gap = np.zeros(len(pairs), float)
        self.update_closure()

    # ---- progress -> per pair closure
    def closure(self, p=None):
        p = self.progress if p is None else p
        p = float(np.clip(p, 0.0, 1.0))
        if not self.directional or self.window <= 1e-9:
            if self.directional:      # step window: all pairs at their own threshold
                return (p * 1.000001 >= self.t).astype(float) * (p > 0)
            return np.full(len(self.t), p)
        w = self.window
        return np.clip((p * (1.0 + w) - self.t) / w, 0.0, 1.0)

    def update_closure(self):
        self.s = self.closure()
        return self.s

    def head_t(self, p=None):
        p = self.progress if p is None else p
        w = max(self.window, 1e-6) if self.directional else 0.0
        return float(np.clip(p * (1.0 + w) - 0.5 * w, 0.0, 1.0))

    def head_position(self, x):
        """World position of the zipper head (between the sealed and open part)."""
        if len(self.pairs) == 0:
            return np.zeros(3)
        mid = 0.5 * (x[self.pairs[:, 0]] + x[self.pairs[:, 1]])
        th = self.head_t()
        return np.array([np.interp(th, self.t, mid[:, k]) for k in range(3)])

    def progress_from_point(self, P, x=None):
        """Slider value for a pull object at world point ``P`` (nearest pair along the seam)."""
        mid = self.rest_mid if x is None else 0.5 * (x[self.pairs[:, 0]] + x[self.pairs[:, 1]])
        k = int(np.argmin(np.linalg.norm(mid - np.asarray(P, float), axis=1)))
        th = float(self.t[k])
        if not self.directional or self.window <= 1e-9:
            return th
        w = self.window
        return float(np.clip((th + 0.5 * w) / (1.0 + w), 0.0, 1.0))

    def compliance(self):
        base = (1.0 - float(np.clip(self.strength, 0.0, 1.0))) * 2e-3 * self.soft
        return base

    # ---- hook
    def attach(self, S):
        self._S = S
        e = S.stretch_edges
        in_a = np.zeros(S.n, bool)
        in_b = np.zeros(S.n, bool)
        in_a[self.pairs[:, 0]] = True
        in_b[self.pairs[:, 1]] = True
        m = (in_a[e[:, 0]] & in_a[e[:, 1]]) | (in_b[e[:, 0]] & in_b[e[:, 1]])
        self._side_edges = np.nonzero(m)[0]
        self._base_rest = S.stretch_rest[self._side_edges].copy()
        pair_of = np.full(S.n, -1, np.int64)
        pair_of[self.pairs[:, 0]] = np.arange(len(self.pairs))
        pair_of[self.pairs[:, 1]] = np.arange(len(self.pairs))
        self._pair_of_edge = np.stack([pair_of[e[self._side_edges, 0]],
                                       pair_of[e[self._side_edges, 1]]], axis=1)
        return self

    def reset(self):
        self.torn[:] = False
        self.armed[:] = False
        self.lam[:] = 0.0

    def detach(self, S):
        if getattr(self, '_side_edges', None) is not None and len(self._side_edges):
            S.stretch_rest[self._side_edges] = self._base_rest

    def effective_closure(self):
        s = self.s * self.active
        s = np.where(self.torn, 0.0, s)
        return s

    def begin_step(self, S, dt):
        self.update_closure()
        self.lam[:] = 0.0
        self._it = 0
        # gather: compress the seam edges near the zipper head so the cloth buckles into folds
        if len(self._side_edges):
            if self.gather > 0:
                s_pair = self.s
                se = 0.5 * (s_pair[self._pair_of_edge[:, 0]] + s_pair[self._pair_of_edge[:, 1]])
                f = 4.0 * se * (1.0 - se)
                S.stretch_rest[self._side_edges] = self._base_rest * (1.0 - self.gather * f)
            else:
                S.stretch_rest[self._side_edges] = self._base_rest
        # fold-over ruffle: a small alternating push out of the surface around the head
        if self.ruffle > 0 and len(self.pairs):
            f = 4.0 * self.s * (1.0 - self.s)
            hot = f > 0.05
            if hot.any():
                nrm = vertex_normals(S.x, S.tris)
                k = np.nonzero(hot)[0]
                sgn = np.where(k % 2 == 0, 1.0, -0.6)
                for side in (0, 1):
                    v = self.pairs[k, side]
                    free = S.inv_mass[v] > 0
                    S.v[v[free]] += (nrm[v[free]] * (self.ruffle * 30.0 * dt) * (f[k][free] * sgn[free])[:, None])

    def solve(self, S, h):
        if len(self.pairs) == 0:
            return
        a, b = self.pairs[:, 0], self.pairs[:, 1]
        d = S.x[b] - S.x[a]
        L = np.linalg.norm(d, axis=1)
        first = (self._it % max(1, int(S.iterations))) == 0
        self._it += 1
        if first:
            self.pre_gap = L.copy()
            if self.tear > 0:
                # a pair can only tear once it has actually been closed (armed)
                sealed = (self.s >= 0.999) & self.active & ~self.torn
                self.torn |= sealed & self.armed & (L > self.tear)
        s = self.effective_closure()
        live = s > 1e-4
        if not live.any():
            return
        target = self.gap0 * (1.0 - s)
        comp = self.compliance()
        weld = (self.strength >= 0.999) & (s >= 0.999)
        alpha = np.where(weld, 0.0, comp) / (h * h)
        w = S.inv_mass
        n = d / np.maximum(L, 1e-12)[:, None]
        wsum = w[a] + w[b] + alpha
        dl = (-(L - target) - alpha * self.lam) / np.maximum(wsum, 1e-30)
        ok = live & (wsum > 0) & (L > 1e-12)
        # coincident vertices of a weld need no direction
        dl = np.where(ok, dl, 0.0)
        self.lam += dl
        corr = dl[:, None] * n
        S.x[a] -= w[a, None] * corr
        S.x[b] += w[b, None] * corr

    def post_step(self, S, dt):
        if self.tear > 0 and len(self.pairs):
            g = self.gaps(S.x)
            self.armed |= (self.s >= 0.999) & self.active & ~self.torn & (g < max(self.weld_tol * 5.0, 0.5 * self.tear))

    # ---- queries
    def gaps(self, x):
        if len(self.pairs) == 0:
            return np.zeros(0)
        return np.linalg.norm(x[self.pairs[:, 1]] - x[self.pairs[:, 0]], axis=1)


# ----------------------------------------------------------------------------- Accents

ACCENT_KINDS = {
    'BUTTON': dict(label="Button", dynamic=False, scale=0.012),
    'BOW': dict(label="Bow", dynamic=False, scale=0.04),
    'JEWEL': dict(label="Jewel", dynamic=False, scale=0.008),
    'SEQUIN': dict(label="Sequin", dynamic=False, scale=0.005),
    'CHARM': dict(label="Charm / pendant", dynamic=True, scale=0.012),
    'TASSEL': dict(label="Tassel", dynamic=True, scale=0.015),
    'CHAIN': dict(label="Chain", dynamic=True, scale=0.006),
    'BROOCH': dict(label="Brooch", dynamic=False, scale=0.03),
}


def points_along_chain(rest, chain, count, closed=False):
    """``count`` points evenly spaced by arc length along a vertex chain."""
    p = np.asarray(rest, float)[np.asarray(chain, np.int64)]
    if closed:
        p = np.vstack([p, p[:1]])
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    count = max(1, int(count))
    if closed:
        u = np.arange(count) / count * cum[-1]
    else:
        u = np.linspace(0.0, cum[-1], count) if count > 1 else np.array([0.5 * cum[-1]])
    return np.stack([np.interp(u, cum, p[:, k]) for k in range(3)], axis=1)


def scatter_points(rest, tris, count=None, density=None, seed=0, weights=None):
    """Area-weighted random surface points.  ``density`` is points per m^2 (overrides count)."""
    rest = np.asarray(rest, float)
    tris = np.asarray(tris, np.int64)
    p = rest[tris]
    area = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    if weights is not None:
        wv = np.asarray(weights, float)[tris].mean(axis=1)
        area = area * np.clip(wv, 0.0, None)
    tot = float(area.sum())
    if tot <= 0:
        return np.zeros((0, 3))
    if density is not None and density > 0:
        count = int(round(float(density) * tot))
    count = max(0, int(count or 0))
    rng = np.random.default_rng(int(seed))
    ti = rng.choice(len(tris), size=count, p=area / tot)
    u = rng.random((count, 2))
    flip = u.sum(axis=1) > 1.0
    u[flip] = 1.0 - u[flip]
    b = np.stack([1.0 - u.sum(axis=1), u[:, 0], u[:, 1]], axis=1)
    return np.einsum('kj,kjc->kc', b, p[ti])


def surface_frames(x, tris, tri, bary):
    """Tangent / binormal / normal frames (K,3,3; columns t, b, n) and points at the binding."""
    T = tris[tri]
    p = x[T]
    pos = np.einsum('kj,kjc->kc', bary, p)
    e0 = p[:, 1] - p[:, 0]
    nrm = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    t = e0 - (e0 * nrm).sum(1, keepdims=True) * nrm
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    b = np.cross(nrm, t)
    F = np.stack([t, b, nrm], axis=2)
    return pos, F


class AccentSet:
    """Decorative instances bound to the cloth at barycentric points."""

    def __init__(self, rest, tris, points, kind='BUTTON', offset=0.0, scale=None, align=True,
                 chain_segments=0, chain_length=0.0, chain_stiffness=0.9, chain_damping=0.04,
                 collide=True, collide_radius=0.006, seed=0):
        cfg = ACCENT_KINDS.get(kind, ACCENT_KINDS['BUTTON'])
        self.kind = kind
        self.rest = np.asarray(rest, float)
        self.tris = np.asarray(tris, np.int64)
        self.scale = float(cfg['scale'] if scale is None else scale)
        self.align = bool(align)
        self.offset = float(offset)
        self.collide = bool(collide)
        self.collide_radius = float(collide_radius)
        self.chain_stiffness = float(chain_stiffness)
        self.chain_damping = float(chain_damping)
        P = np.asarray(points, float).reshape(-1, 3)
        self.tri, self.bary, cp, _d = closest_on_triangles(P, self.rest, self.tris)
        # local offset (point -> surface) in the rest frame, so it survives deformation
        _pos, F0 = surface_frames(self.rest, self.tris, self.tri, self.bary)
        self.local = np.einsum('kji,kj->ki', F0, P - cp)       # F0^T (P - cp)
        self.local[:, 2] += self.offset
        self.K = len(P)
        self.M = int(chain_segments)
        self.chain_seg = float(chain_length) / max(self.M, 1)
        self.dynamic = self.M > 0
        self.vertex_group_weights = None
        self.rng_seed = int(seed)
        self.enabled = True
        self.reset(None)

    # positions of the attachment points for deformed cloth ``x``
    def anchors(self, x):
        pos, F = surface_frames(x, self.tris, self.tri, self.bary)
        return pos + np.einsum('kij,kj->ki', F, self.local), F

    def reset(self, x):
        if x is None:
            x = self.rest
        a, F = self.anchors(x)
        self.F = F
        self.anchor = a
        if self.dynamic:
            down = np.array([0.0, 0.0, -1.0])
            self.p = a[:, None, :] + down[None, None, :] * self.chain_seg * np.arange(1, self.M + 1)[None, :, None]
            self.pp = self.p.copy()
        else:
            self.p = None

    def transforms(self, x):
        """World 4x4 matrices (K,4,4) of the accents (and chain particle paths via ``chains``)."""
        a, F = self.anchors(x)
        self.anchor, self.F = a, F
        R = F if self.align else np.broadcast_to(np.eye(3), F.shape)
        M = np.zeros((self.K, 4, 4), float)
        M[:, :3, :3] = R * self.scale
        M[:, 3, 3] = 1.0
        pos = a
        if self.dynamic:
            # a hanging accent sits at the end of its chain, aligned along the last link
            end = self.p[:, -1, :]
            prev = self.p[:, -2, :] if self.M > 1 else a
            ax = end - prev
            ax /= np.maximum(np.linalg.norm(ax, axis=1, keepdims=True), 1e-12)
            side = F[:, :, 0]
            side = side - (side * ax).sum(1, keepdims=True) * ax
            side /= np.maximum(np.linalg.norm(side, axis=1, keepdims=True), 1e-9)
            third = np.cross(ax, side)
            Rh = np.stack([side, third, ax], axis=2)        # local z follows the chain
            M[:, :3, :3] = Rh * self.scale
            pos = end
        M[:, :3, 3] = pos
        return M

    def chains(self):
        """(K, M+1, 3) polylines: anchor then the particles (empty for static accents)."""
        if not self.dynamic:
            return np.zeros((self.K, 0, 3))
        return np.concatenate([self.anchor[:, None, :], self.p], axis=1)

    def step(self, S, dt):
        """Advance pendant sub-particles one full cloth step."""
        a, F = self.anchors(S.x)
        self.anchor, self.F = a, F
        if not self.dynamic:
            return
        nsub = max(1, int(round(dt / (1.0 / 240.0))))
        h = dt / nsub
        g = S.gravity
        damp = max(0.0, 1.0 - self.chain_damping * 60.0 * h)
        # the anchor moved during the step: interpolate it
        a0 = getattr(self, '_a_prev', a)
        for sidx in range(nsub):
            anc = a0 + (a - a0) * ((sidx + 1) / nsub)
            vel = (self.p - self.pp) * damp
            self.pp = self.p.copy()
            self.p = self.p + vel + g[None, None, :] * h * h
            for _ in range(4):
                prev = anc
                for m in range(self.M):
                    cur = self.p[:, m, :]
                    pr = prev if m == 0 else self.p[:, m - 1, :]
                    d = cur - pr
                    L = np.linalg.norm(d, axis=1, keepdims=True)
                    corr = d * ((self.chain_seg - L) / np.maximum(L, 1e-12))
                    if m == 0:
                        self.p[:, m, :] = cur + corr * self.chain_stiffness
                    else:
                        self.p[:, m - 1, :] = pr - 0.5 * corr * self.chain_stiffness
                        self.p[:, m, :] = cur + 0.5 * corr * self.chain_stiffness
            if self.collide:
                self._collide(S)
        self._a_prev = a

    def _collide(self, S):
        pts = self.p.reshape(-1, 3)
        r = self.collide_radius
        if S.ground_z is not None:
            low = pts[:, 2] < S.ground_z + r
            pts[low, 2] = S.ground_z + r
        # stay on the outer side of the garment the accent hangs on (face-normal side of its anchor)
        tri, bary, cpc, _d = closest_on_triangles(pts, S.x, S.tris)
        T = S.tris[tri]
        p3 = S.x[T]
        fn = np.cross(p3[:, 1] - p3[:, 0], p3[:, 2] - p3[:, 0])
        fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
        side = ((pts - cpc) * fn).sum(axis=1)
        ref = np.repeat(self.F[:, :, 2], self.M, axis=0)       # outward normal at each anchor
        flip = np.sign((fn * ref).sum(axis=1))
        flip = np.where(flip == 0, 1.0, flip)
        sd_side = side * flip
        low = sd_side < r
        if low.any():
            pts[low] = cpc[low] + fn[low] * flip[low, None] * r
        if S.collider is not None:
            from . import sdf
            cp, sd, _ti = sdf.closest_surface(S.collider, pts)
            hit = sd < r
            if hit.any():
                diff = pts[hit] - cp[hit]
                ln = np.linalg.norm(diff, axis=1, keepdims=True)
                nn = np.where(ln > 1e-12, diff / np.maximum(ln, 1e-12), np.array([0.0, 0.0, 1.0]))
                nn[sd[hit] < 0] *= -1.0
                pts[hit] = cp[hit] + nn * r
        self.p = pts.reshape(self.p.shape)

    # hook
    def post_step(self, S, dt):
        if self.enabled:
            self.step(S, dt)


# ----------------------------------------------------------------------------- attach helpers

def attach(S, obj):
    """Register ``obj`` (Trim, SeamSeal, AccentSet) on the solver's extra-constraint hook."""
    if hasattr(obj, 'attach'):
        obj.attach(S)
    if obj not in S.extra_constraints:
        S.extra_constraints.append(obj)
    return obj


def detach(S, obj):
    if hasattr(obj, 'detach'):
        obj.detach(S)
    if obj in S.extra_constraints:
        S.extra_constraints.remove(obj)


def apply_stiff_weight(S, vertex_weights, scale=2.0):
    """Utility: multiply stretch / bend stiffness of the edges touching weighted vertices by
    ``1 + (scale - 1) * w`` (implemented as extra constraints)."""
    w = np.clip(np.asarray(vertex_weights, float), 0.0, 1.0)
    e = S.stretch_edges
    we = 0.5 * (w[e[:, 0]] + w[e[:, 1]])
    m = we > 0.01
    if not m.any():
        return None
    extra = _Pairs(e[m, 0], e[m, 1], S.stretch_rest[m],
                   S.stretch_compliance / np.maximum((scale - 1.0) * we[m], 1e-6))
    holder = type('StiffWeight', (), {})()
    holder.begin_step = lambda S_, dt: extra.reset()
    holder.solve = lambda S_, h: extra.solve(S_, h, omega=0.9)
    S.extra_constraints.append(holder)
    return holder
