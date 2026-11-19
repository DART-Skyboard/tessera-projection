"""Surface sources and fluid volumes for Fluid mode (NumPy, no bpy).  Tessera Projection / LOTBCBP.

Three kinds of source are layered on top of core.fluidmode.FluidSolver without changing its time step:

* ViscositySurface  faces that drag the adjacent fluid.  Every particle within `reach` of a face sample is relaxed toward the
                    surface velocity in the tangential direction: rate = coefficient / (rho reach^2) x no-slip fraction x
                    proximity weight, applied as dv = -(1 - exp(-rate dt)) (v - v_wall)_t (unconditionally stable).  The normal
                    component is left to the collision bodies.  The reaction force on the surface is accumulated (`force`).
* EmissionSurface   faces that emit particles of one fluid: rate (particles/s), speed along the face normal (+ an extra velocity),
                    a spread cone, temperature.  Particles are appended to the solver (`grow`).
* fluid volumes     a box (the matrix and half-extents of an Empty) filled with particles of a fluid at its density, pressure and
                    temperature: `box_mesh` gives the closed mesh that Fluid mode already fills (shape), `gas_density` the amount.

`install(S, viscosity=[...], emission=[...])` wraps S.step / S.snapshot / S.restore on that solver instance (the solver module is
untouched): emission happens before the step, the viscous coupling after it, and snapshots carry the grown arrays so the frame cache
and Reset stay exact.  The emission random stream is keyed by (seed, step index), so replays are deterministic."""
import math

import numpy as np

from . import fluidmode as FM

PERFLUID = {'lo', 'hi', 'rho0', 'Vp', 'molar', 'mass_p', 'visc', 'cp', 'k', 'beta', 'tref', 'coh', 'misc', 'gas', 'cool', 'free', 'drag',
            'diff', 'group', 'sp_misc', 'sp_of_fluid'}


# ----------------------------------------------------------------------------- geometry
def box_mesh(matrix, half):
    """Closed box (8 vertices, 12 triangles, outward normals) of an Empty: `matrix` 4x4 world matrix, `half` half-extents in local units."""
    h = np.asarray(half, float)
    c = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * h
    M = np.asarray(matrix, float)
    V = c @ M[:3, :3].T + M[:3, 3]
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    ctr = V.mean(0)
    tris = []
    for a, b, cc, d in quads:
        for tr in ((a, b, cc), (a, cc, d)):
            tr = np.array(tr)
            p = V[tr]
            n = np.cross(p[1] - p[0], p[2] - p[0])
            tris.append(tr if np.dot(n, p.mean(0) - ctr) > 0 else tr[::-1])
    return V, np.array(tris)


def box_volume(matrix, half):
    M = np.asarray(matrix, float)
    sc = np.linalg.norm(M[:3, :3], axis=0)
    return float(np.prod(2.0 * np.asarray(half, float) * sc))


def gas_density(rho_ref, T=293.15, P=FM.P_ATM, T_ref=293.15):
    """Ideal gas: density at temperature T and pressure P from the preset density at T_ref and 1 atm."""
    return float(rho_ref) * (T_ref / max(T, 1.0)) * (P / FM.P_ATM)


def sample_faces(verts, tris, dx, kmax=12):
    """Sample points of triangles: each triangle is split k x k (k = ceil(longest edge / dx), at most kmax) and the sub-triangle
    centroids are returned with the face normal and the sub-triangle area.  Returns (points, normals, areas, face_index)."""
    V = np.asarray(verts, float)
    T = np.asarray(tris, int)
    if len(T) == 0:
        z = np.zeros((0, 3))
        return z, z.copy(), np.zeros(0), np.zeros(0, int)
    A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    cr = np.cross(B - A, C - A)
    area = 0.5 * np.linalg.norm(cr, axis=1)
    nrm = cr / np.maximum(2.0 * area, 1e-30)[:, None]
    el = np.maximum(np.maximum(np.linalg.norm(B - A, axis=1), np.linalg.norm(C - B, axis=1)), np.linalg.norm(A - C, axis=1))
    k = np.clip(np.ceil(el / max(dx, 1e-9)).astype(int), 1, kmax)
    P, N, W, F = [], [], [], []
    for kk in np.unique(k):
        sel = np.flatnonzero(k == kk)
        # barycentric centroids of the k^2 sub-triangles
        bc = []
        for i in range(kk):
            for j in range(kk - i):
                bc.append(((i + 1.0 / 3.0) / kk, (j + 1.0 / 3.0) / kk))           # "up" triangles
                if i + j < kk - 1:
                    bc.append(((i + 2.0 / 3.0) / kk, (j + 2.0 / 3.0) / kk))       # "down" triangles
        bc = np.array(bc)
        u, v = bc[:, 0], bc[:, 1]
        pts = A[sel][:, None, :] + (B[sel] - A[sel])[:, None, :] * u[None, :, None] + (C[sel] - A[sel])[:, None, :] * v[None, :, None]
        P.append(pts.reshape(-1, 3))
        N.append(np.repeat(nrm[sel], len(bc), axis=0))
        W.append(np.repeat(area[sel] / (kk * kk), len(bc)))
        F.append(np.repeat(sel, len(bc)))
    return np.concatenate(P), np.concatenate(N), np.concatenate(W), np.concatenate(F)


def cross_pairs(P, Q, r):
    """All (i, j) with |P_i - Q_j| < r (sorted-grid spatial hash of Q).  Returns index arrays (i, j)."""
    e = np.zeros(0, np.int64)
    if len(P) == 0 or len(Q) == 0:
        return e, e
    lo = np.minimum(P.min(0), Q.min(0)) - r
    cq = np.floor((Q - lo) / r).astype(np.int64) + 1
    cp = np.floor((P - lo) / r).astype(np.int64) + 1
    dims = np.maximum(cq.max(0), cp.max(0)) + 3
    kq = (cq[:, 0] * dims[1] + cq[:, 1]) * dims[2] + cq[:, 2]
    kp = (cp[:, 0] * dims[1] + cp[:, 1]) * dims[2] + cp[:, 2]
    order = np.argsort(kq, kind='stable')
    sk = kq[order]
    I, J = [], []
    r2 = r * r
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
                ia = np.repeat(np.arange(len(P)), cnt)
                st = np.cumsum(cnt) - cnt
                ja = order[np.arange(tot) - np.repeat(st, cnt) + np.repeat(lo_, cnt)]
                d = Q[ja] - P[ia]
                m = np.einsum('ij,ij->i', d, d) < r2
                I.append(ia[m]); J.append(ja[m])
    if not I:
        return e, e
    return np.concatenate(I), np.concatenate(J)


# ----------------------------------------------------------------------------- surface base
class _Surface:
    """Sample points of a face set in the object's local frame, moved to world space by set_matrix (velocity from the matrix change)."""
    kind = 'SURFACE'

    def __init__(self, name, local_pts, local_nrm, areas, matrix=None):
        self.name = name
        self.lp = np.asarray(local_pts, float).reshape(-1, 3)
        self.ln = np.asarray(local_nrm, float).reshape(-1, 3)
        self.area = np.asarray(areas, float)
        self.M = np.eye(4) if matrix is None else np.asarray(matrix, float)
        self.p = self.lp @ self.M[:3, :3].T + self.M[:3, 3]
        self.n = self._rot(self.ln)
        self.vel = np.zeros_like(self.p)
        self.force = np.zeros(3)
        self.total = 0.0

    def _rot(self, nl):
        R = self.M[:3, :3]
        nn = nl @ np.linalg.inv(R).T       # normals transform with the inverse transpose
        return nn / np.maximum(np.linalg.norm(nn, axis=1), 1e-30)[:, None]

    def set_matrix(self, M, dt=None):
        """Move the surface to world matrix M; with dt, the surface velocity is the finite difference of the sample positions."""
        M = np.asarray(M, float)
        p1 = self.lp @ M[:3, :3].T + M[:3, 3]
        self.vel = (p1 - self.p) / dt if dt else np.zeros_like(p1)
        self.M, self.p = M, p1
        self.n = self._rot(self.ln)

    @property
    def total_area(self):
        return float(self.area.sum())


class ViscositySurface(_Surface):
    kind = 'VISCOSITY'

    def __init__(self, name, local_pts, local_nrm, areas, matrix=None, coefficient=1.0, noslip=1.0, reach=None, fluids=None):
        super().__init__(name, local_pts, local_nrm, areas, matrix)
        self.coef = float(coefficient)
        self.noslip = float(np.clip(noslip, 0.0, 1.0))
        self.reach = reach
        self.fluids = fluids               # None = every fluid, else a list of solver fluid indices
        self.body = None                   # index of the solver Body this surface is part of (two-way: its force / torque move that body)
        self.torque = np.zeros(3)          # about the body's centre of mass (or the origin without a body)
        self.pressure_force = np.zeros(3)  # normal force of the fluid's excess pressure on the faces

    def apply(self, S, dt):
        """Tangential viscous drag on the particles next to the faces; returns the force on the surface (N, the reaction)."""
        self.force = np.zeros(3)
        self.torque = np.zeros(3)
        self.pressure_force = np.zeros(3)
        if S.n == 0 or len(self.p) == 0 or self.coef <= 0:
            return self.force
        reach = self.reach or 1.5 * S.dx
        idx = np.arange(S.n) if self.fluids is None else np.flatnonzero(np.isin(S.fid, self.fluids))
        if len(idx) == 0:
            return self.force
        i, j = cross_pairs(S.x[idx], self.p, reach)
        if len(i) == 0:
            return self.force
        d = np.linalg.norm(S.x[idx][i] - self.p[j], axis=1)
        # a sample stands for its patch: weight by the patch area relative to a particle cross-section so that coarse faces still couple
        w = (1.0 - d / reach) ** 2 * np.minimum(1.0, self.area[j] / (S.dx * S.dx) + 0.25)
        nl = len(idx)
        ws = np.bincount(i, w, nl)
        vw = np.stack([np.bincount(i, w * self.vel[j][:, a], nl) for a in range(3)], axis=1)
        nn = np.stack([np.bincount(i, w * self.n[j][:, a], nl) for a in range(3)], axis=1)
        hit = ws > 1e-9
        vw[hit] /= ws[hit][:, None]
        nlen = np.linalg.norm(nn, axis=1)
        nn[nlen > 1e-9] /= nlen[nlen > 1e-9][:, None]
        rho = np.maximum(S.rho_particles()[idx], 1e-9)
        rate = self.coef / (rho * reach * reach) * self.noslip * np.minimum(ws, 1.0)
        frac = np.where(hit, 1.0 - np.exp(-rate * dt), 0.0)
        u = S.v[idx] - vw
        ut = u - np.einsum('ij,ij->i', u, nn)[:, None] * nn
        dv = -frac[:, None] * ut
        S.v[idx] += dv
        m = S.mass_ref[idx]
        fp = -(m[:, None] * dv) / max(dt, 1e-12)          # reaction of every particle's drag on the surface
        self.force = fp.sum(0)
        com = S.bodies[self.body].com_world(S.bodies[self.body].M1) if self.body is not None and self.body < len(S.bodies) else self.p.mean(0)
        self.torque = np.cross(S.x[idx] - com, fp).sum(0)
        # the fluid's excess (static) pressure pushes the faces along the normal, away from the fluid side
        pr = getattr(S, 'pressure', None)
        if pr is not None and len(pr) == S.n:
            pex = np.zeros(nl)
            pw = np.bincount(i, w * (pr[idx][i] - S.env.pressure_at(S.x[idx][i][:, 2])), nl)
            pex[hit] = pw[hit] / ws[hit]
            sd = np.einsum('ij,ij->i', S.x[idx][i] - self.p[j], self.n[j])
            sgn = np.zeros(nl)
            sgn[hit] = np.bincount(i, w * np.sign(sd), nl)[hit] / ws[hit]
            # force on the surface from one particle's share of the patch: pressure * patch area / (particles on the patch), here per surface sample
            cnt = np.bincount(j, minlength=len(self.p)).astype(float)
            fj = -(pex[i] * sgn[i] * self.area[j] / np.maximum(cnt[j], 1.0))[:, None] * self.n[j] * w[:, None] / np.maximum(ws[i], 1e-9)[:, None]
            self.pressure_force = fj.sum(0)
            self.force = self.force + self.pressure_force
            self.torque = self.torque + np.cross(self.p[j] - com, fj).sum(0)
        self.total += float(np.linalg.norm(self.force)) * dt
        return self.force


class EmissionSurface(_Surface):
    kind = 'EMISSION'

    def __init__(self, name, local_pts, local_nrm, areas, matrix=None, fluid=0, rate=200.0, speed=0.5, velocity=(0, 0, 0), spread=15.0,
                 temperature=None, max_particles=20000, start=0.0, end=0.0, seed=1, reverse=False):
        super().__init__(name, local_pts, local_nrm, areas, matrix)
        self.fluid = int(fluid)
        self.rate, self.speed = float(rate), float(speed)
        self.velocity = np.asarray(velocity, float)
        self.spread = math.radians(spread)
        self.temperature = temperature
        self.max_particles = int(max_particles)
        self.start, self.end = float(start), float(end)       # seconds; end <= 0 = never
        self.seed = int(seed)
        self.sign = -1.0 if reverse else 1.0
        self.acc = 0.0
        self.emitted = 0

    def emit(self, S, dt):
        t = S.t
        if t < self.start or (self.end > 0 and t >= self.end) or len(self.p) == 0 or self.rate <= 0:
            return 0
        f = self.fluid
        have = int(S.hi[f] - S.lo[f])
        self.acc += self.rate * dt
        k = int(self.acc)
        self.acc -= k
        k = min(k, max(self.max_particles - have, 0))
        if k <= 0:
            return 0
        rng = np.random.default_rng([self.seed, int(round(t * 1e4)) & 0x7fffffff, f])
        w = self.area / self.area.sum()
        s = rng.choice(len(self.p), size=k, p=w)
        n = self.n[s] * self.sign
        # in-plane jitter and a small push off the surface so the new particle does not sit inside the wall
        r = rng.standard_normal((k, 3))
        r -= np.einsum('ij,ij->i', r, n)[:, None] * n
        r /= np.maximum(np.linalg.norm(r, axis=1), 1e-9)[:, None]
        X = self.p[s] + r * rng.random((k, 1)) * 0.5 * S.dx + n * (0.6 * S.dx)
        # direction: normal tilted by a random angle inside the spread cone
        ang = self.spread * np.sqrt(rng.random(k))
        dirv = n * np.cos(ang)[:, None] + r * np.sin(ang)[:, None]
        V = dirv * self.speed + self.velocity + self.vel[s]
        # each particle is born at a random moment of the step: age it by u (free flight under gravity for a liquid), so the frame's particles
        # form a continuous stream instead of a bunch that moves as one
        u = rng.random((k, 1)) * dt
        g = np.asarray(S.env.g, float) if not S.gas[f] else np.zeros(3)
        X = X + V * u + 0.5 * g[None, :] * u * u
        V = V + g[None, :] * u
        T = np.full(k, S.defs[f]['temperature'] if self.temperature is None else self.temperature)
        grow(S, f, X, V, T)
        self.emitted += k
        return k


# ----------------------------------------------------------------------------- growing the solver
def per_particle_names(S):
    n = S.n
    out = []
    for name, a in vars(S).items():
        if name.startswith('_') or name in PERFLUID or not isinstance(a, np.ndarray) or a.ndim < 1 or a.shape[0] != n:
            continue
        out.append(name)
    return out


def grow(S, f, X, V, T):
    """Append particles of fluid f (positions X, velocities V, temperatures T) to the solver.  Every per-particle array grows; the
    fluid slices after f shift.  Returns the index range of the new particles."""
    X = np.asarray(X, float).reshape(-1, 3)
    k = len(X)
    n0 = S.n
    pos = int(S.hi[f])
    if k == 0:
        return pos, pos
    for name in per_particle_names(S):
        a = getattr(S, name)
        if name == 'x':
            new = X
        elif name == 'v':
            new = np.asarray(V, float).reshape(-1, 3)
        elif name == 'T':
            new = np.asarray(T, float).reshape(-1)
        elif name == 'fid':
            new = np.full(k, f, a.dtype)
        elif name == 'mass_ref':
            new = np.full(k, S.mass_p[f])
        elif name == 'wall_dist':
            new = np.full(k, 1e9)
        elif name == 'C':
            new = np.zeros((k, a.shape[1]))
            new[:, S.sp_of_fluid[f]] = 1.0
        elif name == 'cluster':
            new = (int(a.max()) + 1 if len(a) else 0) + np.arange(k)
        elif name == 'rho0_p':
            new = np.full(k, S.rho0[f])
        elif name == 'beta_p':
            new = np.full(k, S.beta[f])
        elif name == 'cp_p':
            new = np.full(k, S.cp[f])
        else:
            new = np.zeros((k,) + a.shape[1:], a.dtype)
        setattr(S, name, np.insert(a, pos, new.astype(a.dtype, copy=False), axis=0))
    S.hi[f] += k
    for g in range(f + 1, S.nf):
        S.lo[g] += k
        S.hi[g] += k
    S.n = n0 + k
    return pos, pos + k


# ----------------------------------------------------------------------------- installing on a solver
class SurfaceState:
    def __init__(self, viscosity, emission):
        self.viscosity, self.emission = list(viscosity), list(emission)
        self.active = True

    def counters(self):
        return dict(acc=[e.acc for e in self.emission], emitted=[e.emitted for e in self.emission],
                    total=[v.total for v in self.viscosity])

    def set_counters(self, d):
        for e, a, n in zip(self.emission, d['acc'], d['emitted']):
            e.acc, e.emitted = a, n
        for v, t in zip(self.viscosity, d['total']):
            v.total = t


def install(S, viscosity=(), emission=(), prepare=True):
    """Hook the sources into the solver instance S (idempotent: a second call replaces the sources).  `prepare` sets the per-particle
    mass / volume of the emission fluids (their parcels are one lattice cell of the fluid)."""
    if prepare:
        for e in emission:
            f = e.fluid
            S.Vp[f] = S.dx ** 3
            S.mass_p[f] = S.rho0[f] * S.Vp[f]
    st = getattr(S, 'surf', None)
    if st is not None:
        st.viscosity, st.emission = list(viscosity), list(emission)
        return st
    st = SurfaceState(viscosity, emission)
    S.surf = st
    step0, snap0, rest0 = S.step, S.snapshot, S.restore

    def step(dt, sources=None):
        if st.active:
            for e in st.emission:
                e.emit(S, dt)
        step0(dt, sources=sources)
        if st.active:
            for v in st.viscosity:
                if v.body is not None and v.body < len(S.bodies):
                    b = S.bodies[v.body]
                    v.set_matrix(b.M1, dt)                  # the surface rides on its body
                v.apply(S, dt)
                if v.body is not None and v.body < len(S.bodies) and S.bodies[v.body].dynamic:
                    b = S.bodies[v.body]
                    b.fx, b.tx = v.force.copy(), v.torque.copy()     # drives the body from the next step on (one-step lag, explicit)

    def snapshot():
        s = snap0()
        names = [nm for nm in per_particle_names(S) if nm not in s]
        s['_srf'] = dict(n=S.n, lo=S.lo.copy(), hi=S.hi.copy(), arrays={nm: getattr(S, nm).copy() for nm in names}, counters=st.counters())
        return s

    def restore(s):
        g = s.get('_srf')
        if g is not None:
            S.n, S.lo, S.hi = g['n'], g['lo'].copy(), g['hi'].copy()
            for nm, a in g['arrays'].items():
                setattr(S, nm, a.copy())
            st.set_counters(g['counters'])
        rest0(s)

    S.step, S.snapshot, S.restore = step, snapshot, restore
    return st
