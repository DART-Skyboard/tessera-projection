"""Fluid mode solver (NumPy, no bpy): volumetric, particle-based (Lagrangian) fluid dynamics for general fluids.

Tessera Projection / LOTBCBP.  The particles are the fluid's representation: a liquid is a set of equal-volume
particles that carry mass, velocity and temperature; a free gas is a set of parcels that carry the same fields and
change volume with ambient pressure and their own temperature.  Rendering as points, assigned meshes or a ray-marched
volume happens in the Blender layer.

What is solved (per sub-step)
-----------------------------
* Momentum: gravity, buoyancy (Archimedes against the ambient atmosphere, Boussinesq density rho(T) = rho0 (1 - beta
  (T - Tref)) for liquids, ideal gas rho = P M / (R T) with the barometric ambient pressure P(z) for gases), drag toward
  the wind (free fluids only), pressure and viscosity.
* Pressure: position-based fluids (PBF, Macklin and Mueller 2013) on the number density of the liquid particles.  The
  constraint projection is weighted by 1/rho_i (a mass-weighted PBD projection), so the pressure force is the internal
  force between particles and heavier fluid really is displaced less than lighter fluid: a light blob in a heavy
  carrier feels the carrier's pressure gradient and rises, a heavy blob sinks (multi-phase, density-contrast correct to
  first order).  Pressure is one-sided (no tension), which gives a free surface.
* Viscosity: XSPH-style velocity smoothing toward the neighbours of the same (or a miscible) fluid, with a rate set by
  the kinematic viscosity nu = mu / rho (clamped for stability, so very viscous fluids saturate).
* Heat: sources (lamps) deposit power on nearby particles; pairwise conductive exchange with conductance k_ij dx
  (1 - r/h) (energy conserving, stability-limited); relaxation to the ambient temperature (and the ground temperature)
  scaled by the particle's exposure (free surface, wall proximity, free gas).  A thermal speed-up factor scales all heat
  transport so slow real processes (a lava lamp warms over an hour) fit an animation.
* Surface cohesion: Akinci-style attractive/repulsive pair force between particles of the same fluid, scaled by a
  surface tension coefficient; immiscible fluids get no cross cohesion and no velocity mixing, miscible ones do.
* Collision: every body is a LOTBCBP collision body (the SDF projection of core.sdf.PartField with on-demand bricks):
  containers (fluid stays inside), colliders (fluid stays outside), the ground plane.  Rigid body motion is an
  update of the body's matrix; the wall's surface velocity v_lin + omega x r at the contact point is applied at contact
  (with a friction coefficient), so moving walls and rotating gears drive the fluid.

What is approximated (honest limits)
------------------------------------
Pressure is a position-based projection with a few iterations, not a converged pressure solve (a few percent density
error in deep columns, and it is not time accurate for strong transients); the kernel resolution is the particle spacing
(no sub-particle structure, no turbulence model, no thin films); a gas is a parcel set that does not push on liquid
particles (gas bubbles in liquid are not modelled); viscosity and cohesion are calibrated, not exact; mixing of miscible
fluids is by the velocity smoothing only (no species diffusion); fluid does not move solid objects (solid -> fluid is
exact through the collision projection, fluid -> solid is one-way: no force is applied to rigid bodies).
"""
import math

import numpy as np

R_GAS = 8.314462618
P_ATM = 101325.0

# ----------------------------------------------------------------------------- properties and units


def derive_properties(mass=0.0, volume=0.0, density=0.0, weight=0.0, g=9.80665):
    """Fill in whatever is missing from what is given (0 = not given).  Precedence: volume, then mass (or weight / g),
    then density.  Returns dict(mass, volume, density, weight)."""
    g = abs(g) or 9.80665
    m, v, d, w = float(mass), float(volume), float(density), float(weight)
    if m <= 0 and w > 0:
        m = w / g
    if m > 0 and v > 0:
        d = m / v
    elif m > 0 and d > 0:
        v = m / d
    elif v > 0 and d > 0:
        m = v * d
    return dict(mass=m, volume=v, density=d, weight=m * g if m > 0 else 0.0)


# per-fluid presets: density at 293.15 K and 1 atm
PRESETS = {
    'WATER': dict(label="Water", kind='LIQUID', density=997.0, visc=1.0e-3, cp=4181.0, k=0.60, beta=2.1e-4, cohesion=0.072, color=(0.25, 0.55, 0.95), miscible=True),
    'VEG_OIL': dict(label="Vegetable oil", kind='LIQUID', density=920.0, visc=0.05, cp=2000.0, k=0.17, beta=7.0e-4, cohesion=0.032, color=(0.95, 0.80, 0.25), miscible=False),
    'MINERAL_OIL': dict(label="Mineral oil", kind='LIQUID', density=850.0, visc=0.03, cp=1900.0, k=0.14, beta=7.0e-4, cohesion=0.030, color=(0.90, 0.85, 0.55), miscible=False),
    'LAVA_WAX': dict(label="Lava lamp wax", kind='LIQUID', density=1010.0, visc=0.05, cp=2200.0, k=0.25, beta=1.5e-3, cohesion=0.040, color=(1.0, 0.25, 0.10), miscible=False),
    'HONEY': dict(label="Honey", kind='LIQUID', density=1420.0, visc=10.0, cp=2400.0, k=0.50, beta=5.0e-4, cohesion=0.060, color=(0.90, 0.55, 0.10), miscible=True),
    'AIR': dict(label="Air", kind='GAS', density=1.204, visc=1.8e-5, cp=1005.0, k=0.0257, beta=0.0, cohesion=0.0, color=(0.85, 0.90, 1.0), miscible=True),
    'HELIUM': dict(label="Helium", kind='GAS', density=0.1664, visc=1.96e-5, cp=5193.0, k=0.152, beta=0.0, cohesion=0.0, color=(0.95, 0.75, 0.95), miscible=True),
    'CO2': dict(label="Carbon dioxide", kind='GAS', density=1.842, visc=1.47e-5, cp=844.0, k=0.0166, beta=0.0, cohesion=0.0, color=(0.55, 0.55, 0.60), miscible=True),
    'CUSTOM': dict(label="Custom", kind='LIQUID', density=1000.0, visc=1.0e-3, cp=4000.0, k=0.5, beta=3.0e-4, cohesion=0.05, color=(0.5, 0.8, 0.9), miscible=True),
}
PRESET_ITEMS = [(k, v['label'], '') for k, v in PRESETS.items()]


def fluid_def(preset='WATER', **over):
    """A complete fluid definition dict from a preset plus overrides (see FluidSolver.build for the keys)."""
    p = PRESETS[preset]
    d = dict(name=p['label'], kind=p['kind'], density=p['density'], mass=0.0, volume=0.0, weight=0.0, temperature=293.15,
             velocity=(0.0, 0.0, 0.0), visc=p['visc'], cp=p['cp'], k=p['k'], beta=p['beta'], cohesion=p['cohesion'],
             miscible=p['miscible'], color=p['color'], cool=0.0, drag=None, diffusion=None, count=2000, fill=1.0,
             shape=None, body=-1, tref=None, group=None)
    d.update(over)
    return d


# ----------------------------------------------------------------------------- environment


class FEnv:
    """Ambient atmosphere: the 0.2.0 Environment card data (pressure, temperature, gravity, wind, gas mix -> molar mass).

    pressure is the pressure at height z_ref; pressure_at(z) follows the isothermal barometric formula
    P(z) = P0 exp(-M g (z - z_ref) / (R T)); density_at(z) = P(z) M / (R T)."""

    def __init__(self, gravity=(0.0, 0.0, -9.80665), pressure=P_ATM, temperature=288.15, molar=0.0289647, wind=(0.0, 0.0, 0.0),
                 gust=0.0, z_ref=0.0, ground_z=None, ground_friction=0.4, ground_temp=None):
        self.g = np.asarray(gravity, float)
        self.pressure, self.temperature, self.molar = float(pressure), float(temperature), float(molar)
        self.wind = np.asarray(wind, float)
        self.gust = float(gust)
        self.z_ref = float(z_ref)
        self.ground_z = ground_z
        self.ground_friction = float(ground_friction)
        self.ground_temp = ground_temp

    def scale_height(self):
        return R_GAS * self.temperature / (self.molar * max(np.linalg.norm(self.g), 1e-9))

    def pressure_at(self, z):
        return self.pressure * np.exp(-(np.asarray(z, float) - self.z_ref) / self.scale_height())

    def density_at(self, z):
        return self.pressure_at(z) * self.molar / (R_GAS * self.temperature)

    @staticmethod
    def from_atmosphere(preset_state, **kw):
        """preset_state: core.atmosphere.preset_state(key) / state(...) dict plus pressure, temperature, gravity."""
        s = preset_state
        return FEnv(gravity=(0, 0, -s['gravity']), pressure=s['pressure'], temperature=s['temperature'], molar=s['molar'], **kw)


# ----------------------------------------------------------------------------- neighbour search and kernels


def neighbor_pairs(pos, h):
    """All unordered pairs (i, j) with |x_i - x_j| < h by a sorted uniform grid (spatial hash).  O(n) for even density."""
    n = len(pos)
    e = np.zeros(0, np.int64)
    if n < 2:
        return e, e
    lo = pos.min(0)
    c = np.floor((pos - lo) / h).astype(np.int64) + 1
    dims = c.max(0) + 2
    key = (c[:, 0] * dims[1] + c[:, 1]) * dims[2] + c[:, 2]
    order = np.argsort(key, kind='stable')
    sk, sp = key[order], pos[order]
    idx = np.arange(n)
    offs = [(0, 0, 0)] + [(a, b, cc) for a in (-1, 0, 1) for b in (-1, 0, 1) for cc in (-1, 0, 1) if (a, b, cc) > (0, 0, 0)]
    I, J = [], []
    h2 = h * h
    for (a, b, cc) in offs:
        off = (a * dims[1] + b) * dims[2] + cc
        nk = sk + off
        hi_ = np.searchsorted(sk, nk, 'right')
        lo_ = idx + 1 if off == 0 else np.searchsorted(sk, nk, 'left')
        cnt = np.maximum(hi_ - lo_, 0)
        tot = int(cnt.sum())
        if tot == 0:
            continue
        ia = np.repeat(idx, cnt)
        st = np.cumsum(cnt) - cnt
        ja = np.arange(tot) - np.repeat(st, cnt) + np.repeat(lo_, cnt)
        d = sp[ja] - sp[ia]
        m = np.einsum('ij,ij->i', d, d) < h2
        I.append(order[ia[m]]); J.append(order[ja[m]])
    if not I:
        return e, e
    return np.concatenate(I), np.concatenate(J)


def scatter_pair(n, I, J, vec):
    """out[i] += vec for I, out[j] -= vec for J (bincount is much faster than np.add.at)."""
    out = np.empty((n, 3))
    for ax in range(3):
        out[:, ax] = np.bincount(I, vec[:, ax], n) - np.bincount(J, vec[:, ax], n)
    return out


def kernel_rest_density(dx, h):
    """sum of W over a cubic lattice of spacing dx (including the particle itself): the rest number density."""
    r = int(math.ceil(h / dx)) + 1
    g = np.arange(-r, r + 1) * dx
    X, Y, Z = np.meshgrid(g, g, g, indexing='ij')
    d = np.sqrt(X * X + Y * Y + Z * Z).ravel()
    q = d / h
    return float(np.sum(np.where(q < 1.0, (1.0 - q) ** 3, 0.0)))


# ----------------------------------------------------------------------------- collision bodies (LOTBCBP projection)


class Body:
    """A collision body: a PartField (the LOTBCBP SDF with on-demand bricks) plus a rigid transform.

    kind 'CONTAINER': fluid is legal inside the cavity.  'COLLIDER': fluid is legal outside the solid.
    fluids: fluid indices that see this body (None = all).  set_matrix() moves it; the solver interpolates the matrix over
    the frame and applies the wall surface velocity at contact."""

    def __init__(self, field, matrix=None, kind='CONTAINER', fluids=None, friction=0.2, name="", lotbcbp=True):
        self.field, self.kind, self.fluids, self.friction, self.name = field, kind, fluids, float(friction), name
        self.lotbcbp = lotbcbp
        # on-demand bricks only pay off (and are only accurate) for meshes with small triangles: a body made of a few
        # large flat faces (a box) is already exact in the coarse field, so refinement is skipped for it
        s_ = field.soup
        edge = float(max(np.linalg.norm(s_.B - s_.A, axis=1).max(), np.linalg.norm(s_.C - s_.B, axis=1).max())) if len(s_) else 0.0
        self.bricks_ok = edge < 6.0 * field.h
        M = np.eye(4) if matrix is None else np.asarray(matrix, float)
        self.M0 = self.M1 = M.copy()
        self.n_refined_calls = 0
        self.open = False          # an open vessel spills when overfilled, a sealed one is reported as over-pressured
        # two-way coupling: fluid force / torque are measured on every body with two_way on; a body with mass > 0 also moves under them
        self.two_way = True
        self.mass = 0.0            # kg; 0 = immovable (animated or fixed), the fluid still measures force on it
        self.constraint = 'FREE'   # FREE (6 DOF) | SLIDER (translate along `axis`) | HINGE (rotate about `axis` through `pivot`)
        self.axis = np.array([0.0, 0.0, 1.0])
        self.pivot = None
        self.limits = None         # (lo, hi): metres along the slider axis / radians about the hinge, measured from the start pose
        self.stick = 0.0           # static friction force / torque the fluid must exceed before the body moves (N or N m)
        self.kfric = 0.0           # viscous friction (N s/m or N m s/rad)
        self.ext = np.zeros(3)     # external force (N), e.g. an engine's thrust
        self.vel = np.zeros(3)
        self.omega = np.zeros(3)
        self.travel = 0.0          # displacement along the slider axis / angle about the hinge since the start pose
        self.F = np.zeros(3)       # force of the fluid on the body, last step (N)
        self.T = np.zeros(3)       # torque about the centre of mass (N m)
        self._F = np.zeros(3)
        self._T = np.zeros(3)
        self.smooth = 0.15         # low-pass weight of the newest fluid force sample (explicit coupling of a light body is otherwise unstable)
        self._Fs = np.zeros(3)
        self._Ts = np.zeros(3)
        self.rel = (0.0, 1.0)      # substep window of a dynamic body (its pose is predicted per substep)
        self.M_start = M.copy()
        lo_, hi_ = field.bmin, field.bmax
        self.com_local = 0.5 * (np.asarray(lo_, float) + np.asarray(hi_, float))

    @property
    def dynamic(self):
        return bool(self.two_way and self.mass > 0.0)

    def com_world(self, M=None):
        M = self.M1 if M is None else M
        return M[:3, :3] @ self.com_local + M[:3, 3]

    def inertia(self):
        lo, hi = self.bbox_world()
        L = float(np.max(hi - lo))
        return max(self.mass * L * L / 6.0, 1e-9)

    def reset_motion(self):
        self.vel[:] = 0.0
        self.omega[:] = 0.0
        self.travel = 0.0
        self.F[:] = 0.0
        self.T[:] = 0.0
        self._F[:] = 0.0
        self._T[:] = 0.0
        self._Fs[:] = 0.0
        self._Ts[:] = 0.0

    def point_velocity(self, P, dt):
        """Velocity of body-fixed world points P over the current M0 -> M1 window."""
        Pl, _R, _sc = self.local(P, self.M1)
        Rn1, t1, s1 = self._split(self.M1)
        Rn0, t0, s0 = self._split(self.M0)
        return (((Pl * s1) @ Rn1.T + t1) - ((Pl * s0) @ Rn0.T + t0)) / dt

    def pose_after(self, dt, vel=None, omega=None):
        """Matrix after moving with (vel, omega) for dt from M1 (translation of the centre of mass, rotation about it or about the hinge)."""
        vel = self.vel if vel is None else vel
        omega = self.omega if omega is None else omega
        M = self.M1.copy()
        ang = float(np.linalg.norm(omega)) * dt
        c = self.com_world() if (self.constraint != 'HINGE' or self.pivot is None) else self.pivot_world()
        if ang > 1e-12:
            k = omega / np.linalg.norm(omega)
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            R = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)
            M[:3, :3] = R @ M[:3, :3]
            M[:3, 3] = R @ (M[:3, 3] - c) + c
        M[:3, 3] += vel * dt
        return M

    def pivot_world(self):
        return self.M_start[:3, :3] @ np.asarray(self.pivot, float) + self.M_start[:3, 3] if self.pivot is not None else self.com_world()

    def integrate(self, dt, g):
        """Advance the body one substep under the measured fluid force / torque, gravity, external force, friction and its constraint."""
        if not self.dynamic:
            self.M0 = self.M1
            return
        m = self.mass
        ax = np.asarray(self.axis, float)
        ax = ax / max(np.linalg.norm(ax), 1e-12)
        self._Fs = (1 - self.smooth) * self._Fs + self.smooth * self._F
        self._Ts = (1 - self.smooth) * self._Ts + self.smooth * self._T
        F = self._Fs + m * np.asarray(g, float) + self.ext
        T = self._Ts.copy()
        if self.constraint == 'SLIDER':
            f = float(F @ ax)
            v = float(self.vel @ ax)
            f = self._frictionised(f, v, self.stick, self.kfric)
            v += f / m * dt
            tr = self.travel + v * dt
            if self.limits is not None:
                lo, hi = self.limits
                if tr > hi:
                    tr, v = hi, min(v, 0.0)
                if tr < lo:
                    tr, v = lo, max(v, 0.0)
            self.vel = ax * v
            self.omega[:] = 0.0
            dv = ax * (tr - self.travel)
            self.travel = tr
            self.M0 = self.M1
            M = self.M1.copy()
            M[:3, 3] += dv
            self.M1 = M
            return
        if self.constraint == 'HINGE':
            piv = self.pivot_world()
            r = self.com_world() - piv
            Tp = T + np.cross(self.com_world() - piv, F)
            t = float(Tp @ ax)
            w = float(self.omega @ ax)
            Ih = self.inertia() + m * float(r @ r)
            t = self._frictionised(t, w, self.stick, self.kfric)
            w += t / Ih * dt
            ang = self.travel + w * dt
            if self.limits is not None:
                lo, hi = self.limits
                if ang > hi:
                    ang, w = hi, min(w, 0.0)
                if ang < lo:
                    ang, w = lo, max(w, 0.0)
            da = ang - self.travel
            self.travel = ang
            self.omega = ax * w
            self.vel[:] = 0.0
            self.M0 = self.M1
            self.M1 = self.pose_after(1.0, vel=np.zeros(3), omega=ax * da)
            return
        # free body
        fm = float(np.linalg.norm(F))
        if self.stick > 0 and fm <= self.stick and float(np.linalg.norm(self.vel)) < 1e-6:
            F = np.zeros(3)
        elif self.stick > 0:
            F = F - self.stick * F / max(fm, 1e-12)
        F = F - self.kfric * self.vel
        self.vel = self.vel + F / m * dt
        self.omega = (self.omega + T / self.inertia() * dt) * math.exp(-(self.kfric / max(m, 1e-9) + 0.5) * dt)
        wn = float(np.linalg.norm(self.omega))
        if wn > 30.0:
            self.omega *= 30.0 / wn
        self.M0 = self.M1
        self.M1 = self.pose_after(dt)

    @staticmethod
    def _frictionised(f, v, stick, kfric):
        if abs(v) < 1e-9 and abs(f) <= stick:
            return 0.0
        f = f - math.copysign(stick, v if abs(v) > 1e-9 else f)
        return f - kfric * v

    def set_matrix(self, M):
        """New transform for the coming frame (the previous one becomes the start of the interpolation)."""
        self.M0 = self.M1
        self.M1 = np.asarray(M, float).copy()

    def settle(self):
        self.M0 = self.M1

    def moving(self):
        return not np.allclose(self.M0, self.M1, atol=1e-12)

    def matrix_at(self, s):
        if self.dynamic:                       # a dynamic body advances inside each substep: s is the solver's substep fraction
            a, b = self.rel
            s = (s - a) / max(b - a, 1e-12)
        return self.M0 + (self.M1 - self.M0) * s

    @staticmethod
    def _split(M):
        R = M[:3, :3]
        sc = float(np.cbrt(abs(np.linalg.det(R)))) or 1.0
        return R / sc, M[:3, 3], sc

    def local(self, P, M):
        Rn, t, sc = self._split(M)
        return ((P - t) @ Rn) / sc, Rn, sc

    def _dist_local(self, Pl):
        if self.kind == 'CONTAINER':
            return -self.field.fields(Pl)[0]
        return self.field.sample_sd(Pl)

    def distance(self, P, M):
        """Legal-side distance in world units (positive where fluid may be)."""
        Pl, _R, sc = self.local(P, M)
        return self._dist_local(Pl) * sc

    def distance_normal(self, P, M, refine_band=None):
        """(distance, unit normal pointing toward the legal side) with a forward-difference gradient of the field."""
        Pl, Rn, sc = self.local(P, M)
        if self.lotbcbp and self.bricks_ok and self.field.level > 0 and refine_band is not None:
            near = np.flatnonzero(np.abs(self._dist_local(Pl)) < refine_band * self.field.h)
            if len(near):
                self.field.request_refine(Pl[near], max_new=32)
                self.n_refined_calls += 1
        d0 = self._dist_local(Pl)
        e = 0.35 * self.field.h
        gr = np.empty_like(Pl)
        for a in range(3):
            Q = Pl.copy(); Q[:, a] += e
            gr[:, a] = (self._dist_local(Q) - d0) / e
        gn = gr / np.maximum(np.linalg.norm(gr, axis=1, keepdims=True), 1e-9)
        return d0 * sc, gn @ Rn.T, Rn, sc

    def bbox_world(self):
        lo, hi = self.field.bmin, self.field.bmax
        c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        w = c @ self.M1[:3, :3].T + self.M1[:3, 3]
        return w.min(0), w.max(0)


# ----------------------------------------------------------------------------- geometry sampling


def mesh_volume(V, T):
    P = np.asarray(V, float)[np.asarray(T)]
    return abs(float(np.einsum('ij,ij->i', P[:, 0], np.cross(P[:, 1], P[:, 2])).sum()) / 6.0)


def is_closed(T):
    T = np.asarray(T)
    e = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
    _u, c = np.unique(e, axis=0, return_counts=True)
    return bool(len(c)) and bool(np.all(c == 2))


def lattice_in_box(lo, hi, dx):
    """Points (i + 1/2) dx anchored at the origin, so every fluid shares one lattice (exact exclusion between fluids)."""
    i0 = np.floor(np.asarray(lo) / dx - 0.5).astype(int)
    i1 = np.ceil(np.asarray(hi) / dx - 0.5).astype(int)
    ax = [np.arange(i0[a], i1[a] + 1) for a in range(3)]
    I = np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3)
    return I


def lat_pos(I, dx):
    return (I + 0.5) * dx


def lat_key(I):
    I = np.asarray(I, np.int64)
    return ((I[:, 0] + 1048576) * 2097152 + (I[:, 1] + 1048576)) * 2097152 + (I[:, 2] + 1048576)


def interior_lattice(verts, tris, dx, inset=0.0):
    """Lattice indices whose centre is inside the closed mesh (signed-distance sign test, the projection's pseudo-normal
    test), at least `inset` inside."""
    from . import sdf as SDF
    soup = SDF.soup_from_mesh(verts, tris)
    lo, hi = soup.bounds()
    I = lattice_in_box(lo, hi, dx)
    d = SDF.signed_distance(soup, lat_pos(I, dx))
    return I[d < -inset]


def surface_points(verts, tris, dx, rng):
    """Area-weighted random points on an open / flat mesh (a sheet of fluid lying on a surface): one layer."""
    P = np.asarray(verts, float)[np.asarray(tris)]
    cr = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    area = 0.5 * np.linalg.norm(cr, axis=1)
    n = max(int(area.sum() / (dx * dx)), 1)
    tid = rng.choice(len(P), size=n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    f = u + v > 1
    u[f], v[f] = 1 - u[f], 1 - v[f]
    q = P[tid]
    return q[:, 0] + (q[:, 1] - q[:, 0]) * u[:, None] + (q[:, 2] - q[:, 0]) * v[:, None]


# ----------------------------------------------------------------------------- solver


class FluidSolver:
    """Particle fluid system.  Build with FluidSolver.build(fluid_defs, bodies, env, ...).

    Arrays (n particles, grouped by fluid): x, v, T, fid.  Per fluid f: lo[f]:hi[f] slice, rho0, mass_p, Vp, ..."""

    def __init__(self):
        self.t = 0.0
        self.escapes = 0
        self.fill_report = []

    # ------------------------------------------------------------------ construction
    @staticmethod
    def build(defs, bodies=(), env=None, dx=None, max_particles=20000, seed=1, thermal_speedup=1.0, iterations=4,
              max_substeps=4, vmax=20.0, lotbcbp=True, drift_iterations=None, fill_seed='LEVEL', species=None, rules=None):
        """defs: list of fluid_def dicts.  Recognised keys: kind, density, mass, volume, weight, temperature, velocity,
        visc, cp, k, beta, cohesion, miscible, cool (1/s), drag (1/s), diffusion (m2/s), count (gas parcels), fill (fraction
        of a container's cavity when no shape is given), shape (None | dict(kind='MESH', verts, tris) | dict(kind='POINTS', pts)),
        body (index into bodies, -1 = no container), tref (Boussinesq reference temperature)."""
        S = FluidSolver()
        S.fill_seed = fill_seed
        S.env = env or FEnv()
        S.bodies = list(bodies)
        S.rng = np.random.default_rng(seed)
        S.relax, S.warm, S.flip_ratio, S.cg_iters = 1.0, 0.0, 0.95, 120
        S._solid_cache, S._p_prev, S._p_origin, S.cg_last = None, None, None, 0
        S.speedup, S.iters, S.max_sub, S.vmax, S.lotbcbp = float(thermal_speedup), int(iterations), int(max_substeps), float(vmax), lotbcbp
        for b in S.bodies:
            b.lotbcbp = lotbcbp
        g = float(np.linalg.norm(S.env.g)) or 9.80665
        nf = len(defs)
        D = [dict(d) for d in defs]
        # ---- unit derivation (per fluid) and particle spacing estimate
        for d in D:
            d['gas'] = d['kind'] == 'GAS'
            if d.get('tref') is None:
                d['tref'] = d['temperature']
            d.setdefault('group', None)
        body_vol = {}
        for bi, b in enumerate(S.bodies):
            if b.kind == 'CONTAINER':
                lo, hi = b.bbox_world()
                body_vol[bi] = float(np.prod(hi - lo)) * 0.5
        est = 0.0
        for d in D:
            sh = d['shape']
            vol = d['volume']
            if vol <= 0 and sh is not None and sh['kind'] == 'MESH' and is_closed(sh['tris']):
                vol = mesh_volume(sh['verts'], sh['tris'])
            if vol <= 0 and d['mass'] > 0 and d['density'] > 0:
                vol = d['mass'] / d['density']
            if vol <= 0 and sh is None and d['body'] >= 0:
                vol = body_vol.get(d['body'], 0.0) * d['fill']
            d['_vol_est'] = vol
            if not d['gas']:
                est += vol
        auto_dx = dx is None
        if dx is None:
            dx = (est / max(max_particles, 1)) ** (1.0 / 3.0) if est > 0 else 0.01
            dx = max(dx, 1e-4)
            n_pts = sum(len(d['shape']['pts']) for d in D if d['shape'] is not None and d['shape']['kind'] == 'POINTS' and not d['gas'])
            if n_pts and est <= 0:
                dx = 0.01
        S.dx = float(dx)
        S.h = 1.8 * S.dx
        S.hg = 1.6 * S.dx
        S.sig0 = kernel_rest_density(S.dx, S.h)
        # ---- containers' interior lattices (shared among their fluids)
        ground = S.env.ground_z
        occupied = set()
        cont_cache = {}

        gv = np.asarray(S.env.g, float)
        ghat = gv / (np.linalg.norm(gv) or 1.0)

        def _level(I, raw=False):
            """Fill order: the level is measured along gravity (lowest point first), not along world Z."""
            return (I if raw else lat_pos(I, S.dx)) @ ghat * -1.0

        def interior_of(bi):
            if bi not in cont_cache:
                b = S.bodies[bi]
                lo, hi = b.bbox_world()
                I = lattice_in_box(lo, hi, S.dx)
                P = lat_pos(I, S.dx)
                ok = b.distance(P, b.M1) >= 0.45 * S.dx
                cont_cache[bi] = I[ok]
            return cont_cache[bi]

        placed = [None] * nf
        # 1) shaped liquids (and shaped gases)
        for f, d in enumerate(D):
            sh = d['shape']
            if sh is None:
                continue
            if sh['kind'] == 'POINTS':
                P = np.asarray(sh['pts'], float)
                if d['gas']:
                    placed[f] = P
                else:
                    placed[f] = P
                    for k in lat_key(np.floor(P / S.dx).astype(int)):
                        occupied.add(int(k))
                continue
            V, T = np.asarray(sh['verts'], float), np.asarray(sh['tris'])
            if is_closed(T):
                I = interior_lattice(V, T, S.dx)
                if d['body'] >= 0 and not d['gas']:
                    pass
                vol = d['volume']
                if d['volume'] <= 0 and d['mass'] > 0 and d['density'] > 0:
                    vol = d['mass'] / d['density']
                if vol > 0 and len(I) * S.dx ** 3 > vol * 1.02 and not d['gas']:       # less than the shape: fill from the bottom
                    nkeep = max(int(round(vol / S.dx ** 3)), 1)
                    I = I[np.argsort(_level(I), kind='stable')[:nkeep]]
                if not d['gas']:
                    keys = lat_key(I)
                    I = I[[int(k) not in occupied for k in keys]]
                    for k in lat_key(I):
                        occupied.add(int(k))
                placed[f] = lat_pos(I, S.dx)
            else:
                placed[f] = surface_points(V, T, S.dx, S.rng)
                if not d['gas']:
                    for k in lat_key(np.floor(placed[f] / S.dx).astype(int)):
                        occupied.add(int(k))
        # 2) container-filled liquids: densest first, from the bottom up
        by_body = {}
        for f, d in enumerate(D):
            if d['shape'] is None and d['body'] >= 0 and not d['gas']:
                by_body.setdefault(d['body'], []).append(f)
        S.fill_report = []
        for bi, fl in by_body.items():
            I_all = interior_of(bi)
            n_all = len(I_all)
            free = np.array([int(k) not in occupied for k in lat_key(I_all)]) if n_all else np.zeros(0, bool)
            I = I_all[free]
            I = I[np.argsort(_level(I), kind='stable')]
            total = len(I)
            # capacity of the real cavity (collision-body projection sampled at dx/2).  Particles are cell-sized (mass = rho dx^3) and the first
            # lattice node sits ~0.45 dx off the wall, so the lattice itself holds ~12% less than the cavity: the seeding is lowest-point-first
            # along gravity with n = volume / dx^3 (mass exact), and the extra particles of a near-brim fill go to the highest free spots;
            # the dynamic fill (settle) then closes the wall gaps and brings the free surface to the exact level
            b = S.bodies[bi]
            lo_b, hi_b = b.bbox_world()
            Pf = lat_pos(lattice_in_box(lo_b, hi_b, S.dx / 2.0), S.dx / 2.0)
            dfine = b.distance(Pf, b.M1)
            v_cav = float(np.count_nonzero(dfine >= 0.0)) * (S.dx / 2.0) ** 3
            v_free = v_cav * (total / max(n_all, 1)) if n_all else 0.0
            v_free = max(v_free, v_cav - len(occupied) * 0.0)   # capacity of the cavity (shaped fluids inside it are not subtracted)
            pos = 0
            rep = dict(body=bi, capacity=v_cav, free_capacity=0.98 * v_cav, requested=0.0, overflow=0.0, fluids=[], open=bool(getattr(b, 'open', False)))
            n_cap = int(round(0.98 * v_cav / S.dx ** 3))
            n_used = 0
            if S.fill_seed == 'LEVEL':
                # analytic level-set fill: the cavity (fine samples inside the collision body) is ordered along gravity; each fluid, densest first,
                # takes the next slab of the requested volume, so the free surface is at the exact level and several fluids are stratified by
                # density.  Lattice nodes inside a slab are seeded; the few extra particles that mass conservation needs (the lattice leaves a
                # half-cell wall gap) go to random fine points of the same slab.
                inside = dfine >= 0.0
                Pin = Pf[inside]
                dsel = dfine[inside]
                if occupied:      # space taken by shaped fluids is not available to this fill
                    freef = np.array([int(k) not in occupied for k in lat_key(np.floor(Pin / S.dx).astype(int))])
                    Pin, dsel = Pin[freef], dsel[freef]
                Lin = _level(Pin, raw=True)
                order = np.argsort(Lin, kind='stable')
                Pin, dsel, Lin = Pin[order], dsel[order], Lin[order]
                Vf = (S.dx / 2.0) ** 3
                vfree = len(Pin) * Vf
                Lnode = _level(I)
                cum = 0.0
                taken = np.zeros(len(I), bool)
                for f in sorted(fl, key=lambda q: -D[q]['density']):
                    d = D[f]
                    vol = d['volume']
                    if vol <= 0 and d['mass'] > 0 and d['density'] > 0:
                        vol = d['mass'] / d['density']
                    if vol <= 0:
                        vol = d['fill'] * vfree
                    rep['requested'] += vol
                    vol_c = min(vol, max(0.98 * vfree - cum, 0.0))
                    k0, k1 = int(round(cum / Vf)), int(round((cum + vol_c) / Vf))
                    k1 = min(k1, len(Lin))
                    if k1 <= k0:
                        placed[f] = np.zeros((0, 3)); rep['fluids'].append((f, vol, 0)); continue
                    La = Lin[k0]; Lb = Lin[k1 - 1]
                    sel = (Lnode >= La) & (Lnode <= Lb) & ~taken
                    taken |= sel
                    P = lat_pos(I[sel], S.dx)
                    n = int(round(vol_c / S.dx ** 3))
                    if n > len(P):
                        cand = np.flatnonzero(dsel[k0:k1] >= 0.5 * S.dx) + k0
                        if len(cand):
                            extra = Pin[S.rng.choice(cand, size=min(n - len(P), len(cand)), replace=len(cand) < n - len(P))]
                            P = np.vstack([P, extra])
                    elif n < len(P):
                        P = P[np.argsort(_level(P, raw=True), kind='stable')[:n]]
                    placed[f] = P
                    rep['fluids'].append((f, vol, len(P)))
                    cum += vol_c
                rep['overflow'] = max(rep['requested'] - 0.98 * vfree, 0.0)
                S.fill_report.append(rep)
                continue
            for f in sorted(fl, key=lambda q: -D[q]['density']):
                d = D[f]
                vol = d['volume']
                if vol <= 0 and d['mass'] > 0 and d['density'] > 0:
                    vol = d['mass'] / d['density']
                if vol <= 0:
                    vol = d['fill'] * v_cav
                rep['requested'] += vol
                n = int(round(vol / S.dx ** 3))
                n = max(min(n, n_cap - n_used), 0)
                n_use = min(n, total - pos)
                P = lat_pos(I[pos:pos + n_use], S.dx)
                if n > n_use:                      # near-brim: the remaining particles go to the highest free spots of the cavity
                    hi_pts = Pf[dfine >= 0.3 * S.dx]
                    hi_pts = hi_pts[np.argsort(-_level(hi_pts, raw=True), kind='stable')]
                    k = n - n_use
                    P = np.vstack([P, hi_pts[S.rng.choice(min(len(hi_pts), 4 * k), size=min(k, len(hi_pts)), replace=False)]])
                placed[f] = P
                rep['fluids'].append((f, vol, n))
                pos += n_use
                n_used += n
            rep['overflow'] = max(rep['requested'] - 0.98 * v_cav, 0.0)
            S.fill_report.append(rep)
        # 3) gases in a container or free without a shape
        for f, d in enumerate(D):
            if placed[f] is not None or not d['gas']:
                continue
            n = int(d['count'])
            if d['body'] >= 0:
                I = interior_of(d['body'])
                if len(I) == 0:
                    placed[f] = np.zeros((0, 3))
                    continue
                keys = lat_key(I)
                I = I[[int(k) not in occupied for k in keys]] if len(I) else I
                sel = S.rng.choice(len(I), size=min(n, len(I)), replace=False) if len(I) else np.zeros(0, int)
                placed[f] = lat_pos(I[sel], S.dx) + (S.rng.random((len(sel), 3)) - 0.5) * S.dx
            else:
                placed[f] = np.zeros((0, 3))
        # gas parcels from a lattice shape: thin to `count`
        for f, d in enumerate(D):
            if d['gas'] and placed[f] is not None and len(placed[f]) > d['count'] > 0:
                placed[f] = placed[f][S.rng.choice(len(placed[f]), size=int(d['count']), replace=False)]
        # ---- assemble arrays
        xs, vs, fids = [], [], []
        S.lo, S.hi = np.zeros(nf, int), np.zeros(nf, int)
        n0 = 0
        for f, d in enumerate(D):
            P = np.zeros((0, 3)) if placed[f] is None else np.asarray(placed[f], float)
            jit = 0.03 * S.dx * (S.rng.random(P.shape) - 0.5)
            P = P + (jit if not d['gas'] else 0.0)
            S.lo[f], S.hi[f] = n0, n0 + len(P)
            n0 += len(P)
            xs.append(P); fids.append(np.full(len(P), f, int))
            vs.append(np.tile(np.asarray(d['velocity'], float), (len(P), 1)))
        n_liq = sum(int(S.hi[f] - S.lo[f]) for f in range(nf) if not D[f]['gas'])
        if auto_dx and n_liq > 1.15 * max_particles:      # the estimate was low: refine the spacing once and rebuild
            return FluidSolver.build(defs, bodies, env, dx=S.dx * (n_liq / max_particles) ** (1.0 / 3.0), max_particles=max_particles,
                                     seed=seed, thermal_speedup=thermal_speedup, iterations=iterations, max_substeps=max_substeps,
                                     vmax=vmax, lotbcbp=lotbcbp, fill_seed=fill_seed, species=species, rules=rules)
        S.n = n0
        S.x = np.concatenate(xs) if n0 else np.zeros((0, 3))
        S.v = np.concatenate(vs) if n0 else np.zeros((0, 3))
        S.fid = np.concatenate(fids) if n0 else np.zeros(0, int)
        S.defs = D
        S.nf = nf
        # ---- per fluid tables and derived properties
        arr = lambda key, dt=float: np.array([d[key] for d in D], dt)
        S.gas = arr('gas', bool)
        S.rho0 = np.zeros(nf); S.Vp = np.zeros(nf); S.molar = np.zeros(nf)
        for f, d in enumerate(D):
            nfp = max(S.hi[f] - S.lo[f], 1)
            if d['gas']:
                vol = d['volume'] if d['volume'] > 0 else (d['mass'] / d['density'] if d['mass'] > 0 and d['density'] > 0 else 0.0)
                if vol <= 0:
                    shv = d['shape']
                    vol = mesh_volume(shv['verts'], shv['tris']) if shv is not None and shv['kind'] == 'MESH' and is_closed(shv['tris']) else \
                        (len(placed[f]) * S.dx ** 3 if len(placed[f]) else 0.0)
                    if d['shape'] is None and d['body'] >= 0:
                        vol = len(interior_of(d['body'])) * S.dx ** 3 * d['fill']
                S.Vp[f] = max(vol, 1e-12) / nfp
            else:
                S.Vp[f] = S.dx ** 3
            prop = derive_properties(d['mass'], S.Vp[f] * nfp, d['density'] if d['mass'] <= 0 or d['volume'] <= 0 else 0.0, d['weight'], g)
            S.rho0[f] = prop['density'] if prop['density'] > 0 else d['density']
            if d['mass'] > 0 and (S.hi[f] - S.lo[f]) > 0 and not d['gas']:
                S.rho0[f] = d['mass'] / (S.Vp[f] * nfp) if d['volume'] > 0 or d['shape'] is None else S.rho0[f]
            S.molar[f] = S.rho0[f] * R_GAS * d['tref'] / P_ATM      # ideal gas, valid for gases
        S.mass_p = S.rho0 * S.Vp
        S.visc, S.cp, S.k = arr('visc'), arr('cp'), arr('k')
        S.beta, S.tref, S.coh = arr('beta'), arr('tref'), arr('cohesion')
        S.misc = arr('miscible', bool)
        S.cool = np.array([d['cool'] if d['cool'] else (0.15 if d['gas'] else 0.05) for d in D])
        S.free = np.array([d['body'] < 0 for d in D])
        S.drag = np.array([(d['drag'] if d['drag'] is not None else (0.6 if d['gas'] else 0.02)) for d in D]) * S.free
        S.diff = np.array([(d['diffusion'] if d['diffusion'] is not None else (0.01 if d['gas'] else 0.0)) for d in D])
        S.group = np.array([(d['group'] if d['group'] is not None else f) for f, d in enumerate(D)], int)
        S.T = np.concatenate([np.full(S.hi[f] - S.lo[f], D[f]['temperature']) for f in range(nf)]) if n0 else np.zeros(0)
        S.mass_ref = S.mass_p[S.fid] if n0 else np.zeros(0)
        S._init_species(D, species, rules)
        S.wall_dist = np.full(S.n, 1e9)
        S.gust_vec = np.zeros(3)
        S.lam_prev = np.zeros(S.n)
        S.dt_prev = 0.0
        S.sources = []
        if ground is None:
            S.ground_z = None
        else:
            S.ground_z = float(ground)
        return S

    # ------------------------------------------------------------------ species / composition
    def _init_species(self, D, species, rules):
        """Per-particle composition (one-hot of the fluid's species), cluster ids and the derived per-particle properties.  Mixing / merge / reaction
        rules only run when `rules` are given or a fluid has mix_rate > 0, so existing scenes behave exactly as before."""
        from . import species as SP
        sp = []
        self.sp_of_fluid = []
        for f, d in enumerate(D):
            nm = d.get('species') or d['name']
            if any(s['name'] == nm for s in sp):
                nm = "%s#%d" % (nm, f)
            sp.append(SP.species_def(nm, d['color'], self.rho0[f], d['cp'], d['beta'], d.get('priority', 0), d['visc']))
            self.sp_of_fluid.append(len(sp) - 1)
        for s_ in (species or []):
            if not any(q['name'] == s_['name'] for q in sp):
                sp.append(SP.species_def(**s_))
        self.SS = SP.SpeciesSystem(sp, [SP.rule_def(**r) for r in (rules or [])])
        self.sp_misc = np.array([bool(D[f]['miscible']) for f in range(len(D))] + [True] * (len(sp) - len(D)), bool)
        self.mixing = bool(rules) or any(float(d.get('mix_rate', 0.0)) > 0 for d in D)
        for r in range(len(self.SS.rules)):
            pass
        for f, d in enumerate(D):                     # per fluid mixing rate: default pair rate for its own species with others
            mr = float(d.get('mix_rate', 0.0))
            if mr > 0:
                for g in range(len(D)):
                    a, b = self.sp_of_fluid[f], self.sp_of_fluid[g]
                    if a != b and not self.SS.out[a, b] and D[g]['miscible'] and d['miscible']:
                        self.SS.out[a, b] = self.SS.out[b, a] = SP.MISCIBLE
                        self.SS.rate[a, b] = self.SS.rate[b, a] = max(mr, self.SS.rate[a, b])
                self.SS.rate[a, a] = mr
        n = self.n
        self.C = np.zeros((n, self.SS.ns))
        if n:
            self.C[np.arange(n), np.array(self.sp_of_fluid)[self.fid]] = 1.0
        self.cluster = np.arange(n)
        self.rho0_p = self.rho0[self.fid].copy() if n else np.zeros(0)
        self.beta_p = self.beta[self.fid].copy() if n else np.zeros(0)
        self.cp_p = self.cp[self.fid].copy() if n else np.zeros(0)
        self.mix_stats = dict(merges=0, reactions=0)

    def colours(self):
        """Per-particle colour derived from the composition (subtractive pigment mix)."""
        return self.SS.colour(self.C) if self.n else np.zeros((0, 3))

    def _derive(self, idx):
        C = self.C[idx]
        self.rho0_p[idx] = self.SS.density(C)
        self.beta_p[idx] = self.SS.beta_mix(C)
        self.cp_p[idx] = self.SS.cp_mix(C)

    def _mix(self, dt, idx_l, il, jl):
        from . import species as SP
        C, T = self.C[idx_l].copy(), self.T[idx_l].copy()
        mass, cp = self.mass_ref[idx_l], self.cp_p[idx_l]
        sm = self.sp_misc

        def dflt(a, b):
            return SP.MISCIBLE if (sm[a] and sm[b]) else SP.IMMISCIBLE
        info = SP.mix_step(self.SS, C, T, mass, cp, self.v[idx_l], self.x[idx_l], il, jl, self.h, dt, dflt)
        self.C[idx_l] = C
        self.T[idx_l] = T
        self._derive(idx_l)
        cl = self.cluster[idx_l].copy()
        self.cluster[idx_l] = SP.propagate_clusters(cl, self.SS.dominant(C), il, jl, info['absorb_pairs'])
        self.mix_stats['merges'] += info['merges']
        self.mix_stats['reactions'] += info['reactions']

    # ------------------------------------------------------------------ state helpers
    def pressure_at(self, z):
        return self.env.pressure_at(z)

    def rho_particles(self, T=None):
        """Current density of each particle (kg/m3): Boussinesq for liquids, ideal gas at ambient pressure for gases."""
        T = self.T if T is None else T
        f = self.fid
        rho_l = self.rho0_p * np.clip(1.0 - self.beta_p * (T - self.tref[f]), 0.2, 5.0)
        Pz = self.env.pressure_at(self.x[:, 2])
        rho_g = self.rho0_p * (self.tref[f] / np.maximum(T, 1.0)) * (Pz / P_ATM)
        return np.where(self.gas[f], rho_g, rho_l)

    def parcel_radius(self):
        """Collision radius of every particle (gas parcels follow their own volume m / rho)."""
        f = self.fid
        rho = self.rho_particles()
        Vg = self.mass_p[f] / np.maximum(rho, 1e-9)
        rg = np.cbrt(3.0 * Vg / (4.0 * np.pi))
        return np.where(self.gas[f], np.minimum(rg, 4.0 * self.dx), 0.5 * self.dx)

    def snapshot(self):
        return dict(x=self.x.copy(), v=self.v.copy(), T=self.T.copy(), t=self.t, gust=self.gust_vec.copy(), C=self.C.copy(), cluster=self.cluster.copy(),
                    bodies=[(b.M1.copy(), b.vel.copy(), b.omega.copy(), b.travel, b.F.copy(), b.T.copy()) for b in self.bodies])

    def restore(self, s):
        self.x, self.v, self.T, self.t = s['x'].copy(), s['v'].copy(), s['T'].copy(), s['t']
        self.gust_vec = s.get('gust', np.zeros(3)).copy()
        for b, st in zip(self.bodies, s.get('bodies', [])):
            if b.dynamic:
                b.M0 = b.M1 = st[0].copy()
                b.vel, b.omega, b.travel = st[1].copy(), st[2].copy(), st[3]
            b.F, b.T = st[4].copy(), st[5].copy()
        if 'C' in s and len(s['C']) == self.n:
            self.C, self.cluster = s['C'].copy(), s['cluster'].copy()
            self._derive(np.arange(self.n))

    def totals(self):
        """mass (kg, reference mass: conserved), volume (m3, actual), thermal energy proxy and fluid-wise numbers."""
        out = []
        rho = self.rho_particles()
        for f in range(self.nf):
            s = slice(self.lo[f], self.hi[f])
            m = float(self.mass_ref[s].sum())
            vol = float((self.mass_ref[s] / np.maximum(rho[s], 1e-9)).sum()) if self.gas[f] else float(self.Vp[f] * (self.hi[f] - self.lo[f]))
            out.append(dict(name=self.defs[f]['name'], n=int(self.hi[f] - self.lo[f]), mass=m, volume=vol,
                            density=m / vol if vol > 0 else 0.0, T=float(self.T[s].mean()) if self.hi[f] > self.lo[f] else 0.0))
        return out

    # ------------------------------------------------------------------ stepping
    def step(self, dt, sources=None):
        """Advance by dt seconds.  sources: list of (position, power W, radius m) heat sources (negative power cools).
        Bodies keep their M0 -> M1 interpolation for this call; they are settled afterwards."""
        if sources is not None:
            self.sources = sources
        if self.n == 0:
            self.t += dt
            return
        vmax_now = float(np.sqrt(np.max(np.einsum('ij,ij->i', self.v, self.v)))) if self.n else 0.0
        nsub = int(np.clip(math.ceil(max(vmax_now, 0.5 * np.linalg.norm(self.env.g) * dt) * dt / (0.8 * self.hg)), 1, self.max_sub))
        dyn = [b for b in self.bodies if b.dynamic]
        for b in self.bodies:
            b.F[:] = 0.0
            b.T[:] = 0.0
        for k in range(nsub):
            sub = dt / nsub
            for b in self.bodies:
                b._F[:] = 0.0
                b._T[:] = 0.0
            for b in dyn:                              # predict the pose over this substep from the present velocity
                b.M0 = b.M1
                b.rel = (k / nsub, (k + 1) / nsub)
                b.M1 = b.pose_after(sub) if b.constraint != 'SLIDER' else _slide_pose(b, sub)
            self._substep(sub, k / nsub, (k + 1) / nsub)
            for b in self.bodies:
                b.F += b._F / nsub
                b.T += b._T / nsub
            for b in dyn:
                b.M1 = b.M0                           # integrate from the start-of-substep pose with the measured force
                b.integrate(sub, self.env.g)
                self._body_contacts(b, sub)
        for b in self.bodies:
            if not b.dynamic:
                b.settle()
            else:
                b.M0 = b.M1
        self.t += dt

    def outside_count(self):
        """Particles whose centre is outside the cavity of a container that holds them (0 when confinement works)."""
        n = 0
        for bi, b in enumerate(self.bodies):
            if b.kind != 'CONTAINER':
                continue
            idx = self._bodies_for(bi)
            if len(idx):
                n += int(np.count_nonzero(b.distance(self.x[idx], b.M1) < 0.0))
        return n

    def _fluid_pairs(self, mask):
        idx = np.flatnonzero(mask)
        if len(idx) < 2:
            return idx, np.zeros(0, np.int64), np.zeros(0, np.int64)
        i, j = neighbor_pairs(self.x[idx], self.h * 1.25)
        return idx, i, j

    # ------------------------------------------------------------------ grid (FLIP) helpers
    def _grid_setup(self, xl, bodies_idx):
        """Grid covering the liquid particles (and the containers that hold liquid), aligned to multiples of the cell size."""
        hg = self.hg
        lo, hi = xl.min(0) - 3 * hg, xl.max(0) + 3 * hg
        for bi in bodies_idx:
            blo, bhi = self.bodies[bi].bbox_world()
            lo, hi = np.minimum(lo, blo - 2 * hg), np.maximum(hi, bhi + 2 * hg)
        origin = np.floor(lo / hg) * hg
        dims = np.ceil((hi - origin) / hg).astype(int) + 1
        return origin, dims

    def _solid_mask(self, origin, dims, liquid_fluids, s1):
        """Cells whose centre is on the illegal side of a body (container wall / collider), by the LOTBCBP field."""
        key = (tuple(origin), tuple(dims), tuple(tuple(b.matrix_at(s1).ravel()) for b in self.bodies))
        if self._solid_cache is not None and self._solid_cache[0] == key:
            return self._solid_cache[1]
        hg = self.hg
        S = np.zeros(dims, bool)
        Sown = -np.ones(dims, np.int16)
        ax = [origin[a] + (np.arange(dims[a]) + 0.5) * hg for a in range(3)]
        for bi, b in enumerate(self.bodies):
            if b.fluids is not None and not (set(b.fluids) & set(liquid_fluids)):
                continue
            M = b.matrix_at(s1)
            blo, bhi = b.bbox_world()
            sl = []
            for a in range(3):
                ia = np.flatnonzero((ax[a] > blo[a] - 2 * hg) & (ax[a] < bhi[a] + 2 * hg)) if b.kind == 'COLLIDER' else np.arange(dims[a])
                sl.append(ia)
            if any(len(s_) == 0 for s_ in sl):
                continue
            G = np.stack(np.meshgrid(ax[0][sl[0]], ax[1][sl[1]], ax[2][sl[2]], indexing='ij'), -1).reshape(-1, 3)
            d = b.distance(G, M)
            m = (d < 0.0).reshape(len(sl[0]), len(sl[1]), len(sl[2]))
            S[np.ix_(sl[0], sl[1], sl[2])] |= m
            sub = Sown[np.ix_(sl[0], sl[1], sl[2])]
            sub[m] = bi
            Sown[np.ix_(sl[0], sl[1], sl[2])] = sub
        if self.ground_z is not None:
            zc = origin[2] + (np.arange(dims[2]) + 0.5) * hg
            S |= (zc < self.ground_z)[None, None, :]
        self._solid_cache = (key, S)
        self._solid_owner = Sown
        return S

    @staticmethod
    def _corners(g, shape):
        """Trilinear corner flat indices and weights of grid-space points g for an array of `shape`."""
        i0 = np.floor(g).astype(np.int64)
        f = g - i0
        out = []
        for cx in (0, 1):
            for cy in (0, 1):
                for cz in (0, 1):
                    ii = i0 + np.array([cx, cy, cz])
                    w = (f[:, 0] if cx else 1 - f[:, 0]) * (f[:, 1] if cy else 1 - f[:, 1]) * (f[:, 2] if cz else 1 - f[:, 2])
                    ok = np.all((ii >= 0) & (ii < np.array(shape)), axis=1)
                    flat = (ii[:, 0] * shape[1] + ii[:, 1]) * shape[2] + ii[:, 2]
                    out.append((np.where(ok, flat, 0), np.where(ok, w, 0.0)))
        return out

    def _pressure_force(self, p, L, S, origin, hg):
        Sown = self._solid_owner
        hg2 = hg * hg
        for c in range(3):
            a_lo = tuple(slice(0, -1) if a == c else slice(None) for a in range(3))
            a_hi = tuple(slice(1, None) if a == c else slice(None) for a in range(3))
            for fl_side, sign in ((0, 1.0), (1, -1.0)):
                if fl_side == 0:
                    m = L[a_lo] & S[a_hi]
                    own = Sown[a_hi]
                    pr = p[a_lo]
                else:
                    m = L[a_hi] & S[a_lo]
                    own = Sown[a_lo]
                    pr = p[a_hi]
                ii = np.argwhere(m)
                if len(ii) == 0:
                    continue
                ow = own[m]
                f = sign * pr[m] * hg2
                pos = origin + (ii + 0.5) * hg
                pos[:, c] = origin[c] + (ii[:, c] + 1) * hg
                for bi in np.unique(ow):
                    if bi < 0 or not self.bodies[bi].two_way:
                        continue
                    sel = ow == bi
                    b = self.bodies[bi]
                    F = np.zeros(3)
                    F[c] = float(f[sel].sum())
                    r = pos[sel] - b.com_world(b.M1)
                    Fv = np.zeros((int(sel.sum()), 3)); Fv[:, c] = f[sel]
                    b._F += F
                    b._T += np.cross(r, Fv).sum(0)
                    self._acc_flip = True

    def _flip(self, dt, idx_l, rho, s1):
        """Grid pressure projection with variable density for the liquid particles (FLIP / PIC transfer).

        Gravity and the other accelerations are already in v.  The grid velocity is made divergence free with
        div((dt / rho) grad p) = div u*, p = 0 in air cells (free surface), zero normal velocity at solid cells."""
        hg = self.hg
        xl, vl, rl = self.x[idx_l], self.v[idx_l], rho[idx_l]
        lf = sorted(set(int(q) for q in self.fid[idx_l]))
        bidx = [bi for bi, b in enumerate(self.bodies) if b.kind == 'CONTAINER' and (b.fluids is None or set(b.fluids) & set(lf))]
        origin, dims = self._grid_setup(xl, bidx)
        if int(np.prod(dims)) > 6_000_000:
            return
        S = self._solid_mask(origin, dims, lf, s1)
        g = (xl - origin) / hg
        cidx = np.clip(np.floor(g).astype(np.int64), 0, np.array(dims) - 1)
        cflat = (cidx[:, 0] * dims[1] + cidx[:, 1]) * dims[2] + cidx[:, 2]
        ncell = int(np.prod(dims))
        cnt = np.bincount(cflat, minlength=ncell).reshape(dims)
        rsum = np.bincount(cflat, rl, ncell).reshape(dims)
        S = S & (cnt == 0)            # a cell with fluid in it is fluid, even if its centre is inside the wall (the wall is resolved by the particle projection)
        L = cnt > 0
        rho_c = np.where(cnt > 0, rsum / np.maximum(cnt, 1), 0.0)
        if not L.any():
            return
        # ---- particles to grid (staggered face velocities)
        U, Uw, Uold = [], [], []
        for c in range(3):
            shape = tuple(dims + np.eye(3, dtype=int)[c])
            off = np.array([0.5, 0.5, 0.5]); off[c] = 0.0
            cs = self._corners(g - off, shape)
            nsz = int(np.prod(shape))
            num = np.zeros(nsz); den = np.zeros(nsz)
            for flat, w in cs:
                num += np.bincount(flat, w * vl[:, c], nsz)
                den += np.bincount(flat, w, nsz)
            u = np.where(den > 1e-9, num / np.maximum(den, 1e-9), 0.0).reshape(shape)
            U.append(u); Uw.append((den > 1e-9).reshape(shape)); Uold.append(cs)
        # wall faces (solid neighbour): static wall velocity
        for c in range(3):
            wall = np.zeros(U[c].shape, bool)
            lo_s = S[tuple(slice(0, -1) if a == c else slice(None) for a in range(3))]
            hi_s = S[tuple(slice(1, None) if a == c else slice(None) for a in range(3))]
            wall[tuple(slice(1, -1) if a == c else slice(None) for a in range(3))] = lo_s | hi_s
            U[c][wall] = 0.0
            # a moving body drives the fluid next to it: the wall face takes the body's velocity (normal component)
            mv = [bi for bi, b_ in enumerate(self.bodies) if b_.dynamic or b_.moving()]
            if mv and wall.any():
                Sown = self._solid_owner
                own = -np.ones(U[c].shape, np.int16)
                ow_lo = Sown[tuple(slice(0, -1) if a == c else slice(None) for a in range(3))]
                ow_hi = Sown[tuple(slice(1, None) if a == c else slice(None) for a in range(3))]
                inner_sl = tuple(slice(1, -1) if a == c else slice(None) for a in range(3))
                own[inner_sl] = np.where(ow_lo >= 0, ow_lo, ow_hi)
                for bi in mv:
                    mm = wall & (own == bi)
                    if not mm.any():
                        continue
                    ii = np.argwhere(mm)
                    pos = origin + (ii + 0.5) * hg
                    pos[:, c] = origin[c] + ii[:, c] * hg
                    U[c][mm] = self.bodies[bi].point_velocity(pos, dt)[:, c]
            Uw[c] = Uw[c] | wall
        U0 = [u.copy() for u in U]
        # ---- viscosity (explicit, clamped) on the grid is applied through the particles (XSPH) afterwards
        # ---- pressure solve
        div = np.zeros(dims)
        for c in range(3):
            hi_i = tuple(slice(1, None) if a == c else slice(None) for a in range(3))
            lo_i = tuple(slice(0, -1) if a == c else slice(None) for a in range(3))
            div += (U[c][hi_i] - U[c][lo_i]) / hg
        tcoef, diag = [], np.zeros(dims)
        for c in range(3):
            a_lo = tuple(slice(0, -1) if a == c else slice(None) for a in range(3))
            a_hi = tuple(slice(1, None) if a == c else slice(None) for a in range(3))
            Lc, Ln, Sc, Sn = L[a_lo], L[a_hi], S[a_lo], S[a_hi]
            rf = np.where(Lc & Ln, 0.5 * (rho_c[a_lo] + rho_c[a_hi]), np.where(Lc, rho_c[a_lo], rho_c[a_hi]))
            t = np.where((Lc | Ln) & ~(Sc | Sn), dt / (hg * hg * np.maximum(rf, 1e-6)), 0.0)
            tcoef.append((t, Lc, Ln, a_lo, a_hi))
            diag[a_lo] += t * Lc
            diag[a_hi] += t * Ln
        rhs = np.where(L, -div, 0.0)

        def A(p):
            y = diag * p
            for t, Lc, Ln, a_lo, a_hi in tcoef:
                y[a_lo] -= t * Ln * p[a_hi]
                y[a_hi] -= t * Lc * p[a_lo]
            return np.where(L, y, 0.0)
        p = np.zeros(dims)
        if self._p_prev is not None and self._p_prev.shape == tuple(dims) and self._p_origin is not None and np.allclose(self._p_origin, origin):
            p = np.where(L, self._p_prev, 0.0)
        r = rhs - A(p)
        Minv = np.where(L & (diag > 0), 1.0 / np.maximum(diag, 1e-30), 0.0)
        z = Minv * r
        d_ = z.copy()
        rz = float((r * z).sum())
        rr0 = float((rhs * rhs).sum()) + 1e-30
        for it in range(self.cg_iters):
            if float((r * r).sum()) < 1e-8 * rr0:
                break
            Ad = A(d_)
            alpha = rz / max(float((d_ * Ad).sum()), 1e-30)
            p += alpha * d_
            r -= alpha * Ad
            z = Minv * r
            rz_new = float((r * z).sum())
            d_ = z + (rz_new / max(rz, 1e-30)) * d_
            rz = rz_new
        self.cg_last = it
        self._p_prev, self._p_origin = p, origin.copy()
        # ---- force of the pressure on the bodies (fluid faces next to a solid cell of the body), for the two-way coupling
        if any(b.two_way for b in self.bodies):
            self._pressure_force(p, L, S, origin, hg)
        # ---- velocity update from the pressure gradient
        for c in range(3):
            t, Lc, Ln, a_lo, a_hi = tcoef[c]
            rf = np.where(Lc & Ln, 0.5 * (rho_c[a_lo] + rho_c[a_hi]), np.where(Lc, rho_c[a_lo], rho_c[a_hi]))
            open_ = t > 0
            gp = (p[a_hi] - p[a_lo]) / hg
            inner = tuple(slice(1, -1) if a == c else slice(None) for a in range(3))
            U[c][inner] -= np.where(open_, dt / np.maximum(rf, 1e-6) * gp, 0.0)
        # ---- grid to particles (FLIP / PIC blend)
        pic = np.zeros_like(vl); dflip = np.zeros_like(vl)
        for c in range(3):
            du = (U[c] - U0[c]).ravel(); uu = U[c].ravel()
            for flat, w in Uold[c]:
                pic[:, c] += w * uu[flat]
                dflip[:, c] += w * du[flat]
        a = self.flip_ratio
        self.v[idx_l] = a * (vl + dflip) + (1.0 - a) * pic

    def _substep(self, dt, s0, s1):
        env, f, n = self.env, self.fid, self.n
        x, v = self.x, self.v
        rho = self.rho_particles()
        gas = self.gas[f]
        if env.gust > 0:
            tau = 2.0
            self.gust_vec += -self.gust_vec * dt / tau + env.gust * math.sqrt(2.0 * dt / tau) * self.rng.standard_normal(3)
        wind = env.wind + self.gust_vec
        rho_env = env.density_at(x[:, 2])
        a = env.g[None, :] * (1.0 - rho_env / np.maximum(rho, 1e-9))[:, None]
        a += self.drag[f][:, None] * (wind[None, :] - v)
        idx_l, il, jl = self._fluid_pairs(~gas)
        # ---- surface cohesion (same fluid or miscible partners), pair force
        if len(il) and (self.coh[f[idx_l]] > 0).any():
            I, J = idx_l[il], idx_l[jl]
            fi_, fj_ = f[I], f[J]
            same = fi_ == fj_
            both_m = self.misc[fi_] & self.misc[fj_]
            gam = np.where(same, self.coh[fi_], np.where(both_m, 0.5 * (self.coh[fi_] + self.coh[fj_]), 0.0))
            rij = x[I] - x[J]
            d = np.linalg.norm(rij, axis=1)
            m = (d < self.h) & (d > 1e-9) & (gam > 0)
            if m.any():
                q = d[m]; h = self.h
                c = np.where(2 * q > h, (h - q) ** 3 * q ** 3, 2 * (h - q) ** 3 * q ** 3 - h ** 6 / 64.0) / (h ** 6 / 64.0)
                scale = gam[m] / (np.minimum(rho[I[m]], rho[J[m]]) * self.dx ** 2) / 12.0
                fv = (-(scale * c) / q)[:, None] * rij[m]
                a += scatter_pair(n, I[m], J[m], fv)
        v += a * dt
        sp = np.linalg.norm(v, axis=1)
        over = sp > self.vmax
        if over.any():
            v[over] *= (self.vmax / sp[over])[:, None]
        # ---- liquid pressure: grid projection with variable density (FLIP)
        if len(idx_l):
            self._flip(dt, idx_l, rho, s1)
            # viscosity: smoothing toward same / miscible neighbours at a rate set by nu = mu / rho (clamped)
            if len(il) and (self.visc[f[idx_l]] > 0).any():
                I, J = idx_l[il], idx_l[jl]
                ok = (f[I] == f[J]) | (self.misc[f[I]] & self.misc[f[J]])
                rij = x[I] - x[J]
                d = np.linalg.norm(rij, axis=1)
                m = ok & (d < self.h)
                if m.any():
                    w = (1 - d[m] / self.h) ** 3
                    Im, Jm = I[m], J[m]
                    sw = np.bincount(Im, w, n) + np.bincount(Jm, w, n)
                    dv = scatter_pair(n, Im, Jm, w[:, None] * (self.v[Jm] - self.v[Im]))
                    nu = self.visc[f] / np.maximum(rho, 1e-9)
                    fac = np.minimum(0.6, 8.0 * nu * dt / self.h ** 2)
                    # blob coherence: a cohesive (surface-tension) phase also couples its own particles' velocities (XSPH toward same-phase
                    # neighbours only), so interface particles are not sheared off the blob by the surrounding carrier
                    fac = np.maximum(fac, np.minimum(0.45, 0.25 * self.coh[f]))
                    self.v += (fac / np.maximum(sw, 1e-9))[:, None] * dv
        v = self.v
        xs = x + v * dt
        prad = self.parcel_radius()
        # ---- gas parcels: soft repulsion between parcels
        gas_idx = np.flatnonzero(gas)
        gp = None
        if len(gas_idx) > 1:
            dmax = 2.0 * float(prad[gas_idx].max()) * 1.1
            ig, jg = neighbor_pairs(xs[gas_idx], dmax * 1.5)
            gp = (gas_idx[ig], gas_idx[jg])
        # ---- drift correction for the liquid (position based, density constraint, no velocity change) and gas repulsion
        rho_ref = float(np.mean(self.rho0[~self.gas])) if (~self.gas).any() else 1.0
        wl = rho_ref / np.maximum(rho[idx_l], 1e-9) if len(idx_l) else np.zeros(0)
        gi, gj = (idx_l[il], idx_l[jl]) if len(il) else (np.zeros(0, int), np.zeros(0, int))
        sig_last = None
        for it in range(self.iters):
            if len(il):
                rij = xs[gi] - xs[gj]
                d = np.linalg.norm(rij, axis=1)
                m = (d < self.h) & (d > 1e-9)
                ii, jj, dm, rm = il[m], jl[m], d[m], rij[m]
                q = dm / self.h
                W = (1 - q) ** 3
                gw = (-3.0 * (1 - q) ** 2 / self.h / dm)[:, None] * rm
                nl = len(idx_l)
                sig = np.full(nl, 1.0) + np.bincount(ii, W, nl) + np.bincount(jj, W, nl)
                C = sig / self.sig0 - 1.0
                lam = np.zeros(nl)
                pos_ = C > 0
                gsum = scatter_pair(nl, ii, jj, gw)
                g2 = np.einsum('ij,ij->i', gw, gw)
                den = (wl * np.einsum('ij,ij->i', gsum, gsum) + np.bincount(ii, wl[jj] * g2, nl) + np.bincount(jj, wl[ii] * g2, nl)) / self.sig0 ** 2
                lam[pos_] = -C[pos_] / (den[pos_] + 1e-12)
                sl = (lam[ii] + lam[jj])[:, None] * gw / self.sig0 * self.relax
                dxl = wl[:, None] * scatter_pair(nl, ii, jj, sl)
                nm = np.linalg.norm(dxl, axis=1)
                dxl *= np.minimum(1.0, 0.25 * self.dx / np.maximum(nm, 1e-12))[:, None]
                xs[idx_l] += dxl
                sig_last = sig
            if gp is not None:
                I, J = gp
                rij = xs[I] - xs[J]
                d = np.linalg.norm(rij, axis=1)
                dmin = 1.6 * (prad[I] + prad[J]) * 0.5 * 1.25
                m = (d < dmin) & (d > 1e-9)
                if m.any():
                    corr = (0.25 * (dmin[m] - d[m]) / d[m])[:, None] * rij[m]
                    xs += scatter_pair(n, I[m], J[m], corr)
            self._project(xs, prad, s1)
        if self.iters == 0:
            self._project(xs, prad, s1)
        # gas velocity follows positions (liquids keep the grid velocity)
        vn = v.copy()
        if len(gas_idx):
            vn[gas_idx] = ((xs - x) / dt)[gas_idx]
        dif = self.diff[f]
        if (dif > 0).any():
            xs += self.rng.standard_normal((n, 3)) * np.sqrt(2.0 * dif * dt)[:, None]
            self._project(xs, prad, s1)
        self._contacts(xs, vn, prad, dt, s0, s1)
        sp = np.linalg.norm(vn, axis=1)
        over = sp > self.vmax
        if over.any():
            vn[over] *= (self.vmax / sp[over])[:, None]
        self.x[:] = xs
        self.v[:] = vn
        self._heat(dt, rho, sig_last, (idx_l, il, jl))
        if self.mixing and len(idx_l) and len(il):
            self._mix(dt, idx_l, il, jl)


    # ------------------------------------------------------------------ collision projection
    def _bodies_for(self, bi):
        b = self.bodies[bi]
        if b.fluids is None:
            return np.arange(self.n)
        mask = np.isin(self.fid, list(b.fluids))
        return np.flatnonzero(mask)

    def _body_contacts(self, b, dt):
        """A dynamic body does not sink through the container walls or the ground: sample points of its surface are kept on the legal side."""
        sp = b.field.soup
        if len(sp) == 0:
            return
        pts = np.vstack([sp.A, (sp.A + sp.B + sp.C) / 3.0])
        if len(pts) > 600:
            pts = pts[np.linspace(0, len(pts) - 1, 600).astype(int)]
        for _ in range(3):
            W = pts @ b.M1[:3, :3].T + b.M1[:3, 3]
            push = np.zeros(3)
            nsum = np.zeros(3)
            for o in self.bodies:
                if o is b or o.kind != 'CONTAINER':
                    continue
                d = o.distance(W, o.M1)
                bad = np.flatnonzero(d < 0.0)
                if len(bad) == 0:
                    continue
                j = bad[np.argmin(d[bad])]
                _d, nrm, _R, _s = o.distance_normal(W[j:j + 1], o.M1)
                push += nrm[0] * (-d[j])
                nsum += nrm[0]
            if self.ground_z is not None:
                dz = float(W[:, 2].min()) - self.ground_z
                if dz < 0:
                    push[2] += -dz
                    nsum[2] += 1.0
            if float(np.linalg.norm(push)) < 1e-9:
                break
            if b.constraint == 'SLIDER':
                ax = np.asarray(b.axis, float) / max(np.linalg.norm(b.axis), 1e-12)
                push = ax * float(push @ ax)
            M = b.M1.copy()
            M[:3, 3] += push
            b.M1 = M
            b.travel += float(push @ (np.asarray(b.axis, float) / max(np.linalg.norm(b.axis), 1e-12))) if b.constraint == 'SLIDER' else 0.0
            n = nsum / max(np.linalg.norm(nsum), 1e-12)
            vn = float(b.vel @ n)
            if vn < 0:
                b.vel = b.vel - vn * n

    def _project(self, xs, prad, s1):
        """Push every particle to the legal side of every body (and the ground) by its radius."""
        for bi, b in enumerate(self.bodies):
            idx = self._bodies_for(bi)
            if len(idx) == 0:
                continue
            M = b.matrix_at(s1)
            P = xs[idx]
            lo, hi = b.bbox_world()
            pad = 3 * self.dx
            if b.kind == 'COLLIDER':
                near = np.flatnonzero(np.all((P > lo - pad) & (P < hi + pad), axis=1))
                if len(near) == 0:
                    continue
            else:
                near = np.arange(len(idx))
            d0 = b.distance(P[near], M)
            viol = np.flatnonzero(d0 < prad[idx[near]])
            if len(viol) == 0:
                continue
            sel = near[viol]
            d, nrm, _R, _s = b.distance_normal(P[sel], M, refine_band=2.0)
            self.escapes += int(np.count_nonzero(d < 0.0)) if b.kind == 'CONTAINER' else 0   # centre beyond the wall: counted
            push = np.maximum(prad[idx[sel]] - d, 0.0)
            xs[idx[sel]] += nrm * push[:, None]
            if b.kind == 'CONTAINER':         # safety: a second pass for anything still outside the cavity
                d2 = b.distance(xs[idx[sel]], M)
                bad = np.flatnonzero(d2 < 0.0)
                for _ in range(3):
                    if len(bad) == 0:
                        break
                    d3, n3, _R, _s = b.distance_normal(xs[idx[sel[bad]]], M, refine_band=2.0)
                    xs[idx[sel[bad]]] += n3 * (prad[idx[sel[bad]]] - d3 + 0.25 * self.dx)[:, None]
                    bad = bad[b.distance(xs[idx[sel[bad]]], M) < 0.0]
        gz = self.ground_z
        if gz is not None:
            viol = xs[:, 2] < gz + prad
            xs[viol, 2] = gz + prad[viol]

    def _contacts(self, xs, vn, prad, dt, s0, s1):
        """Wall surface velocity at contact (v_wall = (M(s1) p - M(s0) p) / dt for the body-fixed point p), friction on the
        tangential relative velocity, outward normal velocity removed.  Also stores the wall distance for cooling."""
        wd = np.full(self.n, 1e9)
        for bi, b in enumerate(self.bodies):
            idx = self._bodies_for(bi)
            if len(idx) == 0:
                continue
            M1 = b.matrix_at(s1)
            P = xs[idx]
            if b.kind == 'COLLIDER':
                lo, hi = b.bbox_world()
                near = np.flatnonzero(np.all((P > lo - 4 * self.dx) & (P < hi + 4 * self.dx), axis=1))
            else:
                near = np.arange(len(idx))
            if len(near) == 0:
                continue
            d0 = b.distance(P[near], M1)
            wd[idx[near]] = np.minimum(wd[idx[near]], d0)
            c = np.flatnonzero(d0 < prad[idx[near]] + 0.6 * self.dx)
            if len(c) == 0:
                continue
            sel = near[c]
            gi = idx[sel]
            d, nrm, _R, _s = b.distance_normal(P[sel], M1)
            M0 = b.matrix_at(s0)
            Pl, _r, sc = b.local(P[sel], M1)
            w1 = Pl * sc @ _r.T * 0 + (Pl * sc) @ (M1[:3, :3] / sc).T + M1[:3, 3] if False else None
            # body-fixed point in world at s0 and s1
            Rn1, t1, s_1 = b._split(M1)
            Rn0, t0, s_0 = b._split(M0)
            p_l = Pl                                                   # local coordinates at the end of the sub-step
            pw1 = (p_l * s_1) @ Rn1.T + t1
            pw0 = (p_l * s_0) @ Rn0.T + t0
            vw = (pw1 - pw0) / dt
            vr = vn[gi] - vw
            vn_c = np.einsum('ij,ij->i', vr, nrm)
            vr_n = np.minimum(vn_c, 0.0)[:, None] * nrm                 # keep the part moving into the wall removed below
            vr_t = vr - vn_c[:, None] * nrm
            mu = min(max(b.friction, 0.0), 1.0)
            v_old = vn[gi].copy()
            vn[gi] = vw + np.maximum(vn_c, 0.0)[:, None] * nrm + (1.0 - mu) * vr_t
            if b.two_way:
                # momentum the wall gave the particles, reversed: tangential drag for every particle; the normal part only for gas parcels
                # (liquids get their wall pressure from the grid projection, counted in _pressure_force)
                dv = vn[gi] - v_old
                gasc = self.gas[self.fid[gi]]
                dvn = np.einsum('ij,ij->i', dv, nrm)
                dv_use = np.where(gasc[:, None], dv, dv - dvn[:, None] * nrm)
                imp = -self.mass_ref[gi][:, None] * dv_use / dt
                b._F += imp.sum(0)
                b._T += np.cross(xs[gi] - b.com_world(b.M1), imp).sum(0)
        if self.ground_z is not None:
            gd = xs[:, 2] - self.ground_z
            wd = np.minimum(wd, gd)
            c = np.flatnonzero(gd < prad + 0.6 * self.dx)
            if len(c):
                mu = min(max(self.env.ground_friction, 0.0), 1.0)
                vz = vn[c, 2]
                vn[c, 2] = np.maximum(vz, 0.0)
                vn[c, :2] *= (1.0 - mu)
        self.wall_dist = wd

    # ------------------------------------------------------------------ heat
    def _heat(self, dt, rho, sig, pairs):
        f, n, env = self.fid, self.n, self.env
        dts = dt * self.speedup
        Cap = self.mass_ref * self.cp_p
        T = self.T
        dQ = np.zeros(n)
        # sources (lamps): power shared by the particles within the radius with a (1 - r/R)^2 weight
        for (pos, power, rad) in self.sources:
            d = np.linalg.norm(self.x - np.asarray(pos, float), axis=1)
            w = np.where(d < rad, (1.0 - d / max(rad, 1e-9)) ** 2, 0.0)
            sw = w.sum()
            if sw > 0:
                dQ += power * dts * w / sw
        # conduction between liquid neighbours (and gas parcels among themselves): conservative pair exchange
        idx_l, il, jl = pairs
        pr = []
        if len(il):
            pr.append((idx_l[il], idx_l[jl], self.h))
        gas_idx = np.flatnonzero(self.gas[f])
        if len(gas_idx) > 1:
            ig, jg = neighbor_pairs(self.x[gas_idx], 3.0 * self.dx)
            pr.append((gas_idx[ig], gas_idx[jg], 3.0 * self.dx))
        for (I, J, hh) in pr:
            if len(I) == 0:
                continue
            d = np.linalg.norm(self.x[I] - self.x[J], axis=1)
            m = d < hh
            if not m.any():
                continue
            I, J, d = I[m], J[m], d[m]
            ki, kj = self.k[f[I]], self.k[f[J]]
            kij = 2 * ki * kj / np.maximum(ki + kj, 1e-12)
            G = kij * self.dx * (1.0 - d / hh) * dts
            load = np.bincount(I, G / Cap[I], n) + np.bincount(J, G / Cap[J], n)
            lim = np.minimum(1.0, 0.45 / np.maximum(load, 1e-12))
            G = G * np.minimum(lim[I], lim[J])
            q = G * (T[J] - T[I])
            dQ += np.bincount(I, q, n) - np.bincount(J, q, n)
        # relaxation to the ambient / ground temperature by exposure (free surface, wall proximity, free gas)
        expo = np.clip(1.0 - (self.wall_dist - 0.5 * self.dx) / (2.0 * self.dx), 0.0, 1.0)
        if sig is not None and len(idx_l):
            sf = np.zeros(n)
            sf[idx_l] = np.clip(2.0 * (1.0 - sig / self.sig0), 0.0, 1.0)
            expo = np.maximum(expo, sf)
        expo = np.where(self.gas[f] | self.free[f], 1.0, expo)
        Tenv = env.temperature
        rate = np.clip(self.cool[f] * expo * dts, 0.0, 0.95)
        T_target = np.full(n, Tenv)
        if env.ground_temp is not None and self.ground_z is not None:
            near = (self.x[:, 2] - self.ground_z) < 2 * self.dx
            T_target[near] = env.ground_temp
        T += dQ / Cap + (T_target - T) * rate


def _slide_pose(b, dt):
    ax = np.asarray(b.axis, float)
    ax = ax / max(np.linalg.norm(ax), 1e-12)
    M = b.M1.copy()
    M[:3, 3] += ax * float(b.vel @ ax) * dt
    return M


def step_many(S, n, dt=1.0 / 24.0, **kw):
    for _ in range(n):
        S.step(dt, **kw)
    return S
