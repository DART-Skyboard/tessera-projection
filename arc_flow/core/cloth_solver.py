"""Pure NumPy XPBD cloth solver (no bpy).

XPBD stretch / shear / isometric-bend distance constraints, optional dihedral
bending, pin and armature skin-weight attachments, gravity, damping,
aerodynamic drag & lift against a wind or fluid velocity field, LOTBCBP SDF
collision-body projection (via ``core.sdf``) and a spatial-hash self-collision
push-out.

Garment presets (Silk, Cotton, Denim, Leather, Cape) set mass density,
stiffness, damping and aero coefficients.  Quality profiles trade substeps,
iterations and tessellation budget for 60 fps real-time vs. high-detail.
"""
from collections import defaultdict

import numpy as np

from . import sdf
from . import cloth_selfcol as SC

# ----------------------------------------------------------------------------- presets & quality

# density: kg / m^2 (areal).  stretch / shear / bend are XPBD stiffness (1/compliance),
# with bend applied to both isometric (opposite-vertex) and dihedral constraints.
# drag / lift are dimensionless aero coefficients.  thickness / margin in metres.
PRESETS = {
    'SILK': dict(
        label="Silk / Light Skirt",
        density=0.08, stretch=2.4e4, shear=9.0e3, bend=12.0, damping=0.12,
        thickness=0.0015, margin=0.004, drag=1.35, lift=0.55, friction=0.15,
    ),
    'COTTON': dict(
        label="Cotton / Shirt",
        density=0.18, stretch=4.0e4, shear=1.6e4, bend=40.0, damping=0.18,
        thickness=0.0025, margin=0.005, drag=1.05, lift=0.35, friction=0.25,
    ),
    'DENIM': dict(
        label="Denim / Jacket",
        density=0.42, stretch=8.0e4, shear=3.5e4, bend=180.0, damping=0.24,
        thickness=0.0040, margin=0.006, drag=0.90, lift=0.22, friction=0.35,
    ),
    'LEATHER': dict(
        label="Leather / Belt",
        density=0.70, stretch=1.4e5, shear=7.0e4, bend=520.0, damping=0.30,
        thickness=0.0050, margin=0.007, drag=0.75, lift=0.12, friction=0.45,
    ),
    'CAPE': dict(
        label="Cape / Banner",
        density=0.22, stretch=3.2e4, shear=1.1e4, bend=16.0, damping=0.14,
        thickness=0.0030, margin=0.006, drag=1.50, lift=0.60, friction=0.20,
    ),
}

PRESET_ITEMS = [(k, v['label'], v['label']) for k, v in PRESETS.items()]

QUALITY = {
    'REALTIME': dict(label="Real-time (60 fps)", substeps=2, iterations=4,
                     self_collision=True, self_collision_mode='FAST', use_dihedral=False, tessellation_cap=20000, tessellation_edge=0.08),
    'BALANCED': dict(label="Balanced", substeps=4, iterations=8,
                     self_collision=True, use_dihedral=True, tessellation_cap=80000, tessellation_edge=0.04),
    'QUALITY': dict(label="High detail", substeps=8, iterations=12,
                    self_collision=True, use_dihedral=True, tessellation_cap=200000, tessellation_edge=0.02),
}

QUALITY_ITEMS = [(k, v['label'], v['label']) for k, v in QUALITY.items()]

# Chaos Cloth / Unity coefficient hints map SI stiffness into a 0–1 game range.
_STRETCH_REF = 1.4e5
_BEND_REF = 520.0


# ----------------------------------------------------------------------------- mesh helpers

def grid_mesh(nx, ny, width=1.0, height=1.0, origin=(0.0, 0.0, 0.0), axis='xy'):
    """Regular (nx * ny) vertex grid, triangulated into 2*(nx-1)*(ny-1) faces.

    ``axis`` is the plane of the cloth: 'xy' (horizontal), 'xz' (vertical, hangs in Z),
    'yz' (vertical, hangs in Z, extends in Y).  Returns verts (V,3), tris (F,3),
    quads (Q,4) in consistent winding.
    """
    nx, ny = int(nx), int(ny)
    if nx < 2 or ny < 2:
        raise ValueError("grid_mesh needs nx, ny >= 2")
    xs = np.linspace(0.0, width, nx)
    ys = np.linspace(0.0, height, ny)
    origin = np.asarray(origin, float).reshape(3)
    vv = []
    for j in range(ny):
        for i in range(nx):
            if axis == 'xy':
                vv.append((xs[i], ys[j], 0.0))
            elif axis == 'yz':
                vv.append((0.0, xs[i], ys[j]))
            else:  # 'xz'  x across, z up
                vv.append((xs[i], 0.0, ys[j]))
    verts = np.asarray(vv, float) + origin
    tris, quads = [], []
    def vid(i, j):
        return j * nx + i
    for j in range(ny - 1):
        for i in range(nx - 1):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            # two triangles of the quad (a,b,c,d)
            tris.append((a, b, c)); tris.append((a, c, d))
            quads.append((a, b, c, d))
    return verts, np.asarray(tris, np.int64), np.asarray(quads, np.int64)


def top_row_pins(nx, ny):
    """Vertex indices of the last row of a ``grid_mesh`` (the 'top' in j)."""
    return np.arange((ny - 1) * nx, ny * nx, dtype=np.int64)


def band_pins(verts, axis=2, lo=0.45, hi=0.55):
    """Vertices whose coordinate along ``axis`` sits in the [lo, hi] fraction of the extent."""
    verts = np.asarray(verts, float)
    c = verts[:, int(axis)]
    z0, z1 = float(c.min()), float(c.max())
    a, b = z0 + lo * (z1 - z0), z0 + hi * (z1 - z0)
    return np.nonzero((c >= a) & (c <= b))[0].astype(np.int64)


def shoulder_pins(verts, frac=0.12, top_frac=0.85, axis_up=2, axis_across=0):
    """Two clusters near the top-left and top-right of a hanging garment (shoulder ring)."""
    verts = np.asarray(verts, float)
    up, ac = verts[:, int(axis_up)], verts[:, int(axis_across)]
    z0, z1 = float(up.min()), float(up.max())
    x0, x1 = float(ac.min()), float(ac.max())
    top = up >= z0 + top_frac * (z1 - z0)
    span = max(x1 - x0, 1e-9)
    left = top & (ac <= x0 + frac * span)
    right = top & (ac >= x1 - frac * span)
    return np.unique(np.concatenate([np.nonzero(left)[0], np.nonzero(right)[0]])).astype(np.int64)


def unique_edges(tris):
    """Undirected unique edges of a triangle mesh, (E, 2) with i < j."""
    tris = np.asarray(tris, np.int64)
    if len(tris) == 0:
        return np.zeros((0, 2), np.int64)
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e.sort(axis=1)
    return np.unique(e, axis=0)


def topology(tris, nverts=None):
    """Mesh edges, opposite-vertex (shear / isometric-bend) pairs, and dihedral hinges.

    Returns dict with:
      edges (E,2)           unique mesh edges
      shear (S,2)           opposite vertices of each adjacent triangle pair (the 'other' diagonal)
      hinges (H,4)          (i, j, k, l) shared edge i-j, opposite verts k, l
      neighbors             list of sets, 1-ring adjacency
    """
    tris = np.asarray(tris, np.int64)
    edges = unique_edges(tris)
    n = int(nverts if nverts is not None else (int(tris.max()) + 1 if len(tris) else 0))
    neighbors = [set() for _ in range(n)]
    for a, b in edges:
        neighbors[int(a)].add(int(b))
        neighbors[int(b)].add(int(a))
    # map undirected edge -> list of (third vertex, triangle index)
    inc = defaultdict(list)
    for t, (a, b, c) in enumerate(tris):
        for u, v, w in ((a, b, c), (b, c, a), (c, a, b)):
            key = (int(u), int(v)) if u < v else (int(v), int(u))
            inc[key].append((int(w), t))
    shear, hinges = [], []
    for (i, j), hits in inc.items():
        if len(hits) < 2:
            continue
        # every unique pair of incident triangles forms a hinge
        for a in range(len(hits)):
            for b in range(a + 1, len(hits)):
                k, l = hits[a][0], hits[b][0]
                if k == l:
                    continue
                hinges.append((i, j, k, l))
                p, q = (k, l) if k < l else (l, k)
                shear.append((p, q))
    shear = np.unique(np.asarray(shear, np.int64).reshape(-1, 2), axis=0) if shear else np.zeros((0, 2), np.int64)
    hinges = np.asarray(hinges, np.int64) if hinges else np.zeros((0, 4), np.int64)
    return dict(edges=edges, shear=shear, hinges=hinges, neighbors=neighbors)


def vertex_areas(verts, tris):
    """One-third of incident triangle area at each vertex, (V,)."""
    verts = np.asarray(verts, float)
    tris = np.asarray(tris, np.int64)
    area = np.zeros(len(verts), float)
    if len(tris) == 0:
        return area
    p = verts[tris]
    a = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    for k in range(3):
        np.add.at(area, tris[:, k], a / 3.0)
    return area


def _compliance(stiffness):
    s = float(stiffness)
    return 1.0 / s if s > 0.0 else 1e12


def _color_groups(indices, nverts):
    """Greedy vertex-disjoint colouring of constraint index rows (each row is a list of verts)."""
    indices = np.asarray(indices, np.int64)
    if len(indices) == 0:
        return []
    adj = [[] for _ in range(int(nverts))]
    for k, row in enumerate(indices):
        for v in np.unique(row):
            if 0 <= int(v) < nverts:
                adj[int(v)].append(k)
    colors = np.full(len(indices), -1, np.int32)
    for k, row in enumerate(indices):
        used = set()
        for v in np.unique(row):
            if 0 <= int(v) < nverts:
                for e in adj[int(v)]:
                    if colors[e] >= 0:
                        used.add(int(colors[e]))
        c = 0
        while c in used:
            c += 1
        colors[k] = c
    return [np.nonzero(colors == c)[0] for c in range(int(colors.max()) + 1)]


def _pack_pairs(n, pairs):
    out = set()
    for i, j in pairs:
        if i == j:
            continue
        a, b = (int(i), int(j)) if i < j else (int(j), int(i))
        out.add(a * n + b)
    return out


# ----------------------------------------------------------------------------- solver


class ClothSolver:
    """XPBD cloth on a triangle mesh.

    Parameters
    ----------
    verts, tris : rest-pose mesh in world metres
    pins : vertex indices held to ``pin_targets`` (defaults to the rest position)
    preset, quality : keys into PRESETS / QUALITY (overridable by kwargs)
    """

    def __init__(self, verts, tris, pins=None, pin_targets=None, preset='COTTON',
                 quality='BALANCED', gravity=(0.0, 0.0, -9.81), air_density=1.225,
                 ground_z=None, **kwargs):
        verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
        tris = np.asarray(tris, dtype=np.int64)
        if tris.ndim == 1:
            tris = tris.reshape(-1, 3)
        elif tris.ndim == 2 and tris.shape[1] == 4:
            # two triangles per quad, preserving winding
            tris = np.ascontiguousarray(np.vstack((tris[:, [0, 1, 2]], tris[:, [0, 2, 3]])))
        elif tris.ndim != 2 or tris.shape[1] != 3:
            raise ValueError("ClothSolver faces must be triangles or quads")
        if len(verts) < 3 or len(tris) < 1:
            raise ValueError("ClothSolver needs at least one triangle")
        self.rest = verts.copy()
        self.x = verts.copy()
        self.x_prev = verts.copy()
        self.v = np.zeros_like(verts)
        self.tris = tris.copy()
        self.n = len(verts)
        self.t = 0.0
        self.frame = 0

        pr = dict(PRESETS[preset if preset in PRESETS else 'COTTON'])
        q = dict(QUALITY[quality if quality in QUALITY else 'BALANCED'])
        pr.update(q)
        pr.update(kwargs)
        self.preset = preset if preset in PRESETS else 'COTTON'
        self.quality = quality if quality in QUALITY else 'BALANCED'
        self.density = float(pr['density'])
        self.thickness = float(pr['thickness'])
        self.margin = float(pr.get('margin', self.thickness * 2.0))
        self.friction = float(pr.get('friction', 0.2))
        self.drag = float(pr['drag'])
        self.lift = float(pr['lift'])
        self.damping = float(pr['damping'])
        self.substeps = int(pr.get('substeps', 4))
        self.iterations = int(pr.get('iterations', 8))
        self.self_collision = bool(pr.get('self_collision', True))
        # 'FAST' (cached candidates, one pass: real-time) | 'ACCURATE' (re-detect each substep, two passes)
        self.self_collision_mode = str(pr.get('self_collision_mode', 'ACCURATE'))
        # contact thickness / friction against itself (0 / -1 -> follow cloth thickness / friction)
        self.self_thickness = float(pr.get('self_thickness', 0.0))
        self.self_inloop = False
        self.self_friction = float(pr.get('self_friction', -1.0))
        self.self_layers = None
        self.self_ignore_pairs = ()
        self._selfcol = None
        self.tessellation_cap = int(pr.get('tessellation_cap', 80000))
        self.tessellation_edge = float(pr.get('tessellation_edge', 0.0))
        self.use_dihedral = bool(pr.get('use_dihedral', True))
        self.use_isometric_bend = bool(pr.get('use_isometric_bend', True))
        self.vmax = float(pr.get('vmax', 40.0))
        self.gravity = np.asarray(gravity, float).reshape(3)
        self.air_density = float(air_density)
        self.ground_z = None if ground_z is None else float(ground_z)
        self.wind0 = np.zeros(3, float)

        areas = vertex_areas(verts, tris)
        areas = np.maximum(areas, 1e-8)
        self.mass = self.density * areas
        self.inv_mass = 1.0 / self.mass
        self.area0 = areas.copy()

        topo = topology(tris, self.n)
        self.stretch_edges = topo['edges']
        self.shear_edges = topo['shear']
        self.hinges = topo['hinges']
        self.neighbors = topo['neighbors']
        self.stretch_rest = _edge_lengths(self.rest, self.stretch_edges)
        self.shear_rest = _edge_lengths(self.rest, self.shear_edges)
        self.bend_edges = self.shear_edges  # isometric bend uses the opposite-vertex pair
        self.bend_rest = self.shear_rest.copy()
        self.hinge_rest = _hinge_angles(self.rest, self.hinges) if len(self.hinges) else np.zeros(0, float)

        self.stretch_compliance = _compliance(pr['stretch'])
        self.shear_compliance = _compliance(pr['shear'])
        # isometric bend shares the 'bend' stiffness; dihedral uses a slightly softer copy
        self.bend_compliance = _compliance(pr['bend'])
        self.dihedral_compliance = _compliance(max(pr['bend'] * 0.35, 1e-6))

        self._groups_stretch = _color_groups(self.stretch_edges, self.n)
        self._groups_shear = _color_groups(self.shear_edges, self.n)
        self._groups_bend = _color_groups(self.bend_edges, self.n)
        self._groups_hinge = _color_groups(self.hinges, self.n) if len(self.hinges) else []

        self._lam_stretch = np.zeros(len(self.stretch_edges), float)
        self._lam_shear = np.zeros(len(self.shear_edges), float)
        self._lam_bend = np.zeros(len(self.bend_edges), float)
        self._lam_hinge = np.zeros(len(self.hinges), float)
        self._lam_pin = None

        # pins
        self.pin_hard = np.zeros(self.n, bool)
        self.pin_target = self.rest.copy()
        self.pin_vel = np.zeros_like(self.rest)
        self.pin_compliance = np.full(self.n, 1e12, float)  # unused unless soft
        self.pin_soft = np.zeros(self.n, bool)
        if pins is not None:
            self.set_pins(pins, pin_targets)

        # skinning (optional)
        self.skin_idx = None   # (N, K) int, -1 unused
        self.skin_w = None     # (N, K)
        self.inv_bind = None   # (B, 4, 4)
        self.skin_threshold = 0.65
        self.anim_drive = 1.0

        # collision
        self.collider = None          # sdf.TriSoup or None
        self.collider_prev = None     # verts of previous collider sample (for velocity)
        self._collider_v = None

        self.last_ms = 0.0
        self.last_stretch = 0.0
        self.tension = np.zeros(self.n, float)

    # ---------------------------------------------------------------- pins / skin

    def set_pins(self, indices, targets=None, hard=True, compliance=0.0):
        """Pin vertices.  ``targets`` defaults to the current position of each index."""
        idx = np.unique(np.asarray(indices, np.int64).ravel())
        idx = idx[(idx >= 0) & (idx < self.n)]
        self.pin_hard[:] = False
        self.pin_soft[:] = False
        self.inv_mass[:] = 1.0 / self.mass
        if len(idx) == 0:
            return
        if targets is None:
            tgt = self.x[idx].copy()
        else:
            tgt = np.asarray(targets, float).reshape(-1, 3)
            if len(tgt) == 1:
                tgt = np.broadcast_to(tgt, (len(idx), 3)).copy()
            elif len(tgt) != len(idx):
                raise ValueError("pin targets must match pin indices")
        self.pin_target[idx] = tgt
        if hard:
            self.pin_hard[idx] = True
            self.inv_mass[idx] = 0.0
            self.x[idx] = tgt
            self.x_prev[idx] = tgt
            self.v[idx] = 0.0
        else:
            self.pin_soft[idx] = True
            self.pin_compliance[idx] = float(compliance)
        self._lam_pin = np.zeros(int(self.pin_soft.sum()), float)

    def set_pin_targets(self, targets, indices=None, dt=None):
        """Move pin targets (e.g. a driving bone).  Optional ``dt`` sets pin velocity."""
        if indices is None:
            mask = self.pin_hard | self.pin_soft
            indices = np.nonzero(mask)[0]
        else:
            indices = np.asarray(indices, np.int64)
            mask = np.zeros(self.n, bool)
            mask[indices] = True
        tgt = np.asarray(targets, float).reshape(-1, 3)
        if len(tgt) == 1:
            tgt = np.broadcast_to(tgt, (len(indices), 3)).copy()
        if dt is not None and dt > 0:
            self.pin_vel[indices] = (tgt - self.pin_target[indices]) / float(dt)
        self.pin_target[indices] = tgt

    def set_skin(self, bone_indices, weights, bind_matrices, threshold=0.65, anim_drive=1.0):
        """Armature skin-weight attachments.

        ``bone_indices`` (N, K) and ``weights`` (N, K) are the K leading influences
        per vertex (index -1 = unused).  ``bind_matrices`` is (B, 4, 4) world-space
        bind pose; pass the current pose each step as ``bone_matrices`` to ``step``.
        Vertices whose weights sum to at least ``threshold`` become hard pins.
        """
        idx = np.asarray(bone_indices, np.int64)
        w = np.asarray(weights, float)
        if idx.shape != w.shape or idx.shape[0] != self.n:
            raise ValueError("skin arrays must be (N, K)")
        self.skin_idx = idx
        self.skin_w = w
        bind = np.asarray(bind_matrices, float).reshape(-1, 4, 4)
        self.inv_bind = np.linalg.inv(bind)
        self.skin_threshold = float(threshold)
        self.anim_drive = float(np.clip(anim_drive, 0.0, 1.0))
        # bind-pose targets: identity deformation = rest
        self.update_skin_targets(bind)
        heavy = w.sum(axis=1) >= self.skin_threshold
        if heavy.any():
            pins = np.nonzero(heavy)[0]
            self.set_pins(pins, self.pin_target[pins], hard=True)

    def update_skin_targets(self, bone_matrices, dt=None):
        if self.skin_idx is None:
            return
        M = np.asarray(bone_matrices, float).reshape(-1, 4, 4)
        # deformed = sum_j w_ij * (M_j @ inv_bind_j @ rest_h)
        rest_h = np.ones((self.n, 4), float)
        rest_h[:, :3] = self.rest
        acc = np.zeros((self.n, 3), float)
        wsum = np.zeros(self.n, float)
        K = self.skin_idx.shape[1]
        for k in range(K):
            bi = self.skin_idx[:, k]
            ww = self.skin_w[:, k]
            valid = (bi >= 0) & (ww > 0)
            if not valid.any():
                continue
            # per-vertex bone transform
            T = M[bi[valid]] @ self.inv_bind[bi[valid]]
            p = np.einsum('nij,nj->ni', T, rest_h[valid])
            acc[valid] += ww[valid, None] * p[:, :3]
            wsum[valid] += ww[valid]
        good = wsum > 1e-8
        tgt = self.pin_target.copy()
        tgt[good] = acc[good] / wsum[good, None]
        driven = good & ((self.pin_hard | self.pin_soft) | (wsum >= self.skin_threshold))
        if dt is not None and dt > 0:
            self.pin_vel[driven] = (tgt[driven] - self.pin_target[driven]) / float(dt)
        self.pin_target[driven] = tgt[driven]
        # vertices above threshold that were not yet pinned become pins
        new = good & (wsum >= self.skin_threshold) & ~self.pin_hard
        if new.any():
            idx = np.nonzero(new)[0]
            # don't wipe existing pins: just add
            self.pin_hard[idx] = True
            self.inv_mass[idx] = 0.0

    # ---------------------------------------------------------------- collider / wind

    def set_pattern_rest(self, pattern_verts):
        """Use a cutting-pattern pose for stretch / shear / bend rest lengths.

        ``self.rest`` (and therefore armature skinning / reset) stays at the
        bind pose, which may be a gathered wrap; the pattern supplies the
        surplus length that buckles into folds.
        """
        p = np.asarray(pattern_verts, float).reshape(-1, 3)
        if len(p) != self.n:
            raise ValueError("pattern rest must have one vertex per cloth vertex")
        self.stretch_rest = _edge_lengths(p, self.stretch_edges)
        self.shear_rest = _edge_lengths(p, self.shear_edges)
        self.bend_rest = self.shear_rest.copy()
        if len(self.hinges):
            self.hinge_rest = _hinge_angles(p, self.hinges)
        self.pattern_rest = p.copy()

    def set_collider_mesh(self, verts, tris, margin=None, tessellate=True):

        """Build (or replace) the LOTBCBP collision soup from a world-space mesh."""
        verts = np.asarray(verts, float).reshape(-1, 3)
        tris = np.asarray(tris, np.int64).reshape(-1, 3)
        if margin is not None:
            self.margin = float(margin)
        if len(verts) == 0 or len(tris) == 0:
            self.collider = None
            return
        soup = sdf.soup_from_mesh(verts, tris)
        edge = float(self.tessellation_edge)
        if tessellate and edge > 0 and len(soup) > 0:
            soup = sdf.virtual_subdivide(soup, edge, cap=int(self.tessellation_cap))
        self.collider = soup
        self._collider_v = None

    def set_collider_soup(self, soup, margin=None):
        if margin is not None:
            self.margin = float(margin)
        self.collider = soup

    def clear_collider(self):
        self.collider = None

    # ---------------------------------------------------------------- stepping

    def step(self, dt, wind=None, gravity=None, bone_matrices=None,
             collider_verts=None, collider_tris=None, fluid_velocity=None):
        """Advance the cloth by ``dt`` seconds.  Returns the new positions (N, 3)."""
        import time as _time
        t0 = _time.perf_counter()
        dt = float(dt)
        if dt <= 0:
            return self.x
        if bone_matrices is not None:
            self.update_skin_targets(bone_matrices, dt=dt)
        if collider_verts is not None:
            self.set_collider_mesh(collider_verts, collider_tris if collider_tris is not None else np.zeros((0, 3), np.int64))
        g = np.asarray(self.gravity if gravity is None else gravity, float).reshape(3)
        if wind is None:
            wind = self.wind0
        nsub = max(1, int(self.substeps))
        h = dt / nsub
        # pin velocity from target displacement over the full step
        pinned = self.pin_hard
        if pinned.any():
            self.pin_vel[pinned] = (self.pin_target[pinned] - self.x[pinned]) / dt

        self._lam_stretch[:] = 0.0
        self._lam_shear[:] = 0.0
        self._lam_bend[:] = 0.0
        self._lam_hinge[:] = 0.0

        # Hard pins travel with pin_vel during substeps and snap to the exact
        # target only at the end of the full step, so a moving rig imparts inertia
        # smoothly instead of teleporting on the first substep.
        # Untangle teleports / pre-existing intersections while x_prev still
        # holds the last accepted (or rest) pose.  Skip the extra work when
        # motion is already within a collision margin.
        if self.self_collision:
            disp = float(np.max(np.linalg.norm(self.x - self.x_prev, axis=1)))
            if disp > 2.0 * max(self.thickness, 1e-4):
                self._untangle(ref=self.x_prev, passes=12)
                self.x_prev = self.x.copy()
                self.v[:] = 0.0
                if pinned.any():
                    self.v[pinned] = self.pin_vel[pinned]

        for _ in range(nsub):
            self._integrate(h, g, wind, fluid_velocity)
            for _it in range(max(1, int(self.iterations))):
                self._solve_distance(h, self.stretch_edges, self.stretch_rest,
                                     self.stretch_compliance, self._lam_stretch, self._groups_stretch)
                self._solve_distance(h, self.shear_edges, self.shear_rest,
                                     self.shear_compliance, self._lam_shear, self._groups_shear)
                if self.use_isometric_bend and len(self.bend_edges):
                    self._solve_distance(h, self.bend_edges, self.bend_rest,
                                         self.bend_compliance, self._lam_bend, self._groups_bend)
                if self.use_dihedral and len(self.hinges):
                    self._solve_dihedral(h)
                self._solve_soft_pins(h)
                if self.self_collision and self.self_inloop and _it < max(1, int(self.iterations)) - 1:
                    # contacts re-applied between constraint iterations so stretch cannot drag vertices back across
                    self.self_collider().solve(self.x, self.x_prev, self.inv_mass, substep_index=_,
                                               reuse=(_it > 0), passes=1)
            self._collide()
            if self.self_collision:
                self._self_collide(ref=self.x_prev, passes=1, substep_index=_)
            self.v = (self.x - self.x_prev) / h
            damp = max(0.0, 1.0 - self.damping * h)
            self.v *= damp
            if self.vmax > 0:
                sp = np.linalg.norm(self.v, axis=1)
                m = sp > self.vmax
                if m.any():
                    self.v[m] *= (self.vmax / sp[m])[:, None]
            if pinned.any():
                self.v[pinned] = self.pin_vel[pinned]
            self.x_prev = self.x.copy()

        self._apply_hard_pins()
        if pinned.any():
            self.v[pinned] = self.pin_vel[pinned]
        self.t += dt
        self.frame += 1
        self._update_tension()
        self.last_ms = 1000.0 * (_time.perf_counter() - t0)
        return self.x

    def _integrate(self, h, g, wind, fluid_velocity):
        self.x_prev = self.x.copy()
        F = self._aero_forces(wind, fluid_velocity)
        a = F * self.inv_mass[:, None]
        a += g.reshape(1, 3)
        free = self.inv_mass > 0
        self.v[free] = self.v[free] + a[free] * h
        self.x[free] = self.x[free] + self.v[free] * h
        # kinematic pins travel with their target (linear interpolation per substep is
        # achieved by integrating pin_vel; the final substep snaps in _apply_hard_pins)
        if self.pin_hard.any():
            self.x[self.pin_hard] = self.x[self.pin_hard] + self.pin_vel[self.pin_hard] * h

    def _apply_hard_pins(self):
        if self.pin_hard.any():
            self.x[self.pin_hard] = self.pin_target[self.pin_hard]

    def _sample_flow(self, P, wind, fluid_velocity):
        if callable(fluid_velocity):
            flow = np.asarray(fluid_velocity(P), float).reshape(-1, 3)
        elif fluid_velocity is not None:
            flow = np.asarray(fluid_velocity, float).reshape(-1, 3)
            if flow.shape[0] == 1:
                flow = np.broadcast_to(flow, P.shape).copy()
        elif callable(wind):
            flow = np.asarray(wind(P), float).reshape(-1, 3)
        else:
            flow = np.asarray(wind, float).reshape(-1, 3)
            if flow.shape[0] == 1:
                flow = np.broadcast_to(flow, P.shape).copy()
        return flow

    def _aero_forces(self, wind, fluid_velocity):
        """Per-vertex aerodynamic drag & lift from triangle relative velocity (N, 3)."""
        x, v, tris = self.x, self.v, self.tris
        p0, p1, p2 = x[tris[:, 0]], x[tris[:, 1]], x[tris[:, 2]]
        cr = np.cross(p1 - p0, p2 - p0)
        twice = np.linalg.norm(cr, axis=1)
        area = 0.5 * twice
        nrm = cr / np.maximum(twice, 1e-12)[:, None]
        v_tri = (v[tris[:, 0]] + v[tris[:, 1]] + v[tris[:, 2]]) / 3.0
        c_tri = (p0 + p1 + p2) / 3.0
        v_flow = self._sample_flow(c_tri, wind, fluid_velocity)
        v_rel = v_tri - v_flow
        vn = (v_rel * nrm).sum(axis=1)
        rho = self.air_density
        # quadratic pressure (works for both windings) + tangential 'sail' lift
        F = ((-0.5 * rho * self.drag * area * np.abs(vn) * vn)[:, None] * nrm)
        v_t = v_rel - vn[:, None] * nrm
        F += ((-0.5 * rho * self.lift * area * np.abs(vn))[:, None] * v_t)
        Fv = np.zeros_like(x)
        for k in range(3):
            np.add.at(Fv, tris[:, k], F / 3.0)
        return Fv

    def _solve_distance(self, h, edges, rest, compliance, lam, groups):
        """Colour-grouped Gauss-Seidel XPBD distance constraints."""
        if len(edges) == 0:
            return
        alpha = float(compliance) / (h * h)
        x, w = self.x, self.inv_mass
        for g in groups:
            i, j = edges[g, 0], edges[g, 1]
            d = x[j] - x[i]
            L = np.linalg.norm(d, axis=1)
            n = d / np.maximum(L, 1e-12)[:, None]
            C = L - rest[g]
            wsum = w[i] + w[j] + alpha
            dlam = np.zeros(len(g), float)
            live = (L > 1e-12) & (wsum > 0)
            dlam[live] = (-C[live] - alpha * lam[g][live]) / wsum[live]
            lam[g] = lam[g] + dlam
            corr = dlam[:, None] * n
            x[i] = x[i] - w[i, None] * corr
            x[j] = x[j] + w[j, None] * corr

    def _solve_dihedral(self, h):
        """Vectorised Jacobi XPBD dihedral via a rotation-about-hinge gradient.

        The signed flap angle is 0 in the bind (usually flat) pose.  Opposite
        vertices are rotated about the shared edge; an equal and opposite
        correction is applied to the two edge vertices so the constraint is
        momentum-conserving.
        """
        H = self.hinges
        if len(H) == 0:
            return
        alpha = float(self.dihedral_compliance) / (h * h)
        x, w = self.x, self.inv_mass
        rest = self.hinge_rest
        lam = self._lam_hinge
        i, j, k, l = H[:, 0], H[:, 1], H[:, 2], H[:, 3]
        p1, p2, p3, p4 = x[i], x[j], x[k], x[l]
        e = p2 - p1
        e_len = np.linalg.norm(e, axis=1, keepdims=True)
        axis = e / np.maximum(e_len, 1e-12)
        r3 = p3 - p1
        r4 = p4 - p1
        r3p = r3 - (r3 * axis).sum(axis=1, keepdims=True) * axis
        r4p = r4 - (r4 * axis).sum(axis=1, keepdims=True) * axis
        n1 = np.cross(e, p3 - p1)
        n2 = np.cross(p4 - p1, e)
        l1 = np.linalg.norm(n1, axis=1, keepdims=True)
        l2 = np.linalg.norm(n2, axis=1, keepdims=True)
        n1u = n1 / np.maximum(l1, 1e-12)
        n2u = n2 / np.maximum(l2, 1e-12)
        d = np.clip((n1u * n2u).sum(axis=1), -1.0, 1.0)
        sin = (np.cross(n1u, n2u) * axis).sum(axis=1)
        phi = np.arctan2(sin, d)
        C = phi - rest
        r3n = np.linalg.norm(r3p, axis=1)
        r4n = np.linalg.norm(r4p, axis=1)
        good = ((l1[:, 0] > 1e-10) & (l2[:, 0] > 1e-10) & (e_len[:, 0] > 1e-10)
                & (r3n > 1e-10) & (r4n > 1e-10))
        if not good.any():
            return
        q3 = -np.cross(axis, r3p) / np.maximum((r3p * r3p).sum(axis=1, keepdims=True), 1e-12)
        q4 =  np.cross(axis, r4p) / np.maximum((r4p * r4p).sum(axis=1, keepdims=True), 1e-12)
        q1 = -0.5 * (q3 + q4)
        q2 = q1
        wsum = (w[i] * (q1 * q1).sum(1) + w[j] * (q2 * q2).sum(1)
                + w[k] * (q3 * q3).sum(1) + w[l] * (q4 * q4).sum(1) + alpha)
        dlam = np.zeros(len(H), float)
        live = good & (wsum > 1e-12)
        dlam[live] = (-C[live] - alpha * lam[live]) / wsum[live]
        lam += dlam
        s = dlam[:, None]
        dx = np.zeros_like(x)
        np.add.at(dx, i, w[i, None] * s * q1)
        np.add.at(dx, j, w[j, None] * s * q2)
        np.add.at(dx, k, w[k, None] * s * q3)
        np.add.at(dx, l, w[l, None] * s * q4)
        # Jacobi under-relaxation keeps a heavily-connected drape stable
        # while still unfolding a isolated hinge (see tests/test_cloth_solver.py).
        x += 0.3 * dx

    def _solve_soft_pins(self, h):
        idx = np.nonzero(self.pin_soft)[0]
        if len(idx) == 0:
            return
        alpha = self.pin_compliance[idx] / (h * h)
        d = self.x[idx] - self.pin_target[idx]
        # C = |x - target|, gradient = unit(d); treat as a distance-to-point constraint
        L = np.linalg.norm(d, axis=1)
        n = np.zeros_like(d)
        ok = L > 1e-12
        n[ok] = d[ok] / L[ok, None]
        w = self.inv_mass[idx]
        wsum = w + alpha
        if self._lam_pin is None or len(self._lam_pin) != len(idx):
            self._lam_pin = np.zeros(len(idx), float)
        dlam = np.zeros(len(idx), float)
        live = wsum > 0
        dlam[live] = (-L[live] - alpha[live] * self._lam_pin[live]) / wsum[live]
        self._lam_pin = self._lam_pin + dlam
        self.x[idx] = self.x[idx] + w[:, None] * dlam[:, None] * n

    def _collide(self):
        x = self.x
        free = self.inv_mass > 0
        if self.ground_z is not None:
            floor = self.ground_z + self.margin
            hit = free & (x[:, 2] < floor)
            if hit.any():
                x[hit, 2] = floor
                inward = self.v[hit, 2] < 0
                if inward.any():
                    idx = np.nonzero(hit)[0][inward]
                    self.v[idx, 2] = 0.0
                    self.v[idx, :2] *= max(0.0, 1.0 - self.friction)
        if self.collider is None or not free.any():
            return
        pts = x[free]
        cp, sd, _ti = sdf.closest_surface(self.collider, pts)
        margin = self.margin
        hit = sd < margin
        if not hit.any():
            return
        # outward unit normal: (p - cp) aligned with the sign of sd; fall back to the
        # vector from the closest point when the query sits exactly on the surface
        diff = pts[hit] - cp[hit]
        ln = np.linalg.norm(diff, axis=1, keepdims=True)
        n = np.zeros_like(diff)
        ok = ln[:, 0] > 1e-12
        n[ok] = diff[ok] / ln[ok]
        # if we are inside (sd < 0) the vector p-cp already points inward, so flip
        inside = sd[hit] < 0
        n[inside] *= -1.0
        # degenerate: use a vertical fallback
        bad = ~ok
        if bad.any():
            n[bad] = np.array([0.0, 0.0, 1.0])
        pts = pts.copy()
        pts[hit] = cp[hit] + n * margin
        x[free] = pts
        # kill inward velocity and apply simple Coulomb friction in the tangent
        idx = np.nonzero(free)[0][hit]
        vn = (self.v[idx] * n).sum(axis=1)
        inward = vn < 0
        if inward.any():
            self.v[idx[inward]] -= vn[inward, None] * n[inward]
            self.v[idx[inward]] *= max(0.0, 1.0 - self.friction)

    def self_collider(self):
        """The (lazily built) vectorised self-collision state for this mesh."""
        sc = self._selfcol
        if sc is None or len(sc.tris) != len(self.tris):
            sc = SC.SelfCollider(self.tris, self.stretch_edges, self.n)
            if self.self_layers is not None:
                sc.set_layers(self.self_layers, self.self_ignore_pairs)
            self._selfcol = sc
        sc.thickness = max(float(self.self_thickness if self.self_thickness > 0 else self.thickness), 1e-4)
        sc.friction = float(self.self_friction if self.self_friction >= 0 else self.friction)
        sc.mode = 'FAST' if str(self.self_collision_mode).upper() == 'FAST' else 'ACCURATE'
        if sc.rest is None:
            sc.set_rest(self.rest)
        return sc

    def set_self_layers(self, layer_ids, ignore_pairs=()):
        """Per-vertex layer ids (>= 0) with layer pairs that ignore each other; -1 excludes the vertex."""
        self.self_layers = np.asarray(layer_ids, np.int64).copy()
        self.self_ignore_pairs = tuple((int(a), int(b)) for a, b in ignore_pairs)
        if self._selfcol is not None:
            self._selfcol.set_layers(self.self_layers, self.self_ignore_pairs)

    def _self_collide(self, ref=None, passes=1, substep_index=0):
        """Vectorised vertex-vs-triangle self-collision (``core.cloth_selfcol``)."""
        sc = self.self_collider()
        xp = self.x_prev if ref is None else ref
        n = 0
        for _ in range(max(1, int(passes))):
            n += sc.solve(self.x, xp, self.inv_mass, substep_index=substep_index)
        return n

    def min_self_distance(self):
        """Smallest vertex-to-triangle distance between topologically distant parts of the cloth."""
        return self.self_collider().min_distance(self.x, limit=max(self.self_collider().thickness * 1.5, 1e-3))

    def _penetrating_vertices(self, band=None):
        """Free vertices with a negative signed distance to a non-adjacent triangle."""
        thick = max(self.thickness if band is None else float(band), 1e-4)
        # reuse the intersection counter's pair walk via a side channel
        x, w, tris = self.x, self.inv_mass, self.tris
        if len(tris) == 0:
            return np.zeros(0, np.int64)
        hits = self._self_intersection_hits(thick, proximity=False)
        if len(hits) == 0:
            return np.zeros(0, np.int64)
        return np.unique(hits)

    def _untangle(self, ref=None, passes=12):
        """Pull teleported intersecting vertices back toward the last accepted pose, then push out."""
        ref = self.x_prev if ref is None else ref
        band = max(self.thickness, 1e-4)
        for _ in range(max(1, int(passes))):
            idx = self._penetrating_vertices(band)
            if len(idx):
                extra = set(map(int, idx))
                for i in list(extra):
                    extra.update(int(j) for j in self.neighbors[i])
                idx = np.fromiter((i for i in extra if self.inv_mass[i] > 0), np.int64)
                if len(idx):
                    self.x[idx] = self.x[idx] + 0.5 * (ref[idx] - self.x[idx])
            self._self_collide(ref=ref, passes=1)
            if len(self._penetrating_vertices(band)) == 0:
                break

    def _self_intersection_hits(self, thick, proximity=False):
        """Vertex indices of each self-intersection (or proximity) contact; may contain duplicates."""
        x, tris = self.x, self.tris
        sc = self.self_collider()
        v, t = sc._candidates(x, x, thick)
        v, t = sc._filter_topology(v, t)
        v, t = v[self.inv_mass[v] > 0], t[self.inv_mass[v] > 0]
        if len(v) == 0:
            return np.zeros(0, np.int64)
        A, B, C = x[tris[t, 0]], x[tris[t, 1]], x[tris[t, 2]]
        q, _b, _in = SC.closest_point_triangle(x[v], A, B, C)
        d = x[v] - q
        if proximity:
            mask = np.linalg.norm(d, axis=1) < thick
        else:
            nrm = np.cross(B - A, C - A)
            nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1), 1e-12)[:, None]
            mask = ((d * nrm).sum(axis=1) < -1e-6) & _in
        return v[mask]

    def count_self_intersections(self, thickness=None, proximity=False):
        """Self-collision diagnostics.

        By default, counts *penetrations*: free vertices that have gone through a
        non-adjacent triangle (signed distance < 0).  With ``proximity=True``,
        counts every pair closer than ``thickness`` (used to verify the push-out).
        """
        thick = max(self.thickness if thickness is None else float(thickness), 1e-4)
        return int(len(self._self_intersection_hits(thick, proximity=proximity)))

    def _update_tension(self):
        edges, rest = self.stretch_edges, self.stretch_rest
        if len(edges) == 0:
            self.tension[:] = 0.0
            self.last_stretch = 0.0
            return
        d = self.x[edges[:, 1]] - self.x[edges[:, 0]]
        L = np.linalg.norm(d, axis=1)
        strain = (L / np.maximum(rest, 1e-12)) - 1.0
        acc = np.zeros(self.n, float)
        cnt = np.zeros(self.n, float)
        val = np.maximum(strain, 0.0)
        np.add.at(acc, edges[:, 0], val)
        np.add.at(acc, edges[:, 1], val)
        np.add.at(cnt, edges[:, 0], 1.0)
        np.add.at(cnt, edges[:, 1], 1.0)
        self.tension = acc / np.maximum(cnt, 1.0)
        self.last_stretch = float(np.max(strain)) if len(strain) else 0.0

    # ---------------------------------------------------------------- snapshot / package

    def snapshot(self):
        return dict(x=self.x.copy(), v=self.v.copy(), x_prev=self.x_prev.copy(),
                    t=self.t, frame=self.frame, pin_target=self.pin_target.copy(),
                    pin_vel=self.pin_vel.copy())

    def restore(self, snap):
        self.x = snap['x'].copy()
        self.v = snap['v'].copy()
        self.x_prev = snap['x_prev'].copy()
        self.t = float(snap['t'])
        self.frame = int(snap['frame'])
        self.pin_target = snap['pin_target'].copy()
        self.pin_vel = snap['pin_vel'].copy()

    def penetration(self, verts=None):
        """Signed distance of each vertex to the collider (empty array if none)."""
        if self.collider is None:
            return np.zeros(0, float)
        pts = self.x if verts is None else np.asarray(verts, float).reshape(-1, 3)
        _cp, sd, _ti = sdf.closest_surface(self.collider, pts)
        return sd

    def max_penetration(self):
        """How far (metres) any free vertex sits *inside* the collider past the margin.  0 = clean."""
        if self.collider is None:
            return 0.0
        sd = self.penetration()
        free = self.inv_mass > 0
        if not free.any():
            return 0.0
        return float(np.maximum(self.margin - sd[free], 0.0).max())

    def package_dict(self, name="garment", colliders=None, extra=None):
        """Game-engine cloth package (rest mesh, pins, stiffness, aero, collider metadata)."""
        pins = np.nonzero(self.pin_hard | self.pin_soft)[0]
        stretch = 1.0 / max(self.stretch_compliance, 1e-18)
        bend = 1.0 / max(self.bend_compliance, 1e-18)
        pkg = dict(
            format="tessera-cloth-package",
            version=1,
            name=name,
            units="m",
            preset=self.preset,
            quality=self.quality,
            rest_mesh=dict(
                vertices=self.rest.tolist(),
                triangles=self.tris.tolist(),
                n_vertices=int(self.n),
                n_triangles=int(len(self.tris)),
            ),
            pins=dict(
                indices=pins.tolist(),
                weights=(1.0 - (self.pin_compliance[pins] / (self.pin_compliance[pins] + 1.0))).tolist() if len(pins) else [],
                hard=self.pin_hard[pins].astype(int).tolist() if len(pins) else [],
            ),
            constraints=dict(
                stretch_stiffness=stretch,
                shear_stiffness=1.0 / max(self.shear_compliance, 1e-18),
                bend_stiffness=bend,
                stretch_compliance=self.stretch_compliance,
                shear_compliance=self.shear_compliance,
                bend_compliance=self.bend_compliance,
                dihedral_compliance=self.dihedral_compliance,
                damping=self.damping,
                density=self.density,
                thickness=self.thickness,
                collision_margin=self.margin,
                friction=self.friction,
                drag=self.drag,
                lift=self.lift,
                air_density=self.air_density,
                gravity=self.gravity.tolist(),
                substeps=self.substeps,
                iterations=self.iterations,
                self_collision=bool(self.self_collision),
                self_collision_mode=str(self.self_collision_mode),
                self_thickness=float(self.self_thickness),
                vmax=self.vmax,
            ),
            colliders=list(colliders or []),
            engine_hints=dict(
                unreal_chaos=dict(
                    stretch_stiffness=float(np.clip(stretch / _STRETCH_REF, 0.0, 1.0)),
                    shear_stiffness=float(np.clip((1.0 / max(self.shear_compliance, 1e-18)) / _STRETCH_REF, 0.0, 1.0)),
                    bend_stiffness=float(np.clip(bend / _BEND_REF, 0.0, 1.0)),
                    damping=float(np.clip(self.damping, 0.0, 1.0)),
                    collision_thickness=self.thickness,
                    anim_drive_stiffness=float(self.anim_drive),
                    drag=self.drag,
                    lift=self.lift,
                    density=self.density,
                ),
                unity=dict(
                    stretching_stiffness=float(np.clip(stretch / _STRETCH_REF, 0.0, 1.0)),
                    bending_stiffness=float(np.clip(bend / _BEND_REF, 0.0, 1.0)),
                    damping=float(self.damping),
                    world_acceleration_scale=1.0,
                    world_velocity_scale=1.0,
                    friction=self.friction,
                    collision_mass_scale=0.0,
                    use_virtual_particles=True,
                    thickness=self.thickness,
                ),
                web=dict(
                    stretch=stretch, bend=bend, damping=self.damping,
                    drag=self.drag, lift=self.lift, density=self.density,
                    thickness=self.thickness, margin=self.margin,
                ),
            ),
        )
        if extra:
            pkg.update(extra)
        return pkg


def _edge_lengths(verts, edges):
    if len(edges) == 0:
        return np.zeros(0, float)
    d = verts[edges[:, 1]] - verts[edges[:, 0]]
    return np.linalg.norm(d, axis=1)


def _hinge_angles(verts, hinges):
    if len(hinges) == 0:
        return np.zeros(0, float)
    i, j, k, l = hinges[:, 0], hinges[:, 1], hinges[:, 2], hinges[:, 3]
    p1, p2, p3, p4 = verts[i], verts[j], verts[k], verts[l]
    e = p2 - p1
    n1 = np.cross(e, p3 - p1)
    n2 = np.cross(p4 - p1, e)
    l1 = np.maximum(np.linalg.norm(n1, axis=1, keepdims=True), 1e-12)
    l2 = np.maximum(np.linalg.norm(n2, axis=1, keepdims=True), 1e-12)
    e_len = np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
    n1u, n2u = n1 / l1, n2 / l2
    d = np.clip((n1u * n2u).sum(axis=1), -1.0, 1.0)
    sin = (np.cross(n1u, n2u) * (e / e_len)).sum(axis=1)
    return np.arctan2(sin, d)


def apply_preset(solver, preset):
    """Retarget stiffness / mass / aero of an existing solver to a named preset."""
    if preset not in PRESETS:
        raise KeyError(preset)
    pr = PRESETS[preset]
    solver.preset = preset
    solver.density = float(pr['density'])
    solver.thickness = float(pr['thickness'])
    solver.margin = float(pr['margin'])
    solver.friction = float(pr['friction'])
    solver.drag = float(pr['drag'])
    solver.lift = float(pr['lift'])
    solver.damping = float(pr['damping'])
    areas = vertex_areas(solver.rest, solver.tris)
    solver.mass = solver.density * np.maximum(areas, 1e-8)
    solver.inv_mass = 1.0 / solver.mass
    solver.inv_mass[solver.pin_hard] = 0.0
    solver.stretch_compliance = _compliance(pr['stretch'])
    solver.shear_compliance = _compliance(pr['shear'])
    solver.bend_compliance = _compliance(pr['bend'])
    solver.dihedral_compliance = _compliance(max(pr['bend'] * 0.35, 1e-6))
    return solver
