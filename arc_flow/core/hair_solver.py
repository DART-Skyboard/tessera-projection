"""Real-time hair-curve physics (XPBD / Verlet). Pure NumPy, no bpy.

Tessera Projection Hair drives Blender Empty Hair Curves (and any packed
polyline groom) with inextensible strands, rest-pose / curl memory,
root-attachment inertia, anisotropic aerodynamic coupling, and
tessellation-aware SDF collision against LOTBCBP PartFields
(`arc_flow.core.sdf`).

A groom is a packed point array plus curve offsets (the same layout
Blender's `Curves.curve_offset_data` uses). Guide strands are simulated;
child strands are skinned to the k nearest guides and stay cheap enough
for a live 60 fps preview. Root transforms (object / bone / deforming
scalp) inject linear and angular velocity so a posed or controller-driven
rig moves the hair with the character.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import sdf as SDF
from . import primitives as MS


PACKAGE_FORMAT = "tessera-hair-physics-v1"
PACKAGE_VERSION = 1
_EPS = 1e-12


# ----------------------------------------------------------------------------- parameters


class HairParams:
    """Tunable solver knobs. Defaults target a 60 fps game-engine preview."""

    __slots__ = (
        "dt", "substeps", "gravity", "damping",
        "stretch_compliance", "stretch_iterations",
        "bend_compliance", "bend_iterations",
        "rest_shape", "rest_shape_root", "rest_shape_tip",
        "self_collision", "collision_radius", "collision_iterations", "collision_friction",
        "sdf_resolution", "sdf_level", "sdf_refine",
        "wind_drag", "wind_tangent", "fluid_coupling",
        "guide_ratio", "child_k", "child_smooth",
        "pinned_prefix", "ground_z", "max_segment_budget", "max_points",
        "gravity_scale", "bend_tip_scale", "stiffness_curve", "max_speed",
        "hair_collision", "strand_radius", "repel_strength", "volume", "clumping",
        "clump_range", "strand_friction", "max_pairs", "adaptive_substeps", "hair_every",
    )

    def __init__(self, **kw):
        self.dt = 1.0 / 24.0
        self.substeps = 4
        self.gravity = np.array((0.0, 0.0, -9.81), dtype=np.float64)
        self.damping = 0.92
        self.stretch_compliance = 0.0
        self.stretch_iterations = 8
        self.bend_compliance = 2e-4
        self.bend_iterations = 4
        self.rest_shape = 0.35
        self.rest_shape_root = 0.85
        self.rest_shape_tip = 0.12
        self.self_collision = True
        self.collision_radius = 0.008
        self.collision_iterations = 2
        self.collision_friction = 0.3
        self.sdf_resolution = 32
        self.sdf_level = 2
        self.sdf_refine = True
        self.wind_drag = 0.8
        self.wind_tangent = 0.15
        self.fluid_coupling = 1.0
        self.guide_ratio = 0.25
        self.child_k = 3
        self.child_smooth = 1.0
        self.pinned_prefix = 1
        self.ground_z = None
        self.max_segment_budget = 0
        self.max_points = 0
        # strand-scale gravity / stiffness profile
        self.gravity_scale = 1.0       # global multiplier on gravity (per strand: set_per_strand(gravity=))
        self.bend_tip_scale = 3.0      # bend compliance at the tip relative to the root (floppier tips)
        self.stiffness_curve = 1.0     # exponent of the root-to-tip stiffness ramp (>1 keeps the root stiff longer)
        self.max_speed = 30.0          # m/s clamp (stability under a hard swat)
        # hair-hair interaction (spatial hash, vectorised)
        self.hair_collision = True
        self.strand_radius = 0.004     # fibre radius used for hair-hair contact (m)
        self.repel_strength = 0.5      # 0..1 how much overlap is removed per substep
        self.volume = 0.0              # >0 inflates the hair-hair contact radius (puffier groom)
        self.clumping = 0.0            # >0 pulls neighbouring strands together (locks)
        self.clump_range = 0.012       # reach of the clumping attraction beyond contact (m)
        self.strand_friction = 0.05    # velocity sharing between touching strands
        self.max_pairs = 600000
        self.adaptive_substeps = True  # add substeps when a collider moves fast
        self.hair_every = 1            # strand-strand contact runs on every N-th substep (2 = half the cost on dense grooms)
        self.update(**kw)

    def update(self, **kw):
        for k, v in kw.items():
            if k not in self.__slots__:
                raise AttributeError("unknown HairParams field %r" % k)
            if k == "gravity":
                v = np.asarray(v, dtype=np.float64).reshape(3)
            setattr(self, k, v)
        return self

    def as_dict(self):
        d = {}
        for k in self.__slots__:
            v = getattr(self, k)
            if isinstance(v, np.ndarray):
                v = v.tolist()
            d[k] = v
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__slots__})


# ----------------------------------------------------------------------------- packing / topology helpers


def pack_strands(strands):
    """List of (n_i, 3) arrays -> (positions (N,3), offsets (S+1,))."""
    chunks = [np.asarray(s, dtype=np.float64).reshape(-1, 3) for s in strands if len(s)]
    if not chunks:
        return np.zeros((0, 3), np.float64), np.zeros(1, np.int64)
    counts = np.array([len(s) for s in chunks], np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return np.vstack(chunks), offsets


def unpack_strands(positions, offsets):
    P = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    off = np.asarray(offsets, dtype=np.int64).ravel()
    return [P[off[i]:off[i + 1]].copy() for i in range(len(off) - 1)]


def strand_counts(offsets):
    return np.diff(np.asarray(offsets, dtype=np.int64).ravel())


def _as_M44(M):
    M = np.asarray(M, dtype=np.float64)
    if M.shape == (4, 4):
        return M
    if M.ndim == 3 and M.shape[-2:] == (4, 4):
        return M
    raise ValueError("expected a 4x4 matrix or an (S,4,4) stack, got %s" % (M.shape,))


def xform_points(P, M):
    """Apply a single 4x4 matrix to (N,3) points."""
    M = _as_M44(M)
    return P @ M[:3, :3].T + M[:3, 3]


def inv_xform_points(P, M):
    M = _as_M44(M)
    R, t = M[:3, :3], M[:3, 3]
    try:
        Rinv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        Rinv = R.T
    return (P - t) @ Rinv.T


def xform_per_strand(P, Ms, strand_id):
    """Apply per-strand 4x4 matrices to packed points."""
    Ms = _as_M44(Ms)
    R = Ms[strand_id, :3, :3]
    t = Ms[strand_id, :3, 3]
    return np.einsum("nij,nj->ni", R, P) + t


def invert_M(Ms):
    Ms = _as_M44(Ms)
    single = Ms.ndim == 2
    if single:
        Ms = Ms[None]
    R, t = Ms[:, :3, :3], Ms[:, :3, 3]
    try:
        Rinv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        Rinv = np.transpose(R, (0, 2, 1))
    out = np.tile(np.eye(4), (len(Ms), 1, 1))
    out[:, :3, :3] = Rinv
    out[:, :3, 3] = -np.einsum("nij,nj->ni", Rinv, t)
    return out[0] if single else out


def rigid_velocity(M_prev, M_curr, dt, points):
    """Linear + angular velocity of a rigid frame, sampled at `points` (world)."""
    dt = max(float(dt), 1e-8)
    Mp, Mc = _as_M44(M_prev), _as_M44(M_curr)
    v_lin = (Mc[:3, 3] - Mp[:3, 3]) / dt
    dR = Mc[:3, :3] @ Mp[:3, :3].T
    # rotation vector from the skew-symmetric part of (R - R.T) / 2
    skew = 0.5 * (dR - dR.T)
    omega = np.array((skew[2, 1], skew[0, 2], skew[1, 0])) / dt
    r = np.asarray(points, dtype=np.float64).reshape(-1, 3) - Mc[:3, 3]
    return v_lin + np.cross(omega, r)


def _normalize(v, fallback=None):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    if fallback is None:
        fallback = np.zeros(v.shape[-1], dtype=v.dtype)
        fallback[0] = 1.0
    return np.where(n > _EPS, v / np.maximum(n, _EPS), fallback)


def _triangle_frames(A, B, C):
    """Orthonormal frames at the triangle centroids: X along AB, Z = normal."""
    x = _normalize(B - A)
    n = _normalize(np.cross(B - A, C - A), fallback=np.array((0.0, 0.0, 1.0)))
    y = _normalize(np.cross(n, x))
    z = _normalize(np.cross(x, y))
    return x, y, z


def _barycentric(P, A, B, C):
    v0, v1, v2 = B - A, C - A, P - A
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    denom = np.maximum(d00 * d11 - d01 * d01, _EPS)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return u, v, w


# ----------------------------------------------------------------------------- groom builders (tests + example scene)


def fibonacci_hemisphere(n, radius=1.0, up=(0.0, 0.0, 1.0), seed=0):
    """Evenly distributed points on the upper hemisphere (scalp roots)."""
    n = max(int(n), 1)
    i = np.arange(n, dtype=np.float64) + 0.5
    # golden-spiral on a hemisphere (z in [0, 1])
    z = 1.0 - i / n
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    theta = np.pi * (1.0 + 5 ** 0.5) * i
    P = np.stack((r * np.cos(theta), r * np.sin(theta), z), axis=1) * float(radius)
    up = np.asarray(up, dtype=np.float64)
    # default already +Z; rotate if a different up is requested
    if abs(up[2] - 1.0) > 1e-9 or abs(up[0]) + abs(up[1]) > 1e-9:
        zaxis = _normalize(up.reshape(1, 3))[0]
        xaxis = _normalize(np.cross(np.array((0.0, 1.0, 0.0) if abs(zaxis[1]) < 0.9 else (1.0, 0.0, 0.0)), zaxis))
        yaxis = np.cross(zaxis, xaxis)
        R = np.stack((xaxis, yaxis, zaxis), axis=1)
        P = P @ R.T
    return P


def make_strand(root, direction, length, n_pts, curl=0.0, curl_axis=None, sag=0.35, gravity_dir=(0.0, 0.0, -1.0)):
    """A single styled polyline: leaves along `direction`, then sags toward gravity, optional curl."""
    n_pts = max(int(n_pts), 2)
    root = np.asarray(root, dtype=np.float64).reshape(3)
    d0 = _normalize(np.asarray(direction, dtype=np.float64).reshape(1, 3))[0]
    g = _normalize(np.asarray(gravity_dir, dtype=np.float64).reshape(1, 3))[0]
    t = np.linspace(0.0, 1.0, n_pts)
    # blend leaving-normal -> hanging
    dirs = _normalize((1.0 - sag * t)[:, None] * d0 + (sag * t)[:, None] * g)
    step = float(length) / (n_pts - 1)
    pts = np.empty((n_pts, 3), np.float64)
    pts[0] = root
    if curl_axis is None:
        curl_axis = _normalize(np.cross(d0, g).reshape(1, 3), fallback=np.array((0.0, 1.0, 0.0)))[0]
    ca, sa = np.cos(curl * t), np.sin(curl * t)
    for i in range(1, n_pts):
        di = dirs[i]
        # Rodrigues rotation about curl_axis
        k = curl_axis
        rot = di * ca[i] + np.cross(k, di) * sa[i] + k * np.dot(k, di) * (1.0 - ca[i])
        pts[i] = pts[i - 1] + _normalize(rot.reshape(1, 3))[0] * step
    return pts


def make_hemisphere_groom(n_strands=80, n_pts=10, radius=0.12, length=0.28, center=(0, 0, 1.6),
                          curl=0.6, sag=0.55, seed=0, length_jitter=0.12):
    """A spherical-scalp groom: packed (positions, offsets). Deterministic for a given seed."""
    rng = np.random.RandomState(int(seed))
    center = np.asarray(center, dtype=np.float64).reshape(3)
    roots = fibonacci_hemisphere(n_strands, radius=radius) + center
    normals = _normalize(roots - center)
    lengths = float(length) * (1.0 + length_jitter * (rng.rand(n_strands) * 2.0 - 1.0))
    curls = float(curl) * (0.4 + 1.2 * rng.rand(n_strands))
    signs = np.where(rng.rand(n_strands) < 0.5, -1.0, 1.0)
    strands = [make_strand(roots[i], normals[i], lengths[i], n_pts, curl=signs[i] * curls[i], sag=sag)
               for i in range(n_strands)]
    return pack_strands(strands)


def strand_frames(P):
    """Parallel-transport-ish normal / binormal for every point of one polyline (n,3) -> (N, B) each (n,3)."""
    P = np.asarray(P, dtype=np.float64)
    T = np.zeros_like(P)
    T[:-1] = P[1:] - P[:-1]
    T[-1] = T[-2] if len(P) > 1 else (0.0, 0.0, -1.0)
    T = _normalize(T, fallback=np.array((0.0, 0.0, -1.0)))
    ref = np.array((0.0, 1.0, 0.0)) if abs(T[0, 1]) < 0.9 else np.array((1.0, 0.0, 0.0))
    N = np.empty_like(T)
    B = np.empty_like(T)
    n = np.cross(T[0], ref)
    n = n / max(np.linalg.norm(n), _EPS)
    for i in range(len(P)):
        n = n - T[i] * np.dot(n, T[i])
        nn = np.linalg.norm(n)
        n = n / nn if nn > _EPS else np.cross(T[i], ref)
        N[i] = n
        B[i] = np.cross(T[i], n)
    return N, B


def generate_children(positions, offsets, per_guide=8, spread=0.012, clump=0.55, clump_power=1.6,
                      frizz=0.0, frizz_freq=5.0, tip_spread=0.0, length_jitter=0.08, seed=0):
    """Surround every guide strand with child strands (cheap grooming: root scatter, tip clumping, frizz, flyaways).

    Child k of a guide starts `spread` metres around the root in the strand's normal plane; `clump`
    (0..1) pulls the child back onto the guide toward the tip (1 = one tight lock), `tip_spread` fans the
    tips out again, `frizz` adds a smooth random wiggle that grows toward the tip. Returns packed
    (positions, offsets) of the *children only* (guides first in the caller's array, then these).
    """
    rng = np.random.RandomState(int(seed))
    P = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    off = np.asarray(offsets, dtype=np.int64).ravel()
    out = []
    for si in range(len(off) - 1):
        g = P[off[si]:off[si + 1]]
        n = len(g)
        if n < 2:
            continue
        N, B = strand_frames(g)
        t = np.linspace(0.0, 1.0, n)
        for _k in range(int(per_guide)):
            ang = rng.rand() * 2.0 * np.pi
            rad = spread * np.sqrt(rng.rand())
            base = np.cos(ang) * N + np.sin(ang) * B                          # (n,3) direction per point
            # offset profile: scatter at the root, converge (clump), then optional tip fan-out
            prof = rad * (1.0 - clump * t ** clump_power) + tip_spread * rng.rand() * t ** 2
            prof[0] = rad * 0.35
            pts = g + base * prof[:, None]
            if frizz > 0.0:
                ph1, ph2 = rng.rand(2) * 2 * np.pi
                f = frizz * (t ** 1.4)
                pts += N * (np.sin(t * frizz_freq * 2 * np.pi + ph1) * f)[:, None] \
                    + B * (np.sin(t * frizz_freq * 1.7 * 2 * np.pi + ph2) * f)[:, None]
            if length_jitter > 0.0:
                k = 1.0 + length_jitter * (rng.rand() * 2.0 - 1.0)
                pts = pts[0] + (pts - pts[0]) * k
            out.append(pts)
    if not out:
        return np.zeros((0, 3), np.float64), np.zeros(1, np.int64)
    return pack_strands(out)


def resample_strand(pts, n_pts):
    """Uniform arc-length resample of a polyline, keeping the root and tip."""
    P = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    n_pts = max(int(n_pts), 2)
    if len(P) == n_pts:
        return P.copy()
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(arc[-1])
    if total < _EPS:
        return np.repeat(P[:1], n_pts, axis=0)
    s = np.linspace(0.0, total, n_pts)
    out = np.empty((n_pts, 3), np.float64)
    for k in range(3):
        out[:, k] = np.interp(s, arc, P[:, k])
    return out


def apply_segment_budget(positions, offsets, max_segments=0, max_points=0):
    """Downsample every strand so no strand exceeds `max_segments` points and the groom stays under `max_points`."""
    P = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    off = np.asarray(offsets, dtype=np.int64).ravel()
    counts = np.diff(off)
    n_strands = len(counts)
    cap = int(max_segments) if max_segments and max_segments > 0 else int(counts.max() if len(counts) else 2)
    cap = max(cap, 2)
    if max_points and max_points > 0 and n_strands * cap > max_points:
        cap = max(2, int(max_points) // max(n_strands, 1))
    if np.all(counts <= cap) and (not max_points or len(P) <= max_points):
        return P.copy(), off.copy()
    strands = [resample_strand(P[off[i]:off[i + 1]], min(int(counts[i]), cap)) for i in range(n_strands)]
    return pack_strands(strands)


def select_guides(n_strands, ratio=0.25, seed=0, min_guides=1):
    """Deterministic, evenly spaced guide indices (stable sort, no hash randomisation)."""
    n = int(n_strands)
    if n <= 0:
        return np.zeros(0, np.int64)
    k = max(int(min_guides), int(np.ceil(n * float(ratio))))
    k = min(k, n)
    # even spacing, offset by a tiny seed so different grooms can pick a different phase
    phase = (int(seed) % max(n, 1)) / max(n, 1)
    u = (np.arange(k, dtype=np.float64) + phase) / k
    idx = np.floor(u * n).astype(np.int64) % n
    # unique, sorted for determinism; fill any collision from leftover indices
    seen = []
    used = np.zeros(n, bool)
    for i in idx:
        if not used[i]:
            used[i] = True
            seen.append(int(i))
    if len(seen) < k:
        rest = np.flatnonzero(~used)
        need = k - len(seen)
        seen.extend(rest[:need].tolist())
    return np.array(sorted(seen), dtype=np.int64)


def child_skin_weights(child_roots, guide_roots, k=3, smooth=1.0):
    """k-nearest guide weights for each child root. Ties broken by smaller guide index.

    Returns (indices (C,k), weights (C,k)) with rows summing to 1.
    """
    C = np.asarray(child_roots, dtype=np.float64).reshape(-1, 3)
    G = np.asarray(guide_roots, dtype=np.float64).reshape(-1, 3)
    k = max(1, min(int(k), len(G)))
    if len(C) == 0 or len(G) == 0:
        return np.zeros((len(C), k), np.int64), np.zeros((len(C), k), np.float64)
    # (C,G) squared distances; add a tiny index jitter so ties are stable
    d2 = np.sum((C[:, None, :] - G[None, :, :]) ** 2, axis=2)
    d2 = d2 + np.arange(len(G), dtype=np.float64)[None, :] * 1e-18
    idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
    # sort the k neighbours of each child by distance (then index)
    row = np.arange(len(C))[:, None]
    part = d2[row, idx]
    order = np.argsort(part, axis=1)
    idx = idx[row, order]
    dist = np.sqrt(np.maximum(d2[row, idx], 0.0))
    # inverse-distance weights; identical roots get a one-hot
    power = max(float(smooth), 1e-6)
    w = 1.0 / np.maximum(dist, 1e-8) ** power
    zero = dist[:, 0] < 1e-10
    w[zero] = 0.0
    w[zero, 0] = 1.0
    w = w / np.maximum(w.sum(axis=1, keepdims=True), _EPS)
    return idx.astype(np.int64), w


def sample_strands_at(positions, offsets, s):
    """Sample packed strands at a normalised parameter s in [0,1] (0 = root, 1 = tip).

    `s` is a scalar or an (N,) array broadcast against the strands.
    """
    off = np.asarray(offsets, dtype=np.int64).ravel()
    P = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    n = len(off) - 1
    s = np.asarray(s, dtype=np.float64)
    if s.ndim == 0:
        s = np.full(n, float(s))
    s = np.clip(s, 0.0, 1.0)
    counts = np.diff(off)
    # arc-length parameter
    out = np.empty((n, 3), np.float64)
    for i in range(n):
        a, b = int(off[i]), int(off[i + 1])
        pts = P[a:b]
        if b - a <= 1:
            out[i] = pts[0]
            continue
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(arc[-1])
        if total < _EPS:
            out[i] = pts[0]
            continue
        out[i] = [np.interp(s[i] * total, arc, pts[:, k]) for k in range(3)]
    return out


def interpolate_children(guide_pos, guide_off, child_rest, child_off, guide_rest=None,
                         indices=None, weights=None, k=3, smooth=1.0):
    """Skin child rest curves to the current guide deformation (same-s sample + weighted delta).

    At t=0 (guide_pos == guide_rest) this returns `child_rest` exactly.
    """
    gP = np.asarray(guide_pos, dtype=np.float64).reshape(-1, 3)
    gOff = np.asarray(guide_off, dtype=np.int64).ravel()
    cP = np.asarray(child_rest, dtype=np.float64).reshape(-1, 3)
    cOff = np.asarray(child_off, dtype=np.int64).ravel()
    if guide_rest is None:
        guide_rest = gP
    gR = np.asarray(guide_rest, dtype=np.float64).reshape(-1, 3)
    g_roots_r = gR[gOff[:-1]]
    c_roots = cP[cOff[:-1]]
    if indices is None or weights is None:
        indices, weights = child_skin_weights(c_roots, g_roots_r, k=k, smooth=smooth)
    out = cP.copy()
    # per-child parametric coordinate of each point, then deform by weighted guide deltas
    for ci in range(len(cOff) - 1):
        a, b = int(cOff[ci]), int(cOff[ci + 1])
        pts = cP[a:b]
        if b - a <= 1:
            out[a:b] = pts
            continue
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(arc[-1]) if float(arc[-1]) > _EPS else 1.0
        s = arc / total
        acc = np.zeros_like(pts)
        wsum = 0.0
        for m, w in zip(indices[ci], weights[ci]):
            g_curr = sample_strands_at(gP, gOff, s)[m] if False else _sample_one(gP, gOff, int(m), s)
            g_bind = _sample_one(gR, gOff, int(m), s)
            acc += w * (g_curr - g_bind)
            wsum += w
        out[a:b] = pts + acc
    return out


def _sample_one(P, off, si, s):
    a, b = int(off[si]), int(off[si + 1])
    pts = P[a:b]
    s = np.asarray(s, dtype=np.float64)
    if b - a <= 1:
        return np.repeat(pts[:1], len(s), axis=0) if s.ndim else pts[0]
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(arc[-1])
    if total < _EPS:
        base = pts[0]
        return np.repeat(base.reshape(1, 3), len(s), axis=0) if s.ndim else base
    tgt = np.clip(s, 0.0, 1.0) * total
    if tgt.ndim == 0:
        return np.array([np.interp(float(tgt), arc, pts[:, k]) for k in range(3)])
    return np.stack([np.interp(tgt, arc, pts[:, k]) for k in range(3)], axis=1)


# ----------------------------------------------------------------------------- SDF collision


def make_collider(verts, tris, resolution=32, level=2, mode="SOLID", adaptive=False):
    """Build a LOTBCBP `PartField` for a hair collider (head, body, environment)."""
    return SDF.PartField(np.asarray(verts, float), np.asarray(tris, int),
                         mode=mode, resolution=int(resolution), level=int(level), adaptive=adaptive)


def collider_from_primitive(kind="sphere", radius=0.12, center=(0, 0, 0), resolution=32, level=2, **kw):
    if kind == "sphere":
        v, t = MS.uvsphere(n=kw.get("n", 24), m=kw.get("m", 12), r=radius, center=center)
    elif kind == "box":
        sx = kw.get("sx", radius * 2); sy = kw.get("sy", radius * 2); sz = kw.get("sz", radius * 2)
        v, t = MS.box(sx, sy, sz, center=center)
    else:
        raise ValueError("unknown primitive %r" % kind)
    return make_collider(v, t, resolution=resolution, level=level, mode=kw.get("mode", "SOLID"))


def project_points_sdf(P, field, matrix=None, radius=0.008, refine=True, iterations=1, friction=0.0, prev=None):
    """Push world-space points out of a PartField's solid (sd < radius). Returns (P, n_hits, normals).

    `matrix` is the collider's world transform (field is in local / bind space). When `prev`
    is given, inward velocity along the contact normal is cancelled (optional friction) and a
    4-tuple (P, n_hits, normals, prev) is returned. Points far from the collider's bounding box
    are skipped (cheap broad phase), so a small glove costs almost nothing for distant hair.
    """
    X = np.asarray(P, dtype=np.float64).reshape(-1, 3).copy()
    if len(X) == 0:
        return X, 0, np.zeros((0, 3), np.float64)
    M = np.eye(4) if matrix is None else _as_M44(matrix)
    hits = 0
    n_world = np.zeros_like(X)
    h = float(getattr(field, "h", 0.02) or 0.02)
    rad = float(radius)
    eye3 = np.eye(3)
    bmin = getattr(field, "bmin", None)
    bmax = getattr(field, "bmax", None)
    # world-space scale of the local -> world map (broad phase margin stays in local units)
    sc = float(np.cbrt(abs(np.linalg.det(M[:3, :3])))) or 1.0
    pad = (rad + 3.0 * h) / max(sc, 1e-6)
    for _ in range(max(int(iterations), 1)):
        Xl_all = inv_xform_points(X, M)
        if bmin is not None and bmax is not None:
            near = np.all((Xl_all >= np.asarray(bmin) - pad) & (Xl_all <= np.asarray(bmax) + pad), axis=1)
            idx_near = np.flatnonzero(near)
            if len(idx_near) == 0:
                break
        else:
            idx_near = np.arange(len(X))
        Xl = Xl_all[idx_near]
        if refine and hasattr(field, "request_refine"):
            band = rad / max(sc, 1e-6) + 2.5 * h
            try:
                sd0 = field.sample_sd(Xl, bricks=True)
                cand = Xl[sd0 < band]
                if len(cand):
                    field.request_refine(cand, band=band, max_new=96)
            except Exception:
                pass
        sd = field.sample_sd(Xl, bricks=True)
        inside = sd < rad / max(sc, 1e-6)
        if not np.any(inside):
            break
        hits = max(hits, int(inside.sum()))
        # finite-difference gradient only on penetrating points
        sel = np.flatnonzero(inside)
        Psel = Xl[sel]
        eps = max(0.35 * h, 1e-5)
        g = np.empty((len(sel), 3), dtype=np.float64)
        for ax in range(3):
            e = eye3[ax] * eps
            g[:, ax] = (field.sample_sd(Psel + e, bricks=True) - field.sample_sd(Psel - e, bricks=True)) / (2.0 * eps)
        nloc = _normalize(g, fallback=np.array((0.0, 0.0, 1.0)))
        Xl_new = Psel + nloc * (rad / max(sc, 1e-6) - sd[sel])[:, None]
        gsel = idx_near[sel]
        X[gsel] = xform_points(Xl_new, M)
        n_world[gsel] = _normalize(nloc @ M[:3, :3].T)
    if prev is not None and hits:
        prev = np.asarray(prev, dtype=np.float64).reshape(-1, 3).copy()
        vn = np.einsum("ij,ij->i", X - prev, n_world)[:, None]
        # cancel inward (and optionally some tangent for friction)
        inward = (vn[:, 0] < 0.0) & (np.linalg.norm(n_world, axis=1) > 0.5)
        if np.any(inward):
            prev[inward] = prev[inward] + n_world[inward] * vn[inward]
            if friction:
                v = X[inward] - prev[inward]
                vt = v - n_world[inward] * np.einsum("ij,ij->i", v, n_world[inward])[:, None]
                prev[inward] = prev[inward] + float(friction) * vt
        return X, hits, n_world, prev
    return X, hits, n_world


def lerp_matrix(Ma, Mb, t):
    """Linear blend of two 4x4 rigid frames (fine for the small per-frame motion of a collider)."""
    Ma, Mb = _as_M44(Ma), _as_M44(Mb)
    return Ma + (Mb - Ma) * float(t)


def hair_hair_pairs(x, cell, strand_id, reach, max_pairs=600000):
    """Spatial-hash broad phase. Returns (i, j, d, n) for point pairs of *different* strands closer than `reach`.

    Fully vectorised (sort + searchsorted over the 14 half-space neighbour cells); `n` is the unit vector i -> j.
    """
    N = len(x)
    if N < 2:
        z = np.zeros(0, np.int64)
        return z, z, np.zeros(0), np.zeros((0, 3))
    cell = max(float(cell), 1e-6)
    c = np.floor((x - x.min(axis=0)) / cell).astype(np.int64) + 1
    dims = c.max(axis=0) + 3
    key = (c[:, 0] * dims[1] + c[:, 1]) * dims[2] + c[:, 2]
    order = np.argsort(key, kind="stable")
    ks = key[order]
    half = [(0, 0, 0)]
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if (dx, dy, dz) > (0, 0, 0):
                    half.append((dx, dy, dz))
    I, J = [], []
    total = 0
    for dx, dy, dz in half:
        q = ((c[:, 0] + dx) * dims[1] + (c[:, 1] + dy)) * dims[2] + (c[:, 2] + dz)
        lo = np.searchsorted(ks, q, side="left")
        hi = np.searchsorted(ks, q, side="right")
        cnt = hi - lo
        tot = int(cnt.sum())
        if tot == 0:
            continue
        if total + tot > max_pairs * 4:
            break
        ii = np.repeat(np.arange(N), cnt)
        start = np.repeat(lo - (np.cumsum(cnt) - cnt), cnt)
        jj = order[start + np.arange(tot)]
        if (dx, dy, dz) == (0, 0, 0):
            keep = jj > ii
            ii, jj = ii[keep], jj[keep]
        I.append(ii)
        J.append(jj)
        total += len(ii)
    if not I:
        z = np.zeros(0, np.int64)
        return z, z, np.zeros(0), np.zeros((0, 3))
    i = np.concatenate(I)
    j = np.concatenate(J)
    keep = strand_id[i] != strand_id[j]
    i, j = i[keep], j[keep]
    dv = x[j] - x[i]
    d = np.linalg.norm(dv, axis=1)
    keep = d < reach
    i, j, d, dv = i[keep], j[keep], d[keep], dv[keep]
    if len(i) > max_pairs:
        sel = np.argsort(d)[:max_pairs]
        i, j, d, dv = i[sel], j[sel], d[sel], dv[sel]
    n = dv / np.maximum(d, 1e-9)[:, None]
    return i, j, d, n


# ----------------------------------------------------------------------------- solver


class HairSolver:
    """Packed-strand XPBD / Verlet hair solver.

    Simulate with `step()`. Roots are pinned to the current attachment
    transform (object, bone, or deforming scalp). Guide strands carry the
    dynamics; `child_positions()` skins any attached children.
    """

    def __init__(self, positions, offsets, params=None, rest=None, radii=None):
        self.params = params if isinstance(params, HairParams) else HairParams(**(params or {}))
        P, off = apply_segment_budget(positions, offsets,
                                      self.params.max_segment_budget, self.params.max_points)
        self.positions = np.asarray(P, dtype=np.float64).reshape(-1, 3).copy()
        self.offsets = np.asarray(off, dtype=np.int64).ravel().copy()
        self.rest = (np.asarray(rest, dtype=np.float64).reshape(-1, 3).copy()
                     if rest is not None else self.positions.copy())
        if len(self.rest) != len(self.positions):
            raise ValueError("rest pose length %d != %d points" % (len(self.rest), len(self.positions)))
        self.radii = (np.asarray(radii, dtype=np.float64).reshape(-1)
                      if radii is not None else np.full(len(self.positions), 0.004))
        if len(self.radii) != len(self.positions):
            self.radii = np.full(len(self.positions), float(np.mean(self.radii) if len(self.radii) else 0.004))
        self.prev = self.positions.copy()
        self.n_strands = max(len(self.offsets) - 1, 0)
        self.n_points = len(self.positions)
        self.dt_used = float(self.params.dt)
        self._build_topology()
        self._capture_rest_metrics()
        # attachments
        self.root_M = np.tile(np.eye(4), (max(self.n_strands, 1), 1, 1))
        self.root_M_prev = self.root_M.copy()
        self.bind_M = None
        self.rest_local = None
        self._surface_bind = None
        # forces / collision
        self.colliders = []          # list of dict(field, matrix, radius)
        self.wind = np.zeros(3, np.float64)   # uniform, or replaced by a callable
        self.wind_fn = None
        self.fluid_fn = None
        # guides / children
        self.guide_idx = np.arange(self.n_strands, dtype=np.int64)
        self.child_rest = None
        self.child_off = None
        self.child_idx = None
        self.child_w = None
        self._stiffness = np.ones(self.n_strands, np.float64)
        self._damping = np.ones(self.n_strands, np.float64)
        self._gravity = np.ones(self.n_strands, np.float64)
        self._child_fast = None
        self.last_contacts = 0
        self.last_pairs = 0
        self.time = 0.0
        self.frame = 0

    # ----- topology / rest

    def _build_topology(self):
        off = self.offsets
        n, npt = self.n_strands, self.n_points
        counts = np.diff(off) if n else np.zeros(0, np.int64)
        sid = np.repeat(np.arange(n, dtype=np.int64), counts) if n else np.zeros(0, np.int64)
        local = np.arange(npt, dtype=np.int64) - np.repeat(off[:-1], counts) if n else np.zeros(0, np.int64)
        is_seg = np.ones(npt, bool)
        if n:
            is_seg[off[1:] - 1] = False
        if npt:
            is_seg[-1] = False if (n and off[-1] == npt) else is_seg[-1]
            # last point of the packed array is always a tip when offsets are well formed
            if n:
                is_seg[off[1:] - 1] = False
        self.strand_id = sid
        self.local_index = local
        self.e_i = np.flatnonzero(is_seg).astype(np.int64)
        self.e_j = (self.e_i + 1).astype(np.int64)
        self.e_parity = (local[self.e_i] & 1) if len(self.e_i) else np.zeros(0, np.int64)
        # bend edges: i -- k (skip one), same strand
        is_bend = np.ones(npt, bool)
        if n and npt:
            is_bend[off[1:] - 1] = False
            tips2 = off[:-1] + np.maximum(counts - 2, 0)
            is_bend[tips2] = False
            is_bend &= (counts[sid] >= 3) if n else is_bend
        self.b_i = np.flatnonzero(is_bend & (local < (counts[sid] - 2) if n and npt else False)).astype(np.int64)
        # safer rebuild of bend indices
        if n and npt:
            keep = np.zeros(npt, bool)
            keep[:npt] = False
            for s in range(n):
                a, b = int(off[s]), int(off[s + 1])
                if b - a >= 3:
                    keep[a:b - 2] = True
            self.b_i = np.flatnonzero(keep).astype(np.int64)
        else:
            self.b_i = np.zeros(0, np.int64)
        self.b_k = (self.b_i + 2).astype(np.int64)
        self.b_parity = (local[self.b_i] & 1) if len(self.b_i) else np.zeros(0, np.int64)
        self._rebuild_mass()
        self.param = np.zeros(npt, np.float64)
        if n:
            counts_safe = np.maximum(counts - 1, 1)
            self.param = local / counts_safe[sid]

    def _rebuild_mass(self):
        w = np.ones(self.n_points, np.float64)
        pin = max(int(self.params.pinned_prefix), 0)
        if self.n_strands and pin:
            for s in range(self.n_strands):
                a, b = int(self.offsets[s]), int(self.offsets[s + 1])
                w[a:min(a + pin, b)] = 0.0
        self.inv_mass = w

    def _capture_rest_metrics(self):
        P = self.rest
        if len(self.e_i):
            self.seg_rest = np.linalg.norm(P[self.e_j] - P[self.e_i], axis=1)
        else:
            self.seg_rest = np.zeros(0, np.float64)
        if len(self.b_i):
            self.bend_rest = np.linalg.norm(P[self.b_k] - P[self.b_i], axis=1)
        else:
            self.bend_rest = np.zeros(0, np.float64)
        # along-strand shape stiffness lerp
        t = self.param
        tc = np.power(np.clip(t, 0.0, 1.0), max(float(self.params.stiffness_curve), 1e-3))
        self.shape_w = ((1.0 - tc) * float(self.params.rest_shape_root)
                        + tc * float(self.params.rest_shape_tip))
        # bend compliance grows toward the tip: stiff root, floppy ends
        if len(self.b_i):
            self.bend_scale = 1.0 + (float(self.params.bend_tip_scale) - 1.0) * self.param[self.b_i]
        else:
            self.bend_scale = np.zeros(0, np.float64)

    def set_params(self, **kw):
        self.params.update(**kw)
        self._rebuild_mass()
        self._capture_rest_metrics()
        return self

    def set_rest(self, rest=None):
        self.rest = self.positions.copy() if rest is None else np.asarray(rest, dtype=np.float64).reshape(-1, 3).copy()
        if self.bind_M is not None:
            self.rest_local = xform_per_strand(self.rest, invert_M(self.bind_M), self.strand_id)
        self._capture_rest_metrics()
        return self

    def set_per_strand(self, stiffness=None, damping=None, gravity=None):
        if gravity is not None:
            self._gravity = np.broadcast_to(np.asarray(gravity, dtype=np.float64).ravel(), (self.n_strands,)).copy()
        if stiffness is not None:
            self._stiffness = np.broadcast_to(np.asarray(stiffness, dtype=np.float64).ravel(), (self.n_strands,)).copy()
        if damping is not None:
            self._damping = np.broadcast_to(np.asarray(damping, dtype=np.float64).ravel(), (self.n_strands,)).copy()
        return self

    # ----- attachments (object / bone / scalp)

    def bind_roots(self, matrices=None):
        """Capture the bind-time root frames. `matrices` is (S,4,4) world; default = identity at each rest root."""
        if matrices is None:
            M = np.tile(np.eye(4), (max(self.n_strands, 1), 1, 1))
            if self.n_strands:
                M[:, :3, 3] = self.rest[self.offsets[:-1]]
            matrices = M
        self.bind_M = _as_M44(matrices).copy()
        if self.bind_M.ndim == 2:
            self.bind_M = np.repeat(self.bind_M[None], max(self.n_strands, 1), axis=0)
        if len(self.bind_M) != max(self.n_strands, 1):
            raise ValueError("bind_roots: expected %d matrices, got %d" % (self.n_strands, len(self.bind_M)))
        self.root_M = self.bind_M.copy()
        self.root_M_prev = self.bind_M.copy()
        self.rest_local = xform_per_strand(self.rest, invert_M(self.bind_M), self.strand_id)
        return self

    def set_root_transforms(self, matrices, dt=None):
        """Update attachment frames. Linear + angular impulse is applied on the next `step()`."""
        M = _as_M44(matrices)
        if M.ndim == 2:
            M = np.repeat(M[None], max(self.n_strands, 1), axis=0)
        if self.bind_M is None:
            self.bind_roots(M)
        self.root_M_prev = self.root_M
        self.root_M = M.copy()
        if dt is not None:
            self.dt_used = float(dt)
        return self

    def bind_to_mesh(self, verts, tris):
        """Bind each root to the closest triangle of a scalp / body mesh (local or world, same space as rest)."""
        V = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
        T = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
        soup = SDF.soup_from_mesh(V, T)
        roots = self.rest[self.offsets[:-1]] if self.n_strands else np.zeros((0, 3))
        cp, sd, ti = SDF.closest_surface(soup, roots)
        A, B, C = V[T[ti, 0]], V[T[ti, 1]], V[T[ti, 2]]
        u, v, w = _barycentric(cp, A, B, C)
        fx, fy, fz = _triangle_frames(A, B, C)
        offset = np.einsum("ij,ij->i", roots - cp, fz)
        origin = cp + offset[:, None] * fz
        F = np.stack((fx, fy, fz), axis=2)
        # Bind frames live at the roots, in the scalp triangle basis, so later
        # update_from_mesh (which writes the same convention) cannot mix in an
        # object/bone translation and yank the groom off the surface.
        M = np.tile(np.eye(4), (max(self.n_strands, 1), 1, 1))
        if self.n_strands:
            M[:, :3, :3] = F
            M[:, :3, 3] = origin
        self.bind_roots(M)
        self._surface_bind = dict(tri=ti.astype(np.int64), bary=np.stack((u, v, w), axis=1),
                                  offset=offset, soup_sig=(len(V), len(T)),
                                  bind_frame=F)
        return self

    def update_from_mesh(self, verts, tris, dt=None):
        """Recompute root frames from a deforming scalp (same connectivity as `bind_to_mesh`)."""
        if self._surface_bind is None:
            self.bind_to_mesh(verts, tris)
        V = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
        T = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
        b = self._surface_bind
        ti = b["tri"]
        A, B, C = V[T[ti, 0]], V[T[ti, 1]], V[T[ti, 2]]
        bary = b["bary"]
        cp = bary[:, 0:1] * A + bary[:, 1:2] * B + bary[:, 2:3] * C
        fx, fy, fz = _triangle_frames(A, B, C)
        origin = cp + b["offset"][:, None] * fz
        F = np.stack((fx, fy, fz), axis=2)
        M = np.tile(np.eye(4), (max(self.n_strands, 1), 1, 1))
        if self.n_strands:
            M[:, :3, :3] = F
            M[:, :3, 3] = origin
        self.set_root_transforms(M, dt=dt)
        return self

    # ----- collision / wind

    def set_colliders(self, items):
        """`items` is a list of PartField, or (field, matrix), or dict(field=, matrix=, radius=)."""
        self.colliders = []
        for it in (items or []):
            self.add_collider(it)
        return self

    def add_collider(self, field, matrix=None, radius=None, friction=None, name=""):
        """Register a collision body. `radius` (thickness) and `friction` override the global values per collider."""
        if isinstance(field, dict):
            matrix = field.get("matrix", matrix)
            radius = field.get("radius", radius)
            friction = field.get("friction", friction)
            name = field.get("name", name)
            field = field["field"]
        elif isinstance(field, (tuple, list)) and not hasattr(field, "sample_sd"):
            if len(field) >= 2:
                matrix = field[1]
            if len(field) >= 3:
                radius = field[2]
            field = field[0]
        M = np.eye(4) if matrix is None else _as_M44(matrix).copy()
        self.colliders.append(dict(field=field, matrix=M, matrix_prev=M.copy(),
                                   radius=None if radius is None else float(radius),
                                   friction=None if friction is None else float(friction),
                                   name=str(name), enabled=True))
        return self

    def set_collider_matrix(self, index, matrix):
        """Move a collider. The previous matrix is kept so the next step sweeps the motion and
        transfers the collider's velocity to the hair it hits (a swatting hand / glove)."""
        c = self.colliders[index]
        M = _as_M44(matrix).copy()
        c["matrix_prev"] = c["matrix"]
        c["matrix"] = M
        return self

    def update_collider(self, index, field=None, matrix=None, radius=None, friction=None, enabled=None):
        c = self.colliders[index]
        if field is not None:
            c["field"] = field
        if matrix is not None:
            self.set_collider_matrix(index, matrix)
        if radius is not None:
            c["radius"] = float(radius)
        if friction is not None:
            c["friction"] = float(friction)
        if enabled is not None:
            c["enabled"] = bool(enabled)
        return self

    def set_wind(self, velocity_or_fn):
        if callable(velocity_or_fn):
            self.wind_fn = velocity_or_fn
            self.wind = np.zeros(3, np.float64)
        else:
            self.wind_fn = None
            self.wind = np.asarray(velocity_or_fn, dtype=np.float64).reshape(3)
        return self

    def set_fluid_velocity(self, fn):
        """Optional callable `fn(P) -> (N,3)` sampled each substep (Tessera fluid / wind-tunnel field)."""
        self.fluid_fn = fn
        return self

    # ----- guides / children

    def auto_guides(self, ratio=None, seed=0, min_guides=1):
        r = self.params.guide_ratio if ratio is None else ratio
        self.guide_idx = select_guides(self.n_strands, ratio=r, seed=seed, min_guides=min_guides)
        return self

    def set_guides(self, indices):
        self.guide_idx = np.asarray(indices, dtype=np.int64).ravel()
        return self

    def attach_children(self, positions, offsets, k=None, smooth=None):
        self.child_rest = np.asarray(positions, dtype=np.float64).reshape(-1, 3).copy()
        self.child_off = np.asarray(offsets, dtype=np.int64).ravel().copy()
        k = self.params.child_k if k is None else k
        smooth = self.params.child_smooth if smooth is None else smooth
        g_off = self.offsets
        # children skin against the *guide* rest roots
        gidx = self.guide_idx if len(self.guide_idx) else np.arange(self.n_strands)
        g_roots = self.rest[g_off[gidx]]
        c_roots = self.child_rest[self.child_off[:-1]]
        idx, w = child_skin_weights(c_roots, g_roots, k=k, smooth=smooth)
        # `idx` is into the guide list; store it as such
        self.child_idx = idx
        self.child_w = w
        self._child_guide_idx = gidx
        self._build_child_fast(gidx)
        return self

    def _build_child_fast(self, gidx):
        """Precompute, per child point and per skin guide, the guide point pair + blend that carry the child
        (vectorised `child_positions` afterwards; the python loop runs once here)."""
        cOff, cP = self.child_off, self.child_rest
        npc = len(cP)
        k = self.child_idx.shape[1] if self.child_idx is not None and self.child_idx.ndim == 2 else 0
        if not k or npc == 0:
            self._child_fast = None
            return
        i0 = np.zeros((npc, k), np.int64)
        fr = np.zeros((npc, k), np.float64)
        W = np.zeros((npc, k), np.float64)
        bind = np.zeros((npc, k, 3), np.float64)
        garc = {}
        for ci in range(len(cOff) - 1):
            a, b = int(cOff[ci]), int(cOff[ci + 1])
            pts = cP[a:b]
            if b - a < 2:
                continue
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            arc = np.concatenate([[0.0], np.cumsum(seg)])
            sc = arc / (arc[-1] if arc[-1] > _EPS else 1.0)
            for kk in range(k):
                m = int(self.child_idx[ci, kk])
                gs = int(gidx[m])
                ga, gb = int(self.offsets[gs]), int(self.offsets[gs + 1])
                if gs not in garc:
                    gp = self.rest[ga:gb]
                    gl = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(gp, axis=0), axis=1))])
                    garc[gs] = gl
                gl = garc[gs]
                tot = gl[-1] if gl[-1] > _EPS else 1.0
                tgt = sc * tot
                j = np.clip(np.searchsorted(gl, tgt, side="right") - 1, 0, gb - ga - 2)
                span = np.maximum(gl[j + 1] - gl[j], _EPS)
                f = np.clip((tgt - gl[j]) / span, 0.0, 1.0)
                i0[a:b, kk] = ga + j
                fr[a:b, kk] = f
                W[a:b, kk] = self.child_w[ci, kk]
                bind[a:b, kk] = self.rest[ga + j] * (1 - f)[:, None] + self.rest[ga + j + 1] * f[:, None]
        self._child_fast = dict(i0=i0, fr=fr, W=W, bind=bind)

    def _children_fast(self):
        F = self._child_fast
        P = self.positions
        out = self.child_rest.copy()
        for kk in range(F["i0"].shape[1]):
            i0 = F["i0"][:, kk]
            f = F["fr"][:, kk][:, None]
            cur = P[i0] * (1 - f) + P[i0 + 1] * f
            out += F["W"][:, kk][:, None] * (cur - F["bind"][:, kk])
        return out


    def _project_packed(self, positions, offsets):
        """Push non-root points of a packed groom out of every collider (used for skinned children)."""
        if not self.colliders or positions is None or len(positions) == 0:
            return positions
        x = np.asarray(positions, dtype=np.float64).reshape(-1, 3).copy()
        off = np.asarray(offsets, dtype=np.int64).ravel()
        free = np.ones(len(x), bool)
        if len(off) > 1:
            free[off[:-1]] = False
        if not np.any(free):
            return x
        p = self.params
        for col in self.colliders:
            if not col.get("enabled", True):
                continue
            rad = p.collision_radius if col["radius"] is None else col["radius"]
            xf, _hits, _n = project_points_sdf(x[free], col["field"], matrix=col["matrix"], radius=rad,
                                               refine=False, iterations=max(int(p.collision_iterations), 1))
            x[free] = xf
        if p.ground_z is not None:
            z0 = float(p.ground_z) + p.collision_radius
            under = free & (x[:, 2] < z0)
            x[under, 2] = z0
        return x

    def child_positions(self):

        if self.child_rest is None:
            return None
        if self._child_fast is not None:
            return self._project_packed(self._children_fast(), self.child_off)
        gidx = getattr(self, "_child_guide_idx", self.guide_idx)
        # pack only the guide strands for the interpolator (same offsets? no — extract)
        g_pos, g_off = _extract_strands(self.positions, self.offsets, gidx)
        g_rest, g_off_r = _extract_strands(self.rest, self.offsets, gidx)
        # offsets of the extracted guides must match
        out = interpolate_children(g_pos, g_off, self.child_rest, self.child_off,
                                   guide_rest=g_rest, indices=self.child_idx, weights=self.child_w,
                                   k=self.params.child_k, smooth=self.params.child_smooth)
        return self._project_packed(out, self.child_off)

    # ----- integration

    @property
    def velocities(self):
        dt = max(float(self.dt_used), 1e-8)
        return (self.positions - self.prev) / dt

    def reset(self):
        for col in self.colliders:
            col["matrix_prev"] = col["matrix"]
        self.positions[:] = self.rest
        self.prev[:] = self.rest
        if self.bind_M is not None:
            self.root_M = self.bind_M.copy()
            self.root_M_prev = self.bind_M.copy()
        self.time = 0.0
        self.frame = 0
        return self

    def _pin_roots(self, x, prev, dt):
        """Snap pinned prefix to the current root frame and seed Verlet with attachment velocity."""
        if self.rest_local is None or not self.n_strands:
            return
        target = xform_per_strand(self.rest_local, self.root_M, self.strand_id)
        pin = self.inv_mass == 0.0
        if not np.any(pin):
            return
        x[pin] = target[pin]
        dt = max(float(dt), 1e-8)
        Mc, Mp = self.root_M, self.root_M_prev
        v_lin = (Mc[:, :3, 3] - Mp[:, :3, 3]) / dt
        dR = np.einsum("nij,nkj->nik", Mc[:, :3, :3], Mp[:, :3, :3])
        skew = 0.5 * (dR - np.transpose(dR, (0, 2, 1)))
        omega = np.stack((skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]), axis=1) / dt
        sid = self.strand_id[pin]
        r = x[pin] - Mc[sid, :3, 3]
        v = v_lin[sid] + np.cross(omega[sid], r)
        prev[pin] = x[pin] - v * dt

    def _rest_targets(self):
        if self.rest_local is not None:
            return xform_per_strand(self.rest_local, self.root_M, self.strand_id)
        return self.rest

    def _project_stretch(self, x, compliance, iters, dt):
        if not len(self.e_i) or iters <= 0:
            return
        alpha = float(compliance) / max(dt * dt, 1e-12)
        wi, wj = self.inv_mass, self.inv_mass
        rest = self.seg_rest
        for _ in range(int(iters)):
            for par in (0, 1):
                m = self.e_parity == par
                if not np.any(m):
                    continue
                i, j = self.e_i[m], self.e_j[m]
                d = x[j] - x[i]
                L = np.linalg.norm(d, axis=1)
                n = d / np.maximum(L, _EPS)[:, None]
                C = L - rest[m]
                wsum = wi[i] + wj[j]
                sl = wsum > 0.0
                if not np.any(sl):
                    continue
                dl = np.zeros_like(C)
                dl[sl] = -C[sl] / (wsum[sl] + alpha)
                i_s, j_s = i[sl], j[sl]
                x[i_s] = x[i_s] - wi[i_s, None] * dl[sl, None] * n[sl]
                x[j_s] = x[j_s] + wj[j_s, None] * dl[sl, None] * n[sl]

    def _project_bend(self, x, compliance, iters, dt):
        if not len(self.b_i) or iters <= 0:
            return
        alpha0 = float(compliance) / max(dt * dt, 1e-12)
        wi, wk = self.inv_mass, self.inv_mass
        rest = self.bend_rest
        for _ in range(int(iters)):
            for par in (0, 1):
                m = self.b_parity == par
                if not np.any(m):
                    continue
                alpha = alpha0 * self.bend_scale[m]
                i, k = self.b_i[m], self.b_k[m]
                d = x[k] - x[i]
                L = np.linalg.norm(d, axis=1)
                n = d / np.maximum(L, _EPS)[:, None]
                C = L - rest[m]
                wsum = wi[i] + wk[k]
                sl = wsum > 0.0
                if not np.any(sl):
                    continue
                dl = np.zeros_like(C)
                dl[sl] = -C[sl] / (wsum[sl] + alpha[sl])
                i_s, k_s = i[sl], k[sl]
                x[i_s] = x[i_s] - wi[i_s, None] * dl[sl, None] * n[sl]
                x[k_s] = x[k_s] + wk[k_s, None] * dl[sl, None] * n[sl]

    def _apply_rest_shape(self, x):
        k = float(self.params.rest_shape)
        if k <= 0.0:
            return
        tgt = self._rest_targets()
        w = self.shape_w * k * self._stiffness[self.strand_id]
        # pinned points stay put (inv_mass == 0)
        w = w * (self.inv_mass > 0)
        x += w[:, None] * (tgt - x)

    def _aero(self, x, prev, dt):
        drag = float(self.params.wind_drag) * float(self.params.fluid_coupling)
        if drag <= 0.0 and self.fluid_fn is None and self.wind_fn is None and np.all(self.wind == 0):
            return np.zeros_like(x)
        v = (x - prev) / max(dt, 1e-8)
        if self.wind_fn is not None:
            u = np.asarray(self.wind_fn(x), dtype=np.float64).reshape(-1, 3)
        else:
            u = np.broadcast_to(self.wind, x.shape).copy()
        if self.fluid_fn is not None:
            u = u + np.asarray(self.fluid_fn(x), dtype=np.float64).reshape(-1, 3)
        vrel = u - v
        # strand tangent from the current edge (root uses next, tip uses prev)
        tan = np.zeros_like(x)
        if len(self.e_i):
            e = _normalize(x[self.e_j] - x[self.e_i])
            tan[self.e_i] = e
            tip = self.e_j.copy()
            # last write wins at shared vertices; tips get the incoming edge
            tan[tip] = e
        tan = _normalize(tan, fallback=np.array((0.0, 0.0, 1.0)))
        vtan = np.einsum("ij,ij->i", vrel, tan)[:, None] * tan
        vnor = vrel - vtan
        # quadratic anisotropic drag (more resistance across the fibre)
        kt = float(self.params.wind_tangent)
        acc = drag * (np.linalg.norm(vnor, axis=1)[:, None] * vnor
                      + kt * np.linalg.norm(vtan, axis=1)[:, None] * vtan)
        # stability clamp
        amax = 200.0
        an = np.linalg.norm(acc, axis=1)
        scale = np.minimum(1.0, amax / np.maximum(an, 1e-8))
        acc *= scale[:, None]
        acc[self.inv_mass == 0] = 0.0
        return acc

    def _collider_pose(self, col, frac):
        """World matrix of a collider `frac` of the way through this frame's motion (swept collision)."""
        if frac >= 1.0:
            return col["matrix"]
        return lerp_matrix(col["matrix_prev"], col["matrix"], frac)

    def _collide(self, x, prev, h=None, frac=1.0):
        """Project points out of every collider. Returns (x, prev, contact) where `contact` carries the
        per-point normal, collider velocity and friction used by `_contact_friction` after integration."""
        p = self.params
        free = self.inv_mass > 0.0
        N = len(x)
        cn = np.zeros((N, 3), np.float64)
        cv = np.zeros((N, 3), np.float64)
        cf = np.zeros(N, np.float64)
        dt_frame = max(float(self.dt_used), 1e-8)
        n_contact = 0
        for col in self.colliders:
            if not col.get("enabled", True) or not np.any(free):
                continue
            rad = p.collision_radius if col["radius"] is None else col["radius"]
            fr = p.collision_friction if col["friction"] is None else col["friction"]
            M = self._collider_pose(col, frac)
            fi = np.flatnonzero(free)
            xf = x[fi]
            xo, hits, nw = project_points_sdf(xf, col["field"], matrix=M, radius=rad,
                                              refine=p.sdf_refine, iterations=p.collision_iterations)
            if not hits:
                continue
            hit = np.linalg.norm(nw, axis=1) > 0.5
            gi = fi[hit]
            x[gi] = xo[hit]
            # velocity of the collider material point that touched the strand (exact for rigid motion)
            Mp = col["matrix_prev"]
            if Mp is not col["matrix"] and not np.array_equal(Mp, col["matrix"]):
                xl = inv_xform_points(x[gi], M)
                vc = (x[gi] - xform_points(xl, Mp)) / dt_frame if frac >= 1.0 else \
                     (xform_points(xl, col["matrix"]) - xform_points(xl, Mp)) / dt_frame
            else:
                vc = np.zeros((len(gi), 3))
            # keep the strongest contact if several colliders overlap
            take = cf[gi] <= fr
            gi_t = gi[take]
            cn[gi_t] = nw[hit][take]
            cv[gi_t] = vc[take]
            cf[gi_t] = fr
            n_contact += int(len(gi))
        self.last_contacts = n_contact
        if p.ground_z is not None:
            z0 = float(p.ground_z) + p.collision_radius
            under = x[:, 2] < z0
            if np.any(under):
                x[under, 2] = z0
                going = under & ((x[:, 2] - prev[:, 2]) < 0)
                prev[going, 2] = x[going, 2]
        return x, prev, (cn, cv, cf)

    def _contact_friction(self, x, prev, contact, h):
        """Coulomb-style tangential damping against the (possibly moving) collider, applied to the Verlet velocity."""
        cn, cv, cf = contact
        m = cf > 0.0
        if not np.any(m):
            return
        v = x[m] - prev[m]
        vrel = v - cv[m] * h
        n = cn[m]
        vn = np.einsum("ij,ij->i", vrel, n)[:, None]
        vt = vrel - n * vn
        # stick to the surface (kills the push-out rebound) but keep what the collider gave the strand
        vn_keep = np.maximum(vn, 0.0) * 0.5
        v_new = cv[m] * h + vt * (1.0 - cf[m])[:, None] + n * vn_keep
        prev[m] = x[m] - v_new

    def _hair_hair(self, x, prev, h):
        """Cheap strand-strand contact: vectorised spatial-hash repulsion (keeps volume), optional
        clump attraction (locks) and velocity sharing (hair-hair friction). Samples points and segment midpoints."""
        p = self.params
        if (not p.hair_collision) or (not getattr(p, 'self_collision', True)) or self.n_strands < 2:
            self.last_pairs = 0
            return
        r = max(float(p.strand_radius), 1e-5) * (1.0 + max(float(p.volume), 0.0))
        k_rep = float(np.clip(p.repel_strength, 0.0, 1.0))
        k_clump = max(float(p.clumping), 0.0)
        k_fric = float(np.clip(p.strand_friction, 0.0, 1.0))
        if k_rep <= 0.0 and k_clump <= 0.0 and k_fric <= 0.0:
            self.last_pairs = 0
            return
        contact = 2.0 * r
        reach = contact + (float(p.clump_range) if k_clump > 0.0 else 0.0)
        # sample set: points + midpoints of every segment
        mid = 0.5 * (x[self.e_i] + x[self.e_j])
        S = np.concatenate([x, mid])
        sid = np.concatenate([self.strand_id, self.strand_id[self.e_i]])
        ownA = np.concatenate([np.arange(len(x)), self.e_i])
        ownB = np.concatenate([np.arange(len(x)), self.e_j])
        pin = self.inv_mass == 0.0
        i, j, d, n = hair_hair_pairs(S, max(reach, 1e-5), sid, reach, max_pairs=int(p.max_pairs))
        self.last_pairs = int(len(i))
        if not len(i):
            return
        disp = np.zeros_like(x)
        cnt = np.zeros(len(x), np.float64)
        wgt = np.zeros(len(i), np.float64)
        pen = d < contact
        # repulsion: move the pair apart by the overlap
        if k_rep > 0.0:
            wgt[pen] += k_rep * (contact - d[pen]) * 0.5
        # clump attraction in the band beyond contact, fading out at `reach`
        if k_clump > 0.0:
            band = ~pen
            span = max(reach - contact, 1e-6)
            wgt[band] -= 0.25 * min(k_clump, 4.0) * (d[band] - contact) * (1.0 - (d[band] - contact) / span)
        act = wgt != 0.0
        if np.any(act):
            ii, jj, nn, ww = i[act], j[act], n[act], wgt[act]
            dv = nn * ww[:, None]
            for samp, sign in ((ii, -1.0), (jj, 1.0)):
                a, b = ownA[samp], ownB[samp]
                for own in (a, b):
                    for c3 in range(3):
                        disp[:, c3] += np.bincount(own, weights=sign * dv[:, c3], minlength=len(x))
                    cnt += np.bincount(own, minlength=len(x))
            # points that are both ends of a point sample were counted twice with the same value: average handles it
            ok = (cnt > 0) & (~pin)
            x[ok] += disp[ok] / cnt[ok][:, None]
        if k_fric > 0.0:
            vel = x - prev
            vs = np.concatenate([vel, 0.5 * (vel[self.e_i] + vel[self.e_j])])
            near = d < contact * 1.5
            if np.any(near):
                ii, jj = i[near], j[near]
                dvv = 0.5 * k_fric * (vs[jj] - vs[ii])
                acc = np.zeros_like(x)
                c2 = np.zeros(len(x), np.float64)
                for samp, sign in ((ii, 1.0), (jj, -1.0)):
                    for own in (ownA[samp], ownB[samp]):
                        for c3 in range(3):
                            acc[:, c3] += np.bincount(own, weights=sign * dvv[:, c3], minlength=len(x))
                        c2 += np.bincount(own, minlength=len(x))
                ok = (c2 > 0) & (~pin)
                prev[ok] -= acc[ok] / c2[ok][:, None]

    def _frame_substeps(self, nsub):
        """More substeps when a collider moves fast relative to its own size (anti-tunnelling)."""
        p = self.params
        if not p.adaptive_substeps or not self.colliders:
            return nsub
        worst = 0.0
        for col in self.colliders:
            if not col.get("enabled", True):
                continue
            Mp, Mc = col["matrix_prev"], col["matrix"]
            if Mp is Mc:
                continue
            f = col["field"]
            size = float(np.min(np.asarray(getattr(f, "bmax", np.ones(3))) - np.asarray(getattr(f, "bmin", np.zeros(3)))))
            size = max(size, 0.02)
            c = 0.5 * (np.asarray(getattr(f, "bmax", np.zeros(3))) + np.asarray(getattr(f, "bmin", np.zeros(3))))
            move = float(np.linalg.norm(xform_points(c[None], Mc) - xform_points(c[None], Mp)))
            rot = float(np.linalg.norm(Mc[:3, :3] - Mp[:3, :3])) * 0.5 * size
            worst = max(worst, (move + rot) / (0.5 * size))
        extra = int(np.ceil(worst))
        return int(min(max(nsub, extra), 4 * nsub))

    def step(self, dt=None, root_M=None, substeps=None):
        """Advance one frame. Optional `root_M` is forwarded to `set_root_transforms`."""
        p = self.params
        if root_M is not None:
            self.set_root_transforms(root_M, dt=dt)
        dt = float(p.dt if dt is None else dt)
        self.dt_used = dt
        nsub = max(int(p.substeps if substeps is None else substeps), 1)
        if substeps is None:
            nsub = self._frame_substeps(nsub)
        h = dt / nsub
        g = np.asarray(p.gravity, dtype=np.float64).reshape(3) * float(p.gravity_scale)
        gstrand = self._gravity[self.strand_id]
        x = self.positions
        prev = self.prev
        damp0 = float(np.clip(p.damping, 0.0, 1.0))
        vmax = float(p.max_speed) * h
        for _s in range(nsub):
            frac = (_s + 1.0) / nsub
            self._pin_roots(x, prev, h)
            acc = g[None, :] * gstrand[:, None]
            acc += self._aero(x, prev, h)
            acc[self.inv_mass == 0] = 0.0
            # per-strand damping scale
            damp = damp0 * self._damping[self.strand_id]
            damp = np.clip(damp, 0.0, 1.0)
            x_old = x.copy()
            x[:] = x + (x - prev) * damp[:, None] + acc * (h * h)
            self._pin_roots(x, prev, h)
            self._apply_rest_shape(x)
            self._project_stretch(x, p.stretch_compliance, p.stretch_iterations, h)
            self._project_bend(x, p.bend_compliance, p.bend_iterations, h)
            if _s % max(int(p.hair_every), 1) == 0:
                self._hair_hair(x, prev, h)
            self._project_stretch(x, p.stretch_compliance, max(2, p.stretch_iterations // 2), h)
            x, prev, contact = self._collide(x, prev, h, frac)
            self._pin_roots(x, prev, h)
            # clamp runaway displacement, then Verlet velocity = this substep's displacement
            dx = x - x_old
            nrm = np.linalg.norm(dx, axis=1)
            big = nrm > vmax
            if np.any(big):
                x[big] = x_old[big] + dx[big] * (vmax / nrm[big])[:, None]
            prev[:, :] = x_old
            self._contact_friction(x, prev, contact, h)
            self._pin_roots(x, prev, h)
        self.positions = x
        self.prev = prev
        self.time += dt
        self.frame += 1
        # the previous root frame becomes the current one after the step is consumed
        self.root_M_prev = self.root_M.copy()
        for col in self.colliders:
            col["matrix_prev"] = col["matrix"]
        return self

    # ----- metrics / cache / package

    def segment_lengths(self):
        if not len(self.e_i):
            return np.zeros(0, np.float64)
        return np.linalg.norm(self.positions[self.e_j] - self.positions[self.e_i], axis=1)

    def length_error(self):
        L = self.segment_lengths()
        if not len(L):
            return 0.0
        return float(np.max(np.abs(L - self.seg_rest) / np.maximum(self.seg_rest, 1e-8)))

    def snapshot(self):
        return dict(positions=self.positions.copy(), prev=self.prev.copy(),
                    root_M=self.root_M.copy(), root_M_prev=self.root_M_prev.copy(),
                    time=self.time, frame=self.frame)

    def restore(self, snap):
        self.positions = np.asarray(snap["positions"], dtype=np.float64).reshape(-1, 3).copy()
        self.prev = np.asarray(snap["prev"], dtype=np.float64).reshape(-1, 3).copy()
        self.root_M = np.asarray(snap["root_M"], dtype=np.float64).copy()
        self.root_M_prev = np.asarray(snap["root_M_prev"], dtype=np.float64).copy()
        self.time = float(snap["time"])
        self.frame = int(snap["frame"])
        return self

    def frame_record(self):
        rec = dict(frame=int(self.frame), time=float(self.time),
                   positions=self.positions.copy(), offsets=self.offsets.copy())
        ch = self.child_positions()
        if ch is not None:
            rec["child_positions"] = ch
            rec["child_offsets"] = self.child_off.copy()
        return rec

    def rest_package(self, extra=None, collider_meta=None):
        """JSON-serialisable game-engine hair physics package (arrays as lists; use `write_package` for .npz)."""
        gidx = np.asarray(self.guide_idx, dtype=np.int64)
        pkg = dict(
            format=PACKAGE_FORMAT,
            version=PACKAGE_VERSION,
            engine_targets=["unreal_groom", "unity", "web"],
            n_strands=int(self.n_strands),
            n_points=int(self.n_points),
            offsets=self.offsets.tolist(),
            rest=self.rest.tolist(),
            radii=self.radii.tolist(),
            guides=dict(indices=gidx.tolist(),
                        ratio=float(self.params.guide_ratio),
                        counts=strand_counts(self.offsets).tolist()),
            per_strand=dict(
                stiffness=self._stiffness.tolist(),
                damping=self._damping.tolist(),
                segment_count=strand_counts(self.offsets).tolist(),
            ),
            params=self.params.as_dict(),
            collision=dict(
                radius=float(self.params.collision_radius),
                sdf_resolution=int(self.params.sdf_resolution),
                sdf_level=int(self.params.sdf_level),
                colliders=list(collider_meta or []),
            ),
        )
        if self.child_rest is not None:
            pkg["children"] = dict(offsets=self.child_off.tolist(), rest=self.child_rest.tolist(),
                                   guide_indices=self.child_idx.tolist(), weights=self.child_w.tolist())
        if extra:
            pkg.update(extra)
        return pkg


def _extract_strands(positions, offsets, indices):
    P = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    off = np.asarray(offsets, dtype=np.int64).ravel()
    idx = np.asarray(indices, dtype=np.int64).ravel()
    chunks = [P[off[i]:off[i + 1]] for i in idx]
    return pack_strands(chunks)


# ----------------------------------------------------------------------------- cache / package I/O


def write_npz_cache(path, frames, offsets, rest=None, meta=None):
    """Write a point-cache `.npz`: positions_0001, positions_0002, ..., offsets, rest, frames, times."""
    payload = dict(offsets=np.asarray(offsets, dtype=np.int64),
                   n_frames=np.array(len(frames), np.int64))
    if rest is not None:
        payload["rest"] = np.asarray(rest, dtype=np.float64)
    fr = []
    tm = []
    for rec in frames:
        f = int(rec["frame"])
        fr.append(f)
        tm.append(float(rec.get("time", 0.0)))
        payload["positions_%04d" % f] = np.asarray(rec["positions"], dtype=np.float32)
        if "child_positions" in rec:
            payload["child_%04d" % f] = np.asarray(rec["child_positions"], dtype=np.float32)
            payload["child_offsets"] = np.asarray(rec["child_offsets"], dtype=np.int64)
    payload["frames"] = np.asarray(fr, dtype=np.int64)
    payload["times"] = np.asarray(tm, dtype=np.float64)
    if meta:
        payload["meta_json"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def read_npz_cache(path):
    data = np.load(path, allow_pickle=False)
    frames = []
    for f in data["frames"].tolist():
        rec = dict(frame=int(f), positions=np.asarray(data["positions_%04d" % f], dtype=np.float64),
                   offsets=np.asarray(data["offsets"], dtype=np.int64))
        if "times" in data.files:
            # times aligned with frames
            pass
        key = "child_%04d" % f
        if key in data.files:
            rec["child_positions"] = np.asarray(data[key], dtype=np.float64)
            rec["child_offsets"] = np.asarray(data["child_offsets"], dtype=np.int64)
        frames.append(rec)
    meta = None
    if "meta_json" in data.files:
        meta = json.loads(bytes(data["meta_json"].tolist()).decode("utf-8") if data["meta_json"].dtype != np.uint8
                          else data["meta_json"].tobytes().decode("utf-8"))
    rest = np.asarray(data["rest"], dtype=np.float64) if "rest" in data.files else None
    times = np.asarray(data["times"], dtype=np.float64) if "times" in data.files else None
    if times is not None:
        for rec, t in zip(frames, times):
            rec["time"] = float(t)
    return dict(frames=frames, offsets=np.asarray(data["offsets"], dtype=np.int64), rest=rest, meta=meta)


def write_package(folder, solver, frames=None, collider_meta=None, extra=None):
    """Write a game-engine hair physics package: `hair_physics.json` + `rest_curves.npz` [+ `cache.npz`]."""
    os.makedirs(folder, exist_ok=True)
    pkg = solver.rest_package(extra=extra, collider_meta=collider_meta)
    # heavy arrays live in npz; json keeps metadata + small arrays for engines that want a single file too
    rest_path = os.path.join(folder, "rest_curves.npz")
    np.savez_compressed(rest_path,
                        positions=np.asarray(solver.rest, dtype=np.float32),
                        offsets=np.asarray(solver.offsets, dtype=np.int64),
                        radii=np.asarray(solver.radii, dtype=np.float32),
                        stiffness=np.asarray(solver._stiffness, dtype=np.float32),
                        damping=np.asarray(solver._damping, dtype=np.float32))
    pkg["rest_curves"] = "rest_curves.npz"
    if frames:
        cache_path = os.path.join(folder, "cache.npz")
        write_npz_cache(cache_path, frames, solver.offsets, rest=solver.rest)
        pkg["cache"] = "cache.npz"
        pkg["n_cached_frames"] = len(frames)
    json_path = os.path.join(folder, "hair_physics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(pkg, f, indent=2)
    return json_path


def read_package(folder):
    json_path = folder if folder.endswith(".json") else os.path.join(folder, "hair_physics.json")
    with open(json_path, "r", encoding="utf-8") as f:
        pkg = json.load(f)
    base = os.path.dirname(json_path)
    rest_file = os.path.join(base, pkg.get("rest_curves", "rest_curves.npz"))
    rest = np.load(rest_file) if os.path.isfile(rest_file) else None
    cache = None
    if pkg.get("cache"):
        cp = os.path.join(base, pkg["cache"])
        if os.path.isfile(cp):
            cache = read_npz_cache(cp)
    return pkg, rest, cache
