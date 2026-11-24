"""Solid-body fluid blending and smooth volumetric density fields (pure NumPy, no bpy).

Particles are splatted with a compactly supported Wendland C2 kernel into a regular density (and
optional colour / temperature / velocity) grid, optionally relaxed with a mean-curvature
(surface-tension) term, then meshed with the extension's surface nets.  A short volume-preserving
Laplacian smooth plus area-weighted normals give a continuous liquid skin (water-like) instead of
a cloud of disconnected beads.  The same splat is reused for combustion plumes and wind-tunnel
airflow: a blurred density / temperature / speed volume that looks like a solid slipstream or a
smooth flame rather than sparse streaks.

The older `core.fluidsurface.fluid_surface` / `SurfaceBuilder` path is left unchanged (existing
merge tests depend on its kernel and defaults).  This module is the higher-quality solid-body and
volume blender used by Fluid mode Surface rendering, combustion plume blending, and the wind tunnel.
"""
import numpy as np

from . import cavity
from .fluidsurface import gaussian_blur, mesh_components


def wendland(d, R):
    """Wendland C2 kernel (1-q)^4 (1+4q) for q = d/R < 1, else 0.  Compact, C2, positive."""
    q = np.clip(np.asarray(d, float) / max(float(R), 1e-12), 0.0, 1.0)
    t = 1.0 - q
    return (t * t * t * t) * (1.0 + 4.0 * q)


def _lattice_norm(dx, R, h):
    """Kernel sum of a packed cubic lattice of spacing dx, so bulk density normalises to 1."""
    r = int(np.ceil(R / max(dx, 1e-12))) + 2
    g = np.arange(-r, r + 1) * dx
    X, Y, Z = np.meshgrid(g, g, g, indexing='ij')
    return float(max(wendland(np.sqrt(X * X + Y * Y + Z * Z), R).sum(), 1e-12))


def splat_fields(x, *, dx, radius=2.2, cell=0.5, colors=None, temp=None, vel=None, weight=None, blur=0.85):
    """Wendland-splat particles onto a world-aligned grid.

    Returns dict(lo, h, dens, colors|None, temp|None, vel|None, dims).  `dens` is 1 in the bulk of a
    packed lattice of spacing `dx`.  Colour / temperature / velocity are density-weighted means.
    """
    x = np.asarray(x, float)
    if len(x) == 0:
        z = np.zeros((1, 1, 1))
        return dict(lo=np.zeros(3), h=cell * dx, dens=z, colors=None, temp=None, vel=None, dims=np.array(z.shape))
    h = float(cell) * float(dx)
    R = float(radius) * float(dx)
    pad = int(np.ceil(R / h)) + 3
    lo_i = np.floor(x.min(0) / h).astype(int) - pad
    hi_i = np.floor(x.max(0) / h).astype(int) + pad + 1
    dims = hi_i - lo_i + 1
    # guard huge grids (keep the field tractable in unit tests and mid-size demos)
    max_cells = 2_000_000
    if int(np.prod(dims)) > max_cells:
        scale = (int(np.prod(dims)) / max_cells) ** (1.0 / 3.0)
        h *= scale
        pad = int(np.ceil(R / h)) + 3
        lo_i = np.floor(x.min(0) / h).astype(int) - pad
        hi_i = np.floor(x.max(0) / h).astype(int) + pad + 1
        dims = hi_i - lo_i + 1
    g = x / h - lo_i
    i0 = np.floor(g).astype(int)
    f = g - i0
    wgt = np.ones(len(x)) if weight is None else np.asarray(weight, float)
    # trilinear splat of the Wendland weight evaluated at the particle (the kernel support is
    # applied as an extra Gaussian blur of sigma ~ R/h, which is cheaper than a large stencil)
    dens = np.zeros(dims)
    for cx in (0, 1):
        for cy in (0, 1):
            for cz in (0, 1):
                w = (f[:, 0] if cx else 1 - f[:, 0]) * (f[:, 1] if cy else 1 - f[:, 1]) * (f[:, 2] if cz else 1 - f[:, 2]) * wgt
                ii = i0 + np.array([cx, cy, cz])
                # clip any particle that landed on the padded rim
                ok = np.all((ii >= 0) & (ii < dims), axis=1)
                if not ok.any():
                    continue
                flat = (ii[ok, 0] * dims[1] + ii[ok, 1]) * dims[2] + ii[ok, 2]
                dens += np.bincount(flat, w[ok], int(np.prod(dims))).reshape(dims)

    def _chan(arr, nchan):
        out = np.zeros(tuple(dims) + ((nchan,) if nchan > 1 else ()))
        for cx in (0, 1):
            for cy in (0, 1):
                for cz in (0, 1):
                    w = (f[:, 0] if cx else 1 - f[:, 0]) * (f[:, 1] if cy else 1 - f[:, 1]) * (f[:, 2] if cz else 1 - f[:, 2]) * wgt
                    ii = i0 + np.array([cx, cy, cz])
                    ok = np.all((ii >= 0) & (ii < dims), axis=1)
                    if not ok.any():
                        continue
                    flat = (ii[ok, 0] * dims[1] + ii[ok, 1]) * dims[2] + ii[ok, 2]
                    if nchan == 1:
                        out += np.bincount(flat, w[ok] * arr[ok], int(np.prod(dims))).reshape(dims)
                    else:
                        for c in range(nchan):
                            out[..., c] += np.bincount(flat, w[ok] * arr[ok, c], int(np.prod(dims))).reshape(dims)
        return out

    csum = _chan(np.asarray(colors, float), 3) if colors is not None else None
    tsum = _chan(np.asarray(temp, float), 1) if temp is not None else None
    vsum = _chan(np.asarray(vel, float), 3) if vel is not None else None
    # blur in cells: Wendland support R maps to sigma ~ 0.45 R/h plus the caller's blur
    sg = max(0.45 * R / h, 0.35) * float(blur)
    dens = gaussian_blur(dens, sg)
    if csum is not None:
        csum = gaussian_blur(csum, sg)
    if tsum is not None:
        tsum = gaussian_blur(tsum, sg)
    if vsum is not None:
        vsum = gaussian_blur(vsum, sg)
    # one particle contributes ~ (h/dx)^3 of a packed cell after trilinear + blur; normalise to 1 in the bulk
    dens /= (h / max(float(dx), 1e-12)) ** 3
    if csum is not None:
        csum /= (h / max(float(dx), 1e-12)) ** 3
    if tsum is not None:
        tsum /= (h / max(float(dx), 1e-12)) ** 3
    if vsum is not None:
        vsum /= (h / max(float(dx), 1e-12)) ** 3
    return dict(lo=lo_i * h, h=h, dens=dens, colors=csum, temp=tsum, vel=vsum, dims=np.array(dims))


def _laplacian(a):
    """6-neighbour Laplacian with Neumann (replicate) boundaries, same shape as `a`."""
    p = np.pad(a, 1, mode='edge')
    return (p[0:-2, 1:-1, 1:-1] + p[2:, 1:-1, 1:-1] + p[1:-1, 0:-2, 1:-1] + p[1:-1, 2:, 1:-1]
            + p[1:-1, 1:-1, 0:-2] + p[1:-1, 1:-1, 2:] - 6.0 * a)


def apply_tension(dens, strength=0.35, iters=2):
    """Mean-curvature (surface-tension) relaxation: dens += s * lap(dens) near the iso, slightly rounding necks."""
    out = np.asarray(dens, float).copy()
    s = float(strength)
    if s <= 0.0:
        return out
    for _ in range(max(int(iters), 0)):
        out = out + s * 0.15 * _laplacian(out)
    return out


def face_normals(verts, tris):
    """Area-weighted vertex normals (n,3), unit length, 0 on isolated vertices."""
    v, t = np.asarray(verts, float), np.asarray(tris, int)
    if len(v) == 0 or len(t) == 0:
        return np.zeros((len(v), 3))
    p = v[t]
    fn = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, t[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return np.where(ln > 1e-16, n / ln, 0.0)


def laplacian_smooth(verts, tris, iters=4, lam=0.45, volume=True):
    """Umbrella-operator smooth of a triangle mesh, optionally rescaled to preserve signed volume."""
    v = np.asarray(verts, float).copy()
    t = np.asarray(tris, int)
    if len(v) == 0 or len(t) == 0 or iters <= 0:
        return v

    def _vol(p):
        q = p[t]
        return float(np.einsum('ij,ij->', q[:, 0], np.cross(q[:, 1], q[:, 2])) / 6.0)

    v0 = _vol(v) if volume else None
    # adjacency
    n = len(v)
    adj_i = np.concatenate([t[:, 0], t[:, 1], t[:, 2], t[:, 1], t[:, 2], t[:, 0]])
    adj_j = np.concatenate([t[:, 1], t[:, 2], t[:, 0], t[:, 0], t[:, 1], t[:, 2]])
    # unique undirected edges
    pair = np.stack([np.minimum(adj_i, adj_j), np.maximum(adj_i, adj_j)], axis=1)
    pair = np.unique(pair, axis=0)
    i, j = pair[:, 0], pair[:, 1]
    deg = np.bincount(np.concatenate([i, j]), minlength=n).astype(float)
    for _ in range(int(iters)):
        acc = np.zeros_like(v)
        np.add.at(acc, i, v[j])
        np.add.at(acc, j, v[i])
        avg = np.where(deg[:, None] > 0, acc / np.maximum(deg[:, None], 1.0), v)
        v = (1.0 - lam) * v + lam * avg
    if volume and v0 is not None and abs(v0) > 1e-16:
        v1 = _vol(v)
        if abs(v1) > 1e-16 and (v0 * v1) > 0:
            c = v.mean(0)
            v = c + (v - c) * (abs(v0 / v1) ** (1.0 / 3.0))
    return v


def signed_volume(verts, tris):
    if len(verts) == 0 or len(tris) == 0:
        return 0.0
    p = np.asarray(verts, float)[np.asarray(tris, int)]
    return float(np.einsum('ij,ij->', p[:, 0], np.cross(p[:, 1], p[:, 2])) / 6.0)


def solid_body_surface(x, colors=None, dx=0.01, radius=2.2, iso=0.35, cell=0.5,
                       tension=0.4, smooth=5, weight=None):
    """Mesh a continuous solid-body fluid surface from particles.

    Returns (verts, tris, vcolors, normals, info) where info has volume, components, coverage
    (fraction of particles enclosed by a loose bounding test / non-zero volume flag), and the
    density grid AABB.  The surface is a single smooth body when the particles form a packed
    liquid; separated clusters stay separate.
    """
    x = np.asarray(x, float)
    empty = (np.zeros((0, 3)), np.zeros((0, 3), int), None, np.zeros((0, 3)),
             dict(volume=0.0, components=0, coverage=0.0, n_verts=0, n_tris=0))
    if len(x) == 0:
        return empty
    fields = splat_fields(x, dx=dx, radius=radius, cell=cell, colors=colors, weight=weight, blur=0.9)
    dens = apply_tension(fields['dens'], strength=tension)
    # field is negative inside the fluid (surface nets convention)
    field = float(iso) - dens
    # keep a positive outer rim so the surface closes
    field[0, :, :] = np.maximum(field[0, :, :], 0.05)
    field[-1, :, :] = np.maximum(field[-1, :, :], 0.05)
    field[:, 0, :] = np.maximum(field[:, 0, :], 0.05)
    field[:, -1, :] = np.maximum(field[:, -1, :], 0.05)
    field[:, :, 0] = np.maximum(field[:, :, 0], 0.05)
    field[:, :, -1] = np.maximum(field[:, :, -1], 0.05)
    verts, tris = cavity.surface_nets(field, fields['lo'], fields['h'])
    if len(verts) == 0:
        return empty
    verts = laplacian_smooth(verts, tris, iters=int(smooth), lam=0.4, volume=True)
    nrm = face_normals(verts, tris)
    vc = None
    if fields['colors'] is not None:
        h = fields['h']
        lo = fields['lo']
        dims = np.array(fields['dens'].shape)
        vi = np.clip(np.round((verts - lo) / h).astype(int), 0, dims - 1)
        den = fields['dens'][vi[:, 0], vi[:, 1], vi[:, 2]]
        vc = fields['colors'][vi[:, 0], vi[:, 1], vi[:, 2]] / np.maximum(den, 1e-6)[:, None]
        vc = np.clip(vc, 0.0, 1.0)
    vol = abs(signed_volume(verts, tris))
    comp = mesh_components(len(verts), tris)
    # coverage: fraction of particles that fall inside the iso (density >= iso at their cell)
    h, lo, dims = fields['h'], fields['lo'], np.array(fields['dens'].shape)
    gi = np.clip(np.round((x - lo) / h).astype(int), 0, dims - 1)
    inside = fields['dens'][gi[:, 0], gi[:, 1], gi[:, 2]] >= float(iso) * 0.5
    coverage = float(inside.mean()) if len(x) else 0.0
    info = dict(volume=vol, components=int(comp), coverage=coverage, n_verts=len(verts), n_tris=len(tris),
                dens_max=float(fields['dens'].max()), aabb=(fields['lo'], fields['lo'] + fields['h'] * dims))
    return verts, tris, vc, nrm, info


class SolidSurfaceBuilder:
    """Frame-coherent solid-body surface (same idea as fluidsurface.SurfaceBuilder, Wendland + tension).

    `update(x, colors, frame=)` returns (verts, tris, vcolors) so it can replace SurfaceBuilder at
    the call site.  Asking for the same frame twice is cached; a jump in time resets the blend.
    """

    def __init__(self, dx, radius=2.2, iso=0.35, cell=0.5, tension=0.4, smooth=4, alpha=0.55):
        self.dx = float(dx)
        self.radius, self.iso, self.cell, self.tension, self.smooth = float(radius), float(iso), float(cell), float(tension), int(smooth)
        self.alpha = float(alpha)
        self.prev = None
        self.frame = None
        self.last = None
        self.last_full = None

    def reset(self):
        self.prev = self.frame = self.last = self.last_full = None

    def update(self, x, colors, frame=None):
        if frame is not None and frame == self.frame and self.last is not None:
            return self.last
        if frame is not None and self.frame is not None and abs(frame - self.frame) != 1:
            self.prev = None
        x = np.asarray(x, float)
        if len(x) == 0:
            self.prev = None
            self.frame = frame
            self.last = (np.zeros((0, 3)), np.zeros((0, 3), int), None)
            self.last_full = (*self.last, np.zeros((0, 3)), dict(volume=0.0, components=0, coverage=0.0, n_verts=0, n_tris=0))
            return self.last
        colors = np.asarray(colors, float) if colors is not None else np.ones((len(x), 3)) * 0.5
        fields = splat_fields(x, dx=self.dx, radius=self.radius, cell=self.cell, colors=colors, blur=0.9)
        dens, csum, lo, h = fields['dens'], fields['colors'], fields['lo'], fields['h']
        if self.prev is not None:
            plo, pd, pc, ph = self.prev
            # resample previous onto the current grid (they share the same world-aligned lattice when h matches)
            if abs(ph - h) < 1e-12:
                # embed both on a common origin
                p0 = np.minimum(lo, plo)
                # shift current
                def _embed(src, slo, shape_extra=None):
                    off = np.round((slo - p0) / h).astype(int)
                    dims = np.maximum(off + np.array(src.shape[:3]), np.array(dens.shape))  # placeholder
                    return off, src
                # simpler: just use the current frame if grids drifted a lot; otherwise blend overlapping region
                dens = self.alpha * dens + (1.0 - self.alpha) * _resample(pd, plo, ph, lo, h, dens.shape)
                if csum is not None and pc is not None:
                    csum = self.alpha * csum + (1.0 - self.alpha) * _resample(pc, plo, ph, lo, h, csum.shape)
        self.prev = (lo, dens, csum, h)
        dens_t = apply_tension(dens, strength=self.tension)
        field = self.iso - dens_t
        field[0, :, :] = np.maximum(field[0, :, :], 0.05)
        field[-1, :, :] = np.maximum(field[-1, :, :], 0.05)
        field[:, 0, :] = np.maximum(field[:, 0, :], 0.05)
        field[:, -1, :] = np.maximum(field[:, -1, :], 0.05)
        field[:, :, 0] = np.maximum(field[:, :, 0], 0.05)
        field[:, :, -1] = np.maximum(field[:, :, -1], 0.05)
        verts, tris = cavity.surface_nets(field, lo, h)
        if len(verts):
            verts = laplacian_smooth(verts, tris, iters=self.smooth, lam=0.4, volume=True)
            nrm = face_normals(verts, tris)
            vc = None
            if csum is not None:
                dims = np.array(dens.shape)
                vi = np.clip(np.round((verts - lo) / h).astype(int), 0, dims - 1)
                den = dens[vi[:, 0], vi[:, 1], vi[:, 2]]
                vc = np.clip(csum[vi[:, 0], vi[:, 1], vi[:, 2]] / np.maximum(den, 1e-6)[:, None], 0.0, 1.0)
            vol = abs(signed_volume(verts, tris))
            info = dict(volume=vol, components=int(mesh_components(len(verts), tris)), coverage=1.0 if vol > 0 else 0.0,
                        n_verts=len(verts), n_tris=len(tris))
        else:
            nrm, vc, info = np.zeros((0, 3)), None, dict(volume=0.0, components=0, coverage=0.0, n_verts=0, n_tris=0)
        self.last = (verts, tris, vc)
        self.last_full = (verts, tris, vc, nrm, info)
        self.frame = frame
        return self.last


def _resample(src, slo, sh, tlo, th, tshape):
    """Nearest-neighbour resample of a grid from (slo, sh) onto (tlo, th, tshape)."""
    out = np.zeros(tshape, dtype=src.dtype)
    # only when cell sizes match (the usual case: world-aligned fixed h)
    if abs(sh - th) > 1e-9:
        return out
    off = np.round((slo - tlo) / th).astype(int)
    src_lo = np.maximum(0, -off)
    dst_lo = np.maximum(0, off)
    src_hi = np.minimum(src.shape[:3], src_lo + (np.array(tshape[:3]) - dst_lo))
    dst_hi = dst_lo + (src_hi - src_lo)
    if np.any(src_hi <= src_lo) or np.any(dst_hi <= dst_lo):
        return out
    sl = tuple(slice(int(a), int(b)) for a, b in zip(src_lo, src_hi))
    dl = tuple(slice(int(a), int(b)) for a, b in zip(dst_lo, dst_hi))
    out[dl] = src[sl]
    return out


def upsample_volume_points(x, *, dx, radius=2.4, cell=0.7, temp=None, vel=None, weight=None, iso=0.08):
    """Turn a sparse particle cloud into a dense, smoothly blended point set (one point per occupied voxel).

    Used to feed Blender Points-to-Volume so a combustion plume or a wind-tunnel slipstream renders as a
    continuous volume rather than disconnected streaks.  Returns (P, density, temp_or_None, vel_or_None).
    """
    fields = splat_fields(x, dx=dx, radius=radius, cell=cell, temp=temp, vel=vel, weight=weight, blur=1.0)
    dens, lo, h = fields['dens'], fields['lo'], fields['h']
    mask = dens >= float(iso)
    if not mask.any():
        return (np.zeros((0, 3)), np.zeros(0), None if temp is None else np.zeros(0),
                None if vel is None else np.zeros((0, 3)))
    ii = np.argwhere(mask)
    P = lo + (ii + 0.5) * h
    d = dens[mask]
    T = fields['temp'][mask] / np.maximum(d, 1e-6) if fields['temp'] is not None else None
    V = fields['vel'][mask] / np.maximum(d, 1e-6)[:, None] if fields['vel'] is not None else None
    return P, d, T, V


def airflow_field(body_P, body_N, body_T=None, *, v_inf, T_inf, bounds, cell=0.04, delta=0.06, wake_len=None):
    """Procedural continuous slipstream / heated boundary-layer volume around a collision body.

    A regular grid is filled with a slip condition against the closest surface sample (the normal
    component of v_inf is cancelled in a layer of thickness `delta`), a slower wake downstream of
    the body, and a temperature excess equal to the wall temperature shed into that layer and
    advected with the flow.  Returns dict(P, dens, temp, vel, lo, h, shape) of the occupied voxels
    (and the full grids under keys dens_grid, temp_grid, vel_grid).
    """
    body_P = np.asarray(body_P, float)
    body_N = np.asarray(body_N, float)
    if len(body_P) == 0:
        return dict(P=np.zeros((0, 3)), dens=np.zeros(0), temp=np.zeros(0), vel=np.zeros((0, 3)),
                    lo=np.zeros(3), h=cell, shape=(0, 0, 0), dens_grid=None, temp_grid=None, vel_grid=None)
    lo = np.asarray(bounds[0], float)
    hi = np.asarray(bounds[1], float)
    h = float(cell)
    dims = np.maximum(np.ceil((hi - lo) / h).astype(int), 1)
    # cap
    if int(np.prod(dims)) > 1_200_000:
        h *= (int(np.prod(dims)) / 1_200_000) ** (1.0 / 3.0)
        dims = np.maximum(np.ceil((hi - lo) / h).astype(int), 1)
    gx = lo[0] + (np.arange(dims[0]) + 0.5) * h
    gy = lo[1] + (np.arange(dims[1]) + 0.5) * h
    gz = lo[2] + (np.arange(dims[2]) + 0.5) * h
    # sample on a flattened grid, batched closest-point
    XX, YY, ZZ = np.meshgrid(gx, gy, gz, indexing='ij')
    Q = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)
    vi = np.asarray(v_inf, float).reshape(3)
    speed = float(np.linalg.norm(vi)) or 1.0
    vhat = vi / speed
    # closest body sample (chunked to stay memory-safe)
    nq = len(Q)
    d2 = np.empty(nq)
    jmin = np.empty(nq, np.int64)
    chunk = 8000
    # downsample body samples if needed
    P = body_P
    N = body_N
    Ts = np.full(len(P), float(T_inf)) if body_T is None else np.asarray(body_T, float)
    if len(P) > 2500:
        rng = np.random.default_rng(0)
        pick = rng.choice(len(P), 2500, replace=False)
        P, N, Ts = P[pick], N[pick], Ts[pick]
    for a in range(0, nq, chunk):
        q = Q[a:a + chunk]
        # (c, m)
        D = ((q[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        j = np.argmin(D, axis=1)
        d2[a:a + chunk] = D[np.arange(len(q)), j]
        jmin[a:a + chunk] = j
    dist = np.sqrt(np.maximum(d2, 0.0))
    nrm = N[jmin]
    Tw = Ts[jmin]
    # inside: signed distance ~ dot(Q-P, n); body normals point out
    signed = np.einsum('ij,ij->i', Q - P[jmin], nrm)
    inside = signed < 0.0
    # slip: cancel the incoming normal component close to the wall
    layer = np.exp(-np.maximum(signed, 0.0) / max(float(delta), 1e-4))
    vn = np.einsum('j,ij->i', vi, nrm)
    vel = np.broadcast_to(vi, Q.shape).copy()
    hit = vn < 0.0
    vel[hit] = vel[hit] - (layer[hit] * vn[hit])[:, None] * nrm[hit]
    # wake: voxels downstream of the body, within a radius that grows slowly
    rel = Q - P.mean(0)
    along = rel @ vhat
    rad = np.linalg.norm(rel - along[:, None] * vhat, axis=1)
    body_rad = np.linalg.norm((P - P.mean(0)) - ((P - P.mean(0)) @ vhat)[:, None] * vhat, axis=1)
    R0 = float(np.percentile(body_rad, 90)) if len(body_rad) else 0.2
    wake_len = float(wake_len if wake_len is not None else 3.0 * R0 + 0.4)
    in_wake = (along > 0.0) & (along < wake_len) & (rad < (R0 * (1.0 + 0.35 * along / max(wake_len, 1e-6))))
    vel[in_wake] *= (0.45 + 0.55 * (along[in_wake] / max(wake_len, 1e-6)))[:, None]
    # density: solid slipstream everywhere outside the body, slightly denser in the boundary layer and wake
    dens = np.ones(nq)
    dens[inside] = 0.0
    dens += 0.85 * layer * (~inside)
    dens[in_wake] += 0.55
    # temperature: ambient plus wall heat shed into the layer and carried into the wake
    temp = np.full(nq, float(T_inf))
    shed = np.maximum(Tw - float(T_inf), 0.0)
    temp += shed * layer * 0.85
    # wake carries a decaying fraction of the heat
    temp[in_wake] += shed[in_wake] * 0.45 * np.exp(-along[in_wake] / max(0.5 * wake_len, 1e-6))
    # drop empty / inside voxels for the point cloud
    keep = (~inside) & (dens > 0.15)
    shape = tuple(int(x) for x in dims)
    dens_g = dens.reshape(shape)
    temp_g = temp.reshape(shape)
    vel_g = vel.reshape(shape + (3,))
    dens_g = gaussian_blur(dens_g, 0.7)
    temp_g = gaussian_blur(temp_g, 0.7)
    vel_g = gaussian_blur(vel_g, 0.7)
    dens, temp, vel = dens_g.ravel(), temp_g.ravel(), vel_g.reshape(-1, 3)
    keep = (dens > 0.12) & (~inside)
    return dict(P=Q[keep], dens=dens[keep], temp=temp[keep], vel=vel[keep], lo=lo, h=h, shape=shape,
                dens_grid=dens_g, temp_grid=temp_g, vel_grid=vel_g)


def infer_source_kind(x):
    """Classify a particle set as a curve-like filament, a thin sheet, or a filled volume (PCA)."""
    x = np.asarray(x, float)
    if len(x) < 8:
        return 'volume'
    c = x - x.mean(0)
    cov = (c.T @ c) / max(len(x) - 1, 1)
    w = np.sort(np.clip(np.linalg.eigvalsh(cov), 0.0, None))
    if w[2] <= 1e-16:
        return 'volume'
    # one long axis, two tiny ones: a tendril / curve source
    if (w[0] + w[1]) / w[2] < 0.14:
        return 'curve'
    # two long axes, one tiny: a thin surface / sheet source
    if w[0] / max(w[1] + w[2], 1e-16) < 0.08:
        return 'surface'
    return 'volume'


# tighter kernels keep curve / thin-surface sources sharp instead of bloating them into a volume
_PLUME_PRESET = {
    'curve':   dict(radius=1.35, cell=0.42, blur=0.48),
    'surface': dict(radius=1.65, cell=0.50, blur=0.62),
    'volume':  dict(radius=2.60, cell=0.65, blur=1.05),
}


def blend_plume(x, temp, *, dx, soot=None, fade=None, radius=None, cell=None, iso=0.05, source='auto', blur=None):
    """Smooth combustion-plume / wisp volume from particles.

    `source` is 'auto', 'curve', 'surface' or 'volume'.  Curve and thin-surface sources (Houdini-style
    emission along a line or sheet) use a tighter Wendland kernel so the tendril stays sharp with far
    fewer particles than a filled-volume emitter.  Returns (P, density, temp, soot_or_None).
    """
    kind = infer_source_kind(x) if source == 'auto' else source
    pr = _PLUME_PRESET.get(kind, _PLUME_PRESET['volume'])
    radius = pr['radius'] if radius is None else radius
    cell = pr['cell'] if cell is None else cell
    blur = pr['blur'] if blur is None else blur
    w = np.ones(len(x))
    if fade is not None:
        w = w * np.clip(np.asarray(fade, float), 0.05, 1.0)
    fields = splat_fields(x, dx=dx, radius=radius, cell=cell, temp=temp, weight=w, blur=blur)
    if soot is not None:
        sfields = splat_fields(x, dx=dx, radius=radius, cell=cell, temp=soot, weight=w, blur=1.05)
        soot_grid = sfields['temp']
    else:
        soot_grid = None
    dens, lo, h = fields['dens'], fields['lo'], fields['h']
    mask = dens >= float(iso)
    if not mask.any():
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0), None
    ii = np.argwhere(mask)
    P = lo + (ii + 0.5) * h
    d = dens[mask]
    T = fields['temp'][mask] / np.maximum(d, 1e-6) if fields['temp'] is not None else np.full(len(P), 300.0)
    S = soot_grid[mask] / np.maximum(d, 1e-6) if soot_grid is not None else None
    return P, d, T, S


# ----------------------------------------------------------------------------- curve / thin-surface sourcing (wispy plumes)
def source_from_curve(V, step, *, rng=None, jitter=0.0):
    """Particle seeds along a polyline (curve) at `step` spacing.

    Curve / thin-surface sourcing puts every particle where the detail is: a plume seeded from a
    line or a thin sheet keeps sharp, wispy filaments with far fewer particles than a filled
    volume (Houdini-style curve sourcing).  V (n,3) ordered polyline points.
    """
    V = np.asarray(V, float)
    if len(V) < 2:
        return V.copy()
    seg = np.linalg.norm(np.diff(V, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    k = max(int(s[-1] / max(float(step), 1e-9)), 1)
    t = np.linspace(0.0, s[-1], k + 1)
    P = np.stack([np.interp(t, s, V[:, a]) for a in range(3)], axis=1)
    if jitter > 0.0:
        rng = rng or np.random.default_rng(0)
        P = P + rng.normal(scale=float(jitter), size=P.shape)
    return P


def source_from_thin_surface(V, T, step, *, rng=None, thickness=0.0):
    """Area-weighted seeds on a thin (open) surface at roughly `step` spacing, optionally given a small thickness."""
    V, T = np.asarray(V, float), np.asarray(T, int)
    rng = rng or np.random.default_rng(0)
    if len(T) == 0:
        return np.zeros((0, 3))
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    n = max(int(area.sum() / max(float(step) ** 2, 1e-12)), 1)
    tri = rng.choice(len(T), n, p=area / max(area.sum(), 1e-16))
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    P = a[tri] + u[:, None] * (b[tri] - a[tri]) + v[:, None] * (c[tri] - a[tri])
    if thickness > 0.0:
        nrm = np.cross(b[tri] - a[tri], c[tri] - a[tri])
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-16)
        P = P + nrm * rng.uniform(-0.5, 0.5, n)[:, None] * float(thickness)
    return P


def wisp_detail(P, dens, h):
    """Sharpness of a blended volume: mean |grad density| / mean density over occupied voxels (higher = crisper wisps)."""
    if len(P) < 4:
        return 0.0
    ii = np.round((P - P.min(0)) / h).astype(int)
    dims = ii.max(0) + 3
    g = np.zeros(dims)
    g[ii[:, 0] + 1, ii[:, 1] + 1, ii[:, 2] + 1] = dens
    gx, gy, gz = np.gradient(g, h)
    gm = np.sqrt(gx * gx + gy * gy + gz * gz)[ii[:, 0] + 1, ii[:, 1] + 1, ii[:, 2] + 1]
    return float(gm.mean() / max(float(dens.mean()), 1e-9))


# ----------------------------------------------------------------------------- whitewater (foam / spray / bubbles)
def whitewater(x, v, *, dx, normals_hint=None, body_P=None, body_N=None, speed_ref=None,
               foam_thresh=0.35, spray_thresh=0.6):
    """Classify liquid particles into foam / spray / bubble emitters (Houdini FLIP-whitewater style).

    Uses three cheap per-particle potentials, each in 0..1:
      trapped air  ~ relative velocity divergence with neighbours (turbulent mixing)
      wave crest   ~ low neighbour count (surface) and upward / outward velocity
      impact       ~ speed toward a nearby collision surface (body_P / body_N)
    Returns dict(foam (n,) 0..1, spray mask, foam mask, bubble mask).
    """
    x = np.asarray(x, float)
    v = np.asarray(v, float)
    n = len(x)
    if n == 0:
        z = np.zeros(0)
        return dict(foam=z, spray=z.astype(bool), foam_mask=z.astype(bool), bubble=z.astype(bool))
    from . import thermal as TH
    h = 2.2 * float(dx)
    I, J = TH.cross_pairs(x, x, h)
    keep = I != J
    I, J = I[keep], J[keep]
    cnt = np.bincount(I, minlength=n).astype(float)
    full = max(np.percentile(cnt, 90), 1.0)
    surface = np.clip(1.0 - cnt / full, 0.0, 1.0)
    dv = np.linalg.norm(v[I] - v[J], axis=1)
    turb = np.bincount(I, dv, n) / np.maximum(cnt, 1.0)
    sp = np.linalg.norm(v, axis=1)
    vref = float(speed_ref) if speed_ref else max(float(np.percentile(sp, 95)), 1e-6)
    trapped = np.clip(turb / vref, 0.0, 1.0)
    crest = surface * np.clip(sp / vref, 0.0, 1.0)
    impact = np.zeros(n)
    if body_P is not None and len(body_P):
        bI, bJ = TH.cross_pairs(x, np.asarray(body_P, float), 2.5 * float(dx))
        if len(bI):
            nn = np.asarray(body_N, float)[bJ]
            into = np.clip(-np.einsum('ij,ij->i', v[bI], nn) / vref, 0.0, 1.0)
            impact = np.maximum(impact, np.bincount(bI, into, n) / np.maximum(np.bincount(bI, minlength=n), 1))
    foam = np.clip(0.45 * trapped + 0.35 * crest + 0.6 * impact, 0.0, 1.0)
    spray = (surface > 0.55) & (foam > spray_thresh)
    foam_mask = (foam > foam_thresh) & ~spray
    bubble = (surface < 0.2) & (trapped > foam_thresh)
    return dict(foam=foam, spray=spray, foam_mask=foam_mask, bubble=bubble)


def foam_on_surface(verts, x, foam, dx):
    """Per-vertex foam weight (0..1) on a blended liquid surface from particle foam potentials."""
    from . import thermal as TH
    verts = np.asarray(verts, float)
    if len(verts) == 0 or len(x) == 0:
        return np.zeros(len(verts))
    h = 2.5 * float(dx)
    I, J = TH.cross_pairs(verts, np.asarray(x, float), h)
    if len(I) == 0:
        return np.zeros(len(verts))
    w = np.clip(1.0 - np.linalg.norm(verts[I] - x[J], axis=1) / h, 0.0, None) ** 2
    num = np.bincount(I, w * np.asarray(foam)[J], len(verts))
    den = np.bincount(I, w, len(verts))
    return np.clip(np.where(den > 0, num / np.maximum(den, 1e-12), 0.0), 0.0, 1.0)



def wispy_streamlines(body_P, body_N, *, v_inf, bounds, n_curves=28, steps=90, dt=None, seed=3):
    """Curve-sourced twisting slipstream tendrils around a collision body.

    A rake of curves is seeded upstream (far fewer points than a filled volume), advected with a
    slip condition, a slower swirling wake, and a gentle helical twist, then resampled as thin
    curve sources.  Returns dict(curves=list of (k,3) polylines, P, dens, temp) for a sharp blended
    wisp volume.  This is the Houdini "emit from a curve, keep the tendril" path for wind-tunnel air.
    """
    body_P = np.asarray(body_P, float)
    body_N = np.asarray(body_N, float)
    lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    vi = np.asarray(v_inf, float).reshape(3)
    speed = float(np.linalg.norm(vi)) or 1.0
    vhat = vi / speed
    dt = float(dt) if dt is not None else 0.35 * (hi[0] - lo[0]) / (speed * max(steps, 1))
    rng = np.random.default_rng(seed)
    # downsample the body for the closest-point query
    P = body_P
    N = body_N
    if len(P) > 1800:
        pick = rng.choice(len(P), 1800, replace=False)
        P, N = P[pick], N[pick]
    ctr = P.mean(0) if len(P) else 0.5 * (lo + hi)
    # rake in the YZ plane at the upstream bound
    n_y, n_z = max(int(np.sqrt(n_curves * 1.6)), 4), max(int(n_curves / max(int(np.sqrt(n_curves * 1.6)), 4)), 3)
    ys = np.linspace(lo[1] * 0.85 + ctr[1] * 0.15, hi[1] * 0.85 + ctr[1] * 0.15, n_y)
    zs = np.linspace(lo[2] * 0.7 + ctr[2] * 0.3, hi[2] * 0.7 + ctr[2] * 0.3, n_z)
    seeds = np.array([[lo[0] + 0.02, y, z] for y in ys for z in zs])
    # keep seeds that would graze the body (not far in the free stream)
    if len(P):
        dmin = np.sqrt(((seeds[:, None, :] - P[None, :, :]) ** 2).sum(-1).min(1))
        R0 = float(np.percentile(np.linalg.norm(P - ctr, axis=1), 88))
        seeds = seeds[dmin < 1.35 * R0]
    if len(seeds) == 0:
        return dict(curves=[], P=np.zeros((0, 3)), dens=np.zeros(0), temp=np.zeros(0))
    curves = []
    for s0 in seeds:
        q = s0.copy()
        path = [q.copy()]
        swirl = 0.0
        for k in range(int(steps)):
            # closest body sample
            d2 = ((q - P) ** 2).sum(-1)
            j = int(np.argmin(d2))
            rel = q - P[j]
            dist = float(np.sqrt(max(d2[j], 0.0)))
            nrm = N[j]
            signed = float(rel @ nrm)
            if signed < 0.02:
                q = q + nrm * (0.03 - signed)
                path.append(q.copy())
                continue
            layer = np.exp(-max(signed, 0.0) / 0.08)
            vel = vi.copy()
            vn = float(vel @ nrm)
            if vn < 0.0:
                vel = vel - layer * vn * nrm
            # helical wake twist downstream of the body
            along = float((q - ctr) @ vhat)
            if along > 0.0:
                radial = (q - ctr) - along * vhat
                rlen = float(np.linalg.norm(radial))
                if rlen > 1e-6:
                    twist = np.cross(vhat, radial / rlen)
                    swirl = 0.22 * speed * np.exp(-along / 1.2) * (0.6 + 0.4 * np.sin(along * 7.0 + s0[1] * 9.0))
                    vel = vel * (0.55 + 0.45 * min(along / 0.8, 1.0)) + twist * swirl
            q = q + vel * dt
            path.append(q.copy())
            if q[0] > hi[0] or abs(q[1]) > hi[1] + 0.1 or q[2] < lo[2] - 0.1 or q[2] > hi[2] + 0.1:
                break
        curves.append(np.array(path))
    # resample each curve as a thin particle source and blend with the curve kernel
    pts = []
    for c in curves:
        if len(c) < 4:
            continue
        pts.append(source_from_curve(c, step=0.035, rng=rng, jitter=0.004))
    if not pts:
        return dict(curves=curves, P=np.zeros((0, 3)), dens=np.zeros(0), temp=np.zeros(0))
    X = np.vstack(pts)
    Pv, dens, temp, _ = blend_plume(X, np.full(len(X), 300.0), dx=0.03, source='curve', iso=0.05)
    return dict(curves=curves, P=Pv, dens=dens, temp=temp, seeds=len(seeds), n_source=len(X))
