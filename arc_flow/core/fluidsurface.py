"""Fluid surface from particles (no bpy): a smoothed particle density field is sampled on a grid and meshed with the extension's surface nets, so blobs that touch
become ONE welded surface and separate blobs stay separate.  Per-vertex colour comes from the particles' composition colours (density weighted)."""
import numpy as np

from . import cavity


def fluid_surface(x, colors=None, dx=0.01, radius=1.6, iso=0.5, cell=None):
    """x (n,3) particle positions, colors (n,3) or None.  Returns (verts (m,3), tris (k,3), vcolors (m,3) | None).
    radius: kernel support in particle spacings; iso: density level (1 = full lattice density) that defines the surface."""
    x = np.asarray(x, float)
    if len(x) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), int), None
    h = (cell or 0.6 * dx)
    R = radius * dx
    pad = int(np.ceil(R / h)) + 2
    lo = x.min(0) - pad * h
    hi = x.max(0) + pad * h
    dims = np.ceil((hi - lo) / h).astype(int) + 1
    dens = np.zeros(dims)
    csum = np.zeros(tuple(dims) + (3,)) if colors is not None else None
    r = int(np.ceil(R / h))
    g = (x - lo) / h
    base = np.floor(g).astype(int)
    offs = np.arange(-r, r + 2)
    for ox in offs:
        for oy in offs:
            for oz in offs:
                idx = base + np.array([ox, oy, oz])
                ok = np.all((idx >= 0) & (idx < dims), axis=1)
                d = np.linalg.norm((idx - g) * h, axis=1)
                w = np.clip(1.0 - d / R, 0.0, None) ** 3
                m = ok & (w > 0)
                if m.any():
                    ii = idx[m]
                    np.add.at(dens, (ii[:, 0], ii[:, 1], ii[:, 2]), w[m])
                    if csum is not None:
                        np.add.at(csum, (ii[:, 0], ii[:, 1], ii[:, 2]), w[m, None] * colors[m])
    # normalise so that a full lattice of spacing dx has density 1 in the bulk
    lat = np.arange(-r - 1, r + 2) * dx
    X, Y, Z = np.meshgrid(lat, lat, lat, indexing='ij')
    dd = np.sqrt(X * X + Y * Y + Z * Z)
    full = float((np.clip(1.0 - dd / R, 0.0, None) ** 3).sum())
    dens /= max(full, 1e-12)
    field = iso - dens
    verts, tris = cavity.surface_nets(field, lo, h)
    if len(verts) == 0:
        return verts, tris, None
    vc = None
    if csum is not None:
        vi = np.clip(np.round((verts - lo) / h).astype(int), 0, dims - 1)
        dv = dens[vi[:, 0], vi[:, 1], vi[:, 2]] * full
        vc = csum[vi[:, 0], vi[:, 1], vi[:, 2]] / np.maximum(dv, 1e-9)[:, None]
    return verts, tris, vc


def _blur_axis(a, w, axis):
    r = (len(w) - 1) // 2
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r, r)
    ap = np.pad(a, pad)
    out = np.zeros_like(a)
    n = a.shape[axis]
    for k, wk in enumerate(w):
        sl = [slice(None)] * a.ndim
        sl[axis] = slice(k, k + n)
        out += wk * ap[tuple(sl)]
    return out


def gaussian_blur(a, sigma):
    """Separable Gaussian blur of a 3-D (or 4-D with a trailing channel) array, sigma in cells."""
    r = max(int(np.ceil(3.0 * sigma)), 1)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / max(sigma, 1e-6)) ** 2)
    k /= k.sum()
    for ax in range(3):
        a = _blur_axis(a, k, ax)
    return a


class SurfaceBuilder:
    """Smooth, frame-coherent surface of one or several fluids: all particles are splatted (trilinear) into one density field on a lattice aligned to a fixed
    world grid, blurred with a Gaussian (a metaball-style union: touching blobs of different fluids join with a neck), blended in time (exponential average
    of the density and colour grids, so the surface and the colours change continuously with no flicker) and meshed with surface nets.  Vertex colour is the
    density-weighted colour of the particles, so colours blend across a seam.  Frame cache: asking for the same frame twice returns the same mesh; jumping
    in time resets the blend."""

    def __init__(self, dx, sigma=0.85, iso=0.45, cell=0.55, alpha=0.5):
        self.dx, self.sigma, self.iso, self.cell_f, self.alpha = float(dx), float(sigma), float(iso), float(cell), float(alpha)
        self.h = self.cell_f * self.dx
        self.prev = None
        self.frame = None
        self.last = None

    def reset(self):
        self.prev = None
        self.frame = None
        self.last = None

    def _fields(self, x, colors):
        h = self.h
        pad = int(np.ceil(3.0 * self.sigma * self.dx / h)) + 3
        lo_i = np.floor(x.min(0) / h).astype(int) - pad
        hi_i = np.floor(x.max(0) / h).astype(int) + pad + 1
        dims = hi_i - lo_i + 1
        g = x / h - lo_i
        i0 = np.floor(g).astype(int)
        f = g - i0
        n = len(x)
        cnt = np.zeros(dims)
        csum = np.zeros(tuple(dims) + (3,))
        for cx in (0, 1):
            for cy in (0, 1):
                for cz in (0, 1):
                    w = (f[:, 0] if cx else 1 - f[:, 0]) * (f[:, 1] if cy else 1 - f[:, 1]) * (f[:, 2] if cz else 1 - f[:, 2])
                    ii = i0 + np.array([cx, cy, cz])
                    flat = (ii[:, 0] * dims[1] + ii[:, 1]) * dims[2] + ii[:, 2]
                    cnt += np.bincount(flat, w, int(np.prod(dims))).reshape(dims)
                    for c in range(3):
                        csum[..., c] += np.bincount(flat, w * colors[:, c], int(np.prod(dims))).reshape(dims)
        sg = self.sigma * self.dx / h
        cnt = gaussian_blur(cnt, sg)
        csum = gaussian_blur(csum, sg)
        # density 1 in the bulk of a lattice of spacing dx: (h/dx)^3 particles per cell
        cnt /= (h / self.dx) ** 3
        csum /= (h / self.dx) ** 3
        return lo_i, cnt, csum

    def _blend(self, lo_i, cnt, csum):
        if self.prev is None:
            return lo_i, cnt, csum
        plo, pc, pcs = self.prev
        lo = np.minimum(lo_i, plo)
        hi = np.maximum(lo_i + np.array(cnt.shape), plo + np.array(pc.shape))
        dims = hi - lo
        A = np.zeros(dims); AC = np.zeros(tuple(dims) + (3,))
        B = np.zeros(dims); BC = np.zeros(tuple(dims) + (3,))
        a0 = lo_i - lo; b0 = plo - lo
        A[a0[0]:a0[0] + cnt.shape[0], a0[1]:a0[1] + cnt.shape[1], a0[2]:a0[2] + cnt.shape[2]] = cnt
        AC[a0[0]:a0[0] + cnt.shape[0], a0[1]:a0[1] + cnt.shape[1], a0[2]:a0[2] + cnt.shape[2]] = csum
        B[b0[0]:b0[0] + pc.shape[0], b0[1]:b0[1] + pc.shape[1], b0[2]:b0[2] + pc.shape[2]] = pc
        BC[b0[0]:b0[0] + pc.shape[0], b0[1]:b0[1] + pc.shape[1], b0[2]:b0[2] + pc.shape[2]] = pcs
        a = self.alpha
        return lo, a * A + (1 - a) * B, a * AC + (1 - a) * BC

    def update(self, x, colors, frame=None):
        """Mesh for the particles x (n,3) with colours (n,3): returns (verts, tris, vertex colours)."""
        if frame is not None and frame == self.frame and self.last is not None:
            return self.last
        if frame is not None and self.frame is not None and abs(frame - self.frame) != 1:
            self.prev = None
        x = np.asarray(x, float)
        if len(x) == 0:
            self.prev = None
            self.frame = frame
            self.last = (np.zeros((0, 3)), np.zeros((0, 3), int), None)
            return self.last
        colors = np.asarray(colors, float) if colors is not None else np.ones((len(x), 3)) * 0.5
        lo_i, cnt, csum = self._fields(x, colors)
        lo_i, cnt, csum = self._blend(lo_i, cnt, csum)
        self.prev = (lo_i, cnt, csum)
        h = self.h
        field = self.iso - cnt
        verts, tris = cavity.surface_nets(field, lo_i * h, h)
        if len(verts) == 0:
            self.last = (verts, tris, None)
        else:
            vi = np.clip(np.round((verts - lo_i * h) / h).astype(int), 0, np.array(cnt.shape) - 1)
            den = cnt[vi[:, 0], vi[:, 1], vi[:, 2]]
            vc = csum[vi[:, 0], vi[:, 1], vi[:, 2]] / np.maximum(den, 1e-6)[:, None]
            self.last = (verts, tris, np.clip(vc, 0.0, 1.0))
        self.frame = frame
        return self.last


def mesh_components(n_verts, tris):
    """Number of connected components of a triangle mesh (vertex union-find)."""
    parent = np.arange(n_verts)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for t in tris:
        r0 = find(t[0])
        for k in (1, 2):
            rk = find(t[k])
            if rk != r0:
                parent[rk] = r0
    used = np.unique(tris.ravel()) if len(tris) else np.zeros(0, int)
    return len({find(v) for v in used})
