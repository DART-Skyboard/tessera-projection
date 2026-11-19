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
