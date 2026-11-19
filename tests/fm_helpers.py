"""Shared helpers for the Fluid mode NumPy tests (no bpy)."""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'arc_flow'))
from core import fluidmode as FM  # noqa: E402
from core import sdf as SDF  # noqa: E402


def box(lo, hi):
    """Closed box mesh, outward normals."""
    lo, hi = np.array(lo, float), np.array(hi, float)
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    ctr = c.mean(0)
    out = []
    for a, b, cc, d in quads:
        for tr in ((a, b, cc), (a, cc, d)):
            tr = np.array(tr)
            p = c[tr]
            n = np.cross(p[1] - p[0], p[2] - p[0])
            out.append(tr if np.dot(n, p.mean(0) - ctr) > 0 else tr[::-1])
    return c, np.array(out)


def container(lo, hi, res=40, matrix=None, fluids=None, friction=0.2):
    V, T = box(lo, hi)
    return FM.Body(SDF.PartField(V, T, 'SHELL', resolution=res, level=2), matrix, 'CONTAINER', fluids, friction)


def collider(V, T, res=32, matrix=None, friction=0.2):
    return FM.Body(SDF.PartField(V, T, 'SOLID' if False else 'AUTO', resolution=res, level=2), matrix, 'COLLIDER', None, friction)
