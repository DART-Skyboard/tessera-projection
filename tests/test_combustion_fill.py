"""Combustion containers of non-cylindrical shape fill to the right level (python3 tests/test_combustion_fill.py).

The combustion world fills a container from its quantity (volume x density at the tank's pressure / temperature), by sampling the real cavity through the
same collision-body projection as Fluid mode and keeping the lowest points along the Environment gravity."""
import os
import sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fm_helpers import lathe  # noqa: E402  (also puts arc_flow on the path)
from core import sdf, world, fluids  # noqa: E402
from core import primitives as MS  # noqa: E402

fails = [0]


def ok(c, msg):
    print(("PASS " if c else "FAIL ") + msg, flush=True)
    if not c:
        fails[0] += 1


def tank_world(prof, pct, P=5e5):
    V, T = lathe(prof)
    fl = [dict(name='water', spec=fluids.FLUIDS['WATER'], temp=293.15, color=(0, 0, 1))]
    parts = [world.PartSpec('tank', sdf.PartField(V, T, 'SHELL', 40), np.eye(4), 'CONTAINER', fluid=0, fill_value=pct, pressure=P, n_particles=3000)]
    return world.World(parts, fl, world.Env(), world.Params(continuous=False, net_res=36, vmax=40, seed=3)), V, T


def level_of(prof, frac):
    zs = np.linspace(prof[0][1], prof[-1][1], 4001)
    p = np.asarray(prof, float)
    vz = np.cumsum(np.pi * np.interp(zs, p[:, 1], p[:, 0]) ** 2) * (zs[1] - zs[0])
    return zs[np.searchsorted(vz, frac * vz[-1])]


CONE = [(0, 0), (0.9, 0), (0.4, 3.0), (0, 3.0)]
SPH = [(0, -1.0)] + [(np.sin(a), -np.cos(a)) for a in np.linspace(0.12, np.pi - 0.12, 17)] + [(0, 1.0)]
for name, prof in (("cone", CONE), ("sphere", SPH)):
    for pct in (30.0, 60.0):
        w, V, T = tank_world(prof, pct)
        s = w.sources[0]
        alive = w.state == 1
        z = w.pos[alive, 2]
        zt = level_of(prof, pct / 100.0)
        # expected mass = filled volume x density at the tank's pressure and temperature
        rho = fluids.density_at(fluids.FLUIDS['WATER'], 5e5, 293.15)
        ok(abs(z.max() - zt) < 0.12 * (prof[-1][1] - prof[0][1]) and z.max() <= zt + 0.1,
           "combustion %s tank %d%%: particles reach z=%.2f, expected level %.2f" % (name, pct, z.max(), zt))
        ok(abs(s['fill_vol'] / s['cav_vol'] - pct / 100.0) < 1e-6 and abs(s['mass'] - s['fill_vol'] * rho) < 1e-6 * s['mass'] + 1e-9,
           "combustion %s tank: mass %.1f kg = fill volume %.3f m3 x density %.1f kg/m3" % (name, s['mass'], s['fill_vol'], rho))
        r = np.hypot(w.pos[alive, 0], w.pos[alive, 1])
        rc = np.interp(w.pos[alive, 2], np.asarray(prof)[:, 1], np.asarray(prof)[:, 0])
        ok(float((r - rc).max()) < 0.06, "combustion %s tank: particles stay inside the contour (max overshoot %.3f)" % (name, float((r - rc).max())))
print("FAILURES:", fails[0])
raise SystemExit(1 if fails[0] else 0)
