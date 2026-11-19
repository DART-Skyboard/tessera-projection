"""Thermal readback core (NumPy only):  python3 tests/test_thermal.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'arc_flow'))
from core import thermal as TH

fails = [0]
def ok(c, m):
    print(("PASS " if c else "FAIL ") + m)
    if not c: fails[0] += 1

rng = np.random.default_rng(1)
# cross_pairs against brute force
P = rng.random((200, 3)); X = rng.random((500, 3)); h = 0.15
I, J = TH.cross_pairs(P, X, h)
D = np.linalg.norm(P[:, None] - X[None], axis=2)
bi, bj = np.nonzero(D < h)
ok(set(zip(I.tolist(), J.tolist())) == set(zip(bi.tolist(), bj.tolist())) and len(I) == len(bi), "cross_pairs equals brute force (%d pairs)" % len(I))
ok(len(TH.cross_pairs(np.zeros((0, 3)), X, h)[0]) == 0, "empty query")

# packed lattice of 330 K air at rest around a 300 K wall: coverage ~1, wall relaxes to 330 K
dx = 0.05
g = np.arange(0.0, 1.0, dx)
X = np.array([[x, y, z] for x in g for y in g for z in g]) + np.array([0, 0, 0.5 * dx])
n = len(X)
T = np.full(n, 330.0); V = np.zeros((n, 3)); cp = np.full(n, 1005.0); k = np.full(n, 0.0257); rho = np.full(n, 1.2); mu = np.full(n, 1.8e-5)
P = np.array([[0.5, 0.5, 0.5 * dx]]); N = np.array([[0, 0, 1.0]])
pr = TH.probe(P, X, V, T, cp, k, rho, mu, 2.2 * dx, dx)
ok(abs(pr['T'][0] - 330.0) < 1e-9, "probe mean temperature")
ok(0.5 < pr['coverage'][0] <= 1.0, "packed half space coverage %.2f" % pr['coverage'][0])
Ts = np.array([300.0])
for _ in range(400):
    Ts, info = TH.surface_step(Ts, P, N, X, V, T, cp, k, rho, mu, 0.1, h=2.2 * dx, dx=dx, ambient=300.0, L=1.0, C_a=50.0)
ok(abs(Ts[0] - 330.0) < 1.0, "wall relaxes to the fluid temperature (%.2f K)" % Ts[0])
# monotone and bounded (no overshoot) for a big step
Ts2, _ = TH.surface_step(np.array([300.0]), P, N, X, V, T, cp, k, rho, mu, 1e6, h=2.2 * dx, dx=dx, ambient=300.0, L=1.0, C_a=500.0)
ok(300.0 <= Ts2[0] <= 330.0 + 1e-6, "unconditionally stable step (%.3f K)" % Ts2[0])

# recovery heating of a fast flow: stagnation temperature rise  r v^2 / 2 cp
V2 = np.tile([200.0, 0, 0], (n, 1))
Ts = np.array([300.0])
for _ in range(2000):
    Ts, info = TH.surface_step(Ts, P, N, X, np.copy(V2), np.full(n, 300.0), cp, k, rho, mu, 0.1, h=2.2 * dx, dx=dx, ambient=300.0, L=1.0, C_a=500.0)
exp = 300.0 + 0.85 * 200.0 ** 2 / (2 * 1005.0)
ok(abs(Ts[0] - exp) < 0.5, "recovery temperature %.2f K (expected %.2f)" % (Ts[0], exp))

# no particles: ambient plus radiation from a heat source, inverse square, facing side only
P3 = np.array([[1.0, 0, 0], [2.0, 0, 0], [-1.0, 0, 0]]); N3 = np.array([[-1.0, 0, 0], [-1.0, 0, 0], [-1.0, 0, 0]])
q = TH.radiative_flux(P3, [((0, 0, 0), 1000.0, 0.01)], 0.7, N3)
ok(abs(q[0] / q[1] - 4.0) < 1e-6 and q[2] == 0.0, "radiation falls with the inverse square and ignores the back side")
Ts = np.full(3, 293.15)
for _ in range(50):
    Ts, _ = TH.surface_step(Ts, P3, N3, np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), 1.0,
                            h=0.1, dx=0.05, ambient=293.15, L=1.0, C_a=4900.0, sources=[((0, 0, 0), 1000.0, 0.01)])
ok(Ts[0] > Ts[1] > 293.15 and abs(Ts[2] - 293.15) < 1e-9, "irradiated wall heats nearest first (%.2f, %.2f, %.2f)" % tuple(Ts))

# re-radiation caps the heating of a wall next to a strong source
T1 = np.array([293.15])
for _ in range(4000):
    T1, _i = TH.surface_step(T1, np.array([[0.1, 0, 0]]), np.array([[-1.0, 0, 0]]), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), 1.0,
                             h=0.1, dx=0.05, ambient=293.15, L=1.0, C_a=4900.0, sources=[((0, 0, 0), 2500.0, 0.04)], emissivity=0.7)
ok(300.0 < T1[0] < 1500.0, "radiating wall saturates (%.0f K) instead of running away" % T1[0])

# patch sources conserve the sign of the exchange
Pp = rng.random((400, 3)); area = np.full(400, 0.01)
info = dict(coverage=np.ones(400), h_conv=np.full(400, 20.0), T_fluid=np.full(400, 300.0))
src = TH.patch_sources(Pp, area, np.full(400, 350.0), info, 0.5)
tot = sum(s[1] for s in src)
ok(src and abs(tot - 20.0 * 50.0 * 4.0 * 1.0) / (20.0 * 50.0 * 4.0) < 0.2 and all(s[1] > 0 for s in src), "hot wall heats the fluid (%.0f W total, %d patches)" % (tot, len(src)))
ok(all(s[1] < 0 for s in TH.patch_sources(Pp, area, np.full(400, 250.0), info, 0.5)), "cold wall cools the fluid")

# ramp
c = TH.ramp_color(np.array([0.0, 0.5, 1.0]), 'INFERNO')
ok(c.shape == (3, 3) and c[0].sum() < c[2].sum(), "ramp colours")
print("FAILED %d" % fails[0] if fails[0] else "ALL PASS")
sys.exit(1 if fails[0] else 0)
