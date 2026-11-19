"""Surface sources and fluid volumes (NumPy, no Blender): viscous drag, emission, growth / snapshot, box volume."""
import numpy as np
from fm_helpers import FM, box
from core import fsurf as FS

fails = []


def ok(c, msg):
    print(("PASS " if c else "FAIL ") + msg)
    if not c:
        fails.append(msg)


# ---- geometry
V, T = box((0, 0, 0), (1, 1, 1))
P, N, A, F = FS.sample_faces(V, T, 0.25)
ok(abs(A.sum() - 6.0) < 1e-9, "face samples keep the area (%.6f)" % A.sum())
ok(np.allclose(np.linalg.norm(N, axis=1), 1.0), "unit normals")
ok(len(P) > 12 * 4, "triangles are subdivided to the spacing (%d samples)" % len(P))
M = np.eye(4); M[:3, :3] = np.diag([2.0, 1.0, 0.5]); M[:3, 3] = (1, 2, 3)
Vb, Tb = FS.box_mesh(M, (1, 1, 1))
ok(FM.is_closed(Tb) and abs(FM.mesh_volume(Vb, Tb) - 8 * 2 * 1 * 0.5) < 1e-9 and abs(FS.box_volume(M, (1, 1, 1)) - 8.0) < 1e-9, "box_mesh closed, volume 8")
ok(abs(FS.gas_density(1.2, 293.15, FM.P_ATM) - 1.2) < 1e-12 and FS.gas_density(1.2, 586.3, FM.P_ATM) < 0.61 and FS.gas_density(1.2, 293.15, 2 * FM.P_ATM) > 2.39, "ideal gas density")
i, j = FS.cross_pairs(np.array([[0, 0, 0.], [5, 5, 5.]]), np.array([[0.1, 0, 0], [5, 5, 5.2], [9, 9, 9.]]), 0.3)
ok(sorted(zip(i.tolist(), j.tolist())) == [(0, 0), (1, 1)], "cross_pairs")

# ---- viscous surface: a still wall drags a stream flowing along it; normal motion untouched
env = FM.FEnv(gravity=(0, 0, 0))
d = FM.fluid_def('WATER', name="w", shape=dict(kind='MESH', verts=np.array(box((0, 0, 0.0), (0.4, 0.4, 0.3))[0]), tris=box((0, 0, 0.0), (0.4, 0.4, 0.3))[1]),
                 velocity=(1.0, 0.0, 0.0), cohesion=0.0, visc=0.0)
S = FM.FluidSolver.build([d], [], env, dx=0.04, max_particles=3000)
S.v[:] = (1.0, 0.0, 0.0)
wall_v = np.array([[0, 0, -0.01], [0.4, 0, -0.01], [0.4, 0.4, -0.01]]), np.array([[0, 0, -0.01], [0.4, 0.4, -0.01], [0, 0.4, -0.01]])
Vw = np.array([[-0.5, -0.5, -0.01], [1.0, -0.5, -0.01], [1.0, 1.0, -0.01], [-0.5, 1.0, -0.01]])
Tw = np.array([[0, 1, 2], [0, 2, 3]])
Pw, Nw, Aw, _ = FS.sample_faces(Vw, Tw, 0.04)
vs = FS.ViscositySurface("wall", Pw, np.array([[0, 0, 1.0]] * len(Pw)), Aw, coefficient=200.0, noslip=1.0)
FS.install(S, viscosity=[vs])
near = S.x[:, 2] < 0.03
far = S.x[:, 2] > 0.2
for _ in range(10):
    S.step(1 / 24.0)
vx = S.v[:, 0]
ok(near.sum() > 10 and vx[near].mean() < 0.9 * vx[far].mean() and vx[far].mean() > 0.9, "wall drags the near layer: near %.3f far %.3f" % (vx[near].mean(), vx[far].mean()))
ok(vs.force[0] > 0, "reaction force on the wall points along the flow (%.4g N)" % vs.force[0])
vs.noslip = 0.0
v_before = S.v[:, 0].copy()
S.step(1 / 24.0)
ok(abs(vs.force[0]) < 1e-12, "no-slip fraction 0 = no drag")
# a moving wall drives the fluid
S2 = FM.FluidSolver.build([dict(d, velocity=(0.0, 0.0, 0.0))], [], env, dx=0.04, max_particles=3000)
vs2 = FS.ViscositySurface("belt", Pw, np.array([[0, 0, 1.0]] * len(Pw)), Aw, coefficient=200.0)
FS.install(S2, viscosity=[vs2])
M2 = np.eye(4)
for k in range(8):
    M2 = M2.copy(); M2[0, 3] += 0.5 / 24.0
    vs2.set_matrix(M2, dt=1 / 24.0)
    S2.step(1 / 24.0)
low = S2.x[:, 2] < 0.03
ok(S2.v[low, 0].mean() > 0.1 and S2.v[~low & (S2.x[:, 2] > 0.2), 0].mean() < S2.v[low, 0].mean(), "a moving wall (0.5 m/s) drags the fluid along: near %.3f" % S2.v[low, 0].mean())

# ---- emission
d1 = FM.fluid_def('WATER', name="tank", shape=dict(kind='MESH', verts=box((0, 0, 0), (0.2, 0.2, 0.1))[0], tris=box((0, 0, 0), (0.2, 0.2, 0.1))[1]))
d2 = FM.fluid_def('WATER', name="jet", color=(1, 0.2, 0.2))
d3 = FM.fluid_def('AIR' if 'AIR' in FM.PRESETS else list(FM.PRESETS)[-1], name="last", count=8)
S = FM.FluidSolver.build([d1, d2, d3], [], FM.FEnv(), dx=0.04, max_particles=3000)
n1, n3 = S.hi[0] - S.lo[0], S.hi[2] - S.lo[2]
Pe = np.array([[0.1, 0.1, 0.5]]); Ne = np.array([[0, 0, -1.0]])
em = FS.EmissionSurface("nozzle", Pe, Ne, np.array([1e-3]), fluid=1, rate=48.0, speed=1.5, spread=10, max_particles=60, seed=3)
FS.install(S, emission=[em])
ok(S.hi[1] - S.lo[1] == 0 and S.mass_p[1] > 0, "emission fluid starts empty with a particle mass %.4g kg" % S.mass_p[1])
snap0 = S.snapshot()
for _ in range(24):
    S.step(1 / 24.0)
nj = S.hi[1] - S.lo[1]
ok(46 <= nj <= 50, "48 particles/s for 1 s -> %d particles" % nj)
ok(S.hi[0] - S.lo[0] == n1 and S.hi[2] - S.lo[2] == n3 and S.n == n1 + nj + n3, "other fluids untouched, n consistent")
ok(all(len(getattr(S, k)) == S.n for k in ('x', 'v', 'T', 'fid', 'mass_ref', 'C', 'cluster', 'rho0_p', 'wall_dist')), "all per-particle arrays grew")
ok(np.all(S.fid[S.lo[1]:S.hi[1]] == 1) and np.all(S.fid[S.lo[2]:S.hi[2]] == 2) and np.all(np.isfinite(S.x)), "fluid ids stay grouped")
ok(S.x[S.lo[1]:S.hi[1], 2].mean() < 0.5, "jet falls from the nozzle (mean z %.3f)" % S.x[S.lo[1]:S.hi[1], 2].mean())
snap1 = S.snapshot()
for _ in range(60):
    S.step(1 / 24.0)
ok(S.hi[1] - S.lo[1] == 60, "max_particles cap holds (%d)" % (S.hi[1] - S.lo[1]))
S.restore(snap1)
ok(S.hi[1] - S.lo[1] == nj and S.n == snap1['_srf']['n'], "restore a mid-run snapshot restores the grown arrays")
a = S.x.copy()
S.step(1 / 24.0); b = S.x.copy()
S.restore(snap1); S.step(1 / 24.0)
ok(np.allclose(S.x, b), "emission is deterministic after a restore")
S.restore(snap0)
ok(S.hi[1] - S.lo[1] == 0 and S.n == n1 + n3 and em.emitted == 0, "restore of the start snapshot empties the emitter")

print("surfaces: %d failure(s)" % len(fails))
raise SystemExit(1 if fails else 0)
