"""Solid-body fluid blending and aerodynamic friction heating (NumPy only):

    python3 tests/test_fluid_blending.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'arc_flow'))
from core import fluid_blend as FB
from core import friction_heat as FH
from assets import fighter as FG

fails = [0]


def ok(c, m):
    print(("PASS " if c else "FAIL ") + m)
    if not c:
        fails[0] += 1


def _cube(lo, hi, dx):
    g = [np.arange(a, b + 1e-12, dx) for a, b in zip(lo, hi)]
    X, Y, Z = np.meshgrid(*g, indexing='ij')
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)


# ----------------------------------------------------------------------------- solid-body fluid surface
dx = 0.02
X = _cube((-0.08, -0.08, -0.08), (0.08, 0.08, 0.08), dx)
verts, tris, vc, nrm, info = FB.solid_body_surface(X, colors=np.tile((0.2, 0.5, 0.9), (len(X), 1)), dx=dx)
ok(info['components'] == 1, "packed cube is one connected surface (%d)" % info['components'])
ok(info['volume'] > 0.5 * (0.16 ** 3) and info['volume'] < 1.6 * (0.16 ** 3),
   "enclosed volume is a continuous liquid body (%.5f m3, box=%.5f)" % (info['volume'], 0.16 ** 3))
ok(info['coverage'] >= 0.95, "continuous volume coverage %.3f" % info['coverage'])
ok(len(verts) > 0 and len(tris) > 0 and np.isfinite(verts).all() and np.isfinite(nrm).all() and np.isfinite(tris).all(),
   "mesh is finite (no NaN/Inf), %d verts %d tris" % (len(verts), len(tris)))
ln = np.linalg.norm(nrm, axis=1)
ok(np.all((ln > 0.98) | (ln < 1e-12)) and ln.mean() > 0.99, "smooth unit normals (mean length %.4f)" % ln.mean())
# two well-separated blobs stay 2; two touching blobs merge to 1
A = _cube((-0.10, -0.04, -0.04), (-0.04, 0.04, 0.04), dx)
B = _cube((0.04, -0.04, -0.04), (0.10, 0.04, 0.04), dx)
_,_,_,_, far = FB.solid_body_surface(np.vstack([A, B]), dx=dx, radius=2.2, iso=0.35)
C = _cube((-0.05, -0.04, -0.04), (0.01, 0.04, 0.04), dx)
D = _cube((-0.01, -0.04, -0.04), (0.05, 0.04, 0.04), dx)
_,_,_,_, near = FB.solid_body_surface(np.vstack([C, D]), dx=dx, radius=2.2, iso=0.35)
ok(far['components'] == 2, "separated blobs stay 2 components (%d)" % far['components'])
ok(near['components'] == 1, "touching blobs merge into a single solid body (%d)" % near['components'])

sb = FB.SolidSurfaceBuilder(dx)
out1 = sb.update(X, np.ones((len(X), 3)) * 0.4, frame=0)
out2 = sb.update(X, np.ones((len(X), 3)) * 0.4, frame=0)
ok(out1[0] is out2[0] or (len(out1[0]) == len(out2[0]) and np.allclose(out1[0], out2[0])),
   "SolidSurfaceBuilder caches the same frame")
ok(len(out1[0]) > 0 and np.isfinite(out1[0]).all(), "builder mesh is finite")

# combustion / plume blending: a sparse streak becomes a continuous volume
rng = np.random.default_rng(2)
streak = np.stack([np.linspace(0, 0.4, 40), 0.03 * rng.normal(size=40), 0.03 * rng.normal(size=40)], axis=1)
temp = 800.0 + 1200.0 * np.linspace(1, 0.2, 40)
P, dens, T, S = FB.blend_plume(streak, temp, dx=0.02, radius=2.8, cell=0.7, iso=0.04)
ok(len(P) > len(streak), "plume blender fills a continuous volume (%d voxels from %d parcels)" % (len(P), len(streak)))
ok(np.isfinite(P).all() and np.isfinite(dens).all() and np.isfinite(T).all() and dens.min() > 0, "plume fields finite and positive")
ok(T.max() > T.min() + 50.0, "plume temperature varies smoothly (%.0f .. %.0f K)" % (T.min(), T.max()))

# ----------------------------------------------------------------------------- aerodynamic friction heating
body = FG.build()
Vv, Ff = body['verts'], body['faces']
# area-weighted vertex normals
fn = np.cross(Vv[Ff][:, 1] - Vv[Ff][:, 0], Vv[Ff][:, 2] - Vv[Ff][:, 0])
N = np.zeros_like(Vv)
for k in range(3):
    np.add.at(N, Ff[:, k], fn)
N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
P = Vv
L = float((P.max(0) - P.min(0)).max())
air = dict(T_inf=288.15, rho=1.225, mu=1.8e-5, cp=1005.0, k=0.0257, L=L)

def ev(speed, **kw):
    kw = dict(air, **kw)
    return FH.aero_evaluate(P, N, v_inf=(speed, 0, 0), roughness=kw.pop('roughness', 0.0), **{k: kw[k] for k in ('T_inf', 'rho', 'mu', 'cp', 'k', 'L')}, **{k: v for k, v in kw.items() if k == 'roughness'})

# ev signature is messy because of the above. just call directly.
def ev(speed, roughness=0.0, rho=1.225):
    return FH.aero_evaluate(P, N, v_inf=(speed, 0, 0), T_inf=288.15, rho=rho, mu=1.8e-5, cp=1005.0, k=0.0257,
                            L=L, roughness=roughness)

a50, a150, a300 = ev(50), ev(150), ev(300)
ok(np.all(np.isfinite(a300['T_aw'])) and np.all(np.isfinite(a300['q'])) and np.all(a300['T_aw'] >= 288.15 - 1e-6),
   "aero fields finite and at least ambient")
ok(a50['T_aw'].max() < a150['T_aw'].max() < a300['T_aw'].max(),
   "adiabatic wall temperature rises with wind speed (%.2f, %.2f, %.2f K)" % (a50['T_aw'].max(), a150['T_aw'].max(), a300['T_aw'].max()))
ok(a50['q'].max() < a150['q'].max() < a300['q'].max(),
   "friction heat flux rises with wind speed (%.0f, %.0f, %.0f W/m2)" % (a50['q'].max(), a150['q'].max(), a300['q'].max()))
ok(a300['Mach'].max() > 0.8, "Mach number tracks the free stream (%.3f at 300 m/s)" % a300['Mach'].max())

# leading edge / high-shear vs downstream / leeward
xs = P[:, 0]
nose = xs <= np.quantile(xs, 0.08)
tail = xs >= np.quantile(xs, 0.92)
ok(a300['q'][nose].mean() > a300['q'][tail].mean() * 1.4,
   "heat flux peaks at the nose / leading edges and cools downstream (nose %.0f, tail %.0f W/m2)" %
   (a300['q'][nose].mean(), a300['q'][tail].mean()))
# leeward = outward normal pointing down-stream (wind is +X); windward / leading edge points into the wind
lee = N[:, 0] > 0.55
hit = N[:, 0] < -0.55
ok(hit.any() and lee.any() and a300['q'][hit].mean() > a300['q'][lee].mean() * 1.5,
   "stagnation / high-shear zones out-heat the leeward face (%.0f vs %.0f)" % (a300['q'][hit].mean(), a300['q'][lee].mean()))

rough = ev(200, roughness=5e-4)
smooth = ev(200, roughness=0.0)
# compare on grazing samples
g = smooth['graze'] > 0.7
ok(g.any() and rough['h'][g].mean() > smooth['h'][g].mean() * 1.15,
   "sand-grain roughness raises grazing-skin heat transfer (%.0f -> %.0f W/m2K)" % (smooth['h'][g].mean(), rough['h'][g].mean()))
thin = ev(200, rho=0.3)
ok(thin['q'][nose].mean() < smooth['q'][nose].mean(),
   "heat flux falls with air density (%.0f vs %.0f W/m2)" % (thin['q'][nose].mean(), smooth['q'][nose].mean()))

# early-time wall temperature: nose heats faster than the tail, leeward stays cool
Ts = np.full(len(P), 288.15)
Ts, info = FH.surface_step(Ts, P, N, 0.05, v_inf=(250, 0, 0), T_inf=288.15, rho=1.225, mu=1.8e-5, cp=1005, k=0.0257,
                           L=L, C_a=2000.0, speedup=1.0, emissivity=0.3)
dT = Ts - 288.15
ok(dT[nose].mean() > dT[tail].mean() and dT[hit].mean() > dT[lee].mean() and dT[lee].mean() >= -1e-6,
   "early wall heating is fastest at the leading edge and cooler downstream / leeward (nose %.3f, tail %.3f, lee %.3f K)" %
   (dT[nose].mean(), dT[tail].mean(), dT[lee].mean()))
ok(np.isfinite(Ts).all() and Ts.min() >= 288.15 - 1e-6, "stepped wall temperatures finite and bounded below by ambient")

# airflow volume around the fighter
lo, hi = P.min(0) - 0.2, P.max(0) + np.array([0.8, 0.3, 0.3])
# downsample body for a quicker field in this test
rng = np.random.default_rng(0)
pick = rng.choice(len(P), min(len(P), 800), replace=False)
field = FB.airflow_field(P[pick], N[pick], body_T=288.15 + 40.0 * a300['q'][pick] / max(float(a300['q'].max()), 1.0),
                         v_inf=(200, 0, 0), T_inf=288.15, bounds=(lo, hi), cell=0.07, delta=0.08)
ok(len(field['P']) > 200 and np.isfinite(field['P']).all() and np.isfinite(field['dens']).all() and np.isfinite(field['temp']).all(),
   "slipstream volume is a continuous finite field (%d voxels)" % len(field['P']))
ok(field['temp'].max() > 288.15 + 5.0, "heated boundary layer sheds warmer air than ambient (max %.1f K)" % field['temp'].max())
ok(field['dens'].max() > 0.5 and field['P'][:, 0].max() > P[:, 0].max(),
   "wake extends downstream of the body")

# fighter mesh itself is a finite closed solid
ok(len(Vv) > 200 and np.isfinite(Vv).all() and FB.signed_volume(Vv, Ff) > 0, "procedural fighter is a closed solid (V=%.4f m3)" % FB.signed_volume(Vv, Ff))

# curve / thin-surface sourcing: sharper wisps with fewer particles than a filled volume
rng = np.random.default_rng(1)
tt = np.linspace(0, 1, 400)
curve = np.stack([0.4 * tt, 0.05 * np.sin(12 * tt), 0.05 * np.cos(9 * tt)], 1)
Pc = FB.source_from_curve(curve, 0.01, rng=rng, jitter=0.002)
Pv = rng.uniform([0, -0.06, -0.06], [0.4, 0.06, 0.06], (len(Pc) * 6, 3))
a = FB.blend_plume(Pc, np.full(len(Pc), 900.0), dx=0.01, cell=0.6)
b = FB.blend_plume(Pv, np.full(len(Pv), 900.0), dx=0.01, cell=0.6)
da, db = FB.wisp_detail(a[0], a[1], 0.006), FB.wisp_detail(b[0], b[1], 0.006)
ok(len(Pc) < len(Pv) / 4 and da > db, "curve-sourced plume is sharper (%.1f vs %.1f) with %d vs %d particles" % (da, db, len(Pc), len(Pv)))
sq = np.array([[0, 0, 0], [0.2, 0, 0], [0.2, 0.2, 0], [0, 0.2, 0.]]); Ts_ = np.array([[0, 1, 2], [0, 2, 3]])
Ps = FB.source_from_thin_surface(sq, Ts_, 0.01, rng=rng, thickness=0.002)
ok(len(Ps) > 100 and np.abs(Ps[:, 2]).max() <= 0.0011, "thin-surface sourcing seeds a sheet (%d points)" % len(Ps))

# whitewater: impact on a collision surface and turbulent crest make foam; calm bulk does not
g = _cube((-0.06, -0.06, 0.0), (0.06, 0.06, 0.08), 0.01)
vel = np.zeros_like(g); vel[:, 2] = -0.05
hit = g[:, 2] < 0.015
vel[hit, 2] = -2.0
ww = FB.whitewater(g, vel, dx=0.01, body_P=np.array([[x, y, -0.005] for x in np.arange(-.06, .061, .01) for y in np.arange(-.06, .061, .01)]),
                   body_N=np.tile([0, 0, 1.0], (169, 1)), speed_ref=2.0)
ok(ww['foam'][hit].mean() > ww['foam'][~hit].mean() + 0.2, "whitewater foam forms where water strikes the collision surface (%.2f vs %.2f)" % (ww['foam'][hit].mean(), ww['foam'][~hit].mean()))
vs, ts, _, _, _ = FB.solid_body_surface(g, dx=0.01)
fv = FB.foam_on_surface(vs, g, ww['foam'], 0.01)
ok(len(fv) == len(vs) and np.isfinite(fv).all() and fv.max() > 0.3, "foam is transferred to the blended surface (max %.2f)" % fv.max())

print("FAILED %d" % fails[0] if fails[0] else "ALL PASS")
sys.exit(1 if fails[0] else 0)
