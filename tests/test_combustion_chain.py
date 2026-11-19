"""Combustion end to end on the single-engine demo: tanks fill, propellants merge and ignite in the chamber, the neck builds pressure, the throat chokes, the nozzle
expands the gas, the jet hits the ground, flame turns into smoke, the exhaust fades at the domain radius, the particle budget holds, Continuous flow ON / OFF.
blender --factory-startup -b --python tests/test_combustion_chain.py"""
import math
import os
import sys
import time
import numpy as np
import bpy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import arc_flow
arc_flow.register()
from arc_flow import runtime as RT, demos
from arc_flow.core import flame as FL
from arc_flow.core import burn as BURN

fails = [0]


def ok(c, msg):
    print(("PASS " if c else "FAIL ") + msg, flush=True)
    if not c:
        fails[0] += 1


sc = bpy.context.scene
af = sc.arc_flow
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
demos.build_single(sc)
af.domain_radius = 10.0
af.particle_budget = 16000
af.continuous = True
w = RT.build_world(sc)
dt = 1 / 24.0
GROUND = w.env.ground_z

# ---- 1. tanks fill (shared level-set selection): the 75 percent sphere tank is filled to the exact level
tank = next(s for s in w.sources if s['kind'] == 'C')
sl = slice(tank['lo'], tank['lo'] + tank['n0'])
zc = w.parts[tank['part']].matrix[2, 3]
zlev = w.pos[sl][:, 2].max()
rt = 1.0
hh = next(h for h in np.linspace(0, 2 * rt, 2001) if math.pi * h * h * (3 * rt - h) / 3.0 >= 0.75 * 4.0 / 3.0 * math.pi * rt ** 3)
ok(abs((zlev - (zc - rt)) - hh) < 0.12, "tank fill: free surface %.2f m above the bottom, exact level for 75 percent of a sphere %.2f m (shared level selection)" % (zlev - (zc - rt), hh))
ok(abs(tank['fill_vol'] / tank['cav_vol'] - 0.75) < 0.01, "tank fill volume = 75 percent of the cavity volume")

# ---- run: ignition, pressure build-up, choked flow, expansion, ground, smoke, domain, budget
P_hist, ign_hist, alive_max, vent_sum, mdot_err = [], [], 0, 0.0, []
nz = w.nozzle
g, R = w.exh_gamma, w.exh_R
ok(nz is not None and nz['r_t'] < nz['r_e'], "nozzle found from the openings: throat radius %.2f m, exit radius %.2f m (area ratio %.1f)" % (nz['r_t'], nz['r_e'], (nz['r_e'] / nz['r_t']) ** 2))
t0 = time.time()
first_reaction = None
for f in range(1, 121):
    w.step(dt)
    b = w.bstats
    P_hist.append(b['P_c'])
    ign_hist.append(w.ignition)
    alive_max = max(alive_max, b['alive'])
    if first_reaction is None and b['reacted'] > 0:
        first_reaction = f
    vent_sum += b['mdot_out'] * dt
b = w.bstats
print("  (120 frames in %.0f s, %d alive)" % (time.time() - t0, b['alive']), flush=True)
ok(first_reaction is not None and first_reaction <= 12, "propellants merge and ignite in the chamber: first reaction in frame %s, %d fuel/oxidiser pairs burnt (%.0f kg)" % (first_reaction, b['reacted'], b['burnt_mass']))
inch = np.flatnonzero((w.state == 1) & (w.kind == 1))
ok(len(inch) > 0 and abs(w.temp[inch].mean() - 3670.0) < 50, "flame gas in the chamber is at the flame temperature (%.0f K)" % w.temp[inch].mean())
ok(w.ignition > 0.5, "ignition level %.2f" % w.ignition)
ok(P_hist[0] < 1.3 * w.env.pressure and max(P_hist) > 10 * w.env.pressure, "neck pressure builds: %.2f bar in frame 1 to %.1f bar peak" % (P_hist[0] / 1e5, max(P_hist) / 1e5))
ok(b['choked'] and w.env.pressure / b['P_c'] < FL.critical_pressure_ratio(g), "flow is choked: ambient / chamber = %.3f below the critical ratio %.3f" % (w.env.pressure / b['P_c'], FL.critical_pressure_ratio(g)))
T0 = max(w.chambers[0]['T'], 300.0)
ok(abs(b['v_throat'] - math.sqrt(2 * g * R * T0 / (g + 1))) < 1e-6, "throat velocity is sonic: %.0f m/s" % b['v_throat'])
A_t = math.pi * nz['r_t'] ** 2
errs = [abs(BURN.mdot_out(w, P, T0) - A_t * P * math.sqrt(g / (R * T0)) * (2 / (g + 1)) ** ((g + 1) / (2 * (g - 1)))) / (A_t * P * math.sqrt(g / (R * T0))) for P in (2e6, 5e6, 8e6)]
ok(max(errs) < 1e-9, "choked mass flow = A* p_c sqrt(gamma/(R T)) (2/(gamma+1))^((gamma+1)/(2(gamma-1))), linear in chamber pressure (%.0f kg/s at 50 bar)" % BURN.mdot_out(w, 5e6, T0))
ok(0 < BURN.mdot_out(w, 1.2e5, T0) < BURN.mdot_out(w, 1.6e5, T0) < A_t * FL.choked_mass_flux(1.6e5, T0, g, R), "below the critical pressure ratio the flow is subsonic and smaller than the choked flow would be")
ex = FL.exit_state(b['P_c'], T0, g, R, (nz['r_e'] / nz['r_t']) ** 2)
Ae = math.pi * nz['r_e'] ** 2
Fth = b['mdot_out'] * ex['v'] + (ex['p'] - w.env.pressure) * Ae
ok(abs(b['thrust'] - Fth) < 1e-6 * max(Fth, 1.0) and b['thrust'] > 0 and np.dot(b['thrust_vec'], nz['normal']) < 0, "nozzle thrust F = mdot v_e + (p_e - p_a) A_e = %.0f kN, pointing against the exhaust (Isp %.0f s)" % (b['thrust'] / 1e3, b['isp']))
ok(abs(b['v_exit'] - ex['v']) < 1e-6 and ex['M'] > 2.5, "nozzle expansion: exit Mach %.2f, %.0f m/s, exit pressure %.2f bar (area ratio %.1f)" % (ex['M'], ex['v'], ex['p'] / 1e5, (nz['r_e'] / nz['r_t']) ** 2))
phis = np.array([nz['phi_t'] * 2.0, nz['phi_t'] * 1.05, nz['phi_t'] * 0.5, 0.0])
rads = np.array([1.0, nz['r_t'], 0.5 * (nz['r_t'] + nz['r_e']), nz['r_e']])
v = BURN.gas_speed_cap(w, phis, rads) / b['vscale']
ok(v[0] < v[1] < v[2] < v[3] and abs(v[1] / b['v_throat'] - 1) < 0.05, "gas speeds up along the nozzle: chamber %.0f, throat %.0f (Mach 1), mid %.0f, exit %.0f m/s" % (tuple(v)))
ok(b['vscale'] < 1.0, "particle speeds are scaled by %.3f so the fastest stays under the %.0f m/s world limit (physical numbers above)" % (b['vscale'], w.p.vmax))
ok(abs(b['vented_mass'] - vent_sum) / vent_sum < 0.15, "mass leaves the nozzle only at the choked flow: %.0f kg vented against %.0f kg allowed by the throat" % (b['vented_mass'], vent_sum))
ok(b['ground'] > 0, "ground impact: %d wall-jet contacts" % b['ground'])
gas = np.flatnonzero((w.state == 2) & (w.kind > 0) & (w.pos[:, 2] < GROUND + 0.6))
cx = nz['centre'][:2]
r_ground = np.linalg.norm(w.pos[gas][:, :2] - cx, axis=1)
ok(len(gas) > 100 and np.median(r_ground) > 0.6 * nz['r_e'], "the jet spreads along the ground as a wall jet (median radius %.2f m, exit radius %.2f m, %d particles)" % (np.median(r_ground), nz['r_e'], len(gas)))
ok(b['smoke'] > 100, "flame turns into smoke as it cools below %.0f K: %d particles converted" % (w.p.smoke_T, b['smoke']))
sm = np.flatnonzero((w.kind == 2) & (w.state == 2))
ok(len(sm) > 0 and w.temp[sm].max() < w.p.smoke_T + 1.0, "smoke particles are below the conversion temperature")
sc_ = np.array(FL.flame_smoke(0.0)); sc2 = np.array(FL.flame_smoke(1.0))
ok(abs(np.array(FL.flame_smoke(0.5)) - (sc_ + sc2) / 2).max() < 1e-9, "smoke colour comes from the web app's flame model (steam grey %s to soot %s)" % (np.round(sc_, 2), np.round(sc2, 2)))
gas_all = np.flatnonzero((w.state > 0) & (w.kind > 0))
rr = np.linalg.norm(w.pos[gas_all] - w.dom_c, axis=1)
ok(b['removed_domain'] > 0 and rr.max() <= af.domain_radius + 1e-6, "exhaust is removed at the %.0f m domain radius (%d removed, farthest alive %.2f m)" % (af.domain_radius, b['removed_domain'], rr.max()))
fz = w.fade[gas_all]
ok(np.all(fz[rr < 0.75 * af.domain_radius] > 0.999) and np.any(fz < 0.9) and np.allclose(fz, FL.fade_factor(rr, af.domain_radius, w.p.domain_fade)), "exhaust fades smoothly over the outer quarter of the domain (%d faded, min %.2f)" % (int((fz < 0.999).sum()), fz.min()))
ok(alive_max <= 16000, "particle budget holds: at most %d alive (budget 16000), %d smoke particles removed for it" % (alive_max, b['removed_budget']))
inside = int(np.sum((w.state == 1) & (w.pin == tank['part'])))
ok(inside > 0.5 * tank['n0'], "Continuous flow ON: the tank is refilled as it is used (%d of %d particles inside after 5 s)" % (inside, tank['n0']))
reacted_on = b['reacted']

# ---- Continuous flow OFF: the tanks deplete, the engine runs dry, the plume ends and fades
af.continuous = False
af.particle_budget = 16000
w2 = RT.build_world(sc)
hist = []
t0 = time.time()
for f in range(1, 241):
    w2.step(dt)
    if f % 20 == 0:
        hist.append((f, int(np.sum((w2.state == 1) & (w2.kind == 0))), w2.bstats['P_c'], int(w2.bstats['reacted']), int(((w2.state == 2) & (w2.kind == 1)).sum()), int(((w2.state > 0)).sum())))
print("  (OFF, 240 frames in %.0f s)" % (time.time() - t0), flush=True)
for hrow in hist:
    print("   frame %d: confined propellant %d, p_c %.2f bar, pairs burnt %d, free flame gas %d, alive %d" % (hrow[0], hrow[1], hrow[2] / 1e5, hrow[3], hrow[4], hrow[5]))
r_early = hist[1][3] - hist[0][3]
r_late = hist[-1][3] - hist[-2][3]
tank_in = int(np.sum((w2.state == 1) & (w2.pin == tank['part'])))
ok(tank_in < 0.05 * tank['n0'] and hist[-1][3] < 0.1 * reacted_on, "Continuous flow OFF: the tank runs empty (%d of %d inside; the feed line keeps %d particles it cannot push out without tank pressure) and only %d pairs burnt against %d with flow ON" % (tank_in, tank['n0'], hist[-1][1], hist[-1][3], reacted_on))
ok(r_late == 0 and r_early > 0, "OFF: the engine shuts down (pairs burnt in the last 20 frames: %d, early: %d)" % (r_late, r_early))
ok(hist[-1][2] < 2 * w2.env.pressure, "OFF: chamber pressure returns to ambient (%.2f bar)" % (hist[-1][2] / 1e5))
ok(hist[-1][4] < 0.1 * max(h[4] for h in hist) + 5, "OFF: the flame is gone, only smoke is left to fade (free flame gas %d of a peak %d)" % (hist[-1][4], max(h[4] for h in hist)))
print("FAILURES:", fails[0])
sys.exit(1 if fails[0] else 0)
