"""Combustion chain per engine (NumPy only, a stub world): channel groups, per-engine throat / choked flow / thrust, gas speed along each nozzle,
every channel can burn like a chamber (python3 tests/test_combustion_groups.py)."""
import math
import os
import sys
from types import SimpleNamespace as NS
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fm_helpers  # noqa: E402,F401
from core import burn as B  # noqa: E402
from core import flame as FL  # noqa: E402
from core import fluids as F  # noqa: E402

fails = [0]


def ok(c, msg):
    print(("PASS " if c else "FAIL ") + msg, flush=True)
    if not c:
        fails[0] += 1


class Field:
    def __init__(self, v):
        self.v = v

    def cavity_volume(self):
        return self.v


def part(role, v):
    return NS(role=role, field=Field(v))


def stub():
    # engine 0: line(0) - chamber(1) - nozzle(2); engine 1: line(3) - chamber(4) - nozzle(5, smaller throat); a lone channel (6) with one open end
    parts = [part('TRANSPORT', .05), part('CHAMBER', 2.), part('CHAMBER', 1.), part('TRANSPORT', .05), part('CHAMBER', 2.), part('CHAMBER', 1.), part('TRANSPORT', .2), part('CONTAINER', 4.)]
    op = lambda i, r, st, partner=-1, c=(0, 0, 0): dict(part=i, centre=np.array(c, float), normal=np.array([0, 0, -1.]), radius=r, state=st, partner=partner)
    ops = [op(0, .12, 'CONNECTED', 1), op(1, .4, 'CONNECTED', 0), op(1, .33, 'CONNECTED', 2), op(2, .33, 'CONNECTED', 1), op(2, .77, 'OPEN', c=(0, 0, -2)),
           op(3, .12, 'CONNECTED', 4), op(4, .4, 'CONNECTED', 3), op(4, .2, 'CONNECTED', 5), op(5, .2, 'CONNECTED', 4), op(5, .5, 'OPEN', c=(11, 0, -2)),
           op(6, .1, 'OPEN', c=(30, 0, 0)), op(6, .1, 'PLUGGED'), op(0, .12, 'CONNECTED', 7)]
    w = NS(active=list(range(8)), parts=parts, openings=ops, scale=np.ones(8), net=None, env=NS(pressure=101325.0, temp=288.0),
           fluids=[dict(spec=dict(role='FUEL'), name='RP-1'), dict(spec=dict(role='OXIDIZER'), name='LOX')],
           _world_bounds=lambda: (np.zeros(3), np.array([12., 1, 1])), _gs=lambda a, p: np.zeros(len(p)), p=NS(vmax=60.0))
    B._geometry(w)
    return w


w = stub()
ok(len(w.nozzles) == 3 and [c['group'] for c in w.chambers] == [0, 0, 0, 1, 1, 1, 2], "channel groups: two engines (line + chamber + nozzle) and one lone channel; tank excluded (%s)" % [c['group'] for c in w.chambers])
ok(len(w.chambers) == 7 and all(c['part'] in (0, 1, 2, 3, 4, 5, 6) for c in w.chambers), "every channel part is a ledger (a chamber is just another channel)")
n0, n1, n2 = w.nozzles
ok(abs(n0['r_t'] - .33) < 1e-9, "engine 0 throat from its chamber-part openings (%.2f)" % n0['r_t'])
ok(n0['r_t'] > n1['r_t'] and n0['r_e'] == .77 and n1['r_e'] == .5, "each engine has its own throat / exit (%.2f/%.2f and %.2f/%.2f)" % (n0['r_t'], n0['r_e'], n1['r_t'], n1['r_e']))
ok(n2 is not None and n2['r_e'] == n2['r_t'] == .1, "lone channel: its open end is both throat and exit (no expansion)")
ok(n2['centre'][0] == 30, "lone channel vents at its own open end")

# per-engine choked flow and thrust from independent chamber pressures
g, R = w.exh_gamma, w.exh_R
w.bstats = dict(P_c=0, mdot_in=0, mdot_out=0)
w.mass = np.array([1e-3])
for c, (m, T) in zip(w.chambers, ((0, 300), (300.0, 3600.0), (0, 300), (0, 300), (60.0, 3600.0), (0, 300), (0, 300))):
    c['m'], c['T'] = m, T
w.p.vmax = 60.0
B.chamber_update(w, 1 / 24.0)
e0, e1, e2 = w.engines
ok(e0['P_c'] > 1.3 * w.env.pressure and e1['P_c'] > 1.3 * w.env.pressure and e2['P_c'] == w.env.pressure, "pressure per engine (%.1f, %.1f bar), the idle channel stays at ambient" % (e0['P_c'] / 1e5, e1['P_c'] / 1e5))
ok(e0['choked'] and e1['choked'] and not e2['choked'], "both burning engines choke, the idle channel does not")
for e, nz in ((e0, n0), (e1, n1)):
    A = math.pi * nz['r_t'] ** 2
    mdot_ref = A * FL.choked_mass_flux(e['P_c'], e['T'], g, R)
    ok(abs(B.mdot_out(w, e['P_c'], e['T'], nz) - mdot_ref) < 1e-9 * mdot_ref, "engine throat %.2f m: choked mass flow matches A* p_c sqrt(gamma/RT) (...) = %.0f kg/s" % (nz['r_t'], mdot_ref))
ok(e0['thrust'] > 0 and e1['thrust'] > 0 and abs(w.bstats['thrust'] - e0['thrust'] - e1['thrust']) < 1e-9, "thrust per engine, total is the sum (%.0f + %.0f N)" % (e0['thrust'], e1['thrust']))
ok(abs(np.dot(e0['thrust_vec'], n0['normal']) + e0['thrust']) < 1e-6, "thrust points opposite to the exit normal")
ok(w.bstats['P_c'] == max(e0['P_c'], e1['P_c']), "reported chamber pressure is the highest of the engines")

# gas speed along the nozzle with per-engine throat
phi = np.array([5.0, 5.0, -1.0, -1.0])
rad = np.array([n0['r_t'], n1['r_t'], n0['r_t'] * 2, n1['r_t'] * 2])
grp = np.array([0, 1, 0, 1])
for nz in w.nozzles[:2]:
    nz['phi_t'] = 0.0
v = B.gas_speed_cap(w, phi, rad, grp)
ok(v[0] > 0 and v[1] > 0 and v[2] > v[0] * 0.99 and v[3] > v[1] * 0.99, "gas speeds up after each engine's own throat (%s m/s)" % np.round(v, 1))

# particle groups: gas born in a chamber keeps its engine, others join the nearest nozzle
w.home = np.array([4, -1, 0, -1])
w.pos = np.array([[0, 0, 0.], [10.5, 0, -1], [0, 0, 0], [0.3, 0, -2.1]])
ok(list(B.particle_group(w, np.arange(4))) == [1, 1, 0, 0], "particle -> engine: born-in chamber wins, else the nearest nozzle exit (%s)" % list(B.particle_group(w, np.arange(4))))

print("FAILED %d" % fails[0] if fails[0] else "all passed")
sys.exit(1 if fails[0] else 0)
