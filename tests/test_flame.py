"""Flame model port vs the web app's JavaScript (reference values from node), propellant chemistry and nozzle gas dynamics
(python3 tests/test_flame.py)."""
import json
import math
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fm_helpers  # noqa: E402,F401  (puts arc_flow on the path)
from core import flame as FL  # noqa: E402

fails = [0]


def ok(c, msg):
    print(("PASS " if c else "FAIL ") + msg, flush=True)
    if not c:
        fails[0] += 1


def close(a, b, tol=1e-9):
    return all(abs(x - y) <= tol * max(1.0, abs(y)) for x, y in zip(a, b))


R = json.load(open(os.path.join(HERE, 'flame_reference.json')))
ok(all(abs(FL.FLAME[k] - v) < 1e-12 for k, v in R['FLAME'].items() if isinstance(v, (int, float))), "FLAME constants equal the web app's")
ok(all(close(FL.flame_emission(e['tn'], e['s'], e['c'], e['ab']), e['v']) for e in R['emission']), "emission (%d samples) matches the JavaScript model" % len(R['emission']))
ok(all(close(FL.flame_bb(e['T']), e['v']) for e in R['bb']), "blackbody chromaticity ramp matches")
ok(all(close(FL.flame_smoke(e['s']), e['v']) for e in R['smoke']), "smoke colour (steam grey to soot black) matches")
ok(all(abs(FL.FUEL_SPECIES[k]['ofStoich'] - v) < 1e-9 for k, v in R['ofStoich'].items()), "stoichiometric O/F: %s" % {k: round(v, 2) for k, v in R['ofStoich'].items()})
worst = 0.0
for e in R['products']:
    p = FL.comb_products(e['k'], e['of'], e['T'])
    for g, v in e['mass'].items():
        worst = max(worst, abs(p['mass'][g] - v))
ok(worst < 1e-9, "combustion products (%d mixtures, with water-gas shift) match, worst mass-fraction difference %.1e" % (len(R['products']), worst))
good = True
for e in R['species']:
    s = FL.flame_species(e['k'], 0, 3000.0, e['pRel'], dict(N2=0.781, O2=0.209))
    good &= all(abs(s[k] - e[k]) < 1e-9 for k in ('soot', 'chem', 'yf', 'nox', 'smkMax', 'steam'))
ok(good, "soot / chemiluminescence / smoke opacity / NOx indices match")
ok(abs(sum(FL.comb_products('rp1', 2.4, 3000.0)['mass'].values()) - 1.0) < 1e-12, "exhaust mass fractions sum to 1")
# physical ordering
ok(FL.flame_species('rp1', 0, 3000)['soot'] > FL.flame_species('lh2', 0, 3000)['soot'], "RP-1 is sooty, hydrogen is not")
ok(FL.flame_species('lh2', 0, 3000)['chem'] > FL.flame_species('rp1', 0, 3000)['chem'], "hydrogen flame is the chemiluminescent (blue) one")
# nozzle: choked flow and area-Mach expansion against textbook values
g, Rg, T0, p0 = 1.2, 8.314 / 0.024, 3500.0, 7.0e6
ok(abs(FL.critical_pressure_ratio(1.4) - 0.5283) < 1e-4, "critical pressure ratio of a gamma 1.4 gas = 0.5283 (%.4f)" % FL.critical_pressure_ratio(1.4))
ok(abs(FL.critical_pressure_ratio(1.2) - 0.5645) < 1e-4, "critical pressure ratio of gamma 1.2 = 0.5645")
mf = FL.choked_mass_flux(101325.0, 300.0, 1.4, 287.0)
ok(abs(mf / (101325.0 / math.sqrt(300.0)) - 0.0404) < 1e-4, "choked mass flux of air = 0.0404 p0 / sqrt(T0) (%.1f kg/s/m2 at 1 atm, 300 K)" % mf)
ok(abs(FL.area_ratio(2.0, 1.4) - 1.6875) < 1e-3, "A/A* at Mach 2 (gamma 1.4) = 1.6875")
ok(abs(FL.mach_from_area(1.6875, 1.4) - 2.0) < 1e-3, "supersonic Mach from area ratio 1.6875 = %.3f" % FL.mach_from_area(1.6875, 1.4))
ex = FL.exit_state(p0, T0, g, Rg, 16.0)
ok(ex['M'] > 3.0 and ex['v'] > 2500 and ex['p'] < 0.01 * p0, "area ratio 16: exit Mach %.2f, %.0f m/s, p_e/p0 = %.4f (gamma 1.2, 3500 K, 7 MPa chamber)" % (ex['M'], ex['v'], ex['p'] / p0))
ok(FL.throat_velocity(T0, g, Rg) < ex['v'] and abs(FL.throat_velocity(T0, g, Rg) - math.sqrt(g * Rg * 2 * T0 / (g + 1))) < 1e-6, "throat velocity is sonic (%.0f m/s), below the exit velocity" % FL.throat_velocity(T0, g, Rg))
print("FAILURES:", fails[0])
sys.exit(1 if fails[0] else 0)
