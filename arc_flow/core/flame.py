"""Flame, propellant chemistry and nozzle gas dynamics for combustion mode.

The FLAME constants, the blackbody / soot / chemiluminescence emission model, the propellant table, the equilibrium-ish product split (oxygen to carbon first,
then hydrogen, water-gas shift, soot yield from the C/O ratio) and the smoke colours are a line-for-line port of the shared flame model in the web app
(templates/arc-flow.html, region flame-model).  tests/flame_reference.json holds values produced by running that JavaScript under node, and tests/test_flame.py
checks this port against it.  The nozzle part (choked flow, area-Mach expansion) is textbook isentropic 1-D gas dynamics.
"""
import math

import numpy as np

FLAME = dict(
    TLO=600.0, THI=3800.0, TREF=3000.0,
    BB=[(600, 0.18, 0.0, 0.0), (900, 0.55, 0.05, 0.0), (1200, 0.90, 0.18, 0.02), (1600, 1.0, 0.40, 0.07), (2000, 1.0, 0.58, 0.20),
        (2400, 1.0, 0.74, 0.40), (2800, 1.0, 0.88, 0.65), (3200, 1.0, 0.96, 0.88), (3800, 0.92, 0.95, 1.0)],
    WHITE_LO=2500.0, WHITE_HI=3300.0,
    GAS_EPS=0.35, SOOT_GAIN=1.6, SOOT_TMAX=2100.0, SOOT_BURN=0.7,
    CHEM_GAIN=0.55, CHEM_TLO=1000.0, CHEM_THI=2200.0, CHEM_RGB=(0.45, 0.62, 1.0), CHEM_DIM=0.6,
    AB_LUM=1.5,
    AB_MIX_XS=2.0, AB_END_XS=16.0, AB_DT_K=1100.0, AB_YF_REF=0.12, AB_GROUND_DECAY=0.9,
    SMOKE_SOOT=(0.22, 0.19, 0.17), SMOKE_STEAM=(0.52, 0.55, 0.58),
    T_AMB=288.0,
)

SOOT_C_FRAC = 0.45
SOOT_REF, CHEM_K, K_SOOT, K_STEAM = 0.04, 1.6, 10.0, 1.0

# fuel / oxidiser formulae (atoms per mole, molar mass g/mol), nominal O/F, autoignition temperature (0 = hypergolic), flame temperature fraction
FUEL_SPECIES = {
    'rp1': dict(label='RP-1 kerosene / LOX', fuel=dict(C=1, H=1.95, N=0, M=13.97), ox=dict(O=2, M=32.0), ofNom=2.4, tAuto=500.0, tK=1.00),
    'ch4': dict(label='Methane / LOX', fuel=dict(C=1, H=4, N=0, M=16.04), ox=dict(O=2, M=32.0), ofNom=3.5, tAuto=810.0, tK=0.97),
    'lh2': dict(label='Hydrogen / LOX', fuel=dict(C=0, H=2, N=0, M=2.016), ox=dict(O=2, M=32.0), ofNom=6.0, tAuto=770.0, tK=0.93),
    'hyp': dict(label='Hypergolic MMH / N2O4', fuel=dict(C=1, H=6, N=2, M=46.07), ox=dict(O=4, M=92.01), ofNom=1.9, tAuto=0.0, tK=1.00),
}
for _k, _f in FUEL_SPECIES.items():
    _f['ofStoich'] = (2 * _f['fuel']['C'] + _f['fuel']['H'] / 2) / _f['ox']['O'] * _f['ox']['M'] / _f['fuel']['M']

_MOLAR = dict(H2O=18.015, H2=2.016, CO=28.01, CO2=44.01, C=12.011, N2=28.014, O2=32.0)


def _ss(a, b, x):
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def _mix(a, b, t):
    return [a[i] + (b[i] - a[i]) * t for i in range(3)]


def flame_tk(tn):
    return FLAME['TLO'] + min(max(tn, 0.0), 1.0) * (FLAME['THI'] - FLAME['TLO'])


def flame_tn(T):
    return min(max((T - FLAME['TLO']) / (FLAME['THI'] - FLAME['TLO']), 0.0), 1.0)


def flame_rad(T):
    return (T / FLAME['TREF']) ** 4


def flame_bb(T):
    k = FLAME['BB']
    if T <= k[0][0]:
        return list(k[0][1:])
    for i in range(1, len(k)):
        if T < k[i][0]:
            return _mix(k[i - 1][1:], k[i][1:], (T - k[i - 1][0]) / (k[i][0] - k[i - 1][0]))
    return list(k[-1][1:])


def flame_emission(tn, soot, chem, ab):
    F = FLAME
    T = flame_tk(tn)
    rad = flame_rad(T) * (1 + F['AB_LUM'] * ab)
    soot_e = soot * (1 - F['SOOT_BURN'] * ab)
    gas = [c * rad * F['GAS_EPS'] * (1 - F['CHEM_DIM'] * chem) for c in flame_bb(T)]
    mx = max(gas)
    gas = _mix(gas, [mx, mx, mx], _ss(F['WHITE_LO'], F['WHITE_HI'], T))
    sb = flame_bb(min(T, F['SOOT_TMAX']))
    sk = flame_rad(T) * (1 + F['AB_LUM'] * ab) * soot_e * F['SOOT_GAIN']
    ck = chem * _ss(F['CHEM_TLO'], F['CHEM_THI'], T) * F['CHEM_GAIN'] * (0.25 + 0.75 * math.sqrt(rad))
    return [gas[i] + sb[i] * sk + F['CHEM_RGB'][i] * ck for i in range(3)]


def flame_smoke(soot):
    return _mix(FLAME['SMOKE_STEAM'], FLAME['SMOKE_SOOT'], min(max(soot, 0.0), 1.0))


def wgs_equilibrate(mol, T):
    """Water-gas shift CO + H2O <=> CO2 + H2 to equilibrium at T (K): K(T) = exp(4577.8/T - 4.33)."""
    if not T > 300:
        return mol
    K = math.exp(4577.8 / T - 4.33)
    co, w, c2, h = mol['CO'], mol['H2O'], mol['CO2'], mol['H2']
    a = 1 - K
    b = c2 + h + K * (co + w)
    c = c2 * h - K * co * w
    lo, hi = -min(c2, h), min(co, w)
    if abs(a) < 1e-9:
        xi = -c / b
    else:
        d = math.sqrt(max(b * b - 4 * a * c, 0.0))
        r1, r2 = (-b + d) / (2 * a), (-b - d) / (2 * a)
        xi = r1 if (lo - 1e-12 <= r1 <= hi + 1e-12) else r2
    xi = min(max(xi, lo), hi)
    mol['CO'] -= xi
    mol['H2O'] -= xi
    mol['CO2'] += xi
    mol['H2'] += xi
    return mol


def comb_products(fuel_key, of=0.0, T=0.0):
    """Moles per mole of fuel and mass fractions of the exhaust at mass mixture ratio `of`."""
    f = FUEL_SPECIES.get(fuel_key, FUEL_SPECIES['rp1'])
    F = f['fuel']
    o = of if of > 0 else f['ofNom']
    O = o * F['M'] / f['ox']['M'] * f['ox']['O']
    O0 = O
    mol = dict(H2O=0.0, H2=0.0, CO=0.0, CO2=0.0, C=0.0, N2=F['N'] / 2.0, O2=0.0)
    mol['CO'] = min(F['C'], O)
    O -= mol['CO']
    mol['C'] = F['C'] - mol['CO']
    mol['H2O'] = min(F['H'] / 2.0, O)
    O -= mol['H2O']
    mol['H2'] = F['H'] / 2.0 - mol['H2O']
    r = F['C'] / max(O0, 1e-9)
    k = min(mol['CO'] / 2.0, F['C'] * SOOT_C_FRAC * _ss(0.30, 0.70, r))
    mol['CO'] -= 2 * k
    mol['CO2'] += k
    mol['C'] += k
    up = min(mol['CO'], O)
    mol['CO'] -= up
    mol['CO2'] += up
    O -= up
    mol['O2'] = O / 2.0
    if T > 0:
        wgs_equilibrate(mol, T)
    mass, tot = {}, 0.0
    for g in mol:
        mass[g] = mol[g] * _MOLAR[g]
        tot += mass[g]
    for g in mass:
        mass[g] /= tot
    return dict(mol=mol, mass=mass, of=o, phi=f['ofStoich'] / o)


def flame_species(fuel_key, of, T=None, p_rel=1.0, gas=None):
    """Exhaust description: soot, chem, unburnt fraction yf, smoke opacity, steam, nox, products."""
    f = FUEL_SPECIES.get(fuel_key, FUEL_SPECIES['rp1'])
    pr = comb_products(fuel_key, of, T)
    m = pr['mass']
    g = gas or {}
    p_rel = max(p_rel, 0.0)
    soot = min(1.0, m['C'] / SOOT_REF)
    hmass = f['fuel']['H'] / f['fuel']['M']
    lean = max(0.0, pr['of'] / f['ofStoich'] - 1)
    chem = min(1.0, CHEM_K * hmass * (1 - 0.6 * soot) * (1 + 0.3 * min(lean, 1)))
    cond = _ss(0.002, 0.05, p_rel)
    tau = K_SOOT * m['C'] + K_STEAM * m['H2O'] * cond
    nox = min(((g.get('N2', 0.0)) * (g.get('O2', 0.0))) / (0.781 * 0.209) * min(p_rel, 1.0), 1.0)
    return dict(soot=soot, chem=chem, yf=m['CO'] + m['H2'] + m['C'], of=pr['of'], phi=pr['phi'], products=m, nox=nox,
                smkMax=1 - math.exp(-tau / 0.5), steam=m['H2O'] * cond)


def fuel_key_for(fuel_name, ox_name=''):
    """Map a fuel / oxidiser fluid name to a propellant row of the table."""
    n = (fuel_name + ' ' + ox_name).lower()
    if 'hydrazine' in n or 'mmh' in n or 'n2o4' in n or 'hypergol' in n:
        return 'hyp'
    if 'lh2' in n or 'hydrogen' in n:
        return 'lh2'
    if 'ch4' in n or 'methane' in n:
        return 'ch4'
    return 'rp1'


# ------------------------------------------------------------------ nozzle gas dynamics (isentropic, perfect gas)
R_UNIV = 8.314462618


def r_specific(molar_kg_mol):
    return R_UNIV / molar_kg_mol


def critical_pressure_ratio(gamma):
    """p* / p0 at which the throat reaches Mach 1."""
    return (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))


def choked_mass_flux(p0, T0, gamma, R):
    """Mass flow per unit throat area (kg/s/m2) of a choked nozzle."""
    return p0 * math.sqrt(gamma / (R * T0)) * (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))


def throat_velocity(T0, gamma, R):
    """Sonic velocity at the throat: a* = sqrt(2 gamma R T0 / (gamma + 1))."""
    return math.sqrt(2.0 * gamma * R * T0 / (gamma + 1.0))


def area_ratio(M, gamma):
    """A / A* for Mach M."""
    return (1.0 / M) * ((2.0 / (gamma + 1.0)) * (1.0 + 0.5 * (gamma - 1.0) * M * M)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))


def mach_from_area(ar, gamma, supersonic=True):
    """Solve A/A* = ar for the supersonic (or subsonic) Mach number by bisection."""
    ar = max(ar, 1.0)
    lo, hi = (1.0, 60.0) if supersonic else (1e-4, 1.0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        a = area_ratio(mid, gamma)
        if (a < ar) == supersonic:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def exit_state(p0, T0, gamma, R, ar):
    """Exit Mach, temperature, pressure and velocity of an isentropic nozzle of area ratio `ar` = Ae / A*."""
    Me = mach_from_area(ar, gamma, True)
    f = 1.0 + 0.5 * (gamma - 1.0) * Me * Me
    Te = T0 / f
    pe = p0 * f ** (-gamma / (gamma - 1.0))
    return dict(M=Me, T=Te, p=pe, v=Me * math.sqrt(gamma * R * Te))


def chamber_pressure(m_gas, T, V, R):
    """Ideal-gas pressure of m_gas kg of exhaust at T in the volume V."""
    return m_gas * R * T / max(V, 1e-12)


def fade_factor(r, r_domain, fade_frac=0.25):
    """1 inside, falling smoothly to 0 at the domain radius over the outer `fade_frac` of it (numpy-friendly)."""
    r0 = r_domain * (1.0 - fade_frac)
    t = np.clip((np.asarray(r, float) - r0) / max(r_domain - r0, 1e-9), 0.0, 1.0)
    return 1.0 - t * t * (3.0 - 2.0 * t)


# ------------------------------------------------------------------ vectorised (numpy) versions for per-particle colours
_BB = np.array(FLAME['BB'], float)


def flame_bb_np(T):
    T = np.asarray(T, float)
    return np.stack([np.interp(T, _BB[:, 0], _BB[:, c]) for c in (1, 2, 3)], axis=-1)


def _ss_np(a, b, x):
    t = np.clip((np.asarray(x, float) - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def flame_emission_np(tn, soot, chem, ab=0.0):
    """Array version of flame_emission (same formula): tn, soot, chem, ab broadcast; returns (..., 3)."""
    F = FLAME
    tn = np.clip(np.asarray(tn, float), 0, 1)
    T = F['TLO'] + tn * (F['THI'] - F['TLO'])
    soot, chem, ab = np.broadcast_arrays(np.asarray(soot, float), np.asarray(chem, float), np.asarray(ab, float))
    rad = (T / F['TREF']) ** 4 * (1 + F['AB_LUM'] * ab)
    soot_e = soot * (1 - F['SOOT_BURN'] * ab)
    gas = flame_bb_np(T) * (rad * F['GAS_EPS'] * (1 - F['CHEM_DIM'] * chem))[..., None]
    mx = gas.max(axis=-1, keepdims=True)
    wf = _ss_np(F['WHITE_LO'], F['WHITE_HI'], T)[..., None]
    gas = gas + (mx - gas) * wf
    sb = flame_bb_np(np.minimum(T, F['SOOT_TMAX']))
    sk = ((T / F['TREF']) ** 4 * (1 + F['AB_LUM'] * ab) * soot_e * F['SOOT_GAIN'])[..., None]
    ck = (chem * _ss_np(F['CHEM_TLO'], F['CHEM_THI'], T) * F['CHEM_GAIN'] * (0.25 + 0.75 * np.sqrt(rad)))[..., None]
    return gas + sb * sk + np.array(F['CHEM_RGB']) * ck
