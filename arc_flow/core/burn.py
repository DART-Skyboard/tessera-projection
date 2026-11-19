"""Combustion chemistry and exhaust life cycle for the particle world (core/world.py calls into this when Params.chemistry is on).

Chain, in the order a propellant particle experiences it:
  1. tanks fill (World._init_container, shared level selection in core/fillsettle.py),
  2. fuel and oxidiser particles meet in the chamber: pairs in the same cell react (igniter assumed, or autoignition temperature) and become flame gas.  The pair's O/F
     sets soot / chemiluminescence through the web app's flame model (core/flame.py); the pair's mass goes into the chamber's gas ledger,
  3. the chamber is a lumped control volume: p_c = m R T / V from the gas ledger, drained through the throat at the choked mass flow (or the subsonic flow when the
     pressure ratio is below critical).  Only that much mass may leave the nozzle exit per second, so the pressure builds in the neck until inflow and outflow balance,
  4. the gas follows the nozzle: velocity from the isentropic area-Mach relation of the local cross-section (sonic at the throat, supersonic after it), scaled by a
     single factor so the fastest particle stays under Params.vmax (reported, the physical numbers are kept in stats),
  5. at the exit the jet hits the ground and spreads as a wall jet (stagnation: vertical momentum turns into radial),
  6. the flame gas cools by entrainment; below `smoke_T` it is converted to smoke (colour from the flame model: steam grey to soot black),
  7. beyond 75 percent of the domain radius (300 m default) particles fade out and are removed at the radius; a particle budget removes the oldest smoke first.
"""
import math

import numpy as np

from . import flame as FL
from . import fluids as F


def setup(w):
    N = w.N
    w.kind = np.zeros(N, np.int8)          # 0 propellant, 1 flame gas, 2 smoke
    w.soot = np.zeros(N)
    w.chem = np.zeros(N)
    w.smk = np.zeros(N)                    # smoke opacity ceiling (flame model)
    w.fade = np.ones(N)
    w.home = -np.ones(N, np.int32)         # chamber ledger a gas particle was born in
    _geometry(w)
    reset(w)


def reset(w):
    w.kind[:] = 0
    w.soot[:] = 0
    w.chem[:] = 0
    w.smk[:] = 0
    w.fade[:] = 1.0
    w.home[:] = -1
    w.bstats = dict(reacted=0, burnt_mass=0.0, vented_mass=0.0, smoke=0, faded=0, removed_domain=0, removed_budget=0, ground=0,
                    P_c=w.env.pressure, mdot_in=0.0, mdot_out=0.0, choked=False, M_exit=0.0, v_exit=0.0, v_throat=0.0, vscale=1.0, p_exit=0.0, alive=0)
    for c in w.chambers:
        c['m'] = 0.0
        c['budget'] = 0.0
        c['born'] = 0.0
        c['T'] = 300.0


def _geometry(w):
    """Chambers (control volumes), throat and exit radii from the part openings."""
    w.chambers = []
    w.nozzle = None
    ch = [i for i in w.active if w.parts[i].role == 'CHAMBER']
    rad = []
    for o in w.openings:
        if w.parts[o['part']].role == 'CHAMBER':
            rad.append(o)
    for i in ch:
        q = w.parts[i]
        V = max(q.field.cavity_volume(), 1e-9) * w.scale[i] ** 3
        w.chambers.append(dict(part=i, V=V, m=0.0, budget=0.0, born=0.0, T=300.0, P=w.env.pressure))
    exits = [o for o in rad if o['state'] == 'OPEN']
    if rad and exits:
        throat = min(rad, key=lambda o: o['radius'])
        ex = max(exits, key=lambda o: o['radius'])
        phi_t = float(w._gs(w.net['dist'], throat['centre'][None])[0]) if w.net is not None else 0.0
        w.nozzle = dict(r_t=throat['radius'], r_e=ex['radius'], phi_t=phi_t, centre=ex['centre'])
    w.dom_c = np.asarray(w._world_bounds()[0] + w._world_bounds()[1], float) * 0.5 if w.active else np.zeros(3)
    ids = {fl['spec']['role']: n for n, fl in enumerate(w.fluids) if fl['spec']['role'] in ('FUEL', 'OXIDIZER')}
    w.fuel_i, w.ox_i = ids.get('FUEL', -1), ids.get('OXIDIZER', -1)
    names = [fl['name'] for fl in w.fluids]
    w.fkey = FL.fuel_key_for(names[w.fuel_i] if w.fuel_i >= 0 else '', names[w.ox_i] if w.ox_i >= 0 else '')
    exh = F.FLUIDS['EXHAUST']
    w.exh_M, w.exh_cp = exh['molar'], exh['cp']
    w.exh_R = F.R_GAS / w.exh_M
    cv = w.exh_cp - w.exh_R
    w.exh_gamma = float(w.exh_cp / cv) if cv > 1 else 1.2
    w.exh_gamma = min(max(w.exh_gamma, 1.15), 1.4)


def _noz_state(w, T0):
    """Physical choked / expanded state of the nozzle at the current chamber pressure and temperature."""
    p = w.bstats['P_c']
    nz = w.nozzle
    if nz is None:
        return None
    g, R = w.exh_gamma, w.exh_R
    ar = (nz['r_e'] / max(nz['r_t'], 1e-9)) ** 2
    ex = FL.exit_state(max(p, 1.0), T0, g, R, ar)
    return dict(ar=ar, exit=ex, vt=FL.throat_velocity(T0, g, R))


def mdot_out(w, P, T):
    """Mass flow through the throat for chamber pressure P (Pa), temperature T: choked above the critical pressure ratio, subsonic below."""
    nz = w.nozzle
    if nz is None or P <= w.env.pressure:
        return 0.0
    g, R = w.exh_gamma, w.exh_R
    At = math.pi * nz['r_t'] ** 2
    pe = w.env.pressure
    if pe / P <= FL.critical_pressure_ratio(g):
        return At * FL.choked_mass_flux(P, T, g, R)
    x = pe / P
    return At * P * math.sqrt(2 * g / ((g - 1) * R * T) * (x ** (2 / g) - x ** ((g + 1) / g)))


def react(w, dt):
    """Pair fuel and oxidiser particles that share a cell inside a chamber; the pairs become flame gas."""
    p = w.p
    if w.fuel_i < 0 or w.ox_i < 0 or not w.chambers:
        return
    if not (p.igniter or w.ignition > 0.05):
        return
    ch_parts = np.array([c['part'] for c in w.chambers])
    sel = np.flatnonzero((w.state == 1) & (w.kind == 0) & np.isin(w.pin, ch_parts))
    if len(sel) == 0:
        return
    fu = sel[w.pfluid[sel] == w.fuel_i]
    ox = sel[w.pfluid[sel] == w.ox_i]
    if len(fu) == 0 or len(ox) == 0:
        return
    cell = max(2.0 * float(np.median(w.rad[sel])), 1e-3)

    def keys(ix):
        c = np.floor(w.pos[ix] / cell).astype(np.int64)
        k = (c[:, 0] * 73856093) ^ (c[:, 1] * 19349663) ^ (c[:, 2] * 83492791)
        o = np.argsort(k, kind='stable')
        ks = k[o]
        first = np.r_[True, ks[1:] != ks[:-1]]
        start = np.maximum.accumulate(np.where(first, np.arange(len(ks)), 0))
        rank = np.arange(len(ks)) - start
        return ix[o], ks * 4096 + np.minimum(rank, 4095)
    fi, fk = keys(fu)
    oi, ok = keys(ox)
    _c, a, b = np.intersect1d(fk, ok, return_indices=True)
    if len(a) == 0:
        return
    pf, po = fi[a], oi[b]
    keep = w.rng.random(len(pf)) < (1.0 - math.exp(-p.burn_rate * dt))
    if not p.igniter:                                   # no igniter: the pair itself must be above the autoignition temperature (0 = hypergolic)
        t_auto = FL.FUEL_SPECIES[w.fkey]['tAuto']
        keep &= (t_auto == 0.0) | (0.5 * (w.temp[pf] + w.temp[po]) >= t_auto)
    pf, po = pf[keep], po[keep]
    if len(pf) == 0:
        return
    f_spec = w.fluids[w.fuel_i]['spec']
    tk = FL.FUEL_SPECIES[w.fkey]['tK']
    of = np.clip(w.mass[po] / np.maximum(w.mass[pf], 1e-12), 0.5, 12.0)
    Tf = f_spec['flame_t'] * tk
    sp = [FL.flame_species(w.fkey, float(o), Tf, w.env.pressure / F.P_ATM) for o in np.unique(np.round(of, 1))]
    lut = {round(float(o), 1): s for o, s in zip(np.unique(np.round(of, 1)), sp)}
    both = np.concatenate([pf, po])
    for ii, o in zip((pf, po), (of, of)):
        for j, i in enumerate(ii):
            s = lut[round(float(np.round(o[j], 1)), 1)]
            w.soot[i], w.chem[i], w.smk[i] = s['soot'], s['chem'], s['smkMax']
    w.kind[both] = 1
    w.temp[both] = Tf
    w.rad[both] *= p.gas_expand
    owner = w.pin[both]
    for c_i, c in enumerate(w.chambers):
        m = owner == c['part']
        if m.any():
            dm = float(w.mass[both][m].sum())
            c['m'] += dm
            c['born'] += dm
            c['T'] = Tf
            w.home[both[m]] = c_i
    w.bstats['reacted'] += len(pf)
    w.bstats['burnt_mass'] += float(w.mass[both].sum())


def chamber_update(w, dt):
    """Integrate the chamber gas ledgers over dt: p_c = m R T / V, drained at the throat mass flow."""
    pe = w.env.pressure
    R = w.exh_R
    P_max = 0.0
    din = 0.0
    dout = 0.0
    for c in w.chambers:
        din += c.pop('born', 0.0) / max(dt, 1e-9)
        c['born'] = 0.0
        T = max(c['T'], 300.0)
        t = 0.0
        out_tot = 0.0
        n = 0
        while t < dt - 1e-12 and n < 400:
            P = pe + c['m'] * R * T / c['V']
            md = mdot_out(w, P, T)
            tau = c['m'] / max(md, 1e-9) if md > 0 else dt
            h = min(dt - t, max(0.2 * tau, dt / 400.0))
            dm = min(md * h, c['m'])
            c['m'] -= dm
            out_tot += dm
            t += h
            n += 1
        c['budget'] = min(c['budget'] + out_tot, 2.0 * out_tot + 4.0 * max(float(w.mass.max()), 1e-6))
        c['P'] = pe + c['m'] * R * T / c['V']
        P_max = max(P_max, c['P'])
        dout += out_tot / max(dt, 1e-9)
    w.bstats['P_c'] = P_max if w.chambers else pe
    w.bstats['mdot_in'] = din
    w.bstats['mdot_out'] = dout
    if w.nozzle is not None and w.chambers:
        T0 = max(w.chambers[0]['T'], 300.0)
        st = _noz_state(w, T0)
        w.bstats['choked'] = bool(pe / max(w.bstats['P_c'], 1.0) <= FL.critical_pressure_ratio(w.exh_gamma))
        w.bstats['v_throat'] = st['vt']
        w.bstats['M_exit'] = st['exit']['M'] if w.bstats['choked'] else 0.0
        w.bstats['v_exit'] = st['exit']['v'] if w.bstats['choked'] else 0.0
        w.bstats['p_exit'] = st['exit']['p']
        v_phys = max(st['exit']['v'], st['vt'])
        w.bstats['vscale'] = min(1.0, w.p.vmax / max(v_phys, 1e-9))


def gas_rho(w, T):
    return w.env.pressure * w.exh_M / (F.R_GAS * np.maximum(T, 200.0)) + 0.0


def vent_gate(w, idx, h):
    """Which of the gas particles that crossed the exit this substep may leave (the choked mass flow sets the allowed mass); the rest are held back."""
    allow = np.zeros(len(idx), bool)
    if not w.chambers:
        return np.ones(len(idx), bool)
    order = np.argsort(w.age[idx])[::-1]
    for k in order:
        i = idx[k]
        c = w.chambers[w.home[i]] if 0 <= w.home[i] < len(w.chambers) else w.chambers[0]
        if c['budget'] >= w.mass[i]:
            c['budget'] -= w.mass[i]
            allow[k] = True
            w.bstats['vented_mass'] += float(w.mass[i])
    return allow


def gas_speed_cap(w, phi, rad_loc):
    """Local gas speed along the nozzle: isentropic area-Mach (sonic at the throat, supersonic downstream), scaled to the world's speed limit."""
    nz = w.nozzle
    s = w.bstats['vscale']
    g, R = w.exh_gamma, w.exh_R
    T0 = max(w.chambers[0]['T'], 300.0) if w.chambers else 3000.0
    ar = np.clip((rad_loc / max(nz['r_t'], 1e-9)) ** 2, 1.0, 400.0)
    Ms = np.interp(ar, _AR_TABLE(g)[0], _AR_TABLE(g)[1])
    Mb = np.interp(ar, _AR_TABLE(g)[0], _AR_TABLE(g)[2])
    M = np.where(phi < nz['phi_t'], Ms, Mb)
    Tl = T0 / (1.0 + 0.5 * (g - 1.0) * M * M)
    return M * np.sqrt(g * R * Tl) * s


_AR_CACHE = {}


def _AR_TABLE(g):
    if g not in _AR_CACHE:
        ar = np.linspace(1.0, 400.0, 800)
        ms = np.array([FL.mach_from_area(a, g, True) for a in ar])
        mb = np.array([FL.mach_from_area(a, g, False) for a in ar])
        _AR_CACHE[g] = (ar, ms, mb)
    return _AR_CACHE[g]


def free_gas(w, fre, x, v, h):
    """Free-flight bookkeeping for exhaust gas and smoke: cooling, flame to smoke conversion, ground wall jet.  Returns the modified (x, v)."""
    k = w.kind[fre]
    g = k > 0
    if not g.any():
        return x, v
    gi = fre[g]
    amb = w.env.temp
    p = w.p
    w.temp[gi] += (amb - w.temp[gi]) * (1.0 - math.exp(-h / p.cool_tau))
    # entrainment grows the plume
    w.rad[gi] = np.minimum(w.rad[gi] * math.exp(h * p.plume_growth), p.max_plume_radius)
    return x, v


def ground_jet(w, fre, x, v, below, h):
    """Stagnation at the ground: vertical momentum turns into a radial wall jet for gas particles."""
    kg = w.kind[fre] > 0
    sel = below & kg
    if not sel.any():
        return
    vz = v[sel, 2]
    radial = x[sel] - np.array([w.nozzle['centre'][0], w.nozzle['centre'][1], x[sel, 2].mean()]) if w.nozzle is not None else x[sel] * 0
    radial[:, 2] = 0
    rn = np.linalg.norm(radial, axis=1, keepdims=True)
    u = radial / np.maximum(rn, 1e-6)
    inc = np.maximum(-vz, 0.0)
    v[sel, 0] += u[:, 0] * 0.8 * inc
    v[sel, 1] += u[:, 1] * 0.8 * inc
    v[sel, 2] = np.maximum(vz, 0.0) * 0.0
    v[sel, 0] *= (1.0 - min(1.0 * h, 0.5))
    v[sel, 1] *= (1.0 - min(1.0 * h, 0.5))
    w.bstats['ground'] += int(sel.sum())


def life_cycle(w, dt):
    """Flame to smoke conversion, domain fade / removal and the particle budget (once per frame)."""
    p = w.p
    al = np.flatnonzero(w.state == 2)
    if len(al) == 0:
        w.bstats['alive'] = int((w.state > 0).sum())
        return
    fl = al[(w.kind[al] == 1) & (w.temp[al] < p.smoke_T)]
    if len(fl):
        w.kind[fl] = 2
        w.life[fl] = w.age[fl] + p.smoke_life
        w.bstats['smoke'] += len(fl)
    gas = al[w.kind[al] > 0]
    if len(gas):
        r = np.linalg.norm(w.pos[gas] - w.dom_c, axis=1)
        w.fade[gas] = FL.fade_factor(r, p.domain_radius, p.domain_fade)
        out = gas[r >= p.domain_radius]
        if len(out):
            w.state[out] = 0
            w.bstats['removed_domain'] += len(out)
        w.bstats['faded'] = int((w.fade[gas] < 0.999).sum())
    w.fade[al[w.kind[al] == 0]] = 1.0
    if p.max_alive > 0:
        alive = np.flatnonzero(w.state > 0)
        over = len(alive) - p.max_alive
        if over > 0:
            cand = np.flatnonzero((w.state == 2) & (w.kind == 2))
            if len(cand) < over:
                cand = np.flatnonzero((w.state == 2) & (w.kind > 0))
            cand = cand[np.argsort(-w.age[cand])][:over]
            w.state[cand] = 0
            w.bstats['removed_budget'] += len(cand)
    w.bstats['alive'] = int((w.state > 0).sum())


def afterburn(w, idx):
    """Unreacted propellant that leaves the chamber while the engine is lit burns on contact with the hot jet: it becomes flame gas at a lower temperature
    (fuel particles sooty, oxidiser particles clean)."""
    idx = idx[w.kind[idx] == 0]
    if len(idx) == 0 or not (w.p.igniter or w.ignition >= 0.2):
        return
    f_spec = w.fluids[w.fuel_i]['spec']
    tk = FL.FUEL_SPECIES[w.fkey]['tK']
    Tf = f_spec['flame_t'] * tk * 0.75
    rich = FL.flame_species(w.fkey, FL.FUEL_SPECIES[w.fkey]['ofNom'] * 0.7, Tf, w.env.pressure / F.P_ATM)
    lean = FL.flame_species(w.fkey, FL.FUEL_SPECIES[w.fkey]['ofNom'] * 1.4, Tf, w.env.pressure / F.P_ATM)
    isf = w.pfluid[idx] == w.fuel_i
    w.kind[idx] = 1
    w.temp[idx] = Tf
    w.rad[idx] *= w.p.gas_expand
    w.soot[idx] = np.where(isf, rich['soot'], lean['soot'])
    w.chem[idx] = np.where(isf, rich['chem'], lean['chem'])
    w.smk[idx] = np.where(isf, rich['smkMax'], lean['smkMax'])
    w.bstats['afterburn'] = w.bstats.get('afterburn', 0) + len(idx)
