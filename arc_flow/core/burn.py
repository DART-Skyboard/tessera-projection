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


LEDGER_ROLES = ('CHAMBER', 'TRANSPORT')       # a combustion chamber is just another channel by geometry: every channel behaves like a chamber


def _groups(w, parts):
    """Connected groups of channel / chamber parts (an engine = its lines + chamber + nozzle): union over the openings that open into a neighbouring channel."""
    par = {i: i for i in parts}

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    for o in w.openings:
        if o['part'] in par and o.get('partner', -1) in par:
            par[find(o['part'])] = find(o['partner'])
    roots = {}
    return [roots.setdefault(find(i), len(roots)) for i in parts], len(roots)


def _geometry(w):
    """Chambers (control volumes), one nozzle (throat, exit) per connected group of channel parts, from the part openings."""
    w.chambers = []
    w.nozzles = []
    w.nozzle = None
    led = [i for i in w.active if w.parts[i].role in LEDGER_ROLES]
    gid, ng = _groups(w, led)
    grp_of = dict(zip(led, gid))
    for g in range(ng):
        members = [i for i in led if grp_of[i] == g]
        rad = [o for o in w.openings if o['part'] in members and w.parts[o['part']].role == 'CHAMBER']
        exits = [o for o in rad if o['state'] == 'OPEN']
        if not exits:                                   # no nozzle part: the channel's own open ends are the exit (no expansion, throat = exit)
            rad = [o for o in w.openings if o['part'] in members]
            exits = [o for o in rad if o['state'] == 'OPEN']
        nz = None
        if rad and exits:
            throat = min(rad, key=lambda o: o['radius'])
            ex = max(exits, key=lambda o: o['radius'])
            phi_t = float(w._gs(w.net['dist'], throat['centre'][None])[0]) if w.net is not None else 0.0
            nz = dict(r_t=throat['radius'], r_e=ex['radius'], phi_t=phi_t, centre=ex['centre'], normal=np.asarray(ex['normal'], float), group=g)
        w.nozzles.append(nz)
    for i in led:
        q = w.parts[i]
        V = max(q.field.cavity_volume(), 1e-9) * w.scale[i] ** 3
        w.chambers.append(dict(part=i, V=V, m=0.0, budget=0.0, born=0.0, T=300.0, P=w.env.pressure, group=grp_of[i]))
    w.nozzle = next((n for n in w.nozzles if n is not None), None)
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
    w.engines = [dict(P_c=w.env.pressure, T=300.0, mdot_in=0.0, mdot_out=0.0, choked=False, M_exit=0.0, v_exit=0.0, v_throat=0.0, vscale=1.0, p_exit=0.0,
                      thrust=0.0, thrust_vec=np.zeros(3), isp=0.0, nozzle=nz) for nz in w.nozzles]


def _nz(w, c):
    """Nozzle that drains chamber ledger c."""
    return w.nozzles[c['group']] if w.nozzles and 0 <= c['group'] < len(w.nozzles) else None


def _noz_state(w, T0, nz=None, p=None):
    """Physical choked / expanded state of a nozzle at a chamber pressure and temperature."""
    nz = nz or w.nozzle
    p = w.bstats['P_c'] if p is None else p
    if nz is None:
        return None
    g, R = w.exh_gamma, w.exh_R
    ar = (nz['r_e'] / max(nz['r_t'], 1e-9)) ** 2
    ex = FL.exit_state(max(p, 1.0), T0, g, R, ar)
    return dict(ar=ar, exit=ex, vt=FL.throat_velocity(T0, g, R))


def mdot_out(w, P, T, nz=None):
    """Mass flow through the throat for chamber pressure P (Pa), temperature T: choked above the critical pressure ratio, subsonic below."""
    nz = nz or w.nozzle
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
    """Integrate the chamber gas ledgers over dt: p_c = m R T / V, drained at the throat mass flow of the ledger's nozzle (one per engine / channel group)."""
    pe = w.env.pressure
    R = w.exh_R
    ng = max(len(w.nozzles), 1)
    G = [dict(P_c=pe, T=300.0, mdot_in=0.0, mdot_out=0.0, m=0.0) for _ in range(ng)]
    for c in w.chambers:
        g = G[c['group']]
        nz = _nz(w, c)
        g['mdot_in'] += c.pop('born', 0.0) / max(dt, 1e-9)
        c['born'] = 0.0
        T = max(c['T'], 300.0)
        t = 0.0
        out_tot = 0.0
        n = 0
        while t < dt - 1e-12 and n < 400:
            P = pe + c['m'] * R * T / c['V']
            md = mdot_out(w, P, T, nz)
            tau = c['m'] / max(md, 1e-9) if md > 0 else dt
            h = min(dt - t, max(0.2 * tau, dt / 400.0))
            dm = min(md * h, c['m'])
            c['m'] -= dm
            out_tot += dm
            t += h
            n += 1
        c['budget'] = min(c['budget'] + out_tot, 2.0 * out_tot + 4.0 * max(float(w.mass.max()), 1e-6))
        c['P'] = pe + c['m'] * R * T / c['V']
        if c['P'] > g['P_c']:
            g['P_c'], g['T'] = c['P'], T
        g['m'] += c['m']
        g['mdot_out'] += out_tot / max(dt, 1e-9)
    eng = []
    for k, g in enumerate(G):
        nz = w.nozzles[k] if k < len(w.nozzles) else None
        e = dict(P_c=g['P_c'], T=g['T'], mdot_in=g['mdot_in'], mdot_out=g['mdot_out'], choked=False, M_exit=0.0, v_exit=0.0, v_throat=0.0, vscale=1.0, p_exit=0.0,
                 thrust=0.0, thrust_vec=np.zeros(3), isp=0.0, nozzle=nz)
        if nz is not None:
            st = _noz_state(w, max(g['T'], 300.0), nz, g['P_c'])
            e['choked'] = bool(pe / max(g['P_c'], 1.0) <= FL.critical_pressure_ratio(w.exh_gamma))
            e['v_throat'] = st['vt']
            e['p_exit'] = st['exit']['p']
            if e['choked']:
                e['M_exit'], e['v_exit'] = st['exit']['M'], st['exit']['v']
            e['vscale'] = min(1.0, w.p.vmax / max(max(st['exit']['v'], st['vt']), 1e-9))
            # thrust of the nozzle on the engine (reaction to the exhaust): F = mdot v_e + (p_e - p_amb) A_e, pointing opposite to the exit normal
            if e['choked']:
                e['thrust'] = e['mdot_out'] * st['exit']['v'] + (st['exit']['p'] - pe) * math.pi * nz['r_e'] ** 2
            e['thrust_vec'] = -nz['normal'] * e['thrust']
            e['isp'] = e['thrust'] / max(e['mdot_out'] * 9.80665, 1e-9) if e['choked'] else 0.0
        eng.append(e)
    w.engines = eng
    b = w.bstats
    top = max(eng, key=lambda e: e['P_c'])
    b['P_c'] = top['P_c'] if w.chambers else pe
    b['mdot_in'] = sum(e['mdot_in'] for e in eng)
    b['mdot_out'] = sum(e['mdot_out'] for e in eng)
    if w.nozzle is not None and w.chambers:
        act = [e for e in eng if e['nozzle'] is not None]
        b['choked'] = any(e['choked'] for e in act)
        b['v_throat'] = top['v_throat'] if top['nozzle'] is not None else max(e['v_throat'] for e in act)
        b['M_exit'] = max(e['M_exit'] for e in act)
        b['v_exit'] = max(e['v_exit'] for e in act)
        b['p_exit'] = top['p_exit'] if top['nozzle'] is not None else act[0]['p_exit']
        b['vscale'] = min(e['vscale'] for e in act)
        b['thrust'] = sum(e['thrust'] for e in act)
        b['thrust_vec'] = sum((e['thrust_vec'] for e in act), np.zeros(3))
        b['isp'] = b['thrust'] / max(b['mdot_out'] * 9.80665, 1e-9) if b['choked'] else 0.0


def particle_group(w, idx):
    """Engine group of gas particles: the group of the chamber they were born in, else the group of the nearest nozzle exit."""
    idx = np.asarray(idx)
    h = w.home[idx]
    lg = np.array([c['group'] for c in w.chambers], np.int32) if w.chambers else np.zeros(1, np.int32)
    out = np.where(h >= 0, lg[np.clip(h, 0, len(lg) - 1)], -1)
    miss = out < 0
    if miss.any():
        cen = np.array([n['centre'] if n is not None else np.full(3, 1e9) for n in w.nozzles])
        d = np.linalg.norm(w.pos[idx[miss]][:, None, :] - cen[None], axis=2)
        out[miss] = np.argmin(d, axis=1)
    return out


def nz_r_e(w):
    return w.nozzle['r_e']


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
        c = w.chambers[w.home[i]] if 0 <= w.home[i] < len(w.chambers) else max(w.chambers, key=lambda q: q['budget'])
        if c['budget'] >= w.mass[i]:
            c['budget'] -= w.mass[i]
            allow[k] = True
            w.bstats['vented_mass'] += float(w.mass[i])
    return allow


def gas_speed_cap(w, phi, rad_loc, grp=None):
    """Local gas speed along the nozzle: isentropic area-Mach (sonic at the throat, supersonic downstream), scaled to the world's speed limit.
    grp: engine group of each particle (per-engine throat, stagnation temperature and speed scale)."""
    g, R = w.exh_gamma, w.exh_R
    n = len(w.engines) if getattr(w, 'engines', None) else 0
    if grp is None or n == 0:
        grp = np.zeros(len(phi), np.int32)
    nzs = [e['nozzle'] or w.nozzle for e in w.engines] if n else [w.nozzle]
    r_t = np.array([z['r_t'] for z in nzs])[grp]
    phi_t = np.array([z['phi_t'] for z in nzs])[grp]
    sc = np.array([e['vscale'] for e in w.engines])[grp] if n else np.full(len(phi), w.bstats['vscale'])
    T0 = np.array([max(e['T'], 300.0) for e in w.engines])[grp] if n else np.full(len(phi), 3000.0)
    T0 = np.where(T0 <= 300.0, max((c['T'] for c in w.chambers), default=3000.0), T0)
    ar = np.clip((rad_loc / np.maximum(r_t, 1e-9)) ** 2, 1.0, 400.0)
    tab = _AR_TABLE(g)
    Ms = np.interp(ar, tab[0], tab[1])
    Mb = np.interp(ar, tab[0], tab[2])
    M = np.where(phi < phi_t, Ms, Mb)
    Tl = T0 / (1.0 + 0.5 * (g - 1.0) * M * M)
    return M * np.sqrt(g * R * Tl) * sc


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
    if w.nozzles and any(n is not None for n in w.nozzles):
        cen = np.array([n['centre'] if n is not None else np.full(3, 1e9) for n in w.nozzles])
        k = np.argmin(np.linalg.norm(x[sel][:, None, :2] - cen[None, :, :2], axis=2), axis=1)    # nearest nozzle axis: each engine has its own wall jet
        radial = x[sel] - np.column_stack([cen[k, 0], cen[k, 1], x[sel, 2]])
    else:
        radial = x[sel] * 0
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
