"""Phase and chemistry transitions for Fluid mode (no bpy): a JSON rule table, animated, with mass and enthalpy bookkeeping.

A rule moves composition from species `from` to species `to` over time (the fraction changes continuously, so colour, density, heat capacity and
opacity interpolate and nothing pops):

    {"name": "boil", "from": "Water", "to": "Steam", "kind": "GAS", "tmin": 373.15, "width": 4.0, "rate": 1.5, "q": -2.26e6}

  kind    GAS      the product is a gas: a particle whose gas fraction passes 0.5 becomes a gas parcel (hysteresis: back below 0.35), so it rises,
                   expands and carries pressure; the liquid surface and the solver's liquid projection stop seeing it.
          CRYSTAL  the product is a solid: needs a nucleus.  `seeds` random nuclei (or `S.nuc` set by the caller) convert first; the solid then grows
                   outward, one neighbour shell per 1/rate seconds, so a crystal grows from its nucleation points.  A particle whose solid fraction
                   passes 0.6 is frozen in place (position and velocity pinned); back below 0.45 it is free again (melting).
          CHEM     plain conversion (oxidation, rust tint, burning): only the composition, colour and density change.
  tmin / tmax    temperature window (K) in which the rule runs; `width` smooths the onset (rate scales 0 -> 1 over `width` K from the edge)
  pmin / pmax    optional absolute pressure window (Pa, uses the solver pressure output); rmin / rmax optional density window (kg/m3)
  rate           GAS / CRYSTAL: converted fraction per second (a full change takes 1 / rate s); CHEM: first-order, a fraction `rate` of the remaining
                 reactant per second (exponential approach)
  q              J/kg released by the conversion (negative = absorbed: latent heat of vaporisation; positive: fusion / burning).  The heat is taken
                 from / given to the particle's own enthalpy, m cp T, exactly: mass is relabelled 1:1 so mass is conserved, and the enthalpy changes
                 by q x converted mass only.
  nuc_r          crystal growth reach (m, default 1.6 dx);  seeds  number of nucleation points (default 1 for CRYSTAL)
Species carry `phase` ('liquid', 'gas', 'solid').  Presets: see PRESETS (water -> steam, wax melt / solidify, fuel + oxidiser -> flame -> smoke,
salt / crystal growth, rust tint)."""
import json
import numpy as np


def rule_def(**kw):
    r = dict(name='phase', kind='CHEM', a=None, b=None, tmin=-1e9, tmax=1e9, width=4.0, rate=1.0, q=0.0, pmin=-1e18, pmax=1e18, rmin=0.0, rmax=1e18,
             nuc_r=None, seeds=None)
    if 'from' in kw:
        kw['a'] = kw.pop('from')
    if 'to' in kw:
        kw['b'] = kw.pop('to')
    r.update(kw)
    return r


def _hyst(frac, state, on, off):
    return np.where(state, frac >= off, frac >= on)


def init(S, rules):
    """Allocate the per-particle phase state on solver S (called once from _init_species)."""
    S.phase_rules = [rule_def(**r) for r in (rules or [])]
    S.gasp = S.gas[S.fid].copy() if S.n else np.zeros(0, bool)
    S.frozen = np.zeros(S.n, bool)
    S.nuc = np.zeros(S.n, bool)
    S.phase_stats = dict(converted=0.0, heat=0.0)
    S._nuc_done = set()
    names = {nm for r in S.phase_rules for nm in (r['a'], r['b'])}
    for f, si in enumerate(S.sp_of_fluid):
        if S.SS.sp[si]['name'] in names:
            S.tref[f] = 293.15          # species densities are defined at 293.15 K, whatever temperature the fluid was filled at
    ph = [s.get('phase', 'liquid') for s in S.SS.sp]
    S._sp_gas = np.array([p == 'gas' for p in ph], float)
    S._sp_solid = np.array([p == 'solid' for p in ph], float)


def step(S, dt):
    """One substep of every rule (called by the solver after mixing).  Returns the converted mass (kg)."""
    if not S.phase_rules or S.n == 0:
        return 0.0
    SS, C, T = S.SS, S.C, S.T
    n = S.n
    moved = np.zeros(n, bool)
    tot = 0.0
    heat = 0.0
    for r in S.phase_rules:
        a, b = SS.idx[r['a']], SS.idx[r['b']]
        el = (C[:, a] > 1e-9) & (T >= r['tmin']) & (T <= r['tmax'])
        if r['pmin'] > -1e17 or r['pmax'] < 1e17:
            P = S.pressure if getattr(S, 'pressure', None) is not None and len(S.pressure) == n else S.env.pressure_at(S.x[:, 2])
            el &= (P >= r['pmin']) & (P <= r['pmax'])
        if r['rmin'] > 0 or r['rmax'] < 1e17:
            rho = S.rho_particles()
            el &= (rho >= r['rmin']) & (rho <= r['rmax'])
        w = r['width']
        edge = np.ones(n)
        if r['tmin'] > -1e8:
            edge = np.minimum(edge, np.clip((T - r['tmin']) / max(w, 1e-6), 0.0, 1.0))
        if r['tmax'] < 1e8:
            edge = np.minimum(edge, np.clip((r['tmax'] - T) / max(w, 1e-6), 0.0, 1.0))
        rate = r['rate'] * np.maximum(edge, 0.02)
        if r['kind'] == 'CRYSTAL':
            from .fluidmode import neighbor_pairs
            key = id(r)
            if key not in S._nuc_done:
                S._nuc_done.add(key)
                cand = np.flatnonzero(el & (C[:, a] > 0.5))
                wd = S.wall_dist[cand] if len(S.wall_dist) == n else None
                if wd is not None and ((wd > -1.0) & (wd < 2.0 * S.dx)).any():     # heterogeneous nucleation: at the container walls first
                    cand = cand[(wd > -1.0) & (wd < 2.0 * S.dx)]
                k = int(r['seeds'] if r['seeds'] is not None else 1)
                if len(cand) and k > 0:
                    S.nuc[S.rng.choice(cand, size=min(k, len(cand)), replace=False)] = True
            solid = C[:, b] >= 0.5       # a neighbour must be half grown before it can seed the next shell, so the front advances at a visible speed
            reach = r['nuc_r'] or 1.6 * S.dx
            near = np.zeros(n, bool)
            cand = np.flatnonzero(el)
            sol = np.flatnonzero(solid)
            if len(sol) and len(cand):
                sub = np.concatenate([sol, cand])
                I, J = neighbor_pairs(S.x[sub], reach)
                ns = len(sol)
                m1 = (I < ns) & (J >= ns)
                m2 = (J < ns) & (I >= ns)
                near[sub[J[m1]]] = True
                near[sub[I[m2]]] = True
            el &= near | S.nuc | solid
        if r['kind'] == 'CHEM':       # first order: a fraction `rate` of the remaining reactant per second (exponential approach)
            ds = np.where(el, C[:, a] * (1.0 - np.exp(-rate * dt)), 0.0)
        else:                         # GAS / CRYSTAL: progress per second (a complete conversion takes 1 / rate seconds)
            ds = np.where(el, np.minimum(C[:, a], rate * dt), 0.0)
        if not ds.any():
            continue
        idx = np.flatnonzero(ds > 0)
        m = S.mass_ref[idx]
        T0 = 293.15                      # enthalpy reference: species enthalpies are zero at 293.15 K, so converting at constant T does not change T
        H = m * S.cp_p[idx] * (T[idx] - T0) + m * r['q'] * ds[idx]
        C[idx, a] -= ds[idx]
        C[idx, b] += ds[idx]
        S._derive(idx)
        T[idx] = T0 + H / np.maximum(m * S.cp_p[idx], 1e-12)
        moved[idx] = True
        tot += float((m * ds[idx]).sum())
        heat += float((m * r['q'] * ds[idx]).sum())
    if moved.any():
        idx = np.flatnonzero(moved)
        gf = C[idx] @ S._sp_gas
        sf = C[idx] @ S._sp_solid
        S.gasp[idx] = S.gas[S.fid[idx]] | _hyst(gf, S.gasp[idx] & ~S.gas[S.fid[idx]], 0.5, 0.35)
        S.frozen[idx] = _hyst(sf, S.frozen[idx], 0.6, 0.45)
        S.v[idx[S.frozen[idx]]] = 0.0
    S.phase_stats['converted'] += tot
    S.phase_stats['heat'] += heat
    return tot


# ----------------------------------------------------------------------------- presets (JSON authoring): species + rules
PRESETS = {
    'water_steam': dict(
        species=[dict(name='Water', color=(0.25, 0.55, 0.95), density=997.0, cp=4181.0, beta=2.1e-4, priority=0, visc=1e-3, phase='liquid'),
                 dict(name='Steam', color=(0.93, 0.95, 1.0), density=0.75, cp=2010.0, beta=0.0, priority=0, visc=1.3e-5, phase='gas')],
        phase=[dict(name='boil', kind='GAS', a='Water', b='Steam', tmin=373.15, width=3.0, rate=1.2, q=-2.26e6),
               dict(name='condense', kind='CHEM', a='Steam', b='Water', tmax=370.0, width=3.0, rate=1.0, q=2.26e6)]),
    'wax': dict(
        species=[dict(name='Wax melt', color=(1.0, 0.45, 0.12), density=900.0, cp=2200.0, beta=7e-4, priority=0, visc=0.05, phase='liquid'),
                 dict(name='Wax solid', color=(0.98, 0.9, 0.7), density=930.0, cp=2100.0, beta=3e-4, priority=0, visc=0.05, phase='solid')],
        phase=[dict(name='solidify', kind='CRYSTAL', a='Wax melt', b='Wax solid', tmax=325.0, width=4.0, rate=1.5, q=1.7e5, seeds=3),
               dict(name='melt', kind='CHEM', a='Wax solid', b='Wax melt', tmin=330.0, width=4.0, rate=1.5, q=-1.7e5)]),
    'fuel_flame_smoke': dict(
        species=[dict(name='Fuel', color=(0.9, 0.7, 0.2), density=800.0, cp=2000.0, beta=7e-4, priority=0, visc=2e-3, phase='liquid'),
                 dict(name='Flame', color=(1.0, 0.55, 0.12), density=0.3, cp=1500.0, beta=0.0, priority=0, visc=3e-5, phase='gas'),
                 dict(name='Smoke', color=(0.12, 0.12, 0.13), density=0.5, cp=1200.0, beta=0.0, priority=0, visc=2e-5, phase='gas')],
        phase=[dict(name='burn', kind='GAS', a='Fuel', b='Flame', tmin=600.0, width=30.0, rate=2.0, q=4.0e6),
               dict(name='soot', kind='CHEM', a='Flame', b='Smoke', tmax=950.0, width=100.0, rate=1.5, q=0.0)]),
    'salt_crystal': dict(
        species=[dict(name='Brine', color=(0.75, 0.85, 0.95), density=1200.0, cp=3500.0, beta=3e-4, priority=0, visc=1.5e-3, phase='liquid'),
                 dict(name='Salt crystal', color=(0.97, 0.97, 1.0), density=2160.0, cp=860.0, beta=1e-4, priority=0, visc=1.5e-3, phase='solid')],
        phase=[dict(name='crystallise', kind='CRYSTAL', a='Brine', b='Salt crystal', tmax=300.0, width=3.0, rate=0.8, q=0.0, seeds=4)]),
    'rust': dict(
        species=[dict(name='Iron', color=(0.55, 0.57, 0.6), density=7800.0, cp=450.0, beta=3e-5, priority=0, visc=1e-3, phase='liquid'),
                 dict(name='Rust', color=(0.55, 0.27, 0.1), density=5200.0, cp=650.0, beta=3e-5, priority=0, visc=1e-3, phase='liquid')],
        phase=[dict(name='oxidise', kind='CHEM', a='Iron', b='Rust', rate=0.15, q=0.0)]),
}


def preset(key):
    """A deep copy of a preset: dict(species=[...], phase=[...]); pass species= and phase= to FluidSolver.build, or to_json for the authoring file."""
    return json.loads(json.dumps(PRESETS[key]))


def to_json(key):
    return json.dumps(PRESETS[key], indent=1)


def from_json(text):
    d = json.loads(text)
    return d.get('species', []), d.get('phase', [])
