"""Species, composition, mixing and merge / reaction rules for Fluid mode (no bpy).  A simplified, documented model, not a chemistry engine.

Every particle carries a composition vector C (mass fractions of species, summing to 1), plus temperature, velocity, a cluster id and a priority
(the priority of its dominant species).  Derived per particle:  colour (subtractive pigment mix: weighted geometric mean of the species colours, so
yellow + blue gives green), density (ideal mixing, 1/rho = sum w/rho_s), heat capacity and thermal expansion (mass weighted).

Rules per species pair (editable table, JSON-extensible; default: same species MISCIBLE, different species follow the fluids' `miscible` flag):
  IMMISCIBLE  stays separate (interface cohesion of the solver),
  MISCIBLE    composition, colour and enthalpy diffuse between neighbouring particles (Fick-style exchange, rate set by `rate` and by the relative speed),
  ABSORB      the higher-PRIORITY species absorbs the other on contact: lower-priority particles drift toward the dominant composition at `rate`, snap to it
              at a 50 percent majority, and take over its cluster id; cluster ids propagate through connected particles from the root,
  REACT       a + b -> c when the pair is in contact and T in [tmin, tmax], relative speed >= vmin, density ratio in [rmin, rmax] and the dominant
              fractions >= cmin; heat `q` (J/kg, positive = exothermic) goes into the two particles' temperature; c has its own properties.
Mass is conserved (particles keep their mass; species mass is exchanged, ABSORB / REACT relabel mass 1:1).  Counts of merges and reactions are kept."""
import json
import numpy as np

IMMISCIBLE, MISCIBLE, ABSORB, REACT = 'IMMISCIBLE', 'MISCIBLE', 'ABSORB', 'REACT'


def species_def(name, color=(0.5, 0.5, 0.5), density=1000.0, cp=4186.0, beta=2e-4, priority=0, visc=1e-3):
    return dict(name=name, color=tuple(color), density=float(density), cp=float(cp), beta=float(beta), priority=int(priority), visc=float(visc))


def rule_def(a, b, outcome=MISCIBLE, **kw):
    r = dict(a=a, b=b, outcome=outcome, rate=2.0, product=None, tmin=0.0, tmax=1e9, vmin=0.0, rmin=0.0, rmax=1e9, cmin=0.6, q=0.0)
    r.update(kw)
    return r


class SpeciesSystem:
    def __init__(self, species, rules=()):
        self.sp = list(species)
        self.idx = {s['name']: i for i, s in enumerate(self.sp)}
        self.rules = [dict(r) for r in rules]
        self.ns = len(self.sp)
        self.color = np.array([s['color'] for s in self.sp], float).reshape(-1, 3)
        self.dens = np.array([s['density'] for s in self.sp], float)
        self.cp = np.array([s['cp'] for s in self.sp], float)
        self.beta = np.array([s['beta'] for s in self.sp], float)
        self.prio = np.array([s['priority'] for s in self.sp], int)
        self.visc = np.array([s['visc'] for s in self.sp], float)
        self.stats = dict(merges=0, reactions=0)
        self._table()

    def _table(self):
        ns = self.ns
        self.out = np.full((ns, ns), '', dtype=object)
        self.rate = np.zeros((ns, ns))
        self.react = {}
        self.is_absorber = np.zeros(ns, bool)
        for r in self.rules:
            a, b = self.idx[r['a']], self.idx[r['b']]
            for p, q in ((a, b), (b, a)):
                self.out[p, q] = r['outcome']
                self.rate[p, q] = r['rate']
            if r['outcome'] == ABSORB:
                self.is_absorber[a if self.prio[a] >= self.prio[b] else b] = True
            if r['outcome'] == REACT:
                self.react[(a, b)] = self.react[(b, a)] = r

    def outcome(self, a, b, default=IMMISCIBLE):
        o = self.out[a, b]
        if o:
            return o
        return MISCIBLE if a == b else default

    # ---- JSON preset: {"species": [...], "rules": [...]}
    @staticmethod
    def from_json(text):
        d = json.loads(text)
        return SpeciesSystem([species_def(**s) for s in d.get('species', [])], [rule_def(**r) for r in d.get('rules', [])])

    def to_json(self):
        return json.dumps(dict(species=self.sp, rules=self.rules), indent=1)

    # ---- derived per-particle properties from the composition C (n, ns)
    def colour(self, C):
        w = C / np.maximum(C.sum(1, keepdims=True), 1e-12)
        lg = np.log(np.clip(self.color, 0.02, 1.0))
        return np.exp(w @ lg)

    def density(self, C):
        w = C / np.maximum(C.sum(1, keepdims=True), 1e-12)
        return 1.0 / np.maximum(w @ (1.0 / self.dens), 1e-12)

    def cp_mix(self, C):
        w = C / np.maximum(C.sum(1, keepdims=True), 1e-12)
        return w @ self.cp

    def beta_mix(self, C):
        w = C / np.maximum(C.sum(1, keepdims=True), 1e-12)
        return w @ self.beta

    def dominant(self, C):
        return np.argmax(C, axis=1)


def _scatter(n, I, J, vi, vj):
    out = np.zeros((n,) + vi.shape[1:])
    if vi.ndim == 1:
        out += np.bincount(I, vi, n) + np.bincount(J, vj, n)
    else:
        for c in range(vi.shape[1]):
            out[:, c] += np.bincount(I, vi[:, c], n) + np.bincount(J, vj[:, c], n)
    return out


def mix_step(SS, C, T, mass, cp, v, x, I, J, h, dt, liquid_default):
    """One mixing / merging / reaction step over neighbour pairs (I, J) (indices into the particle arrays).  Modifies C, T (in place).
    liquid_default: callable(a, b) -> outcome for species pairs without a rule (from the fluids' miscible flags).
    Returns dict(merges, reactions, cluster_pairs)."""
    n = len(C)
    info = dict(merges=0, reactions=0, absorb_pairs=(np.zeros(0, int), np.zeros(0, int)))
    if len(I) == 0:
        return info
    rij = x[I] - x[J]
    d = np.linalg.norm(rij, axis=1)
    m = d < h
    I, J, d = I[m], J[m], d[m]
    if len(I) == 0:
        return info
    dom = SS.dominant(C)
    da, db = dom[I], dom[J]
    out = np.array([SS.outcome(a, b, liquid_default(a, b)) for a, b in zip(da, db)], dtype=object) if len(I) < 40000 else \
        np.array([SS.out[a, b] or (MISCIBLE if a == b else liquid_default(a, b)) for a, b in zip(da, db)], dtype=object)
    w = (1.0 - d / h) ** 2
    vrel = np.linalg.norm(v[I] - v[J], axis=1)
    # ---- MISCIBLE: Fick-style exchange of composition and enthalpy (conserves species mass and heat)
    sel = np.flatnonzero(out == MISCIBLE)
    if len(sel):
        i, j = I[sel], J[sel]
        rate = np.maximum(SS.rate[da[sel], db[sel]], 0.0)
        rate = np.where(rate > 0, rate, 2.0)
        k = np.clip(rate * dt * w[sel] * (1.0 + np.minimum(vrel[sel] / 0.05, 9.0)) / 6.0, 0.0, 0.25)
        mu = mass[i] * mass[j] / (mass[i] + mass[j])
        dm = (k * mu)[:, None] * (C[j] - C[i])                 # species mass moved from j to i
        C_i_m = C * mass[:, None]
        add = _scatter(n, i, j, dm, -dm)
        C_i_m = C_i_m + add
        Ci, Cj = mass[i] * cp[i], mass[j] * cp[j]
        dE = k * (Ci * Cj / (Ci + Cj)) * (T[j] - T[i])
        T[:] = T + (np.bincount(i, dE, n) - np.bincount(j, dE, n)) / np.maximum(mass * cp, 1e-12)
        C[:] = C_i_m / mass[:, None]
    # ---- ABSORB: the higher priority species converts the other on contact
    sel = np.flatnonzero(out == ABSORB)
    if len(sel):
        i, j = I[sel], J[sel]
        pi, pj = SS.prio[da[sel]], SS.prio[db[sel]]
        up = pi > pj                                  # i dominates j
        dom_p = np.where(up, i, j)
        low_p = np.where(up, j, i)
        dsp = np.where(up, da[sel], db[sel])
        rate = np.maximum(SS.rate[da[sel], db[sel]], 0.1)
        step = np.clip(rate * dt * w[sel], 0.0, 0.5)
        # fraction moved toward the dominant species (mass relabelled 1:1)
        tgt = np.zeros_like(C)
        cnt = np.bincount(low_p, minlength=n).astype(float)
        np.add.at(tgt, (low_p, dsp), step)
        mv = np.minimum(tgt.sum(1), 1.0)
        lowmask = mv > 0
        if lowmask.any():
            Cl = C[lowmask]
            tg = tgt[lowmask]
            tot = np.maximum(tg.sum(1, keepdims=True), 1e-12)
            share = np.minimum(tg.sum(1), 1.0)[:, None]
            # remove `share` of the mass from the other species proportionally, give it to the dominant ones
            Cl_new = Cl * (1.0 - share) + share * tg / tot
            C[lowmask] = Cl_new
            info['merges'] = int(lowmask.sum())
        domi = SS.dominant(C)
        snap = (C.max(1) >= 0.5) & (C.max(1) < 1.0 - 1e-12) & SS.is_absorber[domi]
        if snap.any():
            C[snap] = np.eye(SS.ns)[domi[snap]]
        info['absorb_pairs'] = (dom_p, low_p)
    # ---- REACT: a + b -> c inside the stated window
    sel = np.flatnonzero(out == REACT)
    if len(sel):
        used = np.zeros(n, bool)
        for s in sel:
            i, j = I[s], J[s]
            if used[i] or used[j]:
                continue
            r = SS.react.get((da[s], db[s]))
            if r is None:
                continue
            Tm = 0.5 * (T[i] + T[j])
            rho_i, rho_j = SS.dens[da[s]], SS.dens[db[s]]
            ratio = max(rho_i, rho_j) / max(min(rho_i, rho_j), 1e-9)
            if not (r['tmin'] <= Tm <= r['tmax'] and vrel[s] >= r['vmin'] and r['rmin'] <= ratio <= r['rmax'] and min(C[i, da[s]], C[j, db[s]]) >= r['cmin']):
                continue
            c = SS.idx[r['product']]
            for p in (i, j):
                C[p] = 0.0
                C[p, c] = 1.0
                T[p] += r['q'] / max(cp[p], 1e-9)       # J/kg released into each particle
            used[i] = used[j] = True
            info['reactions'] += 1
    SS.stats['merges'] += info['merges']
    SS.stats['reactions'] += info['reactions']
    return info


def propagate_clusters(cluster, dom, I, J, absorb_pairs, rounds=3):
    """Cluster ids: absorbed particles take their absorber's id; ids then propagate (minimum) through neighbouring particles of the same dominant species."""
    ap, lp = absorb_pairs
    if len(ap):
        cluster[lp] = np.minimum(cluster[lp], cluster[ap])
    same = dom[I] == dom[J]
    Is, Js = I[same], J[same]
    for _ in range(rounds):
        mn = np.minimum(cluster[Is], cluster[Js])
        np.minimum.at(cluster, Is, mn)
        np.minimum.at(cluster, Js, mn)
    return cluster
