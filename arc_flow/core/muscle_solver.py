"""Tessera Projection muscle solver (pure numpy, no bpy).

Given a skin mesh (vertices + edges + normals), per-vertex region labels (painted colours or auto-seeded from the
skeleton), optional contour strokes (ordered vertex chains) and optional bone attachment points, the solver

  1. diffuses the painted regions over the whole skin (screened-Poisson / harmonic region projection) into a smooth,
     balanced partition of unity (soft membership per region),
  2. splits every region into connected muscle instances (left / right biceps, one cheek per head, ...),
  3. finds each instance's fibre axis (a contour stroke, else bone attachments, else the principal axis) and
     tendon ends (zero flex at the two ends, belly in between),
  4. builds per-vertex flex weights and the unit-flex displacement fields (bulge along the skin normal plus accent
     slide along the fibre axis toward the joint end),
  5. provides runtime helpers: joint-angle / speed driven flex (rest / walk / run gains) and a jiggle spring.

Everything here is deterministic and offline (one solve = tens to a few hundred ms for a 5-20k vertex skin).  What runs
in real time afterwards is evaluation of the baked fields (shape keys, Geometry Nodes, engine sidecar)."""
import colorsys
import json
import math

import numpy as np

# --------------------------------------------------------------------------------------------------------------------
# Region catalog.  Colours are static: they are derived from the catalog position, so a "cheek" is the same colour on every
# character, every preset and biped as well as quadruped.  Append new regions at the END only (keeps colours stable).
# fields: key, label, group, tier (1 cartoon, 2 standard, 3 cinematic), bodies ('B' biped, 'Q' quadruped, 'BQ'),
#         bone keywords, t range along the bone (0 head .. 1 tail), facing, default emphasis, rest tone, range (rad)
# --------------------------------------------------------------------------------------------------------------------
GROUPS = ("Head and face", "Neck", "Shoulders", "Arms", "Forearms", "Hands", "Chest", "Back", "Abdomen",
          "Hips and glutes", "Thighs", "Calves", "Feet", "Tail", "Custom")

_R = []


def _r(key, label, group, tier, bodies, bones, tr=(0.0, 1.0), facing='any', emph=1.0, tone=0.06, rng=1.2, sub=1):
    _R.append(dict(key=key, label=label, group=group, tier=tier, bodies=bodies, bones=tuple(bones),
                   t=tr, facing=facing, emphasis=emph, tone=tone, range=rng, sub=sub))


# head and face (bone: head)
_r("frontalis", "Forehead (frontalis)", "Head and face", 3, "BQ", ("head",), (0.78, 1.0), 'front', 0.6)
_r("orbicularis_oculi", "Eye ring (orbicularis oculi)", "Head and face", 3, "BQ", ("head",), (0.60, 0.78), 'front', 0.5)
_r("nasalis", "Nose (nasalis)", "Head and face", 3, "BQ", ("head",), (0.44, 0.60), 'front', 0.5)
_r("zygomaticus", "Cheek (zygomaticus)", "Head and face", 2, "BQ", ("head",), (0.30, 0.60), 'out', 0.8)
_r("orbicularis_oris", "Lips (orbicularis oris)", "Head and face", 3, "BQ", ("head",), (0.26, 0.40), 'front', 0.7)
_r("mentalis", "Chin (mentalis)", "Head and face", 3, "BQ", ("head",), (0.08, 0.26), 'front', 0.5)
_r("masseter", "Jaw (masseter)", "Head and face", 2, "BQ", ("head",), (0.10, 0.45), 'out', 0.8)
# neck
_r("sternocleidomastoid", "Neck (sternocleidomastoid)", "Neck", 1, "BQ", ("neck",), (0.0, 1.0), 'any', 1.0, rng=0.9)
_r("trapezius", "Trapezius", "Shoulders", 2, "BQ", ("neck", "shoulder", "clavicle"), (0.0, 0.5), 'back', 0.9)
_r("deltoid", "Deltoid", "Shoulders", 1, "BQ", ("upper_arm", "shoulder", "arm_up", "foreleg_up", "humerus"), (0.0, 0.35), 'out', 1.0)
# arms
_r("biceps", "Biceps", "Arms", 1, "B", ("upper_arm", "arm_up"), (0.25, 0.85), 'front', 1.0, rng=1.6)
_r("triceps", "Triceps", "Arms", 1, "B", ("upper_arm", "arm_up"), (0.25, 0.85), 'back', 1.0, rng=1.6)
_r("brachialis", "Brachialis", "Arms", 3, "B", ("upper_arm", "arm_up"), (0.5, 0.95), 'out', 0.6, rng=1.6)
# forearms
_r("forearm_flexors", "Forearm flexors", "Forearms", 2, "B", ("forearm", "arm_low"), (0.1, 0.8), 'in', 0.9, rng=1.2)
_r("forearm_extensors", "Forearm extensors", "Forearms", 2, "B", ("forearm", "arm_low"), (0.1, 0.8), 'out', 0.9, rng=1.2)
_r("brachioradialis", "Brachioradialis", "Forearms", 3, "B", ("forearm", "arm_low"), (0.0, 0.45), 'front', 0.6, rng=1.2)
# hands
_r("thenar", "Thumb pad (thenar)", "Hands", 3, "BQ", ("hand",), (0.0, 0.5), 'in', 0.6, rng=0.9)
_r("palm", "Palm and fingers", "Hands", 3, "BQ", ("hand", "finger"), (0.3, 1.0), 'any', 0.5, rng=0.9)
# chest, back, abdomen
_r("pectoralis", "Chest (pectoralis)", "Chest", 1, "B", ("chest", "spine"), (0.2, 1.0), 'front', 1.0, rng=0.9)
_r("serratus", "Ribs (serratus)", "Chest", 3, "BQ", ("chest", "spine"), (0.1, 0.7), 'out', 0.6, rng=0.9)
_r("latissimus", "Lats (latissimus)", "Back", 2, "BQ", ("chest", "spine"), (0.0, 0.6), 'back', 0.9, rng=0.9)
_r("erector_spinae", "Back (erector spinae)", "Back", 2, "BQ", ("spine", "chest", "pelvis", "hips"), (0.0, 1.0), 'top', 0.8, rng=0.8)
_r("rectus_abdominis", "Abs (rectus abdominis)", "Abdomen", 2, "B", ("spine", "pelvis", "hips"), (0.3, 1.0), 'front', 0.9, rng=0.8)
_r("obliques", "Obliques", "Abdomen", 3, "BQ", ("spine", "pelvis", "hips"), (0.3, 1.0), 'out', 0.6, rng=0.8)
# legs
_r("gluteus", "Glutes (gluteus)", "Hips and glutes", 2, "B", ("pelvis", "hips", "thigh", "leg_up"), (0.0, 0.3), 'back', 1.0, rng=1.0)
_r("quadriceps", "Quads (quadriceps)", "Thighs", 1, "B", ("thigh", "leg_up"), (0.2, 0.9), 'front', 1.0, rng=1.6)
_r("hamstrings", "Hamstrings", "Thighs", 1, "B", ("thigh", "leg_up"), (0.2, 0.9), 'back', 1.0, rng=1.6)
_r("adductors", "Inner thigh (adductors)", "Thighs", 3, "BQ", ("thigh", "leg_up"), (0.1, 0.7), 'in', 0.6, rng=1.4)
_r("gastrocnemius", "Calf (gastrocnemius)", "Calves", 1, "B", ("shin", "calf", "leg_low"), (0.0, 0.6), 'back', 1.0, rng=1.4)
_r("tibialis", "Shin (tibialis)", "Calves", 3, "B", ("shin", "calf", "leg_low"), (0.1, 0.8), 'front', 0.6, rng=1.4)
# feet
_r("foot_arch", "Foot arch", "Feet", 3, "BQ", ("foot",), (0.0, 0.7), 'bottom', 0.5, rng=0.8)
_r("toes", "Toes", "Feet", 3, "BQ", ("toe", "foot"), (0.6, 1.0), 'any', 0.5, rng=0.8)
# quadruped specific (appended after the biped set; colours stay stable)
_r("haunch", "Haunch", "Thighs", 1, "Q", ("thigh", "leg_up", "hindleg", "hind_up"), (0.0, 0.7), 'out', 1.0, rng=1.4)
_r("foreleg_shoulder", "Fore shoulder", "Shoulders", 1, "Q", ("foreleg_up", "shoulder", "scapula", "arm_up"), (0.0, 0.6), 'out', 1.0, rng=1.2)
_r("tail_base", "Tail", "Tail", 2, "BQ", ("tail",), (0.0, 1.0), 'any', 0.8, rng=1.0)
_r("lower_leg_q", "Lower leg", "Calves", 2, "Q", ("shin", "calf", "leg_low", "forearm", "arm_low"), (0.0, 0.7), 'back', 0.8, rng=1.3)
_r("limb_mid", "Limb belly (custom creatures)", "Custom", 2, "BQ", ("limb", "seg", "tentacle"), (0.1, 0.9), 'any', 0.8, rng=1.3)
_r("head_mass", "Head mass (extra heads)", "Head and face", 2, "BQ", ("head",), (0.1, 0.9), 'any', 0.5, rng=0.8)

# bilateral muscles: a patch that crosses the skin's mid-plane is split into a left and a right muscle
PAIRED = {"zygomaticus", "masseter", "orbicularis_oculi", "deltoid", "trapezius", "pectoralis", "serratus", "latissimus",
          "obliques", "gluteus", "quadriceps", "hamstrings", "adductors", "gastrocnemius", "tibialis", "haunch",
          "foreleg_shoulder", "lower_leg_q", "biceps", "triceps", "brachialis", "forearm_flexors", "forearm_extensors",
          "brachioradialis", "thenar", "palm", "foot_arch", "toes", "erector_spinae", "sternocleidomastoid"}
for _d in _R:
    _d['pair'] = _d['key'] in PAIRED
CATALOG = tuple(_R)
CATALOG_INDEX = {r['key']: i for i, r in enumerate(CATALOG)}
CID_CUSTOM_BASE = 1000


def catalog_cid(key):
    """Stable integer id stored in the skin's `tp_muscle_id` attribute (0 = unpainted)."""
    return CATALOG_INDEX[key] + 1


def region_color(key_or_index):
    """Static region colour (r, g, b).  Golden-angle hue stepping on the catalog index: stable for ever."""
    i = CATALOG_INDEX[key_or_index] if isinstance(key_or_index, str) else int(key_or_index)
    h = (i * 0.61803398875 + 0.07) % 1.0
    s = (0.78, 0.55, 0.92)[i % 3]
    v = (0.95, 0.88, 0.78)[(i // 3) % 3]
    return colorsys.hsv_to_rgb(h, s, v)


DETAIL_ITEMS = (('CARTOON', "Cartoon", "A few big muscles: biceps, triceps, neck, chest, thighs, calves", 1),
                ('STANDARD', "Standard", "Common game muscle set", 2),
                ('CINEMATIC', "Cinematic", "Full set including facial, finger and foot muscles", 3))
DETAIL_TIER = {d[0]: d[3] for d in DETAIL_ITEMS}

# preset id -> dict(body, gender, label, per-region emphasis multipliers)
PRESETS = {
    'BIPED': dict(label="Biped (generic)", body='B', mult={}),
    'HUMAN_MALE': dict(label="Human male", body='B',
                       mult=dict(pectoralis=1.25, deltoid=1.25, biceps=1.2, triceps=1.2, trapezius=1.2, latissimus=1.2,
                                 rectus_abdominis=1.1, zygomaticus=0.9, gluteus=0.9)),
    'HUMAN_FEMALE': dict(label="Human female", body='B',
                         mult=dict(pectoralis=0.6, deltoid=0.8, biceps=0.75, triceps=0.75, trapezius=0.8, latissimus=0.75,
                                   gluteus=1.2, quadriceps=1.1, adductors=1.2, gastrocnemius=1.0, zygomaticus=1.1,
                                   orbicularis_oris=1.15)),
    'QUADRUPED': dict(label="Quadruped (generic)", body='Q', mult={}),
    'FELINE': dict(label="Quadruped: feline", body='Q', mult=dict(haunch=1.2, foreleg_shoulder=1.2, tail_base=1.2,
                                                                  erector_spinae=1.15)),
    'CUSTOM': dict(label="Custom creature", body='B', mult={}),
}
PRESET_ITEMS = tuple((k, v['label'], "") for k, v in PRESETS.items())

CREATURE_DEFAULT = dict(heads=1, arms=2, legs=2, tails=0, fingers=0, toes=0, spine_segments=3, joints_per_limb=2)

# Region category CARDS (the stacked cards in the Muscle tab).  Every catalog group belongs to exactly one card; a card
# lists the muscles of its body region with their static colours and has its own Emphasis / Flex / Accent sliders that
# multiply the per-muscle values.  fields: key, label, icon, groups
CARDS = (
    ('HEAD', "Head and face", 'MONKEY', ("Head and face", "Neck")),
    ('ARM', "Arms and hands", 'BONE_DATA', ("Shoulders", "Arms", "Forearms", "Hands")),
    ('TORSO', "Torso and back", 'ARMATURE_DATA', ("Chest", "Back", "Abdomen")),
    ('LEG', "Legs and feet", 'CON_KINEMATIC', ("Hips and glutes", "Thighs", "Calves", "Feet")),
    ('TAIL', "Tail and custom", 'OUTLINER_OB_ARMATURE', ("Tail", "Custom")),
)
CARD_OF_GROUP = {g: c[0] for c in CARDS for g in c[3]}
CARD_LABEL = {c[0]: c[1] for c in CARDS}
RESOLUTION_MIN, RESOLUTION_MAX = 0.85, 10.0


def card_of(group):
    """Card key for a catalog group (unknown / user groups land on the custom card)."""
    return CARD_OF_GROUP.get(group, 'TAIL')


def card_regions(regions, card):
    """Subset of region dicts (or objects with .group) on one card, catalog order kept."""
    out = []
    for r in regions:
        g = r['group'] if isinstance(r, dict) else getattr(r, 'group', '')
        if card_of(g) == card:
            out.append(r)
    return out


def effective_params(region, card=None):
    """Per-muscle flex / accent / emphasis after the card multipliers (card = dict(flex, accent, emphasis) or None)."""
    c = card or {}
    return dict(flex=float(region['flex']) * float(c.get('flex', 1.0)),
                accent=float(region['accent']) * float(c.get('accent', 1.0)),
                emphasis=float(region['emphasis']) * float(c.get('emphasis', 1.0)),
                enabled=bool(region.get('enabled', True)))


_ARM_GROUPS = {"Arms", "Forearms"}
_HAND_KEYS = {"thenar", "palm"}
_LEG_GROUPS = {"Thighs", "Calves", "Hips and glutes"}


def regions_for(preset='BIPED', detail='STANDARD', creature=None):
    """Region list (dicts, emphasis already multiplied by the preset) for a preset, detail level and creature spec."""
    cr = dict(CREATURE_DEFAULT)
    cr.update(creature or {})
    p = PRESETS[preset]
    body = p['body']
    tier = DETAIL_TIER[detail]
    custom = preset == 'CUSTOM'
    out = []
    for r in CATALOG:
        if body not in r['bodies'] and not custom:
            continue
        if r['tier'] > tier and r['key'] != 'limb_mid':
            continue
        k, g = r['key'], r['group']
        if cr['heads'] <= 0 and g == "Head and face":
            continue
        if k == 'head_mass' and cr['heads'] < 2:
            continue
        if cr['arms'] <= 0 and (g in _ARM_GROUPS or k in ("deltoid", "foreleg_shoulder")):
            continue
        if cr['arms'] <= 0 and k in _HAND_KEYS:
            continue
        if cr['legs'] <= 0 and (g in _LEG_GROUPS or k in ("foot_arch", "toes", "haunch", "lower_leg_q")):
            continue
        if cr['tails'] <= 0 and g == "Tail":
            continue
        if k == 'toes' and cr['toes'] <= 0 and cr['legs'] > 0 and custom:
            continue
        if k == 'palm' and cr['fingers'] <= 0 and custom:
            continue
        if k == 'limb_mid' and not (custom and (cr['joints_per_limb'] > 2 or cr['arms'] + cr['legs'] > 4)):
            continue
        if body == 'Q' and not custom and g == "Forearms":
            continue
        d = dict(r)
        d['cid'] = catalog_cid(k)
        d['color'] = region_color(k)
        d['emphasis'] = float(r['emphasis']) * float(p['mult'].get(k, 1.0))
        d['flex'] = 1.0
        d['accent'] = 0.6
        out.append(d)
    return out


# --------------------------------------------------------------------------------------------------------------------
# graph helpers
# --------------------------------------------------------------------------------------------------------------------
class Graph:
    """Vertex adjacency with 1/length edge weights, stored for fast vectorised neighbour sums (np.add.reduceat)."""

    def __init__(self, verts, edges):
        verts = np.asarray(verts, np.float64)
        e = np.asarray(edges, np.int64).reshape(-1, 2)
        n = len(verts)
        self.n = n
        if len(e):
            e = np.unique(np.sort(e, axis=1), axis=0)
        self.edges = e
        src = np.concatenate([e[:, 0], e[:, 1]])
        dst = np.concatenate([e[:, 1], e[:, 0]])
        ln = np.linalg.norm(verts[src] - verts[dst], axis=1)
        self.mean_len = float(ln.mean()) if len(ln) else 1.0
        w = self.mean_len / np.maximum(ln, 1e-9 * self.mean_len + 1e-12)
        o = np.argsort(dst, kind='stable')
        self.src, self.dst, self.w = src[o], dst[o], w[o]
        self.deg = np.bincount(self.dst, minlength=n)
        self.wsum = np.bincount(self.dst, weights=self.w, minlength=n)
        nz = np.nonzero(self.deg)[0]
        self.nz = nz
        starts = np.concatenate([[0], np.cumsum(self.deg[nz])[:-1]]) if len(nz) else np.zeros(0, np.int64)
        self.starts = starts

    def nbr_sum(self, u, weighted=True):
        """sum over neighbours of (w *) u for each vertex; u is (n,) or (n,k)."""
        out = np.zeros_like(u, dtype=np.float64)
        if not len(self.nz):
            return out
        vals = u[self.src]
        if weighted:
            vals = vals * (self.w[:, None] if u.ndim == 2 else self.w)
        out[self.nz] = np.add.reduceat(vals, self.starts, axis=0)
        return out

    def smooth(self, u, alpha=0.5, iters=1):
        u = np.asarray(u, np.float64)
        d = np.maximum(self.wsum, 1e-12)
        d = d[:, None] if u.ndim == 2 else d
        for _ in range(int(iters)):
            u = (1 - alpha) * u + alpha * np.where(d > 1e-11, self.nbr_sum(u) / d, u)
        return u

    def components(self, mask):
        """Connected components of the sub-graph induced by `mask`. Returns (n,) ints (-1 outside the mask)."""
        mask = np.asarray(mask, bool)
        lab = np.where(mask, np.arange(self.n), self.n + 1).astype(np.int64)
        e = self.edges
        if not len(e):
            return np.where(mask, lab, -1)
        ok = mask[e[:, 0]] & mask[e[:, 1]]
        a, b = e[ok, 0], e[ok, 1]
        for _ in range(self.n + 2):
            m = np.minimum(lab[a], lab[b])
            new = lab.copy()
            np.minimum.at(new, a, m)
            np.minimum.at(new, b, m)
            # pointer jumping speeds up long chains
            new = np.where(mask, new[np.minimum(new, self.n - 1)], new)
            if np.array_equal(new, lab):
                break
            lab = new
        out = -np.ones(self.n, np.int64)
        if mask.any():
            _, inv = np.unique(lab[mask], return_inverse=True)
            out[mask] = inv
        return out


def vertex_normals_from_faces(verts, faces):
    """Area-weighted vertex normals for triangles / quads given as lists of index tuples."""
    v = np.asarray(verts, np.float64)
    n = np.zeros_like(v)
    for f in faces:
        f = list(f)
        for i in range(1, len(f) - 1):
            a, b, c = v[f[0]], v[f[i]], v[f[i + 1]]
            fn = np.cross(b - a, c - a)
            n[f[0]] += fn
            n[f[i]] += fn
            n[f[i + 1]] += fn
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(ln, 1e-12)


# --------------------------------------------------------------------------------------------------------------------
# skeleton-driven auto seeding (rough starting layout the artist then repaints / refines)
# --------------------------------------------------------------------------------------------------------------------
_FACE = {'front': (0, -1, 0), 'back': (0, 1, 0), 'top': (0, 0, 1), 'bottom': (0, 0, -1)}


def auto_seed(verts, normals, bone_names, heads, tails, weights, regions, min_dot=0.25, dominance=0.45):
    """Label vertices with a region index from the skeleton.  weights is (n, nbones) skin weights.
    A region claims a vertex when the vertex is dominated by a bone whose name contains one of the region's keywords,
    when its projection on that bone lies inside the region's t range and when the radial direction matches the facing.
    Returns (labels (n,) region index or -1, score (n,))."""
    v = np.asarray(verts, np.float64)
    n = len(v)
    W = np.asarray(weights, np.float64)
    names = [str(b).lower() for b in bone_names]
    H = np.asarray(heads, np.float64)
    T = np.asarray(tails, np.float64)
    seg = T - H
    seglen2 = np.maximum((seg ** 2).sum(1), 1e-12)
    labels = -np.ones(n, np.int64)
    best = np.full(n, -1.0)
    wsum = np.maximum(W.sum(1), 1e-12)
    for ri, r in enumerate(regions):
        bi = [i for i, nm in enumerate(names) if any(k in nm for k in r['bones'])]
        if not bi:
            continue
        wr = W[:, bi].sum(1) / wsum
        ok = wr >= dominance
        if not ok.any():
            continue
        # nearest of the matching bones, per vertex
        best_bone = np.zeros(n, np.int64)
        best_d = np.full(n, 1e30)
        t_at = np.zeros(n)
        rad = np.zeros((n, 3))
        for b in bi:
            tt = ((v - H[b]) @ seg[b]) / seglen2[b]
            tc = np.clip(tt, 0, 1)
            pt = H[b] + tc[:, None] * seg[b]
            d = np.linalg.norm(v - pt, axis=1)
            m = d < best_d
            best_d = np.where(m, d, best_d)
            best_bone = np.where(m, b, best_bone)
            t_at = np.where(m, tt, t_at)
            rad = np.where(m[:, None], v - pt, rad)
        t0, t1 = r['t']
        ok &= (t_at >= t0) & (t_at <= t1)
        rl = np.linalg.norm(rad, axis=1)
        rd = rad / np.maximum(rl, 1e-12)[:, None]
        f = r['facing']
        if f == 'any':
            dot = np.ones(n) * 0.5
        elif f in _FACE:
            dot = rd @ np.array(_FACE[f], float)
        else:
            sx = np.sign(v[:, 0])
            sx = np.where(sx == 0, 1.0, sx)
            dot = rd[:, 0] * sx * (1.0 if f == 'out' else -1.0)
        ok &= dot >= (min_dot if f != 'any' else 0.0)
        score = dot * (0.5 + 0.5 * wr)
        take = ok & (score > best)
        labels[take] = ri
        best[take] = score[take]
    return labels, best


# --------------------------------------------------------------------------------------------------------------------
# region projection
# --------------------------------------------------------------------------------------------------------------------
def project_regions(graph, seed_labels, n_regions, res=1.0, leak=0.25, iters=None):
    """Harmonic region membership with a screened leak.  seed_labels (n,) in -1..R-1.  Returns u (n,R) with u=1 on the
    region's own seeds and 0 on other regions' seeds; the leak makes influence fade over several edges so unpainted
    areas stay unassigned.  Higher `res` -> lower leak (wider reach per iteration budget is not needed: sharper)."""
    n = graph.n
    R = max(1, n_regions)
    lab = np.asarray(seed_labels, np.int64)
    seeded = lab >= 0
    fixed = np.zeros((n, R))
    fixed[np.nonzero(seeded)[0], lab[seeded]] = 1.0
    u = fixed.copy()
    lk = float(leak) / max(float(res), 0.05)
    den = graph.wsum + lk * 1.0
    den = np.where(den > 1e-12, den, 1.0)[:, None]
    if iters is None:
        iters = int(min(400, 60 + 60 * math.sqrt(max(res, 0.1))))
    free = ~seeded
    for it in range(int(iters)):
        nu = graph.nbr_sum(u) / den
        nu[~free] = fixed[~free]
        if it % 8 == 7 and np.max(np.abs(nu - u)) < 1e-5:
            u = nu
            break
        u = nu
    return u


class Instance:
    """One connected muscle patch of a region (e.g. the left biceps)."""
    __slots__ = ("key", "region", "side", "idx", "w", "bulge", "slide", "a", "b", "axis", "bone", "accent_end",
                 "half_width", "length", "source", "name", "t")

    def to_dict(self, prec=5):
        return dict(name=self.name, region=self.key, side=self.side, bone=self.bone, accent_end=self.accent_end,
                    axis=[np.round(self.a, 5).tolist(), np.round(self.b, 5).tolist()], source=self.source,
                    half_width=round(float(self.half_width), 6), length=round(float(self.length), 6),
                    idx=self.idx.tolist(), w=np.round(self.w, prec).tolist(),
                    bulge=np.round(self.bulge, prec + 3).tolist(), slide=np.round(self.slide, prec + 3).tolist())

    @staticmethod
    def from_dict(d):
        i = Instance()
        i.name, i.key, i.side, i.bone = d['name'], d['region'], d['side'], d['bone']
        i.region = d['region']
        i.accent_end = d['accent_end']
        i.a, i.b = np.array(d['axis'][0]), np.array(d['axis'][1])
        i.axis = (i.b - i.a) / max(np.linalg.norm(i.b - i.a), 1e-12)
        i.source = d['source']
        i.half_width, i.length = d['half_width'], d['length']
        i.idx = np.array(d['idx'], np.int64)
        i.w = np.array(d['w'])
        i.bulge = np.array(d['bulge']).reshape(-1, 3)
        i.slide = np.array(d['slide']).reshape(-1, 3)
        i.t = np.zeros(len(i.idx))
        return i


def _pca_axis(P):
    c = P.mean(0)
    X = P - c
    if len(P) < 3:
        return c, np.array([0, 0, 1.0])
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    a = vt[0]
    return c, a / max(np.linalg.norm(a), 1e-12)


def _polyline_param(P, line):
    """For points P (m,3) and polyline (k,3): t in 0..1 (arc-length fraction of the nearest point) and distance."""
    line = np.asarray(line, float)
    seg = line[1:] - line[:-1]
    sl = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0], np.cumsum(sl)])
    total = max(cum[-1], 1e-12)
    best_d = np.full(len(P), 1e30)
    best_t = np.zeros(len(P))
    for k in range(len(seg)):
        s2 = max(sl[k] ** 2, 1e-18)
        tt = np.clip(((P - line[k]) @ seg[k]) / s2, 0, 1)
        q = line[k] + tt[:, None] * seg[k]
        d = np.linalg.norm(P - q, axis=1)
        m = d < best_d
        best_d = np.where(m, d, best_d)
        best_t = np.where(m, (cum[k] + tt * sl[k]) / total, best_t)
    return best_t, best_d


def _side(c, tol):
    if c[0] > tol:
        return "L"
    if c[0] < -tol:
        return "R"
    return "C"


def solve(verts, edges, normals, seed_labels, regions, strokes=None, attachments=None, bone_info=None, res=1.0,
          bulge_frac=0.22, min_patch=6, membership_threshold=0.35, background=0.12):
    """Run the muscle projection.

    verts (n,3), edges (E,2), normals (n,3); seed_labels (n,) region index (into `regions`) or -1;
    regions: list of region dicts (needs 'key');  strokes: list of {'region': ri, 'verts': [vertex ids in order]};
    attachments: {ri: [points (3,), ...]} bone / bone-geometry attachment points that fix the tendon ends;
    bone_info: dict(names, heads, tails, weights (n,nb)) used to pick the driving bone and the accent end.
    Returns dict(instances=[Instance], membership (n,R), label (n,), total_w (n,), graph)."""
    v = np.asarray(verts, np.float64)
    nrm = np.asarray(normals, np.float64)
    g = Graph(v, edges)
    R = len(regions)
    lab = np.asarray(seed_labels, np.int64).copy()
    strokes = strokes or []
    for s in strokes:
        ids = np.asarray(s['verts'], np.int64)
        ids = ids[(ids >= 0) & (ids < g.n)]
        lab[ids] = s['region']
    res = float(max(res, 0.05))
    if R == 0 or not (lab >= 0).any():
        return dict(instances=[], membership=np.zeros((g.n, max(R, 1))), label=-np.ones(g.n, np.int64),
                    total_w=np.zeros(g.n), graph=g)
    u = project_regions(g, lab, R, res=res)
    # partition of unity with a background competitor so unpainted skin keeps membership 0
    m = u / (u.sum(1, keepdims=True) + background)
    # smoothing of the memberships: coarse resolution blends more across neighbours
    sm = int(round(2 + 8 * max(0.0, 1.0 - res) - 1.5 * max(0.0, min(res - 1.0, 1.0))))
    if sm > 0:
        m = g.smooth(m, 0.5, sm)
    label = np.where(m.max(1) >= membership_threshold, m.argmax(1), -1)
    insts = []
    stroke_by_region = {}
    for s in strokes:
        stroke_by_region.setdefault(int(s['region']), []).append(np.asarray(s['verts'], np.int64))
    total = np.zeros(g.n)
    bi = bone_info or {}
    for ri, reg in enumerate(regions):
        mask = label == ri
        if mask.sum() < min_patch:
            continue
        comp = g.components(mask)
        if reg.get('pair') and comp.max() >= 0:
            xm = float(np.median(v[:, 0]))
            ext = max(float(np.ptp(v[:, 0])), 1e-9)
            nxt = int(comp.max()) + 1
            for ci in range(int(comp.max()) + 1):
                sel = np.nonzero(comp == ci)[0]
                xs = v[sel, 0] - xm
                left, right = xs > 0.01 * ext, xs < -0.01 * ext
                if left.sum() >= min_patch and right.sum() >= min_patch:
                    comp[sel[right]] = nxt
                    nxt += 1
        sizes = np.bincount(comp[comp >= 0]) if comp.max() >= 0 else np.zeros(0)
        floor = max(min_patch, 0.2 * float(sizes.max() if len(sizes) else 0))
        for ci in range(int(comp.max()) + 1):
            idx = np.nonzero(comp == ci)[0]
            if len(idx) < floor:
                continue
            P = v[idx]
            c, a = _pca_axis(P)
            source = 'pca'
            line = None
            ends = None
            # contour stroke inside this patch (majority of stroke vertices in patch)
            inset = set(idx.tolist())
            for st in stroke_by_region.get(ri, []):
                if len(st) >= 2 and np.mean([x in inset for x in st]) >= 0.5:
                    pts = v[st]
                    line = pts if line is None else line
                    source = 'stroke'
                    break
            att = (attachments or {}).get(ri)
            if line is None and att and len(att) >= 2:
                A = np.asarray(att, float)
                reach = 0.9 * float(np.linalg.norm(P.max(0) - P.min(0))) + 1e-9
                A = A[np.linalg.norm(A - c, axis=1) <= reach]
            if line is None and att and len(att) >= 2 and len(A) >= 2:
                proj = (A - c) @ a
                line = np.stack([A[np.argmin(proj)], A[np.argmax(proj)]])
                source = 'attach'
            if line is None:
                proj = (P - c) @ a
                lo, hi = np.percentile(proj, 2), np.percentile(proj, 98)
                line = np.stack([c + a * lo, c + a * hi])
            t, d = _polyline_param(P, line)
            if source == 'pca':
                pass
            elif source == 'stroke':
                # strokes cover the belly; extend a little so the tendon ends lie just beyond the stroke ends
                pass
            hw = float(max(np.percentile(d, 90), 1e-6))
            a_pt, b_pt = line[0], line[-1]
            length = float(np.linalg.norm(b_pt - a_pt))
            ax = (b_pt - a_pt) / max(length, 1e-12)
            # belly profile: zero at both tendon ends, one in the middle
            tt = np.clip(t, 0.0, 1.0)
            prof = np.sin(np.pi * tt) ** 0.8
            # lateral falloff
            lat = np.exp(-(d / (1.15 * hw)) ** 2)
            # resolution > 1 splits the belly into fibre bundles (striation); < 1 stays smooth
            nb = max(1, int(round(res * reg.get('sub', 1))))
            if nb > 1 and res > 1.0:
                side_vec = np.cross(ax, np.array([0.0, 0.0, 1.0]))
                if np.linalg.norm(side_vec) < 1e-6:
                    side_vec = np.cross(ax, np.array([1.0, 0.0, 0.0]))
                side_vec /= np.linalg.norm(side_vec)
                s_coord = (P - c) @ side_vec / max(2 * hw, 1e-9)
                depth = min(1.0, 0.25 * (res - 1.0) + 0.1)
                # mean of the striation factor is (1 - depth / 2): divide it out so a finer resolution adds detail
                # without making the muscle weaker overall (strength stays the artist's Flex / Emphasis choice)
                prof = prof * (1.0 - depth + depth * 0.5 * (1.0 + np.cos(2 * np.pi * nb * s_coord))) / (1.0 - 0.5 * depth)
            w_full = np.zeros(g.n)
            w_full[idx] = prof * lat * m[idx, ri]
            # accent end: the end nearest the driving bone's head (the joint it crosses)
            bone, acc_end = None, 'B'
            if bi and len(bi.get('names', [])):
                W = np.asarray(bi['weights'])[idx]
                names = [str(x).lower() for x in bi['names']]
                cand = [i for i, nm in enumerate(names) if any(k in nm for k in reg['bones'])]
                if not cand:
                    cand = list(range(len(names)))
                sc = W[:, cand].sum(0)
                bsel = cand[int(np.argmax(sc))]
                bone = bi['names'][bsel]
                H = np.asarray(bi['heads'])[bsel]
                T = np.asarray(bi['tails'])[bsel]
                da = min(np.linalg.norm(a_pt - H), np.linalg.norm(a_pt - T))
                db = min(np.linalg.norm(b_pt - H), np.linalg.norm(b_pt - T))
                # the joint end is the one closer to a bone end (head / tail) of the driving bone
                acc_end = 'A' if da < db else 'B'
            inst = Instance()
            inst.key = inst.region = reg['key']
            inst.side = _side(P.mean(0), 0.02 * max(np.ptp(v[:, 0]), 1e-9))
            inst.idx = idx
            inst.t = tt
            inst.a, inst.b, inst.axis = a_pt, b_pt, ax
            inst.source = source
            inst.half_width, inst.length = hw, length
            inst.bone = bone
            inst.accent_end = acc_end
            inst.w = w_full[idx]
            insts.append(inst)
    # unique names + per-side counters; resolve overlaps by capping the summed weight at 1
    counter = {}
    for inst in insts:
        k = (inst.key, inst.side)
        counter[k] = counter.get(k, 0) + 1
        suffix = "" if inst.side == "C" and counter[k] == 1 else ("." + inst.side)
        inst.name = inst.key + suffix + ("" if counter[k] == 1 else ".%03d" % (counter[k] - 1))
        total[inst.idx] += inst.w
    over = total > 1.0
    for inst in insts:
        if over[inst.idx].any():
            inst.w = inst.w / np.maximum(total[inst.idx], 1.0)
    # extra even-balancing: smooth each instance's weights over the whole skin graph (region borders blend)
    sm_w = int(round(1 + 3 * max(0.0, 1.0 - res)))
    total = np.zeros(g.n)
    for inst in insts:
        full = np.zeros(g.n)
        full[inst.idx] = inst.w
        full = g.smooth(full, 0.5, sm_w)
        inst.w = full[inst.idx]
        # displacement fields (unit flex, unit accent, unit emphasis)
        amp = bulge_frac * inst.half_width
        amp = min(amp, 0.35 * inst.half_width)
        inst.bulge = nrm[inst.idx] * (amp * inst.w)[:, None]
        toward = (inst.a - inst.b) if inst.accent_end == 'A' else (inst.b - inst.a)
        toward = toward / max(np.linalg.norm(toward), 1e-12)
        # skin slides toward the joint end, strongest where the belly meets the tendon (sin(2 pi t)-like skew)
        tt = inst.t
        skew = np.sin(np.pi * np.clip(tt if inst.accent_end == 'B' else 1 - tt, 0, 1) * 0.999)
        inst.slide = toward[None, :] * (0.6 * amp * inst.w * (0.5 + 0.5 * skew))[:, None]
        total[inst.idx] += inst.w
    return dict(instances=insts, membership=m, label=label, total_w=np.minimum(total, 1.0), graph=g)


# --------------------------------------------------------------------------------------------------------------------
# displacement assembly + runtime helpers
# --------------------------------------------------------------------------------------------------------------------
def assemble(n, instances, region_params, flex_by_instance=None, accent_scale=1.0):
    """Sum instance fields into a displacement (n,3).  region_params[key] = dict(flex, accent, emphasis, enabled).
    flex_by_instance: {name: 0..1.5 flex value} (default 1)."""
    out = np.zeros((n, 3))
    for inst in instances:
        p = region_params.get(inst.key)
        if p is None or not p.get('enabled', True):
            continue
        f = 1.0 if flex_by_instance is None else float(flex_by_instance.get(inst.name, 1.0))
        k = float(p.get('flex', 1.0)) * float(p.get('emphasis', 1.0)) * f
        out[inst.idx] += k * inst.bulge + k * float(p.get('accent', 1.0)) * accent_scale * inst.slide
    return out


STATE_SPEEDS = (0.0, 1.2, 4.0)   # rad/s of the joint: rest, walk, run


def flex_drive(angle, speed, tone=0.06, rng=1.2, gains=(0.6, 1.0, 1.5)):
    """Flex (0..1.5) from a joint angle (rad) and angular speed (rad/s).
    Muscle tone at rest + |angle| / range, scaled by a rest / walk / run gain interpolated from the joint speed."""
    gains = np.asarray(gains, float)
    gain = np.interp(np.abs(speed), STATE_SPEEDS, gains)
    base = np.abs(angle) / max(float(rng), 1e-6)
    return np.clip(tone + base * gain, 0.0, 1.5)


def jiggle_step(x, v, target, dt, k=120.0, damping=9.0, amount=1.0):
    """Damped spring on a flex value (semi-implicit Euler).  Used by the engine runtime; included here so the solver's
    behaviour is reproducible and tested."""
    a = amount * (k * (target - x) - damping * v)
    v = v + a * dt
    x = x + v * dt
    return x, v


# --------------------------------------------------------------------------------------------------------------------
# data export
# --------------------------------------------------------------------------------------------------------------------
def result_to_json(instances, regions, meta=None):
    """JSON text, one instance per line (keeps Blender text-block writes fast)."""
    head = json.dumps(dict(version=1, meta=meta or {},
                           regions=[{k: (list(v) if isinstance(v, (tuple, np.ndarray)) else v)
                                     for k, v in r.items() if k in ('key', 'label', 'group', 'cid', 'color', 'flex',
                                                                    'accent', 'emphasis', 'tone', 'range', 'bones')}
                                    for r in regions]))
    body = ",\n".join(json.dumps(i.to_dict(4)) for i in instances)
    return head[:-1] + ', "instances": [\n' + body + "\n]}"


def instances_from_json(text):
    d = json.loads(text)
    return [Instance.from_dict(x) for x in d['instances']], d
