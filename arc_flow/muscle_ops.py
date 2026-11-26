"""Tessera Projection muscle solver operators and helpers (Blender side).

What is real time and what is baked
  * Solve (offline, one shot, ~0.1 - 2 s for 5 - 20k vertices): region projection, tendon / fibre axis, flex weights and
    unit-flex displacement fields.  Results are stored on the skin (point attributes tp_muscle_*), in a text block
    (TP_Muscle_Solve, JSON) and cached in memory.
  * Real time: the baked fields are evaluated by shape keys + drivers (joint angle, animation-state gain), by the
    "Tessera Muscles" Geometry Nodes group (Flex / Accent / Region inputs) and by the game-engine sidecar (JSON).
  * Not simulated in Blender: jiggle.  Its spring parameters are exported for the engine runtime.
"""
import json
import math
import os

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty
from bpy.types import Operator

from .core import muscle_solver as MS

ATTR_ID = "tp_muscle_id"
ATTR_COLOR = "tp_muscle_color"
ATTR_SOLVED_COLOR = "tp_muscle_solved_color"
ATTR_BULGE = "tp_muscle_bulge"
ATTR_SLIDE = "tp_muscle_slide"
ATTR_FLEX = "tp_muscle_flex"
KEY_PREFIX = "TP_Muscle_"
TEXT_SOLVE = "TP_Muscle_Solve"
TEXT_SIDECAR = "TP_Muscle_Sidecar"
PROP_STROKES = "tp_muscle_strokes"
CONTOUR_OBJ = "TP Muscle Contours"
CTRL_PREFIX = "CTRL_"

_CACHE = {}      # skin object name -> list of Instance


def _ms(scene):
    return scene.arc_flow_muscle


def _log(scene, text):
    _ms(scene).status = text
    return text


# --------------------------------------------------------------------------------------------------------------------
# mesh <-> numpy
# --------------------------------------------------------------------------------------------------------------------
def skin_arrays(obj):
    me = obj.data
    n = len(me.vertices)
    v = np.empty(n * 3, np.float64)
    me.vertices.foreach_get('co', v)
    e = np.empty(len(me.edges) * 2, np.int64)
    me.edges.foreach_get('vertices', e)
    nr = np.empty(n * 3, np.float64)
    me.vertex_normals.foreach_get('vector', nr)
    return v.reshape(-1, 3), e.reshape(-1, 2), nr.reshape(-1, 3)


def basis_coords(obj):
    me = obj.data
    if me.shape_keys and me.shape_keys.key_blocks:
        kb = me.shape_keys.key_blocks[0]
        a = np.empty(len(kb.data) * 3, np.float64)
        kb.data.foreach_get('co', a)
        return a.reshape(-1, 3)
    return skin_arrays(obj)[0]


def read_ids(obj):
    me = obj.data
    a = me.attributes.get(ATTR_ID)
    if a is None or a.domain != 'POINT':
        return np.zeros(len(me.vertices), np.int64)
    out = np.empty(len(me.vertices), np.int64)
    a.data.foreach_get('value', out)
    return out


def write_ids(obj, ids):
    me = obj.data
    a = me.attributes.get(ATTR_ID)
    if a is None or a.domain != 'POINT' or a.data_type != 'INT':
        if a is not None:
            me.attributes.remove(a)
        a = me.attributes.new(ATTR_ID, 'INT', 'POINT')
    a.data.foreach_set('value', np.asarray(ids, np.int32))
    me.update()


def write_colors(obj, ms):
    """Paint the static region colours into the colour attribute and make it the active / render colour."""
    me = obj.data
    ids = read_ids(obj)
    table = {r.cid: tuple(r.color) for r in ms.regions}
    rgba = np.empty((len(ids), 4), np.float32)
    rgba[:] = (0.62, 0.60, 0.58, 1.0)
    for cid in np.unique(ids):
        if cid and int(cid) in table:
            rgba[ids == cid, :3] = table[int(cid)]
    a = me.color_attributes.get(ATTR_COLOR)
    if a is None or a.domain != 'POINT' or a.data_type != 'FLOAT_COLOR':
        if a is not None:
            me.color_attributes.remove(a)
        a = me.color_attributes.new(ATTR_COLOR, 'FLOAT_COLOR', 'POINT')
    a.data.foreach_set('color', rgba.ravel())
    try:
        me.color_attributes.active_color = a
        me.color_attributes.render_color_index = list(me.color_attributes.keys()).index(ATTR_COLOR)
    except Exception:
        pass
    me.update()


def write_solved_colors(obj, ms, label, strength):
    """Colour of the solved region partition (what the projection made of the paint), shaded by membership strength."""
    me = obj.data
    table = {i: tuple(r.color) for i, r in enumerate(ms.regions)}
    rgba = np.empty((len(label), 4), np.float32)
    rgba[:] = (0.36, 0.35, 0.34, 1.0)
    st = np.clip(np.asarray(strength) * 1.6, 0.0, 1.0)
    for ri in np.unique(label):
        if ri >= 0 and int(ri) in table:
            sel = label == ri
            c = np.array(table[int(ri)])
            rgba[sel, :3] = (np.array([0.36, 0.35, 0.34])[None, :] * (1 - st[sel, None]) + c[None, :] * st[sel, None])
    a = me.color_attributes.get(ATTR_SOLVED_COLOR)
    if a is None or a.domain != 'POINT' or a.data_type != 'FLOAT_COLOR':
        if a is not None:
            me.color_attributes.remove(a)
        a = me.color_attributes.new(ATTR_SOLVED_COLOR, 'FLOAT_COLOR', 'POINT')
    a.data.foreach_set('color', rgba.ravel())
    me.update()


def set_active_color(obj, name):
    me = obj.data
    a = me.color_attributes.get(name)
    if a is None:
        return False
    me.color_attributes.active_color = a
    me.color_attributes.render_color_index = list(me.color_attributes.keys()).index(name)
    return True


def show_colors_in_viewports(ctx, on=True):
    for win in ctx.window_manager.windows:
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                sh = area.spaces.active.shading
                if on:
                    sh.type = 'SOLID' if sh.type not in ('SOLID', 'MATERIAL', 'RENDERED') else sh.type
                    for ct in ('VERTEX', 'ATTRIBUTE'):
                        try:
                            sh.color_type = ct
                            break
                        except Exception:
                            continue
                else:
                    sh.color_type = 'MATERIAL'


def cid_to_index(ms):
    return {r.cid: i for i, r in enumerate(ms.regions)}


def region_dicts(ms):
    out = []
    eff = region_params(ms)
    for r in ms.regions:
        cat = MS.CATALOG[MS.CATALOG_INDEX[r.key]] if r.key in MS.CATALOG_INDEX else {}
        e = eff[r.key]
        out.append(dict(key=r.key, label=r.label, group=r.group, cid=r.cid, color=tuple(r.color), enabled=r.enabled,
                        flex=e['flex'], accent=e['accent'], emphasis=e['emphasis'], tone=r.tone, range=r.joint_range,
                        bones=tuple(cat.get('bones', ())) or ((r.bone,) if r.bone else ()),
                        pair=bool(cat.get('pair', False)), t=cat.get('t', (0, 1)), facing=cat.get('facing', 'any'), sub=cat.get('sub', 1),
                        driver_bone=r.bone))
    return out


def ensure_cards(ms):
    """Create the region category cards once (Head and face, Arms and hands, Torso and back, Legs and feet, Tail and
    custom).  Existing cards keep their sliders."""
    have = {c.key for c in ms.cards}
    for key, label, _icon, _groups in MS.CARDS:
        if key not in have:
            c = ms.cards.add()
            c.key, c.label = key, label
    return ms.cards


def card_params(ms):
    """{card key: dict(flex, accent, emphasis)} from the card sliders (missing cards = 1.0)."""
    return {c.key: dict(flex=c.flex, accent=c.accent, emphasis=c.emphasis) for c in ms.cards}


def region_card(r):
    return MS.card_of(r.group)


def region_params(ms):
    """Per-muscle slider values with the region-card multipliers applied (what the solve outputs are built from)."""
    cards = card_params(ms)
    return {r.key: MS.effective_params(dict(flex=r.flex, accent=r.accent, emphasis=r.emphasis, enabled=r.enabled),
                                       cards.get(region_card(r)))
            for r in ms.regions}


def refresh_stroke_counts(obj, ms):
    st = load_strokes(obj) if obj is not None else []
    for r in ms.regions:
        r.n_strokes = sum(1 for x in st if x.get('cid') == r.cid)


def muscle_marked(ms, r):
    """True when the artist gave this muscle something to solve from: pencil lines, painted vertices or geometry."""
    if ms.skin is not None:
        refresh_painted_counts(ms.skin, ms)
        refresh_stroke_counts(ms.skin, ms)
    return r.n_strokes > 0 or r.n_painted > 0 or r.muscle_obj is not None or r.bone_obj is not None


def complete_muscle(scene, index=None):
    """'Complete' in the Muscle Pencil workflow: book the active muscle into its region card's list.  Returns the card key.
    Raises RuntimeError when the muscle has no marks yet."""
    ms = _ms(scene)
    ensure_cards(ms)
    i = ms.index if index is None else index
    r = ms.regions[i]
    if not muscle_marked(ms, r):
        raise RuntimeError("Mark %s first: draw a few pencil lines, paint it, or assign muscle / bone geometry" % r.label)
    r.completed = True
    r.enabled = True
    card = region_card(r)
    done = [x for x in ms.regions if region_card(x) == card and x.completed]
    nxt = next((j for j, x in enumerate(ms.regions) if region_card(x) == card and not x.completed and j != i), None)
    if nxt is not None:
        ms.index = nxt
    _log(scene, "%s added to %s (%d muscles complete%s)." % (r.label, MS.CARD_LABEL[card], len(done),
                                                            (", next: " + ms.regions[nxt].label) if nxt is not None else ""))
    return card


def add_catalog_region(ms, key):
    """Add one catalog muscle to the scene list (used by Add Region and the card Add menu).  Returns its index."""
    for j, r in enumerate(ms.regions):
        if r.key == key:
            return j
    d = next(r for r in MS.CATALOG if r['key'] == key)
    r = ms.regions.add()
    r.key, r.label, r.group, r.cid = d['key'], d['label'], d['group'], MS.catalog_cid(d['key'])
    r.color = MS.region_color(d['key'])
    r.emphasis, r.tone, r.joint_range = d['emphasis'], d['tone'], d['range']
    r.accent = 0.6
    ensure_cards(ms)
    return len(ms.regions) - 1


def write_export_properties(scene):
    """Store the muscle rig as custom properties on the skin object so glTF (extras) and FBX (user properties) carry it
    into a game engine together with the shape keys (morph targets) and the bone weights."""
    ms = _ms(scene)
    skin = ms.skin
    if skin is None:
        return None
    params = region_params(ms)
    keys = skin.data.shape_keys.key_blocks if skin.data.shape_keys is not None else ()
    muscles = []
    for inst in get_instances(scene):
        p = params.get(inst.key)
        if p is None or not p['enabled']:
            continue
        r = next((x for x in ms.regions if x.key == inst.key), None)
        muscles.append(dict(name=inst.name, key=inst.key, shape_key=key_name(inst) if key_name(inst) in keys else "",
                            card=MS.card_of(r.group) if r else "", bone=inst.bone or "", accent_end=inst.accent_end,
                            flex=round(p['flex'] * p['emphasis'], 4), accent=round(p['accent'], 4),
                            tone=round(r.tone, 4) if r else 0.06, joint_range=round(r.joint_range, 4) if r else 1.2))
    data = dict(format="tessera-muscles", version=1, preset=ms.preset, detail=ms.detail,
                resolution=round(ms.resolution, 4), speed_gains=[ms.gain_rest, ms.gain_walk, ms.gain_run],
                speed_curve_rad_per_s=list(MS.STATE_SPEEDS),
                cards={c.key: dict(flex=c.flex, accent=c.accent, emphasis=c.emphasis) for c in ms.cards},
                muscles=muscles)
    skin["tp_muscles"] = json.dumps(data, separators=(',', ':'))
    skin["tp_muscle_count"] = len(muscles)
    return data


# --------------------------------------------------------------------------------------------------------------------
# skeleton data
# --------------------------------------------------------------------------------------------------------------------
def find_armature(skin):
    for m in skin.modifiers:
        if m.type == 'ARMATURE' and m.object is not None:
            return m.object
    return skin.parent if skin.parent is not None and skin.parent.type == 'ARMATURE' else None


def bone_info(skin, arm):
    """Bone names, head / tail in skin-local rest space, and an (n, nb) skin weight matrix from the vertex groups."""
    if skin is None or arm is None:
        return None
    to_skin = skin.matrix_world.inverted() @ arm.matrix_world
    names, heads, tails = [], [], []
    for b in arm.data.bones:
        if not b.use_deform:
            continue
        names.append(b.name)
        heads.append(np.array(to_skin @ b.head_local))
        tails.append(np.array(to_skin @ b.tail_local))
    if not names:
        return None
    gmap = {vg.index: names.index(vg.name) for vg in skin.vertex_groups if vg.name in names}
    n = len(skin.data.vertices)
    W = np.zeros((n, len(names)))
    for v in skin.data.vertices:
        for g in v.groups:
            j = gmap.get(g.group)
            if j is not None:
                W[v.index, j] = g.weight
    return dict(names=names, heads=np.array(heads), tails=np.array(tails), weights=W)


# --------------------------------------------------------------------------------------------------------------------
# presets / painting / seeds / strokes
# --------------------------------------------------------------------------------------------------------------------
def apply_preset(scene):
    ms = _ms(scene)
    cr = dict(heads=ms.heads, arms=ms.arms, legs=ms.legs, tails=ms.tails, fingers=ms.fingers, toes=ms.toes,
              joints_per_limb=ms.joints_per_limb, spine_segments=3)
    regs = MS.regions_for(ms.preset, ms.detail, cr)
    keep = {r.key: (r.flex, r.accent) for r in ms.regions}
    ms.regions.clear()
    for d in regs:
        r = ms.regions.add()
        r.key, r.label, r.group, r.cid = d['key'], d['label'], d['group'], d['cid']
        r.color = d['color']
        r.emphasis, r.tone, r.joint_range = d['emphasis'], d['tone'], d['range']
        r.flex, r.accent = keep.get(d['key'], (1.0, d['accent']))
    ms.index = 0
    ms.solved = False
    ensure_cards(ms)
    _log(scene, "%s, %s detail: %d muscle regions." % (MS.PRESETS[ms.preset]['label'], ms.detail.title(), len(ms.regions)))
    return len(ms.regions)


def paint_vertices(obj, ms, center, radius, cid, erase=False):
    """Pencil: set tp_muscle_id on every vertex within `radius` (skin local units) of `center` (vector)."""
    v = basis_coords(obj)
    d = np.linalg.norm(v - np.asarray(center, float), axis=1)
    sel = d <= radius
    ids = read_ids(obj)
    ids[sel] = 0 if erase else int(cid)
    write_ids(obj, ids)
    write_colors(obj, ms)
    return int(sel.sum())


def refresh_painted_counts(obj, ms):
    ids = read_ids(obj)
    for r in ms.regions:
        r.n_painted = int((ids == r.cid).sum())


def load_strokes(obj):
    try:
        return json.loads(obj.data.get(PROP_STROKES, "[]"))
    except Exception:
        return []


def save_strokes(obj, strokes):
    obj.data[PROP_STROKES] = json.dumps(strokes)


def update_contour_curve(scene, skin):
    """Visible polyline curves for the contour strokes (like Grease Pencil strokes on the surface)."""
    ms = _ms(scene)
    strokes = load_strokes(skin)
    cu = bpy.data.curves.get(CONTOUR_OBJ)
    if cu is None:
        cu = bpy.data.curves.new(CONTOUR_OBJ, 'CURVE')
    cu.dimensions = '3D'
    cu.splines.clear()
    cu.bevel_depth = 0.0025
    v = basis_coords(skin)
    _, _, nrm = skin_arrays(skin)
    for st in strokes:
        ids = [i for i in st['verts'] if i < len(v)]
        if len(ids) < 2:
            continue
        sp = cu.splines.new('POLY')
        sp.points.add(len(ids) - 1)
        for p, i in zip(sp.points, ids):
            q = v[i] + nrm[i] * 0.004
            p.co = (q[0], q[1], q[2], 1.0)
    ob = bpy.data.objects.get(CONTOUR_OBJ)
    if ob is None:
        ob = bpy.data.objects.new(CONTOUR_OBJ, cu)
        scene.collection.objects.link(ob)
    ob.parent = skin
    ob.matrix_parent_inverse.identity()
    ob.color = (1.0, 0.85, 0.1, 1.0)
    ob.show_in_front = True
    return ob


def selected_vertex_chain(obj):
    """Selected vertices of the mesh in edit mode, ordered along the selected edges when they form a chain."""
    import bmesh
    me = obj.data
    if obj.mode == 'EDIT':
        bm = bmesh.from_edit_mesh(me)
    else:
        bm = bmesh.new()
        bm.from_mesh(me)
    sel = {v.index for v in bm.verts if v.select}
    adj = {i: [] for i in sel}
    for e in bm.edges:
        a, b = e.verts[0].index, e.verts[1].index
        if a in sel and b in sel and e.select:
            adj[a].append(b)
            adj[b].append(a)
    if obj.mode != 'EDIT':
        bm.free()
    if not sel:
        return []
    start = next((i for i in sorted(sel) if len(adj[i]) <= 1), min(sel))
    chain, seen, cur = [], set(), start
    while cur is not None and cur not in seen:
        chain.append(cur)
        seen.add(cur)
        nxt = [j for j in adj[cur] if j not in seen]
        cur = nxt[0] if nxt else None
    rest = sorted(sel - seen)
    return chain + rest


def seed_labels(skin, ms):
    ids = read_ids(skin)
    c2i = cid_to_index(ms)
    lab = -np.ones(len(ids), np.int64)
    for cid, ri in c2i.items():
        lab[ids == cid] = ri
    return lab


def muscle_obj_seeds(skin, ms, lab):
    """Skin vertices lying on the projection of an assigned muscle mesh are seeded with that region."""
    v = basis_coords(skin)
    ed = skin_arrays(skin)[1]
    mean_edge = float(np.mean(np.linalg.norm(v[ed[:, 0]] - v[ed[:, 1]], axis=1)))
    att = {}
    inv = skin.matrix_world.inverted()
    for ri, r in enumerate(ms.regions):
        pts = []
        if r.muscle_obj is not None and r.muscle_obj.type == 'MESH':
            mo = r.muscle_obj
            P = np.array([tuple(inv @ (mo.matrix_world @ p.co)) for p in mo.data.vertices])
            if len(P):
                ext = np.sort(np.ptp(P, axis=0))
                thick = float(ext[:2].mean())
                d = np.min(np.linalg.norm(v[:, None, :] - P[None, ::max(1, len(P) // 400), :], axis=2), axis=1)
                hit = d <= (0.6 * thick + mean_edge)
                lab[hit & (lab < 0)] = ri
                c, a = MS._pca_axis(P)
                pr = (P - c) @ a
                pts += [P[np.argmin(pr)], P[np.argmax(pr)]]
        if r.bone_obj is not None and r.bone_obj.type == 'MESH':
            bo = r.bone_obj
            P = np.array([tuple(inv @ (bo.matrix_world @ p.co)) for p in bo.data.vertices])
            if len(P):
                c, a = MS._pca_axis(P)
                pr = (P - c) @ a
                pts += [P[np.argmin(pr)], P[np.argmax(pr)]]
        att[ri] = pts
    return att


def bone_end_attachments(skin, arm, ms, att):
    if arm is None:
        return att
    to_skin = skin.matrix_world.inverted() @ arm.matrix_world
    for ri, r in enumerate(ms.regions):
        if r.attach_bone_ends and r.bone and r.bone in arm.data.bones:
            b = arm.data.bones[r.bone]
            att.setdefault(ri, [])
            att[ri] += [np.array(to_skin @ b.head_local), np.array(to_skin @ b.tail_local)]
    return att


def run_auto_seed(scene, overwrite=False):
    ms = _ms(scene)
    skin, arm = ms.skin, ms.armature
    info = bone_info(skin, arm)
    if info is None:
        raise RuntimeError("Auto-seed needs a skeleton with deform bones and skin weights")
    v, _, nrm = skin_arrays(skin)
    lab, _ = MS.auto_seed(v, nrm, info['names'], info['heads'], info['tails'], info['weights'], region_dicts(ms))
    ids = read_ids(skin)
    for ri, r in enumerate(ms.regions):
        take = (lab == ri) & ((ids == 0) | overwrite)
        ids[take] = r.cid
    write_ids(skin, ids)
    write_colors(skin, ms)
    refresh_painted_counts(skin, ms)
    return int((lab >= 0).sum())


# --------------------------------------------------------------------------------------------------------------------
# solve / bake
# --------------------------------------------------------------------------------------------------------------------
def run_solve(scene):
    import time
    ms = _ms(scene)
    skin = ms.skin
    if skin is None:
        raise RuntimeError("Assign a skin mesh first")
    t0 = time.perf_counter()
    v = basis_coords(skin)
    _, edges, _ = skin_arrays(skin)
    nrm = skin_arrays(skin)[2]
    lab = seed_labels(skin, ms)
    att = muscle_obj_seeds(skin, ms, lab)
    arm = ms.armature or find_armature(skin)
    att = bone_end_attachments(skin, arm, ms, att)
    c2i = cid_to_index(ms)
    strokes = [dict(region=c2i[s['cid']], verts=s['verts']) for s in load_strokes(skin) if s['cid'] in c2i]
    info = bone_info(skin, arm) if arm else None
    regs = region_dicts(ms)
    out = MS.solve(v, edges, nrm, lab, regs, strokes=strokes, attachments=att, bone_info=info, res=ms.resolution)
    insts = out['instances']
    for inst in insts:
        r = next((x for x in ms.regions if x.key == inst.key), None)
        if r is not None and r.bone and r.bone in (info['names'] if info else []):
            inst.bone = r.bone
        if r is not None and r.accent_end in ('A', 'B'):
            if r.accent_end != inst.accent_end:
                inst.accent_end = r.accent_end
                toward = (inst.a - inst.b) if r.accent_end == 'A' else (inst.b - inst.a)
                nrmv = np.linalg.norm(inst.slide, axis=1)
                toward = toward / max(np.linalg.norm(toward), 1e-12)
                inst.slide = toward[None, :] * nrmv[:, None]
    _CACHE[skin.name] = insts
    write_solved_colors(skin, ms, out['label'], out['membership'].max(axis=1) if out['membership'].size else np.zeros(len(v)))
    for r in ms.regions:
        r.n_instances = sum(1 for i in insts if i.key == r.key)
    refresh_painted_counts(skin, ms)
    store_solve(scene, insts)
    write_attributes(scene, insts)
    ms.solved = True
    ms.n_instances = len(insts)
    ms.last_ms = 1000.0 * (time.perf_counter() - t0)
    _log(scene, "Solved %d muscles in %d regions in %.0f ms (%d vertices, resolution %.2f)."
         % (len(insts), len({i.key for i in insts}), ms.last_ms, len(v), ms.resolution))
    return insts


def store_solve(scene, insts):
    ms = _ms(scene)
    txt = bpy.data.texts.get(TEXT_SOLVE) or bpy.data.texts.new(TEXT_SOLVE)
    txt.clear()
    txt.write(MS.result_to_json(insts, region_dicts(ms), meta=dict(skin=ms.skin.name, resolution=ms.resolution,
                                                                      preset=ms.preset, detail=ms.detail)))
    txt.use_fake_user = True


def get_instances(scene):
    ms = _ms(scene)
    skin = ms.skin
    if skin is None:
        return []
    insts = _CACHE.get(skin.name)
    if insts is None:
        txt = bpy.data.texts.get(TEXT_SOLVE)
        if txt is not None:
            try:
                insts, _ = MS.instances_from_json(txt.as_string())
                _CACHE[skin.name] = insts
            except Exception:
                insts = []
    return insts or []


def instance_displacement(inst, ms, n):
    p = region_params(ms)
    return MS.assemble(n, [inst], p)


def write_attributes(scene, insts):
    ms = _ms(scene)
    skin = ms.skin
    me = skin.data
    n = len(me.vertices)
    D_b = np.zeros((n, 3))
    D_s = np.zeros((n, 3))
    W = np.zeros(n)
    best = np.zeros(n)
    rid = np.zeros(n, np.int64)
    params = region_params(ms)
    cidmap = {r.key: r.cid for r in ms.regions}
    for inst in insts:
        p = params.get(inst.key)
        if p is None or not p['enabled']:
            continue
        k = p['flex'] * p['emphasis']
        D_b[inst.idx] += k * inst.bulge
        D_s[inst.idx] += k * p['accent'] * inst.slide
        W[inst.idx] += inst.w
        upd = inst.w > best[inst.idx]
        rid[inst.idx[upd]] = cidmap.get(inst.key, 0)
        best[inst.idx[upd]] = inst.w[upd]
    for name, arr, kind, dom in ((ATTR_BULGE, D_b, 'FLOAT_VECTOR', 'POINT'), (ATTR_SLIDE, D_s, 'FLOAT_VECTOR', 'POINT')):
        a = me.attributes.get(name)
        if a is None or a.data_type != kind:
            if a is not None:
                me.attributes.remove(a)
            a = me.attributes.new(name, kind, dom)
        a.data.foreach_set('vector', arr.astype(np.float32).ravel())
    a = me.attributes.get(ATTR_FLEX)
    if a is None or a.data_type != 'FLOAT':
        if a is not None:
            me.attributes.remove(a)
        a = me.attributes.new(ATTR_FLEX, 'FLOAT', 'POINT')
    a.data.foreach_set('value', np.minimum(W, 1.0).astype(np.float32))
    write_ids_keep = read_ids(skin)
    solved_id = me.attributes.get("tp_muscle_solved_id")
    if solved_id is None:
        solved_id = me.attributes.new("tp_muscle_solved_id", 'INT', 'POINT')
    solved_id.data.foreach_set('value', rid.astype(np.int32))
    me.update()
    return write_ids_keep


def key_name(inst):
    return KEY_PREFIX + inst.name


def bake_shape_keys(scene, drivers=None):
    """One shape key per muscle instance (delta from Basis = unit flex with the current region sliders)."""
    ms = _ms(scene)
    skin = ms.skin
    insts = get_instances(scene)
    if not insts:
        raise RuntimeError("Solve first: there are no muscles to bake")
    me = skin.data
    if me.shape_keys is None:
        skin.shape_key_add(name="Basis", from_mix=False)
    keys = me.shape_keys.key_blocks
    for kb in list(keys):
        if kb.name.startswith(KEY_PREFIX):
            skin.shape_key_remove(kb)
    basis = basis_coords(skin)
    n = len(basis)
    params = region_params(ms)
    made = 0
    for inst in insts:
        p = params.get(inst.key)
        if p is None or not p['enabled']:
            continue
        d = MS.assemble(n, [inst], params)
        kb = skin.shape_key_add(name=key_name(inst), from_mix=False)
        kb.data.foreach_set('co', (basis + d).astype(np.float32).ravel())
        kb.slider_min, kb.slider_max = 0.0, 1.5
        r = next((x for x in ms.regions if x.key == inst.key), None)
        kb.value = float(r.tone) if r is not None else 0.0
        kb.relative_key = keys[0]
        made += 1
    use_drivers = ms.use_drivers if drivers is None else drivers
    nd = setup_drivers(scene) if use_drivers else 0
    write_export_properties(scene)
    _log(scene, "Baked %d shape keys%s." % (made, (", %d driven by joint rotation" % nd) if use_drivers else ""))
    return made


def setup_drivers(scene):
    """Drive each muscle key from its bone's local rotation: tone + max(|rot X|, |rot Z|) / range * state gain."""
    ms = _ms(scene)
    skin = ms.skin
    arm = ms.armature or find_armature(skin)
    me = skin.data
    if me.shape_keys is None or arm is None:
        return 0
    # clear old drivers
    ad = me.shape_keys.animation_data
    if ad is not None:
        for fc in list(ad.drivers):
            if fc.data_path.startswith('key_blocks["' + KEY_PREFIX):
                ad.drivers.remove(fc)
    count = 0
    for inst in get_instances(scene):
        kb = me.shape_keys.key_blocks.get(key_name(inst))
        if kb is None or not inst.bone or inst.bone not in arm.pose.bones:
            continue
        r = next((x for x in ms.regions if x.key == inst.key), None)
        tone = float(r.tone) if r else 0.06
        rng = float(r.joint_range) if r else 1.2
        fc = kb.driver_add('value')
        d = fc.driver
        d.type = 'SCRIPTED'
        for vn, tt in (('x', 'ROT_X'), ('z', 'ROT_Z')):
            var = d.variables.new()
            var.name = vn
            var.type = 'TRANSFORMS'
            t = var.targets[0]
            t.id = arm
            t.bone_target = inst.bone
            t.transform_type = tt
            t.transform_space = 'LOCAL_SPACE'
        g = d.variables.new()
        g.name = 'g'
        g.type = 'SINGLE_PROP'
        g.targets[0].id_type = 'SCENE'
        g.targets[0].id = scene
        g.targets[0].data_path = 'arc_flow_muscle.state_gain'
        d.expression = "min(1.5, %.4f + max(abs(x), abs(z)) / %.4f * g)" % (tone, rng)
        count += 1
    return count


def apply_preview(scene, key):
    ms = _ms(scene)
    skin = ms.skin
    if skin is None or skin.data.shape_keys is None:
        return
    r = next((x for x in ms.regions if x.key == key), None)
    if r is None:
        return
    for inst in get_instances(scene):
        if inst.key != key:
            continue
        kb = skin.data.shape_keys.key_blocks.get(key_name(inst))
        if kb is None:
            continue
        ad = skin.data.shape_keys.animation_data
        driven = ad is not None and any(fc.data_path == 'key_blocks["%s"].value' % kb.name for fc in ad.drivers)
        if driven:
            for fc in ad.drivers:
                if fc.data_path == 'key_blocks["%s"].value' % kb.name:
                    fc.mute = r.preview > 0.0
        kb.value = r.preview if r.preview > 0.0 else r.tone


# --------------------------------------------------------------------------------------------------------------------
# sidecar export
# --------------------------------------------------------------------------------------------------------------------
def build_sidecar(scene):
    ms = _ms(scene)
    skin = ms.skin
    insts = get_instances(scene)
    n = len(skin.data.vertices)
    basis = basis_coords(skin)
    params = region_params(ms)
    keys, regions = [], []
    for r in ms.regions:
        regions.append(dict(key=r.key, label=r.label, group=r.group, card=MS.card_of(r.group), id=r.cid,
                            color=[round(c, 4) for c in r.color], completed=bool(r.completed),
                            flex=params[r.key]['flex'], accent=params[r.key]['accent'], emphasis=params[r.key]['emphasis'],
                            enabled=r.enabled, rest_tone=r.tone,
                            joint_range_rad=r.joint_range, instances=r.n_instances))
    for inst in insts:
        p = params.get(inst.key)
        if p is None or not p['enabled']:
            continue
        r = next((x for x in ms.regions if x.key == inst.key), None)
        d = MS.assemble(n, [inst], params)[inst.idx]
        keys.append(dict(
            name=key_name(inst), region=inst.key, side=inst.side, bone=inst.bone, source=inst.source,
            accent_end=inst.accent_end, axis=[np.round(inst.a, 5).tolist(), np.round(inst.b, 5).tolist()],
            vertices=inst.idx.tolist(), weights=np.round(inst.w, 5).tolist(),
            morph_delta=np.round(d, 6).tolist(),
            control=dict(driver_bone=inst.bone, rotation_channels=["local_x", "local_z"],
                         formula="flex = clamp(tone + max(|x|,|z|) / joint_range * state_gain(speed), 0, 1.5)",
                         tone=r.tone if r else 0.06, joint_range_rad=r.joint_range if r else 1.2,
                         speed_curve_rad_per_s=list(MS.STATE_SPEEDS),
                         speed_gains=[ms.gain_rest, ms.gain_walk, ms.gain_run],
                         jiggle=dict(enabled=bool(ms.use_jiggle), spring=ms.jiggle_k, damping=ms.jiggle_damping))))
    rig = []
    arm = ms.armature or find_armature(skin)
    if arm is not None:
        for b in arm.data.bones:
            if b.name.startswith(CTRL_PREFIX):
                tgt = b.name[len(CTRL_PREFIX):]
                rig.append(dict(control=b.name, drives=tgt if tgt in arm.data.bones else None))
    return dict(format="tessera-muscles", version=1, skin=skin.name, vertex_count=n,
                resolution_factor=ms.resolution, preset=ms.preset, detail=ms.detail, units="skin local",
                regions=regions, muscles=keys, control_rig=rig,
                notes="morph_delta is the per-vertex offset at key value 1 (glTF morph target equivalent); "
                      "flex evaluates per frame from the driver bone's local rotation and angular speed")


def write_sidecar(scene, path):
    data = build_sidecar(scene)
    with open(path, 'w') as f:
        json.dump(data, f)
    txt = bpy.data.texts.get(TEXT_SIDECAR) or bpy.data.texts.new(TEXT_SIDECAR)
    txt.clear()
    txt.write(json.dumps({k: v for k, v in data.items() if k != 'muscles'}))
    return data


def export_game(scene, base_path, fmt):
    """Export skin + armature with shape keys, weights and vertex colours; returns the file path."""
    ms = _ms(scene)
    skin = ms.skin
    arm = ms.armature or find_armature(skin)
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    skin.select_set(True)
    if arm is not None:
        arm.select_set(True)
    bpy.context.view_layer.objects.active = skin
    write_export_properties(scene)
    path = bpy.path.abspath(base_path)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    if fmt == 'GLB':
        path = path if path.lower().endswith('.glb') else path + '.glb'
        kw = dict(filepath=path, export_format='GLB', use_selection=True, export_morph=True, export_skins=True,
                  export_vertex_color='ACTIVE', export_apply=False, export_animations=False, export_extras=True)
        bpy.ops.export_scene.gltf(**kw)
    else:
        path = path if path.lower().endswith('.fbx') else path + '.fbx'
        bpy.ops.export_scene.fbx(filepath=path, use_selection=True, add_leaf_bones=False, colors_type='SRGB',
                                 object_types={'ARMATURE', 'MESH'}, use_mesh_modifiers=False, bake_anim=False,
                                 use_custom_props=True)
    return path


# --------------------------------------------------------------------------------------------------------------------
# control rig (minimal FK / IK helper)
# --------------------------------------------------------------------------------------------------------------------
def _circle_shape():
    ob = bpy.data.objects.get("TP_Muscle_CtrlCircle")
    if ob is not None:
        return ob
    me = bpy.data.meshes.new("TP_Muscle_CtrlCircle")
    n = 24
    verts = [(0.5 * math.cos(2 * math.pi * i / n), 0.0, 0.5 * math.sin(2 * math.pi * i / n)) for i in range(n)]
    me.from_pydata(verts, [(i, (i + 1) % n) for i in range(n)], [])
    ob = bpy.data.objects.new("TP_Muscle_CtrlCircle", me)
    bpy.context.scene.collection.objects.link(ob)
    ob.hide_render = True
    ob.hide_set(True)
    return ob


def build_control_rig(scene, ik=True):
    """CTRL_<bone> circle controllers (FK) copy-rotating every deform bone, plus optional IK handles on forearms / shins."""
    ms = _ms(scene)
    arm = ms.armature or find_armature(ms.skin)
    if arm is None:
        raise RuntimeError("No armature")
    shape = _circle_shape()
    prev_active = bpy.context.view_layer.objects.active
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    if arm.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.mode_set(mode='EDIT')
    eb = arm.data.edit_bones
    deform = [b.name for b in eb if b.use_deform and not b.name.startswith(CTRL_PREFIX)]
    made_ctrl, made_ik = [], []
    for name in deform:
        if (CTRL_PREFIX + name) in eb:
            continue
        src = eb[name]
        c = eb.new(CTRL_PREFIX + name)
        c.head = src.head
        c.tail = src.tail
        c.roll = src.roll
        c.use_deform = False
        made_ctrl.append(c.name)
    for name in deform:
        c = eb.get(CTRL_PREFIX + name)
        src = eb[name]
        if c is not None and src.parent is not None and src.parent.name in deform:
            c.parent = eb.get(CTRL_PREFIX + src.parent.name)
    if ik:
        for name in deform:
            low = name.lower()
            if any(k in low for k in ('forearm', 'shin', 'calf')) and ("IK_" + name) not in eb:
                src = eb[name]
                h = eb.new("IK_" + name)
                h.head = src.tail
                h.tail = src.tail + (src.tail - src.head).normalized() * 0.08
                h.use_deform = False
                made_ik.append((name, h.name))
    bpy.ops.object.mode_set(mode='POSE')
    pb = arm.pose.bones
    for name in deform:
        c = pb.get(CTRL_PREFIX + name)
        if c is None:
            continue
        c.custom_shape = shape
        c.use_custom_shape_bone_size = True if hasattr(c, 'use_custom_shape_bone_size') else False
        con = pb[name].constraints.get("TP FK") or pb[name].constraints.new('COPY_ROTATION')
        con.name = "TP FK"
        con.target = arm
        con.subtarget = c.name
        con.target_space = 'LOCAL'
        con.owner_space = 'LOCAL'
    for name, hname in made_ik:
        pb[hname].custom_shape = shape
        con = pb[name].constraints.get("TP IK") or pb[name].constraints.new('IK')
        con.name = "TP IK"
        con.target = arm
        con.subtarget = hname
        con.chain_count = 2
        con.influence = 0.0       # IK / FK blend: raise the constraint influence (or key it) to switch a limb to IK
    bpy.ops.object.mode_set(mode='OBJECT')
    if prev_active is not None:
        bpy.context.view_layer.objects.active = prev_active
    return made_ctrl, [h for _, h in made_ik]


# --------------------------------------------------------------------------------------------------------------------
# example: procedural biped skin + armature
# --------------------------------------------------------------------------------------------------------------------
_JOINTS = {   # name: (x, y, z, radius)
    'pelvis': (0, 0, 0.95, 0.15), 'spine': (0, 0, 1.15, 0.14), 'chest': (0, 0, 1.40, 0.165), 'cn': (0, 0, 1.50, 0.14),
    'neck': (0, 0, 1.60, 0.058), 'headb': (0, 0, 1.68, 0.09), 'headm': (0, 0, 1.75, 0.125), 'headt': (0, 0, 1.83, 0.085),
    'nose': (0, -0.13, 1.735, 0.026),
    'sh': (0.20, 0, 1.50, 0.062), 'el': (0.46, 0, 1.25, 0.046), 'wr': (0.70, 0, 1.03, 0.034), 'hand': (0.80, 0, 0.93, 0.036),
    'hp': (0.10, 0, 0.93, 0.10), 'kn': (0.11, -0.01, 0.50, 0.062), 'an': (0.11, 0, 0.10, 0.046), 'toe': (0.11, -0.17, 0.045, 0.04),
}
_EDGES = [('pelvis', 'spine'), ('spine', 'chest'), ('chest', 'cn'), ('cn', 'neck'), ('neck', 'headb'), ('headb', 'headm'), ('headm', 'headt'), ('headm', 'nose'),
          ('cn', 'sh'), ('sh', 'el'), ('el', 'wr'), ('wr', 'hand'), ('pelvis', 'hp'), ('hp', 'kn'), ('kn', 'an'), ('an', 'toe')]
_BONES = [  # name, head point, tail point, parent  (+x side gets .L, mirrored .R)
    ('pelvis', (0, 0, .95), (0, 0, 1.15), None), ('spine', (0, 0, 1.15), (0, 0, 1.40), 'pelvis'),
    ('chest', (0, 0, 1.40), (0, 0, 1.52), 'spine'), ('neck', (0, 0, 1.52), (0, 0, 1.62), 'chest'),
    ('head', (0, 0, 1.62), (0, 0, 1.84), 'neck'),
]
_SIDE_BONES = [
    ('shoulder', (0, 0, 1.50), (.20, 0, 1.50), 'chest'), ('upper_arm', (.20, 0, 1.50), (.46, 0, 1.25), 'shoulder'),
    ('forearm', (.46, 0, 1.25), (.70, 0, 1.03), 'upper_arm'), ('hand', (.70, 0, 1.03), (.82, 0, .92), 'forearm'),
    ('thigh', (.10, 0, .93), (.11, -.01, .50), 'pelvis'), ('shin', (.11, -.01, .50), (.11, 0, .10), 'thigh'),
    ('foot', (.11, 0, .10), (.11, -.06, .045), 'shin'), ('toe', (.11, -.06, .045), (.11, -.18, .04), 'foot'),
]


def build_example_biped(scene, subdiv=2):
    """Procedural closed skin (Skin + Subdivision modifiers evaluated to a mesh) bound to a 20-bone armature."""
    import bmesh
    for n in ("Muscle Skin", "Muscle Rig"):
        o = bpy.data.objects.get(n)
        if o is not None:
            bpy.data.objects.remove(o, do_unlink=True)
    verts, idx = [], {}
    for k, (x, y, z, r) in _JOINTS.items():
        for sgn, suf in ((1, ''), (-1, 'R')) if k in ('sh', 'el', 'wr', 'hand', 'hp', 'kn', 'an', 'toe') else ((1, ''),):
            nm = k + suf
            idx[nm] = len(verts)
            verts.append((sgn * x, y, z, r))
    me = bpy.data.meshes.new("Muscle Skin Cage")
    edges = []
    for a, b in _EDGES:
        for suf in ('', 'R'):
            aa = a + suf if (a + suf) in idx else a
            bb = b + suf if (b + suf) in idx else b
            if aa != bb and (aa, bb) not in edges and (bb, aa) not in edges:
                edges.append((aa, bb))
    me.from_pydata([v[:3] for v in verts], [(idx[a], idx[b]) for a, b in edges], [])
    cage = bpy.data.objects.new("Muscle Skin Cage", me)
    scene.collection.objects.link(cage)
    bm = bmesh.new()
    bm.from_mesh(me)
    lay = bm.verts.layers.skin.verify()
    bm.verts.ensure_lookup_table()
    for i, v in enumerate(bm.verts):
        r = verts[i][3]
        v[lay].radius = (r, r)
        v[lay].use_root = (i == idx['pelvis'])
    bm.to_mesh(me)
    bm.free()
    md = cage.modifiers.new("Skin", 'SKIN')
    md.branch_smoothing = 0.6
    sub = cage.modifiers.new("Subsurf", 'SUBSURF')
    sub.levels = subdiv
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    skin_me = bpy.data.meshes.new_from_object(cage.evaluated_get(dg))
    skin_me.name = "Muscle Skin"
    bpy.data.objects.remove(cage, do_unlink=True)
    bpy.data.meshes.remove(me)
    skin = bpy.data.objects.new("Muscle Skin", skin_me)
    scene.collection.objects.link(skin)
    for p in skin_me.polygons:
        p.use_smooth = True
    # armature
    ad = bpy.data.armatures.new("Muscle Rig")
    arm = bpy.data.objects.new("Muscle Rig", ad)
    scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    eb = ad.edit_bones
    segs = []

    def add(name, h, t, parent):
        b = eb.new(name)
        b.head, b.tail = h, t
        if parent:
            b.parent = eb[parent]
        segs.append((name, np.array(h), np.array(t)))

    for nm, h, t, par in _BONES:
        add(nm, h, t, par)
    for suf, sg in ((".L", 1), (".R", -1)):
        for nm, h, t, par in _SIDE_BONES:
            hh = (sg * h[0], h[1], h[2])
            tt = (sg * t[0], t[1], t[2])
            add(nm + suf, hh, tt, (par + suf) if par in ('shoulder', 'upper_arm', 'forearm', 'thigh', 'shin', 'foot') else par)
    bpy.ops.object.mode_set(mode='OBJECT')
    # distance based weights (deterministic, no heat solve)
    co = np.empty(len(skin_me.vertices) * 3)
    skin_me.vertices.foreach_get('co', co)
    co = co.reshape(-1, 3)
    D = np.empty((len(co), len(segs)))
    for j, (nm, h, t) in enumerate(segs):
        s = t - h
        tt = np.clip(((co - h) @ s) / max(s @ s, 1e-12), 0, 1)
        D[:, j] = np.linalg.norm(co - (h + tt[:, None] * s), axis=1)
    W = 1.0 / (D + 0.012) ** 4
    keep = np.argsort(-W, axis=1)[:, :3]
    mask = np.zeros_like(W, bool)
    np.put_along_axis(mask, keep, True, axis=1)
    W = np.where(mask, W, 0.0)
    W /= W.sum(1, keepdims=True)
    vgs = [skin.vertex_groups.new(name=s[0]) for s in segs]
    for j, vg in enumerate(vgs):
        nz = np.nonzero(W[:, j] > 0.02)[0]
        for i in nz:
            vg.add([int(i)], float(W[i, j]), 'REPLACE')
    skin.parent = arm
    am = skin.modifiers.new("Armature", 'ARMATURE')
    am.object = arm
    ms = _ms(scene)
    ms.skin, ms.armature = skin, arm
    return skin, arm


# --------------------------------------------------------------------------------------------------------------------
# operators
# --------------------------------------------------------------------------------------------------------------------
class _Op:
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, ctx):
        return getattr(ctx.scene, 'arc_flow_muscle', None) is not None


def _need_skin(self, ctx):
    ms = _ms(ctx.scene)
    if ms.skin is None:
        self.report({'ERROR'}, "Assign a skin mesh first")
        return None
    return ms


class AF_OT_muscle_assign(_Op, Operator):
    bl_idname = "arcflow.muscle_assign"
    bl_label = "Use Selected"
    bl_description = "Use the selected mesh as the skin and the selected or modifier armature as the skeleton"

    def execute(self, ctx):
        ms = _ms(ctx.scene)
        skin = ctx.active_object if ctx.active_object is not None and ctx.active_object.type == 'MESH' else None
        if skin is None:
            skin = next((o for o in ctx.selected_objects if o.type == 'MESH'), None)
        arm = next((o for o in ctx.selected_objects if o.type == 'ARMATURE'), None)
        if skin is None:
            self.report({'ERROR'}, "Select a mesh object (and optionally its armature)")
            return {'CANCELLED'}
        ms.skin = skin
        ms.armature = arm or find_armature(skin)
        _log(ctx.scene, "Skin %s, skeleton %s." % (skin.name, ms.armature.name if ms.armature else "none"))
        return {'FINISHED'}


class AF_OT_muscle_apply_preset(_Op, Operator):
    bl_idname = "arcflow.muscle_apply_preset"
    bl_label = "Apply Preset"
    bl_description = "Build the muscle region list from the preset, detail level and creature counts"

    def execute(self, ctx):
        n = apply_preset(ctx.scene)
        return {'FINISHED'} if n else {'CANCELLED'}


class AF_OT_muscle_region_add(_Op, Operator):
    bl_idname = "arcflow.muscle_region_add"
    bl_label = "Add Region"
    bl_description = "Add a catalog muscle region that the preset left out"

    key: EnumProperty(name="Region", items=lambda s, c: [(r['key'], r['label'], "") for r in MS.CATALOG])

    def invoke(self, ctx, event):
        return ctx.window_manager.invoke_props_dialog(self)

    def execute(self, ctx):
        ms = _ms(ctx.scene)
        if any(r.key == self.key for r in ms.regions):
            return {'CANCELLED'}
        ms.index = add_catalog_region(ms, self.key)
        return {'FINISHED'}


class AF_OT_muscle_region_remove(_Op, Operator):
    bl_idname = "arcflow.muscle_region_remove"
    bl_label = "Remove Region"
    bl_description = "Remove the active region from the list"

    def execute(self, ctx):
        ms = _ms(ctx.scene)
        if not ms.regions:
            return {'CANCELLED'}
        ms.regions.remove(ms.index)
        ms.index = max(0, min(ms.index, len(ms.regions) - 1))
        return {'FINISHED'}


class AF_OT_muscle_example(_Op, Operator):
    bl_idname = "arcflow.muscle_example"
    bl_label = "Example: Procedural Biped"
    bl_description = "Create a procedural biped skin with an armature and apply the Biped preset"

    def execute(self, ctx):
        skin, arm = build_example_biped(ctx.scene)
        apply_preset(ctx.scene)
        _log(ctx.scene, "Example biped: %d skin vertices, %d bones. Auto-seed, then Solve." % (len(skin.data.vertices), len(arm.data.bones)))
        return {'FINISHED'}


class AF_OT_muscle_paint_selected(_Op, Operator):
    bl_idname = "arcflow.muscle_paint_selected"
    bl_label = "Assign Selection"
    bl_description = "Paint the active region onto the vertices selected in Edit Mode (or erase)"

    erase: BoolProperty(name="Erase", default=False)

    def execute(self, ctx):
        ms = _need_skin(self, ctx)
        if ms is None or not ms.regions:
            return {'CANCELLED'}
        skin = ms.skin
        was = skin.mode
        if was == 'EDIT':
            skin.update_from_editmode()
        sel = np.zeros(len(skin.data.vertices), bool)
        tmp = np.zeros(len(skin.data.vertices), bool)
        skin.data.vertices.foreach_get('select', tmp)
        sel |= tmp
        if not sel.any():
            self.report({'WARNING'}, "No vertices selected")
            return {'CANCELLED'}
        ids = read_ids(skin)
        ids[sel] = 0 if self.erase else ms.regions[ms.index].cid
        write_ids(skin, ids)
        write_colors(skin, ms)
        refresh_painted_counts(skin, ms)
        return {'FINISHED'}


class AF_OT_muscle_paint_clear(_Op, Operator):
    bl_idname = "arcflow.muscle_paint_clear"
    bl_label = "Clear Paint"
    bl_description = "Remove every painted region colour from the skin"

    def execute(self, ctx):
        ms = _need_skin(self, ctx)
        if ms is None:
            return {'CANCELLED'}
        write_ids(ms.skin, np.zeros(len(ms.skin.data.vertices)))
        write_colors(ms.skin, ms)
        refresh_painted_counts(ms.skin, ms)
        return {'FINISHED'}


class AF_OT_muscle_auto_seed(_Op, Operator):
    bl_idname = "arcflow.muscle_auto_seed"
    bl_label = "Auto-Seed From Skeleton"
    bl_description = "Lay out a rough starting paint from the bones and skin weights. Repaint and add contours to refine"

    overwrite: BoolProperty(name="Overwrite paint", default=False)

    def execute(self, ctx):
        ms = _need_skin(self, ctx)
        if ms is None:
            return {'CANCELLED'}
        try:
            n = run_auto_seed(ctx.scene, self.overwrite)
        except Exception as ex:
            self.report({'ERROR'}, str(ex))
            return {'CANCELLED'}
        _log(ctx.scene, "Auto-seed painted %d vertices from the skeleton." % n)
        return {'FINISHED'}


class AF_OT_muscle_show_colors(_Op, Operator):
    bl_idname = "arcflow.muscle_show_colors"
    bl_label = "Show Region Colours"
    bl_description = "Show the painted region colours (or the solved partition) in the viewport"

    solved: BoolProperty(name="Solved regions", default=False, description="Show the regions the solver projected instead of the raw paint")

    def execute(self, ctx):
        ms = _need_skin(self, ctx)
        if ms is None:
            return {'CANCELLED'}
        write_colors(ms.skin, ms)
        if self.solved:
            set_active_color(ms.skin, ATTR_SOLVED_COLOR)
        show_colors_in_viewports(ctx, True)
        return {'FINISHED'}


class AF_OT_muscle_paint(Operator):
    bl_idname = "arcflow.muscle_paint"
    bl_label = "Paint Regions"
    bl_description = ("Region pencil: drag on the skin to paint the active region colour. Ctrl or the Erase option erases. "
                      "[ and ] change the radius. Right click or Esc to finish")
    bl_options = {'REGISTER', 'UNDO', 'BLOCKING'}

    @classmethod
    def poll(cls, ctx):
        ms = getattr(ctx.scene, 'arc_flow_muscle', None)
        return ms is not None and ms.skin is not None and ctx.area is not None and ctx.area.type == 'VIEW_3D'

    def _hit_vertex(self, ctx, event):
        from bpy_extras import view3d_utils
        ms = _ms(ctx.scene)
        skin = ms.skin
        region, rv3d = ctx.region, ctx.region_data
        co = (event.mouse_region_x, event.mouse_region_y)
        origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, co)
        direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, co)
        inv = skin.matrix_world.inverted()
        o, d = inv @ origin, (inv.to_3x3() @ direction).normalized()
        ok, loc, nrm, fi = skin.ray_cast(o, d)
        if not ok:
            return None
        poly = skin.data.polygons[fi]
        v = basis_coords(skin)
        cand = np.array(poly.vertices)
        k = cand[np.argmin(np.linalg.norm(v[cand] - np.array(loc), axis=1))]
        return v[k]

    def modal(self, ctx, event):
        ms = _ms(ctx.scene)
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            ctx.window.cursor_modal_restore()
            _log(ctx.scene, "Region paint finished.")
            return {'FINISHED'}
        if event.type == 'LEFT_BRACKET' and event.value == 'PRESS':
            ms.brush_radius *= 0.85
        elif event.type == 'RIGHT_BRACKET' and event.value == 'PRESS':
            ms.brush_radius /= 0.85
        elif event.type == 'LEFTMOUSE':
            self._down = event.value == 'PRESS' or (event.value != 'RELEASE' and self._down)
            if event.value == 'RELEASE':
                self._down = False
        if event.type in {'MOUSEMOVE', 'LEFTMOUSE'} and self._down and ms.regions:
            p = self._hit_vertex(ctx, event)
            if p is not None:
                paint_vertices(ms.skin, ms, p, ms.brush_radius, ms.regions[ms.index].cid, ms.brush_erase or event.ctrl)
                refresh_painted_counts(ms.skin, ms)
                ctx.area.tag_redraw()
            return {'RUNNING_MODAL'}
        if event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'MIDDLEMOUSE'} or event.alt:
            return {'PASS_THROUGH'}
        return {'RUNNING_MODAL'}

    def invoke(self, ctx, event):
        self._down = False
        ms = _ms(ctx.scene)
        if not ms.regions:
            self.report({'ERROR'}, "Apply a preset first")
            return {'CANCELLED'}
        write_colors(ms.skin, ms)
        show_colors_in_viewports(ctx, True)
        ctx.window.cursor_modal_set('PAINT_BRUSH')
        ctx.window_manager.modal_handler_add(self)
        _log(ctx.scene, "Painting %s: drag on the skin, [ ] size, Ctrl erase, Esc done." % ms.regions[ms.index].label)
        return {'RUNNING_MODAL'}


class AF_OT_muscle_contour(AF_OT_muscle_paint):
    bl_idname = "arcflow.muscle_contour"
    bl_label = "Draw Contour"
    bl_description = ("Draw a few contour lines across the surface for the active region. The line is the fibre axis; "
                      "the solve puts the tendon ends just beyond its two ends. Right click or Esc to finish")

    def modal(self, ctx, event):
        ms = _ms(ctx.scene)
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            self._finish(ctx)
            ctx.window.cursor_modal_restore()
            return {'FINISHED'}
        if event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                self._pts = []
            elif event.value == 'RELEASE':
                self._finish(ctx)
            self._down = event.value == 'PRESS'
            return {'RUNNING_MODAL'}
        if event.type == 'MOUSEMOVE' and self._down:
            p = self._hit_vertex(ctx, event)
            if p is not None:
                v = basis_coords(ms.skin)
                k = int(np.argmin(np.linalg.norm(v - p, axis=1)))
                if not self._pts or self._pts[-1] != k:
                    self._pts.append(k)
            return {'RUNNING_MODAL'}
        if event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'MIDDLEMOUSE'} or event.alt:
            return {'PASS_THROUGH'}
        return {'RUNNING_MODAL'}

    def _finish(self, ctx):
        ms = _ms(ctx.scene)
        if len(getattr(self, '_pts', [])) >= 2:
            st = load_strokes(ms.skin)
            st.append(dict(cid=ms.regions[ms.index].cid, verts=[int(i) for i in self._pts]))
            save_strokes(ms.skin, st)
            update_contour_curve(ctx.scene, ms.skin)
            refresh_stroke_counts(ms.skin, ms)
            _log(ctx.scene, "Muscle Pencil line added to %s (%d lines). Press Complete when the muscle is marked."
                 % (ms.regions[ms.index].label, ms.regions[ms.index].n_strokes))
        self._pts = []

    def invoke(self, ctx, event):
        self._pts = []
        return super().invoke(ctx, event)


class AF_OT_muscle_contour_selection(_Op, Operator):
    bl_idname = "arcflow.muscle_contour_selection"
    bl_label = "Contour From Selection"
    bl_description = "Turn the selected edge chain (Edit Mode) into a contour stroke for the active region"

    def execute(self, ctx):
        ms = _need_skin(self, ctx)
        if ms is None or not ms.regions:
            return {'CANCELLED'}
        chain = selected_vertex_chain(ms.skin)
        if len(chain) < 2:
            self.report({'WARNING'}, "Select at least two connected vertices")
            return {'CANCELLED'}
        st = load_strokes(ms.skin)
        st.append(dict(cid=ms.regions[ms.index].cid, verts=[int(i) for i in chain]))
        save_strokes(ms.skin, st)
        update_contour_curve(ctx.scene, ms.skin)
        refresh_stroke_counts(ms.skin, ms)
        _log(ctx.scene, "Contour stroke added (%d strokes)." % len(st))
        return {'FINISHED'}


class AF_OT_muscle_contour_clear(_Op, Operator):
    bl_idname = "arcflow.muscle_contour_clear"
    bl_label = "Clear Contours"
    bl_description = "Remove every contour stroke"

    def execute(self, ctx):
        ms = _need_skin(self, ctx)
        if ms is None:
            return {'CANCELLED'}
        save_strokes(ms.skin, [])
        ob = bpy.data.objects.get(CONTOUR_OBJ)
        if ob is not None:
            bpy.data.objects.remove(ob, do_unlink=True)
        return {'FINISHED'}


class AF_OT_muscle_solve(_Op, Operator):
    bl_idname = "arcflow.muscle_solve"
    bl_label = "Solve Muscles"
    bl_description = "Project the painted regions over the whole skin and compute tendons, flex weights and displacement fields"

    def execute(self, ctx):
        ms = _need_skin(self, ctx)
        if ms is None:
            return {'CANCELLED'}
        try:
            run_solve(ctx.scene)
        except Exception as ex:
            self.report({'ERROR'}, str(ex))
            return {'CANCELLED'}
        self.report({'INFO'}, ms.status)
        return {'FINISHED'}


class AF_OT_muscle_bake(_Op, Operator):
    bl_idname = "arcflow.muscle_bake"
    bl_label = "Bake to Shape Keys"
    bl_description = "Create one shape key per solved muscle with the current slider values, driven by the bone rotation"

    def execute(self, ctx):
        ms = _need_skin(self, ctx)
        if ms is None:
            return {'CANCELLED'}
        try:
            bake_shape_keys(ctx.scene)
        except Exception as ex:
            self.report({'ERROR'}, str(ex))
            return {'CANCELLED'}
        self.report({'INFO'}, ms.status)
        return {'FINISHED'}


class AF_OT_muscle_gn(_Op, Operator):
    bl_idname = "arcflow.muscle_gn"
    bl_label = "Create Geometry Nodes"
    bl_description = "Add the Tessera Muscles Geometry Nodes group as a modifier on the skin (before the Armature modifier)"

    def execute(self, ctx):
        ms = _need_skin(self, ctx)
        if ms is None:
            return {'CANCELLED'}
        if not ms.solved:
            self.report({'ERROR'}, "Solve the muscles first")
            return {'CANCELLED'}
        from . import muscle_gn
        write_attributes(ctx.scene, get_instances(ctx.scene))
        muscle_gn.add_modifier(ms.skin)
        _log(ctx.scene, "Added the Tessera Muscles modifier. Use Flex, Accent and Region inputs; combine with other nodes.")
        return {'FINISHED'}


class AF_OT_muscle_export_json(_Op, Operator):
    bl_idname = "arcflow.muscle_export_json"
    bl_label = "Export Data (JSON Sidecar)"
    bl_description = "Write per-muscle flex weights, morph deltas and the control mapping for a game-engine runtime"

    def execute(self, ctx):
        ms = _need_skin(self, ctx)
        if ms is None:
            return {'CANCELLED'}
        if not get_instances(ctx.scene):
            self.report({'ERROR'}, "Solve the muscles first")
            return {'CANCELLED'}
        base = bpy.path.abspath(ms.export_path)
        path = base if base.lower().endswith('.json') else base + '.muscles.json'
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        data = write_sidecar(ctx.scene, path)
        _log(ctx.scene, "Sidecar written: %d muscles to %s" % (len(data['muscles']), os.path.basename(path)))
        return {'FINISHED'}


class AF_OT_muscle_export_game(_Op, Operator):
    bl_idname = "arcflow.muscle_export_game"
    bl_label = "Export GLB / FBX"
    bl_description = "Export the skin and skeleton with shape keys, skin weights and region colours"

    def execute(self, ctx):
        ms = _need_skin(self, ctx)
        if ms is None:
            return {'CANCELLED'}
        try:
            path = export_game(ctx.scene, ms.export_path, ms.export_format)
        except Exception as ex:
            self.report({'ERROR'}, "Export failed: %s" % ex)
            return {'CANCELLED'}
        _log(ctx.scene, "Exported %s" % os.path.basename(path))
        return {'FINISHED'}


class AF_OT_muscle_control_rig(_Op, Operator):
    bl_idname = "arcflow.muscle_control_rig"
    bl_label = "Build Control Rig"
    bl_description = "Create circle controllers that drive the deform bones (FK) and IK handles for forearms and shins"

    ik: BoolProperty(name="IK handles", default=True)

    def execute(self, ctx):
        ms = _need_skin(self, ctx)
        if ms is None:
            return {'CANCELLED'}
        try:
            c, k = build_control_rig(ctx.scene, self.ik)
        except Exception as ex:
            self.report({'ERROR'}, str(ex))
            return {'CANCELLED'}
        _log(ctx.scene, "Control rig: %d FK controllers, %d IK handles." % (len(c), len(k)))
        return {'FINISHED'}


class AF_OT_muscle_complete(_Op, Operator):
    bl_idname = "arcflow.muscle_complete"
    bl_label = "Complete Muscle"
    bl_description = ("Book the active muscle into its region card's list once it is marked (a few Muscle Pencil lines, "
                      "paint, or muscle / bone geometry). The next unmarked muscle of the card becomes active")

    index: IntProperty(name="Muscle", default=-1, options={'SKIP_SAVE'})

    def execute(self, ctx):
        ms = _need_skin(self, ctx)
        if ms is None or not ms.regions:
            return {'CANCELLED'}
        try:
            complete_muscle(ctx.scene, None if self.index < 0 else self.index)
        except RuntimeError as ex:
            self.report({'WARNING'}, str(ex))
            return {'CANCELLED'}
        self.report({'INFO'}, ms.status)
        return {'FINISHED'}


class AF_OT_muscle_select(_Op, Operator):
    bl_idname = "arcflow.muscle_select"
    bl_label = "Make Active"
    bl_description = "Make this muscle the active one for the Muscle Pencil, paint and Complete"
    bl_options = {'INTERNAL'}

    index: IntProperty(default=0, options={'SKIP_SAVE'})

    def execute(self, ctx):
        ms = _ms(ctx.scene)
        if 0 <= self.index < len(ms.regions):
            ms.index = self.index
        return {'FINISHED'}


class AF_OT_muscle_card_add(_Op, Operator):
    bl_idname = "arcflow.muscle_card_add"
    bl_label = "Add Muscle to Card"
    bl_description = "Add the chosen catalog muscle of this body region and make it active for the Muscle Pencil"

    card: StringProperty(default="", options={'SKIP_SAVE'})

    def execute(self, ctx):
        ms = _ms(ctx.scene)
        ensure_cards(ms)
        c = next((c for c in ms.cards if c.key == self.card), None)
        if c is None or c.pick in ('', 'NONE'):
            return {'CANCELLED'}
        ms.index = add_catalog_region(ms, c.pick)
        return {'FINISHED'}


class AF_OT_muscle_cards_init(_Op, Operator):
    bl_idname = "arcflow.muscle_cards_init"
    bl_label = "Create Region Cards"
    bl_description = "Create the region category cards (Head and face, Arms and hands, Torso and back, Legs and feet, Tail)"

    def execute(self, ctx):
        ensure_cards(_ms(ctx.scene))
        return {'FINISHED'}


CLASSES = (AF_OT_muscle_complete, AF_OT_muscle_select, AF_OT_muscle_card_add, AF_OT_muscle_cards_init, AF_OT_muscle_assign, AF_OT_muscle_apply_preset, AF_OT_muscle_region_add, AF_OT_muscle_region_remove,
           AF_OT_muscle_example, AF_OT_muscle_paint_selected, AF_OT_muscle_paint_clear, AF_OT_muscle_auto_seed,
           AF_OT_muscle_show_colors, AF_OT_muscle_paint, AF_OT_muscle_contour, AF_OT_muscle_contour_selection,
           AF_OT_muscle_contour_clear, AF_OT_muscle_solve, AF_OT_muscle_bake, AF_OT_muscle_gn,
           AF_OT_muscle_export_json, AF_OT_muscle_export_game, AF_OT_muscle_control_rig)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
