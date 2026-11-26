"""Tessera Projection Cloth Trim, Accents and Seam Seal Control Solving: the Blender bridge.

The math lives in core/cloth_trim.py.  This module (a) builds the core objects for every garment
from the scene properties and plugs them into the cloth solver's ``extra_constraints`` hook,
(b) writes strips, accent instances, hanging chains and the zipper pull after every cloth step,
and (c) provides the operators and the jacket-zipper example.

The cloth bridge is extended by wrapping ``cloth_ops.build_solver`` and ``cloth_ops.step_scene``
at registration time (restored on unregister), so the shared cloth files only gain registration lines.
"""
import math

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty
from bpy.types import Operator

from . import cloth_ops as CO
from .core import cloth_trim as CT

ACCENT_COLLECTION = "Tessera Accents"
_ORIG = {}


def trims(scene):
    return scene.arc_flow_cloth_trim


# ----------------------------------------------------------------------------- helpers
def _garment_for(scene, name):
    cl = CO.cloth(scene)
    if name:
        for g in cl.garments:
            if g.name == name:
                return g
    if cl.garments:
        return cl.garments[max(0, min(int(cl.index), len(cl.garments) - 1))]
    return None


def _belongs(scene, item, g):
    gg = _garment_for(scene, item.garment)
    return gg is not None and gg.name == g.name


def _group_indices(obj, name, thr=0.5):
    if not name or obj.vertex_groups.get(name) is None:
        return np.zeros(0, np.int64), np.zeros(len(obj.data.vertices))
    w = CO._pin_weights(obj, name)
    return np.nonzero(w >= thr)[0], w


def _sig(item, keys):
    return tuple(getattr(item, k) if not hasattr(getattr(item, k), 'name') else getattr(item, k).name for k in keys)


TRIM_KEYS = ('kind', 'source', 'loop_index', 'group', 'width', 'thickness', 'mass_per_m', 'stiffness', 'bend',
             'edge_weight', 'flare', 'flare_count', 'enabled')
ACCENT_KEYS = ('kind', 'placement', 'group', 'count', 'density', 'seed', 'size', 'offset', 'align', 'dynamic',
               'chain_segments', 'chain_length', 'chain_stiffness', 'chain_damping', 'collide', 'collide_radius',
               'enabled')
SEAM_STRUCT_KEYS = ('mode', 'group', 'group_a', 'group_b', 'reverse', 'enabled')


def _structure_sig(scene, g, obj):
    ts = trims(scene)
    return (tuple((t.name,) + _sig(t, TRIM_KEYS) for t in ts.trims if _belongs(scene, t, g)),
            tuple((a.name,) + _sig(a, ACCENT_KEYS) + (tuple(scene.cursor.location) if a.placement == 'CURSOR' else (),)
                  for a in ts.accents if _belongs(scene, a, g)),
            tuple((s.name,) + _sig(s, SEAM_STRUCT_KEYS) for s in ts.seams if _belongs(scene, s, g)),
            len(obj.data.vertices))


# ----------------------------------------------------------------------------- build / attach
class _Bound:
    """Everything attached to one solver."""

    def __init__(self):
        self.sig = None
        self.trims = {}     # name -> CT.Trim
        self.accents = {}   # name -> CT.AccentSet
        self.seams = {}     # name -> CT.SeamSeal
        self.objects = []


def _build_trim(S, obj, t):
    if t.source in ('LOWEST', 'HIGHEST'):
        chain, closed = CT.edge_band_chain(S.rest, S.tris, S.stretch_edges, t.source)
    elif t.source == 'BOUNDARY':
        ch = CT.boundary_chains(S.tris)
        if not ch:
            return None
        chain, closed = ch[min(int(t.loop_index), len(ch) - 1)]
    else:
        idx, _w = _group_indices(obj, t.group)
        if len(idx) < 3:
            return None
        chain, closed = CT.order_chain(S.rest, idx, S.stretch_edges)
    if len(chain) < 3:
        return None
    tr = CT.Trim(chain, closed, kind=t.kind, width=t.width, thickness=t.thickness, mass_per_m=t.mass_per_m,
                 stiffness=t.stiffness, bend=t.bend, edge_weight=t.edge_weight, flare=t.flare,
                 flare_count=t.flare_count, name=t.name)
    tr.enabled = bool(t.enabled)
    return CT.attach(S, tr)


def _accent_points(scene, S, obj, a):
    idx, w = _group_indices(obj, a.group)
    if a.placement == 'CURSOR':
        loc = np.array(scene.cursor.location, float)
        return loc.reshape(1, 3)
    if a.placement == 'CURVE':
        if len(idx) < 2:
            return np.zeros((0, 3))
        chain, closed = CT.order_chain(S.rest, idx, S.stretch_edges)
        return CT.points_along_chain(S.rest, chain, a.count, closed)
    if a.placement == 'VERTICES':
        return S.rest[idx] if len(idx) else np.zeros((0, 3))
    weights = w if (obj.vertex_groups.get(a.group) is not None) else None
    return CT.scatter_points(S.rest, S.tris, count=a.count, density=a.density if a.density > 0 else None,
                             seed=a.seed, weights=weights)


def _build_accent(scene, S, obj, a):
    pts = _accent_points(scene, S, obj, a)
    if len(pts) == 0:
        return None
    segs = int(a.chain_segments) if a.dynamic else 0
    ac = CT.AccentSet(S.rest, S.tris, pts, kind=a.kind, offset=a.offset, scale=1.0, align=a.align,
                      chain_segments=segs, chain_length=a.chain_length if a.dynamic else 0.0,
                      chain_stiffness=a.chain_stiffness, chain_damping=a.chain_damping,
                      collide=a.collide, collide_radius=a.collide_radius, seed=a.seed)
    ac.enabled = bool(a.enabled)
    return CT.attach(S, ac)


def _build_seam(S, obj, sm):
    ia, _ = _group_indices(obj, sm.group_a)
    ib, _ = _group_indices(obj, sm.group_b)
    idx, _ = _group_indices(obj, sm.group)
    if len(ia) and len(ib):
        pairs = CT.pair_seam(S.rest, S.stretch_edges, np.concatenate([ia, ib]), ia, ib)
    else:
        pairs = CT.pair_seam(S.rest, S.stretch_edges, idx)
    if len(pairs) == 0:
        return None
    seal = CT.SeamSeal(S.rest, pairs, mode=sm.mode, progress=sm.progress, strength=sm.strength,
                       tear=sm.tear_gap if sm.use_tear else 0.0, gather=sm.gather, ruffle=sm.ruffle,
                       reverse=sm.reverse)
    return CT.attach(S, seal)


def sync(scene, g, S):
    """Make the solver ``S`` of garment ``g`` carry exactly the trims, accents and seams in the scene."""
    if g.obj is None:
        return None
    ts = trims(scene)
    sig = _structure_sig(scene, g, g.obj)
    B = getattr(S, '_trim_bound', None)
    if B is None or B.sig != sig or getattr(S, '_trim_n', None) != S.n:
        if B is not None:
            for o in list(B.trims.values()) + list(B.accents.values()) + list(B.seams.values()):
                CT.detach(S, o)
        B = _Bound()
        B.sig = sig
        for t in ts.trims:
            if _belongs(scene, t, g) and t.enabled:
                o = _build_trim(S, g.obj, t)
                if o is not None:
                    B.trims[t.name] = o
                    t.n_verts = len(o.chain)
        for a in ts.accents:
            if _belongs(scene, a, g) and a.enabled:
                o = _build_accent(scene, S, g.obj, a)
                if o is not None:
                    B.accents[a.name] = o
                    a.n_instances = o.K
        for sm in ts.seams:
            if _belongs(scene, sm, g) and sm.enabled:
                o = _build_seam(S, g.obj, sm)
                if o is not None:
                    B.seams[sm.name] = o
                    sm.n_pairs = len(o.pairs)
        S._trim_bound = B
        S._trim_n = S.n
    # live parameters: the slider, seal strength, tear limit and so on apply without a rebuild
    for sm in ts.seams:
        o = B.seams.get(sm.name)
        if o is None:
            continue
        if sm.drive_from_pull and sm.pull is not None:
            P = np.array(sm.pull.matrix_world.translation, float)
            val = o.progress_from_point(P, S.x)
            if abs(val - sm.progress) > 1e-4:
                sm.progress = float(val)
        o.progress = float(sm.progress)
        o.strength = float(sm.strength)
        o.tear = float(sm.tear_gap) if sm.use_tear else 0.0
        o.gather = float(sm.gather)
        o.ruffle = float(sm.ruffle)
        sm.n_torn = int(o.torn.sum())
    for t in ts.trims:
        o = B.trims.get(t.name)
        if o is not None:
            o.enabled = bool(t.enabled)
    return B


# ----------------------------------------------------------------------------- scene objects
def _accent_collection(scene):
    c = bpy.data.collections.get(ACCENT_COLLECTION)
    if c is None:
        c = bpy.data.collections.new(ACCENT_COLLECTION)
    if c.name not in scene.collection.children:
        scene.collection.children.link(c)
    return c


def _default_accent_mesh(kind):
    import bmesh
    me = bpy.data.meshes.get("Tessera Accent " + kind)
    if me is not None:
        return me
    me = bpy.data.meshes.new("Tessera Accent " + kind)
    bm = bmesh.new()
    if kind in ('BUTTON', 'SEQUIN'):
        bmesh.ops.create_cone(bm, cap_ends=True, segments=20, radius1=0.5, radius2=0.45, depth=0.28)
    elif kind in ('TASSEL', 'CHAIN'):
        bmesh.ops.create_cone(bm, cap_ends=True, segments=12, radius1=0.35, radius2=0.12, depth=1.0)
    elif kind == 'BOW':
        bmesh.ops.create_icosphere(bm, subdivisions=2, radius=0.5)
        for v in bm.verts:
            v.co.x *= 1.0
            v.co.y *= 0.55
            v.co.z *= 0.32
    else:
        bmesh.ops.create_icosphere(bm, subdivisions=2 if kind != 'JEWEL' else 1, radius=0.5)
    bm.to_mesh(me)
    bm.free()
    me.update()
    return me


def _src_factor(a):
    if a.source is not None and a.source.type == 'MESH':
        d = max(max(a.source.dimensions), 1e-6)
        return float(a.size) / d
    return float(a.size)


def _shape_mesh(a):
    """Per-accent-set mesh (a copy of the source or the built-in shape) so materials stay local."""
    src = a.source.data if (a.source is not None and a.source.type == 'MESH') else _default_accent_mesh(a.kind)
    key = "%s|%s" % (src.name, a.kind)
    nm = "%s Shape" % a.name
    me = bpy.data.meshes.get(nm)
    if me is not None and me.get("tessera_shape") != key:
        for o in bpy.data.objects:
            if o.data is me:
                o.data = src
        bpy.data.meshes.remove(me)
        me = None
    if me is None:
        me = src.copy()
        me.name = nm
        me["tessera_shape"] = key
    if a.material is not None and (len(me.materials) == 0 or me.materials[0] != a.material):
        me.materials.clear()
        me.materials.append(a.material)
    return me


def _ensure_instances(scene, a, K):
    coll = _accent_collection(scene)
    obs = [o for o in coll.objects if o.get("tessera_accent") == a.name]
    obs.sort(key=lambda o: int(o.get("tessera_index", 0)))
    me = _shape_mesh(a)
    while len(obs) > K:
        bpy.data.objects.remove(obs.pop(), do_unlink=True)
    for ob in obs:
        if ob.data is not me:
            ob.data = me
    for k in range(len(obs), K):
        o = bpy.data.objects.new("%s %d" % (a.name, k), me)
        o["tessera_accent"] = a.name
        o["tessera_index"] = k
        coll.objects.link(o)
        obs.append(o)
    return obs


def _line_object(scene, name, V, E, thickness, roots=()):
    """Polyline object (chains) drawn with a Skin modifier so loose edges get thickness."""
    import bmesh
    ob = bpy.data.objects.get(name)
    fresh = False
    if ob is None:
        me = bpy.data.meshes.new(name)
        ob = bpy.data.objects.new(name, me)
        _accent_collection(scene).objects.link(ob)
        sk = ob.modifiers.new("Skin", 'SKIN')
        sk.use_smooth_shade = True
        fresh = True
    me = ob.data
    if fresh or len(me.vertices) != len(V) or len(me.edges) != len(E):
        me.clear_geometry()
        me.from_pydata(V.tolist(), [tuple(map(int, e)) for e in E], [])
        bm = bmesh.new()
        bm.from_mesh(me)
        lay = bm.verts.layers.skin.verify()
        for v in bm.verts:
            v[lay].radius = (float(thickness), float(thickness))
            v[lay].use_root = v.index in set(roots)
        bm.to_mesh(me)
        bm.free()
    else:
        me.vertices.foreach_set('co', V.astype(np.float32).ravel())
    me.update()
    return ob


def _strip_object(scene, t, verts, faces):
    nm = "Trim " + t.name
    ob = t.strip_obj if (t.strip_obj is not None and t.strip_obj.name in bpy.data.objects) else bpy.data.objects.get(nm)
    if ob is None:
        me = bpy.data.meshes.new(nm)
        me.from_pydata(verts.tolist(), [], [tuple(map(int, f)) for f in faces])
        me.update()
        ob = bpy.data.objects.new(nm, me)
        _accent_collection(scene).objects.link(ob)
        ob.modifiers.new("Thickness", 'SOLIDIFY')
        t.strip_obj = ob
    me = ob.data
    if len(me.vertices) != len(verts):
        me.clear_geometry()
        me.from_pydata(verts.tolist(), [], [tuple(map(int, f)) for f in faces])
    else:
        me.vertices.foreach_set('co', verts.astype(np.float32).ravel())
    me.update()
    m = ob.modifiers.get("Thickness")
    if m is not None:
        m.thickness = float(t.thickness)
        m.offset = 1.0
    if t.material is not None and (len(me.materials) == 0 or me.materials[0] != t.material):
        me.materials.clear()
        me.materials.append(t.material)
    return ob


def refresh_objects(scene, g, S):
    """Write strips, accent instances, chains and the zipper pull for garment ``g`` from solver state."""
    B = getattr(S, '_trim_bound', None)
    if B is None:
        return
    from mathutils import Matrix
    ts = trims(scene)
    for t in ts.trims:
        o = B.trims.get(t.name)
        if o is not None and t.show_strip and t.enabled:
            v, f = o.strip(S.x, S.tris, lift=0.5 * float(t.thickness))
            _strip_object(scene, t, v, f)
    for a in ts.accents:
        o = B.accents.get(a.name)
        if o is None or not a.enabled:
            continue
        obs = _ensure_instances(scene, a, o.K)
        M = o.transforms(S.x)
        fac = _src_factor(a)
        M[:, :3, :3] /= max(o.scale, 1e-12)
        M[:, :3, :3] *= fac
        for k, ob in enumerate(obs):
            ob.matrix_world = Matrix(M[k].tolist())
        if o.dynamic and a.show_chain:
            C = o.chains()
            V = C.reshape(-1, 3)
            m = C.shape[1]
            E = [(k * m + j, k * m + j + 1) for k in range(o.K) for j in range(m - 1)]
            _line_object(scene, "%s Chains" % a.name, V, E, max(a.collide_radius * 0.5, 0.0015), roots=[k * m for k in range(o.K)])
    for sm in ts.seams:
        o = B.seams.get(sm.name)
        if o is None:
            continue
        if sm.pull is not None and not sm.drive_from_pull:
            sm.pull.location = tuple(float(c) for c in o.head_position(S.x))


# ----------------------------------------------------------------------------- bridge wrappers
def _wrap_build(orig):
    def build_solver(scene, g):
        S = orig(scene, g)
        try:
            sync(scene, g, S)
        except Exception as ex:  # never break the cloth solver
            print("Tessera cloth trim sync:", ex)
        return S
    build_solver._tessera_trim = True
    return build_solver


def _wrap_step(orig):
    def step_scene(scene, dt, deps=None):
        ts = getattr(scene, 'arc_flow_cloth_trim', None)
        active = ts is not None and (len(ts.trims) or len(ts.accents) or len(ts.seams))
        if active:
            for g in CO.cloth(scene).garments:
                S = CO.RT.solvers.get(g.name)
                if S is not None and g.enabled and g.obj is not None:
                    try:
                        sync(scene, g, S)
                    except Exception as ex:
                        print("Tessera cloth trim sync:", ex)
        n = orig(scene, dt, deps)
        if active:
            for g in CO.cloth(scene).garments:
                S = CO.RT.solvers.get(g.name)
                if S is not None and g.enabled and g.obj is not None:
                    try:
                        refresh_objects(scene, g, S)
                    except Exception as ex:
                        print("Tessera cloth trim objects:", ex)
        return n
    step_scene._tessera_trim = True
    return step_scene


def _install():
    if not getattr(CO.build_solver, '_tessera_trim', False):
        _ORIG['build'] = CO.build_solver
        CO.build_solver = _wrap_build(CO.build_solver)
    if not getattr(CO.step_scene, '_tessera_trim', False):
        _ORIG['step'] = CO.step_scene
        CO.step_scene = _wrap_step(CO.step_scene)


def _uninstall():
    if 'build' in _ORIG:
        CO.build_solver = _ORIG.pop('build')
    if 'step' in _ORIG:
        CO.step_scene = _ORIG.pop('step')


def rebuild(scene):
    """Rebuild solvers' trims now (also draws the objects once at the current state)."""
    for g in CO.cloth(scene).garments:
        if g.enabled and g.obj is not None:
            S = CO.get_solver(scene, g)
            sync(scene, g, S)
            refresh_objects(scene, g, S)


# ----------------------------------------------------------------------------- operators
class _TOp:
    @classmethod
    def poll(cls, ctx):
        return getattr(ctx.scene, 'arc_flow_cloth_trim', None) is not None


def _unique(coll, base):
    names = {x.name for x in coll}
    nm, k = base, 1
    while nm in names:
        k += 1
        nm = "%s.%d" % (base, k)
    return nm


def _active_garment_name(scene):
    g = _garment_for(scene, "")
    return g.name if g is not None else ""


class ARCFLOW_OT_trim_add(_TOp, Operator):
    bl_idname = "arcflow.trim_add"
    bl_label = "Add Trim"
    bl_description = "Add a hem, piping, lace border, belt or collar band to the active garment"
    bl_options = {'REGISTER', 'UNDO'}
    kind: EnumProperty(name="Kind", items=[(k, v['label'], v['label']) for k, v in CT.TRIM_KINDS.items()], default='HEM')

    def execute(self, ctx):
        ts = trims(ctx.scene)
        t = ts.trims.add()
        pr = CT.TRIM_KINDS[self.kind]
        t.name = _unique([x for x in ts.trims if x != t], pr['label'].split(' / ')[0])
        t.kind = self.kind
        for k in ('width', 'thickness', 'mass_per_m', 'stiffness', 'bend', 'edge_weight', 'flare', 'flare_count'):
            setattr(t, k, pr[k])
        t.source = 'HIGHEST' if self.kind in ('BAND', 'COLLAR') else 'LOWEST'
        t.garment = _active_garment_name(ctx.scene)
        ts.trim_index = len(ts.trims) - 1
        return {'FINISHED'}


class ARCFLOW_OT_trim_remove(_TOp, Operator):
    bl_idname = "arcflow.trim_remove"
    bl_label = "Remove Trim"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        ts = trims(ctx.scene)
        if 0 <= ts.trim_index < len(ts.trims):
            t = ts.trims[ts.trim_index]
            if t.strip_obj is not None:
                bpy.data.objects.remove(t.strip_obj, do_unlink=True)
            ts.trims.remove(ts.trim_index)
            ts.trim_index = max(0, ts.trim_index - 1)
        for g in CO.cloth(ctx.scene).garments:
            S = CO.RT.solvers.get(g.name)
            if S is not None:
                sync(ctx.scene, g, S)
        return {'FINISHED'}


class ARCFLOW_OT_accent_add(_TOp, Operator):
    bl_idname = "arcflow.accent_add"
    bl_label = "Add Accents"
    bl_description = "Add buttons, bows, jewels, sequins, chains, charms, tassels or brooches bound to the active garment"
    bl_options = {'REGISTER', 'UNDO'}
    kind: EnumProperty(name="Kind", items=[(k, v['label'], v['label']) for k, v in CT.ACCENT_KINDS.items()], default='BUTTON')

    def execute(self, ctx):
        ts = trims(ctx.scene)
        a = ts.accents.add()
        pr = CT.ACCENT_KINDS[self.kind]
        a.name = _unique([x for x in ts.accents if x != a], pr['label'].split(' / ')[0] + "s")
        a.kind = self.kind
        a.size = pr['scale']
        a.dynamic = bool(pr['dynamic'])
        a.garment = _active_garment_name(ctx.scene)
        if self.kind in ('SEQUIN', 'JEWEL'):
            a.placement = 'DENSITY'
            a.count = 60
        ts.accent_index = len(ts.accents) - 1
        return {'FINISHED'}


class ARCFLOW_OT_accent_remove(_TOp, Operator):
    bl_idname = "arcflow.accent_remove"
    bl_label = "Remove Accents"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        ts = trims(ctx.scene)
        if 0 <= ts.accent_index < len(ts.accents):
            a = ts.accents[ts.accent_index]
            for o in list(bpy.data.objects):
                if o.get("tessera_accent") == a.name:
                    bpy.data.objects.remove(o, do_unlink=True)
            ch = bpy.data.objects.get("%s Chains" % a.name)
            if ch is not None:
                bpy.data.objects.remove(ch, do_unlink=True)
            ts.accents.remove(ts.accent_index)
            ts.accent_index = max(0, ts.accent_index - 1)
        for g in CO.cloth(ctx.scene).garments:
            S = CO.RT.solvers.get(g.name)
            if S is not None:
                sync(ctx.scene, g, S)
        return {'FINISHED'}


class ARCFLOW_OT_accent_use_selected(_TOp, Operator):
    bl_idname = "arcflow.accent_use_selected"
    bl_label = "Use Selected Object"
    bl_description = "Instance the selected mesh object on the cloth for the active accent set"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, ctx):
        ts = getattr(ctx.scene, 'arc_flow_cloth_trim', None)
        return ts is not None and len(ts.accents) > 0 and ctx.active_object is not None and ctx.active_object.type == 'MESH'

    def execute(self, ctx):
        ts = trims(ctx.scene)
        a = ts.accents[min(ts.accent_index, len(ts.accents) - 1)]
        a.source = ctx.active_object
        a.size = max(max(ctx.active_object.dimensions), 0.001)
        return {'FINISHED'}


class ARCFLOW_OT_seam_add(_TOp, Operator):
    bl_idname = "arcflow.seam_add"
    bl_label = "Add Seam Seal"
    bl_description = "Add a zipper, button placket, Velcro, magnetic strip or laces on a seam vertex group"
    bl_options = {'REGISTER', 'UNDO'}
    mode: EnumProperty(name="Closure", items=[(k, v['label'], v['label']) for k, v in CT.SEAM_MODES.items()], default='ZIPPER')

    def execute(self, ctx):
        ts = trims(ctx.scene)
        s = ts.seams.add()
        pr = CT.SEAM_MODES[self.mode]
        s.name = _unique([x for x in ts.seams if x != s], pr['label'])
        s.mode = self.mode
        s.gather = pr['gather']
        s.ruffle = pr['ruffle']
        s.garment = _active_garment_name(ctx.scene)
        ts.seam_index = len(ts.seams) - 1
        return {'FINISHED'}


class ARCFLOW_OT_seam_remove(_TOp, Operator):
    bl_idname = "arcflow.seam_remove"
    bl_label = "Remove Seam Seal"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        ts = trims(ctx.scene)
        if 0 <= ts.seam_index < len(ts.seams):
            ts.seams.remove(ts.seam_index)
            ts.seam_index = max(0, ts.seam_index - 1)
        for g in CO.cloth(ctx.scene).garments:
            S = CO.RT.solvers.get(g.name)
            if S is not None:
                sync(ctx.scene, g, S)
        return {'FINISHED'}


class ARCFLOW_OT_seam_pull_add(_TOp, Operator):
    bl_idname = "arcflow.seam_pull_add"
    bl_label = "Add Zipper Pull"
    bl_description = "Create an empty that rides the zipper head (tick Pull drives slider to zip by moving it)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, ctx):
        ts = getattr(ctx.scene, 'arc_flow_cloth_trim', None)
        return ts is not None and len(ts.seams) > 0

    def execute(self, ctx):
        ts = trims(ctx.scene)
        s = ts.seams[min(ts.seam_index, len(ts.seams) - 1)]
        o = bpy.data.objects.new("Zipper Pull", None)
        o.empty_display_type = 'ARROWS'
        o.empty_display_size = 0.05
        ctx.scene.collection.objects.link(o)
        g = _garment_for(ctx.scene, s.garment)
        S = CO.RT.solvers.get(g.name) if g is not None else None
        B = getattr(S, '_trim_bound', None) if S is not None else None
        seal = B.seams.get(s.name) if B is not None else None
        if seal is not None:
            o.location = tuple(float(c) for c in seal.head_position(S.x))
        s.pull = o
        return {'FINISHED'}


class ARCFLOW_OT_seam_set(_TOp, Operator):
    bl_idname = "arcflow.seam_set"
    bl_label = "Open / Close Seam"
    bl_options = {'REGISTER', 'UNDO'}
    value: FloatProperty(default=1.0, min=0.0, max=1.0)

    @classmethod
    def poll(cls, ctx):
        ts = getattr(ctx.scene, 'arc_flow_cloth_trim', None)
        return ts is not None and len(ts.seams) > 0

    def execute(self, ctx):
        ts = trims(ctx.scene)
        ts.seams[min(ts.seam_index, len(ts.seams) - 1)].progress = self.value
        return {'FINISHED'}


class ARCFLOW_OT_seam_animate(_TOp, Operator):
    bl_idname = "arcflow.seam_animate"
    bl_label = "Key Zip Animation"
    bl_description = "Keyframe the closed value from 0 to 1 over a frame range"
    bl_options = {'REGISTER', 'UNDO'}
    frame_from: IntProperty(name="From frame", default=-1)
    frame_to: IntProperty(name="To frame", default=-1)

    @classmethod
    def poll(cls, ctx):
        ts = getattr(ctx.scene, 'arc_flow_cloth_trim', None)
        return ts is not None and len(ts.seams) > 0

    def execute(self, ctx):
        sc = ctx.scene
        ts = trims(sc)
        s = ts.seams[min(ts.seam_index, len(ts.seams) - 1)]
        f0 = self.frame_from if self.frame_from >= 0 else sc.frame_start + 6
        f1 = self.frame_to if self.frame_to >= 0 else max(f0 + 1, sc.frame_start + 54)
        s.progress = 0.0
        s.keyframe_insert('progress', frame=f0)
        s.progress = 1.0
        s.keyframe_insert('progress', frame=f1)
        return {'FINISHED'}


class ARCFLOW_OT_trim_group_from_selection(Operator):
    bl_idname = "arcflow.trim_group_from_selection"
    bl_label = "Selected Vertices to Group"
    bl_description = "Store the selected vertices of the active mesh in a vertex group (edge, seam or accent selection)"
    bl_options = {'REGISTER', 'UNDO'}
    group: StringProperty(name="Group", default="Seam")

    @classmethod
    def poll(cls, ctx):
        return ctx.active_object is not None and ctx.active_object.type == 'MESH'

    def execute(self, ctx):
        ob = ctx.active_object
        if ob.mode == 'EDIT':
            ob.update_from_editmode()
        idx = [v.index for v in ob.data.vertices if v.select]
        if not idx:
            self.report({'WARNING'}, "No vertices are selected")
            return {'CANCELLED'}
        vg = ob.vertex_groups.get(self.group) or ob.vertex_groups.new(name=self.group)
        if ob.mode == 'EDIT':
            bpy.ops.object.mode_set(mode='OBJECT')
            vg.add(idx, 1.0, 'REPLACE')
            bpy.ops.object.mode_set(mode='EDIT')
        else:
            vg.add(idx, 1.0, 'REPLACE')
        self.report({'INFO'}, "%d vertices stored in group %s" % (len(idx), self.group))
        return {'FINISHED'}


class ARCFLOW_OT_trim_refresh(_TOp, Operator):
    bl_idname = "arcflow.trim_refresh"
    bl_label = "Apply Trim Setup"
    bl_description = "Rebuild the trims, accents and seams on the running cloth solver and redraw their objects"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        rebuild(ctx.scene)
        return {'FINISHED'}


# ----------------------------------------------------------------------------- example: jacket with zipper
def _jacket_mesh(body_z0, body_h, rx, ry, nz=26, nth=56, gap=0.20, ease=0.014, z_lo=0.05, z_hi=0.95):
    """Open tube around the torso (slit at the front, -Y) as vertices, faces and the row/column ids."""
    def prof(t):
        s = 0.86 + 0.18 * math.sin(math.pi * min(1.0, t * 1.1)) + 0.10 * max(0.0, t - 0.75) / 0.25
        return s
    V = []
    for j in range(nz):
        t = z_lo + (z_hi - z_lo) * j / (nz - 1)
        z = body_z0 + body_h * t
        s = prof(t)
        for i in range(nth):
            th = gap + (2 * math.pi - 2 * gap) * i / (nth - 1)
            phi = -math.pi / 2 + th
            V.append(((rx * s + ease) * math.cos(phi), (ry * s + ease) * math.sin(phi), z))
    F = []
    for j in range(nz - 1):
        for i in range(nth - 1):
            a = j * nth + i
            F.append((a, a + 1, a + 1 + nth, a + nth))
    return V, F, nz, nth


def load_jacket_example(scene):
    """Mannequin torso with a slit jacket: zipper seam, hem and collar trims, buttons, jewels, hanging charms."""
    coll = bpy.data.collections.get("Tessera Cloth Trim Example") or bpy.data.collections.new("Tessera Cloth Trim Example")
    if coll.name not in scene.collection.children:
        scene.collection.children.link(coll)
    for o in list(coll.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    ac = bpy.data.collections.get(ACCENT_COLLECTION)
    if ac is not None:
        for o in list(ac.objects):
            bpy.data.objects.remove(o, do_unlink=True)
    cl = CO.cloth(scene)
    cl.garments.clear()
    cl.colliders.clear()
    ts = trims(scene)
    ts.trims.clear()
    ts.accents.clear()
    ts.seams.clear()
    CO.RT.reset()

    z0, h = 0.70, 0.80
    rx, ry = 0.17, 0.11
    tv, tf = CO._capsule_torso(rx=rx, ry=ry, h=h, n=40, rings=18)
    torso = CO._new_obj(scene, "Jacket Mannequin Torso", tv, tf, coll)
    torso.location = (0.0, 0.0, z0)
    hv, hf = CO._uv_sphere(0.105, 32, 16)
    head = CO._new_obj(scene, "Jacket Mannequin Head", hv, hf, coll)
    head.location = (0.0, 0.0, z0 + h + 0.20)
    nv, nf = CO._capsule_torso(rx=0.05, ry=0.05, h=0.14, n=20, rings=4)
    neck = CO._new_obj(scene, "Jacket Mannequin Neck", nv, nf, coll)
    neck.location = (0.0, 0.0, z0 + h - 0.01)
    for o in (torso, head, neck):
        for p in o.data.polygons:
            p.use_smooth = True
    JV, JF, nz, nth = _jacket_mesh(z0, h, rx, ry)
    jacket = CO._new_obj(scene, "Jacket", JV, JF, coll)
    for p in jacket.data.polygons:
        p.use_smooth = True
    # vertex groups: pins (shoulder row, away from the seam), seam lips, chest buttons, charms, hem sequin weights
    def vid(i, j):
        return j * nth + i
    vg = jacket.vertex_groups
    pins = [vid(i, nz - 1) for i in range(3, nth - 3)]
    vg.new(name="ClothPins").add(pins, 1.0, 'REPLACE')
    vg.new(name="Seam").add([vid(0, j) for j in range(nz)] + [vid(nth - 1, j) for j in range(nz)], 1.0, 'REPLACE')
    vg.new(name="Buttons").add([vid(nth - 6, j) for j in range(3, nz - 4)], 1.0, 'REPLACE')
    charm = vg.new(name="Charms")
    charm.add([vid(5, nz - 3), vid(nth - 6, nz - 3)], 1.0, 'REPLACE')
    sq = vg.new(name="Sequins")
    for j in range(0, nz // 2):
        sq.add([vid(i, j) for i in range(nth)], 1.0 - j / (nz / 2), 'REPLACE')
    for o in (torso, head, neck):
        CO.add_collider(scene, o)
    g = CO.assign_garment(scene, jacket, preset='DENIM', quality='REALTIME', pin_mode='NONE')
    g.pin_group = "ClothPins"
    g.self_collision = False
    g.follow_preset = True
    g.margin = 0.006
    cl.use_scene_wind = False
    cl.wind = (0.0, 0.0, 0.0)
    # materials
    def mat(name, col, metal=0.0, rough=0.5):
        m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
        m.use_nodes = True
        b = m.node_tree.nodes.get("Principled BSDF")
        if b is not None:
            b.inputs["Base Color"].default_value = (col[0], col[1], col[2], 1.0)
            b.inputs["Metallic"].default_value = metal
            b.inputs["Roughness"].default_value = rough
        m.diffuse_color = (col[0], col[1], col[2], 1.0)
        return m
    jacket.data.materials.append(mat("Jacket Cloth", (0.10, 0.16, 0.30), 0.0, 0.85))
    m_hem = mat("Hem Leather", (0.30, 0.15, 0.07), 0.0, 0.45)
    m_gold = mat("Accent Gold", (0.95, 0.65, 0.18), 1.0, 0.25)
    m_jewel = mat("Accent Jewel", (0.85, 0.05, 0.20), 0.0, 0.12)
    # trims
    t = ts.trims.add()
    t.name = "Hem"; t.kind = 'BAND'; t.source = 'LOWEST'; t.garment = g.name
    t.width = 0.045; t.thickness = 0.006; t.mass_per_m = 0.12; t.bend = 1800.0; t.edge_weight = 3.0; t.material = m_hem
    t = ts.trims.add()
    t.name = "Collar"; t.kind = 'COLLAR'; t.source = 'HIGHEST'; t.garment = g.name
    t.width = 0.03; t.thickness = 0.005; t.mass_per_m = 0.06; t.bend = 900.0; t.edge_weight = 2.0; t.material = m_hem
    # seam
    s = ts.seams.add()
    s.name = "Zipper"; s.mode = 'ZIPPER'; s.garment = g.name; s.group = "Seam"
    s.progress = 0.0; s.strength = 1.0; s.gather = 0.32; s.ruffle = 0.012
    ts.seam_index = 0
    # zipper pull: a small slider tab that rides the zipper head
    pv = [(-0.5, -0.3, -0.5), (0.5, -0.3, -0.5), (0.5, 0.3, -0.5), (-0.5, 0.3, -0.5),
          (-0.5, -0.3, 0.5), (0.5, -0.3, 0.5), (0.5, 0.3, 0.5), (-0.5, 0.3, 0.5)]
    pf = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    pull = CO._new_obj(scene, "Zipper Pull", pv, pf, coll)
    pull.scale = (0.022, 0.012, 0.040)
    pull.data.materials.append(m_gold)
    s.pull = pull
    # accents
    a = ts.accents.add()
    a.name = "Buttons"; a.kind = 'BUTTON'; a.garment = g.name; a.placement = 'CURVE'; a.group = "Buttons"
    a.count = 5; a.size = 0.018; a.offset = 0.004; a.material = m_gold
    a = ts.accents.add()
    a.name = "Charms"; a.kind = 'CHARM'; a.garment = g.name; a.placement = 'VERTICES'; a.group = "Charms"
    a.size = 0.022; a.dynamic = True; a.chain_segments = 5; a.chain_length = 0.13; a.offset = 0.006; a.material = m_gold
    a = ts.accents.add()
    a.name = "Sequins"; a.kind = 'SEQUIN'; a.garment = g.name; a.placement = 'DENSITY'; a.group = "Sequins"
    a.density = 130.0; a.size = 0.011; a.offset = 0.003; a.seed = 4; a.material = m_jewel
    ts.status = "Jacket example: zip the Closed slider 0 to 1, or Key Zip Animation."
    return "Jacket example loaded: seam, hem and collar trims, buttons, charms and sequins. Step the cloth (or play) while moving the Closed slider."


class ARCFLOW_OT_trim_example(Operator):
    bl_idname = "arcflow.trim_example"
    bl_label = "Example: Jacket with zipper"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        msg = load_jacket_example(ctx.scene)
        self.report({'INFO'}, msg)
        return {'FINISHED'}


CLASSES = (ARCFLOW_OT_trim_add, ARCFLOW_OT_trim_remove, ARCFLOW_OT_accent_add, ARCFLOW_OT_accent_remove,
           ARCFLOW_OT_accent_use_selected, ARCFLOW_OT_seam_add, ARCFLOW_OT_seam_remove, ARCFLOW_OT_seam_pull_add,
           ARCFLOW_OT_seam_set, ARCFLOW_OT_seam_animate, ARCFLOW_OT_trim_group_from_selection,
           ARCFLOW_OT_trim_refresh, ARCFLOW_OT_trim_example)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    _install()
    try:
        from . import surfaces_panel
        surfaces_panel.register_example("Cloth", "Jacket with zipper", "arcflow.trim_example")
    except Exception:
        pass


def unregister():
    _uninstall()
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
