"""Surface sources and fluid volumes (Tessera Projection, Fluid mode): operators and the bridge to the Fluid mode solver.

Nothing in fluidmode_ops.py is edited: at register time its build_solver / sources / apply_state / _shape_of are wrapped (see `install_hooks`).
  build_solver   before: volume sources refresh their fluid entry from the Empty (size, mass at the pressure); after: the face selections are
                 sampled and core.fsurf.install hooks viscosity / emission into the solver instance.
  sources        called once per simulated frame: the surfaces follow their objects (velocity from the matrix change) and switch on.
  apply_state    emitter point meshes grow to the current particle count; scrubbing to an earlier frame restores that frame's arrays.
  _shape_of      a volume Empty gives its box mesh, which Fluid mode fills with particles."""
import math
import traceback

import bpy
import numpy as np
from bpy.props import EnumProperty, IntProperty, StringProperty
from bpy.types import Operator

from . import fluidmode_ops as FO
from . import gn
from . import surfaces_gn as SGN
from .core import fluidmode as FM
from .core import fsurf as FS

RT = FO.RT


# ----------------------------------------------------------------------------- helpers
def srf(scene):
    return scene.arc_flow_srf


def active_source(scene, index=-1):
    s = srf(scene)
    i = s.index if index < 0 else index
    return s.items[i] if 0 <= i < len(s.items) else None


def unique_name(scene, base):
    names = {s.name for s in srf(scene).items}
    nm, k = base, 1
    while nm in names:
        k += 1
        nm = "%s.%d" % (base, k)
    return nm


def attr_name(src):
    return "af_srf_" + gn.slug(src.name)


def volume_half(obj):
    return float(obj.empty_display_size) if obj is not None and obj.type == 'EMPTY' else 0.5


def volume_size(obj):
    """Box dimensions in m: 2 x display size x scale."""
    s = obj.matrix_world.to_scale() if obj is not None else (1, 1, 1)
    h = volume_half(obj)
    return (2 * h * abs(s[0]), 2 * h * abs(s[1]), 2 * h * abs(s[2]))


def fluid_of(scene, src):
    for fl in scene.arc_flow_fm.fluids:
        if fl.name == src.fluid:
            return fl
    return None


def fluid_index(scene, src):
    for i, fl in enumerate(scene.arc_flow_fm.fluids):
        if fl.name == src.fluid:
            return i
    return -1


def sync_source(scene, src, refresh_volume=True):
    """Create / update the Fluid mode entry that a source feeds (emission and volume sources)."""
    if src.kind == 'VISCOSITY' or scene is None or not isinstance(scene, bpy.types.Scene):
        return None
    fm = scene.arc_flow_fm
    fl = fluid_of(scene, src)
    if fl is None:
        fl = FO.add_fluid(scene, src.preset, name="%s fluid" % src.name)
        src.fluid = fl.name
    if fl.preset != src.preset:
        fl.preset = src.preset
    fl.temperature = src.temperature
    fl.no_container = True
    fl.container = None
    if src.kind == 'EMISSION':
        fl.shape = None
        fl.mass = fl.volume = fl.weight = 0.0
        fl.velocity = (0.0, 0.0, 0.0)
    else:
        fl.shape = src.obj
        fl.velocity = tuple(src.velocity0)
        fl.count = src.quantity
        rho_ref = FM.PRESETS[src.preset]['density'] if src.preset in FM.PRESETS else fl.density
        rho_ref = src.density or rho_ref
        if fl.kind == 'GAS':
            fl.density = FS.gas_density(rho_ref, src.temperature, FM.P_ATM)
            V = FS.box_volume(np.array(src.obj.matrix_world), (volume_half(src.obj),) * 3) if src.obj is not None else 0.0
            fl.mass = FS.gas_density(rho_ref, src.temperature, src.pressure) * V
            fl.volume = 0.0
        else:
            fl.density = rho_ref
            fl.mass = fl.volume = 0.0
        fl.weight = 0.0
    FO.mark_dirty()
    return fl


def _faces_world(src, dx):
    """World-space sample points / normals / areas of the stored faces of the source mesh (None when nothing is stored)."""
    obj = src.obj
    if obj is None or obj.type != 'MESH' or not src.face_attr:
        return None
    me = obj.data
    a = me.attributes.get(src.face_attr)
    if a is None:
        return None
    sel = np.zeros(len(me.polygons), bool)
    a.data.foreach_get('value', sel)
    if not sel.any():
        return None
    me.calc_loop_triangles()
    nt = len(me.loop_triangles)
    tri = np.zeros(nt * 3, np.int32)
    me.loop_triangles.foreach_get('vertices', tri)
    pidx = np.zeros(nt, np.int32)
    me.loop_triangles.foreach_get('polygon_index', pidx)
    tri = tri.reshape(-1, 3)[sel[pidx]]
    V = np.zeros(len(me.vertices) * 3, np.float64)
    me.vertices.foreach_get('co', V)
    M = np.array(obj.matrix_world, float)
    Vw = V.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
    P, N, A, _f = FS.sample_faces(Vw, tri, dx)
    if len(P) == 0:
        return None
    Mi = np.linalg.inv(M)
    lp = P @ Mi[:3, :3].T + Mi[:3, 3]
    ln = N @ M[:3, :3]
    ln /= np.maximum(np.linalg.norm(ln, axis=1), 1e-30)[:, None]
    return lp, ln, A, M


def attach(scene, S):
    """Sample the stored faces and hook the surface sources into the freshly built solver S."""
    vis, emi, bound = [], [], []
    for src in srf(scene).items:
        if not src.enabled or src.kind == 'VOLUME':
            continue
        smp = _faces_world(src, S.dx)
        if smp is None:
            continue
        lp, ln, A, M = smp
        if src.kind == 'VISCOSITY':
            core = FS.ViscositySurface(src.name, lp, ln, A, M, coefficient=src.coef, noslip=src.noslip, reach=src.reach or None)
            vis.append(core)
        else:
            f = fluid_index(scene, src)
            if f < 0:
                continue
            core = FS.EmissionSurface(src.name, lp, ln, A, M, fluid=f, rate=src.rate, speed=src.speed, velocity=tuple(src.velocity), spread=src.spread,
                                      temperature=src.temperature, max_particles=src.max_particles, start=src.t_start, end=src.t_end,
                                      seed=scene.arc_flow_fm.seed, reverse=src.reverse)
            emi.append(core)
        bound.append((src.name, src.obj, core, scene.frame_current))
    if not (vis or emi):
        return None
    st = FS.install(S, viscosity=vis, emission=emi)
    st.active = False          # the fill phase does not emit; sources() switches it on for the simulation
    st.bound = bound
    return st


def _refresh_matrices(scene, S, frame):
    st = getattr(S, 'surf', None)
    if st is None:
        return
    dt = 1.0 / FO.fps(scene)
    for i, (name, obj, core, last) in enumerate(getattr(st, 'bound', [])):
        if obj is None:
            continue
        M = np.array(obj.matrix_world, float)
        core.set_matrix(M, dt if (frame is not None and last is not None and frame - last == 1) else None)
        st.bound[i] = (name, obj, core, frame)


def _readback(scene, S):
    st = getattr(S, 'surf', None)
    if st is None:
        return
    srcs = {s.name: s for s in srf(scene).items}
    for (name, obj, core, _l) in getattr(st, 'bound', []):
        s = srcs.get(name)
        if s is None:
            continue
        if core.kind == 'VISCOSITY':
            s.out_force = tuple(float(x) for x in core.force)
        else:
            s.out_count = int(core.emitted)
        _push_modifier(s)


def _push_modifier(src):
    obj = src.obj
    if obj is None:
        return
    host = obj if obj.type == 'MESH' and src.kind != 'VOLUME' else None
    if host is None:
        return
    mod = host.modifiers.get("Tessera Source")
    if mod is not None and mod.node_group is not None:
        SGN.set_inputs(mod, src)


# ----------------------------------------------------------------------------- hooks into fluidmode_ops
_orig = {}


def _build_solver(scene):
    try:
        for src in srf(scene).items:
            if src.kind != 'VISCOSITY':
                sync_source(scene, src)
    except Exception:
        traceback.print_exc()
    S = _orig['build_solver'](scene)
    try:
        attach(scene, S)
    except Exception as ex:
        RT.msg = "Surface sources: %s" % ex
        traceback.print_exc()
    return S


def _sources(scene, frame=None):
    out = _orig['sources'](scene, frame)
    S = RT.solver
    st = getattr(S, 'surf', None) if S is not None else None
    if st is not None:
        try:
            _readback(scene, S)
            _refresh_matrices(scene, S, frame)
        except Exception:
            traceback.print_exc()
        st.active = True
    return out


def _emitter_fluids(scene, S):
    st = getattr(S, 'surf', None)
    return sorted({e.fluid for e in st.emission}) if st is not None else []


def _resize_points(scene, S, counts):
    for fi, n in counts.items():
        fm = scene.arc_flow_fm
        if fi >= len(fm.fluids) or n <= 0:
            continue
        o = fm.fluids[fi].points_obj
        if o is None:
            continue
        me = o.data
        if len(me.vertices) < n:
            me.vertices.add(n - len(me.vertices))
        elif len(me.vertices) > n:
            o.data = FO._points_mesh(me.name, n)


def _apply_state(scene, snap=None):
    S = RT.solver
    st = getattr(S, 'surf', None) if S is not None else None
    if st is None or not st.emission:
        return _orig['apply_state'](scene, snap)
    cur = None
    if snap is not None and '_srf' in snap:
        g = snap['_srf']
        if g['n'] != S.n or not (np.array_equal(g['lo'], S.lo) and np.array_equal(g['hi'], S.hi)):
            cur = S.snapshot()
            S.restore(snap)
            snap = None
    try:
        _resize_points(scene, S, {f: int(S.hi[f] - S.lo[f]) for f in _emitter_fluids(scene, S)})
        _orig['apply_state'](scene, snap)
    finally:
        if cur is not None:
            S.restore(cur)
    _readback(scene, S)


def _shape_of(obj, deps):
    if obj.type == 'EMPTY' and obj.get('af_volume'):
        h = volume_half(obj)
        V, T = FS.box_mesh(np.array(obj.matrix_world, float), (h, h, h))
        return dict(kind='MESH', verts=V, tris=T)
    return _orig['_shape_of'](obj, deps)


def install_hooks():
    if _orig:
        return
    for nm, fn in (('build_solver', _build_solver), ('sources', _sources), ('apply_state', _apply_state), ('_shape_of', _shape_of)):
        _orig[nm] = getattr(FO, nm)
        setattr(FO, nm, fn)


def remove_hooks():
    for nm, fn in list(_orig.items()):
        setattr(FO, nm, fn)
    _orig.clear()


# ----------------------------------------------------------------------------- operators
def _need_source(op, ctx):
    src = active_source(ctx.scene, op.index)
    if src is None:
        op.report({'ERROR'}, "No surface source selected")
    return src


class _IndexOp:
    index: IntProperty(default=-1, options={'SKIP_SAVE', 'HIDDEN'})


class AF_OT_srf_add(Operator):
    bl_idname = "arcflow.srf_add"
    bl_label = "Add Surface"
    bl_description = "Add a viscosity or emission surface source. Use 'Use selected object', then 'Edit faces' and 'Apply'"
    bl_options = {'REGISTER', 'UNDO'}
    kind: EnumProperty(items=[('VISCOSITY', "Viscosity surface", ""), ('EMISSION', "Emission surface", "")], default='VISCOSITY')

    def execute(self, ctx):
        sc = ctx.scene
        s = srf(sc)
        src = s.items.add()
        src.kind = self.kind
        src.name = unique_name(sc, "Viscosity surface" if self.kind == 'VISCOSITY' else "Emission surface")
        o = ctx.active_object
        if o is not None and o.type == 'MESH':
            src.obj = o
        s.index = len(s.items) - 1
        if self.kind == 'EMISSION':
            src.preset = 'WATER'
            sync_source(sc, src)
        FO.mark_dirty()
        return {'FINISHED'}


class AF_OT_srf_remove(_IndexOp, Operator):
    bl_idname = "arcflow.srf_remove"
    bl_label = "Remove Source"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        sc = ctx.scene
        src = _need_source(self, ctx)
        if src is None:
            return {'CANCELLED'}
        fm = sc.arc_flow_fm
        fl = fluid_of(sc, src)
        if fl is not None:
            fm.fluids.remove(list(fm.fluids).index(fl))
            fm.index = max(0, min(fm.index, len(fm.fluids) - 1))
        if src.kind == 'VOLUME' and src.obj is not None:
            for c in list(src.obj.children):
                if c.get('af_volume_points'):
                    bpy.data.objects.remove(c, do_unlink=True)
            bpy.data.objects.remove(src.obj, do_unlink=True)
        elif src.obj is not None:
            m = src.obj.modifiers.get("Tessera Source")
            if m is not None:
                src.obj.modifiers.remove(m)
            if src.face_attr and src.face_attr in src.obj.data.attributes:
                src.obj.data.attributes.remove(src.obj.data.attributes[src.face_attr])
        s = srf(sc)
        s.items.remove(s.index if self.index < 0 else self.index)
        s.index = max(0, min(s.index, len(s.items) - 1))
        FO.mark_dirty()
        return {'FINISHED'}


class AF_OT_srf_use_selected(_IndexOp, Operator):
    bl_idname = "arcflow.srf_use_selected"
    bl_label = "Use selected object"
    bl_description = "Use the active mesh object as the surface of this source"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        src = _need_source(self, ctx)
        if src is None:
            return {'CANCELLED'}
        o = ctx.active_object
        if o is None or o.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh object first")
            return {'CANCELLED'}
        if src.obj is not None and src.obj != o and src.face_attr and src.face_attr in src.obj.data.attributes:
            src.obj.data.attributes.remove(src.obj.data.attributes[src.face_attr])
            src.face_attr, src.face_count, src.face_area = "", 0, 0.0
        src.obj = o
        FO.mark_dirty()
        self.report({'INFO'}, "%s uses %s. Press 'Edit faces' to choose the faces." % (src.name, o.name))
        return {'FINISHED'}


class AF_OT_srf_edit_faces(_IndexOp, Operator):
    bl_idname = "arcflow.srf_edit_faces"
    bl_label = "Edit faces"
    bl_description = "Switch to Edit Mode to select the faces of this surface (the stored faces are preselected), then press Apply"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        src = _need_source(self, ctx)
        if src is None:
            return {'CANCELLED'}
        o = src.obj
        if o is None or o.type != 'MESH':
            self.report({'ERROR'}, "Press 'Use selected object' first")
            return {'CANCELLED'}
        if ctx.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        for ob in ctx.view_layer.objects:
            ob.select_set(False)
        o.select_set(True)
        ctx.view_layer.objects.active = o
        me = o.data
        a = me.attributes.get(src.face_attr) if src.face_attr else None
        if a is not None:
            sel = np.zeros(len(me.polygons), bool)
            a.data.foreach_get('value', sel)
            me.polygons.foreach_set('select', sel)
            me.update()
        bpy.ops.object.mode_set(mode='EDIT')
        try:
            bpy.ops.mesh.select_mode(type='FACE')
        except Exception:
            pass
        self.report({'INFO'}, "Select the faces of '%s', then press Apply" % src.name)
        return {'FINISHED'}


def store_faces(src, sel):
    """Write the face selection as a boolean face attribute and update the face count and area."""
    me = src.obj.data
    nm = attr_name(src)
    if src.face_attr and src.face_attr != nm and src.face_attr in me.attributes:
        me.attributes.remove(me.attributes[src.face_attr])
    if nm in me.attributes:
        me.attributes.remove(me.attributes[nm])
    a = me.attributes.new(nm, 'BOOLEAN', 'FACE')
    a.data.foreach_set('value', np.asarray(sel, bool))
    src.face_attr = nm
    src.face_count = int(np.count_nonzero(sel))
    ar = np.zeros(len(me.polygons), np.float32)
    me.polygons.foreach_get('area', ar)
    sc = np.abs(np.array(src.obj.matrix_world.to_scale()))
    src.face_area = float(ar[np.asarray(sel, bool)].sum() * (sc.prod() ** (2.0 / 3.0)))
    me.update()


class AF_OT_srf_apply(_IndexOp, Operator):
    bl_idname = "arcflow.srf_apply"
    bl_label = "Apply"
    bl_description = "Store the selected faces as this surface source (leaves Edit Mode) and connect it to the fluid"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        sc = ctx.scene
        src = _need_source(self, ctx)
        if src is None:
            return {'CANCELLED'}
        o = src.obj
        if o is None or o.type != 'MESH':
            self.report({'ERROR'}, "Press 'Use selected object' first")
            return {'CANCELLED'}
        if o.mode == 'EDIT':
            bpy.ops.object.mode_set(mode='OBJECT')     # flushes the edit selection into the mesh
        sel = np.zeros(len(o.data.polygons), bool)
        o.data.polygons.foreach_get('select', sel)
        if not sel.any():
            self.report({'ERROR'}, "No faces selected: press 'Edit faces', select faces, then Apply")
            return {'CANCELLED'}
        store_faces(src, sel)
        if src.kind == 'EMISSION':
            sync_source(sc, src)
        SGN.ensure_modifier(o, src)
        FO.mark_dirty()
        self.report({'INFO'}, "%s: %d faces stored (%.4g m2)" % (src.name, src.face_count, src.face_area))
        return {'FINISHED'}


def make_volume(scene, location=(0.0, 0.0, 1.0), size=1.0, name="Fluid Volume", preset='AIR', temperature=None, quantity=1500, rotation=(0, 0, 0)):
    s = srf(scene)
    src = s.items.add()
    src.kind = 'VOLUME'
    src.name = unique_name(scene, name)
    src.preset = preset
    src.quantity = quantity
    if temperature is not None:
        src.temperature = temperature
    o = bpy.data.objects.new(src.name, None)
    o.empty_display_type = 'CUBE'
    o.empty_display_size = 0.5
    o.show_in_front = True
    o.hide_render = True
    o.location = location
    o.rotation_euler = rotation
    o.scale = (size, size, size)
    o["af_volume"] = True
    col = bpy.context.collection or scene.collection
    col.objects.link(o)
    src.obj = o
    s.index = len(s.items) - 1
    sync_source(scene, src)
    # helper mesh that carries the exposed Geometry Nodes group (an Empty cannot hold modifiers)
    me = bpy.data.meshes.new(src.name + " points")
    me.vertices.add(1)
    h = bpy.data.objects.new(src.name + " points", me)
    h.parent = o
    h.hide_render = True
    h["af_volume_points"] = True
    col.objects.link(h)
    fl = fluid_of(scene, src)
    SGN.ensure_modifier(h, src, volume_size(o), fl.density if fl is not None else None)
    FO.mark_dirty()
    return src


class AF_OT_srf_add_volume(Operator):
    bl_idname = "arcflow.srf_add_volume"
    bl_label = "Add Fluid Volume"
    bl_description = ("Add a wireframe box (an Empty that does not render). Scale, move and rotate it with Blender's manipulators: Build fills the box with "
                      "free-floating fluid or gas that interacts with the Environment from the first frame")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        sc = ctx.scene
        loc = tuple(sc.cursor.location)
        src = make_volume(sc, loc)
        for ob in ctx.view_layer.objects:
            ob.select_set(False)
        src.obj.select_set(True)
        ctx.view_layer.objects.active = src.obj
        sc.arc_flow_fm.mode = 'FLUID'
        self.report({'INFO'}, "Fluid volume added at the 3D cursor. Scale the box, then Build.")
        return {'FINISHED'}


class AF_OT_srf_select_object(_IndexOp, Operator):
    bl_idname = "arcflow.srf_select_object"
    bl_label = "Select object"
    bl_description = "Select the object of this source (the box of a volume)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        src = _need_source(self, ctx)
        if src is None or src.obj is None:
            return {'CANCELLED'}
        if ctx.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        for ob in ctx.view_layer.objects:
            ob.select_set(False)
        src.obj.select_set(True)
        ctx.view_layer.objects.active = src.obj
        return {'FINISHED'}


# ----------------------------------------------------------------------------- examples built from these sources
def load_surface_example(scene, which='EMIT'):
    """Examples: EMIT = an emission nozzle over a viscous belt; VOLUME = a hot gas box and a cool gas box."""
    import bmesh
    from mathutils import Vector
    fm = scene.arc_flow_fm
    FO._clear_demo(scene)
    for s in list(srf(scene).items):
        srf(scene).items.remove(0)
    for o in list(bpy.data.objects):
        if o.get('af_volume') or o.get('af_volume_points') or o.name.startswith('SRF_'):
            bpy.data.objects.remove(o, do_unlink=True)
    col = FO._demo_common(scene)
    fm.mode = 'FLUID'
    fm.max_particles = 6000
    fm.fill_mode = 'OFF'
    fm.use_scene_range = False
    scene.frame_start, scene.frame_end = 1, 72
    if which == 'VOLUME':
        a = make_volume(scene, (-0.9, 0, 0.5), 0.8, "Hot gas", 'AIR', 420.0, 900)
        b = make_volume(scene, (0.9, 0, 0.5), 0.8, "Cool gas", 'HELIUM' if 'HELIUM' in FM.PRESETS else 'AIR', 280.0, 900)
        for s in (a, b):
            fl = fluid_of(scene, s)
            fl.temp_color = True
            fl.t_lo, fl.t_hi = 270.0, 430.0
            fl.particle_scale = 1.2
        srf(scene).index = 0
        scene.arc_flow.env.ground = True
        return "Two gas boxes float free in the Environment: the hot one rises, the other follows its own buoyancy. Build and play."
    # EMIT: a belt (viscosity) below a nozzle (emission), both plain planes
    def plane(name, size, loc, rot=(0, 0, 0)):
        bm = bmesh.new()
        bmesh.ops.create_grid(bm, x_segments=4, y_segments=4, size=size)
        me = bpy.data.meshes.new(name)
        bm.to_mesh(me)
        bm.free()
        o = bpy.data.objects.new(name, me)
        o.location, o.rotation_euler = loc, rot
        col.objects.link(o)
        return o
    belt = plane("SRF_Belt", 0.8, (-0.5, 0.0, 0.0))
    nozzle = plane("SRF_Nozzle", 0.06, (-0.1, 0.0, 0.7), (math.pi, 0, 0))     # faces down
    for o in (belt, nozzle):
        for p in o.data.polygons:
            p.select = True
    s = srf(scene)
    v = s.items.add(); v.kind = 'VISCOSITY'; v.name = "Belt"; v.obj = belt; v.coef = 400.0; v.noslip = 1.0
    store_faces(v, np.ones(len(belt.data.polygons), bool))
    SGN.ensure_modifier(belt, v)
    e = s.items.add(); e.kind = 'EMISSION'; e.name = "Nozzle"; e.obj = nozzle; e.preset = 'WATER'; e.rate = 300.0; e.speed = 0.4; e.spread = 8.0
    e.max_particles = 4000
    store_faces(e, np.ones(len(nozzle.data.polygons), bool))
    sync_source(scene, e)
    SGN.ensure_modifier(nozzle, e)
    fl = fluid_of(scene, e)
    fl.color = (0.2, 0.55, 1.0)
    # the belt slides sideways (keyframed): the viscous layer drags the fluid with it
    belt.location = (-0.5, 0, 0)
    belt.keyframe_insert('location', frame=1)
    belt.location = (0.4, 0, 0)
    belt.keyframe_insert('location', frame=72)
    for fc in (belt.animation_data.action.fcurves if hasattr(belt.animation_data.action, 'fcurves') else []):
        for kp in fc.keyframe_points:
            kp.interpolation = 'LINEAR'
    s.index = 0
    scene.frame_set(1)
    return "A nozzle emits water onto a sliding belt that drags the fluid with it (viscosity surface). Build and play."


class AF_OT_srf_example(Operator):
    bl_idname = "arcflow.srf_example"
    bl_label = "Load Example"
    bl_options = {'REGISTER', 'UNDO'}
    which: StringProperty(default='EMIT')

    @classmethod
    def description(cls, ctx, props):
        return {'EMIT': "Emission surface above a moving viscosity surface (a belt that drags the fluid)",
                'VOLUME': "Two fluid volumes: hot and cool gas boxes floating in the Environment"}.get(props.which, "")

    def execute(self, ctx):
        msg = load_surface_example(ctx.scene, self.which)
        FO.mark_dirty()
        self.report({'INFO'}, msg)
        return {'FINISHED'}


CLASSES = (AF_OT_srf_add, AF_OT_srf_remove, AF_OT_srf_use_selected, AF_OT_srf_edit_faces, AF_OT_srf_apply, AF_OT_srf_add_volume,
           AF_OT_srf_select_object, AF_OT_srf_example)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    install_hooks()


def unregister():
    remove_hooks()
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
