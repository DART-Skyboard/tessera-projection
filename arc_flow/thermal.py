"""Thermal readback (Tessera Projection): surface temperature of collision bodies from the fluid physics.

Per selected object (or its selected faces) the extension keeps a wall temperature per mesh point or face, updated every frame
from the nearby solver particles, convection and heat sources (core/thermal.py), writes it to the mesh attribute `af_temperature`
(and a colour attribute `af_thermal_color` for the Solid viewport), builds a shader from it (works in EEVEE and Cycles), an optional
Geometry Nodes group that outputs the values, and can give the exchanged heat back to the fluid as heat sources."""
import json
import math

import bpy
import numpy as np
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, EnumProperty, FloatProperty, PointerProperty
from bpy.types import Operator, Panel, PropertyGroup

from . import fluidmode_ops as FO
from .core import thermal as TH
from .nodeutil import Builder, add_socket, enable_nodes, fresh_group, inp, out

ATTR = "af_temperature"
COLOR = "af_thermal_color"
MASK = "af_th_mask"
MATIDX = "af_th_matidx"
MOD_NAME = "Tessera Surface Temperature"
GROUP = {'POINT': "Tessera Thermal Surface (Point)", 'FACE': "Tessera Thermal Surface (Face)"}


# ----------------------------------------------------------------------------- properties
def _changed(self, ctx):
    o = self.id_data
    try:
        refresh_look(o)
        STATE.pop(o.name, None)
    except Exception:
        pass


def _enable_changed(self, ctx):
    o = self.id_data
    if self.on:
        enable(o)
    else:
        disable(o)


def _look_changed(self, ctx):
    try:
        refresh_look(self.id_data)
    except Exception:
        pass


class AF_ThermalObj(PropertyGroup):
    on: BoolProperty(name="Thermal readback", default=False, update=_enable_changed,
                     description="Compute the surface temperature of this object from the fluid physics each frame and write it to the 'af_temperature' attribute")
    domain: EnumProperty(name="Domain", items=[('POINT', "Points", "One temperature per vertex"), ('FACE', "Faces", "One temperature per face")], default='POINT', update=_changed)
    t_min: FloatProperty(name="Range min", default=288.15, description="K at the cold end of the ramp", update=_look_changed)
    t_max: FloatProperty(name="Range max", default=340.0, description="K at the hot end of the ramp", update=_look_changed)
    ramp: EnumProperty(name="Ramp", items=TH.RAMP_ITEMS, default='INFERNO', update=_look_changed)
    shader: BoolProperty(name="Thermal shader", default=True, update=_look_changed, description="Replace the surface material with the temperature ramp (EEVEE and Cycles). Off restores the original material")
    emission: FloatProperty(name="Glow", default=0.3, min=0.0, max=20.0, update=_look_changed, description="Emission strength of the ramp colour so it reads in any light")
    solid: BoolProperty(name="Solid viewport colour", default=True, update=_look_changed, description="Also write the 'af_thermal_color' attribute and make it the active colour so Solid mode (colour: Attribute) shows it")
    gn: BoolProperty(name="Geometry Nodes output", default=True, update=_look_changed, description="Add the 'Tessera Thermal Surface' Geometry Nodes modifier (normalised value, colour, min / mean / max outputs)")
    couple: BoolProperty(name="Heat the fluid", default=False, description="The heat the surface exchanges with the fluid is given back to the particles as heat sources", update=_changed)
    wall_heat: FloatProperty(name="Wall heat capacity", default=4900.0, min=1.0, description="J/(m2 K): density x specific heat x thickness of the skin (2 mm aluminium = 4900)", update=_changed)
    absorb: FloatProperty(name="Absorptivity", default=0.7, min=0.0, max=1.0, description="Fraction of the radiation from heat sources that the surface absorbs", update=_changed)
    emissivity: FloatProperty(name="Emissivity", default=0.3, min=0.0, max=1.0, description="Thermal radiation of the surface to the ambient (limits how hot a wall next to a strong source gets)", update=_changed)
    recovery: FloatProperty(name="Recovery factor", default=0.85, min=0.0, max=1.0, description="Fraction of the kinetic temperature rise v^2 / 2cp recovered at the wall (turbulent boundary layer 0.85)", update=_changed)
    probe: FloatProperty(name="Probe radius", default=2.2, min=1.2, max=6.0, description="Fluid sampling radius in particle spacings", update=_changed)
    speedup: FloatProperty(name="Wall speed-up", default=1.0, min=0.0, description="Multiplies the wall's heating time so slow real heating fits an animation (1 = real time)", update=_changed)


# ----------------------------------------------------------------------------- state
class _State:
    def __init__(self):
        self.sig = None
        self.frame = None
        self.T0 = None
        self.cache = {}
        self.info = None
        self.P = None
        self.area = None
        self.Ts = None


STATE = {}


def _fps(scene):
    return scene.render.fps / scene.render.fps_base


# ----------------------------------------------------------------------------- geometry sampling
def _poly_data(me):
    npoly = len(me.polygons)
    area = np.empty(npoly, np.float64); me.polygons.foreach_get('area', area)
    tot = np.empty(npoly, np.int64); me.polygons.foreach_get('loop_total', tot)
    nl = len(me.loops)
    lv = np.empty(nl, np.int64); me.loops.foreach_get('vertex_index', lv)
    lp = np.repeat(np.arange(npoly), tot)
    return area, tot, lv, lp


def mask_of(o):
    """Boolean per face: the faces that take part (all when no mask attribute is stored)."""
    me = o.data
    a = me.attributes.get(MASK)
    if a is None or a.domain != 'FACE':
        return np.ones(len(me.polygons), bool)
    v = np.empty(len(me.polygons), np.int32)
    a.data.foreach_get('value', v)
    return v != 0


def sample_geometry(o, deps, domain):
    """World-space samples: positions P, unit normals N, areas, selection index into the attribute domain, size of the domain.
    Uses the evaluated mesh (shape keys, armatures, parent / object animation) when its topology matches the original."""
    me0 = o.data
    ev = o.evaluated_get(deps)
    me = ev.to_mesh()
    try:
        same = len(me.vertices) == len(me0.vertices) and len(me.polygons) == len(me0.polygons)
        src = me if same else me0
        M = np.array(ev.matrix_world if same else o.matrix_world)
        nv, npoly = len(src.vertices), len(src.polygons)
        co = np.empty(nv * 3, np.float64); src.vertices.foreach_get('co', co); co = co.reshape(-1, 3)
        area, tot, lv, lp = _poly_data(src)
        fmask = mask_of(o)
        if len(fmask) != npoly:
            fmask = np.ones(npoly, bool)
        R = M[:3, :3]
        if domain == 'POINT':
            vn = np.empty(nv * 3, np.float64); src.vertex_normals.foreach_get('vector', vn); vn = vn.reshape(-1, 3)
            va = np.bincount(lv, (area / np.maximum(tot, 1))[lp], nv)
            vm = np.zeros(nv, bool); vm[lv[fmask[lp]]] = True
            sel = np.flatnonzero(vm)
            P = co[sel] @ R.T + M[:3, 3]
            N = vn[sel] @ np.linalg.inv(R).T
            ar = va[sel] * (abs(np.linalg.det(R)) ** (2.0 / 3.0))
            size = nv
        else:
            cen = np.empty(npoly * 3, np.float64); src.polygons.foreach_get('center', cen); cen = cen.reshape(-1, 3)
            nrm = np.empty(npoly * 3, np.float64); src.polygons.foreach_get('normal', nrm); nrm = nrm.reshape(-1, 3)
            sel = np.flatnonzero(fmask)
            P = cen[sel] @ R.T + M[:3, 3]
            N = nrm[sel] @ np.linalg.inv(R).T
            ar = area[sel] * (abs(np.linalg.det(R)) ** (2.0 / 3.0))
            size = npoly
        N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
        dims = (co @ R.T).max(0) - (co @ R.T).min(0) if nv else np.ones(3)
        return P, N, ar, sel, size, float(max(dims.max(), 1e-3))
    finally:
        ev.to_mesh_clear()


# ----------------------------------------------------------------------------- solver access
def lamps(scene, frame=None):
    """Heat sources (position, power W, radius) of the scene: the objects with 'Heat source' on (the wall absorbs their radiation)."""
    if frame is not None and frame < scene.arc_flow_fm.heat_start:
        return []
    return [(tuple(o.matrix_world.translation), o.arc_flow_fm.heat_power, o.arc_flow_fm.heat_radius)
            for o in scene.objects if o.arc_flow_fm.heat_on]


def _particles(scene, frame):
    """(X, V, T, cp, k, rho, mu, dx, sources) of the cached solver state at `frame` (the closest earlier one), or empty arrays."""
    S = FO.RT.solver
    srcs = lamps(scene, frame)
    if S is None or S.n == 0:
        z = np.zeros(0)
        return np.zeros((0, 3)), np.zeros((0, 3)), z, z, z, z, z, 0.05, srcs
    snap = FO.RT.cache.get(frame)
    if snap is None and FO.RT.cache:
        ks = [f for f in FO.RT.cache if f <= frame]
        snap = FO.RT.cache[max(ks)] if ks else None
    if snap is None:
        x, v, T = S.x, S.v, S.T
    else:
        x, v, T = snap['x'], snap['v'], snap['T']
    f = S.fid
    keep = (S.x, S.T)
    S.x, S.T = x, T
    try:
        rho = S.rho_particles()
    finally:
        S.x, S.T = keep
    dx_eff = float(np.cbrt(np.mean(S.Vp[f])))      # liquids: the lattice spacing; sparse gas parcels: the volume each one stands for
    return x, v, T, S.cp[f], S.k[f], rho, S.visc[f], dx_eff, srcs


# ----------------------------------------------------------------------------- per-frame update
def _ensure_attrs(o, domain, create_color):
    me = o.data
    dom = 'POINT' if domain == 'POINT' else 'FACE'
    a = me.attributes.get(ATTR)
    if a is not None and a.domain != dom:
        me.attributes.remove(a)
        a = None
    if a is None:
        a = me.attributes.new(ATTR, 'FLOAT', dom)
    if create_color:
        cdom = 'POINT' if domain == 'POINT' else 'CORNER'
        c = me.color_attributes.get(COLOR)
        if c is not None and c.domain != cdom:
            me.color_attributes.remove(c)
            c = None
        if c is None:
            c = me.color_attributes.new(COLOR, 'FLOAT_COLOR', cdom)
    return a


def write_attributes(o, full, lp=None):
    p = o.arc_flow_th
    me = o.data
    a = _ensure_attrs(o, p.domain, p.solid)
    a.data.foreach_set('value', full.astype(np.float32))
    if p.solid:
        rgb = TH.ramp_color(TH.normalise(full, p.t_min, p.t_max), p.ramp)
        rgba = np.concatenate([rgb, np.ones((len(rgb), 1))], axis=1).astype(np.float32)
        if p.domain == 'FACE':
            area, tot, lv, lpoly = _poly_data(me)
            rgba = rgba[lpoly]
        c = me.color_attributes.get(COLOR)
        if c is not None and len(c.data) == len(rgba):
            c.data.foreach_set('color', rgba.ravel())
    me.update_tag()


def update_object(scene, o, frame=None):
    """Bring the wall temperature of object `o` to `frame` and write it to the mesh.  Returns the sample temperatures (selection only)."""
    p = o.arc_flow_th
    frame = scene.frame_current if frame is None else frame
    deps = bpy.context.evaluated_depsgraph_get()
    P0, N0, area, sel, size, L = sample_geometry(o, deps, p.domain)
    st = STATE.setdefault(o.name, _State())
    sig = (p.domain, size, len(sel), round(p.wall_heat, 3), round(p.absorb, 3), round(p.recovery, 3), round(p.emissivity, 3), round(p.probe, 3), round(p.speedup, 3),
           scene.frame_start, round(scene.arc_flow.env.temperature, 3))
    if st.sig != sig:
        st.sig, st.frame, st.cache, st.info = sig, None, {}, None
    ambient = float(scene.arc_flow.env.temperature)
    T0 = np.full(len(sel), ambient)
    dx_last = 0.05
    dt = 1.0 / _fps(scene)
    if frame in st.cache:
        Ts = st.cache[frame]
    else:
        if st.frame is not None and frame > st.frame:
            frames = range(st.frame + 1, frame + 1)
            Ts = st.cache.get(st.frame, T0)
        else:
            f0 = scene.frame_start if frame - scene.frame_start <= 400 else frame
            frames = range(f0, frame + 1)
            Ts = T0
        Ts = Ts.copy()
        for fr in frames:
            X, V, T, cp, k, rho, mu, dx, srcs = _particles(scene, fr)
            if dx <= 0:
                dx = 0.05
            dx_last = dx
            h = p.probe * dx
            off = P0 + N0 * 0.5 * dx
            Ts, info = TH.surface_step(Ts, off, N0, X, V, T, cp, k, rho, mu, dt, h=h, dx=dx, ambient=ambient, L=L, C_a=p.wall_heat,
                                       recovery=p.recovery, sources=srcs, alpha=p.absorb, speedup=p.speedup, emissivity=p.emissivity)
            st.cache[fr] = Ts
            st.info = info
        st.frame = frame
    st.P, st.area, st.Ts, st.sel = P0, area, Ts, sel
    st.dx = dx_last
    full = np.full(size, ambient)
    full[sel] = Ts
    write_attributes(o, full)
    return Ts


def active_objects(scene):
    return [o for o in scene.objects if o.type == 'MESH' and getattr(o, 'arc_flow_th', None) is not None and o.arc_flow_th.on]


def update_all(scene, frame=None):
    n = 0
    for o in active_objects(scene):
        try:
            update_object(scene, o, frame)
            n += 1
        except Exception as ex:
            import traceback
            traceback.print_exc()
            FO.RT.msg = "Thermal readback (%s): %s" % (o.name, ex)
    return n


@persistent
def on_frame_change(scene, *args):
    if not active_objects(scene):
        return
    if FO.RT.dirty and FO.RT.solver is not None and not FO.RT.running:
        pass
    update_all(scene)


# ----------------------------------------------------------------------------- body -> fluid coupling
_orig_sources = None      # captured at register time, so wrappers installed by other modules stay in the chain


def sources_with_bodies(scene, frame=None):
    out_ = list(_orig_sources(scene, frame))
    if frame is not None and frame < scene.arc_flow_fm.heat_start:
        return out_
    for o in active_objects(scene):
        st = STATE.get(o.name)
        p = o.arc_flow_th
        if not p.couple or st is None or st.info is None or st.Ts is None:
            continue
        dx = max(getattr(st, 'dx', 0.05), 1e-3)
        try:
            for pos, power, rad in TH.patch_sources(st.P, st.area, st.Ts, st.info, 4.0 * dx):
                out_.append((pos, float(np.clip(power, -2e3, 2e3)), rad))
        except Exception:
            pass
    return out_


def _install_coupling():
    # the lamps (_orig_sources) are what the wall absorbs; the body's own patches are added only on the way to the fluid solver
    global _orig_sources
    if FO.sources is not sources_with_bodies:
        _orig_sources = FO.sources
    FO.sources = sources_with_bodies


def _remove_coupling():
    if FO.sources is sources_with_bodies:
        FO.sources = _orig_sources


# ----------------------------------------------------------------------------- look: shader, GN, solid colour
def _build_shader(o):
    p = o.arc_flow_th
    name = "Tessera Thermal %s" % o.name
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    nt = enable_nodes(m)
    nt.nodes.clear()
    b = Builder(nt)
    at = b.node('ShaderNodeAttribute', label="Temperature", y=0)
    at.attribute_type = 'GEOMETRY'
    at.attribute_name = ATTR
    mr = b.node('ShaderNodeMapRange', label="Range (K)", y=0)
    mr.clamp = True
    inp(mr, 'From Min').default_value = p.t_min
    inp(mr, 'From Max').default_value = p.t_max
    cr = b.node('ShaderNodeValToRGB', label="Ramp", y=0)
    st = TH.RAMPS[p.ramp]
    ramp = cr.color_ramp
    while len(ramp.elements) > 2:
        ramp.elements.remove(ramp.elements[-1])
    ramp.elements[0].position, ramp.elements[0].color = st[0][0], (*st[0][1], 1.0)
    ramp.elements[1].position, ramp.elements[1].color = st[-1][0], (*st[-1][1], 1.0)
    for pos, col in st[1:-1]:
        e = ramp.elements.new(pos)
        e.color = (*col, 1.0)
    bs = b.node('ShaderNodeBsdfPrincipled', y=0)
    bs.inputs['Roughness'].default_value = 0.45
    em = bs.inputs.get('Emission Color')
    if em is not None:
        bs.inputs['Emission Strength'].default_value = p.emission
    o_ = b.node('ShaderNodeOutputMaterial', y=0)
    b.link(out(at, 'Fac'), inp(mr, 'Value'))
    b.link(out(mr, 'Result'), inp(cr, 'Fac'))
    b.link(out(cr, 'Color'), inp(bs, 'Base Color'))
    if em is not None:
        b.link(out(cr, 'Color'), em)
    b.link(out(bs, 'BSDF'), inp(o_, 'Surface'))
    m.diffuse_color = (0.8, 0.4, 0.1, 1.0)
    return m


def thermal_material(o):
    return _build_shader(o)


def _apply_material(o):
    """Show the thermal material: on the selected faces only (an extra slot) or on the whole object (all slots)."""
    me = o.data
    p = o.arc_flow_th
    m = _build_shader(o)
    if not o.get('af_th_slots'):
        o['af_th_slots'] = json.dumps([s.material.name if s.material else "" for s in o.material_slots])
        o['af_th_matcount'] = len(o.material_slots)
    fmask = mask_of(o)
    whole = bool(fmask.all())
    if whole:
        if len(me.materials) == 0:
            me.materials.append(m)
        else:
            for i in range(len(me.materials)):
                me.materials[i] = m
    else:
        if MATIDX not in me.attributes:
            idx = np.zeros(len(me.polygons), np.int32)
            me.polygons.foreach_get('material_index', idx)
            a = me.attributes.new(MATIDX, 'INT', 'FACE')
            a.data.foreach_set('value', idx)
        if m.name not in [x.name for x in me.materials if x]:
            me.materials.append(m)
        slot = [x.name if x else "" for x in me.materials].index(m.name)
        base = np.zeros(len(me.polygons), np.int32)
        me.attributes[MATIDX].data.foreach_get('value', base)
        base[fmask] = slot
        me.polygons.foreach_set('material_index', base)
    me.update()


def _restore_material(o):
    me = o.data
    raw = o.get('af_th_slots')
    if raw is not None:
        names = json.loads(raw)
        n0 = int(o.get('af_th_matcount', len(names)))
        a = me.attributes.get(MATIDX)
        if a is not None:
            idx = np.zeros(len(me.polygons), np.int32)
            a.data.foreach_get('value', idx)
            me.polygons.foreach_set('material_index', idx)
            me.attributes.remove(a)
        while len(me.materials) > n0:
            me.materials.pop(index=len(me.materials) - 1)
        for i, nm in enumerate(names[:len(me.materials)]):
            me.materials[i] = bpy.data.materials.get(nm) if nm else None
        for k in ('af_th_slots', 'af_th_matcount'):
            if k in o:
                del o[k]
    me.update()


def _gn_group(domain):
    name = GROUP[domain]
    ng = bpy.data.node_groups.get(name)
    if ng is not None and ng.nodes:
        return ng
    ng = fresh_group(name)
    add_socket(ng, "Geometry", 'NodeSocketGeometry', 'INPUT')
    add_socket(ng, "Min K", 'NodeSocketFloat', 'INPUT', 288.15, desc="Temperature at the cold end of the ramp")
    add_socket(ng, "Max K", 'NodeSocketFloat', 'INPUT', 340.0, desc="Temperature at the hot end of the ramp")
    add_socket(ng, "Geometry", 'NodeSocketGeometry', 'OUTPUT')
    for nm, desc in (("Min Temperature", "K"), ("Mean Temperature", "K"), ("Max Temperature", "K")):
        add_socket(ng, nm, 'NodeSocketFloat', 'OUTPUT', desc=desc)
    b = Builder(ng)
    gi, go = b.group_io()
    nm_ = b.node('GeometryNodeInputNamedAttribute', data_type='FLOAT', y=120, x=-200)
    inp(nm_, 'Name').default_value = ATTR
    T = out(nm_, 'Attribute')
    mr = b.node('ShaderNodeMapRange', y=120, x=0)
    mr.clamp = True
    b.link(T, inp(mr, 'Value'))
    b.link(out(gi, 'Min K'), inp(mr, 'From Min'))
    b.link(out(gi, 'Max K'), inp(mr, 'From Max'))
    cr = b.node('ShaderNodeValToRGB', y=120, x=200)
    b.link(out(mr, 'Result'), inp(cr, 'Fac'))
    st = TH.RAMPS['INFERNO']
    ramp = cr.color_ramp
    while len(ramp.elements) > 2:
        ramp.elements.remove(ramp.elements[-1])
    ramp.elements[0].position, ramp.elements[0].color = st[0][0], (*st[0][1], 1.0)
    ramp.elements[1].position, ramp.elements[1].color = st[-1][0], (*st[-1][1], 1.0)
    for pos, col in st[1:-1]:
        e = ramp.elements.new(pos)
        e.color = (*col, 1.0)
    dom = 'POINT' if domain == 'POINT' else 'FACE'
    s1 = b.node('GeometryNodeStoreNamedAttribute', data_type='FLOAT', domain=dom, y=0, x=500)
    inp(s1, 'Name').default_value = "af_temperature_norm"
    b.link(out(gi, 'Geometry'), inp(s1, 'Geometry'))
    b.link(out(mr, 'Result'), inp(s1, 'Value'))
    s2 = b.node('GeometryNodeStoreNamedAttribute', data_type='FLOAT_COLOR', domain=dom, y=0, x=720)
    inp(s2, 'Name').default_value = "af_temperature_color"
    b.link(out(s1, 'Geometry'), inp(s2, 'Geometry'))
    b.link(out(cr, 'Color'), inp(s2, 'Value'))
    b.link(out(s2, 'Geometry'), inp(go, 'Geometry'))
    stat = b.node('GeometryNodeAttributeStatistic', data_type='FLOAT', domain=dom, y=-200, x=500)
    b.link(out(gi, 'Geometry'), inp(stat, 'Geometry'))
    b.link(T, inp(stat, 'Attribute'))
    b.link(out(stat, 'Min'), inp(go, 'Min Temperature'))
    b.link(out(stat, 'Mean'), inp(go, 'Mean Temperature'))
    b.link(out(stat, 'Max'), inp(go, 'Max Temperature'))
    return ng


def _sync_modifier(o):
    p = o.arc_flow_th
    mod = o.modifiers.get(MOD_NAME)
    if not (p.on and p.gn):
        if mod is not None:
            o.modifiers.remove(mod)
        return None
    ng = _gn_group(p.domain)
    if mod is None:
        mod = o.modifiers.new(MOD_NAME, 'NODES')
    mod.node_group = ng
    from . import gn
    gn.set_modifier_inputs(mod, {"Min K": p.t_min, "Max K": p.t_max})
    return mod


def _solid_view(o):
    p = o.arc_flow_th
    me = o.data
    if p.solid and p.on:
        c = me.color_attributes.get(COLOR)
        if c is not None:
            me.color_attributes.active_color = c
            me.color_attributes.render_color_index = list(me.color_attributes).index(c)


def refresh_look(o):
    p = o.arc_flow_th
    if not p.on:
        return
    if p.shader:
        _apply_material(o)
    else:
        _restore_material(o)
    _sync_modifier(o)
    _solid_view(o)


def enable(o):
    if o.type != 'MESH':
        return
    p = o.arc_flow_th
    STATE.pop(o.name, None)
    _ensure_attrs(o, p.domain, p.solid)
    refresh_look(o)
    sc = bpy.context.scene
    try:
        update_object(sc, o)
    except Exception as ex:
        FO.RT.msg = "Thermal readback: %s" % ex
    try:
        FO.mark_dirty()
    except Exception:
        pass


def disable(o):
    STATE.pop(o.name, None)
    _restore_material(o)
    m = o.modifiers.get(MOD_NAME)
    if m is not None:
        o.modifiers.remove(m)
    me = o.data
    for nm in (ATTR, MASK):
        a = me.attributes.get(nm)
        if a is not None:
            me.attributes.remove(a)
    c = me.color_attributes.get(COLOR)
    if c is not None:
        me.color_attributes.remove(c)
    mt = bpy.data.materials.get("Tessera Thermal %s" % o.name)
    if mt is not None and mt.users == 0:
        bpy.data.materials.remove(mt)
    me.update()


def set_selection(o):
    """Faces selected in Edit Mode become the readback surface; in Object Mode the whole object."""
    import bmesh
    me = o.data
    if o.mode == 'EDIT':
        bm = bmesh.from_edit_mesh(me)
        lay = bm.faces.layers.int.get(MASK) or bm.faces.layers.int.new(MASK)
        sel = 0
        for f in bm.faces:
            f[lay] = 1 if f.select else 0
            sel += int(f.select)
        bmesh.update_edit_mesh(me)
        return sel
    a = me.attributes.get(MASK)
    if a is not None:
        me.attributes.remove(a)
    return len(me.polygons)


# ----------------------------------------------------------------------------- operators and panel
class AF_OT_thermal_select(Operator):
    bl_idname = "arcflow.thermal_select"
    bl_label = "Use Selected Faces"
    bl_description = "Edit Mode: the selected faces become the thermal readback surface. Object Mode: the whole object"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        o = ctx.object
        if o is None or o.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh object")
            return {'CANCELLED'}
        n = set_selection(o)
        p = o.arc_flow_th
        if o.mode != 'EDIT':
            if not p.on:
                p.on = True
            else:
                STATE.pop(o.name, None)
                refresh_look(o)
                update_all(ctx.scene)
        self.report({'INFO'}, "%d face(s) selected for thermal readback" % n)
        return {'FINISHED'}


class AF_OT_thermal_update(Operator):
    bl_idname = "arcflow.thermal_update"
    bl_label = "Update Thermal Readback"
    bl_description = "Recompute the surface temperature for the current frame"

    def execute(self, ctx):
        for o in active_objects(ctx.scene):
            STATE.pop(o.name, None)
            refresh_look(o)
        n = update_all(ctx.scene)
        self.report({'INFO'}, "Updated %d object(s)" % n)
        return {'FINISHED'}


class AF_PT_thermal(Panel):
    bl_idname = "AF_PT_thermal"
    bl_label = "Thermal readback (active object)"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Tessera"
    bl_parent_id = "AF_PT_fm_main"

    @classmethod
    def poll(cls, ctx):
        fm = getattr(ctx.scene, 'arc_flow_fm', None)
        return fm is not None and fm.mode == 'FLUID' and ctx.object is not None and ctx.object.type == 'MESH'

    def draw(self, ctx):
        l = self.layout
        o = ctx.object
        p = o.arc_flow_th
        l.prop(p, "on", toggle=True, icon='LIGHT_SUN')
        if not p.on:
            l.label(text="Surface temperature from the fluid physics", icon='INFO')
            return
        l.operator("arcflow.thermal_select", icon='FACESEL')
        col = l.column(align=True)
        col.prop(p, "domain", expand=True)
        r = col.row(align=True)
        r.prop(p, "t_min"); r.prop(p, "t_max")
        col.prop(p, "ramp")
        col = l.column(align=True)
        col.prop(p, "shader"); col.prop(p, "solid"); col.prop(p, "gn"); col.prop(p, "couple")
        if p.shader:
            col.prop(p, "emission")
        box = l.box()
        box.label(text="Wall model")
        box.prop(p, "wall_heat"); box.prop(p, "absorb"); box.prop(p, "emissivity"); box.prop(p, "recovery"); box.prop(p, "probe"); box.prop(p, "speedup")
        st = STATE.get(o.name)
        if st is not None and st.Ts is not None and len(st.Ts):
            l.label(text="Now: min %.1f  mean %.1f  max %.1f K" % (st.Ts.min(), st.Ts.mean(), st.Ts.max()), icon='INFO')
        l.operator("arcflow.thermal_update", icon='FILE_REFRESH')


CLASSES = (AF_ThermalObj, AF_OT_thermal_select, AF_OT_thermal_update, AF_PT_thermal)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Object.arc_flow_th = PointerProperty(type=AF_ThermalObj)
    if on_frame_change not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(on_frame_change)
    _install_coupling()
    try:
        from . import tessera_nodes
        for dom in ('POINT', 'FACE'):
            tessera_nodes.register_group(GROUP[dom], (lambda d=dom: _gn_group(d)),
                                         "Surface temperature of a collision body (%s domain): normalised value, ramp colour, min / mean / max" % dom.lower())
    except Exception:
        pass


def unregister():
    _remove_coupling()
    if on_frame_change in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(on_frame_change)
    del bpy.types.Object.arc_flow_th
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
