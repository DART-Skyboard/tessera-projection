"""Wind tunnel scene builder (Tessera Projection, Fluid mode).

Builds a hollow tunnel (a box with a real wall thickness) that is the Environment container of the physics, a wind emitter at the inlet,
a gas fluid that fills the test section and flows with the wind, and takes any aircraft (or other) collision bodies.  The tunnel's
atmosphere is the Blender scene, an assigned container, or a preset (Earth, Mars, Venus, Titan, Moon, vacuum, custom gas).
Objects flagged as *plain collision bodies* (an item left on the table) only respond to the fluid: they take no heat source, no
thermal readback and no Systems role.  Moving parts (object / shape-key / armature animation) are moving colliders; deforming
bodies re-project their collision field every frame.  Independent engines (parts with Container / Channel / Chamber roles) inside the
tunnel keep running as their own Systems next to the wind (the Systems panel detects them)."""
import math

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty
from bpy.types import Operator, Panel, PropertyGroup

from . import fluidmode_ops as FO
from .core import atmosphere as ATM
from .core import fluidmode as FM

COLLECTION = "WindTunnel"
TUNNEL, AIRVOL, EMITTER = "Wind Tunnel", "Tunnel Air Volume", "Wind Emitter"
AIR_NAME = "Tunnel Air"
R_GAS = 8.314462618
MAX_WIND = 1200.0        # high-speed / transonic tunnel (m/s); the particle clamp follows 1.5 x wind, see build_solver_wide

ATMOSPHERES = [('EARTH', "Earth", "1 atm, 288 K, 1 g, nitrogen / oxygen"), ('MARS', "Mars", "610 Pa carbon dioxide"),
               ('VENUS', "Venus", "92 bar, 737 K carbon dioxide"), ('TITAN', "Titan", "1.45 bar, 94 K nitrogen / methane"),
               ('MOON', "Moon", "Vacuum with lunar gravity: no gas flow"), ('VACUUM', "Vacuum", "No gas: only radiation and the walls"),
               ('CUSTOM', "Custom gas", "Your own pressure, temperature, gravity and gas")]
MOON = dict(pressure=3e-10, temperature=250.0, gravity=1.62)


def _upd(self, ctx):
    pass


class AF_WTScene(PropertyGroup):
    source: EnumProperty(name="Environment", items=[('SCENE', "Blender 3D scene", "Open scene: gravity of the scene, the Environment card as it is"),
                                                    ('CONTAINER', "Assigned container", "The physics lives inside an object you pick (room, vacuum chamber, your own tunnel)"),
                                                    ('PRESET', "Atmosphere preset", "A planet or a custom gas inside the tunnel")], default='PRESET')
    container: PointerProperty(name="Container", type=bpy.types.Object, poll=lambda s, o: o.type == 'MESH')
    atmosphere: EnumProperty(name="Atmosphere", items=ATMOSPHERES, default='EARTH')
    pressure: FloatProperty(name="Pressure", default=101325.0, min=0.0, description="Pa (custom gas)")
    temperature: FloatProperty(name="Temperature", default=293.15, min=1.0, description="K (custom gas)")
    gravity: FloatProperty(name="Gravity", default=9.80665, min=0.0, description="m/s2 (custom gas)")
    gas: EnumProperty(name="Gas", items=ATM.COMPOUND_ITEMS, default='N2', description="Compound of the custom gas")
    length: FloatProperty(name="Length", default=4.5, min=0.5, unit='LENGTH')
    width: FloatProperty(name="Width", default=1.8, min=0.3, unit='LENGTH')
    height: FloatProperty(name="Height", default=1.5, min=0.3, unit='LENGTH')
    wall: FloatProperty(name="Wall thickness", default=0.06, min=0.005, unit='LENGTH', description="Real thickness of the tunnel shell")
    open_ends: BoolProperty(name="Open inlet and outlet", default=True, description="Off: a closed box")
    wind: FloatProperty(name="Wind speed", default=14.0, min=0.0, max=MAX_WIND, description="m/s along +X. The particle speed clamp follows the wind (1.5 x). High speeds (hundreds of m/s) show aerodynamic friction heating on leading edges")
    count: IntProperty(name="Air parcels", default=9000, min=500, max=60000)
    emit: BoolProperty(name="Continuous wind", default=True, description="The Wind Emitter (an Emission surface at the inlet) keeps feeding fresh air into the tunnel")
    emit_cap: FloatProperty(name="Emission cap", default=1500.0, min=0.0, description="Most parcels per second the wind emitter adds (keeps the run time in check)")
    blend_airflow: BoolProperty(name="Blend airflow volume", default=True,
                                description="Draw the tunnel air as a continuous volumetric slipstream / heated boundary layer around collision bodies, not as sparse streaks")
    aero_heat: BoolProperty(name="Friction heating", default=True,
                            description="Compute aerodynamic friction and stagnation heating on tunnel collision bodies (tessera_temp_k / tessera_friction_heat)")
    status: StringProperty(default="")


class AF_WTObject(PropertyGroup):
    plain: BoolProperty(name="Plain collision body", default=False,
                        description="An object that merely responds to the fluid (an item left on the table): it is a collider only, with no heat source, no thermal readback and no Systems role",
                        update=lambda s, c: _plain_changed(s.id_data))
    body: BoolProperty(name="Tunnel collision body", default=False, description="Marked as a body inside the tunnel (aircraft, model, part)")


def _plain_changed(o):
    if o.get('af_wt_plain') == o.arc_flow_wt.plain:
        return
    o['af_wt_plain'] = bool(o.arc_flow_wt.plain)
    if o.arc_flow_wt.plain:
        o.arc_flow_fm.collider = True
        o.arc_flow_fm.heat_on = False
        if getattr(o, 'arc_flow_th', None) is not None and o.arc_flow_th.on:
            o.arc_flow_th.on = False
        try:
            o.arc_flow.role = 'NONE'
        except Exception:
            pass


# ----------------------------------------------------------------------------- environment
def gas_state(scene):
    """(pressure Pa, temperature K, gravity m/s2, mix dict or None) for the tunnel's atmosphere settings."""
    wt = scene.arc_flow_wt
    k = wt.atmosphere
    if k in ATM.PRESETS:
        p = ATM.PRESETS[k]
        return p['pressure'], p['temperature'], p['gravity'], dict(p['mix'])
    if k == 'MOON':
        return MOON['pressure'], MOON['temperature'], MOON['gravity'], {'N2': 1.0}
    if k == 'VACUUM':
        return 0.0, 293.15, abs(scene.arc_flow.env.gravity[2]) or 9.80665, {'N2': 1.0}
    return wt.pressure, wt.temperature, wt.gravity, {wt.gas: 1.0}


def apply_environment(scene):
    """Write the Environment card from the tunnel settings (scene / container / preset)."""
    wt = scene.arc_flow_wt
    e = scene.arc_flow.env
    e.wind = (wt.wind, 0.0, 0.0)
    e.ground = False
    if wt.source == 'SCENE':
        e.source = 'WORLD'
        return "Blender 3D scene"
    if wt.source == 'CONTAINER':
        e.source = 'OBJECT'
        e.obj = wt.container
        return "container %s" % (wt.container.name if wt.container else "(none)")
    P, T, g, mix = gas_state(scene)
    e.source = 'OBJECT' if bpy.data.objects.get(TUNNEL) else 'WORLD'
    if e.source == 'OBJECT':
        e.obj = bpy.data.objects.get(TUNNEL)
    if wt.atmosphere in ATM.PRESETS:
        e.preset = wt.atmosphere
    else:
        e.preset = 'CUSTOM'
    e.pressure, e.temperature = P, T
    e.use_scene_gravity = False
    e.gravity = (0.0, 0.0, -g)
    e.use_mix = True
    e.mix.clear()
    for cid, fr in mix.items():
        c = e.mix.add()
        c.compound, c.fraction = cid, fr
    return dict(ATMOSPHERES_D)[wt.atmosphere]


ATMOSPHERES_D = {k: v for k, v, _d in ATMOSPHERES}


def air_properties(scene):
    """Properties of the gas for the Air fluid: density, viscosity, cp, k from the tunnel atmosphere (ideal gas, Sutherland, Pr 0.71)."""
    wt = scene.arc_flow_wt
    P, T, g, mix = gas_state(scene)
    if wt.source == 'SCENE':
        e = scene.arc_flow.env
        P, T = e.pressure, e.temperature
        mix = {c.compound: c.fraction for c in e.mix} if e.use_mix and len(e.mix) else {'N2': 0.7808, 'O2': 0.2095, 'AR': 0.0097}
    st = ATM.state(mix, P, T, g)
    cp = R_GAS * st['gamma'] / (st['gamma'] - 1.0) / st['molar']
    return dict(density=max(st['rho'], 1e-9), visc=st['mu'], cp=cp, k=cp * st['mu'] / 0.71, temperature=T, pressure=P)


# ----------------------------------------------------------------------------- geometry
def _box_shell(L, W, H, t, open_ends):
    """Hollow box: outer and inner surfaces, normals pointing away from the wall material.  Axis X is the flow direction."""
    bm = bmesh.new()
    o = [(sx * L / 2, sy * W / 2, sz * H / 2) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    i = [(sx * (L / 2 - (0 if open_ends else t)), sy * (W / 2 - t), sz * (H / 2 - t)) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    ov = [bm.verts.new(p) for p in o]
    iv = [bm.verts.new(p) for p in i]

    # vertex index = sx*4 + sy*2 + sz  (0/1 for -/+); every face is oriented explicitly below
    side = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]   # x-, x+, y-, y+, z-, z+
    sides = side[2:] if open_ends else side
    outer, inner, rim = [], [], []
    for a, b, c, d in sides:
        outer.append(bm.faces.new((ov[a], ov[b], ov[c], ov[d])))
        inner.append(bm.faces.new((iv[a], iv[b], iv[c], iv[d])))
    if open_ends:      # rim of the open ends joins outer and inner surfaces
        for xs in (0, 1):
            for a, b in [(0, 1), (1, 3), (3, 2), (2, 0)]:
                rim.append((xs, bm.faces.new((ov[xs * 4 + a], ov[xs * 4 + b], iv[xs * 4 + b], iv[xs * 4 + a]))))
    bm.normal_update()
    ctr = Vector((0.0, 0.0, 0.0))
    for f in outer:
        if f.normal.dot(f.calc_center_median() - ctr) < 0:
            f.normal_flip()
    for f in inner:
        if f.normal.dot(f.calc_center_median() - ctr) > 0:
            f.normal_flip()
    for xs, f in rim:
        want = -1.0 if xs == 0 else 1.0
        if f.normal.x * want < 0:
            f.normal_flip()
    return bm


def _plain_box(lo, hi):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co = (lo[0] if v.co.x < 0 else hi[0], lo[1] if v.co.y < 0 else hi[1], lo[2] if v.co.z < 0 else hi[2])
    return bm


def _collection(scene):
    c = bpy.data.collections.get(COLLECTION)
    if c is None:
        c = bpy.data.collections.new(COLLECTION)
    if c.name not in scene.collection.children:
        scene.collection.children.link(c)
    return c


def _mesh_obj(name, bm, col):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = False
    o = bpy.data.objects.new(name, me)
    col.objects.link(o)
    o['af_wt'] = name
    return o


def _remove(name):
    o = bpy.data.objects.get(name)
    if o is not None:
        me = o.data
        bpy.data.objects.remove(o, do_unlink=True)
        if me is not None and me.users == 0:
            bpy.data.meshes.remove(me)


def _tunnel_material():
    m = bpy.data.materials.get("Tunnel Glass") or bpy.data.materials.new("Tunnel Glass")
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes.get('Principled BSDF')
    if b is not None:
        b.inputs['Base Color'].default_value = (0.62, 0.75, 0.82, 1.0)
        b.inputs['Roughness'].default_value = 0.08
        for nm, val in (('Transmission Weight', 0.9), ('IOR', 1.45), ('Alpha', 0.25)):
            if nm in b.inputs:
                b.inputs[nm].default_value = val
    m.diffuse_color = (0.62, 0.75, 0.82, 0.25)
    try:
        m.blend_method = 'BLEND'
    except Exception:
        pass
    return m


def build_tunnel(scene):
    """Create (or rebuild) the tunnel shell, the air volume, the emitter and the Air fluid.  Returns the shell object."""
    wt = scene.arc_flow_wt
    fm = scene.arc_flow_fm
    fm.mode = 'FLUID'
    col = _collection(scene)
    for n in (TUNNEL, AIRVOL, EMITTER):
        _remove(n)
    L, W, H, t = wt.length, wt.width, wt.height, wt.wall
    z0 = H / 2 + 0.0
    shell = _mesh_obj(TUNNEL, _box_shell(L, W, H, t, wt.open_ends), col)
    shell.location.z = z0
    shell.data.materials.append(_tunnel_material())
    shell.display_type = 'WIRE' if False else 'TEXTURED'
    shell.visible_shadow = False
    shell.arc_flow.role = 'NONE'
    shell.arc_flow.cavity_mode = 'SOLID'
    shell.arc_flow_fm.collider = True
    shell.arc_flow_fm.friction = 0.1
    shell.arc_flow_wt.plain = False
    inner = (L / 2 - (0 if wt.open_ends else t), W / 2 - t, H / 2 - t)
    air = _mesh_obj(AIRVOL, _plain_box((-inner[0] + 0.02, -inner[1] + 0.02, -inner[2] + 0.02), (inner[0] - 0.02, inner[1] - 0.02, inner[2] - 0.02)), col)
    air.location.z = z0
    air.display_type = 'WIRE'
    air.hide_render = True
    # wind emitter: a plane just inside the inlet, normal along the flow
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.5)
    for v in bm.verts:
        v.co = (0.0, v.co.x * 2 * (inner[1] - 0.03), v.co.y * 2 * (inner[2] - 0.03))
    em = _mesh_obj(EMITTER, bm, col)
    em.location = (-inner[0] + 0.03, 0.0, z0)
    em.display_type = 'WIRE'
    em.hide_render = True
    em['af_wt_wind'] = float(wt.wind)
    emitter_source(scene, em, inner)
    # the Air fluid
    fl = next((f for f in fm.fluids if f.name == AIR_NAME), None)
    if fl is None:
        fl = FO.add_fluid(scene, 'AIR', AIR_NAME)
    ap = air_properties(scene)
    P = ap['pressure']
    no_gas = P < 1.0 or wt.atmosphere in ('MOON', 'VACUUM') and wt.source == 'PRESET'
    fl.kind = 'GAS'
    fl.no_container = True
    fl.shape = air
    fl.density, fl.visc, fl.cp, fl.k = ap['density'], ap['visc'], ap['cp'], ap['k']
    fl.temperature = ap['temperature']
    fl.velocity = (wt.wind, 0.0, 0.0)
    fl.count = 8 if no_gas else wt.count
    fl.color = (0.85, 0.92, 1.0)
    fl.render_mode = 'POINTS'
    fl.particle_scale = 0.55
    fl.drag = -1.0
    fl.cool = 0.02
    fm.max_particles = max(fm.max_particles, wt.count)
    fm.gust = 0.0
    desc = apply_environment(scene)
    FO.mark_dirty()
    wt.status = "Tunnel %.1f x %.1f x %.1f m, wall %.0f mm, %s, %s, %.0f m/s%s" % (
        L, W, H, t * 1000, "open ends" if wt.open_ends else "closed", desc, wt.wind, " (no gas: vacuum)" if no_gas else "")
    return shell


def emitter_source(scene, em, inner):
    """Make the Wind Emitter plane an Emission surface (surfaces_ops): fresh air enters at the inlet with the wind speed.  The rate keeps the parcel
    density of the initial air slug as far as `Emission cap` allows."""
    wt = scene.arc_flow_wt
    try:
        from . import surfaces_ops as SO
    except Exception:
        return None
    s = SO.srf(scene)
    for i in range(len(s.items) - 1, -1, -1):
        if s.items[i].name == "Wind":
            s.items.remove(i)
    if not wt.emit or wt.wind <= 0.0:
        return None
    area = 4.0 * (inner[1] - 0.03) * (inner[2] - 0.03)
    vp = (2 * inner[0] * 2 * inner[1] * 2 * inner[2]) / max(wt.count, 1)
    rate = min(wt.wind * area / max(vp, 1e-9), wt.emit_cap)
    e = s.items.add()
    e.kind, e.name, e.obj = 'EMISSION', "Wind", em
    e.preset = 'AIR'
    e.rate, e.speed, e.spread = float(rate), float(wt.wind), 1.5
    e.max_particles = int(wt.emit_cap * 8)
    e.temperature = air_properties(scene)['temperature']
    SO.store_faces(e, np.ones(len(em.data.polygons), bool))
    SO.sync_source(scene, e)
    fl = SO.fluid_of(scene, e)
    if fl is not None:
        ap = air_properties(scene)
        fl.density, fl.visc, fl.cp, fl.k = ap['density'], ap['visc'], ap['cp'], ap['k']
        fl.color = (0.85, 0.92, 1.0)
        fl.render_mode = 'POINTS'
        fl.particle_scale = 0.55
    try:
        from . import surfaces_gn as SGN
        SGN.ensure_modifier(em, e)
    except Exception:
        pass
    return e


def mark_body(o, plain=False, thermal=False, heat=None):
    """Make `o` a collision body of the tunnel.  plain: responds only.  thermal: thermal readback on its surface."""
    o.arc_flow_wt.body = True
    o.arc_flow_fm.collider = True
    from . import asset_colliders as AC
    AC.mark_projection_collider(o)
    if plain:
        o.arc_flow_wt.plain = True
    elif thermal and getattr(o, 'arc_flow_th', None) is not None:
        setup_thermal_body(bpy.context.scene, o)
    return o


def stagnation_rise(scene, wind=None):
    """Full adiabatic stagnation temperature rise v^2 / 2cp (K) of the tunnel air at `wind` m/s."""
    ap = air_properties(scene)
    v = float(scene.arc_flow_wt.wind if wind is None else wind)
    return v * v / (2.0 * max(ap['cp'], 1.0)), ap['temperature']


def setup_thermal_body(scene, o, wind=None):
    """Thermal feedback on a tunnel collision body, set up by the add-on itself.

    The wall temperature is integrated every frame from the tunnel air (Fluid-mode parcels) plus
    boundary-layer friction / stagnation heating of the tunnel wind, in real time (time scale 1,
    2 mm aluminium skin).  The look keeps the object's paint and mixes in a warm tint; the range
    tops out at twice the full stagnation rise of the tunnel air, so a fully soaked leading edge
    lands mid-range and nothing saturates, with a 6 K floor so a slow tunnel is not amplified into
    false colour.  Incandescence comes only from the real wall temperature (above ~750 K)."""
    p = o.arc_flow_th
    dT, T0 = stagnation_rise(scene, wind)
    p.aero = True
    p.overlay = True
    p.shader = True
    p.solid = True
    p.couple = False
    p.speedup = 1.0
    p.wall_heat = 4900.0
    p.emissivity = 0.3
    p.recovery = 0.85
    p.emission = 0.3
    p.t_min, p.t_max = T0, T0 + max(2.0 * dT, 6.0)
    if p.on:
        from . import thermal as TM
        TM.STATE.pop(o.name, None)
        TM.refresh_look(o)
    else:
        p.on = True
    return o


# ----------------------------------------------------------------------------- deforming colliders
def _deforming(o):
    if o.type != 'MESH':
        return False
    if o.data.shape_keys is not None and len(o.data.shape_keys.key_blocks) > 1:
        return True
    return any(m.type in ('ARMATURE', 'LATTICE', 'SIMPLE_DEFORM', 'CURVE', 'HOOK', 'WARP', 'WAVE') for m in o.modifiers)


def _refresh_deformers(scene):
    """Rebuild the collision field of colliders whose mesh deforms (shape keys, armature...) when it changed since the last step."""
    S = FO.RT.solver
    if S is None:
        return 0
    from . import runtime
    deps = bpy.context.evaluated_depsgraph_get()
    n = 0
    for b, o in zip(S.bodies, FO.RT.body_objs):
        if not _deforming(o):
            continue
        try:
            v, t = runtime._mesh_arrays(o, deps)
        except Exception:
            continue
        sig = (round(float(v.sum()), 9), round(float((v * v).sum()), 9))
        if getattr(b, '_wt_sig', None) == sig:
            continue
        fld, _v, _t = runtime._field(o, deps)
        b.field = fld
        s_ = fld.soup
        edge = float(max(np.linalg.norm(s_.B - s_.A, axis=1).max(), np.linalg.norm(s_.C - s_.B, axis=1).max())) if len(s_) else 0.0
        b.bricks_ok = edge < 6.0 * fld.h
        b._wt_sig = sig
        n += 1
    return n


_orig_sim_frame = None


def sim_frame_deforming(scene, frame):
    try:
        if FO.RT.solver is not None and not FO.RT.dirty:
            _refresh_deformers(scene)
    except Exception as ex:
        FO.RT.msg = "Deforming collider: %s" % ex
    return _orig_sim_frame(scene, frame)


_orig_build = None


def build_solver_wide(scene):
    S = _orig_build(scene)
    try:
        w = float(np.linalg.norm(scene.arc_flow.env.wind))
        if S is not None and any(f.name == AIR_NAME for f in scene.arc_flow_fm.fluids):
            S.vmax = max(float(S.vmax), 1.5 * w)      # wind-tunnel scenes: the particle speed clamp follows the wind
    except Exception:
        pass
    return S


def _install():
    global _orig_sim_frame, _orig_build
    if FO.build_solver is not build_solver_wide:
        _orig_build = FO.build_solver
        FO.build_solver = build_solver_wide
    if FO.sim_frame is not sim_frame_deforming:
        _orig_sim_frame = FO.sim_frame
    FO.sim_frame = sim_frame_deforming


def _remove_hook():
    if FO.build_solver is build_solver_wide:
        FO.build_solver = _orig_build
    if FO.sim_frame is sim_frame_deforming:
        FO.sim_frame = _orig_sim_frame


# ----------------------------------------------------------------------------- aircraft example
def _asset_mesh(name, data, col, smooth=True):
    v = np.asarray(data['verts'], float) - np.asarray(data['origin'], float)
    me = bpy.data.meshes.new(name)
    me.from_pydata(v.tolist(), [], np.asarray(data['faces']).tolist())
    me.update()
    if smooth:
        for p in me.polygons:
            p.use_smooth = True
    o = bpy.data.objects.new(name, me)
    o.location = tuple(data['origin'])
    col.objects.link(o)
    return o


def _paint(o, rgb, metal=0.6, rough=0.35):
    m = bpy.data.materials.new(o.name + " Paint")
    m.use_nodes = True
    b = m.node_tree.nodes.get('Principled BSDF')
    if b is not None:
        b.inputs['Base Color'].default_value = (*rgb, 1.0)
        b.inputs['Metallic'].default_value = metal
        b.inputs['Roughness'].default_value = rough
    m.diffuse_color = (*rgb, 1.0)
    o.data.materials.append(m)


def add_aircraft(scene, location=None, thermal=True, spin_frames=(1, 250), rpm=1500.0):
    """The example tilt-rotor VTOL (procedural stand-in): airframe plus two spinning rotors and two exhaust heat sources."""
    from .assets import vtol
    col = _collection(scene)
    parts = vtol.build()
    wt = scene.arc_flow_wt
    loc = tuple(location) if location is not None else (0.0, 0.0, wt.height / 2 - 0.02)
    air = _asset_mesh("Aircraft Airframe", parts['Airframe'], col)
    _paint(air, (0.78, 0.80, 0.84), 0.7, 0.30)
    rotors = []
    for nm, key in (("Aircraft Rotor R", 'Rotor R'), ("Aircraft Rotor L", 'Rotor L')):
        r = _asset_mesh(nm, parts[key], col, smooth=False)
        _paint(r, (0.12, 0.12, 0.13), 0.2, 0.5)
        r.parent = air
        r.matrix_parent_inverse = air.matrix_world.inverted()
        rotors.append(r)
        r.rotation_mode = 'XYZ'
        r.rotation_euler = (0.0, 0.0, 0.0)
        r.keyframe_insert('rotation_euler', index=0, frame=spin_frames[0])
        r.rotation_euler[0] = 2 * math.pi * rpm / 60.0 * (spin_frames[1] - spin_frames[0]) / (scene.render.fps / scene.render.fps_base)
        r.keyframe_insert('rotation_euler', index=0, frame=spin_frames[1])
        _linear(r)
    air.location = loc
    air.rotation_euler = (0.0, math.radians(-4.0), 0.0)
    heats = []
    for i, p in enumerate(vtol.EXHAUSTS):
        e = bpy.data.objects.new("Engine Exhaust %s" % ("R" if i == 0 else "L"), None)
        e.empty_display_type = 'SPHERE'
        e.empty_display_size = 0.04
        col.objects.link(e)
        e.parent = air
        e.matrix_parent_inverse = air.matrix_world.inverted()
        e.location = tuple(p)          # parent inverse is identity (the airframe was still at the origin when parented)
        e.arc_flow_fm.heat_on = True
        e.arc_flow_fm.heat_power = 2500.0
        e.arc_flow_fm.heat_radius = 0.12
        heats.append(e)
    nozzles = []
    try:
        from . import surfaces_ops as SO
        for i, p in enumerate(vtol.EXHAUSTS):
            side = "R" if i == 0 else "L"
            bm = bmesh.new()
            bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.5)
            for v in bm.verts:
                v.co = (0.0, v.co.x * 0.07, v.co.y * 0.07)
            nz = _mesh_obj("Engine Nozzle %s" % side, bm, col)
            nz.parent = air
            nz.location = (p[0] - 0.035, p[1], p[2])
            nz.hide_render = True
            nz.display_type = 'WIRE'
            sr = SO.srf(scene)
            e = sr.items.add()
            e.kind, e.name, e.obj, e.preset = 'EMISSION', "Exhaust %s" % side, nz, 'AIR'
            e.rate, e.speed, e.spread, e.temperature = 500.0, 22.0, 6.0, 460.0
            e.max_particles = 3000
            SO.store_faces(e, np.ones(len(nz.data.polygons), bool))
            SO.sync_source(scene, e)
            fl = SO.fluid_of(scene, e)
            if fl is not None:
                fl.color = (1.0, 0.7, 0.4)
                fl.render_mode = 'POINTS'
                fl.particle_scale = 0.5
            try:
                from . import surfaces_gn as SGN
                SGN.ensure_modifier(nz, e)
            except Exception:
                pass
            nozzles.append(nz)
    except Exception as ex:
        FO.RT.msg = "Engine exhaust emitters: %s" % ex
    for o in [air] + rotors:
        mark_body(o)
    air.arc_flow_fm.friction = 0.15
    for r in rotors:
        r.arc_flow_fm.friction = 0.6
    if thermal:
        air.arc_flow_th.t_min, air.arc_flow_th.t_max = 288.15, 345.0
        air.arc_flow_th.speedup = 60.0
        air.arc_flow_th.wall_heat = 600.0
        air.arc_flow_th.on = True
    return air, rotors, heats


def _linear(o):
    ad = o.animation_data
    if ad is None or ad.action is None:
        return
    try:
        fcs = ad.action.fcurves
    except Exception:
        fcs = []
        try:
            for layer in ad.action.layers:
                for strip in layer.strips:
                    for cb in strip.channelbags:
                        fcs += list(cb.fcurves)
        except Exception:
            pass
    for fc in fcs:
        for kp in fc.keyframe_points:
            kp.interpolation = 'LINEAR'


def load_example(scene, with_thermal=True):
    """Example: Wind tunnel.  Clears the previous tunnel objects and builds tunnel, air, aircraft, thermal readback."""
    for n in ("Aircraft Rotor R", "Aircraft Rotor L", "Aircraft Airframe", "Engine Exhaust R", "Engine Exhaust L", "Engine Nozzle R", "Engine Nozzle L"):
        _remove(n)
    wt = scene.arc_flow_wt
    wt.source, wt.atmosphere = 'PRESET', 'EARTH'
    wt.wind = 14.0
    fm = scene.arc_flow_fm
    fm.mode = 'FLUID'
    build_tunnel(scene)
    add_aircraft(scene, thermal=with_thermal)
    fm.speedup = 1.0
    scene.frame_start, scene.frame_end = 1, 90
    FO.mark_dirty()
    return wt.status



AIRFLOW_NAME = "Tessera Airflow Volume"


def add_fighter(scene, location=None, thermal=True, wind=None):
    """Insert the procedural fighter airframe as a tunnel collision body, with aero friction heating on its skin."""
    from .assets import fighter as FG
    from . import thermal as TM
    col = _collection(scene)
    # remove a previous fighter if the operator is clicked twice
    prev = bpy.data.objects.get("Fighter Airframe")
    if prev is not None:
        _remove("Fighter Airframe")
    parts = FG.build()
    wt = scene.arc_flow_wt
    loc = tuple(location) if location is not None else (0.0, 0.0, wt.height / 2 + 0.02)
    air = _asset_mesh("Fighter Airframe", parts, col)
    _paint(air, (0.42, 0.55, 0.68), 0.72, 0.28)
    air.location = loc
    air.rotation_euler = (0.0, math.radians(-1.5), 0.0)
    mark_body(air)
    air.arc_flow_fm.friction = 0.18
    if thermal:
        setup_thermal_body(scene, air, wind=wind)
        if wt.aero_heat:
            # immediate readout before the tunnel air runs: one second of real-time wall heating
            TM.apply_aero_heat(scene, air, v_inf=(float(wt.wind if wind is None else wind), 0.0, 0.0),
                               steps=24, speedup=air.arc_flow_th.speedup)
    if wt.blend_airflow:
        build_airflow_volume(scene, air)
    return air


def load_fighter_example(scene, wind=420.0, with_thermal=True):
    """High-speed fighter wind-tunnel example: aerodynamic friction heating + blended slipstream."""
    for n in ("Aircraft Rotor R", "Aircraft Rotor L", "Aircraft Airframe", "Engine Exhaust R", "Engine Exhaust L",
              "Engine Nozzle R", "Engine Nozzle L", "Fighter Airframe", AIRFLOW_NAME):
        _remove(n)
    wt = scene.arc_flow_wt
    wt.source, wt.atmosphere = 'PRESET', 'EARTH'
    wt.wind = float(wind)
    wt.blend_airflow = True
    wt.aero_heat = True
    wt.length, wt.width, wt.height = 5.0, 2.2, 1.8
    fm = scene.arc_flow_fm
    fm.mode = 'FLUID'
    build_tunnel(scene)
    # hide sparse air parcels when the blended volume is on
    fl = next((f for f in fm.fluids if f.name == AIR_NAME), None)
    if fl is not None and wt.blend_airflow:
        fl.render_mode = 'POINTS'
        fl.particle_scale = 0.15
    air = add_fighter(scene, thermal=with_thermal, wind=wind)
    fm.speedup = 1.0
    scene.frame_start, scene.frame_end = 1, 90
    FO.mark_dirty()
    return wt.status


def build_airflow_volume(scene, body, cell=0.045, delta=0.07):
    """Continuous volumetric slipstream and heated boundary layer around a collision body."""
    from . import thermal as TM
    from .core import fluid_blend as FB
    from .core import friction_heat as FH
    deps = bpy.context.evaluated_depsgraph_get()
    domain = body.arc_flow_th.domain if getattr(body, 'arc_flow_th', None) else 'POINT'
    P, N, area, sel, size, L = TM.sample_geometry(body, deps, domain)
    st = TM.STATE.get(body.name)
    Ts = None
    if st is not None and st.Ts is not None and len(st.Ts) == len(P):
        Ts = st.Ts
    elif 'tessera_temp_k' in body.data.attributes:
        a = body.data.attributes['tessera_temp_k']
        arr = np.empty(len(a.data), np.float32)
        a.data.foreach_get('value', arr)
        if len(arr) >= (sel.max() + 1 if len(sel) else 0):
            Ts = arr[sel].astype(float)
    wt = scene.arc_flow_wt
    v_inf = (float(wt.wind), 0.0, 0.0)
    T_inf = float(scene.arc_flow.env.temperature)
    # bounds: a box around the body, padded downstream for the wake
    if len(P) == 0:
        return None
    lo = P.min(0) - np.array([0.35, 0.45, 0.35])
    hi = P.max(0) + np.array([1.8, 0.45, 0.45])
    field = FB.airflow_field(P, N, body_T=Ts, v_inf=v_inf, T_inf=T_inf, bounds=(lo, hi), cell=cell, delta=delta)
    o = _airflow_object(scene, field, T_inf, body.arc_flow_th.t_max if getattr(body, 'arc_flow_th', None) else T_inf + 80.0)
    try:
        build_airflow_wisps(scene, P, N, v_inf, (lo, hi), T_inf)
    except Exception:
        pass
    return o



WISP_NAME = "Tessera Airflow Wisps"


def build_airflow_wisps(scene, body_P, body_N, v_inf, bounds, T_inf=288.15):
    """Sharp, twisting slipstream tendrils sourced from curves (few particles, high detail)."""
    from .core import fluid_blend as FB
    wis = FB.wispy_streamlines(body_P, body_N, v_inf=v_inf, bounds=bounds, n_curves=32, steps=80)
    field = dict(P=wis['P'], dens=wis['dens'], temp=np.full(len(wis['P']), float(T_inf)) if len(wis['P']) else wis.get('temp', np.zeros(0)), h=0.03)
    if not len(field['P']):
        return None
    # temporarily retarget the volume object builder onto the wisp name
    global AIRFLOW_NAME
    prev, AIRFLOW_NAME = AIRFLOW_NAME, WISP_NAME
    try:
        o = _airflow_object(scene, field, T_inf, T_inf + 40.0)
        # a slightly tighter, brighter cyan filament look
        try:
            from . import gn
            mr = o.modifiers.get("AF Render")
            if mr is not None:
                gn.set_modifier_inputs(mr, {"Density": 1.6, "Voxel Size": 0.028, "Radius Scale": 0.85})
        except Exception:
            pass
        return o
    finally:
        AIRFLOW_NAME = prev


def _airflow_object(scene, field, T_inf, T_hot):

    from . import gn
    name = AIRFLOW_NAME
    o = bpy.data.objects.get(name)
    n = len(field['P'])
    me = bpy.data.meshes.get(name)
    if me is None or o is None or len(me.vertices) != n:
        old = me
        me = bpy.data.meshes.new(name + ".tmp")
        if n:
            me.vertices.add(n)
        if o is not None:
            o.data = me
        if old is not None and old.users == 0:
            bpy.data.meshes.remove(old)
        me.name = name
        if o is None:
            o = bpy.data.objects.new(name, me)
            _collection(scene).objects.link(o)
    if n == 0:
        o.hide_viewport = o.hide_render = True
        return o
    o.hide_viewport = o.hide_render = False
    me.vertices.foreach_set('co', field['P'].astype(np.float32).ravel())
    for nm, typ in (('af_radius', 'FLOAT'), ('af_density', 'FLOAT'), ('af_temperature', 'FLOAT'),
                    (ATTR_TEMP if False else 'tessera_temp_k', 'FLOAT')):
        if me.attributes.get(nm) is None:
            try:
                me.attributes.new(nm, typ, 'POINT')
            except Exception:
                pass
    dens = np.asarray(field['dens'], float)
    temp = np.asarray(field['temp'], float)
    rad = np.full(n, float(field['h']) * 0.85, np.float32)
    # slightly fatter in denser / hotter voxels so the volume closes
    rad = rad * np.clip(0.7 + 0.6 * dens / max(float(dens.max()), 1e-6), 0.6, 1.8).astype(np.float32)
    me.attributes['af_radius'].data.foreach_set('value', rad)
    me.attributes['af_density'].data.foreach_set('value', dens.astype(np.float32))
    me.attributes['af_temperature'].data.foreach_set('value', temp.astype(np.float32))
    if me.attributes.get('tessera_temp_k') is not None:
        me.attributes['tessera_temp_k'].data.foreach_set('value', temp.astype(np.float32))
    me.update()
    mr = o.modifiers.get("AF Render")
    if mr is None:
        mr = o.modifiers.new("AF Render", 'NODES')
        try:
            mr.node_group = gn.render_group('RayMarch', "AF_Airflow")
        except Exception:
            pass
    mat = _airflow_material(T_inf, T_hot)
    try:
        gn.set_group_material(mr.node_group, mat)
        gn.set_modifier_inputs(mr, {"Density": 1.15, "Voxel Size": max(float(field['h']) * 0.85, 1e-3), "Radius Scale": 1.35})
    except Exception:
        if mat.name not in [x.name if x else "" for x in me.materials]:
            me.materials.clear()
            me.materials.append(mat)
    o.visible_shadow = False
    return o


def _airflow_material(T_inf, T_hot):
    """Principled volume: cool cyan slipstream, orange-white heated boundary layer."""
    from .nodeutil import Builder, enable_nodes, inp, out
    name = "Tessera Airflow Volume"
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    nt = enable_nodes(m)
    nt.nodes.clear()
    b = Builder(nt)
    at = b.node('ShaderNodeAttribute', label="Temperature", y=80)
    at.attribute_name = 'af_temperature'
    ad = b.node('ShaderNodeAttribute', label="Density", y=-80)
    ad.attribute_name = 'density'
    mr = b.node('ShaderNodeMapRange', label="Heat", y=80)
    mr.clamp = True
    inp(mr, 'From Min').default_value = float(T_inf)
    inp(mr, 'From Max').default_value = float(max(T_hot, T_inf + 1.0))
    cr = b.node('ShaderNodeValToRGB', label="Air / heat", y=80)
    ramp = cr.color_ramp
    while len(ramp.elements) > 2:
        ramp.elements.remove(ramp.elements[-1])
    ramp.elements[0].position, ramp.elements[0].color = 0.0, (0.55, 0.78, 1.0, 1.0)
    ramp.elements[1].position, ramp.elements[1].color = 1.0, (1.0, 0.55, 0.12, 1.0)
    e = ramp.elements.new(0.45)
    e.color = (0.85, 0.90, 1.0, 1.0)
    e = ramp.elements.new(0.72)
    e.color = (1.0, 0.75, 0.25, 1.0)
    pv = b.node('ShaderNodeVolumePrincipled', y=0)
    if 'Color' in pv.inputs:
        # density-gated emission so the bounding box does not glow
        pass
    o_ = b.node('ShaderNodeOutputMaterial', y=0)
    mu = b.node('ShaderNodeMath', label="Density gate", y=-80)
    mu.operation = 'MULTIPLY'
    mu.inputs[0].default_value = 0.9
    b.link(out(at, 'Fac'), inp(mr, 'Value'))
    b.link(out(mr, 'Result'), inp(cr, 'Fac'))
    if 'Color' in pv.inputs:
        b.link(out(cr, 'Color'), inp(pv, 'Color'))
    # density
    dens_in = pv.inputs.get('Density')
    if dens_in is not None:
        b.link(out(ad, 'Fac'), dens_in)
    # emission
    eco = pv.inputs.get('Emission Color') or pv.inputs.get('Emission')
    if eco is not None:
        b.link(out(cr, 'Color'), eco)
    est = pv.inputs.get('Emission Strength') or pv.inputs.get('Blackbody Intensity')
    if est is not None:
        b.link(out(ad, 'Fac'), inp(mu, 1) if False else mu.inputs[1])
        b.link(out(mr, 'Result'), mu.inputs[0])
        nt.links.new(mu.outputs[0], est)
    b.link(out(pv, 'Volume'), inp(o_, 'Volume'))
    try:
        m.blend_method = 'BLEND'
    except Exception:
        pass
    return m


EXAMPLE = dict(id='WIND_TUNNEL', label="Example: Wind tunnel",
               description="Hollow wind tunnel with an air stream, a tilt-rotor VTOL with spinning rotors and thermal readback on its skin",
               load=load_example)


# ----------------------------------------------------------------------------- operators and panel
class AF_OT_wt_build(Operator):
    bl_idname = "arcflow.wt_build"
    bl_label = "Build Wind Tunnel"
    bl_description = "Create the hollow tunnel (the Environment container of the physics), the wind emitter and the air that flows through it"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        build_tunnel(ctx.scene)
        self.report({'INFO'}, ctx.scene.arc_flow_wt.status)
        return {'FINISHED'}


class AF_OT_wt_add_body(Operator):
    bl_idname = "arcflow.wt_add_body"
    bl_label = "Add Selected as Collision Body"
    bl_description = "Selected meshes become collision bodies in the tunnel (moving parts and deforming meshes follow their animation)"
    bl_options = {'REGISTER', 'UNDO'}
    plain: BoolProperty(name="Plain collision body", default=False, description="Only responds to the fluid: no thermal readback, no heat source, no Systems role")

    def execute(self, ctx):
        n = 0
        for o in ctx.selected_objects:
            if o.type == 'MESH' and not o.get('af_wt') in (TUNNEL, AIRVOL, EMITTER):
                mark_body(o, plain=self.plain, thermal=not self.plain)
                n += 1
        self.report({'INFO'}, "%d collision body(ies)" % n)
        FO.mark_dirty()
        return {'FINISHED'}


class AF_OT_wt_add_aircraft(Operator):
    bl_idname = "arcflow.wt_add_aircraft"
    bl_label = "Add Example Aircraft"
    bl_description = "Insert the example tilt-rotor VTOL (airframe, spinning rotors, engine exhaust heat) into the tunnel with thermal readback on its skin"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        add_aircraft(ctx.scene)
        FO.mark_dirty()
        return {'FINISHED'}



class AF_OT_wt_add_fighter(Operator):
    bl_idname = "arcflow.wt_add_fighter"
    bl_label = "Add Example Fighter"
    bl_description = "Insert the procedural fighter airframe with aerodynamic friction heating and a blended slipstream volume"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        add_fighter(ctx.scene)
        FO.mark_dirty()
        return {'FINISHED'}


class AF_OT_wt_apply_env(Operator):

    bl_idname = "arcflow.wt_apply_env"
    bl_label = "Apply Atmosphere"
    bl_description = "Write the tunnel atmosphere into the Environment card and the Air fluid"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        sc = ctx.scene
        desc = apply_environment(sc)
        fl = next((f for f in sc.arc_flow_fm.fluids if f.name == AIR_NAME), None)
        if fl is not None:
            ap = air_properties(sc)
            fl.density, fl.visc, fl.cp, fl.k, fl.temperature = ap['density'], ap['visc'], ap['cp'], ap['k'], ap['temperature']
            fl.velocity = (sc.arc_flow_wt.wind, 0, 0)
        FO.mark_dirty()
        self.report({'INFO'}, desc)
        return {'FINISHED'}


class AF_PT_windtunnel(Panel):
    bl_idname = "AF_PT_windtunnel"
    bl_label = "Wind tunnel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Tessera"
    bl_parent_id = "AF_PT_fm_main"

    @classmethod
    def poll(cls, ctx):
        fm = getattr(ctx.scene, 'arc_flow_fm', None)
        return fm is not None and fm.mode == 'FLUID'

    def draw(self, ctx):
        l = self.layout
        wt = ctx.scene.arc_flow_wt
        l.prop(wt, "source")
        if wt.source == 'CONTAINER':
            l.prop(wt, "container")
        elif wt.source == 'PRESET':
            l.prop(wt, "atmosphere")
            if wt.atmosphere == 'CUSTOM':
                col = l.column(align=True)
                col.prop(wt, "gas"); col.prop(wt, "pressure"); col.prop(wt, "temperature"); col.prop(wt, "gravity")
        col = l.column(align=True)
        r = col.row(align=True)
        r.prop(wt, "length"); r.prop(wt, "width"); r.prop(wt, "height")
        col.prop(wt, "wall"); col.prop(wt, "open_ends"); col.prop(wt, "wind"); col.prop(wt, "count")
        col.prop(wt, "emit")
        col.prop(wt, "blend_airflow"); col.prop(wt, "aero_heat")
        if wt.emit:
            col.prop(wt, "emit_cap")
        row = l.row(align=True)
        row.scale_y = 1.3
        row.operator("arcflow.wt_build", icon='MOD_WIREFRAME')
        row.operator("arcflow.wt_apply_env", icon='WORLD', text="Apply Atmosphere")
        row = l.row(align=True)
        row.operator("arcflow.wt_add_aircraft", icon='OUTLINER_OB_MESH', text="Add VTOL")
        row.operator("arcflow.wt_add_fighter", icon='MESH_MONKEY', text="Add Fighter")
        row = l.row(align=True)
        op = row.operator("arcflow.wt_add_body", icon='MESH_CUBE', text="Add Collision Body")
        op.plain = False
        op = row.operator("arcflow.wt_add_body", icon='MESH_ICOSPHERE', text="Add Plain Body")
        op.plain = True
        o = ctx.object
        if o is not None and o.type == 'MESH' and o.get('af_wt') not in (TUNNEL, AIRVOL, EMITTER):
            b = l.box()
            b.label(text=o.name, icon='OBJECT_DATA')
            b.prop(o.arc_flow_wt, "body")
            b.prop(o.arc_flow_wt, "plain")
            if _deforming(o):
                b.label(text="Deforming: collision field follows each frame", icon='MOD_ARMATURE')
        if wt.status:
            l.label(text=wt.status[:70], icon='INFO')


CLASSES = (AF_WTScene, AF_WTObject, AF_OT_wt_build, AF_OT_wt_add_body, AF_OT_wt_add_aircraft, AF_OT_wt_add_fighter, AF_OT_wt_apply_env, AF_PT_windtunnel)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    try:
        from . import surfaces_panel
        surfaces_panel.register_loader("Fluid", "Wind tunnel", EXAMPLE['id'], load_example, EXAMPLE['description'])
        surfaces_panel.register_loader("Fluid", "Fighter wind tunnel", "WIND_TUNNEL_FIGHTER", load_fighter_example,
                                       "High-speed fighter: aerodynamic friction heating on the airframe and a blended slipstream")
    except Exception:
        pass
    bpy.types.Scene.arc_flow_wt = PointerProperty(type=AF_WTScene)
    bpy.types.Object.arc_flow_wt = PointerProperty(type=AF_WTObject)
    _install()


def unregister():
    _remove_hook()
    del bpy.types.Object.arc_flow_wt
    del bpy.types.Scene.arc_flow_wt
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
