"""Fluid mode (Tessera Projection / LOTBCBP): Blender bridge for core/fluidmode.py.

Scene -> solver (fluids, containers, colliders, heat sources, Environment card), frame stepping with an in-memory cache,
particle objects (Points / mesh per particle / ray march / splat through the existing render groups), operators and examples."""
import json
import math

import bpy
import numpy as np
from bpy.app.handlers import persistent
from bpy.props import EnumProperty, StringProperty
from bpy.types import Operator

from . import fluidmode_gn as FGN
from . import gn, runtime, shading
from . import spawn as SP
from .core import atmosphere as ATM
from .core import fluidmode as FM

COLLECTION = "TesseraFluid"
AIR_MOLAR = 0.0289647


class FMRuntime:
    def __init__(self):
        self.solver = None
        self.dirty = True
        self.in_handler = False
        self.fill = None
        self.rebuild_pending = False
        self.running = False
        self.scene_name = None
        self.cache = {}
        self.last = None
        self.start_snap = None
        self.body_objs = []
        self.msg = ""
        self.t_build = 0.0
        self.step_times = []
        self.surf = None
        self.surf_key = None


RT = FMRuntime()


def mark_dirty():
    RT.dirty = True


def get_scene():
    return bpy.data.scenes.get(RT.scene_name) if RT.scene_name else bpy.context.scene


# ----------------------------------------------------------------------------- scene -> solver
def _world_mesh(obj, deps):
    v, t = runtime._mesh_arrays(obj, deps)
    M = np.array(obj.matrix_world)
    return v @ M[:3, :3].T + M[:3, 3], t


def _shape_of(obj, deps):
    try:
        V, T = _world_mesh(obj, deps)
    except Exception:
        V = np.array([np.array(obj.matrix_world @ v.co) for v in obj.data.vertices]) if obj.type == 'MESH' else np.zeros((0, 3))
        T = np.zeros((0, 3), int)
    if len(T) == 0:
        return dict(kind='POINTS', pts=V) if len(V) else None
    return dict(kind='MESH', verts=V, tris=T)


def environment(scene, z_hint=0.0):
    af = scene.arc_flow
    e = af.env
    fm = scene.arc_flow_fm
    g = tuple(scene.gravity) if e.use_scene_gravity else tuple(e.gravity)
    molar = AIR_MOLAR
    if e.use_mix and len(e.mix):
        molar = ATM.molar_mass({c.compound: c.fraction for c in e.mix})
    gz = None
    if e.ground:
        gz = z_hint if e.ground_auto else e.ground_z
    return FM.FEnv(gravity=g, pressure=e.pressure, temperature=e.temperature, molar=molar, wind=tuple(e.wind), gust=fm.gust,
                   z_ref=(gz if gz is not None else 0.0), ground_z=gz, ground_friction=fm.ground_friction,
                   ground_temp=fm.ground_temp if fm.ground_temp_on else None)


def _rules_kw(fm):
    import json
    if not fm.rules_json.strip():
        return {}
    d = json.loads(fm.rules_json)
    return dict(species=d.get('species', []), rules=d.get('rules', []), phase=d.get('phase', []))


def _spawn_shapes(fm, defs, geoms, radii, modes, dens, bodies, spacing):
    """Turn each fluid's geometry into the solver's shape (volume mesh, or generated points by the Spawn mode and Density).  Returns the spacing to
    build with: the user's, or (when points were generated) the automatic one so the generated points and the solver lattice agree."""
    rng = np.random.default_rng(fm.seed + 17)
    gen = any(g is not None and SP.needs_dx(g, m, de) for g, m, de in zip(geoms, modes, dens))
    dx = spacing
    if dx is None and gen:
        dx = SP.estimate_dx(defs, geoms, radii, modes, dens, bodies, fm.max_particles)
    for _try in range(3):
        n_liq = 0
        for d, g, r, m, de, fl in zip(defs, geoms, radii, modes, dens, fm.fluids):
            if g is None:
                d['shape'] = None
                fl.d_spawn = ""
                continue
            d['shape'], note = SP.spawn(g, m, de, dx or 0.01, r, rng)
            fl.d_spawn = note
            if d['shape'] is not None and d['shape']['kind'] == 'POINTS' and d['kind'] != 'GAS':
                n_liq += len(d['shape']['pts'])
        if not gen or spacing is not None or n_liq <= 1.15 * fm.max_particles:
            break
        dx *= (n_liq / fm.max_particles) ** (1.0 / 3.0)     # too many generated points: coarser spacing, regenerate
    return dx if gen else spacing


def build_solver(scene):
    fm = scene.arc_flow_fm
    deps = bpy.context.evaluated_depsgraph_get()
    if not RT.in_handler:   # Depsgraph.update() inside a frame-change handler deadlocks Blender's task pool
        deps.update()
    bodies, body_objs, cont_idx = [], [], {}
    defs = []
    geoms, radii, modes, dens = [], [], [], []
    z_low = []
    for fl in fm.fluids:
        preset = fl.preset if fl.preset in FM.PRESETS else 'CUSTOM'
        d = FM.fluid_def(preset, name=fl.name, kind=fl.kind, density=fl.density, mass=fl.mass, volume=fl.volume, weight=fl.weight,
                         temperature=fl.temperature, velocity=tuple(fl.velocity), visc=fl.visc, cp=fl.cp, k=fl.k, beta=fl.beta,
                         cohesion=fl.cohesion, miscible=fl.miscible, cool=fl.cool, drag=None if fl.drag < 0 else fl.drag,
                         count=fl.count, fill=fl.fill / 100.0, color=tuple(fl.color), mix_rate=fl.mix_rate, priority=fl.priority)
        geom, rad = SP.fluid_geometry(fl, deps)
        geoms.append(geom); radii.append(rad); modes.append(fl.spawn_mode); dens.append(fl.spawn_density)
        if geom is not None:
            pts0 = geom['V'] if len(geom['V']) else (geom['psys'] if geom['psys'] is not None else np.zeros((0, 3)))
            if len(pts0):
                z_low.append(float(np.min(pts0[:, 2])))
        if not fl.no_container and fl.container is not None:
            o = fl.container
            if o.name not in cont_idx:
                fld, _v, _t = SP.container_field(o, deps)
                bodies.append(FM.Body(fld, np.array(o.matrix_world), 'CONTAINER', [], 0.2, o.name))
                bodies[-1].open = bool(len(_t)) and not FM.is_closed(np.asarray(_t))
                body_objs.append(o)
                cont_idx[o.name] = len(bodies) - 1
                z_low.append(float(bodies[-1].bbox_world()[0][2]))
            d['body'] = cont_idx[o.name]
            bodies[d['body']].fluids.append(len(defs))
        defs.append(d)
    for o in scene.objects:
        if o.type in SP.CONTAINER_TYPES and o.arc_flow_fm.collider and o.name not in cont_idx:
            fld, _v, _t = SP.container_field(o, deps)
            bodies.append(FM.Body(fld, np.array(o.matrix_world), 'COLLIDER', None, o.arc_flow_fm.friction, o.name))
            body_objs.append(o)
    for b, o in zip(bodies, body_objs):
        _body_props(b, o)
        if b.kind == 'CONTAINER' and not b.fluids:
            b.fluids = None
    env = environment(scene, min(z_low) if z_low else 0.0)
    spacing = fm.spacing if fm.spacing > 0 else None
    spacing = _spawn_shapes(fm, defs, geoms, radii, modes, dens, bodies, spacing)
    S = FM.FluidSolver.build(defs, bodies, env, dx=spacing, max_particles=fm.max_particles, seed=fm.seed, thermal_speedup=fm.speedup,
                             iterations=fm.iterations, max_substeps=fm.max_substeps, lotbcbp=fm.lotbcbp,
                             fill_seed='POUR' if fm.fill_mode == 'POUR' else 'LEVEL', **_rules_kw(fm))
    S.pre_settle = 0
    RT.body_objs = body_objs
    for fi, fl in enumerate(fm.fluids):
        t = S.totals()[fi]
        fl.d_particles, fl.d_mass, fl.d_volume, fl.d_density = t['n'], t['mass'], t['volume'], t['density']
    return S


def sources(scene, frame=None):
    out = []
    if frame is not None and frame < scene.arc_flow_fm.heat_start:
        return out
    for o in scene.objects:
        p = o.arc_flow_fm
        if p.heat_on:
            out.append((tuple(o.matrix_world.translation), p.heat_power, p.heat_radius))
    return out


def fill_phase(scene):
    """Fill / Settle: container-filled liquids are poured into the real cavity under the Environment gravity and settled; the settled state
    is the initial state of the main simulation (heat starts later, at fm.heat_start).  Returns the report dict or None."""
    from .core import fillsettle as FS
    fm = scene.arc_flow_fm
    S = RT.solver
    has = any((not fl.no_container) and fl.container is not None and fl.kind != 'GAS' for fl in fm.fluids)
    if S is None or not has or fm.fill_mode == 'OFF':
        return None
    t0 = __import__('time').time()

    def prog(k, n, ms):
        fm.status = "Filling: step %d / %d, mean speed %.3f m/s" % (k, n, ms)
    rep = FS.settle(S, max_steps=fm.fill_steps, mode=fm.fill_mode, progress=prog)
    rep['seconds'] = __import__('time').time() - t0
    lay = "; ".join("%s %.0f-%.0f mm" % (n, lo * 1e3, hi * 1e3) for n, lo, hi in rep['layers'])
    ov = [o['handling'] for o in rep['overflow'] if o['overflow'] > 0]
    fm.status = "Filled in %d steps (%.1f s): layers %s%s" % (rep['steps'], rep['seconds'], lay, ("; " + "; ".join(ov)) if ov else "")
    return rep


def ensure_world(scene, force=False):
    if RT.solver is None or RT.dirty or force:
        RT.scene_name = scene.name
        t = __import__('time').time()
        RT.solver = build_solver(scene)
        RT.t_build = __import__('time').time() - t
        RT.dirty = False
        RT.cache = {}
        RT.last = None
        RT.fill = fill_phase(scene)
        RT.start_snap = RT.solver.snapshot()
        ensure_objects(scene)
    return RT.solver


# ----------------------------------------------------------------------------- particle objects
def _collection(scene):
    c = bpy.data.collections.get(COLLECTION)
    if c is None:
        c = bpy.data.collections.new(COLLECTION)
    if c.name not in scene.collection.children:
        scene.collection.children.link(c)
    return c


def _points_mesh(name, n):
    me = bpy.data.meshes.get(name)
    if me is None or len(me.vertices) != n:
        old = me
        me = bpy.data.meshes.new(name)
        me.vertices.add(max(n, 1))
        if old is not None:
            for o in list(bpy.data.objects):
                if o.data == old:
                    o.data = me
            bpy.data.meshes.remove(old)
            me.name = name
    for a, t in (('af_radius', 'FLOAT'), ('af_tn', 'FLOAT'), ('af_speed', 'FLOAT'), ('fm_temperature', 'FLOAT'), ('fm_density', 'FLOAT'), ('fm_pressure_solver', 'FLOAT'),
                 ('fm_fluid_id', 'INT'), ('fm_velocity', 'FLOAT_VECTOR'), ('fm_color', 'FLOAT_COLOR')):
        if a not in me.attributes:
            me.attributes.new(a, t, 'POINT')
    return me


def _composition_material():
    """Per-particle composition colour (instancer attribute fm_color on points, geometry attribute on the surface mesh) with alpha blending: overlapping
    translucent particles accumulate and blend their colours like one substance."""
    m = bpy.data.materials.get("TesseraComposition")
    if m is not None and m.node_tree and m.node_tree.nodes:
        return m
    m = bpy.data.materials.get("TesseraComposition") or bpy.data.materials.new("TesseraComposition")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    pb = nt.nodes.new('ShaderNodeBsdfPrincipled')
    pb.inputs['Roughness'].default_value = 0.45
    for k in ('Specular IOR Level', 'Specular'):
        if k in pb.inputs:
            pb.inputs[k].default_value = 0.15
    tr = nt.nodes.new('ShaderNodeBsdfTransparent')
    dk = nt.nodes.new('ShaderNodeMix'); dk.data_type = 'RGBA'; dk.blend_type = 'MULTIPLY'
    dk.inputs[0].default_value = 1.0; dk.inputs[7].default_value = (0.55, 0.55, 0.55, 1.0)
    mx = nt.nodes.new('ShaderNodeMixShader')
    mx.inputs[0].default_value = 0.8
    a1 = nt.nodes.new('ShaderNodeAttribute'); a1.attribute_type = 'INSTANCER'; a1.attribute_name = 'fm_color'
    a2 = nt.nodes.new('ShaderNodeAttribute'); a2.attribute_type = 'GEOMETRY'; a2.attribute_name = 'fm_color'
    ad = nt.nodes.new('ShaderNodeMix'); ad.data_type = 'RGBA'; ad.blend_type = 'ADD'
    ad.inputs[0].default_value = 1.0
    nt.links.new(a1.outputs['Color'], ad.inputs[6])
    nt.links.new(a2.outputs['Color'], ad.inputs[7])
    nt.links.new(ad.outputs[2], dk.inputs[6])
    nt.links.new(dk.outputs[2], pb.inputs['Base Color'])
    nt.links.new(tr.outputs[0], mx.inputs[1])
    nt.links.new(pb.outputs[0], mx.inputs[2])
    nt.links.new(mx.outputs[0], out.inputs['Surface'])
    return m


def _own_material(base, fl, attr):
    cur = getattr(fl, attr)
    if cur is None:
        cur = base.copy()
        cur.name = "%s_%s" % (base.name, fl.name)
        cur['af_fresh'] = False
        setattr(fl, attr, cur)
    return cur


def ensure_objects(scene):
    """Create / refresh the particle objects, node groups and materials of every fluid (cheap to call repeatedly)."""
    fm = scene.arc_flow_fm
    S = RT.solver
    col = _collection(scene)
    gn.ensure_render_groups()
    FGN.ensure_instance_group()
    base_pts, base_vol = shading.points_material(), shading.volume_material()
    for fi, fl in enumerate(fm.fluids):
        n = int(S.hi[fi] - S.lo[fi]) if S is not None and fi < S.nf else 0
        me = _points_mesh("AFFM_%s" % fl.name, n)
        o = bpy.data.objects.get("AFFM_%s" % fl.name)
        if o is None:
            o = bpy.data.objects.new("AFFM_%s" % fl.name, me)
        else:
            o.data = me
        if o.name not in col.objects:
            col.objects.link(o)
        o.hide_select = False
        fl.points_obj = o
        mat = _own_material(base_pts, fl, 'material')
        if S is not None and getattr(S, 'mixing', False):
            mat = _composition_material()
        vmat = _own_material(base_vol, fl, 'volume_material')
        shading.set_params(mat, AF_Color=fl.color, AF_TintMix=1.0 if fl.temp_color else 0.0, AF_Emission=0.0)
        shading.set_params(vmat, AF_ColorA=fl.color, AF_ColorB=fl.color, AF_Density=fl.rm_density, AF_Absorb=0.5)
        # modifier chain: Tessera Fluid (the exposed per-fluid group: points + attributes) -> render group
        mf = o.modifiers.get("Tessera Fluid") or o.modifiers.new("Tessera Fluid", 'NODES')
        mf.node_group = FGN.fluid_group(fl)
        FGN.set_fluid_inputs(mf, fl, fm)
        mr = o.modifiers.get("AF Render") or o.modifiers.new("AF Render", 'NODES')
        kind = {'POINTS': 'Points', 'RAYMARCH': 'RayMarch', 'SPLAT': 'Splat', 'SURFACE': 'Points'}[fl.render_mode]
        if fl.render_mode == 'SURFACE' and fl.kind == 'LIQUID':
            o.hide_render = True     # the surface mesh (AFFMS_*) is rendered instead of the points
        mr.node_group = gn.render_group(kind, "FM_" + fl.name)
        gn.set_group_material(mr.node_group, vmat if kind == 'RayMarch' else mat)
        vox = max(S.dx * 0.9, 1e-3) if S is not None else 0.01
        if kind == 'Points':
            gn.set_modifier_inputs(mr, {"Instance Object": fl.mesh_obj, "Use Object": fl.mesh_obj is not None,
                                        "Instance Collection": fl.mesh_collection, "Use Collection": fl.mesh_collection is not None,
                                        "Scale": 1.0})      # the particle scale is applied to af_radius by apply_state
        elif kind == 'RayMarch':
            gn.set_modifier_inputs(mr, {"Density": fl.rm_density * 0.02, "Voxel Size": vox, "Radius Scale": 2.0})
        else:
            gn.set_modifier_inputs(mr, {"Scale": 2.5})
        o.update_tag()          # modifier inputs set from Python do not tag the object for re-evaluation by themselves
    apply_state(scene)


def refresh_render(scene):
    if RT.solver is not None and scene is not None and getattr(scene, 'arc_flow_fm', None) is not None:
        try:
            ensure_objects(scene)
        except Exception as ex:
            RT.msg = "render refresh: %s" % ex


def _update_merged_surface(scene, S, fls, snap):
    """One smooth, frame-coherent surface for every liquid fluid in Surface mode: all their particles go through a single density field (smooth union:
    touching blobs of different fluids join with a neck, colours blend across the seam), blended in time (core/fluidsurface.py SurfaceBuilder)."""
    from .core import fluidsurface as FSF
    col = _collection(scene)
    C = (snap.get('C', S.C) if snap is not None else S.C)
    X = (snap['x'] if snap is not None else S.x)
    gasp = snap.get('gasp', S.gasp) if snap is not None else S.gasp
    frozen = snap.get('frozen', S.frozen) if snap is not None else S.frozen
    xs, cs = [], []
    for fi, fl in fls:
        sl = slice(S.lo[fi], S.hi[fi])
        if S.hi[fi] == S.lo[fi]:
            continue
        keep = ~(gasp[sl] | frozen[sl])           # transmuted gas parcels and crystals are drawn by their own objects
        if not keep.any():
            continue
        xs.append(X[sl][keep])
        cs.append((S.SS.colour(C[sl]) if getattr(S, 'mixing', False) else np.tile(np.asarray(fl.color, float), (S.hi[fi] - S.lo[fi], 1)))[keep])
    sb = RT.surf
    if sb is None or abs(sb.dx - S.dx) > 1e-12 or RT.surf_key != id(S):
        sb = RT.surf = FSF.SurfaceBuilder(S.dx)
        RT.surf_key = id(S)
    if xs:
        V, T, vc = sb.update(np.vstack(xs), np.vstack(cs), frame=scene.frame_current)
    else:
        V, T, vc = np.zeros((0, 3)), np.zeros((0, 3), int), None
    name = "AFFMS_Liquids"
    me = bpy.data.meshes.get(name) or bpy.data.meshes.new(name)
    o = bpy.data.objects.get(name)
    if o is None:
        o = bpy.data.objects.new(name, me)
    o.data = me
    if o.name not in col.objects:
        col.objects.link(o)
    me.clear_geometry()
    if len(V):
        me.from_pydata(V.tolist(), [], T.tolist())
        if 'fm_color' not in me.attributes:
            me.attributes.new('fm_color', 'FLOAT_COLOR', 'POINT')
        rgba = np.concatenate([vc, np.ones((len(vc), 1))], axis=1).astype(np.float32)
        me.attributes['fm_color'].data.foreach_set('color', rgba.ravel())
        me.polygons.foreach_set('use_smooth', np.ones(len(me.polygons), bool))
    me.materials.clear()
    me.materials.append(_composition_material())
    me.update()


PHASE_GAS_DENSITY = float(__import__('os').environ.get('AF_PHASE_GAS_DENSITY', '4'))
_ICO = None


def _icosahedron():
    global _ICO
    if _ICO is None:
        t = (1.0 + 5 ** 0.5) / 2.0
        V = np.array([(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)], float)
        V /= np.linalg.norm(V, axis=1)[:, None]
        F = np.array([(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
                      (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)])
        _ICO = (V, F)
    return _ICO


def _crystal_material():
    m = bpy.data.materials.get("TesseraCrystal") or bpy.data.materials.new("TesseraCrystal")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    at = nt.nodes.new('ShaderNodeAttribute'); at.attribute_name = 'fm_color'; at.attribute_type = 'GEOMETRY'
    pb = nt.nodes.new('ShaderNodeBsdfPrincipled')
    pb.inputs['Roughness'].default_value = 0.08
    for k, v in (('Transmission Weight', 0.85), ('Transmission', 0.85), ('IOR', 1.5), ('Specular IOR Level', 0.6)):
        if k in pb.inputs:
            pb.inputs[k].default_value = v
    o = nt.nodes.new('ShaderNodeOutputMaterial')
    nt.links.new(at.outputs['Color'], pb.inputs['Base Color'])
    nt.links.new(pb.outputs[0], o.inputs['Surface'])
    return m


def _update_phase_objects(scene, S, snap):
    """Objects for what the transmutation rules made: AFFMG_Phase (points -> soft splats) for the gas parcels that were liquid, AFFMC_Crystals (one flat-shaded,
    randomly oriented icosahedron per frozen particle, growing with the solid fraction)."""
    col = _collection(scene)
    X = snap['x'] if snap is not None else S.x
    C = snap.get('C', S.C) if snap is not None else S.C
    gasp = snap.get('gasp', S.gasp) if snap is not None else S.gasp
    frozen = snap.get('frozen', S.frozen) if snap is not None else S.frozen
    Tn = snap['T'] if snap is not None else S.T
    rad = S.parcel_radius()
    colr = S.SS.colour(C)
    tg = np.flatnonzero(gasp & ~S.gas[S.fid])
    name = "AFFMG_Phase"
    o = bpy.data.objects.get(name)
    if len(tg):
        if o is None:
            me = _points_mesh(name, len(tg))
            o = bpy.data.objects.new(name, me)
            col.objects.link(o)
            mr = o.modifiers.new("AF Render", 'NODES')
            mr.node_group = gn.render_group('RayMarch', "FM_PhaseGas")
            vm = shading.volume_material().copy()
            vm.name = "TesseraPhaseGas"
            shading.set_params(vm, AF_ColorA=(0.95, 0.97, 1.0), AF_ColorB=(0.95, 0.97, 1.0), AF_Density=PHASE_GAS_DENSITY, AF_Absorb=0.5, AF_Noise=0.9, AF_NoiseScale=22.0, AF_Anisotropy=0.4)
            gn.set_group_material(mr.node_group, vm)
            gn.set_modifier_inputs(mr, {"Density": 1.0, "Voxel Size": max(S.dx * 0.55, 1e-3), "Radius Scale": 1.25})
        elif len(o.data.vertices) != len(tg):
            o.data = _points_mesh(name, len(tg))
        me = o.data
        o.hide_render = o.hide_viewport = False
        rgba = np.concatenate([colr[tg], np.ones((len(tg), 1))], axis=1).astype(np.float32)
        me.vertices.foreach_set('co', X[tg].astype(np.float32).ravel())
        me.attributes['af_radius'].data.foreach_set('value', (rad[tg] * 1.0).astype(np.float32))
        for an, av in (('af_alpha', 0.45), ('af_fmix', 1.0)):
            if an not in me.attributes:
                me.attributes.new(an, 'FLOAT', 'POINT')
            me.attributes[an].data.foreach_set('value', np.full(len(tg), av, np.float32))
        if 'af_fcol' not in me.attributes:
            me.attributes.new('af_fcol', 'FLOAT_COLOR', 'POINT')
        me.attributes['af_fcol'].data.foreach_set('color', rgba.ravel())
        me.attributes['fm_color'].data.foreach_set('color', rgba.ravel())
        me.attributes['fm_temperature'].data.foreach_set('value', Tn[tg].astype(np.float32))
        me.update()
        o.update_tag()
    elif o is not None:
        o.hide_render = o.hide_viewport = True
    # crystals
    name = "AFFMC_Crystals"
    fz = np.flatnonzero(frozen)
    o = bpy.data.objects.get(name)
    if len(fz) == 0:
        if o is not None:
            o.hide_render = o.hide_viewport = True
        return
    V0, F0 = _icosahedron()
    rng = np.random.default_rng(7)
    if getattr(RT, 'cryst_az', None) is None or len(RT.cryst_az) < S.n:
        RT.cryst_az = rng.random(S.n) * 2 * np.pi
        RT.cryst_tilt = np.abs(rng.normal(size=S.n)) * 0.28           # shards grow mostly upward, leaning a little
        RT.cryst_s = np.stack([0.55 + 0.25 * rng.random(S.n), 0.55 + 0.25 * rng.random(S.n), 1.5 + 0.9 * rng.random(S.n)], axis=1)
    az, tl = RT.cryst_az[fz], RT.cryst_tilt[fz]
    ca, sa, ct, st = np.cos(az), np.sin(az), np.cos(tl), np.sin(tl)
    # R = Rz(az) @ Rx(tilt)
    R = np.stack([ca, -sa * ct, sa * st, sa, ca * ct, -ca * st, np.zeros_like(ca), st, ct], axis=1).reshape(-1, 3, 3)
    sf = np.clip((C[fz] @ S._sp_solid - 0.5) / 0.5, 0.0, 1.0)
    r = S.dx * 0.95 * (0.5 + 0.5 * sf)
    Vl = V0[None, :, :] * (RT.cryst_s[fz][:, None, :] * r[:, None, None])
    Vw = np.einsum('nij,nkj->nki', R, Vl) + X[fz][:, None, :]
    n = len(fz)
    Vall = Vw.reshape(-1, 3)
    Fall = (F0[None, :, :] + (np.arange(n) * 12)[:, None, None]).reshape(-1, 3)
    cc = np.repeat(colr[fz], 12, axis=0)
    me = bpy.data.meshes.get(name) or bpy.data.meshes.new(name)
    if o is None:
        o = bpy.data.objects.new(name, me)
        col.objects.link(o)
    o.data = me
    o.hide_render = o.hide_viewport = False
    me.clear_geometry()
    me.from_pydata(Vall.tolist(), [], Fall.tolist())
    if 'fm_color' not in me.attributes:
        me.attributes.new('fm_color', 'FLOAT_COLOR', 'POINT')
    me.attributes['fm_color'].data.foreach_set('color', np.concatenate([cc, np.ones((len(cc), 1))], axis=1).astype(np.float32).ravel())
    me.polygons.foreach_set('use_smooth', np.zeros(len(me.polygons), bool))
    me.materials.clear()
    me.materials.append(_crystal_material())
    me.update()


def apply_state(scene, snap=None):
    fm = scene.arc_flow_fm
    S = RT.solver
    if S is None:
        return
    if snap is not None:
        x, v, T = snap['x'], snap['v'], snap['T']
    else:
        x, v, T = S.x, S.v, S.T
    rho = None
    if snap is None:
        rho = S.rho_particles()
    else:
        keep = (S.x, S.v, S.T)
        S.x, S.v, S.T = x, v, T
        rho = S.rho_particles()
        S.x, S.v, S.T = keep
    rad = S.parcel_radius() if snap is None else None
    if rad is None:
        keep = (S.x, S.v, S.T)
        S.x, S.v, S.T = x, v, T
        rad = S.parcel_radius()
        S.x, S.v, S.T = keep
    speed = np.linalg.norm(v, axis=1)
    surf_fls = []
    for fi, fl in enumerate(fm.fluids):
        o = fl.points_obj
        if o is None or fi >= S.nf:
            continue
        lo, hi = S.lo[fi], S.hi[fi]
        me = o.data
        if len(me.vertices) != hi - lo or hi == lo:
            continue
        sl = slice(lo, hi)
        me.vertices.foreach_set('co', x[sl].astype(np.float32).ravel())
        tn = np.clip((T[sl] - fl.t_lo) / max(fl.t_hi - fl.t_lo, 1e-3), 0.0, 1.0).astype(np.float32)
        me.attributes['af_radius'].data.foreach_set('value', (rad[sl] * 1.15 * fl.particle_scale).astype(np.float32))
        me.attributes['af_tn'].data.foreach_set('value', tn)
        me.attributes['af_speed'].data.foreach_set('value', speed[sl].astype(np.float32))
        me.attributes['fm_temperature'].data.foreach_set('value', T[sl].astype(np.float32))
        me.attributes['fm_density'].data.foreach_set('value', rho[sl].astype(np.float32))
        prs = (snap.get('pressure') if snap is not None else getattr(S, 'pressure', None))
        if prs is not None and len(prs) == S.n:
            me.attributes['fm_pressure_solver'].data.foreach_set('value', np.asarray(prs)[sl].astype(np.float32))
        me.attributes['fm_fluid_id'].data.foreach_set('value', np.full(hi - lo, fi, np.int32))
        me.attributes['fm_velocity'].data.foreach_set('vector', v[sl].astype(np.float32).ravel())
        if getattr(S, 'mixing', False):
            col = S.SS.colour(S.C[sl] if snap is None else (snap.get('C', S.C))[sl])
            rgba = np.concatenate([col, np.ones((len(col), 1))], axis=1).astype(np.float32)
            me.attributes['fm_color'].data.foreach_set('color', rgba.ravel())
        if fl.render_mode == 'SURFACE' and not S.gas[fi]:
            surf_fls.append((fi, fl))
        me.update_tag()
    if surf_fls:
        _update_merged_surface(scene, S, surf_fls, snap)
    if getattr(S, 'phase_rules', None):
        _update_phase_objects(scene, S, snap)
    for b, o in zip(S.bodies, RT.body_objs):
        pass


# ----------------------------------------------------------------------------- stepping
def fps(scene):
    return scene.render.fps / scene.render.fps_base


def frame_range(scene):
    return scene.frame_start, scene.frame_end


def sim_frame(scene, frame):
    """Advance the solver so that `frame` exists in the cache.  Returns True if a step ran."""
    import time
    S = ensure_world(scene)
    start, _end = frame_range(scene)
    last = RT.last
    stepped = False
    if last is None or frame < last or frame > last + 240:
        seed = RT.cache.get(frame)
        if frame <= start or seed is None:
            S.restore(RT.start_snap)
            for b in S.bodies:
                b.settle()
            RT.last = start
            RT.cache = {start: S.snapshot()}
            stepped = True
        else:
            S.restore(seed)
            RT.last = frame
    if frame == RT.last:
        sync_bodies()
        return stepped
    dt = 1.0 / fps(scene)
    while RT.last < frame:
        for b, o in zip(S.bodies, RT.body_objs):
            if b.dynamic:
                continue                                   # the object follows the body
            M = np.array(o.matrix_world)
            if not np.allclose(M, b.M1, atol=1e-12):
                b.set_matrix(M)
        t0 = time.time()
        S.step(dt, sources=sources(scene, RT.last + 1))
        RT.step_times.append(time.time() - t0)
        RT.last += 1
        RT.cache[RT.last] = S.snapshot()
        stepped = True
    sync_bodies()
    return stepped


@persistent
def _body_props(b, o):
    p = o.arc_flow_fm
    b.two_way = bool(p.two_way)
    b.mass = float(p.mass)
    b.constraint = p.constraint
    b.axis = np.array(p.axis, float)
    b.limits = (p.limit_lo, p.limit_hi) if p.limit_on else None
    b.stick, b.kfric = float(p.stick), float(p.drag)
    if p.constraint == 'HINGE':
        b.pivot = np.zeros(3)                                # the hinge turns about the object origin
    b.M_start = np.array(o.matrix_world)
    b.reset_motion()


def sync_bodies():
    """Dynamic bodies move their Blender objects (also after scrubbing back through the cache)."""
    S = RT.solver
    if S is None:
        return
    for b, o in zip(S.bodies, RT.body_objs):
        if b.dynamic:
            o.matrix_world = b.M1.tolist()


@persistent
def _defer_rebuild(scene_name):
    if RT.rebuild_pending:
        return
    RT.rebuild_pending = True

    def run():
        RT.rebuild_pending = False
        sc = bpy.data.scenes.get(scene_name)
        if sc is not None and RT.dirty:
            try:
                ensure_world(sc)
                apply_state(sc, RT.start_snap)
            except Exception as ex:
                RT.msg = "Fluid mode: %s" % ex
        return None
    bpy.app.timers.register(run, first_interval=0.0)


def on_frame_change(scene, *args):
    fm = getattr(scene, 'arc_flow_fm', None)
    if fm is None or fm.mode != 'FLUID' or (RT.scene_name and scene.name != RT.scene_name):
        return
    frame = scene.frame_current
    if RT.dirty and RT.solver is not None:
        # Rebuilding creates objects / modifiers and updates the depsgraph: never inside a frame-change handler (Blender deadlocks),
        # so it is deferred to a timer.
        _defer_rebuild(scene.name)
        return
    RT.in_handler = True
    try:
        if RT.running:
            start, end = frame_range(scene)
            if frame > end:
                stop(scene, "Reached end frame")
                return
            sim_frame(scene, frame)
            apply_state(scene, RT.cache.get(frame))
            fm.status = "Simulating frame %d (%.2f s/frame)" % (frame, RT.step_times[-1] if RT.step_times else 0.0)
        elif RT.solver is not None and frame in RT.cache:
            apply_state(scene, RT.cache[frame])
    except Exception as ex:
        RT.msg = "Fluid mode: %s" % ex
        fm.status = RT.msg
        import traceback
        traceback.print_exc()
    finally:
        RT.in_handler = False


def start(scene, play=True):
    fm = scene.arc_flow_fm
    RT.scene_name = scene.name
    ensure_world(scene, force=True)
    RT.running = True
    RT.last = None
    RT.step_times = []
    fm.status = "Running (%d particles, spacing %.2f mm)" % (RT.solver.n, RT.solver.dx * 1000)
    if play:
        s, _e = frame_range(scene)
        if scene.frame_current < s or scene.frame_current > scene.frame_end:
            scene.frame_set(s)
        else:
            on_frame_change(scene)
        try:
            if not bpy.context.screen.is_animation_playing:
                bpy.ops.screen.animation_play()
        except Exception:
            pass


def stop(scene, why="Stopped"):
    RT.running = False
    try:
        if bpy.context.screen and bpy.context.screen.is_animation_playing:
            bpy.ops.screen.animation_cancel(restore_frame=False)
    except Exception:
        pass
    scene.arc_flow_fm.status = why


# ----------------------------------------------------------------------------- operators
def add_fluid(scene, preset='WATER', name=None, primitive=False):
    """primitive=True (the Add Fluid operator): the new fluid is a sphere of points at the 3D cursor that the user edits or replaces."""
    from .fluidmode_props import apply_preset
    fm = scene.arc_flow_fm
    fl = fm.fluids.add()
    fl.name = name or FM.PRESETS[preset]['label']
    n = 1
    base = fl.name
    while sum(1 for f in fm.fluids if f.name == fl.name) > 1:
        n += 1
        fl.name = "%s.%d" % (base, n)
    fl.preset = preset
    apply_preset(fl, preset)
    if fl.kind == 'LIQUID':
        fl.render_mode = 'SURFACE'              # liquids render as one smooth, merged surface by default
    else:
        fl.render_mode = 'RAYMARCH'             # gases render as a soft volume by default, never as beads
    if primitive:
        fl.shape_source, fl.prim_type, fl.spawn_mode = 'PRIMITIVE', 'SPHERE', 'VOLUME'
        fl.prim_loc = tuple(scene.cursor.location)
        if fl.kind == 'GAS':
            fl.prim_size = 0.15
    fm.index = len(fm.fluids) - 1
    mark_dirty()
    return fl


class AF_OT_fm_add_fluid(Operator):
    bl_idname = "arcflow.fm_add_fluid"
    bl_label = "Add Fluid"
    bl_options = {'REGISTER', 'UNDO'}
    preset: EnumProperty(name="Preset", items=FM.PRESET_ITEMS, default='WATER')

    def execute(self, ctx):
        add_fluid(ctx.scene, self.preset, primitive=True)
        return {'FINISHED'}


class AF_OT_fm_remove_fluid(Operator):
    bl_idname = "arcflow.fm_remove_fluid"
    bl_label = "Remove Fluid"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        fm = ctx.scene.arc_flow_fm
        if 0 <= fm.index < len(fm.fluids):
            fl = fm.fluids[fm.index]
            if fl.points_obj is not None:
                bpy.data.objects.remove(fl.points_obj, do_unlink=True)
            fm.fluids.remove(fm.index)
            fm.index = max(0, fm.index - 1)
            mark_dirty()
        return {'FINISHED'}


class AF_OT_fm_assign(Operator):
    bl_idname = "arcflow.fm_assign"
    bl_label = "Assign Selected as Fluid"
    bl_description = ("Make every selected object a fluid: a closed mesh is filled with particles (its interior by the projection's sign test), "
                      "an open mesh gives a sheet on its surface, a point cloud / vertex object gives one particle per vertex")
    bl_options = {'REGISTER', 'UNDO'}
    preset: EnumProperty(name="Fluid", items=FM.PRESET_ITEMS, default='WATER')

    def invoke(self, ctx, event):
        return ctx.window_manager.invoke_props_dialog(self)

    def execute(self, ctx):
        n = 0
        for o in ctx.selected_objects:
            if o.type not in SP.SHAPE_TYPES and not o.particle_systems:
                continue
            fl = add_fluid(ctx.scene, self.preset, name=o.name)
            fl.shape = o
            fl.shape_source = 'OBJECT'
            fl.no_container = True
            n += 1
        if not n:
            self.report({'WARNING'}, "Select one or more geometry objects")
            return {'CANCELLED'}
        self.report({'INFO'}, "%d fluid(s) added. Untick No container and pick a Container to hold them." % n)
        return {'FINISHED'}


class AF_OT_fm_prim_to_object(Operator):
    bl_idname = "arcflow.fm_prim_to_object"
    bl_label = "Make Editable Object"
    bl_description = "Turn the fluid's primitive into a real mesh object (edit it in the viewport) and use that object as the Shape"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        fm = ctx.scene.arc_flow_fm
        if not (0 <= fm.index < len(fm.fluids)):
            return {'CANCELLED'}
        fl = fm.fluids[fm.index]
        V, T = SP.primitive_mesh(fl.prim_type, fl.prim_size)
        me = bpy.data.meshes.new("%s %s" % (fl.name, fl.prim_type.title()))
        me.from_pydata(V.tolist(), [], T.tolist())
        me.update()
        o = bpy.data.objects.new(me.name, me)
        ctx.scene.collection.objects.link(o)
        o.location, o.rotation_euler, o.scale = tuple(fl.prim_loc), tuple(fl.prim_rot), tuple(fl.prim_scale)
        fl.shape, fl.shape_source = o, 'OBJECT'
        mark_dirty()
        return {'FINISHED'}


class AF_OT_fm_heat(Operator):
    bl_idname = "arcflow.fm_heat"
    bl_label = "Selected = Heat Source"
    bl_description = "Mark the active object (a lamp, a burner, an emissive bulb) as a heat source for Fluid mode"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        o = ctx.object
        if o is None:
            return {'CANCELLED'}
        o.arc_flow_fm.heat_on = not o.arc_flow_fm.heat_on
        return {'FINISHED'}


class AF_OT_fm_collider(Operator):
    bl_idname = "arcflow.fm_collider"
    bl_label = "Selected = Fluid Collider"
    bl_description = "Toggle the active mesh as a solid the fluid cannot enter (animate it to push the fluid)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        o = ctx.object
        if o is None or o.type != 'MESH':
            return {'CANCELLED'}
        o.arc_flow_fm.collider = not o.arc_flow_fm.collider
        return {'FINISHED'}


class AF_OT_fm_build(Operator):
    bl_idname = "arcflow.fm_build"
    bl_label = "Build"
    bl_description = "Initialise every fluid from its requirements and start the Fluid mode simulation"

    def execute(self, ctx):
        try:
            start(ctx.scene, play=True)
        except Exception as ex:
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, "Fluid build failed: %s" % ex)
            return {'CANCELLED'}
        return {'FINISHED'}


class AF_OT_fm_stop(Operator):
    bl_idname = "arcflow.fm_stop"
    bl_label = "Stop"

    def execute(self, ctx):
        stop(ctx.scene)
        return {'FINISHED'}


class AF_OT_fm_fill(Operator):
    bl_idname = "arcflow.fm_fill"
    bl_label = "Fill / Settle"
    bl_description = "Pour every container fluid into its cavity under the Environment gravity and settle it; the settled state is the start frame (heat starts at 'Start heat at frame')"

    def execute(self, ctx):
        sc = ctx.scene
        stop(sc)
        RT.scene_name = sc.name
        ensure_world(sc, force=True)
        apply_state(sc, RT.start_snap)
        self.report({'INFO'}, sc.arc_flow_fm.status)
        return {'FINISHED'}


class AF_OT_fm_reset(Operator):
    bl_idname = "arcflow.fm_reset"
    bl_label = "Reset"
    bl_description = "Rebuild from the current settings and go to the start frame"

    def execute(self, ctx):
        stop(ctx.scene)
        ensure_world(ctx.scene, force=True)
        ctx.scene.frame_set(ctx.scene.frame_start)
        return {'FINISHED'}


class AF_OT_fm_bake(Operator):
    bl_idname = "arcflow.fm_bake"
    bl_label = "Bake Range"
    bl_description = "Simulate the whole frame range now (kept in memory for scrubbing and rendering)"

    def execute(self, ctx):
        sc = ctx.scene
        RT.scene_name = sc.name
        ensure_world(sc, force=True)
        RT.running = True
        RT.last = None
        wm = ctx.window_manager
        a, b = frame_range(sc)
        wm.progress_begin(a, b)
        for f in range(a, b + 1):
            sc.frame_set(f)
            wm.progress_update(f)
        wm.progress_end()
        RT.running = False
        sc.arc_flow_fm.status = "Baked %d frames" % (b - a + 1)
        return {'FINISHED'}


# ----------------------------------------------------------------------------- examples (Blender primitives only)
def _emissive_material(name, color, strength):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    em = nt.nodes.new('ShaderNodeEmission')
    em.inputs['Color'].default_value = (*color, 1.0)
    em.inputs['Strength'].default_value = strength
    o = nt.nodes.new('ShaderNodeOutputMaterial')
    nt.links.new(em.outputs['Emission'], o.inputs['Surface'])
    return m


def _glass_material(name="TesseraGlass"):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    gl = nt.nodes.new('ShaderNodeBsdfGlass') if hasattr(bpy.types, 'ShaderNodeBsdfGlass') else nt.nodes.new('ShaderNodeBsdfPrincipled')
    o = nt.nodes.new('ShaderNodeOutputMaterial')
    # clear glass with a faint blue rim (Fresnel) so the cone's silhouette reads without mirror streaks of the lights
    fr = nt.nodes.new('ShaderNodeFresnel')
    fr.inputs['IOR'].default_value = 1.8
    mul = nt.nodes.new('ShaderNodeMath'); mul.operation = 'MULTIPLY'; mul.inputs[1].default_value = 0.55
    em = nt.nodes.new('ShaderNodeEmission')
    em.inputs['Color'].default_value = (0.45, 0.65, 0.9, 1.0); em.inputs['Strength'].default_value = 1.2
    mix = nt.nodes.new('ShaderNodeMixShader')
    nt.links.new(fr.outputs[0], mul.inputs[0])
    nt.links.new(mul.outputs[0], mix.inputs[0])
    nt.links.new(gl.outputs[0], mix.inputs[1])
    nt.links.new(em.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], o.inputs['Surface'])
    m.diffuse_color = (0.8, 0.9, 1.0, 0.15)
    return m


def _mesh_obj(name, bm_fn, loc, col):
    import bmesh
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bm_fn(bm)
    bm.to_mesh(me)
    bm.free()
    o = bpy.data.objects.new(name, me)
    col.objects.link(o)
    o.location = loc
    return o


def _cone(bm, r1, r2, h, seg=40):
    import bmesh
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=seg, radius1=r1, radius2=r2, depth=h)
    bmesh.ops.translate(bm, verts=bm.verts, vec=(0, 0, h / 2))


def _sphere(bm, r, seg=24):
    import bmesh
    bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=seg // 2, radius=r)


def _clear_demo(scene):
    fm = scene.arc_flow_fm
    stop(scene)
    for fl in list(fm.fluids):
        if fl.points_obj is not None:
            bpy.data.objects.remove(fl.points_obj, do_unlink=True)
    fm.fluids.clear()
    for o in list(bpy.data.objects):
        if o.name.startswith(("TS_", "AFFMS_", "AFFMG_", "AFFMC_")):
            bpy.data.objects.remove(o, do_unlink=True)
    RT.solver = None
    mark_dirty()


def _demo_common(scene):
    fm = scene.arc_flow_fm
    fm.mode = 'FLUID'
    _clear_demo(scene)
    af = scene.arc_flow
    e = af.env
    e.preset = 'EARTH'
    e.use_scene_gravity = False
    e.ground = True
    e.ground_auto = True
    e.wind = (0.0, 0.0, 0.0)
    return _collection(scene)


def load_lava_lamp(scene, max_particles=15000):
    col = _demo_common(scene)
    fm = scene.arc_flow_fm
    h, r_b, r_t = 0.36, 0.075, 0.04
    tank = _mesh_obj("TS_LampGlass", lambda bm: _cone(bm, r_b, r_t, h), (0, 0, 0.0), col)
    tank.display_type = 'WIRE'
    tank.data.materials.append(_glass_material())
    tank.arc_flow.cavity_mode = 'SHELL'
    base = _mesh_obj("TS_LampBase", lambda bm: _cone(bm, 0.085, 0.075, 0.06), (0, 0, -0.06), col)
    base.arc_flow_fm.collider = False
    # heat source: an emissive bulb at the bottom with a point light
    bulb = _mesh_obj("TS_LampBulb", lambda bm: _sphere(bm, 0.018), (0, 0, 0.03), col)
    bulb.data.materials.append(_emissive_material("TS_BulbGlow", (1.0, 0.75, 0.35), 25.0))
    bulb.arc_flow_fm.heat_on = True
    bulb.arc_flow_fm.heat_power = 100.0
    bulb.arc_flow_fm.heat_radius = 0.045
    ld = bpy.data.lights.new("TS_LampLight", 'POINT')
    ld.energy = 60.0
    ld.shadow_soft_size = 0.02
    lamp = bpy.data.objects.new("TS_LampLight", ld)
    col.objects.link(lamp)
    lamp.location = (0, 0, 0.05)
    lamp.arc_flow_fm.heat_on = False
    # fluids: a carrier liquid and three waxes poured into the cone (Fill / Settle: the quantities fill the real cavity under gravity and
    # settle into layers, densest at the bottom; the lamp then heats the wax and the blobs rise)
    cav = math.pi * h / 3.0 * (r_b * r_b + r_b * r_t + r_t * r_t)
    carrier = add_fluid(scene, 'WATER', "Carrier")
    carrier.density, carrier.beta, carrier.visc, carrier.k = 985.0, 4.0e-4, 0.01, 0.4
    carrier.color = (0.55, 0.75, 1.0)
    carrier.cohesion, carrier.miscible = 0.04, False
    carrier.no_container, carrier.container, carrier.shape = False, tank, None
    carrier.fill = 90.0
    carrier.volume = 0.0
    carrier.render_mode = 'POINTS'
    carrier.particle_scale = 0.9
    carrier.rm_density = 3.0
    carrier.cool = 0.15
    # three wax blobs (cohesive, immiscible; cold wax is slightly denser than the carrier, warm wax lighter) start inside the cone at the bottom
    for i, (name, color, rr, z, x, y) in enumerate((("Wax red", (1.0, 0.18, 0.08), 0.026, 0.034, 0.0, 0.0),
                                                    ("Wax orange", (1.0, 0.55, 0.05), 0.022, 0.092, 0.012, 0.008),
                                                    ("Wax magenta", (0.9, 0.1, 0.6), 0.02, 0.145, -0.008, -0.006))):
        bl = _mesh_obj("TS_Blob%d" % (i + 1), lambda bm, rr=rr: _sphere(bm, rr), (x, y, z), col)
        bl.display_type = 'WIRE'
        bl.hide_render = True
        fl = add_fluid(scene, 'LAVA_WAX', name)
        fl.density, fl.beta, fl.color = 1034.0 - 4.0 * i, 3.5e-3, color
        fl.temp_color = False
        fl.no_container, fl.container, fl.shape = False, tank, bl
        fl.cohesion, fl.miscible = 2.0, False
        fl.cool = 0.4
        fl.t_lo, fl.t_hi = 290.0, 330.0
    fm.max_particles = max_particles
    fm.speedup = 400.0
    fm.max_substeps = 3
    scene.frame_start, scene.frame_end = 1, 240
    # camera and a floor look
    cam = bpy.data.objects.new("TS_Camera", bpy.data.cameras.new("TS_Camera"))
    col.objects.link(cam)
    cam.location = (0.0, -0.95, 0.24)
    cam.rotation_euler = (math.radians(88.0), 0, 0)
    cam.data.lens = 70
    scene.camera = cam
    mark_dirty()
    return tank


def load_colour_mixing(scene, max_particles=6000):
    """Yellow water (bottom), blue water (poured above), a little red water: miscible, mixing rate on; a rotating paddle swirls them."""
    col = _demo_common(scene)
    fm = scene.arc_flow_fm
    glass = _mesh_obj("TS_Glass", lambda bm: _cone(bm, 0.07, 0.085, 0.16, 48), (0, 0, 0.0), col)
    glass.display_type = 'WIRE'
    glass.data.materials.append(_glass_material())
    glass.arc_flow.cavity_mode = 'SHELL'
    cav = math.pi * 0.16 / 3.0 * (0.07 ** 2 + 0.07 * 0.085 + 0.085 ** 2)
    for name, color, frac, T in (("Yellow water", (1.0, 0.85, 0.08), 0.34, 300.0), ("Blue water", (0.08, 0.3, 1.0), 0.34, 285.0), ("Red water", (0.95, 0.1, 0.1), 0.08, 290.0)):
        fl = add_fluid(scene, 'WATER', name)
        fl.no_container, fl.container, fl.shape = False, glass, None
        fl.volume, fl.color, fl.temperature = frac * cav, color, T
        fl.miscible, fl.mix_rate, fl.cohesion = True, 5.0, 0.0
        fl.cool = 1e-6
        fl.render_mode = 'SURFACE'
    # a paddle that turns (moving collider: its wall velocity stirs the liquid)
    pad = _mesh_obj("TS_Paddle", lambda bm: __import__('bmesh').ops.create_cube(bm, size=1.0) and __import__('bmesh').ops.scale(bm, verts=bm.verts, vec=(0.1, 0.012, 0.11)), (0, 0, 0.07), col)
    pad.arc_flow_fm.collider = True
    pad.rotation_mode = 'XYZ'
    scene.frame_start, scene.frame_end = 1, 240
    pad.rotation_euler = (0, 0, 0)
    pad.keyframe_insert('rotation_euler', frame=1)
    pad.rotation_euler = (0, 0, 14.0)
    pad.keyframe_insert('rotation_euler', frame=240)
    for fc in (pad.animation_data.action.fcurves if hasattr(pad.animation_data.action, 'fcurves') else []):
        for kp in fc.keyframe_points:
            kp.interpolation = 'LINEAR'
    fm.max_particles = max_particles
    fm.speedup = 1.0
    fm.heat_start = 1
    cam = bpy.data.objects.new("TS_Camera", bpy.data.cameras.new("TS_Camera"))
    col.objects.link(cam)
    cam.location = (0.0, -0.6, 0.09)
    cam.rotation_euler = (math.radians(90.0), 0, 0)
    cam.data.lens = 55
    scene.camera = cam
    mark_dirty()


def _beaker(scene, col, r1=0.07, r2=0.085, h=0.16):
    glass = _mesh_obj("TS_Glass", lambda bm: _cone(bm, r1, r2, h, 48), (0, 0, 0.0), col)
    glass.display_type = 'WIRE'
    glass.data.materials.append(_glass_material())
    glass.arc_flow.cavity_mode = 'SHELL'
    return glass, math.pi * h / 3.0 * (r1 ** 2 + r1 * r2 + r2 ** 2)


def _camera(scene, col, loc, lens=55):
    cam = bpy.data.objects.new("TS_Camera", bpy.data.cameras.new("TS_Camera"))
    col.objects.link(cam)
    cam.location = loc
    cam.rotation_euler = (math.radians(90.0), 0, 0)
    cam.data.lens = lens
    scene.camera = cam


def load_steam(scene, max_particles=5000):
    """Water in a beaker over a burner: at 100 C the water turns to steam (rule table `water_steam`), the vapour carries the latent heat away and rises out of
    the glass.  The rules are the JSON on the scene (Species rules)."""
    from .core import phase as PH
    col = _demo_common(scene)
    fm = scene.arc_flow_fm
    glass, _cav = _beaker(scene, col, 0.07, 0.085, 0.5)          # a tall glass so the steam has room to rise
    cav = math.pi * 0.16 / 3.0 * (0.07 ** 2 + 0.07 * 0.085 + 0.085 ** 2)
    pr = PH.preset('water_steam')
    fm.rules_json = json.dumps(dict(species=pr['species'][1:], rules=[], phase=pr['phase']))
    fl = add_fluid(scene, 'WATER', 'Water')
    fl.no_container, fl.container, fl.shape = False, glass, None
    fl.volume, fl.color, fl.temperature, fl.cohesion = 0.5 * cav, tuple(pr['species'][0]['color']), 372.0, 0.0
    fl.render_mode = 'SURFACE'
    fl.cool = 1e-6
    burner = _mesh_obj("TS_Burner", lambda bm: _sphere(bm, 0.02, 12), (0, 0, 0.012), col)
    burner.arc_flow_fm.heat_on = True
    burner.arc_flow_fm.heat_power = 250000.0
    burner.arc_flow_fm.heat_radius = 0.05
    fm.max_particles = max_particles
    fm.speedup = 1.0
    fm.heat_start = 1
    scene.frame_start, scene.frame_end = 1, 200
    _camera(scene, col, (0.0, -1.05, 0.27), lens=50)
    mark_dirty()


def load_crystal(scene, max_particles=5000):
    """A warm salt solution cooling in a dish: nuclei appear and solid crystals grow outward from them (rule table `salt_crystal`), frozen in place and drawn as
    faceted shards, while the rest of the liquid stays a smooth surface."""
    from .core import phase as PH
    col = _demo_common(scene)
    fm = scene.arc_flow_fm
    dish, cav = _beaker(scene, col, 0.09, 0.1, 0.05)
    pr = PH.preset('salt_crystal')
    pr['phase'][0].update(seeds=8, rate=0.5)
    fm.rules_json = json.dumps(dict(species=pr['species'][1:], rules=[], phase=pr['phase']))
    fl = add_fluid(scene, 'WATER', 'Brine')
    fl.no_container, fl.container, fl.shape = False, dish, None
    fl.volume, fl.color, fl.temperature, fl.cohesion = 0.8 * cav, tuple(pr['species'][0]['color']), 290.0, 0.0
    fm.ground_temp_on = False
    fl.render_mode = 'SURFACE'
    fl.cool = 1e-6
    fm.max_particles = max_particles
    fm.speedup = 1.0
    fm.heat_start = 1
    scene.frame_start, scene.frame_end = 1, 200
    _camera(scene, col, (0.0, -0.45, 0.055), lens=60)
    mark_dirty()


def load_gas_plume(scene, max_particles=3000):
    col = _demo_common(scene)
    fm = scene.arc_flow_fm
    src = _mesh_obj("TS_PlumeSource", lambda bm: _sphere(bm, 0.25), (0.0, 0.0, 0.3), col)
    src.display_type = 'WIRE'
    src.hide_render = True
    smoke = add_fluid(scene, 'AIR', "Hot gas")
    smoke.temperature = 650.0
    smoke.count = max_particles
    smoke.volume = 0.0
    smoke.color = (0.9, 0.5, 0.25)
    smoke.temp_color = True
    smoke.t_lo, smoke.t_hi = 288.0, 650.0
    smoke.shape, smoke.no_container = src, True
    smoke.drag = 0.8
    smoke.cool = 0.25
    scene.arc_flow.env.wind = (0.8, 0.0, 0.0)
    scene.arc_flow.env.preset = 'EARTH'
    fm.gust = 0.3
    fm.max_particles = max_particles
    fm.speedup = 1.0
    scene.frame_start, scene.frame_end = 1, 240
    cam = bpy.data.objects.new("TS_Camera", bpy.data.cameras.new("TS_Camera"))
    col.objects.link(cam)
    cam.location = (0.0, -9.0, 3.0)
    cam.rotation_euler = (math.radians(84.0), 0, 0)
    scene.camera = cam
    mark_dirty()


def load_gas_tower(scene, max_particles=3000):
    col = _demo_common(scene)
    fm = scene.arc_flow_fm
    tower = _mesh_obj("TS_Tower", lambda bm: _cone(bm, 0.5, 0.35, 2.5, 32), (0.0, 0.0, 3.0), col)
    tower.display_type = 'WIRE'
    tower.arc_flow.cavity_mode = 'SHELL'
    cloud_src = _mesh_obj("TS_Cloud", lambda bm: _sphere(bm, 0.45), (0.0, 0.0, 0.7), col)
    cloud_src.display_type = 'WIRE'
    cloud_src.hide_render = True
    inside = add_fluid(scene, 'HELIUM', "Tower gas")
    inside.no_container, inside.container, inside.fill = False, tower, 60.0
    inside.count = max_particles // 2
    inside.color = (0.95, 0.6, 0.95)
    cloud = add_fluid(scene, 'CO2', "Free gas cloud")
    cloud.shape, cloud.no_container = cloud_src, True
    cloud.count = max_particles // 2
    cloud.color = (0.4, 0.45, 0.55)
    cloud.drag = 0.8
    scene.arc_flow.env.wind = (0.4, 0.0, 0.0)
    fm.max_particles = max_particles
    scene.frame_start, scene.frame_end = 1, 240
    cam = bpy.data.objects.new("TS_Camera", bpy.data.cameras.new("TS_Camera"))
    col.objects.link(cam)
    cam.location = (0.0, -11.0, 3.2)
    cam.rotation_euler = (math.radians(86.0), 0, 0)
    scene.camera = cam
    mark_dirty()


class AF_OT_fm_demo(Operator):
    bl_idname = "arcflow.fm_demo"
    bl_label = "Load Fluid Example"
    bl_options = {'REGISTER', 'UNDO'}
    which: StringProperty(default='LAVA')

    @classmethod
    def description(cls, ctx, props):
        return {'LAVA': "Lava lamp: a tapered container, a heat source at the bottom and wax blobs in a carrier liquid",
                'PLUME': "Free gas plume: hot gas rising and drifting in the Environment's wind",
                'MIX': "Colour mixing: yellow, blue and red water in a glass; a stirrer swirls them and the colours blend (subtractive mix) and the temperature equilibrates",
                'TOWER': "Elevated gas tower: light gas in an elevated empty tower above a free gas cloud"}.get(props.which, "")

    def execute(self, ctx):
        {'LAVA': load_lava_lamp, 'PLUME': load_gas_plume, 'TOWER': load_gas_tower, 'MIX': load_colour_mixing}[self.which](ctx.scene)
        self.report({'INFO'}, "Example loaded. Press Build.")
        return {'FINISHED'}


CLASSES = (AF_OT_fm_add_fluid, AF_OT_fm_remove_fluid, AF_OT_fm_assign, AF_OT_fm_prim_to_object, AF_OT_fm_heat, AF_OT_fm_collider, AF_OT_fm_build, AF_OT_fm_stop,
           AF_OT_fm_reset, AF_OT_fm_fill, AF_OT_fm_bake, AF_OT_fm_demo)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    if on_frame_change not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(on_frame_change)


def unregister():
    if on_frame_change in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(on_frame_change)
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
