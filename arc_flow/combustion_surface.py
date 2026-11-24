"""Smooth merged surface of the liquid propellants (combustion mode, render type 'Smooth surface').

Liquid propellant particles (not yet reacted: kind 0) of every fluid in Smooth surface mode go through one density field (core/fluidsurface.SurfaceBuilder, the same
builder as Fluid mode): fuel and oxidiser that touch join with a neck, colours blend across the seam, the mesh changes continuously from frame to frame.  The mesh
`AFCS_Liquids` carries the colour attribute `fm_color` and a shader preset material (Water by default; Auto picks Oil for fuels).  Flame gas and smoke are drawn as soft, smoothly blended
volumes (Wendland splat -> dense points -> Points to Volume + shader preset volumes Flame / Smoke / Soot, objects `AFCP_core|flame|smoke|soot`); `scene.arc_flow["plume_volume"] = 0` goes back to
Gaussian splats."""
import bpy
import numpy as np

from . import runtime as RT0
from .core import fluidsurface as FSF
from .core import fluid_blend as FB

NAME = "AFCS_Liquids"
_state = dict(builder=None, key=None)


def _is_gas(world, fi):
    try:
        return world.fluids[fi]['spec']['kind'] == 'GAS'
    except Exception:
        return False


def plume_on(af):
    try:
        return bool(af.get('plume_volume', 1))
    except Exception:
        return True


def gas_mask(world, src, fi, idx):
    """1.0 for particles the splat object draws (flame gas, smoke, gas fluids), 0.0 for liquid propellant drawn by the surface mesh."""
    if _is_gas(world, fi):
        return np.ones(len(idx), np.float32)
    if 'kind' in src and plume_on(bpy.context.scene.arc_flow):
        return (src['state'][idx] == 2).astype(np.float32) * (src['kind'][idx] == 0)     # flame / smoke go to the volume objects
    if 'kind' in src:
        return ((src['kind'][idx] > 0) | (src['state'][idx] == 2)).astype(np.float32)     # free-flying liquid stays particles
    return np.zeros(len(idx), np.float32)


def _material(af, fls, world):
    from . import shader_presets as SP
    f0 = fls[0][1]
    key = 'OIL' if len(fls) == 1 and f0.fluid_role == 'FUEL' else 'WATER'
    name = "TesseraCombustionLiquid [%s]" % key
    mat = bpy.data.materials.get(name)
    if mat is None or mat.get('ts_preset') != key:
        mat = SP.build(key, name=name, reuse=False)
    try:
        SP.set_params(mat, **{'Use fluid colour': 1.0, 'Use fluid colour (absorption)': 1.0, 'Use fluid colour (emission)': 1.0, 'Use_fluid_colour': 1.0})
    except Exception:
        pass
    return mat


def update(scene, world, src):
    """Rebuild the merged liquid surface for the current state (called from runtime.apply_state)."""
    af = scene.arc_flow
    fls = [(fi, fl) for fi, fl in enumerate(af.fluids) if fl.render_mode == 'SURFACE' and not _is_gas(world, fi)]
    o = bpy.data.objects.get(NAME)
    if not fls or world is None:
        if o is not None:
            o.hide_viewport = o.hide_render = True
        return
    state = src['state']
    liquid = state == 1                                    # confined (in a tank or channel); liquid in free flight is drawn as particles
    if 'kind' in src:
        liquid = liquid & (src['kind'] == 0)
    xs, cs = [], []
    for fi, fl in fls:
        idx = RT0.RT.fl_slots[fi] if fi < len(RT0.RT.fl_slots) else np.zeros(0, int)
        idx = idx[liquid[idx]]
        if len(idx):
            xs.append(src['pos'][idx])
            cs.append(np.tile(np.asarray(fl.color, float), (len(idx), 1)))
    if xs:
        X = np.vstack(xs)
        rl = src['rad'][np.concatenate([RT0.RT.fl_slots[fi][liquid[RT0.RT.fl_slots[fi]]] for fi, _ in fls])]
        dx = float(np.median(rl)) / (world.p.radius_scale * 1.6) * 1.612       # spacing of the particles at their own volume
    else:
        dx = 0.1
    sb = _state['builder']
    if sb is None or _state['key'] != id(world) or abs(sb.dx - dx) > 0.15 * sb.dx:
        sb = _state['builder'] = FB.SolidSurfaceBuilder(max(dx, 1e-3), radius=2.3, iso=0.28, cell=0.45, tension=0.35, smooth=4)
        _state['key'] = id(world)
    if xs:
        V, T, vc = sb.update(X, np.vstack(cs), frame=scene.frame_current)
    else:
        V, T, vc = np.zeros((0, 3)), np.zeros((0, 3), int), None
    col = RT0._collection(scene)
    me = bpy.data.meshes.get(NAME) or bpy.data.meshes.new(NAME)
    if o is None:
        o = bpy.data.objects.new(NAME, me)
    o.data = me
    if o.name not in col.objects:
        col.objects.link(o)
    o.hide_viewport = o.hide_render = False
    me.clear_geometry()
    if len(V):
        me.from_pydata(V.tolist(), [], T.tolist())
        if 'fm_color' not in me.attributes:
            me.attributes.new('fm_color', 'FLOAT_COLOR', 'POINT')
        rgba = np.concatenate([vc, np.ones((len(vc), 1))], axis=1).astype(np.float32)
        me.attributes['fm_color'].data.foreach_set('color', rgba.ravel())
        me.polygons.foreach_set('use_smooth', np.ones(len(me.polygons), bool))
    me.materials.clear()
    me.materials.append(_material(af, fls, world))
    me.update()


# ----------------------------------------------------------------------------- plume volumes
# class: (shader preset, radius scale, voxel factor, material overrides)
_PLUME = {
    'core': ('FLAME', 0.9, dict(Density=0.04, Temperature=2900.0, Blackbody_intensity=5.0)),
    'flame': ('FLAME', 1.5, dict(Density=0.03, Temperature=1500.0, Blackbody_intensity=0.9)),
    'smoke': ('SMOKE', 1.7, dict(Density=0.35, Blackbody_intensity=0.0)),
    'soot': ('SOOT', 1.6, dict(Density=0.7)),
}


def _gate_emission(mat):
    """Principled Volume emission is not scaled by density, so it would fill the whole grid bounding box: multiply it by the density grid."""
    nt = mat.node_tree
    pv = next((n for n in nt.nodes if n.bl_idname == 'ShaderNodeVolumePrincipled'), None)
    if pv is None:
        return
    at = nt.nodes.new('ShaderNodeAttribute')
    at.attribute_name = 'density'
    mu = nt.nodes.new('ShaderNodeMath')
    mu.operation = 'MULTIPLY'
    for sk in ('Blackbody Intensity', 'Emission Strength'):
        s_ = pv.inputs[sk]
        if s_.is_linked:
            src = s_.links[0].from_socket
            nt.links.new(src, mu.inputs[0])
        else:
            mu.inputs[0].default_value = s_.default_value
        nt.links.new(at.outputs['Fac'], mu.inputs[1])
        nt.links.new(mu.outputs[0], s_)
        break


def _plume_obj(scene, cls, n):
    from . import gn, shading
    from . import shader_presets as SP
    name = "AFCP_" + cls
    o = bpy.data.objects.get(name)
    me = bpy.data.meshes.get(name)
    if me is None or len(me.vertices) != n:
        old = me
        me = bpy.data.meshes.new(name + ".tmp")
        me.vertices.add(n)
        if 'af_radius' not in me.attributes:
            me.attributes.new('af_radius', 'FLOAT', 'POINT')
        if o is not None:
            o.data = me
        if old is not None:
            bpy.data.meshes.remove(old)
        me.name = name
    if o is None:
        o = bpy.data.objects.new(name, me)
        RT0._collection(scene).objects.link(o)
        mr = o.modifiers.new("AF Render", 'NODES')
        mr.node_group = gn.render_group('RayMarch', "AFCP_" + cls)
        key, rs, over = _PLUME[cls]
        mat = SP.build(key, name="TesseraPlume [%s]" % cls, grid=True, reuse=False)
        SP.set_params(mat, **over)
        _gate_emission(mat)
        gn.set_group_material(mr.node_group, mat)
        o['af_rs'] = rs
    return o, me


def update_plume(scene, world, src):
    af = scene.arc_flow
    on = plume_on(af) and 'kind' in src
    fi_on = [fi for fi, fl in enumerate(af.fluids) if fl.render_mode == 'SURFACE']
    from .core import flame as FLM
    kind, soot, temp = src.get('kind'), src.get('soot'), src['temp']
    groups = {}
    if on and fi_on:
        idx = np.concatenate([RT0.RT.fl_slots[fi] for fi in fi_on if fi < len(RT0.RT.fl_slots)] or [np.zeros(0, int)]).astype(int)
        idx = idx[(kind[idx] > 0) & (src['state'][idx] == 2)]
        fade = src['fade'][idx]
        tnf = np.clip((temp[idx] - FLM.FLAME['TLO']) / (FLM.FLAME['THI'] - FLM.FLAME['TLO']), 0.0, 1.0)
        k = kind[idx]
        sm = src['smk'][idx]
        groups['core'] = (idx[(k == 1) & (tnf > 0.62)], fade[(k == 1) & (tnf > 0.62)])
        groups['flame'] = (idx[(k == 1) & (tnf <= 0.62)], fade[(k == 1) & (tnf <= 0.62)])
        sel = (k == 2) & (soot[idx] < 0.45)
        groups['smoke'] = (idx[sel], fade[sel] * np.clip(sm[sel] * 2.5, 0.3, 1.0))
        sel = (k == 2) & (soot[idx] >= 0.45)
        groups['soot'] = (idx[sel], fade[sel] * np.clip(sm[sel] * 2.5, 0.3, 1.0))
    for cls in _PLUME:
        ii, fd = groups.get(cls, (np.zeros(0, int), np.zeros(0)))
        o = bpy.data.objects.get("AFCP_" + cls)
        if len(ii) == 0:
            if o is not None:
                o.hide_viewport = o.hide_render = True
            continue
        # blend the sparse flame / smoke parcels; curve / thin-surface sources keep a tight kernel
        # so wispy tendrils stay sharp with far fewer particles than a filled-volume emitter
        dx_p = float(np.median(src['rad'][ii])) / 1.2 if len(ii) else 0.02
        P, dens, Tbl, _ = FB.blend_plume(src['pos'][ii], src['temp'][ii], dx=max(dx_p, 1e-4),
                                         fade=fd, source='auto', iso=0.04)
        if len(P) < 8:
            P, dens = src['pos'][ii], np.clip(fd, 0.05, 1.0)
            rad = src['rad'][ii].astype(np.float32) * np.clip(fd, 0.05, 1.0).astype(np.float32)
        else:
            rad = (np.full(len(P), max(dx_p * 0.85, 1e-4)) * np.clip(0.6 + 0.9 * dens / max(float(dens.max()), 1e-6), 0.5, 2.0)).astype(np.float32)
        o, me = _plume_obj(scene, cls, len(P))
        o.hide_viewport = o.hide_render = False
        me.vertices.foreach_set('co', P.astype(np.float32).ravel())
        me.attributes['af_radius'].data.foreach_set('value', rad)
        me.update()
        mr = o.modifiers.get("AF Render")
        from . import gn
        r_med = float(np.median(rad)) if len(rad) else 0.01
        gn.set_modifier_inputs(mr, {"Density": 1.15, "Voxel Size": max(r_med * 0.65, 1e-3), "Radius Scale": float(o.get('af_rs', 1.6))})
