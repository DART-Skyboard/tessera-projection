"""Smooth merged surface of the liquid propellants (combustion mode, render type 'Smooth surface').

Liquid propellant particles (not yet reacted: kind 0) of every fluid in Smooth surface mode go through one density field (core/fluidsurface.SurfaceBuilder, the same
builder as Fluid mode): fuel and oxidiser that touch join with a neck, colours blend across the seam, the mesh changes continuously from frame to frame.  The mesh
`AFCS_Liquids` carries the colour attribute `fm_color` and a shader preset material (Water by default; Auto picks Oil for fuels).  Flame gas and smoke of the same
fluids stay particles (Gaussian splat)."""
import bpy
import numpy as np

from . import runtime as RT0
from .core import fluidsurface as FSF

NAME = "AFCS_Liquids"
_state = dict(builder=None, key=None)


def _is_gas(world, fi):
    try:
        return world.fluids[fi]['spec']['kind'] == 'GAS'
    except Exception:
        return False


def gas_mask(world, src, fi, idx):
    """1.0 for particles the splat object draws (flame gas, smoke, gas fluids), 0.0 for liquid propellant drawn by the surface mesh."""
    if _is_gas(world, fi):
        return np.ones(len(idx), np.float32)
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
        rl = world.rad[np.concatenate([RT0.RT.fl_slots[fi][liquid[RT0.RT.fl_slots[fi]]] for fi, _ in fls])]
        dx = float(np.median(rl)) / (world.p.radius_scale * 1.6) * 1.612       # spacing of the particles at their own volume
    else:
        dx = 0.1
    sb = _state['builder']
    if sb is None or _state['key'] != id(world) or abs(sb.dx - dx) > 0.15 * sb.dx:
        sb = _state['builder'] = FSF.SurfaceBuilder(max(dx, 1e-3), sigma=1.3, iso=0.25, cell=0.4)
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
