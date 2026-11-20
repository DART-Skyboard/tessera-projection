"""Headless Blender test of the shader presets:  blender --factory-startup -b --python tests/test_shaders_bpy.py
Every preset builds in every variant without unresolved sockets or invalid links, parameters are settable, the assignment operators work
on meshes / curves / volumes, fluid defaults map, fluid shaders apply to the render objects."""
import os
import sys

import bpy

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import arc_flow
from arc_flow import shader_presets as SP
arc_flow.register()
fails = [0]


def ok(c, msg):
    print(("PASS " if c else "FAIL ") + msg, flush=True)
    if not c:
        fails[0] += 1


print("Blender", bpy.app.version_string, flush=True)
sc = bpy.context.scene
ok(len(SP.PRESETS) >= 19, "%d presets" % len(SP.PRESETS))

# ---- every preset, every variant, strict socket resolution
bad = []
for k in SP.ORDER:
    for kw in ({}, {'preview': True}, {'particle': True}, {'grid': True}):
        try:
            m = SP.build(k, strict=True, reuse=False, **kw)
            nt = m.node_tree
            if m['ts_unresolved'] or any(not l.is_valid for l in nt.links):
                bad.append((k, kw, 'unresolved/invalid'))
            outs = [n for n in nt.nodes if n.bl_idname == 'ShaderNodeOutputMaterial']
            if len(outs) != 1 or not (outs[0].inputs['Surface'].is_linked or outs[0].inputs['Volume'].is_linked):
                bad.append((k, kw, 'no output'))
            if not SP.params(m):
                bad.append((k, kw, 'no params'))
        except Exception as ex:
            bad.append((k, kw, repr(ex)))
ok(not bad, "all presets build in 4 variants, sockets resolved %s" % bad[:3])

# ---- parameters
m = SP.build('WATER', reuse=False)
ok(SP.set_params(m, Absorption_density=3.5, IOR=1.4, Color=(0.2, 0.4, 0.6)) == [], "set_params")
ok(abs(m.node_tree.nodes['TS_Absorption density'].outputs[0].default_value - 3.5) < 1e-6, "parameter value stored")
ok(SP.set_params(m, Nonexistent=1.0) == ['Nonexistent'], "unknown parameter reported")
m2 = SP.build('GLASS', Absorption_density=0.7, reuse=False)
ok(abs(m2.node_tree.nodes['TS_Absorption density'].outputs[0].default_value - 0.7) < 1e-6, "override at build")

# ---- node structure: attribute inputs present
nt = SP.build('LAVA', reuse=False).node_tree
attrs = {n.attribute_name for n in nt.nodes if n.bl_idname == 'ShaderNodeAttribute'}
ok('fm_temperature' in attrs, "lava reads fm_temperature")
ok('fm_color' in {n.attribute_name for n in SP.build('WATER', reuse=False).node_tree.nodes if n.bl_idname == 'ShaderNodeAttribute'}, "water reads fm_color")
pv = [n for n in SP.build('FLAME', reuse=False).node_tree.nodes if n.bl_idname == 'ShaderNodeVolumePrincipled']
ok(len(pv) == 1 and pv[0].inputs['Blackbody Intensity'].is_linked or pv[0].inputs['Blackbody Intensity'].default_value > 0, "flame blackbody emission")
g = SP.build('SMOKE', grid=True, reuse=False).node_tree
pvn = [n for n in g.nodes if n.bl_idname == 'ShaderNodeVolumePrincipled'][0]
ok(pvn.inputs['Density Attribute'].default_value == 'density', "grid variant reads the density grid")
ok(SP.build('WATER', reuse=False).use_raytrace_refraction if hasattr(bpy.types.Material, 'use_raytrace_refraction') else True, "EEVEE refraction setting on water")

# ---- assignment operator on objects
bpy.ops.mesh.primitive_uv_sphere_add(location=(0, 0, 0))
a = bpy.context.object
bpy.ops.mesh.primitive_cube_add(location=(3, 0, 0))
b = bpy.context.object
bpy.ops.curve.primitive_bezier_circle_add(location=(6, 0, 0))
cu = bpy.context.object
bpy.ops.object.select_all(action='DESELECT')
for o in (a, b, cu):
    o.select_set(True)
bpy.context.view_layer.objects.active = a
sc.arc_flow_sh.preset = 'HONEY'
r = bpy.ops.arcflow.shader_assign(target='SELECTED')
ok(r == {'FINISHED'} and all(o.active_material is not None and o.active_material.get('ts_preset') == 'HONEY' for o in (a, b, cu)), "assign to selected (mesh, cube, curve)")
sc.arc_flow_sh.preset = 'ICE'
bpy.ops.arcflow.shader_assign(target='ACTIVE')
ok(a.active_material.get('ts_preset') == 'ICE' and b.active_material.get('ts_preset') == 'HONEY', "assign to active only")
sc.arc_flow_sh.caustics = True
sc.arc_flow_sh.preset = 'GLASS'
bpy.ops.arcflow.shader_assign(target='ACTIVE')
ok(a.cycles.is_caustics_caster, "refractive assignment tags the caustics caster")
sc.arc_flow_sh.preset = 'WAX'
bpy.ops.arcflow.shader_create()
ok(any(m.name.startswith('TS Wax') and m.users >= 1 for m in bpy.data.materials) and a.active_material.get('ts_preset') == 'WAX', "create shader from preset")
ok(bpy.ops.arcflow.shader_caustics() == {'FINISHED'}, "caustics set-up operator")
bpy.ops.object.empty_add()
e = bpy.context.object
ok(not SP.assign(e, SP.build('WATER')), "empties have no material slots")
try:
    bpy.ops.object.volume_add()
    v = bpy.context.object
    sc.arc_flow_sh.preset = 'SMOKE'
    bpy.ops.object.select_all(action='DESELECT'); v.select_set(True); bpy.context.view_layer.objects.active = v
    bpy.ops.arcflow.shader_assign(target='ACTIVE')
    ok(v.active_material is not None and v.active_material['ts_variant'] == 'grid', "volume object gets the grid variant")
except Exception as ex:
    ok(False, "volume object: %r" % ex)

# ---- fluid defaults and fluid shaders
class F:
    def __init__(s, p, **kw):
        s.preset = p
        s.shader = kw.get('shader', 'AUTO')
        s.render_mode = kw.get('render_mode', 'POINTS')
        s.name = p
        s.color = (0.3, 0.5, 0.9)
ok([SP.shader_for_fluid(F(p)) for p in ('WATER', 'LAVA_WAX', 'HONEY', 'VEG_OIL')] == ['WATER', 'WAX', 'HONEY', 'OIL'], "fluid preset -> default shader")
ok(SP.shader_for_fluid(F('WATER', shader='LAVA')) == 'LAVA', "per-fluid choice")
from arc_flow import fluidmode_ops as FO
fm = sc.arc_flow_fm
fm.mode = 'FLUID'
FO.load_lava_lamp(sc, max_particles=900)
ok(hasattr(fm.fluids[0], 'shader') and fm.fluids[0].shader == 'AUTO', "fluid has a shader property")
fm.fluids[1].shader = 'LAVA'
FO.start(sc, play=False)
n = SP.apply_fluid_shaders(sc)
ok(n >= len(fm.fluids), "apply_fluid_shaders touched %d objects" % n)
mr = fm.fluids[1].points_obj.modifiers["AF Render"]
mat = mr.node_group.nodes['AF_Material'].material
ok(mat is not None and mat.get('ts_preset') == 'LAVA', "fluid points use the fluid's shader (%s)" % (mat.name if mat else None))
ok(abs(mat.node_tree.nodes['TS_Use fluid temperature'].outputs[0].default_value - 1.0) < 1e-6, "fluid variant reads the fluid attributes")
r = bpy.ops.arcflow.shader_assign_fluid()
ok(r == {'FINISHED'}, "assign preset to selected fluid(s)")

print("FAILS:", fails[0])
sys.exit(1 if fails[0] else 0)
