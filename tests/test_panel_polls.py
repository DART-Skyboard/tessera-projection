"""Panel poll() in both modes: shared panels (Examples, Shader presets, Components, Projection Remesh, Systems, Environment, Thermal ...)
are visible in Combustion AND Fluid mode; combustion panels hide in Fluid mode, Fluid panels hide in Combustion mode.
    blender --factory-startup -b --python tests/test_panel_polls.py [-- --installed]"""
import importlib, os, sys
import bpy
HERE = os.path.dirname(os.path.abspath(__file__))
fails = [0]
def ok(c, m):
    print(("PASS " if c else "FAIL ") + m, flush=True)
    if not c: fails[0] += 1
if '--installed' not in sys.argv:
    sys.path.insert(0, os.path.join(HERE, '..')); import arc_flow; arc_flow.register()
else:
    import addon_utils
    pkg = [m.__name__ for m in addon_utils.modules() if m.__name__.endswith(('tessera_projection', 'arc_flow'))][0]
    if pkg not in bpy.context.preferences.addons:
        addon_utils.enable(pkg, default_set=True, persistent=False)
print("Blender", bpy.app.version_string, flush=True)
sc = bpy.context.scene
bpy.ops.mesh.primitive_cube_add()
def vis(mode):
    sc.arc_flow_fm.mode = mode
    out = {}
    for n in dir(bpy.types):
        if n.startswith('AF_PT_') and getattr(bpy.types, n).bl_space_type == 'VIEW_3D':
            c = getattr(bpy.types, n)
            out[n] = bool(c.poll(bpy.context)) if hasattr(c, 'poll') else True
    return out
comb, fl = vis('COMBUSTION'), vis('FLUID')
shared = ['AF_PT_examples', 'AF_PT_shader', 'AF_PT_retopo', 'AF_PT_systems', 'AF_PT_env', 'AF_PT_fm_mode', 'AF_PT_gn_nodes_view3d']
for n in shared:
    ok(hasattr(bpy.types, n), n + " exists")
    ok(comb.get(n) and fl.get(n), "%s visible in both modes (combustion %s, fluid %s)" % (n, comb.get(n), fl.get(n)))
for n in ('AF_PT_main', 'AF_PT_parts', 'AF_PT_fluids', 'AF_PT_solver', 'AF_PT_cache'):
    ok(comb.get(n) and not fl.get(n), "%s combustion only" % n)
for n in ('AF_PT_fm_main', 'AF_PT_fm_fluids', 'AF_PT_srf'):
    ok(fl.get(n) and not comb.get(n), "%s fluid only" % n)
print("FAILURES: %d" % fails[0])
sys.exit(1 if fails[0] else 0)
