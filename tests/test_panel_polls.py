"""Panel poll() with the vertical Tessera tab rail: the root rail panel stays visible in every
fm.mode and ui_tab; category panels are hosted inside the tab card (they do not stack in the
N-panel); the object-properties and Geometry Nodes editor panels keep their own polls.
Switching tabs never changes fm.mode.  Built-in category draw still works in both solver modes.

    blender --factory-startup -b --python tests/test_panel_polls.py [-- --installed]
"""
import importlib, os, sys
import bpy
HERE = os.path.dirname(os.path.abspath(__file__))
fails = [0]
def ok(c, m):
    print(("PASS " if c else "FAIL ") + m, flush=True)
    if not c: fails[0] += 1
if '--installed' not in sys.argv:
    sys.path.insert(0, os.path.join(HERE, '..')); import arc_flow; arc_flow.register()
    pkg = 'arc_flow'
else:
    import addon_utils
    pkg = [m.__name__ for m in addon_utils.modules() if m.__name__.endswith(('tessera_projection', 'arc_flow'))][0]
    if pkg not in bpy.context.preferences.addons:
        addon_utils.enable(pkg, default_set=True, persistent=False)
    importlib.import_module(pkg)
ui_tabs = importlib.import_module(pkg + '.ui_tabs')
print("Blender", bpy.app.version_string, flush=True)
sc = bpy.context.scene
bpy.ops.mesh.primitive_cube_add()

def vis(mode, tab=None):
    sc.arc_flow_fm.mode = mode
    if tab is not None:
        sc.arc_flow.ui_tab = tab
    out = {}
    for n in dir(bpy.types):
        if n.startswith('AF_PT_') and getattr(bpy.types, n).bl_space_type == 'VIEW_3D':
            c = getattr(bpy.types, n)
            out[n] = bool(c.poll(bpy.context)) if hasattr(c, 'poll') else True
    return out

ok(hasattr(bpy.types, 'AF_PT_tabs'), "AF_PT_tabs exists")
ok(hasattr(sc.arc_flow, 'ui_tab'), "scene.arc_flow.ui_tab exists")

# root rail is visible for every mode x tab; hosted category panels stay hidden
for mode in ('COMBUSTION', 'FLUID'):
    for tab in ui_tabs.TAB_IDS:
        v = vis(mode, tab)
        ok(v.get('AF_PT_tabs'), "AF_PT_tabs visible in mode=%s tab=%s" % (mode, tab))
        stacked = [n for n in ui_tabs.HOSTED_PANELS if v.get(n)]
        ok(not stacked, "no hosted panel stacks in the N-panel (mode=%s tab=%s): %s" % (mode, tab, stacked[:5]))

# switching tabs is a view change only
sc.arc_flow_fm.mode = 'COMBUSTION'
sc.arc_flow.ui_tab = 'REMESH'
sc.arc_flow.ui_tab = 'FLUID'
ok(sc.arc_flow_fm.mode == 'COMBUSTION', "setting ui_tab does not change fm.mode")
sc.arc_flow_fm.mode = 'FLUID'
sc.arc_flow.ui_tab = 'COMBUSTION'
ok(sc.arc_flow_fm.mode == 'FLUID', "COMBUSTION tab does not force fm.mode")

# Properties editor and Node Editor panels are not hosted
ok(bool(bpy.types.AF_PT_object_tab.poll(bpy.context)), "AF_PT_object_tab still polls for a mesh")
ok(hasattr(bpy.types, 'AF_PT_gn_nodes_editor'), "AF_PT_gn_nodes_editor remains registered")

# every former shared / mode-gated panel is still registered
for n in ('AF_PT_examples', 'AF_PT_shader', 'AF_PT_retopo', 'AF_PT_systems', 'AF_PT_env',
          'AF_PT_fm_mode', 'AF_PT_gn_nodes_view3d', 'AF_PT_main', 'AF_PT_parts', 'AF_PT_fluids',
          'AF_PT_solver', 'AF_PT_cache', 'AF_PT_fm_main', 'AF_PT_fm_fluids', 'AF_PT_srf',
          'AF_PT_windtunnel', 'AF_PT_thermal'):
    ok(hasattr(bpy.types, n), n + " still registered")

print("FAILURES: %d" % fails[0])
sys.exit(1 if fails[0] else 0)
