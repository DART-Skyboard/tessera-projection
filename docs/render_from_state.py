"""Render a saved Fluid-mode state (npz of x, C, T, gasp, frozen from a simulated frame) for fast look iteration:
blender --factory-startup -b --python docs/render_from_state.py -- <demo> <state.npz> <out.png> [max_particles]"""
import os
import sys
import bpy
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import arc_flow
from arc_flow import fluidmode_ops as FO
arc_flow.register()
demo, npz, out = sys.argv[sys.argv.index('--') + 1:][:3]
maxp = int(sys.argv[sys.argv.index('--') + 4]) if len(sys.argv) > sys.argv.index('--') + 4 else 2500
sc = bpy.context.scene
{'steam': FO.load_steam, 'crystal': FO.load_crystal}[demo](sc, max_particles=maxp)
for nm in ('Cube', 'Light', 'Camera'):
    if nm in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[nm])
FO.start(sc, play=False)
S = FO.RT.solver
d = np.load(npz)
assert len(d['x']) == S.n, "particle count differs (%d vs %d)" % (len(d['x']), S.n)
S.x[:], S.C[:], S.T[:], S.gasp[:], S.frozen[:] = d['x'], d['C'], d['T'], d['gasp'], d['frozen']
S._derive(np.arange(S.n))
FO.apply_state(sc)
for h in (bpy.app.handlers.frame_change_pre, bpy.app.handlers.frame_change_post, bpy.app.handlers.depsgraph_update_post):
    h.clear()           # the render must show this state, not the solver's cached frame
KEY = float(os.environ.get('FLUID_KEY', '400'))
for nm, loc, en, sz in (("TS_Key", (0.9, -1.0, 0.7), KEY, 0.8), ("TS_Rim", (-0.8, 0.9, 0.5), KEY * 0.8, 0.8), ("TS_Under", (0.0, -0.8, -0.1), KEY * 0.3, 0.6)):
    ld = bpy.data.lights.new(nm, 'AREA'); ld.energy = en; ld.size = sz
    lo = bpy.data.objects.new(nm, ld); sc.collection.objects.link(lo); lo.location = loc
    d_ = (__import__('mathutils').Vector((0.0, 0.0, 0.25)) - lo.location)
    lo.rotation_euler = d_.to_track_quat('-Z', 'Y').to_euler()
    lo.visible_camera = False
    lo.visible_glossy = False
w = sc.world or bpy.data.worlds.new("TS_World"); sc.world = w; w.use_nodes = True
w.node_tree.nodes['Background'].inputs['Color'].default_value = (0.012, 0.014, 0.02, 1)
sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'
sc.view_settings.view_transform = 'Standard'
sc.cycles.use_denoising = os.environ.get('FLUID_DENOISE', '1') == '1'
sc.cycles.samples = int(os.environ.get('FLUID_SAMPLES', '48'))
sc.render.resolution_x, sc.render.resolution_y = int(os.environ.get('FLUID_W', '960')), int(os.environ.get('FLUID_H', '540'))
sc.render.resolution_percentage = int(os.environ.get('FLUID_RES_PCT', '100'))
sc.render.filepath = os.path.abspath(out)
print("n gas", int(S.gasp.sum()))
bpy.ops.render.render(write_still=True)
print("rendered", out, flush=True)
for nm in ("AFFMG_Phase", "AFFMC_Crystals", "AFFMS_Liquids"):
    o = bpy.data.objects.get(nm)
    print(nm, None if o is None else (len(o.data.vertices), o.hide_render, [m.name for m in o.modifiers]))
