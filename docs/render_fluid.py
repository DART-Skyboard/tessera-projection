"""Progress renders of Fluid mode (Cycles CPU, 960x540, 32 samples).

    blender --factory-startup -b --python docs/render_fluid.py -- lamp 1 docs/renders/fluid_lamp_t0.png
    args: demo (lamp | plume | tower), frame to render, output png [max_particles]
The demo is built, the solver is stepped headless to the frame, then the frame is rendered."""
import os
import sys
import time

import bpy

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import arc_flow
from arc_flow import fluidmode_ops as FO

arc_flow.register()
args = sys.argv[sys.argv.index('--') + 1:]
demo, frame, out = args[0], int(args[1]), args[2]
maxp = int(args[3]) if len(args) > 3 else 8000
sc = bpy.context.scene
sc.render.fps = 24
{'lamp': FO.load_lava_lamp, 'plume': FO.load_gas_plume, 'tower': FO.load_gas_tower}[demo](sc, max_particles=maxp)
for nm in ('Cube', 'Light', 'Camera'):   # the factory-startup scene objects
    if nm in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[nm])
sc.frame_start, sc.frame_end = 1, max(frame + 2, 10)
FO.start(sc, play=False)
t0 = time.time()
for f in range(1, frame + 1):
    sc.frame_set(f)
print("simulated %d frames of %d particles in %.1f s (%.2f s/frame)" % (frame, FO.RT.solver.n, time.time() - t0, (time.time() - t0) / max(frame, 1)), flush=True)
if demo == 'lamp':   # frame the whole lamp; the carrier liquid as a translucent volume, the wax as solid blobs
    cam = sc.camera
    cam.data.lens = 36
    cam.location = (0.0, -0.95, 0.17)
    cam.rotation_euler = (__import__('math').radians(90.0), 0, 0)
    if os.environ.get('FLUID_CARRIER', 'points') == 'ray':
        sc.arc_flow_fm.fluids[0].render_mode = 'RAYMARCH'
    else:
        sc.arc_flow_fm.fluids[0].particle_scale = float(os.environ.get("FLUID_CS", "0.28"))
        for fl in list(sc.arc_flow_fm.fluids)[1:]:
            fl.particle_scale = 1.25
    if os.environ.get('FLUID_DEBUG') == 'noglass':
        bpy.data.objects['TS_LampGlass'].hide_render = True
    if os.environ.get('FLUID_DEBUG') == 'nolights':
        sc.arc_flow_fm.fluids[0].render_mode = 'POINTS'
    FO.refresh_render(sc)
    FO.apply_state(sc, FO.RT.cache.get(frame))
w = bpy.data.worlds.new("TS_World") if sc.world is None else sc.world
sc.world = w
w.use_nodes = True
bg = w.node_tree.nodes.get('Background')
bg.inputs['Color'].default_value = (0.012, 0.014, 0.02, 1) if demo == 'lamp' else (0.03, 0.04, 0.06, 1)
bg.inputs['Strength'].default_value = 1.0
if demo == 'tower':   # show the gas, not the opaque container shell
    bpy.data.objects['TS_Tower'].hide_render = True
if demo != 'lamp':
    for fl in sc.arc_flow_fm.fluids:
        fl.particle_scale = 3.5
    FO.refresh_render(sc)
    FO.apply_state(sc, FO.RT.cache.get(frame))
    ld = bpy.data.lights.new("TS_Sun", 'SUN'); ld.energy = 4.0
    so = bpy.data.objects.new("TS_Sun", ld); sc.collection.objects.link(so); so.rotation_euler = (0.9, 0.3, 0.6)
if demo == 'lamp':   # a soft key light and a rim light so the carrier and the wax read against the dark room
    for nm, loc, en, sz in (("TS_Key", (0.5, -0.8, 0.45), 120.0, 0.4), ("TS_Rim", (-0.5, 0.5, 0.35), 60.0, 0.3)):
        ld = bpy.data.lights.new(nm, 'AREA'); ld.energy = en; ld.size = sz
        lo = bpy.data.objects.new(nm, ld); sc.collection.objects.link(lo); lo.location = loc
        lo.visible_glossy = False   # no mirror images of the area lights in the glass (they read as white streaks)
        d = (lo.location * -1 + __import__('mathutils').Vector((0, 0, 0.17)))
        lo.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
sc.render.engine = 'CYCLES'
sc.cycles.device = 'CPU'
sc.cycles.samples = 32
sc.cycles.use_denoising = False
sc.render.resolution_x, sc.render.resolution_y = 960, 540
sc.render.resolution_percentage = int(os.environ.get('FLUID_RES_PCT', '100'))
sc.cycles.samples = int(os.environ.get('FLUID_SAMPLES', '32'))
sc.render.filepath = os.path.abspath(out)
t0 = time.time()
bpy.ops.render.render(write_still=True)
print("rendered %s in %.1f s" % (out, time.time() - t0), flush=True)
