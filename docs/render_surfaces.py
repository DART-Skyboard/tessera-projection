"""Renders of the surface sources and fluid volumes (Cycles CPU, small):
    blender --factory-startup -b --python docs/render_surfaces.py -- emit 40 docs/renders/surfaces_emit.png
    blender --factory-startup -b --python docs/render_surfaces.py -- volume 30 docs/renders/surfaces_volume.png
args: example (emit | volume), frame, output png.  Env: FLUID_RES_PCT, FLUID_SAMPLES."""
import math
import os
import sys

import bpy
import mathutils

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import arc_flow
from arc_flow import fluidmode_ops as FO
from arc_flow import surfaces_ops as SO

arc_flow.register()
which, frame, out = sys.argv[sys.argv.index('--') + 1:][:3]
frame = int(frame)
sc = bpy.context.scene
sc.render.fps = 24
for nm in ('Cube', 'Light', 'Camera'):
    if nm in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[nm])
SO.load_surface_example(sc, 'EMIT' if which == 'emit' else 'VOLUME')
sc.frame_start, sc.frame_end = 1, frame + 2
FO.start(sc, play=False)
for f in range(1, frame + 1):
    sc.frame_set(f)
fm = sc.arc_flow_fm
for fl in fm.fluids:
    fl.particle_scale = 1.5 if which == 'emit' else 0.7
# every fluid point object stays; hide the helper marker meshes in the render
for o in bpy.data.objects:
    if o.get('af_volume_points'):
        o.hide_render = True
ld = bpy.data.lights.new("TS_Sun", 'SUN'); ld.energy = 3.5
so = bpy.data.objects.new("TS_Sun", ld); sc.collection.objects.link(so); so.rotation_euler = (0.8, 0.2, 0.6)
cam = bpy.data.cameras.new("TS_Cam"); co = bpy.data.objects.new("TS_Cam", cam); sc.collection.objects.link(co); sc.camera = co
cam.lens = 40
tgt = mathutils.Vector((0.0, 0, 0.25)) if which == 'emit' else mathutils.Vector((0, 0, 1.3))
co.location = (0.0, -2.2, 0.8) if which == 'emit' else (0.0, -12.0, 3.0)
co.rotation_euler = (tgt - co.location).to_track_quat('-Z', 'Y').to_euler()
w = bpy.data.worlds.new("w"); w.use_nodes = True
w.node_tree.nodes['Background'].inputs[0].default_value = (0.05, 0.06, 0.08, 1)
sc.world = w
for o in bpy.data.objects:      # make the belt and nozzle visible as plain grey slabs
    if o.name.startswith("SRF_"):
        m = bpy.data.materials.new(o.name); m.diffuse_color = (0.5, 0.5, 0.55, 1)
        o.data.materials.append(m)
r = sc.render
r.engine = 'CYCLES'
sc.cycles.device = 'CPU'
sc.cycles.samples = int(os.environ.get('FLUID_SAMPLES', '24'))
r.resolution_x, r.resolution_y = 800, 450
r.resolution_percentage = int(os.environ.get('FLUID_RES_PCT', '100'))
r.filepath = os.path.abspath(out)
sc.frame_set(frame)
bpy.ops.render.render(write_still=True)
print("rendered", r.filepath, "particles", FO.RT.solver.n)
