"""Workbench preview stills of a solved party cache (debug aid).

    blender -b --python docs/preview_cache_frames.py -- WORKDIR OUTDIR f1 f2 ...
"""
import math, os, sys
import bpy, numpy as np
from mathutils import Vector
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import arc_flow
arc_flow.register()
from arc_flow import cloth_ops as CO
a = sys.argv[sys.argv.index('--') + 1:]
work, out, frames = a[0], a[1], [int(x) for x in a[2:]]
os.makedirs(out, exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=os.path.join(work, 'party.blend'))
sc = bpy.context.scene
z = np.load(os.path.join(work, 'party_cache.npz'))
X = z['x']
g = sc.arc_flow_cloth.garments[0]
CO.RT.live_running = True
CO.build_solver(sc, g)
sc.render.engine = 'BLENDER_WORKBENCH'
sc.display.shading.light = 'STUDIO'
sc.display.shading.color_type = 'OBJECT'
cols = [(0.1, 0.3, 1, 1), (0.1, 0.9, 0.3, 1), (1, 0.2, 0.2, 1)]
for o, c in zip([o for o, _s, _e in CO.RT.layout[g.name]], cols):
    o.color = c
    for m in o.modifiers:
        m.show_viewport = False; m.show_render = False
for o in bpy.data.objects:
    if o.type == 'MESH' and o.name not in [x.name for x, _s, _e in CO.RT.layout[g.name]] and not o.name.startswith('Mannequin'):
        o.hide_render = True
sc.render.resolution_x, sc.render.resolution_y = 700, 700
sc.view_settings.view_transform = 'Standard'
cam = sc.camera
for f in frames:
    sc.frame_set(f)
    CO.write_garment(g, X[f - 1])
    for k, (ang, nm) in enumerate(((-25, 'a'), (150, 'b'))):
        t = math.radians(ang)
        cam.location = (3.6 * math.sin(t), -3.6 * math.cos(t), 1.4)
        cam.rotation_euler = (Vector((0, 0, 0.7)) - cam.location).to_track_quat('-Z', 'Y').to_euler()
        sc.render.filepath = os.path.join(out, 'f%04d%s.png' % (f, nm))
        bpy.ops.render.render(write_still=True)
