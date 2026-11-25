"""Cycles verification stills of Cloth Trim, Accents and Seam Seal Control Solving.

    bl.sh -r -b --factory-startup --python docs/render_cloth_trim.py -- [outdir] [samples] [width] [height]

Zips the jacket example shut over time (hem and collar trims, buttons, charms on hanging chains, sequins)
and writes four stills to <outdir>/cloth_trim_*.png (default docs/renders):
  cloth_trim_1_open.png  cloth_trim_2_zipping.png  cloth_trim_3_closed.png  cloth_trim_4_accents.png
"""
import math
import os
import sys
import tempfile

import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
os.environ.setdefault('TMPDIR', tempfile.gettempdir())
import arc_flow
arc_flow.register()
from arc_flow import cloth_ops as CO
from arc_flow import cloth_trim_ops as TO

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
OUT = argv[0] if argv else os.path.join(HERE, 'renders')
SAMPLES = int(argv[1] if len(argv) > 1 else os.environ.get('TRIM_SAMPLES', '48'))
W = int(argv[2] if len(argv) > 2 else os.environ.get('TRIM_WIDTH', '1200'))
H = int(argv[3] if len(argv) > 3 else os.environ.get('TRIM_HEIGHT', '1000'))
os.makedirs(OUT, exist_ok=True)

for nm in ('Cube', 'Light', 'Camera'):
    if nm in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[nm], do_unlink=True)
sc = bpy.context.scene
print(TO.load_jacket_example(sc), flush=True)
ts = sc.arc_flow_cloth_trim
seam = ts.seams[0]


def look(o, target):
    o.rotation_euler = (Vector(target) - o.location).to_track_quat('-Z', 'Y').to_euler()


def area(name, loc, target, energy, size, color=(1, 1, 1)):
    d = bpy.data.lights.new(name, 'AREA')
    d.energy, d.size, d.color = energy, size, color
    o = bpy.data.objects.new(name, d)
    sc.collection.objects.link(o)
    o.location = loc
    look(o, target)


def studio():
    w = sc.world or bpy.data.worlds.new("World")
    sc.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (0.045, 0.05, 0.06, 1)
    bg.inputs["Strength"].default_value = 1.0
    bpy.ops.mesh.primitive_plane_add(size=14, location=(0, 0, 0.0))
    fl = bpy.context.active_object
    fl.name = "Studio Floor"
    m = bpy.data.materials.new("Studio Floor")
    m.use_nodes = True
    m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.32, 0.31, 0.30, 1)
    m.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.6
    fl.data.materials.append(m)
    # backdrop wall behind the figure (+Y)
    bpy.ops.mesh.primitive_plane_add(size=14, location=(0, 2.4, 3.0), rotation=(math.pi / 2, 0, 0))
    wl = bpy.context.active_object
    wl.name = "Studio Wall"
    m2 = bpy.data.materials.new("Studio Wall")
    m2.use_nodes = True
    m2.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.20, 0.23, 0.27, 1)
    m2.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.9
    wl.data.materials.append(m2)
    area("Key", (-2.0, -2.2, 2.8), (0, 0, 1.15), 380, 1.6, (1.0, 0.95, 0.88))
    area("Fill", (2.4, -1.6, 1.6), (0, 0, 1.1), 150, 2.0, (0.80, 0.88, 1.0))
    area("Rim", (0.8, 2.0, 2.6), (0, 0, 1.2), 260, 1.0, (1.0, 1.0, 1.0))
    cam = bpy.data.cameras.new("Cam")
    cam.lens = 50
    co = bpy.data.objects.new("Cam", cam)
    sc.collection.objects.link(co)
    sc.camera = co
    return co


cam = studio()
sc.render.engine = 'CYCLES'
sc.cycles.device = 'CPU'
sc.cycles.samples = SAMPLES
sc.cycles.use_denoising = True
sc.cycles.max_bounces = 6
sc.render.resolution_x, sc.render.resolution_y = W, H
sc.render.resolution_percentage = 100
sc.render.image_settings.file_format = 'PNG'
sc.render.film_transparent = False
sc.view_settings.view_transform = 'Standard'


def shot(name, loc, target, lens=50):
    cam.location = loc
    cam.data.lens = lens
    look(cam, target)
    path = os.path.join(OUT, name)
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    print("wrote", path, os.path.getsize(path), "bytes", flush=True)


def step(n, progress_fn=None, f0=0):
    for k in range(n):
        if progress_fn is not None:
            seam.progress = float(progress_fn(f0 + k))
        CO.step_scene(sc, 1 / 60.0)
    return f0 + n


# settle on the mannequin, then zip
f = step(36, lambda k: 0.0)
shot("cloth_trim_1_open.png", (-0.55, -3.0, 1.30), (0.0, 0.0, 1.13), 62)
t0 = f
RAMP = 96


def zip_p(k):
    u = (k - t0) / RAMP
    return max(0.0, min(1.0, u))


f = step(int(RAMP * 0.52), zip_p, f)
shot("cloth_trim_2_zipping.png", (-0.55, -3.0, 1.30), (0.0, 0.0, 1.13), 62)
f = step(RAMP - int(RAMP * 0.52) + 8, zip_p, f)
f = step(40, lambda k: 1.0, f)
shot("cloth_trim_3_closed.png", (-0.55, -3.0, 1.30), (0.0, 0.0, 1.13), 62)
shot("cloth_trim_4_accents.png", (-0.28, -1.15, 1.40), (0.0, -0.1, 1.28), 55)
print("done", flush=True)
