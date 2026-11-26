"""Verification stills for the Tessera muscle solver (Cycles, region colour overlay, flex before / after).

    blender --factory-startup -b --python docs/render_muscle.py -- [outdir] [samples]

Writes muscle_regions.png, muscle_face_regions.png and muscle_flex_before_after.png to outdir (default docs/renders).
"""
import math
import os
import sys
import tempfile

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
os.environ.setdefault('TMPDIR', tempfile.gettempdir())
import arc_flow
arc_flow.register()
from arc_flow import muscle_ops as MO

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
OUT = argv[0] if argv else os.path.join(HERE, 'renders')
SAMPLES = int(argv[1]) if len(argv) > 1 else 48
os.makedirs(OUT, exist_ok=True)
for nm in ('Cube', 'Light', 'Camera'):
    if nm in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[nm], do_unlink=True)
sc = bpy.context.scene
ms = sc.arc_flow_muscle

sc.render.engine = 'CYCLES'
sc.cycles.samples = SAMPLES
sc.cycles.use_denoising = True
sc.cycles.device = 'CPU'
sc.render.film_transparent = False
w = bpy.data.worlds.new("W") if sc.world is None else sc.world
sc.world = w
w.use_nodes = True
w.node_tree.nodes['Background'].inputs[0].default_value = (0.20, 0.21, 0.23, 1)
w.node_tree.nodes['Background'].inputs[1].default_value = 0.35
sc.view_settings.view_transform = 'Standard'


def skin_material(use_color=True):
    m = bpy.data.materials.new("MuscleSkin")
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes['Principled BSDF']
    bsdf.inputs['Roughness'].default_value = 0.55
    if use_color:
        at = nt.nodes.new('ShaderNodeAttribute')
        at.attribute_name = MO.ATTR_SOLVED_COLOR
        nt.links.new(at.outputs['Color'], bsdf.inputs['Base Color'])
    else:
        bsdf.inputs['Base Color'].default_value = (0.78, 0.62, 0.52, 1)
    return m


def add_light_cam(loc, target, lens=85, ortho=None):
    cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
    sc.collection.objects.link(cam)
    cam.location = loc
    d = np.array(target) - np.array(loc)
    cam.rotation_euler = (math.atan2(math.hypot(d[0], d[1]), -d[2]) if False else math.pi / 2 - math.asin(d[2] / np.linalg.norm(d)) * 1.0 if False else 0, 0, 0)
    direction = __import__('mathutils').Vector(d)
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    cam.data.lens = lens
    sc.camera = cam
    for i, (p, e) in enumerate((((-2.5, -3.0, 3.0), 260), ((3.0, -2.0, 1.0), 90))):
        ld = bpy.data.objects.new("L%d" % i, bpy.data.lights.new("L%d" % i, 'AREA'))
        ld.data.energy, ld.data.size = e, 3.0
        ld.location = p
        ld.rotation_euler = (__import__('mathutils').Vector(target) - __import__('mathutils').Vector(p)).to_track_quat('-Z', 'Y').to_euler()
        sc.collection.objects.link(ld)
    return cam


def render(name, x, y):
    sc.render.resolution_x, sc.render.resolution_y = x, y
    sc.render.image_settings.file_format = 'PNG'
    sc.render.filepath = os.path.join(OUT, name)
    bpy.ops.render.render(write_still=True)
    print("rendered", sc.render.filepath, flush=True)


# floor
bpy.ops.mesh.primitive_plane_add(size=12, location=(0, 0, -0.005))
floor = bpy.context.active_object
fm = bpy.data.materials.new("Floor")
fm.use_nodes = True
fm.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.12, 0.12, 0.13, 1)
floor.data.materials.append(fm)

# ---------------------------------------------------------------- cinematic biped: regions, contours, solve
skin, arm = MO.build_example_biped(sc)
ms.preset, ms.detail = 'BIPED', 'CINEMATIC'
MO.apply_preset(sc)
MO.run_auto_seed(sc)
reg = {r.key: r for r in ms.regions}
v = MO.basis_coords(skin)
# contour lines: front of the left thigh, left biceps
def line(p0, p1, n=12):
    ids = []
    for t in np.linspace(0, 1, n):
        p = np.array(p0) * (1 - t) + np.array(p1) * t
        k = int(np.argmin(np.linalg.norm(v - p, axis=1)))
        if not ids or ids[-1] != k:
            ids.append(k)
    return ids
MO.save_strokes(skin, [dict(cid=reg['quadriceps'].cid, verts=line((0.11, -0.12, 0.50), (0.10, -0.12, 0.90))),
                       dict(cid=reg['biceps'].cid, verts=line((0.22, -0.05, 1.46), (0.45, -0.045, 1.26)))])
MO.update_contour_curve(sc, skin)
cm = bpy.data.materials.new("Contour")
cm.use_nodes = True
em = cm.node_tree.nodes['Principled BSDF']
em.inputs['Base Color'].default_value = (1, 0.9, 0.1, 1)
em.inputs['Emission Color'].default_value = (1, 0.8, 0.05, 1)
em.inputs['Emission Strength'].default_value = 2.0
bpy.data.objects[MO.CONTOUR_OBJ].data.materials.append(cm)
MO.run_solve(sc)
MO.set_active_color(skin, MO.ATTR_SOLVED_COLOR)
for r in ms.regions:
    r.flex = 2.5
MO.bake_shape_keys(sc)
skin.data.materials.append(skin_material(True))
arm.hide_render = True
arm.hide_viewport = True

cam = add_light_cam((0, -4.6, 0.98), (0, 0, 0.95), lens=75)
render("muscle_regions.png", 800, 1100)

# head close-up (face regions, cinematic)
cam.location = (0.0, -1.1, 1.70)
cam.rotation_euler = (__import__('mathutils').Vector((0, 1.1, -0.0))).to_track_quat('-Z', 'Y').to_euler()
cam.data.lens = 85
render("muscle_face_regions.png", 800, 800)

# ---------------------------------------------------------------- flex before / after with neutral skin
skin.data.materials.clear()
skin.data.materials.append(skin_material(False))
bpy.data.objects[MO.CONTOUR_OBJ].hide_render = True
cam.location = (0, -4.6, 0.98)
cam.rotation_euler = (__import__('mathutils').Vector((0, 4.6, -0.03))).to_track_quat('-Z', 'Y').to_euler()
cam.data.lens = 75
sk = skin.data.shape_keys
if sk.animation_data:
    for fc in sk.animation_data.drivers:
        fc.mute = True
for k in sk.key_blocks:
    if k.name.startswith(MO.KEY_PREFIX):
        k.value = 0.0
skin.modifiers['Armature'].show_viewport = True
render("_flex_rest.png", 800, 1100)

# pose: elbows and knees bent (chosen so hands come forward and heels go back), muscles at full flex
pb = arm.pose.bones
def best_rot(bone, tip_axis, sign):
    p = pb[bone]
    p.rotation_mode = 'XYZ'
    best, bv = None, -1e9
    for ax in (0, 2):
        for sg in (1, -1):
            e = [0, 0, 0]
            e[ax] = 1.2 * sg
            p.rotation_euler = e
            bpy.context.view_layer.update()
            score = sign * (p.tail.y - p.head.y) * (-1)
            if score > bv:
                best, bv = tuple(e), score
            p.rotation_euler = (0, 0, 0)
    p.rotation_euler = best
bpy.context.view_layer.update()
for s in ('L', 'R'):
    best_rot('forearm.' + s, 1, 1)
    best_rot('shin.' + s, 1, -1)
    best_rot('thigh.' + s, 1, 1)
bpy.context.view_layer.update()
for k in sk.key_blocks:
    if k.name.startswith(MO.KEY_PREFIX):
        k.value = 1.0
render("_flex_after.png", 800, 1100)

def _px(path):
    im = bpy.data.images.load(path)
    w, h = im.size
    arr = np.empty(w * h * 4, np.float32)
    im.pixels.foreach_get(arr)
    return arr.reshape(h, w, 4)


A, B = _px(os.path.join(OUT, "_flex_rest.png")), _px(os.path.join(OUT, "_flex_after.png"))
both = np.concatenate([A, B], axis=1)
out = bpy.data.images.new("ba", both.shape[1], both.shape[0], alpha=False)
out.pixels.foreach_set(both.ravel())
out.filepath_raw = os.path.join(OUT, "muscle_flex_before_after.png")
out.file_format = 'PNG'
out.save()
for f in ("_flex_rest.png", "_flex_after.png"):
    os.remove(os.path.join(OUT, f))
print("done", flush=True)
