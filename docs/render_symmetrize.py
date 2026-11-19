"""Before/after render for Symmetrize: asymmetric triangulated ellipsoid, remeshed without and with Symmetrize,
mirror-asymmetry shown as vertex colour (red = distance of the reflected surface point to the surface).
blender -b --factory-startup --python docs/render_symmetrize.py"""
import math, os, sys
import bpy, bmesh
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from arc_flow import retopo_ops as RO
from arc_flow.core import retopo as R

OUT = os.path.join(HERE, 'renders')
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'; sc.cycles.samples = 8; sc.cycles.use_denoising = False
sc.render.resolution_x = 480; sc.render.resolution_y = 560
sc.render.image_settings.file_format = 'PNG'
w = bpy.data.worlds.new("w"); w.use_nodes = True
w.node_tree.nodes["Background"].inputs[0].default_value = (0.93, 0.93, 0.93, 1); sc.world = w
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam)
cam.location = (0, -9.0, 0.3); cam.rotation_euler = (math.radians(90), 0, 0); cam.data.lens = 85; sc.camera = cam
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", 'SUN')); sun.data.energy = 3; sun.rotation_euler = (0.9, 0.2, 0.3)
sc.collection.objects.link(sun)

bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=1.0)
src = bpy.context.active_object
def bump(co, c, sg, a):
    return a * math.exp(-(co - Vector(c)).length_squared / (2 * sg * sg))
for v in src.data.vertices:
    e = Vector((v.co.x, v.co.y * 1.35, v.co.z * 1.7)); k = 1.0
    for sx in (-1, 1):
        k += bump(e, (0.9 * sx, 0.1, 0.9), 0.30, 0.35)
    k += bump(e, (0.0, -1.1, 1.0), 0.25, 0.20)
    k += bump(e, (0.8, -0.9, -0.5), 0.35, 0.22) - bump(e, (-0.7, -0.9, 0.1), 0.3, 0.12)   # asymmetric bump (+x) and dent (-x), facing the camera
    v.co = e * k
bm = bmesh.new(); bm.from_mesh(src.data); bmesh.ops.triangulate(bm, faces=bm.faces[:])
import random
rnd = random.Random(3)
bmesh.ops.rotate_edges(bm, edges=[e for e in bm.edges if len(e.link_faces) == 2 and rnd.random() < 0.3], use_ccw=False)
bm.to_mesh(src.data); bm.free()
src.rotation_euler = (0, 0, 0)

def mirror_colour(ob, n, p):
    bm = bmesh.new(); bm.from_mesh(ob.data); t = BVHTree.FromBMesh(bm)
    V = np.array([v.co[:] for v in bm.verts]); diag = R._bbox_diag(V); bm.free()
    n = np.array(n); p = np.array(p)
    Q = V - 2.0 * ((V - p) @ n)[:, None] * n[None, :]
    d = np.array([t.find_nearest(Vector(q))[3] for q in Q]) / diag
    a = ob.data.color_attributes.new("asym", 'FLOAT_COLOR', 'POINT')
    c = np.clip(d / 0.012, 0, 1)
    col = np.stack([0.80 + 0.2 * c, 0.80 * (1 - c) + 0.1 * c, 0.82 * (1 - c) + 0.1 * c, np.ones_like(c)], 1)
    a.data.foreach_set("color", col.astype(np.float32).ravel())
    m = bpy.data.materials.new("m"); m.use_nodes = True
    nt = m.node_tree; vc = nt.nodes.new("ShaderNodeAttribute"); vc.attribute_name = "asym"
    nt.links.new(vc.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
    ob.data.materials.clear(); ob.data.materials.append(m)
    for p in ob.data.polygons: p.use_smooth = False
    wf = ob.modifiers.new("wire", 'WIREFRAME'); wf.thickness = 0.012; wf.use_replace = False; wf.material_offset = 1
    wm = bpy.data.materials.new("wm"); wm.use_nodes = True
    wm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.05, 0.1, 0.45, 1)
    ob.data.materials.append(wm)
    return d.max(), d.mean()

# mirror line marker (the detected plane edge-on)
bpy.ops.mesh.primitive_cube_add(size=1); line = bpy.context.active_object
line.scale = (0.006, 0.01, 3.2); line.location = (0, 0.0, 0); line.hide_render = True
lm = bpy.data.materials.new("lm"); lm.use_nodes = True
lm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.1, 0.1, 0.1, 1); line.data.materials.append(lm)

def render(ob, name):
    for o in sc.objects:
        if o.type == 'MESH' and o is not line:
            o.hide_render = o is not ob
    line.hide_render = False
    line.location.y = 2.5          # behind the object: a dark centre line through the plane
    sc.render.filepath = os.path.join(OUT, name + ".png")
    bpy.ops.render.render(write_still=True)

st = dict(crease_density=0.0, resolution=1.0)
off, rep_off = RO.remesh_object(src, R.RemeshSettings(symmetrize=False, use_symmetry=False, **st), suffix="_off")
on, rep_on = RO.remesh_object(src, R.RemeshSettings(symmetrize=True, **st), suffix="_on")
print("on:", rep_on["symmetrize"])
res = {}
for name, ob in (("src", src), ("off", off), ("on", on)):
    res[name] = mirror_colour(ob, rep_on['symmetry']['normal'], rep_on['symmetry']['point'])
    print(name, "mirror dist max %.4f mean %.5f" % res[name])
    render(ob, "_sym_" + name)
# stitch the three renders side by side
imgs = [bpy.data.images.load(os.path.join(OUT, "_sym_%s.png" % n)) for n in ("src", "off", "on")]
w_, h_ = imgs[0].size
px = [np.array(i.pixels[:]).reshape(h_, w_, 4) for i in imgs]
sheet = np.concatenate(px, axis=1)
out = bpy.data.images.new("sheet", w_ * 3, h_, alpha=False)
out.pixels = sheet.ravel().tolist()
out.filepath_raw = os.path.join(OUT, "retopo_symmetrize.png"); out.file_format = 'PNG'; out.save()
for n in ("src", "off", "on"):
    os.remove(os.path.join(OUT, "_sym_%s.png" % n))
