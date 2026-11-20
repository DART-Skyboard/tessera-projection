"""Contact sheet of the shader presets (Cycles CPU).

    blender --factory-startup -b --python docs/render_shaders.py -- docs/renders/shaders_contact.png [samples] [width]
Spheres on a checker floor with a checker backdrop (so refraction is visible); volumes are drawn inside a sphere with a soft edge."""
import math
import os
import sys

import bpy
import mathutils

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from arc_flow import shader_presets as SP

args = sys.argv[sys.argv.index('--') + 1:]
out = args[0]
samples = int(args[1]) if len(args) > 1 else 64
W = int(args[2]) if len(args) > 2 else 1600
sc = bpy.context.scene
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
cols = 7
keys = SP.ORDER
rows = (len(keys) + cols - 1) // cols
sp = 1.15


def checker(name, size, loc, rot, tiles):
    bpy.ops.mesh.primitive_plane_add(size=size, location=loc, rotation=rot)
    o = bpy.context.object
    o.name = name
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes['Principled BSDF']
    ck = nt.nodes.new('ShaderNodeTexChecker')
    ck.inputs['Scale'].default_value = tiles
    ck.inputs['Color1'].default_value = (0.85, 0.85, 0.85, 1)
    ck.inputs['Color2'].default_value = (0.06, 0.06, 0.07, 1)
    tc = nt.nodes.new('ShaderNodeTexCoord')
    nt.links.new(tc.outputs['UV'], ck.inputs['Vector'])
    nt.links.new(ck.outputs['Color'], bsdf.inputs['Base Color'])
    bsdf.inputs['Roughness'].default_value = 0.6
    o.data.materials.append(m)
    return o


wid = cols * sp
checker("Floor", 40.0, (0, 0, 0), (0, 0, 0), 120.0)
checker("Back", 40.0, (0, 1.2, 0), (math.pi / 2, 0, 0), 120.0)
for i, k in enumerate(keys):
    r, c = divmod(i, cols)
    x = (c - (cols - 1) / 2) * sp
    y = -(r - (rows - 1) / 2) * 0.0
    z = 0.5 + (rows - 1 - r) * 1.15
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=96, ring_count=48, location=(x, 0, z))
    o = bpy.context.object
    o.scale = (0.5, 0.5, 0.5)
    o.name = "S_" + k
    bpy.ops.object.shade_smooth()
    over = {}
    if SP.PRESETS[k][2] == 'VOLUME':
        over['Edge_softness'] = 0.9
        over['Density'] = {'SMOKE': 30.0, 'STEAM': 14.0, 'SOOT': 20.0, 'FLAME': 4.0, 'PLASMA': 3.0, 'GAS_CLOUD': 8.0}[k]
    if k == 'FLAME':
        over['Blackbody_intensity'] = 5.0
    if k == 'SMOKE':
        over['Blackbody_intensity'] = 0.0
    mat = SP.build(k, reuse=False, **over)
    SP.assign(o, mat)
    if SP.is_refractive(k):
        SP.caustics_hints(sc, casters=[o])
# lights and world
for nm, loc, en, sz in (("Key", (-3, -5, 6), 260.0, 4.0), ("Rim", (4, -3, 3), 120.0, 3.0)):
    ld = bpy.data.lights.new(nm, 'AREA')
    ld.energy, ld.size = en, sz
    lo = bpy.data.objects.new(nm, ld)
    sc.collection.objects.link(lo)
    lo.location = loc
    lo.rotation_euler = (mathutils.Vector((0, 0, 1.5)) - mathutils.Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
    lo.visible_camera = False
w = bpy.data.worlds.new("W")
sc.world = w
w.use_nodes = True
nt = w.node_tree
bg = nt.nodes['Background']
tc = nt.nodes.new('ShaderNodeTexCoord')
gr = nt.nodes.new('ShaderNodeTexGradient')
ramp = nt.nodes.new('ShaderNodeValToRGB')
ramp.color_ramp.elements[0].color = (0.04, 0.05, 0.07, 1)
ramp.color_ramp.elements[1].color = (0.3, 0.36, 0.46, 1)
mp = nt.nodes.new('ShaderNodeMapping')
mp.inputs['Rotation'].default_value = (math.pi / 2, 0, 0)
nt.links.new(tc.outputs['Generated'], mp.inputs['Vector'])
nt.links.new(mp.outputs[0], gr.inputs['Vector'])
nt.links.new(gr.outputs['Fac'], ramp.inputs['Fac'])
nt.links.new(ramp.outputs['Color'], bg.inputs['Color'])
bg.inputs['Strength'].default_value = 1.0
cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
sc.collection.objects.link(cam)
sc.camera = cam
cam.data.type = 'ORTHO'
cam.data.ortho_scale = cols * sp + 0.3
zc = 0.5 + (rows - 1) * 1.15 / 2
cam.location = (0, -10, zc)
cam.rotation_euler = (math.pi / 2, 0, 0)
sc.render.engine = 'CYCLES'
sc.cycles.device = 'CPU'
sc.cycles.samples = samples
sc.cycles.use_denoising = True
sc.cycles.max_bounces = 16
sc.cycles.transparent_max_bounces = 24
sc.cycles.volume_bounces = 3
sc.cycles.volume_step_rate = 1.0
sc.view_settings.view_transform = 'Standard'
sc.render.resolution_x = W
sc.render.resolution_y = int(W * (rows * 1.15 + 0.3) / (cols * sp + 0.3))
sc.render.filepath = os.path.abspath(out)
sc.render.image_settings.file_format = 'PNG'
bpy.ops.render.render(write_still=True)
