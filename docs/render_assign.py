"""Renders of the Fluid mode assignment model (Cycles CPU, 960x540).

    blender --factory-startup -b --python docs/render_assign.py -- curve|particles|assets frames docs/renders/assign_{name}_{f}.png
curve      a fluid spawned from a curve (channel, Volume) next to a second one spawned along a curve (Points)
particles  a fluid spawned at the points of a Blender particle system
assets     three independent assets meeting on the ground and blending into one surface (frames e.g. 1,30,70)"""
import math
import os
import sys
import time

import bpy
import mathutils

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import arc_flow
from arc_flow import assets_demo
from arc_flow import fluidmode_ops as FO

arc_flow.register()
args = sys.argv[sys.argv.index('--') + 1:]
which, frames, out = args[0], [int(v) for v in args[1].split(',')], args[2]
sc = bpy.context.scene
fm = sc.arc_flow_fm
sc.render.fps = 24
for nm in ('Cube', 'Light', 'Camera'):
    if nm in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[nm])
col = FO._demo_common(sc)
fm.mode = 'FLUID'
fm.fill_mode = 'OFF'
fm.max_particles = 7000
sc.frame_start, sc.frame_end = 1, max(frames) + 2


def spiral(name, r0, z0, turns=2.5, rise=0.06, n=60, centre=(0.0, 0.0)):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('POLY'); sp.points.add(n - 1)
    for i, p in enumerate(sp.points):
        a = 2 * math.pi * turns * i / (n - 1)
        p.co = (centre[0] + r0 * math.cos(a), centre[1] + r0 * math.sin(a), z0 + rise * i / (n - 1) * turns * 3, 1.0)
    o = bpy.data.objects.new(name, cu); col.objects.link(o)
    return o


cam_loc, cam_look = (0.0, -1.25, 0.3), (0.0, 0.0, 0.12)
if which == 'curve':
    a = spiral("Channel A", 0.11, 0.08, centre=(-0.17, 0.0))
    a.arc_flow_fm.channel_radius = 0.022
    fa = FO.add_fluid(sc, 'WATER', "From curve (Volume)")
    fa.shape, fa.no_container, fa.spawn_mode, fa.color, fa.render_mode = a, True, 'VOLUME', (0.1, 0.45, 1.0), 'SURFACE'
    b = spiral("Channel B", 0.11, 0.08, centre=(0.17, 0.0))
    fb = FO.add_fluid(sc, 'WATER', "From curve (Points)")
    fb.shape, fb.no_container, fb.spawn_mode, fb.color, fb.render_mode = b, True, 'POINTS', (1.0, 0.55, 0.1), 'POINTS'
    fb.particle_scale = 1.4
    for o in (a, b):
        o.hide_render = True
    sc.frame_end = max(frames) + 1
elif which == 'particles':
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1, location=(0, 0, 0.14))
    em = bpy.context.object
    em.name = "Emitter"
    md = em.modifiers.new("ps", 'PARTICLE_SYSTEM')
    st = md.particle_system.settings
    st.count = 500; st.frame_start = 1; st.frame_end = 1; st.lifetime = 500; st.physics_type = 'NO'
    st.emit_from = 'VOLUME'; st.normal_factor = 0.0; st.render_type = 'NONE'
    em.hide_render = True
    sc.frame_set(1)
    fl = FO.add_fluid(sc, 'WATER', "From particles")
    fl.shape, fl.no_container, fl.spawn_mode, fl.color, fl.render_mode = em, True, 'PARTICLES', (0.15, 0.7, 0.5), 'POINTS'
    fl.particle_scale = 2.6
    sc.frame_end = max(frames) + 1
else:
    assets_demo.load(sc, max_particles=9000)
    for o in list(sc.objects):
        if o.name.endswith(" part"):
            o.hide_render = True
    cam_loc, cam_look = (0.0, -0.75, 0.16), (0.0, 0.0, 0.05)
    sc.frame_end = max(frames) + 1
sc.frame_start = 1
# stage
bpy.ops.mesh.primitive_plane_add(size=3.0, location=(0, 0, -0.001))
fl_ = bpy.context.object; fl_.name = "Floor"
mt = bpy.data.materials.new("FloorMat"); mt.use_nodes = True
mt.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.02, 0.022, 0.03, 1)
mt.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = 0.5
fl_.data.materials.append(mt)
for nm, loc, en, sz in (("TS_Key", (0.5, -0.7, 0.5), 35.0, 0.5), ("TS_Rim", (-0.5, 0.5, 0.4), 25.0, 0.4)):
    ld = bpy.data.lights.new(nm, 'AREA'); ld.energy = en; ld.size = sz
    lo = bpy.data.objects.new(nm, ld); sc.collection.objects.link(lo); lo.location = loc
    lo.rotation_euler = (mathutils.Vector((0, 0, 0.1)) - lo.location).to_track_quat('-Z', 'Y').to_euler()
cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam")); sc.collection.objects.link(cam)
cam.location = cam_loc
cam.rotation_euler = (mathutils.Vector(cam_look) - mathutils.Vector(cam_loc)).to_track_quat('-Z', 'Y').to_euler()
cam.data.lens = 40
sc.camera = cam
w = bpy.data.worlds.new("W"); sc.world = w; w.use_nodes = True
w.node_tree.nodes['Background'].inputs['Color'].default_value = (0.02, 0.025, 0.035, 1)
sc.render.engine = 'CYCLES'
sc.cycles.device = 'CPU'
sc.cycles.samples = int(os.environ.get('FLUID_SAMPLES', '32'))
sc.cycles.use_denoising = False
sc.render.resolution_x, sc.render.resolution_y = 960, 540
sc.render.resolution_percentage = int(os.environ.get('FLUID_RES_PCT', '100'))
sc.view_settings.view_transform = 'Standard'
FO.start(sc, play=False)
from arc_flow import shader_presets as SPR
for fl in fm.fluids:      # opaque coloured particles (the default water shader is a refractive glass that reads black at this size)
    if fl.render_mode == 'POINTS' and fl.points_obj is not None:
        nd = fl.points_obj.modifiers['AF Render'].node_group.nodes.get('AF_Material')
        if nd is not None:
            nd.material = SPR.build('SLIME', name="AssignParticle_" + fl.name, particle=True, reuse=False, Color=tuple(fl.color) + (1.0,))
FO.apply_state(sc, FO.RT.start_snap)
print("built %d particles" % FO.RT.solver.n, [fl.d_spawn for fl in fm.fluids], flush=True)
t0 = time.time()
for f in range(1, max(frames) + 1):
    sc.frame_set(f)
    if f in frames:
        sc.render.filepath = os.path.abspath(out.format(f=f))
        bpy.ops.render.render(write_still=True)
        print("frame %d rendered, %.0f s" % (f, time.time() - t0), flush=True)
