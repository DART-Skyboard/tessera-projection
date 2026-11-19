"""Progress renders of Fluid mode (Cycles CPU, 960x540, 32 samples).

    blender --factory-startup -b --python docs/render_fluid.py -- lamp 1,80,160 docs/renders/fluid_lamp_{f}.png [max_particles]
    args: demo (lamp | plume | tower), comma separated frames to render, output png ({f} = the frame), optional max particles
The demo is built (Fill / Settle runs on build), the solver is stepped headless and each listed frame is rendered.
Env: FLUID_RES_PCT, FLUID_SAMPLES (quick previews), FLUID_CS (carrier particle scale)."""
import math
import os
import sys
import time

import bpy
import mathutils

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import arc_flow
from arc_flow import fluidmode_ops as FO

arc_flow.register()
args = sys.argv[sys.argv.index('--') + 1:]
demo, frames, out = args[0], [int(v) for v in args[1].split(',')], args[2]
maxp = int(args[3]) if len(args) > 3 else None
sc = bpy.context.scene
sc.render.fps = 24
loader = {'mix': FO.load_colour_mixing, 'lamp': FO.load_lava_lamp, 'plume': FO.load_gas_plume, 'tower': FO.load_gas_tower}[demo]
loader(sc, **({'max_particles': maxp} if maxp else {}))
for nm in ('Cube', 'Light', 'Camera'):   # the factory-startup scene objects
    if nm in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[nm])
sc.frame_start, sc.frame_end = 1, max(frames) + 2
t0 = time.time()
FO.start(sc, play=False)
print("built %d particles, fill %.1f s (%d steps)" % (FO.RT.solver.n, time.time() - t0, FO.RT.fill['steps'] if FO.RT.fill else 0), flush=True)
fm = sc.arc_flow_fm
if demo == 'tower':
    bpy.data.objects['TS_Tower'].hide_render = True
if demo == 'mix':
    for fl in fm.fluids:
        fl.particle_scale = float(os.environ.get('FLUID_PS', '1.6'))
    for nm, loc, en in (("TS_Key", (0.4, -0.7, 0.4), float(os.environ.get("FLUID_KEY", "35")), ), ("TS_Rim", (-0.4, 0.4, 0.3), 12.0)):
        ld = bpy.data.lights.new(nm, 'AREA'); ld.energy = en; ld.size = 0.4
        lo = bpy.data.objects.new(nm, ld); sc.collection.objects.link(lo); lo.location = loc
        lo.visible_camera = False; lo.visible_glossy = False
        lo.rotation_euler = (lo.location * -1 + mathutils.Vector((0, 0, 0.08))).to_track_quat('-Z', 'Y').to_euler()
if demo not in ('lamp', 'mix'):
    for fl in fm.fluids:
        fl.particle_scale = 3.5
    ld = bpy.data.lights.new("TS_Sun", 'SUN'); ld.energy = 4.0
    so = bpy.data.objects.new("TS_Sun", ld); sc.collection.objects.link(so); so.rotation_euler = (0.9, 0.3, 0.6)
if demo == 'lamp':   # frame the whole lamp; the carrier liquid is a faint tint, the wax blobs are larger and solid
    cam = sc.camera
    cam.data.lens = 36
    cam.location = (0.0, -0.95, 0.17)
    cam.rotation_euler = (math.radians(90.0), 0, 0)
    # the carrier liquid: a faint blue tint.  Its particles are hidden in the render and a volume mesh of the liquid (the cone up to the
    # particles' free surface) carries an absorbing blue volume shader, so no spray of particles shows outside the glass
    import bmesh
    zt = float(FO.RT.solver.x[FO.RT.solver.lo[0]:FO.RT.solver.hi[0], 2].max()) + 0.004
    rb, rt, hh = 0.075, 0.04, 0.36
    r_top = (rb + (rt - rb) * zt / hh) - 0.0012
    me = bpy.data.meshes.new("TS_Liquid")
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=64, radius1=rb - 0.0012, radius2=r_top, depth=zt)
    bmesh.ops.translate(bm, verts=bm.verts, vec=(0, 0, zt / 2))
    bm.to_mesh(me); bm.free()
    liq = bpy.data.objects.new("TS_Liquid", me)
    sc.collection.objects.link(liq)
    mt = bpy.data.materials.new("TS_LiquidMat"); mt.use_nodes = True
    nt = mt.node_tree; nt.nodes.clear()
    ob = nt.nodes.new('ShaderNodeOutputMaterial')
    tr = nt.nodes.new('ShaderNodeBsdfTransparent')
    pv = nt.nodes.new('ShaderNodeVolumeAbsorption')
    pv.inputs['Color'].default_value = (0.55, 0.78, 1.0, 1.0); pv.inputs['Density'].default_value = float(os.environ.get("FLUID_RM", "9.0"))
    nt.links.new(tr.outputs[0], ob.inputs['Surface']); nt.links.new(pv.outputs[0], ob.inputs['Volume'])
    liq.data.materials.append(mt)
    fm.fluids[0].points_obj.hide_render = True
    for fl in list(fm.fluids)[1:]:                  # the wax: overlapping spheres read as one glossy blob
        fl.particle_scale = 2.1
        if os.environ.get('FLUID_WAXSURF', '0') == '1':
            fl.render_mode = 'SURFACE'
    for nm, loc, en, sz in (("TS_Key", (0.5, -0.8, 0.45), 120.0, 0.4), ("TS_Rim", (-0.5, 0.5, 0.35), 60.0, 0.3)):
        ld = bpy.data.lights.new(nm, 'AREA'); ld.energy = en; ld.size = sz
        lo = bpy.data.objects.new(nm, ld); sc.collection.objects.link(lo); lo.location = loc
        lo.visible_glossy = False   # no mirror images of the area lights in the glass
        lo.visible_camera = False   # and the lights themselves are not seen through the glass (the white bar)
        lo.rotation_euler = (lo.location * -1 + mathutils.Vector((0, 0, 0.17))).to_track_quat('-Z', 'Y').to_euler()
FO.refresh_render(sc)
FO.apply_state(sc, FO.RT.start_snap)
w = sc.world or bpy.data.worlds.new("TS_World")
sc.world = w
w.use_nodes = True
bg = w.node_tree.nodes.get('Background')
bg.inputs['Color'].default_value = (0.012, 0.014, 0.02, 1) if demo in ('lamp', 'mix') else (0.03, 0.04, 0.06, 1)
bg.inputs['Strength'].default_value = 1.0
sc.render.engine = 'CYCLES'
if demo == 'mix':
    sc.view_settings.view_transform = 'Standard'
sc.cycles.device = 'CPU'
sc.cycles.use_denoising = os.environ.get('FLUID_DENOISE', '0') == '1'
sc.render.resolution_x, sc.render.resolution_y = 960, 540
sc.render.resolution_percentage = int(os.environ.get('FLUID_RES_PCT', '100'))
sc.cycles.samples = int(os.environ.get('FLUID_SAMPLES', '32'))
t0 = time.time()
for f in range(1, max(frames) + 1):
    sc.frame_set(f)
    if f in frames:
        sc.render.filepath = os.path.abspath(out.format(f=f))
        t1 = time.time()
        bpy.ops.render.render(write_still=True)
        print("frame %d (%d particles): simulated %.0f s so far, rendered in %.1f s" % (f, FO.RT.solver.n, t1 - t0, time.time() - t1), flush=True)
