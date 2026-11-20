"""Time sequence of the twin-engine example through the whole combustion chain (filled tanks, merge and ignition, neck pressure and choked nozzles, plume hits the
ground and flattens, flame turns into smoke, exhaust fades at the domain radius).  Liquids use the smooth Surface render type with shader presets, flame gas and
smoke are Gaussian splats.
blender --factory-startup -b --python docs/render_combustion_chain.py -- GOOD|BAD [frames e.g. 1,10,24,45,80] [samples]"""
import math, os, sys, tempfile, time
import bpy
from mathutils import Vector
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import arc_flow
arc_flow.register()
from arc_flow import runtime, gn, shading

argv = sys.argv[sys.argv.index('--') + 1:]
VARIANT = argv[0] if argv else 'GOOD'
FRAMES = [int(x) for x in (argv[1] if len(argv) > 1 else '4,12,24,40').split(',')]
SAMPLES = int(argv[2]) if len(argv) > 2 else 48
OUT = os.environ.get('ENGINE_OUT', os.path.join(HERE, 'renders'))
sc = bpy.context.scene
af = sc.arc_flow
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
af.cache_dir = tempfile.mkdtemp() + os.sep
sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'; sc.cycles.samples = SAMPLES; sc.cycles.use_denoising = True
sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = 1600, 800, int(os.environ.get('ENGINE_PCT', 100))
sc.view_settings.view_transform = 'AgX'
af.sys_variant = VARIANT
bpy.ops.arcflow.demo_twin()
for o in bpy.data.objects:
    if o.arc_flow.role == 'CONTAINER':
        o.arc_flow.particles = int(os.environ.get('TANK_PARTICLES', '900'))
af.domain_radius = float(os.environ.get('DOMAIN', '22'))
af.particle_budget = 26000
af.xray = True
af.xray_emission = 0.0
af.t_hi = 3800.0
runtime.mark_dirty()
sc.frame_start, sc.frame_end = 1, max(FRAMES) + 2
runtime.RT.scene_name = sc.name
runtime.start(sc, record=False, play=False)
runtime.ensure_world(sc, force=False)
for fl in af.fluids:
    fl.render_mode = 'SURFACE'
    if fl.fluid_role == 'FUEL': fl.color = (0.95, 0.55, 0.12)
    elif fl.fluid_role == 'OXIDIZER': fl.color = (0.45, 0.7, 1.0)
    fl.gs_scale = 0.45; fl.gs_blend = 3.0; fl.gs_resolution = 4; fl.gs_density = 0.5
runtime.mark_dirty(); runtime.ensure_world(sc, force=False)
for fl in af.fluids:
    ob = fl.splat_obj
    m = ob.modifiers.get("AF Render") if ob else None
    if m:
        gn.set_modifier_inputs(m, {"Scale": fl.gs_scale, "Density": fl.gs_density, "Resolution": fl.gs_resolution})
    if fl.gs_material:
        shading.set_params(fl.gs_material, AF_Emission=0.8, AF_Sharpness=3.0)
    if ob:
        ob.visible_shadow = False
w = sc.world or bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
bg = w.node_tree.nodes["Background"]; bg.inputs[0].default_value = (0.02, 0.025, 0.04, 1); bg.inputs[1].default_value = 1.0
floor = af.env.ground_z
bpy.ops.mesh.primitive_plane_add(size=400, location=(0, 0, floor)); fl_ = bpy.context.object
m = bpy.data.materials.new("floor"); m.use_nodes = True
b = m.node_tree.nodes['Principled BSDF']; b.inputs['Base Color'].default_value = (0.07, 0.075, 0.085, 1); b.inputs['Roughness'].default_value = 0.55
fl_.data.materials.append(m)
for nm, loc, en, col, sz in (("key", (-4, -14, 12), 6000, (1.0, 0.92, 0.82), 6.0), ("rim", (20, 8, 9), 5000, (0.55, 0.7, 1.0), 5.0)):
    ld = bpy.data.lights.new(nm, 'AREA'); ld.energy = en; ld.color = col; ld.size = sz
    lo = bpy.data.objects.new(nm, ld); sc.collection.objects.link(lo); lo.location = loc
    lo.rotation_euler = (Vector((6, 0, 0)) - lo.location).to_track_quat('-Z', 'Y').to_euler()
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
c = Vector((6.5, 1.5, -0.2)); cam.location = c + Vector((2.0, -34.0, 9.0)); cam.data.lens = 36
cam.rotation_euler = (c - cam.location).to_track_quat('-Z', 'Y').to_euler()
t0 = time.time()
for f in range(1, max(FRAMES) + 1):
    sc.frame_set(f)
    if f in FRAMES:
        sc.render.filepath = os.path.join(OUT, "combustion_chain_%s_f%03d.png" % (VARIANT.lower(), f))
        t1 = time.time()
        bpy.ops.render.render(write_still=True)
        wd = runtime.RT.world
        st = wd.bstats
        print("frame %d: %d alive, engines %s bar, %d smoke, %d faded, %d removed at domain; simulated %.0f s, rendered %.0f s" % (
            f, st['alive'], [round(e['P_c'] / 1e5, 1) for e in wd.engines], st['smoke'], st['faded'], st['removed_domain'], t1 - t0, time.time() - t1), flush=True)
runtime.stop(sc)
