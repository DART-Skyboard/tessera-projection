"""Time sequence of the single-engine demo with the combustion chain (ignition, steady plume on the ground, smoke fading at the domain edge).
blender --factory-startup -b --python docs/render_combustion.py -- [frames e.g. 7,40,150] [samples]
The plume is drawn with the Gaussian splat path (soft overlapping transparent sprites, colour from the web app's flame model)."""
import math, os, sys, tempfile, time
import bpy
from mathutils import Vector
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import arc_flow
arc_flow.register()
from arc_flow import runtime, gn, shading

argv = sys.argv[sys.argv.index('--') + 1:]
FRAMES = [int(x) for x in (argv[0] if argv else '7,40,150').split(',')]
SAMPLES = int(argv[1]) if len(argv) > 1 else 40
OUT = os.environ.get('ENGINE_OUT', os.path.join(HERE, 'renders'))
DOMAIN = float(os.environ.get('DOMAIN', '14'))
sc = bpy.context.scene
af = sc.arc_flow
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
af.cache_dir = tempfile.mkdtemp() + os.sep
sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'; sc.cycles.samples = SAMPLES; sc.cycles.use_denoising = True
sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = 1280, 720, int(os.environ.get('ENGINE_PCT', 100))
sc.view_settings.view_transform = 'AgX'
bpy.ops.arcflow.demo()
af.domain_radius = DOMAIN
af.particle_budget = 22000
af.xray = False
af.t_hi = 3800.0
runtime.mark_dirty()
sc.frame_start, sc.frame_end = 1, max(FRAMES) + 2
runtime.RT.scene_name = sc.name
runtime.start(sc, record=False, play=False)
# render setup on the live objects
runtime.ensure_world(sc, force=False)
for fl in af.fluids:
    fl.render_mode = 'SPLAT'
    fl.gs_scale = 1.25; fl.gs_blend = 3.0; fl.gs_resolution = 3; fl.gs_density = 1.0
runtime.mark_dirty(); runtime.ensure_world(sc, force=False)
for fl in af.fluids:
    for ob, vals in ((fl.splat_obj, {"Scale": fl.gs_scale, "Density": fl.gs_density, "Resolution": fl.gs_resolution}),):
        m = ob.modifiers.get("AF Render") if ob else None
        if m:
            gn.set_modifier_inputs(m, vals)
    if fl.gs_material:
        shading.set_params(fl.gs_material, AF_Emission=1.0, AF_Sharpness=fl.gs_blend)
    for ob in (fl.splat_obj,):
        if ob:
            ob.visible_shadow = False
w = sc.world or bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
bg = w.node_tree.nodes["Background"]; bg.inputs[0].default_value = (0.01, 0.012, 0.02, 1); bg.inputs[1].default_value = 1.0
floor = (af.env.ground_z if not af.env.ground_auto else -3.5)
bpy.ops.mesh.primitive_plane_add(size=400, location=(0, 0, floor)); fl_ = bpy.context.object
m = bpy.data.materials.new("floor"); m.use_nodes = True
b = m.node_tree.nodes['Principled BSDF']; b.inputs['Base Color'].default_value = (0.05, 0.055, 0.065, 1); b.inputs['Roughness'].default_value = 0.6
fl_.data.materials.append(m)
for nm, loc, en, col, sz in (("key", (-6, -10, 8), 2500, (1.0, 0.92, 0.82), 4.0), ("rim", (9, 7, 5), 2600, (0.55, 0.7, 1.0), 3.0)):
    ld = bpy.data.lights.new(nm, 'AREA'); ld.energy = en; ld.color = col; ld.size = sz
    lo = bpy.data.objects.new(nm, ld); sc.collection.objects.link(lo); lo.location = loc
    lo.rotation_euler = (Vector((0, 0, -1)) - lo.location).to_track_quat('-Z', 'Y').to_euler()
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
c = Vector((1.0, 1.0, 2.5)); cam.location = c + Vector((12.0, -30.0, 3.0)); cam.data.lens = 19
cam.rotation_euler = (c - cam.location).to_track_quat('-Z', 'Y').to_euler()
t0 = time.time()
for f in range(1, max(FRAMES) + 1):
    sc.frame_set(f)
    if f in FRAMES:
        sc.render.filepath = os.path.join(OUT, "combustion_f%03d.png" % f)
        t1 = time.time()
        bpy.ops.render.render(write_still=True)
        st = runtime.RT.world.bstats
        print("frame %d: %d alive, p_c %.1f bar, %d smoke, %d faded; simulated %.0f s, rendered in %.0f s" % (f, st['alive'], st['P_c'] / 1e5, st['smoke'], st['faded'], t1 - t0, time.time() - t1), flush=True)
runtime.stop(sc)
