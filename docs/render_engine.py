"""Polished stills of the single rocket-engine demo (Cycles CPU, 1280x720, the extension's own Points / Ray-march / Gaussian paths).
blender --factory-startup -b --python docs/render_engine.py -- <case> [samples]
case: beauty | flow | raymarch | xray | gaussian | all   (all = one sim, every still)"""
import math, os, sys, tempfile
import bpy
from mathutils import Vector
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import arc_flow
arc_flow.register()
from arc_flow import runtime

argv = sys.argv[sys.argv.index('--') + 1:]
CASES = ['beauty', 'flow', 'raymarch', 'xray', 'gaussian'] if argv[0] == 'all' else [argv[0]]
SAMPLES = int(argv[1]) if len(argv) > 1 else 56
OUT = os.environ.get('ENGINE_OUT', os.path.join(HERE, 'renders'))
FRAME = 36
FLOOR_Z = -3.62
sc = bpy.context.scene
af = sc.arc_flow
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
af.cache_dir = tempfile.mkdtemp() + os.sep
sc.render.engine = 'CYCLES'
sc.cycles.device = 'CPU'; sc.cycles.samples = SAMPLES; sc.cycles.use_denoising = True
try:
    sc.cycles.denoiser = 'OPENIMAGEDENOISE'
except Exception:
    pass
sc.cycles.max_bounces = 6; sc.cycles.volume_bounces = 2; sc.cycles.volume_step_rate = 2.5
sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = 1280, 720, int(os.environ.get('ENGINE_PCT', 100))
sc.render.image_settings.file_format = 'PNG'; sc.render.image_settings.color_mode = 'RGB'; sc.render.image_settings.compression = 100
sc.view_settings.view_transform = 'AgX'
try:
    sc.view_settings.look = 'AgX - Medium High Contrast'
except Exception:
    pass


def look_at(o, target):
    o.rotation_euler = (Vector(target) - o.location).to_track_quat('-Z', 'Y').to_euler()


def stage(center, dist, elev, azim, lens=45, fstop=None, focus=None):
    for o in [x for x in bpy.data.objects if x.name.startswith(('cam', 'key', 'rim', 'fill', 'Studio'))]:
        bpy.data.objects.remove(o, do_unlink=True)
    w = sc.world or bpy.data.worlds.new("studio"); w.use_nodes = True; sc.world = w
    bg = w.node_tree.nodes["Background"]; bg.inputs[0].default_value = (0.008, 0.01, 0.016, 1); bg.inputs[1].default_value = 1.0
    c = Vector(center)
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam)
    cam.location = c + Vector((math.cos(elev) * math.sin(azim), -math.cos(elev) * math.cos(azim), math.sin(elev))) * dist
    look_at(cam, c)
    cam.data.lens = lens; sc.camera = cam
    if fstop:
        cam.data.dof.use_dof = True; cam.data.dof.aperture_fstop = fstop
        cam.data.dof.focus_distance = focus or (cam.location - c).length
    for nm, loc, en, col, sz in (("key", (c.x - 6, c.y - 8, c.z + 8), 2200, (1.0, 0.92, 0.82), 4.0),
                                 ("rim", (c.x + 8, c.y + 6, c.z + 4), 2600, (0.55, 0.7, 1.0), 3.0),
                                 ("fill", (c.x + 4, c.y - 10, c.z), 500, (0.8, 0.85, 1.0), 8.0)):
        ld = bpy.data.lights.new(nm, 'AREA'); ld.energy = en; ld.color = col; ld.size = sz
        lo = bpy.data.objects.new(nm, ld); sc.collection.objects.link(lo); lo.location = loc; look_at(lo, c)
    bpy.ops.mesh.primitive_plane_add(size=400, location=(c.x, c.y, FLOOR_Z))
    fl = bpy.context.object; fl.name = "Studio_Floor"
    m = bpy.data.materials.new("floor"); m.use_nodes = True
    nt = m.node_tree; b = nt.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = (0.025, 0.028, 0.035, 1); b.inputs['Roughness'].default_value = 0.28; b.inputs['Metallic'].default_value = 0.3
    # heat glow under the nozzle: radial emission gradient
    tc = nt.nodes.new('ShaderNodeTexCoord'); mp = nt.nodes.new('ShaderNodeMapping'); gr = nt.nodes.new('ShaderNodeTexGradient'); ramp = nt.nodes.new('ShaderNodeValToRGB')
    gr.gradient_type = 'SPHERICAL'
    mp.vector_type = 'POINT'
    mp.inputs['Location'].default_value = (-(2.2 - c.x) * 0.3, c.y * 0.3, 0); mp.inputs['Scale'].default_value = (0.3, 0.3, 0.3)
    nt.links.new(tc.outputs['Object'], mp.inputs['Vector']); nt.links.new(mp.outputs['Vector'], gr.inputs['Vector']); nt.links.new(gr.outputs['Fac'], ramp.inputs['Fac'])
    ramp.color_ramp.interpolation = 'EASE'; ramp.color_ramp.elements[0].position = 0.25; ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
    ramp.color_ramp.elements[1].position = 1.0; ramp.color_ramp.elements[1].color = (1.0, 0.3, 0.04, 1)
    nt.links.new(ramp.outputs['Color'], b.inputs['Emission Color']); b.inputs['Emission Strength'].default_value = 1.6
    fl.data.materials.append(m)


def polish_parts():
    for o in bpy.data.objects:
        if o.name.startswith('AF_') and o.type == 'MESH' and (o.get('af_part') or 'Flange' in o.name or 'Lip' in o.name):
            for p in o.data.polygons:
                p.use_smooth = True
            m = o.data.materials[0] if o.data.materials else None
            if m and m.node_tree:
                b = m.node_tree.nodes.get('Principled BSDF')
                if b:
                    b.inputs['Anisotropic'].default_value = 0.5       # brushed metal
                    b.inputs['Roughness'].default_value = max(b.inputs['Roughness'].default_value, 0.32)


def sim_to(last):
    sc.frame_start, sc.frame_end = 1, max(last + 2, 40)
    runtime.RT.scene_name = sc.name
    runtime.start(sc, record=False, play=False)
    for f in range(1, last + 1):
        sc.frame_set(f)
    runtime.stop(sc)


def setup_fluids(mode, xray, gas_em=3.0, frame=FRAME, xe=None, wall=None, **kw):
    af.xray = xray
    if xe is not None:
        af.xray_emission = xe
    if wall is not None:
        af.xray_opacity = wall
    af.t_hi = 2300.0          # thermal tint range: the 2000 K exhaust reads orange-white, propellant stays cold blue
    runtime.apply_xray(sc)
    for fl in af.fluids:
        gas = fl.fluid_role == 'GENERIC' or 'gas' in fl.name.lower()
        fl.rm_color_a = (0.3, 0.45, 0.8)       # near-white: the volume's absorption colour would tint the transmitted light
        fl.rm_noise = 0.5
        if mode == 'RAYMARCH':
            fl.rm_density = 3.0 if gas else 1.5
            fl.rm_radius = 1.6 if gas else 1.0
            fl.rm_voxel = 0.09
            fl.rm_emission = 0.0
        fl.render_mode = mode
        for k, v in kw.items():
            setattr(fl, k, v)
    runtime.mark_dirty(); runtime.ensure_world(sc, force=False)
    from arc_flow import gn
    for fl in af.fluids:       # push the card values into the live render modifiers (ensure_world may reuse them)
        for ob, vals in ((fl.points_obj, {"Scale": fl.pt_scale}), (fl.splat_obj, {"Scale": fl.gs_scale, "Density": fl.gs_density, "Resolution": fl.gs_resolution}),
                         (fl.volume_obj, {"Density": fl.rm_density * 0.02, "Radius Scale": fl.rm_radius, "Voxel Size": fl.rm_voxel})):
            m = ob.modifiers.get("AF Render") if ob else None
            if m:
                gn.set_modifier_inputs(m, vals)
    from arc_flow import shading
    if mode == 'SPLAT':
        for fl in af.fluids:
            if fl.gs_material:
                shading.set_params(fl.gs_material, AF_Emission=2.6, AF_Sharpness=fl.gs_blend)
    for fl in af.fluids:
        if fl.fluid_role == 'GENERIC' or 'gas' in fl.name.lower():
            for attr, key in (('pt_material', 'AF_Emission'), ('gs_material', 'AF_Emission'), ('rm_material', 'AF_Emission')):
                m = getattr(fl, attr)
                if m:
                    shading.set_params(m, **{key: gas_em})
    if mode == 'RAYMARCH':       # hot exhaust: drive the volume's emission with a fixed flame colour instead of the ignition ramp
        for fl in af.fluids:
            m = fl.rm_material
            pv = next((n for n in m.node_tree.nodes if n.type == 'PRINCIPLED_VOLUME'), None)
            if pv is None:
                continue
            nt = m.node_tree
            for l in list(pv.inputs['Emission Color'].links) + list(pv.inputs['Emission Strength'].links):
                nt.links.remove(l)
            # the propellant that leaves the nozzle is heated: emission ramps up with depth below the nozzle (Z), orange-white at the floor
            pos = nt.nodes.new('ShaderNodeNewGeometry'); sep = nt.nodes.new('ShaderNodeSeparateXYZ'); mr = nt.nodes.new('ShaderNodeMapRange')
            mr.inputs['From Min'].default_value = -1.6; mr.inputs['From Max'].default_value = -3.0
            mr.inputs['To Min'].default_value = 0.0; mr.inputs['To Max'].default_value = gas_em
            nt.links.new(pos.outputs['Position'], sep.inputs['Vector']); nt.links.new(sep.outputs['Z'], mr.inputs['Value'])
            vi = nt.nodes.new('ShaderNodeVolumeInfo'); gate = nt.nodes.new('ShaderNodeMath'); gate.operation = 'MULTIPLY'     # emit only where the fluid is
            nt.links.new(mr.outputs['Result'], gate.inputs[0]); nt.links.new(vi.outputs['Density'], gate.inputs[1])
            nt.links.new(gate.outputs['Value'], pv.inputs['Emission Strength'])
            pv.inputs['Emission Color'].default_value = (1.0, 0.32, 0.05, 1.0)
    for fl in af.fluids:
        for ob in (fl.splat_obj, fl.volume_obj):
            if ob:
                ob.visible_shadow = False       # soft sprites / volumes should not cast hard shadows
    polish_parts()
    sc.frame_set(frame)


def shoot(name):
    sc.render.filepath = os.path.join(OUT, name)
    bpy.ops.render.render(write_still=True)
    print("wrote", sc.render.filepath, flush=True)


bpy.ops.arcflow.demo()
sim_to(FRAME)
for case in CASES:
    if case == 'beauty':
        stage((1.2, 1.2, -0.8), 17.5, 0.32, 0.95, lens=45)
        setup_fluids('POINTS', False, pt_scale=0.55)
        shoot("engine_beauty.png")
    elif case == 'flow':
        stage((0.0, 0.0, -0.3), 13.0, 0.14, 0.0, lens=50)
        setup_fluids('POINTS', True, xe=1.0, wall=0.22, frame=22, pt_scale=0.5)
        shoot("engine_flow_pipes.png")
    elif case == 'raymarch':
        stage((2.0, 0.0, -0.8), 22.0, 0.3, 0.6, lens=50)
        setup_fluids('RAYMARCH', False, gas_em=2.2)
        shoot("engine_plume_raymarch.png")
    elif case == 'xray':
        stage((0.2, 1.0, -0.6), 19.0, 0.28, 0.75, lens=45)
        setup_fluids('POINTS', True, xe=1.0, wall=0.18, pt_scale=0.45)
        shoot("engine_xray.png")
    elif case == 'gaussian':
        stage((1.2, 1.2, -0.8), 17.5, 0.32, 0.95, lens=45)
        setup_fluids('SPLAT', False, gs_scale=1.2, gs_blend=5.0, gs_resolution=2)
        shoot("engine_gaussian.png")
