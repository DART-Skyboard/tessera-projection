"""Renders of the wind tunnel example (thermal readback on the aircraft, air flowing past it).
    blender --factory-startup -b --python docs/render_windtunnel.py -- <hero|viewport|all> [frame] [outdir]
hero      Cycles CPU: the temperature shader on the aircraft, pathlines (smooth tubes through the cached particle positions) coloured by speed.
viewport  Workbench, Solid shading with the 'af_thermal_color' attribute and the particles as small spheres: what the 3D viewport shows.
Env: WT_SAMPLES (default 64), WT_PCT (resolution %, default 100), WT_PATHS (number of pathlines, default 160)."""
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import arc_flow
arc_flow.register()
from arc_flow import fluidmode_ops as FO
from arc_flow import thermal as TM
from arc_flow import windtunnel as WT

argv = sys.argv[sys.argv.index('--') + 1:]
CASES = ['hero', 'viewport'] if argv[0] == 'all' else [argv[0]]
FRAME = int(argv[1]) if len(argv) > 1 else 72
OUT = argv[2] if len(argv) > 2 else os.path.join(HERE, 'renders')
SAMPLES = int(os.environ.get('WT_SAMPLES', '64'))
PCT = int(os.environ.get('WT_PCT', '100'))
NPATH = int(os.environ.get('WT_PATHS', '160'))
sc = bpy.context.scene
sc.render.fps = 24
for nm in ('Cube', 'Light', 'Camera'):
    if nm in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[nm])

WT.load_example(sc)
sc.frame_start, sc.frame_end = 1, FRAME + 2
air = bpy.data.objects["Aircraft Airframe"]
p = air.arc_flow_th
p.t_min, p.t_max, p.speedup = 288.15, 312.0, 400.0
FO.start(sc, play=False)
for f in range(1, FRAME + 1):
    sc.frame_set(f)
print("simulated", FRAME, "frames,", FO.RT.solver.n, "particles", flush=True)
a = air.data.attributes['af_temperature']
arr = np.empty(len(a.data), np.float32)
a.data.foreach_get('value', arr)
print("wall temperature min %.2f mean %.2f max %.2f K" % (arr.min(), arr.mean(), arr.max()), flush=True)


def look_at(o, target):
    o.rotation_euler = (Vector(target) - o.location).to_track_quat('-Z', 'Y').to_euler()


def camera(loc, target, lens):
    cam = bpy.data.cameras.new("WT_Cam")
    co = bpy.data.objects.new("WT_Cam", cam)
    sc.collection.objects.link(co)
    co.location = loc
    look_at(co, target)
    cam.lens = lens
    cam.clip_start = 0.02
    sc.camera = co
    return co


def pathlines():
    """Smooth tubes through the cached positions of tracer particles that pass the aircraft."""
    cache = FO.RT.cache
    frames = sorted(f for f in cache if f <= FRAME)
    n0 = min(len(cache[f]['x']) for f in frames)                # emission grows the arrays: follow the particles that exist from the first frame
    X = np.stack([cache[f]['x'][:n0] for f in frames])        # (frames, n, 3)
    V = np.stack([np.linalg.norm(cache[f]['v'][:n0], axis=1) for f in frames])
    n = X.shape[1]
    ctr = np.array(air.matrix_world.translation)
    d = np.linalg.norm(X - ctr[None, None, :], axis=2).min(axis=0)
    seg = np.linalg.norm(np.diff(X, axis=0), axis=2).sum(axis=0)
    net = np.linalg.norm(X[-1] - X[0], axis=1) + 1e-6
    near = np.flatnonzero((d < 0.45) & (X[:, :, 0].max(axis=0) - X[:, :, 0].min(axis=0) > 1.0) & (seg / net < 1.25))
    rng = np.random.default_rng(3)
    pick = rng.choice(near, size=min(NPATH, len(near)), replace=False) if len(near) else np.zeros(0, int)
    cu = bpy.data.curves.new("WT_Paths", 'CURVE')
    cu.dimensions = '3D'
    cu.bevel_depth = 0.0028
    cu.bevel_resolution = 2
    cu.use_fill_caps = True
    mats = []
    for i, (nm, col) in enumerate((("slow", (0.25, 0.55, 1.0)), ("mid", (0.45, 0.85, 1.0)), ("fast", (0.9, 0.97, 1.0)))):
        m = bpy.data.materials.new("WT_Path_" + nm)
        m.use_nodes = True
        b = m.node_tree.nodes['Principled BSDF']
        b.inputs['Base Color'].default_value = (*col, 1)
        b.inputs['Emission Color'].default_value = (*col, 1)
        b.inputs['Emission Strength'].default_value = 2.2
        b.inputs['Roughness'].default_value = 0.4
        m.diffuse_color = (*col, 1)
        cu.materials.append(m)
    vmean = V.mean(axis=0)
    q = np.quantile(vmean[pick], [0.33, 0.66]) if len(pick) else (0, 0)
    k = 4
    for j in pick:
        pts = X[:, j, :]
        # moving average (smooths the parcel jitter), then a Bezier spline with automatic handles
        sm = np.stack([np.convolve(np.pad(pts[:, c], (k, k), mode='edge'), np.ones(2 * k + 1) / (2 * k + 1), mode='valid') for c in range(3)], axis=1)
        sm = sm[::2]
        sp = cu.splines.new('BEZIER')
        sp.bezier_points.add(len(sm) - 1)
        for bp, pt in zip(sp.bezier_points, sm):
            bp.co = pt
            bp.handle_left_type = bp.handle_right_type = 'AUTO'
        sp.material_index = 0 if vmean[j] < q[0] else (1 if vmean[j] < q[1] else 2)
    o = bpy.data.objects.new("WT_Paths", cu)
    sc.collection.objects.link(o)
    return o


def scene_look():
    tun = bpy.data.objects["Wind Tunnel"]
    m = tun.data.materials[0]
    b = m.node_tree.nodes['Principled BSDF']
    for nm, val in (('Transmission Weight', 0.0), ('Alpha', 0.10)):
        if nm in b.inputs:
            b.inputs[nm].default_value = val
    for fl in sc.arc_flow_fm.fluids:
        if fl.points_obj is not None:
            fl.points_obj.hide_render = True
    # floor below the tunnel
    bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, -0.02))
    fl = bpy.context.object
    fl.name = "Studio_Floor"
    fm = bpy.data.materials.new("floor")
    fm.use_nodes = True
    fb = fm.node_tree.nodes['Principled BSDF']
    fb.inputs['Base Color'].default_value = (0.03, 0.035, 0.045, 1)
    fb.inputs['Roughness'].default_value = 0.35
    fl.data.materials.append(fm)
    w = bpy.data.worlds.new("wt_world")
    w.use_nodes = True
    w.node_tree.nodes['Background'].inputs[0].default_value = (0.012, 0.016, 0.024, 1)
    w.node_tree.nodes['Background'].inputs[1].default_value = 1.0
    sc.world = w
    for nm, loc, en, size, col in (("key", (-1.0, -1.8, 3.2), 350, 2.5, (1.0, 0.95, 0.88)), ("rim", (2.2, 1.6, 2.2), 250, 2.0, (0.6, 0.75, 1.0)),
                                   ("fill", (-2.0, 0.4, 1.0), 80, 3.0, (0.8, 0.88, 1.0))):
        ld = bpy.data.lights.new(nm, 'AREA')
        ld.energy, ld.size, ld.color = en, size, col
        lo = bpy.data.objects.new(nm, ld)
        sc.collection.objects.link(lo)
        lo.location = loc
        look_at(lo, (0, 0, 0.75))


r = sc.render
r.resolution_x, r.resolution_y, r.resolution_percentage = 1280, 720, PCT
r.image_settings.file_format = 'PNG'
r.image_settings.compression = 100
ctr = Vector(air.matrix_world.translation)
for case in CASES:
    sc.frame_set(FRAME)
    for o in [x for x in bpy.data.objects if x.name.startswith(('WT_', 'key', 'rim', 'fill', 'Studio'))]:
        bpy.data.objects.remove(o, do_unlink=True)
    if case == 'hero':
        r.engine = 'CYCLES'
        sc.cycles.device = 'CPU'
        sc.cycles.samples = SAMPLES
        sc.cycles.use_denoising = True
        sc.cycles.max_bounces = 5
        sc.view_settings.view_transform = 'AgX'
        scene_look()
        pathlines()
        camera((1.75, -0.78, 1.15), (ctr.x + 0.05, 0.0, ctr.z - 0.03), 28)
        r.filepath = os.path.join(OUT, 'windtunnel_thermal.png')
    else:
        r.engine = 'BLENDER_WORKBENCH'
        sh = sc.display.shading
        sh.light = 'STUDIO'
        sh.color_type = 'VERTEX'
        sh.show_cavity = False
        sc.display.render_aa = '8'
        w = bpy.data.worlds.new("wt_world")
        w.color = (0.09, 0.10, 0.12)
        sc.world = w
        for fl in sc.arc_flow_fm.fluids:
            if fl.points_obj is not None:
                fl.points_obj.hide_render = False
                fl.particle_scale = 0.2
                FO.apply_state(sc, FO.RT.cache.get(FRAME))
        tun = bpy.data.objects["Wind Tunnel"]
        tun.hide_render = True
        camera((-1.7, -2.6, 1.7), (ctr.x + 0.1, 0.0, ctr.z - 0.1), 32)
        r.filepath = os.path.join(OUT, 'windtunnel_viewport.png')
    bpy.ops.render.render(write_still=True)
    print("rendered", r.filepath, flush=True)
    if case == 'hero':
        for o in [x for x in bpy.data.objects if x.name.startswith('WT_')]:
            pass
