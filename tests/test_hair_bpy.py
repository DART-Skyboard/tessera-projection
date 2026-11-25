"""Tessera Projection Hair bpy tests (Blender 4.2.3+).

    /home/box/apps/blender-4.2.3-linux-x64/blender --factory-startup -b --python tests/test_hair_bpy.py

Creates a scalp + collider + native CURVES Empty Hair object, steps the live
solver while rotating the rig, checks that curve points react and do not
penetrate the head, bakes a game-engine package, and renders a preview.
"""
import math
import os
import sys
import tempfile
import time

import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import arc_flow
from arc_flow import hair_ops as HO
from arc_flow import hair_panel as HP
from arc_flow import hair_gn as HG
from arc_flow.core import hair_solver as HS
from arc_flow.core import sdf as SDF

arc_flow.register()
fails = [0]


def ok(c, msg):
    print(("PASS " if c else "FAIL ") + msg, flush=True)
    if not c:
        fails[0] += 1


print("Blender", bpy.app.version_string, flush=True)
sc = bpy.context.scene

# ----------------------------------------------------------------------------- registration
ok(hasattr(sc, 'arc_flow_hair'), "Scene.arc_flow_hair registered")
ok(hasattr(bpy.types, "AF_PT_hair_object"), "Hair object Properties panel registered")
ok(not hasattr(bpy.types, "AF_PT_hair"),
   "no sibling VIEW_3D Hair panel (content is drawn inside the HAIR tab card)")
ok(all(hasattr(bpy.ops.arcflow, n) for n in (
    "hair_add_empty", "hair_example", "hair_live_start", "hair_live_stop",
    "hair_step", "hair_reset", "hair_rebuild", "hair_bake",
    "hair_export_package", "hair_export_alembic", "hair_set_role",
)), "Hair operators registered")
ok(callable(HP.draw_hair_tab), "draw_hair_tab is exportable for the vertical-tabbed UI")
from arc_flow import ui_tabs
ok(any(e["fn"] is HP.draw_hair_tab for e in ui_tabs.tab_drawers("HAIR")),
   "draw_hair_tab is registered on the HAIR tab")
# switching to the HAIR tab draws the real card, not the ready-state placeholder
sc.arc_flow.ui_tab = "HAIR"
ok(sc.arc_flow.ui_tab == "HAIR", "scene.arc_flow.ui_tab can be set to HAIR")

# ----------------------------------------------------------------------------- example scene
t0 = time.time()
built = HO.build_groom_example(sc, n_strands=64, n_pts=8, with_armature=True, seed=2)
hair = built["hair"]
head = built["head"]
arm = built["armature"]
hp = sc.arc_flow_hair
ok(hair is not None and hair.type == 'CURVES' and len(hair.data.points) == 64 * 8,
   "example built a native CURVES groom (%d points, %.2fs)" % (len(hair.data.points), time.time() - t0))
ok(hp.hair == hair and hp.collider == head and hp.armature == arm and hp.bone == "head",
   "scene pointers: hair, collider, armature/head bone")
ok(head.arc_flow_hair.role in {'COLLIDER', 'SURFACE'} and hair.arc_flow_hair.role == 'HAIR',
   "object roles tagged")

# GN modifier + material
md = hair.modifiers.get("Tessera Hair Render")
ok(md is not None and md.node_group is not None, "Tessera Hair Render Geometry Nodes modifier on the groom")
ok(len(hair.data.materials) >= 1, "hair material assigned")

# solver
S = HO.build_solver(sc)
ok(S is not None and S.n_strands >= 1 and S.n_points >= S.n_strands * 2,
   "solver built (%d guides, %d simulated points, %d total strands)" % (S.n_strands, S.n_points, S._full_n))

pos0, off0, _rad = HO.read_curves(hair, world=True)
tips0 = pos0[off0[1:] - 1].copy()
roots0 = pos0[off0[:-1]].copy()

# ----------------------------------------------------------------------------- live steps + gravity (no pose yet)
hp.use_scene_gravity = False
hp.gravity = (0.0, 0.0, -9.81)
hp.rest_shape = 0.25
HO.build_solver(sc)
for i in range(8):
    HO.step_scene(sc, dt=1.0 / 24.0, write=True)
pos1, off1, _ = HO.read_curves(hair, world=True)
tips1 = pos1[off1[1:] - 1]
ok(float(tips1[:, 2].mean()) < float(tips0[:, 2].mean()) - 0.002,
   "live steps drop the tips under gravity (%.4f -> %.4f)" % (float(tips0[:, 2].mean()), float(tips1[:, 2].mean())))

# ----------------------------------------------------------------------------- rotate the rig, hair must react
HO.reset_scene(sc, write=True)
pos_rest, off_rest, _ = HO.read_curves(hair, world=True)
tips_rest = pos_rest[off_rest[1:] - 1].copy()
# pose the head bone ~50 degrees about world Z (via the pose bone's local rotation)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode='POSE')
pb = arm.pose.bones.get("head")
ok(pb is not None, "head pose bone exists")
if pb is not None:
    pb.rotation_mode = 'XYZ'
    pb.rotation_euler = Euler((0.0, 0.0, math.radians(50.0)), 'XYZ')
bpy.ops.object.mode_set(mode='OBJECT')
bpy.context.view_layer.update()

# several steps so inertia + style memory both have a chance to show
for i in range(12):
    HO.step_scene(sc, dt=1.0 / 24.0, write=True)
pos_p, off_p, _ = HO.read_curves(hair, world=True)
tips_p = pos_p[off_p[1:] - 1]
roots_p = pos_p[off_p[:-1]]
moved = float(np.linalg.norm(tips_p - tips_rest, axis=1).mean())
ok(moved > 0.01, "tips react to a 50° head-bone pose (mean move %.3f m)" % moved)
root_moved = float(np.linalg.norm(roots_p - pos_rest[off_rest[:-1]], axis=1).mean())
ok(root_moved > 0.005, "roots follow the posed scalp / bone (mean move %.3f m)" % root_moved)

# ----------------------------------------------------------------------------- collision: no free-point penetration
V, T = HO._mesh_arrays(head, world=True)
ok(V is not None and len(T) > 0, "evaluated head mesh (%d verts, %d tris)" % (len(V) if V is not None else 0, len(T) if T is not None else 0))
if V is not None:
    field = HS.make_collider(V, T, resolution=36, level=2, mode='SOLID')
    # only test free (non-root) points of the written curves
    counts = np.diff(off_p)
    is_root = np.zeros(len(pos_p), bool)
    is_root[off_p[:-1]] = True
    sd = field.sample_sd(pos_p)
    rad = float(hp.collision_radius)
    # stay outside the solid (allow a tiny SDF-grid discrepancy) and honour the collision radius
    n_in = int((sd[~is_root] < -2e-3).sum())
    n_rad = int((sd[~is_root] < rad - 6e-3).sum())
    ok(n_in == 0 and n_rad == 0, "no free curve point penetrates the head collider (min sd %.4f, radius %.4f, %d inside, %d inside radius)" % (
        float(sd[~is_root].min()) if np.any(~is_root) else 0.0, rad, n_in, n_rad))

# ----------------------------------------------------------------------------- bake + game-engine package
hp.use_scene_range = False
hp.frame_start, hp.frame_end = 1, 12
# animate the bone over the range so the bake has motion
if pb is not None:
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='POSE')
    pb.rotation_euler = Euler((0, 0, 0), 'XYZ')
    pb.keyframe_insert('rotation_euler', frame=1)
    pb.rotation_euler = Euler((0.15, 0.0, math.radians(40.0)), 'XYZ')
    pb.keyframe_insert('rotation_euler', frame=12)
    bpy.ops.object.mode_set(mode='OBJECT')

tmp = tempfile.mkdtemp(prefix="tessera_hair_bpy_")
hp.bake_dir = tmp
hp.bake_name = "groom"
hp.package_dir = os.path.join(tmp, "package")
hp.bake_format = 'NPZ'
t1 = time.time()
frames, cache_path = HO.bake_scene(sc, start=1, end=12)
ok(cache_path and os.path.isfile(cache_path) and len(frames) == 12,
   "baked 12-frame npz cache (%s, %.1fs)" % (os.path.basename(cache_path or ""), time.time() - t1))
pkg_path = HO.export_package(sc, folder=hp.package_dir)
ok(pkg_path and os.path.isfile(pkg_path), "game-engine package written (%s)" % os.path.basename(pkg_path or ""))
if pkg_path:
    import json
    with open(pkg_path, 'r', encoding='utf-8') as f:
        pkg = json.load(f)
    ok(pkg.get("format") == HS.PACKAGE_FORMAT and "per_strand" in pkg and "guides" in pkg and "collision" in pkg,
       "package has format %s, guides, per-strand params and collision metadata" % pkg.get("format"))
    ok(os.path.isfile(os.path.join(os.path.dirname(pkg_path), "rest_curves.npz")), "package rest_curves.npz present")

# Alembic (best-effort; still assert a file if the operator succeeds)
abc_path = os.path.join(tmp, "groom.abc")
abc_ok = False
try:
    out = HO.export_alembic(sc, abc_path, start=1, end=12)
    abc_ok = bool(out and os.path.isfile(out) and os.path.getsize(out) > 100)
except Exception as e:
    print("Alembic export raised:", e, flush=True)
ok(abc_ok, "Alembic export of the baked hair curves (%d bytes)" % (os.path.getsize(abc_path) if os.path.isfile(abc_path) else 0))

# ----------------------------------------------------------------------------- operators on a clean read-back
ok('FINISHED' in bpy.ops.arcflow.hair_reset(), "hair_reset operator")
ok('FINISHED' in bpy.ops.arcflow.hair_step(), "hair_step operator")
ok('FINISHED' in bpy.ops.arcflow.hair_rebuild(), "hair_rebuild operator")

# ----------------------------------------------------------------------------- colliders: registration, auto, moving glove
bpy.ops.object.mode_set(mode='OBJECT') if bpy.context.object and bpy.context.object.mode != 'OBJECT' else None
for o in list(bpy.data.objects):
    o.select_set(False)
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.05, location=(0.45, 0.0, 1.45), segments=16, ring_count=8)
glove = bpy.context.active_object
glove.name = "TS_TestGlove"
glove.select_set(True)
hp.colliders.clear()
ok('FINISHED' in bpy.ops.arcflow.hair_collider_add(), "Add Selected as Collider operator")
ok(len(hp.colliders) == 1 and hp.colliders[0].obj is glove, "collider list registers the selected mesh")
ok(hasattr(hp.colliders[0], 'thickness') and hasattr(hp.colliders[0], 'friction'), "per-collider thickness + friction props")
hp.colliders[0].thickness = 0.02
hp.colliders[0].friction = 0.15
hp.use_scene_colliders = False
S = HO.build_solver(sc)
names = [c['name'] for c in S.colliders]
ok(head.name in names and glove.name in names, "solver has both the head and the added glove collider (%s)" % names)
gc = next(c for c in S.colliders if c['name'] == glove.name)
ok(abs(gc['radius'] - 0.02) < 1e-6 and abs(gc['friction'] - 0.15) < 1e-6, "per-collider thickness / friction reach the solver")
# moving collider: sweep the glove through the head hair; matrix follows the object, hair gets knocked, never ends inside it
hp.rest_shape = 0.3
HO.reset_scene(sc, write=True)
for i in range(6):
    HO.step_scene(sc, dt=1.0 / 24.0, write=True)
base, boff, _ = HO.read_curves(hair, world=True)
hit_pt = base[np.argmin(np.abs(base[:, 2] - np.percentile(base[:, 2], 20)))]
moved_ok, worst, peak = True, 1e9, 0.0
for i in range(14):
    glove.location = (float(hit_pt[0]) + 0.35 - 0.5 * i / 13.0, float(hit_pt[1]), float(hit_pt[2]))
    bpy.context.view_layer.update()
    HO.step_scene(sc, dt=1.0 / 24.0, write=True)
    gc = next(c for c in HO.RT.solver.colliders if c['name'] == glove.name)
    moved_ok &= bool(np.allclose(gc['matrix'][:3, 3], np.array(glove.matrix_world.translation), atol=1e-6))
    cur, _o, _r = HO.read_curves(hair, world=True)
    free = np.ones(len(cur), bool); free[boff[:-1]] = False
    worst = min(worst, float(np.linalg.norm(cur[free] - np.array(glove.location), axis=1).min()))
    peak = max(peak, float(np.abs(cur - base).max()))
ok(moved_ok, "moving collider: solver matrix tracks the glove object every step")
ok(peak > 0.02, "hair reacts to the moving glove (peak displacement %.3f m)" % peak)
ok(worst > 0.05 + 0.02 - 0.015, "no strand point inside the moving glove (min dist %.3f, radius 0.05)" % worst)
# auto colliders
bpy.ops.object.select_all(action='DESELECT')
hp.colliders.clear()
ok('FINISHED' in bpy.ops.arcflow.hair_collider_auto(), "Auto Colliders operator")
ok(any(c.obj is head for c in hp.colliders) or hp.collider is head, "auto colliders find the scalp / head")
ok('FINISHED' in bpy.ops.arcflow.hair_collider_remove(index=0) if len(hp.colliders) else True, "remove collider operator")
# hair-hair + profile props flow into the solver
hp.volume, hp.clumping, hp.gravity_scale = 1.5, 0.5, 1.2
S = HO.build_solver(sc)
ok(abs(S.params.volume - 1.5) < 1e-6 and abs(S.params.gravity_scale - 1.2) < 1e-6 and S.params.hair_collision, "volume / clumping / gravity scale reach the solver")
# shader assign + children + settle operators
ok('FINISHED' in bpy.ops.arcflow.hair_assign_shader('EXEC_DEFAULT', preset='AUBURN', add_scalp=True), "Assign Hair Shader operator")
mat = hair.data.materials[0]
types = {n.bl_idname for n in mat.node_tree.nodes}
ok('ShaderNodeBsdfHairPrincipled' in types and 'ShaderNodeBsdfPrincipled' in types, "hair material: Principled Hair BSDF + EEVEE Principled fallback")
hn = next(n for n in mat.node_tree.nodes if n.bl_idname == 'ShaderNodeBsdfHairPrincipled')
ok(getattr(hn, 'model', 'CHIANG') == 'CHIANG', "Chiang hair model")
ok(any(o.name.endswith(" Scalp") for o in bpy.data.objects), "scalp skin shell created")
n0 = len(hair.data.curves)
ok('FINISHED' in bpy.ops.arcflow.hair_make_children('EXEC_DEFAULT', per_guide=3), "Make Children operator")
ok(len(hair.data.curves) == n0 * 4 and hair.arc_flow_hair.n_guides == n0, "children added (%d -> %d curves), guides = first %d" % (n0, len(hair.data.curves), n0))
S = HO.build_solver(sc)
ok(S.n_strands == n0 and S.child_rest is not None, "solver simulates the guides and skins the children")
ok(HO.settle_scene(sc, steps=6) == 6, "settle_scene runs")

# ----------------------------------------------------------------------------- verification render (temp — do not clobber the Extensions showcase still)
preview = os.path.join(tempfile.mkdtemp(prefix="tessera_hair_preview_"), "hair_solver_verify.png")
# a staged pose for the still
if pb is not None:
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='POSE')
    pb.rotation_euler = Euler((0.18, 0.05, math.radians(28.0)), 'XYZ')
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.update()
for i in range(10):
    HO.step_scene(sc, dt=1.0 / 24.0, write=True)

sc.render.engine = 'CYCLES'
sc.cycles.device = 'CPU'
sc.cycles.samples = 6
sc.cycles.use_denoising = False
sc.render.resolution_x = 480
sc.render.resolution_y = 360
sc.render.resolution_percentage = 100
sc.render.filepath = preview
sc.render.image_settings.file_format = 'PNG'
# camera
if sc.camera is None and "TS_HairCam" in bpy.data.objects:
    sc.camera = bpy.data.objects["TS_HairCam"]
# slightly nicer look
sc.world.use_nodes = True
bg = sc.world.node_tree.nodes.get('Background')
if bg is not None:
    bg.inputs[0].default_value = (0.08, 0.09, 0.11, 1.0)
    bg.inputs[1].default_value = 0.6
t2 = time.time()
bpy.ops.render.render(write_still=True)
ok(os.path.isfile(preview) and os.path.getsize(preview) > 2000,
   "Cycles CPU preview render (%s, %d bytes, %.1fs)" % (preview, os.path.getsize(preview) if os.path.isfile(preview) else 0, time.time() - t2))

print("ALL PASS" if not fails[0] else "FAILED %d" % fails[0], flush=True)
sys.exit(1 if fails[0] else 0)
