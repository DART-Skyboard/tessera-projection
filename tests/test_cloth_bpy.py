"""Headless Blender test of the Tessera cloth solver:

    /home/box/apps/blender-4.2.3-linux-x64/blender --factory-startup -b --python tests/test_cloth_bpy.py

Creates a mannequin collider and a shoulder-pinned cape, runs live steps under
rig motion and wind, checks a non-penetrating drape and live mesh updates, bakes
a game-engine package, and writes a quick verification preview plus a
publication-grade Cycles beauty still.
"""
import json
import os
import shutil
import sys
import tempfile
import time

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, ROOT)
import arc_flow
arc_flow.register()
from arc_flow import cloth_ops as CO
from arc_flow import cloth_panel
from arc_flow.cloth_panel import draw_cloth_tab

fails = [0]


def ok(c, msg):
    print(("PASS " if c else "FAIL ") + msg, flush=True)
    if not c:
        fails[0] += 1


# keep Cycles / temp files off the tight /workspace disk
TMP = tempfile.mkdtemp(prefix="tessera_cloth_", dir="/tmp")
os.environ['TMPDIR'] = TMP
print("Blender", bpy.app.version_string, "tmpdir", TMP, flush=True)

sc = bpy.context.scene
cl = sc.arc_flow_cloth
ok(cl is not None and hasattr(bpy.types, 'AF_PT_cloth'), "cloth properties and panel registered")
ok(callable(draw_cloth_tab), "draw_cloth_tab is importable for the vertical-tab UI")

# ---- example scene
t0 = time.time()
msg = CO.load_cape_example(sc)
print(msg, flush=True)
ok("cape" in msg.lower(), "example returns a status message")
cape = bpy.data.objects.get("Cloth Cape")
torso = bpy.data.objects.get("Cloth Mannequin Torso")
head = bpy.data.objects.get("Cloth Mannequin Head")
neck = bpy.data.objects.get("Cloth Mannequin Neck")
arm = bpy.data.objects.get("Cloth Rig")
ok(all(o is not None for o in (cape, torso, head, neck, arm)), "cape, mannequin parts and rig exist")
ok(cape.arc_flow_cloth.is_garment and torso.arc_flow_cloth.is_collider, "garment / collider flags")
ok(len(cl.garments) == 1 and len(cl.colliders) == 5, "one garment, five collision bodies (torso/head/neck/arms)")
g = cl.garments[0]
ok(g.obj == cape and g.preset == 'CAPE' and g.armature == arm, "garment wired to the cape and the rig")
vg = cape.vertex_groups.get("ClothPins")
ok(vg is not None and g.n_pins >= 4, "shoulder pin group has %d pins" % g.n_pins)
S = CO.RT.solvers[g.name]
ok(S.n == len(cape.data.vertices) and S.collider is not None, "solver built with a collider soup")
rest = np.empty(len(cape.data.vertices) * 3, np.float64)
cape.data.vertices.foreach_get('co', rest)
rest = rest.reshape(-1, 3).copy()

# ---- live steps under wind
cl.use_scene_wind = False
cl.wind = (0.0, 2.2, 0.15)
cl.live_fps = 60.0
t1 = time.time()
for _ in range(48):
    CO.step_scene(sc, 1 / 60.0)
print("48 live steps %.2fs, last %.1f ms, peak stretch %.2f %%, penetration %.3f mm"
      % (time.time() - t1, cl.last_ms, 100 * g.last_stretch, 1000 * g.last_penetration), flush=True)
now = np.empty(len(cape.data.vertices) * 3, np.float64)
cape.data.vertices.foreach_get('co', now)
now = now.reshape(-1, 3)
delta = float(np.linalg.norm(now - rest, axis=1).mean())
ok(delta > 0.01, "live steps update the garment mesh (mean vertex travel %.3f m)" % delta)
ok(np.all(np.isfinite(now)), "mesh stays finite")
ok(g.last_penetration <= 2.0e-3, "non-penetrating drape (worst %.3f mm past the margin)" % (1000 * g.last_penetration))
# the cape should have moved downwind (+Y in the example) and dropped under gravity
hem = rest[:, 2] <= (float(rest[:, 2].min()) + 0.18)
ok(float(now[hem, 1].max()) > float(rest[hem, 1].max()) - 0.02,
   "hem responds to +Y wind (ymax %.3f -> %.3f)" % (rest[hem, 1].max(), now[hem, 1].max()))
ok(float(now[:, 2].max() - now[:, 2].min()) > 0.55, "cape keeps a hanging span after wind (span %.3f m, zmin %.3f -> %.3f)" % (now[:, 2].max() - now[:, 2].min(), rest[:, 2].min(), now[:, 2].min()))

# ---- moving rig inertia: yank the armature in +X and keep stepping
x_before = now.copy()
arm.location = (0.18, 0.0, 0.0)
arm.rotation_euler = (0.0, 0.0, 0.15)
# depsgraph must see the new pose / parented collider positions
bpy.context.view_layer.update()
for _ in range(18):
    CO.step_scene(sc, 1 / 60.0)
after = np.empty(len(cape.data.vertices) * 3, np.float64)
cape.data.vertices.foreach_get('co', after)
after = after.reshape(-1, 3)
ok(float(after[:, 0].mean()) > float(x_before[:, 0].mean()) + 0.02,
   "cape follows a +X rig move (x %.3f -> %.3f)" % (x_before[:, 0].mean(), after[:, 0].mean()))
ok(g.last_penetration <= 3.0e-3, "still non-penetrating after the rig move (%.3f mm)" % (1000 * g.last_penetration))

# reset the rig and park the cloth in a draped pose for the beauty still
arm.location = (0.0, 0.0, 0.0)
arm.rotation_euler = (0.0, 0.0, 0.0)
bpy.context.view_layer.update()
CO.reset_scene(sc)
CO.build_solver(sc, g)
CO.drape_for_beauty(sc, gust_frames=50, settle_frames=40)
print("settled for render: stretch %.2f %%, penetration %.3f mm, last %.1f ms"
      % (100 * g.last_stretch, 1000 * g.last_penetration, cl.last_ms), flush=True)

# ---- bake / export
cl.bake_path = TMP
cl.bake_name = "cape_test"
cl.use_scene_range = False
cl.frame_start, cl.frame_end = 1, 8
# NPZ cache
res = bpy.ops.arcflow.cloth_bake(mode='NPZ')
ok('FINISHED' in res, "npz bake operator finishes")
npz_files = [f for f in os.listdir(TMP) if f.endswith('.npz') and 'cape' in f.lower()]
ok(len(npz_files) >= 1, "npz cache written (%s)" % npz_files)
if npz_files:
    z = np.load(os.path.join(TMP, npz_files[0]), allow_pickle=False)
    ok('positions' in z.files and 'rest' in z.files and z['positions'].ndim == 3,
       "npz has frames x verts x 3 (shape %s)" % (z['positions'].shape,))
# game-engine package
res = bpy.ops.arcflow.cloth_export_package()
ok('FINISHED' in res, "package export finishes")
js = [os.path.join(TMP, f) for f in os.listdir(TMP) if f.endswith('.json')]
ok(len(js) >= 1, "package JSON written")
if js:
    pkg = json.load(open(js[0]))
    ok(pkg.get('format') == 'tessera-cloth-package' and 'unreal_chaos' in pkg.get('engine_hints', {}),
       "package has Chaos / engine hints")
    ok(len(pkg.get('pins', {}).get('indices', [])) >= 4, "package lists pin indices")
    ok(len(pkg.get('colliders', [])) == 5, "package lists the five body colliders")

# ---- quick verification preview (Workbench, cheap)
preview = os.path.join(HERE, 'out', 'cloth_cape_preview.png')
os.makedirs(os.path.dirname(preview), exist_ok=True)
CO.setup_studio(sc)
sc.render.engine = 'BLENDER_WORKBENCH'
sc.display.shading.light = 'STUDIO'
sc.display.shading.color_type = 'MATERIAL'
sc.render.resolution_x, sc.render.resolution_y = 960, 540
sc.render.resolution_percentage = 100
sc.render.image_settings.file_format = 'PNG'
sc.render.filepath = preview
t2 = time.time()
bpy.ops.render.render(write_still=True)
ok(os.path.isfile(preview) and os.path.getsize(preview) > 8_000,
   "verification preview written (%d KB, %.1fs)" % (os.path.getsize(preview) // 1024, time.time() - t2))

# ---- publication-grade Cycles beauty
beauty = os.path.join(ROOT, 'docs', 'renders', 'cloth_cape_beauty.png')
os.makedirs(os.path.dirname(beauty), exist_ok=True)
t3 = time.time()
CO.render_beauty(sc, beauty, samples=64, resolution=(1920, 1080), denoise=True)
el = time.time() - t3
ok(os.path.isfile(beauty) and os.path.getsize(beauty) > 80_000,
   "Cycles beauty still written (%d KB, 64 spp, %.1fs) -> %s" % (os.path.getsize(beauty) // 1024, el, beauty))

print("example+sim setup %.1fs" % (t1 - t0), flush=True)
print("ALL PASS" if not fails[0] else "FAILED %d" % fails[0], flush=True)

# tidy Cycles / tmp leftovers so we don't sit on the tight disk
try:
    shutil.rmtree(TMP, ignore_errors=True)
except Exception:
    pass
# drop any leftover blender cache under /tmp
for name in os.listdir('/tmp'):
    if name.startswith('tessera_cloth_') or name.startswith('blender_'):
        p = os.path.join('/tmp', name)
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)

sys.exit(1 if fails[0] else 0)
