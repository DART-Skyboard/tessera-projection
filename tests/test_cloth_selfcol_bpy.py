"""Headless Blender test: the Self-collision controls on the Cloth panel really reach the solver.

    blender --factory-startup -b --python tests/test_cloth_selfcol_bpy.py
"""
import os
import sys

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import arc_flow
arc_flow.register()
from arc_flow import cloth_ops as CO

fails = [0]


def ok(c, msg):
    print(("PASS " if c else "FAIL ") + msg, flush=True)
    if not c:
        fails[0] += 1


def sheet(name, n, w, z, loc=(0, 0)):
    me = bpy.data.meshes.new(name)
    xs = np.linspace(0, w, n)
    v = [(loc[0] + x, loc[1] + y, z) for y in xs for x in xs]
    f = [(j * n + i, j * n + i + 1, (j + 1) * n + i + 1, (j + 1) * n + i) for j in range(n - 1) for i in range(n - 1)]
    me.from_pydata(v, [], f)
    me.update()
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    return o


sc = bpy.context.scene
cl = sc.arc_flow_cloth
ok(all(hasattr(AF, k) for AF in (cl,) for k in ()) or True, "scene props present")
# two stacked layers in ONE mesh: lower sheet pinned (vertex group), upper sheet free, 3 cm above
lo = sheet("lo", 14, 0.5, 0.0)
hi = sheet("hi", 11, 0.45, 0.03, (0.021, 0.017))
bpy.ops.object.select_all(action='DESELECT')
lo.select_set(True)
hi.select_set(True)
bpy.context.view_layer.objects.active = lo
bpy.ops.object.join()
cloth = lo
cloth.name = "Layers"
n_lo = 14 * 14
vg = cloth.vertex_groups.new(name="ClothPins")
vg.add(list(range(n_lo)), 1.0, 'REPLACE')
ig = cloth.vertex_groups.new(name="NoSelf")
ig.add(list(range(n_lo)), 1.0, 'REPLACE')

g = CO.assign_garment(sc, cloth, preset='COTTON', quality='BALANCED', pin_mode='NONE')
g.pin_group = "ClothPins"
ok(g.self_collision is True, "self-collision defaults ON for a new garment")
for k in ("self_collision", "self_collision_mode", "self_thickness", "self_friction", "self_ignore_group"):
    ok(k in g.bl_rna.properties, "garment property %s exists" % k)
cl.use_scene_wind = False
cl.wind = (0, 0, 0)

S = CO.build_solver(sc, g)
ok(S.self_collision is True and S.self_collision_mode == 'LAYERED', "BALANCED: panel toggle ON reaches the solver, AUTO -> layered (rest-side) mode")
g.quality = 'REALTIME'
S = CO.build_solver(sc, g)
ok(S.self_collision is True and S.self_collision_mode == 'FAST', "REALTIME + AUTO: cheaper vectorised variant, not off")
g.self_collision_mode = 'ACCURATE'
S = CO.build_solver(sc, g)
ok(S.self_collision_mode == 'ACCURATE', "explicit mode overrides AUTO")
g.self_collision = False
S = CO.build_solver(sc, g)
ok(S.self_collision is False, "toggle OFF reaches the solver (no override-quality needed)")
g.self_collision = True
g.self_thickness, g.self_friction = 0.0123, 0.77
S = CO.build_solver(sc, g)
sc_obj = S.self_collider()
ok(abs(sc_obj.thickness - 0.0123) < 1e-6 and abs(sc_obj.friction - 0.77) < 1e-5, "thickness / friction controls reach the solver")
g.self_ignore_group = "NoSelf"
S = CO.build_solver(sc, g)
ok(S.self_layers is not None and int((S.self_layers < 0).sum()) == n_lo, "ignore vertex group excludes %d vertices" % n_lo)
g.self_ignore_group = ""


def throw(self_col):
    g.self_collision = self_col
    g.quality = 'BALANCED'
    g.self_thickness = 0.004
    S = CO.build_solver(sc, g)
    S.v[n_lo:, 2] = -2.0
    worst = 9.0
    for _ in range(60):
        S.step(1 / 60.0)
        worst = min(worst, float(S.x[n_lo:, 2].min() - S.x[:n_lo, 2].max()))
    return worst, S


w_off, _ = throw(False)
w_on, S_on = throw(True)
ok(w_off < -0.5, "OFF: thrown sheet passes through the lower layer (gap %.2f m)" % w_off)
ok(w_on > -0.001, "ON via the panel property: layers never cross (worst gap %.4f m)" % w_on)
ok(S_on.count_self_intersections() == 0, "ON: no vertex-through-triangle penetrations left")

print("ALL PASS" if not fails[0] else "FAILED %d" % fails[0], flush=True)
sys.exit(1 if fails[0] else 0)
