"""Headless checks that every Tessera Projection asset is a self-colliding
collision body by default, and that the storefront dress drapes without
self-intersections.

    blender --factory-startup -b --python-exit-code 1 --python tests/test_self_collision_bpy.py
"""
import os
import sys
import time

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, ROOT)
import arc_flow
arc_flow.register()
from arc_flow import cloth_ops as CO
from arc_flow import asset_colliders as AC
from arc_flow.core import cloth_solver as CS
from arc_flow.core import hair_solver as HS

fails = [0]


def ok(c, msg):
    print(("PASS " if c else "FAIL ") + msg, flush=True)
    if not c:
        fails[0] += 1


def test_assign_defaults():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, 1))
    dress = bpy.context.object
    dress.name = "DressAsset"
    sc = bpy.context.scene
    g = CO.assign_garment(sc, dress, preset='SILK', quality='REALTIME')
    ok(g.self_collision is True, "assign_garment leaves Self-collision on (even Real-time / Fast quality)")
    ok(dress.arc_flow_cloth.is_garment and dress.arc_flow_cloth.is_collider,
       "assigned garment is also a cloth collision body")
    ok(dress.arc_flow_fm.collider and dress.arc_flow_wt.body,
       "assigned garment is a Fluid/Combustion and wind-tunnel collision body")
    ok(AC.is_projection_collider(dress), "registry reports the garment as a projection collider")
    bodies = list(AC.iter_collision_meshes(sc))
    ok(dress in bodies, "iter_collision_meshes yields the assigned garment")
    # second mesh as an explicit collider is visible to the garment soup
    bpy.ops.mesh.primitive_cube_add(size=0.4, location=(0, 0, 0))
    form = bpy.context.object
    CO.add_collider(sc, form)
    cv, ct, names = CO._collider_mesh(sc, bpy.context.evaluated_depsgraph_get(), exclude=dress)
    ok(form.name in names and dress.name not in names,
       "garment soup includes other bodies and excludes itself (%s)" % names)


def test_hair_default_and_strands():
    hp = bpy.context.scene.arc_flow_hair
    ok(hp.self_collision is True, "Hair Self-collision property defaults to on")
    a = np.stack([np.zeros(6), np.zeros(6), np.linspace(0, 0.2, 6)], 1)
    b = a.copy(); b[:, 0] = 0.002
    P, off = HS.pack_strands([a, b])
    S = HS.HairSolver(P, off, HS.HairParams(self_collision=True, collision_radius=0.008,
                                            gravity=(0, 0, 0), rest_shape=0.0, damping=1.0))
    S.step()
    sep = float(np.mean(np.linalg.norm(S.positions[off[1]:off[2]] - S.positions[off[0]:off[1]], axis=1)))
    ok(sep >= 0.014, "hair solver strand-strand separation after a step is %.4f m" % sep)


def test_dress_drape_self_intersections():
    """Drape the showcase storefront dress and count self-contacts before vs after."""
    sys.path.insert(0, os.path.join(ROOT, 'docs', 'showcase'))
    import cloth_storefront_scene as store
    V, F, pins, n, rows = store.dress_mesh()
    # a thin thickness so the godet folds are allowed to sit close but not pass through
    S_off = CS.ClothSolver(V, np.array(F, np.int64), pins=pins, preset='SILK', quality='BALANCED',
                           self_collision=False, thickness=0.004, gravity=(0, 0, -9.81),
                           use_dihedral=True)
    before = S_off.count_self_intersections()
    # push the hem inward so the skirt wraps through itself, then drape
    hem = V[:, 2] < (V[:, 2].min() + 0.25)
    S_off.x[hem, 0] *= 0.15
    S_off.x[hem, 1] *= 0.15
    tangled = S_off.count_self_intersections()
    for _ in range(30):
        S_off.step(1.0 / 24.0)
    after_off = S_off.count_self_intersections()

    S_on = CS.ClothSolver(V, np.array(F, np.int64), pins=pins, preset='SILK', quality='BALANCED',
                          self_collision=True, thickness=0.004, gravity=(0, 0, -9.81),
                          use_dihedral=True)
    S_on.x[hem, 0] *= 0.15
    S_on.x[hem, 1] *= 0.15
    tangled_on = S_on.count_self_intersections()
    t0 = time.perf_counter()
    for _ in range(30):
        S_on.step(1.0 / 24.0)
    dt = time.perf_counter() - t0
    after_on = S_on.count_self_intersections()
    print("DRESS rest %d  tangled %d  drape without self-col %d  tangled-on %d  drape with self-col %d  (%.2fs)"
          % (before, tangled, after_off, tangled_on, after_on, dt), flush=True)
    ok(tangled_on > 0, "crushed dress hem starts with self-contacts (%d)" % tangled_on)
    ok(after_on <= max(8, tangled_on // 15),
       "dress self-intersections drop to near zero with self-collision (before %d, after %d)" % (tangled_on, after_on))
    ok(after_on < after_off,
       "self-collision leaves fewer contacts than the same drape without it (%d < %d)" % (after_on, after_off))
    test_dress_drape_self_intersections.counts = dict(
        rest=before, tangled=tangled_on, after=after_on, after_off=after_off)


if __name__ == '__main__':
    test_assign_defaults()
    test_hair_default_and_strands()
    test_dress_drape_self_intersections()
    print("ALL PASS" if fails[0] == 0 else "FAILED %d" % fails[0], flush=True)
    sys.exit(1 if fails[0] else 0)
