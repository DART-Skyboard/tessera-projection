"""Headless Blender test: garment GROUPS (several mesh objects in one collection solved as one unit).

    blender --factory-startup -b --python tests/test_cloth_group_bpy.py
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
from arc_flow.cloth_panel import draw_cloth_tab

fails = [0]


def ok(c, msg):
    print(("PASS " if c else "FAIL ") + msg, flush=True)
    if not c:
        fails[0] += 1


sc = bpy.context.scene
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)


def sheet(name, n, w, z, loc=(0, 0), coll=None):
    me = bpy.data.meshes.new(name)
    xs = np.linspace(0, w, n)
    v = [(loc[0] + x, loc[1] + y, z) for y in xs for x in xs]
    f = [(j * n + i, j * n + i + 1, (j + 1) * n + i + 1, (j + 1) * n + i) for j in range(n - 1) for i in range(n - 1)]
    me.from_pydata(v, [], f)
    me.update()
    o = bpy.data.objects.new(name, me)
    (coll or sc.collection).objects.link(o)
    return o


coll = bpy.data.collections.new("Jacket")
sc.collection.children.link(coll)
body = sheet("Body", 14, 0.5, 0.0, coll=coll)
pocket = sheet("Pocket", 5, 0.12, 0.0, (0.2, 0.2), coll=coll)     # lies ON the body; its border touches the body surface
collar = sheet("Collar", 8, 0.3, 0.0, (0.5, 0.0), coll=coll)       # shares the body's right border
vg = body.vertex_groups.new(name="ClothPins")
vg.add([j * 14 + i for j in range(14) for i in range(14) if j == 13], 1.0, 'REPLACE')   # top row pinned
loose = sheet("Loose", 4, 0.1, 2.0)                                # not in the collection

# ---- automatic detection from the selection
bpy.ops.object.select_all(action='DESELECT')
for o in (body, pocket, collar):
    o.select_set(True)
bpy.context.view_layer.objects.active = body
r = bpy.ops.arcflow.cloth_assign()
cl = sc.arc_flow_cloth
ok(r == {'FINISHED'} and len(cl.garments) == 1, "Assign selected turns meshes of one collection into ONE garment group")
g = cl.garments[0]
ok(g.collection == coll and len(g.members) == 3, "group = the collection, 3 members (%s)" % [m.name for m in g.members])
ok(all(o.arc_flow_cloth.is_garment for o in (body, pocket, collar)), "members are flagged as garment objects")
S = CO.RT.solvers[g.name]
ok(S.n == 14 * 14 + 25 + 64, "solver merged all member vertices (%d)" % S.n)
ok(S.members is not None and len(np.unique(S.members)) == 3, "one XPBD system with 3 member ids")
ok(g.n_stitches > 0, "touching borders were auto-detected and stitched (%d welds)" % g.n_stitches)
g.members[0].preset = 'WOOL'
g.members[1].preset = 'SILK'
g.members[2].preset = 'LEATHER'
S = CO.build_solver(sc, g)
ok(S.density_v[0] > S.density_v[14 * 14] and S.density_v[-1] > S.density_v[14 * 14], "per-member fabric: wool/leather heavier than silk")

# ---- a new object added to the collection joins the group
extra = sheet("Patch", 4, 0.08, 0.0, (0.05, 0.05), coll=coll)
n_before = len(g.members)
CO.sync_members(sc, g)
ok(len(g.members) == n_before + 1, "object added to the collection joins the group on refresh")
bpy.data.objects.remove(extra)
CO.sync_members(sc, g)
ok(len(g.members) == n_before, "object removed from the collection leaves the group")
S = CO.build_solver(sc, g)

# ---- stepping writes the result back per object, stitches hold
w0 = np.array([v.co[:] for v in pocket.data.vertices])
for _ in range(40):
    CO.step_scene(sc, 1.0 / 60.0)
w1 = np.array([v.co[:] for v in pocket.data.vertices])
ok(np.abs(w1 - w0).max() > 1e-4, "pocket object received its solved positions")
st = S.stitch
q = (st['bary'][:, :, None] * S.x[st['tri']]).sum(1)
ok(float(np.linalg.norm(S.x[st['a']] - q, axis=1).max()) < 0.03, "stitches hold after 40 steps")

# ---- self-collision toggle applies group-wide
g.self_collision = False
S = CO.build_solver(sc, g)
ok(S.self_collision is False, "Self-collision OFF reaches the group solver")
g.self_collision = True
S = CO.build_solver(sc, g)
ok(S.self_collision is True, "Self-collision ON reaches the group solver")

# ---- the whole group collides with an external collider
box = sheet("Table", 12, 1.2, -0.3, (-0.3, -0.3))
CO.add_collider(sc, box)
for k in range(60):
    CO.step_scene(sc, 1.0 / 60.0)
zmin = min(float(np.array([v.co.z for v in o.data.vertices]).min()) for o in (body, pocket, collar))
ok(zmin > -0.3 - 1e-3, "group members stay above the external collider (min z %.3f)" % zmin)

# ---- panel draw with a group, and single-object path untouched
class _L:
    def __getattr__(self, n):
        return lambda *a, **k: _L()
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
try:
    draw_cloth_tab(_L(), bpy.context)
    ok(True, "panel draws with a group")
except Exception as e:
    ok(False, "panel draw failed: %r" % e)

CO.RT.reset()
for gg in list(cl.garments):
    cl.garments.remove(0)
bpy.ops.object.select_all(action='DESELECT')
loose.select_set(True)
bpy.context.view_layer.objects.active = loose
bpy.ops.arcflow.cloth_assign()
g1 = cl.garments[0]
ok(len(g1.members) == 0 and CO.RT.solvers[g1.name].members is None, "single-object garment still takes the plain path")

print("ALL PASS" if not fails[0] else "%d FAILED" % fails[0])
sys.exit(1 if fails[0] else 0)
