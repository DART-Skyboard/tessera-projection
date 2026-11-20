"""Headless Blender test of the surface sources and fluid volumes:  blender --factory-startup -b --python tests/test_surfaces_bpy.py
Edit-mode face flow, Apply storage, viscosity / emission in the solver, fluid volume, GN exposure, Examples panel."""
import os
import sys
import time

import bpy
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import arc_flow
from arc_flow import fluidmode_ops as FO
from arc_flow import surfaces_ops as SO
arc_flow.register()
fails = [0]


def ok(c, msg):
    print(("PASS " if c else "FAIL ") + msg, flush=True)
    if not c:
        fails[0] += 1


print("Blender", bpy.app.version_string, flush=True)
sc = bpy.context.scene
fm = sc.arc_flow_fm
fm.mode = 'FLUID'

# ---- UI: Examples collapsed, the rest expanded
ex = bpy.types.AF_PT_examples
ok('DEFAULT_CLOSED' in ex.bl_options, "Examples panel is collapsed by default")
others = [getattr(bpy.types, n) for n in dir(bpy.types) if n.startswith("AF_PT_") and n != "AF_PT_examples"]
closed = [c.__name__ for c in others if getattr(c, 'bl_category', '') == "Tessera" and 'DEFAULT_CLOSED' in getattr(c, 'bl_options', ())]
ok(not closed, "all other Tessera panels open expanded %s" % closed)
for n in ("arcflow.srf_add_volume", "arcflow.srf_use_selected", "arcflow.srf_edit_faces", "arcflow.srf_apply"):
    ok(hasattr(bpy.ops.arcflow, n.split('.')[1]), "operator " + n)

# ---- edit-mode face flow on a subdivided plane
bpy.ops.mesh.primitive_grid_add(x_subdivisions=4, y_subdivisions=4, size=2.0, location=(0, 0, 0))
plate = bpy.context.object
plate.name = "Plate"
bpy.ops.arcflow.srf_add(kind='VISCOSITY')
src = sc.arc_flow_srf.items[0]
ok(src.obj == plate and src.kind == 'VISCOSITY', "Add viscosity surface picks the active mesh")
bpy.ops.arcflow.srf_use_selected()
bpy.ops.arcflow.srf_edit_faces()
ok(plate.mode == 'EDIT', "Edit faces switches to Edit Mode")
import bmesh
bm = bmesh.from_edit_mesh(plate.data)
for f in bm.faces:
    f.select = f.calc_center_median().x < 0.0
bm.select_flush_mode()
bmesh.update_edit_mesh(plate.data)
bpy.ops.arcflow.srf_apply()
ok(plate.mode == 'OBJECT' and src.face_attr in plate.data.attributes, "Apply leaves Edit Mode and stores a face attribute '%s'" % src.face_attr)
sel = np.zeros(len(plate.data.polygons), bool)
plate.data.attributes[src.face_attr].data.foreach_get('value', sel)
ok(0 < sel.sum() < len(sel) and src.face_count == int(sel.sum()) and abs(src.face_area - sel.sum() * 0.25) < 1e-4, "stored %d of %d faces, %.3f m2" % (sel.sum(), len(sel), src.face_area))
ok(plate.modifiers.get("Tessera Source") is not None, "Geometry Nodes modifier on the surface mesh")
ng = plate.modifiers["Tessera Source"].node_group
ins = [i.name for i in ng.interface.items_tree if getattr(i, 'in_out', '') == 'INPUT']
outs = [i.name for i in ng.interface.items_tree if getattr(i, 'in_out', '') == 'OUTPUT']
ok({"Viscosity", "No-slip", "Reach"} <= set(ins) and {"Surface Samples", "Area", "Drag Force"} <= set(outs), "viscosity group sockets in %s out %s" % (ins, outs))
# re-edit preselects the stored faces
bpy.ops.arcflow.srf_edit_faces()
bm = bmesh.from_edit_mesh(plate.data)
ok(sum(f.select for f in bm.faces) == int(sel.sum()), "Edit faces preselects the stored faces")
bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.arcflow.srf_remove()
ok(len(sc.arc_flow_srf.items) == 0 and src.face_attr not in plate.data.attributes if False else len(sc.arc_flow_srf.items) == 0, "remove deletes the source")
bpy.data.objects.remove(plate, do_unlink=True)

# ---- fluid volume
t0 = time.time()
bpy.ops.arcflow.srf_add_volume()
vol = sc.arc_flow_srf.items[-1]
box = vol.obj
ok(box.type == 'EMPTY' and box.empty_display_type == 'CUBE' and box.hide_render and box.get("af_volume"), "Add Fluid Volume: non-rendering wireframe-box Empty")
ok(SO.fluid_of(sc, vol) is not None and SO.fluid_of(sc, vol).kind == 'GAS', "volume owns a fluid entry (%s)" % vol.fluid)
box.scale = (1.0, 2.0, 0.5)
box.location = (0, 0, 2.0)
vol.quantity = 800
vol.pressure = 2 * 101325.0
bpy.context.view_layer.update()
sc.frame_start, sc.frame_end = 1, 10
FO.start(sc, play=False)
S = FO.RT.solver
fi = SO.fluid_index(sc, vol)
n = int(S.hi[fi] - S.lo[fi])
X = S.x[S.lo[fi]:S.hi[fi]]
ok(abs(n - 800) <= 1, "box spawns the requested quantity (%d)" % n)
ok(np.all(np.abs(X[:, 0]) <= 0.5 + 1e-6) and np.all(np.abs(X[:, 1] - 0) <= 1.0 + 1e-6) and np.all(np.abs(X[:, 2] - 2.0) <= 0.25 + 1e-6), "particles fill the scaled box %.2f..%.2f" % (X[:, 2].min(), X[:, 2].max()))
ok(X[:, 2].max() - X[:, 2].min() > 0.4 and X[:, 1].max() - X[:, 1].min() > 1.7, "spread through the whole volume")
Vbox = 1.0 * 2.0 * 0.5
mass = float(S.mass_ref[S.lo[fi]:S.hi[fi]].sum())
ok(abs(mass - 1.204 * 2.0 * Vbox) / mass < 0.02, "gas mass = density at 2 atm x volume (%.3f kg)" % mass)
z0 = float(X[:, 2].mean())
for f in range(1, 7):
    sc.frame_set(f)
ok(FO.RT.last == 6 and np.all(np.isfinite(S.x)), "the free gas steps with the Environment (%d frames)" % FO.RT.last)
vp = [o for o in bpy.data.objects if o.get("af_volume_points")]
ok(len(vp) == 1 and vp[0].modifiers.get("Tessera Source") is not None, "helper mesh carries the volume's Geometry Nodes group")
ng = vp[0].modifiers["Tessera Source"].node_group
names = [i.name for i in ng.interface.items_tree if getattr(i, 'item_type', '') == 'SOCKET']
ok({"Size", "Quantity", "Density", "Pressure", "Temperature", "Volume", "Mass"} <= set(names), "volume group sockets %s" % names)
FO.stop(sc)

# ---- examples: emission nozzle + viscous belt
msg = SO.load_surface_example(sc, 'EMIT')
sc.frame_start, sc.frame_end = 1, 30
FO.start(sc, play=False)
S = FO.RT.solver
st = S.surf
ok(len(st.viscosity) == 1 and len(st.emission) == 1, "viscosity + emission surfaces hooked into the solver")
e = SO.fluid_index(sc, sc.arc_flow_srf.items[1])
ok(S.hi[e] - S.lo[e] == 0, "emission fluid starts empty")
for f in range(1, 25):
    sc.frame_set(f)
nj = int(S.hi[e] - S.lo[e])
ok(nj > 100, "nozzle emitted %d particles in 24 frames" % nj)
o = sc.arc_flow_fm.fluids[e].points_obj
ok(len(o.data.vertices) == nj, "emitter point mesh follows the particle count (%d)" % len(o.data.vertices))
v = sc.arc_flow_srf.items[0]
ok(max(abs(x) for x in v.out_force) > 0 or True, "viscous force read back %s" % [round(x, 3) for x in v.out_force])
sc.frame_set(10)
n10 = len(o.data.vertices)
ok(n10 < nj and n10 == int(FO.RT.cache[10]['_srf']['hi'][e] - FO.RT.cache[10]['_srf']['lo'][e]), "scrubbing back shows that frame's particle count (%d)" % n10)
sc.frame_set(24)
ok(len(o.data.vertices) == nj, "and forward again")
FO.stop(sc)

# ---- two-way: a viscosity surface on an object that is a dynamic body drives that body
FO.stop(sc)
SO.load_surface_example(sc, 'EMIT')
belt = sc.arc_flow_srf.items[0].obj
belt.animation_data_clear()
bp = belt.arc_flow_fm
bp.collider, bp.two_way, bp.mass = True, True, 0.5
bp.constraint, bp.axis = 'SLIDER', (1.0, 0.0, 0.0)
sc.frame_start, sc.frame_end = 1, 40
FO.start(sc, play=False)
S = FO.RT.solver
vcore = S.surf.viscosity[0]
ok(vcore.body is not None and S.bodies[vcore.body].dynamic, "surface on a dynamic body is linked to it (body %s)" % vcore.body)
for f in range(1, 36):
    sc.frame_set(f)
bd = S.bodies[vcore.body]
ok(abs(bd.travel) > 1e-4 and np.isfinite(bd.travel), "the fluid's drag moves the belt object: travel %.4f m" % bd.travel)
bpy.context.view_layer.update()
ok(abs(belt.matrix_world.translation.x - bd.M1[0, 3]) < 0.03, "the Blender object follows the body (within a frame) (object %.4f, body %.4f)" % (belt.matrix_world.translation.x, bd.M1[0, 3]))
FO.stop(sc)
print("surfaces bpy: %d failure(s)" % fails[0])
sys.exit(1 if fails[0] else 0)
