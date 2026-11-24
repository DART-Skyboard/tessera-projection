"""Headless Blender test of Fluid mode:  blender --factory-startup -b --python blender/tests/test_fluidmode_blender.py
Loads the lava lamp demo, builds, steps frames, checks objects / attributes / node groups / modes, renders one Cycles CPU frame."""
import os
import sys
import tempfile
import time

import bpy
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import arc_flow
from arc_flow import fluidmode_ops as FO
from arc_flow import fluidmode_panel as FP
arc_flow.register()
fails = [0]


def ok(c, msg):
    print(("PASS " if c else "FAIL ") + msg, flush=True)
    if not c:
        fails[0] += 1


print("Blender", bpy.app.version_string, flush=True)
sc = bpy.context.scene
fm = sc.arc_flow_fm
ok(all(hasattr(bpy.types, n) for n in ("AF_PT_fm_mode", "AF_PT_fm_main", "AF_PT_fm_fluids", "AF_PT_fm_solver", "AF_PT_fm_object", "AF_UL_fm_fluids", "AF_PT_tabs")), "Fluid mode panels registered")

# vertical tabs host category panels inside the tab card (they no longer stack / hide by fm.mode)
from arc_flow import ui_tabs as UT
ok(hasattr(sc.arc_flow, "ui_tab") and hasattr(bpy.types, "AF_PT_tabs"), "vertical tab rail registered")
ok(not bpy.types.AF_PT_main.poll(bpy.context) and not FP.AF_PT_fm_main.poll(bpy.context),
   "hosted combustion and fluid panels do not stack in the N-panel")
fm.mode = 'COMBUSTION'
sc.arc_flow.ui_tab = 'FLUID'
ok(fm.mode == 'COMBUSTION' and sc.arc_flow.ui_tab == 'FLUID',
   "opening the Fluid tab does not change fm.mode (categories stay active together)")
fm.mode = 'FLUID'
sc.arc_flow.ui_tab = 'COMBUSTION'
ok(fm.mode == 'FLUID', "opening the Combustion tab does not change fm.mode")

# demo
t0 = time.time()
FO.load_lava_lamp(sc, max_particles=15000)
ok(len(fm.fluids) == 4 and bpy.data.objects.get("TS_LampBulb") is not None and bpy.data.objects["TS_LampBulb"].arc_flow_fm.heat_on, "lava lamp demo: container, heat source, carrier + 3 wax fluids")
sc.frame_start, sc.frame_end = 1, 12
FO.start(sc, play=False)
S = FO.RT.solver
ok(S is not None and S.n > 300, "build self-initialises the fluids (%d particles, spacing %.1f mm, %.1fs)" % (S.n, S.dx * 1000, time.time() - t0))
ok(all(fl.d_particles > 0 for fl in fm.fluids) and min(fl.d_particles for fl in fm.fluids) > 100, "every fluid got particles: %s" % [fl.d_particles for fl in fm.fluids])
rep = FO.RT.fill
ok(rep is not None and rep['steps'] >= 1 and S.outside_count() == 0, "Fill / Settle phase ran (%d steps), all particles inside the cone: %s" % (rep['steps'], fm.status[:90]))
bpy.ops.arcflow.fm_fill()
ok(FO.RT.fill is not None and FO.RT.last is None, "Fill / Settle operator rebuilds and settles the start frame")
FO.start(sc, play=False)
S = FO.RT.solver
for f in range(1, 9):
    sc.frame_set(f)
tm = [fl.points_obj for fl in fm.fluids]
ok(all(o is not None for o in tm) and all(len(o.data.vertices) == fl.d_particles for o, fl in zip(tm, fm.fluids)), "particle objects created with one vertex per particle")
wax = fm.fluids[1].points_obj.data
ok(all(a in wax.attributes for a in ("af_radius", "fm_velocity", "fm_temperature", "fm_density", "fm_fluid_id")), "particle attributes: velocity, temperature, density, fluid id, radius")
T = np.empty(len(wax.vertices), np.float32)
wax.attributes['fm_temperature'].data.foreach_get('value', T)
T0 = float(FO.RT.start_snap['T'][S.lo[1]:S.hi[1]].mean())
ok(T.mean() > T0 + 0.3, "heat source warmed the wax (%.2f K mean, %.2f K after the fill)" % (T.mean(), T0))
ok(FO.RT.last == 8 and len(FO.RT.cache) >= 8, "frames stepped and cached")
# Geometry Nodes exposure
mod = fm.fluids[1].points_obj.modifiers.get("Tessera Fluid")
ng = bpy.data.node_groups.get("Tessera Fluid_Wax_red") or (mod.node_group if mod else None)
names = [i.name for i in mod.node_group.interface.items_tree] if mod and mod.node_group else []
need = ["Mass", "Volume", "Density", "Temperature", "Velocity", "Viscosity", "Colour", "No Container", "Container", "Geometry"]
ok(all(n in names for n in need), "per-fluid node group exposes %s" % need)
ok("Tessera Mesh Per Particle" in bpy.data.node_groups and len(bpy.data.node_groups["Tessera Mesh Per Particle"].nodes) > 5, "mesh-per-particle node group exists")
ok(fm.fluids[1].points_obj.modifiers.get("AF Render") is not None, "render group modifier present")
# scrub back (cache) and forward again
sc.frame_set(3)
x3 = np.empty(len(wax.vertices) * 3, np.float32); wax.vertices.foreach_get('co', x3)
ok(np.allclose(x3, np.array([c for c in FO.RT.cache[3]['x'][S.lo[1]:S.hi[1]]]).ravel(), atol=1e-5), "scrubbing back shows the cached frame")
FO.stop(sc)

# assign selected geometry as a fluid + mesh-per-particle instancing
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1, location=(1.0, 0, 0.5))
src = bpy.context.object
for o in bpy.context.scene.objects:
    o.select_set(False)
src.select_set(True)
bpy.context.view_layer.objects.active = src
n0 = len(fm.fluids)
bpy.ops.arcflow.fm_assign(preset='HELIUM')
ok(len(fm.fluids) == n0 + 1 and fm.fluids[-1].shape == src, "Assign Selected as Fluid adds a fluid for the object")
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(5, 5, 5))
cube = bpy.context.object
fl = fm.fluids[-1]
# NOTE: setting an Object input on any Geometry Nodes modifier hangs the depsgraph in this sandbox's headless Blender 5.2.3 / 5.3.0
# (reproduced with a trivial node group and no add-on code), so the custom-mesh instance path is not exercised here.
fl.count = 200
FO.mark_dirty()
sc.frame_set(1)
FO.start(sc, play=False)
for f in range(1, 4):
    sc.frame_set(f)
ok(fl.points_obj is not None and any(m.type == 'NODES' for m in fl.points_obj.modifiers), "mesh-per-particle fluid set up")
FO.stop(sc)

# colour-mixing demo: composition colours reach the render attribute, the surface path builds a mesh
FO.stop(sc)
FO.load_colour_mixing(sc, max_particles=1500)
FO.start(sc, play=False)
for f in range(1, 6):
    sc.frame_set(f)
S = FO.RT.solver
ok(S.mixing and S.n > 500, "colour-mixing demo builds with mixing on (%d particles)" % S.n)
fm = sc.arc_flow_fm
fl0 = fm.fluids[0]
fl0.render_mode = 'SURFACE'
FO.refresh_render(sc)
sc.frame_set(6)
sobj = [o for o in bpy.data.objects if o.name.startswith('AFFMS_')]
ok(len(sobj) > 0 and len(sobj[0].data.vertices) > 0, "SURFACE render mode builds a surface mesh (%d verts)" % (len(sobj[0].data.vertices) if sobj else 0))
FO.stop(sc)

# two-way coupling through the Blender layer: a collider cube with a mass falls under gravity and its object follows the body
FO.stop(sc)
FO.load_colour_mixing(sc, max_particles=1200)
cube = bpy.data.objects.new("TS_Block", bpy.data.meshes.new("TS_Block"))
import bmesh
bm = bmesh.new(); bmesh.ops.create_cube(bm, size=0.03); bm.to_mesh(cube.data); bm.free()
sc.collection.objects.link(cube)
cube.location = (0.0, 0.0, 0.3)
cube.arc_flow_fm.collider = True
ok(cube.arc_flow_fm.two_way and cube.arc_flow_fm.mass == 0.0, "two-way toggle defaults on, mass 0 (fixed unless given a mass)")
cube.arc_flow_fm.mass = 0.2
FO.mark_dirty()
FO.start(sc, play=False)
z0 = cube.matrix_world.translation.z
for f in range(1, 8):
    sc.frame_set(f)
ok(any(b.dynamic for b in FO.RT.solver.bodies) and cube.matrix_world.translation.z < z0 - 0.01, "a dynamic body moves its Blender object (z %.3f -> %.3f)" % (z0, cube.matrix_world.translation.z))
FO.stop(sc)
bpy.data.objects.remove(cube, do_unlink=True)

# Cycles CPU render (also valid for EEVEE: same materials, instanced points / volume)
cam = sc.camera
sc.render.engine = 'CYCLES'
sc.cycles.device = 'CPU'
sc.cycles.samples = 4
sc.render.resolution_x, sc.render.resolution_y = 160, 120
sc.render.filepath = os.path.join(tempfile.mkdtemp(), "fm.png")
sc.frame_set(5)
t0 = time.time()
bpy.ops.render.render(write_still=True)
ok(os.path.exists(sc.render.filepath) and os.path.getsize(sc.render.filepath) > 500, "Cycles CPU render of the lava lamp (%.1fs)" % (time.time() - t0))
fm.fluids[0].render_mode = 'RAYMARCH'
FO.refresh_render(sc)
sc.render.filepath = os.path.join(tempfile.mkdtemp(), "fm2.png")
bpy.ops.render.render(write_still=True)
ok(os.path.exists(sc.render.filepath) and os.path.getsize(sc.render.filepath) > 500, "Cycles CPU render, ray-march path")
print("FAILURES:", fails[0])
sys.exit(1 if fails[0] else 0)
