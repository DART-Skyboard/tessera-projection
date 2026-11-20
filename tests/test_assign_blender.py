"""Fluid mode assignment model: any geometry as Container / Shape, Spawn modes, Primitive presets, multi-asset interaction.
    blender --factory-startup -b --python tests/test_assign_blender.py"""
import math
import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import arc_flow
from arc_flow import fluidmode_ops as FO
from arc_flow import spawn as SP
from arc_flow.core import fluidmode as FM
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
sc.frame_start, sc.frame_end = 1, 8
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)


def reset():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for fl in fm.fluids:
        if fl.points_obj is not None:
            bpy.data.objects.remove(fl.points_obj, do_unlink=True)
    fm.fluids.clear()
    FO.RT.solver = None
    fm.max_particles = 6000
    fm.fill_mode = 'OFF'
    fm.rules_json = ""


def build():
    FO.mark_dirty()
    FO.start(sc, play=False)
    return FO.RT.solver


# ---- primitives
for k, _n, _d in SP.PRIM_ITEMS:
    V, T = SP.primitive_mesh(k, 0.1)
    closed = FM.is_closed(T) if k != 'PLANE' else False
    vol = FM.mesh_volume(V, T) if closed else 0
    ok(len(T) > 0 and (closed or k == 'PLANE'), "primitive %s: %d tris, %s" % (k, len(T), "closed, %.3g m3" % vol if closed else "open"))
V, T = SP.primitive_mesh('SPHERE', 0.1, 48)
ok(abs(FM.mesh_volume(V, T) - 4 / 3 * math.pi * 1e-3) / (4e-3) < 0.01, "sphere volume matches 4/3 pi r3")

# ---- default new fluid: sphere of points
reset()
fl = FO.add_fluid(sc, 'WATER', primitive=True)
ok(fl.shape_source == 'PRIMITIVE' and fl.prim_type == 'SPHERE', "new fluid defaults to a sphere primitive")
S = build()
ok(S.n > 500 and fl.d_particles == S.n, "default fluid builds a sphere of points (%d particles, spacing %.1f mm)" % (S.n, S.dx * 1e3))
c = S.x.mean(0); r = np.linalg.norm(S.x - c, axis=1).max()
ok(r < 0.1 * 1.15, "particles lie inside the sphere of radius 0.1 (max radius %.3f)" % r)
n_vol = S.n
for kind in ('CUBE', 'CYLINDER', 'CONE', 'TORUS', 'PLANE'):
    fl.prim_type = kind
    fl.spawn_mode = 'AUTO'
    S = build()
    ok(S.n > 20, "primitive %s: %d particles (%s)" % (kind, S.n, fl.d_spawn))
fl.prim_type = 'SPHERE'
for mode in ('VOLUME', 'SURFACE', 'POINTS'):
    fl.spawn_mode = mode
    S = build()
    print("   sphere", mode, S.n, fl.d_spawn)
    if mode == 'SURFACE':
        n_surf = S.n
        rr = np.linalg.norm(S.x - fl.prim_loc, axis=1)
        ok(rr.min() > 0.08, "Surface mode: one layer on the sphere (min radius %.3f)" % rr.min())
    if mode == 'POINTS':
        ok(S.n == len(SP.primitive_mesh('SPHERE', 0.1)[0]), "Vertices mode: one particle per vertex (%d)" % S.n)
fl.spawn_mode = 'VOLUME'
fl.spawn_density = 0.5
S = build()
ok(0.35 * n_vol < S.n < 0.65 * n_vol, "Density 0.5 keeps about half the volume particles (%d of %d)" % (S.n, n_vol))
fl.spawn_density = 1.0
bpy.ops.arcflow.fm_prim_to_object()
ok(fl.shape is not None and fl.shape_source == 'OBJECT' and fl.shape.type == 'MESH', "Make Editable Object converts the primitive to a mesh shape")
S = build()
ok(abs(S.n - n_vol) < 0.1 * n_vol, "the converted mesh spawns the same fluid (%d vs %d)" % (S.n, n_vol))

# ---- curve as a channel (Shape)
reset()
cu = bpy.data.curves.new("chan", 'CURVE'); cu.dimensions = '3D'
sp = cu.splines.new('POLY'); sp.points.add(3)
for p, co in zip(sp.points, [(0, 0, 0.3), (0.3, 0, 0.3), (0.3, 0.3, 0.3), (0.6, 0.3, 0.3)]):
    p.co = (*co, 1)
co_ = bpy.data.objects.new("chan", cu); sc.collection.objects.link(co_)
co_.arc_flow_fm.channel_radius = 0.03
fl = FO.add_fluid(sc, 'WATER'); fl.shape, fl.no_container = co_, True
S = build()
ok(S.n > 100 and fl.d_spawn.startswith("Volume"), "curve Shape: tubed channel filled (%d particles, %s)" % (S.n, fl.d_spawn))
ext = S.x.max(0) - S.x.min(0)
ok(abs(ext[0] - 0.66) < 0.1 and ext[2] < 0.09, "channel extent matches the curve + radius (%.2f x %.2f x %.2f)" % tuple(ext))
fl.spawn_mode = 'POINTS'
S = build()
ok(S.n > 10 and (S.x.max(0) - S.x.min(0))[2] < 0.02, "curve Points mode: a line of particles along the curve (%d)" % S.n)
fl.spawn_mode = 'SURFACE'
S = build()
ok(S.n > 100, "curve Surface mode: the pipe wall (%d)" % S.n)
# a curve with its own bevel uses it
cu.bevel_depth = 0.04
S = FO.RT.solver
fl.spawn_mode = 'AUTO'
S = build()
ok(S.n > 100, "beveled curve spawns through its evaluated mesh (%d)" % S.n)
cu.bevel_depth = 0.0

# ---- curve as a Container (pipe) with a fluid inside, metaball + text shapes
reset()
cu2 = bpy.data.curves.new("pipe", 'CURVE'); cu2.dimensions = '3D'
sp = cu2.splines.new('POLY'); sp.points.add(2)
for p, co in zip(sp.points, [(0, 0, 0.0), (0.4, 0, 0.0), (0.4, 0, 0.4)]):
    p.co = (*co, 1)
pipe = bpy.data.objects.new("pipe", cu2); sc.collection.objects.link(pipe)
pipe.arc_flow_fm.channel_radius = 0.05
fl = FO.add_fluid(sc, 'WATER', primitive=True)
fl.shape_source = 'OBJECT'; fl.no_container = False; fl.container = pipe; fl.fill = 40.0
S = build()
ok(len(S.bodies) == 1 and S.n > 50, "curve Container: a tubed pipe holds the fluid (%d particles)" % S.n)
tube_pts = S.x
ok(np.all(tube_pts[:, 1] > -0.07) and np.all(tube_pts[:, 1] < 0.07), "container fluid stays inside the pipe (|y| max %.3f)" % np.abs(tube_pts[:, 1]).max())
for t_ in range(3):
    S.step(1 / 24.0)
ok(S.outside_count() == 0 or S.outside_count() < 0.05 * S.n, "stepping the pipe fluid: %d particles outside" % S.outside_count())

reset()
mb = bpy.data.metaballs.new("mb"); mb.resolution = 0.01; mb.threshold = 0.6; e = mb.elements.new(); e.radius = 0.12; e.co = (0, 0, 0)
e2 = mb.elements.new(); e2.radius = 0.1; e2.co = (0.12, 0, 0)
mo = bpy.data.objects.new("mb", mb); sc.collection.objects.link(mo); mo.location = (0, 0, 0.5)
fl = FO.add_fluid(sc, 'WATER'); fl.shape, fl.no_container = mo, True
bpy.context.view_layer.update()
S = build()
ok(S.n > 100, "metaball Shape: filled (%d particles, %s)" % (S.n, fl.d_spawn))
reset()
tc = bpy.data.curves.new("txt", 'FONT'); tc.body = "AB"; tc.extrude = 0.02; tc.size = 0.2
to = bpy.data.objects.new("txt", tc); sc.collection.objects.link(to)
fl = FO.add_fluid(sc, 'WATER'); fl.shape, fl.no_container = to, True
S = build()
ok(S.n > 100, "text Shape (extruded): evaluated mesh spawns (%d particles, %s)" % (S.n, fl.d_spawn))
reset()
tc2 = bpy.data.curves.new("txt2", 'FONT'); tc2.body = "AB"; tc2.size = 0.2
to2 = bpy.data.objects.new("txt2", tc2); sc.collection.objects.link(to2)
fl = FO.add_fluid(sc, 'WATER'); fl.shape, fl.no_container = to2, True
S = build()
ok(S.n > 50 and "Surface" in fl.d_spawn, "flat text Shape: a sheet on the glyphs (%d particles)" % S.n)

# ---- particle system as a Shape
reset()
bpy.ops.mesh.primitive_plane_add(size=0.4, location=(0, 0, 0.5))
em = bpy.context.object
ps_mod = em.modifiers.new("ps", 'PARTICLE_SYSTEM')
st = ps_mod.particle_system.settings
st.count = 300; st.frame_start = 1; st.frame_end = 1; st.lifetime = 100; st.normal_factor = 0.0; st.physics_type = 'NO'
st.use_modifier_stack = False
sc.frame_set(1)
bpy.context.view_layer.update()
deps = bpy.context.evaluated_depsgraph_get()
pp = SP.particle_points(em, deps)
ok(pp is not None and len(pp) == 300, "particle_points reads the live Blender particles (%s)" % (None if pp is None else len(pp)))
fl = FO.add_fluid(sc, 'WATER'); fl.shape, fl.no_container = em, True
fl.spawn_mode = 'PARTICLES'
S = build()
ok(S.n == 300 and abs(S.x[:, 2].mean() - 0.5) < 0.05, "particle system Shape: one fluid particle per Blender particle (%d)" % S.n)
fl.spawn_density = 0.5
S = build()
ok(140 <= S.n <= 160, "particle Density 0.5 keeps half (%d)" % S.n)
fl.spawn_density = 1.0
fl.spawn_mode = 'AUTO'
S = build()
ok(S.n == 300, "Auto picks the particles when the object has a particle system (%d)" % S.n)

# ---- three independent assets in one scene
from arc_flow import assets_demo, systems, surfaces_panel
reset()
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
ok(assets_demo.KEY in surfaces_panel.LOADERS, "example registered in the Examples panel")
msg = assets_demo.load(sc, max_particles=5000)
det = systems.detect(sc, rebuild_sim=False)
ok(len(det['systems']) == 3, "Systems list: %d independent systems from the three assets" % len(det['systems']))
sc.frame_start, sc.frame_end = 1, 40
S = build()
ok(len(fm.fluids) == 3 and all(f.d_particles > 100 for f in fm.fluids), "three fluids built: %s" % [f.d_particles for f in fm.fluids])
c0 = [S.x[S.lo[i]:S.hi[i]].mean(0)[0] for i in range(3)]
T0 = [float(S.T[S.lo[i]:S.hi[i]].mean()) for i in range(3)]
for f in range(1, 31):
    sc.frame_set(f)
S = FO.RT.solver
c1 = [S.x[S.lo[i]:S.hi[i]].mean(0)[0] for i in range(3)]
T1 = [float(S.T[S.lo[i]:S.hi[i]].mean()) for i in range(3)]
print("   centroid x", np.round(c0, 3), "->", np.round(c1, 3), " T", np.round(T0, 1), "->", np.round(T1, 1), flush=True)
ok(S.mix_stats['reactions'] > 0, "blue + yellow REACT -> green: %d reactions" % S.mix_stats['reactions'])
nG = int(np.count_nonzero(S.SS.dominant(S.C) == S.SS.idx['Green']))
ok(nG > 0, "green product particles exist (%d)" % nG)
# mixing across assets: composition of the blue asset contains other species where it touched the yellow one
Cb = S.C[S.lo[0]:S.hi[0]]
ok(float(1.0 - Cb[:, 0].mean()) > 0.01, "mixing: the blue asset's particles carry %.1f %% other species" % (100 * (1.0 - Cb[:, 0].mean())))
ok(T1[1] > T0[1] + 0.5 or T1[0] > T0[0] + 0.5, "proximity coupling: the cold assets warmed next to the hot one (%.1f -> %.1f K, %.1f -> %.1f K)" % (T0[0], T1[0], T0[1], T1[1]))
ok(sc.objects.get("AFFMS_Liquids") is not None and len(sc.objects["AFFMS_Liquids"].data.vertices) > 100, "Surface mode: one merged liquid surface across the assets (%d verts)" % len(sc.objects["AFFMS_Liquids"].data.vertices))

# ---- spawn controls as Geometry Nodes
from arc_flow import tessera_nodes as TN, gn as GNm
ng = TN.ensure_group("Tessera Fluid Spawn")
ok(ng is not None and "Tessera Fluid Spawn" in dict((n, d) for n, d in TN.GROUPS), "Tessera Fluid Spawn registered and built")
ins = {i.name for i in ng.interface.items_tree if i.item_type == 'SOCKET' and i.in_out == 'INPUT'}
ok({"Shape", "Use Primitive", "Primitive", "Primitive Size", "Spawn Mode", "Spacing", "Density", "Seed"} <= ins, "spawn group inputs: %s" % sorted(ins))
me = bpy.data.meshes.new("gnprobe"); gp = bpy.data.objects.new("gnprobe", me); sc.collection.objects.link(gp)
gp.modifiers.new("spawn", 'NODES').node_group = ng
from arc_flow.nodeutil import Builder, add_socket, fresh_group, inp, out
pg = fresh_group("SpawnProbe")
add_socket(pg, "Geometry", 'NodeSocketGeometry', 'OUTPUT')
bp = Builder(pg); pgo = bp.node('NodeGroupOutput', x=800)
gnode = bp.node('GeometryNodeGroup', y=0); gnode.node_tree = ng
p2v = bp.node('GeometryNodePointsToVertices', y=0)
bp.link(out(gnode, 'Points'), inp(p2v, 'Points')); bp.link(out(p2v, 'Mesh'), inp(pgo, 'Geometry'))
gp.modifiers[0].node_group = pg
# group inputs are set on the nested node's sockets
def count(**kw):
    for k, v in kw.items():
        inp(gnode, k).default_value = v
    bpy.context.view_layer.update()
    ev = gp.evaluated_get(bpy.context.evaluated_depsgraph_get())
    return len(ev.data.vertices)
n_vol = count(**{"Use Primitive": True, "Primitive": 0, "Primitive Size": 0.1, "Spawn Mode": 0, "Spacing": 0.02, "Density": 1.0})
n_half = count(**{"Density": 0.5})
n_surf = count(**{"Density": 1.0, "Spawn Mode": 1})
n_vert = count(**{"Spawn Mode": 2})
n_cube = count(**{"Spawn Mode": 0, "Primitive": 1})
n_tor = count(**{"Primitive": 4})
print("   GN counts volume %s half %s surface %s vertices %s cube %s torus %s" % (n_vol, n_half, n_surf, n_vert, n_cube, n_tor), flush=True)
ok(n_vol > 300 and 0.35 * n_vol < n_half < 0.65 * n_vol, "node spawn: sphere volume %d points, Density 0.5 -> %d" % (n_vol, n_half))
ok(n_surf > 50 and n_vert > 100 and n_cube > n_vol and n_tor > 20, "node spawn: surface %d, vertices %d, cube %d, torus %d" % (n_surf, n_vert, n_cube, n_tor))
reset()
fl = FO.add_fluid(sc, 'WATER', primitive=True)
fl.spawn_mode, fl.spawn_density, fl.prim_type = 'SURFACE', 0.5, 'CUBE'
S = build()
sc.frame_set(1); sc.frame_set(2)
mod = fl.points_obj.modifiers["Tessera Fluid"]
vals = {}
for it in mod.node_group.interface.items_tree:
    if it.item_type == 'SOCKET' and it.in_out == 'INPUT' and it.name in ("Density", "Spawn Mode", "Spawn Density", "Primitive", "Use Primitive", "Primitive Size"):
        v = None
        for rd in (lambda: mod[it.identifier], lambda: mod.properties.inputs[it.identifier]['value'], lambda: mod.properties.inputs[it.identifier]):
            try:
                v = rd()
                if v is not None:
                    break
            except Exception:
                pass
        try:
            v = v.to_dict()
            v = v.get('value', v)
        except Exception:
            pass
        vals[it.name] = v
print("   per-fluid group", vals, flush=True)
ok(len(vals) == 6, "per-fluid group exposes Density, Spawn Mode, Spawn Density, Use Primitive, Primitive, Primitive Size (mirrored from the panel by set_fluid_inputs)")

print("FAILURES:", fails[0], flush=True)
sys.exit(1 if fails[0] else 0)
