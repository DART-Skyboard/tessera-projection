"""Fluid mode Geometry Nodes sockets on a real simulation:  blender --factory-startup -b --python tests/test_gn_fluid_blender.py"""
import os, sys, time
import bpy
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import arc_flow
from arc_flow import fluidmode_ops as FO, fluidmode_gn as FGN, tessera_nodes as TN
from arc_flow.nodeutil import Builder, add_socket, fresh_group, inp, out
arc_flow.register()
fails = [0]
def ok(c, msg):
    print(("PASS " if c else "FAIL ") + msg, flush=True)
    if not c: fails[0] += 1
print("Blender", bpy.app.version_string, flush=True)
sc = bpy.context.scene; fm = sc.arc_flow_fm
fm.mode = 'FLUID'
FO.load_lava_lamp(sc, max_particles=3000)
sc.frame_start, sc.frame_end = 1, 4
FO.start(sc, play=False)
for f in range(1, 4):
    sc.frame_set(f)
fl = fm.fluids[1]
o = fl.points_obj
mod = o.modifiers["Tessera Fluid"]
ng = mod.node_group
ins = {i.name for i in ng.interface.items_tree if i.item_type == 'SOCKET' and i.in_out == 'INPUT'}
outs = {i.name for i in ng.interface.items_tree if i.item_type == 'SOCKET' and i.in_out == 'OUTPUT'}
need_in = {"Geometry", "Mass", "Volume", "Density", "Temperature", "Velocity", "Viscosity", "Cohesion", "Mixing Rate", "Colour", "Fluid ID", "Gas", "No Container", "Container",
           "Shape", "Fill", "Gas Parcels", "Specific Heat", "Heat Added", "Heat Release", "Reaction Progress", "Gravity", "Ambient Pressure", "Ambient Temperature", "Wind",
           "Render Mode", "Voxel Size", "Scale"}
need_out = {"Geometry", "Density Volume", "Surface", "Render", "Particle Count", "Total Mass", "Mean Density", "Mean Temperature", "Max Temperature", "Mean Pressure", "Max Speed"}
ok(need_in <= ins, "per-fluid group inputs: missing %s" % sorted(need_in - ins))
ok(need_out <= outs, "per-fluid group outputs: missing %s" % sorted(need_out - outs))
# our inputs reached the modifier
def mval(name):
    for it in ng.interface.items_tree:
        if it.item_type == 'SOCKET' and it.in_out == 'INPUT' and it.name == name:
            try:
                return mod[it.identifier]
            except Exception:
                try:
                    return mod.properties.inputs[it.identifier]['value']
                except Exception:
                    return None
print("modifier Density", mval("Density"), "Viscosity", mval("Viscosity"), "Ambient Pressure", mval("Ambient Pressure"))

# probe: our per-fluid group on a copy of the particle mesh, read back through stock Points to Vertices-free mesh output
probe = bpy.data.objects.new("probe", o.data.copy()); sc.collection.objects.link(probe)
def chain(mode, build, extra=False):
    t = fresh_group("Probe")
    add_socket(t, "Geometry", 'NodeSocketGeometry', 'INPUT'); add_socket(t, "Geometry", 'NodeSocketGeometry', 'OUTPUT')
    b = Builder(t); gi, go = b.group_io()
    n = b.node('GeometryNodeGroup'); n.node_tree = ng; b.link(out(gi, 'Geometry'), inp(n, 'Geometry'))
    inp(n, 'Render Mode').default_value = mode
    inp(n, 'Build Volume And Surface').default_value = bool(extra)
    inp(n, 'Voxel Size').default_value = max(float(FO.RT.solver.dx) * 0.9, 1e-3)
    b.link(out(n, build), inp(go, 'Geometry'))
    for m in list(probe.modifiers): probe.modifiers.remove(m)
    m = probe.modifiers.new("p", 'NODES'); m.node_group = t
    t0 = time.time()
    dg = bpy.context.evaluated_depsgraph_get(); dg.update()
    return probe.evaluated_get(dg).data, time.time() - t0
me, dt0 = chain(0, 'Geometry')
names = {a.name for a in me.attributes}
for k in ("fm_velocity", "fm_temperature", "fm_pressure", "fm_density", "fm_mass", "fm_color", "fm_fluid_id", "af_radius", "fm_viscosity", "fm_cohesion", "fm_mix_rate", "fm_ambient_pressure", "fm_wind"):
    ok(k in names, "solver particles carry %s through the fluid group" % k)
P = np.zeros(len(me.vertices)); me.attributes['fm_pressure'].data.foreach_get('value', P)
ok(P.min() >= 101325 - 5 and P.max() > P.min(), "fm_pressure (%.0f .. %.0f Pa)" % (P.min(), P.max()))
fid = np.zeros(len(me.vertices), np.int32); me.attributes['fm_fluid_id'].data.foreach_get('value', fid)
ok((fid == 1).all(), "fm_fluid_id is the solver's id (%s)" % np.unique(fid))
# the solver's temperature is kept (Cloud only fills missing attributes)
T = np.zeros(len(me.vertices)); me.attributes['fm_temperature'].data.foreach_get('value', T)
ok(abs(T.mean() - np.mean([v for v in [fm.fluids[1].temperature]])) < 60 and T.std() >= 0.0, "solver temperature preserved (mean %.2f K)" % T.mean())
me2, dt2 = chain(2, 'Render')
ok(len(me2.polygons) > 20, "Render Mode 2 gives a surface mesh (%d faces)" % len(me2.polygons))
me3, dt3 = chain(0, 'Surface')
ok(len(me3.polygons) == 0, "Surface output stays empty (and unevaluated) unless asked")
me4, _t = chain(0, 'Surface', True)
ok(len(me4.polygons) > 20, "Build Volume And Surface gives the Surface output (%d faces)" % len(me4.polygons))
print("timing: points %.3fs, surface render %.3fs" % (dt0, dt2))
FO.stop(sc)
print("fail count", fails[0])
sys.exit(1 if fails[0] else 0)
