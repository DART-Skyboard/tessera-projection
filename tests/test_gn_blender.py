"""Geometry Nodes parity / interoperability test:  blender --factory-startup -b --python tests/test_gn_blender.py"""
import os, sys
import bpy
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import arc_flow
from arc_flow import tessera_nodes as TN, gn
arc_flow.register()
fails = [0]
def ok(c, msg):
    print(("PASS " if c else "FAIL ") + msg, flush=True)
    if not c: fails[0] += 1
print("Blender", bpy.app.version_string, flush=True)

def evaluated(o):
    dg = bpy.context.evaluated_depsgraph_get(); dg.update()
    return o.evaluated_get(dg)

for n in ("Tessera Environment", "Tessera Particle Cloud", "Tessera Thermal", "Tessera Surface", "Tessera Fluid Source"):
    g = TN.build_group(n)
    ok(g is not None and len(g.nodes) > 3, "%s built (%d nodes)" % (n, len(g.nodes)))

from arc_flow.nodeutil import Builder, add_socket, fresh_group, inp, out

def user_tree(name, build):
    """A user node tree: Geometry in -> build(b, geo_socket) -> Geometry out (points turned into vertices so they can be read back)."""
    ng = fresh_group(name)
    add_socket(ng, "Geometry", 'NodeSocketGeometry', 'INPUT')
    add_socket(ng, "Geometry", 'NodeSocketGeometry', 'OUTPUT')
    b = Builder(ng)
    gi, go = b.group_io()
    res = build(b, out(gi, 'Geometry'))
    b.link(res, inp(go, 'Geometry'))
    return ng

def points_to_mesh(b, sock):
    p = b.node('GeometryNodePointsToVertices')
    b.link(sock, inp(p, 'Points'))
    return out(p, 'Mesh')

def run(obj, ng):
    for m in list(obj.modifiers): obj.modifiers.remove(m)
    m = obj.modifiers.new("t", 'NODES'); m.node_group = ng
    ev = evaluated(obj)
    return ev.data

def attrs(me):
    return {a.name: a for a in me.attributes}

def src_node(b, shape, **vals):
    n = b.node('GeometryNodeGroup'); n.node_tree = TN.ensure_group("Tessera Fluid Source")
    b.link(shape, inp(n, 'Shape'))
    for k, v in vals.items():
        inp(n, k.replace('_', ' ')).default_value = v
    return n

bpy.ops.mesh.primitive_cube_add(size=1.0)
cube = bpy.context.object

# 1. Fluid Source: fill a unit-ish cube with a lattice
def t1(b, geo):
    n = src_node(b, geo, Spacing=0.1, Density=800.0, Temperature=350.0, Viscosity=0.5, Cohesion=0.03)
    return points_to_mesh(b, out(n, 'Particles'))
me = run(cube, user_tree("T1", t1))
A = attrs(me)
nv = len(me.vertices)
ok(300 < nv < 1700, "Source fills the cube with a lattice (%d points for 1 m3 at 0.1 spacing)" % nv)
for k in ("fm_velocity", "fm_temperature", "fm_pressure", "fm_density", "fm_mass", "fm_color", "fm_fluid_id", "af_radius", "fm_viscosity", "fm_cohesion", "fm_reaction", "fm_flame_color", "fm_emission"):
    ok(k in A, "attribute %s on the particles" % k)
if "fm_temperature" in A:
    t = np.zeros(nv); A["fm_temperature"].data.foreach_get('value', t)
    ok(abs(t.mean() - 350.0) < 1e-3, "temperature input reaches the attribute (%.2f)" % t.mean())
if "fm_pressure" in A:
    p = np.zeros(nv); A["fm_pressure"].data.foreach_get('value', p)
    ok(p.min() >= 101325 - 1 and p.max() > p.min(), "hydrostatic pressure grows with depth (%.0f .. %.0f Pa)" % (p.min(), p.max()))
if "af_radius" in A:
    r = np.zeros(nv); A["af_radius"].data.foreach_get('value', r)
    ok(abs(r.mean() - 0.05) < 1e-6, "radius = spacing / 2")

def getf(me, name, n=None, vec=False):
    a = me.attributes[name]
    n = len(me.vertices) if n is None else n
    if vec:
        v = np.zeros(n * 3); a.data.foreach_get('vector', v); return v.reshape(n, 3)
    v = np.zeros(n); a.data.foreach_get('value', v); return v

# 2. other Source modes, Environment, Thermal as stand-alone groups
def mk_group_user(name, extra=None, **vals):
    def t(b, geo):
        n = b.node('GeometryNodeGroup'); n.node_tree = TN.ensure_group(name)
        b.link(geo, inp(n, 'Geometry'))
        for k, v in vals.items():
            inp(n, k.replace('_', ' ')).default_value = v
        return out(n, 'Geometry')
    return user_tree("U_" + name.replace(' ', '_'), t)
def mode_count(mode):
    def t(b, geo):
        n = src_node(b, geo, Spacing=0.1); n.inputs['Spawn Mode'].default_value = mode
        return points_to_mesh(b, out(n, 'Particles'))
    return len(run(cube, user_tree("M%d" % mode, t)).vertices)
ok(mode_count(1) > 200 and mode_count(2) == 8, "Source spawns on the surface (%d) and at the vertices (%d)" % (mode_count(1), mode_count(2)))
def t_cap(b, geo):
    n = src_node(b, geo, Spacing=0.1); n.inputs['Max Particles'].default_value = 100
    return points_to_mesh(b, out(n, 'Particles'))
ok(len(run(cube, user_tree("Cap", t_cap)).vertices) == 100, "Max Particles caps the count")
me = run(cube, mk_group_user("Tessera Environment", Gravity=(0, 0, -9.80665), Ambient_Pressure=101325.0, Wind=(2.0, 0, 0)))
pa = getf(me, 'fm_ambient_pressure')
ok(pa.max() > pa.min() > 90000 and abs(pa[np.argmin([v.co.z for v in me.vertices])] - 101325.0 * np.exp(0.5 / (8.314462618 * 288.15 / (0.0289647 * 9.80665)))) < 1.0,
   "Environment: barometric ambient pressure falls with height (%.1f .. %.1f Pa)" % (pa.min(), pa.max()))
ok(abs(getf(me, 'fm_wind', vec=True)[:, 0].mean() - 2.0) < 1e-5, "Environment: wind vector attribute")
# thermal: heat release x progress warms, flame colour appears
def t_th(b, geo):
    c = b.node('GeometryNodeGroup'); c.node_tree = TN.ensure_group("Tessera Particle Cloud"); b.link(geo, inp(c, 'Geometry')); inp(c, 'Temperature').default_value = 300.0
    t = b.node('GeometryNodeGroup'); t.node_tree = TN.ensure_group("Tessera Thermal"); b.link(out(c, 'Geometry'), inp(t, 'Geometry'))
    inp(t, 'Heat Release').default_value = 2.0e6; inp(t, 'Reaction Progress').default_value = 0.5; inp(t, 'Specific Heat').default_value = 1000.0
    inp(t, 'Gas').default_value = True; inp(t, 'Molar Mass').default_value = 0.029
    return out(t, 'Geometry')
me = run(cube, user_tree("Th", t_th))
T = getf(me, 'fm_temperature'); em = getf(me, 'fm_emission'); fc = np.zeros(len(me.vertices) * 4); me.attributes['fm_flame_color'].data.foreach_get('color', fc)
ok(abs(T.mean() - 1300.0) < 1.0 and em.min() > 0.2 and fc.reshape(-1, 4)[:, 0].min() > 0.5, "Thermal: reaction heat 0.5 x 2 MJ/kg / 1000 J/kgK -> %.0f K, emission %.2f, flame colour set" % (T.mean(), em.mean()))
P = getf(me, 'fm_pressure'); rho = getf(me, 'fm_density')
ok(abs(P.mean() - (rho * 8.314462618 * 1300.0 / 0.029).mean()) / P.mean() < 1e-3, "Thermal: ideal gas pressure")

# 3. interoperability: our output into stock nodes, and a second extension-style group in the chain
def other_ext_group():
    """A group as another extension would ship it: knows nothing about Tessera except the attribute names."""
    ng = fresh_group("OtherExt Heat Haze")
    add_socket(ng, "Geometry", 'NodeSocketGeometry', 'INPUT'); add_socket(ng, "Gain", 'NodeSocketFloat', 'INPUT', 0.01)
    add_socket(ng, "Geometry", 'NodeSocketGeometry', 'OUTPUT')
    b = Builder(ng); gi, go = b.group_io()
    t = b.node('GeometryNodeInputNamedAttribute', data_type='FLOAT'); inp(t, 'Name').default_value = 'fm_temperature'
    m = b.math('MULTIPLY', out(t, 'Attribute'), out(gi, 'Gain'))
    st = b.node('GeometryNodeStoreNamedAttribute', data_type='FLOAT', domain='POINT'); inp(st, 'Name').default_value = 'ext_haze'
    b.link(out(gi, 'Geometry'), inp(st, 'Geometry')); b.link(out(m, 'Value'), inp(st, 'Value'))
    sp = b.node('GeometryNodeSetPosition')
    b.link(out(st, 'Geometry'), inp(sp, 'Geometry'))
    off = b.node('ShaderNodeCombineXYZ'); b.link(out(m, 'Value'), off.inputs[2]); b.link(out(off, 'Vector'), inp(sp, 'Offset'))
    b.link(out(sp, 'Geometry'), inp(go, 'Geometry'))
    return ng
other = other_ext_group()
mat = bpy.data.materials.new("InteropMat")

def t_chain(b, geo):
    n = src_node(b, geo, Spacing=0.1, Temperature=400.0, Render_Mode=2)
    # stock chain on the particles: Set Position -> (other extension) -> Instance on Points -> Set Material
    sp = b.node('GeometryNodeSetPosition'); b.link(out(n, 'Particles'), inp(sp, 'Geometry'))
    off = b.node('ShaderNodeCombineXYZ'); off.inputs[0].default_value = 3.0; b.link(out(off, 'Vector'), inp(sp, 'Offset'))
    o = b.node('GeometryNodeGroup'); o.node_tree = other; b.link(out(sp, 'Geometry'), inp(o, 'Geometry'))
    ico = b.node('GeometryNodeMeshIcoSphere'); inp(ico, 'Radius').default_value = 0.02
    iop = b.node('GeometryNodeInstanceOnPoints'); b.link(out(o, 'Geometry'), inp(iop, 'Points')); b.link(out(ico, 'Mesh'), inp(iop, 'Instance'))
    mt = b.node('GeometryNodeSetMaterial'); b.link(out(iop, 'Instances'), inp(mt, 'Geometry')); inp(mt, 'Material').default_value = mat
    rl = b.node('GeometryNodeRealizeInstances'); b.link(out(mt, 'Geometry'), inp(rl, 'Geometry'))
    # stock chain on the surface: Subdivision Surface -> Set Material
    sd = b.node('GeometryNodeSubdivisionSurface'); b.link(out(n, 'Surface'), inp(sd, 'Mesh')); inp(sd, 'Level').default_value = 1
    sm = b.node('GeometryNodeSetMaterial'); b.link(out(sd, 'Mesh'), inp(sm, 'Geometry')); inp(sm, 'Material').default_value = mat
    # stock chain on the volume: Volume to Mesh
    vm = b.node('GeometryNodeVolumeToMesh'); b.link(out(n, 'Density Volume'), inp(vm, 'Volume'))
    inp(vm, 'Resolution Mode').default_value = 'Size'; inp(vm, 'Voxel Size').default_value = 0.08; inp(vm, 'Threshold').default_value = 0.1
    # mesh to volume of the surface and back (volume nodes accept our geometry)
    mv = b.node('GeometryNodeMeshToVolume'); b.link(out(sm, 'Geometry'), inp(mv, 'Mesh')); inp(mv, 'Resolution Mode').default_value = 'Size'; inp(mv, 'Voxel Size').default_value = 0.1
    vm2 = b.node('GeometryNodeVolumeToMesh'); b.link(out(mv, 'Volume'), inp(vm2, 'Volume')); inp(vm2, 'Resolution Mode').default_value = 'Size'; inp(vm2, 'Voxel Size').default_value = 0.1
    j = b.node('GeometryNodeJoinGeometry')
    for s_ in (out(rl, 'Geometry'), out(sm, 'Geometry'), out(vm, 'Mesh'), out(vm2, 'Mesh')):
        b.link(s_, inp(j, 'Geometry'))
    return out(j, 'Geometry')
me = run(cube, user_tree("Chain", t_chain))
nv = len(me.vertices)
ok(nv > 3000, "stock chain (Set Position, other extension, Instance on Points, Set Material, Subdivision, Volume to Mesh, Mesh to Volume) evaluates: %d vertices" % nv)
A = attrs(me)
ok("ext_haze" in A and "fm_temperature" in A, "attributes cross the other extension's group (ext_haze, fm_temperature)")
if "ext_haze" in A:
    ok(abs(getf(me, 'ext_haze').max() - 4.0) < 1e-3 or getf(me, 'ext_haze').max() > 3.0, "other extension read fm_temperature (haze %.2f)" % getf(me, 'ext_haze').max())
ok(any(m is not None and m.name.startswith("InteropMat") for m in me.materials), "Set Material reached the output (%s)" % [m.name if m else None for m in me.materials])
ok(float(np.array([v.co.x for v in me.vertices]).max()) > 3.2, "Set Position offset applied (x up to %.2f)" % max(v.co.x for v in me.vertices))

# Surface group on its own: attributes transferred to the welded surface
me = run(cube, user_tree("Surf", lambda b, g: out(src_node(b, g, Spacing=0.1, Temperature=333.0), 'Surface')))
ok(len(me.polygons) > 100 and "fm_temperature" in me.attributes and abs(getf(me, 'fm_temperature').mean() - 333.0) < 1e-3, "Surface mesh (%d faces) carries fm_temperature" % len(me.polygons))
ok(all(k in me.attributes for k in ("fm_color", "fm_velocity", "fm_density", "fm_pressure", "fm_fluid_id")), "Surface mesh carries colour, velocity, density, pressure, fluid id")

# 4. library: build, append, current-version check
import tempfile
tmp = os.path.join(tempfile.mkdtemp(), "lib.blend")
TN.build_library(tmp)
ok(os.path.getsize(tmp) > 10000, "library blend written (%d bytes)" % os.path.getsize(tmp))
ok(os.path.isfile(TN.LIB_PATH), "shipped library exists: %s" % TN.LIB_PATH)
if os.path.isfile(TN.LIB_PATH):
    for n in TN.GROUP_NAMES:
        g = bpy.data.node_groups.get(n)
        if g: bpy.data.node_groups.remove(g)
    for n in TN.GROUP_NAMES:
        ok(TN.append_from_library(n) or n in bpy.data.node_groups, "append %s from the library" % n)
    ok(all(n in bpy.data.node_groups and bpy.data.node_groups[n].get('tessera_gn_version') == TN.TESSERA_GN_VERSION for n in TN.GROUP_NAMES), "appended groups carry the current version")
    srcg = bpy.data.node_groups["Tessera Fluid Source"]
    ok(any(n.bl_idname == 'GeometryNodeGroup' and n.node_tree is not None and n.node_tree.name == "Tessera Particle Cloud" for n in srcg.nodes), "appended Source nests the appended Particle Cloud")
    def sig(g):
        return [(i.name, i.in_out, i.socket_type) for i in g.interface.items_tree if i.item_type == 'SOCKET']
    bpy.ops.wm.read_factory_settings(use_empty=True)
    arc_flow.register() if not hasattr(bpy.types, 'AF_MT_gn_add') else None
    lib_sigs = {}
    for n in TN.GROUP_NAMES:
        TN.append_from_library(n)
        lib_sigs[n] = sig(bpy.data.node_groups[n])
    for n in TN.GROUP_NAMES:
        bpy.data.node_groups.remove(bpy.data.node_groups[n])
    for n in ("Tessera Environment", "Tessera Particle Cloud", "Tessera Thermal", "Tessera Surface", "Tessera Fluid Source"):
        TN.build_group(n)
    ok(all(sig(bpy.data.node_groups[n]) == lib_sigs[n] for n in TN.GROUP_NAMES), "shipped library matches the code (run tools/build_node_library.py after changing a group)")

# 5. Add menu entry and operator
print([(n, hasattr(bpy.types, n)) for n in ('AF_MT_gn_add', 'AF_OT_add_tessera_group', 'AF_PT_gn_nodes_editor', 'AF_PT_gn_nodes_view3d')])
ok(hasattr(bpy.types, 'AF_MT_gn_add') and hasattr(bpy.ops.arcflow, 'add_tessera_group') and hasattr(bpy.types, 'AF_PT_gn_nodes_editor'), "Add menu, operator and panel registered")
ok(hasattr(bpy.types, 'NODE_MT_geometry_node_add_all') and any(getattr(f, '__name__', '') == '_add_menu_entry' for f in bpy.types.NODE_MT_geometry_node_add_all._dyn_ui_initialize()), "Geometry Nodes Add menu hook installed")
bpy.ops.mesh.primitive_cube_add(); ob = bpy.context.object
mod = ob.modifiers.new("host", 'NODES'); ng_host = bpy.data.node_groups.new("Host", 'GeometryNodeTree'); mod.node_group = ng_host
class _Ctx0: space_data = None
ok(TN.add_group_node(_Ctx0, "Tessera Thermal") is None, "add_group_node needs a Geometry Nodes editor")
class _Sp2:
    edit_tree = ng_host
    type = 'NODE_EDITOR'; tree_type = 'GeometryNodeTree'
class _Ctx2: space_data = _Sp2
n = TN.add_group_node(_Ctx2, "Tessera Thermal")
ok(n is not None and n.node_tree.name == "Tessera Thermal" and n in list(ng_host.nodes), "Add Tessera node group puts a Thermal group node in the tree")
bpy.ops.object.select_all(action='DESELECT'); ob.select_set(True); bpy.context.view_layer.objects.active = ob
bpy.ops.arcflow.add_tessera_modifier(group="Tessera Surface")
ok(any(m.type == 'NODES' and m.node_group and m.node_group.name == "Tessera Surface" for m in ob.modifiers), "panel button adds a Tessera modifier to the active object")

# 6. other modules can add their groups to the menu / ensure_group
def _dummy():
    g = fresh_group("Tessera Dummy Extra"); g['tessera_gn_version'] = TN.TESSERA_GN_VERSION
    add_socket(g, "Geometry", 'NodeSocketGeometry', 'INPUT'); add_socket(g, "Geometry", 'NodeSocketGeometry', 'OUTPUT')
    b = Builder(g); gi, go = b.group_io(); b.link(out(gi, 'Geometry'), inp(go, 'Geometry')); return g
TN.register_group("Tessera Dummy Extra", _dummy, "test group")
ok(any(i[0] == "Tessera Dummy Extra" for i in TN._enum_items(None, None)) and TN.ensure_group("Tessera Dummy Extra").name == "Tessera Dummy Extra", "register_group adds a group to the menu list and ensure_group")
print("fail count", fails[0])
sys.exit(1 if fails[0] else 0)
