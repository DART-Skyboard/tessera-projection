"""Geometry Nodes side of the Fluid mode spawn controls (Tessera Projection).

"Tessera Fluid Spawn"  Shape (any geometry: mesh, curve, text, point cloud) or a Primitive (Sphere, Cube, Cylinder, Cone, Torus, Plane), a Spawn Mode
                       (0 volume fill, 1 surface, 2 vertices / points), Spacing, Density (fraction kept) and Seed -> Points, Particle Count.
It is the node version of the Shape / Spawn controls of a fluid in the panel (same modes, same density meaning); registered in the Add menu
through tessera_nodes.register_group.  The per-fluid group "Tessera Fluid <name>" mirrors the panel values as the sockets Spawn Mode,
Spawn Density, Primitive and Primitive Size (see fluidmode_gn)."""
import bpy

from . import gn
from .nodeutil import Builder, add_socket, fresh_group, inp, out

NAME = "Tessera Fluid Spawn"
VERSION = 1


def _cmp(b, a_socket, v, y, op='EQUAL'):
    c = b.node('FunctionNodeCompare', data_type='INT', operation=op, y=y)
    ii = [s for s in c.inputs if s.type == 'INT' and s.enabled]
    b.link(a_socket, ii[0])
    ii[1].default_value = v
    return out(c, 'Result')


def build_spawn(name=NAME):
    from . import tessera_nodes as TN
    ng = fresh_group(name)
    add_socket(ng, "Shape", 'NodeSocketGeometry', 'INPUT', desc="Any geometry: closed mesh (volume), mesh / text / curve (surface, vertices), points")
    add_socket(ng, "Use Primitive", 'NodeSocketBool', 'INPUT', True, desc="Spawn from the Primitive instead of the Shape")
    add_socket(ng, "Primitive", 'NodeSocketInt', 'INPUT', 0, 0, 5, desc="0 sphere, 1 cube, 2 cylinder, 3 cone, 4 torus, 5 plane")
    add_socket(ng, "Primitive Size", 'NodeSocketFloat', 'INPUT', 0.1, 1e-3, desc="Radius / half edge / major radius, m")
    add_socket(ng, "Spawn Mode", 'NodeSocketInt', 'INPUT', 0, 0, 2, desc="0 volume fill, 1 surface, 2 vertices / points")
    add_socket(ng, "Spacing", 'NodeSocketFloat', 'INPUT', 0.01, 1e-3, desc="m between particles")
    add_socket(ng, "Density", 'NodeSocketFloat', 'INPUT', 1.0, 0.0, 1.0, desc="Fraction of the generated points that is kept")
    add_socket(ng, "Seed", 'NodeSocketInt', 'INPUT', 0)
    add_socket(ng, "Points", 'NodeSocketGeometry', 'OUTPUT')
    add_socket(ng, "Particle Count", 'NodeSocketInt', 'OUTPUT')
    b = Builder(ng)
    gi, go = b.group_io()
    size = out(gi, 'Primitive Size')
    diam = b.math('MULTIPLY', size, 2.0, y=700)
    sph = b.node('GeometryNodeMeshUVSphere', y=650)
    inp(sph, 'Segments').default_value = 32
    inp(sph, 'Rings').default_value = 16
    b.link(size, inp(sph, 'Radius'))
    cube = b.node('GeometryNodeMeshCube', y=600)
    cmb = b.node('ShaderNodeCombineXYZ', y=600)
    for i in range(3):
        b.link(out(diam, 'Value'), cmb.inputs[i])
    b.link(out(cmb, 'Vector'), inp(cube, 'Size'))
    cyl = b.node('GeometryNodeMeshCylinder', y=550)
    b.link(size, inp(cyl, 'Radius')); b.link(out(diam, 'Value'), inp(cyl, 'Depth'))
    cone = b.node('GeometryNodeMeshCone', y=500)
    b.link(size, inp(cone, 'Radius Bottom')); b.link(out(diam, 'Value'), inp(cone, 'Depth'))
    inp(cone, 'Radius Top').default_value = 0.0
    c1 = b.node('GeometryNodeCurvePrimitiveCircle', y=450)
    inp(c1, 'Resolution').default_value = 48
    b.link(size, inp(c1, 'Radius'))
    c2 = b.node('GeometryNodeCurvePrimitiveCircle', y=400)
    inp(c2, 'Resolution').default_value = 16
    mn = b.math('MULTIPLY', size, 0.35, y=400)
    b.link(out(mn, 'Value'), inp(c2, 'Radius'))
    c2m = b.node('GeometryNodeCurveToMesh', y=420)
    b.link(out(c1, 'Curve'), inp(c2m, 'Curve'))
    b.link(out(c2, 'Curve'), inp(c2m, 'Profile Curve'))
    plane = b.node('GeometryNodeMeshGrid', y=350)
    b.link(out(diam, 'Value'), inp(plane, 'Size X')); b.link(out(diam, 'Value'), inp(plane, 'Size Y'))
    inp(plane, 'Vertices X').default_value = 2
    inp(plane, 'Vertices Y').default_value = 2
    pr = out(gi, 'Primitive')
    g = out(sph, 'Mesh')
    for k, sock in ((1, out(cube, 'Mesh')), (2, out(cyl, 'Mesh')), (3, out(cone, 'Mesh')), (4, out(c2m, 'Mesh')), (5, out(plane, 'Mesh'))):
        g = gn._sw_geo(b, _cmp(b, pr, k, 300 - 20 * k), g, sock, y=300 - 20 * k)
    shape = gn._sw_geo(b, out(gi, 'Use Primitive'), out(gi, 'Shape'), g, y=250)
    sp = out(gi, 'Spacing')
    voxel = b.math('MULTIPLY', sp, 0.5, y=200)
    m2v = b.node('GeometryNodeMeshToVolume', y=200)
    TN._res_mode(m2v, 'Size')
    b.link(shape, inp(m2v, 'Mesh'))
    b.link(out(voxel, 'Value'), inp(m2v, 'Voxel Size'))
    inp(m2v, 'Interior Band Width').default_value = 1000.0
    dpv = b.node('GeometryNodeDistributePointsInVolume', y=200)
    try:
        inp(dpv, 'Mode').default_value = 'Grid'
    except Exception:
        pass
    b.link(out(m2v, 'Volume'), inp(dpv, 'Volume'))
    sc = b.node('ShaderNodeCombineXYZ', y=150)
    for i in range(3):
        b.link(sp, sc.inputs[i])
    b.link(out(sc, 'Vector'), inp(dpv, 'Spacing'))
    inp(dpv, 'Threshold').default_value = 0.1
    area = b.math('MULTIPLY', sp, sp, y=100)
    dens = b.math('DIVIDE', 1.0, out(area, 'Value'), y=100)
    dpf = b.node('GeometryNodeDistributePointsOnFaces', y=100)
    b.link(shape, inp(dpf, 'Mesh'))
    b.link(out(dens, 'Value'), inp(dpf, 'Density'))
    b.link(out(gi, 'Seed'), inp(dpf, 'Seed'))
    m2p = b.node('GeometryNodeMeshToPoints', mode='VERTICES', y=50)
    b.link(shape, inp(m2p, 'Mesh'))
    mode = out(gi, 'Spawn Mode')
    s = gn._sw_geo(b, _cmp(b, mode, 2, 0), out(dpv, 'Points'), out(m2p, 'Points'), y=0)
    s = gn._sw_geo(b, _cmp(b, mode, 1, -40), s, out(dpf, 'Points'), y=-40)
    # density: keep a random fraction
    rv = b.node('FunctionNodeRandomValue', data_type='FLOAT', y=-100)
    b.link(out(gi, 'Seed'), inp(rv, 'Seed'))
    gt = b.node('FunctionNodeCompare', data_type='FLOAT', operation='GREATER_THAN', y=-100)
    ff = [x for x in gt.inputs if x.type == 'VALUE' and x.enabled]
    b.link(out(rv, 'Value'), ff[0])
    b.link(out(gi, 'Density'), ff[1])
    dele = b.node('GeometryNodeDeleteGeometry', y=-100)
    b.link(s, inp(dele, 'Geometry'))
    b.link(out(gt, 'Result'), inp(dele, 'Selection'))
    cnt = b.node('GeometryNodeAttributeDomainSize', component='POINTCLOUD', y=-150)
    b.link(out(dele, 'Geometry'), inp(cnt, 'Geometry'))
    b.link(out(dele, 'Geometry'), inp(go, 'Points'))
    b.link(out(cnt, 'Point Count'), inp(go, 'Particle Count'))
    ng['spawn_version'] = VERSION
    return TN._tag(ng)


def register():
    try:
        from . import tessera_nodes
        tessera_nodes.register_group(NAME, build_spawn, "Primitive or any shape to particles: Volume / Surface / Vertices spawn with a density control (the panel's Spawn controls as nodes)")
    except Exception:
        pass
