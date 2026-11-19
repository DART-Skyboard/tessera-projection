"""Geometry Nodes groups of the surface sources and fluid volumes (Tessera Projection, Fluid mode).

"Tessera Viscosity Surface <name>"  Inputs: Geometry (the mesh), Viscosity, No-slip, Reach, Samples per m2.  Outputs: Geometry (unchanged),
                                    Surface Samples (points on the stored faces with fm_normal), Area (m2 of the stored faces), Drag Force
                                    (the reaction force the solver measured, written back every frame).
"Tessera Emission Surface <name>"   Inputs: Geometry, Rate, Speed, Spread, Temperature, Reverse, Points per m2.  Outputs: Geometry,
                                    Emitter Points (on the stored faces; fm_velocity = normal x speed, fm_temperature), Area, Emitted
                                    (particles emitted so far, written back).
"Tessera Fluid Volume <name>"       Inputs: Size, Quantity, Density, Pressure, Temperature, Velocity.  Outputs: Geometry (Quantity points
                                    filling the box, with fm_velocity and fm_temperature), Volume (m3), Mass (kg), Pressure (Pa).
The groups are built once (your edits survive; delete a group to rebuild it).  The face selection is read from the boolean face attribute
that the Apply button stores on the mesh."""
import bpy

from . import gn
from .nodeutil import Builder, add_socket, fresh_group, inp, out


def group_name(kind, name):
    return "Tessera %s %s" % ({'VISCOSITY': "Viscosity Surface", 'EMISSION': "Emission Surface", 'VOLUME': "Fluid Volume"}[kind], gn.slug(name))


def _face_selection(b, attr, y):
    n = b.node('GeometryNodeInputNamedAttribute', data_type='BOOLEAN', y=y)
    inp(n, 'Name').default_value = attr
    return out(n, 'Attribute')


def _store(b, geo, name, value, data_type, y, domain='POINT'):
    n = b.node('GeometryNodeStoreNamedAttribute', data_type=data_type, domain=domain, y=y)
    inp(n, 'Name').default_value = name
    b.link(geo, inp(n, 'Geometry'))
    b.link(value, inp(n, 'Value'))
    return out(n, 'Geometry')


def _vmath(b, op, a, c=None, y=0):
    n = b.node('ShaderNodeVectorMath', operation=op, y=y)
    b.link(a, n.inputs[0])
    if c is not None:
        if isinstance(c, (int, float)):
            n.inputs[3].default_value = c      # the Scale input
        else:
            b.link(c, n.inputs[1])
    return n


def _area(b, gi, sel, y):
    """Total area of the selected faces: face area summed over the selection."""
    fa = b.node('GeometryNodeInputMeshFaceArea', y=y)
    ss = b.node('GeometryNodeAttributeStatistic', data_type='FLOAT', domain='FACE', y=y)
    b.link(out(gi, 'Geometry'), inp(ss, 'Geometry'))
    b.link(sel, inp(ss, 'Selection'))
    b.link(out(fa, 'Area'), inp(ss, 'Attribute'))
    return out(ss, 'Sum')


def _samples(b, gi, sel, density, y):
    d = b.node('GeometryNodeDistributePointsOnFaces', distribute_method='RANDOM', y=y)
    b.link(out(gi, 'Geometry'), inp(d, 'Mesh'))
    b.link(sel, inp(d, 'Selection'))
    b.link(density, inp(d, 'Density'))
    return d


def build_viscosity_group(name, attr):
    ng = fresh_group(name)
    add_socket(ng, "Geometry", 'NodeSocketGeometry', 'INPUT')
    add_socket(ng, "Viscosity", 'NodeSocketFloat', 'INPUT', 50.0, 0.0, desc="Pa s")
    add_socket(ng, "No-slip", 'NodeSocketFloat', 'INPUT', 1.0, 0.0, 1.0, desc="Fraction of the contact that sticks")
    add_socket(ng, "Reach", 'NodeSocketFloat', 'INPUT', 0.0, 0.0, desc="m (0 = 1.5 particle spacings)")
    add_socket(ng, "Samples per m2", 'NodeSocketFloat', 'INPUT', 400.0, 0.0, desc="Density of the sample points shown on the faces")
    add_socket(ng, "Drag Force", 'NodeSocketVector', 'INPUT', (0.0, 0.0, 0.0), desc="Read-back of the reaction force on the surface (N), written by the solver")
    add_socket(ng, "Geometry", 'NodeSocketGeometry', 'OUTPUT')
    add_socket(ng, "Surface Samples", 'NodeSocketGeometry', 'OUTPUT')
    add_socket(ng, "Area", 'NodeSocketFloat', 'OUTPUT', desc="m2 of the stored faces")
    add_socket(ng, "Drag Force", 'NodeSocketVector', 'OUTPUT')
    b = Builder(ng)
    gi, go = b.group_io()
    sel = _face_selection(b, attr, 300)
    d = _samples(b, gi, sel, out(gi, 'Samples per m2'), 100)
    pts = _store(b, out(d, 'Points'), 'fm_normal', out(d, 'Normal'), 'FLOAT_VECTOR', 100)
    pts = _store(b, pts, 'fm_viscosity', out(gi, 'Viscosity'), 'FLOAT', 100)
    b.link(out(gi, 'Geometry'), inp(go, 'Geometry'))
    b.link(pts, inp(go, 'Surface Samples'))
    b.link(_area(b, gi, sel, -200), inp(go, 'Area'))
    b.link(out(gi, 'Drag Force'), inp(go, 'Drag Force'))
    return ng


def build_emission_group(name, attr):
    ng = fresh_group(name)
    add_socket(ng, "Geometry", 'NodeSocketGeometry', 'INPUT')
    add_socket(ng, "Rate", 'NodeSocketFloat', 'INPUT', 240.0, 0.0, desc="Particles per second")
    add_socket(ng, "Speed", 'NodeSocketFloat', 'INPUT', 0.8, desc="m/s along the face normal")
    add_socket(ng, "Spread", 'NodeSocketFloat', 'INPUT', 10.0, 0.0, 89.0, desc="Cone half angle, degrees")
    add_socket(ng, "Temperature", 'NodeSocketFloat', 'INPUT', 293.15, 1.0, desc="K")
    add_socket(ng, "Reverse", 'NodeSocketBool', 'INPUT', False)
    add_socket(ng, "Points per m2", 'NodeSocketFloat', 'INPUT', 400.0, 0.0, desc="Density of the emitter points shown on the faces")
    add_socket(ng, "Emitted", 'NodeSocketInt', 'INPUT', 0, desc="Read-back: particles emitted so far, written by the solver")
    add_socket(ng, "Geometry", 'NodeSocketGeometry', 'OUTPUT')
    add_socket(ng, "Emitter Points", 'NodeSocketGeometry', 'OUTPUT')
    add_socket(ng, "Area", 'NodeSocketFloat', 'OUTPUT', desc="m2 of the stored faces")
    add_socket(ng, "Emitted", 'NodeSocketInt', 'OUTPUT')
    b = Builder(ng)
    gi, go = b.group_io()
    sel = _face_selection(b, attr, 300)
    d = _samples(b, gi, sel, out(gi, 'Points per m2'), 100)
    sgn = b.node('GeometryNodeSwitch', input_type='FLOAT', y=-100)
    b.link(out(gi, 'Reverse'), inp(sgn, 'Switch'))
    inp(sgn, 'False').default_value = 1.0
    inp(sgn, 'True').default_value = -1.0
    spd = b.math('MULTIPLY', out(gi, 'Speed'), out(sgn, 'Output'), y=-100)
    vel = _vmath(b, 'SCALE', out(d, 'Normal'), y=-100)
    b.link(out(spd, 'Value'), vel.inputs[3])
    pts = _store(b, out(d, 'Points'), 'fm_velocity', out(vel, 'Vector'), 'FLOAT_VECTOR', 100)
    pts = _store(b, pts, 'fm_temperature', out(gi, 'Temperature'), 'FLOAT', 100)
    b.link(out(gi, 'Geometry'), inp(go, 'Geometry'))
    b.link(pts, inp(go, 'Emitter Points'))
    b.link(_area(b, gi, sel, -300), inp(go, 'Area'))
    b.link(out(gi, 'Emitted'), inp(go, 'Emitted'))
    return ng


def build_volume_group(name):
    ng = fresh_group(name)
    add_socket(ng, "Geometry", 'NodeSocketGeometry', 'INPUT')
    add_socket(ng, "Size", 'NodeSocketVector', 'INPUT', (1.0, 1.0, 1.0), desc="Box dimensions in m (the Empty's scale x its display size x 2)")
    add_socket(ng, "Quantity", 'NodeSocketInt', 'INPUT', 1500, 1, desc="Particles / gas parcels")
    add_socket(ng, "Density", 'NodeSocketFloat', 'INPUT', 1.204, 0.0, desc="kg/m3 at the given pressure and temperature")
    add_socket(ng, "Pressure", 'NodeSocketFloat', 'INPUT', 101325.0, 0.0, desc="Pa")
    add_socket(ng, "Temperature", 'NodeSocketFloat', 'INPUT', 293.15, 1.0, desc="K")
    add_socket(ng, "Velocity", 'NodeSocketVector', 'INPUT', (0.0, 0.0, 0.0), desc="m/s (initial)")
    add_socket(ng, "Geometry", 'NodeSocketGeometry', 'OUTPUT')
    add_socket(ng, "Volume", 'NodeSocketFloat', 'OUTPUT', desc="m3")
    add_socket(ng, "Mass", 'NodeSocketFloat', 'OUTPUT', desc="kg = density x volume")
    add_socket(ng, "Pressure", 'NodeSocketFloat', 'OUTPUT', desc="Pa")
    b = Builder(ng)
    gi, go = b.group_io()
    rv = b.node('FunctionNodeRandomValue', data_type='FLOAT_VECTOR', y=250)
    inp(rv, 'Min').default_value = (-0.5, -0.5, -0.5)
    inp(rv, 'Max').default_value = (0.5, 0.5, 0.5)
    pos = _vmath(b, 'MULTIPLY', out(rv, 'Value'), out(gi, 'Size'), y=250)
    pn = b.node('GeometryNodePoints', y=100)
    b.link(out(gi, 'Quantity'), inp(pn, 'Count'))
    b.link(out(pos, 'Vector'), inp(pn, 'Position'))
    geo = _store(b, out(pn, 'Points'), 'fm_velocity', out(gi, 'Velocity'), 'FLOAT_VECTOR', 100)
    geo = _store(b, geo, 'fm_temperature', out(gi, 'Temperature'), 'FLOAT', 100)
    sep = b.node('ShaderNodeSeparateXYZ', y=-150)
    b.link(out(gi, 'Size'), sep.inputs[0])
    xy = b.math('MULTIPLY', out(sep, 'X'), out(sep, 'Y'), y=-150)
    vol = b.math('MULTIPLY', out(xy, 'Value'), out(sep, 'Z'), y=-150)
    mass = b.math('MULTIPLY', out(vol, 'Value'), out(gi, 'Density'), y=-250)
    b.link(geo, inp(go, 'Geometry'))
    b.link(out(vol, 'Value'), inp(go, 'Volume'))
    b.link(out(mass, 'Value'), inp(go, 'Mass'))
    b.link(out(gi, 'Pressure'), inp(go, 'Pressure'))
    return ng


def source_group(src):
    name = group_name(src.kind, src.name)
    ng = bpy.data.node_groups.get(name)
    if ng is None or not ng.nodes:
        if src.kind == 'VISCOSITY':
            ng = build_viscosity_group(name, src.face_attr or "af_srf_" + gn.slug(src.name))
        elif src.kind == 'EMISSION':
            ng = build_emission_group(name, src.face_attr or "af_srf_" + gn.slug(src.name))
        else:
            ng = build_volume_group(name)
    return ng


def set_inputs(mod, src, size=None, fluid_density=None):
    """Push the panel values into the modifier's group inputs."""
    if src.kind == 'VISCOSITY':
        vals = {"Viscosity": src.coef, "No-slip": src.noslip, "Reach": src.reach, "Drag Force": tuple(src.out_force)}
    elif src.kind == 'EMISSION':
        vals = {"Rate": src.rate, "Speed": src.speed, "Spread": src.spread, "Temperature": src.temperature, "Reverse": bool(src.reverse),
                "Emitted": int(src.out_count)}
    else:
        vals = {"Size": tuple(size or (1, 1, 1)), "Quantity": int(src.quantity), "Density": fluid_density or src.density, "Pressure": src.pressure,
                "Temperature": src.temperature, "Velocity": tuple(src.velocity0)}
    gn.set_modifier_inputs(mod, vals)


def ensure_modifier(host, src, size=None, fluid_density=None):
    """The group + a Geometry Nodes modifier 'Tessera Surface' on the host object (the source mesh, or the helper mesh of a volume)."""
    ng = source_group(src)
    mod = host.modifiers.get("Tessera Source")
    if mod is None:
        mod = host.modifiers.new("Tessera Source", 'NODES')
    mod.node_group = ng
    set_inputs(mod, src, size, fluid_density)
    return mod
