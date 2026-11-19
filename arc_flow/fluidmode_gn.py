"""Geometry Nodes for Fluid mode (Tessera Projection): the same per-fluid inputs and outputs as the panel, as real node groups.

"Tessera Fluid <name>"        per fluid.  Inputs: Geometry (the particle object's points), Mass, Volume, Density, Temperature,
                              Velocity, Viscosity, Colour, No Container, Container, Scale.  Outputs: Geometry (points with the
                              attributes fm_velocity, fm_temperature, fm_density, fm_fluid_id, fm_color, af_radius),
                              Particle Count, Mean Temperature, Max Speed.
"Tessera Mesh Per Particle"   instance an object / collection on every particle (scaled by the particle radius, optionally aligned
                              to the velocity).  The material is a Material node inside the group.
The solver writes the point attributes; the groups let you wire the fluids into your own node trees, drive the inputs and
build on the outputs.  Existing groups are kept (your edits survive); delete a group to have it rebuilt."""
import bpy

from . import gn
from .nodeutil import Builder, add_socket, fresh_group, inp, out


def _store(b, geo, name, value_socket, data_type, y):
    n = b.node('GeometryNodeStoreNamedAttribute', data_type=data_type, domain='POINT', y=y)
    inp(n, 'Name').default_value = name
    b.link(geo, inp(n, 'Geometry'))
    b.link(value_socket, inp(n, 'Value'))
    return out(n, 'Geometry')


def _named(b, name, data_type, y):
    n = b.node('GeometryNodeInputNamedAttribute', data_type=data_type, y=y)
    inp(n, 'Name').default_value = name
    return out(n, 'Attribute')


def build_fluid_group(name, fl=None):
    ng = fresh_group(name)
    add_socket(ng, "Geometry", 'NodeSocketGeometry', 'INPUT')
    add_socket(ng, "Mass", 'NodeSocketFloat', 'INPUT', 0.0, 0.0, desc="kg (0 = derived by the solver)")
    add_socket(ng, "Volume", 'NodeSocketFloat', 'INPUT', 0.0, 0.0, desc="m3")
    add_socket(ng, "Density", 'NodeSocketFloat', 'INPUT', 1000.0, 0.0, desc="kg/m3")
    add_socket(ng, "Temperature", 'NodeSocketFloat', 'INPUT', 293.15, 0.0, desc="K (initial)")
    add_socket(ng, "Velocity", 'NodeSocketVector', 'INPUT', (0.0, 0.0, 0.0), desc="m/s (initial)")
    add_socket(ng, "Viscosity", 'NodeSocketFloat', 'INPUT', 1e-3, 0.0, desc="Pa s")
    add_socket(ng, "Colour", 'NodeSocketColor', 'INPUT', (0.25, 0.55, 0.95, 1.0))
    add_socket(ng, "No Container", 'NodeSocketBool', 'INPUT', True, desc="Free fluid moving in the air")
    add_socket(ng, "Container", 'NodeSocketObject', 'INPUT', desc="Container object (when No Container is off)")
    add_socket(ng, "Scale", 'NodeSocketFloat', 'INPUT', 1.0, 0.0, desc="Point radius multiplier")
    add_socket(ng, "Geometry", 'NodeSocketGeometry', 'OUTPUT')
    add_socket(ng, "Free Fluid", 'NodeSocketBool', 'OUTPUT', desc="True when the fluid has no container")
    b = Builder(ng)
    gi, go = b.group_io()
    geo, _rad = gn._alive_filter(b, gi)
    b.link(geo, inp(go, 'Geometry'))
    b.link(out(gi, 'No Container'), inp(go, 'Free Fluid'))
    return ng


def fluid_group(fl):
    name = "Tessera Fluid %s" % gn.slug(fl.name)
    ng = bpy.data.node_groups.get(name)
    if ng is None or not ng.nodes:
        ng = build_fluid_group(name, fl)
    return ng


def set_fluid_inputs(mod, fl, fm):
    gn.set_modifier_inputs(mod, {
        "Mass": fl.d_mass or fl.mass, "Volume": fl.d_volume or fl.volume, "Density": fl.d_density or fl.density,
        "Temperature": fl.temperature, "Velocity": tuple(fl.velocity), "Viscosity": fl.visc,
        "Colour": (fl.color[0], fl.color[1], fl.color[2], 1.0), "No Container": bool(fl.no_container),
        "Scale": 1.0})


def build_instance_group(name="Tessera Mesh Per Particle"):
    ng = fresh_group(name)
    add_socket(ng, "Geometry", 'NodeSocketGeometry', 'INPUT')
    add_socket(ng, "Instance Object", 'NodeSocketObject', 'INPUT')
    add_socket(ng, "Use Object", 'NodeSocketBool', 'INPUT', False)
    add_socket(ng, "Instance Collection", 'NodeSocketCollection', 'INPUT')
    add_socket(ng, "Use Collection", 'NodeSocketBool', 'INPUT', False)
    add_socket(ng, "Scale", 'NodeSocketFloat', 'INPUT', 1.0, 0.0)
    add_socket(ng, "Align To Velocity", 'NodeSocketFloat', 'INPUT', 0.0, 0.0, 1.0, desc="0 = no alignment, 1 = the mesh's Z axis follows the velocity")
    add_socket(ng, "Geometry", 'NodeSocketGeometry', 'OUTPUT')
    b = Builder(ng)
    gi, go = b.group_io()
    geo, rad = gn._alive_filter(b, gi)
    ico = b.node('GeometryNodeMeshIcoSphere', y=-200)
    oi = b.node('GeometryNodeObjectInfo', y=-350, transform_space='ORIGINAL')
    b.link(out(gi, 'Instance Object'), inp(oi, 'Object'))
    inp(oi, 'As Instance').default_value = True
    ci = b.node('GeometryNodeCollectionInfo', y=-500)
    b.link(out(gi, 'Instance Collection'), inp(ci, 'Collection'))
    inp(ci, 'Separate Children').default_value = True
    inp(ci, 'Reset Children').default_value = True
    s1 = gn._sw_geo(b, out(gi, 'Use Object'), out(ico, 'Mesh'), out(oi, 'Geometry'), y=-250)
    s2 = gn._sw_geo(b, out(gi, 'Use Collection'), s1, out(ci, 'Instances'), y=-400)
    mul = b.math('MULTIPLY', rad, out(gi, 'Scale'), y=100)
    iop = b.node('GeometryNodeInstanceOnPoints', y=0)
    b.link(geo, inp(iop, 'Points'))
    b.link(s2, inp(iop, 'Instance'))
    b.link(out(mul, 'Value'), inp(iop, 'Scale'))
    try:
        vel = _named(b, 'fm_velocity', 'FLOAT_VECTOR', 300)
        al = b.node('FunctionNodeAlignRotationToVector', axis='Z' if False else 'Z', y=300)
        b.link(vel, inp(al, 'Vector'))
        b.link(out(gi, 'Align To Velocity'), inp(al, 'Factor'))
        b.link(out(al, 'Rotation'), inp(iop, 'Rotation'))
    except Exception:
        pass
    if iop.inputs.get('Pick Instance') is not None:
        b.link(out(gi, 'Use Collection'), inp(iop, 'Pick Instance'))
    sm = b.node('GeometryNodeSetMaterial', y=0)
    b.link(out(iop, 'Instances'), inp(sm, 'Geometry'))
    b.link(out(gn._matnode(b), 'Material'), inp(sm, 'Material'))
    b.link(out(sm, 'Geometry'), inp(go, 'Geometry'))
    return ng


def ensure_instance_group():
    ng = bpy.data.node_groups.get("Tessera Mesh Per Particle")
    if ng is None or not ng.nodes:
        ng = build_instance_group()
    return ng
