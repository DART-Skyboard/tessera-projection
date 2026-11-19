"""Tessera node groups: five small, reusable Geometry Nodes groups that carry the fluid features as plain geometry and attributes, so they
can be dropped into any node tree (yours, or another extension's) and chained with stock nodes.

  Tessera Fluid Source   a shape (closed mesh, surface, vertices or existing points) becomes a fluid: particle cloud, density volume, surface mesh
  Tessera Particle Cloud normalises the particle attributes (density, temperature, velocity, colour, fluid id, mass, viscosity, cohesion, mixing rate, radius)
  Tessera Thermal        heat input, reaction heat release and progress, pressure (ideal gas / hydrostatic), flame colour and emission
  Tessera Environment    gravity, ambient pressure (barometric), temperature, density and wind as values and per-point attributes
  Tessera Surface        density volume and welded surface mesh of a particle cloud, attributes transferred to the surface

Attribute names (POINT domain unless noted): fm_velocity (vector), fm_temperature, fm_pressure, fm_density, fm_mass, fm_color (colour), fm_fluid_id (int),
fm_viscosity, fm_cohesion, fm_mix_rate, af_radius, fm_reaction, fm_emission, fm_flame_color, fm_ambient_pressure, fm_ambient_density, fm_wind.
The groups are built from code (the single source of truth) and shipped as a library .blend (library/tessera_nodes.blend) that "Add Tessera node group"
appends from; when the library is missing or older than TESSERA_GN_VERSION the groups are built in place."""
import os

import bpy
from bpy.props import EnumProperty
from bpy.types import Menu, Operator, Panel

from . import gn
from .nodeutil import Builder, add_socket, fresh_group, inp, out

TESSERA_GN_VERSION = 3
LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "library", "tessera_nodes.blend")
R_GAS = 8.314462618

GROUPS = (("Tessera Fluid Source", "Shape to particles, volume and surface (all fluid inputs and outputs)"),
          ("Tessera Particle Cloud", "Particle attributes: density, temperature, velocity, colour, mass, viscosity, cohesion, mixing"),
          ("Tessera Surface", "Density volume and welded surface mesh of a particle cloud"),
          ("Tessera Thermal", "Heat, reaction heat release, pressure, flame colour and emission"),
          ("Tessera Environment", "Gravity, ambient pressure, temperature, density and wind"))
CORE_GROUPS = tuple(g[0] for g in GROUPS)       # shipped in the library blend
GROUP_NAMES = CORE_GROUPS
_ITEMS = []


def register_group(name, builder, description="", nested=()):
    """Let another module of the extension add a group to the Add menu / panel / ensure_group (built in code, not part of the library blend).
    builder() -> node group named `name`; nested = names of Tessera groups it uses (they are ensured first by the builder itself)."""
    global GROUPS
    if name not in BUILDERS:
        GROUPS = GROUPS + ((name, description),)
    BUILDERS[name] = builder
    if nested:
        NESTED[name] = tuple(nested)
    _refresh_items()


def _refresh_items():
    _ITEMS[:] = [(n, n.replace("Tessera ", ""), d) for n, d in GROUPS]


def _enum_items(self, context):
    if not _ITEMS:
        _refresh_items()
    return _ITEMS


# ----------------------------------------------------------------------------- helpers
def _named(b, name, dtype, y=0):
    n = b.node('GeometryNodeInputNamedAttribute', data_type=dtype, y=y)
    inp(n, 'Name').default_value = name
    return out(n, 'Attribute'), out(n, 'Exists')


def _switch(b, cond, false, true, kind, y=0):
    n = b.node('GeometryNodeSwitch', input_type=kind, y=y)
    b.link(cond, inp(n, 'Switch'))
    if isinstance(false, (int, float, tuple)):
        inp(n, 'False').default_value = false
    else:
        b.link(false, inp(n, 'False'))
    if isinstance(true, (int, float, tuple)):
        inp(n, 'True').default_value = true
    else:
        b.link(true, inp(n, 'True'))
    return out(n, 'Output')


_KINDS = {'FLOAT': 'FLOAT', 'INT': 'INT', 'FLOAT_VECTOR': 'VECTOR', 'FLOAT_COLOR': 'RGBA'}


def attr_or(b, name, dtype, fallback, y=0, alias=None):
    """Field: the named point attribute when the geometry has it, else `fallback` (a socket).  `alias` = a second attribute name tried next."""
    a, ex = _named(b, name, dtype, y)
    if alias:
        a2, ex2 = _named(b, alias, dtype, y - 60)
        fallback = _switch(b, ex2, fallback, a2, _KINDS[dtype], y)
    return _switch(b, ex, fallback, a, _KINDS[dtype], y)


def store(b, geo, name, value, dtype='FLOAT', y=0, domain='POINT'):
    n = b.node('GeometryNodeStoreNamedAttribute', data_type=dtype, domain=domain, y=y)
    inp(n, 'Name').default_value = name
    b.link(geo, inp(n, 'Geometry'))
    if isinstance(value, (int, float, tuple)):
        inp(n, 'Value').default_value = value
    else:
        b.link(value, inp(n, 'Value'))
    return out(n, 'Geometry')


def stat(b, geo, field, dtype='FLOAT', y=0):
    n = b.node('GeometryNodeAttributeStatistic', data_type=dtype, domain='POINT', y=y)
    b.link(geo, inp(n, 'Geometry'))
    b.link(field, inp(n, 'Attribute'))
    return n


def vmath(b, op, a=None, c=None, y=0, scalar=None):
    n = b.node('ShaderNodeVectorMath', operation=op, y=y)
    if a is not None:
        b.link(a, n.inputs[0])
    if c is not None:
        if isinstance(c, tuple):
            n.inputs[1].default_value = c
        else:
            b.link(c, n.inputs[1])
    if scalar is not None:
        b.link(scalar, n.inputs[3]) if not isinstance(scalar, (int, float)) else setattr(n.inputs[3], 'default_value', scalar)
    return n


def _res_mode(node, value='Size'):
    if hasattr(node, 'resolution_mode'):
        node.resolution_mode = {'Size': 'VOXEL_SIZE', 'Amount': 'VOXEL_AMOUNT', 'Grid': 'GRID'}[value]
    else:
        inp(node, 'Resolution Mode').default_value = value


def _dir(b, gi_gravity, y=0):
    """Unit vector along gravity (falls back to -Z for a zero vector)."""
    nrm = vmath(b, 'NORMALIZE', gi_gravity, y=y)
    return out(nrm, 'Vector')


def _height(b, pos, gdir, y=0):
    """Height above the origin against gravity: -dot(pos, g_hat)."""
    d = vmath(b, 'DOT_PRODUCT', pos, gdir, y=y)
    return out(b.math('MULTIPLY', out(d, 'Value'), -1.0, y=y), 'Value')


def _group_node(b, ng, label, y=0, x=None):
    n = b.node('GeometryNodeGroup', label=label, y=y, x=x)
    n.node_tree = ng
    return n


def _tag(ng):
    ng['tessera_gn_version'] = TESSERA_GN_VERSION
    ng.use_fake_user = True
    return ng


def _current(name):
    ng = bpy.data.node_groups.get(name)
    return ng if (ng is not None and ng.nodes and ng.get('tessera_gn_version', 0) >= TESSERA_GN_VERSION) else None


# ----------------------------------------------------------------------------- Tessera Environment
def build_environment(name="Tessera Environment"):
    ng = fresh_group(name)
    add_socket(ng, "Geometry", 'NodeSocketGeometry', 'INPUT')
    add_socket(ng, "Gravity", 'NodeSocketVector', 'INPUT', (0.0, 0.0, -9.80665), desc="m/s2")
    add_socket(ng, "Ambient Pressure", 'NodeSocketFloat', 'INPUT', 101325.0, 0.0, desc="Pa at the reference height")
    add_socket(ng, "Ambient Temperature", 'NodeSocketFloat', 'INPUT', 288.15, 1.0, desc="K")
    add_socket(ng, "Molar Mass", 'NodeSocketFloat', 'INPUT', 0.0289647, 1e-4, desc="kg/mol of the ambient gas (air 0.0290)")
    add_socket(ng, "Wind", 'NodeSocketVector', 'INPUT', (0.0, 0.0, 0.0), desc="m/s")
    add_socket(ng, "Reference Height", 'NodeSocketFloat', 'INPUT', 0.0, desc="Height (against gravity) where the ambient pressure is given")
    add_socket(ng, "Geometry", 'NodeSocketGeometry', 'OUTPUT')
    add_socket(ng, "Gravity", 'NodeSocketVector', 'OUTPUT')
    add_socket(ng, "Ambient Pressure", 'NodeSocketFloat', 'OUTPUT')
    add_socket(ng, "Ambient Temperature", 'NodeSocketFloat', 'OUTPUT')
    add_socket(ng, "Ambient Density", 'NodeSocketFloat', 'OUTPUT', desc="kg/m3 at the reference height")
    add_socket(ng, "Scale Height", 'NodeSocketFloat', 'OUTPUT', desc="m: R T / (M g)")
    add_socket(ng, "Wind", 'NodeSocketVector', 'OUTPUT')
    b = Builder(ng)
    gi, go = b.group_io()
    g = out(gi, 'Gravity')
    glen = vmath(b, 'LENGTH', g, y=300)
    gmag = b.math('MAXIMUM', out(glen, 'Value'), 1e-6, y=300)
    rt = b.math('MULTIPLY', out(gi, 'Ambient Temperature'), R_GAS, y=250)
    mg = b.math('MULTIPLY', out(gi, 'Molar Mass'), out(gmag, 'Value'), y=250)
    hs = b.math('DIVIDE', out(rt, 'Value'), out(mg, 'Value'), y=250)
    rho0a = b.math('MULTIPLY', out(gi, 'Ambient Pressure'), out(gi, 'Molar Mass'), y=200)
    rho0 = b.math('DIVIDE', out(rho0a, 'Value'), out(rt, 'Value'), y=200)
    pos = b.node('GeometryNodeInputPosition', y=100)
    gdir = _dir(b, g, y=100)
    h = _height(b, out(pos, 'Position'), gdir, y=100)
    dh = b.math('SUBTRACT', h, out(gi, 'Reference Height'), y=100)
    ex = b.math('DIVIDE', out(dh, 'Value'), out(hs, 'Value'), y=100)
    neg = b.math('MULTIPLY', out(ex, 'Value'), -1.0, y=100)
    fac = b.math('EXPONENT', out(neg, 'Value'), y=100)
    p_here = b.math('MULTIPLY', out(gi, 'Ambient Pressure'), out(fac, 'Value'), y=100)
    rho_here = b.math('MULTIPLY', rho0.outputs[0], out(fac, 'Value'), y=60)
    geo = store(b, out(gi, 'Geometry'), 'fm_ambient_pressure', out(p_here, 'Value'), 'FLOAT', y=0)
    geo = store(b, geo, 'fm_ambient_density', out(rho_here, 'Value'), 'FLOAT', y=-60)
    geo = store(b, geo, 'fm_wind', out(gi, 'Wind'), 'FLOAT_VECTOR', y=-120)
    geo = store(b, geo, 'fm_gravity', g, 'FLOAT_VECTOR', y=-180)
    b.link(geo, inp(go, 'Geometry'))
    for k in ('Gravity', 'Ambient Pressure', 'Ambient Temperature', 'Wind'):
        b.link(out(gi, k), inp(go, k))
    b.link(out(rho0, 'Value'), inp(go, 'Ambient Density'))
    b.link(out(hs, 'Value'), inp(go, 'Scale Height'))
    return _tag(ng)


# ----------------------------------------------------------------------------- Tessera Particle Cloud
def build_cloud(name="Tessera Particle Cloud"):
    ng = fresh_group(name)
    add_socket(ng, "Geometry", 'NodeSocketGeometry', 'INPUT')
    add_socket(ng, "Density", 'NodeSocketFloat', 'INPUT', 1000.0, 0.0, desc="kg/m3, used where the points have no fm_density")
    add_socket(ng, "Temperature", 'NodeSocketFloat', 'INPUT', 293.15, 0.0, desc="K, used where the points have no fm_temperature")
    add_socket(ng, "Velocity", 'NodeSocketVector', 'INPUT', (0.0, 0.0, 0.0), desc="m/s, used where the points have no fm_velocity")
    add_socket(ng, "Colour", 'NodeSocketColor', 'INPUT', (0.25, 0.55, 0.95, 1.0), desc="Used where the points have neither fm_color nor af_fcol (combustion)")
    add_socket(ng, "Fluid ID", 'NodeSocketInt', 'INPUT', 0, desc="Used where the points have no fm_fluid_id")
    add_socket(ng, "Mass", 'NodeSocketFloat', 'INPUT', 0.0, 0.0, desc="kg, total. 0 = density x sphere of the particle radius")
    add_socket(ng, "Viscosity", 'NodeSocketFloat', 'INPUT', 1e-3, 0.0, desc="Pa s")
    add_socket(ng, "Cohesion", 'NodeSocketFloat', 'INPUT', 0.07, 0.0, desc="Surface tension, N/m")
    add_socket(ng, "Mixing Rate", 'NodeSocketFloat', 'INPUT', 0.0, 0.0, desc="1/s")
    add_socket(ng, "Radius", 'NodeSocketFloat', 'INPUT', 0.02, 1e-5, desc="m, used where the points have no af_radius")
    add_socket(ng, "Scale", 'NodeSocketFloat', 'INPUT', 1.0, 0.0, desc="Particle radius multiplier")
    add_socket(ng, "Geometry", 'NodeSocketGeometry', 'OUTPUT')
    add_socket(ng, "Particle Count", 'NodeSocketFloat', 'OUTPUT')
    add_socket(ng, "Total Mass", 'NodeSocketFloat', 'OUTPUT', desc="kg")
    add_socket(ng, "Mean Density", 'NodeSocketFloat', 'OUTPUT')
    add_socket(ng, "Mean Temperature", 'NodeSocketFloat', 'OUTPUT')
    add_socket(ng, "Max Speed", 'NodeSocketFloat', 'OUTPUT')
    b = Builder(ng)
    gi, go = b.group_io()
    geo = out(gi, 'Geometry')
    rad = attr_or(b, 'af_radius', 'FLOAT', out(gi, 'Radius'), y=400)
    rad = out(b.math('MULTIPLY', rad, out(gi, 'Scale'), y=400), 'Value')
    geo = store(b, geo, 'af_radius', rad, 'FLOAT', y=0)
    one = b.node('ShaderNodeValue', y=300)
    one.outputs[0].default_value = 1.0
    count = stat(b, geo, out(one, 'Value'), y=300)
    cnt = b.math('MAXIMUM', out(count, 'Sum'), 1.0, y=300)
    geo = store(b, geo, 'fm_density', attr_or(b, 'fm_density', 'FLOAT', out(gi, 'Density'), y=350), 'FLOAT', y=0)
    geo = store(b, geo, 'fm_temperature', attr_or(b, 'fm_temperature', 'FLOAT', out(gi, 'Temperature'), y=330), 'FLOAT', y=0)
    geo = store(b, geo, 'fm_velocity', attr_or(b, 'fm_velocity', 'FLOAT_VECTOR', out(gi, 'Velocity'), y=310), 'FLOAT_VECTOR', y=0)
    geo = store(b, geo, 'fm_color', attr_or(b, 'fm_color', 'FLOAT_COLOR', out(gi, 'Colour'), y=290, alias='af_fcol'), 'FLOAT_COLOR', y=0)
    geo = store(b, geo, 'fm_fluid_id', attr_or(b, 'fm_fluid_id', 'INT', out(gi, 'Fluid ID'), y=270), 'INT', y=0)
    geo = store(b, geo, 'fm_viscosity', attr_or(b, 'fm_viscosity', 'FLOAT', out(gi, 'Viscosity'), y=250), 'FLOAT', y=0)
    geo = store(b, geo, 'fm_cohesion', attr_or(b, 'fm_cohesion', 'FLOAT', out(gi, 'Cohesion'), y=230), 'FLOAT', y=0)
    geo = store(b, geo, 'fm_mix_rate', attr_or(b, 'fm_mix_rate', 'FLOAT', out(gi, 'Mixing Rate'), y=210), 'FLOAT', y=0)
    # per-particle mass: Mass / count when a total is given, else density x 4/3 pi r^3
    dens, _e = _named(b, 'fm_density', 'FLOAT', y=190)
    r3 = b.math('POWER', rad, 3.0, y=190)
    sph = b.math('MULTIPLY', out(r3, 'Value'), 4.18879020479, y=190)
    derived = b.math('MULTIPLY', out(sph, 'Value'), dens, y=190)
    each = b.math('DIVIDE', out(gi, 'Mass'), out(cnt, 'Value'), y=170)
    has = b.math('GREATER_THAN', out(gi, 'Mass'), 0.0, y=170)
    pm = _switch(b, out(has, 'Value'), out(derived, 'Value'), out(each, 'Value'), 'FLOAT', y=170)
    geo = store(b, geo, 'fm_mass', attr_or(b, 'fm_mass', 'FLOAT', pm, y=150), 'FLOAT', y=0)
    b.link(geo, inp(go, 'Geometry'))
    b.link(out(cnt, 'Value'), inp(go, 'Particle Count'))
    b.link(out(stat(b, geo, _named(b, 'fm_mass', 'FLOAT', y=-100)[0], y=-100), 'Sum'), inp(go, 'Total Mass'))
    b.link(out(stat(b, geo, _named(b, 'fm_density', 'FLOAT', y=-160)[0], y=-160), 'Mean'), inp(go, 'Mean Density'))
    b.link(out(stat(b, geo, _named(b, 'fm_temperature', 'FLOAT', y=-220)[0], y=-220), 'Mean'), inp(go, 'Mean Temperature'))
    sp = vmath(b, 'LENGTH', _named(b, 'fm_velocity', 'FLOAT_VECTOR', y=-280)[0], y=-280)
    b.link(out(stat(b, geo, out(sp, 'Value'), y=-280), 'Max'), inp(go, 'Max Speed'))
    return _tag(ng)


# ----------------------------------------------------------------------------- Tessera Thermal
def build_thermal(name="Tessera Thermal"):
    ng = fresh_group(name)
    add_socket(ng, "Geometry", 'NodeSocketGeometry', 'INPUT')
    add_socket(ng, "Gas", 'NodeSocketBool', 'INPUT', False, desc="On: ideal gas pressure. Off: ambient + hydrostatic pressure")
    add_socket(ng, "Gravity", 'NodeSocketVector', 'INPUT', (0.0, 0.0, -9.80665), desc="m/s2")
    add_socket(ng, "Ambient Pressure", 'NodeSocketFloat', 'INPUT', 101325.0, 0.0, desc="Pa")
    add_socket(ng, "Molar Mass", 'NodeSocketFloat', 'INPUT', 0.0289647, 1e-4, desc="kg/mol of the gas (gas pressure)")
    add_socket(ng, "Specific Heat", 'NodeSocketFloat', 'INPUT', 4181.0, 1.0, desc="J/(kg K)")
    add_socket(ng, "Heat Added", 'NodeSocketFloat', 'INPUT', 0.0, desc="J/kg added to every particle (negative cools)")
    add_socket(ng, "Heat Release", 'NodeSocketFloat', 'INPUT', 0.0, 0.0, desc="J/kg released by a complete reaction (reactions / transmutation)")
    add_socket(ng, "Reaction Progress", 'NodeSocketFloat', 'INPUT', 0.0, 0.0, 1.0, desc="0..1; used where the points have no fm_reaction")
    add_socket(ng, "Ignition Temperature", 'NodeSocketFloat', 'INPUT', 700.0, 0.0, desc="K: flame colour and emission start here")
    add_socket(ng, "Flame Temperature", 'NodeSocketFloat', 'INPUT', 2200.0, 1.0, desc="K: flame colour reaches white here")
    add_socket(ng, "Geometry", 'NodeSocketGeometry', 'OUTPUT')
    add_socket(ng, "Mean Pressure", 'NodeSocketFloat', 'OUTPUT', desc="Pa")
    add_socket(ng, "Mean Temperature", 'NodeSocketFloat', 'OUTPUT', desc="K")
    add_socket(ng, "Max Temperature", 'NodeSocketFloat', 'OUTPUT', desc="K")
    b = Builder(ng)
    gi, go = b.group_io()
    geo = out(gi, 'Geometry')
    # reaction progress and temperature
    prog = attr_or(b, 'fm_reaction', 'FLOAT', out(gi, 'Reaction Progress'), y=400)
    geo = store(b, geo, 'fm_reaction', prog, 'FLOAT', y=0)
    t0, _e = _named(b, 'fm_temperature', 'FLOAT', y=380)
    rel = b.math('MULTIPLY', prog, out(gi, 'Heat Release'), y=380)
    q = b.math('ADD', out(rel, 'Value'), out(gi, 'Heat Added'), y=360)
    dt = b.math('DIVIDE', out(q, 'Value'), out(gi, 'Specific Heat'), y=360)
    tt = b.math('ADD', t0, out(dt, 'Value'), y=340)
    tt = b.math('MAXIMUM', out(tt, 'Value'), 1.0, y=340)
    geo = store(b, geo, 'fm_temperature', out(tt, 'Value'), 'FLOAT', y=0)
    # pressure
    t1, _e = _named(b, 'fm_temperature', 'FLOAT', y=300)
    rho, _e = _named(b, 'fm_density', 'FLOAT', y=280)
    rt = b.math('MULTIPLY', t1, R_GAS, y=260)
    p_gas = b.math('MULTIPLY', rho, out(rt, 'Value'), y=260)
    p_gas = b.math('DIVIDE', out(p_gas, 'Value'), out(gi, 'Molar Mass'), y=260)
    gmag = b.math('MAXIMUM', out(vmath(b, 'LENGTH', out(gi, 'Gravity'), y=240), 'Value'), 1e-6, y=240)
    gdir = _dir(b, out(gi, 'Gravity'), y=220)
    pos = b.node('GeometryNodeInputPosition', y=200)
    h = _height(b, out(pos, 'Position'), gdir, y=200)
    hmax = stat(b, geo, h, y=200)
    depth = b.math('SUBTRACT', out(hmax, 'Max'), h, y=180)
    hyd = b.math('MULTIPLY', rho, out(gmag, 'Value'), y=160)
    hyd = b.math('MULTIPLY', out(hyd, 'Value'), out(depth, 'Value'), y=160)
    p_liq = b.math('ADD', out(hyd, 'Value'), out(gi, 'Ambient Pressure'), y=160)
    pr = _switch(b, out(gi, 'Gas'), out(p_liq, 'Value'), out(p_gas, 'Value'), 'FLOAT', y=140)
    geo = store(b, geo, 'fm_pressure', pr, 'FLOAT', y=0)
    # flame colour and emission
    t2, _e = _named(b, 'fm_temperature', 'FLOAT', y=100)
    f = b.node('ShaderNodeMapRange', data_type='FLOAT', y=100)
    f.clamp = True
    b.link(t2, inp(f, 'Value'))
    b.link(out(gi, 'Ignition Temperature'), inp(f, 'From Min'))
    b.link(out(gi, 'Flame Temperature'), inp(f, 'From Max'))
    ramp = b.node('ShaderNodeValToRGB', y=100)
    cr = ramp.color_ramp
    cr.elements[0].position, cr.elements[0].color = 0.0, (0.0, 0.0, 0.0, 1.0)
    cr.elements[1].position, cr.elements[1].color = 1.0, (1.0, 0.95, 0.85, 1.0)
    for pos_, col in ((0.12, (0.45, 0.02, 0.0, 1.0)), (0.35, (1.0, 0.25, 0.02, 1.0)), (0.65, (1.0, 0.6, 0.12, 1.0))):
        e = cr.elements.new(pos_)
        e.color = col
    b.link(out(f, 'Result'), inp(ramp, 'Factor'))
    geo = store(b, geo, 'fm_flame_color', out(ramp, 'Color'), 'FLOAT_COLOR', y=0)
    geo = store(b, geo, 'fm_emission', out(f, 'Result'), 'FLOAT', y=0)
    b.link(geo, inp(go, 'Geometry'))
    b.link(out(stat(b, geo, _named(b, 'fm_pressure', 'FLOAT', y=-100)[0], y=-100), 'Mean'), inp(go, 'Mean Pressure'))
    st = stat(b, geo, _named(b, 'fm_temperature', 'FLOAT', y=-160)[0], y=-160)
    b.link(out(st, 'Mean'), inp(go, 'Mean Temperature'))
    b.link(out(st, 'Max'), inp(go, 'Max Temperature'))
    return _tag(ng)


# ----------------------------------------------------------------------------- Tessera Surface
def build_surface(name="Tessera Surface"):
    ng = fresh_group(name)
    add_socket(ng, "Geometry", 'NodeSocketGeometry', 'INPUT')
    add_socket(ng, "Voxel Size", 'NodeSocketFloat', 'INPUT', 0.04, 1e-4, desc="m")
    add_socket(ng, "Radius", 'NodeSocketFloat', 'INPUT', 0.03, 1e-5, desc="m, used where the points have no af_radius")
    add_socket(ng, "Radius Scale", 'NodeSocketFloat', 'INPUT', 2.0, 0.01, desc="Particle radius multiplier in the density volume")
    add_socket(ng, "Density", 'NodeSocketFloat', 'INPUT', 1.0, 0.0, desc="Density written into the volume grid")
    add_socket(ng, "Threshold", 'NodeSocketFloat', 'INPUT', 0.1, 0.0, desc="Iso level of the surface")
    add_socket(ng, "Smooth", 'NodeSocketBool', 'INPUT', True)
    add_socket(ng, "Geometry", 'NodeSocketGeometry', 'OUTPUT', desc="The surface mesh (alias of Surface)")
    add_socket(ng, "Surface", 'NodeSocketGeometry', 'OUTPUT', desc="Welded surface mesh with fm_color, fm_temperature, fm_velocity, fm_density, fm_pressure, fm_fluid_id")
    add_socket(ng, "Density Volume", 'NodeSocketGeometry', 'OUTPUT', desc="Volume grid 'density' of the particles")
    b = Builder(ng)
    gi, go = b.group_io()
    pts = out(gi, 'Geometry')
    rad = attr_or(b, 'af_radius', 'FLOAT', out(gi, 'Radius'), y=300)
    mul = b.math('MULTIPLY', rad, out(gi, 'Radius Scale'), y=300)
    p2v = b.node('GeometryNodePointsToVolume', y=100)
    _res_mode(p2v, 'Size')
    b.link(pts, inp(p2v, 'Points'))
    b.link(out(gi, 'Density'), inp(p2v, 'Density'))
    b.link(out(gi, 'Voxel Size'), inp(p2v, 'Voxel Size'))
    b.link(out(mul, 'Value'), inp(p2v, 'Radius'))
    v2m = b.node('GeometryNodeVolumeToMesh', y=100)
    _res_mode(v2m, 'Size')
    b.link(out(p2v, 'Volume'), inp(v2m, 'Volume'))
    b.link(out(gi, 'Voxel Size'), inp(v2m, 'Voxel Size'))
    b.link(out(gi, 'Threshold'), inp(v2m, 'Threshold'))
    sm = b.node('GeometryNodeSetShadeSmooth', y=100)
    b.link(out(v2m, 'Mesh'), inp(sm, 'Mesh'))
    b.link(out(gi, 'Smooth'), inp(sm, 'Shade Smooth'))
    surf = out(sm, 'Mesh')
    pos = b.node('GeometryNodeInputPosition', y=0)
    near = b.node('GeometryNodeSampleNearest', domain='POINT', y=0)
    b.link(pts, inp(near, 'Geometry'))
    b.link(out(pos, 'Position'), inp(near, 'Sample Position'))
    for nm, dt, kind in (('fm_color', 'FLOAT_COLOR', 'FLOAT_COLOR'), ('fm_temperature', 'FLOAT', 'FLOAT'), ('fm_velocity', 'FLOAT_VECTOR', 'FLOAT_VECTOR'),
                         ('fm_density', 'FLOAT', 'FLOAT'), ('fm_pressure', 'FLOAT', 'FLOAT'), ('fm_fluid_id', 'INT', 'INT')):
        si = b.node('GeometryNodeSampleIndex', data_type=dt, domain='POINT', y=-80 * (1 + len(ng.nodes) % 7))
        b.link(pts, inp(si, 'Geometry'))
        b.link(_named(b, nm, dt)[0], inp(si, 'Value'))
        b.link(out(near, 'Index'), inp(si, 'Index'))
        surf = store(b, surf, nm, out(si, 'Value'), kind, domain='POINT')
    smat = b.node('GeometryNodeSetMaterial', y=0)
    b.link(surf, inp(smat, 'Geometry'))
    b.link(out(gn._matnode(b), 'Material'), inp(smat, 'Material'))
    b.link(out(smat, 'Geometry'), inp(go, 'Geometry'))
    b.link(out(smat, 'Geometry'), inp(go, 'Surface'))
    b.link(out(p2v, 'Volume'), inp(go, 'Density Volume'))
    return _tag(ng)


# ----------------------------------------------------------------------------- Tessera Fluid Source
SOURCE_INPUTS = (
    # (name, kind, default, lo, hi, description)
    ("Spacing", 'NodeSocketFloat', 0.05, 1e-3, None, "m between particles (volume lattice) / per side (surface)"),
    ("Spawn Mode", 'NodeSocketInt', 0, 0, 3, "0 fill the shape's volume, 1 on its surface, 2 at its vertices, 3 the shape is already a point cloud"),
    ("Seed", 'NodeSocketInt', 0, None, None, "Surface spawn seed"),
    ("Max Particles", 'NodeSocketInt', 100000, 1, None, "Hard cap"),
    ("Density", 'NodeSocketFloat', 1000.0, 0.0, None, "kg/m3"),
    ("Mass", 'NodeSocketFloat', 0.0, 0.0, None, "kg, total (0 = density x particle sphere)"),
    ("Temperature", 'NodeSocketFloat', 293.15, 0.0, None, "K"),
    ("Velocity", 'NodeSocketVector', (0.0, 0.0, 0.0), None, None, "m/s"),
    ("Viscosity", 'NodeSocketFloat', 1e-3, 0.0, None, "Pa s"),
    ("Cohesion", 'NodeSocketFloat', 0.07, 0.0, None, "Surface tension, N/m"),
    ("Mixing Rate", 'NodeSocketFloat', 0.0, 0.0, None, "1/s"),
    ("Colour", 'NodeSocketColor', (0.25, 0.55, 0.95, 1.0), None, None, ""),
    ("Fluid ID", 'NodeSocketInt', 0, None, None, ""),
    ("Gas", 'NodeSocketBool', False, None, None, "Ideal gas pressure instead of hydrostatic"),
    ("Specific Heat", 'NodeSocketFloat', 4181.0, 1.0, None, "J/(kg K)"),
    ("Heat Added", 'NodeSocketFloat', 0.0, None, None, "J/kg"),
    ("Heat Release", 'NodeSocketFloat', 0.0, 0.0, None, "J/kg of a complete reaction"),
    ("Reaction Progress", 'NodeSocketFloat', 0.0, 0.0, 1.0, "0..1"),
    ("Gravity", 'NodeSocketVector', (0.0, 0.0, -9.80665), None, None, "Environment, m/s2"),
    ("Ambient Pressure", 'NodeSocketFloat', 101325.0, 0.0, None, "Environment, Pa"),
    ("Ambient Temperature", 'NodeSocketFloat', 288.15, 1.0, None, "Environment, K"),
    ("Scale", 'NodeSocketFloat', 1.0, 0.0, None, "Particle radius multiplier"),
    ("Render Mode", 'NodeSocketInt', 0, 0, 2, "0 points, 1 density volume, 2 surface mesh (output Render)"),
    ("Voxel Size", 'NodeSocketFloat', 0.0, 0.0, None, "Volume and surface voxel (0 = Spacing x 0.8)"),
    ("Build Volume And Surface", 'NodeSocketBool', True, None, None, "Off: Density Volume and Surface are only built when the Render Mode asks for them (saves time)"),
)


def _link_by_name(b, node, gi, names):
    for n in names:
        try:
            b.link(out(gi, n), inp(node, n))
        except KeyError:
            pass


def build_source(name="Tessera Fluid Source"):
    cloud_g, therm_g, surf_g = ensure_group("Tessera Particle Cloud"), ensure_group("Tessera Thermal"), ensure_group("Tessera Surface")
    ng = fresh_group(name)
    add_socket(ng, "Shape", 'NodeSocketGeometry', 'INPUT', desc="Closed mesh (fill), any mesh (surface / vertices) or points")
    for nm, kind, default, lo, hi, desc in SOURCE_INPUTS:
        add_socket(ng, nm, kind, 'INPUT', default, lo, hi, desc)
    add_socket(ng, "Particles", 'NodeSocketGeometry', 'OUTPUT', desc="Points with the fm_* attributes (alias of Geometry)")
    add_socket(ng, "Geometry", 'NodeSocketGeometry', 'OUTPUT', desc="Points with the fm_* attributes")
    add_socket(ng, "Density Volume", 'NodeSocketGeometry', 'OUTPUT')
    add_socket(ng, "Surface", 'NodeSocketGeometry', 'OUTPUT')
    add_socket(ng, "Render", 'NodeSocketGeometry', 'OUTPUT', desc="Points, density volume or surface mesh by Render Mode")
    add_socket(ng, "Particle Count", 'NodeSocketFloat', 'OUTPUT')
    add_socket(ng, "Total Mass", 'NodeSocketFloat', 'OUTPUT', desc="kg")
    add_socket(ng, "Mean Temperature", 'NodeSocketFloat', 'OUTPUT')
    add_socket(ng, "Mean Pressure", 'NodeSocketFloat', 'OUTPUT')
    add_socket(ng, "Max Speed", 'NodeSocketFloat', 'OUTPUT')
    b = Builder(ng)
    gi, go = b.group_io()
    shape = out(gi, 'Shape')
    sp = out(gi, 'Spacing')
    # volume: mesh -> density grid -> lattice points
    voxel = b.math('MULTIPLY', sp, 0.5, y=500)
    m2v = b.node('GeometryNodeMeshToVolume', y=500)
    _res_mode(m2v, 'Size')
    b.link(shape, inp(m2v, 'Mesh'))
    b.link(out(voxel, 'Value'), inp(m2v, 'Voxel Size'))
    inp(m2v, 'Interior Band Width').default_value = 1000.0
    dpv = b.node('GeometryNodeDistributePointsInVolume', y=500)
    try:
        inp(dpv, 'Mode').default_value = 'Grid'
    except Exception:
        pass
    b.link(out(m2v, 'Volume'), inp(dpv, 'Volume'))
    cmb = b.node('ShaderNodeCombineXYZ', y=450)
    for i in range(3):
        b.link(sp, cmb.inputs[i])
    b.link(out(cmb, 'Vector'), inp(dpv, 'Spacing'))
    inp(dpv, 'Threshold').default_value = 0.1
    # surface
    area = b.math('MULTIPLY', sp, sp, y=350)
    dens = b.math('DIVIDE', 1.0, out(area, 'Value'), y=350)
    dpf = b.node('GeometryNodeDistributePointsOnFaces', y=350)
    b.link(shape, inp(dpf, 'Mesh'))
    b.link(out(dens, 'Value'), inp(dpf, 'Density'))
    b.link(out(gi, 'Seed'), inp(dpf, 'Seed'))
    # vertices
    m2p = b.node('GeometryNodeMeshToPoints', mode='VERTICES', y=250)
    b.link(shape, inp(m2p, 'Mesh'))
    # mode switch
    def eq(v, y):
        c = b.node('FunctionNodeCompare', data_type='INT', operation='EQUAL', y=y)
        ii = [s for s in c.inputs if s.type == 'INT' and s.enabled]
        b.link(out(gi, 'Spawn Mode'), ii[0])
        ii[1].default_value = v
        return out(c, 'Result')
    s = gn._sw_geo(b, eq(2, 200), out(dpv, 'Points'), out(m2p, 'Points'), y=200)
    s = gn._sw_geo(b, eq(1, 150), s, out(dpf, 'Points'), y=150)
    s = gn._sw_geo(b, eq(3, 100), s, shape, y=100)
    # cap
    idx = b.node('GeometryNodeInputIndex', y=60)
    over = b.node('FunctionNodeCompare', data_type='INT', operation='GREATER_EQUAL', y=60)
    ii = [x for x in over.inputs if x.type == 'INT' and x.enabled]
    b.link(out(idx, 'Index'), ii[0])
    b.link(out(gi, 'Max Particles'), ii[1])
    dele = b.node('GeometryNodeDeleteGeometry', y=60)
    b.link(s, inp(dele, 'Geometry'))
    b.link(out(over, 'Result'), inp(dele, 'Selection'))
    pts = out(dele, 'Geometry')
    # radius = Spacing / 2
    rad = b.math('MULTIPLY', sp, 0.5, y=0)
    cl = _group_node(b, cloud_g, "Particle Cloud", y=0)
    b.link(pts, inp(cl, 'Geometry'))
    _link_by_name(b, cl, gi, ("Density", "Temperature", "Velocity", "Colour", "Fluid ID", "Mass", "Viscosity", "Cohesion", "Mixing Rate", "Scale"))
    b.link(out(rad, 'Value'), inp(cl, 'Radius'))
    th = _group_node(b, therm_g, "Thermal", y=0)
    b.link(out(cl, 'Geometry'), inp(th, 'Geometry'))
    _link_by_name(b, th, gi, ("Gas", "Gravity", "Ambient Pressure", "Specific Heat", "Heat Added", "Heat Release", "Reaction Progress"))
    # surface / volume (lazy: only evaluated when used)
    ms = b.math('MULTIPLY', sp, 0.8, y=-150)
    vox = _switch(b, b.math('GREATER_THAN', out(gi, 'Voxel Size'), 0.0, y=-150).outputs[0], out(ms, 'Value'), out(gi, 'Voxel Size'), 'FLOAT', y=-150)
    sf = _group_node(b, surf_g, "Surface", y=-200)
    b.link(out(th, 'Geometry'), inp(sf, 'Geometry'))
    b.link(vox, inp(sf, 'Voxel Size'))
    b.link(out(rad, 'Value'), inp(sf, 'Radius'))
    rm = out(gi, 'Render Mode')
    def rm_is(v, y):
        c = b.node('FunctionNodeCompare', data_type='INT', operation='EQUAL', y=y)
        ii = [x for x in c.inputs if x.type == 'INT' and x.enabled]
        b.link(rm, ii[0]); ii[1].default_value = v
        return out(c, 'Result')
    need = b.node('FunctionNodeBooleanMath', operation='OR', y=-300)
    gt = b.node('FunctionNodeCompare', data_type='INT', operation='GREATER_THAN', y=-300)
    ii = [x for x in gt.inputs if x.type == 'INT' and x.enabled]
    b.link(rm, ii[0]); ii[1].default_value = 0
    b.link(out(gi, 'Build Volume And Surface'), need.inputs[0]); b.link(out(gt, 'Result'), need.inputs[1])
    empty = b.node('GeometryNodeJoinGeometry', y=-400)       # nothing connected = empty geometry
    surf_out = _sw_geo2(b, out(need, 'Boolean'), out(empty, 'Geometry'), out(sf, 'Surface'), -350)
    vol_out = _sw_geo2(b, out(need, 'Boolean'), out(empty, 'Geometry'), out(sf, 'Density Volume'), -420)
    r = _sw_geo2(b, rm_is(2, -480), out(th, 'Geometry'), out(sf, 'Surface'), -480)
    r = _sw_geo2(b, rm_is(1, -520), r, out(sf, 'Density Volume'), -520)
    b.link(out(th, 'Geometry'), inp(go, 'Particles'))
    b.link(out(th, 'Geometry'), inp(go, 'Geometry'))
    b.link(vol_out, inp(go, 'Density Volume'))
    b.link(surf_out, inp(go, 'Surface'))
    b.link(r, inp(go, 'Render'))
    b.link(out(cl, 'Particle Count'), inp(go, 'Particle Count'))
    b.link(out(cl, 'Total Mass'), inp(go, 'Total Mass'))
    b.link(out(th, 'Mean Temperature'), inp(go, 'Mean Temperature'))
    b.link(out(th, 'Mean Pressure'), inp(go, 'Mean Pressure'))
    b.link(out(cl, 'Max Speed'), inp(go, 'Max Speed'))
    return _tag(ng)


def _sw_geo2(b, cond, false_geo, true_geo, y=0):
    return gn._sw_geo(b, cond, false_geo, true_geo, y=y)


# ----------------------------------------------------------------------------- registry / library
BUILDERS = {"Tessera Environment": build_environment, "Tessera Particle Cloud": build_cloud, "Tessera Thermal": build_thermal,
            "Tessera Surface": build_surface, "Tessera Fluid Source": build_source}


def build_group(name):
    return BUILDERS[name]()


def ensure_group(name):
    """The node group `name` in this file: kept when current, else appended from the library, else built."""
    ng = _current(name)
    if ng is not None:
        return ng
    if append_from_library(name):
        ng = _current(name)
        if ng is not None:
            return ng
    return build_group(name)


def append_from_library(name):
    """Append `name` (and the groups it nests) from library/tessera_nodes.blend; False when the library is missing or older than this code."""
    if not os.path.isfile(LIB_PATH):
        return False
    try:
        with bpy.data.libraries.load(LIB_PATH, link=False) as (src, dst):
            if name not in src.node_groups:
                return False
            dst.node_groups = [n for n in src.node_groups if n == name or n in NESTED.get(name, ())]
        loaded = [g for g in dst.node_groups if g is not None]
        if not loaded or any(g.get('tessera_gn_version', 0) < TESSERA_GN_VERSION for g in loaded):
            for g in loaded:
                bpy.data.node_groups.remove(g)
            return False
        # a re-append gives "Name.001" next to an existing older group: point the new one at the canonical name
        for g in loaded:
            base = g.name.rsplit('.', 1)[0] if g.name.rsplit('.', 1)[-1].isdigit() else g.name
            if base != g.name:
                old = bpy.data.node_groups.get(base)
                if old is not None:
                    old.user_remap(g)
                    bpy.data.node_groups.remove(old)
                g.name = base
        return True
    except Exception:
        return False


NESTED = {"Tessera Fluid Source": ("Tessera Particle Cloud", "Tessera Thermal", "Tessera Surface")}


def build_library(path=LIB_PATH):
    """Write library/tessera_nodes.blend with the five groups (run from tools / the test)."""
    for n in CORE_GROUPS:
        g = bpy.data.node_groups.get(n)
        if g is not None:
            bpy.data.node_groups.remove(g)
    for n in ("Tessera Environment", "Tessera Particle Cloud", "Tessera Thermal", "Tessera Surface", "Tessera Fluid Source"):
        build_group(n)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bpy.data.libraries.write(path, {bpy.data.node_groups[n] for n in CORE_GROUPS}, fake_user=True, compress=True)


def add_group_node(context, name):
    """Add a node using `name` to the edited Geometry Nodes tree; returns the node or None."""
    tree = getattr(context.space_data, 'edit_tree', None) if context.space_data is not None else None
    if tree is None or tree.bl_idname != 'GeometryNodeTree':
        return None
    ng = ensure_group(name)
    n = tree.nodes.new('GeometryNodeGroup')
    n.node_tree = ng
    n.label = name.replace("Tessera ", "")
    for o in tree.nodes:
        o.select = False
    n.select = True
    tree.nodes.active = n
    return n


# ----------------------------------------------------------------------------- UI
class AF_OT_add_tessera_group(Operator):
    bl_idname = "arcflow.add_tessera_group"
    bl_label = "Add Tessera node group"
    bl_description = "Add a Tessera node group (Fluid Source, Particle Cloud, Surface, Thermal, Environment) to the Geometry Nodes tree"
    bl_options = {'REGISTER', 'UNDO'}
    group: EnumProperty(name="Group", items=_enum_items)

    @classmethod
    def poll(cls, context):
        sd = context.space_data
        return sd is not None and getattr(sd, 'type', '') == 'NODE_EDITOR' and getattr(sd, 'tree_type', '') == 'GeometryNodeTree' and getattr(sd, 'edit_tree', None) is not None

    def invoke(self, context, event):
        n = add_group_node(context, self.group)
        if n is None:
            self.report({'WARNING'}, "Open a Geometry Nodes tree first")
            return {'CANCELLED'}
        try:
            n.location = context.space_data.cursor_location
        except Exception:
            pass
        try:
            bpy.ops.node.translate_attach('INVOKE_DEFAULT')
        except Exception:
            pass
        return {'FINISHED'}

    def execute(self, context):
        return {'FINISHED'} if add_group_node(context, self.group) is not None else {'CANCELLED'}


class AF_OT_gn_modifier(Operator):
    bl_idname = "arcflow.add_tessera_modifier"
    bl_label = "Add to selected object"
    bl_description = "Add a Geometry Nodes modifier with the chosen Tessera group to the active object"
    bl_options = {'REGISTER', 'UNDO'}
    group: EnumProperty(name="Group", items=_enum_items)

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type in {'MESH', 'POINTCLOUD', 'CURVE'}

    def execute(self, context):
        ng = ensure_group(self.group)
        m = context.object.modifiers.new(self.group, 'NODES')
        m.node_group = ng
        return {'FINISHED'}


class AF_MT_gn_add(Menu):
    bl_idname = "AF_MT_gn_add"
    bl_label = "Add Tessera node group"

    def draw(self, context):
        lay = self.layout
        lay.operator_context = 'INVOKE_REGION_WIN'
        for n, d in GROUPS:
            op = lay.operator("arcflow.add_tessera_group", text=n.replace("Tessera ", ""))
            op.group = n


def _add_menu_entry(self, context):
    self.layout.separator()
    self.layout.menu("AF_MT_gn_add", icon='NODETREE')


class _NodesPanel:
    bl_label = "Tessera node groups"
    bl_category = "Tessera"

    def draw(self, context):
        lay = self.layout
        sd = context.space_data
        in_gn = sd is not None and getattr(sd, 'type', '') == 'NODE_EDITOR'
        col = lay.column(align=True)
        for n, d in GROUPS:
            if in_gn:
                col.operator("arcflow.add_tessera_group", text="Add " + n.replace("Tessera ", ""), icon='NODETREE').group = n
            else:
                col.operator("arcflow.add_tessera_modifier", text="Modifier: " + n.replace("Tessera ", ""), icon='GEOMETRY_NODES').group = n


class AF_PT_gn_nodes_editor(_NodesPanel, Panel):
    bl_idname = "AF_PT_gn_nodes_editor"
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'UI'

    @classmethod
    def poll(cls, context):
        return getattr(context.space_data, 'tree_type', '') == 'GeometryNodeTree'


class AF_PT_gn_nodes_view3d(_NodesPanel, Panel):
    bl_idname = "AF_PT_gn_nodes_view3d"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_options = {'DEFAULT_CLOSED'}


CLASSES = (AF_OT_add_tessera_group, AF_OT_gn_modifier, AF_MT_gn_add, AF_PT_gn_nodes_editor, AF_PT_gn_nodes_view3d)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    for mt in ("NODE_MT_geometry_node_add_all",):
        if hasattr(bpy.types, mt):
            getattr(bpy.types, mt).append(_add_menu_entry)


def unregister():
    for mt in ("NODE_MT_geometry_node_add_all",):
        if hasattr(bpy.types, mt):
            try:
                getattr(bpy.types, mt).remove(_add_menu_entry)
            except Exception:
                pass
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
