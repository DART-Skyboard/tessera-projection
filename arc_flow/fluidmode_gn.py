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
from . import spawn as SP
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


FLUID_VERSION = 4


def build_fluid_group(name, fl=None):
    """Per-fluid group: Tessera Particle Cloud -> Tessera Thermal (+ Tessera Environment attributes) with every fluid feature as a socket;
    Surface / Density Volume are built lazily (Render Mode or Build Volume And Surface)."""
    from . import tessera_nodes as TN
    cloud_g, therm_g = TN.ensure_group("Tessera Particle Cloud"), TN.ensure_group("Tessera Thermal")
    surf_g, env_g = TN.ensure_group("Tessera Surface"), TN.ensure_group("Tessera Environment")
    ng = fresh_group(name)
    add_socket(ng, "Geometry", 'NodeSocketGeometry', 'INPUT')
    add_socket(ng, "Mass", 'NodeSocketFloat', 'INPUT', 0.0, 0.0, desc="kg (0 = derived by the solver)")
    add_socket(ng, "Volume", 'NodeSocketFloat', 'INPUT', 0.0, 0.0, desc="m3 (mirrors the panel)")
    add_socket(ng, "Density", 'NodeSocketFloat', 'INPUT', 1000.0, 0.0, desc="kg/m3")
    add_socket(ng, "Temperature", 'NodeSocketFloat', 'INPUT', 293.15, 0.0, desc="K (initial)")
    add_socket(ng, "Velocity", 'NodeSocketVector', 'INPUT', (0.0, 0.0, 0.0), desc="m/s (initial)")
    add_socket(ng, "Viscosity", 'NodeSocketFloat', 'INPUT', 1e-3, 0.0, desc="Pa s")
    add_socket(ng, "Cohesion", 'NodeSocketFloat', 'INPUT', 0.07, 0.0, desc="Surface tension, N/m")
    add_socket(ng, "Mixing Rate", 'NodeSocketFloat', 'INPUT', 0.0, 0.0, desc="1/s")
    add_socket(ng, "Colour", 'NodeSocketColor', 'INPUT', (0.25, 0.55, 0.95, 1.0))
    add_socket(ng, "Fluid ID", 'NodeSocketInt', 'INPUT', 0)
    add_socket(ng, "Gas", 'NodeSocketBool', 'INPUT', False, desc="Gas phase: ideal gas pressure")
    add_socket(ng, "No Container", 'NodeSocketBool', 'INPUT', True, desc="Free fluid moving in the air")
    add_socket(ng, "Container", 'NodeSocketObject', 'INPUT', desc="Container object (when No Container is off; mirrors the panel)")
    add_socket(ng, "Shape", 'NodeSocketObject', 'INPUT', desc="Shape object (mirrors the panel; use Tessera Fluid Source to spawn from a shape inside a node tree)")
    add_socket(ng, "Fill", 'NodeSocketFloat', 'INPUT', 50.0, 0.0, 100.0, desc="Percent of the container (mirrors the panel)")
    add_socket(ng, "Spawn Mode", 'NodeSocketInt', 'INPUT', 0, 0, 4, desc="Panel Spawn mode: 0 auto, 1 volume, 2 surface, 3 vertices / points, 4 particle system (mirrors the panel; Tessera Fluid Spawn does it in nodes)")
    add_socket(ng, "Spawn Density", 'NodeSocketFloat', 'INPUT', 1.0, 0.0, 1.0, desc="Fraction of the generated particles kept (mirrors the panel)")
    add_socket(ng, "Use Primitive", 'NodeSocketBool', 'INPUT', False, desc="The Shape is a primitive (mirrors the panel)")
    add_socket(ng, "Primitive", 'NodeSocketInt', 'INPUT', 0, 0, 5, desc="0 sphere, 1 cube, 2 cylinder, 3 cone, 4 torus, 5 plane (mirrors the panel)")
    add_socket(ng, "Primitive Size", 'NodeSocketFloat', 'INPUT', 0.1, 1e-3, desc="m (mirrors the panel)")
    add_socket(ng, "Gas Parcels", 'NodeSocketInt', 'INPUT', 2000, 8, desc="Gas parcel count (mirrors the panel)")
    add_socket(ng, "Specific Heat", 'NodeSocketFloat', 'INPUT', 4181.0, 1.0, desc="J/(kg K)")
    add_socket(ng, "Heat Added", 'NodeSocketFloat', 'INPUT', 0.0, desc="J/kg added to every particle")
    add_socket(ng, "Heat Release", 'NodeSocketFloat', 'INPUT', 0.0, 0.0, desc="J/kg of a complete reaction")
    add_socket(ng, "Reaction Progress", 'NodeSocketFloat', 'INPUT', 0.0, 0.0, 1.0, desc="0..1 (used where the points have no fm_reaction)")
    add_socket(ng, "Gravity", 'NodeSocketVector', 'INPUT', (0.0, 0.0, -9.80665), desc="Environment, m/s2")
    add_socket(ng, "Ambient Pressure", 'NodeSocketFloat', 'INPUT', 101325.0, 0.0, desc="Environment, Pa")
    add_socket(ng, "Ambient Temperature", 'NodeSocketFloat', 'INPUT', 288.15, 1.0, desc="Environment, K")
    add_socket(ng, "Molar Mass", 'NodeSocketFloat', 'INPUT', 0.0289647, 1e-4, desc="Environment gas / fluid, kg/mol")
    add_socket(ng, "Wind", 'NodeSocketVector', 'INPUT', (0.0, 0.0, 0.0), desc="Environment, m/s")
    add_socket(ng, "Render Mode", 'NodeSocketInt', 'INPUT', 0, 0, 2, desc="0 points, 1 density volume, 2 surface mesh (output Render)")
    add_socket(ng, "Voxel Size", 'NodeSocketFloat', 'INPUT', 0.04, 1e-4, desc="Density volume and surface voxel")
    add_socket(ng, "Radius Scale", 'NodeSocketFloat', 'INPUT', 2.0, 0.01, desc="Particle radius multiplier in the density volume")
    add_socket(ng, "Build Volume And Surface", 'NodeSocketBool', 'INPUT', False, desc="Build Density Volume and Surface outputs (costly; the Render Mode builds what it needs)")
    add_socket(ng, "Scale", 'NodeSocketFloat', 'INPUT', 1.0, 0.0, desc="Point radius multiplier")
    add_socket(ng, "Geometry", 'NodeSocketGeometry', 'OUTPUT', desc="Points with fm_velocity, fm_temperature, fm_pressure, fm_density, fm_mass, fm_color, fm_fluid_id, ...")
    add_socket(ng, "Free Fluid", 'NodeSocketBool', 'OUTPUT', desc="True when the fluid has no container")
    add_socket(ng, "Density Volume", 'NodeSocketGeometry', 'OUTPUT')
    add_socket(ng, "Surface", 'NodeSocketGeometry', 'OUTPUT')
    add_socket(ng, "Render", 'NodeSocketGeometry', 'OUTPUT', desc="Points, density volume or surface by Render Mode")
    for k in ("Particle Count", "Total Mass", "Mean Density", "Mean Temperature", "Max Temperature", "Mean Pressure", "Max Speed"):
        add_socket(ng, k, 'NodeSocketFloat', 'OUTPUT')
    b = Builder(ng)
    gi, go = b.group_io()
    geo, rad = gn._alive_filter(b, gi)
    cl = TN._group_node(b, cloud_g, "Particle Cloud", y=0)
    b.link(geo, inp(cl, 'Geometry'))
    TN._link_by_name(b, cl, gi, ("Density", "Temperature", "Velocity", "Colour", "Fluid ID", "Mass", "Viscosity", "Cohesion", "Mixing Rate", "Scale"))
    th = TN._group_node(b, therm_g, "Thermal", y=0)
    b.link(out(cl, 'Geometry'), inp(th, 'Geometry'))
    TN._link_by_name(b, th, gi, ("Gas", "Gravity", "Ambient Pressure", "Molar Mass", "Specific Heat", "Heat Added", "Heat Release", "Reaction Progress"))
    ev = TN._group_node(b, env_g, "Environment", y=0)
    b.link(out(th, 'Geometry'), inp(ev, 'Geometry'))
    TN._link_by_name(b, ev, gi, ("Gravity", "Ambient Pressure", "Ambient Temperature", "Molar Mass", "Wind"))
    pts = out(ev, 'Geometry')
    sf = TN._group_node(b, surf_g, "Surface", y=-200)
    b.link(pts, inp(sf, 'Geometry'))
    TN._link_by_name(b, sf, gi, ("Voxel Size", "Radius Scale"))
    rm = out(gi, 'Render Mode')
    gt = b.node('FunctionNodeCompare', data_type='INT', operation='GREATER_THAN', y=-300)
    ii = [x for x in gt.inputs if x.type == 'INT' and x.enabled]
    b.link(rm, ii[0]); ii[1].default_value = 0
    need = b.node('FunctionNodeBooleanMath', operation='OR', y=-300)
    b.link(out(gi, 'Build Volume And Surface'), need.inputs[0]); b.link(out(gt, 'Result'), need.inputs[1])
    empty = b.node('GeometryNodeJoinGeometry', y=-400)
    surf = gn._sw_geo(b, out(need, 'Boolean'), out(empty, 'Geometry'), out(sf, 'Surface'), y=-350)
    vol = gn._sw_geo(b, out(need, 'Boolean'), out(empty, 'Geometry'), out(sf, 'Density Volume'), y=-420)

    def is_mode(v, y):
        c = b.node('FunctionNodeCompare', data_type='INT', operation='EQUAL', y=y)
        jj = [x for x in c.inputs if x.type == 'INT' and x.enabled]
        b.link(rm, jj[0]); jj[1].default_value = v
        return out(c, 'Result')
    r = gn._sw_geo(b, is_mode(2, -480), pts, out(sf, 'Surface'), y=-480)
    r = gn._sw_geo(b, is_mode(1, -520), r, out(sf, 'Density Volume'), y=-520)
    b.link(pts, inp(go, 'Geometry'))
    b.link(out(gi, 'No Container'), inp(go, 'Free Fluid'))
    b.link(vol, inp(go, 'Density Volume'))
    b.link(surf, inp(go, 'Surface'))
    b.link(r, inp(go, 'Render'))
    for k in ("Particle Count", "Total Mass", "Mean Density", "Max Speed"):
        b.link(out(cl, k), inp(go, k))
    for k in ("Mean Temperature", "Max Temperature", "Mean Pressure"):
        b.link(out(th, k), inp(go, k))
    ng['fluid_group_version'] = FLUID_VERSION
    ng.use_fake_user = True
    return ng


def fluid_group(fl):
    name = "Tessera Fluid %s" % gn.slug(fl.name)
    ng = bpy.data.node_groups.get(name)
    if ng is None or not ng.nodes or ng.get('fluid_group_version', 0) < FLUID_VERSION:
        ng = build_fluid_group(name, fl)
    return ng


def _env_values():
    """Gravity, ambient pressure / temperature, molar mass and wind of the running Fluid mode solver (None when there is none)."""
    try:
        from . import fluidmode_ops as FO
        e = FO.RT.solver.env
        return {"Voxel Size": max(float(FO.RT.solver.dx) * 0.9, 1e-3), "Gravity": tuple(float(x) for x in e.gravity), "Ambient Pressure": float(e.pressure), "Ambient Temperature": float(e.temperature),
                "Molar Mass": float(e.molar), "Wind": tuple(float(x) for x in e.wind)}
    except Exception:
        return {}


def set_fluid_inputs(mod, fl, fm):
    vals = {
        "Mass": fl.d_mass or fl.mass, "Volume": fl.d_volume or fl.volume, "Density": fl.d_density or fl.density,
        "Temperature": fl.temperature, "Velocity": tuple(fl.velocity), "Viscosity": fl.visc, "Cohesion": fl.cohesion, "Mixing Rate": fl.mix_rate,
        "Colour": (fl.color[0], fl.color[1], fl.color[2], 1.0), "Fluid ID": _fluid_index(fl, fm), "Gas": fl.kind == 'GAS',
        "No Container": bool(fl.no_container), "Fill": fl.fill, "Gas Parcels": fl.count,
        "Spawn Mode": [k for k, _n, _d in SP.SPAWN_ITEMS].index(fl.spawn_mode), "Spawn Density": fl.spawn_density,
        "Use Primitive": fl.shape_source == 'PRIMITIVE', "Primitive": [k for k, _n, _d in SP.PRIM_ITEMS].index(fl.prim_type), "Primitive Size": fl.prim_size, "Specific Heat": fl.cp, "Scale": 1.0,
        "Render Mode": {'POINTS': 0, 'RAYMARCH': 1, 'SPLAT': 0, 'SURFACE': 2}.get(fl.render_mode, 0), "Radius Scale": 2.0,
        "Build Volume And Surface": False}
    vals.update(_env_values())
    gn.set_modifier_inputs(mod, vals)


def _fluid_index(fl, fm):
    for i, f in enumerate(fm.fluids):
        if f.name == fl.name:
            return i
    return 0


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
