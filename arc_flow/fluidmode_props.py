"""Fluid mode (Tessera Projection / LOTBCBP) settings: PropertyGroups on Scene and Object (saved in the .blend).

Kept in their own module so the combustion settings in props.py stay untouched.  Scene.arc_flow_fm holds the mode switch
(Combustion | Fluid), the fluid list and the solver settings; Object.arc_flow_fm marks heat sources and colliders."""
import bpy
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty,
                       PointerProperty, StringProperty)
from bpy.types import PropertyGroup

from . import spawn as SP
from .core import fluidmode as FM


def _dirty(self, context):
    try:
        from . import fluidmode_ops
        fluidmode_ops.mark_dirty()
    except Exception:
        pass


def _render_changed(self, context):
    try:
        from . import fluidmode_ops
        fluidmode_ops.refresh_render(self.id_data)
    except Exception:
        pass


def _preset_changed(self, context):
    if self.preset in FM.PRESETS:
        apply_preset(self, self.preset)
    _dirty(self, context)


def apply_preset(fl, key):
    p = FM.PRESETS[key]
    fl.kind = p['kind']
    fl.density = p['density']
    fl.visc = p['visc']
    fl.cp = p['cp']
    fl.k = p['k']
    fl.beta = p['beta']
    fl.cohesion = p['cohesion']
    fl.miscible = p['miscible']
    fl.color = p['color']
    if p['kind'] == 'GAS':
        fl.mass = fl.volume = fl.weight = 0.0


def mesh_poll(self, obj):
    """A Container may be any geometry: mesh, curve (channel), surface, text, metaball."""
    return obj.type in SP.CONTAINER_TYPES


def shape_poll(self, obj):
    return obj.type in SP.SHAPE_TYPES or bool(getattr(obj, 'particle_systems', None))


class AF_FMFluid(PropertyGroup):
    name: StringProperty(name="Name", default="Fluid", update=_dirty)
    preset: EnumProperty(name="Preset", items=FM.PRESET_ITEMS, default='WATER', update=_preset_changed)
    kind: EnumProperty(name="Phase", items=[('LIQUID', "Liquid", ""), ('GAS', "Gas", "")], default='LIQUID', update=_dirty)
    # amount (0 = derive from the others / from the geometry)
    mass: FloatProperty(name="Mass", default=0.0, min=0.0, unit='MASS', description="kg.  0 = derive", update=_dirty)
    volume: FloatProperty(name="Volume", default=0.0, min=0.0, unit='VOLUME', description="m3.  0 = the geometry's volume, or the fill fraction of the container", update=_dirty)
    weight: FloatProperty(name="Weight", default=0.0, min=0.0, description="N (mass x gravity of the Environment).  0 = derive", update=_dirty)
    density: FloatProperty(name="Density", default=997.0, min=1e-3, description="kg/m3 at 293 K and 1 atm.  Overridden by mass / volume when both are given", update=_dirty)
    temperature: FloatProperty(name="Temperature", default=293.15, min=1.0, description="Initial temperature, K", update=_dirty)
    velocity: FloatVectorProperty(name="Velocity", default=(0.0, 0.0, 0.0), subtype='VELOCITY', size=3, update=_dirty)
    visc: FloatProperty(name="Viscosity", default=1e-3, min=0.0, description="Dynamic viscosity, Pa s", update=_dirty)
    cp: FloatProperty(name="Specific heat", default=4181.0, min=1.0, description="J/(kg K)", update=_dirty)
    k: FloatProperty(name="Thermal conductivity", default=0.6, min=1e-4, description="W/(m K)", update=_dirty)
    beta: FloatProperty(name="Thermal expansion", default=2.1e-4, min=0.0, precision=6, description="1/K (Boussinesq, liquids).  Gases use 1/T from the ideal gas law", update=_dirty)
    cohesion: FloatProperty(name="Surface tension", default=0.07, min=0.0, description="N/m: cohesion between particles of the same (or a miscible) fluid", update=_dirty)
    mix_rate: FloatProperty(name="Mixing rate", default=0.0, min=0.0, description="1/s: composition, colour and temperature diffuse between neighbouring particles of miscible fluids (0 = no mixing)", update=_dirty)
    priority: IntProperty(name="Priority", default=0, description="Higher priority species absorb lower ones where a rule says ABSORB", update=_dirty)
    miscible: BoolProperty(name="Miscible", default=True, description="Off: no cohesion or velocity mixing with other fluids (oil vs water, lava lamp wax)", update=_dirty)
    cool: FloatProperty(name="Ambient exchange", default=0.05, min=0.0, description="1/s: relaxation toward the Environment temperature for exposed particles (free surface, walls, free gas)", update=_dirty)
    drag: FloatProperty(name="Drag", default=-1.0, description="1/s toward the wind for free fluids.  -1 = automatic", update=_dirty)
    # geometry
    no_container: BoolProperty(name="No container", default=True, description="Ticked: a free fluid or gas moving in the air.  Unticked: held by the container object below", update=_dirty)
    container: PointerProperty(name="Container", type=bpy.types.Object, poll=mesh_poll, description="Cavity, tank or tower (may be elevated)", update=_dirty)
    shape: PointerProperty(name="Shape", type=bpy.types.Object, poll=shape_poll, description="Any geometry as the fluid's shape: mesh, curve (a channel), surface, text, metaball, point cloud or an object with a particle system. A closed shape is filled, an open one gives its surface; see Spawn", update=_dirty)
    shape_source: EnumProperty(name="Shape from", items=[('PRIMITIVE', "Primitive", "A built-in primitive (sphere of points by default): edit its size and position, or replace it by an object"),
                                                         ('OBJECT', "Object", "The Shape object (empty = fill the Container)")], default='OBJECT', update=_dirty)
    prim_type: EnumProperty(name="Primitive", items=SP.PRIM_ITEMS, default='SPHERE', update=_dirty)
    prim_size: FloatProperty(name="Size", default=0.1, min=1e-3, unit='LENGTH', description="Radius (sphere, cylinder, cone), half edge (cube, plane) or major radius (torus)", update=_dirty)
    prim_loc: FloatVectorProperty(name="Location", size=3, default=(0.0, 0.0, 0.5), subtype='TRANSLATION', unit='LENGTH', update=_dirty)
    prim_rot: FloatVectorProperty(name="Rotation", size=3, default=(0.0, 0.0, 0.0), subtype='EULER', update=_dirty)
    prim_scale: FloatVectorProperty(name="Scale", size=3, default=(1.0, 1.0, 1.0), min=1e-3, update=_dirty)
    spawn_mode: EnumProperty(name="Spawn", items=SP.SPAWN_ITEMS, default='AUTO', update=_dirty)
    spawn_density: FloatProperty(name="Density", default=1.0, min=0.02, max=1.0, subtype='FACTOR', description="Fraction of the generated particles that is kept (1 = the full lattice / surface layer / every vertex or Blender particle)", update=_dirty)
    psys_name: StringProperty(name="Particle system", default="", description="Name of the object's particle system (empty = every emitter system)", update=_dirty)
    d_spawn: StringProperty(default="")
    fill: FloatProperty(name="Fill", default=50.0, min=0.0, max=100.0, subtype='PERCENTAGE', description="Percent of the container when there is no Shape and no Volume", update=_dirty)
    count: IntProperty(name="Gas parcels", default=2000, min=8, max=60000, description="Number of parcels a gas is represented by", update=_dirty)
    # appearance / render
    color: FloatVectorProperty(name="Colour", subtype='COLOR', size=3, default=(0.25, 0.55, 0.95), min=0.0, max=1.0, update=_render_changed)
    temp_color: BoolProperty(name="Temperature colour", default=False, description="Colour the particles by temperature (cold to hot ramp)", update=_render_changed)
    t_lo: FloatProperty(name="Cold", default=290.0, description="K at the cold end of the ramp", update=_render_changed)
    t_hi: FloatProperty(name="Hot", default=340.0, description="K at the hot end of the ramp", update=_render_changed)
    render_mode: EnumProperty(name="Render", items=[('POINTS', "Points / mesh per point", "Instance an icosphere or your own mesh on every particle"),
                                                    ('RAYMARCH', "Ray marching", "Volume rendered by the render engine"),
                                                    ('SPLAT', "Gaussian splat", "Soft blobs"),
                                                    ('SURFACE', "Fluid surface", "A welded mesh of the smoothed particle density (surface nets): touching blobs become one surface, separate blobs stay separate; colour from the composition")], default='POINTS', update=_render_changed)
    mesh_obj: PointerProperty(name="Mesh per particle", type=bpy.types.Object, description="Object instanced on every particle (empty = icosphere)", update=_render_changed)
    mesh_collection: PointerProperty(name="Collection per particle", type=bpy.types.Collection, update=_render_changed)
    particle_scale: FloatProperty(name="Particle scale", default=1.0, min=0.01, update=_render_changed)
    rm_density: FloatProperty(name="Volume density", default=40.0, min=0.0, update=_render_changed)
    # derived (read-only readout after Build)
    d_particles: IntProperty(default=0)
    d_mass: FloatProperty(default=0.0)
    d_volume: FloatProperty(default=0.0)
    d_density: FloatProperty(default=0.0)
    # runtime-created
    points_obj: PointerProperty(type=bpy.types.Object)
    material: PointerProperty(type=bpy.types.Material)
    volume_material: PointerProperty(type=bpy.types.Material)
    expand: BoolProperty(default=True)


class AF_FMScene(PropertyGroup):
    mode: EnumProperty(name="Mode", items=[('COMBUSTION', "Combustion", "Fluid flow through connected cavities with combustion (the original Arc Flow solver)"),
                                           ('FLUID', "Fluid", "General fluids: mixing, thermal convection, free gas, lava lamps")], default='COMBUSTION')
    fluids: CollectionProperty(type=AF_FMFluid)
    index: IntProperty(default=0)
    max_particles: IntProperty(name="Max particles", default=20000, min=100, max=100000, description="The particle spacing is chosen so the liquids stay under this count", update=_dirty)
    spacing: FloatProperty(name="Particle spacing", default=0.0, min=0.0, unit='LENGTH', description="0 = automatic from Max particles", update=_dirty)
    iterations: IntProperty(name="Volume iterations", default=4, min=0, max=20, description="Particle spacing (volume drift) correction passes after the grid pressure solve; more = less clumping, slower", update=_dirty)
    max_substeps: IntProperty(name="Max substeps", default=4, min=1, max=16, update=_dirty)
    speedup: FloatProperty(name="Thermal speed-up", default=1.0, min=0.0, description="Multiplies heat transport (lamps, conduction, ambient exchange) so slow real heating fits an animation. 1 = real time", update=_dirty)
    lotbcbp: BoolProperty(name="LOTBCBP refinement", default=True, description="On-demand collision bricks where fluid touches a surface", update=_dirty)
    gust: FloatProperty(name="Wind gusts", default=0.0, min=0.0, description="m/s amplitude of the turbulent wind fluctuation on free fluids", update=_dirty)
    ground_friction: FloatProperty(name="Ground friction", default=0.4, min=0.0, max=1.0, update=_dirty)
    ground_temp_on: BoolProperty(name="Ground heat sink", default=False, update=_dirty)
    ground_temp: FloatProperty(name="Ground temperature", default=288.15, min=1.0, update=_dirty)
    seed: IntProperty(name="Seed", default=1, update=_dirty)
    rules_json: StringProperty(name="Species rules (JSON)", default="", description='{"species": [{"name":..., "color":[r,g,b], "density":..., "priority":...}], "rules": [{"a":..., "b":..., "outcome":"MISCIBLE|IMMISCIBLE|ABSORB|REACT", ...}]}', update=_dirty)
    fill_mode: EnumProperty(name="Fill", items=[('LEVEL', "Level-set fill + settle", "Analytic fill of the cavity to the requested volume along gravity (stratified by density) and a short settle (10 steps)"),
                                                 ('POUR', "Pour fill", "Seed from the bottom and run the full dynamic fill under the Environment gravity until the fluid is at rest"),
                                                 ('QUICK', "Quick fill", "Level-set fill and 3 relaxing steps"),
                                                 ('OFF', "No fill phase", "Start from the seeded lattice")], default='LEVEL', update=_dirty)
    fill_steps: IntProperty(name="Max settle steps", default=100, min=1, max=1000, description="Upper bound of the dynamic fill (frames of simulated time)", update=_dirty)
    heat_start: IntProperty(name="Start heat at frame", default=1, min=0, description="Heat sources switch on at this frame; before it the settled fluid rests")
    use_scene_range: BoolProperty(name="Use scene frame range", default=True)
    status: StringProperty(default="Idle")


class AF_FMObject(PropertyGroup):
    channel_radius: FloatProperty(name="Channel radius", default=0.02, min=1e-4, unit='LENGTH', description="A curve (or an edge-only mesh) without its own bevel is tubed to this radius when it is used as a Container or Shape (a channel / pipe)", update=_dirty)
    heat_on: BoolProperty(name="Heat source", default=False, description="Warms (or, with negative power, cools) the fluid near this object: a lamp, a burner, an ice block", update=_dirty)
    heat_power: FloatProperty(name="Power", default=25.0, description="W", update=_dirty)
    heat_radius: FloatProperty(name="Radius", default=0.05, min=1e-3, unit='LENGTH', update=_dirty)
    collider: BoolProperty(name="Fluid collider", default=False, description="A solid the fluid cannot enter. Moving / rotating objects drive the fluid by their surface velocity", update=_dirty)
    friction: FloatProperty(name="Wall friction", default=0.2, min=0.0, max=1.0, description="0 = free slip, 1 = no slip (the wall surface velocity is passed on)", update=_dirty)
    # two-way coupling: the fluid pushes this body (containers and colliders alike)
    two_way: BoolProperty(name="Fluid pushes this body", default=True, description="Fluid pressure and drag act on the body (net force and torque are measured every step). The body moves only when it has a Mass above zero", update=_dirty)
    mass: FloatProperty(name="Mass", default=0.0, min=0.0, unit='MASS', description="kg. 0 = the body is fixed or animated (the fluid still measures force on it); above 0 it moves under fluid force, gravity and its constraint", update=_dirty)
    constraint: EnumProperty(name="Motion", items=[('FREE', "Free", "Six degrees of freedom"), ('SLIDER', "Slider / piston", "Translate along the axis only"),
                                                     ('HINGE', "Hinge", "Rotate about the axis through the object origin")], default='FREE', update=_dirty)
    axis: FloatVectorProperty(name="Axis", size=3, default=(0.0, 0.0, 1.0), description="World axis of the slider or hinge", update=_dirty)
    limit_on: BoolProperty(name="Limits", default=False, update=_dirty)
    limit_lo: FloatProperty(name="Min", default=0.0, description="m (slider) or radians (hinge) from the start pose", update=_dirty)
    limit_hi: FloatProperty(name="Max", default=1.0, update=_dirty)
    stick: FloatProperty(name="Static friction", default=0.0, min=0.0, description="N (N m for a hinge) the fluid must exceed before the body moves", update=_dirty)
    drag: FloatProperty(name="Viscous friction", default=0.0, min=0.0, description="N s/m: resistance proportional to speed", update=_dirty)


CLASSES = (AF_FMFluid, AF_FMScene, AF_FMObject)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Scene.arc_flow_fm = PointerProperty(type=AF_FMScene)
    bpy.types.Object.arc_flow_fm = PointerProperty(type=AF_FMObject)


def unregister():
    del bpy.types.Object.arc_flow_fm
    del bpy.types.Scene.arc_flow_fm
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
