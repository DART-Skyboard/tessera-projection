"""Surface sources and fluid volumes (Tessera Projection, Fluid mode): settings saved in the .blend.

Scene.arc_flow_srf.items is a list of sources, each one of
  VISCOSITY  a face selection of a mesh that drags the fluid next to it (viscosity coefficient, no-slip fraction),
  EMISSION   a face selection of a mesh that emits particles (rate, velocity, temperature, fluid preset),
  VOLUME     a wireframe-box Empty (scale it with the usual manipulators) filled with a fluid that floats free in the Environment.
Emission and volume sources own a fluid entry of Fluid mode (named in `fluid`) that carries the preset, colour and render settings."""
import bpy
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty, PointerProperty,
                       StringProperty)
from bpy.types import PropertyGroup

from .core import fluidmode as FM

KIND_ITEMS = [('VISCOSITY', "Viscosity surface", "Faces that drag and viscously couple the fluid next to them"),
              ('EMISSION', "Emission surface", "Faces that emit fluid particles"),
              ('VOLUME', "Fluid volume", "A box that holds free-floating fluid or gas")]


def _sync(self, context):
    try:
        from . import surfaces_ops
        surfaces_ops.sync_source(self.id_data, self)
    except Exception:
        pass


def _dirty(self, context):
    try:
        from . import fluidmode_ops
        fluidmode_ops.mark_dirty()
    except Exception:
        pass


class AF_SRFSource(PropertyGroup):
    name: StringProperty(name="Name", default="Source", update=_sync)
    kind: EnumProperty(name="Type", items=KIND_ITEMS, default='VISCOSITY')
    obj: PointerProperty(name="Object", type=bpy.types.Object, description="The mesh whose faces form the surface (or the box Empty of a volume)", update=_dirty)
    fluid: StringProperty(name="Fluid entry", default="", description="The Fluid mode entry this source feeds (emission and volume)")
    face_attr: StringProperty(name="Face attribute", default="", description="Boolean face attribute on the mesh that stores the selected faces")
    face_count: IntProperty(name="Faces", default=0)
    face_area: FloatProperty(name="Area", default=0.0, unit='AREA')
    enabled: BoolProperty(name="Enabled", default=True, update=_dirty)
    expand: BoolProperty(default=True)
    # viscosity
    coef: FloatProperty(name="Viscosity", default=50.0, min=0.0, description="Pa s: dynamic viscosity of the layer next to the faces. Higher = the fluid is held back harder (honey ~ 10, thick syrup ~ 100, glue ~ 1000)", update=_dirty)
    noslip: FloatProperty(name="No-slip", default=1.0, min=0.0, max=1.0, subtype='FACTOR', description="Fraction of the contact that sticks: 1 = fluid takes the surface's velocity, 0 = free slip", update=_dirty)
    reach: FloatProperty(name="Reach", default=0.0, min=0.0, unit='LENGTH', description="Distance from the faces over which the coupling acts. 0 = 1.5 particle spacings", update=_dirty)
    # emission
    rate: FloatProperty(name="Rate", default=240.0, min=0.0, description="Particles per second", update=_dirty)
    speed: FloatProperty(name="Speed", default=0.8, description="m/s along the face normal", update=_dirty)
    velocity: FloatVectorProperty(name="Extra velocity", size=3, subtype='VELOCITY', default=(0.0, 0.0, 0.0), description="Added to the emitted particles (m/s)", update=_dirty)
    spread: FloatProperty(name="Spread", default=10.0, min=0.0, max=89.0, description="Half angle of the emission cone, degrees", update=_dirty)
    reverse: BoolProperty(name="Reverse direction", default=False, description="Emit against the face normals", update=_dirty)
    max_particles: IntProperty(name="Max particles", default=6000, min=1, max=100000, update=_dirty)
    t_start: FloatProperty(name="Start", default=0.0, min=0.0, description="s after the simulation starts", update=_dirty)
    t_end: FloatProperty(name="Stop", default=0.0, min=0.0, description="s; 0 = never", update=_dirty)
    # fluid (emission and volume)
    preset: EnumProperty(name="Fluid", items=FM.PRESET_ITEMS, default='WATER', update=_sync)
    temperature: FloatProperty(name="Temperature", default=293.15, min=1.0, description="K", update=_sync)
    # volume
    quantity: IntProperty(name="Quantity", default=1500, min=8, max=60000, description="Number of particles / gas parcels that fill the box", update=_sync)
    pressure: FloatProperty(name="Pressure", default=FM.P_ATM, min=1.0, description="Pa: initial gas pressure. Scales the amount (density) of a gas; liquids ignore it", update=_sync)
    density: FloatProperty(name="Density", default=0.0, min=0.0, description="kg/m3 at 293 K and 1 atm. 0 = the preset's density", update=_sync)
    velocity0: FloatVectorProperty(name="Initial velocity", size=3, subtype='VELOCITY', default=(0.0, 0.0, 0.0), update=_sync)
    # read-back
    out_force: FloatVectorProperty(size=3, default=(0.0, 0.0, 0.0))
    out_count: IntProperty(default=0)


class AF_SRFScene(PropertyGroup):
    items: CollectionProperty(type=AF_SRFSource)
    index: IntProperty(default=0)


CLASSES = (AF_SRFSource, AF_SRFScene)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Scene.arc_flow_srf = PointerProperty(type=AF_SRFScene)


def unregister():
    del bpy.types.Scene.arc_flow_srf
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
