"""Tessera Projection cloth solver: PropertyGroups on Scene and Object (saved in the .blend)."""
import bpy
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty,
                       FloatVectorProperty, IntProperty, PointerProperty, StringProperty)
from bpy.types import PropertyGroup

from .core import cloth_solver as CS

PRESET_ITEMS = CS.PRESET_ITEMS
QUALITY_ITEMS = CS.QUALITY_ITEMS
PIN_MODE_ITEMS = [
    ('NONE', "None", "Do not auto-assign pins"),
    ('TOP', "Top edge", "Pin the highest ring of vertices (a hanging hem or cape collar)"),
    ('SHOULDER', "Shoulder ring", "Pin the top-left and top-right clusters (a cape or stole)"),
    ('WAIST', "Waist band", "Pin a mid-height band (a skirt or kilt)"),
]
BAKE_ITEMS = [
    ('SHAPEKEYS', "Shape keys", "One shape key per baked frame on the garment mesh"),
    ('NPZ', "Point cache (.npz)", "Compressed NumPy archive of rest, pins and per-frame positions"),
    ('ALEMBIC', "Alembic (.abc)", "Alembic cache for interchange with other DCC and game tools"),
    ('PACKAGE', "Game-engine package", "Rest mesh, pin weights, stiffness and collider metadata (JSON + NPZ)"),
]


def _mesh_poll(_s, o):
    return o is not None and o.type == 'MESH'


def _arm_poll(_s, o):
    return o is not None and o.type == 'ARMATURE'


class AF_ClothCollider(PropertyGroup):
    name: StringProperty(name="Name", default="Collider")
    obj: PointerProperty(name="Object", type=bpy.types.Object, poll=_mesh_poll,
                         description="Collision body the garment is projected against")
    enabled: BoolProperty(name="Enabled", default=True)


class AF_ClothGarment(PropertyGroup):
    name: StringProperty(name="Name", default="Garment")
    obj: PointerProperty(name="Garment", type=bpy.types.Object, poll=_mesh_poll,
                         description="The cloth mesh whose vertices the solver updates")
    armature: PointerProperty(name="Armature", type=bpy.types.Object, poll=_arm_poll,
                              description="Optional armature whose pose drives pin / skin targets")
    pin_group: StringProperty(name="Pin group", default="ClothPins",
                              description="Vertex group holding pinned (or skin-weighted) vertices")
    pin_mode: EnumProperty(name="Auto pin", items=PIN_MODE_ITEMS, default='SHOULDER')
    pin_threshold: FloatProperty(name="Pin threshold", default=0.5, min=0.0, max=1.0, subtype='FACTOR',
                                 description="Vertex-group weight at which a vertex becomes a hard pin")
    enabled: BoolProperty(name="Enabled", default=True)
    expand: BoolProperty(default=True)
    preset: EnumProperty(name="Preset", items=PRESET_ITEMS, default='CAPE')
    quality: EnumProperty(name="Quality", items=QUALITY_ITEMS, default='BALANCED',
                          description="Substeps, iterations, self-collision and tessellation budget")
    follow_preset: BoolProperty(name="Use preset values", default=True,
                                description="Stretch, bend and aero coefficients follow the selected garment preset")
    # overridable physical parameters (used when follow_preset is off, or as live tweaks)
    density: FloatProperty(name="Density", default=0.22, min=0.001, description="kg / m² areal density")
    stretch: FloatProperty(name="Stretch stiffness", default=3.2e4, min=1.0)
    shear: FloatProperty(name="Shear stiffness", default=1.1e4, min=1.0)
    bend: FloatProperty(name="Bend stiffness", default=16.0, min=0.01)
    damping: FloatProperty(name="Damping", default=0.14, min=0.0, max=8.0)
    drag: FloatProperty(name="Aero drag", default=1.5, min=0.0)
    lift: FloatProperty(name="Aero lift", default=0.60, min=0.0)
    thickness: FloatProperty(name="Thickness", default=0.003, min=0.0001, unit='LENGTH')
    margin: FloatProperty(name="Collision margin", default=0.006, min=0.0, unit='LENGTH')
    friction: FloatProperty(name="Friction", default=0.20, min=0.0, max=1.0, subtype='FACTOR')
    self_collision: BoolProperty(name="Self-collision", default=True)
    use_dihedral: BoolProperty(name="Dihedral bend", default=True)
    substeps: IntProperty(name="Substeps", default=4, min=1, max=32)
    iterations: IntProperty(name="Iterations", default=8, min=1, max=64)
    tessellation_edge: FloatProperty(name="Collider tessellation", default=0.04, min=0.0, unit='LENGTH',
                                     description="Max edge length of the virtual LOTBCBP tessellation; 0 leaves the source mesh")
    tessellation_cap: IntProperty(name="Tessellation budget", default=80000, min=0, max=2000000)
    override_quality: BoolProperty(name="Override quality", default=False,
                                   description="Use the substeps, iterations, self-collision and tessellation values below instead of the quality profile")
    # status
    n_verts: IntProperty(name="Vertices", default=0)
    n_pins: IntProperty(name="Pins", default=0)
    last_ms: FloatProperty(name="Last step", default=0.0, description="milliseconds")
    last_stretch: FloatProperty(name="Peak stretch", default=0.0)
    last_penetration: FloatProperty(name="Penetration", default=0.0, unit='LENGTH')


class AF_ClothScene(PropertyGroup):
    garments: CollectionProperty(type=AF_ClothGarment)
    index: IntProperty(default=0)
    colliders: CollectionProperty(type=AF_ClothCollider)
    collider_index: IntProperty(default=0)
    live: BoolProperty(name="Live simulation", default=False,
                       description="Update the garment in the viewport as you pose or drive the character")
    live_fps: FloatProperty(name="Live rate", default=60.0, min=12.0, max=120.0, description="Ticks per second in live mode")
    use_scene_wind: BoolProperty(name="Use Environment wind", default=True,
                                 description="Sample Tessera Environment wind, gravity and air density")
    wind: FloatVectorProperty(name="Wind", default=(0.0, 8.0, 0.0), subtype='VELOCITY', size=3,
                              description="m/s, used when Environment wind is off or missing")
    gravity: FloatVectorProperty(name="Gravity", default=(0.0, 0.0, -9.81), subtype='ACCELERATION', size=3)
    air_density: FloatProperty(name="Air density", default=1.225, min=0.0, description="kg/m³")
    use_fluid_velocity: BoolProperty(name="Couple to Fluid mode", default=True,
                                     description="If a Fluid-mode solver is running, sample its velocity field as the airflow")
    bake_mode: EnumProperty(name="Bake to", items=BAKE_ITEMS, default='NPZ')
    bake_path: StringProperty(name="Bake path", default="//cloth_cache", subtype='DIR_PATH')
    bake_name: StringProperty(name="Cache name", default="cape")
    frame_start: IntProperty(name="Start", default=1, min=0)
    frame_end: IntProperty(name="End", default=60, min=1)
    use_scene_range: BoolProperty(name="Scene frame range", default=True)
    status: StringProperty(name="Status", default="Cloth solver is idle.")
    last_ms: FloatProperty(default=0.0)
    last_fps: FloatProperty(default=0.0)


class AF_ClothObject(PropertyGroup):
    is_garment: BoolProperty(name="Cloth garment", default=False)
    is_collider: BoolProperty(name="Cloth collider", default=False)
    garment_name: StringProperty(name="Garment", default="")


CLASSES = (AF_ClothCollider, AF_ClothGarment, AF_ClothScene, AF_ClothObject)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Scene.arc_flow_cloth = PointerProperty(type=AF_ClothScene)
    bpy.types.Object.arc_flow_cloth = PointerProperty(type=AF_ClothObject)


def unregister():
    del bpy.types.Object.arc_flow_cloth
    del bpy.types.Scene.arc_flow_cloth
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
