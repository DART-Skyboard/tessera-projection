"""Tessera Projection muscle solver: PropertyGroups on the Scene (saved in the .blend)."""
import bpy
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty, FloatVectorProperty,
                       IntProperty, PointerProperty, StringProperty)
from bpy.types import PropertyGroup

from .core import muscle_solver as MS

PRESET_ITEMS = MS.PRESET_ITEMS
DETAIL_ITEMS = tuple((d[0], d[1], d[2]) for d in MS.DETAIL_ITEMS)
STATE_ITEMS = (
    ('REST', "Rest", "Muscle tone only: idle character"),
    ('WALK', "Walk", "Joints move at walking speed: medium intensity"),
    ('RUN', "Run", "Joints move fast: strongest intensity"),
)
ACCENT_ITEMS = (
    ('AUTO', "Auto", "The end nearer to the joint of the driving bone"),
    ('A', "End A", "Slide toward the first tendon end"),
    ('B', "End B", "Slide toward the second tendon end"),
)
EXPORT_ITEMS = (('GLB', "glTF binary (.glb)", "Shape keys, skin weights and vertex colours"),
                ('FBX', "FBX (.fbx)", "Blend shapes, skin weights and vertex colours"))


def _mesh_poll(_s, o):
    return o is not None and o.type == 'MESH'


def _arm_poll(_s, o):
    return o is not None and o.type == 'ARMATURE'


def _update_preview(self, ctx):
    try:
        from . import muscle_ops
        muscle_ops.apply_preview(ctx.scene, self.key)
    except Exception:
        pass


def _update_state(self, ctx):
    gains = {'REST': self.gain_rest, 'WALK': self.gain_walk, 'RUN': self.gain_run}
    self.state_gain = gains.get(self.state, 1.0)


class AF_MuscleRegion(PropertyGroup):
    key: StringProperty(name="Key", default="")
    label: StringProperty(name="Name", default="Muscle")
    group: StringProperty(name="Category", default="")
    cid: IntProperty(name="Region id", default=0, description="Stored in the skin attribute tp_muscle_id; fixed per muscle group")
    color: FloatVectorProperty(name="Colour", size=3, subtype='COLOR', min=0.0, max=1.0, default=(0.8, 0.3, 0.3),
                               description="Static region colour painted on the skin")
    enabled: BoolProperty(name="Enabled", default=True)
    flex: FloatProperty(name="Flex intensity", default=1.0, min=0.0, max=3.0, subtype='FACTOR',
                        description="How strongly this region bulges when it flexes")
    accent: FloatProperty(name="Accent intensity", default=0.6, min=0.0, max=3.0, subtype='FACTOR',
                          description="Slide of the skin along the muscle toward the joint end when it flexes")
    emphasis: FloatProperty(name="Emphasis", default=1.0, min=0.0, max=3.0, subtype='FACTOR',
                            description="Initial per-region emphasis (preset value), multiplies flex and accent")
    tone: FloatProperty(name="Rest tone", default=0.06, min=0.0, max=1.0, description="Flex at rest, before the joint bends")
    joint_range: FloatProperty(name="Joint range", default=1.2, min=0.1, max=6.3, unit='ROTATION',
                               description="Joint angle that gives full flex")
    bone: StringProperty(name="Driving bone", default="",
                         description="Bone whose rotation drives this region (empty = detected from the skin weights)")
    accent_end: EnumProperty(name="Accent end", items=ACCENT_ITEMS, default='AUTO')
    attach_bone_ends: BoolProperty(name="Tendons at bone ends", default=False,
                                   description="Use the driving bone's head and tail as the tendon attachment points")
    muscle_obj: PointerProperty(name="Muscle mesh", type=bpy.types.Object, poll=_mesh_poll,
                                description="Actual muscle geometry for this category (seeds the region and sets the axis)")
    bone_obj: PointerProperty(name="Bone mesh", type=bpy.types.Object, poll=_mesh_poll,
                              description="Actual bone geometry the muscle attaches to (its two ends are the tendon points)")
    preview: FloatProperty(name="Preview", default=0.0, min=0.0, max=1.5, subtype='FACTOR', update=_update_preview,
                           description="Set the region's shape keys directly (drivers are overridden while non-zero)")
    n_painted: IntProperty(default=0)
    n_instances: IntProperty(default=0)


class AF_MuscleScene(PropertyGroup):
    skin: PointerProperty(name="Skin mesh", type=bpy.types.Object, poll=_mesh_poll,
                          description="Character skin mesh the muscles are solved on")
    armature: PointerProperty(name="Armature", type=bpy.types.Object, poll=_arm_poll,
                              description="Skeleton that deforms the skin")
    preset: EnumProperty(name="Preset", items=PRESET_ITEMS, default='BIPED')
    detail: EnumProperty(name="Detail", items=DETAIL_ITEMS, default='STANDARD')
    # creature setup
    heads: IntProperty(name="Heads", default=1, min=0, max=12)
    arms: IntProperty(name="Arms", default=2, min=0, max=12)
    legs: IntProperty(name="Legs", default=2, min=0, max=16)
    tails: IntProperty(name="Tails", default=0, min=0, max=8)
    fingers: IntProperty(name="Fingers per hand", default=0, min=0, max=12)
    toes: IntProperty(name="Toes per foot", default=0, min=0, max=12)
    joints_per_limb: IntProperty(name="Joints per limb", default=2, min=1, max=12)
    regions: CollectionProperty(type=AF_MuscleRegion)
    index: IntProperty(default=0)
    resolution: FloatProperty(name="Resolution factor", default=1.0, min=0.1, max=10.0, step=10, precision=2,
                              description="Muscle projection resolution, like the Remesh multiplier. Below 1 is coarser and smoother "
                                          "(cartoon), above 1 adds finer region blending and fibre bundles (cinematic)")
    brush_radius: FloatProperty(name="Brush radius", default=0.05, min=0.002, max=2.0, unit='LENGTH',
                                description="Radius of the region pencil in the skin's local units")
    brush_erase: BoolProperty(name="Erase", default=False, description="Paint with the eraser instead of the active region colour")
    show_colors: BoolProperty(name="Show region colours", default=True)
    # animation states
    state: EnumProperty(name="Animation state", items=STATE_ITEMS, default='WALK', update=_update_state,
                        description="Intensity profile used by the shape-key drivers (engines use the sidecar speed curve instead)")
    gain_rest: FloatProperty(name="Rest", default=0.6, min=0.0, max=3.0, update=_update_state)
    gain_walk: FloatProperty(name="Walk", default=1.0, min=0.0, max=3.0, update=_update_state)
    gain_run: FloatProperty(name="Run", default=1.5, min=0.0, max=3.0, update=_update_state)
    state_gain: FloatProperty(name="State gain", default=1.0, min=0.0, max=3.0)
    use_jiggle: BoolProperty(name="Jiggle (engine)", default=False,
                             description="Write jiggle spring parameters to the sidecar (applied by the game-engine runtime)")
    jiggle_k: FloatProperty(name="Spring", default=120.0, min=1.0, max=2000.0)
    jiggle_damping: FloatProperty(name="Damping", default=9.0, min=0.0, max=200.0)
    use_drivers: BoolProperty(name="Drivers from joint angle", default=True,
                              description="Drive every muscle shape key from its bone's rotation, so it flexes while you animate")
    export_format: EnumProperty(name="Format", items=EXPORT_ITEMS, default='GLB')
    export_path: StringProperty(name="Export path", default="//muscle_export", subtype='FILE_PATH')
    status: StringProperty(name="Status", default="Pick a skin mesh and a skeleton, then choose a preset.")
    solved: BoolProperty(default=False)
    n_instances: IntProperty(default=0)
    last_ms: FloatProperty(default=0.0)


CLASSES = (AF_MuscleRegion, AF_MuscleScene)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Scene.arc_flow_muscle = PointerProperty(type=AF_MuscleScene)


def unregister():
    del bpy.types.Scene.arc_flow_muscle
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
