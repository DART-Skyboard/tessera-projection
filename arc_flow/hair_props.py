"""Tessera Projection Hair: settings saved in the .blend.

Scene.arc_flow_hair holds the live-solver, bake and game-engine-package knobs
plus pointers to the active CURVES groom, the collision body and an optional
armature.  Object.arc_flow_hair tags a curves object as a hair groom (or a
mesh as a collider / scalp) and stores per-object stiffness overrides.
"""
import bpy
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty, FloatVectorProperty,
                       IntProperty, PointerProperty, StringProperty)
from bpy.types import PropertyGroup


def _curves_poll(self, obj):
    return obj is not None and obj.type == 'CURVES'


def _mesh_poll(self, obj):
    return obj is not None and obj.type == 'MESH'


def _arm_poll(self, obj):
    return obj is not None and obj.type == 'ARMATURE'


def _overlay_changed(self, context):
    try:
        from . import hair_ops
        hair_ops.set_collider_overlay(context.scene, bool(self.show_colliders))
    except Exception:
        pass


def _live_changed(self, context):
    try:
        from . import hair_ops
        if self.live:
            hair_ops.ensure_live(context.scene)
        else:
            hair_ops.stop_live(context.scene)
    except Exception:
        pass


class AF_HairCollider(PropertyGroup):
    """One collision body of the hair solver (head, shoulders, body, a glove / hand, props)."""
    obj: PointerProperty(name="Object", type=bpy.types.Object, poll=_mesh_poll,
                         description="Mesh strands collide with. Moving or posed bodies are re-read every step")
    enabled: BoolProperty(name="Enabled", default=True)
    thickness: FloatProperty(name="Thickness", default=0.008, min=0.0, max=0.1, unit='LENGTH', precision=4,
                             description="Keep strand points this far outside this body (a skin / padding layer)")
    friction: FloatProperty(name="Friction", default=0.3, min=0.0, max=1.0, subtype='FACTOR',
                            description="Tangential grip: 0 = slippery, 1 = hair sticks to this body")
    deforming: BoolProperty(name="Deforming", default=False,
                            description="Rebuild this body's collision field every step (posed / shape-keyed bodies). "
                                        "Rigid movers like a glove do not need it")


class AF_HairScene(PropertyGroup):
    hair: PointerProperty(name="Hair curves", type=bpy.types.Object, poll=_curves_poll,
                          description="The Empty Hair Curves (CURVES) object Tessera Projection Hair drives")
    collider: PointerProperty(name="Collision body", type=bpy.types.Object, poll=_mesh_poll,
                              description="Head, body or environment mesh the strands collide with (LOTBCBP SDF)")
    surface: PointerProperty(name="Scalp / surface", type=bpy.types.Object, poll=_mesh_poll,
                             description="Mesh the strand roots are bound to (defaults to the collision body)")
    armature: PointerProperty(name="Armature", type=bpy.types.Object, poll=_arm_poll,
                              description="Character rig whose pose drives root motion (optional)")
    bone: StringProperty(name="Bone", default="",
                         description="Pose bone the scalp follows; leave empty to use the armature object transform")

    colliders: CollectionProperty(type=AF_HairCollider)
    collider_index: IntProperty(default=0)
    use_colliders: BoolProperty(name="Collision", default=True,
                                description="Master switch: strands collide with every enabled body below (off = hair passes through bodies, for comparison)")
    show_colliders: BoolProperty(name="Show bodies", default=False, update=_overlay_changed,
                                 description="Display the collision bodies as a translucent overlay (capsule proxies are hidden otherwise; other meshes get a wire overlay)")
    use_scene_colliders: BoolProperty(name="Other Tessera collision bodies", default=True,
                                      description="Also collide with every body the Cloth / Fluid / Wind-tunnel tabs marked as a collision body")

    live: BoolProperty(name="Live simulation", default=False, update=_live_changed,
                       description="Tick the hair solver in the viewport as you pose, play, or drive the rig")
    follow_depsgraph: BoolProperty(name="React to posing", default=True,
                                   description="Also step when the armature, scalp or collider is moved, even if the animation is paused")

    # integration
    dt_mode: EnumProperty(name="Time", items=[
        ('SCENE', "Scene FPS", "Use 1 / scene render FPS as the frame dt"),
        ('FIXED', "Fixed dt", "Use the fixed time step below"),
    ], default='SCENE')
    dt: FloatProperty(name="Fixed dt", default=1.0 / 24.0, min=1.0 / 240.0, max=0.1, precision=4,
                      description="Seconds per tick when Time is Fixed dt")
    substeps: IntProperty(name="Substeps", default=4, min=1, max=16,
                          description="Solver substeps per tick (more = stabler, heavier)")
    gravity: FloatVectorProperty(name="Gravity", default=(0.0, 0.0, -9.81), subtype='ACCELERATION', size=3)
    use_scene_gravity: BoolProperty(name="Use scene / Environment gravity", default=True,
                                    description="Read gravity from the Tessera Environment card, or the scene, when available")
    damping: FloatProperty(name="Damping", default=0.92, min=0.0, max=1.0, subtype='FACTOR',
                           description="Velocity keep per substep (1 = no damping)")

    # constraints / style
    stretch_iterations: IntProperty(name="Stretch iterations", default=8, min=1, max=32)
    stretch_compliance: FloatProperty(name="Stretch compliance", default=0.0, min=0.0,
                                      description="0 = inextensible fibres; larger values allow stretch")
    bend_iterations: IntProperty(name="Bend iterations", default=4, min=0, max=16)
    bend_compliance: FloatProperty(name="Bend compliance", default=0.0002, min=0.0)
    rest_shape: FloatProperty(name="Curl / style memory", default=0.35, min=0.0, max=1.0, subtype='FACTOR',
                              description="How strongly strands remember their rest pose as the head moves")
    rest_shape_root: FloatProperty(name="Root stiffness", default=0.85, min=0.0, max=1.0, subtype='FACTOR')
    rest_shape_tip: FloatProperty(name="Tip stiffness", default=0.12, min=0.0, max=1.0, subtype='FACTOR')
    pinned_prefix: IntProperty(name="Pinned root points", default=1, min=1, max=8,
                               description="How many points at the root of each strand stay glued to the scalp")

    # collision / tessellation
    self_collision: BoolProperty(name="Self-collision", default=True,
                                 description="Strands collide with each other (spatial hash, collision radius). On by default")
    collision_radius: FloatProperty(name="Collision radius", default=0.008, min=0.0, unit='LENGTH',
                                    description="Keep strand points this far outside the collision body and each other")
    collision_iterations: IntProperty(name="Collision iterations", default=2, min=1, max=8)
    collision_friction: FloatProperty(name="Collision friction", default=0.3, min=0.0, max=1.0, subtype='FACTOR')
    sdf_resolution: IntProperty(name="SDF resolution", default=32, min=8, max=128,
                                description="Collision field resolution (tessellation of the collision body)")
    sdf_level: IntProperty(name="LOTBCBP level", default=2, min=0, max=4,
                           description="Virtual tessellation / brick refinement level around contacts")
    sdf_refine: BoolProperty(name="Refine at contacts", default=True,
                             description="Request LOTBCBP bricks around hair contacts each step")
    # hair-hair, volume and profiles
    hair_collision: BoolProperty(name="Strand-strand contact", default=True,
                                 description="Hair repels / drags neighbouring strands (vectorised spatial hash) so the groom keeps its volume")
    hair_every: IntProperty(name="Contact stride", default=1, min=1, max=4,
                            description="Run strand-strand contact on every N-th substep (2 halves its cost on dense grooms)")
    strand_radius: FloatProperty(name="Strand radius", default=0.004, min=0.0005, max=0.05, unit='LENGTH', precision=4,
                                 description="Fibre radius used for hair-hair contact")
    repel_strength: FloatProperty(name="Repulsion", default=0.5, min=0.0, max=1.0, subtype='FACTOR')
    volume: FloatProperty(name="Volume", default=0.0, min=0.0, max=4.0,
                          description="Inflates the hair-hair contact radius: puffier, fuller hair")
    clumping: FloatProperty(name="Clumping", default=0.0, min=0.0, max=4.0,
                            description="Pulls neighbouring strands into locks")
    strand_friction: FloatProperty(name="Strand friction", default=0.05, min=0.0, max=1.0, subtype='FACTOR',
                                   description="Velocity sharing between touching strands (hair drags hair)")
    gravity_scale: FloatProperty(name="Gravity scale", default=1.0, min=0.0, max=4.0,
                                 description="Scales gravity for the whole groom (per-strand scale lives on the hair object)")
    bend_tip_scale: FloatProperty(name="Tip floppiness", default=3.0, min=1.0, max=20.0,
                                  description="Bend compliance at the tip relative to the root")
    stiffness_curve: FloatProperty(name="Stiffness curve", default=1.0, min=0.2, max=4.0,
                                   description="Shape of the root-to-tip stiffness ramp (above 1 keeps the root stiff longer)")
    use_ground: BoolProperty(name="Ground collision", default=False)
    ground_z: FloatProperty(name="Ground height", default=0.0, unit='LENGTH')

    # wind / fluid
    use_env_wind: BoolProperty(name="Use Environment wind", default=True,
                               description="Sample Tessera Environment wind (and a fluid velocity field when one is running)")
    wind: FloatVectorProperty(name="Wind velocity", default=(0.0, 0.0, 0.0), subtype='VELOCITY', size=3)
    wind_drag: FloatProperty(name="Wind / fluid drag", default=0.8, min=0.0,
                             description="Quadratic aerodynamic drag (anisotropic: more across the fibre than along it)")
    wind_tangent: FloatProperty(name="Tangent drag", default=0.15, min=0.0, max=1.0, subtype='FACTOR')
    fluid_coupling: FloatProperty(name="Fluid coupling", default=1.0, min=0.0, max=4.0)

    # guides / budget
    guide_ratio: FloatProperty(name="Guide ratio", default=0.25, min=0.02, max=1.0, subtype='FACTOR',
                               description="Fraction of strands simulated as guides; the rest are skinned children")
    child_k: IntProperty(name="Child guides", default=3, min=1, max=8,
                         description="Nearest guides each child strand is skinned to")
    child_smooth: FloatProperty(name="Child smooth", default=1.0, min=0.25, max=4.0)
    max_segment_budget: IntProperty(name="Max points per strand", default=0, min=0, max=128,
                                    description="0 = keep the groom's own segment counts; otherwise resample to this cap")
    max_points: IntProperty(name="Max simulated points", default=0, min=0,
                            description="0 = unlimited. Lower this (and the guide ratio) for a 60 fps game-engine preview")

    # bake / export
    bake_format: EnumProperty(name="Bake format", items=[
        ('NPZ', "NumPy cache (.npz)", "Point cache of every baked frame"),
        ('ABC', "Alembic (.abc)", "Export the animated CURVES object through Blender's Alembic exporter"),
        ('BOTH', "Cache + Alembic", "Write both a .npz cache and an Alembic file"),
    ], default='NPZ')
    use_scene_range: BoolProperty(name="Use scene frame range", default=True)
    frame_start: IntProperty(name="Start", default=1)
    frame_end: IntProperty(name="End", default=48)
    bake_dir: StringProperty(name="Bake folder", subtype='DIR_PATH', default="//tessera_hair_cache/")
    bake_name: StringProperty(name="Name", default="hair")
    package_dir: StringProperty(name="Game-engine package", subtype='DIR_PATH', default="//tessera_hair_package/",
                                description="Folder for hair_physics.json + rest_curves.npz (+ cache) for Unreal / Unity / web")
    status: StringProperty(name="Status", default="Idle")
    last_package: StringProperty(name="Last package", subtype='FILE_PATH', default="")
    last_cache: StringProperty(name="Last cache", subtype='FILE_PATH', default="")
    last_alembic: StringProperty(name="Last Alembic", subtype='FILE_PATH', default="")


class AF_HairObject(PropertyGroup):
    role: EnumProperty(name="Hair role", items=[
        ('NONE', "None", "Not used by Tessera Projection Hair"),
        ('HAIR', "Hair curves", "This CURVES object is a Tessera hair groom"),
        ('COLLIDER', "Collision body", "Strands collide with this mesh"),
        ('SURFACE', "Scalp / surface", "Strand roots bind to this mesh"),
    ], default='NONE')
    bone: StringProperty(name="Bone", default="", description="Optional pose bone this object follows")
    stiffness: FloatProperty(name="Stiffness scale", default=1.0, min=0.0, max=4.0)
    damping: FloatProperty(name="Damping scale", default=1.0, min=0.0, max=4.0)
    is_guide_only: BoolProperty(name="Simulate every strand as a guide", default=False,
                               description="Ignore the scene guide ratio and simulate the whole groom")
    radius: FloatProperty(name="Curve radius", default=0.004, min=0.0, unit='LENGTH')
    gravity_scale: FloatProperty(name="Gravity scale", default=1.0, min=0.0, max=4.0,
                                 description="Per-groom gravity multiplier (multiplies the scene Gravity scale)")
    n_guides: IntProperty(name="Guide strands", default=0, min=0,
                          description="The first N curves are simulation guides and the rest are children (0 = pick by guide ratio)")


CLASSES = (AF_HairCollider, AF_HairScene, AF_HairObject)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Scene.arc_flow_hair = PointerProperty(type=AF_HairScene)
    bpy.types.Object.arc_flow_hair = PointerProperty(type=AF_HairObject)


def unregister():
    del bpy.types.Object.arc_flow_hair
    del bpy.types.Scene.arc_flow_hair
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
