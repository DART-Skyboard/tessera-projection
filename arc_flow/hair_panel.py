"""Tessera Projection Hair panel, plus `draw_hair_tab(layout, context)` so a vertical-tabbed
N-panel can mount this category under a HAIR tab without merge friction.
"""
import bpy
from bpy.types import Panel

from . import hair_ops as HO
from . import panels


_Base = panels._Base


def _hp(ctx):
    return getattr(ctx.scene, 'arc_flow_hair', None)


def _draw_pointers(l, hp):
    col = l.column(align=True)
    col.prop(hp, "hair", text="Hair curves", icon='OUTLINER_OB_CURVES')
    col.prop(hp, "collider", text="Collision body", icon='MESH_ICOSPHERE')
    col.prop(hp, "surface", text="Scalp / surface", icon='MESH_UVSPHERE')
    col.prop(hp, "armature", text="Armature", icon='ARMATURE_DATA')
    if hp.armature is not None:
        col.prop(hp, "bone", text="Bone", icon='BONE_DATA')


def _draw_live(l, hp):
    row = l.row(align=True)
    row.scale_y = 1.3
    if hp.live:
        row.operator("arcflow.hair_live_stop", icon='PAUSE', text="Stop Live")
    else:
        row.operator("arcflow.hair_live_start", icon='PLAY', text="Start Live")
    row.operator("arcflow.hair_step", text="Step", icon='FRAME_NEXT')
    row.operator("arcflow.hair_reset", text="Reset", icon='LOOP_BACK')
    r = l.row(align=True)
    r.prop(hp, "live", text="Live simulation")
    r.prop(hp, "follow_depsgraph", text="React to posing")
    if hp.status:
        l.label(text=hp.status, icon='INFO')


def _draw_solver(l, hp):
    col = l.column(align=True)
    col.prop(hp, "dt_mode", text="Time")
    if hp.dt_mode == 'FIXED':
        col.prop(hp, "dt")
    col.prop(hp, "substeps")
    col.prop(hp, "use_scene_gravity")
    if not hp.use_scene_gravity:
        col.prop(hp, "gravity", text="")
    col.prop(hp, "damping")
    col.separator()
    col.prop(hp, "stretch_iterations")
    col.prop(hp, "stretch_compliance")
    col.prop(hp, "bend_iterations")
    col.prop(hp, "bend_compliance")
    col.prop(hp, "rest_shape")
    row = col.row(align=True)
    row.prop(hp, "rest_shape_root")
    row.prop(hp, "rest_shape_tip")
    col.prop(hp, "pinned_prefix")


def _draw_collision(l, hp):
    col = l.column(align=True)
    col.prop(hp, "collision_radius")
    col.prop(hp, "collision_iterations")
    col.prop(hp, "collision_friction")
    col.prop(hp, "sdf_resolution")
    col.prop(hp, "sdf_level")
    col.prop(hp, "sdf_refine")
    row = col.row(align=True)
    row.prop(hp, "use_ground")
    if hp.use_ground:
        row.prop(hp, "ground_z")


def _draw_wind(l, hp):
    col = l.column(align=True)
    col.prop(hp, "use_env_wind")
    if not hp.use_env_wind:
        col.prop(hp, "wind", text="")
    col.prop(hp, "wind_drag")
    col.prop(hp, "wind_tangent")
    col.prop(hp, "fluid_coupling")


def _draw_guides(l, hp):
    col = l.column(align=True)
    col.prop(hp, "guide_ratio")
    col.prop(hp, "child_k")
    col.prop(hp, "child_smooth")
    col.prop(hp, "max_segment_budget")
    col.prop(hp, "max_points")
    col.label(text="Lower the guide ratio and the point budget for a 60 fps preview", icon='INFO')


def _draw_bake(l, hp):
    col = l.column(align=True)
    col.prop(hp, "bake_format")
    col.prop(hp, "use_scene_range")
    if not hp.use_scene_range:
        r = col.row(align=True)
        r.prop(hp, "frame_start")
        r.prop(hp, "frame_end")
    col.prop(hp, "bake_dir")
    col.prop(hp, "bake_name")
    row = col.row(align=True)
    row.scale_y = 1.2
    row.operator("arcflow.hair_bake", icon='FILE_CACHE')
    row.operator("arcflow.hair_export_alembic", icon='EXPORT')
    col.separator()
    col.prop(hp, "package_dir")
    col.operator("arcflow.hair_export_package", icon='PACKAGE')
    if hp.last_package:
        col.label(text=bpy.path.basename(hp.last_package), icon='CHECKMARK')
    if hp.last_cache:
        col.label(text=bpy.path.basename(hp.last_cache), icon='CHECKMARK')
    if hp.last_alembic:
        col.label(text=bpy.path.basename(hp.last_alembic), icon='CHECKMARK')


def _draw_object(l, ctx):
    ob = ctx.object
    if ob is None or not hasattr(ob, 'arc_flow_hair'):
        l.label(text="Select a curves or mesh object")
        return
    p = ob.arc_flow_hair
    l.prop(p, "role", text="Role")
    if p.role == 'HAIR':
        l.prop(p, "stiffness")
        l.prop(p, "damping")
        l.prop(p, "is_guide_only")
        l.prop(p, "radius")
        l.operator("arcflow.hair_set_role", text="Use as Hair curves").role = 'HAIR'
    elif p.role == 'COLLIDER':
        l.operator("arcflow.hair_set_role", text="Use as Collision body").role = 'COLLIDER'
    elif p.role == 'SURFACE':
        l.prop(p, "bone")
        l.operator("arcflow.hair_set_role", text="Use as Scalp / surface").role = 'SURFACE'
    else:
        row = l.row(align=True)
        if ob.type == 'CURVES':
            row.operator("arcflow.hair_set_role", text="Hair").role = 'HAIR'
        if ob.type == 'MESH':
            row.operator("arcflow.hair_set_role", text="Collider").role = 'COLLIDER'
            row.operator("arcflow.hair_set_role", text="Scalp").role = 'SURFACE'


def draw_hair_tab(layout, context):
    """Mountable Hair category for the vertical-tabbed Tessera N-panel (Worker 1).

    Every function in this tab is available at the same time as the rest of
    Tessera Projection: switching tabs only changes which settings are on
    screen, it does not enable or disable the solver.
    """
    hp = _hp(context)
    if hp is None:
        layout.label(text="Tessera Projection Hair is not registered")
        return
    l = layout
    box = l.box()
    box.label(text="Tessera Projection Hair", icon='OUTLINER_OB_CURVES')
    box.label(text="Paint and style with Blender's Empty Hair Curves; Tessera drives the physics.")
    row = box.row(align=True)
    row.scale_y = 1.2
    row.operator("arcflow.hair_add_empty", icon='ADD')
    row.operator("arcflow.hair_example", icon='FILE_NEW')
    row.operator("arcflow.hair_rebuild", text="", icon='FILE_REFRESH')
    _draw_pointers(l, hp)
    l.separator()
    _draw_live(l, hp)
    l.separator()
    col = l.column()
    col.label(text="Solver", icon='PHYSICS')
    _draw_solver(col, hp)
    col = l.column()
    col.label(text="Collision & tessellation", icon='MOD_SOLIDIFY')
    _draw_collision(col, hp)
    col = l.column()
    col.label(text="Wind & fluid", icon='FORCE_WIND')
    _draw_wind(col, hp)
    col = l.column()
    col.label(text="Guides & resolution", icon='PARTICLE_POINT')
    _draw_guides(col, hp)
    col = l.column()
    col.label(text="Bake & game-engine package", icon='FILE_CACHE')
    _draw_bake(col, hp)
    if context.object is not None:
        col = l.column()
        col.label(text="Active object", icon='OBJECT_DATA')
        _draw_object(col, context)


class AF_PT_hair(_Base, Panel):
    """Standalone Hair panel in the Tessera N-panel (also the source of `draw_hair_tab`)."""
    bl_idname = "AF_PT_hair"
    bl_label = "Hair  |  Tessera Projection"
    bl_options = {'DEFAULT_CLOSED'}
    bl_order = 40

    def draw(self, ctx):
        draw_hair_tab(self.layout, ctx)


class AF_PT_hair_object(Panel):
    bl_idname = "AF_PT_hair_object"
    bl_label = "Tessera Hair"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "object"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, ctx):
        ob = ctx.object
        return ob is not None and hasattr(ob, 'arc_flow_hair') and ob.type in {'CURVES', 'MESH', 'ARMATURE'}

    def draw(self, ctx):
        _draw_object(self.layout, ctx)


CLASSES = (AF_PT_hair, AF_PT_hair_object)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
