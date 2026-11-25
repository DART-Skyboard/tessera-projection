"""Tessera Projection Muscle panel and the `draw_muscle_tab` hook for the vertical-tab UI."""
import bpy
from bpy.types import Panel, UIList

from . import panels

_Base = panels._Base


def _ms(ctx):
    return getattr(ctx.scene, 'arc_flow_muscle', None)


def draw_muscle_tab(layout, context):
    """Draw the Muscle category UI.  Mounted on the Tessera N-panel MUSCLE tab via ui_tabs.register_tab_drawer."""
    ms = _ms(context)
    if ms is None:
        layout.label(text="Muscle properties are not registered.")
        return

    # skin / skeleton
    b = layout.box()
    b.label(text="Skin and skeleton", icon='ARMATURE_DATA')
    b.prop(ms, "skin")
    b.prop(ms, "armature")
    row = b.row(align=True)
    row.operator("arcflow.muscle_assign", icon='RESTRICT_SELECT_OFF', text="Use selected")
    row.operator("arcflow.muscle_example", icon='MOD_ARMATURE', text="Example biped")
    info = layout.box()
    info.label(text=ms.status or "", icon='INFO')

    # preset
    p = layout.box()
    p.label(text="Preset", icon='PRESET')
    row = p.row(align=True)
    row.prop(ms, "preset", text="")
    row.prop(ms, "detail", text="")
    c = p.column(align=True)
    if ms.preset == 'CUSTOM':
        c.label(text="Custom creature: count each category")
    else:
        c.label(text="Switch between Biped and Quadruped here")
    sub = p.box()
    sub.label(text="Creature setup", icon='OUTLINER_OB_ARMATURE')
    grid = sub.grid_flow(row_major=True, columns=2, align=True)
    for k in ("heads", "arms", "legs", "tails", "fingers", "toes"):
        grid.prop(ms, k)
    sub.prop(ms, "joints_per_limb")
    p.operator("arcflow.muscle_apply_preset", icon='FILE_REFRESH', text="Apply preset (%d regions)" % len(ms.regions))

    # regions
    r = layout.box()
    r.label(text="Muscle regions", icon='GROUP_VERTEX')
    row = r.row()
    row.template_list("AF_UL_muscle_regions", "", ms, "regions", ms, "index", rows=6)
    col = row.column(align=True)
    col.operator("arcflow.muscle_region_add", icon='ADD', text="")
    col.operator("arcflow.muscle_region_remove", icon='REMOVE', text="")
    if ms.regions:
        g = ms.regions[max(0, min(ms.index, len(ms.regions) - 1))]
        gb = r.box()
        gb.label(text="%s (%s)" % (g.label, g.group))
        gb.prop(g, "color")
        gb.prop(g, "flex")
        gb.prop(g, "accent")
        gb.prop(g, "emphasis")
        gb.prop(g, "tone")
        gb.prop(g, "joint_range")
        gb.prop(g, "preview")
        gb.prop_search(g, "bone", ms.armature.data, "bones", text="Driving bone") if ms.armature is not None else gb.prop(g, "bone")
        row = gb.row(align=True)
        row.prop(g, "accent_end", text="Accent")
        row.prop(g, "attach_bone_ends", text="Bone ends")
        gb.prop(g, "muscle_obj")
        gb.prop(g, "bone_obj")
        if g.n_painted or g.n_instances:
            gb.label(text="%d painted vertices, %d muscles solved" % (g.n_painted, g.n_instances))

    # paint
    pb = layout.box()
    pb.label(text="Paint regions", icon='BRUSH_DATA')
    pb.prop(ms, "brush_radius")
    row = pb.row(align=True)
    row.prop(ms, "brush_erase", toggle=True)
    row.operator("arcflow.muscle_show_colors", icon='HIDE_OFF', text="Show colours")
    row = pb.row(align=True)
    row.scale_y = 1.3
    row.operator("arcflow.muscle_paint", icon='BRUSH_DATA', text="Paint")
    row.operator("arcflow.muscle_paint_selected", icon='RESTRICT_SELECT_OFF', text="Assign selection")
    row = pb.row(align=True)
    row.operator("arcflow.muscle_auto_seed", icon='BONE_DATA', text="Auto-seed")
    row.operator("arcflow.muscle_paint_clear", icon='TRASH', text="Clear")

    # contours
    cb = layout.box()
    cb.label(text="Contour lines", icon='CURVE_BEZCURVE')
    cb.label(text="A few lines along each muscle; tendons sit at the ends.")
    row = cb.row(align=True)
    row.operator("arcflow.muscle_contour", icon='CURVE_BEZCURVE', text="Draw")
    row.operator("arcflow.muscle_contour_selection", icon='RESTRICT_SELECT_OFF', text="From selection")
    row.operator("arcflow.muscle_contour_clear", icon='TRASH', text="")

    # solve
    sb = layout.box()
    sb.label(text="Solve", icon='PHYSICS')
    sb.prop(ms, "resolution")
    row = sb.row(align=True)
    row.label(text="Cartoon < 1    Cinematic > 1")
    sb.operator("arcflow.muscle_solve", icon='PLAY', text="Solve muscles")
    if ms.solved:
        sb.label(text="%d muscles, %.0f ms (offline solve)" % (ms.n_instances, ms.last_ms))

    # animation state
    ab = layout.box()
    ab.label(text="Animation intensity", icon='ARMATURE_DATA')
    ab.prop(ms, "state")
    row = ab.row(align=True)
    row.prop(ms, "gain_rest", text="Rest")
    row.prop(ms, "gain_walk", text="Walk")
    row.prop(ms, "gain_run", text="Run")
    ab.prop(ms, "use_drivers")
    row = ab.row(align=True)
    row.prop(ms, "use_jiggle")
    if ms.use_jiggle:
        row = ab.row(align=True)
        row.prop(ms, "jiggle_k")
        row.prop(ms, "jiggle_damping")

    # outputs
    ob = layout.box()
    ob.label(text="Outputs", icon='SHAPEKEY_DATA')
    ob.operator("arcflow.muscle_bake", icon='SHAPEKEY_DATA', text="Bake to shape keys")
    ob.operator("arcflow.muscle_gn", icon='GEOMETRY_NODES', text="Create Geometry Nodes")
    ob.label(text="After solving: adds Tessera Muscles to the modifier stack.")
    ob.prop(ms, "export_path")
    ob.prop(ms, "export_format")
    row = ob.row(align=True)
    row.operator("arcflow.muscle_export_game", icon='EXPORT', text="Export GLB / FBX")
    row.operator("arcflow.muscle_export_json", icon='FILE_TEXT', text="JSON sidecar")
    ob.operator("arcflow.muscle_control_rig", icon='CON_KINEMATIC', text="Build control rig")


class AF_UL_muscle_regions(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "color", text="")
        row.label(text=item.label)
        row.prop(item, "flex", text="", slider=True)
        row.prop(item, "enabled", text="")


class AF_PT_muscle(_Base, Panel):
    bl_idname = "AF_PT_muscle"
    bl_label = "Muscle"
    bl_order = 85

    def draw(self, ctx):
        draw_muscle_tab(self.layout, ctx)


CLASSES = (AF_UL_muscle_regions, AF_PT_muscle)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    try:
        from . import ui_tabs
        ui_tabs.register_tab_drawer('MUSCLE', draw_muscle_tab, replace=True)
    except Exception:
        pass


def unregister():
    try:
        from . import ui_tabs
        ui_tabs.unregister_tab_drawer('MUSCLE', draw_muscle_tab)
    except Exception:
        pass
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
