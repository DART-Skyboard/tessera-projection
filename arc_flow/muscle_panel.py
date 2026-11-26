"""Tessera Projection Muscle panel and the `draw_muscle_tab` hook for the vertical-tab UI."""
import bpy
from bpy.types import Panel, UIList

from . import panels

_Base = panels._Base


def _ms(ctx):
    return getattr(ctx.scene, 'arc_flow_muscle', None)


def _draw_muscle_detail(box, ms, g):
    box.prop(g, "flex", text="Flex intensity")
    box.prop(g, "accent", text="Accent intensity")
    row = box.row(align=True)
    row.prop(g, "accent_end", text="Toward")
    row.prop(g, "attach_bone_ends", text="Bone ends")
    box.prop(g, "emphasis", text="Initial emphasis")
    row = box.row(align=True)
    row.prop(g, "tone")
    row.prop(g, "joint_range")
    box.prop(g, "preview")
    if ms.armature is not None:
        box.prop_search(g, "bone", ms.armature.data, "bones", text="Driving bone")
    else:
        box.prop(g, "bone")
    box.prop(g, "muscle_obj", text="Muscle geometry")
    box.prop(g, "bone_obj", text="Bone geometry")
    box.label(text="%d lines, %d painted vertices, %d solved" % (g.n_strokes, g.n_painted, g.n_instances))


def _draw_cards(layout, ms):
    """Stacked region category cards: each shows its muscles' colour strokes, card sliders and the muscle list."""
    from .core import muscle_solver as MS
    if len(ms.cards) < len(MS.CARDS):
        layout.operator("arcflow.muscle_cards_init", icon='ADD', text="Create region cards")
        return
    active = ms.regions[ms.index] if 0 <= ms.index < len(ms.regions) else None
    icons = {c[0]: c[2] for c in MS.CARDS}
    for card in ms.cards:
        idx = [i for i, rg in enumerate(ms.regions) if MS.card_of(rg.group) == card.key]
        box = layout.box()
        head = box.row(align=True)
        head.prop(card, "expanded", text="", icon='TRIA_DOWN' if card.expanded else 'TRIA_RIGHT', emboss=False)
        done = sum(1 for i in idx if ms.regions[i].completed)
        head.label(text="%s  (%d/%d complete)" % (card.label, done, len(idx)), icon=icons.get(card.key, 'GROUP_VERTEX'))
        # colour-coded muscle strokes of this region (one swatch per muscle, static catalog colours)
        if idx:
            strip = box.row(align=True)
            strip.scale_y = 0.5
            for i in idx[:16]:
                strip.prop(ms.regions[i], "color", text="")
        if not card.expanded:
            continue
        col = box.column(align=True)
        col.prop(card, "emphasis", slider=True)
        col.prop(card, "flex", slider=True)
        col.prop(card, "accent", slider=True)
        lst = box.column(align=True)
        for i in idx:
            rg = ms.regions[i]
            row = lst.row(align=True)
            row.prop(rg, "color", text="")
            is_act = active is not None and rg.key == active.key
            op = row.operator("arcflow.muscle_select", text=rg.label, depress=is_act,
                              icon='CHECKMARK' if rg.completed else 'GREASEPENCIL')
            op.index = i
            row.prop(rg, "flex", text="", slider=True)
            row.prop(rg, "enabled", text="")
            if is_act:
                _draw_muscle_detail(box.box(), ms, rg)
        row = box.row(align=True)
        row.prop(card, "pick", text="")
        row.operator("arcflow.muscle_card_add", icon='ADD', text="Add").card = card.key


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
    row.scale_y = 1.2
    row.prop_enum(ms, "preset", 'BIPED', text="Biped")
    row.prop_enum(ms, "preset", 'QUADRUPED', text="Quadruped")
    row.prop_enum(ms, "preset", 'CUSTOM', text="Custom")
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

    # region category cards
    _draw_cards(layout, ms)

    # muscle pencil (lines + fill) for the active muscle
    pb = layout.box()
    if ms.regions:
        g = ms.regions[max(0, min(ms.index, len(ms.regions) - 1))]
        hdr = pb.row(align=True)
        hdr.prop(g, "color", text="")
        hdr.label(text="Muscle Pencil: %s" % g.label, icon='GREASEPENCIL')
    else:
        pb.label(text="Muscle Pencil", icon='GREASEPENCIL')
    pb.label(text="Draw a few lines along the muscle over the skin contour;")
    pb.label(text="the line ends are the tendon / joint attachments.")
    row = pb.row(align=True)
    row.scale_y = 1.3
    row.operator("arcflow.muscle_contour", icon='GREASEPENCIL', text="Muscle Pencil")
    row.operator("arcflow.muscle_complete", icon='CHECKMARK', text="Complete")
    row = pb.row(align=True)
    row.operator("arcflow.muscle_contour_selection", icon='RESTRICT_SELECT_OFF', text="Line from selection")
    row.operator("arcflow.muscle_contour_clear", icon='TRASH', text="")
    sub = pb.box()
    sub.label(text="Fill paint (optional)", icon='BRUSH_DATA')
    sub.prop(ms, "brush_radius")
    row = sub.row(align=True)
    row.prop(ms, "brush_erase", toggle=True)
    row.operator("arcflow.muscle_show_colors", icon='HIDE_OFF', text="Show paint").solved = False
    row.operator("arcflow.muscle_show_colors", icon='HIDE_OFF', text="Show solved").solved = True
    row = sub.row(align=True)
    row.operator("arcflow.muscle_paint", icon='BRUSH_DATA', text="Paint")
    row.operator("arcflow.muscle_paint_selected", icon='RESTRICT_SELECT_OFF', text="Assign selection")
    row = sub.row(align=True)
    row.operator("arcflow.muscle_auto_seed", icon='BONE_DATA', text="Auto-seed")
    row.operator("arcflow.muscle_paint_clear", icon='TRASH', text="Clear")

    # full region list (all muscles, any card)
    r = layout.box()
    r.label(text="All muscles", icon='GROUP_VERTEX')
    row = r.row()
    row.template_list("AF_UL_muscle_regions", "", ms, "regions", ms, "index", rows=4)
    col = row.column(align=True)
    col.operator("arcflow.muscle_region_add", icon='ADD', text="")
    col.operator("arcflow.muscle_region_remove", icon='REMOVE', text="")

    # solve
    sb = layout.box()
    sb.label(text="Solve", icon='PHYSICS')
    sb.prop(ms, "resolution")
    row = sb.row(align=True)
    row.label(text="Cartoon < 1    Cinematic > 1")
    sb.label(text="1.0 = skin tessellation; 0.85 soft ... 10 very fine")
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
    ob.label(text="Tip: after solving, this adds the Tessera Muscles node", icon='INFO')
    ob.label(text="group to the skin; find it under Add > Group in any")
    ob.label(text="Geometry Nodes editor to wire it into your own nodes.")
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
        row.label(text=item.label, icon='CHECKMARK' if item.completed else 'NONE')
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
