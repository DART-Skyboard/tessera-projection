"""Tessera Projection Muscle tab (sidebar > Tessera > Muscle) and the `draw_muscle_tab` hook for the vertical-tab UI.

Layout, top to bottom (every section is a collapsible layout panel):

  Character        skin mesh + skeleton, Use Selected, Example biped, status line
  Preset           Biped / Quadruped / Custom creature, variant (generic, human male / female, feline ...), detail level from
                   Cartoon up to Cinematic / scientific, custom creature counts (heads, arms, legs, tails, fingers, toes,
                   joints per limb)
  Region cards     one stacked card per body region; each card lists every muscle of that region at the chosen detail with
                   its fixed colour swatch, the region's Initial Emphasis (and card Flex / Accent multipliers); the active
                   muscle opens the Muscle Pencil, Complete, Flex / Accent sliders and optional muscle / bone geometry
  Solve            resolution factor (0.85 - 10, default 1.0) and Solve
  Outputs          Create Shape Keys, one-click Create Geometry Nodes (with a helper tip), animation intensity, GLB / FBX
                   export, JSON sidecar, control rig, export notes

Data lives in the lead-owned schema `Scene.arc_flow_muscle` (muscle_props.py).  The UI only reads it and calls operators.
"""
import bpy
from bpy.types import Panel, UIList

from . import panels

_Base = panels._Base


def _ms(ctx):
    return getattr(ctx.scene, 'arc_flow_muscle', None)


def _section(layout, idname, title, icon='NONE', default_closed=False):
    """Collapsible section: Blender 4.1+ layout panel (open state kept per region).  Returns the body layout or None when
    the section is closed.  Older Blender: a plain box that is always open."""
    if hasattr(layout, 'panel'):
        header, body = layout.panel(idname, default_closed=default_closed)
        header.label(text=title, icon=icon)
        return body
    box = layout.box()
    box.label(text=title, icon=icon)
    return box


def _marked(rg):
    """A muscle counts as marked once it is Completed or has paint / pencil lines / geometry on it."""
    return bool(rg.completed or rg.n_painted or rg.n_strokes or rg.muscle_obj or rg.bone_obj)


def _status_icon(rg):
    if rg.completed:
        return 'CHECKMARK'
    if rg.n_strokes or rg.n_painted or rg.muscle_obj or rg.bone_obj:
        return 'GREASEPENCIL'
    return 'DOT'


def _swatch(layout, rg):
    """Fixed colour swatch: muscle colours are static per catalog muscle, so the swatch is shown but not editable."""
    sub = layout.row(align=True)
    sub.ui_units_x = 1.2
    sub.enabled = False
    sub.prop(rg, "color", text="")


def _active_region(ms):
    if not ms.regions:
        return None
    return ms.regions[max(0, min(ms.index, len(ms.regions) - 1))]


# ----------------------------------------------------------------------------------------------------------------------
# sections
# ----------------------------------------------------------------------------------------------------------------------
def _draw_character(layout, ms):
    body = _section(layout, "tp_muscle_character", "Character", 'ARMATURE_DATA')
    if body is None:
        return
    col = body.column(align=True)
    col.prop(ms, "skin", text="Skin")
    col.prop(ms, "armature", text="Skeleton")
    row = body.row(align=True)
    row.operator("arcflow.muscle_assign", icon='RESTRICT_SELECT_OFF', text="Use Selected")
    row.operator("arcflow.muscle_example", icon='MOD_ARMATURE', text="Example Biped")


def _draw_preset(layout, ms, ctx):
    from .core import muscle_solver as MS
    from . import muscle_ui_ops as MUI
    body = _section(layout, "tp_muscle_preset", "Preset", 'PRESET')
    if body is None:
        return
    fam = MUI.preset_family(ms.preset)
    row = body.row(align=True)
    row.scale_y = 1.25
    for key, text, icon in (('BIPED', "Biped", 'OUTLINER_OB_ARMATURE'), ('QUADRUPED', "Quadruped", 'MOD_ARMATURE'),
                            ('CUSTOM', "Custom", 'OUTLINER_DATA_ARMATURE')):
        if not MUI.family_presets(key):
            continue
        row.operator("arcflow.muscle_ui_set_preset", text=text, icon=icon, depress=(fam == key)).family = key
    variants = MUI.family_presets(fam)
    if len(variants) > 1:
        row = body.row(align=True)
        for k in variants:
            row.operator("arcflow.muscle_ui_set_preset", text=MUI.variant_label(k), depress=(ms.preset == k)).preset = k

    body.label(text="Detail level", icon='MOD_MULTIRES')
    grid = body.grid_flow(row_major=True, columns=len(MS.DETAIL_ITEMS), even_columns=True, align=True)
    cur = None
    for d in MS.DETAIL_ITEMS:
        grid.operator("arcflow.muscle_ui_set_preset", text=d[1].replace(" / ", "/"), depress=(ms.detail == d[0])).detail = d[0]
        if ms.detail == d[0]:
            cur = d
    if cur is not None:
        hint = body.column(align=True)
        hint.scale_y = 0.8
        hint.label(text=cur[2][:70], icon='INFO')

    if fam == 'CUSTOM':
        sub = body.box()
        sub.label(text="Custom creature", icon='OUTLINER_OB_ARMATURE')
        grid = sub.grid_flow(row_major=True, columns=2, even_columns=True, align=True)
        for k in ("heads", "arms", "legs", "tails", "fingers", "toes"):
            grid.prop(ms, k)
        sub.prop(ms, "joints_per_limb")
    if MUI.cards_stale(ms):
        warn = body.row(align=True)
        warn.alert = True
        warn.operator("arcflow.muscle_ui_set_preset", icon='FILE_REFRESH', text="Rebuild cards for these settings")
    elif not ms.regions:
        body.operator("arcflow.muscle_ui_set_preset", icon='FILE_REFRESH', text="Build region cards")


def _draw_active_muscle(layout, ms, rg):
    """The open muscle on a card: Muscle Pencil workflow, Flex / Accent, optional geometry, advanced settings."""
    box = layout.box()
    head = box.row(align=True)
    _swatch(head, rg)
    head.label(text=rg.label, icon=_status_icon(rg))
    head.prop(rg, "completed", text="", icon='CHECKBOX_HLT' if rg.completed else 'CHECKBOX_DEHLT', emboss=False)

    row = box.row(align=True)
    row.scale_y = 1.3
    row.operator("arcflow.muscle_contour", icon='GREASEPENCIL', text="Muscle Pencil")
    row.operator("arcflow.muscle_complete", icon='CHECKMARK', text="Complete")
    row = box.row(align=True)
    row.operator("arcflow.muscle_ui_stroke_undo", icon='LOOP_BACK', text="Last Line")
    row.operator("arcflow.muscle_contour_selection", icon='RESTRICT_SELECT_OFF', text="From Selection")
    hint = box.column(align=True)
    hint.scale_y = 0.8
    hint.label(text="Draw a few lines along the muscle; line ends = tendons.")
    hint.label(text="%d lines, %d painted vertices, %d solved" % (rg.n_strokes, rg.n_painted, rg.n_instances))

    col = box.column(align=True)
    col.prop(rg, "flex", text="Flex Intensity", slider=True)
    col.prop(rg, "accent", text="Accent", slider=True)
    row = col.row(align=True)
    row.prop_enum(rg, "accent_end", 'A', text="Toward End A")
    row.prop_enum(rg, "accent_end", 'AUTO', text="Auto")
    row.prop_enum(rg, "accent_end", 'B', text="Toward End B")

    geo = box.column(align=True)
    geo.label(text="Optional geometry (either or both)", icon='MESH_DATA')
    geo.prop(rg, "muscle_obj", text="Muscle")
    geo.prop(rg, "bone_obj", text="Bone")

    adv = _section(box, "tp_muscle_adv", "Advanced", 'PREFERENCES', default_closed=True)
    if adv is not None:
        adv.prop(rg, "emphasis", text="Muscle Emphasis", slider=True)
        r2 = adv.row(align=True)
        r2.prop(rg, "tone")
        r2.prop(rg, "joint_range")
        adv.prop(rg, "preview", slider=True)
        if ms.armature is not None and ms.armature.type == 'ARMATURE':
            adv.prop_search(rg, "bone", ms.armature.data, "bones", text="Driving Bone")
        else:
            adv.prop(rg, "bone", text="Driving Bone")
        adv.prop(rg, "attach_bone_ends")


def _draw_card(layout, ms, card, idx, active, icon):
    from .core import muscle_solver as MS
    box = layout.box()
    head = box.row(align=True)
    head.prop(card, "expanded", text="", icon='DISCLOSURE_TRI_DOWN' if card.expanded else 'DISCLOSURE_TRI_RIGHT',
              emboss=False)
    done = sum(1 for i in idx if ms.regions[i].completed)
    head.label(text=card.label, icon=icon)
    tag = head.row(align=True)
    tag.alignment = 'RIGHT'
    tag.label(text="%d/%d" % (done, len(idx)), icon='CHECKMARK' if idx and done == len(idx) else 'NONE')
    if idx:
        # colour strip: the region's fixed muscle colours at a glance
        strip = box.row(align=True)
        strip.scale_y = 0.35
        strip.enabled = False
        for i in idx[:24]:
            strip.prop(ms.regions[i], "color", text="")
    if not card.expanded:
        return
    col = box.column(align=True)
    col.prop(card, "emphasis", text="Initial Emphasis", slider=True)
    row = col.row(align=True)
    row.prop(card, "flex", text="Flex ×", slider=True)
    row.prop(card, "accent", text="Accent ×", slider=True)

    if not idx:
        box.label(text="No muscles at this detail level", icon='INFO')
    groups = []
    for i in idx:
        g = ms.regions[i].group
        if g not in groups:
            groups.append(g)
    lst = box.column(align=True)
    for g in groups:
        if len(groups) > 1:
            lab = lst.row()
            lab.scale_y = 0.75
            lab.label(text=g)
        for i in idx:
            rg = ms.regions[i]
            if rg.group != g:
                continue
            row = lst.row(align=True)
            _swatch(row, rg)
            is_act = active is not None and i == ms.index
            op = row.operator("arcflow.muscle_select", text=rg.label, depress=is_act, icon=_status_icon(rg))
            op.index = i
            sl = row.row(align=True)
            sl.ui_units_x = 3.2
            sl.prop(rg, "flex", text="", slider=True)
            row.prop(rg, "enabled", text="")
            if is_act:
                _draw_active_muscle(lst, ms, rg)
    row = box.row(align=True)
    row.prop(card, "pick", text="")
    row.operator("arcflow.muscle_card_add", icon='ADD', text="Add").card = card.key


def _draw_cards(layout, ms):
    """Stacked region category cards (one per MS.CARDS entry; the UI follows whatever cards the lead defines)."""
    from .core import muscle_solver as MS
    body = _section(layout, "tp_muscle_cards", "Region Cards", 'GROUP_VERTEX')
    if body is None:
        return
    if not ms.regions:
        body.label(text="Pick a preset to list the muscles", icon='INFO')
        return
    have = {c.key for c in ms.cards}
    if any(c[0] not in have for c in MS.CARDS):
        body.operator("arcflow.muscle_cards_init", icon='ADD', text="Create Region Cards")
        return
    total = len(ms.regions)
    done = sum(1 for r in ms.regions if r.completed)
    row = body.row(align=True)
    row.label(text="%d muscles, %d complete" % (total, done))
    sub = row.row(align=True)
    sub.alignment = 'RIGHT'
    sub.operator("arcflow.muscle_ui_cards_expand", text="", icon='FULLSCREEN_ENTER').expand = True
    sub.operator("arcflow.muscle_ui_cards_expand", text="", icon='FULLSCREEN_EXIT').expand = False
    active = _active_region(ms)
    icons = {c[0]: c[2] for c in MS.CARDS}
    order = {c[0]: n for n, c in enumerate(MS.CARDS)}
    for card in sorted(ms.cards, key=lambda c: order.get(c.key, 99)):
        if card.key not in order:
            continue
        idx = [i for i, rg in enumerate(ms.regions) if MS.card_of(rg.group) == card.key]
        if not idx and card.key not in ('HEAD', 'TORSO'):
            # regions absent for this creature (no tail, no arms ...): keep the card list short
            continue
        _draw_card(body, ms, card, idx, active, icons.get(card.key, 'GROUP_VERTEX'))


def _draw_pencil_settings(layout, ms, ctx):
    body = _section(layout, "tp_muscle_pencil", "Pencil and Paint", 'BRUSH_DATA', default_closed=True)
    if body is None:
        return
    wm = ctx.window_manager
    if hasattr(wm, 'tp_muscle_overlay'):
        body.prop(wm, "tp_muscle_overlay", text="Show Pencil Lines")
    row = body.row(align=True)
    row.operator("arcflow.muscle_contour_clear", icon='TRASH', text="Clear All Lines")
    body.label(text="Fill paint (optional, instead of lines)")
    body.prop(ms, "brush_radius")
    row = body.row(align=True)
    row.operator("arcflow.muscle_paint", icon='BRUSH_DATA', text="Paint")
    row.prop(ms, "brush_erase", toggle=True)
    row = body.row(align=True)
    row.operator("arcflow.muscle_paint_selected", icon='RESTRICT_SELECT_OFF', text="Assign Selection")
    row.operator("arcflow.muscle_paint_clear", icon='TRASH', text="Clear Paint")
    row = body.row(align=True)
    row.operator("arcflow.muscle_auto_seed", icon='BONE_DATA', text="Auto-seed From Bones")
    row = body.row(align=True)
    row.operator("arcflow.muscle_show_colors", icon='HIDE_OFF', text="Show Paint").solved = False
    row.operator("arcflow.muscle_show_colors", icon='HIDE_OFF', text="Show Solved").solved = True
    lst = _section(body, "tp_muscle_all", "All Muscles (list)", 'LINENUMBERS_ON', default_closed=True)
    if lst is not None:
        row = lst.row()
        row.template_list("AF_UL_muscle_regions", "", ms, "regions", ms, "index", rows=4)
        col = row.column(align=True)
        col.operator("arcflow.muscle_region_add", icon='ADD', text="")
        col.operator("arcflow.muscle_region_remove", icon='REMOVE', text="")


def _draw_solve(layout, ms):
    from .core import muscle_solver as MS
    body = _section(layout, "tp_muscle_solve", "Solve", 'PHYSICS')
    if body is None:
        return
    body.prop(ms, "resolution", text="Resolution Factor")
    hint = body.column(align=True)
    hint.scale_y = 0.8
    hint.label(text="1.0 = skin's own tessellation; %.2f softer (cartoon)" % MS.RESOLUTION_MIN)
    hint.label(text="up to %.0f finer fibre detail (cinematic)" % MS.RESOLUTION_MAX)
    row = body.row(align=True)
    row.scale_y = 1.4
    row.operator("arcflow.muscle_solve", icon='PLAY', text="Solve Muscles")
    if ms.solved:
        body.label(text="%d muscles solved in %.0f ms" % (ms.n_instances, ms.last_ms), icon='CHECKMARK')


def _draw_outputs(layout, ms):
    body = _section(layout, "tp_muscle_outputs", "Outputs", 'SHAPEKEY_DATA')
    if body is None:
        return
    col = body.column(align=True)
    col.scale_y = 1.2
    col.operator("arcflow.muscle_bake", icon='SHAPEKEY_DATA', text="Create Shape Keys")
    col.operator("arcflow.muscle_gn", icon='GEOMETRY_NODES', text="Create Geometry Nodes")
    tip = body.box().column(align=True)
    tip.scale_y = 0.8
    tip.label(text="Tip: after Solve, one click adds the Tessera Muscles", icon='INFO')
    tip.label(text="node group to the skin. In any Geometry Nodes editor")
    tip.label(text="find it under Add > Group and wire Flex / Accent in.")

    anim = _section(body, "tp_muscle_anim", "Animation Intensity", 'ARMATURE_DATA', default_closed=True)
    if anim is not None:
        anim.prop(ms, "state")
        row = anim.row(align=True)
        row.prop(ms, "gain_rest", text="Rest")
        row.prop(ms, "gain_walk", text="Walk")
        row.prop(ms, "gain_run", text="Run")
        anim.prop(ms, "use_drivers")
        anim.prop(ms, "use_jiggle")
        if ms.use_jiggle:
            row = anim.row(align=True)
            row.prop(ms, "jiggle_k")
            row.prop(ms, "jiggle_damping")

    exp = _section(body, "tp_muscle_export", "Export", 'EXPORT')
    if exp is not None:
        exp.prop(ms, "export_path", text="")
        row = exp.row(align=True)
        row.prop(ms, "export_format", expand=True)
        row = exp.row(align=True)
        row.operator("arcflow.muscle_export_game", icon='EXPORT', text="Export %s" % ms.export_format)
        row.operator("arcflow.muscle_export_json", icon='FILE_TEXT', text="JSON Sidecar")
        exp.operator("arcflow.muscle_control_rig", icon='CON_KINEMATIC', text="Build Control Rig")
        notes = _section(exp, "tp_muscle_export_notes", "Export Notes", 'HELP', default_closed=True)
        if notes is not None:
            c = notes.column(align=True)
            c.scale_y = 0.8
            for line in ("Create Shape Keys before exporting.",
                         "GLB: shape keys -> morph targets, skin weights,",
                         "  region colours (COLOR_0), muscle rig in node extras.",
                         "FBX: blend shapes, skin weights, vertex colours,",
                         "  muscle rig as user property tp_muscles.",
                         "Engines drive each morph from its bone angle and",
                         "  speed; JSON sidecar has weights + speed curve.",
                         "Muscle data is also saved inside the .blend."):
                c.label(text=line)


def draw_muscle_tab(layout, context):
    """Draw the Muscle category UI.  Mounted on the Tessera N-panel MUSCLE tab via ui_tabs.register_tab_drawer."""
    ms = _ms(context)
    if ms is None:
        layout.label(text="Muscle properties are not registered.")
        return
    info = layout.row()
    info.label(text=(ms.status or "")[:80], icon='INFO')
    _draw_character(layout, ms)
    _draw_preset(layout, ms, context)
    _draw_cards(layout, ms)
    _draw_pencil_settings(layout, ms, context)
    _draw_solve(layout, ms)
    _draw_outputs(layout, ms)


class AF_UL_muscle_regions(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        _swatch(row, item)
        row.label(text=item.label, icon=_status_icon(item))
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
    from . import muscle_ui_ops
    muscle_ui_ops.register()
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
    from . import muscle_ui_ops
    muscle_ui_ops.unregister()
