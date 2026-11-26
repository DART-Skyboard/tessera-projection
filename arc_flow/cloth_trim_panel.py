"""Tessera Projection Cloth tab sections: Cloth Trim, Accents, Seam Seal Control Solving.

Drawn inside the Cloth tab (after the main cloth controls) through ``ui_tabs.register_tab_drawer``;
each section folds with a disclosure arrow and starts expanded, like the other Tessera panels.
"""
import bpy
from bpy.types import Panel, UIList

from . import panels

_Base = panels._Base


def _ts(ctx):
    return getattr(ctx.scene, 'arc_flow_cloth_trim', None)


def _head(layout, ts, prop, title, icon):
    box = layout.box()
    row = box.row(align=True)
    row.prop(ts, prop, text="", icon='TRIA_DOWN' if getattr(ts, prop) else 'TRIA_RIGHT', emboss=False)
    row.label(text=title, icon=icon)
    return box, getattr(ts, prop)


def _active(coll, index):
    if not len(coll):
        return None
    return coll[max(0, min(int(index), len(coll) - 1))]


def draw_trim_section(layout, ts):
    box, open_ = _head(layout, ts, "show_trim", "Cloth Trim", 'MOD_SOLIDIFY')
    if not open_:
        return
    row = box.row()
    row.template_list("AF_UL_cloth_trims", "", ts, "trims", ts, "trim_index", rows=2)
    col = row.column(align=True)
    col.operator("arcflow.trim_add", icon='ADD', text="")
    col.operator("arcflow.trim_remove", icon='REMOVE', text="")
    row = box.row(align=True)
    for k, label in (('HEM', "Hem"), ('PIPING', "Piping"), ('LACE', "Lace"), ('BAND', "Belt"), ('COLLAR', "Collar")):
        row.operator("arcflow.trim_add", text=label).kind = k
    t = _active(ts.trims, ts.trim_index)
    if t is None:
        box.label(text="Finishes a hem, border, belt or collar edge of the active garment.")
        return
    b = box.box()
    row = b.row(align=True)
    row.prop(t, "name", text="")
    row.prop(t, "enabled", text="")
    b.prop(t, "kind")
    b.prop(t, "source")
    if t.source == 'BOUNDARY':
        b.prop(t, "loop_index")
    elif t.source == 'GROUP':
        row = b.row(align=True)
        row.prop(t, "group", text="Group")
        row.operator("arcflow.trim_group_from_selection", text="", icon='GROUP_VERTEX').group = t.group
    b.prop(t, "garment", text="Garment")
    col = b.column(align=True)
    col.prop(t, "width")
    col.prop(t, "thickness")
    col.prop(t, "mass_per_m", text="Mass (kg/m)")
    col = b.column(align=True)
    col.prop(t, "stiffness")
    col.prop(t, "bend")
    b.prop(t, "edge_weight")
    col = b.column(align=True)
    col.prop(t, "flare")
    col.prop(t, "flare_count")
    row = b.row(align=True)
    row.prop(t, "show_strip")
    row.prop(t, "material", text="")
    if t.n_verts:
        b.label(text="%d vertices follow the edge" % t.n_verts)


def draw_accent_section(layout, ts):
    box, open_ = _head(layout, ts, "show_accents", "Accents", 'OUTLINER_OB_POINTCLOUD')
    if not open_:
        return
    row = box.row()
    row.template_list("AF_UL_cloth_accents", "", ts, "accents", ts, "accent_index", rows=2)
    col = row.column(align=True)
    col.operator("arcflow.accent_add", icon='ADD', text="")
    col.operator("arcflow.accent_remove", icon='REMOVE', text="")
    row = box.row(align=True)
    for k, label in (('BUTTON', "Button"), ('JEWEL', "Jewel"), ('SEQUIN', "Sequin"), ('CHARM', "Charm"), ('TASSEL', "Tassel")):
        row.operator("arcflow.accent_add", text=label).kind = k
    a = _active(ts.accents, ts.accent_index)
    if a is None:
        box.label(text="Buttons, bows, jewels, chains and charms bound to the cloth surface.")
        return
    b = box.box()
    row = b.row(align=True)
    row.prop(a, "name", text="")
    row.prop(a, "enabled", text="")
    b.prop(a, "kind")
    row = b.row(align=True)
    row.prop(a, "source", text="Object")
    row.operator("arcflow.accent_use_selected", text="", icon='RESTRICT_SELECT_OFF')
    b.prop(a, "placement")
    if a.placement in ('CURVE', 'VERTICES') or (a.placement == 'DENSITY'):
        row = b.row(align=True)
        row.prop(a, "group", text="Group")
        row.operator("arcflow.trim_group_from_selection", text="", icon='GROUP_VERTEX').group = a.group
    if a.placement in ('CURVE',):
        b.prop(a, "count")
    elif a.placement == 'DENSITY':
        col = b.column(align=True)
        col.prop(a, "density")
        col.prop(a, "seed")
    col = b.column(align=True)
    col.prop(a, "size")
    col.prop(a, "offset")
    b.prop(a, "align")
    b.prop(a, "garment", text="Garment")
    b.prop(a, "dynamic")
    if a.dynamic:
        col = b.column(align=True)
        col.prop(a, "chain_segments")
        col.prop(a, "chain_length")
        col.prop(a, "chain_stiffness")
        col.prop(a, "chain_damping")
        col.prop(a, "show_chain")
        row = b.row(align=True)
        row.prop(a, "collide")
        sub = row.row(align=True)
        sub.enabled = a.collide
        sub.prop(a, "collide_radius", text="")
    b.prop(a, "material")
    if a.n_instances:
        b.label(text="%d accents bound" % a.n_instances)


def draw_seam_section(layout, ts):
    box, open_ = _head(layout, ts, "show_seams", "Seam Seal Control Solving", 'LINKED')
    if not open_:
        return
    row = box.row()
    row.template_list("AF_UL_cloth_seams", "", ts, "seams", ts, "seam_index", rows=2)
    col = row.column(align=True)
    col.operator("arcflow.seam_add", icon='ADD', text="")
    col.operator("arcflow.seam_remove", icon='REMOVE', text="")
    row = box.row(align=True)
    for k, label in (('ZIPPER', "Zipper"), ('BUTTONS', "Placket"), ('VELCRO', "Velcro"), ('MAGNETIC', "Magnet"), ('LACES', "Laces")):
        row.operator("arcflow.seam_add", text=label).mode = k
    s = _active(ts.seams, ts.seam_index)
    if s is None:
        box.label(text="Open and close seams with a slider: zipper, placket, Velcro, magnets, laces.")
        return
    b = box.box()
    row = b.row(align=True)
    row.prop(s, "name", text="")
    row.prop(s, "enabled", text="")
    b.prop(s, "mode")
    row = b.row(align=True)
    row.prop(s, "group", text="Seam")
    row.operator("arcflow.trim_group_from_selection", text="", icon='GROUP_VERTEX').group = s.group
    row = b.row(align=True)
    row.prop(s, "group_a", text="Side A")
    row.prop(s, "group_b", text="Side B")
    b.prop(s, "garment", text="Garment")
    b.prop(s, "progress", slider=True)
    row = b.row(align=True)
    row.operator("arcflow.seam_set", text="Open", icon='REMOVE').value = 0.0
    row.operator("arcflow.seam_set", text="Close", icon='ADD').value = 1.0
    b.operator("arcflow.seam_animate", icon='KEY_HLT', text="Key zip animation")
    col = b.column(align=True)
    col.prop(s, "strength")
    col.prop(s, "reverse")
    row = b.row(align=True)
    row.prop(s, "use_tear")
    sub = row.row(align=True)
    sub.enabled = s.use_tear
    sub.prop(s, "tear_gap", text="")
    col = b.column(align=True)
    col.prop(s, "gather")
    col.prop(s, "ruffle")
    row = b.row(align=True)
    row.prop(s, "pull", text="Pull")
    row.operator("arcflow.seam_pull_add", text="", icon='EMPTY_ARROWS')
    b.prop(s, "drive_from_pull")
    if s.n_pairs:
        b.label(text="%d pairs sewn%s" % (s.n_pairs, ", %d torn" % s.n_torn if s.n_torn else ""))


def draw_trim_tab(layout, context):
    ts = _ts(context)
    if ts is None:
        layout.label(text="Trim properties are not registered.")
        return
    draw_trim_section(layout, ts)
    draw_accent_section(layout, ts)
    draw_seam_section(layout, ts)
    row = layout.row(align=True)
    row.operator("arcflow.trim_refresh", icon='FILE_REFRESH', text="Apply")
    row.operator("arcflow.trim_example", icon='MOD_CLOTH', text="Example: Jacket with zipper")
    if ts.status:
        layout.label(text=ts.status, icon='INFO')


class AF_UL_cloth_trims(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.label(text="", icon='MOD_SOLIDIFY')
        row.prop(item, "name", text="", emboss=False)
        row.label(text=item.kind.title())
        row.prop(item, "enabled", text="")


class AF_UL_cloth_accents(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.label(text="", icon='OUTLINER_OB_POINTCLOUD')
        row.prop(item, "name", text="", emboss=False)
        if item.n_instances:
            row.label(text="%d" % item.n_instances)
        row.prop(item, "enabled", text="")


class AF_UL_cloth_seams(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.label(text="", icon='LINKED')
        row.prop(item, "name", text="", emboss=False)
        row.prop(item, "progress", text="", slider=True)
        row.prop(item, "enabled", text="")


CLASSES = (AF_UL_cloth_trims, AF_UL_cloth_accents, AF_UL_cloth_seams)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    try:
        from . import ui_tabs
        ui_tabs.register_tab_drawer('CLOTH', draw_trim_tab, replace=False)
    except Exception:
        pass


def unregister():
    try:
        from . import ui_tabs
        ui_tabs.unregister_tab_drawer('CLOTH', draw_trim_tab)
    except Exception:
        pass
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
