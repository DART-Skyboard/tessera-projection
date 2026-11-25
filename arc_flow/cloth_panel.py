"""Tessera Projection cloth panel and the `draw_cloth_tab` hook for the vertical-tabbed UI."""
import bpy
from bpy.types import Panel, UIList

from . import panels

_Base = panels._Base


def _cl(ctx):
    return getattr(ctx.scene, 'arc_flow_cloth', None)


def _garment(cl):
    if cl is None or not cl.garments:
        return None
    i = max(0, min(int(cl.index), len(cl.garments) - 1))
    return cl.garments[i]


def draw_cloth_tab(layout, context):
    """Draw the Cloth category UI.  Mounted on the Tessera N-panel CLOTH tab via ui_tabs.register_tab_drawer."""
    cl = _cl(context)
    if cl is None:
        layout.label(text="Cloth properties are not registered.")
        return
    g = _garment(cl)

    # live transport
    row = layout.row(align=True)
    row.scale_y = 1.4
    if cl.live:
        row.operator("arcflow.cloth_live_stop", icon='PAUSE', text="Stop Live")
    else:
        row.operator("arcflow.cloth_live_start", icon='PLAY', text="Start Live")
    row.operator("arcflow.cloth_step", icon='FRAME_NEXT', text="Step")
    row.operator("arcflow.cloth_reset", icon='LOOP_BACK', text="Reset")

    box = layout.box()
    box.label(text=cl.status or "Cloth solver is idle.", icon='INFO')
    meta = box.row(align=True)
    if cl.last_ms > 0:
        meta.label(text="%.1f ms/step" % cl.last_ms)
    if cl.last_fps > 0:
        meta.label(text="%.0f fps" % cl.last_fps)
    box.prop(cl, "live_fps")

    # garments
    b = layout.box()
    b.label(text="Garments", icon='MOD_CLOTH')
    row = b.row()
    row.template_list("AF_UL_cloth_garments", "", cl, "garments", cl, "index", rows=3)
    col = row.column(align=True)
    col.operator("arcflow.cloth_assign", icon='ADD', text="")
    col.operator("arcflow.cloth_remove", icon='REMOVE', text="")
    row = b.row(align=True)
    row.operator("arcflow.cloth_assign", icon='MOD_CLOTH', text="Assign selected")
    row.operator("arcflow.cloth_auto_pin", icon='PINNED', text="Auto-pin")

    if g is not None:
        gb = b.box()
        gb.prop(g, "name")
        row = gb.row(align=True)
        row.prop(g, "obj", text="")
        row.prop(g, "enabled", text="")
        gb.prop(g, "armature")
        row = gb.row(align=True)
        row.prop(g, "pin_group")
        row.prop(g, "pin_mode", text="")
        gb.prop(g, "pin_threshold")
        row = gb.row(align=True)
        row.prop(g, "preset")
        row.operator("arcflow.cloth_apply_preset", text="", icon='PRESET')
        gb.prop(g, "quality")
        gb.prop(g, "follow_preset")
        phys = gb.column(align=True)
        phys.enabled = not g.follow_preset
        for k in ("density", "stretch", "shear", "bend", "damping", "drag", "lift",
                  "thickness", "margin", "friction"):
            phys.prop(g, k)
        sb = gb.box()
        sb.prop(g, "self_collision", icon='MOD_CLOTH')
        sc = sb.column(align=True)
        sc.enabled = bool(g.self_collision)
        sc.prop(g, "self_collision_mode")
        sc.prop(g, "self_thickness")
        sc.prop(g, "self_friction")
        sc.prop_search(g, "self_ignore_group", g.obj, "vertex_groups") if g.obj is not None else None
        gb.prop(g, "override_quality")
        q = gb.column(align=True)
        q.enabled = g.override_quality
        q.prop(g, "substeps")
        q.prop(g, "iterations")
        q.prop(g, "use_dihedral")
        q.prop(g, "tessellation_edge")
        q.prop(g, "tessellation_cap")
        if g.n_verts:
            gb.label(text="%d vertices, %d pins, stretch %.2f %%, penetration %.2f mm"
                     % (g.n_verts, g.n_pins, 100.0 * g.last_stretch, 1000.0 * g.last_penetration))

    # colliders
    c = layout.box()
    c.label(text="Collision bodies", icon='MESH_UVSPHERE')
    row = c.row()
    row.template_list("AF_UL_cloth_colliders", "", cl, "colliders", cl, "collider_index", rows=2)
    col = row.column(align=True)
    col.operator("arcflow.cloth_add_collider", icon='ADD', text="")
    col.operator("arcflow.cloth_remove_collider", icon='REMOVE', text="")
    c.operator("arcflow.cloth_add_collider", icon='MESH_UVSPHERE', text="Add selected as collider")

    # wind / environment
    w = layout.box()
    w.label(text="Wind and gravity", icon='FORCE_WIND')
    w.prop(cl, "use_scene_wind")
    col = w.column(align=True)
    col.enabled = not cl.use_scene_wind
    col.prop(cl, "wind")
    col.prop(cl, "gravity")
    col.prop(cl, "air_density")
    w.prop(cl, "use_fluid_velocity")

    # bake / export
    k = layout.box()
    k.label(text="Bake and game-engine export", icon='FILE_CACHE')
    k.prop(cl, "use_scene_range")
    if not cl.use_scene_range:
        row = k.row(align=True)
        row.prop(cl, "frame_start")
        row.prop(cl, "frame_end")
    k.prop(cl, "bake_mode")
    k.prop(cl, "bake_path")
    k.prop(cl, "bake_name")
    row = k.row(align=True)
    row.operator("arcflow.cloth_bake", icon='FREEZE', text="Bake")
    row.operator("arcflow.cloth_export_package", icon='EXPORT', text="Export package")

    # examples
    ex = layout.box()
    ex.label(text="Examples", icon='FILE_BLEND')
    ex.operator("arcflow.cloth_example", icon='MOD_CLOTH', text="Example: Cape drape").which = 'CAPE'


class AF_UL_cloth_garments(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.label(text="", icon='MOD_CLOTH')
        row.prop(item, "name", text="", emboss=False)
        if item.n_verts:
            row.label(text="%d v" % item.n_verts)
        row.prop(item, "enabled", text="")


class AF_UL_cloth_colliders(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.label(text="", icon='MESH_UVSPHERE')
        row.prop(item, "name", text="", emboss=False)
        if item.obj is not None:
            row.label(text=item.obj.name)
        row.prop(item, "enabled", text="")


class AF_PT_cloth(_Base, Panel):
    bl_idname = "AF_PT_cloth"
    bl_label = "Cloth"
    bl_order = 80

    def draw(self, ctx):
        draw_cloth_tab(self.layout, ctx)


CLASSES = (AF_UL_cloth_garments, AF_UL_cloth_colliders, AF_PT_cloth)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    try:
        from . import ui_tabs
        ui_tabs.register_tab_drawer('CLOTH', draw_cloth_tab)
    except Exception:
        pass


def unregister():
    try:
        from . import ui_tabs
        ui_tabs.unregister_tab_drawer('CLOTH', draw_cloth_tab)
    except Exception:
        pass
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
