"""Panels of the surface sources / fluid volumes and the Examples panel (Tessera Projection).

* AF_PT_examples   a collapsed-by-default panel at the top of the Tessera tab that lists every loadable example (both modes).  Other modules add
                   theirs with register_example(group, label, idname, **props).
* AF_PT_srf        'Surface sources and volumes' (Fluid mode): Add Fluid Volume, viscosity / emission surfaces with the buttons 'Use selected object',
                   'Edit faces' and 'Apply', and the settings of each source.
`prepare()` runs before the panels are registered and clears DEFAULT_CLOSED on every other Tessera panel, so everything but Examples opens expanded."""
import bpy
from bpy.types import Panel, UIList

from . import fluidmode_ops as FO
from . import panels
from . import surfaces_ops as SO

_Base = panels._Base

# group -> [(label, operator idname, {props})]
EXAMPLES = {
    "Combustion": [("Single engine", "arcflow.demo", {}), ("Twin-engine crossfeed", "arcflow.demo_twin", {})],
    "Fluid": [("Lava lamp", "arcflow.fm_demo", {"which": "LAVA"}), ("Colour mixing", "arcflow.fm_demo", {"which": "MIX"}),
              ("Gas plume", "arcflow.fm_demo", {"which": "PLUME"}), ("Gas tower", "arcflow.fm_demo", {"which": "TOWER"}),
              ("Emission nozzle on a viscous belt", "arcflow.srf_example", {"which": "EMIT"}),
              ("Hot and cool gas volumes", "arcflow.srf_example", {"which": "VOLUME"})],
}


def register_example(group, label, idname, **props):
    """Add an example button to the Examples panel (duplicates by label are ignored)."""
    lst = EXAMPLES.setdefault(group, [])
    if not any(e[0] == label for e in lst):
        lst.append((label, idname, props))


class AF_PT_examples(_Base, Panel):
    bl_idname = "AF_PT_examples"
    bl_label = "Examples"
    bl_order = -9
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, ctx):
        l = self.layout
        for group, items in EXAMPLES.items():
            b = l.box()
            b.label(text=group, icon='FILE_BLEND')
            col = b.column(align=True)
            for label, idname, props in items:
                op = col.operator(idname, text="Example: " + label)
                for k, v in props.items():
                    setattr(op, k, v)


def _fm(ctx):
    return getattr(ctx.scene, 'arc_flow_fm', None)


class AF_UL_srf(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.label(text="", icon={'VISCOSITY': 'FORCE_VISCOSITY', 'EMISSION': 'PARTICLES', 'VOLUME': 'MESH_CUBE'}.get(item.kind, 'DOT'))
        row.prop(item, "name", text="", emboss=False)
        if item.kind != 'VOLUME':
            row.label(text="%d faces" % item.face_count if item.face_count else "no faces")
        row.prop(item, "enabled", text="")


class AF_PT_srf(_Base, Panel):
    bl_idname = "AF_PT_srf"
    bl_label = "Surface sources and volumes"
    bl_parent_id = "AF_PT_fm_main"

    @classmethod
    def poll(cls, ctx):
        fm = _fm(ctx)
        return fm is not None and fm.mode == 'FLUID'

    def draw(self, ctx):
        l = self.layout
        s = ctx.scene.arc_flow_srf
        row = l.row(align=True)
        row.operator("arcflow.srf_add_volume", icon='MESH_CUBE')
        row = l.row(align=True)
        op = row.operator("arcflow.srf_add", text="Viscosity surface", icon='FORCE_VISCOSITY')
        op.kind = 'VISCOSITY'
        op = row.operator("arcflow.srf_add", text="Emission surface", icon='PARTICLES')
        op.kind = 'EMISSION'
        r = l.row()
        r.template_list("AF_UL_srf", "", s, "items", s, "index", rows=3)
        r.operator("arcflow.srf_remove", text="", icon='REMOVE')
        if not (0 <= s.index < len(s.items)):
            l.label(text="Add a volume or a surface", icon='INFO')
            return
        src = s.items[s.index]
        if src.kind == 'VOLUME':
            self.draw_volume(l, ctx, src)
        else:
            self.draw_surface(l, ctx, src)

    def draw_surface(self, l, ctx, src):
        b = l.box()
        b.prop(src, "name")
        row = b.row(align=True)
        row.prop(src, "obj", text="")
        row.operator("arcflow.srf_use_selected", icon='RESTRICT_SELECT_OFF')
        row = b.row(align=True)
        editing = src.obj is not None and src.obj.mode == 'EDIT'
        row.operator("arcflow.srf_edit_faces", icon='EDITMODE_HLT', depress=editing)
        row.operator("arcflow.srf_apply", icon='CHECKMARK')
        if src.face_count:
            b.label(text="%d faces stored, %.4g m2" % (src.face_count, src.face_area), icon='FACESEL')
        else:
            b.label(text="No faces stored yet: Edit faces, select, Apply", icon='INFO')
        c = l.box().column(align=True)
        if src.kind == 'VISCOSITY':
            c.label(text="Viscous coupling", icon='FORCE_VISCOSITY')
            for k in ("coef", "noslip", "reach"):
                c.prop(src, k)
            f = src.out_force
            if any(f):
                c.label(text="Drag on surface: (%.3g, %.3g, %.3g) N" % tuple(f))
        else:
            c.label(text="Emission", icon='PARTICLES')
            c.prop(src, "preset")
            for k in ("rate", "speed", "spread", "temperature", "velocity", "reverse", "t_start", "t_end", "max_particles"):
                c.prop(src, k)
            if src.out_count:
                c.label(text="Emitted so far: %d" % src.out_count)
            fl = SO.fluid_of(ctx.scene, src)
            if fl is not None:
                c.prop(fl, "color")
                c.label(text="Look and physics: fluid '%s' in the Fluids list" % fl.name)

    def draw_volume(self, l, ctx, src):
        b = l.box()
        b.prop(src, "name")
        row = b.row(align=True)
        row.prop(src, "obj", text="")
        row.operator("arcflow.srf_select_object", icon='RESTRICT_SELECT_OFF')
        if src.obj is not None:
            sz = SO.volume_size(src.obj)
            b.label(text="Box %.3g x %.3g x %.3g m  =  %.4g m3 (scale the Empty)" % (sz + (sz[0] * sz[1] * sz[2],)), icon='MESH_CUBE')
        c = l.box().column(align=True)
        c.label(text="Fluid in the box", icon='MOD_FLUIDSIM')
        for k in ("preset", "quantity", "density", "pressure", "temperature", "velocity0"):
            c.prop(src, k)
        fl = SO.fluid_of(ctx.scene, src)
        if fl is not None:
            c.prop(fl, "color")
            if fl.d_particles:
                c.label(text="Built: %d particles, %.4g kg" % (fl.d_particles, fl.d_mass))
            c.label(text="Look and physics: fluid '%s' in the Fluids list" % fl.name)


def prepare():
    """Clear DEFAULT_CLOSED on every Tessera panel except Examples (call before the panel modules register)."""
    from . import fluidmode_panel, retopo_panel, systems_ops
    for mod in (panels, fluidmode_panel, systems_ops, retopo_panel, __import__(__name__, fromlist=['x'])):
        for name in dir(mod):
            cls = getattr(mod, name)
            if isinstance(cls, type) and issubclass(cls, Panel) and getattr(cls, 'bl_category', "") == "Tessera" and cls.__name__ != "AF_PT_examples":
                if 'DEFAULT_CLOSED' in getattr(cls, 'bl_options', ()):
                    cls.bl_options = set(cls.bl_options) - {'DEFAULT_CLOSED'}


CLASSES = (AF_PT_examples, AF_UL_srf, AF_PT_srf)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
