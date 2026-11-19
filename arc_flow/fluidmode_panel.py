"""Fluid mode panels (Tessera Projection): the mode switch (Combustion | Fluid), the fluid list and its properties, solver
settings and the heat source / collider flags.  The Environment card is shared by both modes."""
import bpy
from bpy.types import Panel, UIList

from . import fluidmode_ops as FO
from . import panels
from .core import fluidmode as FM

_Base = panels._Base
_SHARED = {"AF_PT_env", "AF_PT_object_tab"}


def _fm(ctx):
    return getattr(ctx.scene, 'arc_flow_fm', None)


def _patch_polls():
    """Hide the combustion panels in Fluid mode (the Environment card stays): wrap each panel's poll."""
    for name in dir(panels):
        cls = getattr(panels, name)
        if not (isinstance(cls, type) and issubclass(cls, Panel)) or cls.__name__ in _SHARED or getattr(cls, '_fm_patched', False):
            continue
        orig = cls.__dict__.get('poll')
        orig_fn = orig.__func__ if isinstance(orig, classmethod) else None

        def make(orig_fn):
            def poll(cls, ctx):
                fm = _fm(ctx)
                if fm is not None and fm.mode == 'FLUID':
                    return False
                return orig_fn(cls, ctx) if orig_fn else True
            return classmethod(poll)
        cls.poll = make(orig_fn)
        cls._fm_patched = True


_patch_polls()


class _FMBase(_Base):
    @classmethod
    def poll(cls, ctx):
        fm = _fm(ctx)
        return fm is not None and fm.mode == 'FLUID'


class AF_PT_fm_mode(_Base, Panel):
    bl_idname = "AF_PT_fm_mode"
    bl_label = "Mode"
    bl_order = -10

    def draw(self, ctx):
        fm = _fm(ctx)
        row = self.layout.row(align=True)
        row.scale_y = 1.3
        row.prop(fm, "mode", expand=True)


class AF_UL_fm_fluids(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "color", text="")
        row.prop(item, "name", text="", emboss=False)
        row.label(text="", icon='MOD_FLUIDSIM' if item.kind == 'LIQUID' else 'FORCE_WIND')
        row.label(text="free" if item.no_container else "tank")


class AF_PT_fm_main(_FMBase, Panel):
    bl_idname = "AF_PT_fm_main"
    bl_label = "Fluid mode  |  Tessera Projection"

    def draw(self, ctx):
        l = self.layout
        fm = _fm(ctx)
        row = l.row(align=True)
        row.scale_y = 1.4
        if FO.RT.running:
            row.operator("arcflow.fm_stop", icon='SNAP_FACE', text="Stop")
        else:
            row.operator("arcflow.fm_build", icon='PLAY', text="Build")
        row.operator("arcflow.fm_fill", icon='MOD_FLUIDSIM', text="Fill / Settle")
        row.operator("arcflow.fm_reset", icon='FILE_REFRESH', text="Reset")
        row.operator("arcflow.fm_bake", icon='FREEZE', text="Bake")
        l.label(text=fm.status, icon='INFO')
        if FO.RT.msg:
            l.label(text=FO.RT.msg[:60], icon='ERROR')
        b = l.box()
        b.label(text="Demos", icon='MOD_FLUIDSIM')
        r = b.row(align=True)
        for which, lab in (('LAVA', "Lava Lamp"), ('PLUME', "Gas Plume"), ('TOWER', "Gas Tower")):
            op = r.operator("arcflow.fm_demo", text=lab)
            op.which = which
        l.label(text="Volumetric, particle-based (Lagrangian) fluids: render the particles as points,", icon='DOT')
        l.label(text="your own mesh per point, or a ray-marched volume.")


class AF_PT_fm_fluids(_FMBase, Panel):
    bl_idname = "AF_PT_fm_fluids"
    bl_label = "Fluids"
    bl_parent_id = "AF_PT_fm_main"

    def draw(self, ctx):
        l = self.layout
        fm = _fm(ctx)
        row = l.row()
        row.template_list("AF_UL_fm_fluids", "", fm, "fluids", fm, "index", rows=3)
        col = row.column(align=True)
        col.operator_menu_enum("arcflow.fm_add_fluid", "preset", text="", icon='ADD')
        col.operator("arcflow.fm_remove_fluid", text="", icon='REMOVE')
        l.operator("arcflow.fm_assign", icon='OBJECT_DATA')
        if not (0 <= fm.index < len(fm.fluids)):
            l.label(text="Add a fluid or assign the selected objects", icon='INFO')
            return
        fl = fm.fluids[fm.index]
        l.prop(fl, "preset")
        r = l.row(align=True)
        r.prop(fl, "kind", expand=True)
        b = l.box()
        b.label(text="Where it is", icon='OUTLINER_OB_MESH')
        b.prop(fl, "shape")
        b.prop(fl, "no_container")
        if not fl.no_container:
            b.prop(fl, "container")
            if fl.shape is None and fl.volume <= 0:
                b.prop(fl, "fill")
        if fl.kind == 'GAS':
            b.prop(fl, "count")
        b = l.box()
        b.label(text="Amount (0 = derive)", icon='MOD_MASS')
        c = b.column(align=True)
        for k in ("mass", "volume", "weight", "density"):
            c.prop(fl, k)
        if fl.d_particles:
            c = b.column(align=True)
            c.enabled = False
            c.label(text="Built: %d particles, %.4g kg, %.4g m3" % (fl.d_particles, fl.d_mass, fl.d_volume))
            c.label(text="density %.5g kg/m3" % fl.d_density)
        b = l.box()
        b.label(text="Physics", icon='PHYSICS')
        c = b.column(align=True)
        for k in ("temperature", "velocity", "visc", "cp", "k"):
            c.prop(fl, k)
        if fl.kind == 'LIQUID':
            c.prop(fl, "beta")
            c.prop(fl, "cohesion")
        c.prop(fl, "miscible")
        c.prop(fl, "cool")
        c.prop(fl, "drag")
        b = l.box()
        b.label(text="Look", icon='MATERIAL')
        b.prop(fl, "render_mode")
        b.prop(fl, "color")
        b.prop(fl, "temp_color")
        if fl.temp_color:
            r = b.row(align=True)
            r.prop(fl, "t_lo"); r.prop(fl, "t_hi")
        if fl.render_mode == 'POINTS':
            b.prop(fl, "mesh_obj")
            b.prop(fl, "mesh_collection")
        b.prop(fl, "particle_scale")
        if fl.render_mode == 'RAYMARCH':
            b.prop(fl, "rm_density")


class AF_PT_fm_solver(_FMBase, Panel):
    bl_idname = "AF_PT_fm_solver"
    bl_label = "Solver"
    bl_parent_id = "AF_PT_fm_main"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, ctx):
        l = self.layout
        fm = _fm(ctx)
        c = l.column(align=True)
        for k in ("fill_mode", "fill_steps", "heat_start", "max_particles", "spacing", "iterations", "max_substeps", "speedup", "gust", "ground_friction", "lotbcbp", "seed"):
            c.prop(fm, k)
        c.prop(fm, "ground_temp_on")
        if fm.ground_temp_on:
            c.prop(fm, "ground_temp")
        if FO.RT.solver is not None and FO.RT.step_times:
            l.label(text="Last step %.2f s (%d particles)" % (FO.RT.step_times[-1], FO.RT.solver.n))


class AF_PT_fm_object(_FMBase, Panel):
    bl_idname = "AF_PT_fm_object"
    bl_label = "Active object: heat source / collider"
    bl_parent_id = "AF_PT_fm_main"

    def draw(self, ctx):
        l = self.layout
        o = ctx.object
        if o is None:
            l.label(text="Select an object", icon='OBJECT_DATA')
            return
        p = o.arc_flow_fm
        l.label(text=o.name, icon='OBJECT_DATA')
        l.prop(p, "heat_on")
        if p.heat_on:
            l.prop(p, "heat_power"); l.prop(p, "heat_radius")
        if o.type == 'MESH':
            l.prop(p, "collider")
            if p.collider:
                l.prop(p, "friction")
        l.label(text="Rotating / moving colliders and containers drive the fluid", icon='DOT')


class AF_PT_fm_env(_FMBase, Panel):
    """The same Environment card as Combustion mode (gravity, pressure, ambient gas, temperature, wind, ground); it acts on every fluid."""
    bl_idname = "AF_PT_fm_env"
    bl_label = "Environment"
    bl_parent_id = "AF_PT_fm_main"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, ctx):
        panels.AF_PT_env.draw(self, ctx)


CLASSES = (AF_PT_fm_mode, AF_UL_fm_fluids, AF_PT_fm_main, AF_PT_fm_fluids, AF_PT_fm_env, AF_PT_fm_solver, AF_PT_fm_object)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
