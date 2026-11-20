"""Shader presets UI (Tessera Projection): scene settings, assignment operators and the 'Shader' panel.

Works on any object with material slots (meshes, curves, text, metaballs, volumes, point clouds) and on Fluid mode's fluids (each fluid
has a ``shader`` choice: AUTO = the default of its fluid preset, OFF = leave its material alone, or any shader preset)."""
import bpy
from bpy.props import BoolProperty, EnumProperty, PointerProperty
from bpy.types import Operator, Panel, PropertyGroup

from . import panels
from . import shader_presets as SP

_PRESET_ENUM = [(k, v[0], v[3]) for k, v in SP.PRESETS.items()]
_FLUID_ENUM = [('AUTO', "Auto", "The default shader of this fluid's preset"), ('OFF', "Keep my material", "Do not touch the fluid's materials")] + _PRESET_ENUM


def _apply_fluids(self, context):
    try:
        SP.apply_fluid_shaders(context.scene)
    except Exception:
        pass


class AF_ShaderSettings(PropertyGroup):
    preset: EnumProperty(name="Shader", items=_PRESET_ENUM, default='WATER', description="Shader preset to assign / create")
    preview: BoolProperty(name="Preview mode", default=False, update=_apply_fluids,
                          description="Cheap Principled-only look (alpha instead of refraction, no volumes) for the viewport and quick EEVEE use")
    caustics: BoolProperty(name="Cycles shadow caustics", default=False,
                           description="Tag refractive objects as caustic casters (Cycles 3.2+): add a caustics receiver floor and light by hand or with the button below")


def _objs(ctx, which):
    if which == 'ACTIVE':
        return [ctx.object] if ctx.object is not None else []
    return list(ctx.selected_objects)


class AF_OT_shader_assign(Operator):
    bl_idname = "arcflow.shader_assign"
    bl_label = "Assign shader"
    bl_description = "Assign the shader preset to objects (all material slots; one is added when there is none)"
    bl_options = {'REGISTER', 'UNDO'}
    target: EnumProperty(items=[('SELECTED', "Selected", ""), ('ACTIVE', "Active", "")], default='SELECTED')
    preset: EnumProperty(name="Shader", items=_PRESET_ENUM, default='WATER', options={'SKIP_SAVE'})
    use_panel: BoolProperty(default=True, options={'SKIP_SAVE', 'HIDDEN'})

    def execute(self, ctx):
        sh = ctx.scene.arc_flow_sh
        key = sh.preset if self.use_panel else self.preset
        objs = _objs(ctx, self.target)
        if not objs:
            self.report({'WARNING'}, "No object to assign to")
            return {'CANCELLED'}
        mat, n = SP.assign_preset(objs, key, preview=sh.preview, caustics=sh.caustics, scene=ctx.scene)
        if n == 0:
            self.report({'WARNING'}, "No object with material slots")
            return {'CANCELLED'}
        self.report({'INFO'}, "%s assigned to %d object(s)" % (mat.name, n))
        return {'FINISHED'}


class AF_OT_shader_create(Operator):
    bl_idname = "arcflow.shader_create"
    bl_label = "Create shader from preset"
    bl_description = "Make a fresh, independent material from the preset (assigned to the active object when it has material slots)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        sh = ctx.scene.arc_flow_sh
        key = sh.preset
        base = SP.material_name(key, sh.preview)
        name = base
        i = 1
        while name in bpy.data.materials:
            name = "%s.%03d" % (base, i)
            i += 1
        mat = SP.build(key, name=name, preview=sh.preview, grid=(SP.PRESETS[key][2] == 'VOLUME' and ctx.object is not None and ctx.object.type == 'VOLUME'),
                       reuse=False)
        mat.use_fake_user = True
        if ctx.object is not None:
            SP.assign(ctx.object, mat)
        self.report({'INFO'}, "Created %s" % mat.name)
        return {'FINISHED'}


class AF_OT_shader_assign_fluid(Operator):
    bl_idname = "arcflow.shader_assign_fluid"
    bl_label = "Assign preset to selected fluid(s)"
    bl_description = "Set the shader of the active fluid and of every fluid whose particle object is selected, then update the materials"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        fm = ctx.scene.arc_flow_fm
        sh = ctx.scene.arc_flow_sh
        sel = {o.name for o in ctx.selected_objects}
        n = 0
        for i, fl in enumerate(fm.fluids):
            if i == fm.index or (fl.points_obj is not None and fl.points_obj.name in sel) or (fl.shape is not None and fl.shape.name in sel):
                fl.shader = sh.preset
                n += 1
        SP.apply_fluid_shaders(ctx.scene)
        self.report({'INFO'}, "%s set on %d fluid(s)" % (SP.PRESETS[sh.preset][0], n))
        return {'FINISHED'} if n else {'CANCELLED'}


class AF_OT_shader_fluids_auto(Operator):
    bl_idname = "arcflow.shader_fluids_auto"
    bl_label = "Auto shaders for all fluids"
    bl_description = "Give every fluid the default shader of its preset (water, oil, wax, honey, gas ...)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        for fl in ctx.scene.arc_flow_fm.fluids:
            fl.shader = 'AUTO'
        SP.apply_fluid_shaders(ctx.scene)
        return {'FINISHED'}


class AF_OT_shader_caustics(Operator):
    bl_idname = "arcflow.shader_caustics"
    bl_label = "Set up shadow caustics"
    bl_description = "Cycles: selected refractive objects become caustic casters, mesh objects that are not selected receive, and all lights emit caustics"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        sel = [o for o in ctx.selected_objects if o.type != 'LIGHT']
        rest = [o for o in ctx.scene.objects if o.type == 'MESH' and o not in sel]
        SP.caustics_hints(ctx.scene, casters=sel, receivers=rest, lights=[o for o in ctx.scene.objects if o.type == 'LIGHT'])
        self.report({'INFO'}, "Caustics: %d caster(s), %d receiver(s)" % (len(sel), len(rest)))
        return {'FINISHED'}


class AF_PT_shader(panels._Base, Panel):
    bl_idname = "AF_PT_shader"
    bl_label = "Shader presets"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, ctx):
        l = self.layout
        sh = ctx.scene.arc_flow_sh
        l.prop(sh, "preset", text="")
        r = l.row(align=True)
        r.operator("arcflow.shader_assign", text="Selected", icon='RESTRICT_SELECT_OFF').target = 'SELECTED'
        r.operator("arcflow.shader_assign", text="Active", icon='OBJECT_DATA').target = 'ACTIVE'
        l.operator("arcflow.shader_create", icon='MATERIAL')
        l.prop(sh, "preview")
        r = l.row(align=True)
        r.prop(sh, "caustics", toggle=True)
        r.operator("arcflow.shader_caustics", text="Set up")
        l.label(text=SP.PRESETS[sh.preset][4][:70], icon='INFO')
        fm = getattr(ctx.scene, 'arc_flow_fm', None)
        if fm is not None and len(fm.fluids):
            b = l.box()
            b.label(text="Fluids", icon='MOD_FLUIDSIM')
            if 0 <= fm.index < len(fm.fluids):
                b.prop(fm.fluids[fm.index], "shader", text=fm.fluids[fm.index].name)
            b.operator("arcflow.shader_assign_fluid", icon='FORWARD')
            b.operator("arcflow.shader_fluids_auto", icon='FILE_REFRESH')
        o = ctx.object
        m = o.active_material if o is not None else None
        if SP.preset_of(m):
            b = l.box()
            b.label(text="%s: %s" % (m.name, m.get('ts_variant', '')), icon='NODE_MATERIAL')
            c = b.column(align=True)
            for n in SP.params(m):
                c.prop(n.outputs[0], "default_value", text=n.label or n.name[3:])


_patched = []


def _hook_fluid_mode():
    """Hook the fluid renderer without editing its module: after ensure_objects (materials of the points objects) and after the surface
    mesh is rebuilt (its material), the preset shader of each fluid is applied.  Fluids set to 'OFF' keep their own material."""
    from . import fluidmode_ops as FO
    if getattr(FO.ensure_objects, '_ts_shader', False):
        return
    orig_ens, orig_surf = FO.ensure_objects, getattr(FO, '_update_surface', None)

    def ensure_objects(scene, *a, **k):
        r = orig_ens(scene, *a, **k)
        try:
            SP.apply_fluid_shaders(scene)
        except Exception as ex:
            FO.RT.msg = "shaders: %s" % ex
        return r

    def update_surface(scene, fl, *a, **k):
        r = orig_surf(scene, fl, *a, **k)
        try:
            so = bpy.data.objects.get("AFFMS_%s" % fl.name)
            m = SP.surface_material(fl, scene)
            if so is not None and m is not None:
                so.data.materials.clear()
                so.data.materials.append(m)
        except Exception as ex:
            FO.RT.msg = "shaders: %s" % ex
        return r
    orig_merged = getattr(FO, '_update_merged_surface', None)

    def update_merged(scene, *a, **k):
        r = orig_merged(scene, *a, **k)
        try:
            SP.apply_merged_surface(scene, bool(scene.arc_flow_sh.preview))
        except Exception as ex:
            FO.RT.msg = "shaders: %s" % ex
        return r
    orig_glass = getattr(FO, '_glass_material', None)

    def glass_material(name="TesseraGlass"):       # the Examples' vessels: a real thin-walled glass shader
        try:
            return SP.build('GLASS_THIN', name=name)
        except Exception:
            return orig_glass(name)
    ensure_objects._ts_shader = True
    ensure_objects._orig, update_surface._orig, ensure_objects._glass = orig_ens, orig_surf, orig_glass
    if orig_glass is not None:
        FO._glass_material = glass_material
    FO.ensure_objects = ensure_objects
    if orig_merged is not None:
        update_merged._orig = orig_merged
        FO._update_merged_surface = update_merged
    if orig_surf is not None:
        FO._update_surface = update_surface


def _unhook_fluid_mode():
    from . import fluidmode_ops as FO
    if getattr(FO.ensure_objects, '_ts_shader', False):
        if hasattr(FO._update_merged_surface, '_orig'):
            FO._update_merged_surface = FO._update_merged_surface._orig
        if hasattr(FO._update_surface, '_orig'):
            FO._update_surface = FO._update_surface._orig
        if FO.ensure_objects._glass is not None:
            FO._glass_material = FO.ensure_objects._glass
        FO.ensure_objects = FO.ensure_objects._orig


CLASSES = (AF_ShaderSettings, AF_OT_shader_assign, AF_OT_shader_create, AF_OT_shader_assign_fluid, AF_OT_shader_fluids_auto,
           AF_OT_shader_caustics, AF_PT_shader)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Scene.arc_flow_sh = PointerProperty(type=AF_ShaderSettings)
    try:
        _hook_fluid_mode()
    except Exception as ex:
        print("shader_ops: fluid hook failed:", ex)
    try:
        from .fluidmode_props import AF_FMFluid
        AF_FMFluid.shader = EnumProperty(name="Shader", items=_FLUID_ENUM, default='AUTO', update=_apply_fluids,
                                         description="Shader preset of this fluid (Auto = default of its fluid preset)")
    except Exception as ex:
        print("shader_ops: no per-fluid shader property:", ex)


def unregister():
    try:
        _unhook_fluid_mode()
    except Exception:
        pass
    try:
        from .fluidmode_props import AF_FMFluid
        del AF_FMFluid.shader
    except Exception:
        pass
    del bpy.types.Scene.arc_flow_sh
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
