"""Vertical Tessera N-panel tabs: register the add-on, switch all six tabs, draw without
error, keep every category's settings intact across switches, and exercise the Hair / Cloth hook.

    /home/box/apps/blender-4.2.3-linux-x64/blender --factory-startup -b --python tests/test_ui_tabs_bpy.py
    blender --factory-startup -b --python tests/test_ui_tabs_bpy.py [-- --installed]
"""
import os
import sys
import traceback
import types

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
fails = [0]


def ok(c, msg):
    print(("PASS " if c else "FAIL ") + msg, flush=True)
    if not c:
        fails[0] += 1


if '--installed' not in sys.argv:
    sys.path.insert(0, os.path.join(HERE, '..'))
    import arc_flow
    arc_flow.register()
    pkg = 'arc_flow'
else:
    import addon_utils
    import importlib
    pkg = [m.__name__ for m in addon_utils.modules() if m.__name__.endswith(('tessera_projection', 'arc_flow'))][0]
    if pkg not in bpy.context.preferences.addons:
        addon_utils.enable(pkg, default_set=True, persistent=False)
    arc_flow = importlib.import_module(pkg)

ui_tabs = __import__(pkg + '.ui_tabs', fromlist=['x'])

print("Blender", bpy.app.version_string, flush=True)
sc = bpy.context.scene
af = sc.arc_flow
fm = sc.arc_flow_fm
errors = []


def err(msg):
    if msg not in errors:
        errors.append(msg)


class FakeLayout:
    def __init__(self, where=""):
        object.__setattr__(self, "_where", where)

    def __setattr__(self, k, v):
        pass

    def __getattr__(self, name):
        def child(*a, **kw):
            return FakeLayout(self._where)

        if name in ("row", "column", "box", "split", "grid_flow", "column_flow"):
            return child
        if name == "panel":
            def pnl(*a, **kw):
                return FakeLayout(self._where), FakeLayout(self._where)
            return pnl
        if name in ("operator", "operator_enum", "operator_menu_enum"):
            def operator(idname, *a, **kw):
                class Op:
                    def __setattr__(self, k, v):
                        pass
                return Op()
            return operator
        if name in ("prop", "prop_enum", "props_enum", "prop_search", "label", "menu",
                    "separator", "template_list", "template_ID", "operator_menu_hold"):
            def fn(*a, **kw):
                return FakeLayout(self._where)
            return fn

        def anything(*a, **kw):
            return FakeLayout(self._where)
        return anything


class Self:
    def __init__(self, cls, layout=None):
        object.__setattr__(self, "_cls", cls)
        object.__setattr__(self, "layout", layout or FakeLayout(cls.__name__))

    def __getattr__(self, name):
        v = getattr(self._cls, name)
        if isinstance(v, types.FunctionType):
            return types.MethodType(v, self)
        return v


# ---- registration / RNA
ok(hasattr(bpy.types, "AF_PT_tabs"), "AF_PT_tabs registered")
ok(hasattr(bpy.ops.arcflow, "set_ui_tab"), "arcflow.set_ui_tab registered")
ok(hasattr(af, "ui_tab"), "scene.arc_flow.ui_tab exists")
ok(tuple(e.identifier for e in af.bl_rna.properties["ui_tab"].enum_items) == ui_tabs.TAB_IDS,
   "ui_tab enum matches TAB_IDS %s" % (tuple(e.identifier for e in af.bl_rna.properties["ui_tab"].enum_items),))
ok(set(ui_tabs.TAB_IDS) == {"REMESH", "FLUID", "COMBUSTION", "HAIR", "CLOTH", "EXAMPLES"},
   "six category tabs")
ok(af.ui_tab == "REMESH", "default tab is REMESH")

# hosted panels stay registered (backward compatible class ids) but do not stack in the N-panel
for n in ui_tabs.HOSTED_PANELS:
    ok(hasattr(bpy.types, n), "hosted panel %s still registered" % n)
ok(hasattr(bpy.types, "AF_PT_object_tab"), "object properties panel kept")
ok(hasattr(bpy.types, "AF_PT_gn_nodes_editor"), "Geometry Nodes editor panel kept")

# ---- polls: root always visible; hosted panels hidden; Properties / Node Editor unchanged
ctx = bpy.context
ok(bool(bpy.types.AF_PT_tabs.poll(ctx)) if hasattr(bpy.types.AF_PT_tabs, "poll") else True,
   "AF_PT_tabs visible")
# panels without poll are treated as visible; hosted ones must poll False
hidden = []
for n in ui_tabs.HOSTED_PANELS:
    cls = getattr(bpy.types, n)
    vis = bool(cls.poll(ctx)) if hasattr(cls, "poll") else True
    if vis:
        hidden.append(n)
ok(not hidden, "hosted category panels do not stack in the N-panel %s" % hidden)

# object tab poll is not wrapped
ot = bpy.types.AF_PT_object_tab
bpy.ops.mesh.primitive_cube_add()
ok(bool(ot.poll(ctx)), "AF_PT_object_tab still polls for a mesh object")

# ---- switching tabs never mutates other settings
fm.mode = "COMBUSTION"
af.domain_radius = 77.5
af.particle_budget = 12345
af.bool_projection = True
fm.max_particles = 4242
fm.spacing = 0.03125
try:
    sc.arc_flow_retopo.target_faces = 2222
    sc.arc_flow_retopo.use_symmetry = True
    retopo_ok = True
except Exception:
    retopo_ok = False
ok(retopo_ok, "retopo properties writable")
snap = dict(
    mode=fm.mode, domain=af.domain_radius, budget=af.particle_budget,
    bool_p=af.bool_projection, max_p=fm.max_particles, spacing=fm.spacing,
    faces=sc.arc_flow_retopo.target_faces, sym=sc.arc_flow_retopo.use_symmetry,
)

# via RNA
for tab in ui_tabs.TAB_IDS:
    af.ui_tab = tab
    ok(af.ui_tab == tab, "RNA switch to %s" % tab)
    ok(ui_tabs.active_tab(ctx) == tab, "active_tab() is %s" % tab)
ok(fm.mode == snap["mode"] and af.domain_radius == snap["domain"] and af.particle_budget == snap["budget"]
   and af.bool_projection == snap["bool_p"] and fm.max_particles == snap["max_p"]
   and fm.spacing == snap["spacing"] and sc.arc_flow_retopo.target_faces == snap["faces"]
   and sc.arc_flow_retopo.use_symmetry == snap["sym"],
   "settings intact after RNA tab switches (mode still %s)" % fm.mode)

# via operator (and set_tab helper)
af.ui_tab = "REMESH"
for tab in ui_tabs.TAB_IDS:
    res = bpy.ops.arcflow.set_ui_tab(tab=tab)
    ok("FINISHED" in res and af.ui_tab == tab, "operator switch to %s: %s" % (tab, res))
ui_tabs.set_tab(ctx, "FLUID")
ok(af.ui_tab == "FLUID", "set_tab() helper")
try:
    ui_tabs.set_tab(ctx, "NOPE")
    raised = False
except ValueError:
    raised = True
ok(raised and af.ui_tab == "FLUID", "set_tab rejects unknown ids without changing the tab")

ok(fm.mode == "COMBUSTION" and af.domain_radius == 77.5 and fm.max_particles == 4242
   and sc.arc_flow_retopo.target_faces == 2222,
   "operator / helper tab switches still leave simulations and settings untouched")

# switching tabs does not flip fm.mode even when visiting the other solver's tab
fm.mode = "FLUID"
ui_tabs.set_tab(ctx, "COMBUSTION")
ok(fm.mode == "FLUID" and af.ui_tab == "COMBUSTION",
   "COMBUSTION tab does not force fm.mode away from FLUID")
ui_tabs.set_tab(ctx, "REMESH")
ok(fm.mode == "FLUID" and af.ui_tab == "REMESH",
   "REMESH tab does not change fm.mode")
fm.mode = "COMBUSTION"
ui_tabs.set_tab(ctx, "FLUID")
ok(fm.mode == "COMBUSTION" and af.ui_tab == "FLUID",
   "FLUID tab does not force fm.mode away from COMBUSTION")

# ---- draw every tab, in both solver modes, empty and with a mesh selected
def draw_tab(tag, tab):
    n_err = len(errors)
    try:
        ui_tabs.draw_tab_body(FakeLayout(tag), ctx, tab)
        ui_tabs.draw_tab_rail(FakeLayout(tag + "/rail"), ctx)
        ui_tabs.draw_shell(FakeLayout(tag + "/shell"), ctx)
        bpy.types.AF_PT_tabs.draw(Self(bpy.types.AF_PT_tabs), ctx)
        if hasattr(bpy.types.AF_PT_tabs, "draw_header"):
            bpy.types.AF_PT_tabs.draw_header(Self(bpy.types.AF_PT_tabs), ctx)
    except Exception:
        err("%s: %s" % (tag, traceback.format_exc().strip().splitlines()[-1]))
    return len(errors) == n_err


for mode in ("COMBUSTION", "FLUID"):
    fm.mode = mode
    for tab in ui_tabs.TAB_IDS:
        af.ui_tab = tab
        ok(draw_tab("sel/%s/%s" % (mode, tab), tab), "draw %s while fm.mode=%s (mesh selected)" % (tab, mode))

# no object selected
for o in list(sc.objects):
    o.select_set(False)
bpy.context.view_layer.objects.active = None
for tab in ui_tabs.TAB_IDS:
    af.ui_tab = tab
    ok(draw_tab("empty/" + tab, tab), "draw %s with no active object" % tab)

# narrow / wide shell (no region in background: draw_shell uses the side-by-side path)
ok(ui_tabs.region_is_narrow(ctx) is False or isinstance(ui_tabs.region_is_narrow(ctx), bool),
   "region_is_narrow returns a bool")

# ---- Hair / Cloth hook
seen = []


def hair_drawer(layout, context):
    seen.append(("HAIR", context.scene.arc_flow.ui_tab, getattr(layout, "_where", "")))
    layout.label(text="Hair drawer OK")


def cloth_drawer(layout, context):
    seen.append(("CLOTH", context.scene.arc_flow.ui_tab))
    layout.label(text="Cloth drawer OK")

# ready-state card before a module registers
ok(not ui_tabs.tab_drawers("HAIR") and not ui_tabs.tab_drawers("CLOTH"),
   "HAIR / CLOTH start with no custom drawer")
ready_h = []
ready_c = []


class Capture(FakeLayout):
    def __init__(self, where, bag):
        FakeLayout.__init__(self, where)
        object.__setattr__(self, "_bag", bag)

    def __getattr__(self, name):
        bag = self._bag
        where = self._where
        if name == "label":
            def label(*a, **kw):
                text = kw.get("text") or (a[0] if a else "")
                if text:
                    bag.append(text)
                return Capture(where, bag)
            return label
        if name in ("row", "column", "box", "split", "grid_flow", "column_flow"):
            def child(*a, **kw):
                return Capture(where, bag)
            return child
        if name == "panel":
            def pnl(*a, **kw):
                return Capture(where, bag), Capture(where, bag)
            return pnl
        return FakeLayout.__getattr__(self, name)


hcap, ccap = Capture("ready/HAIR", ready_h), Capture("ready/CLOTH", ready_c)
ui_tabs.draw_tab_body(hcap, ctx, "HAIR")
ui_tabs.draw_tab_body(ccap, ctx, "CLOTH")
ok(any("register_tab_drawer('HAIR'" in t for t in ready_h) and any("Hair Curves" in t or "Empty Hair" in t for t in ready_h),
   "HAIR ready-state card shown before hook: %s" % ready_h[:6])
ok(any("register_tab_drawer('CLOTH'" in t for t in ready_c),
   "CLOTH ready-state card shown before hook")
ok(not any("demo" in (t or "").lower() for t in ready_h + ready_c),
   "ready-state cards do not say 'demo'")

ui_tabs.register_tab_drawer("HAIR", hair_drawer)
ui_tabs.register_tab_drawer("CLOTH", cloth_drawer)
ok(len(ui_tabs.tab_drawers("HAIR")) == 1 and ui_tabs.tab_drawers("HAIR")[0]["fn"] is hair_drawer,
   "register_tab_drawer('HAIR', fn) hooks the drawer")

after_h, after_c = [], []
ui_tabs.draw_tab_body(Capture("hook/HAIR", after_h), ctx, "HAIR")
ui_tabs.draw_tab_body(Capture("hook/CLOTH", after_c), ctx, "CLOTH")
ok(any(s[0] == "HAIR" for s in seen) and "Hair drawer OK" in after_h,
   "HAIR drawer is called and replaces the ready-state card")
ok(any(s[0] == "CLOTH" for s in seen) and "Cloth drawer OK" in after_c,
   "CLOTH drawer is called")
ok(not any("register_tab_drawer('HAIR'" in t for t in after_h),
   "ready-state card is gone once the HAIR hook is registered")

# unknown tab id
try:
    ui_tabs.register_tab_drawer("LEGS", hair_drawer)
    bad = False
except ValueError:
    bad = True
ok(bad, "register_tab_drawer rejects unknown tab ids")

ui_tabs.unregister_tab_drawer("HAIR", hair_drawer)
ui_tabs.unregister_tab_drawer("CLOTH")
ok(not ui_tabs.tab_drawers("HAIR") and not ui_tabs.tab_drawers("CLOTH"),
   "unregister_tab_drawer restores an empty hook list")
restored = []
ui_tabs.draw_tab_body(Capture("restored/HAIR", restored), ctx, "HAIR")
ok(any("register_tab_drawer('HAIR'" in t for t in restored),
   "HAIR ready-state card returns after unregister")

# appending a section onto a built-in tab does not wipe Remesh
extra = []


def remesh_extra(layout, context):
    extra.append("ok")
    layout.label(text="Remesh extra section")

ui_tabs.register_tab_drawer("REMESH", remesh_extra, replace=False, label="Worker section")
bag = []
ui_tabs.draw_tab_body(Capture("remesh+extra", bag), ctx, "REMESH")
ok(extra == ["ok"] and any("Projection Remesh" in t or "Remesh" in t for t in bag) and "Remesh extra section" in bag,
   "register_tab_drawer on a built-in tab appends without replacing built-in cards")
ui_tabs.unregister_tab_drawer("REMESH", remesh_extra)

# final property check (drawers / draws must not have reset them)
ok(af.domain_radius == 77.5 and af.bool_projection is True
   and fm.max_particles == 4242 and sc.arc_flow_retopo.target_faces == 2222
   and af.particle_budget == 12345 and sc.arc_flow_retopo.use_symmetry is True
   and fm.spacing == 0.03125 and fm.mode in ("COMBUSTION", "FLUID"),
   "all category properties still intact after draws and hook tests (mode=%s)" % fm.mode)

for e in errors:
    print("FAIL " + e, flush=True)
ok(not errors, "no draw errors (%d)" % len(errors))

print("FAILURES: %d" % fails[0], flush=True)
sys.exit(1 if fails[0] else 0)
