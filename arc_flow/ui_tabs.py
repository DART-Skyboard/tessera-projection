"""Vertical tab rail for the Tessera Projection 3D Viewport N-panel.

The Tessera sidebar used to stack every category in one long scroll. This module
draws an in-panel vertical tab list (the same idea as Blender's N-panel category
tabs) and shows only the active category in a compact card beside the rail.

Switching tabs is a view change only: it never starts, stops, resets or hides
running simulations, modifiers or settings in other categories. Remesh, Fluid,
Combustion, Wind Tunnel, Hair and Cloth can all be active in the same scene.

RNA
    scene.arc_flow.ui_tab    enum  REMESH | FLUID | COMBUSTION | HAIR | CLOTH | EXAMPLES

Hair / Cloth hook (Workers 3 and 4)
    from arc_flow.ui_tabs import register_tab_drawer, unregister_tab_drawer

    def draw_hair(layout, context):
        layout.prop(...)

    def register():
        register_tab_drawer('HAIR', draw_hair)

    def unregister():
        unregister_tab_drawer('HAIR', draw_hair)

    register_tab_drawer replaces the built-in ready-state card for HAIR and CLOTH
    (default replace=True). For the other tabs, the default is to append a new
    section after the built-in cards so existing controls stay put. Do not also
    register a sibling Tessera N-panel for the same content: the hook draws
    inside the tab card, which is what keeps the sidebar compact.
"""
import bpy
from bpy.types import Operator, Panel

from . import panels

# (id, label, short rail label, icon, description)
TABS = (
    ('REMESH', "Remesh", "Remesh", 'MOD_REMESH',
     "Projection Remesh, topology analysis, symmetrize, crease density and Boolean projection"),
    ('FLUID', "Fluid", "Fluid", 'MOD_FLUIDSIM',
     "Fluid mode, volumes, viscosity and emission surfaces, solver, meshing and shaders"),
    ('COMBUSTION', "Combustion", "Comb", 'LIGHT_SUN',
     "Combustion, systems, chambers, wind tunnel, thermal, environment and atmosphere"),
    ('HAIR', "Hair", "Hair", 'OUTLINER_OB_CURVES',
     "Real-time hair curves solver"),
    ('CLOTH', "Cloth", "Cloth", 'MOD_CLOTH',
     "Real-time cloth solver"),
    ('MUSCLE', "Muscle", "Muscle", 'BONE_DATA',
     "Muscle projection solver: region painting, contours, shape keys and Geometry Nodes"),
    ('EXAMPLES', "Examples", "Ex", 'FILE_BLEND',
     "Loadable examples and Tessera Geometry Nodes"),
)
TAB_IDS = tuple(t[0] for t in TABS)
TAB_ITEMS = tuple((t[0], t[1], t[4], t[3], i) for i, t in enumerate(TABS))
TAB_INFO = {t[0]: t for t in TABS}

# N-panel classes whose draw() is hosted inside the tab card. Their own VIEW_3D
# Tessera panel is hidden (poll False) so the sidebar does not stack twice.
# Properties-editor and Node Editor panels are not in this set.
HOSTED_PANELS = (
    "AF_PT_main", "AF_PT_parts", "AF_PT_fluids", "AF_PT_physics", "AF_PT_math",
    "AF_PT_env", "AF_PT_xray", "AF_PT_solver", "AF_PT_cache", "AF_PT_systems",
    "AF_PT_fm_mode", "AF_PT_fm_main", "AF_PT_fm_fluids", "AF_PT_fm_solver",
    "AF_PT_fm_object", "AF_PT_fm_env", "AF_PT_srf", "AF_PT_retopo",
    "AF_PT_examples", "AF_PT_shader", "AF_PT_thermal", "AF_PT_windtunnel",
    "AF_PT_gn_nodes_view3d", "AF_PT_cloth", "AF_PT_muscle",
)

# tab_id -> list of callables (layout, context)
_DRAWERS = {}
_ORIG_POLLS = {}
_NARROW_PX = 240
_RAIL_FACTOR = 0.20


def active_tab(ctx):
    """Return the current N-panel tab id (default REMESH). Never creates properties."""
    sc = getattr(ctx, "scene", None)
    af = getattr(sc, "arc_flow", None) if sc is not None else None
    tab = getattr(af, "ui_tab", None) if af is not None else None
    return tab if tab in TAB_INFO else "REMESH"


def set_tab(ctx, tab):
    """Set the N-panel tab. Does not touch fm.mode, solvers, or other settings."""
    if tab not in TAB_INFO:
        raise ValueError("Unknown Tessera tab %r (expected one of %s)" % (tab, ", ".join(TAB_IDS)))
    ctx.scene.arc_flow.ui_tab = tab
    return tab


def tab_poll(tab_id):
    """classmethod poll helper if a future panel should only appear on one tab.

    Prefer register_tab_drawer so the section lands inside the tab card.
    """
    want = str(tab_id).upper()

    def poll(cls, ctx):
        return active_tab(ctx) == want
    return classmethod(poll)


def register_tab_drawer(tab_id, fn, *, replace=None, label="", icon='NONE'):
    """Register a draw function for a tab body.

    fn(layout, context) -> None

    HAIR / CLOTH default to replace=True (the ready-state card is swapped out).
    Other tabs default to replace=False (fn is drawn as an extra section).
    Returns fn so the same object can be passed to unregister_tab_drawer.
    """
    if not callable(fn):
        raise TypeError("register_tab_drawer() needs a callable(layout, context), got %r" % (type(fn).__name__,))
    tab_id = str(tab_id).upper()
    if tab_id not in TAB_INFO:
        raise ValueError("Unknown Tessera tab %r (expected one of %s)" % (tab_id, ", ".join(TAB_IDS)))
    if replace is None:
        replace = tab_id in ("HAIR", "CLOTH")
    entry = {"fn": fn, "label": label or "", "icon": icon or "NONE"}
    if replace:
        _DRAWERS[tab_id] = [entry]
    else:
        lst = _DRAWERS.setdefault(tab_id, [])
        if not any(e["fn"] is fn for e in lst):
            lst.append(entry)
    return fn


def unregister_tab_drawer(tab_id, fn=None):
    """Remove a drawer. If fn is None, remove every custom drawer on that tab."""
    tab_id = str(tab_id).upper()
    if tab_id not in _DRAWERS:
        return
    if fn is None:
        _DRAWERS.pop(tab_id, None)
        return
    _DRAWERS[tab_id] = [e for e in _DRAWERS[tab_id] if e["fn"] is not fn]
    if not _DRAWERS[tab_id]:
        _DRAWERS.pop(tab_id, None)


def tab_drawers(tab_id):
    """Registered custom drawers for a tab (empty tuple if none)."""
    return tuple(_DRAWERS.get(str(tab_id).upper(), ()))


class _DrawSelf:
    """Stand-in `self` so an existing Panel.draw / helper can reuse .layout."""

    def __init__(self, cls, layout):
        object.__setattr__(self, "_cls", cls)
        object.__setattr__(self, "layout", layout)

    def __getattr__(self, name):
        v = getattr(self._cls, name)
        if callable(v) and not isinstance(v, type):
            def bound(*a, __v=v, **k):
                return __v(self, *a, **k)
            return bound
        return v


def _draw_cls(cls, layout, ctx, *, need_mesh=False):
    obj = getattr(ctx, "object", None)
    if need_mesh and (obj is None or getattr(obj, "type", "") != "MESH"):
        layout.label(text="Select a mesh object", icon="MESH_DATA")
        return
    cls.draw(_DrawSelf(cls, layout), ctx)


def _section(layout, title, icon="NONE"):
    box = layout.box()
    if title:
        box.label(text=title, icon=icon)
    return box


def _ready_card(layout, tab_id):
    info = TAB_INFO[tab_id]
    kind = "Hair Curves" if tab_id == "HAIR" else "Cloth"
    b = layout.box()
    b.label(text="Tessera %s solver" % kind, icon=info[3])
    col = b.column(align=True)
    if tab_id == "HAIR":
        col.label(text="Paint and style hair with Blender's Empty Hair")
        col.label(text="curves, then let Tessera drive collisions, gravity")
        col.label(text="and velocity in real time.")
    else:
        col.label(text="Prepare a cloth mesh in Blender, then let Tessera")
        col.label(text="drive collisions, gravity and velocity in real time.")
    col.separator()
    col.label(text="This tab is ready: the solver module is not loaded")
    col.label(text="yet. When it lands it plugs in with:")
    col.label(text="  register_tab_drawer('%s', draw_fn)" % tab_id)
    col.separator()
    col.label(text="Settings and simulations in other tabs stay active.", icon="INFO")


def _draw_remesh(layout, ctx):
    from . import retopo_panel
    body = _section(layout, "Projection Remesh", "MOD_REMESH")
    _draw_cls(retopo_panel.AF_PT_retopo, body, ctx)
    body = _section(layout, "Boolean projection", "MOD_BOOLEAN")
    panels.draw_boolean_projection(body, ctx)


def _draw_fluid(layout, ctx):
    from . import fluidmode_panel as FM
    from . import shader_ops
    from . import surfaces_panel
    body = _section(layout, "Fluid mode", "MOD_FLUIDSIM")
    _draw_cls(FM.AF_PT_fm_main, body, ctx)
    body = _section(layout, "Fluids", "OUTLINER_OB_POINTCLOUD")
    _draw_cls(FM.AF_PT_fm_fluids, body, ctx)
    body = _section(layout, "Surface sources and volumes", "MESH_CUBE")
    _draw_cls(surfaces_panel.AF_PT_srf, body, ctx)
    body = _section(layout, "Solver", "PHYSICS")
    _draw_cls(FM.AF_PT_fm_solver, body, ctx)
    body = _section(layout, "Active object: heat source / collider", "OBJECT_DATA")
    _draw_cls(FM.AF_PT_fm_object, body, ctx)
    body = _section(layout, "Environment", "WORLD")
    _draw_cls(panels.AF_PT_env, body, ctx)
    body = _section(layout, "Shader presets", "MATERIAL")
    _draw_cls(shader_ops.AF_PT_shader, body, ctx)


def _draw_combustion(layout, ctx):
    from . import systems_ops
    from . import thermal
    from . import windtunnel
    body = _section(layout, "Tessera Projection", "PLAY")
    _draw_cls(panels.AF_PT_main, body, ctx)
    body = _section(layout, "Part (active object)", "OBJECT_DATA")
    _draw_cls(panels.AF_PT_parts, body, ctx)
    body = _section(layout, "Systems", "LINKED")
    _draw_cls(systems_ops.AF_PT_systems, body, ctx)
    body = _section(layout, "Fluids and Render", "MOD_FLUIDSIM")
    _draw_cls(panels.AF_PT_fluids, body, ctx)
    body = _section(layout, "Wind tunnel", "FORCE_WIND")
    _draw_cls(windtunnel.AF_PT_windtunnel, body, ctx)
    body = _section(layout, "Thermal readback", "LIGHT_SUN")
    _draw_cls(thermal.AF_PT_thermal, body, ctx, need_mesh=True)
    body = _section(layout, "Environment", "WORLD")
    _draw_cls(panels.AF_PT_env, body, ctx)
    body = _section(layout, "Physics Cards", "PHYSICS")
    _draw_cls(panels.AF_PT_physics, body, ctx)
    body = _section(layout, "Math Cards", "LINENUMBERS_ON")
    _draw_cls(panels.AF_PT_math, body, ctx)
    body = _section(layout, "X-ray and Feedback", "HIDE_OFF")
    _draw_cls(panels.AF_PT_xray, body, ctx)
    body = _section(layout, "Solver (LOTBCBP)", "MOD_REMESH")
    _draw_cls(panels.AF_PT_solver, body, ctx)
    body = _section(layout, "Simulation Data", "FILE_CACHE")
    _draw_cls(panels.AF_PT_cache, body, ctx)


def _draw_examples(layout, ctx):
    from . import surfaces_panel
    from . import tessera_nodes
    body = _section(layout, "Examples", "FILE_BLEND")
    _draw_cls(surfaces_panel.AF_PT_examples, body, ctx)
    body = _section(layout, "Tessera node groups", "NODETREE")
    _draw_cls(tessera_nodes.AF_PT_gn_nodes_view3d, body, ctx)


_BUILTIN = {
    "REMESH": _draw_remesh,
    "FLUID": _draw_fluid,
    "COMBUSTION": _draw_combustion,
    "EXAMPLES": _draw_examples,
}


def draw_tab_rail(layout, ctx, *, horizontal=False, compact=False):
    """Draw the tab switcher. vertical column by default; a wrapping row if horizontal."""
    current = active_tab(ctx)
    root = layout.row(align=True) if horizontal else layout.column(align=True)
    root.scale_y = 1.15 if not horizontal else 1.2
    for tab_id, label, short, icon, desc in TABS:
        text = short if compact or not horizontal else label
        # keep short labels on the rail so a narrow sidebar still fits
        if not horizontal:
            text = short
        op = root.operator("arcflow.set_ui_tab", text=text, icon=icon, depress=(current == tab_id))
        op.tab = tab_id
    return root


def draw_tab_body(layout, ctx, tab=None):
    """Draw only the given (or active) tab's cards into layout. Used by the root panel and tests."""
    tab = tab or active_tab(ctx)
    if tab not in TAB_INFO:
        layout.label(text="Unknown tab %s" % tab, icon="ERROR")
        return tab
    info = TAB_INFO[tab]
    row = layout.row()
    row.label(text=info[1], icon=info[3])
    builtin = _BUILTIN.get(tab)
    custom = tab_drawers(tab)
    if builtin and not (custom and tab in ("HAIR", "CLOTH")):
        builtin(layout, ctx)
    if tab in ("HAIR", "CLOTH") and not custom:
        _ready_card(layout, tab)
    for entry in custom:
        target = layout
        if entry["label"]:
            target = _section(layout, entry["label"], entry["icon"])
        entry["fn"](target, ctx)
    return tab


def draw_shell(layout, ctx):
    """Rail + card. Side-by-side when the region is wide enough; stacked on narrow sidebars."""
    width = 0
    region = getattr(ctx, "region", None)
    if region is not None:
        width = int(getattr(region, "width", 0) or 0)
    narrow = bool(width and width < _NARROW_PX)
    if narrow:
        draw_tab_rail(layout, ctx, horizontal=True, compact=True)
        card = layout.box()
        draw_tab_body(card, ctx)
        return
    split = layout.split(factor=_RAIL_FACTOR, align=True)
    rail = split.column(align=True)
    draw_tab_rail(rail, ctx, horizontal=False, compact=True)
    card = split.column().box()
    draw_tab_body(card, ctx)


def region_is_narrow(ctx):
    region = getattr(ctx, "region", None)
    width = int(getattr(region, "width", 0) or 0) if region is not None else 0
    return bool(width and width < _NARROW_PX)


class AF_OT_set_ui_tab(Operator):
    bl_idname = "arcflow.set_ui_tab"
    bl_label = "Tessera Tab"
    bl_description = "Show this Tessera category. Other categories stay active; running simulations are not changed"
    bl_options = {'INTERNAL', 'REGISTER'}

    tab: bpy.props.EnumProperty(name="Tab", items=TAB_ITEMS, default='REMESH')

    def execute(self, ctx):
        set_tab(ctx, self.tab)
        return {'FINISHED'}


class AF_PT_tabs(panels._Base, Panel):
    """Root Tessera N-panel: vertical tab rail + the active category's card."""
    bl_idname = "AF_PT_tabs"
    bl_label = "Tessera Projection"
    bl_order = -100

    def draw_header(self, ctx):
        info = TAB_INFO[active_tab(ctx)]
        self.layout.label(text="", icon=info[3])

    def draw(self, ctx):
        draw_shell(self.layout, ctx)


def _hide_hosted_polls():
    """Keep hosted panels registered (operators, draw methods, tests) but out of the N-panel stack."""
    for name in HOSTED_PANELS:
        cls = getattr(bpy.types, name, None)
        if cls is None or getattr(cls, "_ts_tab_hosted", False):
            continue
        # live poll may already wrap fluid-mode hiding; restore it on unregister
        _ORIG_POLLS[name] = getattr(cls, "poll", None)

        def poll(cls, ctx):
            return False
        cls.poll = classmethod(poll)
        cls._ts_tab_hosted = True
    _reregister([n for n in _ORIG_POLLS])


def _panel_family(names):
    """The hosted panels plus every registered sub-panel hanging off them (children first)."""
    fam = []
    want = set(names)
    grew = True
    while grew:
        grew = False
        for cls in bpy.types.Panel.__subclasses__():
            nm = getattr(cls, "bl_idname", "") or cls.__name__
            if nm in want:
                continue
            if getattr(cls, "bl_parent_id", "") in want and getattr(bpy.types, nm, None) is cls:
                want.add(nm)
                grew = True
    for nm in want:
        cls = getattr(bpy.types, nm, None)
        if cls is not None:
            fam.append(cls)

    def depth(c):
        d, cur = 0, c
        while getattr(cur, "bl_parent_id", ""):
            d += 1
            cur = getattr(bpy.types, cur.bl_parent_id, None)
            if cur is None or d > 8:
                break
        return d
    fam.sort(key=depth, reverse=True)
    return fam


def _reregister(names):
    """Blender binds a panel's poll callback at registration: a class that had no poll()
    when it was registered never gets one called later. Re-register the hosted family so
    the replaced poll takes effect in the real UI (not only when called from Python)."""
    if not names:
        return
    fam = _panel_family(names)
    done = []
    for cls in fam:                     # children first
        try:
            bpy.utils.unregister_class(cls)
            done.append(cls)
        except Exception:
            pass
    for cls in reversed(done):          # parents first
        try:
            bpy.utils.register_class(cls)
        except Exception as ex:
            print("Tessera tabs: could not re-register %s: %s" % (cls.__name__, ex))


def _restore_hosted_polls():
    for name, orig in list(_ORIG_POLLS.items()):
        cls = getattr(bpy.types, name, None)
        if cls is None:
            continue
        if orig is not None:
            cls.poll = orig
        elif hasattr(cls, "poll") and getattr(cls, "_ts_tab_hosted", False):
            def poll(cls, ctx):
                return True
            cls.poll = classmethod(poll)
        if hasattr(cls, "_ts_tab_hosted"):
            delattr(cls, "_ts_tab_hosted")
    names = list(_ORIG_POLLS)
    _ORIG_POLLS.clear()
    _reregister(names)


CLASSES = (AF_OT_set_ui_tab, AF_PT_tabs)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    _hide_hosted_polls()


def unregister():
    _restore_hosted_polls()
    _DRAWERS.clear()
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
