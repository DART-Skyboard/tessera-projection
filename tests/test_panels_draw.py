"""Draws every Tessera panel, sub-panel, menu and list headlessly with a validating stand-in for UILayout, in several scene states
(empty, each Example loaded, with an object selected, each fluid mode), and checks: no exception, every icon exists, every property
shown exists on its data block, every operator exists and every operator property set on it exists, UI lists / menus exist.
Also: every Example loads, the Examples panel is collapsed and all others expanded, no 'demo' in labels or operator names.

    blender --factory-startup -b --python tests/test_panels_draw.py [-- --installed]"""
import importlib
import os
import re
import sys
import tempfile
import traceback
import types

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
fails = [0]
strings = set()


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
    pkg = [m.__name__ for m in addon_utils.modules() if m.__name__.endswith(('tessera_projection', 'arc_flow'))][0]
    if pkg not in bpy.context.preferences.addons:
        addon_utils.enable(pkg, default_set=True, persistent=False)
    arc_flow = importlib.import_module(pkg)
SP = importlib.import_module(pkg + '.surfaces_panel')
FO = importlib.import_module(pkg + '.fluidmode_ops')
runtime = importlib.import_module(pkg + '.runtime')
ui_tabs = importlib.import_module(pkg + '.ui_tabs')

print("Blender", bpy.app.version_string, flush=True)
ICONS = {e.identifier for e in bpy.types.UILayout.bl_rna.functions['label'].parameters['icon'].enum_items}
errors = []
counts = {'draws': 0, 'calls': 0}


def err(msg):
    if msg not in errors:
        errors.append(msg)


def check_icon(kw, where):
    for key in ('icon', 'icon_only'):
        pass
    ic = kw.get('icon')
    if isinstance(ic, str) and ic not in ICONS:
        err("%s: invalid icon %r" % (where, ic))


def check_prop(data, name, where):
    if name is None or not isinstance(name, str) or name.startswith('['):
        return
    try:
        props = data.bl_rna.properties
    except Exception:
        err("%s: prop() data %r has no RNA" % (where, data))
        return
    if name not in props:
        err("%s: property %r missing on %s" % (where, name, type(data).__name__))


class FakeOp:
    def __init__(self, idname, where):
        object.__setattr__(self, '_n', idname)
        object.__setattr__(self, '_w', where)
        try:
            mod, fn = idname.split('.')
            rna = getattr(getattr(bpy.ops, mod), fn).get_rna_type()
            object.__setattr__(self, '_p', set(rna.properties.keys()))
        except Exception:
            object.__setattr__(self, '_p', None)
            err("%s: operator %r does not exist" % (where, idname))

    def __setattr__(self, k, v):
        if self._p is not None and k not in self._p:
            err("%s: operator %s has no property %r" % (self._w, self._n, k))

    def __getattr__(self, k):
        return None


class FakeLayout:
    def __init__(self, where):
        object.__setattr__(self, '_where', where)

    def __setattr__(self, k, v):
        pass

    def __getattr__(self, name):
        where = self._where

        def child(*a, **kw):
            counts['calls'] += 1
            check_icon(kw, where)
            for k in ('text', 'heading'):
                if isinstance(kw.get(k), str):
                    strings.add(kw[k])
            return FakeLayout(where)

        if name in ('row', 'column', 'box', 'split', 'grid_flow', 'column_flow', 'row_flow', 'menu_pie', 'absolute'):
            return child
        if name in ('panel', 'panel_prop'):
            def pnl(*a, **kw):
                counts['calls'] += 1
                return FakeLayout(where), FakeLayout(where)
            return pnl
        if name == 'prop':
            def prop(data, prop, *a, **kw):
                counts['calls'] += 1
                check_icon(kw, where)
                check_prop(data, prop, where)
                if isinstance(kw.get('text'), str):
                    strings.add(kw['text'])
            return prop
        if name in ('prop_enum', 'props_enum', 'prop_menu_enum', 'prop_tabs_enum'):
            def pe(data, prop, *a, **kw):
                counts['calls'] += 1
                check_icon(kw, where)
                check_prop(data, prop, where)
            return pe
        if name == 'prop_search':
            def ps(data, prop, sdata, sprop, *a, **kw):
                counts['calls'] += 1
                check_icon(kw, where)
                check_prop(data, prop, where)
                check_prop(sdata, sprop, where)
            return ps
        if name in ('operator', 'operator_enum', 'operator_menu_enum', 'operator_menu_hold'):
            def operator(idname, *a, **kw):
                counts['calls'] += 1
                check_icon(kw, where)
                if isinstance(kw.get('text'), str):
                    strings.add(kw['text'])
                return FakeOp(idname, where)
            return operator
        if name == 'label':
            def label(*a, **kw):
                counts['calls'] += 1
                check_icon(kw, where)
                if isinstance(kw.get('text'), str):
                    strings.add(kw['text'])
            return label
        if name == 'menu':
            def menu(idname, *a, **kw):
                counts['calls'] += 1
                check_icon(kw, where)
                if not hasattr(bpy.types, idname):
                    err("%s: menu %r missing" % (where, idname))
            return menu
        if name == 'template_list':
            def tl(listtype, list_id, data, propname, *a, **kw):
                counts['calls'] += 1
                check_prop(data, propname, where)
                if not hasattr(bpy.types, listtype) and listtype != 'UI_UL_list':
                    err("%s: UI list %r missing" % (where, listtype))
                else:
                    cls = getattr(bpy.types, listtype, None)
                    if cls is not None and hasattr(cls, 'draw_item'):
                        coll = getattr(data, propname, [])
                        for i, it in enumerate(coll):
                            try:
                                cls.draw_item(types.SimpleNamespace(), bpy.context, FakeLayout(where + "/" + listtype), data, it, 'NONE', data, propname, i)
                            except Exception:
                                err("%s/%s.draw_item: %s" % (where, listtype, traceback.format_exc().strip().splitlines()[-1]))
            return tl
        if name in ('template_ID', 'template_any_ID', 'template_path_builder'):
            def tid(data, prop, *a, **kw):
                counts['calls'] += 1
                check_prop(data, prop, where)
            return tid
        if name == 'template_color_ramp':
            def tcr(data, prop, *a, **kw):
                counts['calls'] += 1
                check_prop(data, prop, where)
            return tcr
        # label-like and everything else (separator, template_*, context_pointer_set, ...)
        def anything(*a, **kw):
            counts['calls'] += 1
            check_icon(kw, where)
            return FakeLayout(where)
        return anything


class Self:
    """Stand-in `self` of a Panel / Menu: .layout is the validating fake, everything else is the class's."""

    def __init__(self, cls, where):
        object.__setattr__(self, '_cls', cls)
        object.__setattr__(self, 'layout', FakeLayout(where))

    def __setattr__(self, k, v):
        object.__setattr__(self, k, v)

    def __getattr__(self, name):
        v = getattr(self._cls, name)
        if isinstance(v, types.FunctionType):
            return types.MethodType(v, self)
        return v


def ui_classes():
    out = []
    for n in sorted(dir(bpy.types)):
        if n.startswith(('AF_PT_', 'AF_MT_')):
            out.append(getattr(bpy.types, n))
    return out


def draw_all(tag):
    ctx = bpy.context
    n = 0
    for cls in ui_classes():
        where = "%s/%s" % (tag, cls.__name__)
        try:
            if hasattr(cls, 'poll') and not cls.poll(ctx):
                continue
        except Exception:
            err("%s: poll raised %s" % (where, traceback.format_exc().strip().splitlines()[-1]))
            continue
        for fn in ('draw_header', 'draw'):
            f = cls.__dict__.get(fn) or getattr(cls, fn, None)
            if f is None or not isinstance(f, types.FunctionType):
                continue
            try:
                f(Self(cls, where + "." + fn), ctx)
                n += 1
            except Exception:
                err("%s.%s raised: %s" % (where, fn, traceback.format_exc().strip().splitlines()[-1]))
    counts['draws'] += n
    return n


def reset_scene():
    try:
        bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    sc = bpy.context.scene
    try:
        sc.arc_flow.fluids.clear()
        sc.arc_flow_fm.fluids.clear()
    except Exception:
        pass
    try:
        runtime.RT.reset_all()
    except Exception:
        pass


# ---- structure checks
names = [c.__name__ for c in ui_classes()]
print("%d Tessera panel / menu classes: %s" % (len(names), ", ".join(names)), flush=True)
ok(len(names) >= 20, "panels and menus registered")
ex = bpy.types.AF_PT_examples
ok('DEFAULT_CLOSED' in ex.bl_options, "Examples panel collapsed by default")
others = [getattr(bpy.types, n) for n in names if n.startswith('AF_PT_') and n != 'AF_PT_examples' and getattr(getattr(bpy.types, n), 'bl_category', '') == 'Tessera']
closed = [c.__name__ for c in others if 'DEFAULT_CLOSED' in getattr(c, 'bl_options', ())]
ok(not closed, "all other Tessera panels expanded by default %s" % closed)
for want in ('AF_PT_shader', 'AF_PT_thermal', 'AF_PT_windtunnel', 'AF_PT_systems', 'AF_PT_retopo', 'AF_PT_srf', 'AF_PT_fm_main', 'AF_PT_fm_fluids', 'AF_PT_main', 'AF_PT_fluids', 'AF_PT_tabs'):
    ok(hasattr(bpy.types, want), "panel %s registered" % want)

sc = bpy.context.scene
sc.arc_flow.cache_dir = tempfile.mkdtemp() + os.sep
sc.frame_start, sc.frame_end = 1, 60

# ---- draw: empty scene in both modes
reset_scene()
for mode in ('FLUID', 'COMBUSTION'):
    try:
        sc.arc_flow_fm.mode = mode
    except Exception:
        # enum identifiers differ: take every available one
        pass
modes = [e.identifier for e in sc.arc_flow_fm.bl_rna.properties['mode'].enum_items]
print("modes:", modes, flush=True)
tabs = list(ui_tabs.TAB_IDS)
for m in modes:
    sc.arc_flow_fm.mode = m
    for tab in tabs:
        sc.arc_flow.ui_tab = tab
        draw_all("empty/%s/%s" % (m, tab))

# ---- draw with objects selected (a cube, a curve, a light, a camera, a text)
for kind, add in (('cube', lambda: bpy.ops.mesh.primitive_cube_add()), ('curve', lambda: bpy.ops.curve.primitive_bezier_curve_add()),
                  ('light', lambda: bpy.ops.object.light_add(type='POINT')), ('text', lambda: bpy.ops.object.text_add()),
                  ('empty', lambda: bpy.ops.object.empty_add())):
    reset_scene()
    add()
    for m in modes:
        sc.arc_flow_fm.mode = m
        for tab in tabs:
            sc.arc_flow.ui_tab = tab
            draw_all("sel_%s/%s/%s" % (kind, m, tab))

# ---- every Example: loads without error, then draw both modes
items = []
for group, lst in SP.EXAMPLES.items():
    for label, idname, props in lst:
        items.append((group, label, idname, props))
print("%d examples" % len(items), flush=True)
ok(len(items) >= 10, "examples listed (%d)" % len(items))
for group, label, idname, props in items:
    reset_scene()
    for m in modes:
        sc.arc_flow_fm.mode = m
        break
    mod, fn = idname.split('.')
    try:
        res = getattr(getattr(bpy.ops, mod), fn)(**props)
        loaded = 'FINISHED' in res
    except Exception:
        loaded = False
        print(traceback.format_exc(), flush=True)
    ok(loaded, "Example loads: %s / %s" % (group, label))
    for m in modes:
        sc.arc_flow_fm.mode = m
        for tab in tabs:
            sc.arc_flow.ui_tab = tab
            draw_all("ex[%s]/%s/%s" % (label, m, tab))
    # select the first mesh and draw again (object properties, surface faces, thermal)
    meshes = [o for o in sc.objects if o.type == 'MESH']
    if meshes:
        for o in sc.objects:
            o.select_set(False)
        meshes[0].select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        for m in modes:
            sc.arc_flow_fm.mode = m
            for tab in tabs:
                sc.arc_flow.ui_tab = tab
                draw_all("ex[%s]+sel/%s/%s" % (label, m, tab))

# ---- Muscle tab with a populated state (example biped, regions, solved muscles)
reset_scene()
try:
    bpy.ops.arcflow.muscle_example()
    bpy.ops.arcflow.muscle_auto_seed()
    bpy.ops.arcflow.muscle_solve()
    for stage in ('solved', 'jiggle'):
        sc.arc_flow_muscle.use_jiggle = stage == 'jiggle'
        for m in modes:
            sc.arc_flow_fm.mode = m
            sc.arc_flow.ui_tab = 'MUSCLE'
            draw_all("muscle[%s]/%s/MUSCLE" % (stage, m))
except Exception:
    err("muscle state setup raised: %s" % traceback.format_exc().strip().splitlines()[-1])

# ---- UI text: no 'demo'
txt = []
for n in names:
    c = getattr(bpy.types, n)
    txt.append((n, getattr(c, 'bl_label', '')))
for s in strings:
    txt.append(('drawn text', s))
for mod in dir(bpy.ops.arcflow):
    try:
        rna = getattr(bpy.ops.arcflow, mod).get_rna_type()
        txt.append(('operator ' + mod, rna.name))
        txt.append(('operator desc ' + mod, rna.description))
    except Exception:
        pass
bad = [(w, t) for w, t in txt if re.search(r'demo', t or '', re.I)]
ok(not bad, "no 'demo' in any panel label, drawn text, operator label or description %s" % bad[:5])

print("%d draw calls, %d layout calls, %d distinct strings" % (counts['draws'], counts['calls'], len(strings)), flush=True)
for e in errors:
    print("FAIL " + e, flush=True)
ok(not errors, "no draw errors (%d)" % len(errors))
ok(counts['draws'] > 200, "enough draws happened (%d)" % counts['draws'])
print("FAILURES: %d" % fails[0], flush=True)
sys.exit(1 if fails[0] else 0)
