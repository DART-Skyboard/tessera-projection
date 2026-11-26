"""Headless Blender UI tests for the Tessera Muscle tab (owned by the Muscle UI worker):

    blender --factory-startup -b --python tests/test_muscle_ui_bpy.py

1. registration: the add-on registers, the Muscle tab drawer, panel, list and UI operators exist; unregister / register cycles
2. panel draw: `draw_muscle_tab` draws without errors through a validating stand-in UILayout (every property, operator,
   operator property and icon must exist) in many states: empty scene, example biped, every preset, every detail level,
   custom creature, every card's muscle active, collapsed cards, solved; every section, card and muscle label is drawn
3. Muscle Pencil: `arcflow.muscle_pencil_stroke` snaps world points to a test mesh (moved / rotated / scaled object) and
   writes a stroke whose consecutive vertices share a mesh edge (it follows the surface), in the active muscle's id;
   counts, Complete and Remove Last Line react
4. preset switching repopulates the cards: Biped <-> Quadruped <-> Custom, variants, detail levels; work on muscles that
   stay (Complete, sliders, geometry) survives the switch; stale-card detection
5. the Solve / Create Shape Keys / Create Geometry Nodes buttons run on the example
"""
import json
import math
import os
import sys
import traceback
import types

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import arc_flow  # noqa: E402

fails = [0]


def ok(c, msg):
    print(("PASS " if c else "FAIL ") + msg, flush=True)
    if not c:
        fails[0] += 1


# ---------------------------------------------------------------------------------------------------------------- 1
arc_flow.register()
from arc_flow import muscle_panel as MP  # noqa: E402
from arc_flow import muscle_ui_ops as MUI  # noqa: E402
from arc_flow import muscle_ops as MO  # noqa: E402
from arc_flow import ui_tabs  # noqa: E402
from arc_flow.core import muscle_solver as MS  # noqa: E402

print("Blender", bpy.app.version_string, flush=True)
UI_OPS = ("muscle_pencil_stroke", "muscle_ui_set_preset", "muscle_ui_stroke_undo", "muscle_ui_cards_expand")
ok(all(hasattr(bpy.ops.arcflow, o) for o in UI_OPS), "UI operators registered: " + ", ".join(UI_OPS))
ok(hasattr(bpy.types, 'AF_PT_muscle') and hasattr(bpy.types, 'AF_UL_muscle_regions'), "Muscle panel and list registered")
ok('MUSCLE' in ui_tabs.TAB_IDS, "Muscle tab is on the Tessera tab rail")
order = list(ui_tabs.TAB_IDS)
ok(order.index('MUSCLE') == order.index('CLOTH') + 1 and order.index('EXAMPLES') == order.index('MUSCLE') + 1,
   "tab order: ... Hair, Cloth, Muscle, Ex (%s)" % ", ".join(order))
ok(any(e['fn'] is MP.draw_muscle_tab for e in ui_tabs.tab_drawers('MUSCLE')), "Muscle tab drawer mounted")
ok(hasattr(bpy.types.WindowManager, 'tp_muscle_overlay'), "pencil overlay toggle registered")
try:
    for _ in range(2):
        arc_flow.unregister()
        arc_flow.register()
    cyc = hasattr(bpy.ops.arcflow, 'muscle_pencil_stroke') and any(e['fn'] is MP.draw_muscle_tab for e in ui_tabs.tab_drawers('MUSCLE'))
except Exception:
    traceback.print_exc()
    cyc = False
ok(cyc, "unregister / register cycles twice cleanly")
ok(hasattr(bpy.types.WindowManager, 'tp_muscle_overlay'), "overlay property re-registered after the cycles")

# ---------------------------------------------------------------------------------------------------------------- 2
ICONS = {e.identifier for e in bpy.types.UILayout.bl_rna.functions['label'].parameters['icon'].enum_items}
errors = []
strings = []


def err(m):
    if m not in errors:
        errors.append(m)


def check_prop(data, name, where):
    if not isinstance(name, str):
        return
    try:
        props = data.bl_rna.properties
    except Exception:
        err("%s: prop() data %r has no RNA" % (where, data))
        return
    if name not in props:
        err("%s: property %r missing on %s" % (where, name, type(data).__name__))


def check_enum(data, name, value, where):
    try:
        items = {i.identifier for i in data.bl_rna.properties[name].enum_items}
    except Exception:
        return
    if value not in items:
        err("%s: enum %s.%s has no item %r" % (where, type(data).__name__, name, value))


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


class FakeLayout:
    """Validating stand-in for UILayout.  `closed` = set of layout-panel ids drawn closed."""

    def __init__(self, where, closed=()):
        object.__setattr__(self, '_w', where)
        object.__setattr__(self, '_closed', closed)

    def __setattr__(self, k, v):
        if k not in ('scale_y', 'scale_x', 'enabled', 'active', 'alert', 'alignment', 'ui_units_x', 'ui_units_y',
                     'use_property_split', 'use_property_decorate', 'emboss', 'operator_context'):
            err("%s: unknown layout attribute %r" % (self._w, k))

    def _icon(self, kw):
        ic = kw.get('icon')
        if isinstance(ic, str) and ic not in ICONS:
            err("%s: invalid icon %r" % (self._w, ic))
        t = kw.get('text')
        if isinstance(t, str):
            strings.append(t)

    def __getattr__(self, name):
        w, closed = self._w, self._closed
        if name in ('row', 'column', 'box', 'split', 'grid_flow', 'column_flow'):
            return lambda *a, **kw: FakeLayout(w, closed)
        if name == 'panel':
            def pnl(idname, default_closed=False):
                shut = (default_closed if closed == 'DEFAULT' else idname in closed)
                return FakeLayout(w + "/" + idname, closed), (None if shut else FakeLayout(w + "/" + idname, closed))
            return pnl
        if name == 'prop':
            def prop(data, p, **kw):
                self._icon(kw)
                check_prop(data, p, w)
            return prop
        if name == 'prop_enum':
            def pe(data, p, value, **kw):
                self._icon(kw)
                check_prop(data, p, w)
                check_enum(data, p, value, w)
            return pe
        if name == 'prop_search':
            def ps(data, p, sdata, sp, **kw):
                self._icon(kw)
                check_prop(data, p, w)
                check_prop(sdata, sp, w)
            return ps
        if name == 'operator':
            def op(idname, **kw):
                self._icon(kw)
                return FakeOp(idname, w)
            return op
        if name == 'label':
            def label(**kw):
                self._icon(kw)
            return label
        if name == 'template_list':
            def tl(listtype, list_id, data, propname, adata, aprop, **kw):
                check_prop(data, propname, w)
                check_prop(adata, aprop, w)
                cls = getattr(bpy.types, listtype, None)
                if cls is None:
                    err("%s: UI list %r missing" % (w, listtype))
                    return
                for i, it in enumerate(getattr(data, propname)):
                    try:
                        cls.draw_item(types.SimpleNamespace(), bpy.context, FakeLayout(w + "/list", closed), data, it, 'NONE', adata, aprop, i)
                    except Exception:
                        err("%s/list.draw_item: %s" % (w, traceback.format_exc().strip().splitlines()[-1]))
            return tl
        if name == 'separator':
            return lambda *a, **kw: None
        err("%s: unexpected layout call %r" % (w, name))
        return lambda *a, **kw: FakeLayout(w, closed)


def draw(tag, closed=()):
    """closed: ids of layout panels drawn closed, or 'DEFAULT' for each panel's default open state."""
    del strings[:]
    n0 = len(errors)
    closed = closed if closed == 'DEFAULT' else set(closed)
    try:
        MP.draw_muscle_tab(FakeLayout(tag, closed), bpy.context)
        # the hosting panel too (hidden from the N-panel stack by ui_tabs, but its draw must work)
        MP.AF_PT_muscle.draw(types.SimpleNamespace(layout=FakeLayout(tag + "/AF_PT_muscle", closed)), bpy.context)
    except Exception:
        err("%s: draw raised %s" % (tag, traceback.format_exc().strip().splitlines()[-1]))
    return len(errors) == n0


sc = bpy.context.scene
ms = sc.arc_flow_muscle
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
ok(draw("empty"), "draws with an empty scene (no skin, no preset)")
ok("Pick a preset to list the muscles" in strings or "Build region cards" in strings, "empty scene shows the preset hint")

bpy.ops.arcflow.muscle_example()
ok(ms.skin is not None and len(ms.regions) > 0, "example biped loaded: %d muscles" % len(ms.regions))
ok(draw("example"), "draws with the example biped")
need = ["Character", "Preset", "Biped", "Quadruped", "Custom", "Detail level", "Region Cards", "Muscle Pencil", "Complete",
        "Initial Emphasis", "Flex Intensity", "Accent", "Toward End A", "Toward End B", "Muscle", "Bone",
        "Resolution Factor", "Solve Muscles", "Create Shape Keys", "Create Geometry Nodes", "Export Notes"]
# section titles are drawn on the panel header (label), so collect them too
missing = [s for s in need if s not in strings]
ok(not missing, "every control of the spec is on the tab (missing: %s)" % missing)
ok(any(s.startswith("Tip: after Solve") for s in strings), "Create Geometry Nodes has a helper tip")
ok(any("GLB" in s for s in strings) and any("FBX" in s for s in strings), "GLB / FBX export notes are shown")
for d in MS.DETAIL_ITEMS:
    ok(d[1].replace(" / ", "/") in strings, "detail level button: %s" % d[1])
labels = {r.label for r in ms.regions}
active_card = MS.card_of(ms.regions[ms.index].group)
# every muscle label of every card is listed (cards are expanded by default)
ok(labels <= set(strings), "every preset muscle is pre-listed on its card (%d)" % len(labels))
for c in ms.cards:
    if any(MS.card_of(r.group) == c.key for r in ms.regions):
        ok(c.label in strings, "card drawn: %s" % c.label)

# each card's muscles active in turn
for c in MS.CARDS:
    idx = [i for i, r in enumerate(ms.regions) if MS.card_of(r.group) == c[0]]
    if idx:
        ms.index = idx[-1]
        ok(draw("active-" + c[0]), "draws with a %s muscle active (%s)" % (c[1], ms.regions[ms.index].label))

# collapsed cards and closed sections
bpy.ops.arcflow.muscle_ui_cards_expand(expand=False)
ok(all(not c.expanded for c in ms.cards), "Collapse all collapses every card")
ok(draw("collapsed", closed='DEFAULT'), "draws with every card collapsed (sections in their default state)")
ok(not (labels & set(strings)), "collapsed cards hide their muscle rows")
bpy.ops.arcflow.muscle_ui_cards_expand(expand=True)
ok(all(c.expanded for c in ms.cards), "Expand all expands every card")
ids = ("tp_muscle_character", "tp_muscle_preset", "tp_muscle_cards", "tp_muscle_pencil", "tp_muscle_solve", "tp_muscle_outputs")
ok(draw("closed", closed=ids), "draws with every section closed")
ok("Solve Muscles" not in strings and "Muscle Pencil" not in strings, "closed sections hide their bodies")
ok(draw("open-all", closed=()), "draws with every section and sub-section open")
ok("All Muscles (list)" in strings and "Animation Intensity" in strings, "sub-sections drawn when open")

# ---------------------------------------------------------------------------------------------------------------- 3
skin = ms.skin
mw0 = skin.matrix_world.copy()
from mathutils import Matrix, Vector  # noqa: E402
skin.matrix_world = Matrix.Translation((0.7, -0.3, 0.2)) @ Matrix.Rotation(math.radians(30), 4, 'Z') @ Matrix.Scale(1.1, 4)
bpy.context.view_layer.update()
k = next(i for i, r in enumerate(ms.regions) if r.key == 'biceps')
ms.index = k
v = MO.basis_coords(skin)
import numpy as np  # noqa: E402
# pick three far-apart vertices on the left upper arm region via the skin weights of the upper arm bone
bi = MO.bone_info(skin, ms.armature)
ua = [j for j, n in enumerate(bi['names']) if 'upper_arm' in n.lower() or 'upperarm' in n.lower()]
ok(bool(ua), "test skin has an upper-arm bone (%s)" % [bi['names'][j] for j in ua])
w = bi['weights'][:, ua[0]]
cand = np.nonzero(w > 0.6)[0]
along = (v[cand] - bi['heads'][ua[0]]) @ (bi['tails'][ua[0]] - bi['heads'][ua[0]])
order_ = cand[np.argsort(along)]
pick = [order_[int(len(order_) * f)] for f in (0.15, 0.5, 0.85)]
pts_world = [skin.matrix_world @ Vector(v[i] * 1.0) for i in pick]
before = len(MO.load_strokes(skin))
n0 = ms.regions[k].n_strokes
r = bpy.ops.arcflow.muscle_pencil_stroke(points=[{"name": "", "co": tuple(p)} for p in pts_world], space='WORLD')
st = MO.load_strokes(skin)
ok(r == {'FINISHED'} and len(st) == before + 1, "pencil stroke operator adds one stroke (%d -> %d)" % (before, len(st)))
last = st[-1]
ok(last['cid'] == ms.regions[k].cid, "stroke carries the active muscle's id (%s)" % ms.regions[k].label)
ok(last['verts'][0] == pick[0] and last['verts'][-1] == pick[-1], "stroke ends snap to the drawn end points")
ok(pick[1] in last['verts'], "stroke passes through the middle point")
edges = {tuple(sorted(e.vertices)) for e in skin.data.edges}
ok(all(tuple(sorted(p)) in edges for p in zip(last['verts'], last['verts'][1:])),
   "stroke follows the surface: consecutive vertices share a mesh edge (%d vertices)" % len(last['verts']))
ok(len(set(last['verts'])) == len(last['verts']), "stroke has no repeated vertices")
ok(ms.regions[k].n_strokes == n0 + 1, "the muscle's line count updates (%d)" % ms.regions[k].n_strokes)
ok(MUI._overlay_key(sc) is not None, "overlay has strokes to draw")
# local-space points and a jittery back-tracking line
loc = [tuple(v[pick[0]]), tuple(v[pick[1]]), tuple(v[pick[0]] * 0.999 + v[pick[1]] * 0.001), tuple(v[pick[2]])]
r = bpy.ops.arcflow.muscle_pencil_stroke(points=[{"name": "", "co": p} for p in loc], space='LOCAL')
st = MO.load_strokes(skin)
ok(r == {'FINISHED'} and len(set(st[-1]['verts'])) == len(st[-1]['verts']), "local-space stroke; back-tracking removed")
r = bpy.ops.arcflow.muscle_pencil_stroke(points=[{"name": "", "co": tuple(v[pick[0]])}], space='LOCAL')
ok(r == {'CANCELLED'}, "a one-point stroke is rejected")
r = bpy.ops.arcflow.muscle_ui_stroke_undo()
ok(r == {'FINISHED'} and ms.regions[k].n_strokes == n0 + 1, "Remove Last Line removes one line")
r = bpy.ops.arcflow.muscle_complete()
ok(r == {'FINISHED'} and ms.regions[k].completed, "Complete marks the pencilled muscle done")
ok(ms.index != k and MS.card_of(ms.regions[ms.index].group) == MS.card_of(ms.regions[k].group),
   "Complete moves to the next muscle of the same card (%s)" % ms.regions[ms.index].label)
ok(draw("after-complete"), "draws after Complete")
# undo the last line of the completed muscle -> it is no longer marked, so not complete
ms.index = k
bpy.ops.arcflow.muscle_ui_stroke_undo()
ok(not ms.regions[k].completed and ms.regions[k].n_strokes == n0, "removing the only line un-completes the muscle")
bpy.ops.arcflow.muscle_pencil_stroke(points=[{"name": "", "co": tuple(p)} for p in pts_world], space='WORLD')
bpy.ops.arcflow.muscle_complete()
skin.matrix_world = mw0
bpy.context.view_layer.update()

# ---------------------------------------------------------------------------------------------------------------- 4
def keys():
    return [r.key for r in ms.regions]


def want(preset, detail):
    cr = dict(heads=ms.heads, arms=ms.arms, legs=ms.legs, tails=ms.tails, fingers=ms.fingers, toes=ms.toes,
              joints_per_limb=ms.joints_per_limb, spine_segments=3)
    return [d['key'] for d in MS.regions_for(preset, detail, cr)]


delt = next(i for i, r in enumerate(ms.regions) if r.key == 'deltoid')
ms.regions[delt].flex = 2.25
ms.regions[delt].accent_end = 'B'
ms.regions[delt].bone_obj = ms.skin   # any mesh: geometry pointer must survive
ms.regions[delt].completed = True
ms.index = delt
bpy.ops.arcflow.muscle_ui_set_preset(family='QUADRUPED')
ok(ms.preset == 'QUADRUPED' and keys() == want('QUADRUPED', ms.detail), "Quadruped repopulates the cards (%d muscles)" % len(ms.regions))
ok('biceps' not in keys() and 'haunch' in keys(), "quadruped cards: no biceps, haunch listed")
d = next(r for r in ms.regions if r.key == 'deltoid')
ok(d.completed and abs(d.flex - 2.25) < 1e-6 and d.accent_end == 'B' and d.bone_obj == ms.skin,
   "work on a muscle that stays (Complete, Flex, Accent end, bone geometry) survives the switch")
ok(ms.regions[ms.index].key == 'deltoid', "the active muscle stays active across the switch")
ok(draw("quadruped"), "draws with the Quadruped preset")
bpy.ops.arcflow.muscle_ui_set_preset(preset='FELINE')
ok(ms.preset == 'FELINE' and keys() == want('FELINE', ms.detail), "Quadruped variant (feline) repopulates")
ok(draw("feline"), "draws with the feline variant")
bpy.ops.arcflow.muscle_ui_set_preset(family='BIPED')
ok(ms.preset == 'BIPED' and 'biceps' in keys(), "back to Biped: biceps listed again")
for p in ('HUMAN_MALE', 'HUMAN_FEMALE'):
    bpy.ops.arcflow.muscle_ui_set_preset(preset=p)
    ok(ms.preset == p and keys() == want(p, ms.detail), "%s preset repopulates" % MS.PRESETS[p]['label'])
    ok(draw(p), "draws with %s" % MS.PRESETS[p]['label'])
pm = next(r for r in ms.regions if r.key == 'pectoralis').emphasis
bpy.ops.arcflow.muscle_ui_set_preset(preset='HUMAN_MALE')
pm2 = next(r for r in ms.regions if r.key == 'pectoralis').emphasis
ok(pm2 > pm, "male / female presets set different emphasis (pectoralis %.2f vs %.2f)" % (pm2, pm))
counts = []
for dd in MS.DETAIL_ITEMS:
    bpy.ops.arcflow.muscle_ui_set_preset(detail=dd[0])
    counts.append(len(ms.regions))
    ok(ms.detail == dd[0] and keys() == want(ms.preset, dd[0]), "detail %s lists %d muscles" % (dd[1], len(ms.regions)))
    ok(draw("detail-" + dd[0]), "draws at detail %s" % dd[1])
ok(counts == sorted(counts) and counts[0] < counts[-1], "more detail lists more muscles %s" % counts)
bpy.ops.arcflow.muscle_ui_set_preset(detail='CARTOON')
ok(set(keys()) <= {'biceps', 'triceps', 'sternocleidomastoid', 'pectoralis', 'deltoid', 'quadriceps', 'hamstrings',
                   'gastrocnemius'} and 'biceps' in keys(), "Cartoon lists only the common big muscles")
bpy.ops.arcflow.muscle_ui_set_preset(family='CUSTOM', detail='STANDARD')
ms.heads, ms.tails, ms.arms, ms.legs, ms.joints_per_limb = 2, 1, 4, 4, 3
ok(MUI.cards_stale(ms), "changing creature counts flags the cards as stale")
ok(draw("custom-stale") and "Rebuild cards for these settings" in strings, "stale cards show the Rebuild button")
bpy.ops.arcflow.muscle_ui_set_preset()
ok(not MUI.cards_stale(ms) and 'tail_base' in keys() and 'head_mass' in keys() and 'limb_mid' in keys(),
   "custom creature (2 heads, tail, 8 limbs, 3 joints) lists tail, extra-head and limb muscles")
ok(draw("custom2"), "custom creature draws without errors")
n_before = len(ms.regions)
try:
    r = bpy.ops.arcflow.muscle_ui_set_preset(preset='NOPE')
except RuntimeError as ex:
    r = {'CANCELLED'} if 'Unknown preset' in str(ex) else {'ERR'}
ok(r == {'CANCELLED'} and len(ms.regions) == n_before, "an unknown preset is rejected and the cards are untouched")

# ---------------------------------------------------------------------------------------------------------------- 5
bpy.ops.arcflow.muscle_ui_set_preset(preset='BIPED', detail='STANDARD')
bpy.ops.arcflow.muscle_auto_seed()
r = bpy.ops.arcflow.muscle_solve()
ok(r == {'FINISHED'} and ms.solved, "Solve button: %s" % ms.status)
ms.resolution = 0.5
ok(abs(ms.resolution - 0.85) < 1e-6, "resolution factor clamps at 0.85")
ms.resolution = 50
ok(abs(ms.resolution - 10.0) < 1e-6, "resolution factor clamps at 10.0")
ms.resolution = 1.0
r1 = bpy.ops.arcflow.muscle_bake()
ok(r1 == {'FINISHED'} and ms.skin.data.shape_keys is not None and len(ms.skin.data.shape_keys.key_blocks) > 1,
   "Create Shape Keys button: %d keys" % (len(ms.skin.data.shape_keys.key_blocks) if ms.skin.data.shape_keys else 0))
r2 = bpy.ops.arcflow.muscle_gn()
ok(r2 == {'FINISHED'} and any(m.type == 'NODES' for m in ms.skin.modifiers), "Create Geometry Nodes button adds the modifier")
ok(draw("solved"), "draws after solve / bake / geometry nodes")

ok(not errors, "no UI validation errors (%d)" % len(errors))
for e in errors[:40]:
    print("  ERR", e)
print("FAILS", fails[0], flush=True)
sys.exit(1 if fails[0] else 0)
