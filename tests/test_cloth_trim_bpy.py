"""Headless Blender test of Cloth Trim, Accents and Seam Seal Control Solving:

    bl.sh -b --factory-startup --python tests/test_cloth_trim_bpy.py
"""
import os
import sys

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import arc_flow
arc_flow.register()
from arc_flow import cloth_ops as CO
from arc_flow import cloth_trim_ops as TO
from arc_flow import cloth_trim_panel as TP
from arc_flow import ui_tabs

fails = [0]


def ok(c, msg):
    print(("PASS " if c else "FAIL ") + msg, flush=True)
    if not c:
        fails[0] += 1


print("Blender", bpy.app.version_string, flush=True)
sc = bpy.context.scene
ICONS = {e.identifier for e in bpy.types.UILayout.bl_rna.functions['label'].parameters['icon'].enum_items}

ok(hasattr(sc, 'arc_flow_cloth_trim'), "scene property group registered")
for op in ("trim_add", "trim_remove", "accent_add", "accent_remove", "accent_use_selected", "seam_add", "seam_remove",
           "seam_pull_add", "seam_set", "seam_animate", "trim_group_from_selection", "trim_refresh", "trim_example"):
    ok(hasattr(bpy.ops.arcflow, op), "operator arcflow.%s" % op)
ok(CO.build_solver._tessera_trim and CO.step_scene._tessera_trim, "cloth bridge is extended")
ok(any(e["fn"] is TP.draw_trim_tab for e in ui_tabs.tab_drawers('CLOTH')), "panel mounted in the Cloth tab")
ok(any(e["fn"] is arc_flow.draw_cloth_tab for e in ui_tabs.tab_drawers('CLOTH')), "main cloth drawer still mounted")


# ---- panel draw with a validating stand-in layout
class _L:
    def __init__(self):
        self.bad = []

    def __getattr__(self, name):
        if name in ('row', 'column', 'box', 'split', 'column_flow', 'grid_flow'):
            return lambda *a, **k: self
        if name == 'operator':
            def op(idname, *a, **k):
                mod, nm = idname.split('.')
                if not hasattr(getattr(bpy.ops, mod), nm):
                    self.bad.append("operator " + idname)
                if k.get('icon') and k['icon'] not in ICONS:
                    self.bad.append("icon " + k['icon'])
                return type('P', (), {'__setattr__': lambda s, n, v: None})()
            return op
        if name == 'prop':
            def prop(data, p, *a, **k):
                if p not in data.bl_rna.properties:
                    self.bad.append("prop " + p)
                if k.get('icon') and k['icon'] not in ICONS:
                    self.bad.append("icon " + k['icon'])
            return prop
        if name == 'template_list':
            def tl(t, lid, data, p, idata, ip, **k):
                if not hasattr(bpy.types, t):
                    self.bad.append("list " + t)
            return tl
        if name == 'label':
            def label(*a, **k):
                if k.get('icon') and k['icon'] not in ICONS:
                    self.bad.append("icon " + k['icon'])
            return label
        return lambda *a, **k: self

    def __setattr__(self, k, v):
        if k != 'bad':
            return
        object.__setattr__(self, k, v)


def draw_ok(msg):
    lay = _L()
    TP.draw_trim_tab(lay, bpy.context)
    ok(not lay.bad, msg + (" " + str(lay.bad[:4]) if lay.bad else ""))


# ---- example
msg = TO.load_jacket_example(sc)
ts = sc.arc_flow_cloth_trim
ok(len(ts.trims) == 2 and len(ts.accents) == 3 and len(ts.seams) == 1, "jacket example: 2 trims, 3 accent sets, 1 seam")
ok(not any(w in msg.lower() for w in ("demo",)), "no 'demo' wording")
for i in range(3):
    ts.trim_index = ts.accent_index = ts.seam_index = min(i, 1)
    draw_ok("panel draws with the example (selection %d)" % i)

cl = CO.cloth(sc)
g = cl.garments[0]
S = CO.get_solver(sc, g)
B = S._trim_bound
ok(len(B.trims) == 2 and len(B.accents) == 3 and len(B.seams) == 1, "solver carries 2 trims, 3 accent sets and 1 seam")
seal = B.seams["Zipper"]
ok(len(seal.pairs) >= 20, "seam pairs found (%d)" % len(seal.pairs))
ok(len(S.extra_constraints) == 6, "extra constraints hooked into the solver")

fps = 60.0


def step(n):
    for _ in range(n):
        CO.step_scene(sc, 1.0 / fps)


# the open jacket settles on the mannequin
ts.seams[0].progress = 0.0
step(25)
gap_open = float(seal.gaps(S.x).mean())
ts.seams[0].progress = 0.5
step(40)
gap_half = float(seal.gaps(S.x).mean())
ts.seams[0].progress = 1.0
step(60)
g_closed = seal.gaps(S.x)
print("mean gap open %.3f half %.3f closed %.4f (max %.4f)" % (gap_open, gap_half, g_closed.mean(), g_closed.max()), flush=True)
ok(gap_open > gap_half > g_closed.mean(), "gap shrinks as the slider goes 0 -> 0.5 -> 1")
ok(g_closed.max() < 0.01, "sealed jacket gap below 1 cm (%.4f)" % g_closed.max())
ok(np.isfinite(S.x).all(), "positions finite")

# objects written
coll = bpy.data.collections.get("Tessera Accents")
inst = [o for o in coll.objects if o.get("tessera_accent")]
n_expect = sum(a.n_instances for a in ts.accents)
ok(len(inst) == n_expect and n_expect > 20, "accent instances created (%d)" % len(inst))
ok(bpy.data.objects.get("Trim Hem") is not None and bpy.data.objects.get("Trim Collar") is not None, "trim strips exist")
ok(bpy.data.objects.get("Charms Chains") is not None, "hanging chain object exists")
# instance matrix sits on the cloth at its barycentric point
acc = B.accents["Buttons"]
anc, _F = acc.anchors(S.x)
ob = [o for o in inst if o["tessera_accent"] == "Buttons"][0]
ok(np.allclose(np.array(ob.matrix_world.translation), anc[0], atol=1e-4), "button instance follows its barycentric point")
ch = B.accents["Charms"]
ok(ch.p[:, -1, 2].max() < ch.anchor[:, 2].min() - 0.03, "charms hang below their anchors")
# hem trim attached to the lowest edge of the jacket
hem = B.trims["Hem"]
ok(S.rest[hem.chain, 2].mean() < S.rest[:, 2].min() + 0.02, "hem follows the lowest edge")
col = B.trims["Collar"]
ok(S.rest[col.chain, 2].mean() > S.rest[:, 2].max() - 0.02, "collar follows the highest edge")

# pull empty
bpy.context.view_layer.objects.active = None
r = bpy.ops.arcflow.seam_pull_add()
ok('FINISHED' in r and ts.seams[0].pull is not None, "zipper pull empty created")
ts.seams[0].drive_from_pull = True
head = seal.head_position(S.x)
seal2 = seal
mid = 0.5 * (S.rest[seal.pairs[:, 0]] + S.rest[seal.pairs[:, 1]])
k = int(len(mid) * 0.5)
ts.seams[0].pull.location = tuple(float(c) for c in mid[k])
bpy.context.view_layer.update()
step(1)
print("pull at half the seam -> progress %.2f" % ts.seams[0].progress, flush=True)
ok(0.35 < ts.seams[0].progress < 0.65, "moving the pull sets the slider")
ts.seams[0].drive_from_pull = False

# operators add / remove
n0 = len(ts.trims)
ok('FINISHED' in bpy.ops.arcflow.trim_add(kind='LACE') and len(ts.trims) == n0 + 1, "add lace trim")
ok('FINISHED' in bpy.ops.arcflow.trim_remove() and len(ts.trims) == n0, "remove trim")
n0 = len(ts.accents)
ok('FINISHED' in bpy.ops.arcflow.accent_add(kind='TASSEL') and len(ts.accents) == n0 + 1, "add tassel accents")
ok('FINISHED' in bpy.ops.arcflow.accent_remove() and len(ts.accents) == n0, "remove accents")
n0 = len(ts.seams)
ok('FINISHED' in bpy.ops.arcflow.seam_add(mode='VELCRO') and len(ts.seams) == n0 + 1, "add Velcro seam")
ok('FINISHED' in bpy.ops.arcflow.seam_remove() and len(ts.seams) == n0, "remove seam")
ts.seam_index = 0
r = bpy.ops.arcflow.seam_animate(frame_from=10, frame_to=50)
ok('FINISHED' in r and sc.animation_data is not None and sc.animation_data.action is not None, "key zip animation inserts keyframes")
sc.frame_set(10)
p0 = ts.seams[0].progress
sc.frame_set(30)
p1 = ts.seams[0].progress
sc.frame_set(50)
p2 = ts.seams[0].progress
ok(p0 < 0.01 and 0.3 < p1 < 0.7 and p2 > 0.99, "keyframed slider animates 0 -> 1 (%.2f %.2f %.2f)" % (p0, p1, p2))
ok('FINISHED' in bpy.ops.arcflow.trim_refresh(), "apply trim setup")
draw_ok("panel draws after edits")

# every mode builds
for mode in ('ZIPPER', 'BUTTONS', 'VELCRO', 'MAGNETIC', 'LACES'):
    ts.seams[0].mode = mode
    step(1)
    ok(len(S._trim_bound.seams) == 1 and np.isfinite(S.x).all(), "seam mode %s steps" % mode)
# group helper
jk = g.obj
bpy.context.view_layer.objects.active = jk
for v in jk.data.vertices:
    v.select = v.index < 5
ok('FINISHED' in bpy.ops.arcflow.trim_group_from_selection(group="TestGroup") and jk.vertex_groups.get("TestGroup") is not None,
   "selected vertices stored in a group")

# unregister / register cycle
arc_flow.unregister()
ok(not getattr(CO.build_solver, '_tessera_trim', False), "bridge restored on unregister")
arc_flow.register()
ok(getattr(CO.step_scene, '_tessera_trim', False), "bridge reinstalled on register")
print("FAILED %d" % fails[0] if fails[0] else "ALL PASS", flush=True)
sys.exit(1 if fails[0] else 0)
