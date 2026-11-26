"""Headless Blender test of the Tessera muscle solver:

    blender --factory-startup -b --python tests/test_muscle_bpy.py

Builds a procedural biped skin + armature, auto-seeds / paints regions, draws contours, solves, bakes shape keys,
checks that deformation grows with the flex slider and that the skin stays closed, drivers react to joint rotation,
the Geometry Nodes group evaluates, the control rig works, and GLB / FBX / JSON exports carry the data.
"""
import json
import os
import struct
import sys
import tempfile
import time

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import arc_flow
arc_flow.register()
from arc_flow import muscle_ops as MO
from arc_flow import muscle_gn
from arc_flow.core import muscle_solver as MS

fails = [0]


def ok(c, msg):
    print(("PASS " if c else "FAIL ") + msg, flush=True)
    if not c:
        fails[0] += 1


TMP = tempfile.mkdtemp(prefix="tessera_muscle_", dir="/tmp")
print("Blender", bpy.app.version_string, "tmp", TMP, flush=True)
sc = bpy.context.scene
ms = sc.arc_flow_muscle
ok(ms is not None and hasattr(bpy.types, 'AF_PT_muscle') and hasattr(bpy.ops.arcflow, 'muscle_solve'), "muscle properties, panel and operators registered")
ok('MUSCLE' in arc_flow.ui_tabs.TAB_IDS, "MUSCLE tab is in the vertical tab rail")
ok(any(e['fn'] is arc_flow.muscle_panel.draw_muscle_tab for e in arc_flow.ui_tabs.tab_drawers('MUSCLE')), "MUSCLE tab drawer registered")

# ---- example biped
t0 = time.time()
bpy.ops.arcflow.muscle_example()
skin, arm = ms.skin, ms.armature
ok(skin is not None and arm is not None and len(skin.data.vertices) > 1000, "example biped: %d vertices, %d bones" % (len(skin.data.vertices), len(arm.data.bones)))


def edge_face_counts(obj):
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    cnt = np.array([len(e.link_faces) for e in bm.edges])
    bm.free()
    return cnt


cnt = edge_face_counts(skin)
ok((cnt == 2).all(), "rest skin is closed and manifold (%d edges, %d non-manifold)" % (len(cnt), int((cnt != 2).sum())))
ok(len(ms.regions) == len(MS.regions_for('BIPED', 'STANDARD')), "Biped preset created %d regions" % len(ms.regions))

# ---- preset switching
ms.preset = 'QUADRUPED'
bpy.ops.arcflow.muscle_apply_preset()
keys_q = {r.key for r in ms.regions}
ok('haunch' in keys_q and 'biceps' not in keys_q, "switching to Quadruped swaps the region set")
ms.preset = 'BIPED'
ms.detail = 'CARTOON'
bpy.ops.arcflow.muscle_apply_preset()
n_cartoon = len(ms.regions)
ms.detail = 'CINEMATIC'
bpy.ops.arcflow.muscle_apply_preset()
n_cin = len(ms.regions)
ok(n_cartoon < n_cin, "Cartoon (%d) vs Cinematic (%d) region lists" % (n_cartoon, n_cin))
ms.preset, ms.detail = 'BIPED', 'STANDARD'
bpy.ops.arcflow.muscle_apply_preset()

# ---- paint: pencil + auto seed
reg = {r.key: r for r in ms.regions}
v = MO.basis_coords(skin)
ms.index = [r.key for r in ms.regions].index('zygomaticus')
cheek = v[np.argmin(np.linalg.norm(v - np.array([0.06, -0.09, 1.68]), axis=1))]
n = MO.paint_vertices(skin, ms, cheek, 0.05, reg["zygomaticus"].cid)
ids = MO.read_ids(skin)
ok(n > 3 and (ids == reg['zygomaticus'].cid).sum() == n, "pencil paints %d vertices with the cheek id" % n)
ca = skin.data.color_attributes.get(MO.ATTR_COLOR)
col = np.empty(len(ids) * 4, np.float32)
ca.data.foreach_get('color', col)
col = col.reshape(-1, 4)
ok(np.allclose(col[ids == reg['zygomaticus'].cid][0, :3], tuple(reg['zygomaticus'].color), atol=1e-5), "painted vertices carry the static region colour")
MO.paint_vertices(skin, ms, cheek, 0.05, 0, erase=True)
ok((MO.read_ids(skin) == 0).all(), "eraser clears the paint")
nseed = bpy.ops.arcflow.muscle_auto_seed()
ids = MO.read_ids(skin)
ok((ids > 0).sum() > 800, "auto-seed from skeleton painted %d vertices" % (ids > 0).sum())
painted_keys = {r.key for r in ms.regions if r.n_painted}
ok({'quadriceps', 'hamstrings', 'gastrocnemius', 'pectoralis', 'sternocleidomastoid'} <= painted_keys, "auto-seed covers legs, chest and neck: %d regions" % len(painted_keys))

# ---- contour: a stroke down the front of the left thigh, plus selection chain helper
zs = np.linspace(0.46, 0.88, 12)
chain = [int(np.argmin(np.linalg.norm(v - np.array([0.11, -0.13 + 0.0 * z, z]), axis=1))) for z in zs]
chain = [c for i, c in enumerate(chain) if i == 0 or c != chain[i - 1]]
ms.index = [r.key for r in ms.regions].index('quadriceps')
MO.save_strokes(skin, [dict(cid=reg['quadriceps'].cid, verts=chain)])
ob = MO.update_contour_curve(sc, skin)
ok(ob is not None and len(ob.data.splines) == 1 and len(ob.data.splines[0].points) == len(chain), "contour stroke is stored and shown as a curve")
grid = bpy.data.objects.new("grid", bpy.data.meshes.new("grid"))
gm = grid.data
vv = [(i, j, 0) for j in range(3) for i in range(5)]
gm.from_pydata(vv, [(j * 5 + i, j * 5 + i + 1) for j in range(3) for i in range(4)], [])
sc.collection.objects.link(grid)
for vx in gm.vertices:
    vx.select = False
for i in range(5):
    gm.vertices[5 + i].select = True
for e in gm.edges:
    e.select = all(gm.vertices[k].select for k in e.vertices)
chain2 = MO.selected_vertex_chain(grid)
ok(chain2 == [5, 6, 7, 8, 9], "selected edge chain is ordered along the line: %s" % chain2)
bpy.data.objects.remove(grid)

# ---- region category cards + Muscle Pencil "Complete" workflow
ok([c.key for c in ms.cards] == [c[0] for c in MS.CARDS], "region cards created: %s" % [c.label for c in ms.cards])
card_of = {r.key: MS.card_of(r.group) for r in ms.regions}
ok(card_of['zygomaticus'] == 'HEAD' and card_of['biceps'] == 'ARM' and card_of['pectoralis'] == 'TORSO'
   and card_of['quadriceps'] == 'LEG', "muscles land on the head / arm / torso / leg cards")
ok(all(MS.card_of(g) in {c[0] for c in MS.CARDS} for g in MS.GROUPS), "every catalog group belongs to a card")
MO.refresh_stroke_counts(skin, ms)
ok(reg['quadriceps'].n_strokes == 1, "Muscle Pencil line counted on the quadriceps (%d)" % reg['quadriceps'].n_strokes)
qi = [r.key for r in ms.regions].index('quadriceps')
ms.index = qi
ok(bpy.ops.arcflow.muscle_complete() == {'FINISHED'} and reg['quadriceps'].completed, "Complete books the marked quadriceps into its card")
ok(MS.card_of(ms.regions[ms.index].group) == 'LEG' and ms.index != qi and not ms.regions[ms.index].completed,
   "after Complete the next unmarked muscle of the Legs card is active: %s" % ms.regions[ms.index].label)
# a muscle with no marks at all cannot be completed
nk = 'brachialis'
bi = MO.add_catalog_region(ms, nk)
ids_ = MO.read_ids(skin)
ok((ids_ == ms.regions[bi].cid).sum() == 0, "freshly added %s has no paint" % nk)
ms.index = bi
ok(bpy.ops.arcflow.muscle_complete() == {'CANCELLED'} and not ms.regions[bi].completed, "Complete refuses an unmarked muscle")
arm_card = next(c for c in ms.cards if c.key == 'ARM')
items = [i['key'] for i in MS.CATALOG if MS.card_of(i['group']) == 'ARM' and i['key'] not in {r.key for r in ms.regions}]
if items:
    arm_card.pick = items[0]
    n0 = len(ms.regions)
    ok(bpy.ops.arcflow.muscle_card_add(card='ARM') == {'FINISHED'} and len(ms.regions) == n0 + 1
       and ms.regions[ms.index].key == items[0], "card Add menu adds %s to the Arms card" % items[0])
    ms.regions.remove(ms.index)
ms.regions.remove(bi)
ms.index = qi

# ---- solve
t0 = time.time()
r = bpy.ops.arcflow.muscle_solve()
dt = time.time() - t0
insts = MO.get_instances(sc)
ok(ms.solved and len(insts) >= 12, "solve produced %d muscle instances in %.2f s" % (len(insts), dt))
names = {i.name for i in insts}
ok({'quadriceps.L', 'quadriceps.R', 'biceps.L', 'biceps.R'} <= names, "left / right instances are separate: %s" % sorted(names))
q = next(i for i in insts if i.name == 'quadriceps.L')
ok(q.source == 'stroke', "left quadriceps uses the contour stroke as its fibre axis")
ok(q.bone is not None and q.bone.startswith('thigh'), "driving bone detected from the skin weights: %s" % q.bone)
w_all = np.zeros(len(v))
for i in insts:
    w_all[i.idx] += i.w
ok(w_all.max() <= 1.0001 + 1.0 and (w_all > 0).sum() > 600, "flex weights cover %d vertices" % (w_all > 0).sum())
a = skin.data.attributes
ok(all(a.get(n) is not None for n in (MO.ATTR_BULGE, MO.ATTR_SLIDE, MO.ATTR_FLEX, MO.ATTR_ID)), "solve data stored as skin point attributes")
ok(bpy.data.texts.get(MO.TEXT_SOLVE) is not None and len(bpy.data.texts[MO.TEXT_SOLVE].as_string()) > 1000, "solve data stored in a text block (saved in the .blend)")
MO._CACHE.clear()
ok(len(MO.get_instances(sc)) == len(insts), "solve data reloads from the text block")

# ---- resolution factor is a real knob
ms.resolution = 0.3
ok(abs(ms.resolution - 0.85) < 1e-6, "resolution factor clamps at 0.85 (spec range 0.85 .. 10)")
ms.resolution = 12.0
ok(abs(ms.resolution - 10.0) < 1e-6, "resolution factor clamps at 10.0")
ms.resolution = 0.85
MO.run_solve(sc)
coarse = MO.get_instances(sc)
ms.resolution = 3.0
MO.run_solve(sc)
fine = MO.get_instances(sc)


def roughness(ins):
    g = MS.Graph(v, MO.skin_arrays(skin)[1])
    t = 0.0
    for i in ins:
        full = np.zeros(len(v))
        full[i.idx] = i.w
        t += float(np.abs(full[g.edges[:, 0]] - full[g.edges[:, 1]]).sum())
    return t
rc, rf_ = roughness(coarse), roughness(fine)
ok(rf_ > rc, "resolution 3.0 has more weight detail than 0.85 (%.1f vs %.1f)" % (rf_, rc))
ms.resolution = 10.0
MO.run_solve(sc)
r10 = roughness(MO.get_instances(sc))
ok(r10 > rf_, "resolution 10.0 adds more detail than 3.0 (%.1f vs %.1f)" % (r10, rf_))
mass = lambda ins: sum(float(i.w.sum()) for i in ins)
ok(mass(MO.get_instances(sc)) >= 0.9 * mass(coarse), "finer resolution does not weaken the muscles (weight mass %.0f vs %.0f)"
   % (mass(MO.get_instances(sc)), mass(coarse)))
ms.resolution = 1.0
MO.run_solve(sc)
insts = MO.get_instances(sc)

# ---- bake shape keys and check flex slider response
def key_disp(name):
    kb = skin.data.shape_keys.key_blocks[name]
    co = np.empty(len(kb.data) * 3)
    kb.data.foreach_get('co', co)
    return np.linalg.norm(co.reshape(-1, 3) - MO.basis_coords(skin), axis=1)

bpy.ops.arcflow.muscle_bake()
sk = skin.data.shape_keys
kn = [k.name for k in sk.key_blocks if k.name.startswith(MO.KEY_PREFIX)]
ok(len(kn) == len({i.name for i in insts}) and kn, "baked %d muscle shape keys" % len(kn))
d1 = key_disp(MO.KEY_PREFIX + 'quadriceps.L').max()
reg['quadriceps'].flex = 2.0
bpy.ops.arcflow.muscle_bake()
d2 = key_disp(MO.KEY_PREFIX + 'quadriceps.L').max()
ok(d1 > 0.002 and abs(d2 / d1 - 2.0) < 0.05, "key displacement follows the Flex slider: %.4f m -> %.4f m" % (d1, d2))
reg['quadriceps'].flex = 1.0
leg = next(c for c in ms.cards if c.key == 'LEG')
leg.flex = 2.0
bpy.ops.arcflow.muscle_bake()
d3 = key_disp(MO.KEY_PREFIX + 'quadriceps.L').max()
ok(abs(d3 / d1 - 2.0) < 0.05, "Legs card Flex intensity scales its muscles: %.4f m -> %.4f m" % (d1, d3))
d_arm = key_disp(MO.KEY_PREFIX + 'biceps.L').max()
leg.flex = 1.0
bpy.ops.arcflow.muscle_bake()
ok(abs(key_disp(MO.KEY_PREFIX + 'biceps.L').max() - d_arm) < 1e-6, "Legs card slider leaves the Arms card alone")
leg.emphasis = 0.5
bpy.ops.arcflow.muscle_bake()
d4 = key_disp(MO.KEY_PREFIX + 'quadriceps.L').max()
ok(abs(d4 / d1 - 0.5) < 0.05, "Legs card Initial emphasis scales its muscles: %.4f m -> %.4f m" % (d1, d4))
leg.emphasis = 1.0
reg['quadriceps'].accent = 0.0
bpy.ops.arcflow.muscle_bake()
d_a0 = key_disp(MO.KEY_PREFIX + 'quadriceps.L')
reg['quadriceps'].accent = 1.5
bpy.ops.arcflow.muscle_bake()
d_a1 = key_disp(MO.KEY_PREFIX + 'quadriceps.L')
ok(d_a1.max() > d_a0.max() + 1e-6, "Accent slider changes the baked deformation (%.4f -> %.4f)" % (d_a0.max(), d_a1.max()))
reg['quadriceps'].accent = 0.6
bpy.ops.arcflow.muscle_bake()

# evaluated mesh with all keys at 1: closed, finite, larger volume, bounded displacement
for k in sk.key_blocks:
    if k.name.startswith(MO.KEY_PREFIX):
        k.driver_remove('value') if False else None
ad = sk.animation_data
if ad is not None:
    for fc in ad.drivers:
        fc.mute = True


def evaluated(obj, mods=True):
    obj.update_tag()  # Blender 5.2 does not always tag the modifier after an RNA input change
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get('co', co)
    fcount = len(me.polygons)
    tri = []
    for p in me.polygons:
        vs = list(p.vertices)
        for j in range(1, len(vs) - 1):
            tri.append((vs[0], vs[j], vs[j + 1]))
    ev.to_mesh_clear()
    return co.reshape(-1, 3), np.array(tri)


def volume(P, T):
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    return float(np.einsum('ij,ij->i', a, np.cross(b, c)).sum() / 6.0)

skin.modifiers['Armature'].show_viewport = False
for k in sk.key_blocks:
    if k.name.startswith(MO.KEY_PREFIX):
        k.value = 0.0
P0, T = evaluated(skin)
for k in sk.key_blocks:
    if k.name.startswith(MO.KEY_PREFIX):
        k.value = 1.0
P1, _ = evaluated(skin)
disp = np.linalg.norm(P1 - P0, axis=1)
ok(np.isfinite(P1).all() and disp.max() > 0.003, "all muscles flexed: max displacement %.4f m, mean %.5f m" % (disp.max(), disp.mean()))
ok(disp.max() < 0.08, "displacement stays bounded (%.4f m < 0.08 m)" % disp.max())
v0, v1 = volume(P0, T), volume(P1, T)
ok(v1 > v0, "skin volume grows with flex: %.5f -> %.5f m^3 (%.2f %%)" % (v0, v1, 100 * (v1 / v0 - 1)))
ok((edge_face_counts(skin) == 2).all() and len(P1) == len(v), "skin topology stays closed and manifold after flexing")
half = {}
for k in sk.key_blocks:
    if k.name.startswith(MO.KEY_PREFIX):
        k.value = 0.5
P5, _ = evaluated(skin)
ok(np.linalg.norm(P5 - P0, axis=1).max() < disp.max() and np.linalg.norm(P5 - P0, axis=1).max() > 0, "half key value gives half deformation (monotonic)")
skin.modifiers['Armature'].show_viewport = True
if ad is not None:
    for fc in ad.drivers:
        fc.mute = False

# ---- drivers follow joint rotation and animation state
ok(ad is not None and len(ad.drivers) >= 10, "drivers created on %d muscle keys" % (len(ad.drivers) if ad else 0))
kb = sk.key_blocks[MO.KEY_PREFIX + 'biceps.L']
bpy.context.view_layer.update()
v_rest = kb.value
arm_pb = arm.pose.bones[next(i.bone for i in insts if i.name == 'biceps.L')]
arm_pb.rotation_mode = 'XYZ'
arm_pb.rotation_euler = (0.0, 0.0, 1.0)
bpy.context.view_layer.update()
sk.id_data.update_tag()
bpy.context.evaluated_depsgraph_get().update()
v_bent = sk.key_blocks[MO.KEY_PREFIX + 'biceps.L'].value
ok(v_bent > v_rest + 0.2, "biceps key value follows bone rotation: rest %.2f, bent %.2f" % (v_rest, v_bent))
ms.state = 'RUN'
bpy.context.evaluated_depsgraph_get().update()
v_run = sk.key_blocks[MO.KEY_PREFIX + 'biceps.L'].value
ms.state = 'REST'
bpy.context.evaluated_depsgraph_get().update()
v_rst = sk.key_blocks[MO.KEY_PREFIX + 'biceps.L'].value
ok(v_run > v_rst, "animation state changes intensity: rest %.2f < run %.2f" % (v_rst, v_run))
arm_pb.rotation_euler = (0, 0, 0)
ms.state = 'WALK'

# ---- Geometry Nodes
bpy.ops.arcflow.muscle_gn()
mod = skin.modifiers.get("Tessera Muscles")
ok(mod is not None and mod.node_group is not None, "Create Geometry Nodes adds the Tessera Muscles modifier")
idx_gn = list(skin.modifiers).index(mod)
idx_arm = [i for i, m in enumerate(skin.modifiers) if m.type == 'ARMATURE'][0]
ok(idx_gn < idx_arm, "GN modifier sits before the Armature modifier (%d < %d)" % (idx_gn, idx_arm))
ng = mod.node_group
ins_names = {s.name for s in ng.interface.items_tree if s.in_out == 'INPUT'}
out_names = {s.name for s in ng.interface.items_tree if s.in_out == 'OUTPUT'}
ok({'Geometry', 'Flex', 'Accent', 'Region'} <= ins_names and {'Geometry', 'Offset', 'Region ID', 'Flex Weight'} <= out_names,
   "GN sockets: in %s out %s" % (sorted(ins_names), sorted(out_names)))
skin.modifiers['Armature'].show_viewport = False
for k in sk.key_blocks:
    if k.name.startswith(MO.KEY_PREFIX):
        k.value = 0.0
if ad is not None:
    for fc in ad.drivers:
        fc.mute = True
for k in sk.key_blocks:
    if k.name.startswith(MO.KEY_PREFIX):
        k.value = 0.0
from arc_flow import gn as GNH
setin = lambda **k: GNH.set_modifier_inputs(mod, {kk.replace('_', ' '): vv for kk, vv in k.items()})
setin(Flex=0.0)
G0, _ = evaluated(skin)
setin(Flex=1.0)
G1, _ = evaluated(skin)
gd = np.linalg.norm(G1 - G0, axis=1)
ok(gd.max() > 0.003 and np.isfinite(G1).all(), "GN group evaluates: Flex 1 moves skin by up to %.4f m" % gd.max())
setin(Region=reg['quadriceps'].cid)
G2, _ = evaluated(skin)
gd2 = np.linalg.norm(G2 - G0, axis=1)
ok(0 < (gd2 > 1e-7).sum() < (gd > 1e-7).sum(), "Region input isolates one region (%d of %d points)" % ((gd2 > 1e-7).sum(), (gd > 1e-7).sum()))
setin(Region=-1)
setin(Flex=0.0)
ok((edge_face_counts(skin) == 2).all(), "skin still manifold with the GN modifier")
mod.show_viewport = False
skin.modifiers['Armature'].show_viewport = True
if ad is not None:
    for fc in ad.drivers:
        fc.mute = False

# ---- control rig
made_c, made_ik = MO.build_control_rig(sc, ik=True)
ok(len(made_c) >= 15 and len(made_ik) >= 4, "control rig: %d FK controllers, %d IK handles" % (len(made_c), len(made_ik)))
pbs = arm.pose.bones
ok(all(c.name.startswith('CTRL_') for c in [pbs[n] for n in made_c]) and pbs['upper_arm.L'].constraints.get("TP FK") is not None, "controllers drive the deform bones through Copy Rotation")
ok(pbs['forearm.L'].constraints.get("TP IK") is not None and pbs['forearm.L'].constraints["TP IK"].chain_count == 2, "forearm has an IK constraint to its handle")
pbs['CTRL_upper_arm.L'].rotation_mode = 'XYZ'
before = np.array(pbs['upper_arm.L'].matrix.to_euler())
pbs['CTRL_upper_arm.L'].rotation_euler = (0.0, 0.0, 0.8)
bpy.context.view_layer.update()
after = np.array(pbs['upper_arm.L'].matrix.to_euler())
ok(np.abs(after - before).max() > 0.3, "rotating the controller rotates the deform bone")
pbs['CTRL_upper_arm.L'].rotation_euler = (0, 0, 0)
ok(all(not arm.data.bones[n].use_deform for n in made_c), "controllers are non-deforming bones")

# ---- JSON sidecar
ms.export_path = os.path.join(TMP, "biped")
bpy.ops.arcflow.muscle_export_json()
side = json.load(open(os.path.join(TMP, "biped.muscles.json")))
ok(side['format'] == 'tessera-muscles' and len(side['muscles']) == len(kn) and side['vertex_count'] == len(v), "sidecar JSON: %d muscles" % len(side['muscles']))
m0 = side['muscles'][0]
ok(len(m0['vertices']) == len(m0['weights']) == len(m0['morph_delta']) and m0['control']['driver_bone'], "sidecar has per-muscle weights, morph deltas and a control mapping")
ok(len(side['control_rig']) >= 15, "sidecar lists the control rig mapping (%d controllers)" % len(side['control_rig']))

# ---- GLB export
def read_glb(path):
    with open(path, 'rb') as f:
        magic, ver, ln = struct.unpack('<4sII', f.read(12))
        clen, ctype = struct.unpack('<I4s', f.read(8))
        return json.loads(f.read(clen))

for k in sk.key_blocks:
    if k.name.startswith(MO.KEY_PREFIX):
        k.value = 0.0
ms.export_format = 'GLB'
bpy.ops.arcflow.muscle_export_game()
gj = read_glb(os.path.join(TMP, "biped.glb"))
prim = gj['meshes'][0]['primitives'][0]
tn = gj['meshes'][0].get('extras', {}).get('targetNames', [])
ok(len(prim.get('targets', [])) == len(kn), "GLB carries %d morph targets (shape keys)" % len(prim.get('targets', [])))
ok(set(tn) >= {MO.KEY_PREFIX + 'biceps.L'} and len(tn) == len(kn), "GLB morph target names preserved")
ok('JOINTS_0' in prim['attributes'] and 'WEIGHTS_0' in prim['attributes'] and gj.get('skins'), "GLB carries skin weights and the skeleton")
ok('COLOR_0' in prim['attributes'], "GLB carries the region vertex colours (COLOR_0)")
nd_ = next((nd for nd in gj['nodes'] if nd.get('mesh') == 0), {})
ex_ = nd_.get('extras', {})
try:
    tpm = json.loads(ex_.get('tp_muscles', '{}'))
except Exception:
    tpm = {}
ok(tpm.get('format') == 'tessera-muscles' and len(tpm.get('muscles', [])) == len(kn)
   and all(m['shape_key'] for m in tpm['muscles']) and set(tpm.get('cards', {})) == {c[0] for c in MS.CARDS},
   "GLB node extras carry the muscle rig (%d muscles -> shape keys, bones, card sliders)" % len(tpm.get('muscles', [])))

# ---- FBX export
ms.export_format = 'FBX'
bpy.ops.arcflow.muscle_export_game()
fp = os.path.join(TMP, "biped.fbx")
blob = open(fp, 'rb').read()
ok(os.path.getsize(fp) > 10000 and b'Kaydara FBX Binary' in blob[:30], "FBX written (%d bytes)" % os.path.getsize(fp))
ok(blob.count(b'BlendShape') >= 3 or blob.count(b'Shape') >= 3, "FBX contains blend shape data")
ok(b'tp_muscles' in blob, "FBX carries the muscle rig as a user property (tp_muscles)")

# ---- save / reload keeps the data
blend = os.path.join(TMP, "muscle.blend")
bpy.ops.wm.save_as_mainfile(filepath=blend)
bpy.ops.wm.open_mainfile(filepath=blend)
sc = bpy.context.scene
ms = sc.arc_flow_muscle
ok(ms.solved and len(ms.regions) > 10 and bpy.data.objects['Muscle Skin'].data.attributes.get(MO.ATTR_BULGE) is not None
   and bpy.data.texts.get(MO.TEXT_SOLVE) is not None, "muscle data survives save and reload of the .blend")

print("FAILS", fails[0], flush=True)
sys.exit(1 if fails[0] else 0)
