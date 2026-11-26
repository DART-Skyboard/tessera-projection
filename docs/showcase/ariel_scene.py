"""Build the Ariel example scene for the Tessera Projection Muscle and Hair previews (Blender 5.2.3+).

    blender -b --factory-startup --python docs/showcase/ariel_scene.py -- --dm /path/to/dartmeadow-space --out ariel.blend

Reads Ariel (Journey of the Skyboard) READ-ONLY from a local clone of dartsolarpunk/dartmeadow-space:
  characterassests/Meshy_AI_Autumn_biped_Character_output.glb            mesh + skeleton
  characterassests/Meshy_AI_Autumn_biped_Meshy_AI_Meshy_Merged_Animations.glb  animation clips
  characterassests/ariel_hair_strands.bin                                 "DHS1" groom (5,665 strands x 14 points, Head-bone space)
Nothing from the game is copied into the extension package or this repository.

The scene: Ariel as a neutral matte anatomical figure (one clay material, no skin texture, accessories removed; her fitted shirt,
trousers and shoes are joined in as body surface because the game mesh has no legs / hips under them), her strand
hair converted back to Blender Empty Hair Curves on the Scalp, a walk / wave clip on her rig, the Tessera muscle solve
(Human female preset) baked to bone-driven shape keys, and the Tessera hair solver with Scalp / body colliders."""
import json
import math
import os
import struct
import sys

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []


def arg(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


DM = arg('--dm', '/workspace/jots-v383')
OUT = arg('--out', '/workspace/tp7demo/ariel_tessera.blend')
CLIP = arg('--clip', 'Walking')
DETAIL = arg('--detail', 'STANDARD')
CA = os.path.join(DM, 'characterassests')
BODY, SCALP = 'Ariel Reach.001', 'Scalp'
SHIRT, PANTS, SHOES = 'Shirt.001', 'Pants Reach Pose.001', 'Shoes.001'


def read_glb_json(path):
    with open(path, 'rb') as f:
        f.read(12)
        ln, _t = struct.unpack('<I4s', f.read(8))
        return json.loads(f.read(ln))


def node_world_gltf(g, name):
    """World matrix (glTF, Y-up) of the node called `name` at rest (node TRS chain)."""
    parent = {}
    for i, n in enumerate(g['nodes']):
        for c in n.get('children', []):
            parent[c] = i

    def local(n):
        if 'matrix' in n:
            return Matrix([n['matrix'][i::4] for i in range(4)])
        t = Vector(n.get('translation', (0, 0, 0)))
        q = n.get('rotation', (0, 0, 0, 1))
        s = n.get('scale', (1, 1, 1))
        return Matrix.Translation(t) @ Quaternion((q[3], q[0], q[1], q[2])).to_matrix().to_4x4() @ Matrix.Diagonal((*s, 1))
    idx = next(i for i, n in enumerate(g['nodes']) if n.get('name') == name)
    M = Matrix.Identity(4)
    i = idx
    while i is not None:
        M = local(g['nodes'][i]) @ M
        i = parent.get(i)
    return M


Y_UP_TO_Z_UP = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))


def read_dhs1(path):
    b = open(path, 'rb').read()
    jl = struct.unpack_from('<I', b, 4)[0]
    meta = json.loads(b[12:12 + jl])
    off = 12 + jl
    N, K = meta['count'], meta['points']
    pos = np.frombuffer(b, np.float32, N * K * 3, off).reshape(N, K, 3).astype(np.float64)
    off += N * K * 12
    group = np.frombuffer(b, np.uint8, N, off)
    return meta, pos, group


def clay_material():
    m = bpy.data.materials.get("TP Anatomical Clay") or bpy.data.materials.new("TP Anatomical Clay")
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.62, 0.60, 0.58, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.62
    return m


def densify_limbs(body, arm, cuts=2):
    names = {vg.index: vg.name.lower() for vg in body.vertex_groups}
    n = len(body.data.vertices)
    dom = np.full(n, '', object)
    best = np.zeros(n)
    for v in body.data.vertices:
        for g in v.groups:
            if g.weight > best[v.index]:
                best[v.index], dom[v.index] = g.weight, names.get(g.group, '')
    ext = np.array([any(k in d for k in ('hand', 'head', 'jaw', 'eye', 'toe', 'foot')) for d in dom])
    bpy.context.view_layer.objects.active = body
    for o in list(bpy.context.scene.objects):
        if o is not None:
            o.select_set(o is body)
    sel = np.zeros(len(body.data.polygons), bool)
    for p in body.data.polygons:
        sel[p.index] = not ext[list(p.vertices)].any()
    body.data.polygons.foreach_set('select', sel)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_mode(type='FACE')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.mode_set(mode='OBJECT')
    vsel = np.zeros(n, bool)
    for p in body.data.polygons:
        if sel[p.index]:
            vsel[list(p.vertices)] = True
    ev = np.zeros(len(body.data.edges) * 2, np.int64)
    body.data.edges.foreach_get('vertices', ev)
    ev = ev.reshape(-1, 2)
    body.data.vertices.foreach_set('select', vsel)
    body.data.edges.foreach_set('select', vsel[ev[:, 0]] & vsel[ev[:, 1]])
    body.data.polygons.foreach_set('select', sel)
    print("ARIEL body: subdividing %d of %d faces" % (int(sel.sum()), len(sel)), flush=True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.subdivide(number_cuts=cuts, smoothness=0.0)
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.mode_set(mode='OBJECT')
    print("ARIEL body: %d -> %d vertices (limb / torso faces subdivided x%d)" % (n, len(body.data.vertices), cuts), flush=True)


def neutral_torso(body, iters=120):
    """Anatomical-model look: Laplacian-smooth (then Taubin-relax) the surface detail on the front of the chest (no nipples), keeping the
    overall form.  Only chest-front vertices of the spine / shoulder bones move."""
    names = {vg.index: vg.name.lower() for vg in body.vertex_groups}
    me = body.data
    n = len(me.vertices)
    co = np.zeros(n * 3)
    me.vertices.foreach_get('co', co)
    co = co.reshape(-1, 3)
    chest = np.zeros(n, bool)
    for v in me.vertices:
        g = max(v.groups, key=lambda x: x.weight, default=None)
        nm = names.get(g.group, '') if g is not None else ''
        chest[v.index] = nm.startswith('spine') or 'shoulder' in nm
    zc = co[chest, 2]
    if not chest.any():
        return 0
    lo, hi = np.percentile(zc, 35), np.percentile(zc, 97)
    ymid = float(np.median(co[chest, 1]))
    m = chest & (co[:, 2] > lo) & (co[:, 2] < hi) & (co[:, 1] < ymid)
    ev = np.zeros(len(me.edges) * 2, np.int64)
    me.edges.foreach_get('vertices', ev)
    e = ev.reshape(-1, 2)
    deg = np.bincount(e.ravel(), minlength=n).astype(float)
    # soft falloff so the smoothed patch blends into the rest
    w = m.astype(float)
    for _ in range(6):
        acc = np.bincount(e[:, 0], weights=w[e[:, 1]], minlength=n) + np.bincount(e[:, 1], weights=w[e[:, 0]], minlength=n)
        w = np.maximum(w, 0.85 * acc / np.maximum(deg, 1))
    x = co.copy()
    for it in range(iters):
        for lam in ((0.6,) if it < iters * 2 // 3 else (0.5, -0.53)):
            acc = np.zeros((n, 3))
            for j in range(3):
                acc[:, j] = np.bincount(e[:, 0], weights=x[e[:, 1], j], minlength=n) + np.bincount(e[:, 1], weights=x[e[:, 0], j], minlength=n)
            avg = acc / np.maximum(deg, 1)[:, None]
            x += (lam * w)[:, None] * (avg - x)
    me.vertices.foreach_set('co', x.ravel())
    me.update()
    print("ARIEL body: neutral chest, %d vertices smoothed" % int((w > 0.05).sum()), flush=True)
    return int(m.sum())


def build():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    import arc_flow
    try:
        arc_flow.register()
    except Exception:
        pass
    sc = bpy.context.scene
    glb = os.path.join(CA, 'Meshy_AI_Autumn_biped_Character_output.glb')
    bpy.ops.import_scene.gltf(filepath=glb)
    arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
    body = bpy.data.objects[BODY]
    scalp = bpy.data.objects[SCALP]
    # Ariel's body mesh only exists where her clothes leave skin visible (no hips / legs / feet under the pants and shoes).
    # For a complete, neutral anatomical figure the fitted shirt, trousers and shoes are joined into the skin as body
    # surface and everything is shaded as one matte clay material; belt, charms, LEDs and other accessories are removed.
    keep = {SHIRT, PANTS, SHOES}
    for o in list(bpy.data.objects):
        if o.type == 'MESH' and o not in (body, scalp) and o.name not in keep:
            bpy.data.objects.remove(o, do_unlink=True)
    parts = [bpy.data.objects[k] for k in keep if k in bpy.data.objects]
    if parts:
        bpy.context.view_layer.update()
        for o in list(sc.objects):
            if o is not None:
                o.select_set(o in parts or o is body)
        bpy.context.view_layer.objects.active = body
        bpy.ops.object.join()
        body = bpy.context.view_layer.objects.active
        body.name = BODY
    clay = clay_material()
    body.data.materials.clear()
    body.data.materials.append(clay)
    sm = bpy.data.materials.new("TP Scalp")
    sm.use_nodes = True
    sm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.03, 0.025, 0.025, 1.0)
    scalp.data.materials.clear()
    scalp.data.materials.append(sm)
    for o in (body, scalp):
        for p in o.data.polygons:
            p.use_smooth = True
    # animation clip from the merged-animations GLB (same Meshy skeleton, same bone names)
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=os.path.join(CA, 'Meshy_AI_Autumn_biped_Meshy_AI_Meshy_Merged_Animations.glb'))
    act = bpy.data.actions.get(CLIP)
    for o in set(bpy.data.objects) - before:
        bpy.data.objects.remove(o, do_unlink=True)
    if act is not None:
        arm.animation_data_create()
        arm.animation_data.action = act
        if hasattr(arm.animation_data, 'action_slot') and len(getattr(act, 'slots', [])):
            arm.animation_data.action_slot = act.slots[0]
        sc.frame_start, sc.frame_end = 1, max(48, int(act.frame_range[1]))
    sc.render.fps = 24
    # hair: DHS1 -> Empty Hair Curves on the scalp (Head-bone rest frame -> Blender world)
    meta, pos, group = read_dhs1(os.path.join(CA, 'ariel_hair_strands.bin'))
    Mh = Y_UP_TO_Z_UP @ node_world_gltf(read_glb_json(glb), meta.get('headBone', 'Head'))
    P = pos.reshape(-1, 3)
    Pw = (np.array(Mh)[:3, :3] @ P.T).T + np.array(Mh)[:3, 3]
    N, K = pos.shape[0], pos.shape[1]
    cd = bpy.data.hair_curves.new("Ariel Hair") if hasattr(bpy.data, 'hair_curves') else bpy.data.curves.new("Ariel Hair", 'CURVES')
    cd.add_curves([K] * N)
    cd.position_data.foreach_set('vector', Pw.astype(np.float32).ravel())
    rad = cd.attributes.get('radius') or cd.attributes.new('radius', 'FLOAT', 'POINT')
    r = np.repeat(np.linspace(0.0006, 0.0002, K)[None], N, 0).ravel()
    rad.data.foreach_set('value', r.astype(np.float32))
    hair = bpy.data.objects.new("Ariel Hair", cd)
    sc.collection.objects.link(hair)
    try:
        cd.surface = scalp
    except Exception:
        pass
    hm = bpy.data.materials.new("TP Ariel Hair")
    hm.use_nodes = True
    hm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (*meta.get('color', (0.005, 0.005, 0.006)), 1.0)
    cd.materials.append(hm)
    # Ariel's body was modelled to be worn under clothes: hands and head are dense (~2 mm edges), the limbs and torso coarse
    # (~13 mm).  Subdivide the limb / torso faces (weights interpolate) so the muscle solve has skin to shape there.
    densify_limbs(body, arm, cuts=int(arg('--cuts', '2')))
    neutral_torso(body)
    # muscles
    from arc_flow import muscle_ops as MO
    ms = sc.arc_flow_muscle
    ms.skin, ms.armature = body, arm
    ms.preset, ms.detail = 'HUMAN_FEMALE', DETAIL
    MO.apply_preset(sc)
    n = MO.run_auto_seed(sc, True)
    MO.run_solve(sc)
    MO.bake_shape_keys(sc)
    MO.write_colors(body, ms)
    print("ARIEL muscles: seeded %d verts, %d instances, %s" % (n, len(MO.get_instances(sc)), ms.status), flush=True)
    # hair solver + colliders (scalp, body incl. hands)
    hp = sc.arc_flow_hair
    hp.hair, hp.surface, hp.armature, hp.bone = hair, scalp, arm, meta.get('headBone', 'Head')
    from arc_flow import hair_ops as HO
    for o in (scalp, body):
        HO.add_collider_entry(sc, o)
    for c in hp.colliders:
        c.deforming = True
    print("ARIEL hair: %d strands x %d points, %d colliders" % (N, K, len(hp.colliders)), flush=True)
    os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=OUT, compress=True)
    print("SAVED", OUT, flush=True)
    return dict(body=body, scalp=scalp, hair=hair, arm=arm, Pw=Pw)


if __name__ == '__main__':
    d = build()
    # sanity: hair roots sit on the scalp
    import mathutils.bvhtree as BT
    dg = bpy.context.evaluated_depsgraph_get()
    tree = BT.BVHTree.FromObject(d['scalp'], dg)
    roots = d['Pw'].reshape(-1, 14, 3)[:, 0]
    dist = np.array([(tree.find_nearest(Vector(p))[3] or 9) for p in roots[::25]])
    print("ARIEL hair roots to scalp: median %.4f m, p95 %.4f m" % (np.median(dist), np.percentile(dist, 95)), flush=True)
