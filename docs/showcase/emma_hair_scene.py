"""Showcase 1 — Emma + Tessera Projection Hair Curves (real-time solver, Cycles viewport).

    build(glb) -> dict(hair=..., hand=..., step=fn(frame))

Emma_Character.glb (DART Meadow character, kept out of git under demo_assets/)
gets a Random-Walk (Skin) subsurface Principled skin, her 6,000-strand groom is
loaded from emma_hair_strands.bin (head-local glTF space) into a native Empty
Hair Curves object, and the Tessera Projection Hair solver drives it with the
body as SDF collider plus an animated sculpted glove as a second, moving
collider that parts, compresses and releases the strands.
"""
import json
import math
import os
import struct

import bpy
import numpy as np
from mathutils import Euler, Matrix, Quaternion, Vector

from common import (ROOT, addon, sub, clear_scene, borrow, mesh_world, area_light, spot_light,
                    look_camera, cycles_setup, principled)

FPS = 24
FRAMES = 30 * FPS
HEAD_C = Vector((0.0, -0.01, 1.53))


# ----------------------------------------------------------------------------- glTF helpers
def _gltf_json(path):
    b = open(path, 'rb').read()
    jl = struct.unpack_from('<I', b, 12)[0]
    return json.loads(b[20:20 + jl])


def _node_global(g, name):
    nodes = g['nodes']
    parent = {}
    for i, n in enumerate(nodes):
        for c in n.get('children', []):
            parent[c] = i

    def local(n):
        if 'matrix' in n:
            return np.array(n['matrix'], float).reshape(4, 4).T
        t = n.get('translation', [0, 0, 0])
        r = n.get('rotation', [0, 0, 0, 1])
        s = n.get('scale', [1, 1, 1])
        q = Quaternion((r[3], r[0], r[1], r[2]))
        M = Matrix.Translation(t) @ q.to_matrix().to_4x4() @ Matrix.Diagonal((*s, 1.0))
        return np.array(M, float)
    idx = next(i for i, n in enumerate(nodes) if n.get('name') == name)
    M = np.eye(4)
    i = idx
    while i is not None:
        M = local(nodes[i]) @ M
        i = parent.get(i)
    return M


def load_strands(bin_path, glb_path):
    b = open(bin_path, 'rb').read()
    assert b[:4] == b'DHS1'
    jl = struct.unpack_from('<I', b, 4)[0]
    meta = json.loads(b[12:12 + jl])
    N, K = meta['count'], meta['points']
    rest = np.frombuffer(b, np.float32, N * K * 3, 12 + jl).reshape(N * K, 3).astype(np.float64)
    Mh = _node_global(_gltf_json(glb_path), meta.get('headBone', 'Head'))
    w = rest @ Mh[:3, :3].T + Mh[:3, 3]
    blender = np.stack([w[:, 0], -w[:, 2], w[:, 1]], axis=1)  # glTF y-up -> Blender z-up
    off = np.arange(0, N * K + 1, K, dtype=np.int64)
    return blender, off, meta


# ----------------------------------------------------------------------------- materials
def skin_sss(mat):
    """Lifelike Principled skin: Random Walk (Skin) SSS, warm red-dominant radius, soft spec."""
    p = next(n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    try:
        p.subsurface_method = 'RANDOM_WALK_SKIN'
    except Exception:
        p.subsurface_method = 'RANDOM_WALK'
    def put(key, val):
        if key in p.inputs:
            p.inputs[key].default_value = val
    put('Subsurface Weight', 1.0)
    put('Subsurface Radius', (1.0, 0.38, 0.21))
    put('Subsurface Scale', 0.010)
    put('Subsurface IOR', 1.4)
    put('Subsurface Anisotropy', 0.0)
    put('Roughness', 0.44)
    put('IOR', 1.4)
    put('Specular IOR Level', 0.42)
    put('Coat Weight', 0.06)
    put('Coat Roughness', 0.32)
    put('Sheen Weight', 0.04)
    if not p.inputs['Base Color'].is_linked:
        put('Base Color', (0.72, 0.50, 0.40, 1.0))
    return mat


def hair_material(meta):
    m = bpy.data.materials.new("TS_EmmaHair")
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    hb = nt.nodes.new('ShaderNodeBsdfHairPrincipled')
    try:
        hb.model = 'CHIANG'
        hb.parametrization = 'MELANIN'
    except Exception:
        pass
    def put(key, val):
        if key in hb.inputs:
            hb.inputs[key].default_value = val
    put('Melanin', 0.86)
    put('Melanin Redness', 0.45)
    put('Roughness', 0.30)
    put('Radial Roughness', 0.38)
    put('Coat', 0.12)
    put('Random Roughness', 0.22)
    put('Random Color', 0.12)
    # root-to-tip melanin gradient (darker roots, sun-warmed tips like the in-game groom)
    info = nt.nodes.new('ShaderNodeHairInfo')
    ramp = nt.nodes.new('ShaderNodeMapRange')
    ramp.inputs['To Min'].default_value = 0.92
    ramp.inputs['To Max'].default_value = 0.74
    nt.links.new(info.outputs['Intercept'], ramp.inputs['Value'])
    if 'Melanin' in hb.inputs:
        nt.links.new(ramp.outputs['Result'], hb.inputs['Melanin'])
    nt.links.new(hb.outputs[0], out.inputs['Surface'])
    return m


# ----------------------------------------------------------------------------- pose / stage
def pose_arms_down(arm):
    """Relax the T/A-pose: swing each upper arm so shoulder->wrist hangs beside the body."""
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='POSE')
    Mw = arm.matrix_world
    for side, sgn in (('Left', 1.0), ('Right', -1.0)):
        pb = arm.pose.bones.get(side + 'Arm')
        hand = arm.pose.bones.get(side + 'Hand')
        if pb is None or hand is None:
            continue
        head_w = Mw @ pb.head
        d = (Mw @ hand.head) - head_w
        t = Vector((sgn * 0.22, -0.10, -1.0)).normalized()
        q = d.normalized().rotation_difference(t)
        R = Matrix.Translation(head_w) @ q.to_matrix().to_4x4() @ Matrix.Translation(-head_w)
        pb.matrix = Mw.inverted() @ R @ Mw @ pb.matrix
        bpy.context.view_layer.update()
    bpy.ops.object.mode_set(mode='OBJECT')


def studio():
    sc = bpy.context.scene
    bpy.ops.mesh.primitive_plane_add(size=14, location=(0, 1.2, 0))
    floor = bpy.context.object
    floor.name = "TS_Cyclorama"
    import bmesh
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.subdivide(number_cuts=40)
    bpy.ops.object.mode_set(mode='OBJECT')
    for v in floor.data.vertices:
        y = v.co.y + 1.2
        if y > 1.0:
            t = (y - 1.0) / 3.0
            v.co.z = t * t * 4.0
    bpy.ops.object.shade_smooth()
    floor.data.materials.append(principled("TS_Backdrop", (0.11, 0.115, 0.13), rough=0.62))
    w = sc.world
    w.use_nodes = True
    bg = w.node_tree.nodes.get('Background')
    bg.inputs[0].default_value = (0.03, 0.035, 0.045, 1)
    bg.inputs[1].default_value = 0.35
    # key (warm, camera-left, high) / fill (cool) / hair rim / kicker / backdrop wash
    area_light("TS_Key", (1.05, -1.35, 2.25), HEAD_C, 0.9, 120, (1.0, 0.93, 0.85))
    area_light("TS_Fill", (-1.3, -1.0, 1.65), HEAD_C, 1.5, 38, (0.78, 0.85, 1.0))
    area_light("TS_HairRim", (-0.55, 1.05, 2.15), HEAD_C + Vector((0, 0, 0.05)), 0.45, 180, (1.0, 0.9, 0.78))
    area_light("TS_Kicker", (0.9, 0.9, 1.75), HEAD_C, 0.4, 90, (0.9, 0.95, 1.0))
    spot_light("TS_TopSpot", (0.15, -0.25, 2.9), HEAD_C, 260, (1.0, 0.95, 0.88), size_deg=26, blend=0.5, radius=0.12)
    area_light("TS_BackdropWash", (0.0, 1.0, 0.25), (0.0, 3.2, 1.2), 2.5, 160, (0.55, 0.65, 0.9))


def head_collider(zcut=1.22, voxel=0.005):
    """Watertight head + shoulders collision proxy (the game mesh has open eye / mouth holes,
    which flips the SDF sign); invisible to Cycles rays, wire in Solid mode."""
    import bmesh
    deps = bpy.context.evaluated_depsgraph_get()
    bm = bmesh.new()
    for n in ('Emma_Body', 'Emma_Scalp'):
        ob = bpy.data.objects[n]
        me = bpy.data.meshes.new_from_object(ob.evaluated_get(deps))
        me.transform(ob.matrix_world)
        bm.from_mesh(me)
        bpy.data.meshes.remove(me)
    bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], plane_co=(0, 0, zcut),
                           plane_no=(0, 0, 1), clear_inner=True)
    bmesh.ops.holes_fill(bm, edges=bm.edges[:], sides=0)
    me = bpy.data.meshes.new("TS_HeadCollider")
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new("TS_HeadCollider", me)
    bpy.context.scene.collection.objects.link(ob)
    md = ob.modifiers.new("voxel", 'REMESH')
    md.mode = 'VOXEL'
    md.voxel_size = voxel
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.modifier_apply(modifier=md.name)
    for a in ('visible_camera', 'visible_diffuse', 'visible_glossy', 'visible_transmission',
              'visible_volume_scatter', 'visible_shadow'):
        setattr(ob, a, False)
    ob.hide_render = True
    return ob


def make_hand_collider():
    g = borrow(os.path.join(ROOT, 'docs', 'render_hair.py'),
               {'_add_ball', 'make_hand', 'clay_material'})
    hand = g['make_hand']()
    hand.name = "TS_CollisionGlove"
    hand.data.materials.clear()
    hand.data.materials.append(principled("TS_GloveLeather", (0.02, 0.022, 0.026), rough=0.48,
                                          **{'Coat Weight': 0.3, 'Coat Roughness': 0.25}))
    return hand


# ----------------------------------------------------------------------------- hand path (head-relative)
_KEYS = [  # (frame, offset from HEAD_C, euler deg)
    (1,   (0.42, -0.08, 0.06), (10, -8, 150)),
    (60,  (0.22, -0.02, 0.04), (10, -8, 150)),
    (120, (0.105, 0.02, -0.02), (14, -18, 160)),   # press into right side hair
    (180, (0.11, 0.07, -0.13), (24, -30, 170)),    # comb down & back
    (230, (0.30, 0.16, -0.12), (24, -30, 175)),    # release -> spring back
    (300, (0.05, 0.40, 0.02), (90, 0, 270)),       # travel behind
    (360, (0.00, 0.13, 0.04), (95, 0, 270)),       # press into the back of the head
    (420, (-0.02, 0.14, -0.14), (100, 0, 270)),    # sweep down the back
    (470, (-0.10, 0.36, -0.10), (100, 0, 270)),
    (540, (-0.40, -0.04, 0.02), (10, 8, 30)),      # around to her left
    (600, (-0.11, 0.00, -0.02), (14, 18, 20)),     # part the left side
    (650, (-0.13, 0.06, -0.14), (20, 26, 15)),
    (720, (-0.45, 0.02, -0.08), (20, 26, 15)),     # away; hair settles
]


def _smooth(t):
    return t * t * (3 - 2 * t)


def hand_matrix(frame):
    ks = _KEYS
    if frame <= ks[0][0]:
        a = b = ks[0]; t = 0.0
    elif frame >= ks[-1][0]:
        a = b = ks[-1]; t = 0.0
    else:
        for i in range(len(ks) - 1):
            if ks[i][0] <= frame <= ks[i + 1][0]:
                a, b = ks[i], ks[i + 1]
                t = _smooth((frame - a[0]) / float(b[0] - a[0]))
                break
    loc = HEAD_C + Vector(a[1]).lerp(Vector(b[1]), t)
    qa = Euler([math.radians(x) for x in a[2]], 'XYZ').to_quaternion()
    qb = Euler([math.radians(x) for x in b[2]], 'XYZ').to_quaternion()
    return Matrix.Translation(loc) @ qa.slerp(qb, t).to_matrix().to_4x4()


def breeze(frame):
    """Studio-fan breeze on the groom: a slow, gusting draught from camera-right that veers
    a little, so the hair keeps swaying for the whole clip (not only where the glove is)."""
    t = frame / FPS
    ramp = min(1.0, t / 1.5)
    yaw = math.radians(28) * math.sin(2 * math.pi * t / 7.3) + math.radians(8) * math.sin(2 * math.pi * t / 2.1)
    d = Vector((-0.80, 0.55, 0.12))
    d = (Matrix.Rotation(yaw, 3, 'Z') @ d).normalized()
    gust = (1.9 + 0.75 * math.sin(2 * math.pi * t / 3.7) + 0.45 * math.sin(2 * math.pi * t / 1.3 + 0.8)
            + 0.20 * math.sin(2 * math.pi * t / 0.61 + 2.1))
    return d * (ramp * max(0.15, gust))


class HandGuard:
    """Keeps the glove's surface outside the head + shoulders collider by `clear` metres:
    the keyed path is pushed outward from the head centre until every glove sample clears
    the head, so the glove presses and parts the hair layer instead of passing through her."""
    def __init__(self, head_ob, hand_ob, clear=0.014, n=500):
        from mathutils.bvhtree import BVHTree
        me = head_ob.data
        Mw = head_ob.matrix_world
        verts = [Mw @ v.co for v in me.vertices]
        polys = [tuple(p.vertices) for p in me.polygons]
        self.bvh = BVHTree.FromPolygons(verts, polys)
        hv = hand_ob.data.vertices
        step = max(1, len(hv) // n)
        self.local = [hv[i].co.copy() for i in range(0, len(hv), step)]
        self.clear = clear
        self.last_push = 0.0

    def probe(self, M):
        """(signed clearance of the worst glove sample, outward surface normal there)."""
        worst, wn = 1e9, None
        for co in self.local:
            p = M @ co
            loc, nrm, _i, dist = self.bvh.find_nearest(p)
            if loc is None:
                continue
            sd = dist if (p - loc).dot(nrm) >= 0 else -dist
            if sd < worst:
                worst, wn = sd, nrm
        return worst, wn

    def min_dist(self, M):
        return self.probe(M)[0]

    def __call__(self, M):
        """Slide the glove out along the head / shoulder surface normal until it clears."""
        M = M.copy()
        push = 0.0
        for _ in range(32):
            d, n = self.probe(M)
            if d >= self.clear - 1e-4 or n is None:
                break
            dp = (self.clear - d) + 0.001
            push += dp
            M.translation = M.translation + n.normalized() * dp
        self.last_push = push
        return M


# ----------------------------------------------------------------------------- build
def build(glb, strands_bin, n_strands=6000):
    pkg = addon()
    HO = sub('hair_ops')
    HS = sub('core.hair_solver')
    sc = bpy.context.scene
    clear_scene()
    sc.render.fps = FPS
    sc.frame_start, sc.frame_end = 1, FRAMES
    bpy.ops.import_scene.gltf(filepath=glb)
    for o in list(bpy.data.objects):
        if o.name.startswith('Icosphere'):
            bpy.data.objects.remove(o, do_unlink=True)
    arm = bpy.data.objects['Emma_Armature']
    pose_arms_down(arm)
    skin_sss(bpy.data.materials['Emma_Skin'])
    # the scalp cap reads as hair-coloured skin under the groom (like the in-game private material)
    scalp = bpy.data.objects['Emma_Scalp']
    scalp.data.materials.clear()
    scalp.data.materials.append(principled("TS_EmmaScalp", (0.035, 0.018, 0.010), rough=0.7))
    collider = head_collider()

    P, off, meta = load_strands(strands_bin, glb)
    if n_strands < len(off) - 1:
        P = P[:off[n_strands]]
        off = off[:n_strands + 1]
    S = len(off) - 1
    K = int(off[1] - off[0])
    rad = np.empty(len(P))
    t = np.tile(np.linspace(0, 1, K), S)
    rad[:] = (0.0011 * (1 - t) + 0.00035 * t)
    hair = HO.make_curves_object("TS_EmmaHair", P, off, radii=rad, collection=sc.collection, radius=0.0008)
    for md in list(hair.modifiers):
        hair.modifiers.remove(md)
    hair.data.materials.append(hair_material(meta))

    hand = make_hand_collider()
    hand.matrix_world = hand_matrix(1)
    studio()
    cycles_setup(samples=64, preview_samples=12)
    # viewport skin quality: tight adaptive threshold + high-quality OIDN with accurate
    # albedo/normal prefilter, so the Random-Walk (Skin) SSS reads as skin, not a smeared mask
    cy = sc.cycles
    for attr, val in (('preview_adaptive_threshold', 0.03), ('preview_adaptive_min_samples', 0),
                      ('preview_denoising_quality', 'HIGH'), ('preview_denoising_prefilter', 'ACCURATE'),
                      ('preview_denoising_start_sample', 20), ('subsurface_samples', 4)):
        try:
            setattr(cy, attr, val)
        except Exception:
            pass

    hp = sc.arc_flow_hair
    hp.hair = hair
    hp.collider = collider
    hair.arc_flow_hair.is_guide_only = True
    hp.surface = bpy.data.objects['Emma_Scalp']
    hp.dt_mode = 'SCENE'
    hp.dt = 1.0 / FPS
    hp.substeps = 4
    hp.damping = 0.965
    hp.rest_shape = 0.12
    hp.rest_shape_root = 0.92
    hp.rest_shape_tip = 0.02
    hp.pinned_prefix = 2
    hp.collision_radius = 0.004
    hp.collision_iterations = 3
    hp.collision_friction = 0.35
    hp.sdf_resolution = 64
    hp.sdf_refine = False
    hp.sdf_level = 2
    hp.guide_ratio = 0.25
    hp.child_k = 3
    hp.use_env_wind = False
    hp.wind_drag = 1.4
    sc.arc_flow.ui_tab = 'HAIR'
    bpy.context.view_layer.update()

    solver = HO.build_solver(sc)
    # second, moving collider: the sculpted glove (field in glove-local space)
    Vh, Th = mesh_world(hand)
    Mi = np.linalg.inv(np.array(hand.matrix_world, float))
    field_hand = HS.make_collider(Vh @ Mi[:3, :3].T + Mi[:3, 3], Th, resolution=40, level=2, mode='SOLID')
    state = dict(field_hand=field_hand, hp=hp, HO=HO, HS=HS, hand=hand, hair=hair)

    def ensure_hand(solver):
        if len(solver.colliders) < 2:
            solver.add_collider(field_hand, matrix=np.array(hand.matrix_world, float), radius=0.010)

    guard = HandGuard(collider, hand)
    state['guard'] = guard

    def step(frame):
        hand.matrix_world = guard(hand_matrix(frame))
        hp.wind = tuple(breeze(frame))
        sv = HO.RT.solver or HO.build_solver(sc)
        ensure_hand(sv)
        sv.set_collider_matrix(len(sv.colliders) - 1, np.array(hand.matrix_world, float))
        HO.step_scene(sc, dt=1.0 / FPS)
        return HO.RT.solver
    ensure_hand(solver)
    # settle the groom on her shoulders before frame 1
    for _ in range(12):
        step(1)
    state['step'] = step
    return state


def go_live_ui(state):
    """Flip the panel's Live toggle (the real add-on path), then keep stepping deterministic."""
    HO = state['HO']
    hp = state['hp']
    hp.live = True            # ensure_live(): handlers + timer
    HO._clear_handlers()      # the capture driver steps exactly one solver tick per video frame
    HO.RT.live = True


# ----------------------------------------------------------------------------- GUI capture hooks (docs/showcase/capture.py)
DEMO_GLB = os.path.join(ROOT, 'demo_assets', 'Emma_Character.glb')
DEMO_STRANDS = os.environ.get('EMMA_STRANDS', os.path.join(ROOT, 'demo_assets', 'emma_hair_strands.bin'))


def place_view(r3, sp, eye, target, lens, shift=0.0):
    eye, target = Vector(eye), Vector(target)
    d = target - eye
    q = d.to_track_quat('-Z', 'Y')
    r3.view_perspective = 'PERSP'
    r3.view_rotation = q
    right = q @ Vector((1, 0, 0))
    r3.view_location = target + right * shift
    r3.view_distance = d.length
    sp.lens = lens
    sp.clip_start = 0.01


_build = build


def build(glb=None, strands_bin=None, n_strands=6000):
    st = _build(glb or DEMO_GLB, strands_bin or DEMO_STRANDS, n_strands=n_strands)
    st['status'] = lambda: st['hp'].status
    st['properties'] = 'MATERIAL'
    return st


def ui(st, sp, r3, ctx):
    sc = bpy.context.scene
    sc.arc_flow.ui_tab = 'HAIR'
    go_live_ui(st)
    body = bpy.data.objects['Emma_Body']
    bpy.context.view_layer.objects.active = body
    body.select_set(False)
    body.active_material_index = 0
    place_view(r3, sp, HEAD_C + Vector((0.55, -1.02, 0.06)), HEAD_C + Vector((0.0, 0.0, -0.035)), 72, shift=0.07)
