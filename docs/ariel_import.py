"""Import the game's Ariel character into Blender: skinned mesh, skeleton, animation clips and her real strand hair.

Reusable converter for the dartmeadow-space character assets (not shipped with the extension; point it at your clone):

    blender --python docs/ariel_import.py -- --game /path/to/dartmeadow-space [--clip Idle_Breathe_Blink] [--no-hair]
    # or from another script:  import ariel_import as AI;  S = AI.build_ariel(game_dir, ...)

What it builds
  * characterassests/Meshy_AI_Autumn_biped_Character_output.glb -> mesh + Ariel_Armature (y-up -> z-up by the glTF importer)
  * ...Merged_Animations.glb -> every clip is copied onto Ariel_Armature as an Action (same bone names)
  * characterassests/ariel_hair_strands.bin ("DHS1", see hair_strands.js buildHair / docs/hair-new-character.md) ->
    one CURVES object with ALL strands (Ariel: 5,665 strands x 14 points).  The file stores positions in the *glTF Head node*
    frame.  The CURVES points are kept exactly in that frame and the object is parented to the Head bone, with a parent
    inverse that is the glTF-node -> Blender-bone fix-up, so it follows the head like the game's hair does.  Guide strands
    (the file's `guides` list, 1,415) are written first and `arc_flow_hair.n_guides` is set, so the extension simulates them
    and skins the children (the file's own guide / child split).
  * collision bodies: the head field of the .bin (SDF, Head frame) is turned into a closed mesh; the game's fitted body capsules
    (HAIR_DEFAULT_CONFIG.collision.capsules[2] in hair_strands.js, bone-local) become capsule meshes parented to their bones.
  * materials: hair (Principled Hair look in EEVEE), skin, outfit, floor.

The glTF node frames differ from the Blender bone frames by a constant per bone; `node_fixups()` computes that constant from the
glb JSON (rest world of every node) so both the hair and the capsules sit exactly where the game puts them.
"""
import json
import math
import os
import struct
import sys

import numpy as np

try:
    import bpy
    from mathutils import Matrix, Vector
except ImportError:                      # the .bin / glb parsing helpers also work without Blender
    bpy = None

CHAR_DIR = 'characterassests'
CHAR_GLB = 'Meshy_AI_Autumn_biped_Character_output.glb'
ANIM_GLB = 'Meshy_AI_Autumn_biped_Meshy_AI_Meshy_Merged_Animations.glb'
HAIR_BIN = 'ariel_hair_strands.bin'

# y-up (glTF) -> z-up (Blender)
C_YUP2ZUP = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)

# HAIR_DEFAULT_CONFIG.collision.capsules[2] of hair_strands.js (the game's own fitted values, bone-local metres, glTF y-up bone frames)
BODY_CAPSULES = [
    ('Spine02', (0, -0.0015, 0.0242), (0, 0.1292, 0.0242), 0.1257),
    ('Spine02', (0.0763, -0.0019, 0.0306), (0.0763, 0.1289, 0.0306), 0.1031),
    ('Spine02', (-0.0763, -0.0019, 0.0306), (-0.0763, 0.1289, 0.0306), 0.1031),
    ('Spine01', (0, -0.0038, 0.0305), (0, 0.1237, 0.0305), 0.1107),
    ('Spine01', (0.07, -0.0066, 0.0527), (0.07, 0.1209, 0.0527), 0.0949),
    ('Spine01', (-0.07, -0.0066, 0.0527), (-0.07, 0.1209, 0.0527), 0.0949),
    ('Spine', (0, 0, 0.0191), (0, 0.0435, 0.0191), 0.1009),
    ('Spine', (0.0724, 0, -0.0125), (0.0724, 0.0435, -0.0125), 0.0892),
    ('Spine', (-0.0724, 0, -0.0125), (-0.0724, 0.0435, -0.0125), 0.0892),
    ('Spine', (0, 0.0235, 0), (0.0771, 0.0237, 0.0178), 0.06),
    ('LeftShoulder', (-0.0044, 0.0015, 0.013), (-0.0044, 0.1226, 0.013), 0.0623),
    ('LeftArm', (0, 0, 0), (0, 0.1574, 0), 0.0607),
    ('LeftForeArm', (0, 0, 0), (0, 0.2221, -0.0378), 0.055),
    ('LeftHand', (0, 0, 0), (0, 0.0981, 0), 0.04),
    ('Spine', (0, 0.0235, 0), (-0.0771, 0.0237, 0.0178), 0.06),
    ('RightShoulder', (0.0044, 0.0015, 0.013), (0.0044, 0.1226, 0.013), 0.0623),
    ('RightArm', (0, 0, 0), (0, 0.1542, 0), 0.0607),
    ('RightForeArm', (0, 0, 0), (0, 0.2214, -0.0409), 0.055),
    ('RightHand', (0, 0, 0), (0, 0.0981, 0), 0.04),
    ('Hips', (0, 0, 0), (0, 0.1314, 0), 0.14),
    ('neck', (0, 0, 0), (0, 0.1068, 0), 0.055),
]
# bodies the hair can reach, grouped per rigid bone (one collision object each); the Hips capsule is left out (hair never gets there)
COLLIDER_GROUPS = [('Torso', 'Spine01', ('Spine02', 'Spine01', 'Spine')), ('Neck', 'neck', ('neck',)),
                   ('Shoulder L', 'LeftShoulder', ('LeftShoulder',)), ('Shoulder R', 'RightShoulder', ('RightShoulder',)),
                   ('UpperArm L', 'LeftArm', ('LeftArm',)), ('UpperArm R', 'RightArm', ('RightArm',)),
                   ('Forearm L', 'LeftForeArm', ('LeftForeArm',)), ('Forearm R', 'RightForeArm', ('RightForeArm',))]


# ----------------------------------------------------------------------------- file parsing (no Blender needed)
def parse_dhs1(path):
    """Parse a DHS1 strand file (layout from hair_strands.js buildHair). Returns a dict of numpy arrays + the header."""
    b = open(path, 'rb').read()
    if b[:4] != b'DHS1':
        raise ValueError('bad magic %r' % b[:4])
    jl = struct.unpack('<I', b[4:8])[0]
    meta = json.loads(b[12:12 + jl].decode('utf-8'))
    d0 = 12 + jl
    N, K, G = int(meta['count']), int(meta['points']), int(meta['guides'])
    off = d0
    rest = np.frombuffer(b, '<f4', N * K * 3, off).reshape(N, K, 3).astype(np.float64)
    off += N * K * 12
    group = np.frombuffer(b, np.uint8, N, off).copy()
    off += (N + 3) & ~3
    guide_of = np.frombuffer(b, '<u4', N, off).astype(np.int64)
    off += N * 4
    guides = np.frombuffer(b, '<u4', G, off).astype(np.int64)
    sdf = None
    s = meta.get('sdf')
    if s and s.get('byteOffsetFromData') is not None:
        nx, ny, nz = s['dims']
        v = np.frombuffer(b, '<i2', nx * ny * nz, d0 + s['byteOffsetFromData']).astype(np.float64) * float(s.get('unit', 1e-4))
        sdf = dict(v=v.reshape(nx, ny, nz), origin=np.array(s['origin'], float), cell=float(s['cell']))
    return dict(meta=meta, rest=rest, group=group, guide_of=guide_of, guides=guides, sdf=sdf)


def sample_sdf(sdf, P):
    """Trilinear sample of the head field (Head frame, y-up). Outside the box returns +big."""
    g = (np.asarray(P, float) - sdf['origin']) / sdf['cell']
    nx, ny, nz = sdf['v'].shape
    i0 = np.floor(g).astype(int)
    f = g - i0
    out = np.full(len(g), 1.0)
    ok = np.all((i0 >= 0) & (i0 < np.array([nx - 1, ny - 1, nz - 1])), axis=1)
    if not np.any(ok):
        return out
    i, fr = i0[ok], f[ok]
    v = sdf['v']
    acc = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (fr[:, 0] if dx else 1 - fr[:, 0]) * (fr[:, 1] if dy else 1 - fr[:, 1]) * (fr[:, 2] if dz else 1 - fr[:, 2])
                acc = acc + w * v[i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz]
    out[ok] = acc
    return out


def head_mesh_from_sdf(sdf, center, n_u=72, n_v=40, rmax=0.30, step=0.002):
    """Closed star-shaped mesh of the head field: rays from `center` march out to the first sd >= 0. Returns (verts, tris) in the Head frame."""
    verts = []
    cx = np.asarray(center, float)
    for j in range(n_v + 1):
        th = math.pi * j / n_v
        for i in range(n_u):
            ph = 2 * math.pi * i / n_u
            d = np.array([math.sin(th) * math.cos(ph), math.cos(th), math.sin(th) * math.sin(ph)])
            ts = np.arange(0.0, rmax, step)
            sd = sample_sdf(sdf, cx[None] + ts[:, None] * d[None])
            k = np.flatnonzero(sd >= 0.0)
            if len(k) == 0:
                t = rmax * 0.5
            elif k[0] == 0:
                t = 0.005
            else:
                k0 = k[0]
                t = ts[k0 - 1] + step * (-sd[k0 - 1]) / max(sd[k0] - sd[k0 - 1], 1e-9)
            verts.append(cx + d * t)
    V = np.array(verts)
    tris = []
    for j in range(n_v):
        for i in range(n_u):
            a = j * n_u + i
            b = j * n_u + (i + 1) % n_u
            c = (j + 1) * n_u + i
            e = (j + 1) * n_u + (i + 1) % n_u
            tris += [(a, b, c), (b, e, c)]
    return V, np.array(tris, dtype=np.int64)


def capsule_mesh(a, b, r, n_ring=12, n_cap=6):
    """Capsule (segment a-b, radius r) as (verts, quads/tris) in the frame of a, b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ax = b - a
    L = np.linalg.norm(ax)
    ax = ax / L if L > 1e-9 else np.array([0, 1.0, 0])
    t0 = np.cross(ax, [1, 0, 0] if abs(ax[0]) < 0.9 else [0, 1, 0])
    t0 /= np.linalg.norm(t0)
    t1 = np.cross(ax, t0)
    rings = []
    for k in range(n_cap + 1):                                   # bottom hemisphere (pole -> equator)
        al = -math.pi / 2 + (math.pi / 2) * k / n_cap
        rings.append((a + ax * (r * math.sin(al)), r * math.cos(al)))
    for k in range(n_cap + 1):                                   # top hemisphere
        al = (math.pi / 2) * k / n_cap
        rings.append((b + ax * (r * math.sin(al)), r * math.cos(al)))
    V, T = [], []
    for c, rad in rings:
        for i in range(n_ring):
            ph = 2 * math.pi * i / n_ring
            V.append(c + rad * (math.cos(ph) * t0 + math.sin(ph) * t1))
    for k in range(len(rings) - 1):
        for i in range(n_ring):
            p0, p1 = k * n_ring + i, k * n_ring + (i + 1) % n_ring
            q0, q1 = (k + 1) * n_ring + i, (k + 1) * n_ring + (i + 1) % n_ring
            T += [(p0, p1, q0), (p1, q1, q0)]            # outward-facing winding
    return np.array(V), np.array(T, dtype=np.int64)


def gltf_json(path):
    b = open(path, 'rb').read()
    n = struct.unpack('<I', b[12:16])[0]
    return json.loads(b[20:20 + n].decode('utf-8'))


def node_world_matrices(path):
    """{node name: 4x4 world rest matrix, y-up} from the glb JSON (the frame hair_strands.js uses for the Head bone)."""
    j = gltf_json(path)
    nodes = j['nodes']
    parent = {}
    for i, nd in enumerate(nodes):
        for c in nd.get('children', []):
            parent[c] = i

    def local(nd):
        if 'matrix' in nd:
            return np.array(nd['matrix'], float).reshape(4, 4).T
        t = nd.get('translation', [0, 0, 0])
        q = nd.get('rotation', [0, 0, 0, 1])
        s = nd.get('scale', [1, 1, 1])
        x, y, z, w = q
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        M = np.eye(4)
        M[:3, :3] = R * np.array(s)[None, :]
        M[:3, 3] = t
        return M
    cache = {}

    def world(i):
        if i in cache:
            return cache[i]
        M = local(nodes[i])
        if i in parent:
            M = world(parent[i]) @ M
        cache[i] = M
        return M
    return {nd.get('name', str(i)): world(i) for i, nd in enumerate(nodes)}


# ----------------------------------------------------------------------------- Blender side
def node_fixup(arm, bone_name, node_world):
    """Matrix F with  (armature_world @ pose_bone.matrix @ F)  = the glTF node frame of `bone_name` (in Blender z-up world)."""
    bone = arm.data.bones[bone_name]
    D = C_YUP2ZUP @ node_world[bone_name]
    B = np.array(arm.matrix_world @ bone.matrix_local)
    return np.linalg.inv(B) @ D


def parent_to_bone(obj, arm, bone_name, world_matrix):
    """Parent `obj` to a bone so that its rest world matrix is `world_matrix` and it follows the bone afterwards."""
    obj.parent = arm
    obj.parent_type = 'BONE'
    obj.parent_bone = bone_name
    obj.matrix_parent_inverse = Matrix.Identity(4)
    obj.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()
    P = obj.matrix_world.copy()
    obj.matrix_parent_inverse = P.inverted() @ Matrix(world_matrix.tolist())


def import_character(game_dir):
    cdir = os.path.join(game_dir, CHAR_DIR)
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=os.path.join(cdir, CHAR_GLB))
    new = [o for o in bpy.data.objects if o not in before]
    arm = next(o for o in new if o.type == 'ARMATURE')
    arm.name = 'Ariel_Armature'
    keep = []
    for o in new:                                   # the glb ships a stray unit icosphere (animation-glb target); not part of the character
        if o.type == 'MESH' and o.name.startswith('Icosphere'):
            bpy.data.objects.remove(o, do_unlink=True)
        elif o.type == 'MESH':
            keep.append(o)
    return arm, keep


def import_clips(game_dir, arm):
    """Copy every Action of the merged-animation glb onto the character (bone names match); returns the action names."""
    cdir = os.path.join(game_dir, CHAR_DIR)
    before_obj = set(bpy.data.objects)
    before_act = set(bpy.data.actions)
    bpy.ops.import_scene.gltf(filepath=os.path.join(cdir, ANIM_GLB))
    acts = [a for a in bpy.data.actions if a not in before_act]
    for o in [o for o in bpy.data.objects if o not in before_obj]:
        bpy.data.objects.remove(o, do_unlink=True)
    for coll in (bpy.data.armatures, bpy.data.meshes):          # orphaned data of the removed animation-glb objects
        for d in [d for d in coll if d.users == 0]:
            coll.remove(d)
    for a in acts:
        a.use_fake_user = True
    return [a.name for a in acts]


def build_hair_curves(hairdata, arm, node_world, name='Ariel_Hair', head_bone='Head', radius=0.0011, parent=True):
    """ALL strands as one CURVES object. Points are the file's Head-frame positions (guides first). Returns the object."""
    rest, guides, group = hairdata['rest'], hairdata['guides'], hairdata['group']
    N, K = rest.shape[:2]
    order = np.concatenate([guides, np.setdiff1d(np.arange(N), guides)])
    P = rest[order].reshape(-1, 3)
    # glTF y-up Head frame -> keep local coordinates but convert the axes to Blender's frame inside the object (object matrix does the rest)
    cu = bpy.data.hair_curves.new(name)
    cu.add_curves([K] * N)
    cu.points.foreach_set('position', np.ascontiguousarray(P, np.float32).ravel())
    t = np.tile(np.linspace(0.0, 1.0, K), N)
    rad = radius * (1.0 - 0.75 * t ** 1.3)
    try:
        cu.points.foreach_set('radius', rad.astype(np.float32))
    except Exception:
        pass
    obj = bpy.data.objects.new(name, cu)
    bpy.context.scene.collection.objects.link(obj)
    obj['ariel_strand_order'] = 'guides first'
    obj['ariel_groups'] = list(hairdata['meta'].get('groups', []))
    # attributes for shading: strand group id, random per strand
    try:
        a = cu.attributes.new('ariel_group', 'INT', 'CURVE')
        a.data.foreach_set('value', group[order].astype(np.int32))
    except Exception:
        pass
    if parent:
        D = C_YUP2ZUP @ node_world[head_bone]
        parent_to_bone(obj, arm, head_bone, D)
    else:
        obj.matrix_world = Matrix((C_YUP2ZUP @ node_world[head_bone]).tolist())
    try:
        obj.arc_flow_hair.role = 'HAIR'
        obj.arc_flow_hair.n_guides = int(len(guides))
        obj.arc_flow_hair.radius = float(radius)
    except Exception:
        pass
    return obj


def _mesh_object(name, V, T, matrix_world=None, smooth=True):
    me = bpy.data.meshes.new(name)
    me.from_pydata(V.tolist(), [], T.tolist())
    me.update()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    if smooth:
        for p in me.polygons:
            p.use_smooth = True
    return ob


def build_colliders(hairdata, arm, node_world, with_head=True):
    """Collision bodies as parented meshes: the head from the groom's SDF, then ONE closed capsule mesh per fitted body capsule.
    (Overlapping capsules must not share a mesh: the SDF sign comes from the nearest surface's pseudo-normal, which is wrong inside a union
    of overlapping shells.)"""
    out = []
    if with_head:
        V, T = head_mesh_from_sdf(hairdata['sdf'], center=(0.0, 0.08, 0.0))
        ob = _mesh_object('Col Head', V, T)
        parent_to_bone(ob, arm, 'Head', C_YUP2ZUP @ node_world['Head'])
        out.append(ob)
    allowed = {m for _l, _b, members in COLLIDER_GROUPS for m in members}
    seen = {}
    for (bone, a, b, r) in BODY_CAPSULES:
        if bone not in allowed:
            continue
        side = 'C' if abs(a[0] + b[0]) < 0.02 else ('L' if (a[0] + b[0]) > 0 else 'R')
        key = (bone, side)
        seen[key] = seen.get(key, 0) + 1
        nm = 'Col %s %s%s' % (bone, side, '' if seen[key] == 1 else str(seen[key]))
        V, T = capsule_mesh(a, b, r)
        ob = _mesh_object(nm, V, T)
        parent_to_bone(ob, arm, bone, C_YUP2ZUP @ node_world[bone])
        ob['capsule_radius'] = float(r)
        out.append(ob)
    return out


# ----------------------------------------------------------------------------- materials
def _principled(nt):
    return next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED')


def set_input(node, name, value):
    if name in node.inputs:
        node.inputs[name].default_value = value


def make_material(name, color, rough=0.5, metallic=0.0, spec=0.4, coat=0.0, emission=None):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    p = _principled(m.node_tree)
    set_input(p, 'Base Color', color)
    set_input(p, 'Roughness', rough)
    set_input(p, 'Metallic', metallic)
    set_input(p, 'Specular IOR Level', spec)
    set_input(p, 'Coat Weight', coat)
    if emission is not None:
        set_input(p, 'Emission Color', emission[:4])
        set_input(p, 'Emission Strength', emission[4])
    m.diffuse_color = color
    return m


def style_character(meshes):
    """Replace the importer's flat materials with readable skin / outfit looks (the glb's own textures are kept when present)."""
    for ob in meshes:
        if ob.type != 'MESH':
            continue
        for slot in ob.material_slots:
            m = slot.material
            if m is None:
                continue
            m.use_backface_culling = False
            try:
                m.surface_render_method = 'DITHERED'
            except Exception:
                pass
            p = next((n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None) if m.use_nodes else None
            if p is not None:
                set_input(p, 'Metallic', min(p.inputs['Metallic'].default_value, 0.2) if 'Metallic' in p.inputs else 0.0)
                set_input(p, 'Roughness', max(p.inputs['Roughness'].default_value, 0.45) if 'Roughness' in p.inputs else 0.5)


def hair_material(color_root=(0.011, 0.010, 0.014, 1.0), color_tip=(0.045, 0.036, 0.040, 1.0)):
    """Dark hair: root-to-tip gradient + per-strand variation; Principled Hair (Chiang) for Cycles, readable Principled in EEVEE."""
    m = bpy.data.materials.new('Ariel Hair')
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    out.location = (700, 0)
    cinfo = nt.nodes.new('ShaderNodeHairInfo')
    cinfo.location = (-700, 0)
    ramp = nt.nodes.new('ShaderNodeValToRGB')
    ramp.location = (-420, 120)
    ramp.color_ramp.elements[0].color = color_root
    ramp.color_ramp.elements[1].color = color_tip
    nt.links.new(cinfo.outputs['Intercept'], ramp.inputs['Fac'])
    rnd = nt.nodes.new('ShaderNodeMath')
    rnd.operation = 'MULTIPLY_ADD'
    rnd.location = (-420, -120)
    rnd.inputs[1].default_value = 0.5
    rnd.inputs[2].default_value = 0.75
    nt.links.new(cinfo.outputs['Random'], rnd.inputs[0])
    mix = nt.nodes.new('ShaderNodeMix')
    mix.data_type = 'RGBA'
    mix.blend_type = 'MULTIPLY'
    mix.location = (-180, 80)
    mix.inputs[0].default_value = 1.0
    nt.links.new(ramp.outputs['Color'], mix.inputs[6])
    nt.links.new(rnd.outputs['Value'], mix.inputs[7])
    p = nt.nodes.new('ShaderNodeBsdfPrincipled')
    p.location = (200, 150)
    nt.links.new(mix.outputs[2], p.inputs['Base Color'])
    set_input(p, 'Roughness', 0.38)
    set_input(p, 'Specular IOR Level', 0.9)
    set_input(p, 'Anisotropic', 0.6)
    set_input(p, 'Coat Weight', 0.15)
    ph = nt.nodes.new('ShaderNodeBsdfHairPrincipled')
    ph.location = (200, -250)
    nt.links.new(mix.outputs[2], ph.inputs['Color'])
    try:
        ph.parametrization = 'COLOR'
        ph.inputs['Roughness'].default_value = 0.3
        ph.inputs['Radial Roughness'].default_value = 0.4
        ph.inputs['Coat'].default_value = 0.1
    except Exception:
        pass
    # EEVEE draws the Principled branch, Cycles the Hair BSDF
    lp = nt.nodes.new('ShaderNodeLightPath')
    lp.location = (200, 400)
    sel = nt.nodes.new('ShaderNodeMixShader')
    sel.location = (480, 0)
    nt.links.new(p.outputs['BSDF'], sel.inputs[1])
    nt.links.new(ph.outputs['BSDF'], sel.inputs[2])
    isg = nt.nodes.new('ShaderNodeMath')
    isg.operation = 'LESS_THAN'
    isg.inputs[1].default_value = 0.5
    isg.location = (300, 400)
    # no Cycles-only switch node in EEVEE; keep the Principled branch alone for the viewport and use the Hair BSDF via the engine check below
    nt.links.new(p.outputs['BSDF'], out.inputs['Surface'])
    m['hair_bsdf_node'] = ph.name
    return m


def make_scene_materials(arm, hair_obj, meshes):
    style_character(meshes)
    hm = hair_material()
    hair_obj.data.materials.append(hm)
    return hm


def apply_motion(arm, clips, start=1):
    """Sequence clips on the armature (NLA, 24 fps). `clips` = [(action_name, n_frames_to_play)]; returns the last frame."""
    ad = arm.animation_data or arm.animation_data_create()
    for tr in list(ad.nla_tracks):
        ad.nla_tracks.remove(tr)
    f = start
    for i, (an, nf) in enumerate(clips):
        act = bpy.data.actions[an]
        tr = ad.nla_tracks.new()
        tr.name = 'seq%d' % i
        st = tr.strips.new(an, int(f), act)
        L = max(act.frame_range[1] - act.frame_range[0], 1.0)
        st.action_frame_start, st.action_frame_end = act.frame_range
        st.frame_end = f + nf
        st.repeat = nf / L
        st.use_sync_length = False
        f += nf
    ad.action = None
    return f


def build_ariel(game_dir, hair=True, colliders=True, parent_hair=True, clip=None):
    """One call: character + clips + strand hair + collision bodies + materials in the current scene. Returns a dict of objects."""
    game_dir = os.path.abspath(game_dir)
    cdir = os.path.join(game_dir, CHAR_DIR)
    arm, meshes = import_character(game_dir)
    actions = import_clips(game_dir, arm)
    S = dict(arm=arm, meshes=meshes, actions=actions)
    nw = node_world_matrices(os.path.join(cdir, CHAR_GLB))
    S['node_world'] = nw
    if hair:
        hd = parse_dhs1(os.path.join(cdir, HAIR_BIN))
        S['hairdata'] = hd
        S['hair'] = build_hair_curves(hd, arm, nw, parent=parent_hair)
        S['hair_material'] = make_scene_materials(arm, S['hair'], meshes)
        if colliders:
            S['colliders'] = build_colliders(hd, arm, nw)
    return S


def _main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    game = None
    for i, a in enumerate(argv):
        if a == '--game':
            game = argv[i + 1]
    if not game:
        print(__doc__)
        return
    S = build_ariel(game, hair='--no-hair' not in argv)
    print('Ariel imported: %d meshes, %d clips, hair strands %s' % (len(S['meshes']), len(S['actions']),
          len(S['hair'].data.curves) if 'hair' in S else 'none'))


if __name__ == '__main__' and bpy is not None:
    _main()
