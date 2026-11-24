"""Publication-quality Cycles beauty of Tessera Projection Hair.

    blender --factory-startup -b --python docs/render_hair.py -- [outdir] [samples]

A stylized clay studio bust, dense layered auburn Empty Hair Curves driven by
the Tessera XPBD solver (collision parting, elastic compression and spring-back,
gentle wind), rendered with Principled Hair BSDF, 3-point rim lighting and
OIDN denoising — the Extensions-page showcase still.

Env: HAIR_SAMPLES (default 80), HAIR_RES_X (1600), HAIR_RES_Y (900).
"""
import math
import os
import sys
import time

import bpy
import bmesh
import numpy as np
from mathutils import Vector, Matrix, Euler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import arc_flow
from arc_flow import hair_ops as HO
from arc_flow.core import hair_solver as HS
from arc_flow.core import sdf as SDF

arc_flow.register()

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
OUT = argv[0] if argv else os.path.join(HERE, 'renders')
SAMPLES = int(argv[1] if len(argv) > 1 else os.environ.get('HAIR_SAMPLES', '80'))
RES_X = int(os.environ.get('HAIR_RES_X', '1600'))
RES_Y = int(os.environ.get('HAIR_RES_Y', '900'))
os.makedirs(OUT, exist_ok=True)

sc = bpy.context.scene
# start from an empty factory scene
for nm in list(bpy.data.objects.keys()):
    bpy.data.objects.remove(bpy.data.objects[nm], do_unlink=True)


def _metaball_to_mesh(ob, name, voxel=0.010, smooth_iter=7, smooth_factor=0.55):
    """Evaluate a metaball object to a smooth sculpted mesh. Convert invalidates the Object RNA."""
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    bpy.context.view_layer.update()
    bpy.ops.object.convert(target='MESH')
    ob = bpy.context.view_layer.objects.active
    md = ob.modifiers.new("remesh", 'REMESH')
    md.mode = 'VOXEL'
    md.voxel_size = voxel
    bpy.ops.object.modifier_apply(modifier=md.name)
    sm = ob.modifiers.new("smooth", 'SMOOTH')
    sm.iterations = smooth_iter
    sm.factor = smooth_factor
    bpy.ops.object.modifier_apply(modifier=sm.name)
    bpy.ops.object.shade_smooth()
    ob.name = name
    _keep_largest_island(ob)
    return ob


def _keep_largest_island(ob):
    """Drop disconnected metaball crumbs (ears, etc. that failed to union)."""
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bm.verts.ensure_lookup_table()
    seen = set()
    groups = []
    for v in bm.verts:
        if v.index in seen:
            continue
        stack = [v]
        seen.add(v.index)
        grp = [v]
        while stack:
            cur = stack.pop()
            for e in cur.link_edges:
                o = e.other_vert(cur)
                if o.index not in seen:
                    seen.add(o.index)
                    stack.append(o)
                    grp.append(o)
        groups.append(grp)
    if len(groups) > 1:
        groups.sort(key=len, reverse=True)
        drop = [v for g in groups[1:] for v in g]
        bmesh.ops.delete(bm, geom=drop, context='VERTS')
        bm.to_mesh(ob.data)
        ob.data.update()
    n = len(groups)
    bm.free()
    return n


def _count_islands(ob):
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bm.verts.ensure_lookup_table()
    seen = set()
    n = 0
    for v in bm.verts:
        if v.index in seen:
            continue
        n += 1
        stack = [v]
        seen.add(v.index)
        while stack:
            cur = stack.pop()
            for e in cur.link_edges:
                o = e.other_vert(cur)
                if o.index not in seen:
                    seen.add(o.index)
                    stack.append(o)
    bm.free()
    return n


# ----------------------------------------------------------------------------- bust (metaball clay sculpt -> mesh)
def make_bust():
    # threshold 0.4 + generously overlapping radii — the visible metaball
    # surface is only ~0.64 of `radius`, so undersized elements will not union.
    mb = bpy.data.metaballs.new("TS_BustBalls")
    mb.resolution = 0.016
    mb.render_resolution = 0.010
    mb.threshold = 0.40
    ob = bpy.data.objects.new("TS_ShowcaseBust", mb)
    sc.collection.objects.link(ob)

    def el(co, radius, scale=(1, 1, 1), stiffness=2.2, typ='ELLIPSOID'):
        e = mb.elements.new(type=typ)
        e.co = co
        e.radius = radius
        e.stiffness = stiffness
        e.size_x, e.size_y, e.size_z = scale
        return e

    # Skull — smooth stylized mannequin, face toward -Y
    el((0.0, 0.02, 1.62), 0.24, (0.95, 1.02, 1.12), 2.4)
    # Face plane (subtle, not a second skull)
    el((0.0, -0.05, 1.58), 0.15, (0.82, 0.62, 0.90), 2.0)
    # Brow ridge
    el((0.0, -0.10, 1.68), 0.09, (0.88, 0.42, 0.30), 2.0)
    # Soft cheeks (small, heavily overlapping)
    el((-0.07, -0.08, 1.55), 0.07, (0.62, 0.58, 0.52), 1.9)
    el((0.07, -0.08, 1.55), 0.07, (0.62, 0.58, 0.52), 1.9)
    # Chin
    el((0.0, -0.09, 1.48), 0.09, (0.68, 0.58, 0.40), 2.0)
    # Nose
    el((0.0, -0.17, 1.595), 0.055, (0.32, 0.80, 0.48), 2.0)
    # Ears, flush
    el((-0.17, 0.02, 1.60), 0.07, (0.34, 0.48, 0.62), 1.9)
    el((0.17, 0.02, 1.60), 0.07, (0.34, 0.48, 0.62), 1.9)
    # Neck — thick, overlaps chin and chest
    el((0.0, 0.02, 1.40), 0.15, (0.78, 0.70, 1.05), 2.2)
    el((0.0, 0.025, 1.28), 0.14, (0.88, 0.72, 0.70), 2.1)
    # Chest — one broad classical-bust mass (no dumbbell end-caps)
    el((0.0, 0.02, 1.14), 0.26, (1.55, 0.72, 0.70), 2.2)
    el((0.0, -0.02, 1.12), 0.18, (1.20, 0.62, 0.55), 2.0)
    # Shoulder slopes, heavily overlapping the chest
    el((-0.26, 0.02, 1.18), 0.17, (1.20, 0.70, 0.62), 2.0)
    el((0.26, 0.02, 1.18), 0.17, (1.20, 0.70, 0.62), 2.0)
    # Soft arm cut (flatter, still overlapping)
    el((-0.36, 0.02, 1.13), 0.11, (0.75, 0.65, 0.50), 1.9)
    el((0.36, 0.02, 1.13), 0.11, (0.75, 0.65, 0.50), 1.9)

    # shallow eye sockets (negative influence)
    for sx in (-1.0, 1.0):
        e = mb.elements.new(type='BALL')
        e.co = (0.055 * sx, -0.145, 1.638)
        e.radius = 0.038
        e.stiffness = 2.0
        if hasattr(e, 'use_negative'):
            e.use_negative = True

    ob = _metaball_to_mesh(ob, "TS_ShowcaseBust", voxel=0.008, smooth_iter=14, smooth_factor=0.7)
    return ob


def clay_material(ob, color=(0.14, 0.085, 0.065, 1.0), name="TS_BustClay", roughness=0.55, sss=0.04):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial'); out.location = (400, 0)
    prin = nt.nodes.new('ShaderNodeBsdfPrincipled'); prin.location = (0, 0)
    prin.inputs['Base Color'].default_value = color
    if 'Subsurface Weight' in prin.inputs:
        prin.inputs['Subsurface Weight'].default_value = sss
    elif 'Subsurface' in prin.inputs:
        prin.inputs['Subsurface'].default_value = min(0.22, sss * 2.0)
    for key in ('Subsurface Radius',):
        if key in prin.inputs:
            try:
                prin.inputs[key].default_value = (0.8, 0.3, 0.18)
            except Exception:
                pass
    if 'Subsurface Color' in prin.inputs:
        prin.inputs['Subsurface Color'].default_value = (0.78, 0.42, 0.32, 1.0)
    prin.inputs['Roughness'].default_value = roughness
    if 'Specular IOR Level' in prin.inputs:
        prin.inputs['Specular IOR Level'].default_value = 0.32
    elif 'Specular' in prin.inputs:
        prin.inputs['Specular'].default_value = 0.32
    noise = nt.nodes.new('ShaderNodeTexNoise'); noise.location = (-380, -80)
    noise.inputs['Scale'].default_value = 55.0
    if 'Detail' in noise.inputs:
        noise.inputs['Detail'].default_value = 6.0
    bump = nt.nodes.new('ShaderNodeBump'); bump.location = (-180, -80)
    bump.inputs['Strength'].default_value = 0.035
    nt.links.new(noise.outputs['Fac'], bump.inputs['Height'])
    if 'Normal' in prin.inputs:
        nt.links.new(bump.outputs['Normal'], prin.inputs['Normal'])
    nt.links.new(prin.outputs[0], out.inputs['Surface'])
    ob.data.materials.append(mat)
    return mat


# ----------------------------------------------------------------------------- stylized clay hand (collision body)
def _add_ball(loc, radius, scale=(1, 1, 1)):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=loc, segments=20, ring_count=12)
    ob = bpy.context.object
    ob.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return ob


def make_hand():
    """A readable sculpted glove in local space: palm at origin, fingers along +X."""
    parts = []
    # palm + heel + wrist (overlapping spheres fuse under remesh)
    parts.append(_add_ball((0.00, 0.00, 0.00), 0.042, (1.40, 0.68, 1.20)))
    parts.append(_add_ball((0.04, 0.00, 0.00), 0.036, (1.25, 0.62, 1.15)))
    parts.append(_add_ball((-0.055, 0.008, 0.00), 0.030, (1.20, 0.68, 0.95)))
    # thumb — tightly overlapping so remesh fuses a sausage
    parts.append(_add_ball((0.015, -0.028, 0.032), 0.020, (1.15, 0.85, 0.85)))
    parts.append(_add_ball((0.032, -0.042, 0.044), 0.017, (1.15, 0.85, 0.85)))
    parts.append(_add_ball((0.050, -0.056, 0.054), 0.014, (1.15, 0.85, 0.85)))
    # four fingers, 6 overlapping spheres each, curled toward -Y
    for i, z in enumerate((0.036, 0.012, -0.012, -0.034)):
        reach = 0.115 - 0.010 * i
        r0 = 0.016 - 0.0010 * i
        nseg = 6
        for k in range(nseg):
            t = k / (nseg - 1)
            x = 0.048 + reach * t
            y = -0.002 - 0.030 * (t ** 1.35)
            zz = z * (1.0 - 0.10 * t)
            parts.append(_add_ball((x, y, zz), r0 * (1.0 - 0.18 * t)))
    bpy.ops.object.select_all(action='DESELECT')
    for ob in parts:
        ob.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()
    ob = bpy.context.object
    md = ob.modifiers.new("remesh", 'REMESH')
    md.mode = 'VOXEL'
    md.voxel_size = 0.0040
    bpy.ops.object.modifier_apply(modifier=md.name)
    sm = ob.modifiers.new("smooth", 'SMOOTH')
    sm.iterations = 4
    sm.factor = 0.45
    bpy.ops.object.modifier_apply(modifier=sm.name)
    bpy.ops.object.shade_smooth()
    ob.name = "TS_ShowcaseHand"
    clay_material(ob, color=(0.018, 0.020, 0.016, 1.0), name="TS_HandLeather",
                  roughness=0.72, sss=0.0)
    return ob


def hand_world_matrix(offset=(0.0, 0.0, 0.0)):
    """Place the hand at the near-right temple, fingers into the hair.

    `offset` is in world units and is added to the rest (contact) location so
    the collider can be animated in / held / eased back.
    """
    loc = Vector((0.33, -0.14, 1.55)) + Vector(offset)
    # pose B: fingers into the hair, palm toward camera so the glove reads in 3/4
    eul = Euler((math.radians(10), math.radians(-8), math.radians(140)), 'XYZ')
    return Matrix.Translation(loc) @ eul.to_matrix().to_4x4()


def _mat4(M):
    return np.array(M, dtype=np.float64).reshape(4, 4)


# ----------------------------------------------------------------------------- layered, clumped auburn groom on the scalp
def _mesh_world(ob):
    deps = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(deps)
    me = ev.to_mesh()
    try:
        me.calc_loop_triangles()
        nv, nt = len(me.vertices), len(me.loop_triangles)
        v = np.empty(nv * 3, np.float64); me.vertices.foreach_get('co', v)
        n = np.empty(nv * 3, np.float64); me.vertices.foreach_get('normal', n)
        t = np.empty(nt * 3, np.int64); me.loop_triangles.foreach_get('vertices', t)
        M = np.array(ob.matrix_world, dtype=np.float64)
        V = v.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
        N = n.reshape(-1, 3) @ M[:3, :3].T
        N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
        return V, N, t.reshape(-1, 3)
    finally:
        ev.to_mesh_clear()


def draped_strand(root, normal, length, n_pts, head_center, head_r, part_sign, curl=0.0, seed_off=0.0):
    """A rest-pose polyline that leaves the scalp, hugs the skull, then hangs over the shoulders."""
    n_pts = max(int(n_pts), 4)
    root = np.asarray(root, float).reshape(3)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
    center = np.asarray(head_center, float).reshape(3)
    pts = np.empty((n_pts, 3), float)
    pts[0] = root
    # first step: lift along a mix of normal and up — never straight out
    upness = max(0.0, float(normal[2]))
    leave = normal * (0.22 - 0.14 * upness) + np.array((
        part_sign * 0.12,
        0.20 + 0.35 * upness,
        0.88 - 0.75 * upness,
    ))
    leave = leave / max(float(np.linalg.norm(leave)), 1e-12)
    step = float(length) / (n_pts - 1)
    pts[1] = root + leave * step * 0.85
    # hang: mostly down, a little back, gently parted
    hang = np.array((part_sign * 0.16, 0.22, -1.15))
    hang = hang / max(float(np.linalg.norm(hang)), 1e-12)
    caxis = np.cross(hang, np.array((0.0, 0.0, 1.0)))
    if np.linalg.norm(caxis) < 1e-6:
        caxis = np.array((0.0, 1.0, 0.0))
    caxis = caxis / np.linalg.norm(caxis)
    margin = 0.016
    neck_z = center[2] - head_r * 0.55
    for i in range(2, n_pts):
        t = (i - 1) / (n_pts - 2)
        d = (1.0 - t) ** 1.25 * leave + (t ** 0.70) * hang
        ang = curl * t
        ca, sa = math.cos(ang), math.sin(ang)
        d = d * ca + np.cross(caxis, d) * sa + caxis * np.dot(caxis, d) * (1.0 - ca)
        d = d / max(float(np.linalg.norm(d)), 1e-12)
        nxt = pts[i - 1] + d * step
        # soft S-wave for clump texture (amplitude scales with curl)
        nxt = nxt + caxis * (0.018 * curl * math.sin(t * math.pi * 2.15))
        vec = nxt - center
        rad = float(np.linalg.norm(vec))
        if rad < head_r + margin and nxt[2] > neck_z:
            nxt = center + vec / max(rad, 1e-12) * (head_r + margin)
        pts[i] = nxt
    # soft ceiling just above the crown so a few cowlicks stay, but nothing spikes out of frame
    for i in range(1, n_pts):
        if pts[i, 2] > 1.84:
            pts[i, 2] = 1.84
    return pts


def scalp_samples(V, N, n_want, seed=3):
    """Points + normals on the upper/back scalp, with a centre part."""
    z, y, x = V[:, 2], V[:, 1], V[:, 0]
    zmax = float(z.max())
    # scalp cap: include the front hairline and temples, keep a narrow centre part
    mask = (z > zmax - 0.26) & (y > -0.13) & (np.abs(x) < 0.18)
    # forehead only gets the upper band so we don't plant hair on the brow
    forehead = (y < -0.05) & (z < zmax - 0.14)
    mask = mask & ~forehead
    part = np.abs(x) < 0.007
    mask = mask & ~part
    idx = np.flatnonzero(mask)
    if len(idx) < 8:
        # fallback: just the top of the mesh
        mask = z > (zmax - 0.18)
        idx = np.flatnonzero(mask)
    rng = np.random.RandomState(seed)
    w = np.clip((V[idx, 2] - (zmax - 0.20)) * 3.0, 0.15, 1.5)
    w = w * np.clip(0.4 + (V[idx, 1] + 0.06) * 2.6, 0.25, 1.6)
    w = w / max(float(w.sum()), 1e-12)
    pick = rng.choice(idx, size=min(int(n_want) * 3, len(idx)), replace=False, p=w)
    pts = V[pick]
    chosen = [0]
    dmin = np.sum((pts - pts[0]) ** 2, axis=1)
    for _ in range(1, min(int(n_want), len(pick))):
        j = int(np.argmax(dmin))
        chosen.append(j)
        dmin = np.minimum(dmin, np.sum((pts - pts[j]) ** 2, axis=1))
    sel = pick[np.array(chosen)]
    return V[sel], N[sel]


def build_clumped_groom(roots, normals, n_child=14, n_pts=16, seed=5,
                        head_center=(0.0, 0.0, 1.62), head_r=0.15):
    """One draped guide per clump + close children; length/curl vary by clump (layered cut)."""
    rng = np.random.RandomState(seed)
    n_guides = len(roots)
    strands_g, strands_c = [], []
    z_crown = float(np.max(roots[:, 2]))
    for i in range(n_guides):
        r, n = roots[i], normals[i]
        part_sign = 1.0 if r[0] >= 0.0 else -1.0
        # layer: crown shorter, sides/back long enough to break over the shoulders
        height = (r[2] - (z_crown - 0.20)) / 0.22
        back = np.clip(r[1] + 0.04, 0.0, 0.20)
        side = abs(r[0])
        length = (0.24 + 0.12 * (1.0 - np.clip(height, 0, 1))
                  + 0.38 * (back / 0.20) + 0.22 * np.clip(side / 0.14, 0, 1)
                  + 0.03 * rng.rand())
        length = float(np.clip(length, 0.22, 0.72))
        curl = (0.55 + 1.15 * rng.rand()) * (1.0 if rng.rand() < 0.5 else -1.0)
        nseg = n_pts if length > 0.28 else max(10, n_pts - 3)
        guide = draped_strand(r, n, length, nseg, head_center, head_r, part_sign, curl=curl)
        strands_g.append(guide)
        t1 = np.cross(n, np.array((0.0, 0.0, 1.0)))
        if np.linalg.norm(t1) < 1e-6:
            t1 = np.array((1.0, 0.0, 0.0))
        t1 = t1 / np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        for _k in range(int(n_child)):
            rad = 0.003 + 0.009 * rng.rand() ** 1.5
            ang = rng.rand() * 2 * math.pi
            off = t1 * (rad * math.cos(ang)) + t2 * (rad * math.sin(ang))
            clen = length * (0.88 + 0.18 * rng.rand())
            ccurl = curl * (0.75 + 0.5 * rng.rand())
            child = draped_strand(r + off, n, clen, nseg, head_center, head_r, part_sign, curl=ccurl)
            strands_c.append(child)
    return HS.pack_strands(strands_g), HS.pack_strands(strands_c)


def tapered_radii(offsets, r_root=0.0014, r_tip=0.00024):
    off = np.asarray(offsets, dtype=np.int64)
    n = int(off[-1])
    rad = np.empty(n, np.float32)
    for a, b in zip(off[:-1], off[1:]):
        t = np.linspace(0.0, 1.0, int(b - a))
        rad[a:b] = (r_root * (1.0 - t) ** 1.15 + r_tip * t ** 0.85).astype(np.float32)
    return rad


def principled_hair_material():
    mat = bpy.data.materials.new("TS_AuburnHair")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial'); out.location = (420, 0)
    hair = nt.nodes.new('ShaderNodeBsdfHairPrincipled'); hair.location = (80, 0)
    if 'parametrization' in hair.bl_rna.properties:
        try:
            hair.parametrization = 'MELANIN'
        except Exception:
            pass
    info = nt.nodes.new('ShaderNodeHairInfo'); info.location = (-520, 40)

    map_m = nt.nodes.new('ShaderNodeMapRange'); map_m.location = (-240, 80)
    map_m.inputs['From Min'].default_value = 0.0
    map_m.inputs['From Max'].default_value = 1.0
    map_m.inputs['To Min'].default_value = 0.72   # root — deep chestnut
    map_m.inputs['To Max'].default_value = 0.42   # tip — lighter copper
    nt.links.new(info.outputs['Intercept'], map_m.inputs['Value'])
    map_r = nt.nodes.new('ShaderNodeMapRange'); map_r.location = (-240, -80)
    map_r.inputs['To Min'].default_value = 0.62
    map_r.inputs['To Max'].default_value = 0.88
    nt.links.new(info.outputs['Intercept'], map_r.inputs['Value'])
    rnd = nt.nodes.new('ShaderNodeMath'); rnd.location = (-40, 160); rnd.operation = 'MULTIPLY'
    rnd.inputs[1].default_value = 0.10
    if 'Random' in info.outputs:
        nt.links.new(info.outputs['Random'], rnd.inputs[0])
    add = nt.nodes.new('ShaderNodeMath'); add.location = (120, 160); add.operation = 'ADD'
    nt.links.new(map_m.outputs['Result'] if 'Result' in map_m.outputs else map_m.outputs[0], add.inputs[0])
    nt.links.new(rnd.outputs[0], add.inputs[1])

    def sock(node, *names):
        for n in names:
            if n in node.inputs:
                return node.inputs[n]
        return None

    sm = sock(hair, 'Melanin')
    sr = sock(hair, 'Melanin Redness', 'Redness')
    if sm is not None:
        nt.links.new(add.outputs[0], sm)
    if sr is not None:
        nt.links.new(map_r.outputs['Result'] if 'Result' in map_r.outputs else map_r.outputs[0], sr)
    if sock(hair, 'Roughness') is not None:
        map_rough = nt.nodes.new('ShaderNodeMapRange'); map_rough.location = (-240, -240)
        map_rough.inputs['To Min'].default_value = 0.10
        map_rough.inputs['To Max'].default_value = 0.28
        nt.links.new(info.outputs['Intercept'], map_rough.inputs['Value'])
        nt.links.new(map_rough.outputs[0], sock(hair, 'Roughness'))
    if sock(hair, 'Radial Roughness') is not None:
        sock(hair, 'Radial Roughness').default_value = 0.26
    if sock(hair, 'Coat') is not None:
        sock(hair, 'Coat').default_value = 0.55
    if sock(hair, 'IOR') is not None:
        sock(hair, 'IOR').default_value = 1.55
    if sock(hair, 'Random Color') is not None:
        sock(hair, 'Random Color').default_value = 0.15
    if sock(hair, 'Random Roughness') is not None:
        sock(hair, 'Random Roughness').default_value = 0.08
    nt.links.new(hair.outputs[0], out.inputs['Surface'])
    return mat


# ----------------------------------------------------------------------------- studio
def setup_studio():
    # dark cyclorama — large enough that the camera never sees a hard edge
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=40, y_subdivisions=40, size=10.0, location=(0.0, 0.8, 0.0))
    floor = bpy.context.object
    floor.name = "TS_StudioFloor"
    bm = bmesh.new(); bm.from_mesh(floor.data)
    for v in bm.verts:
        y = v.co.y
        if y > 0.45:
            t = (y - 0.45) / 3.2
            v.co.z = (t * t) * 3.6
    bm.to_mesh(floor.data); bm.free()
    floor.data.update()
    bpy.context.view_layer.objects.active = floor
    floor.select_set(True)
    bpy.ops.object.shade_smooth()
    matf = bpy.data.materials.new("TS_StudioBackdrop")
    matf.use_nodes = True
    nt = matf.node_tree
    prin = next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED')
    prin.inputs['Base Color'].default_value = (0.016, 0.016, 0.020, 1.0)
    prin.inputs['Roughness'].default_value = 0.58
    floor.data.materials.append(matf)

    world = sc.world
    world.use_nodes = True
    bg = world.node_tree.nodes.get('Background')
    if bg is not None:
        bg.inputs[0].default_value = (0.008, 0.009, 0.012, 1.0)
        bg.inputs[1].default_value = 0.12

    def area(name, loc, rot, size, energy, color, shape='AREA'):
        lamp = bpy.data.lights.new(name, 'AREA')
        lamp.energy = energy
        lamp.size = size
        lamp.color = color
        if hasattr(lamp, 'spread'):
            lamp.spread = math.radians(150)
        ob = bpy.data.objects.new(name, lamp)
        sc.collection.objects.link(ob)
        ob.location = loc
        ob.rotation_euler = rot
        return ob

    # Key (warm, camera-left, slightly high) — sculpts the clay
    area("TS_Key", (0.85, -1.20, 2.00), (math.radians(58), 0, math.radians(32)), 1.0, 95, (1.0, 0.93, 0.84))
    # Fill (cool, dim, camera-right)
    area("TS_Fill", (-1.05, -0.55, 1.55), (math.radians(80), 0, math.radians(-55)), 1.4, 45, (0.75, 0.82, 1.0))
    # Rim / kicker behind the head — rich hair specular streaks
    area("TS_Rim", (-0.35, 1.25, 1.95), (math.radians(-25), 0, math.radians(195)), 0.40, 520, (1.0, 0.90, 0.78))
    # Hair top light (tight, slightly back)
    area("TS_HairTop", (0.22, 0.55, 2.35), (math.radians(15), 0, math.radians(10)), 0.30, 200, (1.0, 0.94, 0.86))
    # Bounce from below, very soft (clay shadow wrap)
    area("TS_Bounce", (0.1, -0.50, 0.30), (math.radians(90), 0, 0), 1.8, 14, (1.0, 0.9, 0.8))

    cam = bpy.data.cameras.new("TS_ShowcaseCam")
    cam.lens = 50
    cam.sensor_width = 36
    cam.dof.use_dof = True
    cam.dof.aperture_fstop = 4.0
    cob = bpy.data.objects.new("TS_ShowcaseCam", cam)
    sc.collection.objects.link(cob)
    # 3/4 bust: head, shoulders, and the colliding hand, filling the frame
    cob.location = (0.74, -1.80, 1.42)
    loc = Vector(cob.location)
    tgt = Vector((0.03, -0.02, 1.37))
    direction = tgt - loc
    cob.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    cam.dof.focus_object = None
    cam.dof.focus_distance = (tgt - loc).length
    sc.camera = cob
    return cob


# ----------------------------------------------------------------------------- build, simulate, render
print("Building clay bust…", flush=True)
t0 = time.time()
bust = make_bust()
clay_material(bust)
n_isl = _count_islands(bust)
print("  bust: %d verts, %d island(s), %.1fs" % (len(bust.data.vertices), n_isl, time.time() - t0), flush=True)

print("Building collision hand…", flush=True)
hand = make_hand()
hand.matrix_world = hand_world_matrix(offset=(0.0, 0.0, 0.0))
# start the visible object at the held contact pose (final render pose)
bpy.context.view_layer.update()

V, N, T = _mesh_world(bust)
roots, normals = scalp_samples(V, N, n_want=180, seed=4)
zmax = float(V[:, 2].max())
head_mask = V[:, 2] > (zmax - 0.30)
head_center = V[head_mask].mean(axis=0)
# a slightly inset centre so wrap-around sits on the skull, not in front of the face
head_center = np.array((0.0, 0.02, head_center[2]), float)
xy = V[head_mask][:, :2] - head_center[:2]
head_r = float(np.percentile(np.linalg.norm(xy, axis=1), 88) + 0.01)
print("  scalp roots:", len(roots), "  head_c", np.round(head_center, 3), "  head_r %.3f" % head_r, flush=True)
(gP, gOff), (cP, cOff) = build_clumped_groom(
    roots, normals, n_child=16, n_pts=16, seed=7,
    head_center=head_center, head_r=head_r,
)
print("  guides %d  children %d" % (len(gOff) - 1, len(cOff) - 1), flush=True)

# colliders: bust + hand (hand field is in local space, driven by a pose matrix)
field_bust = HS.make_collider(V, T, resolution=48, level=2, mode='SOLID')
Vh, _Nh, Th = _mesh_world(hand)
# convert hand verts into the hand's local space for a movable collider
Mw = np.array(hand.matrix_world, dtype=np.float64)
Minv = np.linalg.inv(Mw)
Vh_local = Vh @ Minv[:3, :3].T + Minv[:3, 3]
field_hand = HS.make_collider(Vh_local, Th, resolution=40, level=2, mode='SOLID')

params = HS.HairParams(
    dt=1.0 / 24.0, substeps=5, gravity=(0.0, 0.0, -9.81),
    damping=0.91, rest_shape=0.50, rest_shape_root=0.90, rest_shape_tip=0.16,
    stretch_iterations=12, bend_iterations=5,
    collision_radius=0.007, collision_iterations=3, collision_friction=0.38,
    sdf_refine=True, wind_drag=1.15, wind_tangent=0.12, fluid_coupling=1.0,
)
solver = HS.HairSolver(gP, gOff, params)
solver.bind_to_mesh(V, T)
solver.add_collider(field_bust, matrix=np.eye(4), radius=0.006)
solver.add_collider(field_hand, matrix=_mat4(hand_world_matrix(offset=(0.18, 0.0, 0.0))), radius=0.012)
solver.attach_children(cP, cOff, k=1, smooth=1.4)  # k=1 keeps tight clumps

print("Simulating groom…", flush=True)
t1 = time.time()
# 1) settle against the bust under a gentle side/back wind
print("  rest z[%.3f %.3f]" % (float(gP[:, 2].min()), float(gP[:, 2].max())), flush=True)
solver.set_wind((0.12, -0.08, 0.0))
for i in range(10):
    solver.step()
# 2) hand approaches — elastic compression / parting around the collider
solver.set_wind((0.04, -0.02, 0.0))
for i in range(12):
    t = (i + 1) / 12.0
    # ease-in
    u = t * t * (3.0 - 2.0 * t)
    off = (0.18 * (1.0 - u), 0.0, 0.0)
    solver.set_collider_matrix(1, _mat4(hand_world_matrix(offset=off)))
    solver.step()
# 3) hold the press, then ease a few centimetres off so the hair spring-back reads
for i in range(6):
    solver.step()
for i in range(8):
    t = (i + 1) / 8.0
    u = t * t * (3.0 - 2.0 * t)
    off = (0.010 * u, 0.0, 0.0)
    solver.set_collider_matrix(1, _mat4(hand_world_matrix(offset=off)))
    solver.step()
# freeze the visible hand at a near-contact pose (part + slight spring-back)
hand.matrix_world = hand_world_matrix(offset=(0.010, 0.0, 0.0))
solver.set_wind((0.0, 0.0, 0.0))
for i in range(8):
    solver.step()
bpy.context.view_layer.update()
print("  %d steps, %.1fs, length-err %.4f" % (solver.frame, time.time() - t1, solver.length_error()), flush=True)

# pack guides + children into one CURVES object
full_strands = HS.unpack_strands(solver.positions, solver.offsets)
ch = solver.child_positions()
if ch is not None:
    full_strands += HS.unpack_strands(ch, solver.child_off)
P, off = HS.pack_strands(full_strands)
rad = tapered_radii(off, r_root=0.0019, r_tip=0.00035)
free = np.ones(len(P), bool)
free[off[:-1]] = False
sd = field_bust.sample_sd(P)
print("  min free sd %.4f  n_inside %d / %d" % (float(sd[free].min()), int((sd[free] < -0.002).sum()), int(free.sum())), flush=True)

hair = HO.make_curves_object("TS_ShowcaseHair", P, off, radii=rad, collection=sc.collection, radius=0.001)
# do NOT add the tube GN — Cycles must see real hair curves for Principled Hair BSDF
hair.data.materials.append(principled_hair_material())
for md in list(hair.modifiers):
    hair.modifiers.remove(md)

setup_studio()

# Cycles
sc.render.engine = 'CYCLES'
sc.cycles.device = 'CPU'
sc.cycles.samples = int(SAMPLES)
sc.cycles.use_denoising = True
if hasattr(sc.cycles, 'denoiser'):
    try:
        sc.cycles.denoiser = 'OPENIMAGEDENOISE'
    except Exception:
        pass
sc.cycles.use_adaptive_sampling = True
if hasattr(sc.cycles, 'adaptive_threshold'):
    sc.cycles.adaptive_threshold = 0.01
sc.view_settings.view_transform = 'Filmic'
sc.view_settings.look = 'Medium Contrast'
sc.render.resolution_x = RES_X
sc.render.resolution_y = RES_Y
sc.render.resolution_percentage = 100
sc.render.image_settings.file_format = 'PNG'
sc.render.image_settings.color_depth = '8'
sc.render.image_settings.compression = 15
sc.render.filepath = os.path.join(OUT, 'hair_solver_preview.png')
sc.render.film_transparent = False

print("camera", tuple(sc.camera.location), "rot", tuple(round(a, 3) for a in sc.camera.rotation_euler), flush=True)
for ob in sc.objects:
    if ob.type in {'MESH', 'CURVES', 'CAMERA'}:
        bb = [ob.matrix_world @ Vector(c) for c in ob.bound_box] if ob.type != 'CAMERA' and hasattr(ob, 'bound_box') and ob.bound_box else []
        if bb:
            xs, ys, zs = zip(*[(v.x, v.y, v.z) for v in bb])
            print("  bbox %-20s x[%.2f %.2f] y[%.2f %.2f] z[%.2f %.2f]" % (ob.name, min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)), flush=True)
        else:
            print("  obj", ob.name, ob.type, tuple(ob.location), flush=True)
print("Rendering Cycles CPU %dx%d, %d samples (OIDN) → %s" % (RES_X, RES_Y, SAMPLES, sc.render.filepath), flush=True)
t2 = time.time()
bpy.ops.render.render(write_still=True)
print("Wrote %s  (%d bytes, %.1fs)" % (sc.render.filepath, os.path.getsize(sc.render.filepath) if os.path.isfile(sc.render.filepath) else 0, time.time() - t2), flush=True)
