"""Tessera Hair collision showcase: a procedural bust (head, neck, shoulders, a top), long + short hair, and a glove that
swats the head. Everything is built here (no downloads) and simulated through the extension's own operators / API:
colliders registered in Scene.arc_flow_hair.colliders, moving glove swept every frame, strand-strand contact, settle, bake.

    blender --factory-startup -b --python docs/render_hair_collision.py -- still  OUTDIR [frame]      one Cycles still
    blender --factory-startup -b --python docs/render_hair_collision.py -- clip   OUTDIR              Cycles frame sequence
    blender --factory-startup --python docs/render_hair_collision.py -- ui       OUTDIR              GUI stepped capture
                                                       (run under xvfb-run + --gpu-backend opengl; see docs/capture_hair_ui.py)
Env: HC_RES_X / HC_RES_Y / HC_SAMPLES / HC_FPS_STRIDE / HC_STYLE (long|short|both).
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
from arc_flow import hair_gn as HG
from arc_flow.core import hair_solver as HS

try:
    arc_flow.register()
except Exception:
    pass

sc = bpy.context.scene
FPS = 24


# ----------------------------------------------------------------------------- bust / hand / helpers (shared with render_hair.py)
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




def _mixin(node, name):
    """The RGBA socket of a Mix node input (the node also carries float / vector twins with the same names)."""
    return next(sk for sk in node.inputs if sk.name == name and sk.type == 'RGBA') if name != 'Factor' else \
        next(sk for sk in node.inputs if sk.name == 'Factor' and sk.type == 'VALUE')


def _mixout(node):
    return next(sk for sk in node.outputs if sk.type == 'RGBA')


# ----------------------------------------------------------------------------- body material (skin head + a top)
def body_material(ob, skin=(0.55, 0.36, 0.28, 1.0), top=(0.03, 0.035, 0.045, 1.0)):
    """One material: warm skin above the neckline, a dark knit top below it (split on object-space height)."""
    mat = bpy.data.materials.new("TS_BustBody")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial'); out.location = (700, 0)
    prin = nt.nodes.new('ShaderNodeBsdfPrincipled'); prin.location = (440, 0)
    geo = nt.nodes.new('ShaderNodeNewGeometry'); geo.location = (-700, 0)
    sep = nt.nodes.new('ShaderNodeSeparateXYZ'); sep.location = (-500, 0)
    nt.links.new(geo.outputs['Position'], sep.inputs[0])
    # neckline: dips in front (-Y) and rises at the sides
    mul = nt.nodes.new('ShaderNodeMath'); mul.operation = 'MULTIPLY_ADD'; mul.location = (-300, -100)
    mul.inputs[1].default_value = 0.16
    nt.links.new(sep.outputs['Y'], mul.inputs[0])
    nt.links.new(sep.outputs['Z'], mul.inputs[2])
    sx = nt.nodes.new('ShaderNodeMath'); sx.operation = 'MULTIPLY'; sx.location = (-300, 80)
    nt.links.new(sep.outputs['X'], sx.inputs[0]); nt.links.new(sep.outputs['X'], sx.inputs[1])
    ad = nt.nodes.new('ShaderNodeMath'); ad.operation = 'SUBTRACT'; ad.location = (-120, 0)
    nt.links.new(mul.outputs[0], ad.inputs[0]); nt.links.new(sx.outputs[0], ad.inputs[1])
    ramp = nt.nodes.new('ShaderNodeMapRange'); ramp.location = (60, 0)
    ramp.inputs['From Min'].default_value = 1.255
    ramp.inputs['From Max'].default_value = 1.275
    nt.links.new(ad.outputs[0], ramp.inputs['Value'])
    mix = nt.nodes.new('ShaderNodeMix'); mix.data_type = 'RGBA'; mix.location = (260, 0)
    _mixin(mix, 'A').default_value = top
    _mixin(mix, 'B').default_value = skin
    nt.links.new(ramp.outputs[0], _mixin(mix, 'Factor'))
    nt.links.new(_mixout(mix), prin.inputs['Base Color'])
    # fabric weave bump on the top only
    noise = nt.nodes.new('ShaderNodeTexNoise'); noise.location = (60, -260)
    noise.inputs['Scale'].default_value = 220.0
    bump = nt.nodes.new('ShaderNodeBump'); bump.location = (260, -260)
    bump.inputs['Strength'].default_value = 0.08
    nt.links.new(noise.outputs['Fac'], bump.inputs['Height'])
    nt.links.new(bump.outputs['Normal'], prin.inputs['Normal'])
    for k, v in (('Roughness', 0.5), ('Subsurface Weight', 0.15), ('Specular IOR Level', 0.35)):
        if k in prin.inputs:
            prin.inputs[k].default_value = v
    if 'Subsurface Radius' in prin.inputs:
        prin.inputs['Subsurface Radius'].default_value = (0.9, 0.35, 0.2)
    nt.links.new(prin.outputs[0], out.inputs['Surface'])
    ob.data.materials.clear()
    ob.data.materials.append(mat)
    return mat


def glove_material(ob):
    mat = bpy.data.materials.new("TS_Glove")
    mat.use_nodes = True
    b = next(n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    b.inputs['Base Color'].default_value = (0.012, 0.014, 0.012, 1.0)
    b.inputs['Roughness'].default_value = 0.55
    ob.data.materials.clear()
    ob.data.materials.append(mat)


# ----------------------------------------------------------------------------- groom (long / short)
STYLES = {
    #        base len, back extra, side extra, min, max, n_guides, children/guide
    'long':  dict(base=0.30, back=0.38, side=0.20, lo=0.26, hi=0.74, guides=150, kids=9, curl=0.8, preset='CHESTNUT'),
    'short': dict(base=0.10, back=0.04, side=0.05, lo=0.09, hi=0.20, guides=130, kids=9, curl=0.6, preset='BLONDE'),
}


def build_guides(V, N, style, seed=4):
    cfg = STYLES[style]
    roots, normals = scalp_samples(V, N, n_want=cfg['guides'], seed=seed)
    zmax = float(V[:, 2].max())
    head_mask = V[:, 2] > (zmax - 0.30)
    hc = V[head_mask].mean(axis=0)
    hc = np.array((0.0, 0.02, hc[2]), float)
    xy = V[head_mask][:, :2] - hc[:2]
    hr = float(np.percentile(np.linalg.norm(xy, axis=1), 88) + 0.01)
    rng = np.random.RandomState(seed + 11)
    z_crown = float(roots[:, 2].max())
    strands = []
    for r, n in zip(roots, normals):
        part = 1.0 if r[0] >= 0 else -1.0
        height = np.clip((r[2] - (z_crown - 0.20)) / 0.22, 0, 1)
        back = np.clip(r[1] + 0.04, 0.0, 0.20) / 0.20
        side = np.clip(abs(r[0]) / 0.14, 0, 1)
        length = cfg['base'] * (1.0 - 0.25 * height) + cfg['back'] * back + cfg['side'] * side + 0.03 * rng.rand()
        length = float(np.clip(length, cfg['lo'], cfg['hi']))
        curl = (0.45 + 0.9 * rng.rand()) * cfg['curl'] * (1.0 if rng.rand() < 0.5 else -1.0)
        nseg = 18 if length > 0.3 else 12
        strands.append(draped_strand(r, n, length, nseg, hc, hr, part, curl=curl))
    return HS.pack_strands(strands), hc, hr


def make_hair_object(name, V, N, style, seed=4):
    cfg = STYLES[style]
    (gP, gOff), hc, hr = build_guides(V, N, style, seed)
    cP, cOff = HS.generate_children(gP, gOff, per_guide=cfg['kids'], spread=0.011, clump=0.6, clump_power=1.4,
                                    frizz=0.0035, frizz_freq=4.0, tip_spread=0.006, length_jitter=0.07, seed=seed + 3)
    allP, allOff = HS.pack_strands(HS.unpack_strands(gP, gOff) + HS.unpack_strands(cP, cOff))
    rad = np.concatenate([np.linspace(0, 1, b - a) for a, b in zip(allOff[:-1], allOff[1:])])
    rad = (0.0017 * (1.0 - 0.82 * rad ** 1.2)).astype(np.float32)
    ob = HO.make_curves_object(name, allP, allOff, radii=rad, collection=sc.collection, radius=0.0012)
    ob.arc_flow_hair.role = 'HAIR'
    ob.arc_flow_hair.n_guides = len(gOff) - 1
    ob.arc_flow_hair.radius = 0.0017
    HG.apply_hair_material(ob, preset=cfg['preset'])
    return ob


def hair_scalp(bust, hair, preset):
    return HG.make_scalp_shell(bust, hair, preset=preset, offset=0.002, reach=0.03)


# ----------------------------------------------------------------------------- glove choreography
def glove_pose(loc, pitch=10.0, yaw=-8.0, roll=140.0):
    eul = Euler((math.radians(pitch), math.radians(yaw), math.radians(roll)), 'XYZ')
    return Matrix.Translation(Vector(loc)) @ eul.to_matrix().to_4x4()


def _smooth(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def glove_track(seconds=30.0):
    """(time -> (loc, pitch, yaw, roll)) keyframe table: swat the side of the head, brush the hair on the shoulder,
    then (short-hair half) a second swat from the other side."""
    K = [  # t, loc, pitch, yaw, roll
        (0.0, (0.95, -0.10, 1.95), 10, -8, 140),
        (3.0, (0.80, -0.10, 1.90), 10, -8, 140),
        (4.0, (0.42, -0.06, 1.66), 10, -8, 140),        # wind-up beside the head
        (4.35, (0.17, -0.04, 1.62), 12, -8, 140),       # swat the side of the head
        (4.55, (0.10, -0.04, 1.60), 12, -8, 140),
        (5.6, (0.50, -0.06, 1.80), 10, -8, 140),
        (7.5, (0.90, -0.10, 1.95), 10, -8, 140),
        (9.0, (0.60, -0.15, 1.20), 20, -8, 120),        # comes down to the shoulder
        (10.0, (0.30, -0.10, 1.26), 20, -8, 120),
        (11.2, (0.24, 0.02, 1.22), 25, -8, 110),        # push the long hair on the shoulder
        (12.0, (0.34, 0.10, 1.10), 25, -8, 110),
        (13.2, (0.85, 0.0, 1.50), 10, -8, 140),
        (17.0, (0.95, -0.10, 1.95), 10, -8, 140),
        (18.0, (0.95, -0.10, 1.95), 10, -8, 140),
    ]
    return K


def eval_track(K, t):
    if t <= K[0][0]:
        k = K[0]
        return glove_pose(k[1], k[2], k[3], k[4])
    for a, b in zip(K[:-1], K[1:]):
        if a[0] <= t <= b[0]:
            u = _smooth((t - a[0]) / max(b[0] - a[0], 1e-6))
            loc = [a[1][i] + (b[1][i] - a[1][i]) * u for i in range(3)]
            return glove_pose(loc, a[2] + (b[2] - a[2]) * u, a[3] + (b[3] - a[3]) * u, a[4] + (b[4] - a[4]) * u)
    k = K[-1]
    return glove_pose(k[1], k[2], k[3], k[4])


def short_track():
    """Second half: glove swats the short-haired head from the other side, twice."""
    return [
        (0.0, (-0.95, -0.10, 1.95), 10, 8, -140),
        (1.5, (-0.80, -0.10, 1.80), 10, 8, -140),
        (2.4, (-0.40, -0.06, 1.68), 10, 8, -140),
        (2.75, (-0.15, -0.04, 1.64), 12, 8, -140),
        (2.95, (-0.08, -0.04, 1.62), 12, 8, -140),
        (4.0, (-0.45, -0.06, 1.85), 10, 8, -140),
        (5.5, (-0.45, -0.06, 1.70), 10, 8, -140),
        (5.85, (-0.13, 0.02, 1.70), 12, 8, -140),      # back-hand tap on top / back of the head
        (6.05, (-0.06, 0.04, 1.76), 12, 8, -140),
        (7.5, (-0.90, -0.10, 1.95), 10, 8, -140),
        (12.0, (-0.95, -0.10, 1.95), 10, 8, -140),
    ]


# ----------------------------------------------------------------------------- studio + camera
def setup_studio():
    bpy.ops.mesh.primitive_plane_add(size=12.0, location=(0.0, 0.5, 0.0))
    floor = bpy.context.object
    floor.name = "TS_StudioFloor"
    m = bpy.data.materials.new("TS_StudioBackdrop")
    m.use_nodes = True
    b = next(n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    b.inputs['Base Color'].default_value = (0.05, 0.055, 0.07, 1.0)
    b.inputs['Roughness'].default_value = 0.7
    floor.data.materials.append(m)
    bpy.ops.mesh.primitive_plane_add(size=12.0, location=(0.0, 2.2, 3.0), rotation=(math.radians(90), 0, 0))
    wall = bpy.context.object
    wall.name = "TS_StudioBackdropWall"
    wall.data.materials.append(m)
    world = sc.world or bpy.data.worlds.new("TS_World")
    sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get('Background')
    if bg is not None:
        bg.inputs[0].default_value = (0.04, 0.05, 0.07, 1.0)
        bg.inputs[1].default_value = 0.6

    def area(name, loc, rot, size, energy, color):
        lamp = bpy.data.lights.new(name, 'AREA')
        lamp.energy = energy
        lamp.size = size
        lamp.color = color
        ob = bpy.data.objects.new(name, lamp)
        sc.collection.objects.link(ob)
        ob.location = loc
        ob.rotation_euler = rot
        return ob
    area("TS_Key", (-1.1, -1.5, 2.3), (math.radians(60), 0, math.radians(-35)), 1.2, 260, (1.0, 0.92, 0.82))
    area("TS_Fill", (1.4, -0.9, 1.6), (math.radians(80), 0, math.radians(55)), 1.6, 90, (0.75, 0.82, 1.0))
    area("TS_Rim", (0.5, 1.3, 2.2), (math.radians(-35), 0, math.radians(200)), 0.6, 450, (1.0, 0.9, 0.78))
    cam = bpy.data.cameras.new("TS_Cam")
    cam.lens = 55
    cob = bpy.data.objects.new("TS_Cam", cam)
    sc.collection.objects.link(cob)
    cob.location = (0.0, -2.45, 1.62)
    tgt = Vector((0.05, 0.0, 1.50))
    cob.rotation_euler = (tgt - cob.location).to_track_quat('-Z', 'Y').to_euler()
    sc.camera = cob
    return cob


# ----------------------------------------------------------------------------- scene assembly
def build_scene(styles=('long', 'short')):
    for nm in list(bpy.data.objects.keys()):
        bpy.data.objects.remove(bpy.data.objects[nm], do_unlink=True)
    bust = make_bust()
    bust.name = "TS_Bust"
    body_material(bust)
    bpy.context.view_layer.update()
    glove = make_hand()
    glove.name = "TS_Glove"
    glove_material(glove)
    glove.matrix_world = glove_pose((0.95, -0.10, 1.95))
    V, N, T = _mesh_world(bust)
    hp = sc.arc_flow_hair
    hairs = {}
    for st in styles:
        h = make_hair_object("TS_Hair_" + st.title(), V, N, st)
        hairs[st] = h
    setup_studio()
    # extension wiring: scalp + colliders (the same list the Tessera Hair panel shows)
    hp.surface = bust
    hp.collider = bust
    hp.armature = None
    hp.use_scene_colliders = False
    hp.colliders.clear()
    c = HO.add_collider_entry(sc, glove, thickness=0.012, friction=0.15)
    c.deforming = False
    sc.render.fps = FPS
    hp.dt_mode = 'SCENE'
    hp.substeps = 5
    hp.collision_radius = 0.007
    hp.collision_iterations = 3
    hp.collision_friction = 0.35
    hp.sdf_resolution = 64
    hp.damping = 0.93
    hp.rest_shape = 0.5
    hp.rest_shape_root = 0.9
    hp.rest_shape_tip = 0.15
    hp.stiffness_curve = 1.6
    hp.stretch_iterations = 12
    hp.bend_iterations = 5
    hp.hair_collision = True
    hp.strand_radius = 0.0035
    hp.repel_strength = 0.5
    hp.volume = 0.5
    hp.clumping = 0.35
    hp.use_env_wind = False
    hp.wind = (0.0, 0.0, 0.0)
    hp.use_scene_gravity = False
    hp.gravity = (0.0, 0.0, -9.81)
    hp.guide_ratio = 0.2
    return dict(bust=bust, glove=glove, hairs=hairs, V=V, N=N, T=T)


def bake_style(S, style, seconds, track, offset_t=0.0, stride=1, progress=None):
    """Settle the groom on the body, then sim `seconds` of glove animation. Returns {frame: positions}."""
    hp = sc.arc_flow_hair
    hair = S['hairs'][style]
    hp.hair = hair
    for st, o in S['hairs'].items():
        o.hide_viewport = o.hide_render = (st != style)
    S['glove'].matrix_world = eval_track(track, 0.0)
    bpy.context.view_layer.update()
    HO.RT.solver = None
    HO.RT.bind_ok = False
    n = HO.settle_scene(sc, steps=70)
    solver = HO.RT.solver
    cache = {}
    nfr = int(seconds * FPS)
    t0 = time.time()
    for f in range(nfr):
        S['glove'].matrix_world = eval_track(track, f / FPS)
        bpy.context.view_layer.update()
        HO.step_scene(sc, dt=1.0 / FPS, write=False)
        full, _ = HO._compose_full(solver)
        cache[f] = full.copy()
        if progress and f % 24 == 0:
            progress(style, f, nfr, time.time() - t0)
    return cache


def set_frame_pose(S, style, cache, f):
    hp = sc.arc_flow_hair
    HO.write_curves(S['hairs'][style], cache[min(f, max(cache))])


# ----------------------------------------------------------------------------- render drivers
def cycles_setup(res_x, res_y, samples):
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'CPU'
    sc.cycles.samples = int(samples)
    sc.cycles.use_denoising = True
    try:
        sc.cycles.denoiser = 'OPENIMAGEDENOISE'
    except Exception:
        pass
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.02
    sc.cycles.max_bounces = 4
    sc.cycles.diffuse_bounces = 2
    sc.cycles.glossy_bounces = 3
    sc.cycles.transmission_bounces = 4
    try:
        sc.cycles.curve_subdivisions = 2
    except Exception:
        pass
    try:
        sc.cycles.hair_shape = 'THICK'
    except Exception:
        pass
    sc.view_settings.view_transform = 'AgX' if 'AgX' in [t.identifier for t in sc.view_settings.bl_rna.properties['view_transform'].enum_items] else 'Filmic'
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = int(res_x), int(res_y), 100
    sc.render.image_settings.file_format = 'PNG'
    sc.render.film_transparent = False


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    mode = argv[0] if argv else 'still'
    out = argv[1] if len(argv) > 1 else os.path.join(HERE, 'renders')
    os.makedirs(out, exist_ok=True)
    rx, ry = int(os.environ.get('HC_RES_X', '960')), int(os.environ.get('HC_RES_Y', '540'))
    samples = int(os.environ.get('HC_SAMPLES', '24'))
    style = os.environ.get('HC_STYLE', 'long')
    stride = int(os.environ.get('HC_FPS_STRIDE', '2'))
    S = build_scene(('long', 'short') if style == 'both' else (style,))
    cycles_setup(rx, ry, samples)
    prog = lambda st, f, n, dt: print("  sim %s %d/%d  %.1fs" % (st, f, n, dt), flush=True)
    if mode == 'still':
        f = int(argv[2]) if len(argv) > 2 else 120
        track = glove_track() if style == 'long' else short_track()
        cache = bake_style(S, style, (f + 1) / FPS, track, progress=prog)
        S['glove'].matrix_world = eval_track(track, f / FPS)
        set_frame_pose(S, style, cache, f)
        sc.render.filepath = os.path.join(out, 'still_%s_%04d.png' % (style, f))
        bpy.ops.render.render(write_still=True)
        print("wrote", sc.render.filepath)
    elif mode == 'clip':
        # long-hair half then short-hair half, rendered every `stride`-th sim frame (interpolated to full fps with ffmpeg)
        plan = [('long', glove_track(), 18.0), ('short', short_track(), 12.0)]
        idx = 0
        for st, track, secs in plan:
            if style != 'both' and st != style:
                continue
            cache = bake_style(S, st, secs, track, progress=prog)
            for o in S['hairs'].values():
                o.hide_render = o.hide_viewport = True
            S['hairs'][st].hide_render = S['hairs'][st].hide_viewport = False
            for f in range(0, len(cache), stride):
                S['glove'].matrix_world = eval_track(track, f / FPS)
                set_frame_pose(S, st, cache, f)
                bpy.context.view_layer.update()
                sc.render.filepath = os.path.join(out, 'f%05d.png' % idx)
                idx += 1
                if os.path.exists(sc.render.filepath):
                    continue
                t0 = time.time()
                bpy.ops.render.render(write_still=True)
                print("rendered %s %d (%.1fs)" % (st, f, time.time() - t0), flush=True)


if __name__ == "__main__" and '--' in sys.argv and sys.argv[sys.argv.index('--') + 1:][:1] in (['still'], ['clip']):
    main()
