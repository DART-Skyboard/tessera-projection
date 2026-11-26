"""Tessera Cloth - futuristic party dress with SELF-COLLISION ON, in a showroom (reproducible scene).

A three-layer gown (satin bodice + A-line skirt, a high-low iridescent overskirt and a shoulder-pinned train)
is ONE cloth garment, so the layers collide with each other (Self-collision checked on the garment), with the
mannequin / plinth / floor (collision bodies) and are blown by a fan.  Stages:

    blender --factory-startup -b --python docs/render_cloth_party.py -- build  [WORKDIR]
    blender --factory-startup -b --python docs/render_cloth_party.py -- sim    [WORKDIR]
    blender --factory-startup -b --python docs/render_cloth_party.py -- render [WORKDIR] [first] [last] [step]
    xvfb-run ... blender --gpu-backend opengl --python docs/render_cloth_party.py -- ui [WORKDIR]   (UI screen capture)

``build``  writes WORKDIR/party.blend (scene, rig animation, wind keys, extension garment set-up)
``sim``    steps the extension's solver frame by frame and writes WORKDIR/party_cache.npz (+ stats json)
``render`` Cycles frames from the cache (clean render, optional)
``ui``     real Blender UI, EEVEE viewport, Tessera Cloth panel visible; replays the cache and grabs the screen

Env: PARTY_COLS (128) PARTY_FRAMES (720) PARTY_FPS (24) PARTY_SUBSTEPS (5) PARTY_ITERS (6).
"""
import json
import math
import os
import sys
import time

import numpy as np

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
STAGE = argv[0] if argv else 'build'
WORK = os.path.abspath(argv[1] if len(argv) > 1 else '/home/claude/tp6/out/party')
os.makedirs(WORK, exist_ok=True)
BLEND = os.path.join(WORK, 'party.blend')
CACHE = os.path.join(WORK, 'party_cache.npz')

COLS = int(os.environ.get('PARTY_COLS', '128'))
FRAMES = int(os.environ.get('PARTY_FRAMES', '720'))
FPS = int(os.environ.get('PARTY_FPS', '24'))
SUBSTEPS = int(os.environ.get('PARTY_SUBSTEPS', '5'))
ITERS = int(os.environ.get('PARTY_ITERS', '6'))

# --------------------------------------------------------------------------------------------- body profile
# (z, rx, ry, cy) of the torso ellipse; front of the figure faces -Y
TORSO = np.array([
    (0.80, 0.150, 0.100, 0.000),
    (0.90, 0.168, 0.110, 0.000),
    (0.97, 0.168, 0.108, 0.000),
    (1.04, 0.118, 0.088, 0.000),
    (1.12, 0.120, 0.090, -0.004),
    (1.20, 0.128, 0.098, -0.010),
    (1.28, 0.145, 0.106, -0.016),
    (1.34, 0.160, 0.100, -0.010),
    (1.40, 0.185, 0.090, 0.000),
    (1.45, 0.130, 0.072, 0.000),
    (1.49, 0.056, 0.056, 0.000),
])


def torso_at(z):
    return tuple(np.interp(z, TORSO[:, 0], TORSO[:, k]) for k in (1, 2, 3))


def loft(rings, close_top=True, close_bottom=True):
    """rings: list of (n,3) arrays with equal n -> verts, quads."""
    n = len(rings[0])
    verts = [p for r in rings for p in r]
    faces = []
    for j in range(len(rings) - 1):
        for i in range(n):
            a = j * n + i
            b = j * n + (i + 1) % n
            faces.append((a, b, b + n, a + n))
    if close_bottom:
        c = len(verts)
        verts.append(np.mean(rings[0], axis=0))
        for i in range(n):
            faces.append((c, (i + 1) % n, i))
    if close_top:
        c = len(verts)
        verts.append(np.mean(rings[-1], axis=0))
        base = (len(rings) - 1) * n
        for i in range(n):
            faces.append((c, base + i, base + (i + 1) % n))
    return np.asarray(verts, float), faces


def ellipse_ring(z, rx, ry, cx=0.0, cy=0.0, n=40):
    a = 2 * np.pi * np.arange(n) / n
    return np.stack([cx + rx * np.sin(a), cy - ry * np.cos(a), np.full(n, z)], 1)


def limb(p0, p1, r0, r1, n=20, rings=8, cap=True):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = np.linalg.norm(d)
    w = d / L
    u = np.cross(w, [0, 1, 0] if abs(w[1]) < 0.9 else [1, 0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(w, u)
    out = []
    for k in range(rings + 1):
        t = k / rings
        r = r0 + (r1 - r0) * t
        c = p0 + d * t
        a = 2 * np.pi * np.arange(n) / n
        out.append(c + r * (np.cos(a)[:, None] * u + np.sin(a)[:, None] * v))
    return loft(out, cap, cap)


def sphere_ellipsoid(c, rx, ry, rz, n=32, m=16):
    rings = []
    for j in range(1, m):
        th = np.pi * j / m
        a = 2 * np.pi * np.arange(n) / n
        rings.append(np.stack([c[0] + rx * np.sin(th) * np.cos(a), c[1] + ry * np.sin(th) * np.sin(a),
                               np.full(n, c[2] + rz * np.cos(th))], 1))
    v, f = loft(rings, True, True)
    return v, f


def merge(parts):
    vs, fs, off = [], [], 0
    for v, f in parts:
        vs.append(v)
        fs += [tuple(i + off for i in q) for q in f]
        off += len(v)
    return np.concatenate(vs), fs


# --------------------------------------------------------------------------------------------- dress geometry
Z_TOP, Z_WAIST, Z_HIP, Z_HEM = 1.34, 1.04, 0.92, 0.20


def skirt_radius(z):
    """(rx, ry) of the A-line skirt shell at height z (below the waist)."""
    t = np.clip((Z_HIP - z) / (Z_HIP - Z_HEM), 0.0, 1.0)
    f = 0.60 * t ** 1.18
    rx0, ry0, _ = torso_at(max(z, 0.97))
    return rx0 + 0.010 + f, ry0 + 0.010 + f


def shell_point(z, theta, extra=0.0, gather=1.0):
    """Point on the dress shell (front = theta 0 at -Y).  Bodice follows the torso; the skirt flares."""
    if z >= Z_WAIST:
        rx, ry, cy = torso_at(z)
        rx, ry = rx + 0.008 + extra, ry + 0.008 + extra
    else:
        rx, ry = skirt_radius(z)
        rx, ry, cy = rx + extra, ry + extra, 0.0
    return np.array([gather * rx * np.sin(theta), cy - gather * ry * np.cos(theta), z])


def build_dress(cols=COLS):
    """One mesh, three disconnected layers.  Returns dict(verts, pattern, quads, layer, mat, pins, rows...)."""
    rng = np.random.RandomState(7)
    th = 2 * np.pi * np.arange(cols) / cols
    V, P, Q, layer, mat, uvs = [], [], [], [], [], []
    pinw = []

    def pleat(theta, t, amp, seed):
        # hand-tailored gentle pleats: several lobes, growing toward the hem
        return amp * t * (0.55 * np.sin(9 * theta + seed) + 0.30 * np.sin(15 * theta + 2 * seed) + 0.15 * np.sin(4 * theta))

    # ---- layer A: bodice (pinned) + A-line skirt
    zs_b = np.linspace(Z_TOP, Z_WAIST, 13)
    zs_s = np.linspace(Z_WAIST, Z_HEM, 41)[1:]
    zs = np.concatenate([zs_b, zs_s])
    nA = len(zs)
    base = 0
    for r, z in enumerate(zs):
        t = max(0.0, (Z_WAIST - z) / (Z_WAIST - Z_HEM))
        for i in range(cols):
            p = shell_point(z, th[i] + 0.0)
            d = pleat(th[i], t, 0.020, 0.3)
            pat = shell_point(z, th[i], gather=1.0 + 0.30 * t ** 1.1)
            # push pleat radially
            rad = np.array([np.sin(th[i]), -np.cos(th[i]), 0.0])
            V.append(p + rad * d)
            P.append(pat + rad * d * 1.3)
            layer.append(0)
            pinw.append(1.0 if r < 13 else 0.0)
            uvs.append((i / cols, 1.0 - r / (nA - 1)))
    for r in range(nA - 1):
        for i in range(cols):
            a = base + r * cols + i
            b = base + r * cols + (i + 1) % cols
            Q.append((a, b, b + cols, a + cols))
            hem = r >= nA - 3
            mat.append(0 if r < 12 else (2 if hem else 1))
    baseA = len(V)

    # ---- layer B: high-low iridescent overskirt, pinned at the waist
    rowsB = 30
    zw = 1.065
    base = len(V)
    for r in range(rowsB):
        s = r / (rowsB - 1)
        for i in range(cols):
            hem_z = 0.60 + 0.26 * math.cos(th[i]) * 0.0 - 0.20 * (-math.cos(th[i])) * 0.0
            # front (theta 0) high, back (theta pi) long: z_hem = 0.50 + 0.25*cos(theta)
            hz = 0.46 + 0.30 * math.cos(th[i])
            z = zw + (hz - zw) * s
            t = max(0.0, (Z_WAIST - z) / (Z_WAIST - Z_HEM))
            extra = 0.014 + 0.17 * s ** 1.3
            p = shell_point(z, th[i], extra=extra)
            d = pleat(th[i], s, 0.030, 1.1)
            rad = np.array([np.sin(th[i]), -np.cos(th[i]), 0.0])
            pat = shell_point(z, th[i], extra=extra, gather=1.0 + 0.28 * s)
            V.append(p + rad * d)
            P.append(pat + rad * d * 1.3)
            layer.append(1)
            pinw.append(1.0 if r < 2 else 0.0)
            uvs.append((i / cols, 1.0 - s))
    for r in range(rowsB - 1):
        for i in range(cols):
            a = base + r * cols + i
            b = base + r * cols + (i + 1) % cols
            Q.append((a, b, b + cols, a + cols))
            mat.append(4 if r >= rowsB - 3 else 3)

    # ---- layer C: train / cape from the shoulder blades, falls to the floor behind
    ccols, crows = 44, 44
    base = len(V)
    span = 0.95   # +- radians around the back (theta = pi)
    for r in range(crows):
        s = r / (crows - 1)
        z = 1.40 + (0.10 - 1.40) * s
        for i in range(ccols):
            u = (i / (ccols - 1) - 0.5) * 2.0
            theta = math.pi + u * span * (0.55 + 0.45 * s)
            extra = 0.030 + 0.40 * s ** 1.5
            p = shell_point(z, theta, extra=extra)
            d = 0.035 * s * math.sin(7 * u * math.pi * 0.5 + 0.4) * math.sin(math.pi * (i / (ccols - 1)))
            rad = np.array([np.sin(theta), -np.cos(theta), 0.0])
            pat = shell_point(z, theta, extra=extra, gather=1.0 + 0.22 * s)
            V.append(p + rad * d)
            P.append(pat + rad * d * 1.3)
            layer.append(2)
            pinw.append(1.0 if r < 2 else 0.0)
            uvs.append((i / (ccols - 1), 1.0 - s))
    for r in range(crows - 1):
        for i in range(ccols - 1):
            a = base + r * ccols + i
            Q.append((a, a + 1, a + 1 + ccols, a + ccols))
            mat.append(6 if (i < 2 or i >= ccols - 3 or r >= crows - 3) else 5)
    V, P = np.asarray(V), np.asarray(P)
    return dict(verts=V, pattern=P, quads=Q, layer=np.asarray(layer), mat=mat, pins=np.asarray(pinw), uv=np.asarray(uvs),
                nA=baseA)


# --------------------------------------------------------------------------------------------- materials
def _mat(name):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    return m, nt


def _principled(nt, **kw):
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    for k, v in kw.items():
        key = k.replace('_', ' ')
        if key in bsdf.inputs:
            bsdf.inputs[key].default_value = v
    return bsdf


def _out(nt, shader):
    o = nt.nodes.new('ShaderNodeOutputMaterial')
    nt.links.new(shader.outputs[0], o.inputs['Surface'])
    return o


def _ramp(nt, stops):
    r = nt.nodes.new('ShaderNodeValToRGB')
    el = r.color_ramp.elements
    while len(el) > len(stops):
        el.remove(el[-1])
    while len(el) < len(stops):
        el.new(0.5)
    for e, (p, c) in zip(el, stops):
        e.position = p
        e.color = c
    return r


def mat_satin(name, deep, light, sheen=1.0):
    m, nt = _mat(name)
    lw = nt.nodes.new('ShaderNodeLayerWeight')
    lw.inputs['Blend'].default_value = 0.35
    ramp = _ramp(nt, [(0.0, light), (1.0, deep)])
    nt.links.new(lw.outputs['Facing'], ramp.inputs['Fac'])
    b = _principled(nt, Roughness=0.30, Sheen_Weight=sheen, Sheen_Roughness=0.30, Specular_IOR_Level=0.7,
                    Coat_Weight=0.15, Coat_Roughness=0.2, Anisotropic=0.3)
    nt.links.new(ramp.outputs['Color'], b.inputs['Base Color'])
    if 'Sheen Tint' in b.inputs:
        b.inputs['Sheen Tint'].default_value = (0.5, 0.75, 1.0, 1.0)
    _out(nt, b)
    return m


def mat_iridescent(name):
    m, nt = _mat(name)
    lw = nt.nodes.new('ShaderNodeLayerWeight')
    lw.inputs['Blend'].default_value = 0.45
    ramp = _ramp(nt, [(0.0, (0.55, 0.18, 0.95, 1)), (0.4, (0.10, 0.45, 0.95, 1)), (0.75, (0.05, 0.80, 0.85, 1)),
                      (1.0, (0.02, 0.16, 0.45, 1))])
    nt.links.new(lw.outputs['Fresnel'], ramp.inputs['Fac'])
    b = _principled(nt, Roughness=0.18, Metallic=0.35, Sheen_Weight=0.8, Specular_IOR_Level=1.0, Coat_Weight=0.5,
                    Coat_Roughness=0.08)
    nt.links.new(ramp.outputs['Color'], b.inputs['Base Color'])
    nt.links.new(ramp.outputs['Color'], b.inputs['Emission Color'])
    b.inputs['Emission Strength'].default_value = 0.35
    _out(nt, b)
    return m


def mat_emissive(name, color, strength=6.0):
    m, nt = _mat(name)
    e = nt.nodes.new('ShaderNodeEmission')
    e.inputs['Color'].default_value = color
    e.inputs['Strength'].default_value = strength
    _out(nt, e)
    return m


def mat_cape(name):
    m, nt = _mat(name)
    uv = nt.nodes.new('ShaderNodeUVMap')
    wave = nt.nodes.new('ShaderNodeTexWave')
    wave.wave_type = 'BANDS'
    wave.bands_direction = 'Y'
    wave.inputs['Scale'].default_value = 18.0
    wave.inputs['Distortion'].default_value = 5.0
    wave.inputs['Detail'].default_value = 1.0
    nt.links.new(uv.outputs['UV'], wave.inputs['Vector'])
    ramp = _ramp(nt, [(0.0, (0, 0, 0, 1)), (0.80, (0, 0, 0, 1)), (0.86, (1, 1, 1, 1)), (1.0, (0, 0, 0, 1))])
    nt.links.new(wave.outputs['Fac'], ramp.inputs['Fac'])
    b = _principled(nt, Roughness=0.32, Sheen_Weight=0.9, Sheen_Roughness=0.4, Coat_Weight=0.2, Metallic=0.1)
    b.inputs['Base Color'].default_value = (0.006, 0.012, 0.09, 1)
    nt.links.new(ramp.outputs['Color'], b.inputs['Emission Color'])
    b.inputs['Emission Strength'].default_value = 2.2
    # cyan tint of the circuit lines
    mix = nt.nodes.new('ShaderNodeMix')
    mix.data_type = 'RGBA'
    mix.blend_type = 'MULTIPLY'
    mix.inputs['Factor'].default_value = 1.0
    nt.links.new(ramp.outputs['Color'], mix.inputs['A'])
    mix.inputs['B'].default_value = (0.1, 0.8, 1.0, 1.0)
    nt.links.new(mix.outputs['Result'], b.inputs['Emission Color'])
    _out(nt, b)
    return m


def mat_metal(name, color, rough):
    m, nt = _mat(name)
    b = _principled(nt, Metallic=1.0, Roughness=rough)
    b.inputs['Base Color'].default_value = color
    _out(nt, b)
    return m


def mat_plain(name, color, rough=0.5, metallic=0.0, coat=0.0):
    m, nt = _mat(name)
    b = _principled(nt, Roughness=rough, Metallic=metallic, Coat_Weight=coat)
    b.inputs['Base Color'].default_value = color
    _out(nt, b)
    return m


# --------------------------------------------------------------------------------------------- scene build
def new_obj(name, verts, faces, coll, smooth=True):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, v)) for v in verts], [], [tuple(map(int, f)) for f in faces])
    me.update()
    if smooth:
        for p in me.polygons:
            p.use_smooth = True
    o = bpy.data.objects.new(name, me)
    coll.objects.link(o)
    return o


def box_obj(name, size, loc, coll, mat=None, rot=(0, 0, 0)):
    x, y, z = size[0] / 2, size[1] / 2, size[2] / 2
    v = [(-x, -y, -z), (x, -y, -z), (x, y, -z), (-x, y, -z), (-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)]
    f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    o = new_obj(name, v, f, coll, smooth=False)
    o.location = loc
    o.rotation_euler = rot
    if mat:
        o.data.materials.append(mat)
    return o


def cyl_obj(name, r, h, loc, coll, mat=None, n=48, rot=(0, 0, 0), r2=None):
    r2 = r if r2 is None else r2
    a = 2 * np.pi * np.arange(n) / n
    ring0 = np.stack([r * np.cos(a), r * np.sin(a), np.full(n, -h / 2)], 1)
    ring1 = np.stack([r2 * np.cos(a), r2 * np.sin(a), np.full(n, h / 2)], 1)
    v, f = loft([ring0, ring1], True, True)
    o = new_obj(name, v, f, coll)
    o.location = loc
    o.rotation_euler = rot
    if mat:
        o.data.materials.append(mat)
    return o


def torus_obj(name, R, r, loc, coll, mat, nu=64, nv=12, scale=(1, 1)):
    verts, faces = [], []
    for i in range(nu):
        u = 2 * np.pi * i / nu
        for j in range(nv):
            v = 2 * np.pi * j / nv
            verts.append(((R + r * np.cos(v)) * np.cos(u) * scale[0], (R + r * np.cos(v)) * np.sin(u) * scale[1], r * np.sin(v)))
    for i in range(nu):
        for j in range(nv):
            a = i * nv + j
            b = i * nv + (j + 1) % nv
            c = ((i + 1) % nu) * nv + (j + 1) % nv
            d = ((i + 1) % nu) * nv + j
            faces.append((a, b, c, d))
    o = new_obj(name, verts, faces, coll)
    o.location = loc
    o.data.materials.append(mat)
    return o


def build_room(coll):
    """Boutique showroom: reflective dark floor, navy walls with LED slats, neon arch, display plinths, fan, sign."""
    floor_m, nt = _mat("Showroom Floor")
    b = _principled(nt, Roughness=0.10, Metallic=0.0, Coat_Weight=0.3, Specular_IOR_Level=0.9)
    chk = nt.nodes.new('ShaderNodeTexChecker')
    chk.inputs['Scale'].default_value = 7.0
    chk.inputs['Color1'].default_value = (0.012, 0.016, 0.030, 1)
    chk.inputs['Color2'].default_value = (0.030, 0.036, 0.060, 1)
    nt.links.new(chk.outputs['Color'], b.inputs['Base Color'])
    _out(nt, b)
    wall_m = mat_plain("Showroom Wall", (0.020, 0.035, 0.085, 1), 0.55)
    slat_m = mat_emissive("LED Slat", (0.2, 0.75, 1.0, 1), 4.0)
    pink_m = mat_emissive("Neon Magenta", (1.0, 0.15, 0.75, 1), 7.0)
    cyan_m = mat_emissive("Neon Cyan", (0.1, 0.85, 1.0, 1), 7.0)
    white_m = mat_emissive("Sign White", (0.85, 0.95, 1.0, 1), 9.0)
    gold = mat_metal("Brushed Gold", (1.0, 0.70, 0.25, 1), 0.22)
    chrome = mat_metal("Chrome", (0.92, 0.94, 0.98, 1), 0.04)
    pearl = mat_plain("Plinth Pearl", (0.86, 0.88, 0.95, 1), 0.18, 0.0, 0.6)

    box_obj("Showroom Floor", (11.0, 9.0, 0.10), (0, 1.0, -0.05), coll, floor_m)
    box_obj("Back Wall", (11.0, 0.15, 4.2), (0, 3.6, 2.1), coll, wall_m)
    box_obj("Left Wall", (0.15, 9.0, 4.2), (-5.5, 1.0, 2.1), coll, wall_m)
    box_obj("Right Wall", (0.15, 9.0, 4.2), (5.5, 1.0, 2.1), coll, wall_m)
    box_obj("Ceiling", (11.0, 9.0, 0.10), (0, 1.0, 4.25), coll, mat_plain("Ceiling", (0.03, 0.04, 0.08, 1), 0.7))
    # back wall LED slats
    for k in range(-8, 9):
        if abs(k) < 3:
            continue
        box_obj("LED Slat %d" % k, (0.05, 0.03, 3.2), (k * 0.55, 3.5, 2.0), coll, slat_m)
    # neon arch frame behind the figure
    t = torus_obj("Neon Arch", 1.25, 0.022, (0, 3.35, 1.25), coll, cyan_m, nu=96, nv=8)
    t.rotation_euler = (math.pi / 2, 0, 0)
    t2 = torus_obj("Neon Arch Outer", 1.45, 0.016, (0, 3.35, 1.25), coll, pink_m, nu=96, nv=8)
    t2.rotation_euler = (math.pi / 2, 0, 0)
    # ceiling LED rings
    for k, x in enumerate((-2.6, 0.0, 2.6)):
        r = torus_obj("Ceiling Ring %d" % k, 0.75, 0.02, (x, 1.2, 4.12), coll, white_m, nu=64, nv=8)
    # plinth stage (the mannequin stands on it) with a gold rim
    cyl_obj("Stage Plinth", 0.62, 0.12, (0, 0, 0.06), coll, pearl)
    torus_obj("Stage Rim", 0.62, 0.012, (0, 0, 0.115), coll, gold, nu=72, nv=8)
    torus_obj("Stage Glow", 0.66, 0.010, (0, 0, 0.012), coll, cyan_m, nu=72, nv=8)
    # display cubes on both sides
    for sx in (-1, 1):
        box_obj("Display Cube %d" % sx, (0.55, 0.55, 0.9), (sx * 2.6, 2.4, 0.45), coll, pearl)
        box_obj("Display Cube Top %d" % sx, (0.57, 0.57, 0.02), (sx * 2.6, 2.4, 0.91), coll, gold)
        cyl_obj("Display Orb Stand %d" % sx, 0.05, 0.28, (sx * 2.6, 2.4, 1.06), coll, chrome, n=24)
        sph = new_obj("Display Orb %d" % sx, *sphere_ellipsoid((0, 0, 0), 0.14, 0.14, 0.14), coll)
        sph.location = (sx * 2.6, 2.4, 1.32)
        sph.data.materials.append(mat_emissive("Orb Glow %d" % sx, (0.2, 0.7, 1.0, 1) if sx < 0 else (1.0, 0.25, 0.8, 1), 3.0))
    # circular rug
    cyl_obj("Rug", 2.0, 0.012, (0, -0.3, 0.006), coll, mat_plain("Rug", (0.03, 0.05, 0.14, 1), 0.9), n=96)
    # pedestal fan on the left (visible source of the breeze)
    cyl_obj("Fan Pole", 0.025, 0.95, (-2.6, -0.3, 0.52), coll, chrome, n=16)
    cyl_obj("Fan Base", 0.2, 0.04, (-2.6, -0.3, 0.02), coll, chrome, n=32)
    fan = cyl_obj("Fan Cage", 0.30, 0.10, (-2.6, -0.3, 1.1), coll, chrome, n=48, rot=(0, math.pi / 2, 0))
    fan.display_type = 'WIRE'
    hub = cyl_obj("Fan Hub", 0.05, 0.14, (-2.6, -0.3, 1.1), coll, gold, n=20, rot=(0, math.pi / 2, 0))
    # sign: text object
    bpy.ops.object.text_add(location=(0, 3.45, 3.25))
    sign = bpy.context.active_object
    sign.name = "Sign NEBULA COUTURE"
    sign.data.body = "NEBULA  COUTURE"
    sign.data.align_x = 'CENTER'
    sign.data.size = 0.46
    sign.data.extrude = 0.02
    sign.rotation_euler = (math.pi / 2, 0, 0)
    sign.data.materials.append(white_m)
    for c in list(sign.users_collection):
        c.objects.unlink(sign)
    coll.objects.link(sign)
    bpy.ops.object.text_add(location=(0, 3.45, 2.80))
    sub = bpy.context.active_object
    sub.name = "Sign Tagline"
    sub.data.body = "FUTURE  SILK  /  ATELIER  N"
    sub.data.align_x = 'CENTER'
    sub.data.size = 0.14
    sub.data.extrude = 0.008
    sub.rotation_euler = (math.pi / 2, 0, 0)
    sub.data.materials.append(mat_emissive("Sign Pink", (1.0, 0.35, 0.8, 1), 6.0))
    for c in list(sub.users_collection):
        c.objects.unlink(sub)
    coll.objects.link(sub)
    return dict(gold=gold, chrome=chrome)


def add_light(name, kind, loc, target, energy, color, size=1.0, spot=0.0, coll=None):
    from mathutils import Vector
    d = bpy.data.lights.new(name, kind)
    d.energy = energy
    d.color = color
    if kind == 'AREA':
        d.size = size
    if kind == 'SPOT':
        d.spot_size = spot
        d.spot_blend = 0.5
        d.shadow_soft_size = 0.1
    o = bpy.data.objects.new(name, d)
    (coll or bpy.context.scene.collection).objects.link(o)
    o.location = loc
    o.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
    return o


def build():
    from importlib import import_module
    import arc_flow
    CO = import_module('arc_flow.cloth_ops')

    bpy.ops.wm.read_factory_settings(use_empty=True)
    arc_flow.register()
    scene = bpy.context.scene
    scene.name = "Nebula Couture"
    scene.render.fps = FPS
    scene.frame_start, scene.frame_end = 1, FRAMES
    coll = bpy.data.collections.new("Showroom")
    scene.collection.children.link(coll)
    bodyc = bpy.data.collections.new("Mannequin")
    scene.collection.children.link(bodyc)

    # ---------------- mannequin (collision bodies)
    pearl = mat_plain("Mannequin Pearl", (0.88, 0.90, 0.97, 1), 0.22, 0.0, 0.5)
    chrome = mat_metal("Mannequin Chrome", (0.9, 0.93, 1.0, 1), 0.07)
    gold = bpy.data.materials.get("Brushed Gold") or mat_metal("Brushed Gold", (1.0, 0.70, 0.25, 1), 0.22)
    rings = [ellipse_ring(z, rx, ry, 0.0, cy, 40) for z, rx, ry, cy in TORSO]
    body_parts = {}
    tv, tf = loft(rings, True, True)
    body_parts["Torso"] = (tv, tf, pearl)
    hv, hf = sphere_ellipsoid((0, 0.008, 1.64), 0.088, 0.105, 0.118)
    body_parts["Head"] = (hv, hf, chrome)
    nv, nf = limb((0, 0, 1.46), (0, 0, 1.56), 0.052, 0.046, n=20, rings=3)
    body_parts["Neck"] = (nv, nf, pearl)
    for sx, nm in ((-1, "L"), (1, "R")):
        av, af = limb((sx * 0.20, 0.0, 1.38), (sx * 0.31, -0.02, 0.98), 0.046, 0.034, n=20, rings=8)
        body_parts["Arm " + nm] = (av, af, pearl)
        lv, lf = limb((sx * 0.085, 0.0, 0.93), (sx * 0.070, 0.0, 0.12), 0.088, 0.036, n=24, rings=10)
        body_parts["Leg " + nm] = (lv, lf, pearl)
    body_objs = []
    for nm, (v, f, m) in body_parts.items():
        o = new_obj("Mannequin " + nm, v, f, bodyc)
        o.data.materials.append(m)
        body_objs.append(o)
    # gold accents on the figure: choker + belt (follow the rig)
    choker = torus_obj("Gold Choker", 0.056, 0.007, (0, 0, 1.50), bodyc, gold, nu=48, nv=8)
    rx, ry, _ = torso_at(1.065)
    belt = torus_obj("Gold Belt", 1.0, 0.009 / max(rx, 0.1), (0, 0, 1.065), bodyc, gold, nu=64, nv=8)
    belt.scale = (rx + 0.030, ry + 0.030, 1.0)
    brooch = new_obj("Gold Brooch", *sphere_ellipsoid((0, 0, 0), 0.028, 0.012, 0.028, n=20, m=10), bodyc)
    brooch.location = (0, -(ry + 0.043), 1.065)
    brooch.data.materials.append(gold)
    gem = new_obj("Brooch Gem", *sphere_ellipsoid((0, 0, 0), 0.014, 0.012, 0.014, n=16, m=8), bodyc)
    gem.location = (0, -(ry + 0.052), 1.065)
    gem.data.materials.append(mat_emissive("Gem", (0.3, 0.9, 1.0, 1), 8.0))

    # ---------------- room
    build_room(coll)

    # ---------------- rig: one 'Spine' bone at the pelvis; the figure twirls on its plinth
    arm = CO._make_spine_rig(bodyc, "Party Rig")
    for o in body_objs + [choker, belt, brooch, gem]:
        mw = o.matrix_world.copy()
        o.parent = arm
        o.matrix_parent_inverse = arm.matrix_world.inverted()
        o.matrix_world = mw

    # ---------------- dress
    D = build_dress(COLS)
    dress = new_obj("Nebula Gown", D['verts'], D['quads'], bodyc)
    me = dress.data
    mats = [mat_satin("Gown Bodice Satin", (0.01, 0.06, 0.42, 1), (0.10, 0.45, 1.0, 1), 1.0),
            mat_satin("Gown Skirt Satin", (0.015, 0.09, 0.55, 1), (0.12, 0.55, 1.0, 1), 1.0),
            mat_emissive("Gown Hem Glow", (0.15, 0.85, 1.0, 1), 5.0),
            mat_iridescent("Gown Overskirt Iridescent"),
            mat_emissive("Overskirt Hem Glow", (1.0, 0.3, 0.85, 1), 5.0),
            mat_cape("Gown Train Circuit"),
            mat_emissive("Train Edge Glow", (0.1, 0.8, 1.0, 1), 4.0)]
    for m in mats:
        me.materials.append(m)
    for p, mi in zip(me.polygons, D['mat']):
        p.material_index = mi
        p.use_smooth = True
    uv = me.uv_layers.new(name="UVMap")
    for poly in me.polygons:
        for li in poly.loop_indices:
            vi = me.loops[li].vertex_index
            uv.data[li].uv = tuple(D['uv'][vi])
    # vertex groups: the rig bone + pins + layer groups (for the panel's per-layer / ignore controls)
    pins = np.nonzero(D['pins'] > 0.5)[0]
    vg = dress.vertex_groups.new(name="Spine")
    vg.add([int(i) for i in pins], 1.0, 'REPLACE')
    CO.write_pin_group(dress, pins, "ClothPins")
    for k, nm in enumerate(("Layer Skirt", "Layer Overskirt", "Layer Train")):
        g = dress.vertex_groups.new(name=nm)
        g.add([int(i) for i in np.nonzero(D['layer'] == k)[0]], 1.0, 'REPLACE')
    dress["tessera_cloth_pattern"] = [float(c) for c in D['pattern'].ravel()]
    dress["party_layers"] = [int(i) for i in D['layer']]
    sub = dress.modifiers.new("Gown Subdiv", 'SUBSURF')
    sub.levels, sub.render_levels = 0, 1

    # ---------------- extension wiring
    cl = scene.arc_flow_cloth
    g = CO.assign_garment(scene, dress, preset='SILK', quality='BALANCED', pin_mode='NONE', armature=arm)
    g.name = "Nebula Gown"
    g.pin_group = "ClothPins"
    g.follow_preset = True
    g.override_quality = True
    g.substeps, g.iterations = SUBSTEPS, ITERS
    g.thickness = 0.0035
    g.margin = 0.006
    g.self_collision = True                 # <- the point of this scene
    g.self_collision_mode = 'ACCURATE'
    g.self_thickness = 0.006
    g.self_friction = 0.35
    for o in body_objs:
        CO.add_collider(scene, o)
    # plinth + floor are environment colliders
    for nm in ("Stage Plinth", "Showroom Floor"):
        CO.add_collider(scene, bpy.data.objects[nm])
    cl.use_scene_wind = False
    cl.use_fluid_velocity = False
    cl.wind = (3.0, 0.4, 0.2)
    cl.status = "Nebula Gown: 3 layers, self-collision ON."
    animate(scene, arm, cl)
    scene.frame_set(1)

    # ---------------- lights / world / camera
    world = bpy.data.worlds.new("Showroom World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes['Background']
    bg.inputs['Color'].default_value = (0.02, 0.035, 0.09, 1)
    bg.inputs['Strength'].default_value = 0.8
    lc = bpy.data.collections.new("Lights")
    scene.collection.children.link(lc)
    add_light("Key Spot", 'SPOT', (-1.4, -3.2, 3.8), (0, 0, 1.0), 900, (1.0, 0.95, 0.9), spot=0.9, coll=lc)
    add_light("Top Spot", 'SPOT', (0.0, 0.4, 4.0), (0, 0, 0.9), 700, (0.9, 0.95, 1.0), spot=0.7, coll=lc)
    add_light("Rim Cyan", 'AREA', (2.4, 2.6, 2.6), (0, 0, 1.1), 500, (0.2, 0.8, 1.0), size=1.5, coll=lc)
    add_light("Rim Magenta", 'AREA', (-2.6, 2.4, 2.4), (0, 0, 1.1), 420, (1.0, 0.25, 0.8), size=1.5, coll=lc)
    add_light("Front Fill", 'AREA', (1.8, -3.5, 1.6), (0, 0, 1.0), 220, (0.8, 0.9, 1.0), size=3.0, coll=lc)
    add_light("Floor Bounce", 'AREA', (0, -0.5, 0.25), (0, 0, 1.0), 120, (0.4, 0.7, 1.0), size=2.5, coll=lc)

    camd = bpy.data.cameras.new("Showroom Cam")
    cam = bpy.data.objects.new("Showroom Cam", camd)
    scene.collection.objects.link(cam)
    camd.lens = 42
    camd.dof.use_dof = False
    cam.location = (2.4, -4.6, 1.5)
    from mathutils import Vector
    cam.rotation_euler = (Vector((0, 0, 1.0)) - cam.location).to_track_quat('-Z', 'Y').to_euler()
    scene.camera = cam

    CO.build_solver(scene, g)
    S = CO.RT.solvers[g.name]
    print("garment verts", S.n, "tris", len(S.tris), "pins", int(S.pin_hard.sum()), "self_collision", S.self_collision,
          S.self_collision_mode, "th", S.self_thickness)
    bpy.ops.wm.save_as_mainfile(filepath=BLEND)
    print("saved", BLEND)


def animate(scene, arm, cl):
    """Pirouettes of the figure + fan gusts; keys are on the rig and on the cloth wind vector."""
    def smooth(a, b, t):
        t = min(max((t - a) / (b - a), 0.0), 1.0)
        return t * t * (3 - 2 * t)
    secs = FRAMES / FPS
    for k in range(0, FRAMES + 1, 6):
        t = k / FPS
        rz = (smooth(3.0, 9.5, t) * 1.5 * 2 * math.pi
              - smooth(13.5, 20.5, t) * 2.0 * 2 * math.pi * 0.5
              + smooth(23.0, 27.0, t) * 0.5 * 2 * math.pi)
        sway = 0.06 * math.sin(t * 1.3) * smooth(1.0, 4.0, t)
        arm.rotation_euler = (0.0, 0.0, rz)
        arm.location = (sway, 0.0, 0.0)
        arm.keyframe_insert('rotation_euler', frame=k + 1)
        arm.keyframe_insert('location', frame=k + 1)
        gust = 2.5 + 2.8 * (0.5 + 0.5 * math.sin(t * 0.9)) + 1.6 * math.sin(t * 2.7) ** 2
        env = 1.0 if t > 1.0 else t
        cl.wind = (gust * env, 0.35 * math.sin(t * 0.5), 0.15 + 0.2 * math.sin(t * 1.7))
        cl.keyframe_insert('wind', frame=k + 1)
    for fc in (arm.animation_data.action.fcurves if hasattr(arm.animation_data.action, 'fcurves') else []):
        for kp in fc.keyframe_points:
            kp.interpolation = 'BEZIER'


# --------------------------------------------------------------------------------------------- simulation
def sim():
    import arc_flow
    arc_flow.register()
    from importlib import import_module
    CO = import_module('arc_flow.cloth_ops')
    bpy.ops.wm.open_mainfile(filepath=BLEND)
    scene = bpy.context.scene
    cl = scene.arc_flow_cloth
    g = cl.garments[0]
    CO.RT.reset()
    CO.RT.live_running = True
    scene.frame_set(1)
    S = CO.build_solver(scene, g)
    print("sim:", S.n, "verts, self_collision", S.self_collision, S.self_collision_mode, "substeps", S.substeps,
          "iters", S.iterations, flush=True)
    start = int(os.environ.get('PARTY_START', '1'))
    frames = [S.x.astype(np.float32).copy()]
    ms, sc_ms, pairs, contacts, sdist, pen = [], [], [], [], [], []
    # settle: let the gown fall into shape before the shot starts (no timeline motion)
    cl.wind = (0.0, 0.0, 0.0)
    for k in range(int(os.environ.get('PARTY_SETTLE', '24'))):
        CO.step_scene(scene, 1.0 / FPS)
        if os.environ.get('PARTY_DEBUG'):
            print("settle %d  %.0f ms  max|x| %.2f  maxv %.2f  selfcol %s" % (
                k, g.last_ms, float(np.abs(S.x).max()), float(np.linalg.norm(S.v, axis=1).max()),
                S._selfcol.stats if S._selfcol else None), flush=True)
    fps = FPS
    t_wall = time.time()
    for f in range(2, FRAMES + 1):
        scene.frame_set(f)
        CO.step_scene(scene, 1.0 / fps)
        frames.append(S.x.astype(np.float32).copy())
        ms.append(g.last_ms)
        sc = S._selfcol
        if sc is not None:
            sc_ms.append(sc.stats['ms'])
            pairs.append(sc.stats['pairs'])
            contacts.append(sc.stats['contacts'])
        if f % 12 == 0:
            sdist.append((f, float(S.min_self_distance())))
        pen.append(g.last_penetration)
        if f % 12 == 0:
            print("frame %d  step %.0f ms  self-col %.0f ms  pairs %d contacts %d  min self dist %.4f  body pen %.4f  [%.0f s]"
                  % (f, ms[-1], sc_ms[-1] if sc_ms else 0, pairs[-1] if pairs else 0, contacts[-1] if contacts else 0,
                     sdist[-1][1], pen[-1], time.time() - t_wall), flush=True)
            np.savez(CACHE + '.part.npz', x=np.stack(frames))
    X = np.stack(frames)
    mind_arr = np.full(len(X), np.nan)
    for f, d in sdist:
        mind_arr[f - 1] = d
    np.savez_compressed(CACHE, x=X, fps=FPS, ms=np.asarray(ms), contacts=np.asarray(contacts), pairs=np.asarray(pairs), mind=mind_arr)
    stats = dict(verts=int(S.n), frames=len(X), step_ms_mean=float(np.mean(ms)), step_ms_max=float(np.max(ms)),
                 selfcol_ms_mean=float(np.mean(sc_ms)) if sc_ms else 0.0, pairs_mean=float(np.mean(pairs)) if pairs else 0,
                 contacts_mean=float(np.mean(contacts)) if contacts else 0, min_self_distance=sdist,
                 max_body_penetration=float(np.max(pen)), thickness=S.self_thickness or S.thickness,
                 self_intersections_final=int(S.count_self_intersections()))
    with open(os.path.join(WORK, 'party_stats.json'), 'w') as fh:
        json.dump(stats, fh, indent=1)
    print("sim done", stats['step_ms_mean'], "ms/step mean", flush=True)


# --------------------------------------------------------------------------------------------- dispatch
if STAGE == 'build':
    build()
elif STAGE == 'sim':
    sim()
elif STAGE in ('render', 'ui'):
    sys.path.insert(0, HERE)
    import render_cloth_party_stage2 as S2     # noqa: capture / render helpers live beside this file
    S2.run(STAGE, WORK, BLEND, CACHE, argv[2:])
