"""Showcase 2 — boutique storefront: a ladies' dress on a dress form, a pedestal fan
blowing from behind, driven live by the Tessera Projection Cloth solver (CLOTH tab).

The bodice is pinned to the form; the flared, godet-folded skirt is free cloth
(Silk / Light Skirt preset, Balanced quality: XPBD stretch / shear / dihedral bend,
LOTBCBP SDF collision against the form, pole and pedestal, aerodynamic drag + lift).
The fan's airflow is fed to the solver's aero path every tick (oscillating sweep +
gusts), and its blades spin with the airflow.
"""
import math
import os

import bpy
import bmesh
import numpy as np
from mathutils import Euler, Matrix, Vector

from common import (ROOT, addon, sub, clear_scene, area_light, spot_light, cycles_setup, principled)

FPS = 24
FRAMES = 30 * FPS
FAN_POS = Vector((-0.92, 0.80, 0.0))
DRESS_C = Vector((0.0, 0.0, 0.95))

# dress-form profile: (z, rx); ry = 0.74 rx
FORM = [(0.92, 0.12), (0.95, 0.158), (1.00, 0.168), (1.06, 0.140), (1.10, 0.124), (1.18, 0.128),
        (1.24, 0.140), (1.30, 0.156), (1.36, 0.152), (1.42, 0.160), (1.46, 0.170), (1.49, 0.150),
        (1.52, 0.090), (1.54, 0.056), (1.62, 0.050), (1.64, 0.030), (1.645, 0.0)]


def _rx(z):
    zs = [p[0] for p in FORM]
    rs = [p[1] for p in FORM]
    return float(np.interp(z, zs, rs))


def _obj(name, verts, faces, smooth=True, coll=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    me.update()
    ob = bpy.data.objects.new(name, me)
    (coll or bpy.context.scene.collection).objects.link(ob)
    if smooth:
        for p in me.polygons:
            p.use_smooth = True
    return ob


def lathe(name, prof, n=64, ry=0.74, cap_bottom=True):
    verts, faces = [], []
    for z, r in prof:
        for i in range(n):
            a = 2 * math.pi * i / n
            verts.append((r * math.sin(a), r * ry * math.cos(a), z))
    rows = len(prof)
    for j in range(rows - 1):
        for i in range(n):
            a, b = j * n + i, j * n + (i + 1) % n
            faces.append((a, b, b + n, a + n))
    if cap_bottom:
        c = len(verts)
        verts.append((0, 0, prof[0][0]))
        for i in range(n):
            faces.append((c, (i + 1) % n, i))
    return _obj(name, verts, faces)


def cyl(name, r, z0, z1, n=32, loc=(0, 0, 0)):
    bpy.ops.mesh.primitive_cylinder_add(vertices=n, radius=r, depth=z1 - z0, location=(loc[0], loc[1], (z0 + z1) / 2))
    ob = bpy.context.object
    ob.name = name
    bpy.ops.object.shade_smooth()
    return ob


def bevel(ob, w=0.004, seg=3):
    md = ob.modifiers.new("Bevel", 'BEVEL')
    md.width = w
    md.segments = seg
    md.limit_method = 'ANGLE'
    return ob


# ----------------------------------------------------------------------------- materials
def mats():
    CG = sub('cloth_gn')
    M = {}
    M['dress'] = CG.fabric_material(name="TS_DressSilk", tint=(0.006, 0.075, 0.05), sheen=0.9)
    p = M['dress'].node_tree.nodes.get('Principled BSDF') or next(n for n in M['dress'].node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    for k, v in (('Roughness', 0.36), ('Anisotropic', 0.55), ('Sheen Weight', 0.6), ('Specular IOR Level', 0.55)):
        if k in p.inputs:
            p.inputs[k].default_value = v
    if 'Sheen Tint' in p.inputs and p.inputs['Sheen Tint'].type != 'VALUE':
        p.inputs['Sheen Tint'].default_value = (0.75, 1.0, 0.88, 1.0)
    M['form'] = principled("TS_FormLinen", (0.78, 0.70, 0.60), rough=0.75, **{'Sheen Weight': 0.3})
    M['brass'] = principled("TS_Brass", (0.86, 0.66, 0.36), rough=0.22, metallic=1.0)
    M['walnut'] = principled("TS_Walnut", (0.16, 0.08, 0.04), rough=0.35, **{'Coat Weight': 0.5, 'Coat Roughness': 0.08})
    M['plinth'] = principled("TS_Plinth", (0.92, 0.90, 0.87), rough=0.28, **{'Coat Weight': 0.3})
    M['wall'] = principled("TS_WallBlush", (0.42, 0.28, 0.27), rough=0.8)
    M["panel"] = principled("TS_WallPanel", (0.36, 0.30, 0.28), rough=0.6)
    M['fan'] = principled("TS_FanEnamel", (0.03, 0.16, 0.15), rough=0.25, **{'Coat Weight': 0.6})
    M['chrome'] = principled("TS_Chrome", (0.9, 0.9, 0.92), rough=0.08, metallic=1.0)
    M['frame'] = principled("TS_WindowFrame", (0.03, 0.035, 0.04), rough=0.4)
    M['gold'] = principled("TS_GoldLetters", (1.0, 0.76, 0.38), rough=0.18, metallic=1.0)
    M['floor'] = floor_material()
    M['green'] = principled("TS_Leaf", (0.05, 0.22, 0.06), rough=0.5, **{'Subsurface Weight': 0.2})
    M['pot'] = principled("TS_Terracotta", (0.55, 0.25, 0.14), rough=0.8)
    return M


def floor_material():
    m = bpy.data.materials.new("TS_HerringboneOak")
    m.use_nodes = True
    nt = m.node_tree
    p = nt.nodes.get('Principled BSDF')
    tc = nt.nodes.new('ShaderNodeTexCoord')
    mp = nt.nodes.new('ShaderNodeMapping')
    mp.inputs['Scale'].default_value = (3.5, 3.5, 3.5)
    br = nt.nodes.new('ShaderNodeTexBrick')
    br.inputs['Scale'].default_value = 1.0
    br.inputs['Mortar Size'].default_value = 0.004
    br.inputs['Brick Width'].default_value = 0.9
    br.inputs['Row Height'].default_value = 0.15
    br.inputs['Color1'].default_value = (0.36, 0.20, 0.10, 1)
    br.inputs['Color2'].default_value = (0.28, 0.15, 0.07, 1)
    br.inputs['Mortar'].default_value = (0.10, 0.06, 0.03, 1)
    nz = nt.nodes.new('ShaderNodeTexNoise')
    nz.inputs['Scale'].default_value = 40.0
    mix = nt.nodes.new('ShaderNodeMixRGB')
    mix.blend_type = 'MULTIPLY'
    mix.inputs['Fac'].default_value = 0.35
    nt.links.new(tc.outputs['Object'], mp.inputs['Vector'])
    nt.links.new(mp.outputs['Vector'], br.inputs['Vector'])
    nt.links.new(mp.outputs['Vector'], nz.inputs['Vector'])
    nt.links.new(br.outputs['Color'], mix.inputs['Color1'])
    nt.links.new(nz.outputs['Color'], mix.inputs['Color2'])
    nt.links.new(mix.outputs['Color'], p.inputs['Base Color'])
    p.inputs['Roughness'].default_value = 0.32
    if 'Coat Weight' in p.inputs:
        p.inputs['Coat Weight'].default_value = 0.35
    return m


def give(ob, m):
    ob.data.materials.clear()
    ob.data.materials.append(m)
    return ob


# ----------------------------------------------------------------------------- set
def storefront(M):
    sc = bpy.context.scene
    W, D, H = 4.2, 3.2, 2.9
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0.4, 0))
    fl = bpy.context.object; fl.name = "TS_Floor"; fl.scale = (W, D, 1); give(fl, M['floor'])
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 1.75, H / 2), rotation=(math.radians(90), 0, 0))
    bw = bpy.context.object; bw.name = "TS_BackWall"; bw.scale = (W, H, 1); give(bw, M['wall'])
    for sx in (-1, 1):
        bpy.ops.mesh.primitive_plane_add(size=1, location=(sx * W / 2, 0.4, H / 2), rotation=(0, math.radians(90), 0))
        sw = bpy.context.object; sw.name = "TS_SideWall_%s" % ('L' if sx < 0 else 'R'); sw.scale = (H, D, 1); give(sw, M['wall'])
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0.4, H))
    ce = bpy.context.object; ce.name = "TS_Ceiling"; ce.scale = (W, D, 1); give(ce, M['panel'])
    # arched back panel (wainscot frame) behind the display
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 1.72, 1.15))
    pn = bpy.context.object; pn.name = "TS_DisplayPanel"; pn.scale = (1.9, 0.04, 2.3); bevel(pn, 0.01); give(pn, M['panel'])
    for x in (-0.98, 0.98):
        bpy.ops.mesh.primitive_cube_add(size=1, location=(x, 1.69, 1.15))
        tr = bpy.context.object; tr.name = "TS_PanelTrim"; tr.scale = (0.05, 0.05, 2.32); bevel(tr, 0.008); give(tr, M['brass'])
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 1.69, 0.06))
    sk = bpy.context.object; sk.name = "TS_Skirting"; sk.scale = (W, 0.06, 0.12); give(sk, M['walnut'])
    # lettering
    bpy.ops.object.text_add(location=(0, 1.68, 2.05), rotation=(math.radians(90), 0, math.radians(180)))
    tx = bpy.context.object; tx.name = "TS_Lettering"
    tx.data.body = "ATELIER  TESSERA"
    tx.data.align_x = 'CENTER'
    tx.data.size = 0.17
    tx.data.extrude = 0.008
    tx.data.bevel_depth = 0.002
    tx.data.materials.append(M['gold'])
    tx.rotation_euler = (math.radians(90), 0, 0)
    tx.location = (0, 1.68, 2.02)
    # storefront window mullions framing the shot (camera side, y = -1.1)
    for x in (-2.0, 2.0):
        bpy.ops.mesh.primitive_cube_add(size=1, location=(x, -1.15, H / 2))
        m = bpy.context.object; m.name = "TS_WindowMullion"; m.scale = (0.12, 0.12, H); give(m, M['frame'])
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, -1.15, H - 0.08))
    m = bpy.context.object; m.name = "TS_WindowHeader"; m.scale = (W, 0.14, 0.16); give(m, M['frame'])
    # pedestal / plinth
    p1 = cyl("TS_Plinth", 0.55, 0.0, 0.10, 96); bevel(p1, 0.01); give(p1, M['plinth'])
    p2 = cyl("TS_PlinthBand", 0.565, 0.035, 0.05, 96); give(p2, M['brass'])
    # potted plants either side (simple sculpted leaves)
    for x, s in ((1.3, 1.0), (-1.45, 0.85)):
        pot = cyl("TS_Pot", 0.16 * s, 0.0, 0.34 * s, 48, loc=(x, 1.25, 0)); give(pot, M['pot'])
        rng = np.random.default_rng(int(abs(x) * 10))
        for k in range(16):
            a = rng.uniform(0, 2 * math.pi); tilt = rng.uniform(0.25, 0.9)
            bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=0.2 * s,
                                                 location=(x + 0.1 * math.cos(a) * s, 1.25 + 0.1 * math.sin(a) * s, 0.55 * s + rng.uniform(0, 0.35) * s))
            lf = bpy.context.object; lf.name = "TS_Leaf"
            lf.scale = (0.32, 0.07, 1.0)
            lf.rotation_euler = (tilt * math.cos(a), tilt * math.sin(a), a)
            bpy.ops.object.shade_smooth(); give(lf, M['green'])
    w = sc.world
    w.use_nodes = True
    bg = w.node_tree.nodes.get('Background')
    bg.inputs[0].default_value = (0.55, 0.62, 0.75, 1)
    bg.inputs[1].default_value = 0.08


def lights():
    tgt = DRESS_C + Vector((0, 0, 0.1))
    spot_light("TS_TrackSpot_Key", (0.85, -0.85, 2.85), tgt, 520, (1.0, 0.86, 0.70), size_deg=34, blend=0.45, radius=0.06)
    spot_light("TS_TrackSpot_Fill", (-1.0, -0.6, 2.85), tgt, 160, (1.0, 0.9, 0.8), size_deg=40, blend=0.6, radius=0.08)
    spot_light("TS_TrackSpot_Rim", (0.6, 1.3, 2.8), tgt, 520, (1.0, 0.92, 0.82), size_deg=34, blend=0.5, radius=0.05)
    spot_light("TS_TrackSpot_Letters", (0, 0.9, 2.85), (0, 1.7, 1.9), 220, (1.0, 0.85, 0.65), size_deg=50, blend=0.7)
    area_light("TS_WindowDaylight", (0.4, -2.2, 1.8), tgt, 2.4, 90, (0.85, 0.92, 1.0), size_y=1.6)
    area_light("TS_CoveWash", (0, 1.2, 2.82), (0, 1.75, 0.6), 2.5, 30, (1.0, 0.8, 0.7), size_y=0.2)
    for x in (-1.5, 1.5):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.05, location=(x, 1.65, 1.75))
        bulb = bpy.context.object; bulb.name = "TS_Sconce"
        em = bpy.data.materials.new("TS_SconceGlow"); em.use_nodes = True
        pe = em.node_tree.nodes.get('Principled BSDF')
        pe.inputs['Emission Color'].default_value = (1.0, 0.75, 0.45, 1)
        pe.inputs['Emission Strength'].default_value = 25.0
        bulb.data.materials.append(em)


# ----------------------------------------------------------------------------- dress form + dress
def dress_form(M):
    form = lathe("TS_DressForm", FORM, n=72)
    give(form, M['form'])
    pole = cyl("TS_FormPole", 0.016, 0.10, 0.93, 24); give(pole, M['brass'])
    cap = cyl("TS_FormNeckCap", 0.056, 1.62, 1.65, 32); give(cap, M['walnut'])
    knob = cyl("TS_FormBase", 0.12, 0.10, 0.13, 48); bevel(knob, 0.008); give(knob, M['walnut'])
    return [form, pole, knob]


def dress_mesh(n=96, rows_bodice=14, rows_skirt=50, z_top=1.40, z_waist=1.08, z_hem=0.30, r_hem=0.50):
    verts = []
    zs_b = np.linspace(z_top, z_waist, rows_bodice)
    for z in zs_b:
        rx = _rx(z) + 0.007
        for i in range(n):
            a = 2 * math.pi * i / n
            # sweetheart neckline dip at the front
            zz = z
            if z == zs_b[0]:
                zz = z - 0.035 * max(0.0, math.cos(a + math.pi)) ** 6 + 0.015 * math.cos(2 * (a + math.pi))
            verts.append((rx * math.sin(a), rx * 0.74 * math.cos(a) * 1.04, zz))
    for j in range(1, rows_skirt + 1):
        v = j / rows_skirt
        z = z_waist + (z_hem - z_waist) * v
        r0 = _rx(z_waist) + 0.007
        base = r0 + (r_hem - r0) * (v ** 0.85)
        ell = 0.74 * 1.04 + (1.0 - 0.74 * 1.04) * min(1.0, v * 3.0)
        for i in range(n):
            a = 2 * math.pi * i / n
            fold = 1.0 + 0.11 * (v ** 1.1) * math.cos(9 * a + 0.4) + 0.04 * (v ** 1.4) * math.sin(5 * a + 1.1)
            r = base * fold
            r = max(r, _rx(z) + 0.012) if z > 0.92 else r
            verts.append((r * math.sin(a), r * ell * math.cos(a), z))
    rows = rows_bodice + rows_skirt
    faces = []
    for j in range(rows - 1):
        for i in range(n):
            a, b = j * n + i, j * n + (i + 1) % n
            faces.append((a, a + n, b + n, b))
    pins = np.arange(0, rows_bodice * n, dtype=np.int64)
    return np.array(verts), faces, pins, n, rows


def build_dress(M, CO):
    V, F, pins, n, rows = dress_mesh()
    ob = _obj("TS_Dress", V, F)
    uv = ob.data.uv_layers.new(name="UVMap")
    for poly in ob.data.polygons:
        for li in poly.loop_indices:
            vi = ob.data.loops[li].vertex_index
            uv.data[li].uv = ((vi % n) / n * 4.0, (vi // n) / rows * 3.0)
    CO.write_pin_group(ob, pins, "ClothPins")
    sol = ob.modifiers.new("Cloth Thickness", 'SOLIDIFY'); sol.thickness = 0.003; sol.offset = 0.0
    s = ob.modifiers.new("Cloth Subdiv", 'SUBSURF'); s.levels = 1; s.render_levels = 1
    give(ob, M['dress'])
    # waist sash
    sash = lathe("TS_DressSash", [(1.07, _rx(1.07) + 0.012), (1.10, _rx(1.10) + 0.013), (1.125, _rx(1.125) + 0.012)], n=112, cap_bottom=False)
    give(sash, principled("TS_SashSatin", (0.75, 0.62, 0.38), rough=0.25, **{'Sheen Weight': 0.4, 'Anisotropic': 0.6}))
    return ob


# ----------------------------------------------------------------------------- fan
def pedestal_fan(M):
    root = bpy.data.objects.new("TS_PedestalFan", None)
    bpy.context.scene.collection.objects.link(root)
    parts = []
    b = cyl("TS_FanBase", 0.17, 0.0, 0.035, 48); bevel(b, 0.01); give(b, M['fan']); parts.append(b)
    p = cyl("TS_FanPole", 0.018, 0.035, 1.02, 24); give(p, M['chrome']); parts.append(p)
    head = bpy.data.objects.new("TS_FanHead", None)
    bpy.context.scene.collection.objects.link(head)
    head.location = (0, 0, 1.08)
    head.parent = root
    for o in parts:
        o.parent = root
    # motor housing along -Y of the head (the head's local +Y points at the dress after aiming)
    def hpart(o):
        o.parent = head
        return o
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=0.085, location=(0, 0.08, 0))
    mh = bpy.context.object; mh.name = "TS_FanMotor"; mh.scale = (1, 1.35, 1); bpy.ops.object.shade_smooth(); give(mh, M['fan']); hpart(mh)
    # cage: rings + spokes
    for k, (y, r) in enumerate(((-0.05, 0.26), (-0.075, 0.2), (-0.09, 0.12), (0.0, 0.265), (0.03, 0.22))):
        bpy.ops.mesh.primitive_torus_add(major_radius=r, minor_radius=0.0035 if k != 0 else 0.007, major_segments=72, minor_segments=8,
                                         location=(0, y, 0), rotation=(math.radians(90), 0, 0))
        t = bpy.context.object; t.name = "TS_FanCageRing"; give(t, M['chrome']); hpart(t)
    for k in range(24):
        a = 2 * math.pi * k / 24
        bpy.ops.mesh.primitive_cylinder_add(vertices=6, radius=0.0022, depth=0.27,
                                            location=(0.13 * math.cos(a), -0.075, 0.13 * math.sin(a)),
                                            rotation=(0, -a + math.pi / 2, 0))
        s = bpy.context.object; s.name = "TS_FanSpoke"; give(s, M['chrome']); hpart(s)
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.045, depth=0.03, location=(0, -0.035, 0), rotation=(math.radians(90), 0, 0))
    badge = bpy.context.object; badge.name = "TS_FanBadge"; give(badge, M['brass']); hpart(badge)
    rotor = bpy.data.objects.new("TS_FanRotor", None)
    bpy.context.scene.collection.objects.link(rotor)
    rotor.parent = head
    rotor.location = (0, -0.03, 0)
    for k in range(3):
        a = 2 * math.pi * k / 3
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=0.1, location=(0, 0, 0))
        bl = bpy.context.object; bl.name = "TS_FanBlade"
        bl.scale = (0.55, 0.08, 1.0)
        bpy.ops.object.transform_apply(scale=True)
        for v in bl.data.vertices:
            v.co.z += 0.105
        bl.rotation_euler = (0, a, 0)
        bl.rotation_euler.rotate(Euler((0, 0, 0)))
        bpy.ops.object.shade_smooth()
        md = bl.modifiers.new("twist", 'SIMPLE_DEFORM'); md.deform_method = 'TWIST'; md.angle = math.radians(25); md.deform_axis = 'Z'
        give(bl, principled("TS_FanBladeTint", (0.05, 0.25, 0.23), rough=0.3, **{'Transmission Weight': 0.0}))
        bl.parent = rotor
    root.location = FAN_POS
    # aim the head at the skirt
    d = (DRESS_C + Vector((0, 0, -0.15))) - (FAN_POS + Vector((0, 0, 1.08)))
    yaw = math.atan2(d.x, -d.y)
    head.rotation_euler = (math.radians(8), 0, yaw)
    return root, head, rotor


# ----------------------------------------------------------------------------- wind (fan airflow)
def fan_wind(frame, head):
    t = frame / FPS
    sweep = math.radians(16) * math.sin(2 * math.pi * t / 9.0)   # oscillating fan head
    base = (DRESS_C - Vector((FAN_POS.x, FAN_POS.y, DRESS_C.z)))
    base.z = 0.0
    base = base.normalized()
    base.z = 0.16                       # a little lift under the hem
    base = (Matrix.Rotation(sweep, 3, 'Z') @ base).normalized()
    ramp = min(1.0, max(0.0, (t - 0.6) / 2.4))                      # fan spins up
    gust = 1.0 + 0.32 * math.sin(2 * math.pi * t / 3.3) + 0.18 * math.sin(2 * math.pi * t / 1.15 + 1.2)
    speed = 6.5 * ramp * gust
    return base * speed, sweep, ramp


# ----------------------------------------------------------------------------- build
def build():
    addon()
    CO = sub('cloth_ops')
    sc = bpy.context.scene
    clear_scene()
    sc.render.fps = FPS
    sc.frame_start, sc.frame_end = 1, FRAMES
    M = mats()
    storefront(M)
    lights()
    colliders = dress_form(M)
    dress = build_dress(M, CO)
    fan_root, fan_head, rotor = pedestal_fan(M)
    cycles_setup(samples=64, preview_samples=10)
    cl = sc.arc_flow_cloth
    cl.garments.clear()
    cl.colliders.clear()
    CO.RT.reset()
    g = CO.assign_garment(sc, dress, preset='SILK', quality='BALANCED', pin_mode='TOP')
    g.pin_group = "ClothPins"
    # real-time budget for a 30 s capture: dihedral bending on, self-collision off
    g.override_quality = True
    g.substeps = 3
    g.iterations = 6
    g.self_collision = False
    g.use_dihedral = True
    g.tessellation_edge = 0.05
    for o in colliders:
        CO.add_collider(sc, o)
    cl.use_scene_wind = False
    cl.use_fluid_velocity = True
    cl.gravity = (0.0, 0.0, -9.81)
    cl.wind = (0.0, 0.0, 0.0)
    sc.arc_flow.ui_tab = 'CLOTH'
    bpy.context.view_layer.update()
    CO.build_solver(sc, g)
    st = dict(CO=CO, cl=cl, g=g, dress=dress, rotor=rotor, head=fan_head, ang=0.0)

    def step(frame):
        w, sweep, ramp = fan_wind(frame, fan_head)
        cl.wind = tuple(w)
        st['ang'] += ramp * math.radians(52)      # ~1250 rpm look at 24 fps without strobing
        rotor.rotation_euler = (0, st['ang'], 0)
        fan_head.rotation_euler.z = st.setdefault('yaw0', fan_head.rotation_euler.z) + sweep
        CO.step_scene(sc, 1.0 / FPS)
        cl.status = "Live  %.1f ms   frame %d   wind %.1f m/s" % (cl.last_ms, frame, w.length)
    # let the skirt settle under gravity before the fan starts
    for _ in range(18):
        CO.step_scene(sc, 1.0 / FPS)
    st['step'] = step
    st['status'] = lambda: cl.status
    st['properties'] = 'MATERIAL'
    return st


def ui(st, sp, r3, ctx):
    from emma_hair_scene import place_view
    sc = bpy.context.scene
    sc.arc_flow.ui_tab = 'CLOTH'
    sc.arc_flow_cloth.live = True
    CO = st['CO']
    CO.RT.live_running = True      # the capture driver ticks exactly one solver step per video frame
    d = st['dress']
    bpy.context.view_layer.objects.active = d
    d.select_set(False)
    place_view(r3, sp, (1.55, -2.75, 1.25), (-0.12, 0.25, 0.92), 32, shift=0.42)
