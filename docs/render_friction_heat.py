"""Publication stills for CE-12: fighter aero friction heat + curve-sourced wispy slipstream,
and cohesive water with whitewater foam on a collision body.

    blender --factory-startup -b --python docs/render_friction_heat.py -- [all|fighter|water] [outdir]

Env: WT_SAMPLES (default 80), WT_PCT (resolution %, default 100). Cycles CPU, OIDN denoising.
"""
import math
import os
import sys
import tempfile

import bpy
import numpy as np
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import arc_flow
arc_flow.register()
from arc_flow import thermal as TM
from arc_flow import windtunnel as WT
from arc_flow.core import fluid_blend as FB

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else ['all']
CASE = argv[0] if argv else 'all'
OUT = argv[1] if len(argv) > 1 else os.path.join(HERE, 'renders')
os.makedirs(OUT, exist_ok=True)
SAMPLES = int(os.environ.get('WT_SAMPLES', '80'))
PCT = int(os.environ.get('WT_PCT', '100'))
CASES = ['fighter', 'water'] if CASE == 'all' else [CASE]
TEX = tempfile.mkdtemp(prefix='ce12-tex-')
sc = bpy.context.scene

for nm in ('Cube', 'Light', 'Camera'):
    if nm in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[nm], do_unlink=True)


def _img(path, rgb, n=768):
    """Write a small 8-bit RGB PNG (no extra deps)."""
    try:
        from PIL import Image
        Image.fromarray(rgb).save(path)
        return path
    except Exception:
        pass
    import struct, zlib
    h, w = rgb.shape[:2]
    raw = b''.join(b'\x00' + rgb[i].tobytes() for i in range(h))

    def chunk(tag, data):
        return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)

    ihdr = struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)
    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', ihdr) + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))
    return path


def make_textures():
    rng = np.random.default_rng(4)

    def fbm(h, w, octaves=5, seed=0):
        y, x = np.mgrid[0:h, 0:w].astype(float)
        v = np.zeros((h, w))
        a, s = 1.0, 4.0
        tot = 0.0
        for o in range(octaves):
            # value noise via bilinear upsample of a coarse grid
            gh, gw = max(int(h / s), 2), max(int(w / s), 2)
            g = rng.random((gh, gw))
            yi = (y * (gh - 1) / (h - 1))
            xi = (x * (gw - 1) / (w - 1))
            y0, x0 = np.floor(yi).astype(int), np.floor(xi).astype(int)
            y1, x1 = np.clip(y0 + 1, 0, gh - 1), np.clip(x0 + 1, 0, gw - 1)
            fy, fx = yi - y0, xi - x0
            n = (g[y0, x0] * (1 - fx) + g[y0, x1] * fx) * (1 - fy) + (g[y1, x0] * (1 - fx) + g[y1, x1] * fx) * fy
            v += a * n
            tot += a
            a *= 0.5
            s *= 0.5
        return v / tot

    n = 768
    c = fbm(n, n, seed=1)
    # concrete / asphalt floor
    al = np.clip(0.12 + 0.18 * c + 0.04 * fbm(n, n, seed=2), 0, 1)
    concrete = np.stack([al * 0.95, al * 0.97, al * 1.0], -1)
    concrete = (concrete * 255).astype(np.uint8)
    # rock
    r = fbm(n, n, seed=7)
    rock = np.stack([0.18 + 0.22 * r, 0.15 + 0.18 * r, 0.12 + 0.14 * r], -1)
    veins = (fbm(n, n, seed=9) > 0.62).astype(float) * 0.08
    rock = np.clip(rock + veins[..., None], 0, 1)
    rock = (rock * 255).astype(np.uint8)
    # brushed metal (cool)
    m = 0.35 + 0.25 * fbm(n, n, seed=11)
    streaks = fbm(n, 16, seed=13)
    streaks = np.repeat(streaks, n // 16 + 1, axis=1)[:, :n]
    metal = np.stack([m * 0.55 + 0.08 * streaks, m * 0.62 + 0.08 * streaks, m * 0.70 + 0.10 * streaks], -1)
    metal = np.clip(metal, 0, 1)
    metal = (metal * 255).astype(np.uint8)
    return dict(concrete=_img(os.path.join(TEX, 'concrete.png'), concrete),
                rock=_img(os.path.join(TEX, 'rock.png'), rock),
                metal=_img(os.path.join(TEX, 'metal.png'), metal))


TEXS = make_textures()


def look_at(o, target):
    o.rotation_euler = (Vector(target) - o.location).to_track_quat('-Z', 'Y').to_euler()


def camera(loc, target, lens, name="Cam"):
    cam = bpy.data.cameras.new(name)
    co = bpy.data.objects.new(name, cam)
    sc.collection.objects.link(co)
    co.location = loc
    look_at(co, target)
    cam.lens = lens
    cam.clip_start = 0.02
    cam.dof.use_dof = False
    sc.camera = co
    return co


def tex_material(name, image, *, rough=0.55, metal=0.0, scale=4.0, spec=0.4):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial'); out.location = (500, 0)
    pr = nt.nodes.new('ShaderNodeBsdfPrincipled'); pr.location = (200, 0)
    pr.inputs['Roughness'].default_value = rough
    if 'Metallic' in pr.inputs:
        pr.inputs['Metallic'].default_value = metal
    img = nt.nodes.new('ShaderNodeTexImage'); img.location = (-200, 0)
    img.image = bpy.data.images.load(image)
    img.interpolation = 'Smart'
    tc = nt.nodes.new('ShaderNodeTexCoord'); tc.location = (-600, 0)
    mp = nt.nodes.new('ShaderNodeMapping'); mp.location = (-420, 0)
    mp.inputs['Scale'].default_value = (scale, scale, scale)
    nt.links.new(tc.outputs['Object'], mp.inputs['Vector'])
    nt.links.new(mp.outputs['Vector'], img.inputs['Vector'])
    nt.links.new(img.outputs['Color'], pr.inputs['Base Color'])
    nt.links.new(pr.outputs['BSDF'], out.inputs['Surface'])
    return m


def foam_material(name="Tessera Whitewater"):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    pr = nt.nodes.new('ShaderNodeBsdfPrincipled')
    pr.inputs['Base Color'].default_value = (0.93, 0.95, 0.98, 1)
    pr.inputs['Roughness'].default_value = 0.42
    if 'Subsurface Weight' in pr.inputs:
        pr.inputs['Subsurface Weight'].default_value = 0.35
        if 'Subsurface Radius' in pr.inputs:
            try:
                pr.inputs['Subsurface Radius'].default_value = (0.4, 0.35, 0.3)
            except Exception:
                pass
    elif 'Subsurface' in pr.inputs:
        pr.inputs['Subsurface'].default_value = 0.35
    if 'Coat Weight' in pr.inputs:
        pr.inputs['Coat Weight'].default_value = 0.15
    bump = nt.nodes.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.35
    noise = nt.nodes.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = 28.0
    noise.inputs['Detail'].default_value = 8.0
    nt.links.new(noise.outputs['Fac'], bump.inputs['Height'])
    nt.links.new(bump.outputs['Normal'], pr.inputs['Normal'])
    nt.links.new(pr.outputs['BSDF'], out.inputs['Surface'])
    try:
        m.blend_method = 'OPAQUE'
    except Exception:
        pass
    return m


def studio(target=(0, 0, 0.4), *, bg=(0.018, 0.020, 0.024, 1), key=280, sun=3.2):
    w = bpy.data.worlds.new("ce12_world")
    w.use_nodes = True
    w.node_tree.nodes['Background'].inputs[0].default_value = bg
    w.node_tree.nodes['Background'].inputs[1].default_value = 0.55
    sc.world = w
    # sharp-ish sun for clean contact shadows
    sd = bpy.data.lights.new("sun", 'SUN')
    sd.energy, sd.angle, sd.color = sun, math.radians(8.0), (1.0, 0.97, 0.92)
    so = bpy.data.objects.new("sun", sd)
    sc.collection.objects.link(so)
    so.location = (3.0, -2.4, 5.0)
    look_at(so, target)
    for nm, loc, en, size, col in (
        ("key", (1.8, -3.0, 3.2), key, 1.4, (1.0, 0.98, 0.94)),
        ("rim", (-2.6, 1.8, 2.0), 110, 1.8, (0.65, 0.82, 1.0)),
        ("fill", (2.4, 2.2, 0.9), 90, 3.2, (0.95, 0.97, 1.0)),
    ):
        ld = bpy.data.lights.new(nm, 'AREA')
        ld.energy, ld.size, ld.color = en, size, col
        lo = bpy.data.objects.new(nm, ld)
        sc.collection.objects.link(lo)
        lo.location = loc
        look_at(lo, target)



def add_slipstream_mesh(body, v_inf, T_inf=288.15, iso=0.32):
    """Closed mesh of the denser slipstream (boundary layer + wake), cheap surface shader."""
    from arc_flow.core import cavity
    mw = body.matrix_world
    rot = mw.to_3x3()
    P = np.array([mw @ v.co for v in body.data.vertices], float)
    N = np.array([rot @ v.normal for v in body.data.vertices], float)
    N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
    Tw = None
    if 'tessera_temp_k' in body.data.attributes:
        Tw = np.empty(len(body.data.vertices), np.float32)
        body.data.attributes['tessera_temp_k'].data.foreach_get('value', Tw)
    lo = P.min(0) - np.array([0.35, 0.45, 0.35])
    hi = P.max(0) + np.array([1.8, 0.45, 0.45])
    field = FB.airflow_field(P, N, body_T=Tw, v_inf=v_inf, T_inf=T_inf,
                             bounds=(lo, hi), cell=0.05, delta=0.07)
    dens = np.asarray(field['dens_grid'], float)
    # free stream sits near 1.0; only mesh the denser boundary layer + wake
    occupied = dens > 0.2
    base = float(np.percentile(dens[occupied], 35)) if occupied.any() else 1.0
    excess = dens - base
    f = float(iso) - excess
    f[0, :, :] = np.maximum(f[0, :, :], 0.05)
    f[-1, :, :] = np.maximum(f[-1, :, :], 0.05)
    f[:, 0, :] = np.maximum(f[:, 0, :], 0.05)
    f[:, -1, :] = np.maximum(f[:, -1, :], 0.05)
    f[:, :, 0] = np.maximum(f[:, :, 0], 0.05)
    f[:, :, -1] = np.maximum(f[:, :, -1], 0.05)
    verts, tris = cavity.surface_nets(f, field['lo'], field['h'])
    if len(verts) == 0:
        print("slipstream mesh empty", flush=True)
        return None
    verts = FB.laplacian_smooth(verts, tris, iters=5, lam=0.4, volume=True)
    me = bpy.data.meshes.new("TesseraSlipstream")
    me.from_pydata(verts.tolist(), [], tris.tolist())
    me.polygons.foreach_set('use_smooth', np.ones(len(me.polygons), bool))
    # temperature tint as a color attribute
    tg = field.get('temp_grid')
    if tg is not None:
        h, origin = field['h'], field['lo']
        dims = np.array(tg.shape)
        vi = np.clip(np.round((verts - origin) / h).astype(int), 0, dims - 1)
        Tv = tg[vi[:, 0], vi[:, 1], vi[:, 2]]
        heat = np.clip((Tv - T_inf) / max(float(np.percentile(Tv, 98) - T_inf), 1.0), 0, 1)
        col = np.stack([0.55 + 0.45 * heat, 0.78 - 0.20 * heat, 1.0 - 0.85 * heat], -1)
        me.attributes.new('slip_color', 'FLOAT_COLOR', 'POINT')
        rgba = np.concatenate([col, np.ones((len(col), 1))], 1).astype(np.float32)
        me.attributes['slip_color'].data.foreach_set('color', rgba.ravel())
    o = bpy.data.objects.new("Tessera Slipstream Body", me)
    sc.collection.objects.link(o)
    m = bpy.data.materials.new("Tessera Slipstream")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    tr = nt.nodes.new('ShaderNodeBsdfTransparent')
    pr = nt.nodes.new('ShaderNodeBsdfPrincipled')
    pr.inputs['Base Color'].default_value = (0.62, 0.84, 1.0, 1)
    pr.inputs['Roughness'].default_value = 0.18
    if 'Emission Color' in pr.inputs:
        pr.inputs['Emission Color'].default_value = (0.75, 0.88, 1.0, 1)
        if 'Emission Strength' in pr.inputs:
            pr.inputs['Emission Strength'].default_value = 0.35
    attr = nt.nodes.new('ShaderNodeAttribute')
    attr.attribute_name = 'slip_color'
    nt.links.new(attr.outputs['Color'], pr.inputs['Base Color'])
    if 'Emission Color' in pr.inputs:
        nt.links.new(attr.outputs['Color'], pr.inputs['Emission Color'])
    mix = nt.nodes.new('ShaderNodeMixShader')
    lw = nt.nodes.new('ShaderNodeLayerWeight')
    lw.inputs['Blend'].default_value = 0.45
    mix.inputs['Fac'].default_value = 0.22
    nt.links.new(lw.outputs['Fresnel'], mix.inputs['Fac'])
    nt.links.new(tr.outputs['BSDF'], mix.inputs[1])
    nt.links.new(pr.outputs['BSDF'], mix.inputs[2])
    nt.links.new(mix.outputs['Shader'], out.inputs['Surface'])
    try:
        m.blend_method = 'HASHED'
        m.shadow_method = 'NONE'
    except Exception:
        pass
    me.materials.append(m)
    o.visible_shadow = False
    print("slipstream mesh: %d verts, %d tris, dens %.2f..%.2f base %.2f excess_max %.2f" % (len(verts), len(tris), float(dens.min()), float(dens.max()), base, float(excess.max())), flush=True)
    return o


def wisp_curve_material(name="Tessera Wisp Filament"):
    """Surface shader for curve-sourced slipstream tendrils (cheaper and sharper than a second volume)."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    pr = nt.nodes.new('ShaderNodeBsdfPrincipled')
    pr.inputs['Base Color'].default_value = (0.78, 0.90, 1.0, 1)
    pr.inputs['Roughness'].default_value = 0.22
    if 'Emission Color' in pr.inputs:
        pr.inputs['Emission Color'].default_value = (0.70, 0.90, 1.0, 1)
        if 'Emission Strength' in pr.inputs:
            pr.inputs['Emission Strength'].default_value = 2.4
    elif 'Emission' in pr.inputs:
        try:
            pr.inputs['Emission'].default_value = (0.70, 0.90, 1.0, 1)
        except Exception:
            pass
    nt.links.new(pr.outputs['BSDF'], out.inputs['Surface'])
    return m




def still_streamlines(body_P, body_N, v_inf, bounds, n_curves=28, steps=140):
    """Publication still: smooth slipstream filaments with a gentle helical wake."""
    P = np.asarray(body_P, float)
    N = np.asarray(body_N, float)
    lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    vi = np.asarray(v_inf, float).reshape(3)
    speed = float(np.linalg.norm(vi)) or 1.0
    vhat = vi / speed
    ds, dt = 0.024, 0.024 / speed
    rng = np.random.default_rng(5)
    if len(P) > 1800:
        pick = rng.choice(len(P), 1800, replace=False)
        P, N = P[pick], N[pick]
    ctr = P.mean(0)
    R0 = float(np.percentile(np.linalg.norm(P - ctr, axis=1), 90))
    # a tight rake just upstream of the silhouette
    seeds = []
    for y in np.linspace(-0.8 * R0, 0.8 * R0, 6):
        for z in np.linspace(-0.15 * R0, 0.70 * R0, 5):
            if (y / R0) ** 2 + (z / (0.9 * R0)) ** 2 < 1.05:
                seeds.append(np.array([float(P[:, 0].min()) - 0.16, ctr[1] + y, ctr[2] + z]))
    curves = []
    for s0 in seeds:
        q = s0.astype(float).copy()
        path = [q.copy()]
        for _ in range(int(steps)):
            d2 = ((q - P) ** 2).sum(-1)
            j = int(np.argmin(d2))
            nrm = N[j]
            signed = float((q - P[j]) @ nrm)
            vel = vi.copy()
            # smooth slip: cancel inward normal component near the wall, no hard projection
            vn = float(vel @ nrm)
            layer = np.exp(-max(signed, 0.0) / 0.11)
            if vn < 0.0:
                vel = vel - layer * vn * nrm
            if signed < 0.03:
                vel = vel - (vel @ nrm) * nrm
                q = q + nrm * (0.03 - signed) * 0.5
            along = float((q - ctr) @ vhat)
            radial = (q - ctr) - along * vhat
            rlen = float(np.linalg.norm(radial))
            if along > 0.0 and rlen > 1e-6 and rlen < 1.35 * R0:
                twist = np.cross(vhat, radial / rlen)
                swirl = 0.18 * speed * np.exp(-along / 1.4) * (0.6 + 0.4 * np.sin(along * 4.0 + s0[1] * 7.0))
                vel = vel * (0.70 + 0.30 * min(along / 1.0, 1.0)) + twist * swirl
            q = q + vel * dt
            path.append(q.copy())
            if q[0] > hi[0] + 0.1:
                break
        path = np.array(path)
        if len(path) < 12:
            continue
        # Laplacian smooth the polyline so tiny slip steps do not read as jaggies
        for _ in range(5):
            path[1:-1] = 0.25 * path[:-2] + 0.50 * path[1:-1] + 0.25 * path[2:]
        curves.append(path)
    return dict(curves=curves, n_source=sum(len(c) for c in curves))


def add_wisp_curves(body, v_inf):
    """Draw wispy_streamlines as beveled curve filaments (Houdini curve-source look)."""
    mw = body.matrix_world
    rot = mw.to_3x3()
    P = np.array([mw @ v.co for v in body.data.vertices], float)
    N = np.array([rot @ v.normal for v in body.data.vertices], float)
    N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
    lo = P.min(0) - np.array([0.35, 0.45, 0.35])
    hi = P.max(0) + np.array([1.8, 0.45, 0.45])
    wis = still_streamlines(P, N, v_inf, (lo, hi))
    cu = bpy.data.curves.new("TesseraWispCurves", 'CURVE')
    cu.dimensions = '3D'
    cu.bevel_depth = 0.0028
    cu.bevel_resolution = 3
    cu.fill_mode = 'FULL'
    cu.use_fill_caps = True
    n_ok = 0
    for path in wis.get('curves') or []:
        if len(path) < 8:
            continue
        if float(np.linalg.norm(path[-1] - path[0])) < 0.2:
            continue
        dmin = float(np.sqrt(((path[:, None, :] - P[None, :, :]) ** 2).sum(-1).min()))
        if dmin > 0.55:
            continue  # far free-stream rake line
        sp = cu.splines.new('POLY')
        sp.points.add(len(path) - 1)
        for i, pt in enumerate(path):
            # slight taper toward the tail of the filament
            rad = 1.0 - 0.45 * (i / max(len(path) - 1, 1))
            sp.points[i].co = (float(pt[0]), float(pt[1]), float(pt[2]), 1.0)
            try:
                sp.points[i].radius = rad
            except Exception:
                pass
        n_ok += 1
    o = bpy.data.objects.new("Tessera Airflow Wisp Curves", cu)
    sc.collection.objects.link(o)
    cu.materials.append(wisp_curve_material())
    o.visible_shadow = False
    print("wisp curves: %d filaments, %d source particles" % (n_ok, int(wis.get('n_source', 0))), flush=True)
    return o


def render_to(path, size=(1920, 1080), *, caustics=False):
    r = sc.render
    r.engine = 'CYCLES'
    sc.cycles.device = 'CPU'
    sc.cycles.samples = SAMPLES
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = 'OPENIMAGEDENOISE'
    sc.cycles.max_bounces = 8 if caustics else 6
    sc.cycles.transparent_max_bounces = 8 if caustics else 4
    sc.cycles.caustics_reflective = bool(caustics)
    sc.cycles.caustics_refractive = bool(caustics)
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.02
    if hasattr(sc.cycles, 'adaptive_min_samples'):
        sc.cycles.adaptive_min_samples = 2
    if hasattr(sc.cycles, 'volume_step_rate'):
        sc.cycles.volume_step_rate = 2.5
    if hasattr(sc.cycles, 'volume_max_steps'):
        sc.cycles.volume_max_steps = 256
    if hasattr(sc.cycles, 'volume_bounces'):
        sc.cycles.volume_bounces = 1
    if hasattr(sc.cycles, 'sample_clamp_indirect'):
        sc.cycles.sample_clamp_indirect = 10.0
    r.threads_mode = 'FIXED'
    r.threads = max(int(os.cpu_count() or 8), 1)
    if hasattr(sc.view_settings, 'view_transform'):
        # AgX in 4.2 when available, else Filmic
        try:
            sc.view_settings.view_transform = 'AgX'
        except TypeError:
            sc.view_settings.view_transform = 'Filmic'
    r.resolution_x, r.resolution_y, r.resolution_percentage = size[0], size[1], PCT
    r.image_settings.file_format = 'PNG'
    r.image_settings.color_mode = 'RGB'
    r.image_settings.compression = 12
    r.filepath = path
    r.film_transparent = False
    bpy.ops.render.render(write_still=True)
    print("rendered", path, flush=True)


def clear_scene():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for m in list(bpy.data.meshes):
        if m.users == 0:
            bpy.data.meshes.remove(m)


def _rock_mesh(rng, n=18):
    # displaced icosphere-like blob
    t = (1.0 + 5 ** 0.5) / 2.0
    V = np.array([(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t),
                  (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)], float)
    V /= np.linalg.norm(V, axis=1)[:, None]
    F = np.array([(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
                  (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
                  (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)])
    # one loop subdivision
    def sub(v, f):
        edge = {}
        mid = []
        def midp(a, b):
            k = (min(a, b), max(a, b))
            if k not in edge:
                edge[k] = len(v) + len(mid)
                mid.append(0.5 * (v[a] + v[b]))
            return edge[k]
        nf = []
        for a, b, c in f:
            ab, bc, ca = midp(a, b), midp(b, c), midp(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        nv = np.vstack([v, np.array(mid)])
        return nv, np.array(nf)
    V, F = sub(V, F)
    V, F = sub(V, F)
    # radial displacement
    nrm = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    V = V * (0.13 + 0.055 * rng.random(len(V))[:, None])
    V *= np.array([1.15, 0.95, 0.75])
    return V, F


if 'fighter' in CASES:
    clear_scene()
    WIND = float(os.environ.get('CE12_WIND', '480'))
    WT.load_fighter_example(sc, wind=WIND, with_thermal=True)
    air = bpy.data.objects["Fighter Airframe"]
    TM.apply_aero_heat(sc, air, v_inf=(WIND, 0.0, 0.0), steps=36, speedup=120.0)
    # Cycles volumes are slow and fragile on this CPU box; mesh the denser
    # boundary-layer / wake and draw curve-sourced filaments instead.
    for nm in ("Tessera Airflow Volume", "Tessera Airflow Wisps"):
        o = bpy.data.objects.get(nm)
        if o is not None:
            bpy.data.objects.remove(o, do_unlink=True)
    # A meshed envelope of the whole airflow field reads as a melted blob; the
    # Houdini-style still uses curve-sourced filaments around the heated airframe.
    add_wisp_curves(air, (WIND, 0.0, 0.0))
    print("render objects:",
          ", ".join("%s%s" % (o.name, " [hidden]" if o.hide_render else "")
                    for o in bpy.data.objects), flush=True)
    T = np.empty(len(air.data.attributes['tessera_temp_k'].data), np.float32)
    air.data.attributes['tessera_temp_k'].data.foreach_get('value', T)
    Q = np.empty(len(air.data.attributes['tessera_friction_heat'].data), np.float32)
    air.data.attributes['tessera_friction_heat'].data.foreach_get('value', Q)
    print("fighter wall T %.2f .. %.2f K, q max %.0f W/m2, wind %.0f m/s" % (T.min(), T.max(), Q.max(), WIND), flush=True)
    for o in list(bpy.data.objects):
        if o.name in ("Wind Tunnel", "Tunnel Air Volume", "Wind Emitter") or o.get('af_wt') in ("Wind Tunnel", "Tunnel Air Volume", "Wind Emitter"):
            o.hide_render = True
            o.hide_viewport = True
            try:
                bpy.data.objects.remove(o, do_unlink=True)
            except Exception:
                pass
    for fl in sc.arc_flow_fm.fluids:
        if fl.points_obj is not None:
            fl.points_obj.hide_render = True
    bpy.ops.mesh.primitive_plane_add(size=80, location=(0.4, 0, -0.22))
    floor = bpy.context.object
    floor.name = "Studio_Floor"
    floor.data.materials.append(tex_material("concrete", TEXS['concrete'], rough=0.62, scale=3.5))
    # a faint metal sheen already lives in the aero overlay; leave it
    studio(Vector(air.matrix_world.translation), key=300, sun=2.6)
    ctr = Vector(air.matrix_world.translation)
    camera((2.20, -2.40, 1.18), (ctr.x + 0.30, 0.0, ctr.z + 0.04), 52)
    render_to(os.path.join(OUT, 'windtunnel_fighter_friction_heat.png'), caustics=False)

if 'water' in CASES:
    clear_scene()
    rng = np.random.default_rng(6)
    dx = 0.014
    def cube(lo, hi):
        g = [np.arange(a, b + 1e-12, dx) for a, b in zip(lo, hi)]
        X, Y, Z = np.meshgrid(*g, indexing='ij')
        return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    # a basin of packed water around a rock
    water_P = cube((-0.22, -0.16, 0.00), (0.22, 0.16, 0.11))
    rv, rf = _rock_mesh(rng)
    # sit the boulder in a corner of the basin so the solid-body water stays a clean volume
    rv = rv + np.array([0.16, 0.11, 0.00])
    vel = np.zeros_like(water_P)
    d2 = ((water_P[:, None, :] - rv[None, :, :]) ** 2).sum(-1).min(1)
    vel[:, 0] = 0.55
    vel[d2 < 0.05 ** 2, 0] = 1.1
    vel[d2 < 0.05 ** 2, 2] = 0.35
    ww = FB.whitewater(water_P, vel, dx=dx, body_P=rv,
                       body_N=(rv - rv.mean(0)) / np.maximum(np.linalg.norm(rv - rv.mean(0), axis=1, keepdims=True), 1e-9),
                       speed_ref=1.4)
    col = np.tile((0.22, 0.50, 0.88), (len(water_P), 1))
    verts, tris, vc, nrm, info = FB.solid_body_surface(water_P, colors=col, dx=dx, radius=2.3, iso=0.30, tension=0.5, smooth=6)
    fv = FB.foam_on_surface(verts, water_P, ww['foam'], dx)
    print("water surface", info, "foam verts", int((fv > 0.35).sum()), "spray", int(ww['spray'].sum()), flush=True)
    me = bpy.data.meshes.new("TesseraWaterBody")
    me.from_pydata(verts.tolist(), [], tris.tolist())
    me.polygons.foreach_set('use_smooth', np.ones(len(me.polygons), bool))
    if 'tessera_foam' not in me.attributes:
        me.attributes.new('tessera_foam', 'FLOAT', 'POINT')
    me.attributes['tessera_foam'].data.foreach_set('value', fv.astype(np.float32))
    water = bpy.data.objects.new("TesseraWaterBody", me)
    sc.collection.objects.link(water)
    try:
        from arc_flow import shader_presets as SP
        mat = SP.build('WATER', name="Tessera Solid Water", reuse=False,
                       Color=(0.42, 0.70, 1.0), Absorption_density=3.8)
    except Exception:
        mat = bpy.data.materials.new("Tessera Solid Water")
        mat.use_nodes = True
        b = mat.node_tree.nodes['Principled BSDF']
        b.inputs['Base Color'].default_value = (0.42, 0.70, 1.0, 1)
        b.inputs['Roughness'].default_value = 0.05
        for nm, val in (('Transmission Weight', 0.97), ('IOR', 1.333), ('Alpha', 0.95)):
            if nm in b.inputs:
                b.inputs[nm].default_value = val
    me.materials.append(mat)
    # soften the glass a little and mix in a creamy foam cap from tessera_foam
    try:
        nt = mat.node_tree
        pr = next(n for n in nt.nodes if n.bl_idname == 'ShaderNodeBsdfPrincipled')
        pr.inputs['Roughness'].default_value = 0.12
        attr = nt.nodes.new('ShaderNodeAttribute')
        attr.attribute_name = 'tessera_foam'
        attr.location = (-360, 180)
        foam_bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
        foam_bsdf.location = (40, 220)
        foam_bsdf.inputs['Base Color'].default_value = (0.93, 0.95, 0.98, 1)
        foam_bsdf.inputs['Roughness'].default_value = 0.45
        if 'Subsurface Weight' in foam_bsdf.inputs:
            foam_bsdf.inputs['Subsurface Weight'].default_value = 0.25
        mix = nt.nodes.new('ShaderNodeMixShader')
        mix.location = (320, 80)
        out = next(n for n in nt.nodes if n.bl_idname == 'ShaderNodeOutputMaterial')
        # reroute water BSDF through the mix
        for link in list(nt.links):
            if link.to_node == out and link.to_socket.name == 'Surface':
                nt.links.remove(link)
        cr = nt.nodes.new('ShaderNodeValToRGB')
        cr.location = (-140, 180)
        cr.color_ramp.elements[0].position = 0.28
        cr.color_ramp.elements[0].color = (0, 0, 0, 1)
        cr.color_ramp.elements[1].position = 0.75
        cr.color_ramp.elements[1].color = (1, 1, 1, 1)
        nt.links.new(attr.outputs['Fac'], cr.inputs['Fac'])
        nt.links.new(cr.outputs['Color'], mix.inputs['Fac'])
        nt.links.new(pr.outputs['BSDF'], mix.inputs[1])
        nt.links.new(foam_bsdf.outputs['BSDF'], mix.inputs[2])
        nt.links.new(mix.outputs['Shader'], out.inputs['Surface'])
    except Exception as ex:
        print("foam mix", ex, flush=True)
    # foam cap: a slightly inflated copy of the high-foam vertices (a thin surface of bubbles)
    foam_idx = np.flatnonzero(fv > 0.35)
    if len(foam_idx) > 8:
        # splat foam as a small icosphere cloud at those verts, plus spray particles
        foam_pts = verts[foam_idx] + 0.004 * nrm[foam_idx]
        spray_pts = water_P[ww['spray']] + np.array([0, 0, 0.012]) if ww['spray'].any() else np.zeros((0, 3))
        pts = np.vstack([foam_pts, spray_pts]) if len(spray_pts) else foam_pts
        fme = bpy.data.meshes.new("TesseraWhitewater")
        fme.from_pydata(pts.tolist(), [], [])
        fo = bpy.data.objects.new("TesseraWhitewater", fme)
        sc.collection.objects.link(fo)
        # instance tiny foam balls
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=0.006, location=(0, 0, -10))
        ball = bpy.context.object
        ball.name = "FoamBall"
        ball.data.materials.append(foam_material())
        ball.hide_render = True
        fo.instance_type = 'VERTS'
        # GN instance on points is more reliable
        ng = bpy.data.node_groups.new("CE12_FoamInst", 'GeometryNodeTree')
        import arc_flow.nodeutil as NU
        try:
            NU.add_socket(ng, "Geometry", 'NodeSocketGeometry', 'INPUT')
            NU.add_socket(ng, "Geometry", 'NodeSocketGeometry', 'OUTPUT')
        except Exception:
            pass
        # build a minimal group via nodes
        ng.nodes.clear()
        if hasattr(ng, 'interface'):
            # 4.x
            pass
        gi = ng.nodes.new('NodeGroupInput'); gi.location = (-300, 0)
        go = ng.nodes.new('NodeGroupOutput'); go.location = (400, 0)
        # ensure sockets
        if not ng.interface.items_tree if hasattr(ng, 'interface') else True:
            pass
        try:
            if hasattr(ng, 'interface') and not any(getattr(i, 'name', '') == 'Geometry' for i in ng.interface.items_tree):
                ng.interface.new_socket(name='Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
                ng.interface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
        except Exception:
            pass
        oi = ng.nodes.new('GeometryNodeObjectInfo'); oi.location = (-80, -160)
        oi.transform_space = 'ORIGINAL'
        try:
            oi.inputs['As Instance'].default_value = True
        except Exception:
            pass
        # set object later
        iop = ng.nodes.new('GeometryNodeInstanceOnPoints'); iop.location = (80, 0)
        # scale by foam
        try:
            ng.links.new(gi.outputs[0], iop.inputs['Points'])
            ng.links.new(oi.outputs['Geometry'], iop.inputs['Instance'])
            ng.links.new(iop.outputs['Instances'], go.inputs[0])
        except Exception as ex:
            print("foam GN link", ex, flush=True)
        md = fo.modifiers.new("Foam", 'NODES')
        md.node_group = ng
        # bind the ball
        # Object Info default: set via a driver-less pointer if the socket exists on the modifier
        try:
            oi.inputs['Object'].default_value = ball
        except Exception:
            pass
        ball.parent = fo
        ball.hide_set(True)
    # rock
    rme = bpy.data.meshes.new("CollisionRock")
    rme.from_pydata(rv.tolist(), [], rf.tolist())
    rme.polygons.foreach_set('use_smooth', np.ones(len(rme.polygons), bool))
    rock = bpy.data.objects.new("CollisionRock", rme)
    sc.collection.objects.link(rock)
    rm = tex_material("rock", TEXS['rock'], rough=0.55, scale=2.2)
    # wet look near the waterline
    nt = rm.node_tree
    pr = next(n for n in nt.nodes if n.bl_idname == 'ShaderNodeBsdfPrincipled')
    pr.inputs['Roughness'].default_value = 0.28
    if 'Coat Weight' in pr.inputs:
        pr.inputs['Coat Weight'].default_value = 0.45
        if 'Coat Roughness' in pr.inputs:
            pr.inputs['Coat Roughness'].default_value = 0.08
    rme.materials.append(rm)
    bpy.ops.mesh.primitive_plane_add(size=6, location=(0, 0, -0.002))
    floor = bpy.context.object
    floor.data.materials.append(tex_material("wfloor", TEXS['concrete'], rough=0.7, scale=2.0))
    # a dark basin rim
    studio((0.02, 0, 0.07), bg=(0.16, 0.18, 0.20, 1), key=160, sun=2.2)
    for nm in ("key", "rim", "fill"):
        o = bpy.data.objects.get(nm)
        if o is not None and o.data and hasattr(o.data, 'size'):
            o.data.size = max(float(o.data.size) * 1.8, 2.4)
    camera((0.72, -0.95, 0.48), (0.02, 0.0, 0.055), 48, name="WaterCam")
    render_to(os.path.join(OUT, 'fluid_solid_body_water.png'), caustics=True)

# cleanup generated textures (the PNGs are packed into the render, not needed on disk)
import shutil
try:
    shutil.rmtree(TEX, ignore_errors=True)
except Exception:
    pass
print("done", flush=True)
