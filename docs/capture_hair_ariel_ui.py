"""Stepped UI screen recording of Tessera Hair on the game's Ariel character (real Blender GUI, Material Preview / EEVEE viewport with scene
lights + shadows, sidebar Hair panel visible).  Plays back frames baked by render_hair_ariel.py (bake) and grabs the X11 screen.

    xvfb-run -a -s "-screen 0 1600x900x24" blender --gpu-backend opengl --factory-startup --window-geometry 0 0 1600 900 \\
        --python docs/capture_hair_ariel_ui.py -- OUTDIR GAME_DIR ON_PREFIX OFF_PREFIX

Video plan (30 s at 12 grabs / s of scene time): 0-14.75 s collision ON, collision bodies hidden then shown as a translucent overlay;
14.75-18.75 s the same motion with Collision OFF (panel switch flipped, bodies still shown); 18.75-22.75 s ON again; then the rest of the clip.
Env: HA_TIMES (comma list of video times for a quick look instead of the full run), HA_STRIDE (grabs every N sim frames, default 2), HA_MAX (max grabs), HA_W / HA_H, HA_SHADING (MATERIAL|RENDERED)."""
import json
import os
import subprocess
import sys
import time

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ['ARIEL_RUN'] = ''
import render_hair_ariel as R
from arc_flow import hair_ops as HO

sc = bpy.context.scene
argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
OUT, GAME, ON_P, OFF_P = argv[:4]
os.makedirs(OUT, exist_ok=True)
STRIDE = int(os.environ.get('HA_STRIDE', '2'))
SHADING = os.environ.get('HA_SHADING', 'MATERIAL')
MAXG = int(os.environ.get('HA_MAX', '100000'))
W, H = int(os.environ.get('HA_W', '1600')), int(os.environ.get('HA_H', '900'))
FPS = 24
OFF_T0, OFF_T1 = 14.75, 18.75          # the 'Acting_Sad' droop: the hair falls forward over the shoulders and chest
log = lambda *a: print(*a, flush=True)


def source_for(tv):
    """Video time (s) -> (cache name, scene time (s), caption, show_bodies, collision_on)."""
    if tv < 9.0:
        return 'on', tv, 'Collision ON - Ariel, real strand hair (5,665 strands)', False, True
    if tv < OFF_T0:
        return 'on', tv, 'Collision ON - collision bodies shown (capsules + head field)', True, True
    if tv < OFF_T1:
        return 'off', tv, 'Collision OFF - same motion, hair passes through the body', True, False
    if tv < OFF_T1 + (OFF_T1 - OFF_T0):
        return 'on', tv - (OFF_T1 - OFF_T0), 'Collision ON - same motion, hair rests on shoulders and chest', True, True
    return 'on', tv - (OFF_T1 - OFF_T0), 'Collision ON', False, True


def run():
    win = bpy.context.window_manager.windows[0]
    t0 = time.time()
    import ctypes
    x11 = ctypes.CDLL('libX11.so.6')
    xtst = ctypes.CDLL('libXtst.so.6')
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XFlush.argtypes = [ctypes.c_void_p]
    xtst.XTestFakeMotionEvent.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_ulong]
    xtst.XTestFakeButtonEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]
    xdpy = x11.XOpenDisplay(None)

    def mouse_move(x, y):
        xtst.XTestFakeMotionEvent(xdpy, -1, int(x), int(y), 0)
        x11.XFlush(xdpy)

    def mouse_button(b, down):
        xtst.XTestFakeButtonEvent(xdpy, b, 1 if down else 0, 0)
        x11.XFlush(xdpy)
    bpy.context.preferences.view.show_splash = False
    x11.XKeysymToKeycode.restype = ctypes.c_ubyte
    x11.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    xtst.XTestFakeKeyEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]

    def key(sym):
        kc = x11.XKeysymToKeycode(xdpy, sym)
        xtst.XTestFakeKeyEvent(xdpy, kc, 1, 0); x11.XFlush(xdpy)
        xtst.XTestFakeKeyEvent(xdpy, kc, 0, 0); x11.XFlush(xdpy)
    for _ in range(2):
        mouse_move(700, 880); yield 0.3
        key(0xff1b); yield 0.5
        mouse_move(400, 300); yield 0.2
        mouse_button(1, True); mouse_button(1, False); yield 0.5
    mouse_move(700, 880)
    yield 1.0
    S = R.build_scene(GAME)
    log("scene built %.1fs" % (time.time() - t0))
    hp = sc.arc_flow_hair
    for o in S['colliders']:
        HO.mark_proxy(o)
    HO.set_collider_overlay(sc, False)
    caches = {'on': np.load(ON_P + '.npy', mmap_mode='r'), 'off': np.load(OFF_P + '.npy', mmap_mode='r')}
    metrics = {'on': json.load(open(ON_P + '.json')), 'off': json.load(open(OFF_P + '.json'))}
    log("caches", {k: v.shape for k, v in caches.items()})
    # the panel shows the solver's last contact count: during playback it carries the count measured in the bake for that frame
    import types
    HO.RT.solver = types.SimpleNamespace(last_contacts=0, n_points=int(caches['on'].shape[1]))
    # --- UI
    sc.frame_start, sc.frame_end = 1, 1 + R.SECONDS * FPS
    sc.arc_flow.ui_tab = 'HAIR'
    area = next(a for a in win.screen.areas if a.type == 'VIEW_3D')
    sp = area.spaces.active
    sp.show_region_ui = True
    sp.show_region_toolbar = False
    sp.overlay.show_stats = False
    sp.overlay.show_extras = False
    sp.show_gizmo = False
    sp.clip_start = 0.01
    sp.clip_end = 100
    sp.shading.type = 'RENDERED' if SHADING == 'RENDERED' else 'MATERIAL'
    if SHADING != 'RENDERED':
        sp.shading.use_scene_lights = True
        sp.shading.use_scene_world = True
    try:
        sc.render.engine = 'BLENDER_EEVEE'
    except Exception:
        sc.render.engine = 'BLENDER_EEVEE_NEXT'
    ee = sc.eevee
    for k, v in (('use_shadows', True), ('taa_samples', 8), ('taa_render_samples', 4), ('use_raytracing', False),
                 ('use_shadow_jitter_viewport', False), ('shadow_ray_count', 1), ('shadow_step_count', 6), ('use_volumetric_shadows', False),
                 ('use_gtao', False), ('use_bloom', False)):
        if hasattr(ee, k):
            try:
                setattr(ee, k, v)
            except Exception:
                pass
    from mathutils import Vector
    r3d = sp.region_3d
    r3d.view_perspective = 'PERSP'
    tgt = Vector((0.0, 0.10, 1.22))
    eye = Vector((2.05, -1.25, 1.50))
    r3d.view_location = tgt
    r3d.view_distance = (eye - tgt).length
    r3d.view_rotation = (tgt - eye).to_track_quat('-Z', 'Y')
    sp.lens = 55
    for k in ('show_floor', 'show_axis_x', 'show_axis_y', 'show_cursor', 'show_text', 'show_object_origins', 'show_relationship_lines',
              'show_bones', 'show_outline_selected'):
        if hasattr(sp.overlay, k):
            try:
                setattr(sp.overlay, k, False)
            except Exception:
                pass
    state = dict(draws=0, caption='')

    def ui_region():
        a = next(a for a in win.screen.areas if a.type == 'VIEW_3D')
        return next(r for r in a.regions if r.type == 'UI')

    def set_tab():
        try:
            ui_region().active_panel_category = 'Tessera'
        except Exception as e:
            log("tab:", e)
    set_tab()
    import blf

    def on_draw():
        state['draws'] += 1
        if state['caption']:
            blf.size(0, 26)
            blf.color(0, 1.0, 1.0, 1.0, 1.0)
            blf.enable(0, blf.SHADOW)
            blf.shadow(0, 3, 0.0, 0.0, 0.0, 1.0)
            blf.position(0, 28, 850, 0)
            blf.draw(0, state['caption'])
    bpy.types.SpaceView3D.draw_handler_add(on_draw, (), 'WINDOW', 'POST_PIXEL')

    def redraw():
        for a in win.screen.areas:
            a.tag_redraw()
    for _ in range(25):
        redraw()
        yield 0.4
    set_tab()
    for dx in (1, 0, -2, 3, -4, 6):
        if ui_region().width > 420:
            break
        edge = ui_region().x + dx
        mouse_move(edge, 420); yield 0.5
        mouse_button(1, True); yield 0.4
        x = edge
        while x > 700:
            x -= 25
            mouse_move(x, 420); yield 0.15
        mouse_button(1, False); yield 0.5
        mouse_move(700, 880); yield 0.5
        log("drag try dx=%d -> x=%d w=%d" % (dx, ui_region().x, ui_region().width))
    set_tab()
    log("sidebar x=%d w=%d" % (ui_region().x, ui_region().width))
    n_total = int(round(R.SECONDS * FPS / STRIDE))
    times = [float(x) for x in os.environ['HA_TIMES'].split(',')] if os.environ.get('HA_TIMES') else None
    manifest = []
    tg = time.time()
    last = dict(show=None, coll=None)
    plan = times if times is not None else [gi * STRIDE / FPS for gi in range(n_total)]
    for gi, tv in enumerate(plan[:MAXG]):
        name, ts, caption, show, coll = source_for(tv)
        frames = caches[name]
        k = min(int(round(ts * FPS / STRIDE)), len(frames) - 1)
        sim_f = 1 + k * STRIDE
        if last['coll'] != coll:
            hp.use_colliders = coll
            for c in hp.colliders:
                c.enabled = coll
            last['coll'] = coll
        if last['show'] != show:
            hp.show_colliders = show
            last['show'] = show
        sc.frame_set(sim_f)
        bpy.context.view_layer.update()
        HO.write_curves(S['hair'], np.asarray(frames[k], dtype=np.float64))
        mm = metrics[name][min(sim_f - 1, len(metrics[name]) - 1)]
        if HO.RT.solver is not None:
            HO.RT.solver.last_contacts = int(mm.get('contacts', 0)) if coll else 0
        hp.status = "Cache playback  t = %.1f s   %s" % (ts, 'collision ON' if coll else 'collision OFF')
        state['caption'] = caption
        d0 = state['draws']
        set_tab()
        redraw()
        t1 = time.time()
        while state['draws'] <= d0 and time.time() - t1 < 90:
            yield 0.05
            redraw()
        yield 0.2
        fn = os.path.join(OUT, 'f%05d.jpg' % gi)
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'x11grab', '-draw_mouse', '0', '-video_size', '%dx%d' % (W, H),
                        '-i', os.environ['DISPLAY'], '-frames:v', '1', '-q:v', '3', fn], check=True)
        manifest.append(dict(file=os.path.basename(fn), video_t=tv, source=name, scene_t=ts, collision=coll, bodies=show))
        if gi % 5 == 0:
            log("grab %d/%d  %.1fs/grab" % (gi, n_total, (time.time() - tg) / (gi + 1)))
            json.dump(manifest, open(os.path.join(OUT, 'manifest.json'), 'w'))
    json.dump(manifest, open(os.path.join(OUT, 'manifest.json'), 'w'))
    log("capture done")
    bpy.ops.wm.quit_blender()
    yield None


def start():
    gen = run()

    def tick():
        try:
            v = next(gen)
        except StopIteration:
            return None
        except Exception:
            import traceback
            traceback.print_exc()
            bpy.ops.wm.quit_blender()
            return None
        return 0.05 if v is None else v
    bpy.app.timers.register(tick, first_interval=2.0)


start()
