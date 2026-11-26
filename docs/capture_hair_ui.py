"""Stepped UI screen recording of Tessera Hair collision (real Blender GUI, viewport in Rendered / EEVEE).

    xvfb-run -a -s "-screen 0 1600x900x24" blender --gpu-backend opengl --factory-startup \
        --window-geometry 0 0 1600 900 --python docs/capture_hair_ui.py -- OUTDIR

Bakes the long-hair and short-hair swat (24 fps sim, the extension's own solver), then walks the timeline: each output
step sets the frame, waits for the 3D view to redraw and grabs the X11 screen with ffmpeg x11grab. Assemble with ffmpeg.
Env: HC_UI_STRIDE (sim frames per grab, default 3), HC_UI_SHADING (RENDERED|MATERIAL|SOLID), HC_UI_MAX (max grabs).
"""
import importlib.util
import json
import os
import subprocess
import sys
import time

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("hcs", os.path.join(HERE, "render_hair_collision.py"))
hcs = importlib.util.module_from_spec(spec)
sys.modules["hcs"] = hcs
spec.loader.exec_module(hcs)
sc = bpy.context.scene

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
OUT = argv[0] if argv else '/tmp/hair_ui'
os.makedirs(OUT, exist_ok=True)
STRIDE = int(os.environ.get('HC_UI_STRIDE', '3'))
SHADING = os.environ.get('HC_UI_SHADING', 'RENDERED')
MAXG = int(os.environ.get('HC_UI_MAX', '100000'))
W, H = int(os.environ.get('HC_UI_W', '1600')), int(os.environ.get('HC_UI_H', '900'))
log = lambda *a: print(*a, flush=True)


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
    for _ in range(2):                         # dismiss the splash: Escape + click away from it, then park the mouse
        mouse_move(700, 880); yield 0.3
        key(0xff1b); yield 0.5
        mouse_move(400, 300); yield 0.2
        mouse_button(1, True); mouse_button(1, False); yield 0.5
    mouse_move(700, 880)
    yield 1.0
    S = hcs.build_scene(('long', 'short'))
    log("scene built %.1fs" % (time.time() - t0))
    plan = [('long', hcs.glove_track(), 18.0), ('short', hcs.short_track(), 12.0)]
    caches = {}
    for st, track, secs in plan:
        caches[st] = hcs.bake_style(S, st, secs, track, progress=lambda s, f, n, dt: log("  sim %s %d/%d %.0fs" % (s, f, n, dt)))
    log("baked %.1fs" % (time.time() - t0))
    # --- UI
    sc.frame_start, sc.frame_end = 1, 18 * 24 + 12 * 24
    sc.render.fps = 24
    sc.arc_flow.ui_tab = 'HAIR'
    area = next(a for a in win.screen.areas if a.type == 'VIEW_3D')
    sp = area.spaces.active
    sp.show_region_ui = True
    sp.show_region_toolbar = False
    sp.overlay.show_stats = False
    sp.overlay.show_extras = False
    sp.show_gizmo = False
    sp.clip_start = 0.01
    if SHADING == 'RENDERED':
        sp.shading.type = 'RENDERED'
        sc.render.engine = 'BLENDER_EEVEE' if 'BLENDER_EEVEE' in [e.identifier for e in sc.render.bl_rna.properties['engine'].enum_items] else 'BLENDER_EEVEE_NEXT'
    elif SHADING == 'MATERIAL':
        sp.shading.type = 'MATERIAL'
    else:
        sp.shading.type = 'SOLID'
    ee = sc.eevee
    for k, v in (('use_shadows', True), ('taa_samples', 4), ('taa_render_samples', 4), ('use_raytracing', False),
                 ('use_volumetric_shadows', False), ('use_gtao', False), ('use_bloom', False), ('use_shadow_jitter_viewport', False)):
        if hasattr(ee, k):
            try:
                setattr(ee, k, v)
            except Exception:
                pass
    # view the bust from the camera-ish front 3/4
    r3d = sp.region_3d
    from mathutils import Vector, Euler
    r3d.view_perspective = 'PERSP'
    r3d.view_location = (0.42, 0.0, 1.42)
    r3d.view_distance = 2.5
    r3d.view_rotation = Euler((1.5708 - 0.04, 0.0, 0.0), 'XYZ').to_quaternion()
    sp.lens = 50
    for k in ('show_floor', 'show_axis_x', 'show_axis_y', 'show_cursor', 'show_text', 'show_object_origins', 'show_relationship_lines'):
        if hasattr(sp.overlay, k):
            setattr(sp.overlay, k, False)
    sc.camera = bpy.data.objects['TS_Cam']
    for r in area.regions:
        if r.type == 'UI':
            try:
                r.active_panel_category = 'Tessera'
            except Exception:
                pass
    state = dict(draws=0)

    def ui_region():
        a = next(a for a in win.screen.areas if a.type == 'VIEW_3D')
        return next(r for r in a.regions if r.type == 'UI')

    def set_tab():
        try:
            ui_region().active_panel_category = 'Tessera'
        except Exception as e:
            log("tab:", e)
    set_tab()

    def on_draw():
        state['draws'] += 1
    bpy.types.SpaceView3D.draw_handler_add(on_draw, (), 'WINDOW', 'POST_PIXEL')

    def redraw():
        for a in win.screen.areas:
            a.tag_redraw()

    # wait for the first EEVEE shader compile, widen the sidebar (drag its edge), select the Tessera tab
    for _ in range(25):
        redraw()
        yield 0.4
    set_tab()
    r = ui_region()
    edge = r.x - 3 if r.x > 0 else 1200
    mouse_move(edge, 500); yield 0.4
    mouse_button(1, True); yield 0.3
    x = edge
    while x > 1030:
        x -= 30
        mouse_move(x, 500); yield 0.12
    mouse_button(1, False); yield 0.4
    mouse_move(700, 880); yield 0.4
    set_tab()
    log("sidebar x=%d w=%d" % (ui_region().x, ui_region().width))
    manifest = []
    total = len(caches['long']) + len(caches['short'])
    frames = list(range(0, total, STRIDE))[:MAXG]
    n = 0
    tg = time.time()
    for k, fr in enumerate(frames):
        style, f = ('long', fr) if fr < len(caches['long']) else ('short', fr - len(caches['long']))
        track = plan[0][1] if style == 'long' else plan[1][1]
        sc.arc_flow_hair.hair = S['hairs'][style]
        for st, o in S['hairs'].items():
            o.hide_viewport = o.hide_render = (st != style)
            for sk in bpy.data.objects:
                if sk.name == o.name + ".001":
                    pass
        S['glove'].matrix_world = hcs.eval_track(track, f / hcs.FPS)
        hcs.set_frame_pose(S, style, caches[style], f)
        sc.frame_current = fr + 1
        sc.arc_flow_hair.status = "Cache playback  %s hair   t = %.1f s" % (style, fr / hcs.FPS)
        d0 = state['draws']
        set_tab()
        redraw()
        t1 = time.time()
        while state['draws'] <= d0 and time.time() - t1 < 60:
            yield 0.05
            redraw()
        yield 0.15
        fn = os.path.join(OUT, 'f%05d.jpg' % n)
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'x11grab', '-draw_mouse', '0', '-video_size', '%dx%d' % (W, H),
                        '-i', os.environ['DISPLAY'], '-frames:v', '1', '-q:v', '3', fn], check=True)
        manifest.append(dict(file=os.path.basename(fn), sim_frame=fr, style=style))
        n += 1
        if k % 5 == 0:
            log("grab %d/%d  %.1fs/grab" % (k, len(frames), (time.time() - tg) / (k + 1)))
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
