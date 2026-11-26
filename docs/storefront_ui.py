"""REAL Blender UI capture of the storefront group simulation (Layout workspace, N-panel Tessera Cloth tab).

    xvfb-run -a -s "-screen 0 1920x1080x24" blender --gpu-backend opengl docs/storefront_ui.py -- WORKDIR [step]

Opens WORKDIR/storefront_group.blend in the real Blender window (Material Preview with lights, shadows, EEVEE), loads
WORKDIR/sf_cache.npz (the solved group, Self-collision checked), and for every solver frame sets the timeline frame
(playhead moves, fan jet / blades animate through the scene's own playback script), writes the cached positions into the
TS_Dress / TS_DressSash objects, waits for a real redraw and grabs the X11 screen.  Frames are assembled at the scene
rate (24 fps) with NO slow-motion padding.  The capture is stepped (the 6.5k-vertex solve is not real time on this box).
Env: UI_W/UI_H, UI_STEP (1), UI_FIRST/UI_LAST, UI_SHADING (MATERIAL), UI_OUT, UI_TAB_XY, UI_SIDEBAR_X, UI_SCROLL.
"""
import json
import math
import os
import subprocess
import sys
import time

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
WORK = os.path.abspath(argv[0] if argv else '/home/claude/tp6/out/storefront')
os.environ.setdefault('UI_STEP', argv[1] if len(argv) > 1 else '1')

# ------------------------------------------------------------------------------------------------ UI capture
def _install_playback(scene, CO):
    """The scene's own TS_StorefrontPlayback (fan jet, blade spin) wired to this branch's cloth_ops."""
    src = bpy.data.texts['TS_StorefrontPlayback'].as_string()
    src = src.replace("def _cloth_ops():", "def _cloth_ops():\n    from arc_flow import cloth_ops as _c\n    return _c\n\ndef _cloth_ops_orig():", 1)
    src = src.replace("    _on_load()\n\n\ndef unregister", "    pass\n\n\ndef unregister", 1)
    ns = {'__name__': 'TS_StorefrontPlayback'}
    exec(compile(src, 'TS_StorefrontPlayback', 'exec'), ns)
    ns['register']()
    ns['_STATE']['hooked'] = True
    CO._fluid_field = lambda sc: ns['_apply_fan'](sc)[0]


def ui(work, blend, cache):
    bpy.ops.wm.open_mainfile(filepath=blend)
    import arc_flow
    arc_flow.register()
    from importlib import import_module
    CO = import_module('arc_flow.cloth_ops')
    from mathutils import Euler, Vector

    W = int(os.environ.get('UI_W', '1920'))
    H = int(os.environ.get('UI_H', '1080'))
    STEP = int(os.environ.get('UI_STEP', '2'))
    SHADING = os.environ.get('UI_SHADING', 'MATERIAL')
    OUT = os.environ.get('UI_OUT', os.path.join(work, 'ui_frames'))
    os.makedirs(OUT, exist_ok=True)

    z = np.load(cache)
    X = z['x']
    ms = z['ms'] if 'ms' in z.files else np.zeros(len(X))
    contacts = z['contacts'] if 'contacts' in z.files else np.zeros(len(X))
    mind = z['mind'] if 'mind' in z.files else np.full(len(X), np.nan)
    nfr = len(X)
    FIRST = int(os.environ.get('UI_FIRST', '1'))
    LAST = int(os.environ.get('UI_LAST', str(nfr)))

    wm = bpy.context.window_manager
    state = dict(draws=0, n=0)
    win = wm.windows[0]

    def area(kind):
        return next(a for a in win.screen.areas if a.type == kind)

    def region(a, kind='WINDOW'):
        return next(r for r in a.regions if r.type == kind)

    def on_draw():
        state['draws'] += 1
    bpy.types.SpaceView3D.draw_handler_add(on_draw, (), 'WINDOW', 'POST_PIXEL')

    def redraw_all():
        for a in win.screen.areas:
            a.tag_redraw()

    def settle(wait=0.2, timeout=120.0):
        d0 = state['draws']
        redraw_all()
        t0 = time.time()
        while state['draws'] <= d0 and time.time() - t0 < timeout:
            yield 0.05
            redraw_all()
        yield wait

    def grab(path):
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'x11grab', '-draw_mouse', '0',
                        '-video_size', '%dx%d' % (W, H), '-i', os.environ['DISPLAY'], '-frames:v', '1', '-q:v', '2', path],
                       check=True)

    scene = bpy.context.scene
    cl = scene.arc_flow_cloth
    g = cl.garments[0]
    CO.RT.live_running = True          # replaying a cache: the frame handler must not step the solver
    CO.build_solver(scene, g)
    _install_playback(scene, CO)
    try:
        scene.arc_flow.ui_tab = 'CLOTH'
    except Exception as e:
        print("ui_tab", e, flush=True)

    # ---- screen set-up: Layout workspace (viewport + N-panel, outliner, properties, timeline)
    pr = bpy.context.preferences
    pr.view.show_splash = False
    pr.view.ui_scale = float(os.environ.get('UI_SCALE', '1.0'))
    v3d_area = area('VIEW_3D')
    sp = v3d_area.spaces.active
    sp.show_region_ui = True
    sp.show_region_toolbar = True
    sp.overlay.show_stats = False
    sp.overlay.show_floor = False
    sp.overlay.show_axis_x = sp.overlay.show_axis_y = False
    sp.overlay.show_extras = False
    sp.overlay.show_relationship_lines = False
    sp.show_gizmo = False
    sp.clip_start, sp.clip_end = 0.05, 60.0
    sp.lens = 38
    ee = scene.eevee
    for k, v in (('use_raytracing', False), ('use_shadows', True), ('taa_samples', int(os.environ.get('UI_SAMPLES', '1'))),
                 ('taa_render_samples', 16), ('use_volumetric_shadows', False), ('shadow_ray_count', 1),
                 ('shadow_step_count', 6)):
        try:
            setattr(ee, k, v)
        except Exception as e:
            print("eevee", k, e, flush=True)
    scene.render.engine = 'BLENDER_EEVEE' if 'BLENDER_EEVEE' in [i.identifier for i in
                                                                     scene.render.bl_rna.properties['engine'].enum_items] else 'BLENDER_EEVEE_NEXT'
    try:
        scene.view_settings.view_transform = 'AgX'
    except Exception:
        pass
    shading_used = SHADING
    sp.shading.type = 'RENDERED' if SHADING == 'RENDERED' else 'MATERIAL'
    try:
        sp.shading.use_scene_lights_render = True
        sp.shading.use_scene_world_render = True
    except Exception:
        pass
    # N-panel: Tessera category, widened

    def pick_target():
        return Vector((0.0, 0.0, 0.85))

    def set_view(f):
        t = f / 24.0
        az = math.radians(-28.0 + 38.0 * math.sin(t * 0.21) + 8.0 * math.sin(t * 0.53))
        el = math.radians(8.0 + 5.0 * math.sin(t * 0.17 + 1.0))
        dist = 3.9 - 0.7 * (0.5 + 0.5 * math.sin(t * 0.14 + 0.5))
        rv = sp.region_3d
        rv.view_perspective = 'PERSP'
        rv.view_location = pick_target()
        rv.view_distance = dist
        rv.view_rotation = Euler((math.pi / 2 - el, 0.0, az + 0.0), 'XYZ').to_quaternion()

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

    def click(x, y):
        mouse_move(x, y)
        yield 0.4
        mouse_button(1, True)
        yield 0.2
        mouse_button(1, False)
        yield 0.6

    def widen_sidebar(target_left):
        r = region(v3d_area, 'UI')
        edge = r.x - 3
        print("sidebar region x=%d w=%d" % (r.x, r.width), flush=True)
        mouse_move(edge, 600)
        yield 0.4
        mouse_button(1, True)
        yield 0.3
        x = edge
        while x > target_left:
            x -= 40
            mouse_move(x, 600)
            yield 0.15
        mouse_button(1, False)
        yield 0.4
        mouse_move(900, 900)
        yield 0.4
        print("sidebar now x=%d w=%d" % (region(v3d_area, 'UI').x, region(v3d_area, 'UI').width), flush=True)

    def scroll_sidebar(clicks):
        r = region(v3d_area, 'UI')
        mouse_move(r.x + 150, 600)
        yield 0.3
        for _ in range(abs(clicks)):
            mouse_button(5 if clicks > 0 else 4, True)
            mouse_button(5 if clicks > 0 else 4, False)
            yield 0.15
        mouse_move(900, 900)
        yield 0.3

    manifest = []
    frames = list(range(FIRST, LAST + 1, STEP))
    print("UI capture: %d frames (of %d) step %d shading %s %dx%d" % (len(frames), nfr, STEP, shading_used, W, H), flush=True)

    def seq():
        # warm-up: first redraw compiles the EEVEE shaders (can take minutes under llvmpipe)
        set_view(1)
        yield from settle(0.5, timeout=900.0)
        tx, ty = [int(v) for v in os.environ.get('UI_TAB_XY', '1562,328').split(',')]
        yield from click(tx, ty)                      # the 'Tessera' tab of the N-panel
        yield from settle(0.5)
        yield from widen_sidebar(int(os.environ.get('UI_SIDEBAR_X', '1130')))
        yield from settle(0.5)
        yield from scroll_sidebar(int(os.environ.get('UI_SCROLL', '0')))
        yield from settle(0.5)
        t_all = time.time()
        for k, f in enumerate(frames):
            t0 = time.time()
            scene.frame_set(f)
            cur = f - 1
            X0 = X[cur]
            CO.write_garment(g, X0.astype(np.float64))
            g.last_ms = float(ms[cur - 1]) if cur >= 1 and cur - 1 < len(ms) else 0.0
            cl.last_ms = g.last_ms
            cl.last_fps = 1000.0 / max(g.last_ms, 1.0)
            cl.status = "Replaying solved cache - self-collision ON, %d contacts, frame %d / %d" % (
                int(contacts[cur - 1]) if cur >= 1 and cur - 1 < len(contacts) else 0, f, nfr)
            set_view(f)
            yield from settle(0.15)
            fn = os.path.join(OUT, 'f%05d.jpg' % f)
            grab(fn)
            manifest.append(dict(frame=f, file=os.path.basename(fn)))
            if k % 5 == 0:
                with open(os.path.join(OUT, 'manifest.json'), 'w') as fh:
                    json.dump(manifest, fh)
                print("captured %d/%d (frame %d) %.1f s each, elapsed %.0f s" % (k + 1, len(frames), f, time.time() - t0,
                                                                                 time.time() - t_all), flush=True)
        with open(os.path.join(OUT, 'manifest.json'), 'w') as fh:
            json.dump(manifest, fh)
        print("capture done", flush=True)
        bpy.ops.wm.quit_blender()

    gen = seq()

    def tick():
        try:
            return next(gen)
        except StopIteration:
            return None
        except Exception as e:
            import traceback
            traceback.print_exc()
            bpy.ops.wm.quit_blender()
            return None
    bpy.app.timers.register(tick, first_interval=3.0, persistent=True)




if __name__ == '__main__':
    ui(WORK, os.path.join(WORK, 'storefront_group.blend'), os.path.join(WORK, 'sf_cache.npz'))
