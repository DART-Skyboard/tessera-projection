"""Full-Blender-UI capture driver for the 0.6.0 showcase (real GUI, real Cycles viewport).

Run on a private virtual display (never the user's):
    Xvfb :99 -screen 0 1920x1080x24 &
    DISPLAY=:99 LIBGL_ALWAYS_SOFTWARE=1 BLENDER_USER_RESOURCES=<cfg with the 0.6.0 zip installed> \
      blender --window-geometry 0 0 1920 1080 --python docs/showcase/capture.py -- \
      <scene> <outdir> [--frames N] [--hero F] [--hero-samples S] [--video-samples S] [--start F]

<scene> is a module in docs/showcase exposing build() -> state (state['step'](frame)) and
ui(state, space, region_3d, ctx) to place the viewport and pick the Tessera tab.
Every video frame: advance the Tessera solver one tick, let the Cycles *viewport*
re-render until it is converged (pixels stable), then grab the whole X screen
(the Blender window: viewport, Tessera N-panel, outliner, properties, timeline).
"""
import argparse
import hashlib
import importlib
import os
import subprocess
import sys
import time

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument('scene')
ap.add_argument('outdir')
ap.add_argument('--frames', type=int, default=0)
ap.add_argument('--start', type=int, default=1)
ap.add_argument('--hero', type=int, default=0, help='frame to grab the high-sample PNG at (0 = none)')
ap.add_argument('--hero-samples', type=int, default=96)
ap.add_argument('--video-samples', type=int, default=10)
ap.add_argument('--stable', type=float, default=1.6, help='seconds the viewport must stay unchanged')
ap.add_argument('--display', default=os.environ.get('DISPLAY', ':99'))
ap.add_argument('--quit', action='store_true')
ap.add_argument('--widen', type=int, default=120)
ap.add_argument('--tab-dy', type=int, default=250)
ap.add_argument('--text-dy', type=int, default=0)
A = ap.parse_args(argv)
os.makedirs(os.path.join(A.outdir, 'frames'), exist_ok=True)
SCREEN_W, SCREEN_H = 1920, 1080
LOG = open(os.path.join(A.outdir, 'capture.log'), 'a')


def log(*a):
    msg = ' '.join(str(x) for x in a)
    print('[capture]', msg, flush=True)
    LOG.write(time.strftime('%H:%M:%S ') + msg + '\n')
    LOG.flush()


def win():
    return bpy.context.window_manager.windows[0]


def area(kind):
    for a in win().screen.areas:
        if a.type == kind:
            return a
    return None


def region(a, kind='WINDOW'):
    for r in a.regions:
        if r.type == kind:
            return r
    return None


def grab(path=None, crop=None):
    """Grab the X screen. crop=(x, y, w, h) in X coords returns raw bytes for comparing."""
    if crop is None:
        subprocess.run(['ffmpeg', '-loglevel', 'error', '-f', 'x11grab', '-video_size', '%dx%d' % (SCREEN_W, SCREEN_H),
                        '-i', A.display, '-frames:v', '1', '-y', path], check=True)
        return None
    x, y, w, h = crop
    r = subprocess.run(['ffmpeg', '-loglevel', 'error', '-f', 'x11grab', '-video_size', '%dx%d' % (w, h),
                        '-i', '%s+%d,%d' % (A.display, x, y), '-frames:v', '1', '-f', 'rawvideo',
                        '-pix_fmt', 'rgb24', '-'], capture_output=True, check=True)
    return hashlib.md5(r.stdout).hexdigest()


def text_mask(crop):
    """Hash of the near-white overlay text pixels ('Sample N/M') — independent of the render behind it."""
    import numpy as np
    x, y, w, h = crop
    r = subprocess.run(['ffmpeg', '-loglevel', 'error', '-f', 'x11grab', '-video_size', '%dx%d' % (w, h),
                        '-i', '%s+%d,%d' % (A.display, x, y), '-frames:v', '1', '-f', 'rawvideo',
                        '-pix_fmt', 'rgb24', '-'], capture_output=True, check=True)
    a = np.frombuffer(r.stdout, np.uint8).reshape(h, w, 3)
    m = (a.min(axis=2) > 205)
    return hashlib.md5(np.packbits(m).tobytes()).hexdigest(), int(m.sum())


def sample_text_crop():
    a = area('VIEW_3D')
    r = region(a)
    top = SCREEN_H - (r.y + r.height)
    return (r.x + 62, top + 100 + A.text_dy, 150, 18)


def ui_tab_click():
    """The sidebar category is read-only from Python in 5.3: click the 'Tessera' tab like a user."""
    a = area('VIEW_3D')
    ui = region(a, 'UI')
    top = SCREEN_H - (ui.y + ui.height)
    x = ui.x + ui.width - 10
    vr = region(a)
    log('view region x=%d y=%d w=%d h=%d top=%d' % (vr.x, vr.y, vr.width, vr.height, SCREEN_H - (vr.y + vr.height)))
    log('ui region x=%d y=%d w=%d h=%d top=%d' % (ui.x, ui.y, ui.width, ui.height, top))
    y = top + A.tab_dy
    env = dict(os.environ, DISPLAY=A.display)
    try:
        wid = subprocess.run(['xdotool', 'search', '--name', 'Blender'], env=env, capture_output=True,
                             text=True).stdout.split()
        if wid:
            subprocess.run(['xdotool', 'windowfocus', wid[0]], env=env)
    except Exception as ex:
        log('focus', ex)
    subprocess.run(['xdotool', 'mousemove', str(x - 2), str(y - 2)], env=env)
    time.sleep(0.3)
    subprocess.run(['xdotool', 'mousemove', str(x), str(y)], env=env)
    time.sleep(0.3)
    subprocess.run(['xdotool', 'click', '1'], env=env)
    time.sleep(0.5)
    if A.widen:
        # widen the sidebar by dragging its left edge, like a user would
        ex, ey = ui.x + 1, top + ui.height // 2
        subprocess.run(['xdotool', 'mousemove', str(ex), str(ey)], env=env)
        time.sleep(0.4)
        subprocess.run(['xdotool', 'mousedown', '1'], env=env)
        for k in range(1, 11):
            time.sleep(0.08)
            subprocess.run(['xdotool', 'mousemove', str(ex - A.widen * k // 10), str(ey)], env=env)
        time.sleep(0.2)
        subprocess.run(['xdotool', 'mouseup', '1'], env=env)
        time.sleep(0.4)
    subprocess.run(['xdotool', 'mousemove', '960', '1070'], env=dict(os.environ, DISPLAY=A.display))


def viewport_crop():
    a = area('VIEW_3D')
    r = region(a)
    ui = region(a, 'UI')
    w = r.width - (ui.width if ui is not None and a.spaces[0].show_region_ui else 0)
    x = r.x
    y = SCREEN_H - (r.y + r.height)
    return (x + 8, y + 40, max(64, w - 16), r.height - 80)


def redraw():
    for a in win().screen.areas:
        a.tag_redraw()


S = dict(phase='init', frame=A.start, last=None, since=0.0, t_frame=0.0, n=0, state=None)


def build():
    mod = importlib.import_module(A.scene)
    t = time.time()
    st = mod.build()
    log('built', A.scene, '%.1fs' % (time.time() - t))
    sc = bpy.context.scene
    a = area('VIEW_3D')
    sp = a.spaces[0]
    sp.show_region_ui = True
    sp.show_region_toolbar = True
    try:
        region(a, 'UI').active_panel_category = 'Tessera'
    except Exception as ex:
        log('category', ex)
    sp.shading.type = 'RENDERED'
    ov = sp.overlay
    ov.show_floor = False
    ov.show_axis_x = ov.show_axis_y = ov.show_axis_z = False
    ov.show_relationship_lines = False
    ov.show_outline_selected = False
    ov.show_extras = False
    ov.show_bones = False
    ov.show_cursor = False
    ov.show_object_origins = False
    ov.show_text = True
    ov.show_stats = False
    sp.show_gizmo = True
    sp.show_gizmo_navigate = True
    bpy.ops.object.select_all(action='DESELECT') if bpy.context.mode == 'OBJECT' else None
    for o in sc.objects:
        o.select_set(False)
    with bpy.context.temp_override(window=win(), area=a, region=region(a)):
        mod.ui(st, sp, sp.region_3d, bpy.context)
    # properties editor: show what the scene module asks for (e.g. the SSS skin material)
    pa = area('PROPERTIES')
    if pa is not None and st.get('properties'):
        try:
            pa.spaces[0].context = st['properties']
        except Exception as ex:
            log('props ctx', ex)
    # timeline: frame the whole clip
    sc.frame_current = A.start
    S['state'] = st
    S['mod'] = mod
    sc.cycles.preview_samples = A.hero_samples if A.hero == A.start else A.video_samples
    redraw()


def tick():
    try:
        return _tick()
    except Exception:
        import traceback
        log(traceback.format_exc())
        S['phase'] = 'done'
        if A.quit:
            bpy.ops.wm.quit_blender()
        return None


def _tick():
    sc = bpy.context.scene
    if S['phase'] == 'init':
        build()
        S['phase'] = 'warm'
        S['since'] = time.time()
        return 2.0
    if S['phase'] == 'warm':
        ui_tab_click()
        S['phase'] = 'step'
        return 1.0
    if S['phase'] == 'step':
        f = S['frame']
        hero = (A.hero and f == A.hero)
        sc.cycles.preview_samples = A.hero_samples if hero else A.video_samples
        S['t_frame'] = time.time()
        S['state']['step'](f)
        sc.frame_current = f
        redraw()
        S['last'] = None
        S['phase'] = 'settle'
        S['since'] = time.time()
        return 0.5
    if S['phase'] == 'settle':
        now = time.time()
        th, npx = text_mask(sample_text_crop())
        key = sc.cycles.preview_samples
        ref = S.setdefault('ref', {}).get(key)
        h = grab(crop=viewport_crop())
        changed = (h != S['last'])
        S['last'] = h
        if changed:
            S['since'] = now
        if ref is None:
            # learn what the finished 'Sample M/M' text looks like: wait for a long, quiet viewport
            if now - S['since'] < max(A.stable, 6.0):
                return 0.4
            S['ref'][key] = th
            log('learned final sample text for %d samples (%d px)' % (key, npx))
        else:
            if th != ref or now - S['since'] < 0.6:
                if now - S['t_frame'] > 240:
                    log('frame %d: timeout waiting for convergence' % S['frame'])
                else:
                    return 0.3
        f = S['frame']
        if A.hero and f == A.hero:
            p = os.path.join(A.outdir, 'hero_%04d.png' % f)
            grab(p)
            log('hero', p)
        if A.frames:
            grab(os.path.join(A.outdir, 'frames', '%05d.png' % (f - A.start + 1)))
        S['n'] += 1
        el = now - S['t_frame']
        if S['n'] % 10 == 1 or f == A.hero:
            log('frame %d done in %.1fs  %s' % (f, el, S['state'].get('status', lambda: '')()))
        last = A.start + max(A.frames, 1) - 1
        if A.hero:
            last = max(last, A.hero)
        if f >= last:
            S['phase'] = 'done'
            log('DONE')
            if A.quit:
                bpy.ops.wm.quit_blender()
            return None
        S['frame'] = f + 1
        S['phase'] = 'step'
        return 0.01
    return None


try:
    bpy.context.preferences.view.show_splash = False
except Exception:
    pass
bpy.app.timers.register(tick, first_interval=3.0)
