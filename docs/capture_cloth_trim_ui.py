"""Real Blender UI screenshots (Material Preview / EEVEE, software OpenGL) of Cloth Trim, Accents and Seam Seal
Control Solving: the jacket example zipping shut, with the Cloth tab panels visible in the sidebar.

    xvfb-run -a -s "-screen 0 1700x3600x24" bl.sh -r --gpu-backend opengl --factory-startup \
        --window-geometry 0 0 1700 3600 --python docs/capture_cloth_trim_ui.py -- OUTDIR

Writes OUTDIR/cloth_trim_1_open.png, _2_zipping.png, _3_closed.png, _4_accents.png.
"""
import math
import os
import sys
import time

import bpy
from mathutils import Euler, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import arc_flow
arc_flow.register()
from arc_flow import cloth_ops as CO
from arc_flow import cloth_trim_ops as TO

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
OUT = argv[0] if argv else os.path.join(HERE, 'renders')
os.makedirs(OUT, exist_ok=True)
W, H = int(os.environ.get('UI_W', '1700')), int(os.environ.get('UI_H', '3600'))
sc = bpy.context.scene
log = lambda *a: print(*a, flush=True)


def run():
    win = bpy.context.window_manager.windows[0]
    yield 0.5
    for nm in ('Cube', 'Light', 'Camera'):
        if nm in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[nm], do_unlink=True)
    log(TO.load_jacket_example(sc))
    ts = sc.arc_flow_cloth_trim
    seam = ts.seams[0]
    sc.arc_flow.ui_tab = 'CLOTH'
    # floor so the figure stands on something
    bpy.ops.mesh.primitive_plane_add(size=2.4, location=(0, 0, 0))
    bpy.context.active_object.name = "Studio Floor"
    area = next(a for a in win.screen.areas if a.type == 'VIEW_3D')
    sp = area.spaces.active
    sp.show_region_ui = True
    sp.show_region_toolbar = False
    sp.overlay.show_stats = False
    sp.overlay.show_extras = False
    sp.show_gizmo = False
    sp.shading.type = 'MATERIAL'          # Material Preview (EEVEE)
    sp.clip_start = 0.01
    r3d = sp.region_3d
    r3d.view_perspective = 'PERSP'
    for r in area.regions:
        if r.type == 'UI':
            try:
                r.active_panel_category = 'Tessera'
            except Exception:
                pass
    with bpy.context.temp_override(window=win, area=area):
        bpy.ops.screen.screen_full_area(use_hide_panels=False)
    yield 1.0
    area = next(a for a in win.screen.areas if a.type == 'VIEW_3D')   # the full-screen area is a new one
    sp = area.spaces.active
    r3d = sp.region_3d
    bpy.ops.object.select_all(action='DESELECT')

    def tessera_tab():
        for r in area.regions:
            if r.type == 'UI':
                r.active_panel_category = 'Tessera'
        for a in win.screen.areas:
            a.tag_redraw()
    tessera_tab()
    yield 1.0

    def view(dist, loc, rz=-0.5, rx=math.pi / 2 - 0.05):
        r3d.view_rotation = Euler((rx, 0, rz)).to_quaternion()
        r3d.view_location = Vector(loc)
        r3d.view_distance = dist

    def step(n, fn):
        for k in range(n):
            seam.progress = float(fn(k))
            CO.step_scene(sc, 1 / 60.0)

    def shot(name):
        for a in win.screen.areas:
            a.tag_redraw()
        return os.path.join(OUT, name)

    def grab(name):
        tessera_tab()
        path = os.path.join(OUT, name)
        import subprocess
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "x11grab", "-video_size", "%dx%d" % (W, H),
                        "-i", os.environ.get("DISPLAY", ":0"), "-frames:v", "1", path], check=False)
        log("shot", path)

    t0 = time.time()
    ts.seam_index = 0
    ts.trim_index = 0
    ts.accent_index = 1
    # settle the open jacket
    for c in range(3):
        step(12, lambda k: 0.0)
        yield 0.05
    view(1.5, (0, 0, 1.0), rx=math.pi / 2 - 0.2)
    yield 2.5
    for a in win.screen.areas:
        a.tag_redraw()
    yield 1.5
    grab("cloth_trim_1_open.png")
    log("settle+shot1 %.0fs" % (time.time() - t0))
    RAMP = 90
    done = 0
    while done < RAMP // 2:
        step(10, lambda k: (done + k) / RAMP)
        done += 10
        yield 0.05
    seam.progress = done / RAMP
    CO.step_scene(sc, 1 / 60.0)
    yield 2.5
    for a in win.screen.areas:
        a.tag_redraw()
    yield 1.5
    grab("cloth_trim_2_zipping.png")
    while done < RAMP + 10:
        step(10, lambda k: min(1.0, (done + k) / RAMP))
        done += 10
        yield 0.05
    for c in range(4):
        step(10, lambda k: 1.0)
        yield 0.05
    yield 2.5
    for a in win.screen.areas:
        a.tag_redraw()
    yield 1.5
    grab("cloth_trim_3_closed.png")
    ts.accent_index = 1
    view(0.75, (0.0, -0.1, 1.28), rz=-0.35, rx=math.pi / 2 - 0.12)
    yield 2.5
    for a in win.screen.areas:
        a.tag_redraw()
    yield 1.5
    grab("cloth_trim_4_accents.png")
    log("done %.0fs" % (time.time() - t0))
    bpy.ops.wm.quit_blender()


gen = run()


def tick():
    try:
        return next(gen)
    except StopIteration:
        return None
    except Exception:
        import traceback
        traceback.print_exc()
        bpy.ops.wm.quit_blender()
        return None


bpy.app.timers.register(tick, first_interval=2.0)
