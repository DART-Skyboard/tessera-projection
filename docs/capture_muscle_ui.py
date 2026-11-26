"""Still screenshots of the full Blender UI with the Tessera Muscle tab open (0.7.0 extension page / README).

    Xvfb :97 -screen 0 1920x1080x24 -ac +extension GLX +render -noreset &
    DISPLAY=:97 LIBGL_ALWAYS_SOFTWARE=1 blender --factory-startup --window-geometry 0 0 1920 1080 \\
        /workspace/tp7demo/ariel_tessera.blend --python docs/capture_muscle_ui.py -- --out docs/screenshots/0.7.0

Uses the lead's Ariel example scene (docs/showcase/ariel_scene.py) when it is the open file, else builds the procedural
example biped.  The sidebar category cannot be switched from Python in Blender 5.x, so the script clicks the 'Tessera'
tab, widens the sidebar and scrolls it with xdotool like a user, and grabs the X screen with ffmpeg (x11grab).

It also drives the real interactive Muscle Pencil (arcflow.muscle_contour) with XTEST mouse drags over the cheek and
checks that the strokes land on the skin (an end-to-end test of the modal operator); the result is written to
<out>/capture_report.json.
"""
import json
import math
import os
import subprocess
import sys
import time

import bpy
import numpy as np
from mathutils import Quaternion, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import arc_flow  # noqa: E402

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []


def arg(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


OUT = os.path.abspath(arg('--out', os.path.join(HERE, 'screenshots', '0.7.0')))
DISPLAY = os.environ.get('DISPLAY', ':97')
SW, SH = 1920, 1080
WIDEN = int(arg('--widen', '230'))
os.makedirs(OUT, exist_ok=True)
REPORT = {'shots': [], 'pencil_modal': None, 'notes': []}

try:
    arc_flow.register()
except Exception as ex:   # already registered from an installed copy
    print("[cap] register:", ex, flush=True)
from arc_flow import muscle_ops as MO  # noqa: E402
from arc_flow import muscle_ui_ops as MUI  # noqa: E402
from arc_flow import ui_tabs  # noqa: E402
from arc_flow.core import muscle_solver as MS  # noqa: E402

DRAWN = {'t': 0.0}


def _probe(layout, ctx):
    DRAWN['t'] = time.time()


ui_tabs.register_tab_drawer('MUSCLE', _probe)


def log(*a):
    print("[cap]", *a, flush=True)


def win():
    return bpy.context.window_manager.windows[0]


def area3d():
    return next(a for a in win().screen.areas if a.type == 'VIEW_3D')


def reg(a, kind):
    return next(r for r in a.regions if r.type == kind)


def xdo(cmds, wait=True):
    sh = 'wid=$(xdotool search --name Blender | head -1); [ -n "$wid" ] && xdotool windowfocus $wid; ' + '; '.join(cmds)
    p = subprocess.Popen(['bash', '-c', sh], env=dict(os.environ, DISPLAY=DISPLAY))
    return p


def grab(path, crop=None):
    if crop is None:
        cmd = ['ffmpeg', '-loglevel', 'error', '-f', 'x11grab', '-video_size', '%dx%d' % (SW, SH), '-i', DISPLAY,
               '-frames:v', '1', '-y', path]
        subprocess.run(cmd, check=True)
        return None
    x, y, w, h = crop
    r = subprocess.run(['ffmpeg', '-loglevel', 'error', '-f', 'x11grab', '-video_size', '%dx%d' % (w, h), '-i',
                        '%s+%d,%d' % (DISPLAY, x, y), '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                       capture_output=True, check=True)
    return hash(r.stdout)


def redraw():
    for a in win().screen.areas:
        a.tag_redraw()


# ----------------------------------------------------------------------------------------------------------------------
# scene preparation
# ----------------------------------------------------------------------------------------------------------------------
sc = bpy.context.scene
ms = sc.arc_flow_muscle
if ms.skin is None:
    log("no muscle skin in the open file: building the procedural example biped")
    bpy.ops.arcflow.muscle_example()
skin, arm = ms.skin, ms.armature
if arm is not None:
    arm.data.pose_position = 'REST'      # anatomical rest pose for clean pencil lines
bpy.context.view_layer.update()


def bone_pts(name_part, t0, t1, n=3):
    """Points along a deform bone (skin-local, rest), from fraction t0 to t1."""
    if arm is None:
        return None
    b = next((b for b in arm.data.bones if name_part.lower() == b.name.lower()), None)
    if b is None:
        b = next((b for b in arm.data.bones if name_part.lower() in b.name.lower()), None)
    if b is None:
        return None
    m = skin.matrix_world.inverted() @ arm.matrix_world
    h, t = m @ b.head_local, m @ b.tail_local
    return [h.lerp(t, t0 + (t1 - t0) * k / (n - 1)) for k in range(n)]


def surf(p, direction=(0, 1, 0), lift=1.0):
    """Surface point hit by a ray from outside the skin through p (skin-local) along `direction` (default: from the front)."""
    d = Vector(direction).normalized()
    o = Vector(p) - d * lift
    ok, loc, _n, _f = skin.ray_cast(o, d)
    return tuple(loc) if ok else None


def head_box():
    hb = [vg.index for vg in skin.vertex_groups if vg.name.lower() in ('head', 'head_end')]
    v = MO.basis_coords(skin)
    if not hb:
        return v.min(0), v.max(0)
    w = np.zeros(len(v))
    for vx in skin.data.vertices:
        for g in vx.groups:
            if g.group in hb:
                w[vx.index] = max(w[vx.index], g.weight)
    sel = v[w > 0.5]
    return sel.min(0), sel.max(0)


HB0, HB1 = head_box()


def face_pt(fx, fz):
    """Front surface point of the face at head-box fractions (fx: 0 centre .. 0.5 side, + = her left; fz: 0 chin .. 1 top)."""
    cx = (HB0[0] + HB1[0]) / 2
    x = cx + fx * (HB1[0] - HB0[0])
    z = HB0[2] + fz * (HB1[2] - HB0[2])
    return surf((x, (HB0[1] + HB1[1]) / 2, z), (0, 1, 0), lift=1.0)


def add_line(key, pts, local=True):
    i = next((j for j, r in enumerate(ms.regions) if r.key == key), None)
    pts = [p for p in pts if p is not None]
    if i is None or len(pts) < 2:
        log("skip line", key, len(pts))
        return False
    r = bpy.ops.arcflow.muscle_pencil_stroke(points=[{"name": "", "co": tuple(p)} for p in pts], space='LOCAL', index=i)
    return r == {'FINISHED'}


def front_line(bone, t0, t1, key, dx=0.0, dz=0.0, back=False):
    bp = bone_pts(bone, t0, t1, 4)
    if bp is None:
        return False
    d = (0, -1, 0) if back else (0, 1, 0)
    return add_line(key, [surf(p + Vector((dx, 0, dz)), d) for p in bp])


def prepare_strokes():
    MO.save_strokes(skin, [])
    # face (her left side, +X): cheek from the mouth corner toward the outer eye corner, jaw, neck
    add_line('zygomaticus', [face_pt(-0.10, 0.30), face_pt(-0.20, 0.42), face_pt(-0.30, 0.55)])
    add_line('masseter', [face_pt(0.30, 0.18), face_pt(0.33, 0.33), face_pt(0.35, 0.48)])
    add_line('masseter', [face_pt(0.26, 0.16), face_pt(0.29, 0.32), face_pt(0.31, 0.47)])
    for side, dx in (('Left', 0.0), ('Right', 0.0)):
        front_line(side + 'Arm', 0.25, 0.85, 'biceps', dz=0.012)
        front_line(side + 'Arm', 0.28, 0.82, 'biceps', dz=-0.004)
        front_line(side + 'Arm', 0.0, 0.3, 'deltoid', dz=0.03)
        front_line(side + 'ForeArm', 0.1, 0.75, 'forearm_extensors', dz=0.01)
        front_line(side + 'UpLeg', 0.2, 0.85, 'quadriceps', dx=0.01)
        front_line(side + 'UpLeg', 0.25, 0.8, 'quadriceps', dx=-0.03)
        front_line(side + 'Leg', 0.05, 0.55, 'gastrocnemius', back=True)
    for sx in (-0.035, 0.035):
        bp = bone_pts('Spine', 0.0, 1.0, 4)
        sp2 = bone_pts('Hips', 0.6, 1.0, 2)
        if bp and sp2:
            add_line('rectus_abdominis', [surf(p + Vector((sx, 0, 0))) for p in (sp2 + bp)])
    s2 = bone_pts('Spine02', 0.0, 1.0, 2)
    for side in (1, -1):
        sh = bone_pts(('Left' if side > 0 else 'Right') + 'Arm', 0.0, 0.0, 2)
        if s2 and sh:
            a, b = s2[0] + Vector((0.01 * side, 0, 0)), sh[0] + Vector((0, 0, -0.04))
            add_line('pectoralis', [surf(a.lerp(b, k / 4.0)) for k in range(5)])
    n = len(MO.load_strokes(skin))
    log("prepared", n, "pencil lines")
    for key in ('deltoid', 'biceps', 'pectoralis', 'rectus_abdominis', 'quadriceps', 'masseter', 'gastrocnemius'):
        i = next((j for j, r in enumerate(ms.regions) if r.key == key), None)
        if i is not None and ms.regions[i].n_strokes:
            ms.regions[i].completed = True
    return n


def view_setup(kind):
    a = area3d()
    sp = a.spaces.active
    rv = sp.region_3d
    sp.show_region_ui = True
    sp.show_region_toolbar = True
    ov = sp.overlay
    ov.show_floor = True
    ov.show_axis_x = ov.show_axis_y = False
    ov.show_relationship_lines = False
    ov.show_outline_selected = False
    ov.show_extras = False
    ov.show_bones = False
    ov.show_cursor = False
    ov.show_object_origins = False
    ov.show_stats = False
    sp.shading.type = 'SOLID'
    sp.shading.light = 'STUDIO'
    sp.shading.show_cavity = True
    sp.shading.cavity_type = 'WORLD'
    sp.shading.show_shadows = False
    if kind in ('overview', 'solve'):
        MO.set_active_color(skin, MO.ATTR_SOLVED_COLOR)
        sp.shading.color_type = 'VERTEX'
    else:
        sp.shading.color_type = 'MATERIAL'
    rv.view_perspective = 'PERSP'
    rv.view_rotation = Quaternion((0.7071068, 0.7071068, 0.0, 0.0))   # front view
    mw = skin.matrix_world
    v = MO.basis_coords(skin)
    lo, hi = v.min(0), v.max(0)
    if kind in ('overview', 'solve'):
        c = mw @ Vector(((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2))
        rv.view_location = c
        rv.view_distance = (hi[2] - lo[2]) * 1.45
    elif kind == 'face':
        c = mw @ Vector(((HB0[0] + HB1[0]) / 2, (HB0[1] + HB1[1]) / 2, HB0[2] + 0.35 * (HB1[2] - HB0[2])))
        rv.view_location = c
        rv.view_distance = (HB1[2] - HB0[2]) * 2.9
    elif kind == 'arm':
        bp = bone_pts('LeftArm', 0.0, 1.0, 2) or [Vector(((lo[0] + hi[0]) / 2, 0, hi[2] * 0.8))] * 2
        c = mw @ ((bp[0] + bp[1]) / 2 + Vector((-0.05, 0, -0.05)))
        rv.view_location = c
        rv.view_distance = (hi[2] - lo[2]) * 0.62
    for o in sc.objects:
        o.select_set(False)
    redraw()


# ----------------------------------------------------------------------------------------------------------------------
# state machine (timers keep Blender's event loop free for the XTEST input)
# ----------------------------------------------------------------------------------------------------------------------
S = {'phase': 'start', 't': 0.0, 'proc': None, 'dy': 40, 'shot': 0, 'last': None, 'stable': 0}


def ui_geom():
    a = area3d()
    ui = reg(a, 'UI')
    top = SH - (ui.y + ui.height)
    return a, ui, top


def click_tab(dy):
    a, ui, top = ui_geom()
    x, y = ui.x + ui.width - 10, top + dy
    return xdo(['xdotool mousemove %d %d' % (x, y), 'sleep 0.3', 'xdotool click 1', 'sleep 0.5',
                'xdotool mousemove 900 1070'])


def widen():
    a, ui, top = ui_geom()
    ex, ey = ui.x + 3, top + 300
    cmds = ['xdotool mousemove %d %d' % (ex, ey), 'sleep 0.4', 'xdotool mousedown 1', 'sleep 0.3']
    for k in range(1, 11):
        cmds += ['xdotool mousemove %d %d' % (ex - WIDEN * k // 10, ey), 'sleep 0.08']
    cmds += ['sleep 0.2', 'xdotool mouseup 1', 'sleep 0.3', 'xdotool mousemove 900 1070']
    return xdo(cmds)


def scroll_sidebar(clicks):
    a, ui, top = ui_geom()
    x, y = ui.x + ui.width // 2 - 20, top + 500
    btn = 5 if clicks > 0 else 4
    cmds = ['xdotool mousemove %d %d' % (x, y), 'sleep 0.3']
    cmds += ['xdotool click %d' % btn, 'sleep 0.06'] * abs(clicks)
    cmds += ['sleep 0.3', 'xdotool mousemove 900 1070']
    return xdo(cmds)


def to_screen(p_local):
    """Skin-local rest point -> X screen pixel (via the 3D viewport's WINDOW region)."""
    from bpy_extras import view3d_utils
    a = area3d()
    r = reg(a, 'WINDOW')
    rv = a.spaces.active.region_3d
    co = view3d_utils.location_3d_to_region_2d(r, rv, skin.matrix_world @ Vector(p_local))
    if co is None:
        return None
    return int(r.x + co.x), int(SH - (r.y + co.y))


def pencil_drag_cmds(lines):
    cmds = []
    for pts in lines:
        scr = [to_screen(p) for p in pts if p is not None]
        scr = [s for s in scr if s is not None]
        if len(scr) < 2:
            continue
        cmds += ['xdotool mousemove %d %d' % scr[0], 'sleep 0.3', 'xdotool mousedown 1', 'sleep 0.15']
        for (x0, y0), (x1, y1) in zip(scr, scr[1:]):
            for k in range(1, 9):
                cmds += ['xdotool mousemove %d %d' % (x0 + (x1 - x0) * k // 8, y0 + (y1 - y0) * k // 8), 'sleep 0.05']
        cmds += ['sleep 0.2', 'xdotool mouseup 1', 'sleep 0.6']
    return cmds


SHOTS = []   # filled in start


def set_card_state(expanded_keys):
    for c in ms.cards:
        c.expanded = c.key in expanded_keys


def act(key):
    i = next((j for j, r in enumerate(ms.regions) if r.key == key), None)
    if i is not None:
        ms.index = i


def wait_stable(min_s=2.5):
    """True once the screen stopped changing for two consecutive grabs (software GL is slow on 160k vertices)."""
    if time.time() - S['t'] < min_s:
        return False
    h = grab(None, (0, 0, SW, SH))
    if h == S['last']:
        S['stable'] += 1
    else:
        S['stable'] = 0
    S['last'] = h
    return S['stable'] >= 1


def step():
    try:
        return _step()
    except Exception:
        import traceback
        traceback.print_exc()
        REPORT['notes'].append("exception: " + traceback.format_exc().splitlines()[-1])
        finish()
        return None


def finish():
    with open(os.path.join(OUT, 'capture_report.json'), 'w') as f:
        json.dump(REPORT, f, indent=1)
    log("DONE", json.dumps(REPORT))
    if '--quit' in argv:
        bpy.ops.wm.quit_blender()


def _step():
    ph = S['phase']
    if ph == 'start':
        bpy.context.preferences.view.show_splash = False
        sc.arc_flow.ui_tab = 'MUSCLE'
        view_setup('overview')
        S['phase'], S['t'] = 'tab', time.time()
        S['proc'] = click_tab(S['dy'])
        return 2.0
    if ph == 'tab':
        if S['proc'] is not None and S['proc'].poll() is None:
            return 0.3
        if time.time() - DRAWN['t'] < 3.0 and DRAWN['t'] > S['t']:
            log("Tessera tab visible after click at dy", S['dy'])
            S['phase'] = 'widen'
            S['proc'] = widen()
            return 1.0
        S['dy'] += 30
        if S['dy'] > 700:
            REPORT['notes'].append("could not find the Tessera sidebar tab")
            finish()
            return None
        S['t'] = time.time()
        S['proc'] = click_tab(S['dy'])
        return 1.2
    if ph == 'widen':
        if S['proc'].poll() is None:
            return 0.3
        a, ui, top = ui_geom()
        log("sidebar width now", ui.width)
        n = prepare_strokes()
        REPORT['prepared_lines'] = n
        # interactive Muscle Pencil on the cheek (real modal operator + XTEST drags)
        act('zygomaticus')
        set_card_state({'HEAD'})
        view_setup('face')
        S['phase'], S['t'] = 'pencil_wait', time.time()
        return 2.5
    if ph == 'pencil_wait':
        a = area3d()
        r = reg(a, 'WINDOW')
        i = ms.index
        S['n0'] = ms.regions[i].n_strokes
        with bpy.context.temp_override(window=win(), area=a, region=r):
            res = bpy.ops.arcflow.muscle_contour('INVOKE_DEFAULT')
        log("pencil invoke", res)
        lines = [[face_pt(-0.12, 0.27), face_pt(-0.22, 0.40), face_pt(-0.32, 0.52)],
                 [face_pt(-0.14, 0.33), face_pt(-0.23, 0.45), face_pt(-0.31, 0.57)]]
        cmds = pencil_drag_cmds(lines) + ['sleep 0.5', 'xdotool key Escape', 'sleep 0.3', 'xdotool mousemove 900 1070']
        S['proc'] = xdo(cmds)
        S['phase'] = 'pencil_drag'
        return 0.5
    if ph == 'pencil_drag':
        if S['proc'].poll() is None:
            return 0.3
        i = ms.index
        rg = ms.regions[i]
        st = [s for s in MO.load_strokes(skin) if s.get('cid') == rg.cid]
        edges = {tuple(sorted(e.vertices)) for e in skin.data.edges}
        good = all(tuple(sorted(p)) in edges for s in st[-2:] for p in zip(s['verts'], s['verts'][1:]))
        REPORT['pencil_modal'] = dict(muscle=rg.label, lines_before=S['n0'], lines_after=rg.n_strokes,
                                      surface_following=bool(good),
                                      verts_per_line=[len(s['verts']) for s in st[-2:]])
        log("interactive pencil:", REPORT['pencil_modal'])
        rg.completed = rg.n_strokes > 0
        act('masseter')
        S['phase'], S['shot'] = 'shot_setup', 0
        return 0.5
    if ph == 'shot_setup':
        if S['shot'] >= len(SHOTS):
            finish()
            return None
        shot = SHOTS[S['shot']]
        shot['setup']()
        S['proc'] = scroll_sidebar(-60)
        S['phase'] = 'shot_scroll0'
        return 0.5
    if ph == 'shot_scroll0':
        if S['proc'].poll() is None:
            return 0.3
        shot = SHOTS[S['shot']]
        S['proc'] = scroll_sidebar(shot.get('scroll', 0)) if shot.get('scroll', 0) else None
        S['phase'], S['t'], S['stable'], S['last'] = 'shot_wait', time.time(), 0, None
        return 0.5
    if ph == 'shot_wait':
        if S['proc'] is not None and S['proc'].poll() is None:
            return 0.3
        redraw()
        if not wait_stable(3.0):
            return 1.0
        shot = SHOTS[S['shot']]
        path = os.path.join(OUT, shot['name'])
        grab(path)
        REPORT['shots'].append(path)
        log("shot", path)
        S['shot'] += 1
        S['phase'] = 'shot_setup'
        return 0.3
    return None


def s_overview():
    act('zygomaticus')
    set_card_state({'HEAD'})
    view_setup('overview')


def s_face():
    act('zygomaticus')
    set_card_state({'HEAD'})
    view_setup('face')


def s_arm():
    act('biceps')
    set_card_state({'ARM'})
    view_setup('arm')


def s_solve():
    set_card_state(set())
    act('biceps')
    view_setup('solve')


def s_detail():
    MUI.switch_preset(sc, detail='SCIENTIFIC' if 'SCIENTIFIC' in MS.DETAIL_TIER else 'CINEMATIC')
    set_card_state({'HEAD', 'HAND'})
    act('zygomaticus')
    view_setup('face')


SHOTS[:] = [
    dict(name='muscle_01_tab_overview.png', setup=s_overview),
    dict(name='muscle_02_pencil_face.png', setup=s_face, scroll=9),
    dict(name='muscle_03_arm_card_flex_accent.png', setup=s_arm, scroll=9),
    dict(name='muscle_04_solve_outputs.png', setup=s_solve, scroll=10),
    dict(name='muscle_05_scientific_detail.png', setup=s_detail, scroll=4),
]

bpy.app.timers.register(step, first_interval=4.0)
