"""Showcase 3 — FLUID (cohesive solid-body water + whitewater foam) and COMBUSTION (fighter wind
tunnel + aerodynamic friction heat + wispy curve streamlines) as full-UI Cycles viewport stills.

The scenes are built by docs/render_friction_heat.py (via export_fh_blends.py) and appended here,
so the capture shows exactly the add-on output. Env TS_STILL=<blend path>, TS_TAB=FLUID|COMBUSTION.
"""
import os

import bpy
from mathutils import Vector

from common import addon

BLEND = os.environ.get('TS_STILL', '')
TAB = os.environ.get('TS_TAB', 'FLUID')


def build():
    addon()
    old = bpy.context.scene
    with bpy.data.libraries.load(BLEND, link=False) as (src, dst):
        dst.scenes = list(src.scenes)
    sc = dst.scenes[0]
    bpy.context.window_manager.windows[0].scene = sc
    try:
        bpy.data.scenes.remove(old)
    except Exception:
        pass
    sc.arc_flow.ui_tab = TAB
    st = dict(scene=sc, step=lambda f: None, status=lambda: TAB, properties=os.environ.get('TS_PROPS', 'MATERIAL'))
    return st


def ui(st, sp, r3, ctx):
    sc = st['scene']
    sc.arc_flow.ui_tab = TAB
    cam = sc.camera
    r3.view_perspective = 'PERSP'
    if cam is not None:
        M = cam.matrix_world
        fwd = M.to_3x3() @ Vector((0, 0, -1))
        dist = float(os.environ.get('TS_DIST', '2.0'))
        r3.view_rotation = M.to_quaternion()
        right = M.to_3x3() @ Vector((1, 0, 0))
        shift = float(os.environ.get('TS_SHIFT', '0.0'))
        r3.view_location = M.translation + fwd * dist + right * shift * dist
        r3.view_distance = dist
        sp.lens = cam.data.lens * float(os.environ.get('TS_LENS_SCALE', '0.8'))
    act = os.environ.get('TS_ACTIVE')
    if act and act in sc.objects:
        bpy.context.view_layer.objects.active = sc.objects[act]
    for o in sc.objects:
        o.select_set(False)
