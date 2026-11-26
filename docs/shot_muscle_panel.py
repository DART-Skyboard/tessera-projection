"""Screenshot of the Tessera Muscle tab in a real (software OpenGL) Blender window:

    xvfb-run -a -s "-screen 0 1500x2400x24" sh -c 'blender --factory-startup --gpu-backend opengl --python docs/shot_muscle_panel.py & sleep 45; ffmpeg -f x11grab -video_size 1500x2400 -i $DISPLAY -frames:v 1 out.png; wait'
"""
import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import arc_flow
arc_flow.register()
from arc_flow import muscle_ops as MO

OUT = sys.argv[sys.argv.index('--') + 1] if '--' in sys.argv else '/tmp/muscle_panel.png'
state = {'n': 0}


def step():
    try:
        return _step()
    except Exception as ex:
        import traceback
        traceback.print_exc()
        os._exit(1)


def _step():
    n = state['n']
    state['n'] += 1
    sc = bpy.context.scene
    win = bpy.context.window_manager.windows[0]
    area = next((a for a in win.screen.areas if a.type == 'VIEW_3D'), None)
    if n == 0:
        bpy.ops.arcflow.muscle_example()
        bpy.ops.arcflow.muscle_auto_seed()
        bpy.ops.arcflow.muscle_solve()
        sc.arc_flow.ui_tab = 'MUSCLE'
        # the sidebar category tab cannot be switched from Python: host the Tessera panel in the Item tab for this picture
        cls = bpy.types.AF_PT_tabs
        bpy.utils.unregister_class(cls)
        cls.bl_category = 'Item'
        bpy.utils.register_class(cls)
        bpy.context.preferences.view.ui_scale = 0.8
        sc.arc_flow_muscle.index = 10
        sp = area.spaces.active
        sp.show_region_ui = True
        skin = sc.arc_flow_muscle.skin
        MO.set_active_color(skin, MO.ATTR_SOLVED_COLOR)
        m = bpy.data.materials.new("MuscleSkin")
        m.use_nodes = True
        at = m.node_tree.nodes.new('ShaderNodeAttribute')
        at.attribute_name = MO.ATTR_SOLVED_COLOR
        m.node_tree.links.new(at.outputs['Color'], m.node_tree.nodes['Principled BSDF'].inputs['Base Color'])
        skin.data.materials.append(m)
        sp.shading.type = 'MATERIAL'   # Material Preview (EEVEE)
        sp.region_3d.view_perspective = 'PERSP'
        bpy.ops.object.select_all(action='DESELECT')
        for nm in ('Cube', 'Light', 'Camera'):
            if nm in bpy.data.objects:
                bpy.data.objects.remove(bpy.data.objects[nm], do_unlink=True)
        for o in (skin, sc.arc_flow_muscle.armature):
            o.select_set(True)
        for r in area.regions:
            if r.type == 'UI':
                print("UI region", r.width, [c for c in dir(r) if 'categ' in c])
                try:
                    r.active_panel_category = 'Tessera'
                    print("category now", r.active_panel_category)
                except Exception as ex:
                    print("category", ex)
                try:
                    r.width = 520
                except Exception as ex:
                    print("width", ex)
        reg = next(r for r in area.regions if r.type == 'WINDOW')
        with bpy.context.temp_override(window=win, area=area, region=reg):
            bpy.ops.view3d.view_axis(type='FRONT')
            bpy.ops.view3d.view_selected()
        with bpy.context.temp_override(window=win, area=area):
            bpy.ops.screen.screen_full_area(use_hide_panels=False)
        return 0.5
    if n == 1:
        for a in win.screen.areas:
            a.tag_redraw()
        return 1.0
    if n >= 2:
        for a in win.screen.areas:
            a.tag_redraw()
        if n == 3:
            print("READY", flush=True)
        if n > 40:                      # an external grabber (ffmpeg x11grab) takes the picture
            bpy.ops.wm.quit_blender()
            return None
        return 1.5
    return None


bpy.app.timers.register(step, first_interval=2.0)
