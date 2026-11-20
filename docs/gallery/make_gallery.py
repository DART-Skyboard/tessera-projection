#!/usr/bin/env python3
"""Tessera Projection gallery media: real Blender 5.3 UI screenshots + demo video frames, captured from a GUI session
running on a virtual X display (software OpenGL).

Host mode (plain python3):  `python3 docs/gallery/make_gallery.py all` runs everything (about 30 min on 2 cores).
    python3 docs/gallery/make_gallery.py capture [step ...]     build the 0.4.0 zip, install it into a clean config, run the GUI,
                                                                grab raw 1920x1080 frames into $GALLERY_WORK (default /tmp/tessera_gallery)
    python3 docs/gallery/make_gallery.py assemble               write the PNGs and the featured image into docs/gallery/ (needs PIL)
Blender mode (started by the host with --python): drives the UI with bpy.app.timers and grabs the X root window with ImageMagick `import`.
Env: BLENDER (path to blender 5.3), GALLERY_WORK, GALLERY_VIDEO (output mp4)."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
WORK = os.environ.get('GALLERY_WORK', '/tmp/tessera_gallery')
RES = os.environ.get('GALLERY_RES', '/tmp/tessera_gallery_res')
BLENDER = os.environ.get('BLENDER', '/home/claude/blender53/blender-5.3.0-linux-x64/blender')

try:
    import bpy
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False


# ================================================================================ Blender side
if IN_BLENDER:
    import math
    import time

    from mathutils import Vector

    SCALE = float(os.environ.get('GALLERY_UI_SCALE', '1.0'))

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

    def override(a=None):
        a = a or area('VIEW_3D')
        return bpy.context.temp_override(window=win(), screen=win().screen, area=a, region=region(a))

    def redraw():
        for a in win().screen.areas:
            a.tag_redraw()

    def shotg(name, away=True):
        """Generator version: moves the pointer away and lets Blender process it (tooltips close) before the grab."""
        if away:
            mouse(600, 900)
            yield 0.4
            mouse(610, 905)
            yield 1.5
        shot(name, away=False)

    def shot(name, hold=1.0, keep=True, away=True):
        """Grab the whole X screen (the real Blender window).  name 'x' -> work/x.png ; frames go to frames/<seq>_NNNN.png with a hold time."""
        if away:
            mouse(600, 900)
            time.sleep(0.5)
        subprocess.run(['import', '-window', 'root', os.path.join(WORK, name + '.png')], check=True)
        print("SHOT", name, flush=True)

    # ---- real mouse / key events through the X server (XTEST), so the UI is driven like a user would
    _xd = []

    def _x():
        if not _xd:
            sys.path.insert(0, os.environ.get('XLIB_PATH', '/tmp/tessera_gallery_xlib'))
            from Xlib import X, display
            from Xlib.ext import xtest
            d = display.Display()
            _xd.extend([d, X, xtest])
            focus()
        return _xd

    def focus():
        """No window manager on the virtual display: give the Blender window the input focus ourselves."""
        d, X, _xt = _xd
        found = []

        def walk(w, depth=0):
            for c in w.query_tree().children:
                try:
                    nm = c.get_wm_name()
                except Exception:
                    nm = None
                if nm and 'Blender' in str(nm):
                    found.append(c)
                if depth < 2:
                    walk(c, depth + 1)
        walk(d.screen().root)
        print("X windows:", [str(w.get_wm_name()) for w in found], flush=True)
        if found:
            found[0].set_input_focus(X.RevertToParent, X.CurrentTime)
            d.sync()

    def mouse(x, y):
        d, X, xt = _x()
        xt.fake_input(d, X.MotionNotify, x=int(x), y=int(y))
        d.sync()

    def _key(sym, down):
        d, X, xt = _x()
        xt.fake_input(d, X.KeyPress if down else X.KeyRelease, d.keysym_to_keycode(sym))
        d.sync()

    def _btn(b, down):
        d, X, xt = _x()
        xt.fake_input(d, X.ButtonPress if down else X.ButtonRelease, b)
        d.sync()

    # the helpers below are generators (use `yield from`): Blender must get time to process each event
    def click(x, y, button=1, ctrl=False):
        mouse(x - 3, y - 2)
        yield 0.3
        mouse(x, y)
        yield 0.4
        if ctrl:
            _key(0xffe3, True)
            yield 0.2
        _btn(button, True)
        yield 0.15
        _btn(button, False)
        yield 0.3
        if ctrl:
            _key(0xffe3, False)
            yield 0.2

    def drag(x0, y0, x1, y1, steps=10):
        mouse(x0 + 2, y0)
        yield 0.3
        mouse(x0, y0)
        yield 0.4
        _btn(1, True)
        yield 0.3
        for i in range(1, steps + 1):
            mouse(x0 + (x1 - x0) * i / steps, y0 + (y1 - y0) * i / steps)
            yield 0.15
        _btn(1, False)
        yield 0.5

    def scroll(x, y, n, ctrl=False):
        """n > 0 scrolls down (button 5), n < 0 up (button 4)."""
        mouse(x, y)
        yield 0.4
        if ctrl:
            _key(0xffe3, True)
            yield 0.2
        for _i in range(abs(n)):
            _btn(5 if n > 0 else 4, True)
            _btn(5 if n > 0 else 4, False)
            yield 0.25
        if ctrl:
            _key(0xffe3, False)
            yield 0.2

    def sidebar(on=True, category="Tessera"):
        a = area('VIEW_3D')
        a.spaces[0].show_region_ui = on
        r = region(a, 'UI')
        try:
            r.active_panel_category = category
        except Exception as ex:
            print("category", ex)

    def layout():
        sp = area('VIEW_3D').spaces[0]
        sp.shading.type = 'SOLID'
        sp.overlay.show_overlays = True
        sidebar(True)

    def view(center, dist, yaw=35.0, pitch=70.0, lens=50.0, shift_px=230):
        """Place the viewport camera (yaw about Z, pitch from +Z); the scene is shifted left so the sidebar does not cover it."""
        from mathutils import Euler
        r3 = area('VIEW_3D').spaces[0].region_3d
        r3.view_perspective = 'PERSP'
        q = Euler((math.radians(pitch), 0, math.radians(yaw)), 'XYZ').to_quaternion()
        r3.view_rotation = q
        right = q @ Vector((1, 0, 0))
        r3.view_location = Vector(center) + right * (shift_px * dist * 0.72 * 50.0 / lens / 1574.0)
        r3.view_distance = dist
        area('VIEW_3D').spaces[0].lens = lens

    def clear_scene():
        for o in list(bpy.data.objects):
            bpy.data.objects.remove(o, do_unlink=True)

    def boot(open_ids=None, widen=True):
        """Wait for the window, close the splash, set which Tessera panels start open, switch the sidebar to the Tessera tab, widen it."""
        yield 4.0
        mouse(900, 500)
        yield 1.0
        _key(0xff1b, True); _key(0xff1b, False)
        yield 1.0
        mouse(300, 700)
        yield 1.0
        sp = area('VIEW_3D').spaces[0]
        sp.show_region_ui = True
        sp.shading.type = 'SOLID'
        if open_ids is not None:
            panel_states(open_ids)     # before the Tessera tab is first drawn, so the panels start in this state
        yield 2.0
        yield from scroll(1562, 300, 4, ctrl=True)
        yield 1.0
        if widen:
            yield from drag(1285, 200, 1130, 200)
        yield 1.0

    def suzanne_tris():
        bpy.ops.mesh.primitive_monkey_add(size=2.0, location=(0, 0, 0))
        o = bpy.context.object
        o.name = "Suzanne_tris"
        m = o.modifiers.new("sub", 'SUBSURF')
        m.levels = 1
        bpy.ops.object.modifier_apply(modifier="sub")
        m = o.modifiers.new("tri", 'TRIANGULATE')
        bpy.ops.object.modifier_apply(modifier="tri")
        return o

    def solid_wire(on=True):
        sp = area('VIEW_3D').spaces[0]
        sp.shading.type = 'SOLID'
        sp.shading.color_type = 'MATERIAL'
        sp.shading.show_xray = False
        sp.overlay.show_wireframes = on
        sp.overlay.wireframe_threshold = 1.0
        sp.overlay.show_floor = True

    def wait_for(cond, timeout=120.0, every=1.0):
        t0 = time.time()
        while not cond() and time.time() - t0 < timeout:
            yield every

    def material_view():
        sp = area('VIEW_3D').spaces[0]
        sp.shading.type = 'MATERIAL'
        sp.overlay.show_wireframes = False

    def solid_color(color_type='OBJECT'):
        sp = area('VIEW_3D').spaces[0]
        sp.shading.type = 'SOLID'
        sp.shading.color_type = color_type
        sp.shading.show_xray = False
        sp.overlay.show_wireframes = False

    def tessera_panels():
        """All Tessera sidebar panel classes of the 3D viewport, in registration order (parents before children)."""
        out = []
        def walk(c):
            for k in c.__subclasses__():
                walk(k)
                out.append(k)
        walk(bpy.types.Panel)
        res = [c for c in out if getattr(c, 'bl_category', '') == 'Tessera' and getattr(c, 'bl_space_type', '') == 'VIEW_3D']
        return res

    def panel_states(open_ids):
        """Open exactly the sidebar panels in open_ids (others start collapsed): the panels are re-registered with DEFAULT_CLOSED set
        accordingly, which is what the add-on's own bl_options do on a fresh start."""
        cls = tessera_panels()
        for c in reversed(cls):
            try:
                bpy.utils.unregister_class(c)
            except Exception as ex:
                print("unreg", c, ex)
        for c in cls:
            opts = set(getattr(c, 'bl_options', ())) - {'DEFAULT_CLOSED'}
            if c.bl_idname not in open_ids:
                opts.add('DEFAULT_CLOSED')
            c.bl_options = opts
            bpy.utils.register_class(c)
        print("PANELS", [c.bl_idname for c in cls], flush=True)

    def poke_sidebar():
        """Force the sidebar to rebuild its panel list (mode switches change which panels exist)."""
        sp = area('VIEW_3D').spaces[0]
        sp.show_region_ui = False
        sp.show_region_ui = True
        redraw()
        mouse(1400, 600)
        yield 1.0
        mouse(1410, 610)
        yield 1.5

    def fit_view(pitch=70.0, yaw=25.0, factor=2.4):
        """Frame every visible mesh object (left of the sidebar)."""
        pts = [o.matrix_world @ Vector(c) for o in bpy.data.objects if o.type == 'MESH' and not o.hide_get() for c in o.bound_box]
        lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
        hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
        view((lo + hi) / 2, max((hi - lo).length * factor * 0.5, 0.2), yaw=yaw, pitch=pitch)

    def op(fn, *a, **k):
        """Call a bpy operator with the 3D viewport as context (what a button in its sidebar does)."""
        with override():
            return fn(*a, **k)

    def run_until_frame(n, timeout=300):
        yield from wait_for(lambda: bpy.context.scene.frame_current >= n, timeout, 1.0)
        yield 1.0

    def stop_play():
        try:
            if bpy.context.screen.is_animation_playing:
                bpy.ops.screen.animation_play()
        except Exception:
            pass

    def fm_stop():
        op(bpy.ops.arcflow.fm_stop)
        stop_play()

    # ------------------------------------------------------------------ scenes
    def step_remesh():
        yield from boot({'AF_PT_retopo'})
        clear_scene()
        src = suzanne_tris()
        for v in src.data.vertices:            # a slightly asymmetric scan-like mesh, so Symmetrize has something to fix
            if v.co.x > 0.2:
                v.co.y += 0.012 * math.sin(7 * v.co.z)
        src.select_set(False)
        solid_wire()
        view((0, 0, 0.1), 6.0, yaw=18, pitch=82)
        yield 3.0
        yield from shotg('01_remesh_before')
        op(bpy.ops.arcflow.quad_analyse)
        yield 2.5
        yield from shotg('01_remesh_analyse')
        before = set(bpy.data.objects)
        op(bpy.ops.arcflow.quad_remesh)
        yield from wait_for(lambda: set(bpy.data.objects) - before, 120)
        new = list(set(bpy.data.objects) - before)
        yield 2.0
        if new:
            new[0].select_set(False)
            src.location.x = -1.35
            new[0].location.x = 1.35
        view((0, 0, 0.1), 7.2, yaw=10, pitch=84)
        yield 4.0
        yield from shotg('01_projection_remesh')

    def engine_scene(loader, open_ids, dist, center):
        yield from boot(open_ids)
        clear_scene()
        loader()
        material_view()
        view(center, dist)
        yield 3.0

    def step_single():
        yield from engine_scene(lambda: op(bpy.ops.arcflow.demo), {'AF_PT_main', 'AF_PT_systems'}, 22, (3.0, 0, 1.0))
        op(bpy.ops.arcflow.systems_detect)
        op(bpy.ops.arcflow.build)
        yield from run_until_frame(14)
        yield from shotg('02_single_engine')
        op(bpy.ops.arcflow.stop)
        stop_play()
        yield 1.0

    def step_twin():
        yield from engine_scene(lambda: op(bpy.ops.arcflow.demo_twin), {'AF_PT_main', 'AF_PT_systems'}, 26, (5.5, 0, 1.0))
        op(bpy.ops.arcflow.build)
        yield from run_until_frame(22)
        yield from shotg('03_twin_enhanced')
        op(bpy.ops.arcflow.stop)
        stop_play()
        yield 2.0
        bpy.context.scene.frame_set(1)
        bpy.context.scene.arc_flow.sys_variant = 'BAD'      # crossfeed variant: engine A's hot chamber into engine B's fuel tank
        yield 2.0
        op(bpy.ops.arcflow.build)
        yield from run_until_frame(22)
        yield from shotg('03_twin_failure')
        op(bpy.ops.arcflow.stop)
        stop_play()
        yield 1.0

    def step_env():
        yield from engine_scene(lambda: op(bpy.ops.arcflow.demo), {'AF_PT_main', 'AF_PT_env'}, 22, (3.0, 0, 1.0))
        bpy.context.scene.arc_flow.env.preset = 'MARS'
        yield 6.0
        yield from shotg('04_environment')
        yield from click(1330, 551)       # open the planet preset menu
        yield 2.0
        yield from shotg('04_environment_presets')
        _key(0xff1b, True); _key(0xff1b, False)
        yield 1.0

    def step_cavity():
        yield from engine_scene(lambda: op(bpy.ops.arcflow.demo), {'AF_PT_main', 'AF_PT_solver'}, 22, (3.0, 0, 1.0))
        solid_wire(False)
        area('VIEW_3D').spaces[0].shading.color_type = 'MATERIAL'
        bpy.context.scene.arc_flow.bool_projection = True
        op(bpy.ops.arcflow.build_cavities)
        yield 4.0
        sp = area('VIEW_3D').spaces[0]
        sp.shading.show_xray = True
        yield 3.0
        yield from shotg('07_fluid_cavity')

    FLUID_OPEN = {'AF_PT_fm_mode', 'AF_PT_fm_main', 'AF_PT_fm_fluids'}

    def fluid_scene(loader, open_ids, center, dist, nframes, shot_name, yaw=15.0, pitch=88.0, hide=('Carrier',), colour=True, build=True, mode='FLUID', camera_view=False):
        yield from boot(open_ids)
        clear_scene()
        loader()
        for o in bpy.data.objects:
            if o.type == 'CAMERA':
                o.hide_set(True)      # camera gizmos only clutter the viewport
        bpy.context.scene.arc_flow_fm.mode = mode
        if colour:
            solid_color('OBJECT')
        if camera_view:
            sp = area('VIEW_3D').spaces[0]
            sp.region_3d.view_perspective = 'CAMERA'
            sp.shading.type = 'SOLID'
            sp.shading.color_type = 'VERTEX'
        elif center is None:
            fit_view(pitch=pitch)
        else:
            view(center, dist, yaw=yaw, pitch=pitch)
        yield 2.0
        yield from poke_sidebar()
        if build:
            op(bpy.ops.arcflow.fm_build)
            yield from wait_for(lambda: bpy.context.scene.frame_current >= 2, 400)
            for fl in bpy.context.scene.arc_flow_fm.fluids:
                ob = bpy.data.objects.get("AFFM_" + fl.name)
                if ob is not None and any(fl.name.startswith(h) for h in hide):
                    ob.hide_set(True)      # viewport only: the opaque carrier-liquid particles would hide the wax blobs
                if ob is not None and colour:
                    ob.color = (fl.color[0], fl.color[1], fl.color[2], 1.0)
            yield from run_until_frame(nframes, 500)
        yield from shotg(shot_name)
        if build:
            fm_stop()
            yield 1.0

    def FOmod():
        import importlib
        return importlib.import_module('bl_ext.user_default.tessera_projection.fluidmode_ops')

    def step_lava():
        yield from fluid_scene(lambda: FOmod().load_lava_lamp(bpy.context.scene, max_particles=4000), FLUID_OPEN, (0, 0, 0.17), 0.9, 30, '05_fluid_lava_lamp', pitch=88)

    def step_mix():
        yield from fluid_scene(lambda: FOmod().load_colour_mixing(bpy.context.scene, max_particles=2500), FLUID_OPEN, (0, 0, 0.09), 0.55, 14, '06_colour_mixing')

    def step_components():
        yield from fluid_scene(lambda: op(bpy.ops.arcflow.srf_example, which='EMIT'), {'AF_PT_fm_mode', 'AF_PT_fm_main', 'AF_PT_srf'}, None, 0, 0,
                               '08a_components_surfaces', colour=False, build=False, pitch=70)

    def step_shader():
        def load():
            FOmod().load_colour_mixing(bpy.context.scene, max_particles=2500)
            sh = bpy.context.scene.arc_flow_sh
            sh.preset = 'GLASS'
            op(bpy.ops.arcflow.shader_fluids_auto)
            g = bpy.data.objects.get('TS_Glass')
            if g is not None:
                bpy.context.view_layer.objects.active = g
                g.select_set(True)
                op(bpy.ops.arcflow.shader_assign, target='ACTIVE')      # the preset's tunable parameters then show in the panel
        yield from fluid_scene(load, {'AF_PT_shader'}, (0, 0, 0.09), 0.55, 0, '08b_shader_presets', colour=False, build=False, pitch=80, mode='COMBUSTION')

    def step_examples():
        def load():
            op(bpy.ops.arcflow.example_load, key='WIND_TUNNEL')
        yield from fluid_scene(load, {'AF_PT_examples'}, None, 0, 0, '08c_examples', colour=False, build=False, pitch=70, mode='COMBUSTION')

    def step_windtunnel():
        def load():
            op(bpy.ops.arcflow.example_load, key='WIND_TUNNEL')
            a = bpy.data.objects.get('Aircraft Airframe')
            if a is not None:                      # active object -> the Thermal readback panel
                bpy.context.view_layer.objects.active = a
                a.select_set(True)
                th = a.arc_flow_th
                th.on = True
                th.legend = True                   # colour bar (attached to the scene camera)
            cam = bpy.data.objects.new("Gallery_Camera", bpy.data.cameras.new("Gallery_Camera"))
            bpy.context.scene.collection.objects.link(cam)
            cam.data.lens = 32
            cam.location = (0.0, -6.2, 1.3)
            cam.rotation_euler = (math.radians(88), 0, 0)
            bpy.context.scene.camera = cam
            bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y = 1280, 720
        yield from fluid_scene(load, {'AF_PT_fm_mode', 'AF_PT_fm_main', 'AF_PT_thermal'}, None, 0, 14, '08d_wind_tunnel_thermal', colour=False, pitch=75,
                               hide=('Air',), camera_view=True)

    def step_crystal():
        yield from fluid_scene(lambda: FOmod().load_crystal(bpy.context.scene, max_particles=3000), FLUID_OPEN, (0, 0, 0.05), 0.5, 24, '08e_transmutation_crystal', pitch=75)

    def step_steam():
        yield from fluid_scene(lambda: FOmod().load_steam(bpy.context.scene, max_particles=3000), FLUID_OPEN, (0, 0, 0.15), 0.9, 24, '08f_transmutation_steam', pitch=80)

    def step_render():
        yield from engine_scene(lambda: op(bpy.ops.arcflow.demo), {'AF_PT_main', 'AF_PT_systems'}, 22, (3.0, 0, 1.0))
        solid_wire(False)
        op(bpy.ops.arcflow.build)
        yield from run_until_frame(10)
        op(bpy.ops.arcflow.stop)
        stop_play()
        yield 2.0
        stage_render()
        bpy.context.preferences.view.render_display_type = 'AREA'
        done = []
        bpy.app.handlers.render_complete.append(lambda *a: done.append(1))
        with override():
            bpy.ops.render.render('INVOKE_DEFAULT')
        yield from wait_for(lambda: done, 900, 2.0)
        yield 4.0
        yield from shotg('08_render_window')

    def stage_render():
        sc = bpy.context.scene
        sc.render.engine = 'CYCLES'
        sc.cycles.device = 'CPU'
        sc.cycles.samples = 32
        sc.cycles.use_denoising = False
        sc.render.resolution_x, sc.render.resolution_y = 960, 540
        sc.render.resolution_percentage = 100
        sc.view_settings.view_transform = 'Standard'
        w = bpy.data.worlds.new("studio")
        w.use_nodes = True
        bg = w.node_tree.nodes["Background"]
        bg.inputs[0].default_value = (0.012, 0.014, 0.022, 1)
        sc.world = w
        c = Vector((3.0, 0, 1.0))
        cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
        sc.collection.objects.link(cam)
        cam.location = c + Vector((math.sin(0.35), -math.cos(0.35), 0.45)) * 17
        cam.rotation_euler = (c - cam.location).to_track_quat('-Z', 'Y').to_euler()
        cam.data.lens = 40
        sc.camera = cam
        for nm, loc, en, col in (("key", (-3, -9, 9), 2600, (1.0, 0.93, 0.85)), ("rim", (12, 7, 5), 1800, (0.55, 0.7, 1.0))):
            ld = bpy.data.lights.new(nm, 'AREA')
            ld.energy, ld.color, ld.size = en, col, 4.0
            lo = bpy.data.objects.new(nm, ld)
            sc.collection.objects.link(lo)
            lo.location = loc
            lo.rotation_euler = (c - lo.location).to_track_quat('-Z', 'Y').to_euler()

    def build_gn_example():
        ng = bpy.data.node_groups.new("Tessera Fluid Example", 'GeometryNodeTree')
        ng.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
        ng.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
        n = ng.nodes
        gin, gout = n.new('NodeGroupInput'), n.new('NodeGroupOutput')
        m2p = n.new('GeometryNodeMeshToPoints')
        rad = n.new('GeometryNodeSetPointRadius')
        cloud = n.new('GeometryNodeGroup')
        import importlib
        TN = importlib.import_module('bl_ext.user_default.tessera_projection.tessera_nodes')
        cloud.node_tree = TN.ensure_group('Tessera Particle Cloud')
        cloud.label = "Particle Cloud"
        thermal = n.new('GeometryNodeGroup')
        thermal.node_tree = TN.ensure_group('Tessera Thermal')
        thermal.label = "Thermal"
        for node, x in ((gin, -900), (m2p, -650), (rad, -420), (cloud, -190), (thermal, 120), (gout, 420)):
            node.location = (x, 0)

        def first(sockets, typ):
            return next(sk for sk in sockets if sk.type == typ)
        L = ng.links.new
        L(gin.outputs[0], m2p.inputs['Mesh'])
        L(m2p.outputs['Points'], rad.inputs['Points'])
        L(rad.outputs[0], first(cloud.inputs, 'GEOMETRY'))
        L(first(cloud.outputs, 'GEOMETRY'), first(thermal.inputs, 'GEOMETRY'))
        L(first(thermal.outputs, 'GEOMETRY'), gout.inputs[0])
        return ng

    def step_gn():
        yield from boot({'AF_PT_main'}, widen=False)
        clear_scene()
        bpy.ops.mesh.primitive_ico_sphere_add(radius=1.0, location=(0, 0, 0))
        o = bpy.context.object
        op(bpy.ops.arcflow.add_tessera_modifier, group='Tessera Particle Cloud')   # also creates the library groups
        ng = build_gn_example()
        o.modifiers[0].node_group = ng
        bpy.context.window.workspace = bpy.data.workspaces['Geometry Nodes']
        yield 6.0
        for a in win().screen.areas:
            print("AREA", a.type, a.x, a.y, a.width, a.height, flush=True)
        def tap(sym, mods=()):
            for m in mods:
                _key(m, True)
            yield 0.2
            _key(sym, True); yield 0.15; _key(sym, False); yield 0.4
            for m in mods:
                _key(m, False)
            yield 0.3
        mouse(800, 750); yield 0.5
        yield from tap(0x20, (0xffe3,))          # Ctrl+Space: maximise the node editor
        yield 2.0
        mouse(800, 500); yield 0.5
        yield from tap(0xff50)                    # Home: view all
        yield 1.5
        yield from shotg('gn_wired', away=False)
        mouse(400, 300); yield 0.5
        yield from tap(0x41, (0xffe1,))           # Shift+A: Add menu
        yield 1.0
        mouse(330, 700); yield 0.3
        mouse(335, 776); yield 1.8
        yield from shotg('gn_add_menu', away=False)

    STEPS = {k[5:]: v for k, v in list(locals().items()) if k.startswith('step_') and callable(v)}

    def steps():
        only = [x for x in os.environ.get('GALLERY_ONLY', '').split(',') if x] or list(STEPS)
        for k in only:
            yield from STEPS[k]()

    def run():
        gen = steps()

        def tick():
            try:
                return next(gen)
            except StopIteration:
                bpy.ops.wm.quit_blender()
                return None
            except Exception:
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
                os._exit(3)
        bpy.app.timers.register(tick, first_interval=2.0)

    run()


# ================================================================================ host side
else:
    def sh(cmd, **kw):
        print('+', ' '.join(cmd) if isinstance(cmd, list) else cmd, flush=True)
        return subprocess.run(cmd, check=True, **kw)

    def prepare():
        env = dict(os.environ, BLENDER_USER_RESOURCES=RES)
        if not os.environ.get('GALLERY_ZIP'):
            sh([os.path.join(ROOT, 'build.sh'), BLENDER], cwd=ROOT, stdout=subprocess.DEVNULL)
        import glob
        subprocess.run(['rm', '-rf', RES]); os.makedirs(RES)
        import glob
        zipf = os.environ.get('GALLERY_ZIP') or sorted(glob.glob(os.path.join(ROOT, 'dist', 'tessera_projection-*.zip')))[-1]
        sh([BLENDER, '--factory-startup', '-b', '--command', 'extension', 'install-file', '-r', 'user_default', '--enable', zipf], env=env)
        prefs = os.path.join(WORK, 'prefs.py')
        open(prefs, 'w').write(
            "import bpy\np = bpy.context.preferences\np.view.show_splash = False\np.view.ui_scale = %s\n"
            % os.environ.get('GALLERY_UI_SCALE', '1.0'))
        sh([BLENDER, '-b', '--python', prefs], env=env, stdout=subprocess.DEVNULL)
        # Blender 5.3 has no 'MOD_MASS' icon: the Fluids panel of 0.4.0 stops drawing at the "Amount" box.  Gallery-only workaround on the
        # INSTALLED copy (the repo source is untouched): use an icon that exists.
        fp = glob.glob(os.path.join(RES, 'extensions', 'user_default', 'tessera_projection', 'fluidmode_panel.py'))[0]
        txt = open(fp).read()
        open(fp, 'w').write(txt.replace("icon='MOD_MASS'", "icon='MOD_FLUIDSIM'"))
        return env

    def need_xlib():
        xl = os.environ.get('XLIB_PATH', '/tmp/tessera_gallery_xlib')
        if not os.path.isdir(os.path.join(xl, 'Xlib')):
            sh([sys.executable, '-m', 'pip', 'install', '--quiet', '--target', xl, 'python-xlib'])
        return xl

    def capture(only):
        os.makedirs(WORK, exist_ok=True)
        env = prepare()
        env['XLIB_PATH'] = need_xlib()
        names = only or ['remesh', 'single', 'twin', 'env', 'lava', 'mix', 'cavity', 'components', 'shader', 'examples', 'windtunnel', 'crystal', 'steam', 'gn', 'render']
        for name in names:
            env['GALLERY_ONLY'] = name
            for attempt in range(3):      # the machine is shared: a step can be OOM-killed, just run it again
                try:
                    run_gui(env)
                    break
                except subprocess.CalledProcessError as ex:
                    print('step', name, 'failed (%s), attempt %d' % (ex.returncode, attempt + 1), flush=True)

    def run_gui(env):
        sh(['xvfb-run', '-a', '-s', '-screen 0 1920x1080x24', BLENDER, '--gpu-backend', 'opengl', '--window-geometry', '0', '0', '1920', '1080',
            '--python', os.path.abspath(__file__)], env=env)

    # raw capture name -> published name; the list is appendable (add a scene to STEPS in the Blender section, then one line here)
    PUBLISH = [
        ('01_projection_remesh', '01_projection_remesh'), ('01_remesh_before', '01a_remesh_before_triangles'),
        ('02_single_engine', '02_systems_single_engine'), ('03_twin_enhanced', '03_twin_crossfeed_enhanced'),
        ('03_twin_failure', '03b_twin_crossfeed_failure'), ('04_environment_presets', '04_environment_presets'),
        ('05_fluid_lava_lamp', '05_fluid_mode_lava_lamp'), ('06_colour_mixing', '06_colour_mixing'),
        ('07_fluid_cavity', '07_boolean_fluid_cavity'), ('04_environment', '04a_environment_card'),
        ('08a_components_surfaces', '08a_components_surfaces'), ('08b_shader_presets', '08b_shader_presets'),
        ('08c_examples', '08c_examples_dropdown'), ('08d_wind_tunnel_thermal', '08d_wind_tunnel_thermal'),
        ('08e_transmutation_crystal', '08e_transmutation_crystal'), ('08f_transmutation_steam', '08f_transmutation_steam'),
        ('gn_wired', '08g_geometry_nodes_wired'), ('gn_add_menu', '08h_geometry_nodes_add_tessera'),
        ('08_render_window', '09_cycles_render_window'),
    ]
    HERO_SOURCE = '01_projection_remesh'

    def assemble():
        from PIL import Image, ImageDraw, ImageFont
        out = os.path.join(HERE)
        for raw, name in PUBLISH:
            src = os.path.join(WORK, raw + '.png')
            if not os.path.exists(src):
                print('missing', raw)
                continue
            im = Image.open(src).convert('RGB')
            dst = os.path.join(out, name + '.png')
            im.save(dst, optimize=True)
            if os.path.getsize(dst) > 1.4e6:
                im.quantize(256).save(dst, optimize=True)
            print(name, os.path.getsize(dst) // 1024, 'KB')
        # featured image: a real UI screenshot with a title block over the empty part of the viewport
        im = Image.open(os.path.join(WORK, HERO_SOURCE + '.png')).convert('RGBA')
        ov = Image.new('RGBA', im.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        d.rounded_rectangle((60, 150, 640, 400), 18, fill=(18, 20, 26, 215))
        im = Image.alpha_composite(im, ov)
        logo = Image.open(os.path.join(ROOT, 'branding', 'tessera_1024.png')).convert('RGBA').resize((150, 150))
        im.alpha_composite(logo, (90, 180))
        d = ImageDraw.Draw(im)

        def font(sz, bold=True):
            for f in ('/usr/share/fonts/truetype/google-fonts/Poppins-%s.ttf' % ('Bold' if bold else 'Regular'),
                      '/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf' % ('-Bold' if bold else '')):
                if os.path.exists(f):
                    return ImageFont.truetype(f, sz)
            return ImageFont.load_default()
        d.text((265, 190), "Tessera", font=font(56), fill=(255, 255, 255, 255))
        d.text((265, 255), "Projection", font=font(56), fill=(255, 255, 255, 255))
        d.text((90, 355), "Quad remesh, fluids and systems for Blender", font=font(24, False), fill=(190, 200, 215, 255))
        hero = os.path.join(out, 'featured_1920x1080.png')
        im.convert('RGB').save(hero, optimize=True)
        print('featured', os.path.getsize(hero) // 1024, 'KB')

    if __name__ == '__main__':
        cmd = sys.argv[1] if len(sys.argv) > 1 else 'capture'
        if cmd == 'capture':
            capture(sys.argv[2:])
        elif cmd == 'assemble':
            assemble()
        elif cmd == 'all':
            capture([])
            assemble()
