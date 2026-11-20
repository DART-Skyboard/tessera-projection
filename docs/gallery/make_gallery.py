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

    def shot(name, hold=1.0, keep=True):
        """Grab the whole X screen (the real Blender window).  name 'x' -> work/x.png ; frames go to frames/<seq>_NNNN.png with a hold time."""
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

    def boot():
        """Wait for the window, close the splash, open the Tessera sidebar tab and widen the sidebar."""
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
        yield 2.0
        yield from scroll(1562, 300, 4, ctrl=True)
        yield 1.0
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

    def step_remesh():
        yield from boot()
        clear_scene()
        src = suzanne_tris()
        src.select_set(False)
        solid_wire()
        view((0, 0, 0.1), 6.0, yaw=18, pitch=82)
        yield from click(1212, 97)        # collapse the main panel
        yield from click(1212, 125)       # open Projection Remesh
        yield 2.5
        shot('01_remesh_before')
        yield from click(1158, 465)       # Symmetry on
        yield 1.5
        yield from click(1253, 159)       # Analyse Topology
        yield 2.0
        shot('01_remesh_analyse')
        before = set(bpy.data.objects)
        yield from click(1448, 159)       # Projection Remesh
        yield from wait_for(lambda: set(bpy.data.objects) - before, 90)
        new = list(set(bpy.data.objects) - before)
        print("NEW OBJECTS", [o.name for o in new], flush=True)
        yield 2.0
        if new:
            new[0].select_set(False)
            src.location.x = -1.35
            new[0].location.x = 1.35
        view((0, 0, 0.1), 7.2, yaw=10, pitch=84)
        yield 4.0
        shot('01_projection_remesh')

    def material_view():
        sp = area('VIEW_3D').spaces[0]
        sp.shading.type = 'MATERIAL'
        sp.overlay.show_wireframes = False

    def collapse_extra():
        """Tessera main panel at boot: Part and Fluids sub-panels open; close them (bottom first, so positions do not shift)."""
        yield from click(1377, 356)
        yield from click(1380, 293)
        yield 1.5

    def run_until_frame(n, timeout=240):
        yield from wait_for(lambda: bpy.context.scene.frame_current >= n, timeout, 1.0)
        yield 1.0

    def step_single():
        yield from boot()
        clear_scene()
        yield from collapse_extra()
        yield from click(1350, 532)       # Load Demo: Single Engine
        material_view()
        view((3.0, 0, 1.0), 22)
        yield 2.0
        yield from click(1253, 579)       # Re-detect
        yield 2.0
        yield from click(1220, 132)       # Build
        yield from run_until_frame(14)
        shot('02_single_engine')
        yield from click(1220, 132)       # Stop
        yield 1.5

    def step_twin():
        yield from boot()
        clear_scene()
        yield from collapse_extra()
        yield from click(1350, 552)       # Load Demo: Twin Engines + Crossfeed
        material_view()
        view((5.5, 0, 1.0), 26)
        yield 3.0
        yield from click(1220, 132)       # Build
        yield from run_until_frame(18)
        shot('03_twin_enhanced')
        yield from click(1220, 132)       # Stop
        yield 2.0
        bpy.context.scene.arc_flow.sys_variant = 'BAD'     # crossfeed variant: engine A's hot chamber into engine B's fuel tank
        yield 2.0
        yield from click(1220, 132)       # Build again
        yield from run_until_frame(18)
        shot('03_twin_failure')
        yield from click(1220, 132)
        yield 1.5

    def step_env():
        yield from boot()
        clear_scene()
        bpy.ops.arcflow.demo()
        material_view()
        view((3.0, 0, 1.0), 22)
        yield from collapse_extra()
        yield from click(1254, 501)       # close Systems
        yield from click(1210, 397)       # open Environment
        yield 2.0
        bpy.context.scene.arc_flow.env.preset = 'MARS'
        yield 2.0
        yield from click(1330, 488)       # open the planet preset menu
        yield 2.0
        shot('04_environment_presets')
        _key(0xff1b, True); _key(0xff1b, False)
        yield 1.5
        shot('04_environment')

    def fluid_demo(btn_x, name, center, dist, nframes, shot_name, yaw=25.0, pitch=78.0, maxp=None):
        yield from boot()
        clear_scene()
        yield from click(1212, 97)        # collapse the main panel
        yield 1.5
        yield from click(1440, 187)       # Mode: Fluid
        yield 2.0
        if maxp:
            from bl_ext.user_default.tessera_projection import fluidmode_ops as FO   # lighter particle count (shared 8 GB machine)
            {'lava': FO.load_lava_lamp, 'mix': FO.load_colour_mixing}[name](bpy.context.scene, max_particles=maxp)
        else:
            yield from click(btn_x, 222)  # demo button
        yield 3.0
        sp = area('VIEW_3D').spaces[0]      # Solid + object colour: much lighter than Material Preview on this shared 8 GB machine
        sp.shading.type = 'SOLID'
        sp.shading.color_type = 'OBJECT'
        sp.overlay.show_wireframes = False
        view(center, dist, yaw=yaw, pitch=pitch)
        yield 2.0
        for o in bpy.data.objects:
            if o.type == 'CAMERA' or o.name == 'AFFM_Carrier' or o.name.endswith('_Carrier'):
                o.hide_set(True)          # viewport only: camera gizmo and the (opaque) carrier-liquid particles
        yield from click(1204, 132)       # Build (Fill / Settle runs first)
        yield from wait_for(lambda: bpy.context.scene.frame_current >= 2, 400)
        for fl in bpy.context.scene.arc_flow_fm.fluids:
            ob = bpy.data.objects.get("AFFM_" + fl.name)
            print("FLUIDOBJ", fl.name, ob, flush=True)
            if ob is not None and fl.name.startswith('Carrier'):
                ob.hide_set(True)      # viewport only: the opaque carrier-liquid particles would hide the wax blobs
            if ob is not None:
                ob.color = (fl.color[0], fl.color[1], fl.color[2], 1.0)
        yield from run_until_frame(nframes, 400)
        shot(shot_name)
        yield from click(1204, 132)       # Stop
        yield 1.5

    def step_lava():
        yield from fluid_demo(1198, 'lava', (0, 0, 0.17), 0.9, 30, '05_fluid_lava_lamp', yaw=15, pitch=88, maxp=4000)

    def step_mix():
        yield from fluid_demo(1294, 'mix', (0, 0, 0.09), 0.55, 14, '06_colour_mixing', yaw=15, pitch=88, maxp=2500)

    def step_cavity():
        yield from boot()
        clear_scene()
        bpy.ops.arcflow.demo()
        solid_wire(False)
        area('VIEW_3D').spaces[0].shading.color_type = 'MATERIAL'
        view((3.0, 0, 1.0), 22)
        yield from collapse_extra()
        yield from click(1254, 501)       # close Systems
        yield from click(1224, 449)       # open Solver (LOTBCBP)
        yield from click(1158, 645)       # Boolean projection on
        yield from click(1340, 673)       # Build Fluid Cavity
        yield 4.0
        sp = area('VIEW_3D').spaces[0]
        sp.shading.type = 'SOLID'
        sp.shading.show_xray = True
        sp.shading.color_type = 'OBJECT'
        yield 3.0
        shot('07_fluid_cavity')

    def step_render():
        yield from boot()
        clear_scene()
        yield from collapse_extra()
        yield from click(1350, 532)       # Load Demo: Single Engine
        solid_wire(False)
        area('VIEW_3D').spaces[0].shading.color_type = 'MATERIAL'
        view((3.0, 0, 1.0), 22)
        yield 2.0
        yield from click(1220, 132)       # Build
        yield from run_until_frame(10)
        yield from click(1220, 132)       # Stop
        yield 2.0
        stage_render()
        bpy.context.preferences.view.render_display_type = 'AREA'
        with override():
            bpy.ops.render.render('INVOKE_DEFAULT')
        done = []
        bpy.app.handlers.render_complete.append(lambda *a: done.append(1))
        yield from wait_for(lambda: done, 900, 2.0)
        yield 4.0
        shot('08_render_window')

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

    STEPS = {'lava': step_lava, 'mix': step_mix, 'cavity': step_cavity, 'render': step_render, 'remesh': step_remesh, 'single': step_single, 'twin': step_twin, 'env': step_env}

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
        sh([os.path.join(ROOT, 'build.sh'), BLENDER], cwd=ROOT, stdout=subprocess.DEVNULL)
        subprocess.run(['rm', '-rf', RES]); os.makedirs(RES)
        import glob
        zipf = sorted(glob.glob(os.path.join(ROOT, 'dist', 'tessera_projection-*.zip')))[-1]
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
        names = only or ['remesh', 'single', 'twin', 'env', 'lava', 'mix', 'cavity', 'render']
        for name in names:
            env['GALLERY_ONLY'] = name
            run_gui(env)

    def run_gui(env):
        sh(['xvfb-run', '-a', '-s', '-screen 0 1920x1080x24', BLENDER, '--gpu-backend', 'opengl', '--window-geometry', '0', '0', '1920', '1080',
            '--python', os.path.abspath(__file__)], env=env)

    # raw capture name -> published name; the list is appendable (add a scene to STEPS in the Blender section, then one line here)
    PUBLISH = [
        ('01_projection_remesh', '01_projection_remesh'), ('01_remesh_before', '01a_remesh_before_triangles'),
        ('02_single_engine', '02_systems_single_engine'), ('03_twin_enhanced', '03_twin_crossfeed_enhanced'),
        ('03_twin_failure', '03b_twin_crossfeed_failure'), ('04_environment_presets', '04_environment_presets'),
        ('05_fluid_lava_lamp', '05_fluid_mode_lava_lamp'), ('06_colour_mixing', '06_colour_mixing'),
        ('07_fluid_cavity', '07_boolean_fluid_cavity'), ('08_render_window', '08_cycles_render_window'),
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
