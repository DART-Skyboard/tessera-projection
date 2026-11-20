"""Tessera Projection demo video (real Blender UI, Blender 5.3 GUI under Xvfb + software OpenGL).

Two modes in one file:

  python3 make_video.py [--blender PATH] [--out FILE] [--work DIR]      orchestrator: build zip, install into a clean
                                                                       config, run the GUI under xvfb-run, assemble mp4
  blender --python make_video.py -- capture DIR                        (used by the orchestrator) drives the GUI

Capture: a bpy.app.timers generator changes state (demo operators, solver frames, view orbit, panel category),
waits for a real viewport redraw, then grabs the X11 root window with ffmpeg x11grab (one real UI screenshot per state).
Assembly: the screenshots are a time-lapse at SRC_FPS, repeated up to 30 fps, with burned-in captions (PIL) and
title / free-software / end cards (PIL, logo from branding/tessera_1024.png).  See README.md.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
SRC_FPS = 10          # screenshots per second of video time
OUT_FPS = 30
W, H = 1920, 1080

try:
    import bpy  # inside Blender -> capture mode
except ImportError:
    bpy = None


# ======================================================================================== capture (inside Blender)
def capture(outdir):
    import math
    import time
    import addon_utils
    from mathutils import Vector, Euler

    os.makedirs(outdir, exist_ok=True)
    only = os.environ.get('VIDEO_SECTIONS', '')
    only = set(only.split(',')) if only else None
    manifest = []
    OFFSET = int(os.environ.get('VIDEO_FRAME_OFFSET', '0'))
    state = dict(n=OFFSET, draws=0)
    wm = bpy.context.window_manager
    win = wm.windows[0]

    pkg = next(m for m in addon_utils.modules() if m.__name__.endswith('tessera_projection'))
    modname = pkg.__name__
    import importlib
    A = importlib.import_module(modname)
    from importlib import import_module
    runtime = import_module(modname + '.runtime')
    FO = import_module(modname + '.fluidmode_ops')
    systems = import_module(modname + '.systems')
    RO = import_module(modname + '.retopo_ops')

    # ---------------------------------------------------------------- helpers
    def area3d():
        return next(a for a in win.screen.areas if a.type == 'VIEW_3D')

    def region(kind='WINDOW'):
        return next(r for r in area3d().regions if r.type == kind)

    def ov():
        return bpy.context.temp_override(window=win, screen=win.screen, area=area3d(), region=region())

    def on_draw():
        state['draws'] += 1
    bpy.types.SpaceView3D.draw_handler_add(on_draw, (), 'WINDOW', 'POST_PIXEL')

    def redraw_all():
        for a in win.screen.areas:
            a.tag_redraw()

    def settle(wait=0.2, need_draw=True):
        """tag a redraw and wait until the 3D view really drew a new frame (generator)."""
        d0 = state['draws']
        redraw_all()
        t0 = time.time()
        while need_draw and state['draws'] <= d0 and time.time() - t0 < 12.0:
            yield 0.05
            redraw_all()
        yield wait

    def snap(title, sub='', hold=3, draw=True):
        yield from settle(0.2, draw)
        n = state['n']
        state['n'] += 1
        fn = os.path.join(outdir, 'f%05d.jpg' % n)
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'x11grab', '-draw_mouse', '0', '-video_size', '%dx%d' % (W, H),
                        '-i', os.environ['DISPLAY'], '-frames:v', '1', '-q:v', '2', fn], check=True)
        manifest.append(dict(file=os.path.basename(fn), hold=hold, title=title, sub=sub))
        with open(os.path.join(outdir, 'manifest_%05d.json' % OFFSET), 'w') as fh:
            json.dump(manifest, fh)

    QUICK = bool(os.environ.get('VIDEO_QUICK'))

    def q(n):
        return min(n, 3) if QUICK else n

    # ---- X11 test extension: used only to drag the sidebar edge wider (everything else is bpy)
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

    def widen_sidebar(target_left=1230):
        r = region('UI')
        edge = r.x - 3 if r.x > 0 else 1536
        print("sidebar region x=%d w=%d" % (r.x, r.width), flush=True)
        mouse_move(edge, 600); yield 0.4
        mouse_button(1, True); yield 0.3
        x = edge
        while x > target_left:
            x -= 40
            mouse_move(x, 600); yield 0.15
        mouse_button(1, False); yield 0.4
        mouse_move(900, 900); yield 0.4
        print("sidebar now x=%d w=%d" % (region('UI').x, region('UI').width), flush=True)

    def scroll_sidebar(clicks):
        """wheel over the sidebar (positive = down)."""
        r = region('UI')
        mouse_move(r.x + 150, 600); yield 0.3
        for _ in range(abs(clicks)):
            b = 5 if clicks > 0 else 4
            mouse_button(b, True); mouse_button(b, False); yield 0.12
        mouse_move(900, 900); yield 0.3

    PANELS = [c for c in bpy.types.Panel.__subclasses__() if getattr(c, 'bl_category', '') == 'Tessera' and getattr(c, 'bl_space_type', '') == 'VIEW_3D']
    VISIBLE = set()
    ALWAYS = {'AF_PT_fm_mode'}

    def _patch_polls():
        """Show only the cards a scene is about: every Tessera N-panel gets a poll that also checks VISIBLE (the panels' own
        expand state is not scriptable and 0.5.0 opens all of them, which would push the relevant card off the screen)."""
        for c in PANELS:
            if getattr(c, '_video_patched', False):
                continue
            orig = getattr(c, 'poll', None)

            def make(orig):
                def poll(cls, ctx):
                    if cls.bl_idname not in VISIBLE and cls.bl_idname not in ALWAYS:
                        return False
                    return orig(ctx) if orig is not None else True
                return classmethod(poll)
            c.poll = make(orig)
            c._video_patched = True
    _patch_polls()

    def open_panels(*ids, main=True):
        VISIBLE.clear()
        VISIBLE.update(ids)
        if main:
            VISIBLE.update(('AF_PT_main', 'AF_PT_fm_main'))
        set_tab()
        yield from settle(0.3)
        set_tab()
        yield from settle(0.3)

    def setup_ui():
        pr = bpy.context.preferences
        pr.view.ui_scale = 1.35
        pr.view.show_splash = False
        a = area3d()
        with bpy.context.temp_override(window=win, screen=win.screen, area=a, region=a.regions[-1]):
            bpy.ops.screen.screen_full_area()
        sp = area3d().spaces.active
        sp.show_region_ui = True
        sp.show_region_toolbar = False
        sp.overlay.show_stats = False
        sp.show_gizmo = False
        sp.overlay.show_extras = False      # no camera / light icons
        sp.shading.type = 'SOLID'
        sp.shading.light = 'STUDIO'
        sp.shading.color_type = 'SINGLE'
        sp.shading.single_color = (0.80, 0.82, 0.86)
        sp.clip_start = 0.001
        sp.clip_end = 500.0
        sp.lens = 40
        ee = bpy.context.scene.eevee            # cheap viewport EEVEE: the software-GL material preview otherwise needs GBs
        for k, v in (('use_raytracing', False), ('use_shadows', False), ('taa_samples', 1), ('use_volumetric_shadows', False), ('use_gtao', False), ('use_bloom', False)):
            try:
                setattr(ee, k, v)
            except Exception as e:
                print("eevee", k, e, flush=True)

    def set_tab():
        try:
            region('UI').active_panel_category = 'Tessera'
        except Exception as e:
            print("tab:", e, flush=True)

    def clear_scene():
        sc = bpy.context.scene
        for fn in (runtime.stop, FO.stop):
            try:
                fn(sc)
            except Exception:
                pass
        sc.frame_set(1)
        for o in list(bpy.data.objects):
            bpy.data.objects.remove(o, do_unlink=True)
        sc.arc_flow.fluids.clear()
        sc.arc_flow_fm.fluids.clear()
        sc.arc_flow_fm.mode = 'COMBUSTION'
        sp = area3d().spaces.active
        sp.overlay.show_wireframes = False
        sp.region_3d.view_perspective = 'PERSP'

    def view(center, dist, elev_deg, azim_deg):
        rv = area3d().spaces.active.region_3d
        rv.view_perspective = 'PERSP'
        rv.view_location = Vector(center)
        rv.view_distance = dist
        rv.view_rotation = Euler((math.radians(90.0 - elev_deg), 0.0, math.radians(azim_deg)), 'XYZ').to_quaternion()

    def bounds(skip=('TS_Ground',)):
        pts = []
        for o in bpy.data.objects:
            if o.type == 'MESH' and not o.hide_get() and o.name not in skip:
                pts += [o.matrix_world @ Vector(c) for c in o.bound_box]
        if not pts:
            return (0, 0, 0), 1.0
        lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
        hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
        return tuple((lo + hi) / 2), max((hi - lo).length / 2, 0.05)

    def stop_playback():
        try:
            if win.screen.is_animation_playing:
                with bpy.context.temp_override(window=win, screen=win.screen):
                    bpy.ops.screen.animation_cancel(restore_frame=False)
        except Exception as e:
            print("cancel:", e, flush=True)

    def begin(kind, last):
        """start the solver without the timeline playback (cancelling playback stops the simulation with the timeline in 0.5.0),
        then sim_frames() advances it frame by frame with frame_set."""
        sc = bpy.context.scene
        sc.frame_start, sc.frame_end = 1, max(last + 4, 40)
        sc.frame_set(1)
        if kind == 'comb':
            runtime.RT.scene_name = sc.name
            runtime.start(sc, record=False, play=False)
            runtime.RT.t_start = time.time() + 1.0e7      # runtime._poll stops a run that is not on a playing timeline after 3 s; we step it by hand
        else:
            FO.start(sc, play=False)

    def sim_frames(sc, last, step, title, sub_fn, hold, orbit=None):
        """advance the real solver one frame at a time (frame_set -> frame_change handler); snapshot every `step` frames."""
        f0 = sc.frame_current + 1
        if QUICK:
            last = f0 + 5
        for f in range(f0, last + 1):
            sc.frame_set(f)
            if (f % step == 0) or f == last:
                if orbit:
                    view(*orbit(f))
                print("frame %d rss %.0f MB" % (f, int(open('/proc/self/statm').read().split()[1]) * 4096 / 1e6), flush=True)
                yield from snap(title, sub_fn(), hold=hold)

    # ---------------------------------------------------------------- sections
    # Each section is a generator: set the scene up with the extension's own operators / loaders, then `yield from snap(...)` per state.
    # Append a new scene = write sec_xxx() and add it to SECTIONS below.
    def shade(kind):
        sp = area3d().spaces.active
        sp.shading.type = 'MATERIAL' if kind == 'MATERIAL' else 'SOLID'
        if kind in ('SOLID', 'SINGLE'):
            sp.shading.color_type = 'SINGLE'
        elif kind == 'VERTEX':
            sp.shading.color_type = 'VERTEX'

    def sec_remesh():
        import bmesh
        import random
        clear_scene(); shade('SOLID')
        sc = bpy.context.scene
        yield from open_panels('AF_PT_retopo', main=False)
        bpy.ops.mesh.primitive_monkey_add(size=2.6, location=(0, 0, 0))
        src = bpy.context.active_object
        src.name = "Suzanne_triangles"
        md = src.modifiers.new('s', 'SUBSURF'); md.levels = 1
        bpy.ops.object.modifier_apply(modifier=md.name)
        bm = bmesh.new(); bm.from_mesh(src.data); bmesh.ops.triangulate(bm, faces=bm.faces[:])
        rnd = random.Random(4)
        for v in bm.verts:                       # break the mirror symmetry a little (one-sided noise) so Symmetrize has work to do
            if v.co.x > 0.05:
                v.co += Vector((rnd.uniform(-1, 1), rnd.uniform(-1, 1), rnd.uniform(-1, 1))) * 0.012
        bm.to_mesh(src.data); bm.free()
        bpy.ops.object.shade_flat()
        sp = area3d().spaces.active
        sp.overlay.show_wireframes = True
        sp.overlay.wireframe_threshold = 1.0
        sp.overlay.wireframe_opacity = 1.0
        rp = sc.arc_flow_retopo
        rp.auto = False; rp.crease_density = 10.0; rp.relax_iterations = 40; rp.resolution = 0.5; rp.crease_angle = math.radians(30)
        src.select_set(False)
        cap = "Projection Remesh"
        n_in = len(src.data.polygons)
        az = [-25.0]

        def orbit(n, d=3.0):
            for _ in range(n):
                az[0] += d
                view((0, 0, 0), 4.4, 12, az[0])
                yield
        for _ in orbit(0):
            pass
        for k in range(q(14)):
            az[0] += 3.0; view((0, 0, 0), 4.4, 12, az[0])
            yield from snap(cap, "Before: %d triangles, slightly asymmetric" % n_in, hold=3)
        for sym, label in ((False, "Symmetrize off: quads follow the asymmetric input"), (True, "Symmetrize on: the mirror plane is detected and the quads become exactly symmetric")):
            rp.symmetrize = sym; rp.use_symmetry = sym
            src.hide_set(False); src.select_set(False)
            bpy.context.view_layer.objects.active = src
            yield from snap(cap, "Projection Remesh: triangles to quads (Symmetrize %s)..." % ("on" if sym else "off"), hold=5)
            with ov():
                bpy.ops.arcflow.quad_remesh()
            out = bpy.context.active_object
            src.hide_set(True); out.select_set(False)
            yield 0.4
            nq = len(out.data.polygons)
            quads = sum(1 for p in out.data.polygons if len(p.vertices) == 4)
            for k in range(q(10)):
                az[0] += 3.0; view((0, 0, 0), 4.4, 12, az[0])
                yield from snap(cap, "%s: %d faces, %.0f%% quads" % ("Symmetrize " + ("on" if sym else "off"), nq, 100.0 * quads / nq), hold=3)
            out.hide_set(True)

    def sec_single():
        clear_scene(); shade('SOLID')
        sc = bpy.context.scene
        yield from open_panels('AF_PT_systems')
        bpy.ops.arcflow.demo()
        yield 0.5
        cap = "Systems detect themselves"
        center, dist, el = (-0.2, 1.6, 0.8), 14.0, 20
        for i in range(q(6)):
            view(center, dist, el, 12 + i * 3.0)
            yield from snap(cap, "Plain meshes: two tanks, two pipes, a chamber and a nozzle", hold=3)
        with ov():
            bpy.ops.arcflow.systems_detect()
        for i in range(q(8)):
            view(center, dist, el, 30 + i * 3.0)
            yield from snap(cap, "Re-detect: containers, transport lines and chamber are found from the geometry", hold=3)
        begin('comb', 30)
        yield 0.3
        yield from sim_frames(sc, 30, 3, cap, lambda: "Build: chamber pressure and thrust come from the detected system", 4,
                              orbit=lambda f: (center, dist, el, 54 + (f - 2) * 1.5))
        runtime.stop(sc)

    def sec_twin(variant):
        clear_scene(); shade('SOLID')
        sc = bpy.context.scene
        af = sc.arc_flow
        yield from open_panels('AF_PT_systems')
        af.sys_variant = variant
        bpy.ops.arcflow.demo_twin()
        yield 0.5
        center, dist, el = (6.2, 1.5, 0.6), 25.0, 24
        cap = "Twin-engine crossfeed"
        good = variant == 'GOOD'
        for i in range(q(4)):
            view(center, dist, el, 14 + i * 2.0)
            yield from snap(cap, "Engine A oxidizer tank piped into engine B's chamber manifold" if good else
                            "Bad link: engine A's hot chamber piped into engine B's fuel tank", hold=3)
        last = 44 if good else 31
        begin('comb', last)
        yield 0.3
        sub = (lambda: "Combustion chain: both engines run, B is enhanced by the crossfeed") if good else \
              (lambda: "Combustion chain: the link ruptures the system and the failure is reported")
        yield from sim_frames(sc, last, 4, cap, sub, 4, orbit=lambda f: (center, dist, el, 22 + (f - 2) * 0.5))
        for i in range(q(3)):
            yield from snap(cap, "Enhanced" if good else "Rupture", hold=3)
        runtime.stop(sc)

    def fluid_scene(name, loader, build_frames, step, cap, sub_start, sub_run, center, dist, el=6, shading='MATERIAL', particles=None):
        clear_scene(); shade(shading)
        sc = bpy.context.scene
        loader(sc, **({'max_particles': particles} if particles else {}))
        yield from open_panels('AF_PT_fm_solver')   # not AF_PT_fm_fluids unless its draw() works in the installed build
        yield 0.5
        for i in range(q(3)):
            view(center, dist, el, 10 + i)
            yield from snap(cap, sub_start, hold=3)
        begin('fluid', build_frames)
        yield 0.3
        print("fm running:", FO.RT.running, sc.arc_flow_fm.status, flush=True)
        yield from sim_frames(sc, build_frames, step, cap, lambda: sub_run, 4, orbit=lambda f: (center, dist, el, 10 + f * 0.25))
        FO.stop(sc)

    def sec_lava():
        yield from fluid_scene('lava', FO.load_lava_lamp, 52, 4, "Fluid mode: lava lamp", "Wax blobs in a carrier liquid, heat source at the bottom",
                               "Heated wax rises, cools at the top and sinks again", (0, 0, 0.17), 0.75, particles=6000)

    def sec_mix():
        yield from fluid_scene('mix', FO.load_colour_mixing, 52, 4, "Fluid mode: colour mixing", "Yellow, blue and red water in a glass",
                               "A stirrer swirls them: subtractive colour mixing, temperature equilibrates", (0, 0, 0.08), 0.42, el=8)

    def sec_crystal():
        yield from fluid_scene('crystal', FO.load_crystal, 60, 5, "Transmutation: crystal growth", "A warm salt solution in a dish",
                               "It cools, nuclei appear and solid crystals grow outward from them", (0, 0, 0.03), 0.3, el=22)

    def sec_steam():
        yield from fluid_scene('steam', FO.load_steam, 60, 5, "Transmutation: steam", "Water in a beaker over a burner",
                               "At 100 C the water turns to steam, carries the latent heat away and rises", (0, 0, 0.2), 0.7, el=8)

    def sec_shaders():
        clear_scene(); shade('MATERIAL')
        sc = bpy.context.scene
        yield from open_panels('AF_PT_shader', main=False)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=1.0, location=(0, 0, 1.0))
        bpy.ops.object.shade_smooth()
        bpy.ops.mesh.primitive_plane_add(size=8, location=(0, 0, 0))
        plane = bpy.context.active_object
        sphere = next(o for o in bpy.data.objects if o.name.startswith('Sphere'))
        bpy.context.view_layer.objects.active = sphere
        sphere.select_set(True)
        cap = "Shader presets"
        for key, nm in (('WATER', 'Water'), ('HONEY', 'Honey'), ('METAL', 'Molten metal'), ('LAVA', 'Lava'), ('MERCURY', 'Mercury')):
            sc.arc_flow_sh.preset = key
            with ov():
                bpy.ops.arcflow.shader_assign(target='ACTIVE')
            yield 0.5
            for i in range(q(3)):
                view((0, 0, 1.0), 6.0, 14, 20 + i * 4)
                yield from snap(cap, "%s: one click applies the preset's node material" % nm, hold=4)

    def sec_surfaces():
        clear_scene(); shade('MATERIAL')
        sc = bpy.context.scene
        yield from open_panels('AF_PT_examples', main=False)
        yield from scroll_sidebar(0)
        cap = "Examples, fluid volumes and surface sources"
        for i in range(q(3)):
            yield from snap(cap, "The Examples card lists every loadable scene", hold=4)
        sc.arc_flow_fm.mode = 'FLUID'
        bpy.ops.arcflow.srf_example(which='VOLUME')
        yield from open_panels('AF_PT_srf')
        center, rad = bounds()
        dist, el = rad * 2.6, 16
        for i in range(q(4)):
            view(center, dist, el, 15 + i * 3)
            yield from snap(cap, "Add Fluid Volume: hot and cool gas boxes floating in the Environment", hold=4)
        bpy.ops.arcflow.srf_example(which='EMIT')
        yield 0.5
        center, rad = bounds()
        dist = rad * 2.6
        for i in range(q(3)):
            view(center, dist, 16, 15 + i * 3)
            yield from snap(cap, "Emission surface above a moving viscosity surface (a belt that drags the fluid)", hold=4)
        begin('fluid', 34)
        yield 0.3
        yield from sim_frames(sc, 34, 4, cap, lambda: "Emission nozzle feeds particles onto the belt, viscosity drags them along", 4,
                              orbit=lambda f: (center, dist, 16, 25 + f * 0.4))
        FO.stop(sc)

    def sec_wind():
        clear_scene(); shade('VERTEX')
        sc = bpy.context.scene
        yield from open_panels('AF_PT_thermal', 'AF_PT_windtunnel')
        sc.arc_flow_wt.emit = False      # 0.5.0 RC: the thermal readback fails while the emitter adds particles (reported on the bus)
        bpy.ops.arcflow.example_load(key='WIND_TUNNEL')
        yield 0.5
        air = bpy.data.objects.get("Aircraft Airframe")
        if air is not None:
            bpy.ops.object.select_all(action='DESELECT')
            bpy.context.view_layer.objects.active = air
            air.arc_flow_th.t_min, air.arc_flow_th.t_max, air.arc_flow_th.speedup, air.arc_flow_th.wall_heat = 288.15, 345.0, 60.0, 600.0
            air.arc_flow_th.legend = True
        cap = "Wind tunnel with thermal readback"
        ctr = tuple(air.matrix_world.translation) if air is not None else (0, 0, 0)
        for i in range(q(3)):
            view(ctr, 3.2, 14, 25 + i * 3)
            yield from snap(cap, "Tilt-rotor aircraft in an air stream; its skin temperature is read back each frame", hold=4)
        begin('fluid', 48)
        yield 0.3
        yield from sim_frames(sc, 48, 4, cap, lambda: "Skin temperature from the fluid physics, with a colour bar (K) beside the aircraft", 4,
                              orbit=lambda f: (ctr, 3.2, 14, 30 + f * 0.4))
        FO.stop(sc)

    def sec_gn():
        clear_scene(); shade('SOLID')
        sc = bpy.context.scene
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=1.0, location=(0, 0, 0))
        ob = bpy.context.active_object
        cap = "Geometry Nodes sockets"
        for grp in ("Tessera Particle Cloud", "Tessera Thermal"):
            with ov():
                bpy.ops.arcflow.add_tessera_modifier(group=grp)
        a = area3d()
        a.type = 'NODE_EDITOR'
        sp = a.spaces.active
        sp.tree_type = 'GeometryNodeTree'
        yield 1.0
        mods = [m for m in ob.modifiers if m.type == 'NODES']
        ui = next((r for r in a.regions if r.type == 'UI'), None)
        if ui is not None:
            try:
                ui.active_panel_category = 'Tessera'
            except Exception as e:
                print("gn tab:", e, flush=True)
        wr = next(r for r in a.regions if r.type == 'WINDOW')
        for m in mods:
            ob.modifiers.active = m
            yield 0.6
            nodes = sorted(m.node_group.nodes, key=lambda n: n.location.x)
            for side, part, txt in (("inputs", nodes[:7], "group inputs"), ("outputs", nodes[-7:], "group outputs")):
                try:
                    with bpy.context.temp_override(window=win, screen=win.screen, area=a, region=wr, space_data=sp):
                        bpy.ops.node.select_all(action='DESELECT')
                        for n in part:
                            n.select = True
                        bpy.ops.node.view_selected()
                except Exception as e:
                    print("gn view:", e, flush=True)
                for i in range(q(3)):
                    redraw_all()
                    yield 1.5
                    yield from snap(cap, "%s: %s with their sockets" % (m.node_group.name, txt), hold=5, draw=False)
        a.type = 'VIEW_3D'
        yield 1.0

    def sec_env():
        clear_scene(); shade('MATERIAL')
        sc = bpy.context.scene
        FO.load_lava_lamp(sc, max_particles=3000)
        yield from open_panels('AF_PT_fm_env')
        e = sc.arc_flow.env
        cap = "Environment presets"
        e.use_mix = True
        for key, nm in (('EARTH', 'Earth'), ('MARS', 'Mars'), ('VENUS', 'Venus'), ('TITAN', 'Titan'), ('JUPITER', 'Jupiter')):
            e.preset = key
            bpy.ops.arcflow.env_apply_preset()
            for i in range(q(3)):
                view((0, 0, 0.17), 0.75, 6, 10 + i * 2)
                yield from snap(cap, "%s: pressure, temperature, gravity and gas mix set together" % nm, hold=4)

    # storyboard: (key, generator function).  Append new scenes here.
    SECTIONS = [('remesh', sec_remesh), ('single', sec_single), ('twin_good', lambda: sec_twin('GOOD')), ('twin_bad', lambda: sec_twin('BAD')),
                ('lava', sec_lava), ('mix', sec_mix), ('crystal', sec_crystal), ('steam', sec_steam), ('shaders', sec_shaders),
                ('surfaces', sec_surfaces), ('wind', sec_wind), ('gn', sec_gn), ('env', sec_env)]

    # ---------------------------------------------------------------- driver
    def main_gen():
        yield 2.0
        setup_ui()
        yield 1.0
        clear_scene()
        for name, fn in SECTIONS:
            if only and name not in only:
                continue
            print("== section", name, flush=True)
            yield from fn()
        print("capture done: %d frames" % state['n'], flush=True)
        yield 0.5
        bpy.ops.wm.quit_blender()

    gen = main_gen()

    def tick():
        try:
            return next(gen)
        except StopIteration:
            return None
        except Exception:
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            bpy.ops.wm.quit_blender()
            return None
    bpy.app.timers.register(tick, first_interval=3.0)


if bpy is not None:
    capture(sys.argv[sys.argv.index('--') + 1:][1])


# ======================================================================================== orchestrator / assembly (outside Blender)
def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def build_and_install(blender, work):
    """build.sh -> dist/tessera_projection-<ver>.zip, then install into a clean config dir; returns that config dir."""
    run([os.path.join(REPO, 'build.sh'), blender], stdout=subprocess.DEVNULL)
    zips = sorted(f for f in os.listdir(os.path.join(REPO, 'dist')) if f.endswith('.zip'))
    cfg = os.path.join(work, 'cfg')
    if os.path.isdir(cfg):
        import shutil
        shutil.rmtree(cfg)
    os.makedirs(cfg)
    env = dict(os.environ, BLENDER_USER_RESOURCES=cfg)
    run([blender, '--factory-startup', '-b', '--command', 'extension', 'install-file', '-r', 'user_default', '--enable',
         os.path.join(REPO, 'dist', zips[-1])], env=env)
    return cfg


# One Blender process per section, retried: software-GL material previews and the solvers need 2-3 GB and the box is shared, so an
# out-of-memory kill of a single section must not lose the whole recording.  Keep SECTION_KEYS in the order of SECTIONS in capture().
SECTION_KEYS = ["remesh", "single", "twin_good", "twin_bad", "lava", "mix", "crystal", "steam", "shaders", "surfaces", "wind", "gn", "env"]


def capture_gui(blender, cfg, frames_dir, keys=None, attempts=5):
    import shutil
    keys = keys or SECTION_KEYS
    os.makedirs(frames_dir, exist_ok=True)
    for key in keys:
        d = os.path.join(frames_dir, key)
        for attempt in range(1, attempts + 1):
            if os.path.isdir(d):
                shutil.rmtree(d)
            os.makedirs(d)
            env = dict(os.environ, BLENDER_USER_RESOURCES=cfg, VIDEO_SECTIONS=key)
            env.pop('WAYLAND_DISPLAY', None)
            log = os.path.join(frames_dir, key + '.log')
            with open(log, 'w') as fh:
                subprocess.run(['nice', '-n', '10', 'xvfb-run', '-a', '-s', '-screen 0 %dx%dx24' % (W, H), blender, '--gpu-backend', 'opengl',
                                '--window-geometry', '0', '0', str(W), str(H), '--python', os.path.abspath(__file__), '--', 'capture', d],
                               env=env, stdout=fh, stderr=subprocess.STDOUT)
            if 'capture done' in open(log).read():
                print("section %s captured (attempt %d)" % (key, attempt), flush=True)
                break
            print("section %s failed (attempt %d), retrying" % (key, attempt), flush=True)
        else:
            raise RuntimeError("section %s failed %d times" % (key, attempts))


def load_manifests(dirs):
    import glob
    out = []
    for d in dirs:
        for mf in sorted(glob.glob(os.path.join(d, 'manifest*.json'))):
            for e in json.load(open(mf)):
                out.append(dict(e, dir=d))
    return out


def _font(size, bold=True):
    from PIL import ImageFont
    p = '/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf' % ('-Bold' if bold else '')
    return ImageFont.truetype(p, size)


def _caption(img, title, sub):
    from PIL import Image, ImageDraw
    ov = Image.new('RGBA', img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    ft, fs = _font(54), _font(32, False)
    tw = max(d.textlength(title, font=ft), d.textlength(sub, font=fs)) if sub else d.textlength(title, font=ft)
    x0, y0 = 36, H - 52 - 150
    d.rounded_rectangle((x0, y0, x0 + tw + 56, y0 + (150 if sub else 90)), 18, fill=(8, 10, 16, 215))
    d.rectangle((x0, y0 + 14, x0 + 8, y0 + (136 if sub else 76)), fill=(255, 150, 40, 255))
    d.text((x0 + 30, y0 + 12), title, font=ft, fill=(255, 255, 255, 255))
    if sub:
        d.text((x0 + 30, y0 + 84), sub, font=fs, fill=(205, 215, 235, 255))
    return Image.alpha_composite(img.convert('RGBA'), ov).convert('RGB')


def _card(lines, logo=True, url=None):
    """Title / free-software / end card (PIL, not a Blender screenshot)."""
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (W, H), (14, 16, 24))
    px = img.load()
    for y in range(H):                       # soft vertical gradient
        k = y / H
        for x in range(0, W, 1):
            pass
    g = Image.linear_gradient('L').resize((W, H))
    img = Image.composite(Image.new('RGB', (W, H), (26, 30, 46)), Image.new('RGB', (W, H), (8, 9, 14)), g.transpose(Image.FLIP_TOP_BOTTOM))
    d = ImageDraw.Draw(img)
    y = 150
    if logo:
        lg = Image.open(os.path.join(REPO, 'branding', 'tessera_1024.png')).convert('RGBA').resize((420, 420), Image.LANCZOS)
        img.paste(lg, ((W - 420) // 2, y), lg)
        y += 470
    for text, size, bold, col in lines:
        f = _font(size, bold)
        d.text(((W - d.textlength(text, font=f)) / 2, y), text, font=f, fill=col)
        y += int(size * 1.45)
    return img


def cards():
    white, grey, orange = (255, 255, 255), (170, 182, 205), (255, 165, 60)
    return dict(
        title=_card([("Tessera Projection", 92, True, white), ("for Blender", 60, False, orange),
                     ("Projection remesh, self-detecting systems, fluids, shaders and a wind tunnel", 34, False, grey)]),
        foss=_card([("Free and open source", 84, True, white), ("GPL-3.0-or-later", 52, True, orange),
                    ("A Blender extension, free to use, study, change and share", 34, False, grey)], logo=False),
        end=_card([("Tessera Projection for Blender", 62, True, white), ("github.com/DART-Skyboard/tessera-projection", 46, True, orange),
                   ("Free and open source, GPL-3.0-or-later", 32, False, grey)]))


def assemble(dirs, out, target=60.0, drop=None):
    from PIL import Image
    if isinstance(dirs, str):
        dirs = [dirs]
    man = [e for e in load_manifests(dirs) if not (drop and drop(e))]
    cd = cards()
    fade = 5
    t_card = dict(title=4.0, foss=3.0, end=3.5)
    n_trans = len({m['title'] for m in man}) * 2 * fade + 3 * 2 * fade
    fixed = sum(t_card.values()) * OUT_FPS + n_trans
    total_hold = sum(m['hold'] for m in man)
    k = max(0.3, min(4.0, (target * OUT_FPS - fixed) / total_hold))
    print("frames %d, holds sum %d, hold scale %.2f" % (len(man), total_hold, k), flush=True)
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '%dx%d' % (W, H), '-r', str(OUT_FPS),
                           '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out],
                          stdin=subprocess.PIPE)
    count = [0]

    def put(img, n=1, bright=1.0):
        if bright < 1.0:
            img = Image.eval(img, lambda v: int(v * bright))
        raw = img.tobytes()
        for _ in range(n):
            ff.stdin.write(raw)
        count[0] += n

    def show_card(name, secs):
        n = int(secs * OUT_FPS)
        for i in range(fade):
            put(cd[name], 1, (i + 1) / fade)
        put(cd[name], n - 2 * fade)
        for i in range(fade):
            put(cd[name], 1, 1 - (i + 1) / fade)

    show_card('title', t_card['title'])
    prev = None
    for idx, m in enumerate(man):
        img = _caption(Image.open(os.path.join(m['dir'], m['file'])).convert('RGB'), m['title'], m['sub'])
        nxt_title = man[idx + 1]['title'] if idx + 1 < len(man) else None
        if m['title'] != prev:
            for i in range(fade):
                put(img, 1, (i + 1) / fade)
        put(img, max(1, int(round(m['hold'] * k))))
        if nxt_title != m['title']:
            for i in range(fade):
                put(img, 1, 1 - (i + 1) / fade)
        prev = m['title']
    show_card('foss', t_card['foss'])
    show_card('end', t_card['end'])
    ff.stdin.close()
    ff.wait()
    print("wrote %s: %d frames = %.1f s" % (out, count[0], count[0] / OUT_FPS), flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--blender', default=os.environ.get('BLENDER', 'blender'))
    ap.add_argument('--work', default='/tmp/tessera_video_work')
    ap.add_argument('--out', default='tessera_demo.mp4')
    ap.add_argument('--target-seconds', type=float, default=60.0)
    ap.add_argument('--skip-build', action='store_true', help='reuse the installed config in --work')
    ap.add_argument('--skip-capture', action='store_true', help='only assemble from --work/frames')
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    frames = os.path.join(a.work, 'frames')
    if not a.skip_capture:
        cfg = os.path.join(a.work, 'cfg') if a.skip_build else build_and_install(a.blender, a.work)
        capture_gui(a.blender, cfg, frames)
    assemble([os.path.join(frames, k) for k in SECTION_KEYS], a.out, a.target_seconds)


if bpy is None and __name__ == '__main__':
    main()
