"""Example: three independent projection fluid assets in one scene (Fluid mode > Examples).

Asset = a part (a mesh with a Systems role) + the fluid it carries.  The three parts do not touch, so the Systems list detects three independent
systems; the three fluids are free bodies in one solver, so when they meet on the ground they interact:
  Blue + Yellow   miscible, REACT rule: blue + yellow -> Green (colour mixing, a little heat released),
  Hot Red         warm fluid next to the cold ones: heat conducts across the contact (proximity coupling),
  all three       Render = Fluid surface: one smooth mesh, necks form where they touch and the colours blend across the seam."""
import bpy

from . import fluidmode_ops as FO

KEY = "assets_three"
LABEL = "Three independent assets"
DESCRIPTION = ("Three fluid assets (blue, yellow, hot red) that the Systems list sees as three independent systems; they meet on the ground and mix, react "
               "(blue + yellow -> green), exchange heat and blend into one smooth surface. Press Build.")

ASSETS = (("Asset Blue", (0.08, 0.3, 1.0), (-0.17, 0.0, 0.10), 285.0),
          ("Asset Yellow", (1.0, 0.85, 0.08), (0.0, 0.0, 0.10), 295.0),
          ("Asset Red", (0.95, 0.12, 0.1), (0.17, 0.0, 0.10), 360.0))
RULES = ('{"species": [{"name": "Green", "color": [0.1, 0.75, 0.2], "density": 1005.0, "cp": 4186.0}], "rules": ['
         '{"a": "Asset Blue", "b": "Asset Yellow", "outcome": "REACT", "product": "Green", "tmin": 250.0, "tmax": 400.0, "vmin": 0.0, "q": 3.0e4, "cmin": 0.5, "rate": 4.0}]}')


def load(scene, max_particles=9000, radius=0.07):
    FO._clear_demo(scene)
    col = FO._demo_common(scene)
    fm = scene.arc_flow_fm
    fm.mode = 'FLUID'
    import math
    for name, color, loc, T in ASSETS:
        part = FO._mesh_obj(name + " part", lambda bm: FO._sphere(bm, radius), loc, col)
        part.display_type = 'WIRE'
        part.arc_flow.role = 'CONTAINER'                 # the Systems list sees every asset as its own container system
        fl = FO.add_fluid(scene, 'WATER', name)
        fl.shape, fl.shape_source, fl.no_container, fl.container = part, 'OBJECT', True, None
        fl.spawn_mode = 'VOLUME'
        fl.color, fl.temperature, fl.render_mode = color, T, 'SURFACE'
        fl.miscible, fl.mix_rate, fl.cohesion, fl.cool = True, 6.0, 0.03, 0.01
        fl.drag = 0.0
    fm.rules_json = RULES
    fm.max_particles = max_particles
    fm.fill_mode = 'OFF'
    fm.speedup = 8.0                                     # thermal: seconds of conduction in a few frames
    scene.frame_start, scene.frame_end = 1, 120
    cam = bpy.data.objects.new("TS_Camera", bpy.data.cameras.new("TS_Camera"))
    col.objects.link(cam)
    cam.location = (0.0, -0.9, 0.16)
    cam.rotation_euler = (math.radians(86.0), 0, 0)
    cam.data.lens = 45
    scene.camera = cam
    FO.mark_dirty()
    return "Three independent assets loaded. Press Build (Systems list: 3 systems)."


def register():
    try:
        from . import surfaces_panel
        surfaces_panel.register_loader("Fluid", LABEL, KEY, load, DESCRIPTION)
    except Exception:
        pass
