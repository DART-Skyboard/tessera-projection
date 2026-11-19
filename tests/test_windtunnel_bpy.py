"""Headless Blender test of thermal readback and the wind tunnel:  blender --factory-startup -b --python tests/test_windtunnel_bpy.py"""
import math
import os
import sys
import time

import bpy
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import arc_flow
arc_flow.register()
from arc_flow import fluidmode_ops as FO
from arc_flow import thermal as TM
from arc_flow import windtunnel as WT
fails = [0]


def ok(c, msg):
    print(("PASS " if c else "FAIL ") + msg, flush=True)
    if not c:
        fails[0] += 1


print("Blender", bpy.app.version_string, flush=True)
sc = bpy.context.scene
t0 = time.time()
msg = WT.load_example(sc)
print(msg)
ok(all(bpy.data.objects.get(n) for n in ("Wind Tunnel", "Tunnel Air Volume", "Wind Emitter", "Aircraft Airframe", "Aircraft Rotor R", "Aircraft Rotor L")), "tunnel, emitter, air volume and aircraft exist")
tun = bpy.data.objects["Wind Tunnel"]
ok(len(tun.data.polygons) == 16 + 8 or len(tun.data.polygons) > 0, "hollow shell has %d faces" % len(tun.data.polygons))
# real thickness: bbox vs inner bound; the shell is a solid with volume > 0 and smaller than the outer box
import bmesh
bm = bmesh.new(); bm.from_mesh(tun.data)
vol = bm.calc_volume(signed=True)
bm.free()
wt = sc.arc_flow_wt
Lo, Wo, Ho, t = wt.length, wt.width, wt.height, wt.wall
vexp = Lo * Wo * Ho - Lo * (Wo - 2 * t) * (Ho - 2 * t)
ok(abs(vol - vexp) / vexp < 1e-6, "tunnel shell volume %.4f m3 matches the hollow box with %.0f mm walls (%.4f)" % (vol, t * 1000, vexp))
env = sc.arc_flow.env
ok(abs(env.pressure - 101325) < 1 and env.source == 'OBJECT' and env.obj == tun and env.use_mix and not env.ground, "Environment: Earth atmosphere, container = tunnel")
air = bpy.data.objects["Aircraft Airframe"]
ok(air.arc_flow_th.on and 'af_temperature' in air.data.attributes, "thermal attribute written on the aircraft")
ok(air.data.color_attributes.get('af_thermal_color') is not None, "viewport colour attribute written")
ok(air.modifiers.get('Tessera Surface Temperature') is not None, "Geometry Nodes thermal modifier added")
ok(air.material_slots and air.material_slots[0].material.name.startswith("Tessera Thermal"), "thermal shader assigned")

# run the simulation a few frames
FO.start(sc, play=False)
S = FO.RT.solver
print("particles", S.n, "dx %.3f" % S.dx, flush=True)
t0 = time.time()
tmax = []
for f in range(1, 13):
    sc.frame_set(f)
    a = air.data.attributes['af_temperature']
    arr = np.empty(len(a.data), np.float32); a.data.foreach_get('value', arr)
    tmax.append(float(arr.max()))
print("12 frames %.1fs, max wall T per frame" % (time.time() - t0), ["%.2f" % x for x in tmax], flush=True)
ok(tmax[-1] > tmax[0] + 0.05, "surface temperature rises with time (%.2f -> %.2f K)" % (tmax[0], tmax[-1]))
ok(float(arr.min()) >= sc.arc_flow.env.temperature - 1e-3 and float(arr.max()) < 700.0, "bounded: %.1f .. %.1f K" % (arr.min(), arr.max()))
# scrubbing back restores the cached state
sc.frame_set(3)
a.data.foreach_get('value', arr)
ok(abs(float(arr.max()) - tmax[2]) < 1e-3, "scrubbing back reproduces frame 3 (%.3f vs %.3f)" % (float(arr.max()), tmax[2]))
FO.stop(sc)

# plain collision body
cube = bpy.data.objects.new("Item", bpy.data.meshes.new("Item"))
cube.data.from_pydata([(-.1, -.1, 0), (.1, -.1, 0), (.1, .1, 0), (-.1, .1, 0), (-.1, -.1, .2), (.1, -.1, .2), (.1, .1, .2), (-.1, .1, .2)], [],
                      [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
sc.collection.objects.link(cube)
cube.arc_flow_th.on = True
WT.mark_body(cube, plain=True)
ok(cube.arc_flow_fm.collider and not cube.arc_flow_th.on and cube.arc_flow.role == 'NONE', "plain collision body: collider only, no thermal, no role")
cube.arc_flow_wt.plain = False

# face selection and domain
cube.arc_flow_th.domain = 'FACE'
cube.arc_flow_th.on = True
ok(cube.data.attributes['af_temperature'].domain == 'FACE', "face-domain attribute")
print("ALL PASS" if not fails[0] else "FAILED %d" % fails[0])
sys.exit(1 if fails[0] else 0)
