"""Twin-engine examples through the whole combustion chain: both engines fill, burn, build neck pressure, choke, expand, hit the ground, make smoke; GOOD: the
crossfeed enhances engine B, BAD: engine B's fuel tank ruptures; combustion Smooth surface render type; a plain user-modelled channel acts as a chamber.
blender --factory-startup -b --python tests/test_combustion_twin.py"""
import os
import sys
import time
import numpy as np
import bpy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import arc_flow
arc_flow.register()
from arc_flow import runtime as RT, demos, systems
from arc_flow.core import burn as BURN

fails = [0]


def ok(c, msg):
    print(("PASS " if c else "FAIL ") + msg, flush=True)
    if not c:
        fails[0] += 1


sc = bpy.context.scene
af = sc.arc_flow
FR = int(os.environ.get('TWIN_FRAMES', '36'))
for variant in ('GOOD', 'BAD'):
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    af.sys_variant = variant
    bpy.ops.arcflow.demo_twin()
    for o in bpy.data.objects:
        if o.arc_flow.role == 'CONTAINER':
            o.arc_flow.particles = 260
    af.domain_radius = 14.0
    af.particle_budget = 14000
    af.continuous = True
    for fl in af.fluids:
        fl.render_mode = 'SURFACE'
    sc.frame_start, sc.frame_end = 1, FR + 1
    RT.RT.scene_name = sc.name
    t0 = time.time()
    RT.start(sc, record=False, play=False)
    hist = []
    for f in range(1, FR + 1):
        sc.frame_set(f)
        w = RT.RT.world
        hist.append([dict(P=e['P_c'], ch=e['choked'], th=e['thrust'], md=e['mdot_out'], vx=e['v_exit']) for e in w.engines])
    w = RT.RT.world
    st = w.bstats
    ok(len(w.engines) == 2 and all(e['nozzle'] is not None for e in w.engines), "%s: two engine groups, each with its own nozzle (x = %s)" % (variant, [round(float(e['nozzle']['centre'][0]), 1) for e in w.engines]))
    ok(all(max(h[k]['P'] for h in hist) > 5 * w.env.pressure for k in (0, 1)), "%s: both engines build neck pressure (peak %s bar)" % (variant, [round(max(h[k]['P'] for h in hist) / 1e5, 1) for k in (0, 1)]))
    ok(all(any(h[k]['ch'] for h in hist) for k in (0, 1)), "%s: both throats choke" % variant)
    ok(all(max(h[k]['th'] for h in hist) > 0 for k in (0, 1)), "%s: both nozzles give thrust (peak %s kN)" % (variant, [round(max(h[k]['th'] for h in hist) / 1e3) for k in (0, 1)]))
    ok(st['reacted'] > 100 and st['ground'] > 0, "%s: pairs burnt %d, ground impacts %d" % (variant, st['reacted'], st['ground']))
    alive = np.flatnonzero(w.state > 0)
    gas = alive[w.kind[alive] > 0]
    ok(len(gas) > 100, "%s: flame gas / smoke in the domain (%d, %d smoke so far)" % (variant, len(gas), st['smoke']))
    gx = w.pos[gas][:, 0]
    ok((gx < 6).sum() > 20 and (gx > 8).sum() > 20, "%s: both engines have their own plume (%d left, %d right of x = 7)" % (variant, (gx < 6).sum(), (gx > 8).sum()))
    ok(RT.RT.world.p.max_alive == 14000 and st['alive'] <= 14000 + 200, "%s: particle budget %d" % (variant, st['alive']))
    # Smooth surface of the liquids
    so = bpy.data.objects.get('AFCS_Liquids')
    ok(so is not None and len(so.data.vertices) > 100 and 'fm_color' in so.data.attributes and so.data.materials[0] is not None, "%s: smooth liquid surface mesh (%d vertices)" % (variant, len(so.data.vertices) if so else 0))
    if variant == 'BAD':
        ok(any(e['part'] == 'AF_B_FuelTank' for f, e in systems.ST.events), "BAD: engine B's fuel tank ruptured")
    else:
        ok(w.p.ignition == 1.0 and not systems.ST.events, "GOOD: crossfeed enhances combustion, no rupture")
    RT.stop(sc)
    print("%s run: %.0f s" % (variant, time.time() - t0), flush=True)

# a channel the user models behaves like a chamber: no CHAMBER role at all, fuel and oxidiser meet in a plain Channel
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
bpy.ops.arcflow.demo()
for o in bpy.data.objects:
    if o.arc_flow.role == 'CHAMBER':
        o.arc_flow.role = 'TRANSPORT'
for o in bpy.data.objects:
    if o.arc_flow.role == 'CONTAINER':
        o.arc_flow.particles = 260
af.domain_radius = 14.0
af.particle_budget = 14000
sc.frame_start, sc.frame_end = 1, 30
RT.RT.scene_name = sc.name
RT.start(sc, record=False, play=False)
for f in range(1, 28):
    sc.frame_set(f)
w = RT.RT.world
ok(all(q.role != 'CHAMBER' for q in w.parts) and w.bstats['reacted'] > 50, "channels only (no Chamber role): fuel and oxidiser react in the channels (%d pairs)" % w.bstats['reacted'])
ok(w.bstats['P_c'] > 2 * w.env.pressure, "channels only: pressure builds in the channel (%.1f bar)" % (w.bstats['P_c'] / 1e5))
RT.stop(sc)
print("FAILED %d" % fails[0] if fails[0] else "all passed")
sys.exit(1 if fails[0] else 0)
