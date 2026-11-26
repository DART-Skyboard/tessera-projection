"""Tessera Cloth storefront scene run on the GARMENT GROUP path (TS_Dress + TS_DressSash solved as one unit).

The scene is the user's own file (kept OUT of the extension package; pass its path in STOREFRONT_BLEND, default
/home/claude/tp6/refs/tessera_cloth_storefront_selfcol.blend).  Stages (headless, ``bl.sh -r`` for sim):

    blender --factory-startup -b --python docs/storefront_group.py -- prep  WORKDIR
    blender --factory-startup -b --python docs/storefront_group.py -- sim   WORKDIR [last_frame]      (resumes)

``prep`` puts TS_Dress and TS_DressSash into one collection and makes it ONE garment group (Silk dress, cotton sash,
auto-stitched where the sash touches the dress); ``sim`` runs the original fan-jet playback script against THIS
branch's arc_flow, frame by frame, with Self-collision on, and writes WORKDIR/sf_cache.npz + sf_cross.json.
Env: SF_EVERY (crossing sample step, 12), SF_SAVE (checkpoint step, 24).
"""
import json
import math
import os
import sys
import time

import numpy as np
import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
STAGE = argv[0] if argv else 'prep'
WORK = os.path.abspath(argv[1] if len(argv) > 1 else '/home/claude/tp6/out/storefront')
os.makedirs(WORK, exist_ok=True)
SRC = os.environ.get('STOREFRONT_BLEND', '/home/claude/tp6/refs/tessera_cloth_storefront_selfcol.blend')
BLEND = os.path.join(WORK, 'storefront_group.blend')
CACHE = os.path.join(WORK, 'sf_cache.npz')
PART = os.path.join(WORK, 'sf_part.npz')
FPS = 24


def load_playback(scene):
    """Run the scene's own TS_StorefrontPlayback text against THIS branch's cloth_ops (fan jet, blade spin)."""
    from arc_flow import cloth_ops as CO
    src = bpy.data.texts['TS_StorefrontPlayback'].as_string()
    src = src.replace("def _cloth_ops():", "def _cloth_ops():\n    from arc_flow import cloth_ops as _c\n    return _c\n\ndef _cloth_ops_orig():", 1)
    ns = {'__name__': 'TS_StorefrontPlayback'}
    # the text calls register() at import; skip its on-load rebuild so the group solver is built by us
    src = src.replace("    _on_load()\n\n\ndef unregister", "    pass\n\n\ndef unregister", 1)
    exec(compile(src, 'TS_StorefrontPlayback', 'exec'), ns)
    ns['register']()
    ns['_STATE']['hooked'] = True
    CO._fluid_field = lambda sc: ns['_apply_fan'](sc)[0]
    return ns


def prep():
    import arc_flow
    arc_flow.register()
    from arc_flow import cloth_ops as CO
    bpy.ops.wm.open_mainfile(filepath=SRC)
    sc = bpy.context.scene
    cl = sc.arc_flow_cloth
    dress, sash = bpy.data.objects['TS_Dress'], bpy.data.objects['TS_DressSash']
    coll = bpy.data.collections.new("TS_Dress Group")
    sc.collection.children.link(coll)
    for o in (dress, sash):
        for c in list(o.users_collection):
            c.objects.unlink(o)
        coll.objects.link(o)
    g = cl.garments[0]
    g.collection = coll
    g.members.clear()
    CO.add_member(g, dress, 'SILK')
    CO.add_member(g, sash, 'COTTON')
    g.use_stitch = True
    g.stitch_tolerance = 0.01
    sash.arc_flow_cloth.is_garment = True
    sash.arc_flow_cloth.garment_name = g.name
    from arc_flow import asset_colliders as AC
    AC.mark_projection_collider(sash)
    g.self_collision = True
    CO.build_solver(sc, g)
    S = CO.RT.solvers[g.name]
    print("group:", [(m.name, m.preset, m.n_verts) for m in g.members], "verts", S.n, "stitches", g.n_stitches,
          "pins", int(S.pin_hard.sum()), "sc", S.self_collision, S.self_collision_mode, "substeps", S.substeps, "iters", S.iterations)
    bpy.ops.wm.save_as_mainfile(filepath=BLEND)


def sim():
    import arc_flow
    arc_flow.register()
    from arc_flow import cloth_ops as CO
    bpy.ops.wm.open_mainfile(filepath=BLEND)
    sc = bpy.context.scene
    cl = sc.arc_flow_cloth
    g = cl.garments[0]
    last = int(argv[2]) if len(argv) > 2 else sc.frame_end
    EVERY = int(os.environ.get('SF_EVERY', '12'))
    SAVE = int(os.environ.get('SF_SAVE', '24'))
    CO.RT.reset()
    CO.RT.live_running = True
    sc.frame_set(1)
    ns = load_playback(sc)
    S = CO.build_solver(sc, g)
    ns['_apply_fan'](sc)
    for k in ('PARTY_SC_PASSES',):
        pass
    S.self_collider().iterations = int(os.environ.get('SF_SC_PASSES', '2'))
    mem = S.members
    form = bpy.data.objects['TS_DressForm']
    fv, ft = CO._mesh_world(form, None)
    print("sim: %d verts, %d members, %d stitches, substeps %d iters %d, mode %s" % (
        S.n, len(g.members), g.n_stitches, S.substeps, S.iterations, S.self_collision_mode), flush=True)
    frames, ms, cross = [S.x.astype(np.float32).copy()], [0.0], []
    start = 2
    if os.path.exists(PART) and not os.environ.get('SF_FRESH'):
        z = np.load(PART)
        frames = list(z['x'])
        ms = list(z['ms'])
        cross = json.load(open(os.path.join(WORK, 'sf_cross.json')))
        S.x[:] = z['sx']; S.x_prev[:] = z['sxp']; S.v[:] = z['sv']
        start = len(frames) + 1
        print("resumed at frame", start, flush=True)
    sc2 = S.self_collider()
    from arc_flow.core import sdf
    EDG = S.stretch_edges

    def sample(f):
        ws = sc2.wrong_side(S.x, labels=mem)
        cr = sc2.crossings(S.x, labels=mem)
        pts = np.concatenate([S.x, 0.5 * (S.x[EDG[:, 0]] + S.x[EDG[:, 1]])])
        if S.collider is not None:
            _cp, sdv, _ti = sdf.closest_surface(S.collider, pts)
            nin = int((sdv[:S.n] < -0.001).sum()); nin_e = int((sdv[S.n:] < -0.001).sum())
            deep = float(-min(sdv.min(), 0.0))
        else:
            nin = nin_e = 0; deep = 0.0
        row = dict(frame=f, form=dict(verts_inside=nin, edge_mid_inside=nin_e, deepest=round(deep, 5)), wrong={"%d>%d" % k: [int(v[0]), round(float(v[1]), 5)] for k, v in ws.items()},
                   cross={"%s>%s" % k: int(v) for k, v in cr.items()}, pen=float(g.last_penetration))
        cross.append(row)
        return row

    if not cross:
        sample(1)
    t_wall = time.time()
    for f in range(start, last + 1):
        sc.frame_set(f)
        CO.step_scene(sc, 1.0 / FPS)
        frames.append(S.x.astype(np.float32).copy())
        ms.append(g.last_ms)
        if f % EVERY == 0 or f == last:
            r = sample(f)
            print("frame %d step %.0f ms [%.0f s] crossings %s wrong %s pen %.4f" % (
                f, g.last_ms, time.time() - t_wall, {k: v for k, v in r['cross'].items() if v}, {k: v[0] for k, v in r['wrong'].items() if v[0]}, r['pen']) + " form %s" % (r['form'],), flush=True)
        if f % SAVE == 0 or f == last:
            np.savez(PART, x=np.stack(frames), ms=np.asarray(ms), sx=S.x, sxp=S.x_prev, sv=S.v)
            json.dump(cross, open(os.path.join(WORK, 'sf_cross.json'), 'w'))
    np.savez_compressed(CACHE, x=np.stack(frames), fps=FPS, ms=np.asarray(ms), rest=S.rest, tris=S.tris, members=S.members,
                        offsets=np.array([[s0, e] for _o, s0, e in CO.RT.layout[g.name]]))
    json.dump(cross, open(os.path.join(WORK, 'sf_cross.json'), 'w'))
    print("sim done", float(np.mean(ms)), "ms/step mean", flush=True)


if STAGE == 'prep':
    prep()
elif STAGE == 'sim':
    sim()
