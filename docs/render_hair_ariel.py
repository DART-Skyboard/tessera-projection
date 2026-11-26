"""Hair collision showcase on the game's real Ariel character (docs/ariel_import.py does the asset conversion).

    blender --factory-startup -b --python docs/render_hair_ariel.py -- bake  OUT.pkl  GAME_DIR [--no-collide] [--seconds 30]
    blender --factory-startup -b --python docs/render_hair_ariel.py -- measure OUT.json GAME_DIR          ON vs OFF penetration counts
    blender --factory-startup --python docs/capture_hair_ariel_ui.py -- OUTDIR GAME_DIR                    GUI stepped capture

Scene: Ariel mesh + skeleton, clips on the NLA (idle / head turns / walk / sad droop / run) plus an additive procedural head-turn
layer, strand hair (all strands, guides simulated) driven by the extension's own solver against capsule / head collision bodies
that follow the skinned skeleton.  The game assets are read from GAME_DIR and are never copied into the repository.
"""
import math
import os
import pickle
import sys
import time

import bpy
import numpy as np
from mathutils import Vector, Matrix, Euler, Quaternion

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
import arc_flow
from arc_flow import hair_ops as HO
from arc_flow.core import hair_solver as HS
import ariel_import as AI

try:
    arc_flow.register()
except Exception:
    pass

sc = bpy.context.scene
FPS = 24
SECONDS = 30

# 30 s timeline: (action, frames played)
TIMELINE = [('Idle_Breathe_Blink', 72, 24), ('Idle_Breathe_Blink', 144, 24), ('Walking_Woman', 138, 0),
            ('Acting_Sad', 96, 0), ('Running', 90, 0), ('Acting_Surprised', 72, 0), ('Idle_Breathe_Blink', 108, 24)]
# procedural head turn layered on top (time s, yaw deg, pitch deg)
HEAD_KEYS = [(0, 0, 0), (3.0, 0, 0), (4.0, 55, 0), (5.2, 55, 0), (6.8, -55, 8), (8.0, -55, 8), (9.0, 0, 0), (10.5, 0, 0),
             (11.5, 40, 0), (13, -40, 0), (14.5, 0, 0), (21, 0, 0), (22, 50, -6), (23.5, -50, 6), (25, 0, 0), (30, 0, 0)]


def add_head_layer(arm):
    """Additive NLA strip with Head + neck yaw / pitch keys (the shipped clips have almost no head turn)."""
    import mathutils
    ad = arm.animation_data
    act = bpy.data.actions.new('Ariel_HeadTurn')
    slot = act.slots.new('OBJECT', arm.name)
    arm.animation_data.action = act
    arm.animation_data.action_slot = slot
    for bn, share in (('Head', 0.7), ('neck', 0.3)):
        pb = arm.pose.bones[bn]
        pb.rotation_mode = 'QUATERNION'
        for t, yaw, pitch in HEAD_KEYS:
            q = Quaternion((0, 1, 0), math.radians(yaw * share)) @ Quaternion((1, 0, 0), math.radians(pitch * share))
            pb.rotation_quaternion = q
            pb.keyframe_insert('rotation_quaternion', frame=1 + t * FPS)
    ad.action = None
    tr = ad.nla_tracks.new()
    tr.name = 'HeadLayer'
    st = tr.strips.new('HeadTurn', 1, act)
    st.blend_type = 'COMBINE'
    st.extrapolation = 'HOLD'
    try:
        st.action_slot = slot
    except Exception:
        pass
    # pose_bone rotations keyed above must not remain on the live pose
    for bn in ('Head', 'neck'):
        arm.pose.bones[bn].rotation_quaternion = (1, 0, 0, 0)
    return act


def build_timeline(arm):
    ad = arm.animation_data or arm.animation_data_create()
    for tr in list(ad.nla_tracks):
        ad.nla_tracks.remove(tr)
    f = 1
    tr = ad.nla_tracks.new()
    tr.name = 'Clips'
    for i, (an, nf, a0) in enumerate(TIMELINE):
        act = bpy.data.actions[an]
        st = tr.strips.new('%s_%d' % (an, i), int(f), act)
        try:
            st.action_slot = act.slots[0]
        except Exception:
            pass
        a1 = act.frame_range[1]
        L = max(a1 - a0, 1.0)
        st.action_frame_start, st.action_frame_end = a0, a1
        st.repeat = nf / L
        st.frame_end = f + nf
        st.blend_in = st.blend_out = 0
        st.extrapolation = 'HOLD' if i == len(TIMELINE) - 1 else 'NOTHING'
        f += nf
    ad.action = None
    add_head_layer(arm)
    sc.frame_start, sc.frame_end = 1, 1 + SECONDS * FPS
    sc.render.fps = FPS
    return f


def setup_studio(arm):
    bpy.ops.mesh.primitive_plane_add(size=14.0, location=(0.0, 0.0, 0.0))
    floor = bpy.context.object
    floor.name = 'Ariel_Floor'
    m = AI.make_material('Ariel Floor', (0.10, 0.11, 0.13, 1.0), rough=0.55)
    floor.data.materials.append(m)
    bpy.ops.mesh.primitive_plane_add(size=14.0, location=(0.0, 2.2, 3.0), rotation=(math.radians(90), 0, 0))
    wall = bpy.context.object
    wall.name = 'Ariel_Backdrop'
    wall.data.materials.append(AI.make_material('Ariel Backdrop', (0.13, 0.15, 0.19, 1.0), rough=0.8))
    world = sc.world or bpy.data.worlds.new('Ariel_World')
    sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get('Background')
    bg.inputs[0].default_value = (0.30, 0.34, 0.42, 1.0)
    bg.inputs[1].default_value = 0.9

    def lamp(name, kind, loc, rot, energy, color, size=1.0):
        L = bpy.data.lights.new(name, kind)
        L.energy = energy
        L.color = color
        if kind == 'AREA':
            L.size = size
        if kind == 'SUN':
            L.angle = math.radians(3)
        L.use_shadow = True
        o = bpy.data.objects.new(name, L)
        sc.collection.objects.link(o)
        o.location = loc
        o.rotation_euler = rot
        return o
    # key light from front-left-above (shadow of the hair on the back / shoulders), cool fill, warm rim from behind
    lamp('Ariel_Sun', 'SUN', (2, -3, 5), (math.radians(48), 0, math.radians(35)), 2.6, (1.0, 0.95, 0.88))
    lamp('Ariel_Fill', 'AREA', (-1.8, -1.6, 1.7), (math.radians(80), 0, math.radians(-50)), 140, (0.7, 0.8, 1.0), 1.6)
    lamp('Ariel_Rim', 'AREA', (0.8, 1.7, 2.1), (math.radians(-45), 0, math.radians(200)), 220, (1.0, 0.85, 0.7), 0.8)
    cam = bpy.data.cameras.new('Ariel_Cam')
    cam.lens = 60
    co = bpy.data.objects.new('Ariel_Cam', cam)
    sc.collection.objects.link(co)
    co.location = (0.55, -2.6, 1.45)
    co.rotation_euler = (Vector((0.0, 0.1, 1.22)) - co.location).to_track_quat('-Z', 'Y').to_euler()
    sc.camera = co
    return co


def configure_hair(S, collide=True):
    hp = sc.arc_flow_hair
    hp.hair = S['hair']
    hp.armature = S['arm']
    hp.bone = 'Head'
    hp.collider = None
    hp.surface = None
    hp.use_scene_colliders = False
    hp.colliders.clear()
    for o in S['colliders']:
        HO.mark_proxy(o)
        c = HO.add_collider_entry(sc, o, thickness=0.004 if o.name == 'Col Head' else 0.006, friction=0.35)
        c.deforming = False
        c.enabled = bool(collide)
    hp.dt_mode = 'SCENE'
    hp.substeps = 4
    hp.collision_radius = 0.005
    hp.collision_iterations = 2
    hp.collision_friction = 0.35
    hp.sdf_resolution = 48
    hp.sdf_level = 1
    hp.damping = 0.93
    hp.rest_shape = 0.30
    hp.rest_shape_root = 0.85
    hp.rest_shape_tip = 0.08
    hp.stiffness_curve = 1.4
    hp.stretch_iterations = 8
    hp.bend_iterations = 3
    hp.hair_collision = True
    hp.strand_radius = 0.0025
    hp.repel_strength = 0.4
    hp.volume = 0.3
    hp.clumping = 0.2
    hp.use_env_wind = False
    hp.wind = (0.0, 0.0, 0.0)
    hp.use_scene_gravity = False
    hp.gravity = (0.0, 0.0, -9.81)
    hp.guide_ratio = 0.25
    hp.pinned_prefix = 2
    hp.sdf_refine = False        # analytic-quality fields from tessellated capsules: the LOTBCBP refine pass only costs time here
    hp.clumping = 0.0
    hp.strand_radius = 0.0014
    hp.volume = 0.3
    hp.use_colliders = True
    hp.show_colliders = False
    return hp


def build_scene(game_dir):
    for nm in list(bpy.data.objects.keys()):
        bpy.data.objects.remove(bpy.data.objects[nm], do_unlink=True)
    S = AI.build_ariel(game_dir)
    build_timeline(S['arm'])
    setup_studio(S['arm'])
    configure_hair(S)
    HO.set_collider_overlay(sc, False)
    sc.frame_set(1)
    bpy.context.view_layer.update()
    return S


# ----------------------------------------------------------------------------- measuring
class Meter:
    """Strand points inside the collision volumes, per frame (independent of the solver's own fields):
    capsules analytically (game's fitted values, posed with the skeleton), the groom's own head SDF, and the posed skin / shirt meshes."""

    def __init__(self, S, skip_first=3):
        self.S = S
        self.arm = S['arm']
        self.nw = S['node_world']
        self.hd = S['hairdata']
        self.skip = skip_first
        N = S['hair'].data.curves.__len__()
        K = self.hd['rest'].shape[1]
        self.mask = np.tile(np.arange(K) >= skip_first, N)          # ignore the pinned root points
        self.fix = {}
        for lab, bone, members in AI.COLLIDER_GROUPS:
            for m in members:
                if m not in self.fix:
                    self.fix[m] = AI.node_fixup(self.arm, m, self.nw)
        self.fix['Head'] = AI.node_fixup(self.arm, 'Head', self.nw)
        self.caps = [c for c in AI.BODY_CAPSULES if any(c[0] in g[2] for g in AI.COLLIDER_GROUPS)]
        self.mesh_names = ['Shirt.001', 'Ariel Reach.001']

    def frame_of(self, bone):
        return np.array(self.arm.matrix_world @ self.arm.pose.bones[bone].matrix, dtype=np.float64) @ self.fix[bone]

    def capsule_inside(self, P):
        inside = np.zeros(len(P), bool)
        per = {}
        for (bn, a, b, r) in self.caps:
            M = self.frame_of(bn)
            Wa = M @ np.array([*a, 1.0])
            Wb = M @ np.array([*b, 1.0])
            ab = Wb[:3] - Wa[:3]
            t = np.clip(((P - Wa[:3]) @ ab) / max(ab @ ab, 1e-12), 0, 1)
            d = np.linalg.norm(P - (Wa[:3] + t[:, None] * ab), axis=1)
            hit = d < r
            inside |= hit
        return inside

    def head_inside(self, P):
        M = self.frame_of('Head')
        L = (P - M[:3, 3]) @ np.linalg.inv(M[:3, :3]).T
        return AI.sample_sdf(self.hd['sdf'], L) < 0.0

    def mesh_inside(self, P, deps):
        from mathutils.bvhtree import BVHTree
        out = {}
        for nm in self.mesh_names:
            ob = bpy.data.objects[nm]
            ev = ob.evaluated_get(deps)
            me = ev.to_mesh()
            try:
                M = np.array(ob.matrix_world)
                V = np.array([(ob.matrix_world @ v.co)[:] for v in me.vertices])
                me.calc_loop_triangles()
                tris = [tuple(t.vertices) for t in me.loop_triangles]
                bvh = BVHTree.FromPolygons([tuple(v) for v in V], tris, epsilon=0.0)
            finally:
                ev.to_mesh_clear()
            lo, hi = V.min(0) - 0.01, V.max(0) + 0.01
            cand = np.flatnonzero(np.all((P >= lo) & (P <= hi), axis=1))
            n_in = 0
            flag = np.zeros(len(P), bool)
            for i in cand:
                p = Vector(P[i])
                r = bvh.find_nearest(p, 0.04)
                if r[0] is None:
                    continue
                if (p - r[0]).dot(r[1]) < -1e-4:
                    flag[i] = True
            out[nm] = flag
        return out

    def measure(self, P, deps=None, with_mesh=False):
        P = np.asarray(P, np.float64)
        Pm = P[self.mask]
        cap = self.capsule_inside(Pm)
        hd = self.head_inside(Pm)
        res = dict(capsule=int(cap.sum()), head=int(hd.sum()), volumes=int((cap | hd).sum()), points=int(len(Pm)))
        if with_mesh:
            fl = self.mesh_inside(Pm, deps)
            for k, v in fl.items():
                res['mesh:' + k] = int(v.sum())
        return res


# ----------------------------------------------------------------------------- bake
def bake(S, collide=True, seconds=SECONDS, stride=2, out_prefix=None, progress=None, mesh_every=6, settle=40, f0=0):
    """Settle on the body, then run the clip. Returns (frames array [kept, N, 3] float32, metrics list)."""
    hp = sc.arc_flow_hair
    for c in hp.colliders:
        c.enabled = bool(collide)
    HO.RT.solver = None
    HO.RT.bind_ok = False
    sc.frame_set(1)
    bpy.context.view_layer.update()
    HO.build_solver(sc)
    solver = HO.RT.solver
    if settle:
        HO.settle_scene(sc, steps=settle)
    meter = Meter(S)
    nfr = int(seconds * FPS)
    metrics = []
    keep = []
    t0 = time.time()
    deps = bpy.context.evaluated_depsgraph_get()
    for k in range(nfr):
        f = 1 + k
        sc.frame_set(f)
        bpy.context.view_layer.update()
        HO.step_scene(sc, dt=1.0 / FPS, write=False)
        full, _ = HO._compose_full(solver)
        if k % stride == 0:
            keep.append(full.astype(np.float32))
        deps = bpy.context.evaluated_depsgraph_get()
        m = meter.measure(full, deps, with_mesh=(k % mesh_every == 0))
        m['frame'] = f
        m['contacts'] = int(solver.last_contacts)
        metrics.append(m)
        if progress and k % 12 == 0:
            progress(k, nfr, time.time() - t0, m)
    return np.array(keep), metrics


def apply_cached_frame(S, frames, k):
    """Write kept frame k (world space) into the CURVES object (matrix_world of the current pose is used for the conversion)."""
    HO.write_curves(S['hair'], frames[min(k, len(frames) - 1)])


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    if not argv:
        print(__doc__)
        return
    mode, out, game = argv[0], argv[1], argv[2]
    opt = argv[3:]
    secs = float(opt[opt.index('--seconds') + 1]) if '--seconds' in opt else SECONDS
    S = build_scene(game)

    def prog(k, n, dt, m):
        print('  sim %d/%d %.0fs  volumes=%d (cap %d head %d) contacts=%d' % (k, n, dt, m['volumes'], m['capsule'], m['head'], m['contacts']), flush=True)
    if mode == 'bake':
        frames, metrics = bake(S, collide='--no-collide' not in opt, seconds=secs, progress=prog)
        import json
        np.save(out + '.npy', frames)
        json.dump(metrics, open(out + '.json', 'w'))
        print('baked', frames.shape)


if __name__ == '__main__' or os.environ.get('ARIEL_RUN'):
    main()
