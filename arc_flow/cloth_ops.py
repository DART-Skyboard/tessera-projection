"""Tessera Projection cloth solver: Blender bridge, live mode, example scene, bake and export.

Live mode: a modal timer operator (ARCFLOW_OT_cloth_live) ticks the NumPy solver at
`live_fps`, reading the evaluated collider meshes and armature pose each tick, so the
garment responds to posing, keyframe playback or a controller/keyboard-driven rig.
A frame_change_post handler also steps the cloth while the timeline plays.
"""
import json
import math
import os
import time

import bpy
import numpy as np
from bpy.props import EnumProperty, IntProperty
from bpy.types import Operator

from . import cloth_gn as CG
from .core import cloth_solver as CS


class _Runtime:
    def __init__(self):
        self.solvers = {}         # garment name -> ClothSolver
        self.sig = {}             # garment name -> build signature
        self.layout = {}          # garment name -> [(object, vertex start, vertex end)]
        self.welds = {}           # garment name -> number of weld constraints
        self.prev_frame = None
        self.live_running = False
        self.last_tick = 0.0

    def reset(self):
        self.solvers.clear()
        self.sig.clear()
        self.layout.clear()
        self.welds.clear()
        self.prev_frame = None


RT = _Runtime()


# ----------------------------------------------------------------------------- helpers
def cloth(scene):
    return scene.arc_flow_cloth


def _mesh_world(obj, deps=None):
    """World-space verts (N,3) and triangles (F,3) of an object (evaluated if deps given)."""
    if deps is not None:
        ev = obj.evaluated_get(deps)
        me = ev.to_mesh()
    else:
        ev, me = None, obj.data
    try:
        me.calc_loop_triangles()
        v = np.empty(len(me.vertices) * 3, np.float64)
        me.vertices.foreach_get('co', v)
        t = np.empty(len(me.loop_triangles) * 3, np.int64)
        me.loop_triangles.foreach_get('vertices', t)
    finally:
        if ev is not None:
            ev.to_mesh_clear()
    M = np.array(obj.matrix_world, float)
    v = v.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
    return v, t.reshape(-1, 3)


def _rest_world(obj):
    """Base (undeformed) mesh of the garment in world space."""
    return _mesh_world(obj, None)


def _write_world(obj, X):
    M = np.array(obj.matrix_world, float)
    Mi = np.linalg.inv(M)
    local = X @ Mi[:3, :3].T + Mi[:3, 3]
    me = obj.data
    if len(me.vertices) != len(local):
        return False
    me.vertices.foreach_set('co', local.astype(np.float32).ravel())
    me.update()
    return True


def _pin_weights(obj, group_name):
    vg = obj.vertex_groups.get(group_name) if group_name else None
    n = len(obj.data.vertices)
    w = np.zeros(n, float)
    if vg is None:
        return w
    gi = vg.index
    for v in obj.data.vertices:
        for g in v.groups:
            if g.group == gi:
                w[v.index] = g.weight
                break
    return w


def _auto_pin_indices(verts_world, mode):
    if mode == 'TOP':
        z = verts_world[:, 2]
        return np.nonzero(z >= z.max() - 1e-4 - 0.02 * max(z.max() - z.min(), 1e-6))[0]
    if mode == 'SHOULDER':
        return CS.shoulder_pins(verts_world, frac=0.18, top_frac=0.93)
    if mode == 'WAIST':
        return CS.band_pins(verts_world, axis=2, lo=0.88, hi=1.0)
    return np.zeros(0, np.int64)


def write_pin_group(obj, indices, name="ClothPins"):
    vg = obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
    vg.remove(list(range(len(obj.data.vertices))))
    if len(indices):
        vg.add([int(i) for i in indices], 1.0, 'REPLACE')
    return vg


def _env_wind(scene):
    cl = cloth(scene)
    wind = np.array(cl.wind, float)
    g = np.array(cl.gravity, float)
    rho = float(cl.air_density)
    if cl.use_scene_wind:
        af = getattr(scene, 'arc_flow', None)
        env = getattr(af, 'env', None) if af is not None else None
        if env is not None:
            w = np.array(env.wind, float)
            if np.linalg.norm(w) > 1e-9:
                wind = w
            rho = float(env.density) if env.density > 0 else rho
            if env.use_scene_gravity:
                g = np.array(scene.gravity, float) if scene.use_gravity else np.zeros(3)
            else:
                g = np.array(env.gravity, float)
    return wind, g, rho


def _fluid_field(scene):
    """Return a callable P -> velocity sampled from a running Fluid-mode solver (or None)."""
    if not cloth(scene).use_fluid_velocity:
        return None
    try:
        from . import fluidmode_ops as FO
        S = FO.RT.solver
    except Exception:
        return None
    if S is None or getattr(S, 'n', 0) < 8:
        return None
    X, V = np.asarray(S.x), np.asarray(S.v)
    if len(X) > 3000:
        step = max(1, len(X) // 2000)
        X, V = X[::step], V[::step]
    h = float(getattr(S, 'dx', 0.05)) * 3.0

    def sample(P):
        out = np.zeros_like(P)
        # cheap kernel average over the nearest particles (chunked brute force)
        for a in range(0, len(P), 256):
            p = P[a:a + 256]
            d2 = ((p[:, None, :] - X[None, :, :]) ** 2).sum(-1)
            wgt = np.maximum(0.0, 1.0 - d2 / (h * h))
            s = wgt.sum(1, keepdims=True)
            out[a:a + 256] = np.where(s > 1e-9, (wgt @ V) / np.maximum(s, 1e-9), 0.0)
        return out
    return sample


def _params(g):
    kw = {}
    if not g.follow_preset:
        kw.update(density=g.density, stretch=g.stretch, shear=g.shear, bend=g.bend, damping=g.damping,
                  drag=g.drag, lift=g.lift, thickness=g.thickness, margin=g.margin, friction=g.friction)
    if g.override_quality:
        kw.update(substeps=g.substeps, iterations=g.iterations,
                  use_dihedral=g.use_dihedral, tessellation_edge=g.tessellation_edge,
                  tessellation_cap=g.tessellation_cap)
    # self-collision is a per-garment toggle (default on) and is never dropped by a quality profile
    kw['self_collision'] = bool(getattr(g, 'self_collision', True))
    mode = getattr(g, 'self_collision_mode', 'AUTO')
    if mode == 'AUTO':
        mode = 'FAST' if g.quality == 'REALTIME' else 'LAYERED'
    kw['self_collision_mode'] = mode
    kw['self_thickness'] = float(getattr(g, 'self_thickness', 0.004))
    kw['self_friction'] = float(getattr(g, 'self_friction', 0.3))
    kw['self_edge_contacts'] = bool(getattr(g, 'self_edge_contacts', False))
    kw.setdefault('thickness', float(g.thickness))
    kw.setdefault('margin', float(g.margin))
    return kw


def _signature(g, rest, mem=None):
    base = (g.obj.name if g.obj else "", len(rest), g.preset, g.quality, g.follow_preset, g.override_quality,
            g.pin_group, round(float(g.pin_threshold), 4),
            tuple(round(float(getattr(g, k)), 6) for k in ("density", "stretch", "shear", "bend", "damping", "drag",
                                                            "lift", "thickness", "margin", "friction",
                                                            "tessellation_edge")),
            int(g.substeps), int(g.iterations), bool(g.self_collision), bool(g.use_dihedral),
            g.self_collision_mode, round(float(g.self_thickness), 6), round(float(g.self_friction), 4), g.self_ignore_group, bool(g.self_edge_contacts))
    if mem:
        base += (bool(g.use_stitch), round(float(g.stitch_tolerance), 6), round(float(g.stitch_stiffness), 1),
                 tuple((o.name, m.preset, m.use_preset, m.weld) + tuple(round(float(getattr(m, k)), 6) for k in
                       ("density", "stretch", "shear", "bend", "damping", "thickness", "friction")) for o, m in mem if m is not None))
    return base


def _armature_pose(arm):
    """(bone names, current world matrices (B,4,4), rest world matrices (B,4,4))."""
    if arm is None or arm.type != 'ARMATURE':
        return None
    M = np.array(arm.matrix_world, float)
    names, cur, rest = [], [], []
    for pb in arm.pose.bones:
        names.append(pb.name)
        cur.append(M @ np.array(pb.matrix, float))
        rest.append(M @ np.array(pb.bone.matrix_local, float))
    return names, np.array(cur), np.array(rest)


def _skin_arrays(obj, arm, bone_names, k=4):
    n = len(obj.data.vertices)
    idx = np.full((n, k), -1, np.int64)
    w = np.zeros((n, k), float)
    lut = {}
    for vg in obj.vertex_groups:
        if vg.name in bone_names:
            lut[vg.index] = bone_names.index(vg.name)
    if not lut:
        return None, None
    for v in obj.data.vertices:
        pairs = sorted(((g.weight, lut[g.group]) for g in v.groups if g.group in lut and g.weight > 0), reverse=True)[:k]
        for c, (ww, bi) in enumerate(pairs):
            idx[v.index, c] = bi
            w[v.index, c] = ww
    return idx, w


def _seed_vertical_folds(S, n_pleats=5, amp=0.042):
    """Push free vertices into long vertical pleats so surplus pattern length buckles."""
    x = S.x
    dx, dy = x[:, 0], x[:, 1]
    ang = np.arctan2(dx, dy)  # 0 at the +Y back
    rad = np.sqrt(dx * dx + dy * dy)
    z = x[:, 2]
    v = (z - z.min()) / max(float(z.max() - z.min()), 1e-9)
    fade = np.clip(1.0 - v, 0.0, 1.0) ** 1.05  # zero at the collar
    # cos so a ridge sits on the visible back, not a node
    delta = amp * fade * np.cos(ang * float(n_pleats))
    rx = np.divide(dx, rad, out=np.zeros_like(dx), where=rad > 1e-6)
    ry = np.divide(dy, rad, out=np.zeros_like(dy), where=rad > 1e-6)
    free = ~S.pin_hard
    S.x[free, 0] = S.x[free, 0] + delta[free] * rx[free]
    S.x[free, 1] = S.x[free, 1] + delta[free] * ry[free]
    S.x_prev = S.x.copy()
    S.v[:] = 0.0



# ----------------------------------------------------------------------------- garment groups
def _is_excluded_member(scene, o):
    """Objects that must never be a member of a garment group: registered collision bodies and other garments."""
    cl = cloth(scene)
    for c in cl.colliders:
        if c.obj == o and not o.arc_flow_cloth.is_garment:
            return True
    return False


def collection_meshes(scene, coll):
    """Mesh objects of ``coll`` (recursive), stable name order, without registered collision bodies."""
    if coll is None:
        return []
    out = [o for o in coll.all_objects if o.type == 'MESH' and not _is_excluded_member(scene, o)]
    return sorted(out, key=lambda o: o.name)


def sync_members(scene, g):
    """Bring ``g.members`` in line with the group collection (new objects are added with the garment preset's
    fabric, removed objects dropped).  Returns the number of members."""
    if g.collection is not None:
        objs = collection_meshes(scene, g.collection)
        have = {m.obj.name: i for i, m in enumerate(g.members) if m.obj is not None}
        for i in reversed(range(len(g.members))):
            m = g.members[i]
            if m.obj is None or m.obj not in objs:
                g.members.remove(i)
        have = {m.obj.name for m in g.members if m.obj is not None}
        for o in objs:
            if o.name not in have:
                add_member(g, o)
    g.n_members = len(g.members)
    return len(g.members)


def add_member(g, obj, preset=None):
    for m in g.members:
        if m.obj == obj:
            return m
    m = g.members.add()
    m.name = obj.name
    m.obj = obj
    m.preset = preset or g.preset
    pr = CS.PRESETS[m.preset]
    for k in ("density", "stretch", "shear", "bend", "damping", "thickness", "friction"):
        setattr(m, k, float(pr[k]))
    return m


def member_fabric(m):
    """Fabric parameter dict of a member (preset values, or the typed ones)."""
    if m.use_preset:
        pr = CS.PRESETS[m.preset]
        return {k: float(pr[k]) for k in ("density", "stretch", "shear", "bend", "damping", "thickness", "friction")}
    return {k: float(getattr(m, k)) for k in ("density", "stretch", "shear", "bend", "damping", "thickness", "friction")}


def garment_members(scene, g):
    """[(object, member-or-None)] of garment ``g`` in layer order.  A plain single-object garment has no members."""
    if g.collection is not None:
        sync_members(scene, g)
    if len(g.members):
        out = [(m.obj, m) for m in g.members if m.enabled and m.obj is not None and m.obj.type == 'MESH']
        if out:
            return out
    return [(g.obj, None)] if g.obj is not None else []


def border_vertices(n, tris):
    """Indices of vertices on a mesh border (edges used by exactly one triangle)."""
    t = np.asarray(tris, np.int64)
    e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    e.sort(axis=1)
    key = e[:, 0] * n + e[:, 1]
    u, cnt = np.unique(key, return_counts=True)
    b = u[cnt == 1]
    return np.unique(np.concatenate([b // n, b % n])) if len(b) else np.zeros(0, np.int64)


def detect_welds(rest, tris, member, tol, allowed=None):
    """Weld constraints between members: every border vertex of a member that lies within ``tol`` of another
    member's surface is attached to it.  Returns (vert, tri(3), bary(3), distance) arrays.  A vertex-to-vertex
    weld has ``bary = (1, 0, 0)``.  ``allowed`` is an optional per-member bool array (members that may weld)."""
    from .core import cloth_selfcol as SC
    n = len(rest)
    empty = (np.zeros(0, np.int64), np.zeros((0, 3), np.int64), np.zeros((0, 3)), np.zeros(0))
    if tol <= 0 or len(np.unique(member)) < 2:
        return empty
    bv = border_vertices(n, tris)
    if allowed is not None:
        bv = bv[np.asarray(allowed, bool)[member[bv]]]
    if len(bv) == 0:
        return empty
    T = np.asarray(tris, np.int64)
    A, B, C = rest[T[:, 0]], rest[T[:, 1]], rest[T[:, 2]]
    lo = np.minimum(np.minimum(A, B), C) - tol
    hi = np.maximum(np.maximum(A, B), C) + tol
    tmem = member[T[:, 0]]
    outv, outt, outb, outd = [], [], [], []
    best = {}
    for s0 in range(0, len(bv), 128):
        ids = bv[s0:s0 + 128]
        P = rest[ids]
        near = ((P[:, None, :] >= lo[None]) & (P[:, None, :] <= hi[None])).all(-1) & (member[ids][:, None] != tmem[None])
        pi, ti = np.nonzero(near)
        if len(pi) == 0:
            continue
        q, bary, _in = SC.closest_point_triangle(P[pi], A[ti], B[ti], C[ti])
        dist = np.linalg.norm(P[pi] - q, axis=1)
        ok = dist <= tol
        for a_, t_, b_, d_ in zip(ids[pi[ok]], ti[ok], bary[ok], dist[ok]):
            cur = best.get(int(a_))
            if cur is None or d_ < cur[0] - 1e-12:
                best[int(a_)] = (float(d_), int(t_), b_)
    seen = set()
    for a_, (d_, t_, b_) in sorted(best.items()):
        tri = T[t_].copy()
        b_ = np.clip(np.asarray(b_, float), 0.0, 1.0)
        b_ = b_ / max(b_.sum(), 1e-12)
        k = int(np.argmax(b_))
        if b_[k] > 0.85:                       # a vertex-to-vertex weld
            tri = np.array([tri[k], tri[k], tri[k]])
            b_ = np.array([1.0, 0.0, 0.0])
            pair = tuple(sorted((a_, int(tri[0]))))
            if pair in seen:
                continue
            seen.add(pair)
        outv.append(a_); outt.append(tri); outb.append(b_); outd.append(d_)
    if not outv:
        return empty
    return np.array(outv, np.int64), np.array(outt, np.int64), np.array(outb, float), np.array(outd, float)


def garment_vertex_count(scene, g):
    return sum(len(o.data.vertices) for o, _m in garment_members(scene, g))


def build_solver(scene, g):
    """(Re)build the NumPy solver for one garment (a single object, or a group of objects solved as one system)."""
    mem = garment_members(scene, g)
    objs = [o for o, _m in mem]
    group = len(objs) > 1 or (len(objs) == 1 and mem[0][1] is not None)
    rests, tris_l, pins_l, offs = [], [], [], []
    off = 0
    for k, (o, m) in enumerate(mem):
        rest_k, tris_k = _rest_world(o)
        pw = _pin_weights(o, g.pin_group)
        pk = np.nonzero(pw >= g.pin_threshold)[0]
        if len(pk) == 0 and g.pin_mode != 'NONE' and k == 0:      # auto-pin the anchor (first) member only
            pk = _auto_pin_indices(rest_k, g.pin_mode)
            write_pin_group(o, pk, g.pin_group or "ClothPins")
        rests.append(rest_k)
        tris_l.append(tris_k + off)
        pins_l.append(pk + off)
        offs.append((o, off, off + len(rest_k)))
        off += len(rest_k)
    rest = np.concatenate(rests)
    tris = np.concatenate(tris_l)
    pins = np.concatenate(pins_l) if pins_l else np.zeros(0, np.int64)
    obj = g.obj if g.obj is not None else objs[0]
    wind, grav, rho = _env_wind(scene)
    kw = _params(g)
    member_ids = None
    if group:
        member_ids = np.concatenate([np.full(e - s0, k, np.int64) for k, (_o, s0, e) in enumerate(offs)])
        kw['members'] = member_ids
        kw['member_params'] = [member_fabric(m) if m is not None else {} for _o, m in mem]
    S = CS.ClothSolver(rest, tris, pins=pins, preset=g.preset, quality=g.quality, gravity=grav,
                       air_density=rho, **kw)
    # a cutting pattern (surplus length) may be stashed on any member object; concatenated, rest where missing
    pats, have_pat = [], False
    for (o, s0, e), rk in zip(offs, rests):
        if "tessera_cloth_pattern" in o:
            try:
                pat = np.asarray(o["tessera_cloth_pattern"], float).reshape(-1, 3)
                if len(pat) == len(rk):
                    pats.append(pat)
                    have_pat = True
                    continue
            except Exception:
                pass
        pats.append(rk)
    # armature skin: weights per member, concatenated
    pose = _armature_pose(g.armature)
    if pose is not None:
        names, cur, restm = pose
        idx_l, w_l, any_skin = [], [], False
        for (o, s0, e) in offs:
            idx, w = _skin_arrays(o, g.armature, names)
            if idx is None:
                idx = np.full((e - s0, 4), -1, np.int64)
                w = np.zeros((e - s0, 4), float)
            else:
                any_skin = True
            idx_l.append(idx)
            w_l.append(w)
        if any_skin:
            S.set_skin(np.concatenate(idx_l), np.concatenate(w_l), restm, threshold=g.pin_threshold)
            if len(pins):
                S.pin_hard[pins] = True
                S.inv_mass[pins] = 0.0
            S._bone_names = names
    S._collider_sig = None
    if g.self_ignore_group:
        ign = np.zeros(S.n, bool)
        for (o, s0, e) in offs:
            if o.vertex_groups.get(g.self_ignore_group) is not None:
                ign[s0:e] = _pin_weights(o, g.self_ignore_group) >= 0.5
        if ign.any():
            S.set_self_layers(np.where(ign, -1, 0))
    # stitches between members that touch
    nweld = 0
    if group and g.use_stitch and g.stitch_tolerance > 0:
        allowed = np.array([bool(m.weld) if m is not None else True for _o, m in mem])
        wv, wt, wb, wd = detect_welds(rest, tris, member_ids, float(g.stitch_tolerance), allowed)
        if len(wv):
            S.set_stitches(wv, wt, wb, wd, stiffness=float(g.stitch_stiffness))
            nweld = len(wv)
    RT.solvers[g.name] = S
    RT.layout[g.name] = offs
    RT.welds[g.name] = nweld
    RT.sig[g.name] = _signature(g, rest, mem if group else None)
    g.n_verts = S.n
    g.n_pins = int(S.pin_hard.sum())
    g.n_members = len(g.members)
    g.n_stitches = nweld
    for k, (o, s0, e) in enumerate(offs):
        o.arc_flow_cloth.is_garment = True
        o.arc_flow_cloth.garment_name = g.name
        if mem[k][1] is not None:
            mem[k][1].n_verts = e - s0
    if have_pat:
        try:
            S.set_pattern_rest(np.concatenate(pats))
            _seed_vertical_folds(S)
        except Exception:
            pass
    try:
        deps = bpy.context.evaluated_depsgraph_get()
        cv, ct, _names = _collider_mesh(scene, deps, exclude=objs)
        if cv is not None:
            S.set_collider_mesh(cv, ct)
            S._collider_sig = (len(cv), float(cv.sum()), float((cv * cv).sum()))
    except Exception:
        pass
    return S


def get_solver(scene, g, rebuild=False):
    S = RT.solvers.get(g.name)
    if S is None or rebuild:
        return build_solver(scene, g)
    if g.obj is not None and S.n != garment_vertex_count(scene, g):
        return build_solver(scene, g)
    return S


def write_garment(g, X, scene=None):
    """Write solver positions ``X`` (N,3 world) back into the member objects, one slice per object."""
    lay = RT.layout.get(g.name)
    if not lay:
        return _write_world(g.obj, X) if g.obj is not None else False
    ok = True
    for o, s0, e in lay:
        ok &= _write_world(o, X[s0:e])
    return ok


def _collider_mesh(scene, deps, exclude=None):
    """World-space triangle soup of every Tessera collision body except ``exclude`` (the garment itself)."""
    from . import asset_colliders as AC
    vs, ts, off = [], [], 0
    names = []
    seen = set()
    if exclude is None:
        skip = set()
    elif isinstance(exclude, (list, tuple, set)):
        skip = {id(o) for o in exclude}
    else:
        skip = {id(exclude)}

    def add(obj):
        nonlocal off
        if obj is None or obj.type != 'MESH' or id(obj) in skip or id(obj) in seen:
            return
        seen.add(id(obj))
        v, t = _mesh_world(obj, deps)
        if len(t) == 0:
            return
        vs.append(v); ts.append(t + off); off += len(v)
        names.append(obj.name)

    cl = cloth(scene)
    for c in cl.colliders:
        if c.enabled:
            add(c.obj)
    from .cloth_props import garment_objects
    for g in cl.garments:
        if g.enabled:
            for o in garment_objects(g):
                add(o)
    for o in AC.iter_collision_meshes(scene, exclude=exclude):
        add(o)
    if not vs:
        return None, None, names
    return np.concatenate(vs), np.concatenate(ts), names


def step_scene(scene, dt, deps=None):
    """Advance every enabled garment by dt seconds and write the result into its mesh."""
    cl = cloth(scene)
    if deps is None:
        deps = bpy.context.evaluated_depsgraph_get()
    wind, grav, rho = _env_wind(scene)
    fluid = _fluid_field(scene)
    t0 = time.perf_counter()
    stepped = 0
    for g in cl.garments:
        if not g.enabled or g.obj is None:
            continue
        S = get_solver(scene, g)
        S.air_density = rho
        # collider soup: every other projection body and garment, never the garment's own members
        own = [o for o, _s, _e in RT.layout.get(g.name, [(g.obj, 0, 0)])]
        cv, ct, names = _collider_mesh(scene, deps, exclude=own)
        if cv is not None:
            sig = (len(cv), float(cv.sum()), float((cv * cv).sum()))
            if getattr(S, '_collider_sig', None) != sig:
                S.set_collider_mesh(cv, ct)
                S._collider_sig = sig
        else:
            S.clear_collider()
        bones = None
        if S.skin_idx is not None and g.armature is not None:
            pose = _armature_pose(g.armature)
            if pose is not None:
                bones = pose[1]
        S.step(dt, wind=wind, gravity=grav, bone_matrices=bones, fluid_velocity=fluid)
        write_garment(g, S.x)
        spd = np.linalg.norm(S.v, axis=1)
        for o, s0, e in RT.layout.get(g.name, [(g.obj, 0, S.n)]):
            CG.write_attributes(o, tension=S.tension[s0:e], speed=spd[s0:e])
        g.last_ms = S.last_ms
        g.last_stretch = S.last_stretch
        g.last_penetration = S.max_penetration()
        stepped += 1
    el = 1000.0 * (time.perf_counter() - t0)
    cl.last_ms = el
    return stepped


def write_garment_rest(g, S):
    lay = RT.layout.get(g.name)
    if not lay:
        _write_world(g.obj, S.rest)
        return
    for o, s0, e in lay:
        _write_world(o, S.rest[s0:e])


def reset_scene(scene):
    for g in cloth(scene).garments:
        S = RT.solvers.get(g.name)
        if S is not None and g.obj is not None:
            write_garment_rest(g, S)
    RT.reset()


# ----------------------------------------------------------------------------- frame handler
@bpy.app.handlers.persistent
def _on_frame(scene, depsgraph=None):
    try:
        cl = getattr(scene, 'arc_flow_cloth', None)
        if cl is None or not cl.garments or RT.live_running:
            return
        if not any(g.enabled and g.obj is not None and g.name in RT.solvers for g in cl.garments):
            return
        f = scene.frame_current
        if RT.prev_frame is None or f <= scene.frame_start:
            if f <= scene.frame_start:
                reset_scene(scene)
            RT.prev_frame = f
            return
        if f == RT.prev_frame + 1:
            fps = scene.render.fps / max(scene.render.fps_base, 1e-6)
            step_scene(scene, 1.0 / fps)
        RT.prev_frame = f
    except Exception as ex:  # never break playback
        print("Tessera cloth frame handler:", ex)


# ----------------------------------------------------------------------------- operators
class _GOp:
    @classmethod
    def poll(cls, ctx):
        return getattr(ctx.scene, 'arc_flow_cloth', None) is not None


def _garment_objs(g):
    from .cloth_props import garment_objects
    return garment_objects(g)


def _active_garment(scene):
    cl = cloth(scene)
    if not cl.garments:
        return None
    return cl.garments[max(0, min(cl.index, len(cl.garments) - 1))]


def assign_garment(scene, obj, preset='CAPE', quality='BALANCED', pin_mode='SHOULDER', armature=None):
    cl = cloth(scene)
    for g in cl.garments:
        if g.obj == obj:
            return g
    g = cl.garments.add()
    base = obj.name
    names = {x.name for x in cl.garments}
    nm, k = base, 1
    while nm in names and nm != g.name:
        k += 1
        nm = "%s.%d" % (base, k)
    g.name = nm
    g.obj = obj
    g.preset = preset
    g.quality = quality
    g.pin_mode = pin_mode
    g.armature = armature
    pr = CS.PRESETS[preset]
    for k2 in ("density", "stretch", "shear", "bend", "damping", "drag", "lift", "thickness", "margin", "friction"):
        setattr(g, k2, float(pr[k2]))
    cl.index = len(cl.garments) - 1
    obj.arc_flow_cloth.is_garment = True
    obj.arc_flow_cloth.garment_name = g.name
    g.self_collision = True
    add_collider(scene, obj)
    from . import asset_colliders as AC
    AC.mark_projection_collider(obj)
    CG.ensure_modifier(obj)
    return g


def gather_group(ctx, source='AUTO'):
    """Objects of a garment group from the selection.  Returns (collection or None, [objects]).

    AUTO: several selected meshes that live in one (non-scene) collection -> that collection, so objects added to it
    later join the group; several meshes in different collections -> the selected set; one mesh with child meshes ->
    the object plus its children; a single mesh in a collection holding other meshes -> that collection."""
    sc = ctx.scene
    sel = [o for o in ctx.selected_objects if o.type == 'MESH' and not o.arc_flow_cloth.is_collider]
    act = ctx.object if ctx.object is not None and ctx.object.type == 'MESH' else None
    if act is not None and act not in sel and not act.arc_flow_cloth.is_collider:
        sel.append(act)
    if not sel:
        return None, []
    def common_coll(objs):
        cs = None
        for o in objs:
            cc = {c for c in o.users_collection if c != sc.collection}
            cs = cc if cs is None else (cs & cc)
        return sorted(cs, key=lambda c: len(c.all_objects)) if cs else []
    if source in ('AUTO', 'COLLECTION'):
        cc = common_coll(sel)
        if cc:
            ms = collection_meshes(sc, cc[0])
            if len(ms) > 1 or source == 'COLLECTION':
                return cc[0], ms
    if source in ('AUTO', 'SELECTION') and len(sel) > 1:
        return None, sorted(sel, key=lambda o: o.name)
    if source in ('AUTO', 'CHILDREN'):
        base = act or sel[0]
        kids = [c for c in base.children_recursive if c.type == 'MESH' and not c.arc_flow_cloth.is_collider]
        if kids:
            return None, [base] + sorted(kids, key=lambda o: o.name)
    return None, sel[:1]


def assign_group(scene, objs, collection=None, preset='CAPE', quality='BALANCED', pin_mode='NONE', armature=None,
                 fabrics=None, name=None):
    """Make ``objs`` ONE garment group (a single self-colliding system).  ``fabrics`` maps object name -> preset."""
    objs = [o for o in objs if o is not None and o.type == 'MESH']
    if not objs:
        return None
    cl = cloth(scene)
    first = objs[0]
    for g0 in cl.garments:
        if g0.obj == first and (len(g0.members) or collection is not None):
            return g0
    g = assign_garment(scene, first, preset=preset, quality=quality, pin_mode=pin_mode, armature=armature)
    if name:
        g.name = name
        first.arc_flow_cloth.garment_name = g.name
    g.collection = collection
    g.members.clear()
    for o in objs:
        add_member(g, o, preset=(fabrics or {}).get(o.name, preset))
        o.arc_flow_cloth.is_garment = True
        o.arc_flow_cloth.garment_name = g.name
        from . import asset_colliders as AC
        AC.mark_projection_collider(o)
        CG.ensure_modifier(o)
    g.n_members = len(g.members)
    g.self_collision = True
    return g


def add_collider(scene, obj):
    cl = cloth(scene)
    for c in cl.colliders:
        if c.obj == obj:
            return c
    c = cl.colliders.add()
    c.name = obj.name
    c.obj = obj
    obj.arc_flow_cloth.is_collider = True
    from . import asset_colliders as AC
    AC.mark_projection_collider(obj)
    cl.collider_index = len(cl.colliders) - 1
    return c


class ARCFLOW_OT_cloth_assign(_GOp, Operator):
    bl_idname = "arcflow.cloth_assign"
    bl_label = "Assign Cloth Garment"
    bl_description = "Make the selected mesh a Tessera cloth garment (auto-pins its top / shoulder ring)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        obs = [o for o in ctx.selected_objects if o.type == 'MESH'] or ([ctx.object] if ctx.object and ctx.object.type == 'MESH' else [])
        if not obs:
            self.report({'WARNING'}, "Select a mesh to use as the garment")
            return {'CANCELLED'}
        arm = next((o for o in ctx.selected_objects if o.type == 'ARMATURE'), None)
        # several selected meshes that live in one collection are ONE garment (a group), detected automatically
        if len(obs) > 1:
            coll, objs = gather_group(ctx, 'COLLECTION')
            if coll is not None and len(objs) > 1 and all(o in objs for o in obs):
                g = assign_group(ctx.scene, objs, collection=coll, armature=arm, name=coll.name)
                build_solver(ctx.scene, g)
                cloth(ctx.scene).status = "Group '%s': %d objects, %d stitches. Press Start Live or play the timeline." % (
                    g.name, len(g.members), g.n_stitches)
                return {'FINISHED'}
        for o in obs:
            if o.arc_flow_cloth.is_collider:
                continue
            g = assign_garment(ctx.scene, o, armature=arm)
            build_solver(ctx.scene, g)
        cloth(ctx.scene).status = "Garment assigned. Press Start Live or play the timeline."
        return {'FINISHED'}


class ARCFLOW_OT_cloth_assign_group(_GOp, Operator):
    bl_idname = "arcflow.cloth_assign_group"
    bl_label = "Assign Garment Group"
    bl_description = ("Make several mesh objects ONE garment group: they are solved together as a single self-colliding unit "
                      "(members collide with each other and with themselves, touching borders are stitched) and the whole group "
                      "collides with every collision body. Uses the shared collection, else the selected set, else the object and its children")
    bl_options = {'REGISTER', 'UNDO'}
    source: EnumProperty(name="From", items=[
        ('AUTO', "Auto", "Shared collection, else selected set, else object and children"),
        ('COLLECTION', "Collection", "Every mesh of the collection that holds the active object"),
        ('SELECTION', "Selection", "The selected meshes"),
        ('CHILDREN', "Object and children", "The active object and its child meshes")], default='AUTO')

    def execute(self, ctx):
        coll, objs = gather_group(ctx, self.source)
        if len(objs) < 2:
            self.report({'WARNING'}, "Select two or more meshes (or one mesh in a collection with others) to form a group")
            return {'CANCELLED'}
        arm = next((o for o in ctx.selected_objects if o.type == 'ARMATURE'), None)
        g = assign_group(ctx.scene, objs, collection=coll, armature=arm,
                         name=(coll.name if coll is not None else None))
        if g is None:
            return {'CANCELLED'}
        build_solver(ctx.scene, g)
        cloth(ctx.scene).status = "Group '%s': %d objects, %d stitches. Self-collision applies to the whole group." % (
            g.name, len(g.members), g.n_stitches)
        return {'FINISHED'}


class ARCFLOW_OT_cloth_group_refresh(_GOp, Operator):
    bl_idname = "arcflow.cloth_group_refresh"
    bl_label = "Refresh Group Members"
    bl_description = "Re-read the group collection (new objects join with the garment preset's fabric), re-detect stitches and rebuild the solver"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        g = _active_garment(ctx.scene)
        if g is None:
            return {'CANCELLED'}
        if g.collection is not None:
            sync_members(ctx.scene, g)
        for o in _garment_objs(g):
            o.arc_flow_cloth.is_garment = True
            o.arc_flow_cloth.garment_name = g.name
        build_solver(ctx.scene, g)
        cloth(ctx.scene).status = "Group '%s': %d objects, %d stitches." % (g.name, max(len(g.members), 1), g.n_stitches)
        return {'FINISHED'}


class ARCFLOW_OT_cloth_member_preset(_GOp, Operator):
    bl_idname = "arcflow.cloth_member_preset"
    bl_label = "Apply Member Fabric"
    bl_description = "Copy the fabric preset's values into the active group member"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        g = _active_garment(ctx.scene)
        if g is None or not len(g.members):
            return {'CANCELLED'}
        m = g.members[max(0, min(g.member_index, len(g.members) - 1))]
        for k, v in CS.PRESETS[m.preset].items():
            if k in ("density", "stretch", "shear", "bend", "damping", "thickness", "friction"):
                setattr(m, k, float(v))
        RT.solvers.pop(g.name, None)
        return {'FINISHED'}


class ARCFLOW_OT_cloth_remove(_GOp, Operator):
    bl_idname = "arcflow.cloth_remove"
    bl_label = "Remove Garment"
    bl_description = "Remove the active garment from the cloth solver (restores its rest shape)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        cl = cloth(ctx.scene)
        g = _active_garment(ctx.scene)
        if g is None:
            return {'CANCELLED'}
        S = RT.solvers.get(g.name)
        if S is not None and g.obj is not None:
            write_garment_rest(g, S)
        RT.solvers.pop(g.name, None)
        for o in _garment_objs(g):
            o.arc_flow_cloth.is_garment = False
        RT.layout.pop(g.name, None)
        cl.garments.remove(cl.index)
        cl.index = max(0, cl.index - 1)
        return {'FINISHED'}


class ARCFLOW_OT_cloth_auto_pin(_GOp, Operator):
    bl_idname = "arcflow.cloth_auto_pin"
    bl_label = "Auto-pin Garment"
    bl_description = "Write the pin vertex group from the Auto pin mode (top edge, shoulder ring or waist band)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        g = _active_garment(ctx.scene)
        if g is None or g.obj is None:
            self.report({'WARNING'}, "No garment")
            return {'CANCELLED'}
        S = RT.solvers.get(g.name)
        if S is not None:
            write_garment_rest(g, S)
        rest, _t = _rest_world(g.obj)
        mode = g.pin_mode if g.pin_mode != 'NONE' else 'TOP'
        idx = _auto_pin_indices(rest, mode)
        write_pin_group(g.obj, idx, g.pin_group or "ClothPins")
        build_solver(ctx.scene, g)
        self.report({'INFO'}, "%d pins (%s)" % (len(idx), mode.lower()))
        return {'FINISHED'}


class ARCFLOW_OT_cloth_apply_preset(_GOp, Operator):
    bl_idname = "arcflow.cloth_apply_preset"
    bl_label = "Apply Garment Preset"
    bl_description = "Copy the preset's density, stiffness, damping and aero values into the garment"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        g = _active_garment(ctx.scene)
        if g is None:
            return {'CANCELLED'}
        pr = CS.PRESETS[g.preset]
        for k in ("density", "stretch", "shear", "bend", "damping", "drag", "lift", "thickness", "margin", "friction"):
            setattr(g, k, float(pr[k]))
        RT.solvers.pop(g.name, None)
        return {'FINISHED'}


class ARCFLOW_OT_cloth_add_collider(_GOp, Operator):
    bl_idname = "arcflow.cloth_add_collider"
    bl_label = "Add Cloth Collider"
    bl_description = "Bind the selected meshes as collision bodies (LOTBCBP signed-distance projection)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        obs = [o for o in ctx.selected_objects if o.type == 'MESH' and not o.arc_flow_cloth.is_garment]
        if not obs:
            self.report({'WARNING'}, "Select a non-garment mesh")
            return {'CANCELLED'}
        for o in obs:
            add_collider(ctx.scene, o)
        return {'FINISHED'}


class ARCFLOW_OT_cloth_remove_collider(_GOp, Operator):
    bl_idname = "arcflow.cloth_remove_collider"
    bl_label = "Remove Cloth Collider"
    bl_description = "Unbind the active collision body"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        cl = cloth(ctx.scene)
        if not cl.colliders:
            return {'CANCELLED'}
        c = cl.colliders[cl.collider_index]
        if c.obj is not None:
            c.obj.arc_flow_cloth.is_collider = False
        cl.colliders.remove(cl.collider_index)
        cl.collider_index = max(0, cl.collider_index - 1)
        for S in RT.solvers.values():
            S._collider_sig = None
        return {'FINISHED'}


class ARCFLOW_OT_cloth_step(_GOp, Operator):
    bl_idname = "arcflow.cloth_step"
    bl_label = "Step Cloth"
    bl_description = "Advance the cloth by one live tick"
    bl_options = {'REGISTER', 'UNDO'}
    count: IntProperty(name="Steps", default=1, min=1, max=10000)

    def execute(self, ctx):
        cl = cloth(ctx.scene)
        dt = 1.0 / max(cl.live_fps, 1.0)
        for _ in range(self.count):
            step_scene(ctx.scene, dt)
        cl.status = "Stepped %d tick(s), %.1f ms last" % (self.count, cl.last_ms)
        return {'FINISHED'}


class ARCFLOW_OT_cloth_reset(_GOp, Operator):
    bl_idname = "arcflow.cloth_reset"
    bl_label = "Reset Cloth"
    bl_description = "Restore every garment to its rest shape and rebuild the solver"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        reset_scene(ctx.scene)
        cloth(ctx.scene).status = "Cloth reset to rest pose."
        return {'FINISHED'}


class ARCFLOW_OT_cloth_live_start(_GOp, Operator):
    """Real-time live mode: ticks the solver from a modal timer so the garment reacts as you pose
    the armature, move colliders, play animation or drive the rig with a controller / keyboard."""
    bl_idname = "arcflow.cloth_live_start"
    bl_label = "Start Live Cloth"
    bl_description = "Run the cloth in real time in the viewport; it reacts to posing, moving colliders and wind"
    _timer = None

    def invoke(self, ctx, event):
        cl = cloth(ctx.scene)
        if not cl.garments:
            self.report({'WARNING'}, "Assign a garment first")
            return {'CANCELLED'}
        cl.live = True
        RT.live_running = True
        RT.last_tick = time.perf_counter()
        self._timer = ctx.window_manager.event_timer_add(1.0 / max(cl.live_fps, 1.0), window=ctx.window)
        ctx.window_manager.modal_handler_add(self)
        cl.status = "Live: move the rig or colliders."
        return {'RUNNING_MODAL'}

    def execute(self, ctx):  # headless / scripted fallback: one tick
        cl = cloth(ctx.scene)
        cl.live = True
        step_scene(ctx.scene, 1.0 / max(cl.live_fps, 1.0))
        return {'FINISHED'}

    def modal(self, ctx, event):
        cl = cloth(ctx.scene)
        if not cl.live or event.type == 'ESC':
            return self._finish(ctx)
        if event.type == 'TIMER':
            now = time.perf_counter()
            dt = min(max(now - RT.last_tick, 1e-4), 1.0 / 20.0)  # clamp hitches for stability
            RT.last_tick = now
            try:
                step_scene(ctx.scene, dt)
            except Exception as ex:
                cl.status = "Live stopped: %s" % ex
                return self._finish(ctx)
            cl.last_fps = 1.0 / dt
            cl.status = "Live %.0f fps, %.1f ms solve" % (cl.last_fps, cl.last_ms)
            for area in ctx.screen.areas if ctx.screen else []:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
        return {'PASS_THROUGH'}

    def _finish(self, ctx):
        if self._timer is not None:
            ctx.window_manager.event_timer_remove(self._timer)
            self._timer = None
        cloth(ctx.scene).live = False
        RT.live_running = False
        cloth(ctx.scene).status = "Live stopped."
        return {'CANCELLED'}


class ARCFLOW_OT_cloth_live_stop(_GOp, Operator):
    bl_idname = "arcflow.cloth_live_stop"
    bl_label = "Stop Live Cloth"
    bl_description = "Stop real-time live mode (keeps the current shape)"

    def execute(self, ctx):
        cloth(ctx.scene).live = False
        RT.live_running = False
        return {'FINISHED'}


# ----------------------------------------------------------------------------- bake / export
def _abs_dir(path):
    d = bpy.path.abspath(path or "//cloth_cache")
    if not d or d.startswith("//"):
        import tempfile
        d = os.path.join(tempfile.gettempdir(), "tessera_cloth")
    os.makedirs(d, exist_ok=True)
    return d


def simulate_frames(scene, f0, f1, record=True):
    """Simulate frames f0..f1 (driving the timeline so rigs / colliders animate) and record positions."""
    cl = cloth(scene)
    reset_scene(scene)
    fps = scene.render.fps / max(scene.render.fps_base, 1e-6)
    frames = {g.name: [] for g in cl.garments if g.enabled and g.obj is not None}
    RT.live_running = True       # stop the frame handler from double-stepping
    try:
        scene.frame_set(f0)
        for g in cl.garments:
            if g.enabled and g.obj is not None:
                build_solver(scene, g)
                frames[g.name].append(RT.solvers[g.name].x.copy())
        for f in range(f0 + 1, f1 + 1):
            scene.frame_set(f)
            step_scene(scene, 1.0 / fps)
            if record:
                for g in cl.garments:
                    if g.name in frames:
                        frames[g.name].append(RT.solvers[g.name].x.copy())
    finally:
        RT.live_running = False
        RT.prev_frame = scene.frame_current
    return frames


def _frame_range(scene):
    cl = cloth(scene)
    if cl.use_scene_range:
        return scene.frame_start, scene.frame_end
    return cl.frame_start, max(cl.frame_end, cl.frame_start + 1)


def bake_shape_keys(obj, S, frames_world, f0):
    if obj.data.shape_keys is None:
        obj.shape_key_add(name="Basis", from_mix=False)
    M = np.array(obj.matrix_world, float)
    Mi = np.linalg.inv(M)
    keys = []
    for k, X in enumerate(frames_world):
        f = f0 + k
        nm = "Tessera Cloth %04d" % f
        sk = obj.data.shape_keys.key_blocks.get(nm) or obj.shape_key_add(name=nm, from_mix=False)
        local = X @ Mi[:3, :3].T + Mi[:3, 3]
        sk.data.foreach_set('co', local.astype(np.float32).ravel())
        sk.value = 0.0
        for ff, val in ((f - 1, 0.0), (f, 1.0), (f + 1, 0.0)):
            sk.value = val
            sk.keyframe_insert('value', frame=ff)
        keys.append(sk)
    return keys


def _bake_garment_keys(g, S, frames, f0):
    for o, s0, e in RT.layout.get(g.name, [(g.obj, 0, S.n)]):
        bake_shape_keys(o, S, [X[s0:e] for X in frames], f0)


def collider_meta(scene):
    out = []
    for c in cloth(scene).colliders:
        if c.obj is None:
            continue
        o = c.obj
        dims = list(o.dimensions)
        cen = list(o.matrix_world.translation)
        shape = 'SPHERE' if abs(dims[0] - dims[1]) < 1e-3 and abs(dims[1] - dims[2]) < 1e-3 else 'CAPSULE' if max(dims) > 1.8 * min(dims) else 'CONVEX'
        out.append(dict(name=o.name, enabled=bool(c.enabled), shape_hint=shape,
                        center=cen, dimensions=dims,
                        radius=0.5 * min(dims), half_height=0.5 * max(dims),
                        matrix_world=[list(r) for r in o.matrix_world],
                        n_triangles=len(o.data.polygons) if o.type == 'MESH' else 0))
    return out


def export_package(scene, g, directory):
    S = get_solver(scene, g)
    pkg = S.package_dict(name=g.name, colliders=collider_meta(scene),
                         extra=dict(blender=bpy.app.version_string, source_object=g.obj.name,
                                    pin_group=g.pin_group))
    base = os.path.join(directory, "%s_cloth" % bpy.path.clean_name(g.name))
    with open(base + ".json", 'w') as fh:
        json.dump(pkg, fh, indent=1)
    np.savez_compressed(base + ".npz", rest=S.rest.astype(np.float32), tris=S.tris.astype(np.int32),
                        pins=np.nonzero(S.pin_hard)[0].astype(np.int32),
                        inv_mass=S.inv_mass.astype(np.float32),
                        stretch_edges=S.stretch_edges.astype(np.int32),
                        stretch_rest=S.stretch_rest.astype(np.float32),
                        shear_edges=S.shear_edges.astype(np.int32),
                        hinges=S.hinges.astype(np.int32))
    return base + ".json", base + ".npz"


class ARCFLOW_OT_cloth_bake(_GOp, Operator):
    bl_idname = "arcflow.cloth_bake"
    bl_label = "Bake Cloth"
    bl_description = "Simulate the frame range and store it as shape keys, a .npz point cache, Alembic or a game-engine package"
    bl_options = {'REGISTER', 'UNDO'}
    mode: EnumProperty(name="Bake to", items=[('SCENE', "Scene setting", "")] + [(i[0], i[1], i[2]) for i in
                       (('SHAPEKEYS', "Shape keys", ""), ('NPZ', "Point cache", ""), ('ALEMBIC', "Alembic", ""),
                        ('PACKAGE', "Game-engine package", ""))], default='SCENE')

    def execute(self, ctx):
        sc = ctx.scene
        cl = cloth(sc)
        mode = cl.bake_mode if self.mode == 'SCENE' else self.mode
        if not cl.garments:
            self.report({'WARNING'}, "No garments")
            return {'CANCELLED'}
        f0, f1 = _frame_range(sc)
        d = _abs_dir(cl.bake_path)
        t0 = time.perf_counter()
        frames = simulate_frames(sc, f0, f1)
        written = []
        for g in cl.garments:
            if g.name not in frames:
                continue
            S = RT.solvers[g.name]
            stack = np.stack(frames[g.name]).astype(np.float32)
            if mode == 'SHAPEKEYS':
                write_garment_rest(g, S)
                _bake_garment_keys(g, S, frames[g.name], f0)
                written.append("%d shape keys on %s" % (len(stack), ", ".join(o.name for o in _garment_objs(g))))
            elif mode == 'NPZ':
                p = os.path.join(d, "%s_%s.npz" % (bpy.path.clean_name(cl.bake_name), bpy.path.clean_name(g.name)))
                np.savez_compressed(p, frames=np.arange(f0, f0 + len(stack), dtype=np.int32), positions=stack,
                                    rest=S.rest.astype(np.float32), tris=S.tris.astype(np.int32),
                                    pins=np.nonzero(S.pin_hard)[0].astype(np.int32),
                                    meta=np.frombuffer(json.dumps(dict(garment=g.name, preset=g.preset, fps=sc.render.fps)).encode(), np.uint8))
                written.append(p)
            elif mode == 'PACKAGE':
                written.extend(export_package(sc, g, d))
                p = os.path.join(d, "%s_%s_anim.npz" % (bpy.path.clean_name(cl.bake_name), bpy.path.clean_name(g.name)))
                np.savez_compressed(p, positions=stack, frames=np.arange(f0, f0 + len(stack), dtype=np.int32))
                written.append(p)
        if mode == 'ALEMBIC':
            # shape keys drive the Alembic export, then are removed again
            for g in cl.garments:
                if g.name in frames:
                    write_garment_rest(g, RT.solvers[g.name])
                    _bake_garment_keys(g, RT.solvers[g.name], frames[g.name], f0)
            p = os.path.join(d, "%s.abc" % bpy.path.clean_name(cl.bake_name))
            sel = [o for o in sc.objects if o.select_get()]
            for o in sc.objects:
                o.select_set(False)
            for g in cl.garments:
                for o in _garment_objs(g):
                    o.select_set(True)
            try:
                bpy.ops.wm.alembic_export(filepath=p, start=f0, end=f1, selected=True, evaluation_mode='VIEWPORT')
                written.append(p)
            finally:
                for o in sc.objects:
                    o.select_set(o in sel)
        cl.status = "Baked %d frames (%s) in %.1fs" % (f1 - f0 + 1, mode.lower(), time.perf_counter() - t0)
        self.report({'INFO'}, cl.status + ": " + ", ".join(os.path.basename(w) for w in written)[:200])
        self._written = written
        return {'FINISHED'}


class ARCFLOW_OT_cloth_export_package(_GOp, Operator):
    bl_idname = "arcflow.cloth_export_package"
    bl_label = "Export Game-Engine Cloth Package"
    bl_description = "Write rest mesh, pin weights, constraint stiffness / damping / aero and collider metadata (JSON + NPZ) for Unreal Chaos Cloth, Unity or web engines"

    def execute(self, ctx):
        cl = cloth(ctx.scene)
        d = _abs_dir(cl.bake_path)
        out = []
        for g in cl.garments:
            if g.enabled and g.obj is not None:
                out.extend(export_package(ctx.scene, g, d))
        if not out:
            self.report({'WARNING'}, "No garments to export")
            return {'CANCELLED'}
        cl.status = "Exported %d package files to %s" % (len(out), d)
        self.report({'INFO'}, cl.status)
        return {'FINISHED'}


# ----------------------------------------------------------------------------- example scene
def _new_obj(scene, name, verts, faces, coll=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, v)) for v in verts], [], [tuple(map(int, f)) for f in faces])
    me.update()
    o = bpy.data.objects.new(name, me)
    (coll or scene.collection).objects.link(o)
    return o


def _uv_sphere(r, n=32, m=16):
    v = [(0, 0, r)]
    for j in range(1, m):
        th = math.pi * j / m
        for i in range(n):
            ph = 2 * math.pi * i / n
            v.append((r * math.sin(th) * math.cos(ph), r * math.sin(th) * math.sin(ph), r * math.cos(th)))
    v.append((0, 0, -r))
    f = []
    for i in range(n):
        f.append((0, 1 + i, 1 + (i + 1) % n))
    for j in range(m - 2):
        for i in range(n):
            a = 1 + j * n + i; b = 1 + j * n + (i + 1) % n
            f.append((a, a + n, b + n, b))
    last = len(v) - 1
    for i in range(n):
        f.append((last, 1 + (m - 2) * n + (i + 1) % n, 1 + (m - 2) * n + i))
    return v, f


def _capsule_torso(rx=0.17, ry=0.11, h=0.56, n=32, rings=14):
    """Smooth elliptic torso: wider at the shoulders, tapering at the waist."""
    v, f = [], []
    for j in range(rings + 1):
        t = j / rings
        z = h * t
        # shoulder bulge near the top, waist taper in the middle
        s = 0.86 + 0.18 * math.sin(math.pi * min(1.0, t * 1.1)) + 0.10 * max(0.0, t - 0.75) / 0.25
        top_round = math.sqrt(max(0.0, 1.0 - max(0.0, (t - 0.92) / 0.08) ** 2)) if t > 0.92 else 1.0
        for i in range(n):
            a = 2 * math.pi * i / n
            v.append((rx * s * top_round * math.cos(a), ry * s * top_round * math.sin(a), z))
    for j in range(rings):
        for i in range(n):
            a = j * n + i; b = j * n + (i + 1) % n
            f.append((a, b, b + n, a + n))
    # caps
    bc = len(v); v.append((0, 0, 0))
    tc = len(v); v.append((0, 0, h))
    for i in range(n):
        f.append((bc, (i + 1) % n, i))
        f.append((tc, rings * n + i, rings * n + (i + 1) % n))
    return v, f



def _make_spine_rig(coll, name="Cloth Rig"):
    arm_data = bpy.data.armatures.new(name)
    arm = bpy.data.objects.new(name, arm_data)
    coll.objects.link(arm)
    bpy.context.view_layer.update()
    prev = bpy.context.view_layer.objects.active
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')
    b = arm_data.edit_bones.new("Spine")
    b.head = (0.0, 0.0, 0.95)
    b.tail = (0.0, 0.0, 1.50)
    bpy.ops.object.mode_set(mode='OBJECT')
    if prev is not None:
        bpy.context.view_layer.objects.active = prev
    return arm


def _area_light(scene, name, loc, target, energy, size, color=(1, 1, 1), shape='AREA'):
    data = bpy.data.lights.new(name, 'AREA')
    data.energy = energy
    data.size = size
    data.color = color
    try:
        data.spread = math.radians(150.0)
    except Exception:
        pass
    o = bpy.data.objects.new(name, data)
    scene.collection.objects.link(o)
    o.location = loc
    from mathutils import Vector
    o.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
    return o



def setup_studio(scene, camera_loc=(2.55, 3.25, 1.48), look_at=(-0.10, 0.16, 1.00), lens=50.0):
    """Publication-grade studio: seamless cyclorama, three-point area lights, 3/4 back camera.

    The camera sits behind the mannequin so the cape is the hero of the frame.
    The cyclorama wall is in -Y, opposite the camera, so it reads as the backdrop.
    """
    from mathutils import Vector

    def look(o, target):
        o.rotation_euler = (Vector(target) - o.location).to_track_quat('-Z', 'Y').to_euler()

    # drop the default cube / light so they cannot sneak into the beauty still
    for nm in ('Cube', 'Light', 'Camera'):
        if nm in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[nm], do_unlink=True)

    # seamless cyclorama: floor under the camera (+Y) curving up into a wall in -Y
    # (behind the figure from the beauty camera). Rebuild each time so layout tweaks stick.
    if "Cloth Studio Cyc" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Cloth Studio Cyc"], do_unlink=True)
    import bmesh
    me = bpy.data.meshes.new("Cloth Studio Cyc")
    bm = bmesh.new()
    nx, ny = 28, 22
    width, depth, rad, wall = 12.0, 10.0, 1.6, 4.2
    y_cam, y_curve = 5.2, -1.6
    for j in range(ny + 1):
        y = y_cam - (j / ny) * depth
        for i in range(nx + 1):
            x = (i / nx - 0.5) * width
            if y >= y_curve:
                yy, z = y, 0.0
            else:
                s = y_curve - y
                if s <= rad:
                    ang = (s / rad) * (math.pi / 2)
                    yy = y_curve - rad * math.sin(ang)
                    z = rad * (1.0 - math.cos(ang))
                else:
                    yy = y_curve - rad
                    z = rad + (s - rad) * (wall / max(depth - (y_cam - y_curve) - rad, 0.01))
            bm.verts.new((x, yy, z))
    bm.verts.ensure_lookup_table()
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            b = a + 1
            c = a + (nx + 1) + 1
            d = a + (nx + 1)
            # winding keeps the floor normal +Z as j walks toward -Y
            bm.faces.new((bm.verts[a], bm.verts[d], bm.verts[c], bm.verts[b]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    cyc = bpy.data.objects.new("Cloth Studio Cyc", me)
    scene.collection.objects.link(cyc)
    for p in me.polygons:
        p.use_smooth = True
    CG.assign(cyc, CG.floor_material())
    # hide any leftover broken backdrop / floor from earlier revisions
    for nm in ("Cloth Studio Backdrop", "Cloth Studio Floor"):
        if nm in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[nm], do_unlink=True)

    for nm in ("Cloth Key Light", "Cloth Fill Light", "Cloth Rim Light", "Cloth Hair Light",
               "Cloth Cyc Light", "Cloth Bounce Light"):
        if nm in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[nm], do_unlink=True)
    # lights are placed for a *back* 3/4 view (camera in +Y, backdrop in -Y)
    _area_light(scene, "Cloth Key Light", (3.5, 0.7, 2.15), (0.0, 0.25, 1.10), energy=260.0, size=1.5, color=(1.00, 0.96, 0.90))
    _area_light(scene, "Cloth Fill Light", (-1.6, 3.2, 1.7), (0.0, 0.15, 1.05), energy=70.0, size=3.2, color=(0.75, 0.84, 1.00))
    _area_light(scene, "Cloth Rim Light", (-0.3, -2.6, 2.1), (0.0, 0.10, 1.20), energy=140.0, size=1.4, color=(1.00, 0.93, 0.86))
    _area_light(scene, "Cloth Hair Light", (0.35, 1.4, 2.7), (0.0, 0.0, 1.65), energy=50.0, size=0.45, color=(1.00, 0.97, 0.93))
    _area_light(scene, "Cloth Cyc Light", (0.0, 0.4, 3.6), (0.0, -2.8, 1.8), energy=200.0, size=5.0, color=(0.92, 0.93, 1.00))
    _area_light(scene, "Cloth Bounce Light", (0.0, 1.2, 0.16), (0.0, 0.2, 1.0), energy=40.0, size=3.5, color=(1.00, 0.95, 0.90))

    world = scene.world or bpy.data.worlds.new("Cloth Studio World")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    bg = nt.nodes.new('ShaderNodeBackground')
    bg.inputs['Color'].default_value = (0.16, 0.15, 0.155, 1)
    bg.inputs['Strength'].default_value = 0.55
    wo = nt.nodes.new('ShaderNodeOutputWorld')
    nt.links.new(bg.outputs['Background'], wo.inputs['Surface'])

    cam = bpy.data.objects.get("Cloth Beauty Cam")
    if cam is None:
        cdata = bpy.data.cameras.new("Cloth Beauty Cam")
        cam = bpy.data.objects.new("Cloth Beauty Cam", cdata)
        scene.collection.objects.link(cam)
    cam.location = camera_loc
    look(cam, look_at)
    cam.data.lens = lens
    cam.data.clip_start = 0.05
    cam.data.clip_end = 40.0
    cam.data.dof.use_dof = True
    cam.data.dof.aperture_fstop = 8.0
    cape = bpy.data.objects.get("Cloth Cape")
    cam.data.dof.focus_object = cape
    if cape is None:
        cam.data.dof.focus_distance = 3.4
    scene.camera = cam
    try:
        scene.view_settings.view_transform = 'AgX'
        scene.view_settings.look = 'AgX - Medium Contrast'
    except TypeError:
        try:
            scene.view_settings.view_transform = 'Filmic'
            scene.view_settings.look = 'Medium Contrast'
        except Exception:
            pass
    scene.view_settings.exposure = 0.05
    scene.view_settings.gamma = 1.0
    return cam


def drape_for_beauty(scene, gust_frames=50, settle_frames=40):
    """A side breeze, then a calmer settle, so the cape shows folds instead of hanging as a flat sheet."""
    cl = cloth(scene)
    cl.use_scene_wind = False
    # a moderate back-and-side breeze: enough to billow, not enough to flip the cape
    cl.wind = (0.9, 1.6, 0.08)
    for _ in range(int(gust_frames)):
        step_scene(scene, 1 / 60.0)
    cl.wind = (0.25, 0.55, 0.0)
    for _ in range(int(settle_frames)):
        step_scene(scene, 1 / 60.0)
    return cl.last_ms


def render_beauty(scene, filepath, samples=64, resolution=(1920, 1080), denoise=True):
    """Cycles beauty still of the current cloth scene (studio lighting, OIDN, ~64 samples)."""
    setup_studio(scene)
    sc = scene
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'CPU'
    sc.cycles.samples = int(samples)
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.01
    sc.cycles.use_denoising = bool(denoise)
    try:
        sc.cycles.denoiser = 'OPENIMAGEDENOISE'
        sc.cycles.denoising_prefilter = 'ACCURATE'
    except Exception:
        pass
    sc.cycles.max_bounces = 8
    sc.cycles.diffuse_bounces = 4
    sc.cycles.glossy_bounces = 4
    sc.cycles.transmission_bounces = 4
    sc.cycles.transparent_max_bounces = 4
    sc.cycles.caustics_reflective = False
    sc.cycles.caustics_refractive = False
    sc.cycles.use_auto_tile = True
    sc.cycles.tile_size = 256
    sc.render.resolution_x, sc.render.resolution_y = int(resolution[0]), int(resolution[1])
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = 'PNG'
    sc.render.image_settings.color_mode = 'RGB'
    sc.render.image_settings.color_depth = '8'
    sc.render.image_settings.compression = 15
    sc.render.filepath = filepath
    sc.render.film_transparent = False
    for owner in (sc.render, getattr(sc, 'cycles', None)):   # moved / removed across 4.2 .. 5.3
        if owner is not None and hasattr(owner, 'use_persistent_data'):
            owner.use_persistent_data = False
    # keep Cycles cache off the tight /workspace disk
    tmp = os.environ.get('TMPDIR') or '/tmp'
    try:
        sc.render.use_overwrite = True
    except Exception:
        pass
    os.makedirs(os.path.dirname(os.path.abspath(filepath)) or '.', exist_ok=True)
    bpy.ops.render.render(write_still=True)
    return filepath


def load_cape_example(scene):
    """Mannequin (head, neck, torso) + an armature + a cape pinned at the shoulders, with wind."""
    coll = bpy.data.collections.get("Tessera Cloth Example") or bpy.data.collections.new("Tessera Cloth Example")
    if coll.name not in scene.collection.children:
        scene.collection.children.link(coll)
    for o in list(coll.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    cl = cloth(scene)
    cl.garments.clear()
    cl.colliders.clear()
    RT.reset()

    # body: torso, neck, head, plus shoulders and upper arms so the silhouette reads as a figure
    tv, tf = _capsule_torso()
    torso = _new_obj(scene, "Cloth Mannequin Torso", tv, tf, coll)
    torso.location = (0.0, 0.0, 0.95)
    hv, hf = _uv_sphere(0.105, 32, 16)
    head = _new_obj(scene, "Cloth Mannequin Head", hv, hf, coll)
    head.location = (0.0, 0.0, 1.67)
    nv, nf = _capsule_torso(rx=0.05, ry=0.05, h=0.14, n=20, rings=4)
    neck = _new_obj(scene, "Cloth Mannequin Neck", nv, nf, coll)
    neck.location = (0.0, 0.0, 1.49)
    body = [torso, head, neck]
    for side, name in ((-1.0, "L"), (1.0, "R")):
        av, af = _capsule_torso(rx=0.058, ry=0.058, h=0.38, n=20, rings=8)
        am = _new_obj(scene, "Cloth Mannequin Arm %s" % name, av, af, coll)
        # top of the capsule meets the torso shoulder; hangs down the side
        am.location = (side * 0.22, 0.02, 1.10)
        body.append(am)
    for o in body:
        for p in o.data.polygons:
            p.use_smooth = True

    # cape: a flared wrap around the back, pinned at the neckline, with tailored
    # vertical folds in the bind pose (open sides + surplus length otherwise just
    # spread into a smooth larger shell).  wrap < pi leaves a front opening so a
    # 3/4 camera can see an arm and the cape's edge.
    nx, ny = 24, 30
    height, z_collar = 1.20, 1.52
    r_neck, r_sh, r_hem, wrap = 0.18, 0.36, 0.42, 1.34
    _, tris, quads = CS.grid_mesh(nx, ny, 1.0, height, origin=(-0.5, 0.0, 0.0), axis='xz')
    iu = (np.arange(nx * ny) % nx) / (nx - 1)
    iv = (np.arange(nx * ny) // nx) / (ny - 1)
    ufrac = (iu - 0.5) * 2.0
    r = np.where(
        iv >= 0.82,
        r_sh + (r_neck - r_sh) * (iv - 0.82) / 0.18,
        r_hem + (r_sh - r_hem) * (iv / 0.82),
    )
    ang = ufrac * wrap
    verts = np.zeros((nx * ny, 3), float)
    verts[:, 0] = r * np.sin(ang)
    verts[:, 1] = r * np.cos(ang)
    verts[:, 2] = (z_collar - height) + iv * height
    # irregular vertical folds — visible on the back, fading at the collar and edges
    s = (ufrac + 1.0) * 0.5
    pleat = (0.062 * np.sin(2 * np.pi * 3.0 * s)
             + 0.036 * np.sin(2 * np.pi * 5.0 * s + 0.7)
             + 0.022 * np.sin(2 * np.pi * 2.0 * s + 1.3)
             + 0.018 * np.sin(2 * np.pi * 7.0 * s + 0.3))
    edge_fade = np.sin(np.pi * np.clip(s, 0.0, 1.0)) ** 0.4
    # keep folds readable through the mid-back, not only at the hem
    collar_fade = np.clip(1.0 - iv * 0.75, 0.15, 1.0) ** 0.65
    disp = pleat * edge_fade * collar_fade
    verts[:, 0] += disp * np.sin(ang)
    verts[:, 1] += disp * np.cos(ang)
    verts[:, 2] -= 0.014 * np.maximum(-pleat, 0.0) * collar_fade
    # a soft roll on the two opening edges so they read as fabric, not a hard trim
    for j in range(ny):
        curl = 0.028 * (1.0 - 0.35 * j / (ny - 1))
        for i, sign in ((0, -1.0), (nx - 1, 1.0)):
            k = j * nx + i
            verts[k, 1] += curl
            verts[k, 0] += sign * curl * 0.25
    cape = _new_obj(scene, "Cloth Cape", verts, quads.tolist(), coll)
    for p in cape.data.polygons:
        p.use_smooth = True
    # UVs for the weave
    uv = cape.data.uv_layers.new(name="UVMap")
    for poly in cape.data.polygons:
        for li in poly.loop_indices:
            vi = cape.data.loops[li].vertex_index
            i, j = vi % nx, vi // nx
            uv.data[li].uv = (i / (nx - 1), j / (ny - 1))
    # solidify + subdivision for a publication-grade surface (solver still writes the cage)
    sol = cape.modifiers.new("Cloth Thickness", 'SOLIDIFY')
    sol.thickness = 0.0035
    sol.offset = 0.0
    sub = cape.modifiers.new("Cloth Subdiv", 'SUBSURF')
    sub.levels = 1
    sub.render_levels = 2
    sub.subdivision_type = 'CATMULL_CLARK'

    # armature driving the shoulders (a single 'Spine' bone the user can pose or drive)
    arm = _make_spine_rig(coll, "Cloth Rig")
    arm.show_in_front = True
    for o in body:
        mw = o.matrix_world.copy()
        o.parent = arm
        o.parent_type = 'OBJECT'
        o.matrix_parent_inverse = arm.matrix_world.inverted()
        o.matrix_world = mw

    # pin the whole neckline (top two rows of the trapezoid) around the collar
    g_pins = np.concatenate([
        np.arange((ny - 1) * nx, ny * nx, dtype=np.int64),
        np.arange((ny - 2) * nx, (ny - 1) * nx, dtype=np.int64),
    ])
    vg = cape.vertex_groups.new(name="Spine")
    vg.add([int(i) for i in g_pins], 1.0, 'REPLACE')
    write_pin_group(cape, g_pins, "ClothPins")

    # materials
    CG.assign(cape, CG.fabric_material(tint=(0.16, 0.015, 0.025)))
    skin = CG.skin_material()
    for o in body:
        CG.assign(o, skin)

    # wire up the solver
    g = assign_garment(scene, cape, preset='CAPE', quality='BALANCED', pin_mode='TOP', armature=arm)
    g.pin_group = "ClothPins"
    for o in body:
        add_collider(scene, o)
    cl.use_scene_wind = False
    cl.wind = (0.0, 1.8, 0.15)
    cl.gravity = (0.0, 0.0, -9.81)
    build_solver(scene, g)
    scene.frame_start = 1
    scene.frame_end = max(scene.frame_end, 96)
    # gentle walk-cycle sway on the rig so playback shows inertia
    arm.animation_data_clear()
    for f, (x, rz) in ((1, (0.0, 0.0)), (25, (0.10, 0.12)), (49, (0.0, 0.0)), (73, (-0.10, -0.12)), (97, (0.0, 0.0))):
        arm.location = (x, 0.0, 0.0)
        arm.rotation_euler = (0.0, 0.0, rz)
        arm.keyframe_insert('location', frame=f)
        arm.keyframe_insert('rotation_euler', frame=f)
    arm.location = (0.0, 0.0, 0.0)
    arm.rotation_euler = (0.0, 0.0, 0.0)
    cl.status = "Cape example loaded: press Start Live or play the timeline."
    return "Cape drape example: mannequin collider, collar-pinned cape, light breeze, rig sway."


EXAMPLES = {'CAPE': load_cape_example}


class ARCFLOW_OT_cloth_example(Operator):
    bl_idname = "arcflow.cloth_example"
    bl_label = "Load Cloth Example"
    bl_description = "Build a mannequin collider, a rig and a shoulder-pinned cape with wind"
    bl_options = {'REGISTER', 'UNDO'}
    which: EnumProperty(name="Example", items=[('CAPE', "Cape drape", "")], default='CAPE')

    def execute(self, ctx):
        msg = EXAMPLES[self.which](ctx.scene)
        self.report({'INFO'}, msg)
        return {'FINISHED'}


CLASSES = (ARCFLOW_OT_cloth_assign, ARCFLOW_OT_cloth_remove, ARCFLOW_OT_cloth_auto_pin, ARCFLOW_OT_cloth_apply_preset,
           ARCFLOW_OT_cloth_add_collider, ARCFLOW_OT_cloth_remove_collider, ARCFLOW_OT_cloth_step, ARCFLOW_OT_cloth_reset,
           ARCFLOW_OT_cloth_live_start, ARCFLOW_OT_cloth_live_stop, ARCFLOW_OT_cloth_bake, ARCFLOW_OT_cloth_export_package,
           ARCFLOW_OT_cloth_example, ARCFLOW_OT_cloth_assign_group, ARCFLOW_OT_cloth_group_refresh,
           ARCFLOW_OT_cloth_member_preset)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    if _on_frame not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(_on_frame)
    try:
        from . import surfaces_panel
        surfaces_panel.register_example("Cloth", "Cape drape", "arcflow.cloth_example", which='CAPE')
    except Exception:
        pass


def unregister():
    if _on_frame in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(_on_frame)
    RT.reset()
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
