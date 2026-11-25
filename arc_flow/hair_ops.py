"""Operators and runtime for Tessera Projection Hair.

Live mode ticks a HairSolver against the active CURVES object as the
armature, scalp or collision body moves (timer + optional depsgraph hook),
so Blender can preview the groom the way a game engine would.  Bake writes
a .npz point cache and/or an Alembic file, plus a game-engine hair physics
package (rest curves, guide weights, per-strand stiffness, collision metadata).
"""
import os
import time

import bpy
import numpy as np
from bpy.props import EnumProperty, IntProperty, StringProperty
from bpy.types import Operator
from bpy_extras.io_utils import ExportHelper
from mathutils import Matrix

from .core import hair_solver as HS
from .core.hair_solver import HairParams, HairSolver


# ----------------------------------------------------------------------------- runtime


class _RT:
    solver = None
    scene = None
    live = False
    updating = False
    last_clock = 0.0
    cache = {}          # frame -> positions (world)
    child_cache = {}
    collider_sig = None
    field = None
    bind_ok = False
    status = "Idle"
    last_ms = 0.0


RT = _RT()


def _hp(scene):
    return scene.arc_flow_hair


def _abspath(scene, path):
    path = bpy.path.abspath(path or "", start=None)
    if path.startswith("//"):
        path = bpy.path.abspath(path)
    return os.path.abspath(path) if path else path


def _mesh_arrays(obj, deps=None, world=True):
    """(verts, tris) of a mesh object; verts in world space when world=True."""
    if obj is None or obj.type != 'MESH':
        return None, None
    deps = deps or bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(deps)
    me = ev.to_mesh()
    try:
        me.calc_loop_triangles()
        nv, nt = len(me.vertices), len(me.loop_triangles)
        v = np.empty(nv * 3, np.float64)
        me.vertices.foreach_get('co', v)
        t = np.empty(nt * 3, np.int64)
        me.loop_triangles.foreach_get('vertices', t)
        V, T = v.reshape(-1, 3), t.reshape(-1, 3)
        if world:
            M = np.array(obj.matrix_world, dtype=np.float64)
            V = V @ M[:3, :3].T + M[:3, 3]
        return V, T
    finally:
        ev.to_mesh_clear()


def read_curves(obj, world=True):
    """Packed (positions, offsets, radii) from a CURVES object."""
    cu = obj.data
    n = len(cu.points)
    pos = np.empty(n * 3, np.float32)
    cu.points.foreach_get('position', pos)
    pos = pos.reshape(-1, 3).astype(np.float64)
    if world:
        M = np.array(obj.matrix_world, dtype=np.float64)
        pos = pos @ M[:3, :3].T + M[:3, 3]
    off = np.empty(len(cu.curve_offset_data), np.int32)
    cu.curve_offset_data.foreach_get('value', off)
    rad = np.empty(n, np.float32)
    try:
        cu.points.foreach_get('radius', rad)
    except Exception:
        rad[:] = 0.004
    return pos, off.astype(np.int64), rad.astype(np.float64)


def write_curves(obj, world_pos, radii=None):
    """Write packed world-space positions back into a CURVES object (as local points)."""
    cu = obj.data
    P = np.asarray(world_pos, dtype=np.float64).reshape(-1, 3)
    M = np.array(obj.matrix_world, dtype=np.float64)
    try:
        Minv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        Minv = np.eye(4)
        Minv[:3, :3] = M[:3, :3].T
        Minv[:3, 3] = -M[:3, :3].T @ M[:3, 3]
    loc = P @ Minv[:3, :3].T + Minv[:3, 3]
    cu.points.foreach_set('position', np.ascontiguousarray(loc, dtype=np.float32).ravel())
    if radii is not None and len(radii) == len(cu.points):
        try:
            cu.points.foreach_set('radius', np.ascontiguousarray(radii, dtype=np.float32))
        except Exception:
            pass
    cu.update_tag()
    obj.update_tag()


def make_curves_object(name, positions, offsets, radii=None, collection=None, radius=0.004):
    """Create a Blender 4.2+ CURVES (Empty Hair) object from packed arrays (local space)."""
    P = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    off = np.asarray(offsets, dtype=np.int64).ravel()
    counts = np.diff(off).astype(int).tolist()
    cu = bpy.data.hair_curves.new(name)
    if counts:
        cu.add_curves(counts)
    if len(cu.points) != len(P):
        raise RuntimeError("hair curves point count %d != %d" % (len(cu.points), len(P)))
    cu.points.foreach_set('position', np.ascontiguousarray(P, dtype=np.float32).ravel())
    if radii is None:
        radii = np.full(len(P), float(radius), dtype=np.float32)
    try:
        cu.points.foreach_set('radius', np.ascontiguousarray(radii, dtype=np.float32))
    except Exception:
        pass
    ob = bpy.data.objects.new(name, cu)
    col = collection or (bpy.context.scene.collection)
    col.objects.link(ob)
    return ob


def _bone_matrix(arm, bone_name):
    """World matrix of a pose bone, or the armature object matrix if the bone is missing."""
    Mw = arm.matrix_world
    if bone_name and arm.pose and bone_name in arm.pose.bones:
        return np.array(Mw @ arm.pose.bones[bone_name].matrix, dtype=np.float64)
    return np.array(Mw, dtype=np.float64)


def _gravity(scene, hp):
    if not hp.use_scene_gravity:
        return tuple(hp.gravity)
    env = getattr(getattr(scene, 'arc_flow', None), 'env', None)
    if env is not None:
        if getattr(env, 'use_scene_gravity', False) and scene.gravity is not None:
            return tuple(scene.gravity)
        return tuple(env.gravity)
    return tuple(scene.gravity) if scene.gravity is not None else tuple(hp.gravity)


def _wind(scene, hp):
    if hp.use_env_wind:
        env = getattr(getattr(scene, 'arc_flow', None), 'env', None)
        if env is not None:
            return tuple(env.wind)
    return tuple(hp.wind)


def _fluid_fn(scene):
    """Best-effort sampler of a running Tessera fluid / wind-tunnel velocity field."""
    try:
        from . import fluidmode_ops as FO
        S = getattr(getattr(FO, 'RT', None), 'solver', None)
        if S is None or not hasattr(S, 'x'):
            return None
        # nearest-neighbour lookup into the live particle velocity when available
        def fn(P, _S=S):
            X = np.asarray(getattr(_S, 'x', None), dtype=np.float64)
            V = np.asarray(getattr(_S, 'v', None), dtype=np.float64) if hasattr(_S, 'v') else None
            if X is None or V is None or len(X) == 0 or len(X) != len(V):
                return np.zeros_like(P)
            # coarse: pick the closest of a random subset to stay real-time
            n = len(X)
            step = max(1, n // 4000)
            Xs, Vs = X[::step], V[::step]
            # chunked naive nearest (small P, subsampled X)
            out = np.zeros_like(P)
            for i0 in range(0, len(P), 256):
                q = P[i0:i0 + 256]
                d2 = np.sum((q[:, None, :] - Xs[None, :, :]) ** 2, axis=2)
                j = np.argmin(d2, axis=1)
                close = d2[np.arange(len(q)), j] < 0.25
                out[i0:i0 + len(q)][close] = Vs[j[close]]
            return out
        return fn
    except Exception:
        return None


def params_from_scene(scene):
    hp = _hp(scene)
    fps = max(int(getattr(scene.render, 'fps', 24) or 24), 1)
    dt = (1.0 / fps) if hp.dt_mode == 'SCENE' else float(hp.dt)
    return HairParams(
        dt=dt,
        substeps=int(hp.substeps),
        gravity=_gravity(scene, hp),
        damping=float(hp.damping),
        stretch_iterations=int(hp.stretch_iterations),
        stretch_compliance=float(hp.stretch_compliance),
        bend_iterations=int(hp.bend_iterations),
        bend_compliance=float(hp.bend_compliance),
        rest_shape=float(hp.rest_shape),
        rest_shape_root=float(hp.rest_shape_root),
        rest_shape_tip=float(hp.rest_shape_tip),
        self_collision=bool(getattr(hp, 'self_collision', True)),
        collision_radius=float(hp.collision_radius),
        collision_iterations=int(hp.collision_iterations),
        collision_friction=float(hp.collision_friction),
        sdf_resolution=int(hp.sdf_resolution),
        sdf_level=int(hp.sdf_level),
        sdf_refine=bool(hp.sdf_refine),
        wind_drag=float(hp.wind_drag),
        wind_tangent=float(hp.wind_tangent),
        fluid_coupling=float(hp.fluid_coupling),
        guide_ratio=float(hp.guide_ratio),
        child_k=int(hp.child_k),
        child_smooth=float(hp.child_smooth),
        pinned_prefix=int(hp.pinned_prefix),
        ground_z=float(hp.ground_z) if hp.use_ground else None,
        max_segment_budget=int(hp.max_segment_budget),
        max_points=int(hp.max_points),
    )


def _collider_sig(obj, hp, deps):
    if obj is None:
        return None
    V, T = _mesh_arrays(obj, deps, world=False)
    if V is None:
        return None
    M = tuple(round(x, 6) for x in np.array(obj.matrix_world, dtype=np.float64).ravel())
    return (obj.name, len(V), len(T), float(V.sum()), int(hp.sdf_resolution), int(hp.sdf_level), M)


def _ensure_field(scene, deps):
    hp = _hp(scene)
    col = hp.collider
    sig = _collider_sig(col, hp, deps)
    if sig is None:
        RT.field = None
        RT.collider_sig = None
        return None
    # rebuild when the mesh changes; keep the field (local space) when only the world matrix moves
    mesh_sig = sig[:-1]
    prev = RT.collider_sig
    if RT.field is None or prev is None or prev[:-1] != mesh_sig:
        V, T = _mesh_arrays(col, deps, world=False)
        RT.field = HS.make_collider(V, T, resolution=hp.sdf_resolution, level=hp.sdf_level, mode='SOLID')
    RT.collider_sig = sig
    return RT.field


def _root_matrices(scene, solver, deps):
    """(S,4,4) current root frames in world space."""
    hp = _hp(scene)
    hair = hp.hair
    n = solver.n_strands
    # start from the hair object's world matrix (or identity translations at rest roots)
    base = np.array(hair.matrix_world, dtype=np.float64) if hair is not None else np.eye(4)
    M = np.repeat(base[None], max(n, 1), axis=0)
    arm = hp.armature
    if arm is not None:
        Bm = _bone_matrix(arm, hp.bone or (arm.arc_flow_hair.bone if hasattr(arm, 'arc_flow_hair') else ""))
        # apply the bone's current transform relative to its bind (the solver bind_M already captured this)
        M[:] = Bm
    surf = hp.surface or hp.collider
    if surf is not None and solver._surface_bind is not None:
        V, T = _mesh_arrays(surf, deps, world=True)
        if V is not None and len(T):
            solver.update_from_mesh(V, T)
            return solver.root_M
    return M



def _add_scene_hair_colliders(scene, solver, deps, skip=None):
    """Register every other Tessera collision body as an extra hair collider."""
    from . import asset_colliders as AC
    hp = _hp(scene)
    rad = float(hp.collision_radius)
    existing = 1 if solver.colliders else 0
    extras = [o for o in AC.iter_collision_meshes(scene, exclude=skip) if o is not hp.collider]
    solver.colliders[:] = solver.colliders[:existing]
    cache = getattr(solver, '_extra_cols', {})
    new_cache = {}
    for o in extras:
        V, T = _mesh_arrays(o, deps, world=False)
        if V is None or T is None or len(T) == 0:
            continue
        mesh_sig = (o.name, len(V), len(T), float(V.sum()), int(hp.sdf_level))
        prev = cache.get(o.name)
        if prev is not None and prev['mesh_sig'] == mesh_sig:
            field = prev['field']
        else:
            field = HS.make_collider(V, T, resolution=max(16, hp.sdf_resolution // 2), level=hp.sdf_level, mode='SOLID')
        Mw = np.array(o.matrix_world, dtype=np.float64)
        solver.add_collider(field, matrix=Mw, radius=rad)
        new_cache[o.name] = dict(field=field, mesh_sig=mesh_sig)
    solver._extra_cols = new_cache


def build_solver(scene, deps=None):
    """(Re)build a HairSolver from the scene pointers. Returns the solver or None."""
    hp = _hp(scene)
    hair = hp.hair
    if hair is None or hair.type != 'CURVES' or not len(hair.data.points):
        RT.solver = None
        RT.bind_ok = False
        hp.status = "No hair curves"
        return None
    deps = deps or bpy.context.evaluated_depsgraph_get()
    pos, off, rad = read_curves(hair, world=True)
    p = params_from_scene(scene)
    guide_only = bool(getattr(hair.arc_flow_hair, 'is_guide_only', False)) or p.guide_ratio >= 0.999
    if not guide_only and p.guide_ratio < 1.0 and len(off) - 1 > 1:
        # split into guides (simulated) and children (skinned)
        gidx = HS.select_guides(len(off) - 1, ratio=p.guide_ratio, seed=scene.arc_flow.seed if hasattr(scene, 'arc_flow') else 0)
        all_idx = np.arange(len(off) - 1, dtype=np.int64)
        child_idx = np.setdiff1d(all_idx, gidx, assume_unique=False)
        g_pos, g_off = HS.pack_strands([pos[off[i]:off[i + 1]] for i in gidx])
        g_rad = np.concatenate([rad[off[i]:off[i + 1]] for i in gidx]) if len(gidx) else rad[:0]
        solver = HairSolver(g_pos, g_off, p, radii=g_rad)
        if len(child_idx):
            c_pos, c_off = HS.pack_strands([pos[off[i]:off[i + 1]] for i in child_idx])
            solver.auto_guides(ratio=1.0, seed=0)
            solver.attach_children(c_pos, c_off)
        solver._all_guide_idx = gidx
        solver._all_child_idx = child_idx
        solver._full_offsets = off
        solver._full_n = len(off) - 1
        solver._full_rest = pos.copy()
    else:
        solver = HairSolver(pos, off, p, radii=rad)
        solver.auto_guides(ratio=1.0, seed=0)
        solver._all_guide_idx = np.arange(solver.n_strands, dtype=np.int64)
        solver._all_child_idx = np.zeros(0, np.int64)
        solver._full_offsets = off
        solver._full_n = solver.n_strands
        solver._full_rest = pos.copy()
    # bind roots: a deforming scalp already carries armature motion, so bind to
    # the mesh (not the bone matrix — mixing the two shifts rest_local off the surface).
    surf = hp.surface or hp.collider
    if surf is not None:
        V, T = _mesh_arrays(surf, deps, world=True)
        if V is not None and len(T):
            solver.bind_to_mesh(V, T)
        elif hp.armature is not None:
            Bm = _bone_matrix(hp.armature, hp.bone)
            solver.bind_roots(np.repeat(Bm[None], max(solver.n_strands, 1), axis=0))
        else:
            solver.bind_roots()
    elif hp.armature is not None:
        Bm = _bone_matrix(hp.armature, hp.bone)
        solver.bind_roots(np.repeat(Bm[None], max(solver.n_strands, 1), axis=0))
    else:
        Mw = np.array(hair.matrix_world, dtype=np.float64)
        solver.bind_roots(np.repeat(Mw[None], max(solver.n_strands, 1), axis=0))
    field = _ensure_field(scene, deps)
    if field is not None and hp.collider is not None:
        solver.set_colliders([])
        solver.add_collider(field, matrix=np.array(hp.collider.matrix_world, dtype=np.float64),
                            radius=float(hp.collision_radius))
        from . import asset_colliders as AC
        AC.mark_projection_collider(hp.collider)
        if hp.surface is not None:
            AC.mark_projection_collider(hp.surface)
        _add_scene_hair_colliders(scene, solver, deps, skip=hp.collider)
    solver.set_wind(_wind(scene, hp))
    ff = _fluid_fn(scene)
    if ff is not None:
        solver.set_fluid_velocity(ff)
    stiff = float(getattr(hair.arc_flow_hair, 'stiffness', 1.0) or 1.0)
    damp = float(getattr(hair.arc_flow_hair, 'damping', 1.0) or 1.0)
    solver.set_per_strand(stiffness=stiff, damping=damp)
    RT.solver = solver
    RT.scene = scene
    RT.bind_ok = True
    RT.cache.clear()
    RT.child_cache.clear()
    hp.status = "Built %d guides / %d strands" % (solver.n_strands, solver._full_n)
    return solver


def _compose_full(solver):
    """World-space packed positions for the original groom (guides + skinned children)."""
    if solver.child_rest is None or not len(getattr(solver, '_all_child_idx', ())):
        return solver.positions, solver.offsets
    gidx = solver._all_guide_idx
    cidx = solver._all_child_idx
    off = solver._full_offsets
    npts = int(off[-1])
    out = np.empty((npts, 3), np.float64)
    # write guides
    gpos, goff = solver.positions, solver.offsets
    for k, si in enumerate(gidx):
        out[off[si]:off[si + 1]] = gpos[goff[k]:goff[k + 1]]
    cpos = solver.child_positions()
    coff = solver.child_off
    for k, si in enumerate(cidx):
        out[off[si]:off[si + 1]] = cpos[coff[k]:coff[k + 1]]
    return out, off


def step_scene(scene, dt=None, write=True, deps=None):
    """Advance the live solver one tick and optionally write back to the CURVES object."""
    if RT.updating:
        return RT.solver
    hp = _hp(scene)
    deps = deps or bpy.context.evaluated_depsgraph_get()
    if RT.solver is None or not RT.bind_ok:
        if build_solver(scene, deps) is None:
            return None
    solver = RT.solver
    # refresh collider field / matrix and environment forces
    field = _ensure_field(scene, deps)
    if field is not None and hp.collider is not None:
        if not solver.colliders:
            solver.add_collider(field, matrix=np.array(hp.collider.matrix_world, dtype=np.float64),
                                radius=float(hp.collision_radius))
        else:
            solver.colliders[0]["field"] = field
            solver.colliders[0]["matrix"] = np.array(hp.collider.matrix_world, dtype=np.float64)
        _add_scene_hair_colliders(scene, solver, deps, skip=hp.collider)
    solver.set_wind(_wind(scene, hp))
    solver.params.update(gravity=_gravity(scene, hp),
                         rest_shape=float(hp.rest_shape),
                         damping=float(hp.damping),
                         wind_drag=float(hp.wind_drag),
                         fluid_coupling=float(hp.fluid_coupling),
                         collision_radius=float(hp.collision_radius),
                         self_collision=bool(getattr(hp, "self_collision", True)))
    ff = _fluid_fn(scene)
    solver.set_fluid_velocity(ff)
    root_M = _root_matrices(scene, solver, deps)
    t0 = time.perf_counter()
    solver.step(dt=dt, root_M=root_M)
    RT.last_ms = (time.perf_counter() - t0) * 1000.0
    full, off = _compose_full(solver)
    RT.cache[int(scene.frame_current)] = full.copy()
    if write and hp.hair is not None:
        RT.updating = True
        try:
            write_curves(hp.hair, full, radii=None)
        finally:
            RT.updating = False
    hp.status = "Live  %.1f ms   frame %d   %d pts" % (RT.last_ms, int(scene.frame_current), solver.n_points)
    return solver


def reset_scene(scene, write=True):
    hp = _hp(scene)
    if RT.solver is not None:
        RT.solver.reset()
        full, _off = _compose_full(RT.solver)
        if write and hp.hair is not None:
            RT.updating = True
            try:
                write_curves(hp.hair, RT.solver._full_rest if hasattr(RT.solver, '_full_rest') else full)
            finally:
                RT.updating = False
    RT.cache.clear()
    hp.status = "Reset"
    return RT.solver


# ----------------------------------------------------------------------------- live timer / handlers


def _timer():
    if not RT.live:
        return None
    scene = RT.scene or (bpy.context.scene if bpy.context else None)
    if scene is None:
        return 0.1
    hp = _hp(scene)
    if not hp.live:
        RT.live = False
        return None
    # real-time tick: use wall-clock capped dt so a hitch doesn't explode the integrator
    now = time.perf_counter()
    p = params_from_scene(scene)
    if RT.last_clock:
        raw = now - RT.last_clock
        dt = float(np.clip(raw, 1e-4, 2.0 * p.dt))
    else:
        dt = p.dt
    RT.last_clock = now
    try:
        step_scene(scene, dt=dt)
    except Exception as e:
        hp.status = "Live error: %s" % e
    return max(1.0 / 60.0, p.dt / max(p.substeps, 1) * p.substeps)


def _on_frame(scene, *args):
    if RT.updating or not RT.live:
        # still apply cache while scrubbing a bake even if live is off
        if not RT.live and RT.cache and int(scene.frame_current) in RT.cache:
            hp = _hp(scene)
            if hp.hair is not None:
                RT.updating = True
                try:
                    write_curves(hp.hair, RT.cache[int(scene.frame_current)])
                finally:
                    RT.updating = False
        return
    try:
        p = params_from_scene(scene)
        step_scene(scene, dt=p.dt)
    except Exception:
        pass


def _on_depsgraph(scene, depsgraph):
    if RT.updating or not RT.live:
        return
    hp = _hp(scene)
    if not hp.follow_depsgraph:
        return
    interesting = False
    for upd in depsgraph.updates:
        idb = getattr(upd, 'id', None)
        if idb is None:
            continue
        name = getattr(idb, 'name', "")
        if hp.hair is not None and name == hp.hair.name and not upd.is_updated_geometry:
            # transform-only updates on the hair object itself are often our own write-back
            continue
        if hp.armature is not None and name == hp.armature.name:
            interesting = True
            break
        if hp.collider is not None and name == hp.collider.name:
            interesting = True
            break
        if hp.surface is not None and name == hp.surface.name:
            interesting = True
            break
    if interesting:
        try:
            step_scene(scene, dt=params_from_scene(scene).dt, deps=depsgraph)
        except Exception:
            pass


def ensure_live(scene):
    RT.scene = scene
    RT.live = True
    RT.last_clock = 0.0
    if RT.solver is None:
        build_solver(scene)
    if _on_frame not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(_on_frame)
    if _on_depsgraph not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph)
    if not bpy.app.timers.is_registered(_timer):
        bpy.app.timers.register(_timer, first_interval=0.0, persistent=True)
    scene.arc_flow_hair.status = "Live"


def stop_live(scene=None):
    RT.live = False
    if bpy.app.timers.is_registered(_timer):
        try:
            bpy.app.timers.unregister(_timer)
        except Exception:
            pass
    if scene is not None:
        scene.arc_flow_hair.live = False if scene.arc_flow_hair.live else False
        scene.arc_flow_hair.status = "Stopped"


def _clear_handlers():
    for bag, fn in ((bpy.app.handlers.frame_change_post, _on_frame),
                    (bpy.app.handlers.depsgraph_update_post, _on_depsgraph)):
        while fn in bag:
            bag.remove(fn)
    if bpy.app.timers.is_registered(_timer):
        try:
            bpy.app.timers.unregister(_timer)
        except Exception:
            pass


# ----------------------------------------------------------------------------- bake / export


def frame_range(scene):
    hp = _hp(scene)
    if hp.use_scene_range:
        return int(scene.frame_start), int(scene.frame_end)
    return int(hp.frame_start), int(hp.frame_end)


def bake_scene(scene, start=None, end=None):
    """Step every frame in the range, write the CURVES object, fill RT.cache. Returns (frames, path or None)."""
    hp = _hp(scene)
    a, b = frame_range(scene)
    if start is not None:
        a = int(start)
    if end is not None:
        b = int(end)
    if b < a:
        a, b = b, a
    was_live = RT.live
    stop_live(scene)
    if build_solver(scene) is None:
        return [], None
    reset_scene(scene, write=True)
    frames = []
    p = params_from_scene(scene)
    cur = int(scene.frame_current)
    try:
        for f in range(a, b + 1):
            scene.frame_set(f)
            step_scene(scene, dt=p.dt, write=True)
            rec = dict(frame=f, time=(f - a) * p.dt, positions=_compose_full(RT.solver)[0].copy())
            if RT.solver.child_rest is not None:
                rec["child_positions"] = RT.solver.child_positions()
                rec["child_offsets"] = RT.solver.child_off.copy()
            frames.append(rec)
    finally:
        scene.frame_set(cur)
    folder = _abspath(scene, hp.bake_dir) or os.path.join(os.path.dirname(bpy.data.filepath) or os.getcwd(), "tessera_hair_cache")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "%s.npz" % (hp.bake_name or "hair"))
    HS.write_npz_cache(path, frames, RT.solver._full_offsets, rest=RT.solver._full_rest,
                       meta=dict(format=HS.PACKAGE_FORMAT, start=a, end=b))
    hp.last_cache = path
    hp.status = "Baked %d frames → %s" % (len(frames), path)
    if was_live:
        hp.live = True
        ensure_live(scene)
    return frames, path


def export_package(scene, folder=None):
    hp = _hp(scene)
    if RT.solver is None and build_solver(scene) is None:
        return None
    folder = _abspath(scene, folder or hp.package_dir)
    os.makedirs(folder, exist_ok=True)
    meta = []
    if hp.collider is not None:
        bb = [list(hp.collider.bound_box[i]) for i in range(8)] if getattr(hp.collider, 'bound_box', None) else []
        meta.append(dict(
            name=hp.collider.name,
            type="mesh_sdf",
            resolution=int(hp.sdf_resolution),
            level=int(hp.sdf_level),
            radius=float(hp.collision_radius),
            matrix=np.array(hp.collider.matrix_world, dtype=float).tolist(),
            bound_box=bb,
        ))
    extra = dict(hair_object=hp.hair.name if hp.hair else "",
                 armature=hp.armature.name if hp.armature else "",
                 bone=hp.bone or "")
    frames = [dict(frame=f, positions=P, offsets=RT.solver._full_offsets)
              for f, P in sorted(RT.cache.items())] or None
    path = HS.write_package(folder, RT.solver, frames=frames, collider_meta=meta, extra=extra)
    hp.last_package = path
    hp.status = "Package → %s" % path
    return path


def export_alembic(scene, filepath, start=None, end=None):
    hp = _hp(scene)
    hair = hp.hair
    if hair is None:
        return None
    a, b = frame_range(scene)
    if start is not None:
        a = int(start)
    if end is not None:
        b = int(end)
    # ensure cache covers the range; bake missing frames if needed
    if not RT.cache or min(RT.cache or [a]) > a or max(RT.cache or [b]) < b:
        bake_scene(scene, start=a, end=b)
    filepath = _abspath(scene, filepath)
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    # apply cache on frame change during the export
    was_live = RT.live
    RT.live = False
    sel = {o: o.select_get() for o in scene.objects}
    try:
        bpy.ops.object.select_all(action='DESELECT')
        hair.hide_set(False)
        hair.select_set(True)
        scene.view_layers[0].objects.active = hair
        kw = dict(filepath=filepath, start=a, end=b, selected=True, flatten=False,
                  visible_objects_only=False, as_background_job=False)
        try:   # keep only the keywords this Blender's exporter knows (visible_objects_only is gone in 5.3)
            known = {p.identifier for p in bpy.ops.wm.alembic_export.get_rna_type().properties}
            kw = {k: v for k, v in kw.items() if k in known}
        except Exception:
            kw.pop('visible_objects_only', None)
        bpy.ops.wm.alembic_export(**kw)
        hp.last_alembic = filepath
        hp.status = "Alembic → %s" % filepath
        return filepath
    finally:
        for o, s in sel.items():
            try:
                o.select_set(s)
            except Exception:
                pass
        if was_live:
            RT.live = True
            ensure_live(scene)


# ----------------------------------------------------------------------------- example / add-hair helpers


def _ensure_active_mesh(context, obj=None):
    """Empty Hair add requires an active mesh (the surface)."""
    if obj is None:
        obj = context.active_object if context.active_object and context.active_object.type == 'MESH' else None
    if obj is None:
        return None
    context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj


def add_empty_hair(context, surface, name="Tessera Hair", n_strands=80, n_pts=10, length=0.28,
                   radius_scale=1.0, seed=0, attach_collider=True):
    """Build a styled hemisphere groom on `surface`, as a native CURVES object bound to it."""
    scene = context.scene
    hp = _hp(scene)
    V, T = _mesh_arrays(surface, world=True)
    if V is None:
        raise RuntimeError("surface must be a mesh")
    # fit a sphere to the upper part of the mesh for a default scalp distribution
    c = V.mean(axis=0)
    up = np.array((0.0, 0.0, 1.0))
    top = V[V[:, 2] >= np.median(V[:, 2])]
    c = np.array((top[:, 0].mean(), top[:, 1].mean(), top[:, 2].mean()))
    r = np.median(np.linalg.norm(top - c, axis=1)) or 0.12
    P, off = HS.make_hemisphere_groom(n_strands=n_strands, n_pts=n_pts, radius=max(r * 0.92, 1e-3),
                                      length=length, center=c, seed=seed, sag=0.85, curl=0.35)
    # snap roots onto the actual mesh surface and lift them a hair-width outside
    from .core import sdf as SDF
    soup = SDF.soup_from_mesh(V, T)
    field = HS.make_collider(V, T, resolution=32, level=1, mode='SOLID')
    roots = P[off[:-1]]
    cp, _sd, _ti = SDF.closest_surface(soup, roots)
    out = P.copy()
    for i, (a, b) in enumerate(zip(off[:-1], off[1:])):
        out[a:b] = P[a:b] + (cp[i] - P[a])
    P, off = _lift_roots(out, off, field, matrix=np.eye(4), radius=0.004)
    rad = np.full(len(P), 0.006 * float(radius_scale))
    # store in the hair object's local space (identity at world)
    ob = make_curves_object(name, P, off, radii=rad, collection=scene.collection)
    ob.arc_flow_hair.role = 'HAIR'
    ob.arc_flow_hair.radius = float(rad[0])
    try:
        ob.data.surface = surface
    except Exception:
        pass
    surface.arc_flow_hair.role = 'SURFACE' if not attach_collider else surface.arc_flow_hair.role
    if attach_collider:
        if surface.arc_flow_hair.role == 'NONE':
            surface.arc_flow_hair.role = 'COLLIDER'
        hp.collider = surface
        hp.surface = surface
    hp.hair = ob
    from . import hair_gn
    hair_gn.apply_hair_gn(ob, radius=float(rad[0]))
    return ob


def _lift_roots(P, off, field, matrix, radius):
    roots = P[off[:-1]].copy()
    Xl = HS.inv_xform_points(roots, matrix)
    sd = field.sample_sd(Xl)
    # finite-difference normals
    eps = max(getattr(field, 'h', 0.01) * 0.4, 1e-5)
    g = np.zeros_like(Xl)
    for ax, e in enumerate(np.eye(3)):
        g[:, ax] = (field.sample_sd(Xl + e * eps) - field.sample_sd(Xl - e * eps)) / (2 * eps)
    n = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
    # move to sd == radius (outside)
    Xl = Xl + n * (float(radius) - sd)[:, None]
    new_roots = HS.xform_points(Xl, matrix)
    out = P.copy()
    for i, (a, b) in enumerate(zip(off[:-1], off[1:])):
        out[a:b] = P[a:b] + (new_roots[i] - P[a])
    return out, off


def build_groom_example(scene, n_strands=96, n_pts=10, with_armature=True, seed=2):
    """A ready-to-play character-hair scene: scalp sphere, collider, armature, Empty Hair, camera, light."""
    # drop leftover example objects if the operator is run twice
    for nm in list(bpy.data.objects.keys()):
        if not nm.startswith("TS_Hair"):
            continue
        ob = bpy.data.objects[nm]
        data = ob.data
        bpy.data.objects.remove(ob, do_unlink=True)
        if data is not None and getattr(data, 'users', 1) == 0:
            for coll in (bpy.data.meshes, bpy.data.hair_curves, bpy.data.armatures, bpy.data.cameras, bpy.data.lights):
                if getattr(data, 'name', None) in coll:
                    try:
                        coll.remove(data)
                    except Exception:
                        pass
                    break
    # head
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=0.12, location=(0.0, 0.0, 1.62))
    head = bpy.context.active_object
    head.name = "TS_HairHead"
    bpy.ops.object.shade_smooth()
    head.arc_flow_hair.role = 'COLLIDER'
    # ground
    bpy.ops.mesh.primitive_plane_add(size=2.4, location=(0.0, 0.0, 0.0))
    ground = bpy.context.active_object
    ground.name = "TS_HairGround"
    # armature
    arm = None
    if with_armature:
        arm_data = bpy.data.armatures.new("TS_HairRigData")
        arm = bpy.data.objects.new("TS_HairRig", arm_data)
        scene.collection.objects.link(arm)
        bpy.context.view_layer.objects.active = arm
        bpy.ops.object.mode_set(mode='EDIT')
        eb = arm_data.edit_bones.new("root")
        eb.head, eb.tail = (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)
        neck = arm_data.edit_bones.new("neck")
        neck.head, neck.tail = (0.0, 0.0, 1.0), (0.0, 0.0, 1.5)
        neck.parent = eb
        headb = arm_data.edit_bones.new("head")
        headb.head, headb.tail = (0.0, 0.0, 1.5), (0.0, 0.0, 1.74)
        headb.parent = neck
        bpy.ops.object.mode_set(mode='OBJECT')
        # parent the scalp to the head bone
        head.parent = arm
        head.parent_type = 'BONE'
        head.parent_bone = "head"
        # bone-parented objects are relative to the bone tip in Blender; compensate
        # After bone parent, the child's matrix is in bone space. Reset to sit on the bone.
        bpy.context.view_layer.update()
        # Place the head at the bone tip in bone-local coordinates.
        # A UV sphere of radius 0.12 centred so the neck meets the chin area: origin at the sphere centre.
        # For BONE parent the child's location is relative to the bone's *tail*? Actually Blender uses the bone head
        # in some versions and the bone matrix (head origin, Y along bone) in others. We'll set location after update
        # so the world position stays (0,0,1.62).
        bpy.context.view_layer.update()
        head.matrix_world = Matrix.Translation((0.0, 0.0, 1.62))
        bpy.context.view_layer.update()
    # hair
    ctx = bpy.context
    bpy.context.view_layer.objects.active = head
    hair = add_empty_hair(ctx, head, name="TS_HairCurves", n_strands=n_strands, n_pts=n_pts,
                          length=0.34, seed=seed, attach_collider=True)
    hp = _hp(scene)
    hp.hair = hair
    hp.collider = head
    hp.surface = head
    if arm is not None:
        hp.armature = arm
        hp.bone = "head"
    hp.use_ground = True
    hp.ground_z = 0.0
    # camera + light
    if "TS_HairCam" not in bpy.data.objects:
        cam = bpy.data.cameras.new("TS_HairCam")
        cam.lens = 50
        cob = bpy.data.objects.new("TS_HairCam", cam)
        scene.collection.objects.link(cob)
        cob.location = (0.55, -1.15, 1.85)
        cob.rotation_euler = (1.35, 0.0, 0.45)
        scene.camera = cob
    if "TS_HairKey" not in bpy.data.objects:
        key = bpy.data.lights.new("TS_HairKey", 'AREA')
        key.energy = 250
        key.size = 0.8
        k = bpy.data.objects.new("TS_HairKey", key)
        scene.collection.objects.link(k)
        k.location = (0.7, -0.6, 2.4)
        fill = bpy.data.lights.new("TS_HairFill", 'AREA')
        fill.energy = 80
        fill.size = 1.2
        f = bpy.data.objects.new("TS_HairFill", fill)
        scene.collection.objects.link(f)
        f.location = (-0.8, -0.2, 1.8)
    # a simple head material + hair GN already applied
    mat = bpy.data.materials.new("TS_HairSkin")
    mat.use_nodes = True
    nt = mat.node_tree
    prin = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if prin is not None:
        prin.inputs['Base Color'].default_value = (0.62, 0.45, 0.35, 1.0)
        if 'Roughness' in prin.inputs:
            prin.inputs['Roughness'].default_value = 0.45
    if head.data.materials:
        head.data.materials[0] = mat
    else:
        head.data.materials.append(mat)
    from . import hair_gn
    hair_gn.apply_hair_material(hair, color=(0.12, 0.07, 0.04, 1.0))
    build_solver(scene)
    hp.status = "Hair groom example ready"
    return dict(hair=hair, head=head, armature=arm, ground=ground)


# ----------------------------------------------------------------------------- operators


class AF_OT_hair_add_empty(Operator):
    bl_idname = "arcflow.hair_add_empty"
    bl_label = "Add Empty Hair Curves"
    bl_description = "Add a native Empty Hair Curves object and attach it to the active mesh (or the Collision body) as a scalp"
    bl_options = {'REGISTER', 'UNDO'}
    n_strands: IntProperty(name="Strands", default=80, min=1, max=8000)
    n_pts: IntProperty(name="Points per strand", default=10, min=2, max=64)
    length: bpy.props.FloatProperty(name="Length", default=0.28, min=0.01, unit='LENGTH')

    def execute(self, ctx):
        hp = _hp(ctx.scene)
        surf = ctx.active_object if ctx.active_object and ctx.active_object.type == 'MESH' else (hp.surface or hp.collider)
        if surf is None or surf.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh (scalp / collision body) first")
            return {'CANCELLED'}
        try:
            ob = add_empty_hair(ctx, surf, n_strands=self.n_strands, n_pts=self.n_pts, length=self.length)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        ctx.view_layer.objects.active = ob
        self.report({'INFO'}, "Added %s" % ob.name)
        return {'FINISHED'}

    def invoke(self, ctx, ev):
        return ctx.window_manager.invoke_props_dialog(self)


class AF_OT_hair_example(Operator):
    bl_idname = "arcflow.hair_example"
    bl_label = "Hair Groom Example"
    bl_description = "Build a scalp, collision body, armature and Empty Hair Curves groom, ready for live simulation or bake"
    bl_options = {'REGISTER', 'UNDO'}
    n_strands: IntProperty(name="Strands", default=96, min=8, max=4000)
    n_pts: IntProperty(name="Points per strand", default=10, min=2, max=64)

    def execute(self, ctx):
        build_groom_example(ctx.scene, n_strands=self.n_strands, n_pts=self.n_pts)
        self.report({'INFO'}, "Hair groom example built")
        return {'FINISHED'}

    def invoke(self, ctx, ev):
        return ctx.window_manager.invoke_props_dialog(self)


class AF_OT_hair_live_start(Operator):
    bl_idname = "arcflow.hair_live_start"
    bl_label = "Start Live Simulation"
    bl_description = "Tick Tessera Projection Hair in the viewport as you pose or play the character"
    bl_options = {'REGISTER'}

    def execute(self, ctx):
        hp = _hp(ctx.scene)
        if hp.hair is None:
            self.report({'ERROR'}, "Assign a Hair curves object first (or build the Hair Groom Example)")
            return {'CANCELLED'}
        hp.live = True
        ensure_live(ctx.scene)
        self.report({'INFO'}, "Live hair simulation on")
        return {'FINISHED'}


class AF_OT_hair_live_stop(Operator):
    bl_idname = "arcflow.hair_live_stop"
    bl_label = "Stop Live Simulation"
    bl_options = {'REGISTER'}

    def execute(self, ctx):
        stop_live(ctx.scene)
        _hp(ctx.scene).live = False
        self.report({'INFO'}, "Live hair simulation off")
        return {'FINISHED'}


class AF_OT_hair_step(Operator):
    bl_idname = "arcflow.hair_step"
    bl_label = "Step Hair"
    bl_description = "Advance the hair solver by one frame without turning Live mode on"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        if step_scene(ctx.scene) is None:
            self.report({'ERROR'}, "No hair curves to step")
            return {'CANCELLED'}
        self.report({'INFO'}, _hp(ctx.scene).status)
        return {'FINISHED'}


class AF_OT_hair_reset(Operator):
    bl_idname = "arcflow.hair_reset"
    bl_label = "Reset Hair"
    bl_description = "Restore the groom to its rest pose and clear the live cache"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        if RT.solver is None:
            build_solver(ctx.scene)
        reset_scene(ctx.scene)
        self.report({'INFO'}, "Hair reset to rest pose")
        return {'FINISHED'}


class AF_OT_hair_rebuild(Operator):
    bl_idname = "arcflow.hair_rebuild"
    bl_label = "Rebuild Solver"
    bl_description = "Rebuild the hair solver from the current curves, collision body and settings"
    bl_options = {'REGISTER'}

    def execute(self, ctx):
        if build_solver(ctx.scene) is None:
            self.report({'ERROR'}, "No hair curves to build from")
            return {'CANCELLED'}
        self.report({'INFO'}, _hp(ctx.scene).status)
        return {'FINISHED'}


class AF_OT_hair_bake(Operator):
    bl_idname = "arcflow.hair_bake"
    bl_label = "Bake Hair"
    bl_description = "Simulate the frame range and write a .npz point cache (and Alembic, if chosen)"
    bl_options = {'REGISTER'}

    def execute(self, ctx):
        hp = _hp(ctx.scene)
        if hp.hair is None:
            self.report({'ERROR'}, "Assign a Hair curves object first")
            return {'CANCELLED'}
        frames, path = bake_scene(ctx.scene)
        if not frames:
            self.report({'ERROR'}, hp.status or "Bake failed")
            return {'CANCELLED'}
        if hp.bake_format in {'ABC', 'BOTH'}:
            folder = _abspath(ctx.scene, hp.bake_dir)
            abc = os.path.join(folder, "%s.abc" % (hp.bake_name or "hair"))
            try:
                export_alembic(ctx.scene, abc)
            except Exception as e:
                self.report({'WARNING'}, "Cache saved, Alembic failed: %s" % e)
                return {'FINISHED'}
        self.report({'INFO'}, hp.status)
        return {'FINISHED'}


class AF_OT_hair_export_package(Operator):
    bl_idname = "arcflow.hair_export_package"
    bl_label = "Export Game-Engine Package"
    bl_description = "Write rest curves, guide weights, per-strand stiffness and collision metadata for Unreal / Unity / web"
    bl_options = {'REGISTER'}

    def execute(self, ctx):
        path = export_package(ctx.scene)
        if not path:
            self.report({'ERROR'}, "No hair solver to export")
            return {'CANCELLED'}
        self.report({'INFO'}, "Wrote %s" % path)
        return {'FINISHED'}


class AF_OT_hair_export_alembic(Operator, ExportHelper):
    bl_idname = "arcflow.hair_export_alembic"
    bl_label = "Export Hair Alembic"
    bl_description = "Export the baked (or just-baked) hair curves as Alembic"
    bl_options = {'REGISTER'}
    filename_ext = ".abc"
    filter_glob: StringProperty(default="*.abc", options={'HIDDEN'})

    def execute(self, ctx):
        try:
            path = export_alembic(ctx.scene, self.filepath)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        if not path:
            self.report({'ERROR'}, "No hair curves to export")
            return {'CANCELLED'}
        self.report({'INFO'}, "Alembic → %s" % path)
        return {'FINISHED'}


class AF_OT_hair_set_role(Operator):
    bl_idname = "arcflow.hair_set_role"
    bl_label = "Set Hair Role"
    bl_options = {'REGISTER', 'UNDO'}
    role: EnumProperty(items=[
        ('HAIR', "Hair curves", ""),
        ('COLLIDER', "Collision body", ""),
        ('SURFACE', "Scalp / surface", ""),
        ('NONE', "None", ""),
    ], default='HAIR')

    def execute(self, ctx):
        ob = ctx.active_object
        if ob is None:
            return {'CANCELLED'}
        ob.arc_flow_hair.role = self.role
        hp = _hp(ctx.scene)
        if self.role == 'HAIR' and ob.type == 'CURVES':
            hp.hair = ob
        elif self.role == 'COLLIDER' and ob.type == 'MESH':
            hp.collider = ob
        elif self.role == 'SURFACE' and ob.type == 'MESH':
            hp.surface = ob
        return {'FINISHED'}


CLASSES = (
    AF_OT_hair_add_empty, AF_OT_hair_example,
    AF_OT_hair_live_start, AF_OT_hair_live_stop, AF_OT_hair_step, AF_OT_hair_reset, AF_OT_hair_rebuild,
    AF_OT_hair_bake, AF_OT_hair_export_package, AF_OT_hair_export_alembic, AF_OT_hair_set_role,
)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    _clear_handlers()
    RT.live = False
    RT.solver = None
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
