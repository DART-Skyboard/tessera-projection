"""Shared collision-body registry for every Tessera Projection asset.

Any object assigned as a Cloth garment, Hair collider / scalp, Fluid or
Combustion collider, or Wind-tunnel body is marked as a collision body that
self-collides (in its own solver) and is visible to every other solver as an
inter-body collider.  Per-asset UI toggles still exist and default to on.
"""

# mesh types that can carry a Tessera collision role
_MESH = {'MESH'}


def mark_projection_collider(obj, enabled=True):
    """Mark ``obj`` as a Tessera collision body across Cloth, Fluid/Combustion and the wind tunnel."""
    if obj is None or getattr(obj, 'type', None) not in _MESH:
        return obj
    if hasattr(obj, 'arc_flow_cloth'):
        obj.arc_flow_cloth.is_collider = bool(enabled)
    if hasattr(obj, 'arc_flow_fm'):
        obj.arc_flow_fm.collider = bool(enabled)
    if hasattr(obj, 'arc_flow_wt'):
        obj.arc_flow_wt.body = bool(enabled)
    return obj


def is_projection_collider(obj):
    if obj is None or getattr(obj, 'type', None) not in _MESH:
        return False
    cl = getattr(obj, 'arc_flow_cloth', None)
    if cl is not None and (getattr(cl, 'is_collider', False) or getattr(cl, 'is_garment', False)):
        return True
    fm = getattr(obj, 'arc_flow_fm', None)
    if fm is not None and getattr(fm, 'collider', False):
        return True
    wt = getattr(obj, 'arc_flow_wt', None)
    if wt is not None and getattr(wt, 'body', False):
        return True
    return False


def iter_collision_meshes(scene, exclude=None):
    """Yield unique mesh objects in ``scene`` that are Tessera collision bodies, skipping ``exclude``."""
    seen = set()
    if exclude is None:
        skip = set()
    elif isinstance(exclude, (list, tuple, set)):
        skip = {id(o) for o in exclude}
    else:
        skip = {id(exclude)}

    def _take(o):
        if o is None or id(o) in seen or id(o) in skip or getattr(o, 'type', None) not in _MESH:
            return False
        seen.add(id(o))
        return True

    cl = getattr(scene, 'arc_flow_cloth', None)
    if cl is not None:
        for g in cl.garments:
            if not g.enabled:
                continue
            objs = [m.obj for m in g.members if m.enabled] if len(g.members) else []
            for o in (objs or [g.obj]):
                if _take(o):
                    yield o
        for c in cl.colliders:
            o = c.obj
            if c.enabled and _take(o):
                yield o
    hp = getattr(scene, 'arc_flow_hair', None)
    if hp is not None:
        for o in (hp.collider, hp.surface):
            if _take(o):
                yield o
    for o in scene.objects:
        if is_projection_collider(o) and _take(o):
            yield o
