"""Tessera Projection Muscle tab: UI-only operators and the Muscle Pencil viewport overlay.

Owned by the Muscle UI worker (see docs/handoff/COORDINATION.md).  The muscle data schema (``muscle_props``) and the solver /
bake / export logic (``muscle_ops``, ``core.muscle_solver``) belong to the lead; this module only *calls* them:

* ``arcflow.muscle_pencil_stroke``  scriptable Muscle Pencil stroke: points (world or skin-local) are snapped to the skin and
  joined along the shortest path over the mesh edges, so the line follows the surface contour like a Grease Pencil stroke
  projected onto the surface.  The interactive pencil (``arcflow.muscle_contour``) commits through the same helper.
* ``arcflow.muscle_ui_set_preset``  switch preset and / or detail level and repopulate the region cards, keeping the work
  already done on muscles that stay in the list (Complete flag, sliders, driving bone, muscle / bone geometry).
* ``arcflow.muscle_ui_stroke_undo``  remove the active muscle's last pencil line.
* ``arcflow.muscle_ui_cards_expand``  expand or collapse every region card.
* a POST_VIEW draw handler that draws every pencil line in its muscle's static colour (the active muscle thicker), plus the
  line being drawn while the pencil is live.
"""
import heapq
import json

import bpy
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, FloatVectorProperty, IntProperty, StringProperty
from bpy.types import Operator, PropertyGroup

from . import muscle_ops as MO
from .core import muscle_solver as MS

# live (in-progress) pencil line, drawn by the overlay: skin-local points and an RGB colour
LIVE = {'skin': None, 'pts': [], 'color': (1.0, 1.0, 1.0)}
_ADJ_CACHE = {}
_DRAW = {'handle': None, 'key': None, 'batches': None}


def _ms(scene):
    return getattr(scene, 'arc_flow_muscle', None)


# --------------------------------------------------------------------------------------------------------------------
# preset families (derived from the lead's MS.PRESETS so new presets show up without UI edits)
# --------------------------------------------------------------------------------------------------------------------
def preset_family(key):
    """'CUSTOM', 'QUADRUPED' or 'BIPED' for a preset key."""
    if key == 'CUSTOM':
        return 'CUSTOM'
    p = MS.PRESETS.get(key, {})
    return 'QUADRUPED' if p.get('body') == 'Q' else 'BIPED'


def family_presets(family):
    """Preset keys of one family, catalog order (e.g. BIPED -> BIPED, HUMAN_MALE, HUMAN_FEMALE)."""
    return [k for k in MS.PRESETS if preset_family(k) == family]


def variant_label(key):
    """Short button label for a preset variant ('Generic', 'Human male', 'Feline', ...)."""
    lab = MS.PRESETS[key]['label']
    if '(generic)' in lab:
        return "Generic"
    if ':' in lab:
        return lab.split(':', 1)[1].strip().capitalize()
    return lab


def detail_items():
    return [(d[0], d[1], d[2]) for d in MS.DETAIL_ITEMS]


def expected_keys(ms):
    """Muscle keys the current preset / detail / creature counts would list (to flag a stale card list)."""
    cr = dict(heads=ms.heads, arms=ms.arms, legs=ms.legs, tails=ms.tails, fingers=ms.fingers, toes=ms.toes,
              joints_per_limb=ms.joints_per_limb, spine_segments=3)
    try:
        return [d['key'] for d in MS.regions_for(ms.preset, ms.detail, cr)]
    except Exception:
        return None


def cards_stale(ms):
    """True when the card list no longer matches the preset settings (preset / detail / counts changed without Apply)."""
    want = expected_keys(ms)
    if want is None:
        return False
    have = [r.key for r in ms.regions]
    return not set(want).issubset(have)


# --------------------------------------------------------------------------------------------------------------------
# preset switching with the artist's work kept
# --------------------------------------------------------------------------------------------------------------------
_KEEP_FIELDS = ('completed', 'enabled', 'flex', 'accent', 'accent_end', 'tone', 'joint_range', 'bone', 'attach_bone_ends',
                'muscle_obj', 'bone_obj')


def switch_preset(scene, preset=None, detail=None):
    """Set preset / detail, rebuild the region list through the lead's ``apply_preset`` and restore per-muscle work.
    Returns the number of muscles listed."""
    ms = _ms(scene)
    if preset:
        if preset not in MS.PRESETS:
            raise ValueError("Unknown preset %r" % preset)
        ms.preset = preset
    if detail:
        if detail not in MS.DETAIL_TIER:
            raise ValueError("Unknown detail level %r" % detail)
        ms.detail = detail
    snap = {}
    for r in ms.regions:
        snap[r.key] = {f: getattr(r, f) for f in _KEEP_FIELDS}
    active_key = ms.regions[ms.index].key if 0 <= ms.index < len(ms.regions) else None
    n = MO.apply_preset(scene)
    for i, r in enumerate(ms.regions):
        s = snap.get(r.key)
        if not s:
            continue
        for f, v in s.items():
            try:
                setattr(r, f, v)
            except Exception:
                pass
    if active_key is not None:
        ms.index = next((i for i, r in enumerate(ms.regions) if r.key == active_key), 0)
    if ms.skin is not None:
        try:
            MO.refresh_stroke_counts(ms.skin, ms)
            MO.refresh_painted_counts(ms.skin, ms)
        except Exception:
            pass
    tag_redraw_all()
    return n


# --------------------------------------------------------------------------------------------------------------------
# Muscle Pencil geometry: snap to the skin and follow the surface along mesh edges
# --------------------------------------------------------------------------------------------------------------------
def _adjacency(skin):
    """CSR edge adjacency (python lists, for the Dijkstra) and a KD-tree of the rest (basis) vertices, cached per mesh."""
    me = skin.data
    key = (me.name, len(me.vertices), len(me.edges))
    hit = _ADJ_CACHE.get(skin.name)
    if hit is not None and hit[0] == key:
        return hit[1], hit[2]
    import numpy as np
    v = MO.basis_coords(skin)
    e = np.empty(len(me.edges) * 2, np.int64)
    me.edges.foreach_get("vertices", e)
    e = e.reshape(-1, 2)
    src = np.concatenate([e[:, 0], e[:, 1]])
    dst = np.concatenate([e[:, 1], e[:, 0]])
    o = np.argsort(src, kind='stable')
    src, dst = src[o], dst[o]
    w = np.linalg.norm(v[src] - v[dst], axis=1)
    indptr = np.zeros(len(v) + 1, np.int64)
    np.cumsum(np.bincount(src, minlength=len(v)), out=indptr[1:])
    adj = (indptr.tolist(), dst.tolist(), w.tolist())
    from mathutils import kdtree
    kd = kdtree.KDTree(len(v))
    for i, p in enumerate(v.tolist()):
        kd.insert(p, i)
    kd.balance()
    _ADJ_CACHE.clear()
    _ADJ_CACHE[skin.name] = (key, adj, kd)
    return adj, kd


def _edge_path(adj, a, b, max_visit=250000):
    """Shortest path a -> b over mesh edges (Dijkstra on the CSR lists).  [a, b] if b is unreachable within max_visit."""
    if a == b:
        return [a]
    indptr, nbr, wt = adj
    dist = {a: 0.0}
    prev = {}
    pq = [(0.0, a)]
    seen = 0
    while pq:
        d, u = heapq.heappop(pq)
        if u == b:
            break
        if d > dist.get(u, 1e30):
            continue
        seen += 1
        if seen > max_visit:
            return [a, b]
        for k in range(indptr[u], indptr[u + 1]):
            w = nbr[k]
            nd = d + wt[k]
            if nd < dist.get(w, 1e30):
                dist[w] = nd
                prev[w] = u
                heapq.heappush(pq, (nd, w))
    if b not in prev:
        return [a, b]
    out = [b]
    while out[-1] != a:
        out.append(prev[out[-1]])
    return out[::-1]


def snap_points(skin, pts_local):
    """Nearest rest-pose skin vertex for every skin-local point (consecutive duplicates removed)."""
    _adj, kd = _adjacency(skin)
    out = []
    for p in pts_local:
        _co, idx, _d = kd.find(tuple(p))
        if idx is not None and (not out or out[-1] != idx):
            out.append(idx)
    return out


def indices_to_chain(skin, snapped, follow_surface=True):
    """Join consecutive vertex indices along the mesh edges (the line hugs the surface contour) and drop back-tracking
    loops (a b a -> a) that jittery mouse input produces.  Returns an ordered vertex chain."""
    snapped = [int(i) for i in snapped]
    snapped = [i for k, i in enumerate(snapped) if k == 0 or i != snapped[k - 1]]
    if len(snapped) < 2 or not follow_surface:
        return snapped
    adj, _kd = _adjacency(skin)
    chain = [snapped[0]]
    for a, b in zip(snapped, snapped[1:]):
        chain.extend(_edge_path(adj, a, b)[1:])
    out = []
    pos = {}
    for i in chain:
        if i in pos:
            cut = pos[i]
            for j in out[cut + 1:]:
                pos.pop(j, None)
            del out[cut + 1:]
            continue
        pos[i] = len(out)
        out.append(i)
    return out


def points_to_chain(skin, pts_local, follow_surface=True):
    """Snap skin-local points to their nearest rest-pose vertices and join them along the surface."""
    return indices_to_chain(skin, snap_points(skin, pts_local), follow_surface)


def hit_vertex(skin, origin_local, dir_local, depsgraph=None):
    """Skin vertex under a ray (skin-local origin / direction): ray-casts the evaluated (posed, shape-keyed) skin and
    returns (vertex index, local hit point), picking the hit face's corner nearest the hit.  Works on a posed character
    because the face index is shared with the rest mesh (armature / shape keys keep the topology).  None when missed."""
    ok, loc, _nrm, fi = skin.ray_cast(origin_local, dir_local, depsgraph=depsgraph) if depsgraph is not None \
        else skin.ray_cast(origin_local, dir_local)
    if not ok or fi < 0:
        return None
    me = skin.data
    ev = None
    if depsgraph is not None:
        try:
            ev = skin.evaluated_get(depsgraph).data
        except Exception:
            ev = None
    if ev is not None and len(ev.vertices) == len(me.vertices) and fi < len(ev.polygons):
        poly = ev.polygons[fi]
        best = min(poly.vertices, key=lambda i: (ev.vertices[i].co - loc).length)
        return int(best), tuple(loc)
    # topology changed by a modifier: snap the hit point to the rest mesh instead
    s = snap_points(skin, [tuple(loc)])
    return (s[0], tuple(loc)) if s else None


def commit_stroke(scene, chain, index=None):
    """Store a vertex chain as a Muscle Pencil line of muscle `index` (default: active) in the skin's stroke list (the
    format the lead's solver reads: dict(cid, verts)), refresh the visible contour curve and the per-muscle counts.
    Returns the muscle's line count."""
    ms = _ms(scene)
    i = ms.index if index is None or index < 0 else index
    r = ms.regions[i]
    st = MO.load_strokes(ms.skin)
    st.append(dict(cid=int(r.cid), verts=[int(v) for v in chain]))
    MO.save_strokes(ms.skin, st)
    try:
        MO.update_contour_curve(scene, ms.skin)
    except Exception:
        pass
    MO.refresh_stroke_counts(ms.skin, ms)
    MO._log(scene, "Muscle Pencil line added to %s (%d lines). Press Complete when the muscle is marked." % (r.label, r.n_strokes))
    tag_redraw_all()
    return r.n_strokes


def undo_stroke(scene, index=None):
    """Remove the last line of muscle `index` (default: active).  Returns True when one was removed."""
    ms = _ms(scene)
    i = ms.index if index is None or index < 0 else index
    r = ms.regions[i]
    st = MO.load_strokes(ms.skin)
    for j in range(len(st) - 1, -1, -1):
        if st[j].get('cid') == r.cid:
            del st[j]
            break
    else:
        return False
    MO.save_strokes(ms.skin, st)
    try:
        MO.update_contour_curve(scene, ms.skin)
    except Exception:
        pass
    MO.refresh_stroke_counts(ms.skin, ms)
    MO.refresh_painted_counts(ms.skin, ms)
    if r.completed and not (r.n_strokes or r.n_painted or r.muscle_obj or r.bone_obj):
        r.completed = False
    tag_redraw_all()
    return True


def tag_redraw_all():
    wm = getattr(bpy.context, 'window_manager', None)
    if wm is None:
        return
    for win in wm.windows:
        for a in win.screen.areas:
            if a.type == 'VIEW_3D':
                a.tag_redraw()


# --------------------------------------------------------------------------------------------------------------------
# viewport overlay: pencil lines in their muscle colours
# --------------------------------------------------------------------------------------------------------------------
def _overlay_key(scene):
    ms = _ms(scene)
    if ms is None or ms.skin is None or ms.skin.type != 'MESH':
        return None
    skin = ms.skin
    raw = skin.data.get(MO.PROP_STROKES, "")
    cols = tuple((r.cid, tuple(round(c, 3) for c in r.color), r.enabled) for r in ms.regions)
    act = ms.regions[ms.index].cid if 0 <= ms.index < len(ms.regions) else -1
    return (skin.name, len(skin.data.vertices), hash(raw), hash(cols), act, _DRAW.get('gen', 0))


def _build_batches(scene):
    import numpy as np
    import gpu
    from gpu_extras.batch import batch_for_shader
    ms = _ms(scene)
    skin = ms.skin
    strokes = MO.load_strokes(skin)
    if not strokes:
        return []
    v, nrm = _display_coords(skin)
    size = float(np.ptp(v, axis=0).max()) if len(v) else 1.0
    off = max(size * 0.002, 1e-4)
    col = {r.cid: tuple(r.color) for r in ms.regions}
    act = ms.regions[ms.index].cid if 0 <= ms.index < len(ms.regions) else -1
    groups = {True: ([], []), False: ([], [])}
    for st in strokes:
        ids = [i for i in st.get('verts', []) if 0 <= i < len(v)]
        if len(ids) < 2:
            continue
        c = col.get(st.get('cid'), (1.0, 0.85, 0.1)) + (1.0,)
        pos, cl = groups[st.get('cid') == act]
        p = v[ids] + nrm[ids] * off
        for a, b in zip(p[:-1], p[1:]):
            pos.extend((tuple(a), tuple(b)))
            cl.extend((c, c))
    shader = gpu.shader.from_builtin('POLYLINE_FLAT_COLOR')
    out = []
    for active, (pos, cl) in groups.items():
        if pos:
            out.append((batch_for_shader(shader, 'LINES', {"pos": pos, "color": cl}), 5.0 if active else 3.0))
    return out


def _display_coords(skin):
    """Vertex positions / normals the lines are drawn on: the evaluated (posed, shape-keyed) skin when its topology
    matches the rest mesh, else the rest pose."""
    import numpy as np
    try:
        dg = bpy.context.evaluated_depsgraph_get()
        me = skin.evaluated_get(dg).data
        if len(me.vertices) == len(skin.data.vertices):
            n = len(me.vertices)
            v = np.empty(n * 3, np.float64)
            me.vertices.foreach_get('co', v)
            nr = np.empty(n * 3, np.float64)
            me.vertex_normals.foreach_get('vector', nr)
            return v.reshape(-1, 3), nr.reshape(-1, 3)
    except Exception:
        pass
    v = MO.basis_coords(skin)
    _, _, nr = MO.skin_arrays(skin)
    return v, nr


@bpy.app.handlers.persistent
def _on_depsgraph(scene, depsgraph):
    """Re-place the overlay lines when the skin or its skeleton moves (pose, frame change, shape keys)."""
    ms = _ms(scene)
    if ms is None or ms.skin is None or _DRAW['handle'] is None:
        return
    names = {ms.skin.name, ms.armature.name if ms.armature is not None else ""}
    for u in depsgraph.updates:
        idb = getattr(u, 'id', None)
        if idb is not None and getattr(idb, 'name', None) in names and (u.is_updated_geometry or u.is_updated_transform):
            _DRAW['gen'] = _DRAW.get('gen', 0) + 1
            return


def _draw_overlay():
    try:
        ctx = bpy.context
        wm = ctx.window_manager
        if not getattr(wm, 'tp_muscle_overlay', True):
            return
        scene = ctx.scene
        key = _overlay_key(scene)
        live = LIVE['pts'] if LIVE['skin'] is not None else []
        if key is None and not live:
            return
        import gpu
        from gpu_extras.batch import batch_for_shader
        if key != _DRAW['key']:
            _DRAW['batches'] = _build_batches(scene) if key is not None else []
            _DRAW['key'] = key
        skin = _ms(scene).skin
        shader = gpu.shader.from_builtin('POLYLINE_FLAT_COLOR')
        vp = gpu.state.viewport_get()
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('LESS_EQUAL')
        gpu.matrix.push()
        gpu.matrix.multiply_matrix(skin.matrix_world)
        shader.bind()
        shader.uniform_float("viewportSize", (vp[2], vp[3]))
        for batch, width in _DRAW['batches'] or ():
            shader.uniform_float("lineWidth", width)
            batch.draw(shader)
        if len(live) >= 2:
            c = tuple(LIVE['color']) + (1.0,)
            pos = []
            for a, b in zip(live[:-1], live[1:]):
                pos.extend((tuple(a), tuple(b)))
            gpu.state.depth_test_set('NONE')
            shader.uniform_float("lineWidth", 4.0)
            batch_for_shader(shader, 'LINES', {"pos": pos, "color": [c] * len(pos)}).draw(shader)
        gpu.matrix.pop()
        gpu.state.depth_test_set('NONE')
        gpu.state.blend_set('NONE')
    except Exception:
        # never break the viewport for an overlay
        pass


# --------------------------------------------------------------------------------------------------------------------
# operators
# --------------------------------------------------------------------------------------------------------------------
class AF_MusclePencilPoint(PropertyGroup):
    co: FloatVectorProperty(name="Point", size=3, subtype='XYZ')


class _Op:
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, ctx):
        return _ms(ctx.scene) is not None


class AF_OT_muscle_pencil_stroke(_Op, Operator):
    bl_idname = "arcflow.muscle_pencil_stroke"
    bl_label = "Muscle Pencil Stroke"
    bl_description = ("Add a Muscle Pencil line from a list of points: each point snaps to the skin and the line follows the "
                      "surface between them. The line is the muscle's fibre direction; its two ends are the tendon attachments")

    points: CollectionProperty(type=AF_MusclePencilPoint, options={'SKIP_SAVE'})
    space: EnumProperty(name="Space", items=(('WORLD', "World", "Points are in world space"),
                                             ('LOCAL', "Skin local", "Points are in the skin object's local space")),
                        default='WORLD')
    index: IntProperty(name="Muscle", default=-1, description="Muscle index in the list (-1 = active muscle)",
                       options={'SKIP_SAVE'})
    follow_surface: BoolProperty(name="Follow surface", default=True,
                                 description="Join the snapped points along the mesh edges so the line hugs the contour")

    def execute(self, ctx):
        ms = _ms(ctx.scene)
        if ms.skin is None or not ms.regions:
            self.report({'ERROR'}, "Assign a skin mesh and apply a preset first")
            return {'CANCELLED'}
        i = ms.index if self.index < 0 else self.index
        if not 0 <= i < len(ms.regions):
            self.report({'ERROR'}, "No muscle %d" % i)
            return {'CANCELLED'}
        inv = ms.skin.matrix_world.inverted()
        from mathutils import Vector
        pts = [tuple(inv @ Vector(p.co)) if self.space == 'WORLD' else tuple(p.co) for p in self.points]
        chain = points_to_chain(ms.skin, pts, self.follow_surface)
        if len(chain) < 2:
            self.report({'WARNING'}, "A pencil line needs two points on different skin vertices")
            return {'CANCELLED'}
        n = commit_stroke(ctx.scene, chain, i)
        self.report({'INFO'}, "%s: %d lines" % (ms.regions[i].label, n))
        return {'FINISHED'}


class AF_OT_muscle_ui_stroke_undo(_Op, Operator):
    bl_idname = "arcflow.muscle_ui_stroke_undo"
    bl_label = "Remove Last Line"
    bl_description = "Remove the active muscle's last Muscle Pencil line"

    def execute(self, ctx):
        ms = _ms(ctx.scene)
        if ms.skin is None or not ms.regions:
            return {'CANCELLED'}
        if not undo_stroke(ctx.scene):
            self.report({'INFO'}, "%s has no pencil lines" % ms.regions[ms.index].label)
            return {'CANCELLED'}
        return {'FINISHED'}


class AF_OT_muscle_ui_set_preset(_Op, Operator):
    bl_idname = "arcflow.muscle_ui_set_preset"
    bl_label = "Set Muscle Preset"
    bl_description = ("Switch the preset or detail level and repopulate the region cards. Muscles that stay keep their "
                      "Complete mark, sliders, bone and geometry")

    preset: StringProperty(name="Preset", default="", options={'SKIP_SAVE'},
                           description="Preset key (BIPED, HUMAN_MALE, QUADRUPED, CUSTOM, ...); empty keeps the current one")
    family: StringProperty(name="Family", default="", options={'SKIP_SAVE'},
                           description="BIPED / QUADRUPED / CUSTOM: switch family, keeping the variant when possible")
    detail: StringProperty(name="Detail", default="", options={'SKIP_SAVE'},
                           description="Detail level key; empty keeps the current one")

    @classmethod
    def description(cls, ctx, props):
        if props.detail:
            d = next((x for x in MS.DETAIL_ITEMS if x[0] == props.detail), None)
            if d:
                return "%s: %s" % (d[1], d[2])
        if props.preset in MS.PRESETS:
            return "Use the %s muscle preset" % MS.PRESETS[props.preset]['label']
        if props.family:
            return "Use the %s muscle setup and repopulate the region cards" % props.family.title()
        return cls.bl_description

    def execute(self, ctx):
        ms = _ms(ctx.scene)
        preset = self.preset or None
        if self.family and not preset:
            fam = family_presets(self.family)
            if not fam:
                self.report({'ERROR'}, "Unknown preset family %r" % self.family)
                return {'CANCELLED'}
            preset = ms.preset if ms.preset in fam else fam[0]
        try:
            n = switch_preset(ctx.scene, preset, self.detail or None)
        except ValueError as ex:
            self.report({'ERROR'}, str(ex))
            return {'CANCELLED'}
        return {'FINISHED'} if n else {'CANCELLED'}


class AF_OT_muscle_ui_cards_expand(_Op, Operator):
    bl_idname = "arcflow.muscle_ui_cards_expand"
    bl_label = "Expand / Collapse Cards"
    bl_description = "Expand or collapse every region card"
    bl_options = {'INTERNAL'}

    expand: BoolProperty(default=True, options={'SKIP_SAVE'})

    def execute(self, ctx):
        ms = _ms(ctx.scene)
        MO.ensure_cards(ms)
        for c in ms.cards:
            c.expanded = self.expand
        return {'FINISHED'}


CLASSES = (AF_MusclePencilPoint, AF_OT_muscle_pencil_stroke, AF_OT_muscle_ui_stroke_undo, AF_OT_muscle_ui_set_preset,
           AF_OT_muscle_ui_cards_expand)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.WindowManager.tp_muscle_overlay = BoolProperty(
        name="Pencil overlay", default=True,
        description="Draw the Muscle Pencil lines in each muscle's colour in the 3D viewport")
    if _DRAW['handle'] is None and not bpy.app.background:
        try:
            _DRAW['handle'] = bpy.types.SpaceView3D.draw_handler_add(_draw_overlay, (), 'WINDOW', 'POST_VIEW')
        except Exception:
            _DRAW['handle'] = None
    if _on_depsgraph not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph)


def unregister():
    if _on_depsgraph in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph)
    if _DRAW['handle'] is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_DRAW['handle'], 'WINDOW')
        except Exception:
            pass
        _DRAW['handle'] = None
    _DRAW['key'] = _DRAW['batches'] = None
    _ADJ_CACHE.clear()
    LIVE['skin'] = None
    LIVE['pts'] = []
    try:
        del bpy.types.WindowManager.tp_muscle_overlay
    except Exception:
        pass
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
