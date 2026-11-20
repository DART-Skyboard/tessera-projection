"""Fluid mode assignment model (Tessera Projection): what a Container / Shape slot accepts and how a shape spawns particles.

Any Blender geometry can fill a Container or Shape slot:
  mesh / surface / text / metaball   the evaluated mesh (closed = a volume, open = a sheet)
  curve (or a mesh of only edges)    a *channel*: the curve is tubed (radius = the object's Channel radius, or its own bevel when it has one)
  point cloud                        its points
  particle systems                   the live Blender particles (Spawn mode Particles, or Auto when the object has particles)
A Shape spawns by mode: Volume (fill the closed shape), Surface (one layer on it), Points (its vertices; for a channel the points
along the curve), Particles (the Blender particle positions).  Density is the fraction of the generated points that is kept (0..1).
A new fluid starts as a Primitive shape (sphere of points) that the user edits or replaces by an object.

The numpy half (primitives, tubing, thinning, `spawn`) needs no Blender and is unit-tested; the bpy half reads the evaluated objects."""
import math

import numpy as np

from .core import fluidmode as FM

SPAWN_ITEMS = [('AUTO', "Auto", "Closed shape = volume, open = surface, wire = channel volume, point cloud = points, an object with particles = its particles"),
               ('VOLUME', "Volume fill", "Fill the inside of a closed shape (a channel is filled along its length)"),
               ('SURFACE', "Surface", "One layer of particles on the shape's surface (a channel: on the pipe wall)"),
               ('POINTS', "Vertices / points", "A particle on every vertex (a curve or channel: points along its length)"),
               ('PARTICLES', "Particle system points", "A particle at every live Blender particle of the object")]
PRIM_ITEMS = [('SPHERE', "Sphere", "Sphere of points (the default new fluid)"), ('CUBE', "Cube", ""), ('CYLINDER', "Cylinder", ""),
              ('CONE', "Cone", ""), ('TORUS', "Torus", ""), ('PLANE', "Plane", "An open sheet: spawns on its surface")]
SHAPE_TYPES = {'MESH', 'CURVE', 'SURFACE', 'META', 'FONT', 'POINTCLOUD'}      # any of these (and particle systems) fills a Shape
CONTAINER_TYPES = {'MESH', 'CURVE', 'SURFACE', 'META', 'FONT'}


# ----------------------------------------------------------------------------- numpy: primitives
def _orient(V, T):
    P = V[T]
    if np.einsum('ij,ij->i', P[:, 0], np.cross(P[:, 1], P[:, 2])).sum() < 0:
        T = T[:, ::-1]
    return V, T


def primitive_mesh(kind, size=0.1, seg=24):
    """(verts, tris) of a primitive centred on the origin.  Sphere / Torus: radius / major radius = size; Cube: half edge = size; Cylinder and Cone:
    radius = size, height = 2 size; Plane: half side = size (open).  Closed shapes are wound outward."""
    s = float(size)
    if kind == 'SPHERE':
        rings = max(seg // 2, 4)
        V = [(0, 0, s)]
        for i in range(1, rings):
            th = math.pi * i / rings
            for j in range(seg):
                ph = 2 * math.pi * j / seg
                V.append((s * math.sin(th) * math.cos(ph), s * math.sin(th) * math.sin(ph), s * math.cos(th)))
        V.append((0, 0, -s))
        T = [(0, 1 + (j + 1) % seg, 1 + j) for j in range(seg)]
        for i in range(rings - 2):
            a, b = 1 + i * seg, 1 + (i + 1) * seg
            for j in range(seg):
                j2 = (j + 1) % seg
                T += [(a + j, a + j2, b + j), (a + j2, b + j2, b + j)]
        last, a = len(V) - 1, 1 + (rings - 2) * seg
        T += [(last, a + j, a + (j + 1) % seg) for j in range(seg)]
        return _orient(np.array(V, float), np.array(T, int))
    if kind == 'CUBE':
        V = np.array([(x, y, z) for x in (-s, s) for y in (-s, s) for z in (-s, s)], float)
        F = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        T = np.array([t for a, b, c, d in F for t in ((a, b, c), (a, c, d))], int)
        return _orient(V, T)
    if kind in ('CYLINDER', 'CONE'):
        ring = [(s * math.cos(2 * math.pi * j / seg), s * math.sin(2 * math.pi * j / seg)) for j in range(seg)]
        V = [(x, y, -s) for x, y in ring]
        T = []
        if kind == 'CYLINDER':
            V += [(x, y, s) for x, y in ring]
            for j in range(seg):
                j2 = (j + 1) % seg
                T += [(j, j2, seg + j), (j2, seg + j2, seg + j)]
            V += [(0, 0, -s), (0, 0, s)]
            c0, c1 = 2 * seg, 2 * seg + 1
            T += [(c0, j2, j) for j, j2 in ((j, (j + 1) % seg) for j in range(seg))]
            T += [(c1, seg + j, seg + (j + 1) % seg) for j in range(seg)]
        else:
            V += [(0, 0, s), (0, 0, -s)]
            ap, c0 = seg, seg + 1
            T += [(j, (j + 1) % seg, ap) for j in range(seg)]
            T += [(c0, (j + 1) % seg, j) for j in range(seg)]
        return _orient(np.array(V, float), np.array(T, int))
    if kind == 'TORUS':
        R, r = s, 0.35 * s
        n, m = max(seg, 12), max(seg // 2, 8)
        V = np.array([((R + r * math.cos(2 * math.pi * j / m)) * math.cos(2 * math.pi * i / n),
                       (R + r * math.cos(2 * math.pi * j / m)) * math.sin(2 * math.pi * i / n), r * math.sin(2 * math.pi * j / m))
                      for i in range(n) for j in range(m)], float)
        T = []
        for i in range(n):
            for j in range(m):
                a, b = i * m + j, ((i + 1) % n) * m + j
                a2, b2 = i * m + (j + 1) % m, ((i + 1) % n) * m + (j + 1) % m
                T += [(a, b, a2), (b, b2, a2)]
        return _orient(V, np.array(T, int))
    if kind == 'PLANE':
        return np.array([(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)], float), np.array([(0, 1, 2), (0, 2, 3)], int)
    raise ValueError(kind)


def prim_world(kind, size, loc=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1)):
    V, T = primitive_mesh(kind, size)
    cx, sx = math.cos(rot[0]), math.sin(rot[0])
    cy, sy = math.cos(rot[1]), math.sin(rot[1])
    cz, sz = math.cos(rot[2]), math.sin(rot[2])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    V = (V * np.asarray(scale, float)) @ (Rz @ Ry @ Rx).T + np.asarray(loc, float)
    return V, T


# ----------------------------------------------------------------------------- numpy: channels (tubed curves)
def polylines(nv, edges):
    """Ordered vertex chains (and whether each is a closed loop) of an edge list."""
    adj = [[] for _ in range(nv)]
    for a, b in np.asarray(edges, int).reshape(-1, 2):
        adj[a].append(b); adj[b].append(a)
    seen, out = set(), []

    def walk(start):
        chain, prev, cur = [start], -1, start
        seen.add(start)
        while True:
            nxt = [n for n in adj[cur] if n != prev and n not in seen]
            if not nxt:
                closed = len(chain) > 2 and start in adj[cur] and prev != start
                return chain, closed
            prev, cur = cur, nxt[0]
            seen.add(cur); chain.append(cur)
    for v in range(nv):                  # open chains first (start at an end), then loops
        if v not in seen and len(adj[v]) == 1:
            out.append(walk(v))
    for v in range(nv):
        if v not in seen and adj[v]:
            out.append(walk(v))
    return out


def tube(poly, radius, closed=False, caps=True, sides=12):
    """Tube mesh around one polyline (parallel-transport frames); caps close the ends (wound outward)."""
    P = np.asarray(poly, float)
    n = len(P)
    if n < 2:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    tan = np.zeros_like(P)
    tan[1:-1] = P[2:] - P[:-2]
    tan[0] = P[1] - P[0]
    tan[-1] = P[-1] - P[-2]
    if closed:
        tan[0] = tan[-1] = P[1] - P[-2] if n > 2 else tan[0]
    tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-12)
    ref = np.array([0.0, 0.0, 1.0]) if abs(tan[0, 2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(tan[0], ref); u /= np.linalg.norm(u)
    ang = 2 * math.pi * np.arange(sides) / sides
    V = []
    for i in range(n):
        u = u - tan[i] * (u @ tan[i])
        nu = np.linalg.norm(u)
        u = u / nu if nu > 1e-9 else np.cross(tan[i], ref)
        w = np.cross(tan[i], u)
        V.append(P[i] + radius * (np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * w))
    V = np.concatenate(V)
    T = []
    for i in range(n - 1):
        a, b = i * sides, (i + 1) * sides
        for j in range(sides):
            j2 = (j + 1) % sides
            T += [(a + j, b + j, a + j2), (a + j2, b + j, b + j2)]
    V = V.reshape(-1, 3)
    if caps and not closed:
        c0, c1 = len(V), len(V) + 1
        V = np.vstack([V, P[0], P[-1]])
        T += [(c0, (j + 1) % sides, j) for j in range(sides)]
        e = (n - 1) * sides
        T += [(c1, e + j, e + (j + 1) % sides) for j in range(sides)]
    return V, np.array(T, int)


def channel_mesh(V, edges, radius, caps=True):
    """All the polylines of a wire as tubes in one mesh (overlapping tubes of branches are fine for the sign test)."""
    Vs, Ts, off = [], [], 0
    for chain, closed in polylines(len(V), edges):
        if len(chain) < 2:
            continue
        tv, tt = tube(np.asarray(V)[chain], radius, closed=closed, caps=caps)
        if len(tt):
            Vs.append(tv); Ts.append(tt + off); off += len(tv)
    if not Vs:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    V2, T2 = np.vstack(Vs), np.vstack(Ts)
    return _orient(V2, T2) if caps else (V2, T2)


def sample_polylines(V, edges, step):
    """Points along a wire at `step` spacing (the curve as a line of particles)."""
    out = []
    for chain, closed in polylines(len(V), edges):
        P = np.asarray(V)[chain + ([chain[0]] if closed else [])]
        if len(P) < 2:
            out.append(P); continue
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        k = max(int(s[-1] / step), 1)
        t = np.linspace(0, s[-1], k + 1)
        out.append(np.stack([np.interp(t, s, P[:, a]) for a in range(3)], 1))
    return np.vstack(out) if out else np.zeros((0, 3))


# ----------------------------------------------------------------------------- numpy: spawning
def thin(P, frac, rng):
    P = np.asarray(P, float)
    if frac >= 0.999 or len(P) == 0:
        return P
    k = max(int(round(len(P) * max(frac, 0.0))), 1)
    return P[np.sort(rng.choice(len(P), size=k, replace=False))]


def _geom_kind(g):
    if g['T'] is not None and len(g['T']):
        return 'MESH'
    if g['E'] is not None and len(g['E']):
        return 'WIRE'
    return 'POINTS'


def resolve_mode(g, mode):
    """The mode actually used for geometry g (a dict with V, T, E and optionally from_psys)."""
    kind = _geom_kind(g)
    if mode == 'AUTO':
        if g.get('psys') is not None:
            return 'PARTICLES'
        if kind == 'MESH':
            return 'VOLUME' if FM.is_closed(g['T']) else 'SURFACE'
        return 'VOLUME' if kind == 'WIRE' else 'POINTS'
    if mode == 'VOLUME' and kind == 'MESH' and not FM.is_closed(g['T']):
        return 'SURFACE'
    if mode == 'VOLUME' and kind == 'POINTS':
        return 'POINTS'
    if mode == 'SURFACE' and kind == 'POINTS':
        return 'POINTS'
    if mode == 'PARTICLES' and g.get('psys') is None:
        return resolve_mode(g, 'AUTO')
    return mode


def volume_estimate(g, mode, density, radius):
    """Volume of the shape as the solver will meet it (m3), for the spacing estimate; 0 when it is not a volume or is generated points."""
    m = resolve_mode(g, mode)
    if m != 'VOLUME':
        return 0.0
    V, T = _solid(g, radius)
    return FM.mesh_volume(V, T) if len(T) else 0.0


def _solid(g, radius):
    kind = _geom_kind(g)
    if kind == 'WIRE':
        return channel_mesh(g['V'], g['E'], radius, caps=True)
    return g['V'], g['T']


def spawn(g, mode, density, dx, radius, rng):
    """Shape dict for the solver (None when nothing spawns) and a short note.  g = geometry dict V (world), T, E, psys (points or None)."""
    m = resolve_mode(g, mode)
    kind = _geom_kind(g)
    note = m.title()
    if m == 'PARTICLES':
        P = np.asarray(g['psys'], float)
        note = "%d Blender particles" % len(P)
    elif m == 'POINTS':
        P = sample_polylines(g['V'], g['E'], dx) if kind == 'WIRE' else np.asarray(g['V'], float)
    elif m == 'SURFACE':
        V, T = (channel_mesh(g['V'], g['E'], radius, caps=False) if kind == 'WIRE' else (g['V'], g['T']))
        P = FM.surface_points(V, T, dx, rng) if len(T) else np.zeros((0, 3))
    else:
        V, T = _solid(g, radius)
        if len(T) == 0:
            return None, "empty"
        if density >= 0.999:
            note = "Volume (channel)" if kind == 'WIRE' else "Volume"
            return dict(kind='MESH', verts=V, tris=T), note
        P = FM.lat_pos(FM.interior_lattice(V, T, dx), dx)
    P = thin(P, density, rng)
    if len(P) == 0:
        return None, "empty"
    return dict(kind='POINTS', pts=P), "%s: %d points" % (note, len(P))


def needs_dx(g, mode, density):
    m = resolve_mode(g, mode)
    return m == 'SURFACE' or (m == 'VOLUME' and density < 0.999) or (m == 'POINTS' and _geom_kind(g) == 'WIRE')


# ----------------------------------------------------------------------------- bpy: reading the scene
def _meta_mother(obj):
    import bpy
    base = obj.name.split('.')[0]
    for o in bpy.data.objects:
        if o.type == 'META' and o.name.split('.')[0] == base and o.name == base:
            return o
    return obj


def evaluated_arrays(obj, deps):
    """(world verts, tris, edges) of any geometry object (local arrays transformed by matrix_world)."""
    import bpy
    src = _meta_mother(obj) if obj.type == 'META' else obj
    ev = src.evaluated_get(deps)
    M = np.array(obj.matrix_world)
    if obj.type == 'POINTCLOUD':
        pts = ev.data.points
        n = len(pts)
        co = np.empty(n * 3, np.float64)
        try:
            pts.foreach_get('co', co)
        except Exception:
            ev.data.attributes['position'].data.foreach_get('vector', co)
        V = co.reshape(-1, 3)
        return V @ M[:3, :3].T + M[:3, 3], np.zeros((0, 3), int), np.zeros((0, 2), int)
    try:
        me = ev.to_mesh()
    except RuntimeError:         # an object without geometry (empty, light ...)
        return np.zeros((0, 3)), np.zeros((0, 3), int), np.zeros((0, 2), int)
    try:
        me.calc_loop_triangles()
        nv, nt, ne = len(me.vertices), len(me.loop_triangles), len(me.edges)
        v = np.empty(nv * 3, np.float64)
        me.vertices.foreach_get('co', v)
        t = np.empty(nt * 3, np.int64)
        me.loop_triangles.foreach_get('vertices', t)
        e = np.empty(ne * 2, np.int64)
        me.edges.foreach_get('vertices', e)
    finally:
        ev.to_mesh_clear()
    V = v.reshape(-1, 3)
    return V @ M[:3, :3].T + M[:3, 3], t.reshape(-1, 3), e.reshape(-1, 2)


def particle_points(obj, deps, name=""):
    """World positions of the live emitter particles of an object's particle system (None when there is none)."""
    if not getattr(obj, 'particle_systems', None) or not len(obj.particle_systems):
        return None
    ev = obj.evaluated_get(deps)
    cand = [ps for ps in ev.particle_systems if ps.settings.type == 'EMITTER' and (not name or ps.name == name)]
    if not cand:
        return None
    out = []
    for ps in cand[:1] if name else cand:
        n = len(ps.particles)
        if not n:
            continue
        loc = np.empty(n * 3, np.float64)
        ps.particles.foreach_get('location', loc)
        alive = np.array([p.alive_state == 'ALIVE' for p in ps.particles])
        out.append(loc.reshape(-1, 3)[alive])
    return np.vstack(out) if out else np.zeros((0, 3))


def object_geometry(obj, deps, psys_name=""):
    V, T, E = evaluated_arrays(obj, deps)
    return dict(V=V, T=T, E=E, psys=particle_points(obj, deps, psys_name))


def channel_radius(obj):
    try:
        return float(obj.arc_flow_fm.channel_radius)
    except Exception:
        return 0.02


def primitive_geometry(fl):
    V, T = prim_world(fl.prim_type, fl.prim_size, tuple(fl.prim_loc), tuple(fl.prim_rot), tuple(fl.prim_scale))
    return dict(V=V, T=T, E=np.zeros((0, 2), int), psys=None)


def fluid_geometry(fl, deps):
    """The geometry dict a fluid's Shape slot / Primitive gives (None: no shape, the container is filled)."""
    if fl.shape_source == 'PRIMITIVE':
        return primitive_geometry(fl), 0.02
    if fl.shape is not None and fl.shape.type == 'EMPTY' and fl.shape.get('af_volume'):      # a Fluid Volume box (surfaces_ops)
        from . import surfaces_ops as SO
        from .core import fsurf as FS
        h = SO.volume_half(fl.shape)
        V, T = FS.box_mesh(np.array(fl.shape.matrix_world, float), (h, h, h))
        return dict(V=V, T=T, E=np.zeros((0, 2), int), psys=None), 0.02
    if fl.shape is not None and (fl.shape.type in SHAPE_TYPES or fl.shape.particle_systems):
        return object_geometry(fl.shape, deps, fl.psys_name), channel_radius(fl.shape)
    return None, 0.02


def container_geometry(obj, deps):
    """(world->local verts, tris) of a container object of any type for the signed-distance field (wires are tubed, left open at the ends)."""
    import bpy
    if obj.type == 'MESH' and len(obj.data.polygons):
        return None
    V, T, E = evaluated_arrays(obj, deps)
    if len(T) == 0 and len(E):
        Mi = np.linalg.inv(np.array(obj.matrix_world))
        Vl = V @ Mi[:3, :3].T + Mi[:3, 3]
        s = abs(np.linalg.det(np.array(obj.matrix_world)[:3, :3])) ** (1.0 / 3.0) or 1.0
        v, t = channel_mesh(Vl, E, channel_radius(obj) / s, caps=False)
        return v, t
    Mi = np.linalg.inv(np.array(obj.matrix_world))
    return V @ Mi[:3, :3].T + Mi[:3, 3], T


def container_field(obj, deps):
    """(field, verts, tris) of any container object; meshes use the shared runtime field, other types a field built here (cached)."""
    from . import runtime
    from .core import sdf as SDF
    g = container_geometry(obj, deps)
    if g is None:
        return runtime._field(obj, deps)
    v, t = g
    p = obj.arc_flow
    sig = (v.shape[0], t.shape[0], float(v.sum()), float((v * v).sum()), p.cavity_mode, p.resolution, p.lotbcbp_level, channel_radius(obj))
    hit = runtime.RT.field_cache.get(obj.name)
    if hit and hit[0] == sig:
        return hit[1], v, t
    fld = SDF.PartField(v, t, p.cavity_mode, p.resolution, p.lotbcbp_level)
    runtime.RT.field_cache[obj.name] = (sig, fld)
    return fld, v, t


def estimate_dx(defs, geoms, radii, modes, dens, bodies, max_particles):
    """The solver's automatic spacing for these fluids (so generated points and the lattice agree)."""
    est = 0.0
    for d, g, r, m, de in zip(defs, geoms, radii, modes, dens):
        if d['kind'] == 'GAS':
            continue
        vol = d['volume']
        if vol <= 0 and g is not None:
            vol = volume_estimate(g, m, de, r)
        if vol <= 0 and d['mass'] > 0 and d['density'] > 0:
            vol = d['mass'] / d['density']
        if vol <= 0 and g is None and d.get('body', -1) >= 0:
            lo, hi = bodies[d['body']].bbox_world()
            vol = float(np.prod(hi - lo)) * 0.5 * d['fill']
        est += max(vol, 0.0)
    dx = (est / max(max_particles, 1)) ** (1.0 / 3.0) if est > 0 else 0.01
    return max(dx, 1e-4)
