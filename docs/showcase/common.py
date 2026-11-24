"""Shared helpers for the 0.6.0 full-Blender-UI showcase scenes (docs/showcase/*).

Every scene drives the *installed* Tessera Projection extension
(bl_ext.user_default.tessera_projection) when it is present, otherwise the
add-on from this source tree, so the N-panel in the captures is the real one.
"""
import ast
import importlib
import math
import os
import sys

import bpy
import bmesh
import numpy as np
from mathutils import Euler, Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))


def addon():
    """Return the Tessera Projection package module (installed extension first)."""
    for name in ('bl_ext.user_default.tessera_projection',):
        try:
            mod = importlib.import_module(name)
            if hasattr(bpy.types.Scene, 'arc_flow'):
                return mod
        except Exception:
            pass
    sys.path.insert(0, ROOT)
    import arc_flow
    if not hasattr(bpy.types.Scene, 'arc_flow'):
        arc_flow.register()
    return arc_flow


def sub(name):
    return importlib.import_module(addon().__name__ + '.' + name)


def clear_scene():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for c in list(bpy.data.collections):
        bpy.data.collections.remove(c)


def borrow(path, names):
    """Load top-level functions from another docs script without running it."""
    src = open(path).read()
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    mod = ast.Module(body=keep, type_ignores=[])
    g = dict(bpy=bpy, bmesh=bmesh, np=np, math=math, Vector=Vector, Matrix=Matrix, Euler=Euler,
             sc=bpy.context.scene)
    exec(compile(mod, path, 'exec'), g)
    return g


def mesh_world(ob):
    deps = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(deps)
    me = ev.to_mesh()
    try:
        me.calc_loop_triangles()
        nv, nt = len(me.vertices), len(me.loop_triangles)
        v = np.empty(nv * 3, np.float64); me.vertices.foreach_get('co', v)
        t = np.empty(nt * 3, np.int64); me.loop_triangles.foreach_get('vertices', t)
        M = np.array(ob.matrix_world, dtype=np.float64)
        return v.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3], t.reshape(-1, 3)
    finally:
        ev.to_mesh_clear()


def area_light(name, loc, target, size, energy, color=(1, 1, 1), spread=None, shape='SQUARE', size_y=None):
    lamp = bpy.data.lights.new(name, 'AREA')
    lamp.energy = energy
    lamp.size = size
    lamp.shape = shape if size_y is None else 'RECTANGLE'
    if size_y is not None:
        lamp.size_y = size_y
    lamp.color = color
    if spread is not None and hasattr(lamp, 'spread'):
        lamp.spread = math.radians(spread)
    ob = bpy.data.objects.new(name, lamp)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = loc
    ob.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
    return ob


def spot_light(name, loc, target, energy, color=(1, 1, 1), size_deg=40, blend=0.35, radius=0.05):
    lamp = bpy.data.lights.new(name, 'SPOT')
    lamp.energy = energy
    lamp.color = color
    lamp.spot_size = math.radians(size_deg)
    lamp.spot_blend = blend
    lamp.shadow_soft_size = radius
    ob = bpy.data.objects.new(name, lamp)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = loc
    ob.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
    return ob


def look_camera(name, loc, target, lens=50.0):
    cam = bpy.data.cameras.new(name)
    cam.lens = lens
    ob = bpy.data.objects.new(name, cam)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = loc
    ob.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
    bpy.context.scene.camera = ob
    return ob


def cycles_setup(samples=24, preview_samples=16, denoise=True):
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    cy = sc.cycles
    cy.device = 'CPU'
    cy.samples = samples
    cy.preview_samples = preview_samples
    cy.use_adaptive_sampling = True
    cy.use_denoising = denoise
    cy.use_preview_denoising = denoise
    for attr, val in (('preview_denoiser', 'OPENIMAGEDENOISE'), ('denoiser', 'OPENIMAGEDENOISE'),
                      ('preview_denoising_start_sample', 1), ('preview_denoising_input_passes', 'RGB_ALBEDO_NORMAL'),
                      ('max_bounces', 6), ('diffuse_bounces', 3), ('glossy_bounces', 3),
                      ('transmission_bounces', 6), ('transparent_max_bounces', 8)):
        try:
            setattr(cy, attr, val)
        except Exception:
            pass
    try:
        sc.view_settings.view_transform = 'AgX'
        sc.view_settings.look = 'AgX - Medium High Contrast'
    except Exception:
        try:
            sc.view_settings.look = 'None'
        except Exception:
            pass
    sc.render.film_transparent = False


def principled(name, base, rough=0.5, metallic=0.0, **inputs):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    p = m.node_tree.nodes.get('Principled BSDF')
    p.inputs['Base Color'].default_value = (*base[:3], 1.0)
    p.inputs['Roughness'].default_value = rough
    p.inputs['Metallic'].default_value = metallic
    for k, v in inputs.items():
        key = k.replace('_', ' ')
        if key in p.inputs:
            p.inputs[key].default_value = v
    return m
