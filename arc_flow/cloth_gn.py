"""Geometry Nodes helpers and Cycles look for the Tessera cloth solver.

"Tessera Cloth Tension"  Stores per-vertex tension and speed as named attributes
                         (af_cloth_tension, af_cloth_speed) and optionally tints the mesh.
The groups are built once; delete a group to rebuild it.
"""
import bpy

from . import gn
from .nodeutil import Builder, add_socket, fresh_group, inp, out

GROUP_TENSION = "Tessera Cloth Tension"
MAT_FABRIC = "Tessera Cloth Fabric"
MAT_SKIN = "Tessera Cloth Skin"
MAT_STUDIO = "Tessera Cloth Studio Floor"
MAT_BACKDROP = "Tessera Cloth Backdrop"


def tension_group(name=GROUP_TENSION):
    """Passthrough group that exposes tension / speed as named attributes for viewport colour."""
    existing = bpy.data.node_groups.get(name)
    if existing is not None and existing.nodes.get("Group Output") is not None:
        return existing
    ng = fresh_group(name)
    add_socket(ng, "Geometry", 'NodeSocketGeometry')
    add_socket(ng, "Tint", 'NodeSocketBool', default=True, desc="Colour the surface by tension")
    add_socket(ng, "Geometry", 'NodeSocketGeometry', io='OUTPUT')
    b = Builder(ng)
    gi = b.node('NodeGroupInput', x=0, y=0)
    go = b.node('NodeGroupOutput', x=900, y=0)
    taut = b.node('GeometryNodeInputNamedAttribute', data_type='FLOAT', x=220, y=80)
    inp(taut, 'Name').default_value = "af_cloth_tension"
    # a cheap 0-1 ramp into a colour for viewport feedback
    col = b.node('ShaderNodeValToRGB', x=440, y=80)
    cr = col.color_ramp
    cr.elements[0].position = 0.0
    cr.elements[0].color = (0.15, 0.35, 0.85, 1)
    if len(cr.elements) < 2:
        cr.elements.new(1.0)
    cr.elements[-1].position = 1.0
    cr.elements[-1].color = (0.95, 0.15, 0.10, 1)
    try:
        b.link(out(taut, 'Attribute'), col.inputs[0])
    except Exception:
        pass
    store = b.node('GeometryNodeStoreNamedAttribute', data_type='FLOAT_COLOR', domain='POINT', x=660, y=0)
    inp(store, 'Name').default_value = "af_cloth_tint"
    b.link(out(gi, 'Geometry'), inp(store, 'Geometry'))
    try:
        b.link(out(col, 'Color'), inp(store, 'Value'))
    except Exception:
        pass
    b.link(out(store, 'Geometry'), inp(go, 'Geometry'))
    return ng


def ensure_modifier(obj):
    """Attach the tension visualiser to a garment if it is not already there."""
    if obj is None or obj.type != 'MESH':
        return None
    tension_group()
    mod = obj.modifiers.get("Tessera Cloth Tension")
    if mod is None:
        mod = obj.modifiers.new("Tessera Cloth Tension", 'NODES')
    ng = bpy.data.node_groups.get(GROUP_TENSION)
    if ng is not None:
        mod.node_group = ng
    return mod


def _fresh_mat(name):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    return mat, nt


def fabric_material(name=MAT_FABRIC, tint=(0.22, 0.05, 0.08), sheen=0.85):
    """Publication-grade woven cloth: Principled sheen, silk-like specular, weave bump."""
    mat, nt = _fresh_mat(name)
    b = Builder(nt)
    outn = b.node('ShaderNodeOutputMaterial', x=1400, y=0)
    bsdf = b.node('ShaderNodeBsdfPrincipled', x=1050, y=0)
    # weave coordinates
    tex = b.node('ShaderNodeTexCoord', x=0, y=0)
    mapn = b.node('ShaderNodeMapping', x=200, y=0)
    mapn.inputs['Scale'].default_value = (18.0, 28.0, 18.0)
    b.link(out(tex, 'UV') if any(s.name == 'UV' for s in tex.outputs) else out(tex, 'Generated'), inp(mapn, 'Vector'))
    waves_u = b.node('ShaderNodeTexWave', x=420, y=120, wave_type='BANDS', bands_direction='X', wave_profile='SIN')
    waves_v = b.node('ShaderNodeTexWave', x=420, y=-80, wave_type='BANDS', bands_direction='Y', wave_profile='SIN')
    waves_u.inputs['Scale'].default_value = 8.0
    waves_v.inputs['Scale'].default_value = 11.0
    b.link(out(mapn, 'Vector'), inp(waves_u, 'Vector'))
    b.link(out(mapn, 'Vector'), inp(waves_v, 'Vector'))
    mul = b.node('ShaderNodeMixRGB', x=640, y=20, blend_type='MULTIPLY')
    mul.inputs['Fac'].default_value = 0.65
    b.link(out(waves_u, 'Color'), mul.inputs['Color1'])
    b.link(out(waves_v, 'Color'), mul.inputs['Color2'])
    # subtle colour variation
    noise = b.node('ShaderNodeTexNoise', x=420, y=-280)
    noise.inputs['Scale'].default_value = 3.5
    noise.inputs['Detail'].default_value = 6.0
    b.link(out(mapn, 'Vector'), inp(noise, 'Vector'))
    col = b.node('ShaderNodeMixRGB', x=800, y=80, blend_type='MIX')
    col.inputs['Fac'].default_value = 0.12
    col.inputs['Color1'].default_value = (tint[0], tint[1], tint[2], 1.0)
    col.inputs['Color2'].default_value = (min(1, tint[0] * 1.35), min(1, tint[1] * 1.2), min(1, tint[2] * 1.15), 1.0)
    b.link(out(noise, 'Fac'), col.inputs['Fac'])
    bump = b.node('ShaderNodeBump', x=820, y=-80)
    bump.inputs['Strength'].default_value = 0.18
    bump.inputs['Distance'].default_value = 0.003
    b.link(out(mul, 'Color'), inp(bump, 'Height'))
    # principled
    def _set(node, name, value):
        if name in node.inputs:
            node.inputs[name].default_value = value
    b.link(out(col, 'Color'), inp(bsdf, 'Base Color'))
    _set(bsdf, 'Roughness', 0.58)
    _set(bsdf, 'Specular', 0.45)
    _set(bsdf, 'Specular IOR Level', 0.45)
    _set(bsdf, 'Metallic', 0.0)
    _set(bsdf, 'Sheen', min(sheen, 0.45))
    _set(bsdf, 'Sheen Weight', min(sheen, 0.45))
    _set(bsdf, 'Sheen Roughness', 0.28)
    _set(bsdf, 'Sheen Tint', (0.85, 0.25, 0.30, 1.0) if 'Sheen Tint' in bsdf.inputs and getattr(bsdf.inputs['Sheen Tint'], 'type', '') != 'VALUE' else 0.7)
    try:
        if bsdf.inputs.get('Sheen Tint') and bsdf.inputs['Sheen Tint'].type == 'VALUE':
            bsdf.inputs['Sheen Tint'].default_value = 0.7
    except Exception:
        pass
    _set(bsdf, 'Coat Weight', 0.04)
    _set(bsdf, 'Coat Roughness', 0.2)
    _set(bsdf, 'IOR', 1.46)
    _set(bsdf, 'Alpha', 1.0)
    try:
        b.link(out(bump, 'Normal'), inp(bsdf, 'Normal'))
    except Exception:
        pass
    b.link(out(bsdf, 'BSDF'), inp(outn, 'Surface'))
    mat.diffuse_color = (tint[0], tint[1], tint[2], 1.0)
    mat.use_backface_culling = False
    try:
        mat.shadow_method = 'OPAQUE'
    except Exception:
        pass
    return mat


def skin_material(name=MAT_SKIN, tint=(0.62, 0.42, 0.32)):
    """Warm skin with subsurface scattering and a soft pore bump."""
    mat, nt = _fresh_mat(name)
    b = Builder(nt)
    outn = b.node('ShaderNodeOutputMaterial', x=1200, y=0)
    bsdf = b.node('ShaderNodeBsdfPrincipled', x=860, y=0)
    tex = b.node('ShaderNodeTexCoord', x=0, y=0)
    noise = b.node('ShaderNodeTexNoise', x=280, y=-80)
    noise.inputs['Scale'].default_value = 180.0
    noise.inputs['Detail'].default_value = 8.0
    noise.inputs['Roughness'].default_value = 0.55
    try:
        b.link(out(tex, 'Object'), inp(noise, 'Vector'))
    except Exception:
        b.link(out(tex, 'Generated'), inp(noise, 'Vector'))
    bump = b.node('ShaderNodeBump', x=540, y=-80)
    bump.inputs['Strength'].default_value = 0.06
    bump.inputs['Distance'].default_value = 0.0015
    b.link(out(noise, 'Fac'), inp(bump, 'Height'))
    def _set(node, nm, value):
        if nm in node.inputs:
            try:
                node.inputs[nm].default_value = value
            except Exception:
                pass
    _set(bsdf, 'Base Color', (tint[0], tint[1], tint[2], 1.0))
    _set(bsdf, 'Roughness', 0.58)
    _set(bsdf, 'Specular', 0.35)
    _set(bsdf, 'Specular IOR Level', 0.35)
    _set(bsdf, 'Subsurface Weight', 0.08)
    _set(bsdf, 'Subsurface', 0.08)
    # Blender 4.2 uses Subsurface Radius + Subsurface Scale / Weight
    if 'Subsurface Radius' in bsdf.inputs:
        try:
            bsdf.inputs['Subsurface Radius'].default_value = (0.8, 0.25, 0.15)
        except Exception:
            pass
    _set(bsdf, 'Subsurface Scale', 0.012)
    if 'Subsurface Color' in bsdf.inputs:
        try:
            bsdf.inputs['Subsurface Color'].default_value = (0.85, 0.35, 0.28, 1.0)
        except Exception:
            pass
    _set(bsdf, 'Sheen Weight', 0.08)
    _set(bsdf, 'Sheen', 0.08)
    try:
        b.link(out(bump, 'Normal'), inp(bsdf, 'Normal'))
    except Exception:
        pass
    b.link(out(bsdf, 'BSDF'), inp(outn, 'Surface'))
    mat.diffuse_color = (tint[0], tint[1], tint[2], 1.0)
    return mat


def floor_material(name=MAT_STUDIO):
    mat, nt = _fresh_mat(name)
    b = Builder(nt)
    outn = b.node('ShaderNodeOutputMaterial', x=700, y=0)
    bsdf = b.node('ShaderNodeBsdfPrincipled', x=420, y=0)
    # dark polished concrete
    tex = b.node('ShaderNodeTexCoord', x=0, y=0)
    noise = b.node('ShaderNodeTexNoise', x=200, y=-40)
    noise.inputs['Scale'].default_value = 18.0
    noise.inputs['Detail'].default_value = 6.0
    try:
        b.link(out(tex, 'Object'), inp(noise, 'Vector'))
    except Exception:
        pass
    mix = b.node('ShaderNodeMixRGB', x=200, y=120, blend_type='MIX')
    mix.inputs['Fac'].default_value = 0.08
    mix.inputs['Color1'].default_value = (0.095, 0.090, 0.096, 1)
    mix.inputs['Color2'].default_value = (0.130, 0.124, 0.128, 1)
    b.link(out(noise, 'Fac'), mix.inputs['Fac'])
    b.link(out(mix, 'Color'), inp(bsdf, 'Base Color'))
    if 'Roughness' in bsdf.inputs:
        bsdf.inputs['Roughness'].default_value = 0.32
    if 'Specular IOR Level' in bsdf.inputs:
        bsdf.inputs['Specular IOR Level'].default_value = 0.4
    elif 'Specular' in bsdf.inputs:
        bsdf.inputs['Specular'].default_value = 0.4
    b.link(out(bsdf, 'BSDF'), inp(outn, 'Surface'))
    mat.diffuse_color = (0.03, 0.03, 0.035, 1.0)
    return mat


def backdrop_material(name=MAT_BACKDROP):
    mat, nt = _fresh_mat(name)
    b = Builder(nt)
    outn = b.node('ShaderNodeOutputMaterial', x=700, y=0)
    emit = b.node('ShaderNodeEmission', x=420, y=0)
    emit.inputs['Color'].default_value = (0.10, 0.085, 0.09, 1)
    emit.inputs['Strength'].default_value = 0.6
    b.link(out(emit, 'Emission'), inp(outn, 'Surface'))
    mat.diffuse_color = (0.10, 0.085, 0.09, 1.0)
    try:
        mat.shadow_method = 'NONE'
    except Exception:
        pass
    return mat


def assign(obj, mat):
    if obj is None or mat is None or obj.type not in {'MESH', 'CURVE', 'SURFACE'}:
        return
    if not obj.data.materials:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat


def write_attributes(obj, tension=None, speed=None):
    """Write per-vertex float attributes used by the tension visualiser."""
    if obj is None or obj.type != 'MESH':
        return
    me = obj.data
    n = len(me.vertices)
    def _write(name, values, domain='POINT'):
        if values is None:
            return
        arr = list(values) if not hasattr(values, 'tolist') else values
        if len(arr) != n:
            return
        attr = me.attributes.get(name)
        if attr is None or attr.domain != domain:
            if attr is not None:
                me.attributes.remove(attr)
            attr = me.attributes.new(name, 'FLOAT', domain)
        try:
            import numpy as np
            buf = np.asarray(arr, np.float32).ravel()
            attr.data.foreach_set('value', buf)
        except Exception:
            for i, v in enumerate(arr):
                attr.data[i].value = float(v)
    _write('af_cloth_tension', tension)
    _write('af_cloth_speed', speed)
