"""Geometry Nodes helpers for Tessera Projection Hair.

`Tessera Hair Render` turns a CURVES groom into a renderable tube mesh
(set radius, curve-to-mesh with a circle profile) and is registered with
the Tessera Add menu via tessera_nodes.register_group.
"""
import bpy

from .nodeutil import Builder, add_socket, fresh_group, inp, out


HAIR_GN_NAME = "Tessera Hair Render"


def build_hair_render(name=HAIR_GN_NAME):
    """Curves in → radius + optional tube mesh out."""
    ng = fresh_group(name)
    add_socket(ng, "Geometry", 'NodeSocketGeometry', 'INPUT')
    add_socket(ng, "Radius", 'NodeSocketFloat', 'INPUT', 0.004, 0.0, desc="Strand radius, metres")
    add_socket(ng, "Sides", 'NodeSocketInt', 'INPUT', 5, 3, 16, desc="Tube profile sides")
    add_socket(ng, "As Tubes", 'NodeSocketBool', 'INPUT', True, desc="On: curve-to-mesh tubes; off: pass the curves through")
    add_socket(ng, "Tip Taper", 'NodeSocketFloat', 'INPUT', 0.8, 0.0, 1.0, desc="0 = constant thickness, 1 = tips taper to a point")
    add_socket(ng, "Geometry", 'NodeSocketGeometry', 'OUTPUT')
    b = Builder(ng)
    gi, go = b.group_io()
    # tapered radius: Radius * (1 - Tip Taper * spline factor^1.3)
    sp = b.node('GeometryNodeSplineParameter', y=-380)
    pw = b.node('ShaderNodeMath', y=-380)
    pw.operation = 'POWER'
    pw.inputs[1].default_value = 1.3
    b.link(out(sp, 'Factor'), pw.inputs[0])
    tp = b.node('ShaderNodeMath', y=-380)
    tp.operation = 'MULTIPLY'
    b.link(out(pw, 'Value'), tp.inputs[0])
    b.link(out(gi, 'Tip Taper'), tp.inputs[1])
    one = b.node('ShaderNodeMath', y=-380)
    one.operation = 'SUBTRACT'
    one.inputs[0].default_value = 1.0
    b.link(out(tp, 'Value'), one.inputs[1])
    rmul = b.node('ShaderNodeMath', y=-380)
    rmul.operation = 'MULTIPLY'
    b.link(out(one, 'Value'), rmul.inputs[0])
    b.link(out(gi, 'Radius'), rmul.inputs[1])
    sr = b.node('GeometryNodeSetCurveRadius', y=0)
    b.link(out(gi, 'Geometry'), inp(sr, 'Curve'))
    b.link(out(rmul, 'Value'), inp(sr, 'Radius'))
    # unit circle profile: Curve to Mesh scales it by the per-point radius
    circ = b.node('GeometryNodeCurvePrimitiveCircle', y=-220)
    try:
        b.link(out(gi, 'Sides'), inp(circ, 'Resolution'))
    except KeyError:
        pass
    try:
        inp(circ, 'Radius').default_value = 1.0
    except Exception:
        pass
    c2m = b.node('GeometryNodeCurveToMesh', y=0)
    b.link(out(sr, 'Curve'), inp(c2m, 'Curve'))
    try:
        b.link(out(circ, 'Curve'), inp(c2m, 'Profile Curve'))
    except KeyError:
        b.link(out(circ, 'Mesh') if False else out(circ, 'Curve'), inp(c2m, 'Profile Curve'))
    try:
        inp(c2m, 'Fill Caps').default_value = True
    except Exception:
        pass
    sw = b.node('GeometryNodeSwitch', y=80)
    # 4.2 uses input_type; 5.x may use a socket
    if hasattr(sw, 'input_type'):
        try:
            sw.input_type = 'GEOMETRY'
        except Exception:
            pass
    b.link(out(gi, 'As Tubes'), inp(sw, 'Switch'))
    b.link(out(sr, 'Curve'), inp(sw, 'False'))
    b.link(out(c2m, 'Mesh'), inp(sw, 'True'))
    b.link(out(sw, 'Output'), inp(go, 'Geometry'))
    return ng


def ensure_hair_gn():
    ng = bpy.data.node_groups.get(HAIR_GN_NAME)
    if ng is None or not ng.nodes:
        ng = build_hair_render()
    return ng


def apply_hair_gn(obj, radius=0.004, sides=5, as_tubes=True, taper=0.8):
    """Put (or refresh) a Tessera Hair Render modifier on a CURVES object."""
    if obj is None or obj.type != 'CURVES':
        return None
    ng = ensure_hair_gn()
    md = obj.modifiers.get("Tessera Hair Render")
    if md is None:
        md = obj.modifiers.new("Tessera Hair Render", 'NODES')
    md.node_group = ng
    # modifier input identifiers are the socket identifiers
    def _set(mod, names, value):
        # resolve the interface identifier from the socket name, then use the
        # version-aware setter (Blender 4.2 mod['Input_2'] ... 5.3 mod.properties.inputs.X.value)
        from .gn import _set_input
        try:
            items = list(ng.interface.items_tree)
        except Exception:
            items = []
        for item in items:
            if getattr(item, 'in_out', '') != 'INPUT':
                continue
            if getattr(item, 'name', '') in names:
                if _set_input(mod, item.identifier, value):
                    return True
        for n in names:          # very old groups keyed by display name
            if _set_input(mod, n, value):
                return True
        return False
    _set(md, ('Radius',), float(radius))
    _set(md, ('Sides',), int(sides))
    _set(md, ('As Tubes', 'As_Tubes'), bool(as_tubes))
    _set(md, ('Tip Taper',), float(taper))
    return md


# name -> (melanin, redness, tint rgb, roughness, radial roughness, coat, random color, base rgb (viewport / EEVEE))
HAIR_PRESETS = {
    'BLACK':      (0.95, 0.10, (1.0, 1.0, 1.0), 0.20, 0.30, 0.10, 0.04, (0.018, 0.015, 0.014)),
    'DARK_BROWN': (0.80, 0.25, (1.0, 0.96, 0.92), 0.26, 0.30, 0.15, 0.08, (0.075, 0.040, 0.025)),
    'CHESTNUT':   (0.62, 0.55, (1.0, 0.92, 0.85), 0.28, 0.32, 0.20, 0.10, (0.09, 0.04, 0.02)),
    'AUBURN':     (0.55, 0.95, (1.0, 0.85, 0.75), 0.30, 0.32, 0.25, 0.12, (0.30, 0.095, 0.035)),
    'BLONDE':     (0.22, 0.30, (1.0, 0.95, 0.85), 0.30, 0.30, 0.30, 0.08, (0.55, 0.38, 0.17)),
    'PLATINUM':   (0.05, 0.10, (1.0, 0.98, 0.95), 0.32, 0.30, 0.30, 0.05, (0.78, 0.70, 0.52)),
    'SILVER':     (0.02, 0.00, (0.92, 0.95, 1.0), 0.34, 0.30, 0.25, 0.05, (0.60, 0.60, 0.62)),
}
HAIR_PRESET_ITEMS = [(k, k.replace('_', ' ').title(), "") for k in HAIR_PRESETS]



def _mixin(node, name):
    """The RGBA socket of a Mix node input (the node also carries float / vector twins with the same names)."""
    return next(sk for sk in node.inputs if sk.name == name and sk.type == 'RGBA') if name != 'Factor' else \
        next(sk for sk in node.inputs if sk.name == 'Factor' and sk.type == 'VALUE')


def _mixout(node):
    return next(sk for sk in node.outputs if sk.type == 'RGBA')


def _sock(node, *names):
    for n in names:
        if n in node.inputs:
            return node.inputs[n]
    return None


def apply_hair_material(obj, color=None, name="Tessera Hair", preset='CHESTNUT', root_dark=0.55, tip_light=0.25,
                        strand_variation=0.12):
    """Realistic hair shader. Cycles: Principled Hair BSDF (Chiang) with melanin, root-to-tip gradient,
    random tone per strand and coat. EEVEE / solid: a Principled BSDF in the same colour (its own output node,
    `target = EEVEE`, since the hair BSDF is Cycles-only). `color` overrides the preset base colour."""
    pr = HAIR_PRESETS.get(str(preset).upper(), HAIR_PRESETS['CHESTNUT'])
    mel, red, tint, rough, rrough, coat, rcol, base = pr
    if color is not None:
        base = tuple(color)[:3]
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    L = nt.links
    info = nt.nodes.new('ShaderNodeHairInfo'); info.location = (-900, 0)
    # root -> tip gradient (Intercept 0 at root): melanin falls toward the tip (sun-bleached ends)
    ramp = nt.nodes.new('ShaderNodeMapRange'); ramp.location = (-640, 160)
    ramp.inputs['To Min'].default_value = 1.0 + root_dark * 0.25
    ramp.inputs['To Max'].default_value = 1.0 - tip_light
    L.new(info.outputs['Intercept'], ramp.inputs['Value'])
    rnd = nt.nodes.new('ShaderNodeMapRange'); rnd.location = (-640, -60)
    rnd.inputs['To Min'].default_value = 1.0 - strand_variation
    rnd.inputs['To Max'].default_value = 1.0 + strand_variation
    L.new(info.outputs['Random'], rnd.inputs['Value'])
    mulm = nt.nodes.new('ShaderNodeMath'); mulm.operation = 'MULTIPLY'; mulm.location = (-400, 100)
    mulm.inputs[0].default_value = mel
    mul2 = nt.nodes.new('ShaderNodeMath'); mul2.operation = 'MULTIPLY'; mul2.location = (-400, 0)
    L.new(ramp.outputs[0], mulm.inputs[1])
    L.new(mulm.outputs[0], mul2.inputs[0])
    L.new(rnd.outputs[0], mul2.inputs[1])
    clampm = nt.nodes.new('ShaderNodeMath'); clampm.operation = 'MINIMUM'; clampm.location = (-220, 40)
    clampm.inputs[1].default_value = 1.0
    L.new(mul2.outputs[0], clampm.inputs[0])
    # Cycles: Principled Hair (Chiang)
    out_c = nt.nodes.new('ShaderNodeOutputMaterial'); out_c.location = (420, 100)
    try:
        out_c.target = 'CYCLES'
    except Exception:
        pass
    try:
        hair = nt.nodes.new('ShaderNodeBsdfHairPrincipled'); hair.location = (160, 100)
        for attr, val in (('model', 'CHIANG'), ('parametrization', 'MELANIN')):
            try:
                setattr(hair, attr, val)
            except Exception:
                pass
        for names, val in ((('Melanin Redness',), red), (('Tint',), tint + (1.0,)), (('Roughness',), rough),
                           (('Radial Roughness',), rrough), (('Coat',), coat), (('IOR',), 1.55),
                           (('Offset',), 0.035), (('Random Color',), rcol), (('Random Roughness',), 0.08)):
            sk = _sock(hair, *names)
            if sk is not None:
                try:
                    sk.default_value = val
                except Exception:
                    pass
        sm = _sock(hair, 'Melanin')
        if sm is not None:
            L.new(clampm.outputs[0], sm)
        L.new(hair.outputs[0], out_c.inputs['Surface'])
    except Exception:
        hair = None
    # EEVEE / fallback: Principled BSDF with the same tone gradient
    out_e = nt.nodes.new('ShaderNodeOutputMaterial'); out_e.location = (420, -240)
    try:
        out_e.target = 'EEVEE'
    except Exception:
        pass
    prin = nt.nodes.new('ShaderNodeBsdfPrincipled'); prin.location = (160, -240)
    mix = nt.nodes.new('ShaderNodeMix'); mix.data_type = 'RGBA'; mix.location = (-120, -300)
    dark = tuple(c * (1.0 - root_dark * 0.45) for c in base) + (1.0,)
    light = tuple(min(1.0, c * (1.0 + tip_light * 0.9) + 0.01) for c in base) + (1.0,)
    _mixin(mix, 'A').default_value = dark
    _mixin(mix, 'B').default_value = light
    L.new(info.outputs['Intercept'], _mixin(mix, 'Factor'))
    mixr = nt.nodes.new('ShaderNodeMix'); mixr.data_type = 'RGBA'; mixr.blend_type = 'MULTIPLY'; mixr.location = (40, -300)
    _mixin(mixr, 'Factor').default_value = 1.0
    L.new(_mixout(mix), _mixin(mixr, 'A'))
    L.new(rnd.outputs[0], _mixin(mixr, 'B'))
    L.new(_mixout(mixr), prin.inputs['Base Color'])
    for n, v in (('Roughness', 0.45), ('Metallic', 0.0), ('Coat Weight', 0.0), ('Specular IOR Level', 0.35)):
        if n in prin.inputs:
            prin.inputs[n].default_value = v
    L.new(prin.outputs[0], out_e.inputs['Surface'])
    if hair is None:
        out_e.target = 'ALL'
    if obj is not None and getattr(obj, 'data', None) is not None and hasattr(obj.data, 'materials'):
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)
    return mat


def make_scalp_shell(surface, hair, preset='CHESTNUT', offset=0.0015, name=None, reach=0.02):
    """A thin scalp skin under the roots: the surface faces nearest the hair roots, pushed out a hair-width, shaded dark
    like the root colour with a soft noise, so no bald skin shows between the strands."""
    import bmesh
    import numpy as np
    from mathutils import Vector
    if surface is None or surface.type != 'MESH':
        return None
    me = surface.data
    cu = hair.data
    n = len(cu.points)
    pos = np.empty(n * 3, np.float32)
    cu.points.foreach_get('position', pos)
    pos = pos.reshape(-1, 3)
    off = np.empty(len(cu.curve_offset_data), np.int32)
    cu.curve_offset_data.foreach_get('value', off)
    Mh = np.array(hair.matrix_world)
    roots_w = pos[off[:-1]] @ Mh[:3, :3].T + Mh[:3, 3]
    Ms = np.array(surface.matrix_world)
    Msi = np.linalg.inv(Ms)
    roots_l = roots_w @ Msi[:3, :3].T + Msi[:3, 3]
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    cent = np.array([f.calc_center_median()[:] for f in bm.faces]).reshape(-1, 3)
    keep = np.zeros(len(cent), bool)
    for i0 in range(0, len(roots_l), 256):
        r = roots_l[i0:i0 + 256]
        d = np.min(np.linalg.norm(cent[:, None, :] - r[None], axis=2), axis=1) if len(cent) * len(r) < 4e6 else None
        if d is None:
            d = np.array([np.min(np.linalg.norm(r - c, axis=1)) for c in cent])
        keep |= d < reach
    bmesh.ops.delete(bm, geom=[f for f, k in zip(bm.faces, keep) if not k], context='FACES')
    bm.verts.ensure_lookup_table()
    for v in bm.verts:
        v.co += v.normal * offset
    new = bpy.data.meshes.new((name or surface.name) + " Scalp")
    bm.to_mesh(new)
    bm.free()
    ob = bpy.data.objects.new((name or surface.name) + " Scalp", new)
    ob.matrix_world = surface.matrix_world
    if surface.parent is not None:
        ob.parent = surface.parent
        ob.parent_type = surface.parent_type
        ob.parent_bone = surface.parent_bone
        ob.matrix_parent_inverse = surface.matrix_parent_inverse
    for c in surface.users_collection:
        c.objects.link(ob)
    pr = HAIR_PRESETS.get(str(preset).upper(), HAIR_PRESETS['CHESTNUT'])
    mat = bpy.data.materials.new("Tessera Scalp")
    mat.use_nodes = True
    bsdf = next(nd for nd in mat.node_tree.nodes if nd.type == 'BSDF_PRINCIPLED')
    bsdf.inputs['Base Color'].default_value = tuple(c * 0.35 for c in pr[7]) + (1.0,)
    bsdf.inputs['Roughness'].default_value = 0.6
    ob.data.materials.append(mat)
    return ob


def register():
    try:
        from . import tessera_nodes
        tessera_nodes.register_group(
            HAIR_GN_NAME, build_hair_render,
            description="Set hair-curve radius and (optionally) convert the groom to renderable tubes",
        )
    except Exception:
        pass


def unregister():
    pass
