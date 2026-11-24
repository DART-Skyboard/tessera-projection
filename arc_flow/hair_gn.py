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
    add_socket(ng, "Geometry", 'NodeSocketGeometry', 'OUTPUT')
    b = Builder(ng)
    gi, go = b.group_io()
    # set curve radius
    sr = b.node('GeometryNodeSetCurveRadius', y=0)
    b.link(out(gi, 'Geometry'), inp(sr, 'Curve'))
    b.link(out(gi, 'Radius'), inp(sr, 'Radius'))
    # circle profile
    circ = b.node('GeometryNodeCurvePrimitiveCircle', y=-220)
    # socket names differ slightly across 4.x; set whichever exist
    for sname, src in (('Resolution', 'Sides'), ('Radius', None)):
        try:
            sock = inp(circ, sname)
        except KeyError:
            continue
        if src is None:
            sock.default_value = 1.0
        else:
            b.link(out(gi, src), sock)
    # a unit circle is then scaled by the curve radius at each point, but Curve to Mesh
    # uses the profile as-is; drive the circle radius from the Radius input too
    try:
        b.link(out(gi, 'Radius'), inp(circ, 'Radius'))
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


def apply_hair_gn(obj, radius=0.004, sides=5, as_tubes=True):
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
        keys = getattr(mod, 'keys', lambda: [])()
        for n in names:
            for k in list(keys) + [n]:
                if n.lower() in str(k).lower() or k == n:
                    try:
                        mod[k] = value
                        return
                    except Exception:
                        pass
        # Blender 4.2: ['Input_2'] style — walk the interface
        try:
            items = list(ng.interface.items_tree)
        except Exception:
            items = []
        for item in items:
            if getattr(item, 'in_out', '') != 'INPUT':
                continue
            if getattr(item, 'name', '') in names:
                ident = item.identifier
                try:
                    mod[ident] = value
                    return
                except Exception:
                    pass
    _set(md, ('Radius',), float(radius))
    _set(md, ('Sides',), int(sides))
    _set(md, ('As Tubes', 'As_Tubes'), bool(as_tubes))
    return md


def apply_hair_material(obj, color=(0.12, 0.07, 0.04, 1.0), name="Tessera Hair"):
    """A Principled Hair BSDF (Cycles) with a Principled BSDF fallback for EEVEE."""
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    out.location = (360, 0)
    # Cycles hair
    try:
        hair = nt.nodes.new('ShaderNodeBsdfHairPrincipled')
        hair.location = (0, 80)
        # colour socket names vary: 'Color', 'Melanin', ...
        if 'Color' in hair.inputs:
            hair.inputs['Color'].default_value = color
        elif 'Melanin' in hair.inputs:
            hair.inputs['Melanin'].default_value = 0.7
        if 'Roughness' in hair.inputs:
            hair.inputs['Roughness'].default_value = 0.3
        nt.links.new(hair.outputs[0], out.inputs['Surface'])
    except Exception:
        hair = None
    prin = nt.nodes.new('ShaderNodeBsdfPrincipled')
    prin.location = (0, -220)
    prin.inputs['Base Color'].default_value = color
    if 'Roughness' in prin.inputs:
        prin.inputs['Roughness'].default_value = 0.35
    if hair is None:
        nt.links.new(prin.outputs[0], out.inputs['Surface'])
    # keep the principled node for EEVEE: a mix isn't necessary headless, Cycles uses Hair
    if obj is not None:
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)
    return mat


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
