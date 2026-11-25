"""Geometry Nodes group "Tessera Muscles" for the Tessera muscle solver.

Inputs   Geometry, Flex (global multiplier), Accent (slide multiplier), Region (-1 = every region, else a region id)
Outputs  Geometry (skin displaced in rest space), plus three fields that stock nodes can use:
         Offset (vector), Region ID (integer) and Flex Weight (float).

The group reads the point attributes the solver writes on the skin (tp_muscle_bulge, tp_muscle_slide, tp_muscle_id,
tp_muscle_flex).  Put the modifier BEFORE the Armature modifier: the offset is a rest-space displacement and the
skeleton then carries it along with the skin weights (the "Create Geometry Nodes" button does this ordering)."""
import bpy

from .nodeutil import Builder, add_socket, fresh_group, inp, out

GROUP_NAME = "Tessera Muscles"
MOD_NAME = "Tessera Muscles"


def build_muscles(name=GROUP_NAME):
    ng = fresh_group(name)
    add_socket(ng, "Geometry", 'NodeSocketGeometry')
    add_socket(ng, "Flex", 'NodeSocketFloat', default=1.0, lo=0.0, hi=4.0, desc="Global flex multiplier for every muscle")
    add_socket(ng, "Accent", 'NodeSocketFloat', default=1.0, lo=0.0, hi=4.0, desc="Multiplier for the slide toward the joint end")
    add_socket(ng, "Region", 'NodeSocketInt', default=-1, lo=-1, hi=100000,
               desc="-1 = all regions; otherwise only the region with this id (tp_muscle_id) moves")
    add_socket(ng, "Geometry", 'NodeSocketGeometry', io='OUTPUT')
    add_socket(ng, "Offset", 'NodeSocketVector', io='OUTPUT', desc="Per-point displacement the group applies")
    add_socket(ng, "Region ID", 'NodeSocketInt', io='OUTPUT', desc="Region id of each skin point (0 = unpainted)")
    add_socket(ng, "Flex Weight", 'NodeSocketFloat', io='OUTPUT', desc="Strongest muscle weight at each point")
    b = Builder(ng)
    gi = b.node('NodeGroupInput', x=-700)
    go = b.node('NodeGroupOutput', x=1000)

    def named(attr, dtype, y):
        n = b.node('GeometryNodeInputNamedAttribute', data_type=dtype, x=-480, y=y)
        inp(n, 'Name').default_value = attr
        return out(n, 'Attribute')

    bulge = named("tp_muscle_bulge", 'FLOAT_VECTOR', 200)
    slide = named("tp_muscle_slide", 'FLOAT_VECTOR', 60)
    rid = named("tp_muscle_id", 'INT', -80)
    wgt = named("tp_muscle_flex", 'FLOAT', -220)

    sc_slide = b.node('ShaderNodeVectorMath', operation='SCALE', x=-260, y=60)
    b.link(slide, sc_slide.inputs[0])
    b.link(out(gi, 'Accent'), sc_slide.inputs['Scale'])
    add = b.node('ShaderNodeVectorMath', operation='ADD', x=-60, y=120)
    b.link(bulge, add.inputs[0])
    b.link(out(sc_slide, 'Vector'), add.inputs[1])
    sc_flex = b.node('ShaderNodeVectorMath', operation='SCALE', x=140, y=120)
    b.link(out(add, 'Vector'), sc_flex.inputs[0])
    b.link(out(gi, 'Flex'), sc_flex.inputs['Scale'])

    all_ = b.node('FunctionNodeCompare', data_type='INT', operation='LESS_THAN', x=-260, y=-120)
    b.link(out(gi, 'Region'), inp(all_, 'A'))
    inp(all_, 'B').default_value = 0
    eq = b.node('FunctionNodeCompare', data_type='INT', operation='EQUAL', x=-260, y=-260)
    b.link(rid, inp(eq, 'A'))
    b.link(out(gi, 'Region'), inp(eq, 'B'))
    sel = b.node('FunctionNodeBooleanMath', operation='OR', x=-40, y=-180)
    b.link(out(all_, 'Result'), sel.inputs[0])
    b.link(out(eq, 'Result'), sel.inputs[1])

    sp = b.node('GeometryNodeSetPosition', x=500, y=0)
    b.link(out(gi, 'Geometry'), inp(sp, 'Geometry'))
    b.link(out(sel, 'Boolean'), inp(sp, 'Selection'))
    b.link(out(sc_flex, 'Vector'), inp(sp, 'Offset'))
    b.link(out(sp, 'Geometry'), inp(go, 'Geometry'))
    b.link(out(sc_flex, 'Vector'), inp(go, 'Offset'))
    b.link(rid, inp(go, 'Region ID'))
    b.link(wgt, inp(go, 'Flex Weight'))
    return ng


def ensure_group(name=GROUP_NAME):
    ng = bpy.data.node_groups.get(name)
    if ng is not None and ng.nodes.get("Group Output") is not None and "Flex" in [s.name for s in ng.interface.items_tree]:
        return ng
    return build_muscles(name)


def add_modifier(obj):
    """Attach the group as a modifier ahead of any Armature modifier.  Returns the modifier."""
    ng = ensure_group()
    mod = obj.modifiers.get(MOD_NAME)
    if mod is None:
        mod = obj.modifiers.new(MOD_NAME, 'NODES')
    mod.node_group = ng
    # rest-space offset must come before the skeleton deformation
    arm_idx = [i for i, m in enumerate(obj.modifiers) if m.type == 'ARMATURE']
    cur = list(obj.modifiers).index(mod)
    if arm_idx and cur > arm_idx[0]:
        try:
            with bpy.context.temp_override(object=obj):
                bpy.ops.object.modifier_move_to_index(modifier=mod.name, index=arm_idx[0])
        except Exception:
            try:
                obj.modifiers.move(cur, arm_idx[0])
            except Exception:
                pass
    return mod


def register():
    try:
        from . import tessera_nodes
        tessera_nodes.register_group(GROUP_NAME, build_muscles,
                                     description="Muscle solver offsets as a rest-space displacement with Flex, Accent and Region inputs")
    except Exception:
        pass


def unregister():
    pass
