"""Tessera Projection Cloth Trim, Accents and Seam Seal: PropertyGroups on the Scene (saved in the .blend)."""
import bpy
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty, IntProperty,
                       PointerProperty, StringProperty)
from bpy.types import PropertyGroup

from .core import cloth_trim as CT

TRIM_KIND_ITEMS = [(k, v['label'], v['label']) for k, v in CT.TRIM_KINDS.items()]
TRIM_SOURCE_ITEMS = [
    ('LOWEST', "Lowest boundary loop", "The open boundary loop with the lowest average height (a skirt or jacket hem)"),
    ('HIGHEST', "Highest boundary loop", "The open boundary loop with the highest average height (a collar, waistband or cuff)"),
    ('BOUNDARY', "Boundary loop by index", "Boundary loops sorted from longest to shortest"),
    ('GROUP', "Vertex group", "An edge or vertex chain painted into a vertex group"),
]
ACCENT_KIND_ITEMS = [(k, v['label'], v['label']) for k, v in CT.ACCENT_KINDS.items()]
ACCENT_PLACE_ITEMS = [
    ('CURVE', "Along a vertex chain", "Evenly spaced along the ordered vertices of a vertex group (a placket, a neckline)"),
    ('DENSITY', "Scatter by density", "Random surface points, optionally weighted by a vertex group"),
    ('VERTICES', "On group vertices", "One accent on every vertex of the group"),
    ('CURSOR', "At the 3D cursor", "One accent bound to the closest surface point of the garment"),
]
SEAM_MODE_ITEMS = [(k, v['label'], v['label']) for k, v in CT.SEAM_MODES.items()]


def _mesh_poll(_s, o):
    return o is not None and o.type == 'MESH'


class AF_ClothTrim(PropertyGroup):
    name: StringProperty(name="Name", default="Hem")
    garment: StringProperty(name="Garment", default="", description="Name of the cloth garment (empty uses the active garment)")
    enabled: BoolProperty(name="Enabled", default=True)
    kind: EnumProperty(name="Kind", items=TRIM_KIND_ITEMS, default='HEM',
                       description="Hem, piping, lace or ruffle border, belt or band, collar or cuff band")
    source: EnumProperty(name="Edge", items=TRIM_SOURCE_ITEMS, default='LOWEST')
    loop_index: IntProperty(name="Loop", default=0, min=0)
    group: StringProperty(name="Vertex group", default="Trim")
    width: FloatProperty(name="Width", default=0.025, min=0.0005, unit='LENGTH', description="Visible strip width on the cloth")
    thickness: FloatProperty(name="Thickness", default=0.004, min=0.0001, unit='LENGTH')
    mass_per_m: FloatProperty(name="Mass", default=0.05, min=0.0, description="kg per metre of trim")
    stiffness: FloatProperty(name="Stretch stiffness", default=6.0e4, min=1.0)
    bend: FloatProperty(name="Bend stiffness", default=250.0, min=0.001,
                        description="Resistance of the strip to bending; the rest shape is remembered")
    edge_weight: FloatProperty(name="Edge stiffening", default=2.0, min=0.0, max=20.0,
                               description="Multiplies the stretch and bend stiffness of the cloth edges beside the trim (a stiff hem makes the skirt edge swirl differently)")
    flare: FloatProperty(name="Flare / ruffle", default=0.0, min=0.0, max=2.0,
                         description="Surplus length along the strip: it buckles into ruffles of this relative amplitude")
    flare_count: IntProperty(name="Ruffles", default=0, min=0, max=200, description="Number of ruffles along the edge")
    show_strip: BoolProperty(name="Show strip", default=True, description="Create a strip mesh that follows the trim")
    material: PointerProperty(name="Material", type=bpy.types.Material,
                              description="Material slot of this trim (separate from the garment)")
    strip_obj: PointerProperty(name="Strip object", type=bpy.types.Object)
    n_verts: IntProperty(default=0)


class AF_ClothAccent(PropertyGroup):
    name: StringProperty(name="Name", default="Buttons")
    garment: StringProperty(name="Garment", default="")
    enabled: BoolProperty(name="Enabled", default=True)
    kind: EnumProperty(name="Kind", items=ACCENT_KIND_ITEMS, default='BUTTON')
    source: PointerProperty(name="Object", type=bpy.types.Object, poll=_mesh_poll,
                            description="Object instanced on the cloth (empty uses a built-in shape)")
    placement: EnumProperty(name="Placement", items=ACCENT_PLACE_ITEMS, default='CURVE')
    group: StringProperty(name="Vertex group", default="Accents")
    count: IntProperty(name="Count", default=5, min=1, max=2000)
    density: FloatProperty(name="Density", default=60.0, min=0.0, description="Accents per square metre")
    seed: IntProperty(name="Seed", default=1)
    size: FloatProperty(name="Size", default=0.012, min=0.0005, unit='LENGTH', description="Largest dimension of each accent")
    offset: FloatProperty(name="Lift", default=0.004, unit='LENGTH', description="Distance above the cloth surface along its normal")
    align: BoolProperty(name="Align to surface", default=True)
    dynamic: BoolProperty(name="Hanging (chain / pendant)", default=False,
                          description="Hang from the attachment point on a chain of sub-particles that swings with the cloth")
    chain_segments: IntProperty(name="Links", default=4, min=1, max=16)
    chain_length: FloatProperty(name="Chain length", default=0.12, min=0.005, unit='LENGTH')
    chain_stiffness: FloatProperty(name="Chain stiffness", default=0.9, min=0.05, max=1.0, subtype='FACTOR')
    chain_damping: FloatProperty(name="Chain damping", default=0.04, min=0.0, max=1.0, subtype='FACTOR')
    collide: BoolProperty(name="Collide", default=True, description="Hanging accents collide with the collision bodies and the floor")
    collide_radius: FloatProperty(name="Collision radius", default=0.006, min=0.0, unit='LENGTH')
    show_chain: BoolProperty(name="Draw chain", default=True)
    material: PointerProperty(name="Material", type=bpy.types.Material)
    n_instances: IntProperty(default=0)


class AF_ClothSeam(PropertyGroup):
    name: StringProperty(name="Name", default="Zipper")
    garment: StringProperty(name="Garment", default="")
    enabled: BoolProperty(name="Enabled", default=True)
    mode: EnumProperty(name="Closure", items=SEAM_MODE_ITEMS, default='ZIPPER')
    group: StringProperty(name="Seam group", default="Seam", description="Vertex group holding both lips of the seam")
    group_a: StringProperty(name="Side A", default="", description="Optional vertex group of the first lip")
    group_b: StringProperty(name="Side B", default="", description="Optional vertex group of the second lip")
    progress: FloatProperty(name="Closed", default=0.0, min=0.0, max=1.0, subtype='FACTOR',
                            description="0 open, 1 sealed; zips in order along the seam. Animate it with keyframes")
    strength: FloatProperty(name="Seal strength", default=1.0, min=0.0, max=1.0, subtype='FACTOR',
                            description="1 welds the sealed pairs; lower values give them compliance so they can stretch")
    reverse: BoolProperty(name="Reverse direction", default=False)
    use_tear: BoolProperty(name="Tear", default=False, description="Break sealed pairs when they are pulled further apart than the limit")
    tear_gap: FloatProperty(name="Tear gap", default=0.03, min=0.001, unit='LENGTH')
    gather: FloatProperty(name="Gather", default=0.30, min=0.0, max=0.8, subtype='FACTOR',
                          description="Shortens the seam edges around the zipper head so the cloth bunches into folds")
    ruffle: FloatProperty(name="Fold-over ruffle", default=0.010, min=0.0, max=0.1, unit='LENGTH',
                          description="Alternating push out of the surface around the zipper head")
    pull: PointerProperty(name="Zipper pull", type=bpy.types.Object, description="Empty or object that rides the head, or drives the slider")
    drive_from_pull: BoolProperty(name="Pull drives slider", default=False,
                                  description="Moving the pull along the seam sets the closed value")
    n_pairs: IntProperty(default=0)
    n_torn: IntProperty(default=0)


class AF_ClothTrimScene(PropertyGroup):
    trims: CollectionProperty(type=AF_ClothTrim)
    trim_index: IntProperty(default=0)
    accents: CollectionProperty(type=AF_ClothAccent)
    accent_index: IntProperty(default=0)
    seams: CollectionProperty(type=AF_ClothSeam)
    seam_index: IntProperty(default=0)
    show_trim: BoolProperty(name="Cloth Trim", default=True)
    show_accents: BoolProperty(name="Accents", default=True)
    show_seams: BoolProperty(name="Seam Seal Control Solving", default=True)
    status: StringProperty(default="")


CLASSES = (AF_ClothTrim, AF_ClothAccent, AF_ClothSeam, AF_ClothTrimScene)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Scene.arc_flow_cloth_trim = PointerProperty(type=AF_ClothTrimScene)


def unregister():
    del bpy.types.Scene.arc_flow_cloth_trim
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
