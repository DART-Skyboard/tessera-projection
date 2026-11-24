"""Arc Flow - LOTBCBP fluid simulation through connected cavities (Blender 5.2.3+ extension).

LOTBCBP = Level Of Tessellation Boolean Collision Body Projection: collision bodies are resolved by a base signed-distance
field plus virtual tessellation bricks that are generated on demand where fluid touches a surface, so the
interaction resolution can exceed both the modelled mesh and the fluid's own particle spacing."""
from . import ops, panels, props, runtime
from . import retopo_ops, retopo_panel  # Quad Remesh (LOTBCBP)
from . import systems_ops  # Systems, crossfeed demos (Tessera Projection)
from . import fluidmode_ops, fluidmode_panel, fluidmode_props  # Fluid mode (Tessera Projection)
from . import thermal, windtunnel  # thermal readback, wind tunnel builder
from . import assets_demo, spawn_gn  # three independent assets example, spawn controls as Geometry Nodes
from . import tessera_nodes  # reusable Tessera node groups, Add menu entry
from . import surfaces_ops, surfaces_panel, surfaces_props  # surface sources, fluid volumes, Examples panel
from . import shader_ops  # realistic shader presets (water, glass, lava, smoke ...)
from . import hair_ops, hair_panel, hair_props, hair_gn  # Tessera Projection Hair (Empty Hair Curves)
from . import ui_tabs  # vertical N-panel tab rail (Remesh / Fluid / Combustion / Hair / Cloth / Examples)


def register():
    surfaces_panel.prepare()      # everything but the Examples panel opens expanded
    props.register()
    ops.register()
    panels.register()
    runtime.register()
    retopo_ops.register(); retopo_panel.register()
    systems_ops.register()
    fluidmode_props.register(); fluidmode_ops.register(); fluidmode_panel.register()
    surfaces_props.register(); surfaces_ops.register(); surfaces_panel.register()
    tessera_nodes.register()
    thermal.register(); windtunnel.register(); assets_demo.register(); spawn_gn.register()
    shader_ops.register()
    hair_props.register(); hair_ops.register(); hair_panel.register(); hair_gn.register()
    ui_tabs.register()  # last so hosted panel polls can be wrapped after every category is registered


def unregister():
    ui_tabs.unregister()
    hair_gn.unregister(); hair_panel.unregister(); hair_ops.unregister(); hair_props.unregister()
    shader_ops.unregister()
    windtunnel.unregister(); thermal.unregister()
    tessera_nodes.unregister()
    surfaces_panel.unregister(); surfaces_ops.unregister(); surfaces_props.unregister()
    fluidmode_panel.unregister(); fluidmode_ops.unregister(); fluidmode_props.unregister()
    systems_ops.unregister()
    retopo_panel.unregister(); retopo_ops.unregister()
    runtime.unregister()
    panels.unregister()
    ops.unregister()
    props.unregister()
