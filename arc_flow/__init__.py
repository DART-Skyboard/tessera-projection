"""Arc Flow - LOTBCBP fluid simulation through connected cavities (Blender 5.2.3+ extension).

LOTBCBP = Level Of Tessellation Boolean Collision Body Projection: collision bodies are resolved by a base signed-distance
field plus virtual tessellation bricks that are generated on demand where fluid touches a surface, so the
interaction resolution can exceed both the modelled mesh and the fluid's own particle spacing."""
from . import ops, panels, props, runtime
from . import retopo_ops, retopo_panel  # Quad Remesh (LOTBCBP)
from . import systems_ops  # Systems, crossfeed demos (Tessera Projection)
from . import fluidmode_ops, fluidmode_panel, fluidmode_props  # Fluid mode (Tessera Projection)
from . import thermal, windtunnel  # thermal readback, wind tunnel builder
from . import tessera_nodes  # reusable Tessera node groups, Add menu entry
from . import surfaces_ops, surfaces_panel, surfaces_props  # surface sources, fluid volumes, Examples panel
from . import shader_ops  # realistic shader presets (water, glass, lava, smoke ...)


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
    thermal.register(); windtunnel.register()
    shader_ops.register()


def unregister():
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
