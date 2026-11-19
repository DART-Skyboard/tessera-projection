# Changelog

## 0.3.1
- Fluid mode: Fill / Settle phase. Container fluids are seeded from the cavity's real shape through the collision-body projection (lowest point first along gravity), poured and settled dynamically under the Environment gravity (or Quick fill); the settled state is the start frame and heat starts at "Start heat at frame". Capacity check with overflow report (open vessel spills, sealed vessel reports over-pressure), layer heights and hydrostatic pressure report.
- Confinement safety: escaped particles are projected back inside and counted; tests for cone, sphere, tilted cylinder, vase-shaped cavity.
- Lava lamp demo: carrier and three waxes fill the cone and settle into layers; clearer glass; renders re-done. Particle scale now applies to the rendered radius.
- Combustion mode keeps its own fill path (documented in docs/TECHNICAL.md).

## 0.3.0
- Fluid mode (panel switch Combustion | Fluid): volumetric, particle-based (Lagrangian) fluid dynamics for general fluids. FLIP/PIC liquids with variable-density projection and Boussinesq convection, Archimedes gas parcels, heat sources and conduction, LOTBCBP containers and colliders (moving walls drive the fluid), Environment-driven gravity, barometric pressure, ambient density, wind and gusts, ground friction.
- Assign Selected as Fluid, fluid presets (water, oils, lava-lamp wax, air, helium, CO2, honey), Lava Lamp / Free Gas Plume / Elevated Gas Tower demos, Geometry Nodes groups (per fluid, mesh per particle).
- New tests (NumPy solver, headless Blender on 5.2.3 and 5.3.0) and renders in docs/renders.

## 0.2.0
- Renamed to Tessera Projection (method: LOTBCBP, Level Of Tessellation Boolean Collision Body Projection).
- Projection Remesh: quad remesh driven by the mesh's own projection (crease and Arc Edge density, symmetry plane detected at any position and rotation, UV/attribute transfer, normals tool, resolution multiplier 0.01 to 12, large-mesh fast path with progress and cancel).
- Boolean projection: Fluid Cavity union and joint cuts.
- Environment card: Earth, Mars, Venus, Titan, Jupiter and editable gas mixes.
- Systems: multi-system detection, twin-engine crossfeed demo, interactive-physics context scan.
- Tested on Blender 5.2.3 LTS and 5.3.0.
