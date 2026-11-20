# Arc Flow Blender extension: Roadmap

Done in 0.2.0: LOTBCBP rename, Boolean projection (Fluid Cavity union, Cut joints), Environment card (atmosphere presets,
gas-mix compound table, drag / spill / expansion coupling, world and ground look), Blender 5.3 compatibility, Extensions-platform
packaging.

Remaining:
- **Phase changes and reactions** driven by per-compound property tables (evaporation, condensation, freezing; combustion beyond the
  flame-temperature model). Explicitly NOT in 0.2.0: the environment gas mix is inert and the atmosphere never changes composition.
- Buoyancy vehicles: balloon gas cell, release valves (needs the mix density already available from the Environment card).
- Multi-stage / multi-propulsion thrust sources feeding a trajectory (Mantis-style).
- Per-voxel temperature in the ray-march volume (Store Named Grid, Blender 5+).
- Incompressible particle-particle interaction (current solver is a pressure-driven tracer model).

Smaller follow-ups: incremental update of the Fluid Cavity field for moving assemblies, an atmosphere altitude profile (scale height is
already reported), EEVEE render check.

## Plan after 0.4.1 (staged, each stage tested and pushed)
1. **0.5.0 two-way fluid-solid coupling.** Particles hand momentum to collision bodies; per-body net force and torque each step; free bodies move (mass, friction, constraints), mechanical bodies (piston, slider, hinge, gear) along their allowed axes; chamber pressure and nozzle thrust act on the chamber shell; moving bodies re-project every frame; per-body toggle (default on).
2. **0.6.0 assignment model.** Container and Shape slots take any geometry (mesh, curve as a tubed channel, metaball, text / surface via the evaluated mesh, particle system); Shape spawn modes Volume, Surface, Vertices, Particle-system points with a density control; default new fluid = a sphere of points with primitive presets (Sphere, Cube, Cylinder, Cone, Torus, Plane); several independent fluid assets interact (mix / react / blend through Surface mode).
3. **0.7.0 one channel model and systems.** A combustion chamber is another channel: chamber behaviour (flow, pressure, ignition, choked flow) is inferred from geometry (narrowest opening, connected cavities) instead of the role; one shared path for Fluid and Combustion where it is safe. Already shared: Environment, level-set fill selection, flame / species chemistry, LOTBCBP collision bodies. Kept separate on purpose: the combustion tracer solver (`core/world.py`) and the Fluid-mode FLIP solver (`core/fluidmode.py`): merging them would change every combustion result and test, so they stay two integrators over one geometry and chemistry layer. Every fluid asset and mechanical assembly is detected as its own system (Systems list), self-initialises, reads the shared Environment and interacts with nearby systems (proximity coupling: mixing, reactions, forces, heat). Geometry Nodes parity: every Fluid feature gets sockets in the groups as combustion has, and every combustion feature that applies to fluid (transmutation, reactions, composition, heat, pressure) is exposed there; outputs are plain geometry and attributes; a test wires our group into a stock-node chain.

## 0.4.1 (done)
- Combustion chain: tanks fill, propellants merge and ignite, neck pressure, choked flow, nozzle expansion, ground wall jet, flame to smoke, 300 m exhaust domain with fade, particle budget, Continuous flow ON / OFF, flame model port, smooth Gaussian render.

## 0.4.0 (done)
- Composition, miscible mixing, absorb/react rules and surface rendering for Fluid mode. Next (0.4.x): combustion mode on the arc-flow.html flow (species and FLAME constants, 300 m domain fade, particle budget, continuous flow), Fill / Settle for combustion tanks, composition-aware viscosity.

## Next (0.3.0)
- Fluid mode: volumetric particle-based fluid dynamics for general fluids (mixing, lava-lamp convection, free gas, fluids without containers, moving colliders, Environment-driven buoyancy and gravity, Geometry Nodes groups). Done in 0.3.0 (see docs/TECHNICAL.md). Next: fluid pushing solids (two-way), per-fluid particle refinement for small wax blobs, custom-mesh instancing verified headless.
- Projection Remesh: curvature-aligned quad layout, exact symmetry in the large-mesh fast path, shape-key transfer, Geometry Nodes exposure.

## XR and AI (planned, after 0.5.0)
Optional link to NVIDIA's XR AI stack (voice agents, headset clients) through our own opt-in bridge and a separate local sidecar; the extension stays purely Blender (no NVIDIA code, network permission only while the bridge is on, off by default). Planned: AI agents and voice commands calling Tessera tools (remesh, simulation, thermal readback, examples, systems); headset scene proportions in, fluid and thermal overlays out; "XR Guided Examples" in the Examples panel. Phase 1 is one voice command driving one operator. Overlays are visual illustrations, not safety predictions. Details, performance limits and estimates: docs/XR_AI_ROADMAP.md.
