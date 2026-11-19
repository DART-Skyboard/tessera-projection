# Changelog

## 0.4.1
- Combustion mode (Burn propellants, default on): fuel and oxidiser particles that meet in a chamber react into flame gas (igniter assumed, or autoignition temperature); the pair's O/F sets soot and chemiluminescence through the web app's flame model, ported to `core/flame.py` and checked against the JavaScript (reference values from node in `tests/flame_reference.json`).
- Chamber gas ledger: p_c = m R T / V, drained at the choked throat mass flow (subsonic below the critical pressure ratio); the nozzle exit only passes that mass; gas speed follows the isentropic area-Mach relation (sonic at the throat, supersonic after), scaled to the world speed limit (physical numbers reported).
- Ground impact (wall jet), flame to smoke conversion below 950 K with the flame model's smoke colours, Exhaust domain radius (300 m default, fade over the outer quarter, removal at the radius), Particle budget, Continuous flow ON refills tanks / OFF runs them dry and the plume ends.
- Shared tank-fill selection (`fillsettle.level_select`) for Fluid mode and combustion tanks; Fill / Settle stays a Fluid-mode operator.
- Render: flame / smoke colour and fade attributes (`af_fcol`, `af_fmix`, `af_alpha`) in the Gaussian splat material; time sequence `docs/render_combustion.py`.
- Tests: `tests/test_flame.py`, `tests/test_combustion_chain.py`. Limits in docs/TECHNICAL.md.

## 0.4.0
- Fluid mode: composition model (`core/species.py`): species mass fractions per particle, miscible Fick exchange (mass and enthalpy conserved), subtractive colour mixing, ideal-mixing density, mass-weighted cp and beta, ABSORB with priority and cluster ids, REACT rules with windows and heat release, JSON rule table.
- New render mode Surface (surface nets on a smoothed particle density, welded coloured mesh); Colour Mixing demo with a rotating paddle; per-fluid mix rate, priority and rules in the panel.
- Tests: mixing, mass and heat conservation, dab fade, ABSORB, REACT, JSON round trip, surface merge; Blender tests for the MIX demo and Surface mode (5.2.3 and 5.3.0). Colour-mixing renders in docs/renders.
- Limits: an absorbed particle keeps its original fluid id (cohesion and per-fluid parameters stay with it); viscosity and heat conduction are not composition-aware; surface mode is ragged for blobs under about 4 spacings, so the lamp wax stays spheres.

## 0.3.2
- Fill: analytic level-set fill (stratified by density, exact level) plus a 10-step settle is the default; Pour fill keeps the full dynamic fill; Quick fill 3 steps. Faster than the 0.3.1 dynamic fill (see docs/TECHNICAL.md).
- Wax blobs: cohesive immiscible phases couple their own particles' velocities, lava lamp wax starts as three blobs inside the cone, cold wax denser / warm wax lighter than the carrier; tests for connectivity, interface mixing, rise and sink, two blobs.
- Lamp render: faint blue liquid volume, glossy blobs, lights hidden from the camera and the glass reflections.
- Combustion fill test for cone and sphere tanks (the combustion world already filled the real cavity).

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
