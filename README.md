<p align="center">
  <img src="branding/tessera_1024.png" width="180" alt="Tessera Projection logo">
</p>

<h1 align="center">Tessera Projection</h1>

<p align="center">
  <b>Collision, Boolean and quad-remesh projection for any mesh in Blender.</b><br>
  A Level Of Tessellation system that lets 3D assets project themselves, so they can collide, cut, flow and remesh without manual setup.
</p>

<p align="center">
  <img alt="Blender 5.2.3+" src="https://img.shields.io/badge/Blender-5.2.3%2B-orange">
  <img alt="Tested on 5.3.0" src="https://img.shields.io/badge/tested-5.2.3%20LTS%20%7C%205.3.0-blue">
  <img alt="License GPL-3.0-or-later" src="https://img.shields.io/badge/license-GPL--3.0--or--later-green">
  <img alt="Version 0.6.0" src="https://img.shields.io/badge/version-0.6.0-informational">
</p>

<p align="center">
  <a href="https://cash.app/$autumndeveloper"><img src="sponsorship/cash-app-mark.png" width="28" height="28" align="absmiddle" alt="Cash App"></a>&nbsp;&nbsp;<a href="https://cash.app/$autumndeveloper"><b>Sponsor on Cash App</b></a>
</p>

---

## What it does

Tessera Projection is built around one idea: **a mesh can be projected against itself and against its neighbours**. That projection (called **LOTBCBP**, *Level Of Tessellation Boolean Collision Body Projection*) gives each asset a collision body whose resolution follows the geometry, finds where cavities connect, and measures its own surface, and the same math drives three tools.

| Tool | What it gives you |
|---|---|
| **Projection Remesh** | A fast quad remesh. Dense, symmetric quads along creases, lips, wrinkles and ears; large quads on broad surfaces. Works on characters, cars, trees, CAD, and scans, at any position or rotation. |
| **Boolean Projection** | Cavities that intersect become one continuous through-cavity, with optional Boolean joint cuts in the walls. |
| **Arc Flow** | A computational fluid dynamics engine: self-initialising fluid and combustion simulation through those connected cavities: containers, channels, chambers and nozzles that start empty and only conduct, finite fills, leaks, plugs, thermal and stress colouring. Fluid mode adds liquids, gases, mixing, transmutation (steam, crystals, burning, rust), viscosity and emission surfaces, fluid volumes, a wind tunnel with thermal readback, 22 shader presets and Geometry Nodes sockets. |
| **Hair** | Real-time physics for Blender's Empty Hair Curves: paint and style a groom with the native curves tools, then let Tessera drive collision-body physics, live armature reactivity and game-engine baking. |
| **Cloth** | A real-time XPBD cloth solver for any garment mesh: pins and armature skinning, SDF collision, aerodynamic drag and lift from wind or a running Fluid-mode field, bake to shape keys / point cache / Alembic and export a game-engine cloth package. |

<p align="center">
  <img src="docs/renders/retopo_input_tris.png" width="46%" alt="Triangulated input mesh">
  <img src="docs/renders/retopo_quad_crease_density.png" width="46%" alt="Quad result with denser loops around eyes, mouth, brow and ears">
</p>
<p align="center"><i>Projection Remesh: triangulated input (left), symmetric quads with crease-driven density (right).</i></p>


## Tessera Projection Hair

Paint, sculpt and layer hair (or fur, or feathers) with Blender's native **Empty Hair Curves**, then let Tessera Projection drive the physics:

- **Live simulation** reacts as you pose an armature, move a character, or drive a rig with a controller — a game-engine-style preview inside the viewport.
- **Collision** uses the same LOTBCBP signed-distance field as the fluid solver, with tessellation / resolution controls so you can tune for 60 fps or a dense offline groom.
- **Wind and fluid coupling** samples the Tessera Environment wind (and a running fluid field when one is active).
- **Bake and export** to a `.npz` point cache, Alembic, and a game-engine hair physics package (rest curves, guide weights, per-strand stiffness, collision metadata) for Unreal, Unity or a web engine.

The Hair category lives in its own Tessera N-panel card (`draw_hair_tab`) so it can sit on a vertical tab next to Combustion, Fluid, Remesh and Examples.

<p align="center">
  <img src="docs/renders/hair_solver_preview.png" width="70%" alt="Tessera Projection Hair: a clay studio bust with a dense auburn Principled Hair groom and a colliding hand">
</p>
<p align="center"><i>Hair Groom Example: a stylized clay bust with clumped auburn Empty Hair Curves, rich Principled Hair BSDF highlights, and elastic collision parting around a hand.</i></p>

## Projection Remesh

- **The mesh is its own reference.** Every new vertex is projected back onto the original surface, and the result is measured against it.
- **Density from the surface itself.** Crease angle, small features and the Arc Edge curvature scale set the target size per region. No face selection in Edit Mode.
- **Symmetrize.** With a confident mirror plane the quads are made truly symmetric: pick the side to trust (Auto, Left, Right, Average) and a strength; the report gives asymmetry before/after and the deviation from your source.
- **Symmetry without setup.** The mirror plane is detected from the mesh, wherever the object sits and however it is rotated. Near-symmetric meshes (sculpts, triangulations that do not line up across the sides) are made symmetric; asymmetric regions are left alone and reported.
- **Retopology and resolution in one step.** The mesh is retopologized from the projection at each point's local position, and the **resolution multiplier (0.01 to 12)** changes the face count: 1.0 is a one-to-one remesh, above 1.0 subdivides up, below 1.0 decimates down. Run it again for more.
- **Keeps your data.** UV maps (including UDIM tiles), colour attributes, vertex groups, materials and generic attributes are transferred; choose to keep the old UVs, add a fresh set, or skip.
- **Modifier-safe.** It is an operator, not a modifier. Read the evaluated mesh with any modifier order, or the base mesh. The original and its stack are never touched.
- **Recalculate Normals (Outside)**, before, after, or on its own.
- **Large meshes.** An automatic fast path above 200,000 faces, with progress and Esc to cancel.

Measured on Blender 5.3.0 (shared 2-core machine, so timings are rough):

| Input faces | Output faces | Quad ratio | Time |
|---|---|---|---|
| 98,280 | 49,615 | 98% | 1.4 s |
| 991,018 | 497,076 | 99% | 13.6 s |
| 3,972,048 | 1,989,188 | 99.7% | 83.7 s |

The fast path pairs triangles into quads, so it keeps the source triangulation's symmetry; see the [technical reference](docs/TECHNICAL.md) for what is exact, what is approximate and the limits.

## Arc Flow: systems that initialise themselves

<p align="center">
  <img src="docs/renders/multi_twin_layout.png" width="70%" alt="Two engines joined by a crossfeed channel">
</p>

<p align="center">
  <img src="docs/renders/engine_beauty.png" width="48%" alt="Single rocket engine example: brushed-metal tanks, channels, chamber and nozzle with a heated exhaust spill">
  <img src="docs/renders/engine_xray.png" width="48%" alt="Same engine in X-ray view: propellant inside the tanks, feed lines and chamber">
</p>

- Parts are grouped into independent **systems**; roles (container, channel, chamber) and requirements are detected automatically and re-detected when parts move or connect.
- **Examples built from Blender's default primitives (Tessera > Examples):** a single engine and twin engines with a crossfeed channel, with a Good and a Bad variant that end in enhanced combustion or a wall rupture.
- **Environment card:** Earth, Mars, Venus, Titan and Jupiter atmospheres with editable gas mixes, driving ambient pressure, drag and plume spread.
- Points, Ray-march and Gaussian render paths (Cycles verified), with Geometry Nodes groups for every card.

The simulation is a simplified, documented model, not a full Navier-Stokes solve everywhere.

## Fluid mode: volumetric, particle-based fluid dynamics

<p align="center">
  <img src="docs/renders/fluid_lamp_t0.png" width="48%" alt="Lava lamp at the start">
  <img src="docs/renders/fluid_lamp_mid.png" width="48%" alt="Lava lamp mid-run: heated wax blobs rising and cooling">
</p>
<p align="center">
  <img src="docs/renders/fluid_lamp_late.png" width="48%" alt="Lava lamp later: blobs have risen, cooled and sunk again">
</p>
<p align="center">
  <img src="docs/renders/fluid_free_gas.png" width="48%" alt="Free hot gas plume in wind">
  <img src="docs/renders/fluid_gas_tower_instanced.png" width="48%" alt="Gas tower, instanced mesh per particle">
</p>
<p align="center">
  <img src="docs/renders/fluid_mix_t0.png" width="32%" alt="Colour mixing, start: yellow, blue and red water">
  <img src="docs/renders/fluid_mix_mid.png" width="32%" alt="Colour mixing, mid: a paddle swirls the colours into each other">
  <img src="docs/renders/fluid_mix_late.png" width="32%" alt="Colour mixing, late: the water has blended to one green-yellow">
</p>

**Merging and mixing (0.4.0).** Particles carry a composition (species mass fractions). Miscible fluids exchange composition and heat by Fick diffusion (mass and enthalpy conserved), the colour is the subtractive (weighted geometric) mix, density, expansion and heat capacity follow the composition. Optional per-species priority lets a dominant fluid absorb a minority (one cluster, one colour), and a JSON reaction table (temperature, relative-velocity, density-ratio and concentration windows, heat in J/kg) transforms species. Render mode **Surface** welds blobs into one coloured mesh (two touching blobs give one surface, apart give two). Demo: *Colour Mixing*.

Open the **Fluid** tab in the Tessera sidebar and turn any mesh into a self-initialising fluid (mass, volume, density, temperature, velocity, viscosity and more, with missing values derived). Fluids with or without a container, mixing, thermal convection (lava lamp), free gas rising or sinking in the Environment (gravity, barometric pressure, ambient density, wind, ground), moving containers and colliders, and Geometry Nodes groups per fluid.

Assigned to a container (the lava lamp's glass cone), a fluid takes the cavity's real contoured shape: the quantity is poured in under the Environment gravity, settles into layers (densest at the bottom), and that settled state is the start frame; the lamp then heats and the blobs move. `Fill / Settle` and `Quick fill` are in the panel.

It solves liquids with FLIP/PIC and a variable-density pressure projection, and gases as buoyant Lagrangian parcels. Solid to fluid is exact; fluid to solid is one-way, and gas does not interact with liquid. Measured: 0.24 s per frame at 5,162 particles and 0.75 s at 18,900 (NumPy, 2 shared cores). Details and limits: [technical reference](docs/TECHNICAL.md#fluid-mode-volumetric-particle-based-lagrangian-fluid-dynamics).

## Gallery

Real Blender 5.3 screenshots of every feature (software OpenGL capture, so viewports use plain shading). Full set and capture script: `docs/gallery/`.

<p align="center">
  <img src="docs/gallery/featured_1920x1080.png" width="96%" alt="Tessera Projection in the Blender 5.3 interface">
</p>
<p align="center">
  <img src="docs/gallery/01_projection_remesh.png" width="48%" alt="Projection Remesh with Symmetrize">
  <img src="docs/gallery/03_twin_crossfeed_enhanced.png" width="48%" alt="Twin engines joined by a crossfeed channel">
  <img src="docs/gallery/05_fluid_mode_lava_lamp.png" width="48%" alt="Fluid mode: lava lamp">
  <img src="docs/gallery/06_colour_mixing.png" width="48%" alt="Colour mixing">
  <img src="docs/gallery/08d_wind_tunnel_thermal.png" width="48%" alt="Wind tunnel with thermal readback">
  <img src="docs/gallery/08e_transmutation_crystal.png" width="48%" alt="Transmutation: crystallisation">
  <img src="docs/gallery/08b_shader_presets.png" width="48%" alt="Realistic shader presets">
  <img src="docs/gallery/09_cycles_render_window.png" width="48%" alt="Cycles render window">
</p>

## What is new in 0.6.0

<p align="center">
  <img src="docs/renders/showcase_hair_emma_ui.png" width="96%" alt="Blender 5.3 UI: Cycles viewport, Emma with a subsurface skin shader and Tessera Hair Curves parting around a collision glove, HAIR tab open">
</p>
<p align="center"><i>Real-time Hair Curves solver in the Blender 5.3 UI: 6,000-strand groom on Emma (subsurface skin), body SDF + moving collision glove, HAIR tab open, Cycles rendered viewport.</i></p>

- **Vertical tabbed N-panel UI.** One *Tessera Projection* sidebar panel with a vertical tab rail (Remesh, Fluid, Combustion, Hair, Cloth, Examples). Switching tabs only changes the view; every category keeps running in the same scene.
- **Real-time Hair Curves solver and game-engine bake.** XPBD / Verlet physics for native Empty Hair Curves: inextensible strands, curl / style memory, scalp binding, live armature reactivity, LOTBCBP SDF collision with any number of (moving) collision bodies, wind and fluid coupling, guide-to-child interpolation; bake to a point cache or Alembic and export a hair physics package for Unreal, Unity and web.
- **Real-time fluid-backed Cloth solver and game-engine bake.** Stretch, shear and dihedral bending, pins and armature skinning, SDF collision, optional self-collision, aerodynamic drag and lift against the Environment wind or a running Fluid-mode field. Silk, Cotton, Denim, Leather and Cape presets with Real-time / Balanced / High-detail budgets; bake to shape keys, `.npz` or Alembic and export a game-engine cloth package.
- **Solid-body fluid and combustion blending.** Liquids are splatted into a density field, relaxed with surface tension and meshed as one cohesive body that merges, necks and separates; whitewater foam is emitted where the surface breaks.
- **Curve-sourced wispy volumetrics.** Smoke, steam and slipstreams can be emitted from curves, giving thin, wispy filaments instead of blobby puffs.
- **Wind-tunnel aerodynamic friction heat.** A boundary-layer model adds stagnation / compression heating and skin-friction shear heating, written to the thermal readback attributes and colour ramp, with curve streamlines around the body.
- **Blender 5.3 ready.** Geometry Nodes modifier inputs, Alembic export and render settings use the 5.3 API; the hidden category panels really hide in the GUI sidebar. Tested on Blender 5.3.0 (headless suite plus an interactive GUI session).

<p align="center">
  <img src="docs/renders/showcase_cloth_storefront_ui.png" width="96%" alt="Blender 5.3 UI: boutique storefront, silk dress on a dress form blown by a pedestal fan, Tessera CLOTH tab open">
</p>
<p align="center"><i>Real-time Cloth solver: a silk dress on a dress form in a boutique window, blown by a pedestal fan (oscillating sweep + gusts), CLOTH tab open, Cycles rendered viewport.</i></p>

<p align="center">
  <img src="docs/renders/showcase_fluid_ui.png" width="48%" alt="FLUID tab: cohesive solid-body water with whitewater foam on a collision rock">
  <img src="docs/renders/showcase_windtunnel_ui.png" width="48%" alt="COMBUSTION tab: fighter in the wind tunnel with aerodynamic friction heat and wispy curve streamlines">
</p>
<p align="center"><i>Left: FLUID tab, cohesive solid-body water and whitewater foam. Right: COMBUSTION tab, wind tunnel with aerodynamic friction heat and curve streamlines.</i></p>

Videos (30 s, real-time Cycles viewport in the full Blender UI): Emma hair + collision glove, and the storefront dress in the fan's airflow, are attached to the 0.6.0 release.

## What was new in 0.5.0

- **Remesh Symmetrize**: truly mirror-symmetric quads, with a before/after report.
- **Fluid mode**: two-way gas pressure, transmutation (steam, crystals, burning, rust), a smooth merged liquid surface, soft volume gases, viscosity and emission surfaces, fluid volumes, any geometry as Container or Shape, several independent assets that mix and react.
- **Shader presets**: 22 Cycles + EEVEE materials (water, glass, oil, lava, smoke, steam, flame ...).
- **Wind tunnel and thermal readback**: surface temperature as attributes and colour, atmosphere presets for Earth, Mars, Venus, Titan and the Moon.
- **Combustion**: per-engine chain with its own throat, choked flow and thrust.
- **Geometry Nodes**: reusable Tessera node groups and an Add menu entry.
- **Faster collision field**: triangle BVH and sparse bricks (see Performance).

Full list: [changelog](CHANGELOG.md).

<!-- Screenshots to add before publishing (placeholders): docs/renders/wind_tunnel_thermal.png, docs/renders/shader_presets_sheet.png, docs/renders/combustion_engine_plume.png -->

## Performance (measured)

Blender 5.3.0 headless, shared 2-core Linux machine, so timings are rough. Details and the machine notes are in the [technical reference](docs/TECHNICAL.md).

| What | Measured |
|---|---|
| Projection Remesh, 98k triangles to quads | 1.4 s |
| Projection Remesh, 991k triangles (fast path) | 13.6 s |
| Projection Remesh, 3.97M triangles (fast path, about 5.4 GB RAM) | 83.7 s |
| Collision field, 2000 near-surface queries on 99k triangles | 0.24 s (was 54 s) |
| Collision field build (resolution 40), 99k triangles | 10 to 18 s (was 703 s) |
| Collision field build (resolution 40), 998k triangles | 29 s, 2.9 GB |
| Fluid mode, 5,162 particles | 0.24 s per frame |
| Fluid mode, 18,900 particles | 0.75 s per frame |

The fluid solver is NumPy on the CPU: frame times grow with particle count and it is not real time. The other fluid features (gas pressure, transmutation, thermal readback) add cost that is not yet benchmarked separately.

## Limits

- The fluid and combustion models are simplified, documented models (weakly compressible, no turbulence model), not a full Navier-Stokes solve; viscosity and cohesion are calibrated, not measured.
- Steam and smoke are soft volumes or splats, not resolved vapour; frozen crystals still count as liquid in the grid projection.
- The wind tunnel aircraft is a procedural stand-in, not a real model.
- The large-mesh remesh path keeps the source triangulation's symmetry; the exact mirror is the accurate path and the reduce mode.
- Verified on Blender 5.2.3 and 5.3.0: full headless test suite, plus an interactive Blender 5.3.0 GUI session (virtual display, software OpenGL, CPU Cycles viewport) for the showcase captures. No GPU render check yet.
- Details and open items: [technical reference](docs/TECHNICAL.md).

## Install

1. Download `tessera_projection-0.6.0.zip` from this repository's Releases (or, once approved, from extensions.blender.org: **Edit > Preferences > Get Extensions**, search *Tessera Projection*). Do not unzip it.
2. In Blender 5.2.3 or newer: **Edit > Preferences > Get Extensions > (v) menu, top right > Install from Disk**, pick the zip.
3. Open the 3D Viewport sidebar (**N**), tab **Tessera**. Inside it, use the vertical tab rail (Remesh, Fluid, Combustion, Hair, Cloth, Examples) to switch category; other categories stay active.

From source: `./build.sh` validates and builds the zip into `dist/`.

## Quick start

- **Remesh:** select a mesh, open *Tessera > Remesh*, press **Analyse Topology** to see triangles and odd faces, then **Projection Remesh**.
- **Fluids:** open *Tessera > Examples*, press **Example: Single engine** or **Example: Twin-engine crossfeed**, then switch to the **Combustion** or **Fluid** tab and press **Build**.

## Status and roadmap

Version 0.6.0 is tested on Blender 5.3.0 (every test, the built extension installed from the zip, and an interactive GUI session). GPU render checks are still to do.

Coming next: curvature-aligned quad layout, exact symmetry in the large-mesh path, shape-key transfer. See [docs/ROADMAP.md](docs/ROADMAP.md) and the [changelog](CHANGELOG.md).

## Documentation

- [Technical reference](docs/TECHNICAL.md): methods, benchmarks, limits, tests
- [Publishing notes](PUBLISHING.md)
- [Extensions-platform listing text, limitations and screenshot plan](docs/LISTING.md)
- Gallery images: `docs/gallery/` (UI screenshots for every feature); more renders in `docs/renders/`

## Support and contributing

Bugs and questions: [open an issue](https://github.com/DART-Skyboard/tessera-projection/issues) with your Blender version, the tool you used and, if you can, a small .blend. Pull requests are welcome.

## Credits and license

Created by Justin Craig Venable, Radical Deepscale LLC (DART Meadow). The Arc Edge measuring math is from the LEATR / Arc Edge documentation. Licensed under **GPL-3.0-or-later**; see [LICENSE](LICENSE).
