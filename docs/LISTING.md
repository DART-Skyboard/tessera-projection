# extensions.blender.org listing: Tessera Projection 0.5.0

Copy-paste material for the new listing. Everything below comes from README.md, CHANGELOG.md and docs/TECHNICAL.md / SHADERS.md; numbers are only the measured ones.

## 1. Tagline and short description

Tagline (manifest, max 64 characters, no trailing period; current text is 57 characters):

```
Collision, Boolean and quad-remesh projection for any mesh
```

Short description (summary field on the listing):

> Project a mesh against itself and its neighbours: a quad remesh with symmetrize, cavity Boolean projection, and a particle-based computational fluid dynamics engine (fluid and combustion) with shader presets, a wind tunnel and Geometry Nodes sockets.

## 2. Full description (Markdown, paste into the listing)

```markdown
# Tessera Projection

A Level Of Tessellation system in which a mesh is projected against itself and against its neighbours. That projection (**LOTBCBP**, Level Of Tessellation Boolean Collision Body Projection) gives every object a collision body whose resolution follows its geometry, finds where cavities connect and measures its own surface. The same math drives the tools below. Open the 3D Viewport sidebar (N), tab **Tessera**.

## Projection Remesh
A quad remesh operator (it never touches the original or its modifiers).
- Every new vertex is projected back onto the source surface; density follows creases, small features and the Arc Edge curvature scale. No face selection needed.
- **Symmetrize**: with a detected mirror plane the quads are made truly mirror-symmetric. Choose the trusted side (Auto, Left, Right, Average) and a strength; the report shows asymmetry before and after and the deviation from your source. The plane is found from the mesh itself, wherever the object sits and however it is rotated.
- **UV and data handling**: UV maps (including UDIM tiles), colour attributes, vertex groups, materials and generic attributes are transferred. Keep the old UVs, add a fresh set, or skip.
- Retopologizes from the projection at each point's local position and changes the resolution with a multiplier of 0.01 to 12 (above 1 subdivides up, below 1 decimates down). Recalculate Normals (Outside), reads the evaluated or the base mesh, automatic fast path above 200,000 faces with progress and Esc to cancel.

## Boolean Projection
Cavities that intersect become one continuous through-cavity (Fluid Cavity union), with optional Boolean joint cuts in the walls.

## Fluid mode
Turn any mesh into a self-initialising fluid (mass, volume, density, temperature, velocity, viscosity ... missing values are derived). Liquids use FLIP/PIC with a variable-density pressure projection; gases are buoyant Lagrangian parcels.
- Containers or no container, moving containers and colliders, free gas in the Environment, thermal convection (lava lamp). A fluid assigned to a container takes the cavity's real shape, settles under gravity into layers and starts from that state (Fill / Settle, Quick fill).
- **Mixing**: particles carry a composition; miscible fluids exchange composition and heat (Fick diffusion), colours mix subtractively, an optional JSON reaction table transforms species. Render mode Surface welds blobs into one coloured mesh.
- **Transmutation and phase change**: water to steam, crystals growing from nuclei at the walls (and melting again), burning, rust, with latent heat bookkeeping and smooth blending. Steam and smoke are soft volumes.
- Two-way gas pressure with liquids and walls; viscosity surfaces and emission surfaces; Add Fluid Volume; any geometry (mesh, curve, text, metaball, particle system) as Container or Shape with spawn modes and a density control; several independent assets that mix and react.

## Combustion
Self-initialising systems of tanks, channels and chambers: parts are grouped into independent systems, roles are detected automatically. Per-engine chain with its own throat, choked flow and thrust, finite fills, leaks, plugs, thermal and stress colouring, and Good / Bad examples (single engine, twin engines with a crossfeed channel) built from Blender's default primitives (Tessera > Examples).

## Components
Container, Channel, Viscosity surface, Emission surface and Fluid volume, plus plain collision bodies, all detected and re-detected as parts move or connect.

## Environment presets
Earth, Mars, Venus, Titan, Jupiter and Moon atmospheres with editable gas mixes, driving gravity, ambient pressure, drag, wind and plume spread.

## Shader presets (22)
Cycles and EEVEE materials with exposed parameters: water, sea, glass (two variants), oil, honey, syrup, milk, wax, lava, mercury, metal, slime, blood, ice, crystal, smoke, steam, soot, flame, plasma, gas.

## Wind tunnel and thermal readback
A wind tunnel builder (atmosphere presets, a procedural aircraft stand-in, collision bodies) and surface temperature written to point/face attributes and colour, with an optional colour bar and optional heat feedback into the fluid.

## Geometry Nodes
Everything the panels control is also a socket or attribute: reusable node groups (Fluid Source, Particle Cloud, Surface, Thermal, Environment, Fluid Spawn, per-fluid groups) with a Tessera entry in the Add menu.

## Measured performance
Blender 5.3.0 headless on a shared 2-core Linux machine (timings are rough):
- Projection Remesh: 98k triangles in 1.4 s; 991k in 13.6 s; 3.97M in 83.7 s (fast path, about 5.4 GB RAM).
- Collision field: 2000 near-surface queries on 99k triangles in 0.24 s; build at resolution 40 on 99k triangles in 10 to 18 s.
- Fluid mode: 0.24 s per frame at 5,162 particles, 0.75 s at 18,900. The solver is NumPy on the CPU and is not real time.

## Limitations
See the list below this description on the listing page, or docs/LISTING.md in the repository.

Created by Justin Craig Venable, Radical Deepscale LLC (DART Meadow). GPL-3.0-or-later.
```

(Paste the Limitations section of this file at the end of the listing description; the platform shows one text.)

## 3. Limitations (be explicit on the listing)

- Projection Remesh is not an edge-flow-aware remesher: principal curvature is estimated but quad edges are **not aligned to it**; poles are only reduced indirectly; crease density moves vertices, it does not add loops; leftover n-gons are split into quads.
- Symmetrize keeps one half of a non-symmetric source. The large-mesh fast path (above 200,000 faces) keeps only the source triangulation's symmetry (the plane is reported, not enforced) and leaves some triangles unpaired (92% quads on a triangulated Suzanne, 98 to 99.7% on the benchmark sphere).
- Fluid and combustion are simplified, documented models (weakly compressible, no turbulence model), not a full Navier-Stokes solve; viscosity and cohesion are calibrated, not measured. The solver is a NumPy CPU solver: frame times grow with particle count, not real time. Features added in 0.5.0 (gas pressure, transmutation, thermal readback) are not benchmarked separately.
- Particle approximation: solid to fluid coupling is exact, fluid to solid is one-way for some paths; gas does not interact with liquid in the grid projection; frozen crystals still count as liquid in that projection; blobs smaller than about 3 particle spacings break up.
- Combustion: particle masses are large, so chamber pressure is a lumped model, not resolved gas dynamics; jet drag and gas buoyancy are scaled so plumes keep their momentum; no multi-stage thrust or buoyancy vehicles yet.
- Steam, smoke and flame are soft volumes or splats, not resolved vapour; per-voxel temperature in the ray-march volume is not implemented; refraction through overlapping particle spheres is not physical (use Surface render mode); dispersion needs Blender 5.3 and is ignored on 5.2.
- EEVEE: shaders are provided for Cycles and EEVEE, but only Cycles has been render-verified; there is no EEVEE render check yet. Cycles has a per-closure budget that heavy presets can hit.
- Not supported as containers: bare point clouds and hair. Hair and other particle systems work as Shape sources only (a fluid particle at every live particle, read when Build runs). Branching channel containers use overlapping tubes without smooth junctions.
- Thermal readback / wind tunnel: no conjugate heat transfer, flat-plate convection correlation rather than a resolved boundary layer, the aircraft is a procedural stand-in that receives no aerodynamic forces, and particle speed is clamped (raised to 1.5x the wind; above about 20 m/s more substeps are needed).
- Geometry Nodes pressure is an estimate, not the solver's projection pressure; composition is exposed as colour and fluid id, not as a species vector.
- Verified headless on Blender 5.2.3 and 5.3.0 only. Interactive UI sessions and GPU render checks are still to do. Large meshes need RAM (3.97M triangles used about 5.4 GB).

## 4. Tags and permissions

Allowed add-on tags (verified against docs.blender.org/manual/en/dev/advanced/extensions/tags.html on 2026-11-20): 3D View, Add Curve, Add Mesh, Animation, Bake, Camera, Compositing, Development, Game Engine, Geometry Nodes, Grease Pencil, Import-Export, Lighting, Material, Modeling, Mesh, Node, Object, Paint, Pipeline, Physics, Render, Rigging, Scene, Sculpt, Sequencer, System, Text Editor, Tracking, User Interface, UV.

Current manifest tags: `Physics`, `Mesh`, `Modeling`, `Animation`, `Geometry Nodes`, `Render` (all valid). Suggested better set, since remeshing is a headline tool (change in `arc_flow/blender_manifest.toml` only if you agree): `Physics`, `Mesh`, `Modeling`, `Geometry Nodes`, `Material`.

Permission `files` (manifest text, shown to users; the platform wants it short, starting with a capital and no trailing period):

```
Read and write simulation cache files chosen by the user
```

Justification for the reviewer (paste in the submission notes): "The extension writes and reads simulation cache files (compressed NumPy `.npz` snapshots, plus optional PLY point-sequence exports) in a folder the user picks (default `arcflow_cache` next to the .blend), so a baked fluid or combustion run can be replayed without re-solving. It reads the cache back to resume or replay. No other files are read or written, nothing is written outside the chosen folder, there is no network access, no subprocess and no downloaded code."

(Verified against `arc_flow/cachefile.py`: `.npz` cache and PLY export.)

## 5. Links

- Support: https://github.com/DART-Skyboard/tessera-projection/issues
- Website: https://github.com/DART-Skyboard/tessera-projection (manifest `website`)

## 6. Release notes for 0.5.0 (first version on the platform)

```markdown
First release on extensions.blender.org.

- Projection Remesh: quad remesh with density from the surface, Symmetrize (trusted side, strength, before/after report), UV / UDIM, colour, vertex group and material transfer, large-mesh fast path.
- Boolean Projection: intersecting cavities become one through-cavity, with optional joint cuts.
- Fluid mode: FLIP/PIC liquids and Lagrangian gases, mixing, transmutation (steam, crystals, burning, rust), smooth merged surface, viscosity and emission surfaces, fluid volumes, any geometry as Container or Shape.
- Combustion: per-engine chain with throat, choked flow and thrust; single and twin-engine examples.
- Environment presets (Earth, Mars, Venus, Titan, Jupiter, Moon), 22 shader presets, wind tunnel with thermal readback.
- Geometry Nodes: reusable node groups and sockets, Tessera entry in the Add menu.
- Collision field: triangle BVH and sparse bricks (99k-triangle build 10 to 18 s, was 703 s).

Tested headless on Blender 5.2.3 LTS and 5.3.0. See the Limitations section.
```

## 7. Screenshot plan (gallery order)

Filenames are the planned names in `docs/gallery/`; the shots worker writes them. Until they exist, the same views are available in `docs/renders/` (second column). Update this table to the real filenames after pulling.

| # | Gallery file | Existing render for now | Caption |
|---|---|---|---|
| 1 | gallery/01_overview.png | docs/renders/multi_twin_layout.png | Tessera Projection: remesh, Boolean cavities, fluid and combustion in one panel |
| 2 | gallery/02_remesh.png | docs/renders/retopo_quad_crease_density.png | Projection Remesh: symmetric quads, denser around eyes, mouth and ears |
| 3 | gallery/03_symmetrize.png | docs/renders/retopo_symmetrize.png | Symmetrize with a before/after asymmetry report |
| 4 | gallery/04_engine.png | docs/renders/engine_beauty.png | Single engine example built from default primitives |
| 5 | gallery/05_combustion_plume.png | docs/renders/combustion_f150.png | Combustion chain: ignition, choked flow and exhaust plume |
| 6 | gallery/06_lava_lamp.png | docs/renders/fluid_lamp_mid.png | Fluid mode: thermal convection in a contoured container |
| 7 | gallery/07_mixing.png | docs/renders/fluid_mix_mid.png | Colour mixing with Fick diffusion |
| 8 | gallery/08_steam.png | docs/renders/transmute_steam_f120.png | Transmutation: water to steam |
| 9 | gallery/09_crystals.png | docs/renders/transmute_crystal_f70.png | Crystals growing from nuclei at the walls |
| 10 | gallery/10_shaders.png | docs/renders/shaders_contact.png | 22 shader presets for Cycles and EEVEE |
| 11 | gallery/11_wind_tunnel.png | docs/renders/windtunnel_thermal.png | Wind tunnel with surface thermal readback |
| 12 | gallery/12_panel.png | (screenshot of the Tessera sidebar) | The Tessera tab: Examples, Remesh, Fluid, Environment |
| 13 | gallery/13_geometry_nodes.png | (screenshot of a Tessera node group) | Geometry Nodes sockets and Tessera node groups |

Platform image rules to check at upload: PNG or JPEG, first image becomes the thumbnail, landscape 16:9 preferred, keep each under the size limit shown on the upload form.

## 8. Pre-submission checklist

- [ ] `./build.sh` passes (runs `blender --command extension validate arc_flow` and `extension build`); zip is `dist/tessera_projection-0.5.0.zip` (name = `<id>-<version>.zip`).
- [ ] Manifest: `id = "tessera_projection"` (never changes after the first upload; note PUBLISHING.md still says `arc_flow`, outdated), `version = "0.5.0"`, `type = "add-on"`, tagline 57 characters without a period, `license = ["SPDX:GPL-3.0-or-later"]`, `blender_version_min = "5.2.3"`, tags from the list above, `[permissions] files` only.
- [ ] Zip contents: the code and `library/tessera_nodes.blend` (a node-group data file, not a binary executable); no `.so/.dll/.exe/.pyd/.whl`, no `__pycache__`, no cache folder, no tokens. `arcflow_cache/` is outside `arc_flow/` and is not shipped.
- [ ] Clean-config install test: set `BLENDER_USER_CONFIG`, `BLENDER_USER_SCRIPTS`, `BLENDER_USER_EXTENSIONS` to temp dirs, `blender -b --command extension install-file -r user_default --enable dist/tessera_projection-0.5.0.zip`, then run the tests with `--installed`.
- [ ] Description, limitations, release notes and screenshots from this file pasted; images match the gallery table.
- [ ] Support link and website set; LICENSE present in the zip or repo.
- [ ] What the review checks: GPL-compatible license, no obfuscated or minified code, permissions match what the code does (files only; no network, no subprocess), no bundled binaries or wheels, code quality and add-on guidelines (https://developer.blender.org/docs/handbook/extensions/addon_guidelines/), clean registration and unregistration, sensible listing text and images, and no claims the extension cannot back up (hence the Limitations list). Answer reviewer comments on the extension page; a changed zip needs a new version number.
- [ ] Do not paste any API token into the listing or the repo (`BLENDER_EXT_TOKEN` only, from the environment).
