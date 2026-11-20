# Shader presets (realistic materials for fluids and any asset)

`arc_flow/shader_presets.py` builds node-based materials in code, `arc_flow/shader_ops.py` is the UI (panel *Shader presets*, `Scene.arc_flow_sh`, per-fluid `shader`).

**Presets:** Water, Sea water, Glass, Glass (thin wall), Oil, Honey, Syrup, Milk / cream, Wax, Lava / magma, Mercury, Molten metal, Slime / goo, Blood, Ice, Crystal; volumes: Smoke, Steam, Soot, Flame, Plasma, Gas cloud.

**Technique (per class):**
- Transparent liquids and glass: Principled BSDF with Transmission Weight 1, roughness ~0, IOR (water 1.333, oil 1.47, honey 1.49, ice 1.31, glass 1.5, gem 1.75); the tint and thickness come from a **Volume Absorption** (+ a little Volume Scatter) inside the closed mesh, so thick parts are darker and more saturated (Beer-Lambert). Ripples / frost / facets are Bump or Voronoi normals. Thin-wall glass uses Principled *Thin Wall* for hollow vessels (no refraction through the wall).
- Subsurface (milk, wax, slime, blood): Principled Subsurface Weight with a per-channel radius (red travels furthest) plus a Coat layer; subsurface scale is in scene units, so on millimetre particle spheres it is replaced by diffuse + coat.
- Lava: blackbody emission from temperature (`fm_temperature` or a parameter) mapped between *Cold* and *Hot* temperature; object-space noise drives a cooled, rough, bumpy crust. Metal glows by blackbody above *Glow starts at*.
- Smoke / steam / soot / flame / plasma / gas: Principled Volume (density, anisotropy, absorption colour, noise breakup, blackbody intensity + temperature or emission), an optional soft edge for instanced unit spheres, grid attributes `density` / `temperature` for OpenVDB volume objects.

**Fluid attributes.** `fm_color` (tint, only when the fluids mix: otherwise the attribute is a white placeholder), `fm_temperature`, `fm_density` are read as instancer attributes (points) and geometry attributes (surface mesh), blended with the parameter by *Use fluid ...* switches. Fluid presets map to defaults (Water -> Water, oils -> Oil, Lava lamp wax -> Wax, Honey -> Honey, Air / Helium -> Gas cloud, CO2 -> Smoke); each fluid can choose another or *Keep my material*.

**Variants** (`build(key, preview=, particle=, grid=)`): full; particle (Transparent + Fresnel sheen with the volume, no refraction, so overlapping particle spheres accumulate colour); preview (Principled only, alpha instead of refraction, no volumes); grid. Socket names are looked up at runtime (Principled sockets differ between 4.x and 5.x); `strict=True` raises on unresolved names (tests).

**Engine settings.** EEVEE: refractive presets set Dithered render method, Raytraced Refraction and transparent shadows; subsurface presets Thickness from Shadow; volumes Accurate intersection. Cycles: volume sampling (Multiple Importance for emissive gas), cubic interpolation for grids. Shadow caustics (Cycles 3.2+): *Set up* marks selected objects as casters, other meshes as receivers and all lights as caustic lights (`caustics_hints`). For EEVEE enable Raytracing in the render properties for true refraction; Cycles needs an environment for mirror metals.

**Hooks.** `shader_ops.register()` wraps `fluidmode_ops.ensure_objects` / `_update_surface` / `_glass_material` at runtime (no edits to the fluid modules): after the render objects are refreshed each fluid's shader is applied; the Examples' vessels use the thin-wall glass.

**Limits.** Refraction through overlapping particle spheres is not physical (use Surface render mode for a refractive liquid, or particle variant); volume presets on spheres need the soft edge and unit-sphere instancing; dispersion needs Blender 5.3 (`Transmission Dispersion Scale`, ignored on 5.2); the per-closure budget of Cycles (64) can be hit by subsurface + coat + transmission on one Principled node.

Sources: Blender manual, Principled BSDF and Principled Volume nodes (docs.blender.org/manual/en/dev/render/shader_nodes/shader/), EEVEE material settings (docs.blender.org/manual/en/dev/render/eevee/material_settings.html), Cycles object settings, shadow caustics (docs.blender.org/manual/en/dev/render/cycles/object_settings/object_data.html), Cycles light settings, Beer-Lambert absorption (Volume Absorption node), Blender 4.0 Principled v2 release notes.
