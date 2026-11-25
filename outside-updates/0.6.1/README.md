# Tessera Projection 0.6.1 — package drop

Separate from the official `arc_flow/` build workflow (`build.sh` / `publish.sh` do not read this folder). This is the extension package that is currently served on extensions.blender.org, stored here so the repository source matches it.

## Download

- [tessera_projection-0.6.1.zip](tessera_projection-0.6.1.zip)
- https://github.com/DART-Skyboard/tessera-projection/raw/main/outside-updates/0.6.1/tessera_projection-0.6.1.zip

## What is in 0.6.1 (compared with the 0.5.0 source that was in `arc_flow/`)

- **Shared collision-body registry** (`asset_colliders.py`): any object assigned as a Cloth garment, Hair collider or scalp, Fluid / Combustion collider or Wind-tunnel body is marked as a Tessera collision body that collides with itself in its own solver and is seen by every other solver as an inter-body collider. Per-asset toggles remain and default to on.
- **Cloth:** vertex-versus-triangle self-collision with an untangle pass for teleported or already intersecting vertices, and self-intersection diagnostics (`count_self_intersections`); cloth operator and property updates.
- **Hair:** strand self-collision (spatial-hash push-out between points on different strands) and hair collider / operator / panel updates.
- **Wind tunnel and thermal:** stagnation-temperature rise, thermal feedback set up on a tunnel collision body by the add-on itself, and an overlay that keeps the paint with a warm tint, with blackbody incandescence only above about 750 K (the Draper point).
- **UI:** vertical tab rail updates; panel poll callbacks are re-registered so hosted panels hide and show correctly in the real UI.
- Manifest version 0.6.1.

Package contents were imported unchanged as the base commit of the 0.6.x line (`arc_flow/` now equals this zip).
