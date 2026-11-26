# log-hair (worker "hair", branch w/hair)

## Audit of 0.6.1 hair collision (before my changes)
- Only ONE primary collider (Scene.arc_flow_hair.collider) + scene collision bodies re-added every step via asset_colliders
  (new add_collider each step => no previous matrix => no collider velocity, no swept motion; a fast glove tunnels).
- `_collide` modified `prev` for friction but `step()` then overwrote `prev[:] = x_old`, so collision friction had NO effect.
- No per-collider thickness/friction, no "add selected" / auto colliders, no colliders list in the panel.
- 0.6.1 `_self_collide` is a pure-python dict loop (very slow). Replaced by a vectorised spatial hash.
- `interpolate_children` was a python loop per child per frame; replaced by a precomputed vectorised skin (180x faster).
- Tessera Hair Render GN scaled the circle profile by Radius AND the curve radius (radius^2 => invisible tubes). Fixed + taper.
- No "assign hair shader" operator existed.

## Done (commits on w/hair, rebased on release/0.6.0 e898366)
- core/hair_solver.py: swept moving colliders (matrix_prev + per-substep interpolation + adaptive substeps), collider velocity
  transfer (swat), working contact friction, per-collider thickness + friction, broad-phase AABB, vectorised hair-hair contact
  (spatial hash, points + segment midpoints) with volume / clumping / strand friction, root-to-tip stiffness curve + bend profile,
  gravity scale (global + per strand), speed clamp, generate_children (scatter, tip clump, frizz), fast vectorised child skinning.
- hair_props/ops/panel: colliders list (Add Selected as Collider, Auto, per-collider thickness/friction/deforming, remove), scene
  collision bodies kept (asset_colliders), n_guides, Make Children, Settle & Set Rest Pose, Assign Hair Shader, scalp shell.
- hair_gn: Principled Hair (Chiang) preset material (root-tip gradient, per-strand random, EEVEE Principled output), tapered tube GN
  (fixed radius^2 bug), scalp shell.
- docs/render_hair_collision.py (scene + Cycles still/clip), docs/capture_hair_ui.py (stepped GUI capture), docs/assemble_hair_ui.sh.
## Tests run
- tests/test_hair_solver.py (numpy): all new tests pass; only the wall-clock budget asserts (60 fps / 120 ms) fail on the loaded
  2-core box (they measured 29 ms/step before hair-hair contact on the same loaded box; hair-hair adds ~10%). Not a logic failure.
- tests/test_hair_bpy.py: ALL PASS (49 checks) on Blender 5.3.0 and on 5.2.3 LTS. test_panels_draw + test_icons: 0 failures on 5.3.0 and 5.2.3.
## Known limits
- UI recording is a stepped capture (software GL, EEVEE Rendered viewport); hair look in EEVEE is thin-curve shading, Cycles Principled Hair is the beauty path.
- Face is a stylised mannequin (no eyes/mouth). Hair-hair contact works on simulated guides; children follow the guides.
