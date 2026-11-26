# worker "cloth" log

## Base
Rebased onto release/0.6.0 cd7c9cb (0.6.1 package). Audit of what 0.6.1 already had for self-collision:
- toggle: `_params()` now always passes `self_collision` (no longer hidden behind Override quality) and the panel shows the checkbox. OK, kept.
- solver: python-loop spatial hash (cell = thickness, per-triangle Python loops), vertex-vs-triangle push-out that only moves the
  vertex, one pass, plus a vertex-vertex loop. Measured slow and one-sided; replaced (same entry points `_self_collide`,
  `count_self_intersections`, `_penetrating_vertices/_untangle` kept).

## Done
- core/cloth_selfcol.py: vectorised NumPy vertex-vs-triangle self-collision (swept-bbox grid broad phase with exact int keys,
  searchsorted vertex lookup, Ericson closest point, 2-hop topological exclusion, per-layer ignore pairs / excluded vertices,
  side from previous positions so tunnelled vertices are pushed back out, mass-weighted barycentric response, Coulomb friction,
  bincount accumulation). Modes FAST (candidates cached, 1 pass; REALTIME preset) and ACCURATE (re-detect each substep, 2 passes).
  Edge-edge contact is NOT implemented (documented in module).
- cloth_solver.py: wired in; REALTIME preset now uses FAST instead of off.
- props/ops/panel: Self-collision box (checkbox, mode, contact thickness, friction, ignore vertex group) - minimal localized edit.
- tests/test_cloth_solver.py: sheet thrown onto a pinned sheet: OFF falls through (min gap -3.26 m), ON/accurate never crosses and rests
  at >= thickness, ON/fast holds (worst -4 mm), layer-ignore pair lets layers pass, friction, flat sheet no contacts.

## Status after rebase / rework (continued)
- selfcol broad phase rewritten: both vertices and triangles hashed over their swept boxes (x and x_prev) +thickness; per-pair
  travel-bound prefilter (plane distance, then closest point). Guards for runaway/non-finite triangles and an overflow cap
  (an earlier version overflowed memory on the dress: OOM kill 137 - fixed).
- Tests: tests/test_cloth_solver.py ALL PASS (python3); tests/test_cloth_selfcol_bpy.py ALL PASS (5.3.0): panel property -> solver
  (toggle, mode AUTO -> FAST for REALTIME, thickness/friction, ignore group), OFF vs ON sheet drop through Blender props;
  tests/test_panels_draw.py PASS.
- Bench (loaded 2-core box, other workers running; ms; two stacked wavy layers, thickness 4 mm):
    5k verts  ACCURATE self-col pass 30 ms / whole step 1098 ms | FAST 17.5 / 263 ms
    20k verts ACCURATE self-col pass 219 ms / whole step 5901 ms | FAST 96 / 1845 ms
  (the XPBD distance/dihedral solve dominates the step, as in the user's 2.9 s/step report; self-collision is 3-15 % of it.)
- Party scene: docs/render_cloth_party.py (build | sim | ui), docs/render_cloth_party_stage2.py (UI capture),
  docs/render_cloth_party_assemble.py. Gown = 8.6k verts, 3 disconnected layers in ONE garment (bodice+skirt, high-low overskirt, train)
  so layers collide through self-collision; mannequin (torso/head/arms/legs) + plinth are colliders; floor is ground_z.
  Sim: 3 substeps, 5 iterations, 1 self-collision pass/substep, ~2.7 s/step, 744 steps (24 settle + 720).
