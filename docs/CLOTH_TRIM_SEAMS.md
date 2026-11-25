# Cloth Trim, Accents and Seam Seal Control Solving

Three functions added to the Tessera Cloth solving menu (Cloth tab, below the main cloth controls). The brief gave
only the three names, the Maya rig clip (a jacket zipper that opens and closes along a seam while the cloth folds
around it, plus colour-painted lip and nose regions) and the boutique dress clip. This page records how they were
interpreted. The numerical core is `arc_flow/core/cloth_trim.py` (pure NumPy); the Blender side is
`cloth_trim_props.py`, `cloth_trim_ops.py` and `cloth_trim_panel.py`.

## How it plugs into the solver

`core/cloth_solver.py` gained one small hook block (marked `>>> trim hook` / `<<< trim hook`): a list
`ClothSolver.extra_constraints`. Each entry may define `begin_step(S, dt)`, `solve(S, h)` (called once per solver
iteration, after the built-in constraints) and `post_step(S, dt)`. Trims, seams and accent sets are entries on that
list, so they take part in the same XPBD substeps as the cloth. Nothing else in the solver changed.
`cloth_ops.build_solver` and `cloth_ops.step_scene` are wrapped at registration (restored on unregister) so the
shared cloth files only gain registration lines in `__init__.py`.

## Cloth Trim

Interpretation: edge finishing bound to a boundary or seam edge: a hem, piping, lace or ruffle border, belt or
band, collar or cuff band. The strip follows an ordered vertex chain of the garment.

* Edge source: lowest or highest boundary edge (works for a skirt hem loop and for the hem of a slit jacket, where
  the whole boundary is one loop), boundary loop by index, or any vertex group (ordered by mesh edges).
* Own physics: lineal mass (kg/m, added to the chain vertices), stretch stiffness on chain edges, bend stiffness
  with rest-shape memory (a rod-bending constraint that keeps the rest curvature: zero for a straight hem), and
  skip-2 / skip-3 distance constraints.
* Edge stiffening: the stretch and bend stiffness of the cloth edges beside the trim are multiplied by the weight,
  so a stiff hem changes how the skirt edge swirls. Verified in `tests/test_cloth_trim.py` (hem curvature drops from
  0.0190 to 0.0086 in the same gusty wind).
* Flare / ruffle: surplus rest length along the strip, modulated by a number of ruffles; the extra length buckles
  into ruffles (the Lace preset).
* Per-trim material slot (a Material pointer on each trim) and an optional strip mesh (ribbon with a Solidify
  modifier) that follows the cloth.
* Presets: Hem, Piping, Lace / ruffle border, Belt / band, Collar / cuff band.

## Accents

Interpretation: decorative objects attached to the cloth: buttons, bows, jewels, sequins, chains, charms,
tassels, brooches.

* Binding: every accent is bound to its closest surface point as (triangle, barycentric weights) with its offset
  stored in the rest frame of that triangle. Each step the point and the frame (tangent from the first edge,
  normal from the face, binormal) are recomputed from the deformed cloth, so the accent follows folds and
  keeps its lift above the surface. Optional alignment to the surface frame.
* Placement: along an ordered vertex chain (a placket, a neckline), scatter by density (area weighted, optional
  vertex-group weights, seeded), on every group vertex, or at the 3D cursor.
* Instances: copies of a chosen mesh object (or a built-in shape per kind) with a size in metres, in the
  collection "Tessera Accents", each with its own matrix written every step.
* Dynamics: a hanging accent (chain, charm, tassel) has sub-particles (links) hung from its anchor, integrated with
  Verlet, distance constraints and damping. They collide with the collision bodies and the floor when Collide is on.
  A line object draws the chain (Wireframe modifier).

## Seam Seal Control Solving

Interpretation: seams selected as a vertex group (both lips; or two side groups) that open and close with a 0 to 1
slider.

* Pairing: lip vertices are matched across the seam (connected components of the group, nearest unused vertex of
  the other lip).
* Closure: each pair has a path parameter along the seam. Zipper mode zips directionally with a window, so the
  closed fraction of pair i is `clip((p (1 + w) - t_i) / w, 0, 1)`; the slider can be keyframed
  ("Key zip animation").
* Sewing: a distance constraint per pair toward the target gap `gap_open * (1 - closure)` with compliance from
  the seal strength; at strength 1 and closure 1 the compliance is zero (a weld).
* Modes: Zipper (every pair, directional), Button placket (every 4th pair), Velcro (soft, wide peel window),
  Magnetic strip (all at once, soft, every 3rd pair), Laces (every 2nd pair, loose).
* Tear: optional. A pair that has been closed tears for good when its gap before correction exceeds the limit
  (the pair is released and stays released until reset).
* Fold-over ruffle: around the zipper head the seam edges are compressed (Gather, shorter rest length), which
  makes the cloth buckle into folds, plus a small alternating push out of the surface (Fold-over ruffle).
* Zipper pull: an empty or any object. By default it rides the zipper head; with "Pull drives slider" the
  slider is taken from the nearest seam position of the object. The example uses a small gold tab mesh.

## Example and renders

Examples panel, Cloth group, "Jacket with zipper": a mannequin and a slit jacket tube with a zipper seam, a belt-type
hem trim and a collar trim, buttons along a chain, hanging gold charms and sequins scattered by weight. See
`docs/render_cloth_trim.py` and `docs/renders/cloth_trim_*.png`.

## Limits

* Pairing is geometric; the two lips must be separate vertices at distance.
* Accent collision is against the collision bodies and the floor, not the cloth itself.
* The slider is read each step; moving it while the cloth is not stepping does not re-solve.
* No self-collision is switched on in the example (the NumPy self-collision is the slowest part of a step).
