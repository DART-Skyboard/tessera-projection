# Tessera Projection Muscles

A muscle layer for character skins that are bound to an armature. The artist picks a preset, paints a few region
colours on the skin, optionally draws contour lines, and the solver projects those marks over the whole skin and
builds the per-muscle flex data. The result is used as Blender shape keys (with drivers), as a Geometry Nodes group,
or as an exported data file for a game engine.

Where it lives: the **Muscle** tab in the Tessera Projection sidebar (vertical tab rail, between Cloth and Examples).

## Real time versus baked (honest summary)

| Part | When it runs |
| --- | --- |
| Region projection, tendon / fibre axes, flex weights, displacement fields | **Solve**: offline, one shot. About 0.5 s for a 3.4k vertex skin and a few seconds at 20k vertices. Not live while painting. |
| Shape keys with joint-angle drivers | **Real time in Blender** (normal shape key and driver evaluation). Re-bake after slider changes. |
| Geometry Nodes group "Tessera Muscles" | **Real time in Blender**: Flex, Accent and Region inputs change instantly. The displacement fields are the solved ones, so a slider change on a region needs a re-solve or re-bake, the GN global Flex does not. |
| Game-engine runtime | Uses the JSON sidecar (morph deltas, weights, control mapping). The engine evaluates the flex formula per frame. |
| Jiggle | **Not simulated in Blender.** The spring parameters are written to the sidecar for the engine runtime; `muscle_solver.jiggle_step` is the reference implementation. |
| Animation-state intensity (rest / walk / run) | In Blender it is a scene setting (Animation state) that scales the driver. Inside an engine it is driven by the joint's angular speed through the speed curve in the sidecar. Blender drivers cannot read speed, so a baked animation uses the state you picked. |

## Workflow

1. **Skin and skeleton.** Select the skin mesh (and its armature) and press *Use selected*, or press *Example biped*
   to create a procedural closed biped skin with a 21 bone armature.
2. **Preset.** Choose Biped (default), Quadruped, Human male, Human female, Feline or Custom creature, and a detail
   level, then *Apply preset*. The region list is built from the catalog:
   * **Cartoon** keeps a few big muscles (8 regions on the biped: neck, deltoid, biceps, triceps, chest, quads, hamstrings, calves).
   * **Standard** adds the common game set (17).
   * **Cinematic** adds facial muscles (forehead, eye ring, nose, cheek, lips, chin, jaw), finger / thumb pads,
     foot arch, toes, obliques, serratus and more (32).
   * Human male / female presets multiply the per-region emphasis (male: chest, shoulders, arms; female: glutes, thighs, cheeks, lips).
   * **Quadruped** swaps the arm and leg set for haunch, fore shoulder, lower leg, tail and back muscles.
   * **Custom creature** uses the *Creature setup* counts (heads, arms, legs, tails, fingers per hand, toes per foot,
     joints per limb). Zero removes a category, extra heads add a head-mass region, more than two joints per limb adds a
     generic *limb belly* region. Several heads, arms or legs need no extra setup: every connected patch of a region
     becomes its own muscle, so one cheek region gives one cheek muscle per head.
3. **Paint.** Every region has a fixed, static colour (derived from its catalog position, so "cheek" has the same colour
   on every character, on bipeds and on quadrupeds, and across presets). Pick a region in the list, then:
   * *Paint*: modal region pencil (drag on the skin, `[` and `]` change the radius, Ctrl or *Erase* erases, Esc ends).
   * *Assign selection*: paint the vertices selected in Edit Mode.
   * *Auto-seed*: a rough starting layout from the bones, skin weights, a t range along each bone and a facing
     direction (front / back / outer / inner). Always meant as a starting point you refine.
   * The paint is stored in the integer point attribute `tp_muscle_id` and mirrored in the colour attribute
     `tp_muscle_color`, which is the active colour attribute, so the viewport (colour type Attribute / Vertex) and the glTF / FBX exporters see it.
4. **Contours.** *Draw* a few lines across the surface along each muscle (or *From selection* for an edge chain in
   Edit Mode). A stroke is the fibre axis; the tendons sit at its two ends: flex weight is zero at the ends and
   largest at the belly (profile `sin(pi t)^0.8` along the stroke, Gaussian fall-off across it). Strokes are stored as
   JSON on the mesh and shown as a curve object. Without a stroke the axis is the region patch's principal axis, or
   the driving bone's head and tail when *Tendons at bone ends* is on.
5. **Geometry and bones.** Per region you can assign actual **muscle geometry** (the skin vertices under it are seeded
   with that region, and its long axis sets the tendon ends) and **bone geometry** (the two ends of the bone mesh are
   tendon attachment points). Use them instead of, or together with, painted colours. Bone-end attachment of armature
   bones is *Tendons at bone ends*. The solver does not simulate the muscle meshes themselves; they are used as
   projection seeds and attachment data, and the skin is the only deformed surface.
6. **Sliders.** Per region: *Flex intensity*, *Accent intensity*, *Emphasis* (initial, from the preset), *Rest tone* and
   *Joint range*. **Accent** is the slide of the skin along the muscle toward the joint end it attaches to (End A / B /
   Auto), added on top of the bulge along the skin normal. *Preview* sets the region's keys by hand.
   The global **Resolution factor** (0.1 to 10, default 1.0, two decimals) works like the Remesh multiplier:
   * below 1 the region memberships are smoothed more and weights are blended wider (cartoon, coarse);
   * above 1 the projection leak is lower (sharper region borders), membership smoothing is lighter, and the belly is
     split into fibre bundles with a cosine striation across the muscle (cinematic, fine). At 1.0 the muscle is smooth.
7. **Solve** runs the projection (see below). **Bake to shape keys** creates one key `TP_Muscle_<muscle>` per
   solved muscle (for example `TP_Muscle_biceps.L`) with the current sliders, and, with *Drivers from joint angle*,
   drives every key from its bone: `value = min(1.5, tone + max(|rot X|, |rot Z|) / joint_range * state_gain)`
   (the local rotation of the driving bone, X and Z because they are the bend axes of most rigs; twist is ignored).
8. **Outputs**: *Create Geometry Nodes*, *Export GLB / FBX*, *JSON sidecar*, *Build control rig*.

## What the solver does

`arc_flow/core/muscle_solver.py` (pure numpy, tested without Blender):

1. Region labels (paint + stroke vertices + muscle-mesh seeds) are diffused over the skin graph with a screened
   harmonic solve (edge weights 1/length, a leak term so influence fades over a few edges): each region's seeds are 1, other
   regions' seeds are 0. Dividing by the sum plus a background term gives a sub-partition of unity (unpainted skin stays 0)
   that blends smoothly across neighbouring regions.
2. A region's argmax patch is split into connected components (left and right biceps, one cheek per head). Bilateral
   muscles that cross the mid plane are split by side. Names are `<region>.L`, `.R` (+X is the character's left) or no suffix for centre muscles.
3. Per component: the fibre line, the arc-length position `t` and distance `d` of every vertex to it, the belly profile,
   lateral fall-off, optional fibre bundles, then the membership weight multiplies it. Weights are capped so overlapping muscles never sum above 1 and are smoothed over the skin graph.
4. Unit-flex fields: **bulge** = skin normal x `0.22 x half width x weight` (capped at 35 % of the muscle's half width,
   so the skin cannot fold through itself) and **slide** = the unit fibre direction toward the accent end x `0.6 x amplitude x weight`. Sliders scale these
   linearly (`flex x emphasis` for both, `accent` for the slide).
5. Runtime helpers: `flex_drive(angle, speed, tone, range, gains)` (tone at rest plus joint angle over range, scaled by a rest /
   walk / run gain interpolated from the joint speed) and `jiggle_step` (damped spring).

Limits to know: the skin is displaced along its own normals and fibre direction, there is no volume-preserving
simulation and no sliding contact between neighbouring muscles; very thin patches can only produce small bulges; paint must
stay on the skin shell (inner surfaces get no special treatment); the auto-seed rules are heuristics tuned for
standard humanoid / quadruped bone names (keywords like `upper_arm`, `forearm`, `thigh`, `shin`, `head`, `spine`, and the
Rigify-style `.L` / `.R` suffix is not required).

## Outputs and data

* **Shape keys**: described above. Closed manifold skins stay closed (shape keys cannot change topology).
* **Geometry Nodes** (`arc_flow/muscle_gn.py`, group **Tessera Muscles**, also in the Tessera Add menu): inputs *Geometry*, *Flex*
  (global multiplier), *Accent* (slide multiplier), *Region* (-1 = all, otherwise one region id). Outputs *Geometry* plus the fields
  *Offset* (vector), *Region ID* (integer) and *Flex Weight* (float), all usable with stock nodes. The group applies a rest-space
  offset read from the skin attributes `tp_muscle_bulge`, `tp_muscle_slide`, `tp_muscle_id`, `tp_muscle_flex`. The button adds it as a
  modifier **before** the Armature modifier so the skeleton carries the offset. The attributes are written by Solve and
  refreshed by the button. In Blender 5.2 LTS the modifier may need an update tag after changing an input from Python.
* **Stored in the .blend**: mesh attributes (`tp_muscle_id`, `tp_muscle_color`, `tp_muscle_bulge`, `tp_muscle_slide`, `tp_muscle_flex`,
  `tp_muscle_solved_id`), strokes (`tp_muscle_strokes` custom property), the text block `TP_Muscle_Solve` (all instances,
  one per line) and the scene settings. Saving and re-opening restores everything (tested).
* **GLB / FBX**: *Export GLB / FBX* exports the skin and armature with shape keys (morph targets / blend shapes), skin
  weights and the active colour attribute (`COLOR_0` in glTF). Tested: a GLB of the example holds one morph target per muscle
  with the names preserved, `JOINTS_0` / `WEIGHTS_0`, a skin, and `COLOR_0`; the FBX is written with blend shape data.
  Drivers do not travel in either format: use the sidecar mapping in the engine.
* **JSON sidecar** (*JSON sidecar* button, `<path>.muscles.json`): per muscle the driver bone, vertices, weights, `morph_delta`
  (the offset at key value 1, equal to the glTF morph target), tone, joint range, the rest / walk / run speed curve, jiggle settings and the formula; plus the region table (id, colour, sliders)
  and the control rig mapping. A small summary is also kept in the text block `TP_Muscle_Sidecar`.

## Control rig helper (minimal)

*Build control rig* adds a `CTRL_<bone>` circle controller to every deform bone, parented like the original chain, and a
Copy Rotation constraint (local space) from each controller to its deform bone, so a controller can be grabbed and rotated like a typical FK control
rig. Forearms and shins also get an `IK_<bone>` handle with an IK constraint (chain of 2) that is created with influence 0: raise the
constraint influence (or key it) to switch that limb to IK. This is intentionally small: no pole targets, no foot roll, no spine IK, no
space switching, no walk-cycle tools. Those are roadmap items.

## Tests

* `python3 tests/test_muscle_solver.py` (numpy only).
* `blender --factory-startup -b --python tests/test_muscle_bpy.py` (Blender 5.3 and 5.2 LTS): procedural biped (closed manifold),
  presets, painting, contours, solve, resolution factor, shape-key response to the Flex and Accent sliders, volume growth, manifold
  after flexing, drivers (joint angle and animation state), the Geometry Nodes group, control rig, JSON sidecar, GLB and FBX export, save / reload.
* `tests/test_panels_draw.py` and `tests/test_ui_tabs_bpy.py` also cover the Muscle tab.
* Stills: `docs/render_muscle.py` writes `docs/renders/muscle_regions.png`, `muscle_face_regions.png`, `muscle_flex_before_after.png`.

## Roadmap / not done

* Modal paint and contour operators need a 3D viewport; their core (pencil, strokes, selection chain) is tested headless, the
  mouse handling was not exercised in an automated run.
* Marking the skin or muscle meshes as shared collision bodies (`asset_colliders.mark_projection_collider`) is not wired in.
* Volume-preserving or sliding-contact muscle simulation, per-muscle jiggle inside Blender, richer rig generation.
* Per-region mirroring of paint between left and right.
