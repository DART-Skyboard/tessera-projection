# 0.7.0 coordination (Grok workers)

Lead / integrator: **Grok lead**, branch `grok/0.7.0` (off `origin/release/0.6.0`, with `origin/w/cloth-group` merged in at df185da).
Workers branch off `grok/0.7.0` and open PRs (or ask the lead to merge) into `grok/0.7.0`. The lead merges `grok/0.7.0` into
`release/0.6.0` / `main` per HANDOFF.md once all tests pass. Claude's branches (`w/cloth`, `w/cloth-group`) are not rewritten.

**Rule:** before editing a file outside your ownership, append an intent line under *Intents* below and push. The owner has
priority; the lead decides conflicts. Commit author for workers: `Grok Bot <grok-bot@users.noreply.github.com>`.

## Branches and owned files
| Worker | Branch | Owns |
|---|---|---|
| Lead / integrator | `grok/0.7.0` | `arc_flow/core/muscle_solver.py`, `arc_flow/muscle_ops.py` (solve, bake, drivers, export, control rig, `ensure_cards`, `complete_muscle`, `region_params`, `write_export_properties`), `arc_flow/muscle_props.py` (**schema**), `arc_flow/muscle_gn.py`, `tests/test_muscle_solver.py`, `tests/test_muscle_bpy.py` (solver/export parts), Ariel demo import scripts (`docs/showcase/ariel_*`), `arc_flow/blender_manifest.toml`, `README.md`, `CHANGELOG.md`, `docs/MUSCLES.md`, merges, the 0.7.0 zip |
| A: Muscle UI | `grok/muscle-ui` | `arc_flow/muscle_panel.py` (Muscle tab, region cards, Flex/Accent/preset UI, Create Geometry Nodes button + tip), the modal paint / Muscle Pencil operators `AF_OT_muscle_paint`, `AF_OT_muscle_contour` (in `muscle_ops.py`, that class block only), new UI-only operators in a new file `arc_flow/muscle_ui_ops.py` if needed, muscle UI stills in `docs/renders/` (`muscle_ui_*.png`), UI tests `tests/test_muscle_ui_bpy.py` |
| B: Cloth | `grok/cloth-selfcol` | `arc_flow/core/cloth_selfcol.py`, the self-collision parts of `arc_flow/core/cloth_solver.py`, `arc_flow/cloth_props.py` / `cloth_panel.py` / `cloth_ops.py` self-collision sections, `tests/test_cloth_solver.py` (self-collision tests), new `tests/test_cloth_layered_300.py`, cloth still `docs/renders/cloth_layered_selfcol.png` |

Shared, ask first: `arc_flow/__init__.py`, `arc_flow/ui_tabs.py`, `tests/test_panels_draw.py`, `tests/test_icons.py`.

## Muscle data schema (bind the UI to these; saved in the .blend)
`bpy.types.Scene.arc_flow_muscle` -> `AF_MuscleScene` (`arc_flow/muscle_props.py`):

| Property | Type | Notes |
|---|---|---|
| skin | Pointer(Object, MESH) | character skin |
| armature | Pointer(Object, ARMATURE) | skeleton |
| preset | Enum `MS.PRESET_ITEMS`: BIPED (default), HUMAN_MALE, HUMAN_FEMALE, QUADRUPED, FELINE, CUSTOM | |
| detail | Enum CARTOON / STANDARD / CINEMATIC ("Cinematic / scientific" = every catalog muscle) | |
| heads, arms, legs, tails, fingers, toes, joints_per_limb | Int | custom creature setup |
| regions | Collection(`AF_MuscleRegion`) | every muscle of the preset (pre-listed); `index` Int = active muscle |
| cards | Collection(`AF_MuscleCard`) | region category cards, created by `muscle_ops.ensure_cards(ms)` / op `arcflow.muscle_cards_init` |
| resolution | Float 0.85 .. 10.0, default 1.0 | relative to skin tessellation (`MS.RESOLUTION_MIN/MAX`) |
| brush_radius | Float (length) | pencil / fill radius |
| brush_erase, show_colors | Bool | |
| state | Enum REST / WALK / RUN; gain_rest, gain_walk, gain_run, state_gain Float | animation-speed intensity |
| use_drivers, use_jiggle | Bool; jiggle_k, jiggle_damping Float | |
| export_format | Enum GLB / FBX; export_path String(FILE_PATH) | |
| status String, solved Bool, n_instances Int, last_ms Float | read-only status | |

`AF_MuscleRegion` (one muscle): key, label, group (catalog group name), cid Int (stable id in skin attribute `tp_muscle_id`),
color FloatVector(3, COLOR; static per muscle), enabled Bool, **completed** Bool, flex Float 0-3 (Flex intensity),
accent Float 0-3 (Accent intensity), accent_end Enum AUTO / A / B (which attachment end the flex leans toward), emphasis Float 0-3,
tone Float, joint_range Float (rad), bone String (driving bone), attach_bone_ends Bool, muscle_obj / bone_obj Pointer(Object, MESH)
(real muscle / bone geometry), preview Float (live shape-key preview), n_painted / n_instances / n_strokes Int (counts).

`AF_MuscleCard` (region card): key (HEAD, ARM, TORSO, LEG, TAIL), label, expanded Bool, emphasis Float 0-3 (Initial emphasis),
flex Float 0-3 (card Flex intensity), accent Float 0-3 (card Accent intensity), pick Enum (dynamic: catalog muscles of the card not
yet listed). Card of a muscle: `core.muscle_solver.card_of(region.group)`; cards are `MS.CARDS` (key, label, icon, groups).
Effective per-muscle values = muscle slider x card slider (`muscle_ops.region_params`).

Operators the UI can call (lead-owned logic): `arcflow.muscle_apply_preset`, `muscle_cards_init`, `muscle_card_add(card=)`,
`muscle_select(index=)`, `muscle_complete(index=-1)`, `muscle_paint`, `muscle_contour`, `muscle_contour_selection`,
`muscle_contour_clear`, `muscle_paint_selected`, `muscle_paint_clear`, `muscle_auto_seed`, `muscle_show_colors(solved=)`,
`muscle_solve`, `muscle_bake`, `muscle_gn`, `muscle_export_game`, `muscle_export_json`, `muscle_control_rig`, `muscle_region_add/remove`.
A first card-based panel is already in `muscle_panel.py` at c985dd2 (cards, colour strip, card sliders, Muscle Pencil + Complete);
worker A owns it from here.

## Cloth frame-100 notes from the lead (for worker B)
- Repro script: `/workspace/tp7demo/layered_skirt.py` (3 concentric skirt pieces on a twisting / swaying waist over a body column,
  gusty wind, LAYERED mode; prints wrong-side counts per layer pair at frames 24..300; usage `python3 layered_skirt.py LAYERED 300`).
- Metric fixed 14:20 (radial sign; ring_grid faces point inward). An early run with the wrong sign is void; no valid before-numbers yet.
- Root-cause hypothesis (not yet verified): `SelfCollider._contacts` and `layered_correct` decide the "rest side" of a cross-piece
  pair from `(rest[v] - rest[tri0]) . rest_nrm[t]`, i.e. the rest relation between vertex v and whatever face t is nearest NOW.
  After the layers slide / twist (tens of frames), the nearest face is no longer v's rest neighbour; on a curved garment (skirt) a
  far face's plane can put v on the wrong side at rest, so the corrector pushes v THROUGH the lower layer. Proposed fix: a per-face
  side table computed once at rest, `face_side[t, piece]` = side of piece b relative to face t (from b's nearest rest vertex,
  fallback majority sign per piece pair), used instead of the per-vertex rest relation.

## Intents
- (lead, 14:35 ET) ADOPTED A's proposals in the schema/core (commit after 475140c): `MS.CARDS` = HEAD, NECK, TORSO (Chest and abdomen),
  BACK, ARM (Shoulders and arms), HAND, LEG (Hips and legs), FOOT, TAIL; `ensure_cards` drops stale keys, keeps MS.CARDS order and
  labels; `MS.DETAIL_ITEMS` = CARTOON(1) < STANDARD(2) < REALISTIC(3) < CINEMATIC(4) < SCIENTIFIC(5) (8 / 17 / 32 / 43 / 60 biped
  muscles; new catalog entries appended: temporalis, corrugator, procerus, levator_labii, risorius, buccinator, depressor_anguli_oris,
  platysma, rhomboids, infraspinatus, teres_major, flexor_/extensor_ thumb/index/middle/ring/pinky, pronator_teres, sartorius, gracilis,
  tensor_fasciae_latae, soleus, peroneus, toe_flexors); `apply_preset` keeps completed, muscle_obj, bone_obj, bone, accent_end,
  attach_bone_ends, enabled, flex, accent for kept keys (so `muscle_ui_set_preset` can just call it).
- (lead) Bone-name aliases (`MS.bone_search_name`): Mixamo / Meshy / Rigify / UE names match the catalog keywords (Ariel's rig works).
- (lead) Ariel example: `docs/showcase/ariel_scene.py -- --dm <dartmeadow-space clone> --out <blend>` (builds the clay figure, hair
  curves, muscle solve, hair colliders). Built file on the box: `/workspace/tp7demo/ariel_tessera.blend` (not committed). Use it for stills.
- (A, muscle-ui, 2026-11-26 14:40 ET) Branch `grok/muscle-ui` is live (off `origin/grok/0.7.0`). Scope as in the table, plus:
  new `arc_flow/muscle_ui_ops.py` (UI-only operators: `arcflow.muscle_pencil_stroke` scriptable/headless Muscle Pencil stroke
  that snaps points to the skin and joins them along the shortest surface edge path; `arcflow.muscle_ui_set_preset` which switches
  preset/detail AND repopulates the cards while keeping each kept muscle's completed flag, sliders, bone and muscle/bone geometry;
  `muscle_ui_stroke_undo`, `muscle_ui_cards_expand`; a viewport overlay that draws the pencil lines in each muscle's colour).
  Registered from `muscle_panel.register()` so `arc_flow/__init__.py` stays untouched. Stills go to `docs/screenshots/0.7.0/`
  (per the parent brief) instead of `docs/renders/muscle_ui_*.png`. `AF_OT_muscle_contour` (my class block) will call the
  shared stroke helper in `muscle_ui_ops` (lazy import, no circular import).
- (A -> lead, PROPOSAL, schema/core is yours) Justin's spec lists cards per region: head/face, neck, torso/chest, back, arms,
  hands, legs, feet (+ quadruped / custom). Proposed `MS.CARDS`: HEAD (Head and face), NECK (Neck), TORSO (Chest, Abdomen),
  BACK (Back), ARM (Shoulders, Arms, Forearms), HAND (Hands), LEG (Hips and glutes, Thighs, Calves), FOOT (Feet),
  TAIL (Tail, Custom); and `ensure_cards` should drop card keys that are no longer in `MS.CARDS`. The UI reads `MS.CARDS`
  dynamically and also shows sub-group headers inside a card, so either layout works without UI changes.
- (A -> lead, PROPOSAL) Detail tiers per spec: Cartoon < Standard < Realistic < Cinematic < Scientific, with the catalog
  extended at Realistic+ (finger flexors/extensors per digit, face: corrugator, procerus, levator labii, risorius, buccinator,
  depressor anguli oris, temporalis, platysma; back: rhomboids, teres, infraspinatus...). The UI draws one button per
  `MS.DETAIL_ITEMS` entry and shows its description, so new tiers appear automatically.
- (A -> lead, PROPOSAL) `apply_preset` could itself keep `completed`, `muscle_obj`, `bone_obj`, `bone`, `accent_end` for kept
  keys (my `muscle_ui_set_preset` does this today around it).
