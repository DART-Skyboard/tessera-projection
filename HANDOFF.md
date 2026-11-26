# Tessera Projection — hand-off (2026-11-26)

Repo: https://github.com/DART-Skyboard/tessera-projection  (Blender extension, manifest id `tessera_projection`, package dir `arc_flow`)
Owner: Justin (DART Meadow / Radical Deepscale LLC). Publishes to extensions.blender.org himself.
**Never** put the Blender extensions API token, or any account passwords, in the repo. They are not in this repo.

## Where the work is (branches)
| Branch | Contents | State |
|---|---|---|
| `main` | 0.6.1 package as uploaded by the user (the other AI's latest) + `outside-updates/0.6.1/` archive + docs | stable base |
| `release/0.6.0` | main + **Muscle solver**, **Hair collision rewrite**, **Cloth Trim / Seam Seal** (merged) | integration branch for 0.7.0; manifest still says 0.6.1 |
| `w/cloth` | cloth self-collision modes (accurate / fast / LAYERED) + panel + tests | NOT merged, works for short sims only (see limits) |
| `w/cloth-group` | worker branch building garment-GROUP solving (several objects in a collection solved as one self-colliding unit) | in progress |

Direct download of the integration branch: https://github.com/DART-Skyboard/tessera-projection/archive/refs/heads/release/0.6.0.zip
(whole repo; the installable extension is the `arc_flow/` folder, built with `build.sh`). There is **no 0.7.0 zip yet**; the latest published zip is `outside-updates/0.6.1/tessera_projection-0.6.1.zip`.

## Plan for 0.7.0 (user confirmed the number)
Merge w/cloth(-group) into release/0.6.0 (expect conflicts in core/cloth_solver.py and cloth_ops/panel/props vs the trim hook), run all tests on Blender 5.3.0 and 5.2.3,
bump manifest to 0.7.0, `blender --command extension validate` + build, put zip + README in `outside-updates/0.7.0/`, update README/CHANGELOG, push to main.

## Feature status (honest)
- **Muscle solver** (core/muscle_solver.py, muscle_*.py, docs/MUSCLES.md): region painting, presets, projection solve, shape keys, GN group, export. Panel/tests pass.
  User's intended UX, to verify against the code: each painted REGION has its own card listing that region's muscles; colours are assigned per muscle as painted; unassigned muscles are skipped
  (cartoon = few muscles, scientific = every muscle); a **Flex** slider; an **Accent** slider chooses which end (origin vs insertion/tendon) the contraction pulls toward. Accents belong to the MUSCLE solver.
- **Cloth Trim / Seam Seal** (core/cloth_trim.py, cloth_trim_*.py, docs/CLOTH_TRIM_SEAMS.md): zipping seams. NOTE the cloth-side "Accents" feature built here is decorative trim accents; user says "Accents" belong to the muscle solver, so it should be renamed/clarified.
- **Hair** (core/hair_solver.py, hair_*.py): swept colliders, hair-hair contact, profiles. The 30 s demo (`docs/capture_hair_ui.py`) did NOT visibly show the collision; user rejected it. Needs a rebuilt demo on the real Ariel character.
- **Cloth self-collision**: implemented, but on the 3-layer party gown, layers drift through each other after ~frame 100 (wrong-side vertices: 6 @f24, 25 @f100, 88 @f150, 121 @f191; train in overskirt 144-176 @f150-191). Not fixed on `w/cloth`.
  User requirement: members of a collection must be solved as ONE system (cross-member self-collision + stitching of touching borders + external colliders), same for hair. In progress on `w/cloth-group`.
- **Fluid volumetric / wispy fire**: done earlier (0.5.x).
- Screenshots/videos: all captured in real Blender UI (Material Preview/EEVEE, software GL via xvfb) — no GPU in the build box. User wants real-time shading, lights, shadows visible and 30 s UI recordings for hair, cloth, muscle.

## Demo assets (user feedback: mannequin demos rejected)
Use the game's Ariel character: repo https://github.com/dartsolarpunk/dartmeadow-space
`characterassests/Meshy_AI_Autumn_biped_Character_output.glb`, `..._Merged_Animations.glb`, `characterassests/ariel_hair_strands.bin` ("DHS1", ~7k strands, 1,415 simulated guides x 14 points, Head-bone frame, head SDF),
`hair_strands.js` (JS solver + DHS1 parser), `docs/hair-new-character.md`, `docs/hair-collision-notes.md`. Emma files in `characterassests/emma/`.
Do not copy these assets into the extension package. The user may also supply 3D scenes made by Grok; otherwise recreate scenes from his reference videos (cloth: futuristic blue anime-style party dress on a mannequin in a showroom).

## Roadmap, not in 0.7.0 unless asked
Wind tunnel v2 (streamlines over wings, per-component readback, no container), finished-example beauty renders, selectable CPU/GPU fluid solver, XR/AI bridge, WebGPU for web Arc Flow,
PrintLink (Blender -> Formlabs printers): see docs/handoff/PRINTLINK_FORMLABS_FRAMEWORK.md (parked).

## How to build / test
- Blender 5.3.0 (and 5.2.3 LTS). Tests are in `tests/` (python3 solver tests; `blender -b --python tests/test_*_bpy.py` for UI/props tests: test_panels_draw, test_icons, test_ui_tabs_bpy, muscle/hair/cloth/trim).
- Package: `./build.sh`; validate with `blender --command extension validate`. Manifest permission reasons must be <= 64 chars; the store review wanted fewer tags.
- GUI capture: `xvfb-run -a -s "-screen 0 WxHx24" blender --gpu-backend opengl --factory-startup --python docs/capture_*.py`. Never `pkill Xvfb`.
- Worker logs and the shared brief are in `docs/handoff/`.
