# Tessera Projection 0.6.0 — shared brief (all workers read this first)

Repo: DART-Skyboard/tessera-projection, fresh clone at /home/claude/tp6/repo (branch release/0.6.0, coordinator merges).
YOU work ONLY in your own git worktree (/home/claude/tp6/wt-<name>, branch w/<name>), commit there often (never push to GitHub, never touch other worktrees). The coordinator merges branches. Commit trailers:
  Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01FQTVSyCcMVSazjTzMiGcBr
Extension package dir: arc_flow/ (manifest tessera_projection). Pure Blender + numpy; no external wheels; GPL-3.0-or-later; no network permission. Never print/commit tokens. UI wording: never the word "demo" (use "example"); "channel" not "pipe". Everything new must be registered in arc_flow/__init__.py, appear as a tab in arc_flow/ui_tabs.py where relevant, and have tests in tests/ (follow tests/test_cloth_bpy.py, tests/test_hair_bpy.py, tests/test_panels_draw.py patterns; test_panels_draw/test_icons must still pass: valid icon names for Blender 5.2 LTS and 5.3).
Blender: use /home/claude/tp6/bl.sh <args> (Blender 5.3.0, nice'd). Heavy sim/render jobs MUST use `bl.sh -r ...` (serialises across workers via flock; the box has only 2 CPU cores, 7 GB RAM; OOM kills happened before: keep memory modest, one Blender at a time per worker, no leftover processes). Blender 5.2.3 LTS is at /home/claude/blender52 (run the key tests there before finishing). Headless GUI shots need xvfb-run + `--gpu-backend opengl` (software).
Reference material in /home/claude/tp6/refs/ (view with Read / extract frames with ffmpeg):
 - ref_cloth_dress.mp4 — reference cloth render: dress on a mannequin in a boutique ("ATELIER TESSERA") with Tessera Cloth panel visible
 - ref_hair_collision.mp4 — hair on a woman with a glove object swatting the head: hair collides with head/body colliders and reacts
 - ref_linkedin_hair_fire.mp4 — hair reacting to touch/grooming with hands; and a note that wispy fire/smoke is better from sources (curves/thin surfaces) than bigger particle counts (already implemented in 0.5.1)
 - ref_rig_regions_zipper.mp4 + ui_sketch_muscle_panel.jpg + muscle_writeup.txt (/home/claude/tp6/muscle_writeup.txt) — muscle-region colour painting, lip/nose regions, zipper/seam behaviour, panel sketch (Muscles tab under the vertical tab rail; Biped/Quadruped presets; per-region colour swatches; sliders)
Honest-quality rule: verify every claim by running it. If something does not work, say so; never claim a test passed that you did not run. Look at your own renders (Read the PNGs) and fix what looks wrong before reporting. Keep a short log at /home/claude/tp6/log-<name>.md (what is done, what failed, what is left) and update it as you go; finish with a final summary in your reply (commit hash, tests run, known limits).
Time budget: aim to finish in about 2 hours of wall time; commit working increments early (first checkpoint within 40 min).

## ADDENDUM 2026-11-26: use the real Ariel character (user feedback)
The mannequin/test head demos were rejected ("ridiculous"). Demos must use the game's Ariel character.
Game repo (read-only clone, already on disk): /home/claude/dartsolarpunk/dartmeadow-space
  characterassests/Meshy_AI_Autumn_biped_Character_output.glb   (Ariel mesh + skeleton, ~8 MB)
  characterassests/Meshy_AI_Autumn_biped_Meshy_AI_Meshy_Merged_Animations.glb (animation clips)
  characterassests/ariel_hair_strands.bin  ("DHS1", 1.3 MB: ~7k strands, 1,415 simulated guides x 14 pts, positions in the Head-bone frame, head SDF + colliders in header)
  characterassests/emma/ (Emma_Character.glb, Emma_Merged_Animations.glb, emma_hair_strands.bin 10 MB, emma_hair_config.json, emma_clips.json) = optional second character
  hair_strands.js (the game's JS hair solver, parser for DHS1 at ~line 206), docs/hair-new-character.md, docs/hair-collision-notes.md (collision tiers, capsules, boxes)
Do NOT copy these assets into the extension package or commit them to the tessera repo (demo inputs only; keep generated renders/videos in /home/claude/tp6/out). Never print or write credentials.
